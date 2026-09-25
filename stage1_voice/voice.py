"""Stage 1: natural-language command -> structured Task (Claude parser + Speechmatics ASR).

STUB mode: without ANTHROPIC_API_KEY (or with VOICE_STUB=1) parse_text returns the
fixed example Task so the stub pipeline keeps running with only pydantic + pyyaml + numpy.
The anthropic / speechmatics imports stay inside the functions for the same reason.
"""

import json
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic import ValidationError

from common import EXAMPLE_COMMAND
from common.types import ActionType, HoldRequirement, Task, TaskStep

from stage1_voice.errors import ParseError, UnknownObjectError
from stage1_voice.vocab import Vocabulary, load_vocab, load_voice_config, validate_task_vocabulary

logger = logging.getLogger(__name__)

_MAX_TOKENS = 4096


def _example_task(command: str) -> Task:
    """The official example Task: stub-mode return value and the few-shot example."""
    return Task(
        command=command,
        steps=[
            TaskStep(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            TaskStep(id=2, action=ActionType.PICK, arm="A", object="plate", depends_on=[1]),
            TaskStep(
                id=3,
                action=ActionType.PLACE,
                arm="A",
                object="plate",
                destination="table",
                depends_on=[2],
            ),
            TaskStep(id=4, action=ActionType.PICK, arm="B", object="mug"),
            TaskStep(id=5, action=ActionType.PICK, arm="A", object="water_bottle", depends_on=[3]),
            TaskStep(
                id=6,
                action=ActionType.POUR,
                arm="A",
                source="water_bottle",
                into="mug",
                depends_on=[4, 5],
                requires_hold=HoldRequirement(arm="B", object="mug"),
            ),
        ],
        constraints=["keep_glasses_away_from_edge"],
    )


def _stub_mode() -> bool:
    return os.environ.get("VOICE_STUB") == "1" or not os.environ.get("ANTHROPIC_API_KEY")


def _build_system_prompt(vocab: Vocabulary) -> str:
    """Assemble the parser system prompt from the pydantic schema + config vocabulary."""
    schema = json.dumps(Task.model_json_schema(), indent=2)
    example = _example_task(EXAMPLE_COMMAND).model_dump_json(indent=2, exclude_none=True)
    return f"""You convert a natural-language command for a dual-arm tabletop robot into a Task JSON object.

The Task must validate against this JSON schema:
{schema}

Allowed names — use ONLY these, exactly as written:
- actions: {list(vocab.actions)}
- objects (for object / source / into): {list(vocab.objects)}
- drawers (for target): {list(vocab.drawers)}
- destinations (for destination): {list(vocab.destinations)}
- arms: {list(vocab.arms)}

Example.
Command: {EXAMPLE_COMMAND}
Task:
{example}

Rules:
- Respond with ONLY a JSON object. No prose, no markdown fences.
- Use only the allowed names listed above.
- Steps are sequential: each step's depends_on lists the ids of steps that must finish first.
  Exception: if a step needs an object held by the other arm (e.g. pouring into a held mug),
  set requires_hold to that arm and object instead of forcing full sequencing.
- A pour with arm X implies that arm X is holding the source object: always emit an
  explicit pick step for the source object with the same arm as its own step before
  the pour, and list that pick's id in the pour step's depends_on — even if the
  command does not mention picking up the source.
- If the command mentions an object that is not in the allowed list, do not invent a name.
  Map it to the closest allowed name only if it is an obvious synonym (e.g. "cup" -> "mug").
  Otherwise leave that step out and add the string "unknown_object:<word>" to constraints."""


def _strip_fences(raw: str) -> str:
    """Defensively strip accidental ```/```json fences around the model output."""
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _parse_and_validate(raw: str, command: str, vocab: Vocabulary) -> Task:
    """Model output text -> Task, or raise (JSONDecodeError | ValidationError | UnknownObjectError)."""
    data = json.loads(_strip_fences(raw))
    if isinstance(data, dict):
        data["command"] = command  # the source of truth is what the user actually said
    task = Task.model_validate(data)
    validate_task_vocabulary(task, vocab)
    return task


def parse_text(text: str) -> Task:
    """Parse a natural-language command into a Task via the Claude API.

    Retries once on bad JSON / schema / vocabulary errors, feeding the error back
    to the model; raises ParseError / UnknownObjectError if it still fails.
    """
    if _stub_mode():
        logger.warning("stage1_voice running in STUB mode — set ANTHROPIC_API_KEY")
        return _example_task(text)

    import anthropic

    vocab = load_vocab()
    voice_cfg = load_voice_config()
    max_retries = int(voice_cfg.get("max_retries", 1))
    client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY

    system = _build_system_prompt(vocab)
    messages: list[dict] = [{"role": "user", "content": text}]
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        response = client.messages.create(
            model=voice_cfg["model"],
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=messages,
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        logger.debug("stage1_voice attempt %d raw model output: %s", attempt + 1, raw)
        try:
            return _parse_and_validate(raw, text, vocab)
        except (json.JSONDecodeError, ValidationError, UnknownObjectError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation: {exc}\n"
                        "Fix the problem and respond with ONLY the corrected JSON object."
                    ),
                }
            )

    if isinstance(last_error, UnknownObjectError):
        raise UnknownObjectError(
            f"command references names outside the allowed vocabulary (after retry): {last_error}"
        ) from last_error
    raise ParseError(f"could not parse command into a valid Task (after retry): {last_error}") from last_error


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using Speechmatics if key is available, else SpeechRecognition with auto-ffmpeg conversion."""
    api_key = os.environ.get("SPEECHMATICS_API_KEY")
    if api_key:
        try:
            from speechmatics.batch_client import BatchClient
            from speechmatics.models import ConnectionSettings

            settings = ConnectionSettings(url="https://asr.api.speechmatics.com/v2", auth_token=api_key)
            config = {"type": "transcription", "transcription_config": {"language": "en"}}
            with BatchClient(settings) as client:
                job_id = client.submit_job(audio=audio_path, transcription_config=config)
                transcript: str = client.wait_for_completion(job_id, transcription_format="txt")
            return transcript.strip()
        except Exception as exc:
            logger.warning("Speechmatics ASR failed (%s); trying SpeechRecognition fallback", exc)

    # Local / Google speech recognition fallback (needs WAV)
    import subprocess
    import tempfile
    from pathlib import Path
    import speech_recognition as sr

    wav_path = Path(audio_path)
    tmp_wav = None
    if wav_path.suffix.lower() != ".wav":
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_wav = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", tmp_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            wav_path = Path(tmp_wav)
        except Exception as exc:
            logger.warning("FFmpeg audio conversion failed: %s", exc)

    try:
        r = sr.Recognizer()
        with sr.AudioFile(str(wav_path)) as source:
            audio_data = r.record(source)
        transcript = r.recognize_google(audio_data)
        return transcript.strip()
    except Exception as exc:
        logger.warning("ASR failed: %s; falling back to default command", exc)
        return EXAMPLE_COMMAND
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def parse_command(audio_path: str) -> Task:
    """Transcribe an audio file and parse it into a Task."""
    transcript = transcribe_audio(audio_path)
    logger.info("stage1_voice transcript for %s: %r", audio_path, transcript)
    return parse_text(transcript)

