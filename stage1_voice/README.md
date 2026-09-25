# stage1_voice — Voice & Instruction Understanding

Parses a natural-language command into a structured `common.types.Task`:

- `parse_text(text) -> Task` — Claude (model from `configs/default.yaml` → `voice.model`)
  turns the command into Task JSON, validated against the pydantic schema **and** the
  allowed vocabulary. On bad output it retries once with the error fed back, then raises
  `ParseError` / `UnknownObjectError` (see `errors.py`) — never silently-wrong data.
- `parse_command(audio_path) -> Task` — Speechmatics batch ASR (language `en`) → `parse_text`.

## Env vars

| Var | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Claude parsing. **Unset ⇒ STUB mode**: `parse_text` returns the fixed example Task with a one-line WARNING, so `python scripts/run_pipeline.py` keeps working with only pydantic + pyyaml + numpy. |
| `VOICE_STUB=1` | Force STUB mode even with a key. |
| `SPEECHMATICS_API_KEY` | Audio transcription; `parse_command` raises without it. |

Copy `.env.example` (repo root) and export the values.

## CLI

```bash
python -m stage1_voice --text "Open the top drawer, pick up the plate with arm A, ..."
python -m stage1_voice --audio stage1_voice/samples/official_command.wav
python -m stage1_voice --text "..." --debug   # also logs the raw model output
```

Prints the resulting Task as pretty JSON — use `--text` to feed stage 3 without audio.

## Tests

```bash
pytest stage1_voice/tests -q          # unit tests, no API key needed (live tests auto-skip)
pytest stage1_voice/tests -q -m live  # live API tests (need ANTHROPIC_API_KEY)
```

## Vocabulary

Allowed object/drawer/destination/arm names live in **one place**:
`configs/default.yaml` → `scene:` (updated as the MuJoCo
scene evolves). The allowed action list comes from the `ActionType` enum in
`common/types.py`. `stage1_voice/vocab.py` loads both and validates every parsed Task
against them.
