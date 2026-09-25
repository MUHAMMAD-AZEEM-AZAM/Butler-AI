#!/usr/bin/env python3
"""Run the full pipeline end-to-end: voice -> perception -> policy -> execute -> verify.

Works today with stubs: python scripts/run_pipeline.py [--command "..."] [--seed 3]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXAMPLE_COMMAND
from common.pipeline import run_once

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the table-setting pipeline end-to-end.")
    parser.add_argument("--command", default=None, help="Natural-language command text.")
    parser.add_argument("--audio", type=Path, default=None, help="Path to audio file (.wav, .m4a, .mp3).")
    parser.add_argument("--mic", action="store_true", help="Record 5 seconds from your laptop microphone.")
    parser.add_argument("--seed", type=int, default=0, help="Randomization seed for reset_scene.")
    args = parser.parse_args()

    command_text = args.command
    if command_text is None:
        if args.mic:
            import speech_recognition as sr
            r = sr.Recognizer()
            print("\n[Microphone] Listening to your laptop microphone for 5 seconds... Speak now!")
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.6)
                audio_data = r.record(source, duration=5)
            try:
                command_text = r.recognize_google(audio_data)
                print(f"[Transcribed]: \"{command_text}\"\n")
            except Exception as exc:
                print(f"[ASR]: Could not parse speech ({exc}); using default command.")
                command_text = EXAMPLE_COMMAND
        elif args.audio and args.audio.is_file():
            from stage1_voice.voice import transcribe_audio
            print(f"[Audio] Transcribing audio file: {args.audio.name}...")
            command_text = transcribe_audio(str(args.audio))
            print(f"[Transcribed]: \"{command_text}\"\n")
        else:
            command_text = EXAMPLE_COMMAND

    print("=" * 72)
    print("PIPELINE  voice -> perception -> policy -> bimanual -> verify")
    print("=" * 72)

    result = run_once(command_text, seed=args.seed)
    for line in result.log:
        print(line)

    if result.success:
        print("\nRESULT: SUCCESS")
        return 0
    print(f"\nRESULT: FAIL (after {result.attempts} attempts)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
