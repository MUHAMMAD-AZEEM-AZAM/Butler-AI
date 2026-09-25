"""CLI: python -m stage1_voice --text "..." | --audio path.wav  -> Task as pretty JSON."""

import argparse
import logging
import sys

from stage1_voice.errors import ParseError, UnknownObjectError
from stage1_voice.voice import parse_command, parse_text


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stage1_voice",
        description="Parse a command into a Task and print it as pretty JSON.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Natural-language command to parse.")
    group.add_argument("--audio", help="Path to an audio file (Speechmatics ASR -> parse).")
    parser.add_argument("--debug", action="store_true", help="Log raw model output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        task = parse_text(args.text) if args.text is not None else parse_command(args.audio)
    except (ParseError, UnknownObjectError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(task.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
