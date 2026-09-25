"""Stage 1: voice/text -> Task."""

from stage1_voice.voice import parse_command, parse_text

__all__ = ["parse_text", "parse_command"]
