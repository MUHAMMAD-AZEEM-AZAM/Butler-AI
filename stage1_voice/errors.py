"""Errors raised by stage1_voice when a command cannot be parsed into a valid Task."""


class ParseError(ValueError):
    """The model output could not be parsed/validated into a Task (after retry)."""


class UnknownObjectError(ValueError):
    """The parsed Task references a name outside the allowed vocabulary (after retry)."""
