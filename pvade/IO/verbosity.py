"""Verbosity helpers used to control PVade terminal output.

The verbosity level is read from the ``PVADE_VERBOSITY`` environment
variable and interpreted as:

* ``0``: quiet mode (suppress INFO messages in terminal)
* ``1``: default mode (INFO on rank 0 only)
* ``2``: verbose mode (INFO on all ranks)
"""

import os

_VERBOSITY_PREFIX = "__PVADE_VERBOSITY_"


def get_verbosity_level(default=1):
    """Return runtime verbosity level from ``PVADE_VERBOSITY``.

    Args:
        default (int, optional): Value used when the env var is unset or
            invalid. Defaults to 1.

    Returns:
        int: Non-negative verbosity level.
    """
    raw_level = os.environ.get("PVADE_VERBOSITY", str(default))

    try:
        return max(0, int(raw_level))
    except ValueError:
        return max(0, int(default))


def should_emit_terminal_message(rank, message_type, verbosity_level, required_level=1):
    """Decide whether a message should be printed to terminal.

    Args:
        rank (int): MPI rank of the process writing the message.
        message_type (str): Either ``"INFO"`` or ``"ERROR"``.
        verbosity_level (int): Active verbosity level.
        required_level (int, optional): Per-message required verbosity level.
            Defaults to 1.

    Returns:
        bool: ``True`` when the message should be echoed to terminal.
    """
    if message_type == "ERROR":
        return True

    if verbosity_level <= 0:
        return False

    if verbosity_level < required_level:
        return False

    if verbosity_level == 1:
        return rank == 0

    return True


def emit_verbosity_print(message, level=1):
    """Emit a print message tagged with its required verbosity level."""
    print(f"{_VERBOSITY_PREFIX}{level}__ {message}")


def parse_verbosity_print(message):
    """Parse and strip optional verbosity prefix from a message.

    Returns:
        tuple: ``(required_level, clean_message)``
    """
    if not message.startswith(_VERBOSITY_PREFIX):
        return 1, message

    try:
        suffix = message[len(_VERBOSITY_PREFIX) :]
        level_text, clean_message = suffix.split("__ ", 1)
        return max(0, int(level_text)), clean_message
    except Exception:
        return 1, message
