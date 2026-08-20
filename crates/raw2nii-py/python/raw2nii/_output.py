"""ANSI color and progress-formatting helpers for the raw2nii console
script. Stdlib-only, no dependency on the compiled extension's internals
so it stays trivial to unit test.
"""

from __future__ import annotations

import os

RESET = "\033[0m"

_CODES = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def color_enabled(stream, no_color: bool) -> bool:
    """Whether ANSI color codes should be written to ``stream``.

    ``--no-color`` and the NO_COLOR convention (https://no-color.org,
    any non-empty value) both force color off; otherwise color is on
    only when ``stream`` reports itself as a tty.
    """
    if no_color:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def colorize(text: str, color: str, enabled: bool) -> str:
    code = _CODES.get(color)
    if not enabled or code is None:
        return text
    return f"{code}{text}{RESET}"


def progress_line(index: int, total: int, path: str, enabled: bool) -> str:
    prefix = colorize(f"[{index}/{total}]", "cyan", enabled)
    return f"{prefix} converting {path}"
