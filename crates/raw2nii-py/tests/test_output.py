"""Tests for raw2nii._output ANSI color/formatting helpers."""

from __future__ import annotations

from raw2nii import _output


class _FakeStream:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_color_enabled_false_when_no_color_flag_set():
    assert _output.color_enabled(_FakeStream(True), no_color=True) is False


def test_color_enabled_false_when_NO_COLOR_env_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert _output.color_enabled(_FakeStream(True), no_color=False) is False


def test_color_enabled_true_when_tty_and_no_overrides(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert _output.color_enabled(_FakeStream(True), no_color=False) is True


def test_color_enabled_false_when_stream_not_a_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert _output.color_enabled(_FakeStream(False), no_color=False) is False


def test_color_enabled_false_when_stream_has_no_isatty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    class _NoIsatty:
        pass

    assert _output.color_enabled(_NoIsatty(), no_color=False) is False


def test_colorize_wraps_text_when_enabled():
    assert _output.colorize("ok", "green", True) == "\033[32mok\033[0m"


def test_colorize_passthrough_when_disabled():
    assert _output.colorize("ok", "green", False) == "ok"


def test_colorize_passthrough_for_unknown_color():
    assert _output.colorize("ok", "not-a-color", True) == "ok"


def test_progress_line_plain():
    line = _output.progress_line(2, 5, "scan.mat", enabled=False)
    assert line == "[2/5] converting scan.mat"


def test_progress_line_colored():
    line = _output.progress_line(2, 5, "scan.mat", enabled=True)
    assert line == "\033[36m[2/5]\033[0m converting scan.mat"
