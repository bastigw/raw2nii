"""Tests for the raw2nii console-script CLI's detailed/colorized output.

These patch raw2nii.cli._convert_many / _build_and_verify_archive so no
real .mat files or native conversion is needed - only argument parsing
and the surrounding output logic is under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raw2nii import cli


def _outcome(status, input_path, **extra):
    return {"status": status, "input": input_path, **extra}


@pytest.fixture
def one_mat_file(tmp_path: Path) -> Path:
    p = tmp_path / "scan.mat"
    p.write_bytes(b"")
    return p


def test_no_color_flag_disables_ansi_codes(monkeypatch, capsys, one_mat_file):
    monkeypatch.setattr(
        cli,
        "_convert_many",
        lambda **kwargs: [
            _outcome("written", str(one_mat_file), output=str(one_mat_file) + ".nii.gz")
        ],
    )
    # Force color_enabled True so we can prove --no-color overrides it.
    monkeypatch.setattr(cli._output, "color_enabled", lambda stream, no_color: not no_color)

    rc = cli.main([str(one_mat_file), "--no-color"])

    assert rc == 0
    out = capsys.readouterr()
    assert "\033[" not in out.out
    assert "\033[" not in out.err


def test_colorized_output_when_color_forced_on(monkeypatch, capsys, one_mat_file):
    monkeypatch.setattr(
        cli,
        "_convert_many",
        lambda **kwargs: [
            _outcome("written", str(one_mat_file), output=str(one_mat_file) + ".nii.gz")
        ],
    )
    monkeypatch.setattr(cli._output, "color_enabled", lambda stream, no_color: True)

    rc = cli.main([str(one_mat_file)])

    assert rc == 0
    out = capsys.readouterr()
    assert "\033[32m" in out.out  # green "written" line


def test_summary_line_reports_counts(monkeypatch, capsys, tmp_path):
    written = tmp_path / "a.mat"
    skipped = tmp_path / "b.mat"
    failed = tmp_path / "c.mat"
    for p in (written, skipped, failed):
        p.write_bytes(b"")

    monkeypatch.setattr(
        cli,
        "_convert_many",
        lambda **kwargs: [
            _outcome("written", str(written), output=str(written) + ".nii.gz"),
            _outcome("skipped", str(skipped), reason="already exists"),
            _outcome("failed", str(failed), error="boom"),
        ],
    )
    monkeypatch.setattr(cli._output, "color_enabled", lambda stream, no_color: False)

    rc = cli.main([str(written), str(skipped), str(failed)])

    assert rc == 1  # one failure
    out = capsys.readouterr()
    assert "1 written, 1 skipped, 1 failed" in out.err


def test_json_log_mode_has_no_ansi_codes_or_extra_lines(monkeypatch, capsys, one_mat_file):
    monkeypatch.setattr(
        cli,
        "_convert_many",
        lambda **kwargs: [
            _outcome("written", str(one_mat_file), output=str(one_mat_file) + ".nii.gz")
        ],
    )
    monkeypatch.setattr(cli._output, "color_enabled", lambda stream, no_color: True)

    rc = cli.main([str(one_mat_file), "--json-log"])

    assert rc == 0
    out = capsys.readouterr()
    assert "\033[" not in out.out
    lines = [line for line in out.out.splitlines() if line]
    assert len(lines) == 2  # one per-file JSON line + one totals JSON line


def test_progress_preamble_printed_by_default(monkeypatch, capsys, one_mat_file):
    monkeypatch.setattr(
        cli,
        "_convert_many",
        lambda **kwargs: [
            _outcome("written", str(one_mat_file), output=str(one_mat_file) + ".nii.gz")
        ],
    )
    monkeypatch.setattr(cli._output, "color_enabled", lambda stream, no_color: False)

    cli.main([str(one_mat_file)])

    out = capsys.readouterr()
    assert "converting 1 file(s)" in out.err
