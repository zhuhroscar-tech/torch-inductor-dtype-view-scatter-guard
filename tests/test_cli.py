"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed."""
from __future__ import annotations

import json

import pytest

import torch_inductor_dtype_view_scatter_guard.core as core
from torch_inductor_dtype_view_scatter_guard.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-inductor-dtype-view-scatter-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert code in (0, 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_output_reports_error_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis and guarding"}
    assert code == 2


def test_torch_unavailable_text_output_shows_fail_headline_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch unavailable: torch is required for diagnosis and guarding" in out
    assert code == 2


def _fake_case(value_bug, alias_bug, guard_ok, n=2):
    return {
        "cache0": [0] * (n * n),
        "data": [float(i) for i in range(n * n)],
        "diag": [0.0] * n,
        "eager_aliases_input": False,
        "compiled_aliases_input": alias_bug,
        "eager_values": [0.0] * (n * n),
        "compiled_values": [float("nan")] * n if value_bug else [0.0] * (n * n),
        "compiled_values_match_eager": not value_bug,
        "compiled_input_corrupted_after_output_mutation": alias_bug,
        "guarded_values_match_eager": guard_ok,
        "guarded_aliases_input_matches_eager": guard_ok,
        "guarded_input_matches_eager_after_output_mutation": guard_ok,
    }


def _fake_report(value_bug, alias_bug, corruption, guard_ok):
    return {
        "torch_version": "0.0.0-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197408",
        "related_issue_url": "https://github.com/pytorch/pytorch/issues/195451",
        "cases": [_fake_case(value_bug, alias_bug, guard_ok)],
        "any_native_value_bug": value_bug,
        "any_native_alias_bug": alias_bug,
        "any_native_corruption": corruption,
        "guard_fully_correct": guard_ok,
    }


def test_no_bugs_shows_info_message_not_fail(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(False, False, False, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[i] no Inductor dtype-view/diagonal_scatter divergence reproduced" in out
    assert code == 0


def test_guard_failed_label_shown_when_guard_ineffective(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, True, True, False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "GUARD-FAILED" in out
    assert "[X] guard did NOT restore eager's values/aliasing contract on at least one case" in out
    assert code == 1


def test_bug_reproduced_shows_fail_headline_and_ok_guard(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, True, True, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] Inductor dtype-view/diagonal_scatter value+aliasing bug reproduced on this host (pytorch#197408)" in out
    assert "[OK] safe_compiled_dtype_view_diagonal_scatter() restores eager's values and non-aliasing contract on every case" in out
    assert code == 0
