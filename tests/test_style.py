"""Tests for the shared style module -- ported verbatim from the fleet's
canonical copy (see torch-multioutput-alias-guard/tests/test_style.py
for the original), since this repo also ships that exact style.py."""
from __future__ import annotations

import io

from torch_inductor_dtype_view_scatter_guard.style import (
    Style,
    bool_badge,
    print_fields,
    resolve_style,
    section,
    status_headline,
)


def test_style_disabled_is_identity():
    s = Style(False)
    assert s.bold("x") == "x"
    assert s.red("x") == "x"
    assert s.dim("x") == "x"


def test_style_enabled_wraps_ansi():
    s = Style(True)
    assert s.red("x") == "\033[31mx\033[0m"


def test_resolve_style_no_color_flag_wins(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    style = resolve_style(no_color_flag=True)
    assert style.enabled is False


def test_resolve_style_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    style = resolve_style(no_color_flag=False)
    assert style.enabled is False


def test_resolve_style_force_color_env_without_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    stream = io.StringIO()
    style = resolve_style(no_color_flag=False, stream=stream)
    assert style.enabled is True


def test_resolve_style_defaults_to_no_color_when_piped(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    stream = io.StringIO()
    style = resolve_style(no_color_flag=False, stream=stream)
    assert style.enabled is False


def test_bool_badge_true_false_none():
    style = Style(False)
    assert bool_badge(style, True) == "yes"
    assert bool_badge(style, False) == "no"
    assert bool_badge(style, None) == "unknown"


def test_print_fields_empty_rows_no_output(capsys):
    print_fields([])
    assert capsys.readouterr().out == ""


def test_print_fields_aligns_columns(capsys):
    print_fields([("a", "1"), ("longer", "2")])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("  a")
    assert lines[1].startswith("  longer")


def test_status_headline_ascii_fallback():
    style = Style(False)
    text = status_headline(style, "ok", "hello")
    assert text == "[OK] hello"


def test_status_headline_ansi_variant():
    style = Style(True)
    text = status_headline(style, "fail", "bad")
    assert "\033[" in text
    assert "bad" in text


def test_section_prints_title(capsys):
    section("my section")
    out = capsys.readouterr().out
    assert "my section" in out
