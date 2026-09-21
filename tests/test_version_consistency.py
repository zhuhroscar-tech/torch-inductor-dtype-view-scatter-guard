"""Guard against version drift between pyproject.toml and __init__.py.

Regression coverage for a real released bug found in
reboot-safety-check: pyproject.toml's [project] version was bumped for
a release, but __init__.py's __version__ (what `--version` actually
prints, and what get_version()-style user-facing checks read) was left
stale. Ported verbatim (module-name-substituted) to close the same gap
fleet-wide.

Deliberately avoids tomllib/tomli (tomllib is 3.11+ only, and some CI
matrices in this fleet still test py3.9/3.10) by parsing the single
`version = "..."` line under [project] with a plain regex -- no extra
dependency needed for a one-line, well-known-format field.
"""
from __future__ import annotations

import re
from pathlib import Path

from torch_inductor_dtype_view_scatter_guard import __version__

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def test_init_version_matches_pyproject_version():
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject_path.read_text()
    match = _VERSION_RE.search(text)
    assert match, 'Could not find `version = "..."` in pyproject.toml'
    pyproject_version = match.group(1)
    assert __version__ == pyproject_version, (
        f"__init__.py __version__ ({__version__!r}) does not match "
        f"pyproject.toml's [project].version ({pyproject_version!r}). "
        "These must be bumped together on every release."
    )
