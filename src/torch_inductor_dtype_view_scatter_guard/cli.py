"""Command-line interface: run the from-scratch diagnosis of the
torch.compile (Inductor) dtype-view + diagonal_scatter value/aliasing
corruption bug (pytorch/pytorch#197408) against the currently
installed torch build, using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-inductor-dtype-view-scatter-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "Inductor backend silently corrupts VALUES (NaN) and the "
            "return-aliasing contract for a dtype-view + custom-op + "
            "diagonal_scatter call pattern (pytorch/pytorch#197408) -- "
            "and verify safe_compiled_dtype_view_diagonal_scatter() "
            "restores eager's values and non-aliasing contract. Never "
            "trusts a cached or previously-reported result, always "
            "re-runs the repro on THIS host's actual installed torch "
            "version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-inductor-dtype-view-scatter-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_value_bug"] or report["any_native_alias_bug"]:
        print(status_headline(style, "fail", "Inductor dtype-view/diagonal_scatter value+aliasing bug reproduced on this host (pytorch#197408)"))
    else:
        print(status_headline(style, "info", "no Inductor dtype-view/diagonal_scatter divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_dtype_view_diagonal_scatter() restores eager's values and non-aliasing contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's values/aliasing contract on at least one case"))

    section("cases (n, data, diag -> eager/compiled/guarded value+aliasing behavior)")
    for c in report["cases"]:
        native_flag = (
            "WRONG-VALUES+ALIASED"
            if (not c["compiled_values_match_eager"]) and c["compiled_aliases_input"]
            else "WRONG-VALUES" if not c["compiled_values_match_eager"]
            else "aliased" if c["compiled_aliases_input"]
            else "ok"
        )
        guard_flag = "guard-ok" if (
            c["guarded_values_match_eager"]
            and c["guarded_aliases_input_matches_eager"]
            and c["guarded_input_matches_eager_after_output_mutation"]
        ) else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"n={int(len(c['diag']))}",
                    f"eager_aliases={c['eager_aliases_input']}  "
                    f"native={native_flag:22s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
