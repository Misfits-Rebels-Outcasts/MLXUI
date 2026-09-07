#!/usr/bin/env python3
"""CFM-R13-1 — regenerate the `uses:`-checks fixture and its golden.

The fixture exercises all four `uses:` codes in one file (E114 escape/absolute,
E116 no-such-file / parse-error / child-invalid, E117 inherited capability,
E115 two-file cycle). The golden is produced by running `mlxflow check --json`
against the Python — never by transcribing what the Swift emits.

Run from anywhere: the sibling `catflow-mlx` checkout is located via this
file's own path, and every file below is written from the constants here (a
hand-edited fixture file would defeat the point).
"""

import json
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent
GOLDEN = Path(__file__).resolve().parents[2] / "goldens" / "check" / "uses_golden.json"
CATFLOW_MLX = Path(__file__).resolve().parents[4].parent / "catflow-mlx"

FILES = {
    "root.cat": """\
catflow 0.8
1. Read Text   in.txt
2. Escape   ../evil.cat
3. Abs   /etc/hosts
4. Missing   gone.cat
5. Broken   broken.cat
6. Invalid   invalid.cat
7. Lib   lib.cat
8. Cycle   cycle_b.cat
uses:
  Escape = ../evil.cat
  Abs = /etc/hosts
  Missing = gone.cat
  Broken = broken.cat
  Invalid = invalid.cat
  Lib = lib.cat
  Cycle = cycle_b.cat
""",
    "broken.cat": """\
catflow 0.8
this is not a valid row
""",
    "invalid.cat": """\
catflow 0.8
1. Frobnicate   x.txt
""",
    "lib.cat": """\
catflow 0.8; improvise
1. Read Text   lib.txt
""",
    "cycle_a.cat": """\
catflow 0.8
1. Cycle   cycle_b.cat
uses:
  Cycle = cycle_b.cat
""",
    "cycle_b.cat": """\
catflow 0.8
1. Cycle   cycle_a.cat
uses:
  Cycle = cycle_a.cat
""",
}


def main() -> int:
    if not CATFLOW_MLX.is_dir():
        print(f"catflow-mlx not found at {CATFLOW_MLX}", file=sys.stderr)
        return 1

    for name, text in FILES.items():
        (FIXTURE_DIR / name).write_text(text, encoding="utf-8")

    root = FIXTURE_DIR / "root.cat"
    result = subprocess.run(
        ["uv", "run", "mlxflow", "check", str(root), "--json"],
        cwd=CATFLOW_MLX, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):  # 0 = clean, 1 = invalid (both have --json payloads)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 2
    payload = json.loads(result.stdout)
    issues = [
        {"row": i["row"], "code": i["code"], "message": i["message"]}
        for i in payload["issues"]
    ]
    GOLDEN.write_text(json.dumps(issues, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN} ({len(issues)} issues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
