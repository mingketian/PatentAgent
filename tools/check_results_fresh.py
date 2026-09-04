"""Verify that results/tables/ still matches the underlying evaluation data.

Regenerates the tables into a temporary directory and compares them cell by cell.
Numeric cells are compared with a small tolerance so that a different pandas or
Python version formatting `1.0` as `1.000` is not reported as stale data.

Run with:  python tools/check_results_fresh.py
Exits non-zero if any committed table disagrees with the source data.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
TOLERANCE = 1e-6


def read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle)]


def cells_agree(committed: str, regenerated: str) -> bool:
    if committed == regenerated:
        return True
    try:
        return abs(float(committed) - float(regenerated)) <= TOLERANCE
    except ValueError:
        return False


def compare(committed_path: Path, regenerated_path: Path) -> list[str]:
    problems: list[str] = []
    committed = read_rows(committed_path)
    regenerated = read_rows(regenerated_path)

    if len(committed) != len(regenerated):
        return [f"{committed_path.name}: {len(committed)} rows committed, "
                f"{len(regenerated)} regenerated"]

    for line, (row_a, row_b) in enumerate(zip(committed, regenerated), start=1):
        if len(row_a) != len(row_b):
            problems.append(f"{committed_path.name}:{line}: column count differs")
            continue
        for column, (cell_a, cell_b) in enumerate(zip(row_a, row_b), start=1):
            if not cells_agree(cell_a, cell_b):
                problems.append(
                    f"{committed_path.name}:{line} col {column}: "
                    f"committed {cell_a!r}, regenerated {cell_b!r}"
                )
    return problems


def main() -> int:
    if not TABLES.exists():
        print("results/tables/ does not exist; run tools/make_results_tables.py", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    import tools.make_results_tables as generator  # noqa: PLC0415

    committed_names = sorted(path.name for path in TABLES.glob("*.csv"))
    backup = Path(tempfile.mkdtemp(prefix="patentagent-tables-"))

    try:
        for name in committed_names:
            shutil.copy2(TABLES / name, backup / name)

        generator.main()

        problems: list[str] = []
        regenerated_names = sorted(path.name for path in TABLES.glob("*.csv"))
        missing = set(committed_names) - set(regenerated_names)
        extra = set(regenerated_names) - set(committed_names)
        problems += [f"{name}: committed but no longer generated" for name in sorted(missing)]
        problems += [f"{name}: generated but not committed" for name in sorted(extra)]

        for name in sorted(set(committed_names) & set(regenerated_names)):
            problems += compare(backup / name, TABLES / name)

        # leave the working tree exactly as it was found
        for name in committed_names:
            shutil.copy2(backup / name, TABLES / name)
    finally:
        shutil.rmtree(backup, ignore_errors=True)

    if problems:
        print("Committed result tables are stale. Run 'make figures' and commit:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"All {len(committed_names)} result tables match the source data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
