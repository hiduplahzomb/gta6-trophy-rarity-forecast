#!/usr/bin/env python3
"""Entrypoint for the gta6-trophy-rarity-forecast entry.

Reads every *.in case from $MAKEFUN_INPUT_DIR (default /input),
forecasts each trophy's earn rate at the case's target, and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output) as one row per
trophy: {"id": n, "q": [q05, q25, q50, q75, q95]}.

Each forecast is validated against its case before anything is
written: the ids must match the case's trophies exactly, the five
quantiles must be non-decreasing and inside (0, 100], and the file
must fit in 65,536 bytes. A malformed case fails fast instead of
producing a bad output.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import forecast  # noqa: E402

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
MAX_OUTPUT_BYTES = 65536


EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def load_case(path):
    """Load one case file, failing fast on malformed input."""
    with open(path) as handle:
        case = json.load(handle)
    if not isinstance(case, dict):
        raise ValueError(f"{path}: case is not a JSON object")
    if not case.get("id"):
        raise ValueError(f"{path}: case has no id")
    trophies = case.get("trophies")
    if not isinstance(trophies, list) or not trophies:
        raise ValueError(f"{path}: case has no trophies")
    for trophy in trophies:
        if not isinstance(trophy, dict) or "id" not in trophy:
            raise ValueError(f"{path}: trophy without an id")
    if not case.get("target"):
        raise ValueError(f"{path}: case has no target")
    if not case.get("at"):
        raise ValueError(f"{path}: case has no checkpoint time")
    levels = case.get("levels")
    if levels is not None and list(levels) != EXPECTED_LEVELS:
        raise ValueError(f"{path}: levels {levels} are not the five "
                         f"levels this entry forecasts {EXPECTED_LEVELS}")
    return case


def check_rows(case, rows):
    """Validate forecast rows against the case before writing."""
    expect_ids = sorted(t["id"] for t in case["trophies"])
    ids = [r["id"] for r in rows]
    if ids != expect_ids:
        raise ValueError(f"{case['id']}: trophy ids mismatch "
                         f"({len(ids)} rows, expected {len(expect_ids)})")
    for row in rows:
        q = row["q"]
        if len(q) != 5:
            raise ValueError(f"{case['id']} id {row['id']}: "
                             f"{len(q)} quantiles, expected 5")
        if not all(isinstance(v, (int, float)) and 0 < v <= 100
                   for v in q):
            raise ValueError(f"{case['id']} id {row['id']}: "
                             f"quantiles out of (0, 100]: {q}")
        if any(q[i] > q[i + 1] for i in range(4)):
            raise ValueError(f"{case['id']} id {row['id']}: "
                             f"quantiles not non-decreasing: {q}")


def main():
    inputs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.in")))
    if not inputs:
        raise SystemExit(f"no .in files in {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in inputs:
        case = load_case(path)
        rows = forecast(case)
        check_rows(case, rows)
        payload = json.dumps({"trophies": rows}, separators=(",", ":"))
        if len(payload.encode()) > MAX_OUTPUT_BYTES:
            raise ValueError(f"{case['id']}: output "
                             f"{len(payload.encode())} bytes > "
                             f"{MAX_OUTPUT_BYTES}")
        out_path = os.path.join(OUTPUT_DIR, case["id"] + ".out")
        with open(out_path, "w") as handle:
            handle.write(payload)
        print(f"{case['id']}: {len(rows)} trophies -> {out_path}")


if __name__ == "__main__":
    main()
