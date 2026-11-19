#!/usr/bin/env python3
"""Entrypoint: gta6-trophy-rarity-forecast-01.

Reads every *.in file in $MAKEFUN_INPUT_DIR (default /input) and writes
one <case-id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output),
each {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Standard library only. Deterministic: files are processed in sorted
order and every forecast comes from the case file alone.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import forecast


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    if not os.path.isdir(input_dir):
        raise SystemExit(f"input dir not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    if not names:
        raise SystemExit(f"no .in files in {input_dir}")

    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast(case)
        # Sanity-check the forecast against the case before writing:
        # one row per trophy, same ids, quantiles non-decreasing.
        want_ids = sorted(t["id"] for t in case["trophies"])
        got_ids = [r["id"] for r in rows]
        if got_ids != want_ids:
            raise SystemExit(f"{case_id}: forecast covers {got_ids}, "
                             f"case needs {want_ids}")
        for r in rows:
            q = r["q"]
            if len(q) != len(case.get("levels") or (0.05, 0.25, 0.5, 0.75,
                                                    0.95)):
                raise SystemExit(f"{case_id}: trophy {r['id']} has "
                                 f"{len(q)} quantiles")
            if any(not (0.0 <= v <= 100.0) for v in q):
                raise SystemExit(f"{case_id}: trophy {r['id']} quantile "
                                 f"out of range 0-100: {q}")
            if any(q[i] > q[i + 1] for i in range(len(q) - 1)):
                raise SystemExit(f"{case_id}: trophy {r['id']} quantiles "
                                 f"not non-decreasing: {q}")
        payload = {"trophies": rows}
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise SystemExit(f"output for {case_id} exceeds 65,536 bytes")
        with open(os.path.join(output_dir, case_id + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(text)
        print(f"{case_id}: {len(rows)} trophies -> "
              f"{os.path.join(output_dir, case_id + '.out')}", flush=True)


if __name__ == "__main__":
    main()
