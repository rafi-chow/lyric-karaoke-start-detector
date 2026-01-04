from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Row:
    track_id: str
    expected: float
    predicted: float
    notes: str

    @property
    def abs_error(self) -> float:
        return abs(self.predicted - self.expected)


def _parse_float(value: str, field: str, track_id: str) -> float:
    value = value.strip()
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"{field} has invalid time '{value}' for track_id={track_id}")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(value)


def load_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"track_id", "expected_sec", "predicted_sec", "notes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for i, r in enumerate(reader, start=2):  # header is line 1
            track_id = (r.get("track_id") or "").strip()
            if not track_id:
                raise ValueError(f"Missing track_id on line {i}")

            expected = _parse_float(r.get("expected_sec") or "", "expected_sec", track_id)
            predicted = _parse_float(r.get("predicted_sec") or "", "predicted_sec", track_id)
            notes = (r.get("notes") or "").strip()
            rows.append(Row(track_id=track_id, expected=expected, predicted=predicted, notes=notes))
    return rows


def pct_within(errors: List[float], threshold: float) -> float:
    if not errors:
        return 0.0
    return 100.0 * sum(1 for e in errors if e <= threshold) / len(errors)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate karaoke_start against a hand-labeled set.")
    ap.add_argument("--csv", default="data/eval_handset.csv", help="Path to eval CSV.")
    ap.add_argument("--print-outliers", action="store_true", help="Print worst errors with notes.")
    ap.add_argument("--topk", type=int, default=3, help="How many outliers to print.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    errors = [r.abs_error for r in rows]

    errors_sorted = sorted(errors)
    med = statistics.median(errors_sorted)
    mean = statistics.mean(errors_sorted)

    within_5 = pct_within(errors, 5.0)
    within_10 = pct_within(errors, 10.0)
    within_15 = pct_within(errors, 15.0)

    print(f"Eval file: {csv_path}")
    print(f"N={len(rows)}")
    print(f"Median abs error: {med:.2f}s")
    print(f"Mean abs error:   {mean:.2f}s")
    print(f"% within 5s:  {within_5:.1f}%")
    print(f"% within 10s: {within_10:.1f}%")
    print(f"% within 15s: {within_15:.1f}%")

    if args.print_outliers:
        # Print rows sorted by error descending
        worst = sorted(rows, key=lambda r: r.abs_error, reverse=True)[: max(args.topk, 1)]
        print("\nWorst cases:")
        for r in worst:
            note = f" | notes: {r.notes}" if r.notes else ""
            print(f"- {r.track_id}: expected={r.expected:.1f}s predicted={r.predicted:.1f}s "
                  f"abs_err={r.abs_error:.1f}s{note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
