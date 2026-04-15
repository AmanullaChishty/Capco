#!/usr/bin/env python3
"""
run_pipeline.py — Run the full matching pipeline and print results.

Usage:
    python run_pipeline.py
    python run_pipeline.py --threshold 0.45
    python run_pipeline.py --all-scores   # show all non-zero pairs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import load_calendar, load_crm, load_labels
from matcher import MATCH_THRESHOLD, detect_calendar_duplicates, run_matching
from evaluate import evaluate, find_best_threshold, print_report

DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Record Matching Pipeline")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help=f"Match score threshold (default: {MATCH_THRESHOLD})")
    parser.add_argument("--all-scores", action="store_true",
                        help="Print all non-zero pair scores, not just matches")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search best threshold on labeled pairs")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    crm_records = load_crm(DATA_DIR / "crm_events.json")
    cal_records = load_calendar(DATA_DIR / "calendar_events.json")
    labels      = load_labels(DATA_DIR / "evaluation_labels.json")
    print(f"  CRM records:      {len(crm_records)}")
    print(f"  Calendar records: {len(cal_records)}")
    print(f"  Labeled pairs:    {len(labels['cross_source_pairs'])}")

    # ── Match ─────────────────────────────────────────────────────────────────
    print("\nRunning matching pipeline...")
    all_results = run_matching(crm_records, cal_records)

    # ── Threshold tuning ──────────────────────────────────────────────────────
    if args.tune:
        best_t, best_report = find_best_threshold(all_results, labels)
        print(f"\nBest threshold by F1: {best_t}")
        print_report(best_report)
        return

    # ── Evaluation ────────────────────────────────────────────────────────────
    report = evaluate(all_results, labels, args.threshold)
    print_report(report)

    # ── Predicted matches ─────────────────────────────────────────────────────
    matches = [r for r in all_results if r.score >= args.threshold]
    print(f"Predicted matches (threshold={args.threshold}): {len(matches)}")
    print()
    print(f"  {'CRM':12s} {'Calendar':12s} {'Score':7s}  Notes")
    print("  " + "-"*60)
    for r in matches:
        note = r.notes[0] if r.notes else ""
        print(f"  {r.crm_id:12s} {r.calendar_id:12s} {r.score:7.4f}  {note}")

    # ── Intra-source duplicates ───────────────────────────────────────────────
    dupes = detect_calendar_duplicates(cal_records)
    print(f"\nDetected calendar duplicates: {len(dupes)}")
    for d in dupes:
        print(f"  {d.record_a} ↔ {d.record_b}  score={d.score:.3f}")
        print(f"    Reason: {d.reason}")

    # ── All scores (optional) ─────────────────────────────────────────────────
    if args.all_scores:
        print(f"\nAll scored pairs ({len(all_results)} non-zero):")
        print(f"  {'CRM':12s} {'Calendar':12s} {'Score':7s}  {'date':6s}  {'owner':6s}  {'company':7s}  {'loc':5s}  {'title':5s}")
        print("  " + "-"*75)
        for r in all_results:
            fs = r.feature_scores
            print(
                f"  {r.crm_id:12s} {r.calendar_id:12s} {r.score:7.4f}"
                f"  {fs['date']:6.3f}  {fs['owner']:6.3f}  {fs['company']:7.3f}"
                f"  {fs['location']:5.3f}  {fs['title']:5.3f}"
                + ("  ← MATCH" if r.is_match else "")
            )


if __name__ == "__main__":
    main()