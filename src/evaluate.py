"""
evaluate.py — Measure system performance against labeled pairs.

Metrics computed:
  Precision, Recall, F1 (standard binary classification)
  Accuracy on the labeled set
  Per-pair breakdown for interpretability

Note on coverage: the labels cover only 13 cross-source pairs (8 positive,
5 negative) out of 440 possible. Results are indicative, not statistically
definitive. We also report unlabeled predicted matches so reviewers can spot
obvious errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from matcher import MatchResult


@dataclass
class PairEval:
    crm_id: str
    calendar_id: str
    label: bool
    predicted: bool
    score: float
    outcome: str   # TP, TN, FP, FN


@dataclass
class EvalReport:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    per_pair: list[PairEval]
    unlabeled_matches: list[MatchResult]
    threshold_used: float


def evaluate(
    results: list[MatchResult],
    labels: dict,
    threshold: float,
) -> EvalReport:
    """
    Compare predictions against ground-truth labels.

    Args:
        results:   All scored pairs (from run_matching, before threshold filter)
        labels:    Parsed evaluation_labels.json
        threshold: Decision boundary used
    """

    # Build lookup: (crm_id, cal_id) → score and prediction
    score_map: dict[tuple, MatchResult] = {
        (r.crm_id, r.calendar_id): r for r in results
    }

    labeled_keys: set[tuple] = set()
    per_pair: list[PairEval] = []

    for pair in labels.get("cross_source_pairs", []):
        crm_id = pair["crm_id"]
        cal_id = pair["calendar_id"]
        label  = pair["match"]
        key    = (crm_id, cal_id)
        labeled_keys.add(key)

        match_result = score_map.get(key)
        score     = match_result.score if match_result else 0.0
        predicted = score >= threshold

        if label and predicted:
            outcome = "TP"
        elif not label and not predicted:
            outcome = "TN"
        elif not label and predicted:
            outcome = "FP"
        else:
            outcome = "FN"

        per_pair.append(PairEval(
            crm_id=crm_id,
            calendar_id=cal_id,
            label=label,
            predicted=predicted,
            score=round(score, 4),
            outcome=outcome,
        ))

    tp = sum(1 for p in per_pair if p.outcome == "TP")
    tn = sum(1 for p in per_pair if p.outcome == "TN")
    fp = sum(1 for p in per_pair if p.outcome == "FP")
    fn = sum(1 for p in per_pair if p.outcome == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(per_pair) if per_pair else 0.0

    # Predicted matches that weren't in the label set
    unlabeled_matches = [
        r for r in results
        if r.is_match and (r.crm_id, r.calendar_id) not in labeled_keys
    ]

    return EvalReport(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        tp=tp, tn=tn, fp=fp, fn=fn,
        per_pair=per_pair,
        unlabeled_matches=unlabeled_matches,
        threshold_used=threshold,
    )


def find_best_threshold(
    results: list[MatchResult],
    labels: dict,
    candidates: Optional[list[float]] = None,
) -> tuple[float, EvalReport]:
    """
    Grid-search over threshold values to maximise F1 on labeled pairs.
    """
    if candidates is None:
        candidates = [i / 20 for i in range(1, 20)]  # 0.05 … 0.95

    best_f1 = -1.0
    best_t  = 0.50
    best_report = None

    for t in candidates:
        # Temporarily set is_match per threshold
        for r in results:
            r.is_match = r.score >= t
        report = evaluate(results, labels, t)
        if report.f1 > best_f1:
            best_f1    = report.f1
            best_t     = t
            best_report = report

    # Restore default threshold
    from matcher import MATCH_THRESHOLD
    for r in results:
        r.is_match = r.score >= MATCH_THRESHOLD

    return best_t, best_report


def print_report(report: EvalReport) -> None:
    print("\n" + "="*60)
    print(f"  EVALUATION REPORT  (threshold={report.threshold_used})")
    print("="*60)
    print(f"  Precision : {report.precision:.3f}")
    print(f"  Recall    : {report.recall:.3f}")
    print(f"  F1        : {report.f1:.3f}")
    print(f"  Accuracy  : {report.accuracy:.3f}")
    print(f"  TP={report.tp}  TN={report.tn}  FP={report.fp}  FN={report.fn}")
    print()
    print("  Per-pair breakdown:")
    print(f"  {'CRM':12s} {'CAL':10s} {'Label':6s} {'Pred':6s} {'Score':7s}  Outcome")
    print("  " + "-"*58)
    for p in sorted(report.per_pair, key=lambda x: x.outcome):
        mark = "✓" if p.outcome in ("TP", "TN") else "✗"
        print(f"  {p.crm_id:12s} {p.calendar_id:10s} "
              f"{'T' if p.label else 'F':6s} "
              f"{'T' if p.predicted else 'F':6s} "
              f"{p.score:7.4f}  {p.outcome} {mark}")

    if report.unlabeled_matches:
        print(f"\n  Unlabeled predicted matches ({len(report.unlabeled_matches)}):")
        for r in report.unlabeled_matches:
            print(f"    {r.crm_id} ↔ {r.calendar_id}  score={r.score:.4f}")

    print("="*60 + "\n")