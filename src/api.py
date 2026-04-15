"""
api.py — Flask REST API serving match predictions.

Endpoints
---------
GET  /matches
     Returns all predicted matches above the configured threshold.
     Query params:
       ?threshold=0.5   override the default threshold
       ?min_score=0.0   return all pairs above this score (for exploration)

GET  /matches/<crm_id>
     Returns all calendar events matched to a specific CRM record.

GET  /matches/<crm_id>/<calendar_id>
     Returns the detailed score breakdown for a specific pair.

GET  /duplicates
     Returns detected intra-source duplicates.

GET  /health
     Liveness check.

Design notes:
  - Data is loaded once at startup; no database.
  - All matching is done in-memory; latency is < 10ms for this data size.
  - In a production system you'd add caching, pagination, and auth.
"""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from ingest import load_calendar, load_crm, load_labels
from matcher import MATCH_THRESHOLD, detect_calendar_duplicates, run_matching, score_pair
from evaluate import evaluate, print_report

DATA_DIR = Path(__file__).parent.parent / "data"

app = Flask(__name__)

# ── Load data at startup ───────────────────────────────────────────────────────

crm_records  = load_crm(DATA_DIR / "crm_events.json")
cal_records  = load_calendar(DATA_DIR / "calendar_events.json")
labels       = load_labels(DATA_DIR / "evaluation_labels.json")

# Build lookup maps
crm_by_id  = {r.crm_id: r  for r in crm_records}
cal_by_id  = {r.event_id: r for r in cal_records}

# Pre-compute all pair scores
all_results = run_matching(crm_records, cal_records)
duplicates  = detect_calendar_duplicates(cal_records)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _result_to_dict(r) -> dict:
    return {
        "crm_id": r.crm_id,
        "calendar_id": r.calendar_id,
        "score": r.score,
        "is_match": r.is_match,
        "feature_scores": r.feature_scores,
        "notes": r.notes,
    }

def _dup_to_dict(d) -> dict:
    return {
        "source": d.source,
        "record_a": d.record_a,
        "record_b": d.record_b,
        "score": d.score,
        "reason": d.reason,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "crm_records": len(crm_records),
                    "calendar_records": len(cal_records)})


@app.get("/matches")
def get_matches():
    threshold = float(request.args.get("threshold", MATCH_THRESHOLD))
    min_score = float(request.args.get("min_score", 0.0))

    filtered = [
        r for r in all_results
        if r.score >= max(threshold, min_score)
    ]
    return jsonify({
        "threshold": threshold,
        "count": len(filtered),
        "matches": [_result_to_dict(r) for r in filtered],
    })


@app.get("/matches/<crm_id>")
def get_matches_for_crm(crm_id: str):
    threshold = float(request.args.get("threshold", MATCH_THRESHOLD))
    crm_id = crm_id.upper()
    if crm_id not in crm_by_id:
        return jsonify({"error": f"CRM record {crm_id!r} not found"}), 404

    matches = [
        r for r in all_results
        if r.crm_id == crm_id and r.score >= threshold
    ]
    return jsonify({
        "crm_id": crm_id,
        "threshold": threshold,
        "count": len(matches),
        "matches": [_result_to_dict(r) for r in matches],
    })


@app.get("/matches/<crm_id>/<calendar_id>")
def get_pair_score(crm_id: str, calendar_id: str):
    crm_id = crm_id.upper()
    calendar_id = calendar_id.upper()
    crm = crm_by_id.get(crm_id)
    cal = cal_by_id.get(calendar_id)
    if not crm:
        return jsonify({"error": f"CRM record {crm_id!r} not found"}), 404
    if not cal:
        return jsonify({"error": f"Calendar event {calendar_id!r} not found"}), 404

    result = score_pair(crm, cal)
    return jsonify(_result_to_dict(result))


@app.get("/duplicates")
def get_duplicates():
    return jsonify({
        "count": len(duplicates),
        "duplicates": [_dup_to_dict(d) for d in duplicates],
    })


@app.get("/evaluate")
def get_evaluation():
    threshold = float(request.args.get("threshold", MATCH_THRESHOLD))
    report = evaluate(all_results, labels, threshold)
    return jsonify({
        "threshold": threshold,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "accuracy": report.accuracy,
        "confusion": {"tp": report.tp, "tn": report.tn,
                      "fp": report.fp, "fn": report.fn},
        "per_pair": [
            {
                "crm_id": p.crm_id, "calendar_id": p.calendar_id,
                "label": p.label, "predicted": p.predicted,
                "score": p.score, "outcome": p.outcome,
            }
            for p in report.per_pair
        ],
        "unlabeled_predicted_matches": [
            _result_to_dict(r) for r in report.unlabeled_matches
        ],
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Print evaluation summary at startup
    from evaluate import print_report
    report = evaluate(all_results, labels, MATCH_THRESHOLD)
    print_report(report)
    app.run(debug=False, port=5000)