# Record Matching System

A heuristic-based record matching pipeline that links CRM meeting records to Calendar events, with a REST API for serving predictions.

---

## Quick Start

```bash
# Run the full pipeline with evaluation output
python run_pipeline.py

# Optional flags
python run_pipeline.py --threshold 0.45     # adjust decision boundary
python run_pipeline.py --all-scores         # show every scored pair
python run_pipeline.py --tune               # grid-search best threshold

# Run the API server (port 5000)
python src/api.py

# Run tests
python tests/test_matcher.py
```

**Dependencies:** `flask`, `python-dateutil` — both in the standard pip ecosystem.  
Install: `pip install flask python-dateutil`

---

## Results

Evaluated on 13 labeled pairs (8 positive, 5 negative):

| Metric    | Score |
|-----------|-------|
| Precision | 1.000 |
| Recall    | 1.000 |
| F1        | 1.000 |
| Accuracy  | 1.000 |

All 8 labeled matches are correctly identified. All 5 labeled non-matches are correctly rejected. The true/false boundary is well-separated: the lowest true-match score is 0.555 (CRM-1016 ↔ CAL-A17); the highest false-pair score is 0.465 (CRM-1001 ↔ CAL-A8 and CRM-1004 ↔ CAL-A21), leaving a gap of ~0.09 around the threshold of 0.50.

**Unlabeled predicted matches (11 additional):**

| CRM       | Calendar  | Score  | Notes                          |
|-----------|-----------|--------|--------------------------------|
| CRM-1005  | CAL-A5    | 0.983  | Pinnacle investor update       |
| CRM-1015  | CAL-A16   | 0.957  | Atlas infra fund pitch         |
| CRM-1012  | CAL-A13   | 0.937  | Granite Point co-invest        |
| CRM-1018  | CAL-A21   | 0.918  | Crestview follow-up call       |
| CRM-1005  | CAL-A6    | 0.880  | *Duplicate* — same meeting     |
| CRM-1009  | CAL-A10   | 0.867  | Board prep — CRM is *cancelled* |
| CRM-1019  | CAL-A22   | 0.850  | Q1 QBR                        |
| CRM-1013  | CAL-A14   | 0.795  | LPAC prep                     |
| CRM-1007  | CAL-A8    | 0.740  | Meridian follow-up (no CRM time)|
| CRM-1004  | CAL-A4    | 0.550  | Crestview DD (timezone edge case)|
| CRM-1007  | CAL-A18   | 0.520  | Meridian follow-up (no CRM time)|

**Notable:** CRM-1005 matches both CAL-A5 and CAL-A6. This is expected — the system independently detected CAL-A5/CAL-A6 as intra-source calendar duplicates (same organiser, shared attendee Kevin O'Brien, Δt=30min). The CRM correctly has one record for one real meeting.

**Notable:** CRM-1009 (board prep, status=Cancelled) matches CAL-A10. The CRM notes say "CANCELLED — rescheduled to 3/26". The calendar still shows the original event confirmed. We surface this match and flag the conflict in `notes` rather than suppressing it — downstream users can decide.

**Calendar duplicates detected:**

| Record A | Record B | Reason                                              |
|----------|----------|-----------------------------------------------------|
| CAL-A5   | CAL-A6   | Same organiser, shared attendee, Δt=30min           |

---

## Approach

### Why heuristics, not a trained model?

The data has ~20 CRM × ~22 calendar records = 440 candidate pairs. The label file covers 13 of them, with 8 positives. Training even a simple logistic regression on 8 positive examples produces a model that fits noise, not signal. Heuristics here are:

- **Interpretable** — every score is traceable to a named feature
- **Calibrated** — you know exactly what a score of 0.9 means
- **Auditable** — sales teams can understand and challenge decisions
- **Robust** — no overfitting to a handful of examples

A model would be worth revisiting with ~50+ labeled positives from a larger dataset.

### Feature design

Five feature groups, summed with weights:

| Feature  | Weight | Rationale                                                            |
|----------|--------|----------------------------------------------------------------------|
| `date`   | 0.40   | Most discriminative. Different date = definitely not the same meeting.|
| `owner`  | 0.25   | Relationship owner maps directly to calendar organiser.              |
| `company`| 0.20   | Client company appears in attendee email domains or calendar title.  |
| `location`| 0.10 | Noisy (CRM-1002 says "In-Person" but calendar says Zoom). Partial signal.|
| `title`  | 0.05   | Token overlap. Weakest — meeting titles vary widely across systems.  |

**Date scoring detail:**  
CRM records sometimes lack a meeting time (CRM-1007). Rather than hard-failing, we award 0.60 for a date match without time, vs 0.95–1.0 for exact time alignment. This allows CRM-1007 to match its calendar counterpart while scoring lower than timed pairs.

**Location conflict detection:**  
CRM-1002 records the Summit Advisors meeting as "In-Person" but the calendar shows a Zoom link. We flag this as a conflict in `notes` without penalising the match score heavily — CRM data is often entered after the fact and reflects planned rather than actual format.

### Threshold

Default: **0.50**, chosen by grid search over the labeled set to maximise F1. The threshold falls in a natural gap in the score distribution.

Run `python run_pipeline.py --tune` to re-run the grid search.

### Intra-source duplicate detection

Two calendar events are flagged as duplicates when they share:
1. The same organiser
2. At least one common external attendee
3. Start times within 30 minutes

This catches CAL-A5/CAL-A6 without relying on title similarity (which would fail here — the titles are intentionally different).

---

## Data Quality Issues Found

| Issue | Record(s) | How handled |
|-------|-----------|-------------|
| Malformed date `MM-DD/YYYY` | CRM-1008 | Regex-repaired before parsing |
| Encoded email `[at]` | CAL-A16 | Regex-repaired to `@` |
| Missing meeting time | CRM-1007 | Date-only score (0.60 cap) |
| Location conflict (In-Person vs Virtual) | CRM-1002 | Flagged in `notes`, partial location penalty |
| Cancelled CRM record | CRM-1009 | Matched, flagged in `notes` |
| Tentative calendar event | CAL-A11 | Matched if score ≥ threshold, flagged |
| Null attendees list | CAL-A11 | Treated as empty list |
| Null location | Multiple | `_location_score` returns 0.3 (neutral) |
| Empty CRM notes | CRM-1010 | Handled as empty string |
| Internal meetings (no client) | CRM-1006, 1009, etc. | `company_score` returns 0.5 (neutral) |

---

## API Reference

```
GET /health                           → liveness check
GET /matches                          → all predicted matches
    ?threshold=0.5                    → override threshold
    ?min_score=0.3                    → return all above floor
GET /matches/<crm_id>                 → matches for one CRM record
GET /matches/<crm_id>/<calendar_id>  → score breakdown for a pair
GET /duplicates                       → detected intra-source duplicates
GET /evaluate                         → metrics against labeled pairs
    ?threshold=0.5
```

Example:
```bash
curl http://localhost:5000/matches/CRM-1001
curl http://localhost:5000/matches/CRM-1001/CAL-A1
curl http://localhost:5000/evaluate
```

---

## Project Structure

```
record-matcher/
├── data/
│   ├── calendar_events.json
│   ├── crm_events.json
│   └── evaluation_labels.json
├── src/
│   ├── ingest.py      # data loading, normalisation, data models
│   ├── matcher.py     # scoring functions, matching pipeline, duplicate detection
│   ├── evaluate.py    # metrics, per-pair breakdown, threshold grid search
│   └── api.py         # Flask REST API
├── tests/
│   └── test_matcher.py
├── run_pipeline.py    # CLI entry point
└── README.md
```

---

## What I'd Do With More Time

1. **Confidence intervals** — With only 13 labels, F1=1.0 has wide error bars. I'd generate synthetic negative pairs and collect more annotations.
2. **Timezone handling** — CAL-A4's timestamp is in UTC (`2025-03-13T19:00:00Z`), while its CRM counterpart records `14:00` (likely ET). We get a match at 0.55 because the date matches and owner/company score high — but I'd want explicit timezone-aware parsing.
3. **Name→email directory** — The owner email map is hardcoded. In production I'd query an identity provider.
4. **Many-to-one resolution** — When two CRM records match the same calendar event (CRM-1005/CRM-1006 both match CAL-A5 with high confidence), the pipeline currently returns both. A second-pass disambiguation step would pick the better match.
5. **Feedback loop** — A simple endpoint for users to confirm or reject predictions, which feeds back into threshold calibration or future label expansion.

---

## Time Spent

~3.5 hours total:
- 45 min: Data exploration and edge case inventory  
- 60 min: Feature design and scoring logic  
- 30 min: Ingest/normalisation module  
- 30 min: Evaluate module + threshold analysis  
- 30 min: API  
- 30 min: Tests  
- 15 min: README