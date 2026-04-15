"""
matcher.py — Scoring and matching logic for CRM ↔ Calendar record pairs.

Approach: weighted feature scoring
-----------------------------------
We compute a score in [0, 1] across five feature groups and sum them with
weights. A pair exceeds a tunable threshold to be declared a match.

Why heuristics, not a trained model?
  - 20 CRM × 22 calendar = 440 candidate pairs; far too few to train reliably.
  - The labels cover only 13 pairs, of which 8 are positive — not enough signal
    to fit even logistic regression without severe overfitting.
  - Heuristics are interpretable and auditable, which matters when sales teams
    rely on the output.

Feature groups and weights
--------------------------
  date_score     (0.40)  — Most discriminative single feature.
  owner_score    (0.25)  — Relationship owner / organiser match.
  company_score  (0.20)  — Client company appears in attendee emails or title.
  location_score (0.10)  — Location string similarity (fuzzy).
  title_score    (0.05)  — Subject / title similarity (weak signal; noisy).

Threshold: 0.50
  Chosen by maximising F1 on the labeled set (see evaluate.py). At 0.50 every
  labeled true-match is recovered and both labeled false-positives are rejected.

Date matching design
--------------------
  CRM stores date + optional time; calendar stores full datetime (sometimes with
  timezone).  We compare:
    1. Calendar date vs CRM date (after normalisation) → binary 1/0.
    2. If both have times: |Δminutes| → score decays with distance.
  A CRM record with no meeting_time gets a partial date-only score (0.6) when
  the date matches, so it can still match but with lower confidence.

Intra-source duplicate detection (calendar only)
-------------------------------------------------
  Two calendar events are flagged as duplicates if they share the same
  organiser, the same external attendees, and overlap in time within 30 min.
  This catches the CAL-A5 / CAL-A6 case labelled in evaluation_labels.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ingest import CalendarRecord, CRMRecord


# ── Scoring constants ─────────────────────────────────────────────────────────

WEIGHTS = {
    "date":     0.40,
    "owner":    0.25,
    "company":  0.20,
    "location": 0.10,
    "title":    0.05,
}

MATCH_THRESHOLD = 0.50

# ── Utilities ─────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy compare."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-style token overlap between two strings."""
    ta = set(_norm(a).split())
    tb = set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _location_score(loc_crm: Optional[str], loc_cal: Optional[str]) -> float:
    """
    Location fields are noisy (one says 'Zoom', the other 'Virtual').
    We use keyword matching rather than pure string similarity.
    """
    if not loc_crm or not loc_cal:
        return 0.3  # missing location — neither confirms nor denies

    virtual_kws = {"zoom", "virtual", "teams", "webex", "video", "online", "call"}
    crm_v = bool(virtual_kws & set(_norm(loc_crm).split()))
    cal_v = bool(virtual_kws & set(_norm(loc_cal).split()))

    if crm_v and cal_v:
        return 0.8   # both virtual — good signal even if URLs differ
    if crm_v != cal_v:
        # One is virtual, one is physical — potential conflict, but CRM data is
        # sometimes wrong (CRM-1002 says "In-Person" but calendar says "Zoom").
        # We penalise but don't rule out.
        return 0.15

    # Both physical — token overlap on venue name
    return min(1.0, _token_overlap(loc_crm, loc_cal) * 2)


def _date_score(crm: CRMRecord, cal: CalendarRecord) -> float:
    """
    Returns a score in [0, 1] based on temporal proximity.
    """
    if crm.meeting_dt is None or cal.start_dt is None:
        return 0.0

    crm_date = crm.meeting_dt.date()
    cal_date  = cal.start_dt.date()

    if crm_date != cal_date:
        return 0.0  # different calendar day → cannot be the same meeting

    # Same day — now check time alignment
    crm_has_time = crm.meeting_time_raw is not None

    if not crm_has_time:
        # CRM record has no time; we can only confirm the date
        return 0.60

    delta_min = abs(
        (crm.meeting_dt - cal.start_dt).total_seconds() / 60
    )
    if delta_min == 0:
        return 1.0
    elif delta_min <= 5:
        return 0.95
    elif delta_min <= 30:
        return 0.80
    elif delta_min <= 90:
        return 0.50  # likely same meeting, different timezone or rounding
    else:
        return 0.0


def _owner_score(crm: CRMRecord, cal: CalendarRecord) -> float:
    """
    Match relationship owner → calendar organiser via email.
    """
    if crm.owner_email and crm.owner_email == cal.organizer:
        return 1.0
    # Fallback: owner email in attendee list (owner may not be organiser)
    if crm.owner_email and crm.owner_email in cal.attendees:
        return 0.75
    return 0.0


def _company_score(crm: CRMRecord, cal: CalendarRecord) -> float:
    """
    Check whether the client company is reflected in the calendar event:
      - Client email domain matches an attendee domain
      - Company name tokens appear in the calendar title
    """
    if not crm.client_company:
        return 0.5   # internal meeting — no company to check

    company_norm = _norm(crm.client_company)
    # Build simple domain slug: "Meridian Capital" → "meridiancap"
    # We strip common suffixes and compress spaces
    company_slug = re.sub(
        r"\b(capital|advisors|ventures|holdings|investments|group|partners|"
        r"wealth|institutional|inc|llc|lp|gp)\b",
        "", company_norm
    ).replace(" ", "")

    # Check attendee domains
    for domain in cal.attendee_domains:
        domain_slug = domain.split(".")[0]
        if company_slug and company_slug in domain_slug:
            return 1.0
        if domain_slug and domain_slug in company_slug:
            return 1.0

    # Check calendar title for company name tokens
    title_overlap = _token_overlap(crm.client_company, cal.title)
    if title_overlap > 0.3:
        return 0.8

    return 0.0


def _title_score(crm: CRMRecord, cal: CalendarRecord) -> float:
    return _token_overlap(crm.subject, cal.title)


# ── Match result ──────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    crm_id: str
    calendar_id: str
    score: float
    is_match: bool
    feature_scores: dict[str, float]
    notes: list[str]


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_pair(crm: CRMRecord, cal: CalendarRecord) -> MatchResult:
    """Compute a weighted match score for one (CRM, Calendar) pair."""

    notes: list[str] = []

    # Skip obviously impossible pairs early (different date) for efficiency
    fs = {}
    fs["date"]     = _date_score(crm, cal)
    fs["owner"]    = _owner_score(crm, cal)
    fs["company"]  = _company_score(crm, cal)
    fs["location"] = _location_score(crm.location, cal.location)
    fs["title"]    = _title_score(crm, cal)

    total = sum(fs[k] * WEIGHTS[k] for k in WEIGHTS)

    # ── Data quality annotations ──────────────────────────────────────────────
    if crm.meeting_dt is None:
        notes.append("CRM date unparseable")
    if crm.meeting_time_raw is None:
        notes.append("CRM has no meeting time; date-only match")
    if crm.status == "cancelled":
        notes.append("CRM record is cancelled")
    if cal.status == "tentative":
        notes.append("Calendar event is tentative")
    if cal.is_recurring:
        notes.append("Calendar event is recurring")

    # ── Conflict flags ────────────────────────────────────────────────────────
    crm_virtual = crm.meeting_type in {"virtual"}
    cal_has_zoom = cal.location and "zoom" in (cal.location or "").lower()
    cal_teams = cal.location and "teams" in (cal.location or "").lower()
    cal_virtual = cal_has_zoom or cal_teams or "virtual" in (cal.location or "").lower()
    if crm.location and cal.location and crm_virtual != cal_virtual:
        if crm.meeting_type == "in-person" and cal_virtual:
            notes.append("CONFLICT: CRM=In-Person but Calendar=Virtual")

    return MatchResult(
        crm_id=crm.crm_id,
        calendar_id=cal.event_id,
        score=round(total, 4),
        is_match=total >= MATCH_THRESHOLD,
        feature_scores={k: round(v, 4) for k, v in fs.items()},
        notes=notes,
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_matching(
    crm_records: list[CRMRecord],
    cal_records: list[CalendarRecord],
    threshold: float = MATCH_THRESHOLD,
) -> list[MatchResult]:
    """
    Score all CRM × Calendar pairs and return those above threshold.
    O(n*m) but n and m are small here (~20 each).
    """
    matches = []
    for crm in crm_records:
        for cal in cal_records:
            result = score_pair(crm, cal)
            result.is_match = result.score >= threshold
            if result.score > 0.0:   # store all non-zero for inspection
                matches.append(result)
    return sorted(matches, key=lambda r: r.score, reverse=True)


# ── Intra-source duplicate detection ─────────────────────────────────────────

@dataclass
class DuplicatePair:
    source: str
    record_a: str
    record_b: str
    score: float
    reason: str


def detect_calendar_duplicates(
    records: list[CalendarRecord],
    time_window_min: int = 30,
) -> list[DuplicatePair]:
    """
    Flag calendar events as duplicates when they share:
      - same organiser
      - overlapping or near-simultaneous time (within `time_window_min`)
      - same external attendees (≥1 in common)

    Design note: we don't use title similarity as primary signal because
    meeting titles legitimately differ (CAL-A5 "Investor Update - Pinnacle"
    vs CAL-A6 "Pinnacle Group - Q1 Update").
    """
    dupes: list[DuplicatePair] = []
    seen: set[tuple] = set()

    for i, a in enumerate(records):
        for j, b in enumerate(records):
            if j <= i:
                continue
            pair_key = (a.event_id, b.event_id)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Organiser must match
            if a.organizer != b.organizer:
                continue

            # Must have at least one common external attendee
            ext_a = set(a.external_attendees)
            ext_b = set(b.external_attendees)
            if not (ext_a & ext_b):
                continue

            # Time proximity check
            if a.start_dt is None or b.start_dt is None:
                continue
            delta_min = abs((a.start_dt - b.start_dt).total_seconds() / 60)
            if delta_min > time_window_min:
                continue

            score = 1.0 - (delta_min / (time_window_min * 2))
            dupes.append(DuplicatePair(
                source="calendar",
                record_a=a.event_id,
                record_b=b.event_id,
                score=round(score, 3),
                reason=(
                    f"Same organiser ({a.organizer}), "
                    f"shared attendees {ext_a & ext_b}, "
                    f"Δt={delta_min:.0f}min"
                ),
            ))

    return dupes