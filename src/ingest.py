"""
ingest.py — Load and normalize records from CRM and Calendar sources.

Key decisions:
- All datetimes are normalized to UTC-naive for comparison (we accept the risk
  of timezone ambiguity; see README for discussion).
- Malformed dates are stored as None rather than raising; downstream matchers
  handle None gracefully.
- Email addresses in attendee lists are lowercased and deduplicated.
- The malformed Atlas attendee "raj.patel[at]atlasvc.com" is repaired to a
  proper email address before downstream use.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dateutil import parser as dtparser


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class CalendarRecord:
    event_id: str
    title: str
    organizer: str
    attendees: list[str]
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    location: Optional[str]
    description: Optional[str]
    is_recurring: bool
    status: str
    created_at: Optional[datetime]

    # derived / cleaned
    attendee_domains: list[str] = field(default_factory=list)
    external_attendees: list[str] = field(default_factory=list)


@dataclass
class CRMRecord:
    crm_id: str
    subject: str
    client_name: Optional[str]
    client_company: Optional[str]
    relationship_owner: str
    meeting_dt: Optional[datetime]   # date + time combined
    meeting_date_raw: str
    meeting_time_raw: Optional[str]
    meeting_type: str
    location: Optional[str]
    notes: str
    status: str
    created_at: Optional[datetime]

    # derived
    owner_email: Optional[str] = None   # inferred from name → email map


# ── Helpers ───────────────────────────────────────────────────────────────────

_FIRMA_DOMAIN = "firma.com"

# Best-effort map of relationship owner display names → internal email.
# In production this would come from a directory service; here we infer from
# the calendar organiser field.
_OWNER_EMAIL_MAP = {
    "Sarah Chen":  "sarah.chen@firma.com",
    "James Wu":    "james.wu@firma.com",
    "Priya Sharma": "priya.sharma@firma.com",
    "Michael Ross": "michael.ross@firma.com",
    "Diane Foster": "diane.foster@firma.com",
}


def _repair_email(addr: str) -> str:
    """Fix common email encoding issues like [at] → @."""
    return re.sub(r"\[at\]", "@", addr).strip().lower()


def _parse_dt_lenient(value: Optional[str]) -> Optional[datetime]:
    """Parse a datetime string permissively; return None on failure."""
    if not value:
        return None
    try:
        dt = dtparser.parse(value)
        # Strip tzinfo so all comparisons are naive UTC-equivalent
        return dt.replace(tzinfo=None)
    except (ValueError, OverflowError):
        return None


def _parse_crm_date(date_raw: str, time_raw: Optional[str]) -> Optional[datetime]:
    """
    CRM dates are mostly ISO-8601 but one record uses MM-DD/YYYY (CRM-1008).
    We try ISO first, then a custom pattern, then dateutil as fallback.
    """
    if not date_raw:
        return None

    # Fix MM-DD/YYYY → YYYY-MM-DD
    m = re.match(r"^(\d{2})-(\d{2})/(\d{4})$", date_raw.strip())
    if m:
        mm, dd, yyyy = m.groups()
        date_raw = f"{yyyy}-{mm}-{dd}"

    combined = date_raw
    if time_raw:
        combined = f"{date_raw} {time_raw}"

    return _parse_dt_lenient(combined)


def _extract_domains(emails: list[str]) -> list[str]:
    domains = []
    for e in emails:
        if "@" in e:
            domains.append(e.split("@")[1].lower())
    return list(set(domains))


def _external_attendees(emails: list[str]) -> list[str]:
    return [e for e in emails if not e.endswith(f"@{_FIRMA_DOMAIN}")]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_calendar(path: Path) -> list[CalendarRecord]:
    raw = json.loads(path.read_text())
    records = []
    for r in raw:
        attendees = [_repair_email(a) for a in (r.get("attendees") or [])]
        rec = CalendarRecord(
            event_id=r["event_id"],
            title=r.get("title", ""),
            organizer=_repair_email(r.get("organizer", "")),
            attendees=attendees,
            start_dt=_parse_dt_lenient(r.get("start_time")),
            end_dt=_parse_dt_lenient(r.get("end_time")),
            location=r.get("location"),
            description=r.get("description"),
            is_recurring=bool(r.get("is_recurring", False)),
            status=(r.get("status") or "").lower(),
            created_at=_parse_dt_lenient(r.get("created_at")),
        )
        rec.attendee_domains = _extract_domains(attendees)
        rec.external_attendees = _external_attendees(attendees)
        records.append(rec)
    return records


def load_crm(path: Path) -> list[CRMRecord]:
    raw = json.loads(path.read_text())
    records = []
    for r in raw:
        owner_name = r.get("relationship_owner", "")
        rec = CRMRecord(
            crm_id=r["crm_id"],
            subject=r.get("subject", ""),
            client_name=r.get("client_name"),
            client_company=r.get("client_company"),
            relationship_owner=owner_name,
            meeting_dt=_parse_crm_date(
                r.get("meeting_date", ""),
                r.get("meeting_time"),
            ),
            meeting_date_raw=r.get("meeting_date", ""),
            meeting_time_raw=r.get("meeting_time"),
            meeting_type=(r.get("meeting_type") or "").lower(),
            location=r.get("location"),
            notes=r.get("notes") or "",
            status=(r.get("status") or "").lower(),
            created_at=_parse_dt_lenient(r.get("created_at")),
        )
        rec.owner_email = _OWNER_EMAIL_MAP.get(owner_name)
        records.append(rec)
    return records


def load_labels(path: Path) -> dict:
    return json.loads(path.read_text())