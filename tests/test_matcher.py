
"""
tests/test_matcher.py — Unit and integration tests for the matching system.
Run with:  python tests/test_matcher.py
"""
import sys, unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest import CalendarRecord, CRMRecord, _parse_crm_date, _parse_dt_lenient, _repair_email, load_calendar, load_crm, load_labels
from matcher import MATCH_THRESHOLD, _date_score, _location_score, _owner_score, _token_overlap, detect_calendar_duplicates, run_matching
from evaluate import evaluate

DATA_DIR = Path(__file__).parent.parent / "data"

def _make_crm(meeting_dt=None, has_time=True, owner_email="sarah.chen@firma.com", location=None):
    r = CRMRecord(crm_id="CRM-TEST", subject="Test Meeting", client_name="Test Client",
        client_company="Test Co", relationship_owner="Sarah Chen", meeting_dt=meeting_dt,
        meeting_date_raw="2025-03-10", meeting_time_raw="14:00" if has_time else None,
        meeting_type="in-person", location=location, notes="", status="confirmed", created_at=None)
    r.owner_email = owner_email
    return r

def _make_cal(start_dt=None, organizer="sarah.chen@firma.com", location=None):
    r = CalendarRecord(event_id="CAL-TEST", title="Test Meeting", organizer=organizer,
        attendees=[organizer], start_dt=start_dt, end_dt=None, location=location,
        description=None, is_recurring=False, status="confirmed", created_at=None)
    r.attendee_domains = []; r.external_attendees = []
    return r

class TestIngest(unittest.TestCase):
    def test_repair_email(self):
        self.assertEqual(_repair_email("raj.patel[at]atlasvc.com"), "raj.patel@atlasvc.com")
    def test_crm_malformed_date(self):
        dt = _parse_crm_date("03-15/2025", "12:00")
        self.assertIsNotNone(dt); self.assertEqual((dt.month, dt.day, dt.year), (3, 15, 2025))
    def test_lenient_parser_none(self):
        self.assertIsNone(_parse_dt_lenient("not-a-date"))
        self.assertIsNone(_parse_dt_lenient(None))
    def test_load_counts(self):
        self.assertEqual(len(load_crm(DATA_DIR/"crm_events.json")), 20)
        self.assertEqual(len(load_calendar(DATA_DIR/"calendar_events.json")), 22)
    def test_external_attendees_filtered(self):
        records = load_calendar(DATA_DIR/"calendar_events.json")
        sync = next(r for r in records if r.event_id == "CAL-A3")
        self.assertEqual(sync.external_attendees, [])
    def test_email_repaired_in_calendar(self):
        records = load_calendar(DATA_DIR/"calendar_events.json")
        a16 = next(r for r in records if r.event_id == "CAL-A16")
        self.assertTrue(any("raj.patel@atlasvc.com" in a for a in a16.attendees))

class TestScoring(unittest.TestCase):
    def test_token_overlap(self):
        self.assertEqual(_token_overlap("foo bar", "foo bar"), 1.0)
        self.assertEqual(_token_overlap("foo bar", "baz qux"), 0.0)
    def test_date_exact(self):
        self.assertEqual(_date_score(_make_crm(datetime(2025,3,10,14,0)), _make_cal(datetime(2025,3,10,14,0))), 1.0)
    def test_date_wrong_day(self):
        self.assertEqual(_date_score(_make_crm(datetime(2025,3,10,14,0)), _make_cal(datetime(2025,3,11,14,0))), 0.0)
    def test_date_no_time_partial(self):
        score = _date_score(_make_crm(datetime(2025,3,10), has_time=False), _make_cal(datetime(2025,3,10,9,0)))
        self.assertGreater(score, 0.5); self.assertLess(score, 1.0)
    def test_location_both_virtual(self):
        self.assertGreaterEqual(_location_score("Microsoft Teams", "Zoom - https://zoom.us/j/123"), 0.7)
    def test_location_conflict(self):
        self.assertLess(_location_score("NYC Office", "Zoom"), 0.5)
    def test_owner_exact(self):
        self.assertEqual(_owner_score(_make_crm(), _make_cal()), 1.0)
    def test_owner_mismatch(self):
        self.assertEqual(_owner_score(_make_crm(), _make_cal(organizer="james.wu@firma.com")), 0.0)

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crm = load_crm(DATA_DIR/"crm_events.json")
        cls.cal = load_calendar(DATA_DIR/"calendar_events.json")
        cls.results = run_matching(cls.crm, cls.cal)
        cls.labels = load_labels(DATA_DIR/"evaluation_labels.json")
        cls.smap = {(r.crm_id, r.calendar_id): r for r in cls.results}

    def test_labeled_positives_matched(self):
        for p in self.labels["cross_source_pairs"]:
            if p["match"]:
                r = self.smap.get((p["crm_id"], p["calendar_id"]))
                self.assertIsNotNone(r)
                self.assertTrue(r.is_match, f"{p['crm_id']} ↔ {p['calendar_id']} score={r.score if r else '?'}")

    def test_labeled_negatives_rejected(self):
        for p in self.labels["cross_source_pairs"]:
            if not p["match"]:
                r = self.smap.get((p["crm_id"], p["calendar_id"]))
                if r: self.assertFalse(r.is_match, f"{p['crm_id']} ↔ {p['calendar_id']} score={r.score}")

    def test_f1_perfect(self):
        report = evaluate(self.results, self.labels, MATCH_THRESHOLD)
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.fp, 0); self.assertEqual(report.fn, 0)

    def test_calendar_duplicate_detected(self):
        dupes = detect_calendar_duplicates(self.cal)
        ids = {(d.record_a, d.record_b) for d in dupes}
        self.assertTrue(("CAL-A5","CAL-A6") in ids or ("CAL-A6","CAL-A5") in ids)

    def test_cancelled_crm_flagged_in_notes(self):
        r = self.smap.get(("CRM-1009", "CAL-A10"))
        self.assertIsNotNone(r)
        self.assertTrue(any("cancelled" in n.lower() for n in r.notes))

if __name__ == "__main__":
    unittest.main(verbosity=2)

