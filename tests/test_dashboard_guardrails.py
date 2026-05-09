import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard


def check(name, condition):
    if not condition:
        raise AssertionError(f"{name} failed")


goldens = dashboard.golden_call_names()

check("golden fixture count", len(goldens) >= 18)
check("known golden protected", dashboard.is_golden_call_name("call_back_set_20260503_114315"))
check("known golden prefix protected", dashboard.is_golden_call_name("call_back_set_20260503_114315_extra"))
check("sold_2 golden protected", dashboard.is_golden_call_name("sold_2"))
check("prospect requested DNC golden protected", dashboard.is_golden_call_name("prospect_asked_for_dnc"))
check("completed-sale callback golden protected", dashboard.is_golden_call_name("callback_accepted_sold"))
check("agent-offered DNC golden protected", dashboard.is_golden_call_name("u90_1_20260508_200221"))
check("repeated-objection good call control golden protected", dashboard.is_golden_call_name("good_call_control"))
check("normal call not protected", not dashboard.is_golden_call_name("normal_real_call_20260503_101010"))

print("Dashboard golden fixture protection tests passed.")

# Golden calls should also be hidden from processing/recent-upload cards.
sample_processing = [
    {"call_name": "good_call_control", "filename": "good_call_control.txt"},
    {"call_name": "sold_2", "filename": "sold_2.txt"},
    {"call_name": "normal_visible_call", "filename": "normal_visible_call.txt"},
]
filtered = [
    item
    for item in sample_processing
    if not dashboard.is_golden_call_name(item.get("call_name", ""))
]
check(
    "golden processing cards hidden",
    filtered == [{"call_name": "normal_visible_call", "filename": "normal_visible_call.txt"}],
)

# Bulk delete support should exist and still protect golden calls.
check("bulk delete route exists", "delete_selected_calls" in dir(dashboard))
check("shared delete helper exists", "delete_call_artifacts_by_id" in dir(dashboard))
check("bulk delete keeps good_call_control protected", dashboard.is_golden_call_name("good_call_control"))

check("sold_clean_call test fixture protected", dashboard.is_protected_call_name("sold_clean_call"))
check("u90_no_call_control test fixture protected", dashboard.is_protected_call_name("u90_no_call_control"))
fixture_names = dashboard.test_fixture_call_names()
check("auto fixture discovery includes sold_clean_call", "sold_clean_call" in fixture_names)
check("auto fixture discovery includes lcr_cancer", "lcr_cancer" in fixture_names)
check("auto fixture discovery includes health poor call control", "health_questions_poor_call_control" in fixture_names)
check("auto fixture discovery protects all fixtures", all(dashboard.is_protected_call_name(name) for name in fixture_names))
check("normal call still not protected by fixture discovery", not dashboard.is_protected_call_name("normal_visible_call"))

# Dashboard call-row helpers should work with current sqlite tuple rows.
sample_call_tuple = (
    123,
    "sample_call",
    "db transcript",
    "db report",
    91,
    "LOW",
    "2026-05-09 12:00:00",
    "SOLD",
    None,
    "SOLD",
    "Auto sold.",
    456,
)

check("tuple call id helper", dashboard.call_row_id(sample_call_tuple) == 123)
check("tuple call name helper", dashboard.call_row_name(sample_call_tuple) == "sample_call")
check("tuple call report helper", dashboard.call_row_report(sample_call_tuple) == "db report")
check("tuple call score helper", dashboard.call_row_score(sample_call_tuple) == 91)
check("tuple call timestamp helper", dashboard.call_row_timestamp(sample_call_tuple) == "2026-05-09 12:00:00")

# The generic helper should also support mapping-style rows for future sqlite.Row usage.
sample_call_mapping = {
    "id": 456,
    "call_name": "mapping_call",
    "report": "mapping report",
    "score": 88,
    "timestamp": "2026-05-09 13:00:00",
}

check("mapping call id helper", dashboard.call_row_id(sample_call_mapping) == 456)
check("mapping call name helper", dashboard.call_row_name(sample_call_mapping) == "mapping_call")
check("mapping call report helper", dashboard.call_row_report(sample_call_mapping) == "mapping report")
check("mapping call score helper", dashboard.call_row_score(sample_call_mapping) == 88)
check("mapping call timestamp helper", dashboard.call_row_timestamp(sample_call_mapping) == "2026-05-09 13:00:00")

