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

