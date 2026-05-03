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
check("normal call not protected", not dashboard.is_golden_call_name("normal_real_call_20260503_101010"))

print("Dashboard golden fixture protection tests passed.")
