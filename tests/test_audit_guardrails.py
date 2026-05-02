import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watcher

def check(name, condition, output):
    if not condition:
        raise AssertionError(f"{name} failed:\n{output}")

def run_case(name, report, transcript, must_contain=(), must_not_contain=()):
    out = watcher.enforce_final_audit_consistency(report, transcript)
    out = watcher.enforce_pass_logic(out)
    out = watcher.enforce_risk_for_automatic_fail(out)
    out = watcher.redact_report_text(out)

    for s in must_contain:
        check(f"{name} contains {s!r}", s in out, out)
    for s in must_not_contain:
        check(f"{name} not contains {s!r}", s not in out, out)
    return out

early_disconnect_report = """SCORE: 90
RISK: LOW
PASS: YES
CALL STAGE REACHED: Banking
EARLY END: NO
NOT REACHED:
- Disclosures

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- Product benefits not explained

TASK CHECKLIST:
- Fact Finding / Warm-up: YES
- Health questions completed: NO
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO
- Payment date explained: YES
- Banking/payment setup explained: PARTIAL
- Banking/account information requested or verified 3 times: NOT REACHED
- Routing number requested or verified 3 times: NOT REACHED
- Account verification evidence count: 0
- Routing verification evidence count: 0

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Banking

BIGGEST MISS:
- None
"""

early_disconnect_transcript = "Agent: Are you there? Can you hear me?"

run_case(
    "early disconnect false banking",
    early_disconnect_report,
    early_disconnect_transcript,
    must_contain=[
        "CALL STAGE REACHED: Fact Finding / Warm-up",
        "EARLY END: YES",
        "Payment date explained: NOT REACHED",
        "Banking/payment setup explained: NOT REACHED",
    ],
    must_not_contain=[
        "CALL STAGE REACHED: Banking",
        "SCORE: 0",
    ],
)

callback_report = """SCORE: 88
RISK: LOW
PASS: YES
CALL STAGE REACHED: Fact Finding / Warm-up
EARLY END: YES
NOT REACHED:
- Disclosures

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- None

TASK CHECKLIST:
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

OBJECTIONS DETECTED:
- Prospect requested callback due to being busy

OBJECTION HANDLING:
- Agent agreed to call back later instead of attempting call control or continuing the sale

SALE OUTCOME:
- Policy sold: NO

BIGGEST MISS:
- None
"""

run_case(
    "callback autofail",
    callback_report,
    "Prospect: Can you call me back later? Agent: Okay, I will call you back.",
    must_contain=[
        "RISK: HIGH",
        "PASS: NO",
        "- Callback set: YES",
        "- Objection occurred without proper call control: YES",
        "- Automatic fail triggered: YES",
    ],
)

false_callback_report = callback_report.replace(
    "Prospect requested callback due to being busy",
    "Agent shared that she actually lives over in Indiana"
).replace(
    "Agent agreed to call back later instead of attempting call control or continuing the sale",
    "Agent continued rapport and fact-finding"
)

run_case(
    "false callback rapport story",
    false_callback_report,
    "Agent: I actually live over in the state of Indiana.",
    must_contain=[
        "- Callback set: NO",
    ],
    must_not_contain=[
        "Prospect requested a callback / delay",
        "- Callback set: YES",
    ],
)

redaction_report = """SCORE: 85
SCORING BREAKDOWN:
- Compliance: 85
- Sales Process: 80
- Product Explanation: 75
- Closing: 70
- Communication Quality: 90
COACHING:
- Use the 3 and 1 Method.
OPENAI COST ESTIMATE:
- Input tokens (est): 12345
- Output tokens (est): 678
"""

out = watcher.redact_report_text(redaction_report)
check("score preserved", "SCORE: 85" in out, out)
check("compliance preserved", "- Compliance: 85" in out, out)
check("3 and 1 preserved", "3 and 1 Method" in out, out)
check("no score zero", "SCORE: 0" not in out, out)

print("All audit guardrail tests passed.")

ivr_callback_report = """SCORE: 80
RISK: HIGH
PASS: AT RISK
CALL STAGE REACHED: Cool Down
EARLY END: NO
NOT REACHED:
- None

COMPLIANCE FAILURES: None

TASK CHECKLIST:
- Product benefits explained: YES
- Three options presented: YES
- Application info collected: YES

SEARCHABLE ANSWERS:
- Was the policy sold? YES

AUTOMATIC FAIL CHECKS:
- Callback set: YES
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.

SALE OUTCOME:
- Policy sold: YES

BIGGEST MISS:
- Agent accepted a callback / delay instead of controlling the objection or continuing the live sales attempt.
"""

ivr_callback_transcript = """Carrier: To receive a callback, press [NUMBER].
Carrier: Estimated wait time is [NUMBER] minutes.
Agent: I am calling to verify current coverage.
Prospect: Okay.
"""

run_case(
    "ivr callback prompt is not callback autofail",
    ivr_callback_report,
    ivr_callback_transcript,
    must_contain=[
        "- Callback set: NO",
        "- Objection occurred without proper call control: NO",
        "- Automatic fail triggered: NO",
        "- Reason: None",
    ],
    must_not_contain=[
        "Prospect requested a callback / delay",
        "Agent accepted a callback / delay",
    ],
)
