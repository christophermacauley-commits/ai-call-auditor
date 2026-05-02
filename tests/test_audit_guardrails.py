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

self_disclosure_labeled_wrong = """Agent: Tell me about your family.

Prospect: I can imagine.
It's funny that you say that they have a good relationship because that's kind of how me and my half-brother are.
The one that moved out here with me and my mom and dad when we left Chicago, we've always been kind of raised around each other.
He's older than me by five years, so me and him have a good relationship.
But me and my other four brothers, I think my oldest brother, I've never even met him but three times.
Then I have another brother that I've probably never met before and I don't ever see or talk to him.
Me and my other sister, we do not get along at all.
But once you do have your own family, it's a lot easier to not focus on the negative things that you have in the relationship with your siblings.
If you know what I mean, that makes a big difference.
"""

fixed_self_disclosure = watcher._repair_agent_self_disclosure_mislabeled_as_prospect(self_disclosure_labeled_wrong)

check(
    "agent self-disclosure relabeled",
    "Agent: I can imagine." in fixed_self_disclosure,
    fixed_self_disclosure,
)

check(
    "agent self-disclosure no prospect label",
    "Prospect: I can imagine." not in fixed_self_disclosure,
    fixed_self_disclosure,
)

print("Speaker-label self-disclosure test passed.")

late_stage_should_not_downgrade_to_who_i_am = """SCORE: 90
RISK: HIGH
PASS: NO
CALL STAGE REACHED: Banking
EARLY END: YES
NOT REACHED:
- Existing coverage
- Beneficiary
- Need amount
- Health questions
- Product benefits
- Three options
- Client choice
- Application information
- Payment date
- Banking/payment setup
- Banking/account verification
- Disclosures
- Third Party Underwriting
- Peace of Mind
- Cool Down

COMPLIANCE FAILURES:
- None

TASK CHECKLIST:
- Agent introduction: YES
- Fact Finding / Warm-up: YES
- Beneficiary identified: NO
- Need amount discussed: NO
- Health questions completed: NO
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO
- Payment date explained: NO
- Banking/payment setup explained: NO
- Banking/account information requested or verified 3 times: NOT REACHED
- Routing number requested or verified 3 times: NOT REACHED
- Account verification evidence count: 0
- Routing verification evidence count: 0

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Early refusal call: no calm call control attempt

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Banking

BIGGEST MISS:
- Failure to attempt calm call control.
"""

late_stage_transcript = """Agent: I usually recommend between [NUMBER] dollars of coverage for cremation and [NUMBER] to [NUMBER] for burial.
Agent: So now that I know that you do take full responsibility for your final expenses, who would be your beneficiary on your policy?
Prospect: My husband.
Agent: Gotcha, what's his name?
Prospect: Rick.
Agent: Now that I have all your answers, I'm hoping I can get you qualified for one of our preferred plans. It'll just take me a few minutes to pull those up.
Prospect: I'm not going to be able to finish this call. Can you just email everything to me?
Agent: I can text you my mobile line.
"""

late_stage_fixed = run_case(
    "late-stage evidence should not downgrade to who i am",
    late_stage_should_not_downgrade_to_who_i_am,
    late_stage_transcript,
    must_contain=[
        "CALL STAGE REACHED: Quotes",
        "- Final stage supporting sale: Quotes",
    ],
    must_not_contain=[
        "CALL STAGE REACHED: Who I Am / What I Do",
        "- Beneficiary — not reached",
        "- Need amount — not reached",
    ],
)

print("Late-stage downgrade regression test passed.")

sold_completion_report = """SCORE: 88
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Third Party Underwriting
EARLY END: YES
NOT REACHED:
- Peace of Mind
- Cool Down

COMPLIANCE FAILURES: None

TASK CHECKLIST:
- Application info collected: PARTIAL
- Payment date explained: YES
- Banking/payment setup explained: PARTIAL

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Application completed through banking, no callback set, no post-sale completion evidence
- Final stage supporting sale: Application Information

SUMMARY:
The application and banking were completed, but the policy was not sold on this call.

BIGGEST MISS:
- None
"""

sold_completion_transcript = """Agent: I am going to do some disclosures.
Agent: I understand this application process was completed over the telephone.
Agent: I understand that this application and all other documents have been read to me for my review and voice signature.
Agent: Do you acknowledge that you provided your banking information and authorize the drafting of insurance premiums from the set account? Yes or no?
Prospect: Yes.
Agent: Do you understand that by stating yes, you're assigning the application electronically?
Prospect: Yes.
Agent: Now we're almost done. I'm just going to fill out the rest of your application.
"""

sold_completion_fixed = run_case(
    "sold completion evidence should stay sold",
    sold_completion_report,
    sold_completion_transcript,
    must_contain=[
        "- Was the policy sold? YES",
        "- Policy sold: YES",
        "- Final stage supporting sale: Third Party Underwriting",
    ],
    must_not_contain=[
        "- Policy sold: NO",
        "- Was the policy sold? NO",
        "no post-sale completion evidence",
    ],
)

print("Sold completion evidence test passed.")

peace_of_mind_after_sale_report = """SCORE: 88
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Third Party Underwriting
EARLY END: YES
NOT REACHED:
- Peace of Mind
- Cool Down

COMPLIANCE FAILURES: None

TASK CHECKLIST:
- Peace of mind completed: NOT REACHED
- Cool down completed: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? YES

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: YES
- Evidence: Application, banking authorization, disclosures, and voice-signature/application completion language were completed.
- Final stage supporting sale: Third Party Underwriting

SUMMARY:
Application and voice signature were completed. Peace of Mind and Cool Down were not reached.

BIGGEST MISS:
- None
"""

peace_of_mind_after_sale_transcript = """Agent: I understand this application process was completed over the telephone.
Agent: I understand that this application and all other documents have been read to me for my review and voice signature.
Prospect: Yes.
Agent: Do you understand that by stating yes, you're assigning the application electronically?
Prospect: Yes.
Agent: Okay. Well, [NAME], we're done. You're good.
Agent: We're not going to forget about you either.
Agent: We're going to mail you the package tomorrow and include everything we talked about and all the information for the program you're qualified for.
Prospect: Thank you.
Agent: Have a great day.
"""

peace_of_mind_after_sale_fixed = run_case(
    "peace of mind after sold call",
    peace_of_mind_after_sale_report,
    peace_of_mind_after_sale_transcript,
    must_contain=[
        "CALL STAGE REACHED: Peace of Mind",
        "EARLY END: NO",
        "- Peace of mind completed: YES",
        "- Cool down completed: NO",
        "- Final stage supporting sale: Peace of Mind",
        "- Cool Down",
    ],
    must_not_contain=[
        "- Peace of Mind\n",
        "Peace of Mind and Cool Down were not reached",
    ],
)

print("Peace of Mind after sale test passed.")

age_disqualification_report = """SCORE: 40
RISK: HIGH
PASS: NO
CALL STAGE REACHED: Who I Am / What I Do
EARLY END: YES
NOT REACHED:
- Health questions
- Product benefits
- Three options

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- Early refusal call: prospect ended call before warm-up; no further progression possible
- Agent did not attempt calm call control when prospect expressed disinterest and ended call

TASK CHECKLIST:
- Recording disclosure: YES
- Agent introduction: YES
- License number: NOT REACHED
- Fact Finding / Warm-up: NOT REACHED
- Health questions completed: NO
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

SALE OUTCOME:
- Policy sold: NO
- Evidence: Agent ended call after informing prospect of disqualification due to age; no sale progression
- Final stage supporting sale: None

BIGGEST MISS:
- Agent did not attempt calm call control.
"""

age_disqualification_transcript = """Agent: What's the birthday?
Prospect: I'm [NUMBER].
Agent: Unfortunately you have to be younger than [NUMBER], so you won't be able to qualify for the plans that I have.
Prospect: Okay.
Agent: I'm sorry. Have a nice day.
"""

run_case(
    "age disqualification should not fail agent",
    age_disqualification_report,
    age_disqualification_transcript,
    must_contain=[
        "PASS: YES",
        "RISK: MEDIUM",
        "SCORE: 90",
        "- Automatic fail triggered: NO",
        "- Reason: None",
        "BIGGEST MISS:\n- None",
    ],
    must_not_contain=[
        "PASS: NO",
        "Agent did not attempt calm call control",
        "Early refusal call",
    ],
)

no_income_lcr_report = """SCORE: 65
RISK: HIGH
PASS: NO
CALL STAGE REACHED: Who I Am / What I Do
EARLY END: YES
NOT REACHED:
- Health questions
- Product benefits
- Three options

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- Early refusal call: agent did not attempt calm call control when prospect expressed disinterest.
- 3 and 1 Method incomplete: Fact Finding / Warm-up not reached, so no rapport or personal disclosure occurred.

TASK CHECKLIST:
- Recording disclosure: YES
- Agent introduction: YES
- License number: YES
- Fact Finding / Warm-up: NOT REACHED
- Health questions completed: NO
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Objection occurred without proper call control

SALE OUTCOME:
- Policy sold: NO
- Evidence: Call ended before application or enrollment; no sale completion evidence
- Final stage supporting sale: Who I Am / What I Do

BIGGEST MISS:
- Objection occurred without proper call control during Who I Am / What I Do stage.
"""

no_income_lcr_transcript = """Agent: Are you working or retired?
Prospect: I'm on medical leave.
Agent: Do you have any kind of income right now at all?
Prospect: Not at all.
Agent: I don't want to sell you a policy if you don't have any income. I don't want to take food off your table.
Agent: I'm not going to make you buy a policy if you don't have any income.
Agent: I hope you have an amazing day and hope you start feeling better.
"""

run_case(
    "no income lcr should not fail agent",
    no_income_lcr_report,
    no_income_lcr_transcript,
    must_contain=[
        "PASS: YES",
        "RISK: MEDIUM",
        "SCORE: 90",
        "- Objection occurred without proper call control: NO",
        "- Automatic fail triggered: NO",
        "- Reason: None",
        "No income",
    ],
    must_not_contain=[
        "PASS: NO",
        "Automatic fail triggered: YES",
        "Early refusal call",
    ],
)

print("Disqualification fairness tests passed.")
