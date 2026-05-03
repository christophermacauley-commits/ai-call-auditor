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
        "RISK: LOW",
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
        "RISK: LOW",
        "SCORE: 90",
        "- Automatic fail triggered: NO",
        "- Reason: None",
        "no income",
    ],
    must_not_contain=[
        "PASS: NO",
        "Automatic fail triggered: YES",
        "Objection occurred without proper call control: YES",
        "Early refusal call",
    ],
)

print("Disqualification fairness tests passed.")

disqualification_coaching_cleanup_report = """SCORE: 40
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
- Attempt calm call control early when prospect expresses disinterest.
- Maintain confident and clear communication.
- Avoid abrupt ending.
- DNQ condition identified but agent did not attempt to redirect or handle per process beyond immediate stop.

TASK CHECKLIST:
- Recording disclosure: YES
- Agent introduction: YES
- Fact Finding / Warm-up: NOT REACHED
- Health questions completed: NO

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

COACHING:
- Attempt calm call control early when prospect expresses disinterest.
- Avoid abrupt ending.

SUMMARY:
The call is scored low due to lack of progression, poor objection handling, and early disengagement.

BIGGEST MISS:
- Agent did not attempt calm call control.
"""

disqualification_coaching_cleanup_transcript = """Agent: Are you working or retired?
Prospect: I don't have any income right now.
Agent: I don't want to sell you a policy if you don't have any income. I don't want to take food off your table.
Agent: I hope you have an amazing day.
"""

run_case(
    "disqualification coaching should not blame agent",
    disqualification_coaching_cleanup_report,
    disqualification_coaching_cleanup_transcript,
    must_contain=[
        "SCORE: 90",
        "RISK: LOW",
        "PASS: YES",
        "- Automatic fail triggered: NO",
        "- Reason: None",
        "Agent appropriately stopped after identifying disqualification / inability to proceed.",
        "BIGGEST MISS:\n- None",
    ],
    must_not_contain=[
        "Attempt calm call control",
        "Maintain confident and clear communication",
        "Avoid abrupt ending",
        "did not attempt to redirect",
        "poor objection handling",
        "Agent did not attempt calm call control",
    ],
)

print("Disqualification coaching cleanup test passed.")

# -------------------------------------------------------------------
# Cross-field invariants before further cleanup/refactor work
# -------------------------------------------------------------------

autofail_unsold_report = """SCORE: 92
RISK: LOW
PASS: YES
CALL STAGE REACHED: Quotes
EARLY END: NO
NOT REACHED:
- Application information
- Payment date
- Banking/payment setup
- Disclosures
- Third Party Underwriting
- Peace of Mind
- Cool Down

COMPLIANCE FAILURES:
- Callback accepted without proper call control

TASK CHECKLIST:
- Product benefits explained: YES
- Three options presented: YES

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: YES
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Quotes

BIGGEST MISS:
- Agent accepted a callback / delay instead of controlling the objection.
"""

autofail_unsold_transcript = """Agent: We can get this done today.
Prospect: Can you call me back tomorrow?
Agent: Yes, I can call you back tomorrow.
"""

run_case(
    "autofail unsold should be pass no risk high",
    autofail_unsold_report,
    autofail_unsold_transcript,
    must_contain=[
        "RISK: HIGH",
        "PASS: NO",
        "- Automatic fail triggered: YES",
        "- Policy sold: NO",
    ],
    must_not_contain=[
        "PASS: AT RISK",
        "PASS: YES",
        "RISK: LOW",
    ],
)

autofail_sold_report = """SCORE: 92
RISK: LOW
PASS: YES
CALL STAGE REACHED: Peace of Mind
EARLY END: NO
NOT REACHED:
- Cool Down

COMPLIANCE FAILURES:
- Callback accepted without proper call control

TASK CHECKLIST:
- Application info collected: YES
- Banking/payment setup explained: YES
- Disclosures completed: YES

SEARCHABLE ANSWERS:
- Was the policy sold? YES

AUTOMATIC FAIL CHECKS:
- Callback set: YES
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.

SALE OUTCOME:
- Policy sold: YES
- Final stage supporting sale: Peace of Mind

BIGGEST MISS:
- Agent accepted a callback / delay instead of controlling the objection.
"""

autofail_sold_transcript = """Agent: The application process was completed over the telephone.
Agent: I understand this application and all other documents have been read to me for my review and voice signature.
Prospect: Yes.
Agent: Okay, you're good. We're not going to forget about you.
Prospect: Can you call me back tomorrow?
Agent: Yes, I can call you back tomorrow.
"""

run_case(
    "autofail sold should be at risk high",
    autofail_sold_report,
    autofail_sold_transcript,
    must_contain=[
        "RISK: HIGH",
        "PASS: AT RISK",
        "- Automatic fail triggered: YES",
        "- Policy sold: YES",
    ],
    must_not_contain=[
        "PASS: NO",
        "RISK: LOW",
    ],
)

clean_disqualification_report = """SCORE: 42
RISK: HIGH
PASS: NO
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- Attempt calm call control early when prospect expressed disinterest.
- Avoid abrupt ending.

TASK CHECKLIST:
- Health questions completed: PARTIAL
- Product benefits explained: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Objection occurred without proper call control

SALE OUTCOME:
- Policy sold: NO
- Evidence: Call ended before application or enrollment.
- Final stage supporting sale: Medical / Health

COACHING:
- Attempt calm call control early when prospect expressed disinterest.

SUMMARY:
The call is scored low due to lack of progression, poor objection handling, and early disengagement.

BIGGEST MISS:
- Agent did not attempt calm call control.
"""

clean_disqualification_transcript = """Agent: Any kidney failure, oxygen, dialysis, hospice, or nursing home?
Prospect: I have kidney failure.
Agent: Unfortunately that means you would not qualify for the plans I have. I am sorry, but I hope you have a good day.
"""

run_case(
    "clean health disqualification should score fairly",
    clean_disqualification_report,
    clean_disqualification_transcript,
    must_contain=[
        "SCORE: 90",
        "RISK: LOW",
        "PASS: YES",
        "- Automatic fail triggered: NO",
        "- Reason: None",
        "Agent appropriately stopped after identifying disqualification / inability to proceed.",
        "BIGGEST MISS:\n- None",
    ],
    must_not_contain=[
        "PASS: NO",
        "RISK: HIGH",
        "Attempt calm call control",
        "poor objection handling",
        "Agent did not attempt calm call control",
    ],
)

not_reached_idempotent_report = """SCORE: 82
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Fact Finding / Warm-up
EARLY END: YES
NOT REACHED:
- Health questions — not reached because the prospect stopped responding / disconnected before the agent could continue
- Product benefits — not reached because the prospect stopped responding / disconnected before the agent could continue
- Three options — not reached because the prospect stopped responding / disconnected before the agent could continue

COMPLIANCE FAILURES: None

TASK CHECKLIST:
- Fact Finding / Warm-up: YES
- Health questions completed: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Fact Finding / Warm-up

BIGGEST MISS:
- Prospect stopped responding / disconnected before the agent could continue.
"""

not_reached_idempotent_transcript = """Agent: Are you there?
Prospect:
Agent: Hello? I cannot hear you.
"""

out_once = run_case(
    "not reached reason should not duplicate",
    not_reached_idempotent_report,
    not_reached_idempotent_transcript,
    must_contain=[
        "not reached because the prospect stopped responding / disconnected before the agent could continue",
    ],
    must_not_contain=[
        "continue — not reached because",
        "continue — not reached because the prospect stopped responding / disconnected before the agent could continue — not reached because",
    ],
)

out_twice = watcher.enforce_final_audit_consistency(out_once, not_reached_idempotent_transcript)
check(
    "not reached reason idempotent on rerun",
    "— not reached because the prospect stopped responding / disconnected before the agent could continue — not reached because" not in out_twice,
    out_twice,
)

print("Cross-field invariant tests passed.")

reason_merge_report = """SCORE: 95
RISK: LOW
PASS: YES
CALL STAGE REACHED: Banking
EARLY END: NO
NOT REACHED:
- Disclosures
- Third Party Underwriting
- Peace of Mind
- Cool Down

COMPLIANCE FAILURES:
- Callback issue
- Existing coverage issue

TASK CHECKLIST:
- Product benefits explained: YES
- Three options presented: YES
- Application info collected: YES

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: YES
- Objection occurred without proper call control: YES
- Existing coverage mentioned but not confirmed: YES
- Credit union mentioned but bank/account not verified: NO
- Automatic fail triggered: YES
- Reason: Existing coverage mentioned but not confirmed

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Banking

BIGGEST MISS:
- Existing coverage was mentioned but not confirmed.
"""

reason_merge_transcript = """Agent: Do you have life insurance now?
Prospect: Yes, I have one active policy.
Agent: Okay.
Prospect: Can you call me back tomorrow?
Agent: Yes, I will call you back tomorrow.
"""

run_case(
    "callback reason should merge with existing coverage reason",
    reason_merge_report,
    reason_merge_transcript,
    must_contain=[
        "- Automatic fail triggered: YES",
        "- Reason: Existing coverage mentioned but not confirmed; Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.",
        "RISK: HIGH",
        "PASS: NO",
    ],
    must_not_contain=[
        "- Reason: Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.\n",
        "- Reason: None",
        "PASS: YES",
    ],
)

print("Autofail reason merge test passed.")

false_lcr_health_questions_only_report = """SCORE: 72
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- 3 and 1 Method incomplete

TASK CHECKLIST:
- Health questions completed: YES
- Product benefits explained: NOT REACHED
- Three options presented: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Health questions were asked but no disqualification was confirmed.
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Prospect stopped responding / disconnected before the agent could continue.
"""

false_lcr_health_questions_only_transcript = """Agent: Have you had kidney failure, oxygen, dialysis, hospice, cancer, heart failure, COPD, or a terminal condition?
Prospect: No.
Agent: And we answered no to all of those health questions, right?
Prospect: Yep.
Agent: Perfect.
Prospect: I am not really interested.
"""

out = watcher.enforce_final_audit_consistency(
    false_lcr_health_questions_only_report,
    false_lcr_health_questions_only_transcript,
)

check(
    "health screening words alone should not trigger fair LCR rewrite",
    "Prospect had a disqualifying health condition" not in out
    and "Agent appropriately stopped after identifying disqualification" not in out
    and "SCORE: 90" not in out,
    out,
)

false_lcr_condition_but_agent_continues_report = """SCORE: 76
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- 3 and 1 Method incomplete

TASK CHECKLIST:
- Health questions completed: PARTIAL
- Product benefits explained: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Prospect mentioned COPD but agent continued asking follow-up questions and did not state disqualification.
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Prospect stopped responding / disconnected before the agent could continue.
"""

false_lcr_condition_but_agent_continues_transcript = """Agent: Is it used for COPD or emphysema?
Prospect: COPD.
Agent: When were you diagnosed with COPD?
Prospect: A couple years ago.
Agent: Okay, besides that, have you had a heart attack or cancer?
Prospect: No.
Agent: Okay, let's keep going.
"""

out = watcher.enforce_final_audit_consistency(
    false_lcr_condition_but_agent_continues_report,
    false_lcr_condition_but_agent_continues_transcript,
)

check(
    "condition mention without stop should not trigger LCR rewrite",
    "Prospect had a disqualifying health condition" not in out
    and "Agent appropriately stopped after identifying disqualification" not in out
    and "SCORE: 90" not in out,
    out,
)

true_lcr_agent_states_no_qualify_report = """SCORE: 40
RISK: HIGH
PASS: NO
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- Attempt calm call control early when prospect expressed disinterest.

TASK CHECKLIST:
- Health questions completed: PARTIAL
- Product benefits explained: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Reason: Objection occurred without proper call control

SALE OUTCOME:
- Policy sold: NO
- Evidence: Call ended before application or enrollment.
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Agent did not attempt calm call control.
"""

true_lcr_agent_states_no_qualify_transcript = """Agent: Are you on dialysis?
Prospect: Yes.
Agent: Unfortunately, because of that condition you would not qualify for the plans I have. I am sorry, and I hope you have a good day.
"""

run_case(
    "true health disqualification still scores fairly",
    true_lcr_agent_states_no_qualify_report,
    true_lcr_agent_states_no_qualify_transcript,
    must_contain=[
        "SCORE: 90",
        "RISK: LOW",
        "PASS: YES",
        "Prospect had a disqualifying health condition.",
        "BIGGEST MISS:\n- None",
    ],
    must_not_contain=[
        "Attempt calm call control",
        "Objection occurred without proper call control: YES",
    ],
)

print("False LCR detection tests passed.")

coverage_hangup_before_verify_report = """SCORE: 80
RISK: HIGH
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES:
- Existing coverage mentioned but not confirmed.

TASK CHECKLIST:
- Existing coverage checked: NO
- Health questions completed: PARTIAL

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Existing coverage mentioned but not confirmed: YES
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Existing coverage was mentioned but not properly confirmed.
"""

coverage_hangup_before_verify_transcript = """Prospect: I have life insurance already.
Agent: Okay, who is that through?
Prospect: Through the government. I don't need any more.
Agent: I understand. Let me explain why I was calling.
Prospect: Bye.
Agent: Hello? Are you there?
"""

run_case(
    "existing coverage hangup before verify should not autofail",
    coverage_hangup_before_verify_report,
    coverage_hangup_before_verify_transcript,
    must_contain=[
        "- Existing coverage mentioned but not confirmed: NO",
        "- Automatic fail triggered: NO",
        "- Reason: None",
    ],
    must_not_contain=[
        "- Existing coverage mentioned but not confirmed: YES",
        "Existing coverage was mentioned but not properly confirmed.",
        "PASS: NO",
    ],
)

needs_stage_after_health_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES: None

TASK CHECKLIST:
- Health questions completed: YES
- Product benefits explained: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Objection occurred without proper call control: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Prospect stopped responding before the agent could continue.
"""

needs_stage_after_health_transcript = """Agent: Okay, we answered no to all the health questions.
Agent: Have you ever had to pay for somebody's funeral?
Prospect: Yes.
Agent: Who passed away in your life?
Prospect: My brother.
Agent: Burial or cremation?
Prospect: Cremation.
Agent: Since you have no coverage, your family would need to come up with that money.
"""

run_case(
    "funeral cost questions after health should reach needs",
    needs_stage_after_health_report,
    needs_stage_after_health_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
    ],
    must_not_contain=[
        "CALL STAGE REACHED: Medical / Health",
    ],
)

print("Coverage hangup and Needs-stage tests passed.")

sold_call_stale_disqualification_report = """SCORE: 90
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Peace of Mind
EARLY END: NO
NOT REACHED:
- Cool Down

SCRIPT / FLOW MISSES:
- Banking/account information requested or verified 3 times incomplete: routing number verification did not meet the three-event standard.

TASK CHECKLIST:
- Routing number requested or verified 3 times: PARTIAL

SEARCHABLE ANSWERS:
- Was the policy sold? YES

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: YES
- Evidence: Prospect had a disqualifying health condition.
- Final stage supporting sale: Peace of Mind

COACHING:
- Agent appropriately stopped after identifying disqualification / inability to proceed. Prospect had a disqualifying health condition.

BIGGEST MISS:
- None

SUMMARY:
The call ended because the prospect was not eligible / could not reasonably proceed. Future sales stages were not reached because continuing the sale was not appropriate.
"""

sold_call_completion_transcript = """Agent: I am going to do some disclosures.
Agent: I understand this application process was completed over the telephone.
Agent: I need your review and voice signature by signing this application.
Agent: You have applied for the Golden Solutions Whole Life Insurance Policy.
Agent: A copy of your completed application will be provided.
Agent: Do you acknowledge that you provided your banking information and authorize the drafting of insurance premiums from this account?
Prospect: Yes.
"""

run_case(
    "sold call does not keep stale disqualification cleanup",
    sold_call_stale_disqualification_report,
    sold_call_completion_transcript,
    must_contain=[
        "- Policy sold: YES",
        "- Evidence: Application, banking authorization, disclosures, and voice-signature/application completion language were completed.",
        "BIGGEST MISS:\n- Routing number verification did not meet the three-event standard after the sale was completed.",
        "The agent completed a sold call",
    ],
    must_not_contain=[
        "Prospect had a disqualifying health condition",
        "Agent appropriately stopped after identifying disqualification",
        "continuing the sale was not appropriate",
    ],
)

needs_stage_final_stage_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: No application completion or banking/payment setup; call ended during health qualification
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- Prospect stopped responding before the agent could continue.
"""

run_case(
    "needs upgrade also updates final supporting stage",
    needs_stage_final_stage_report,
    needs_stage_after_health_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
        "- Final stage supporting sale: Needs",
    ],
    must_not_contain=[
        "- Final stage supporting sale: Medical / Health",
    ],
)

print("Sold stale-disqualification and Needs final-stage regression tests passed.")


needs_summary_and_biggest_miss_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Needs
EARLY END: YES
NOT REACHED:
- Product benefits

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- 3 and 1 Method incomplete: agent asked rapport questions but did not provide meaningful personal self-disclosure tied to the prospect's answers.

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: No application completion or banking/payment setup; call ended during health qualification
- Final stage supporting sale: Medical / Health

BIGGEST MISS:
- None

SUMMARY:
The call ended before need discovery, quotes, or application stages, and no sale was completed.
"""

run_case(
    "needs reports keep final stage summary and biggest miss consistent",
    needs_summary_and_biggest_miss_report,
    needs_stage_after_health_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
        "- Final stage supporting sale: Needs",
        "BIGGEST MISS:\n- 3 and 1 Method incomplete",
        "needs discovery",
    ],
    must_not_contain=[
        "- Final stage supporting sale: Medical / Health",
        "before need discovery",
        "BIGGEST MISS:\n- None",
    ],
)

false_health_dq_clean_screen_report = """SCORE: 90
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Needs
EARLY END: YES
NOT REACHED:
- Product benefits

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- None

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Prospect had a disqualifying health condition.
- Final stage supporting sale: Medical / Health

COACHING:
- Agent appropriately stopped after identifying disqualification / inability to proceed. Prospect had a disqualifying health condition.

BIGGEST MISS:
- None

SUMMARY:
The call ended because the prospect was not eligible / could not reasonably proceed. Future sales stages were not reached because continuing the sale was not appropriate.
"""

false_health_dq_clean_screen_transcript = """Agent: Stroke, heart attack, COPD, kidney failure, oxygen, cancer, or diabetes?
Prospect: No.
Agent: Everything was no, so you are in really good shape.
Prospect: I already have coverage and I don't need any more.
Agent: I understand.
Prospect: Bye.
Agent: Hello? Are you there?
"""

run_case(
    "clean health screen should not become health disqualification cleanup",
    false_health_dq_clean_screen_report,
    false_health_dq_clean_screen_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
        "- Final stage supporting sale: Needs",
        "health screening cleanly",
    ],
    must_not_contain=[
        "Prospect had a disqualifying health condition",
        "Agent appropriately stopped after identifying disqualification",
        "continuing the sale was not appropriate",
        "call ended because the prospect was not eligible",
    ],
)

print("Needs consistency and false health-disqualification cleanup tests passed.")

false_quotes_from_needs_report = """SCORE: 75
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Quotes
EARLY END: YES
NOT REACHED:
- Application information
- Banking
- Disclosures

COMPLIANCE FAILURES: None

SCRIPT / FLOW MISSES:
- Early refusal call: agent did not attempt calm call control when the prospect expressed confusion and affordability concern.

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Call ended before application completion; no banking verification or disclosures completed
- Final stage supporting sale: Quotes

BIGGEST MISS:
- None

SUMMARY:
The agent reached quotes but the call ended before application.
"""

false_quotes_from_needs_transcript = """Agent: These plans cover burial or cremation expenses.
Agent: I will be able to give you the exact cost right now over the phone.
Agent: Okay, we answered no to all the health questions.
Agent: Have you ever had to pay for somebody's funeral?
Prospect: Yes.
Agent: Who passed away in your life?
Prospect: My brother.
Agent: Burial or cremation?
Prospect: Cremation.
Agent: Since you have no coverage, your family would need to come up with that money.
Prospect: I cannot afford anything right now.
Agent: I understand.
Prospect: Bye.
Agent: Hello, are you there?
"""

run_case(
    "generic exact-cost intro should not upgrade Needs call to Quotes",
    false_quotes_from_needs_report,
    false_quotes_from_needs_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
        "- Final stage supporting sale: Needs",
        "before quotes",
    ],
    must_not_contain=[
        "CALL STAGE REACHED: Quotes",
        "- Final stage supporting sale: Quotes",
        "reached quotes",
    ],
)

print("False Quotes-stage cleanup tests passed.")

clean_short_call_should_not_grade_unreached_sections_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Who I Am / What I Do
EARLY END: YES
NOT REACHED:
- Fact Finding / Warm-up
- Product benefits
- Three options

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- 3 and 1 Method incomplete: agent did not complete rapport.
- Product benefits explained incomplete.
- Three options presented incomplete.

TASK CHECKLIST:
- Agent introduction: YES
- 3 and 1 Method used: NO
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Who I Am / What I Do

COACHING:
- Complete the 3 and 1 method next time.

BIGGEST MISS:
- 3 and 1 Method incomplete.
"""

clean_short_call_transcript = """Agent: Hi, this is Ashley calling about the benefits.
Prospect: Not interested.
Agent: I understand.
Prospect: Bye.
"""

run_case(
    "clean short call should not grade unreached future sections",
    clean_short_call_should_not_grade_unreached_sections_report,
    clean_short_call_transcript,
    must_contain=[
        "RISK: LOW",
        "- Product benefits explained: NOT REACHED",
        "- Three options presented: NOT REACHED",
        "- Application info collected: NOT REACHED",
        "confident tonality",
    ],
    must_not_contain=[
        "3 and 1 Method incomplete",
    ],
)

clean_disq_should_not_grade_future_sections_report = """SCORE: 90
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Who I Am / What I Do
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options
- Application information

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- Product benefits explained incomplete.
- Three options presented incomplete.
- Application info collected incomplete.

TASK CHECKLIST:
- Agent introduction: YES
- Product benefits explained: NO
- Three options presented: NO
- Application info collected: NO

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Prospect had no income / affordability barrier, so the agent appropriately did not continue the sale.
- Final stage supporting sale: Who I Am / What I Do

COACHING:
- Product benefits and closing need improvement.

BIGGEST MISS:
- Product benefits explained incomplete.
"""

run_case(
    "clean disqualification should not grade unreached future sections",
    clean_disq_should_not_grade_future_sections_report,
    "Prospect: I do not have any income right now.\nAgent: I understand.",
    must_contain=[
        "RISK: LOW",
        "- Product benefits explained: NOT REACHED",
        "- Three options presented: NOT REACHED",
        "- Application info collected: NOT REACHED",
    ],
    must_not_contain=[
        "Product benefits explained incomplete",
        "Three options presented incomplete",
    ],
)

print("Clean early/unreached-section cleanup tests passed.")

clean_health_needs_hangup_should_not_be_dq_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Medical / Health
EARLY END: YES
NOT REACHED:
- Product benefits
- Three options

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- None

TASK CHECKLIST:
- Health questions completed: YES
- Product benefits explained: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Evidence: Prospect had a disqualifying health condition.
- Final stage supporting sale: Medical / Health

COACHING:
- Agent appropriately stopped after identifying disqualification / inability to proceed. Prospect had a disqualifying health condition.

SUMMARY:
The call ended because the prospect was not eligible / could not reasonably proceed. Future sales stages were not reached because continuing the sale was not appropriate.

BIGGEST MISS:
- None
"""

clean_health_needs_hangup_transcript = """Agent: Any stroke, heart attack, COPD, kidney failure, cancer, diabetes, oxygen, or nursing home?
Prospect: No.
Agent: Everything was no, so you are in really good shape.
Agent: Have you ever had to pay for somebody's funeral?
Prospect: Yes.
Agent: Who passed away?
Prospect: My brother.
Agent: Burial or cremation?
Prospect: Cremation.
Agent: Since you have no coverage, your family would need to come up with that money.
Prospect: I do not need anything else.
Agent: I understand.
Prospect: Bye.
"""

run_case(
    "clean health screen plus Needs should not become health disqualification",
    clean_health_needs_hangup_should_not_be_dq_report,
    clean_health_needs_hangup_transcript,
    must_contain=[
        "CALL STAGE REACHED: Needs",
        "- Final stage supporting sale: Needs",
        "health screening cleanly",
        "reached needs discovery",
    ],
    must_not_contain=[
        "Prospect had a disqualifying health condition",
        "Agent appropriately stopped after identifying disqualification",
        "continuing the sale was not appropriate",
        "The call ended because the prospect was not eligible",
    ],
)

print("Clean health Needs/hangup false-DNQ cleanup tests passed.")

bootc_should_not_be_lead_report = """SCORE: 85
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Opening
EARLY END: YES
NOT REACHED:
- Remaining sales process

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- None

TASK CHECKLIST:
- Agent introduction: NOT REACHED

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: None

COACHING:
- Avoid ending the call abruptly; try to build rapport or clarify prospect needs before concluding.

BIGGEST MISS:
- None
"""

bootc_should_not_be_lead_transcript = """Agent: Hello?
Prospect: Stop calling me.
Agent: Hello, are you there?
"""

run_case(
    "BOOTC call should get low risk and tonality coaching",
    bootc_should_not_be_lead_report,
    bootc_should_not_be_lead_transcript,
    must_contain=[
        "RISK: LOW",
        "CALL STAGE REACHED: BOOTC",
        "confident tonality",
    ],
    must_not_contain=[
        "Avoid ending the call abruptly; try to build rapport or clarify prospect needs before concluding",
        "RISK: MEDIUM",
    ],
)

u90_short_call_should_get_tonality_report = """SCORE: 80
RISK: MEDIUM
PASS: YES
CALL STAGE REACHED: Who I Am / What I Do
EARLY END: YES
NOT REACHED:
- Fact Finding / Warm-up
- Health questions

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
- None

TASK CHECKLIST:
- Agent introduction: YES

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None

SALE OUTCOME:
- Policy sold: NO
- Final stage supporting sale: Who I Am / What I Do

COACHING:
- Avoid ending the call abruptly; try to build rapport or clarify prospect needs before concluding.

BIGGEST MISS:
- None
"""

u90_short_call_should_get_tonality_transcript = """duration_seconds: 65
Agent: Hi, my name is Ashley and I'm calling about the benefits you requested.
Prospect: I am not interested.
Agent: I understand.
Prospect: Bye.
"""

run_case(
    "U90 short call should get low risk and tonality coaching",
    u90_short_call_should_get_tonality_report,
    u90_short_call_should_get_tonality_transcript,
    must_contain=[
        "RISK: LOW",
        "confident tonality",
    ],
    must_not_contain=[
        "Avoid ending the call abruptly; try to build rapport or clarify prospect needs before concluding",
        "RISK: MEDIUM",
    ],
)

print("BOOTC/U90 disposition cleanup tests passed.")

# Direct disposition tests for DB/reprocess edge cases.
bootc_report_for_disposition = """SCORE: 85
RISK: LOW
PASS: YES
CALL STAGE REACHED: BOOTC
EARLY END: YES

SEARCHABLE ANSWERS:
- Was the policy sold? NO

AUTOMATIC FAIL CHECKS:
- Callback set: NO
- Automatic fail triggered: NO
- Reason: None
"""

bootc_disp, _ = watcher.detect_auto_disposition(
    "bootc_fixture",
    "Agent: Hello? Prospect: Stop calling me. Agent: Are you there?",
    bootc_report_for_disposition,
    duration_seconds=45,
)
check("BOOTC report stage outranks U90 disposition", bootc_disp == "BOOTC", bootc_disp)

u90_disp, _ = watcher.detect_auto_disposition(
    "u90_fixture",
    "Agent: Hi, my name is Ashley and I am calling about the benefits you requested. Prospect: Not interested. Agent: I understand.",
    "SCORE: 80\nRISK: LOW\nPASS: YES\nCALL STAGE REACHED: Who I Am / What I Do\n- Policy sold: NO\n- Automatic fail triggered: NO\n",
    duration_seconds=65,
)
check("duration_seconds under 110 triggers U90 disposition", u90_disp == "U90", u90_disp)

print("BOOTC/U90 database disposition priority tests passed.")
