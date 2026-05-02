# AI Auditor Logic Contract

## Core Principle

The audit must distinguish between agent-controllable misses, not-reached stages caused by the call ending, and transcript/speaker-label uncertainty.

## Report Display Rules

- The Detailed Report should not include the full transcript block.
- Remove TRANSCRIPT NOTE (MANDATORY) and embedded TRANSCRIPT blocks from saved reports.
- Decode HTML entities like &#x27;, &quot;, and &amp;.
- Compress long repeated NOT REACHED lists into a short grouped summary.

## Privacy / Redaction Rules

- Transcripts should redact phone numbers, DOB, SSN, bank/account/routing/card numbers, long digit sequences, and sensitive payment details.
- Reports should preserve audit numbers, score breakdowns, token counts, and business terms like 3 and 1 Method.
- Never convert SCORE: [NUMBER] into SCORE: 0. Use Unknown if the score is damaged.

## Callback Rules

A callback autofail requires both:
1. Prospect callback/delay request evidence.
2. Agent acceptance or failure-to-control evidence.

Do not count:
- IVR prompts like “To receive a callback, press [NUMBER].”
- Insurance-company queue callback prompts.
- Rapport stories that mention “later.”
- Prospect disconnecting or stopping response.

If real callback autofail:
- Callback set: YES
- Objection occurred without proper call control: YES
- Automatic fail triggered: YES
- Risk: HIGH

If no real callback evidence:
- Callback set: NO
- Objection occurred without proper call control: NO
- Do not leave callback-based Biggest Miss or Reason.

## Existing Coverage Rules

- Existing coverage mentioned but not confirmed can be an automatic fail.
- Existing coverage confirmed by carrier/insurance-company call should not be autofailed.
- Insurance-company IVR language should not be confused with callback objection language.

## Stage Reached Rules

Stage should be based on latest supported evidence.

If need amount or beneficiary happened, stage cannot be only Who I Am / What I Do.

If health questions started, stage should be at least Medical / Health.

If quote or preferred-plan evidence exists, stage can be Quotes.

If policy sold is NO, product benefits are NO, options are NO, application info is NO, and account/routing evidence is 0, stage cannot be Banking, Application, Disclosures, Peace of Mind, or Cool Down.

Fallback stage order:
1. Quotes
2. Medical / Health
3. Need / Beneficiary
4. Fact Finding / Warm-up
5. Who I Am / What I Do
6. Opening / Handoff

## Early End Rules

If the prospect stops responding, hangs up, refuses, or cannot continue:
- EARLY END: YES
- Later stages are NOT REACHED
- Do not treat later not-reached stages as agent misses
- Do not heavily punish the agent if the agent did nothing wrong

If the agent mishandles an objection or accepts an improper callback/delay, that is agent-controllable.

## Score / Risk / Pass Rules

If Automatic fail triggered: YES, risk should be HIGH.

Sold with automatic fail:
- Audit Result should usually be AT RISK
- Sale Outcome remains YES

Unsold with automatic fail:
- Audit Result should usually be NO
- Risk should be HIGH

Unfinished with no agent fault:
- Do not make it HIGH risk only because later stages were not reached.
- Do not let it look like a clean completed win.

## Speaker Label Rules

PQ should normally only appear during the opening handoff.

Late PQ labels after the selling agent takes control should be treated cautiously.

If a long personal rapport/self-disclosure block is labeled Prospect but clearly appears to be the agent sharing personal details, relabel it Agent without rewriting the transcript words.

## Maintenance Rules

Before committing future logic changes, run:

python3 tests/test_audit_guardrails.py
python3 -m py_compile watcher.py dashboard.py native_app.py
python3 -c "import watcher; print('watcher import ok')"

Every logic fix should include:
1. Rule update if needed.
2. Regression test.
3. Smallest code change that passes tests.
