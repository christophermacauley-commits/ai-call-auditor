import os
import sys
import time
import subprocess
import re
import sqlite3
import shutil
import json
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

load_dotenv()

CALLS_FOLDER = "calls"
TRANSCRIPT_UPLOADS_FOLDER = "transcript_uploads"
PROCESSED_CALLS_FOLDER = "processed_calls"
PROCESSED_TRANSCRIPTS_FOLDER = "processed_transcripts"
TRANSCRIPTS_FOLDER = "transcripts"
TRANSCRIPTS_ROLE_LABELED_FOLDER = "transcripts_role_labeled"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"

# "medium" balances WER vs speed; int8 on CPU is the usual faster-whisper sweet spot (much faster
# than float32/float16 with modest accuracy loss vs full precision).
WHISPER_MODEL = "small"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SCAN_INTERVAL_SECONDS = 5
OLLAMA_TIMEOUT_SECONDS = 300
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")

TRANSCRIPTION_START_PROGRESS = 5
TRANSCRIPTION_DONE_PROGRESS = 75
AI_START_PROGRESS = 80
AI_DONE_PROGRESS = 95
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
USE_STRUCTURED_AUDIT = os.getenv("USE_STRUCTURED_AUDIT", "false").strip().lower() == "true"
STORE_RAW_TRANSCRIPTS = os.getenv("STORE_RAW_TRANSCRIPTS", "false").strip().lower() == "true"
OPENAI_INPUT_COST_PER_1K_TOKENS = 0.0004
OPENAI_OUTPUT_COST_PER_1K_TOKENS = 0.0016

model = None
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text):
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def get_model():
    global model
    if model is None:
        # medium: stronger ASR than small/tiny with acceptable latency on CPU when combined with
        # int8 quantization, VAD skipping silence, and decode settings below (beam 5, no context carry).
        model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )
    return model


def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_name TEXT,
        transcript TEXT,
        report TEXT,
        score INTEGER,
        risk TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS processing_state (
        call_name TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def set_processing_state(call_name, filename, status, progress=0, message=None, error=None):
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT attempts FROM processing_state WHERE call_name=?", (call_name,))
    row = c.fetchone()
    attempts = row[0] if row else 0

    if status in ("processing", "retry"):
        attempts += 1

    if row:
        c.execute("""
            UPDATE processing_state
            SET filename=?,
                status=?,
                progress=?,
                message=?,
                attempts=?,
                error=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE call_name=?
        """, (filename, status, int(progress), message, attempts, error, call_name))
    else:
        c.execute("""
            INSERT INTO processing_state
                (call_name, filename, status, progress, message, attempts, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (call_name, filename, status, int(progress), message, attempts, error))

    conn.commit()
    conn.close()


def save_to_db(call_name, transcript, report, score, risk):
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
    c.execute("""
        INSERT INTO calls (call_name, transcript, report, score, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (call_name, transcript, report, score, risk))

    conn.commit()
    conn.close()


def parse_report(report_text):
    score = None
    risk = None

    for line in report_text.splitlines():
        upper = line.upper().strip()

        if upper.startswith("SCORE:"):
            nums = re.findall(r"\d+", line)
            if nums:
                score = int(nums[0])

        if upper.startswith("RISK:"):
            if "HIGH" in upper:
                risk = "HIGH"
            elif "MEDIUM" in upper:
                risk = "MEDIUM"
            elif "LOW" in upper:
                risk = "LOW"

    return score, risk


def get_ollama_command():
    candidates = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        shutil.which("ollama"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Ollama executable not found.")


def run_ollama(prompt):
    ollama = get_ollama_command()

    result = subprocess.run(
        [ollama, "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=OLLAMA_TIMEOUT_SECONDS
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama failed: {error}")

    return clean_text(result.stdout.decode("utf-8", errors="ignore"))


def build_audit_prompt(transcript, checklist, rubric, output_format, role_label_note=None):
    role_block = f"\n{role_label_note}\n\n" if role_label_note else ""
    return f"""
You are a strict QA auditor for final expense sales calls.

HARD OVERRIDE RULES (MANDATORY):

- If PQ identified the prospect before handoff, it is STRICTLY FORBIDDEN to include any mention of name confirmation in:
  - SCRIPT / FLOW MISSES
  - COACHING
  - SCORING BREAKDOWN
  - SUMMARY

- If the call ends before Fact Finding / Warm-up, it is STRICTLY FORBIDDEN to:
  - Coach the agent to progress the call forward
  - Mention missing Fact Finding / Warm-up or need discovery
  - Penalize for not advancing stages

- If Recording Disclosure is marked YES, it is STRICTLY FORBIDDEN to:
  - Critique timing
  - Suggest improvement
  - Mention it as a miss

- If License Number is not required at the stage reached, it is STRICTLY FORBIDDEN to:
  - Mention it as a miss
  - Include it in coaching

- Call control coaching should ONLY be given if:
  - The prospect shows resistance, objection, or attempts to end the call
  - AND the agent does NOT attempt any call control response

- If the agent does use call control appropriately, do NOT coach on call control.

- If no objection or resistance occurs, it is STRICTLY FORBIDDEN to:
  - Mention call control
  - Suggest improving call control
  - Penalize for lack of call control

- Only include misses and coaching that are DIRECTLY REQUIRED by the stage reached.
- If something is not required at the stage reached, it must be COMPLETELY IGNORED.

CALL STAGE:
Determine the SINGLE furthest stage reached in the call.
There is **no separate** “Warm-up / Rapport” or “Fact finding” **call stage** — those activities are **only** the combined stage **Fact Finding / Warm-up** (exact name in the list below).
STAGE DETECTION MUST BE STRICT (with one important exception below for **Fact Finding / Warm-up**):
- A stage is only reached if the agent clearly **enters** that stage in the transcript (progression), not when every checklist item for that stage is finished (**completion** belongs in TASK CHECKLIST / SCRIPT MISSES / COACHING, not in CALL STAGE REACHED).
- Do NOT infer a later stage from vague conversation.
- Do NOT mark Need unless the agent clearly asks about the customer's need, reason for coverage, family protection, burial/funeral concern, or motivation for buying.
- If the agent only explains who they are or what they do, the stage is Who I Am / What I Do.
- If uncertain between two stages, choose the earlier stage **except** where **Fact Finding / Warm-up — STAGE ENTRY** below applies.

Fact Finding / Warm-up — STAGE ENTRY vs TASK CHECKLIST (MANDATORY):
- **Fact Finding / Warm-up** is **one** stage that includes **all** of: warm-up, rapport building, **3 and 1 Method**, asking personal/background questions, gathering information **before** medical, and building credibility/connection. Do **not** treat these as separate stages in CALL STAGE REACHED or NOT REACHED.
- **CALL STAGE REACHED** for **Fact Finding / Warm-up** is based on **entry**, **not** full completion of every checklist line for that segment.
- Mark **Fact Finding / Warm-up** as **REACHED** as soon as the agent clearly **begins** any of the above (e.g. even **one or two** clear rapport or fact-finding questions **before** heavy **Medical / Health** underwriting).
- **If the agent transitions into Medical / Health** (health questions, medications, height/weight, tobacco, underwriting health, etc.), **Fact Finding / Warm-up must be considered already reached** — do **not** leave CALL STAGE REACHED at Who I Am / What I Do or Opening solely because 3+1 or rapport checklist items were incomplete.
- **NOT REACHED** must **not** list **Fact Finding / Warm-up** if the transcript shows that segment was **entered** or superseded by **Medical / Health**.
- **Separation of concerns:** **CALL STAGE REACHED** = progression; **TASK CHECKLIST** lines for **Fact Finding / Warm-up**, **3 and 1 Method used**, and **Agent shared personal rapport information** = **completion quality** — use **NO** / **PARTIAL** / **YES** (and **NOT REACHED** only when that line’s stage segment was **never entered**, per rules below), not “stage not reached” for the whole call when the segment clearly started.

OPENING
- Agent answered with energy and enthusiasm
- Agent gave the recording disclosure
- Agent introduced themselves
- Agent stated they were licensed (optional)
- License number: YES only if an actual license number was clearly stated
- Agent confirmed who they were speaking with

Choose ONLY ONE from this list:
- PQ / Handoff
- Opening
- Who I Am / What I Do
- Fact Finding / Warm-up
- Medical / Health
- Need
- Features / Benefits
- Change Up
- Pre-Close
- Quotes
- Close
- Application Information
- Payment Date
- Banking
- Disclosures
- Third Party Underwriting
- Peace of Mind
- Cool Down

FURTHEST STAGE — POST-APPLICATION (MANDATORY):
- **CALL STAGE REACHED** must be the **latest** stage in the ordered list above that the agent **clearly performed** in the transcript.
- **POST-SALE / ENROLLMENT ORDER (after Application Information):** Application Information → **Payment Date** → **Banking** → **Disclosures** → **Third Party Underwriting** → **Peace of Mind** → **Cool Down**. **CALL STAGE REACHED** must be the **furthest** stage in this order that was **clearly performed** in the transcript — do **not** skip ahead in the label unless that stage clearly occurred.
- **Do NOT** stop at **Application Information** only because the policy was sold or application details were taken — if the call continued into later enrollment steps, advance the stage accordingly.
- **Do NOT** infer later post-sale stages from any of: **Policy sold**; application completed; banking completed; payment date explained; disclosures read; polite ending; thank you / goodbye; friendly tone; generic warmth; a normal close.
- If the agent collected **banking / account / routing / payment** information for the policy, or **called the bank/CU to verify** account/banking for payment setup, **Banking** is reached (at minimum) unless a **later** listed stage clearly occurred afterward.
- If the agent **read disclosures** (legal/state/product disclosures to the prospect), **Disclosures** is reached when clearly performed.
- **Third Party Underwriting** is **NOT** reached merely because the policy was sold, application information was collected, banking was handled, payment date was handled, disclosures were read, or the agent said the application was complete. **Third Party Underwriting** is reached **only** when the transcript **clearly** shows the agent starting or completing the **post-disclosure** third-party recorded verification step, such as: calling into the **American Amicable recorded line**, starting the **American Amicable** recording, beginning **recorded third-party underwriting**, asking **official recorded verification / underwriting questions after disclosures**, or clearly placing the prospect into the required **carrier/third-party recorded verification** process. **Strong transcript evidence** includes (non-exhaustive): **"Welcome to the American Amicable Group recording system."**; **"For the app ID, only enter the numbers."**; **"Enter the app ID followed by the pound sign."**; **American Amicable Group recording system**; **American Amicable recording system**; carrier IVR-style **recording system** plus **app ID** / **pound sign** instructions; **voice signature** / **recorded verification** through American Amicable. When that language appears, **Third Party Underwriting** is reached — set **CALL STAGE REACHED** to at least **Third Party Underwriting** (unless **Peace of Mind** or **Cool Down** clearly occurred afterward per their strict definitions) and **do not** list **Third Party Underwriting** under **NOT REACHED**. If **Peace of Mind** and **Cool Down** did not occur afterward, list them under **NOT REACHED** instead. If sale/application/banking/disclosures appear but **no** clear recorded third-party underwriting cues (above or equivalent), do **NOT** mark **Third Party Underwriting** as reached and do **NOT** mention **recorded third-party underwriting** in **SALE OUTCOME** evidence or **SUMMARY**. If **Disclosures** were reached and **Third Party Underwriting** was **skipped** when it should have happened next, include a **SCRIPT / FLOW MISS** and **lower the score**, unless the prospect ended the call before the agent had a reasonable chance.
- **Third Party Underwriting — HARD TRIGGER (post-application / post-disclosure context):** If any of these appear in transcript-backed enrollment flow — **Welcome to the American Amicable Group recording system**; **For the app ID, only enter the numbers**; **Enter the app ID followed by the pound sign**; **American Amicable recording system**; **app ID**; **pound sign**; **voice signature**; **recorded line**; **recorded verification** — treat **Third Party Underwriting** as **reached**: **CALL STAGE REACHED** must be **at least Third Party Underwriting** unless **Peace of Mind** or **Cool Down** clearly occurred afterward; **never** list **Third Party Underwriting** only under **NOT REACHED**.
- **Peace of Mind** is reached **only** when the transcript clearly shows the **specific post-sale reassurance section** (not generic politeness). **Do NOT** mark **Peace of Mind** or treat it as reached from: generic politeness; a normal closing; thank you; goodbye; confirming the sale; completing application/payment/banking/disclosures; or simply saying the policy is done. It usually includes language along the lines of reassurance after the sale, **welcome letter** / **mailing tomorrow**, **not going to forget about you**, **personal information** plus **company you qualified for**, confidence about the decision, beneficiary/family protection, or similar **post-sale confidence** before ending — **after** sale/application/payment setup when evidenced. If the agent skips this section and only moves toward ending, **Peace of mind completed** is **NO** when there was reasonable opportunity (not when the prospect hung up or disconnected first).
- **Cool Down** is **separate** from **Peace of Mind**. **Cool Down** means the agent clearly spends time in **casual non-insurance conversation after the sale** (e.g. weather, family, hobbies, location, work, pets — **back-and-forth** away from insurance/application). **Do NOT** mark **Cool Down** or set **CALL STAGE REACHED** to **Cool Down** from: polite ending; thank you; goodbye; confirming the sale; wrapping up the application; warm tone; or a normal close. If there is **no** clear non-insurance small talk after the sale, **Cool down completed** is **NO** and **Cool Down** is **not** the furthest stage reached when it was required and the agent had reasonable opportunity (not when the prospect ended the call first).
- **NOT REACHED** must include **only** stages **after** the furthest reached stage, in order — **never** list a stage as NOT REACHED if it clearly occurred in the transcript.
- Tie-break: if uncertain between two **early** (pre-application) adjacent stages, prefer the **earlier** stage; after **Application Information**, when the transcript **clearly** shows a **later** listed stage occurred **per that stage's strict definition** (especially **Third Party Underwriting**, **Peace of Mind**, **Cool Down**), set **CALL STAGE REACHED** to that **furthest** later stage — do **not** under-select **Application Information** because of sale alone, and do **not** use sale or politeness to infer **Third Party Underwriting**, **Peace of Mind**, or **Cool Down**.

**PEACE OF MIND AND COOL DOWN AFTER SALE (WHEN TO SCORE):**
- If **Policy sold** is **YES** and the agent had a **reasonable opportunity** before the call ended, evaluate **Peace of Mind** and **Cool Down** completion on **transcript evidence** per the strict definitions above — **not** from sale or enrollment steps alone.
- If the agent **skips Peace of Mind** when there was time/opportunity, mark **Peace of mind completed: NO**, include a **SCRIPT / FLOW MISS**, **lower the Sales Process score**, and mention it in **TOP 3 COACHING PRIORITIES** if it is a top issue. Same for **Cool Down** when skipped with opportunity: **Cool down completed: NO**, **SCRIPT / FLOW MISS**, **lower the Sales Process score**, coaching if top issue, and **do not** set **CALL STAGE REACHED** to **Cool Down**.
- **Do NOT** penalize for skipping **Peace of Mind** or **Cool Down** if the **prospect/customer** ended the call, disconnected, or the transcript ended before the agent had a reasonable chance.

BANKING STAGE (CALL STAGE REACHED / NOT REACHED):
- **Banking** is reached when the transcript shows the agent asked for or handled **bank name**, **bank vs credit union status**, **routing number**, **account number**, **checking/savings/payment/draft account**, **read-back** of account/routing details (including **redacted** forms), **prospect confirmation** of redacted banking details, or **called/verified** with a bank or credit union for payment setup.
- If **"Did the agent call the bank to verify banking/account information?"** is **YES**, treat **Banking** as reached for stage detection.
- Do **NOT** list **Banking** under **NOT REACHED** if any banking or payment-account information was collected or verified on the call.
- If **Application Information** was reached and banking/payment-account handling occurred afterward, **CALL STAGE REACHED** should be **Banking** unless **Payment Date**, **Disclosures**, **Third Party Underwriting**, **Peace of Mind**, **Cool Down**, or another later listed stage was clearly reached after Banking.

**REDACTED PLACEHOLDERS — STAGE DETECTION (MANDATORY):**
- Redacted tokens (**[DATE]**, **[NUMBER]**, **[MONEY]**, **[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, **[PHONE]**, etc.) must **not** hide real enrollment progress: infer stages from **surrounding words** and **turn structure**, not from raw digits alone.
- **Payment Date** may be reached when the transcript clearly discusses **policy draft / payment date / premium draft timing** for **this** sale, even if the calendar day appears only as **[DATE]** or **[NUMBER]** / **[MONEY]**.
- **Banking** may be reached when the transcript shows **bank name**, **bank vs credit union**, **routing**, **account**, **checking/savings/payment/draft account**, read-back or confirmation of **redacted** banking details, or placeholders (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, or **[NUMBER]** when context is clearly banking). Do **not** list **Banking** under **NOT REACHED** solely because numbers were redacted.
- **Disclosures** may be reached from **legal/state/product disclosure** language or the agent **reading required disclosures**, even if amounts or dates nearby are redacted.
- **Third Party Underwriting:** use the **HARD TRIGGER** bullet in **FURTHEST STAGE** above — redacted tokens must not hide that stage when surrounding language matches.

PAYMENT DATE STAGE (STRICT — SEPARATE FROM DEPOSIT TIMING):
- **Payment Date** is reached only when the agent **explains, sets, or confirms** the policy **draft/payment date** (or first draft date for the policy premium) with the prospect in the context of this sale.
- **Asking only when Social Security or benefits are deposited** (e.g. which day the check hits) **without** tying it to the **policy draft/payment date** for this policy does **NOT** satisfy **Payment Date** — keep **Payment Date** under **NOT REACHED** unless the draft/payment date was clearly explained or set.

LATE-STAGE SCRIPT / FLOW MISSES (WHEN BANKING REACHED):
- If **Banking** was reached and **Payment Date** was not explained/set/confirmed per above, include a **SCRIPT / FLOW MISS** for missing payment/draft date handling.
- **BANKING / ACCOUNT REPETITION (SCORING):** During **Banking**, the agent should ask for or verify the prospect's **redacted banking/account information** **multiple** times. **Expected:** **3** separate request/verify/read-back cycles when possible; **minimum 2**; agent should **read back** redacted account/payment details for confirmation and the prospect should **confirm**. **YES** only when **3** clear asks/verifications and read-back/confirmation are evident; **PARTIAL** when at least **2** but not **3**, or read-back without full **3**; **NO** when only **one** collection, no read-back, or no meaningful confirmation. If **Banking** is reached but the agent did **not** verify at least **2** times **or** did **not** read back, include a **SCRIPT / FLOW MISS** and **lower the Sales Process score**. Never echo full account or routing numbers — refer only to **redacted banking/account information**.
- **TASK CHECKLIST (BANKING — REQUIRED LINES when Banking is reached):** Include these **four** lines **exactly** in **TASK CHECKLIST** (use **NOT REACHED** on all four only if **Banking** was not reached):
  - Banking/account information requested or verified 3 times: **YES** / **NO** / **PARTIAL** / **NOT REACHED** (**YES** only if **three** separate clear requests/repetitions/verifications; **PARTIAL** if at least **2** but not **3**; **NO** if only once or unclear verification)
  - Banking/account information verified at least 2 times: **YES** / **NO** / **NOT REACHED**
  - Agent read banking/account information back to prospect: **YES** / **NO** / **NOT REACHED** (**YES** only if the agent clearly read **redacted** banking/account details back for confirmation)
  - Prospect confirmed banking/account read-back: **YES** / **NO** / **NOT REACHED** (**YES** only if the prospect clearly confirmed the read-back)
- If **Banking** was reached and **Peace of Mind** and/or **Cool Down** were skipped when the agent had a **reasonable opportunity** (and the customer did **not** cut the call short), add **Peace of Mind** and/or **Cool Down** to **SCRIPT / FLOW MISSES** and **lower the score**; include matching items in **COACHING** / **TOP 3 COACHING PRIORITIES**.

RULES:
- Only select the LAST stage clearly reached.
- Do NOT list multiple stages.
- Do NOT repeat the list.

EARLY END RULE:
- If Cool Down was NOT reached, EARLY END must be YES.
- If Cool Down WAS reached, EARLY END must be NO.
- EARLY END does NOT automatically mean FAIL.
- The call can PASS if the agent correctly completed the required sales checklist items up to the furthest stage reached.
- Do NOT penalize for stages that were not reached unless the agent skipped required steps before ending (does **not** apply when the **customer** hung up — that is not the agent “ending” early).
- Judge early-ended calls only up to the furthest stage reached.

SCORING RULES:

3 AND 1 METHOD — SCORE IMPORTANCE (MAJOR IMPACT, **NOT** AUTOMATIC FAIL):
- Building rapport through the **3 and 1 Method** is one of the most important parts of the sales call when **Fact Finding / Warm-up** was **entered**.
- Missing, weak, or partial **3 and 1** is **not** a compliance **automatic fail** by itself. Do **not** set **Automatic fail triggered: YES**, **RISK: HIGH**, **PASS: NO**, or **PASS: AT RISK** **solely** because of weak, partial, or missing **3 and 1** (those outcomes require their own rules, e.g. coverage, callback, post-sale, payment date, credit union verification).
- If **Fact Finding / Warm-up** was **NOT** reached, do **not** penalize **3 and 1** — keep **3 and 1 Method used** / **Agent shared personal rapport information** as **NOT REACHED** when the segment never started; do **not** treat pre-entry silence as a rapport fail.
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **YES** (only when the **evidence gate** and **A + B** in **Fact Finding / Warm-up — TASK CHECKLIST** are fully met — **three+** topic groups **and** at least **one** meaningful tied self-disclosure you can cite as a **quote or clear paraphrase** — **questions alone are never enough**), apply **no** score deduction **for this item** (other dimensions still score normally).
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **PARTIAL**, lower the **final SCORE** meaningfully — generally about **5–10** points depending on how much was completed; reduce **Sales Process** and **Communication Quality** when rapport was rushed, shallow, interrogative, or lacked meaningful tied self-disclosure.
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **NO**, lower the **final SCORE** heavily — generally about **10–15** points when the warm-up segment clearly occurred; **Sales Process** and **Communication Quality** must both reflect the gap.
- A call with weak or missing **3 and 1** must **not** receive a **near-perfect** final **SCORE** (or near-perfect **Sales Process** / **Communication Quality**) unless the transcript shows **exceptional** strengths elsewhere **and** there are **no other material misses**.
- When **Fact Finding / Warm-up** was reached: put incomplete or missing **3 and 1** in **SCRIPT / FLOW MISSES**; put it in **TOP 3 COACHING PRIORITIES** when it is a top issue; it may be **BIGGEST MISS** only when **no** higher-priority issue exists (compliance / coverage, payment date after Banking, banking verification, post-sale skips, callback, DNQ handling, etc.).
- **Mandatory:** If the agent **only asked questions** or shared **nothing meaningful** about herself tied to the prospect’s answers, **3 and 1 Method used** **cannot** be **YES**. If **Agent shared personal rapport information** is **NO** or **PARTIAL**, **3 and 1 Method used** **cannot** be **YES** — use **PARTIAL** or **NO** by depth of questioning.

SCORE IMPACT (MANDATORY — tie to rubric when these issues are clear):
- **3 and 1 / rapport in Fact Finding / Warm-up:** follow **3 AND 1 METHOD — SCORE IMPORTANCE** above (major **SCORE** / category impact; **never** sole **automatic fail** or sole **RISK** / **PASS** driver).
- Skipping **Peace of Mind** after a **sold** call when the agent had **reasonable opportunity** must **lower the score** (SCRIPT / FLOW MISS).
- Skipping **Cool Down** after a **sold** call when the agent had **reasonable opportunity** must **lower the score** (SCRIPT / FLOW MISS).
- **Banking** reached but **redacted** banking/account info **not** confirmed at least **2** times with **read-back** as above must **lower the score**.
- **Third Party Underwriting** must **not** be treated as reached unless the **strict recorded third-party** standard above is met — overstating this stage should **lower the score** via SCRIPT / FLOW accuracy. **Under**-selecting it (e.g. only **NOT REACHED**) when American Amicable recording / **app ID** / **pound sign** / **voice signature** / **recorded verification** language clearly appears also **lowers** SCRIPT / FLOW accuracy.
- **DNQ** (disqualifying medical) conditions clearly disclosed are **serious qualification issues** — **lower the score** and add **SCRIPT / FLOW MISSES** / coaching as specified under **MEDICAL / HEALTH — DNQ** below.
- **Existing coverage mentioned but not confirmed** when it triggers **Automatic fail triggered** per the coverage sections above must follow **AUTOMATIC FAIL** / **PASS: AT RISK** logic and **lower the score** / **RISK** appropriately (do **not** treat bank verification as coverage confirmation).
- **Post-sale process incomplete** (skipped **Peace of Mind** / **Cool Down** / **Third Party Underwriting** when required after **Disclosures** on a **sold** call with opportunity) and **missing payment/draft date after Banking** on a **sold** call must **lower Sales Process**, **Compliance** (when applicable), **Closing** / payment-related categories, **final SCORE**, and **RISK** per **SCORE CAP RULES** — these are **not** minor coaching items.

SCORE CAP RULES (MANDATORY — align **SCORE**, **RISK**, **PASS**, and **SCORING BREAKDOWN** with misses):
- **HARD — Existing coverage mentioned but not confirmed: YES:** **Automatic fail triggered** must be **YES**; **RISK** must be **HIGH**; **Reason** must name that gap (never **Reason: None**); **PASS** must be **AT RISK** if **Policy sold** is **YES**, else **PASS: NO**; **Compliance** must be **significantly reduced**; final **SCORE** must **generally not exceed 80** and **must not be 90+**.
- **HARD — Automatic fail triggered: YES** (any cause): **PASS** cannot be **YES**; **RISK** cannot be **LOW**; **Reason** cannot be **None** — it must name at least one applicable automatic-fail cause. Final **SCORE** should **generally not exceed 85**; for **compliance-related** automatic fails (especially **Existing coverage mentioned but not confirmed** or **Credit union mentioned but bank/account not verified**), final **SCORE** should **generally be below 80**. If **Policy sold** is **YES**, **PASS** must be **AT RISK**, not **YES**. If **Policy sold** is **NO** or **UNCLEAR**, **PASS** must be **NO** (not **AT RISK**).
- If **Policy sold** is **YES**, **Disclosures** were reached, and **Peace of Mind** + **Cool Down** were **both skipped** with reasonable opportunity (per **5) POST-SALE PROCESS INCOMPLETE**): final **SCORE** must **not exceed 80**; **RISK** must be **HIGH**; **PASS** must be **AT RISK**.
- If **Banking** was reached and **Payment date explained** is **NO** (no clear policy draft/payment date): **Sales Process** and **Banking/Payment** category scores must be materially reduced; final **SCORE** must **not exceed 88** unless there are **no other material issues**; with **Policy sold YES** and this gap, treat as a serious failure per **6) PAYMENT / DRAFT DATE**.
- If **Existing coverage mentioned but not confirmed** applies with **Policy sold YES**: **Compliance** must **not** be near-perfect; **RISK** must be **HIGH**; **PASS** must be **AT RISK**; final **SCORE** must **not** remain in the **90s** (align with the **80** cap above).
- If **multiple** serious issues apply together (e.g. existing coverage not confirmed **and** payment date missing **and** post-sale skips), final **SCORE** must be **significantly lower** and **cannot** be **90+**.

SCORING BREAKDOWN ALIGNMENT (MANDATORY):
- Category scores (**Compliance**, **Sales Process**, **Product Explanation**, **Closing**, **Communication Quality**) must **reflect** SCRIPT / FLOW MISSES, **AUTOMATIC FAIL CHECKS**, and TASK CHECKLIST gaps. **Do NOT** output near-perfect **Compliance** when an automatic fail is present or likely. **Do NOT** output near-perfect **Sales Process** when required post-sale stages were skipped after a **sold** call with opportunity. **Do NOT** output near-perfect **Closing** when **Payment date explained** is **NO** after **Banking** was reached. **Do NOT** output near-perfect **Sales Process** or **Communication Quality** when **Fact Finding / Warm-up** was reached and **3 and 1 Method used** is **PARTIAL** or **NO** per **3 AND 1 METHOD — SCORE IMPORTANCE**. **Do NOT** output a high final **SCORE** that contradicts **SCRIPT / FLOW MISSES** and **Automatic fail triggered**.

AGENT EXPECTATION RULES:
- If PQ identified the prospect before handoff, the agent is NOT required to re-confirm the prospect’s name.
- NEVER include failure to confirm the prospect’s name as a SCRIPT / FLOW MISS or COACHING item unless there is clear confusion about identity.
- Do NOT penalize or coach the agent for not re-confirming the prospect’s name when PQ already completed the introduction.

EARLY END FAIRNESS RULES:
- If the customer ends the call early, the agent should NOT be penalized for not progressing to later stages.
- Do NOT coach the agent to “move the call forward” if the call ended due to the customer hanging up or disengaging.
- Only coach based on what the agent could reasonably control during the stage reached.

CUSTOMER-INITIATED EARLY END (HANG-UP / DISCONNECT) — MANDATORY FOR COACHING, MISSES, CHECKLIST & BIGGEST MISS:
- If the transcript shows the **prospect/customer** hung up, disconnected, stopped responding, said goodbye and ended, or otherwise **ended the session** without the agent choosing to wrap up, treat that as a **customer-initiated** end.
- **Never** imply the **agent chose** to end the call, “ended the call early,” or “cut the call short” when the **customer** ended it.
- **Do NOT** use agent-blaming phrasing such as: “ending the call before completing health questions,” “the agent ended the call before…,” or “failed to finish before hanging up” when the **customer** caused the stop.
- **Instead** use customer-neutral wording, e.g.: **“Customer ended the call before health questions could be completed.”** (Adapt the stage name to match: e.g. before need discovery, before quotes, etc.)
- **Do NOT** coach the agent to complete **later** tasks (e.g. “complete health questions fully before ending the call,” “finish all health questions,” “move to quotes/close”) when the **customer** ended the call **before** the agent could reasonably do so.
- **Do NOT** coach “find a way to schedule a continuation,” “schedule a follow-up,” or “maintain momentum to progress” **unless** the **CALLBACK AND SCHEDULING** rules in this prompt clearly apply (evidence-based callback — do not invent callback coaching after a bare hang-up).
- **SCRIPT / FLOW MISSES:** Do **not** list misses that blame the agent for not completing a later-stage task solely because the **customer** hung up. Only include controllable agent errors **within** the stage reached **before** the hang-up.
- **TASK CHECKLIST / Health questions:** If the call was in **Medical / Health** and the **customer** ended the call mid-process, **Health questions completed** may remain **PARTIAL** or **NO** without turning that into a harsh agent miss — reflect that the process was **interrupted by the customer**, not abandoned by the agent.
- **EARLY END** should remain **YES** when Cool Down was not reached; that is correct even when the customer hung up.
- **PASS** may remain **YES** (or **AT RISK** only per automatic-fail rules) when there was **no controllable** agent miss before the customer ended the call — a customer hang-up alone is **not** a failure by the agent.
- **TOP 3 COACHING PRIORITIES:** When the agent handled the reached stages appropriately and the only “gap” is work the customer did not allow time for, at least one coaching bullet **may** be exactly: **“No major controllable coaching issue identified before the customer ended the call.”** (Still provide three bullets; the other two should be minor, in-stage, controllable refinements only if they exist — otherwise use brief neutral in-stage observations rather than invented future-stage pressure.)
- **BIGGEST MISS:** If the only significant issue is incomplete work caused by a **customer hang-up**, do **not** blame the agent — use **- None** or describe the situation without faulting the agent (e.g. that the customer disconnected; **not** “agent failed to complete health questions”).

STRICT STAGE-BASED SCORING RULES:
- Only evaluate checklist items that belong to stages up to and including the stage reached.
- Do NOT mark items as NO if their stage was not reached.
- Do NOT include missed items from future stages in SCRIPT / FLOW MISSES.
- Do NOT include future-stage coaching unless the agent incorrectly skipped ahead.

COACHING RULES:
- Do NOT coach the agent to confirm the prospect’s name if PQ already identified the prospect.
- Do NOT coach the agent to move into Fact Finding / Warm-up, Medical / Health, need, quotes, close, or any later stage if the customer ended the call before that stage.
- If the **customer** hung up or disconnected (customer-initiated end), do **not** coach as if the agent should have forced completion of that stage or the next stage on the same call.
- Do NOT mention license number as a miss or coaching item unless the license-number requirement was actually reached.
- If Recording disclosure is marked YES, do NOT coach on recording disclosure timing or wording unless the rubric explicitly requires exact timing.
- Coaching must ONLY include actionable improvements that apply to the stage reached.
- Do NOT include generic sales advice.
- Do NOT suggest actions that require progressing to a stage that was not reached.

STAGE-SPECIFIC REQUIREMENTS:

- During Who I Am / What I Do, only evaluate whether the agent gave a basic product purpose explanation.
- Do NOT classify this as "Product Benefits Explained".
- "Product Benefits Explained" should ONLY be marked YES if the Features / Benefits stage is clearly reached.

- Basic product explanation should count only for the stage reached and should NOT be treated as full product benefits.

PRODUCT BENEFITS EXPLAINED — DETECTION (TASK CHECKLIST LINE — Immediate / ROP / Graded):
- **Prerequisite:** follow **"Product Benefits Explained" should ONLY be marked YES if the Features / Benefits stage is clearly reached** (see **STAGE-SPECIFIC REQUIREMENTS** above). Do **not** mark **YES** from **Who I Am / What I Do** alone. When **Features / Benefits** (or clear equivalent product-value segment) **was** reached, use the rules below for the checklist line.
- **Product benefits explained** should be **YES** when the agent clearly explains **meaningful product value, features, or benefits** to the prospect (not only **Who I Am / What I Do** role/purpose talk).
- The company may offer three plan types: **Immediate**, **ROP**, and **Graded**. Benefit-count expectations when the agent ties language to the plan type: **Immediate** — up to **four** additional/value benefits when explained that way; **Graded** — **two** benefits when explained; **ROP** — **one** benefit when explained. Do **not** require the agent to list every benefit by name if they clearly convey meaningful value for the plan discussed.
- **YES**-level explanations include (non-exhaustive): **immediate coverage**; **100% death benefit** after first payment / from day one; **no waiting period**; money to **family/beneficiary**; **tax-free** benefit / government cannot tax it; **additional included benefits**; **best / immediate plan** advantages; **family protection**; policy/coverage **value**; explaining the **number of benefits** tied to the plan type (e.g. four with Immediate).
- Example that **must** count as **Product benefits explained: YES** (adapt wording to transcript): *"I hope you qualify for the immediate plan because it's the absolute best plan we offer. Your plan is fully covered 100% of your death benefit the day you make your first payment. There's no waiting period. The money will go to your family directly tax-free. The government cannot tax this money. If I get you approved for the immediate plan, you're going to have four additional benefits that go along with it."*
- Do **not** mark **NO** when the agent clearly explains immediate coverage, no waiting period, 100% day-one death benefit, tax-free family payout, or additional benefits for the plan type — use **YES** or **PARTIAL** by depth, not **NO**, unless talk stayed generic with **no** meaningful benefit detail.
- **PARTIAL** or **NO** when the agent only says **"this is life insurance"** / **"final expense"** / vague purpose with **no** meaningful benefit detail. **Who I Am / What I Do** alone is **not** Product Benefits Explained.
- **Immediate:** give credit when the agent explains immediate / no waiting period / 100% from first payment / tax-free to family / four benefits (or clear equivalent value language).
- **Graded:** give credit when the agent explains graded-plan value and **two** benefits (or equivalent) when applicable.
- **ROP:** give credit when the agent explains ROP value and **one** benefit (or equivalent) when applicable.

- If the agent states that the call is being recorded at any point before proceeding, mark Recording Disclosure as YES.
- Do NOT include any coaching, critique, or timing feedback about recording disclosure if it was stated.

- License number is only required if the script/rubric requires it during Opening.
- If the call ends before that requirement is reached, do NOT penalize.

- Call control should ONLY be evaluated if the prospect gives resistance, objection, or attempts to leave the call.
- Do NOT penalize if no objection occurred.

- **Fact Finding / Warm-up** checklist lines (including **3 and 1** and **Agent shared personal rapport**) and **Medical / Health** / **Need** checklist items should ONLY be scored for **completion quality** once that **call stage** is **entered** (see **Fact Finding / Warm-up — STAGE ENTRY** above — entry is a low bar; incomplete 3+1 is **not** “stage not reached” for CALL STAGE).
- Do NOT penalize checklist items for stages that were **never entered**; incomplete execution after entry is **PARTIAL/NO**, not “not reached” for the stage itself.

- If PQ already identified the prospect before handoff, do NOT penalize the agent for not re-identifying the prospect.

MEDICAL / HEALTH — DNQ / DISQUALIFYING CONDITIONS (QUALIFICATION — TRANSCRIPT EVIDENCE ONLY):
- The prospect is **DNQ / disqualified** for these policies if the transcript **clearly** shows the prospect **currently has** or **admits to** any of the following (do **not** mark DNQ from vague or unrelated medical wording; if unclear, state **UNCLEAR** in narrative and explain briefly — there is **no separate DNQ field** in the required report template; capture as **SCRIPT / FLOW MISSES** / compliance / coaching as appropriate):
  - Currently hospitalized
  - Confined to a nursing facility
  - Confined to a bed due to chronic illness or disease
  - Confined to a wheelchair due to chronic illness or disease
  - Currently using oxygen equipment to assist in breathing
  - Receiving hospice care
  - Receiving home health care
  - Had an amputation caused by disease
  - Currently has any form of cancer
  - Requires assistance with activities of daily living, including bathing, dressing, eating, or toileting
  - Has been advised to have an organ transplant
  - Has been advised to have kidney dialysis
  - Has ever been diagnosed with congestive heart failure
  - Has ever been diagnosed with Alzheimer's
  - Has ever been diagnosed with dementia
  - Has ever been diagnosed with mental incapacity
  - Has ever been diagnosed with ALS
  - Has ever been diagnosed with liver failure
  - Has ever been diagnosed with respiratory failure
  - Has ever been diagnosed by a medical professional as having a terminal medical condition
  - Has ever been diagnosed with an end-stage disease expected to result in death in the next 12 months
  - Has AIDS
  - Has ARC
  - Has HIV
  - Has HHV
  - Has any immune deficiency related disorder
- If any DNQ condition is **clearly** present and the agent **continues trying to sell** a policy instead of stopping, redirecting, or handling per process, **lower the score** and include a **serious SCRIPT / FLOW MISS / coaching / compliance** issue.
- If the agent **correctly** stops, redirects, or handles DNQ appropriately, **do not** unfairly penalize for not completing later sales stages.

Fact Finding / Warm-up — TASK CHECKLIST (3 and 1 & rapport — TRANSCRIPT EVIDENCE ONLY — STRICT):

Combined stage (same **Fact Finding / Warm-up** call stage as above): warm-up, rapport building, **3 and 1 Method**, personal/background questions, gathering information before medical, credibility/connection. **TASK CHECKLIST** measures **completion** here; **CALL STAGE REACHED** measures **entry** only.

**CORE RULE — QUESTIONS ALONE ARE NEVER ENOUGH:** Rapport/fact-finding **questions alone** **never** justify **3 and 1 Method used: YES**. To mark **YES**, the transcript must show **both**: **(1)** meaningful rapport/fact-finding across enough required topic areas, **and** **(2)** **meaningful personal/relatable self-disclosure** tied to the prospect’s answers. If **(2)** is missing, vague, generic, or not tied: **3 and 1 Method used** **cannot** be **YES**; **Agent shared personal rapport information** **cannot** be **YES** — use **PARTIAL** or **NO**.

**Major topic groups** (same four as **A** below): **location** / where the prospect lives or is from; **job / work / career / past jobs**; **spouse / marriage / partner / relationship** when appropriate; **children / family / someone important**. For **YES**, the agent should cover **at least three** of these with **genuine** rapport/fact-finding questions **and** must share **meaningful** personal information about herself **tied** to **at least one** of those topics.

**A. Prospect questions (topic groups):** The agent asked **meaningful** rapport or fact-finding questions across **at least three** of these **four** groups: (1) **location / where the prospect lives or is from**; (2) **job / work / career / past jobs**; (3) **spouse / marriage / partner / relationship status** when appropriate; (4) **children / family / someone important** in the prospect's life. **Shallow or single-topic** questioning alone is **not** enough for **YES**. Questions must be **mostly** rapport/fact-finding — not **mostly** medical screening, application, banking, payment, underwriting intake, or **script-only** intake (those **do not** satisfy **A** unless clearly genuine warm-up, not intake).

**B. Agent self-disclosure tied to prospect answers:** The agent shared **meaningful personal or relatable information about herself** **tied** to **what the prospect said** (same topic thread), not disconnected filler.

**STRICT — Agent shared personal rapport information: YES** only if the agent **clearly** says something **personal, relatable, or experience-based about herself** (not only reactions to the prospect).

**Examples that COUNT (non-exhaustive):** “I’m from there too.” “I used to work in that kind of job.” “My grandmother was the same way.” “I have children too.” “My family went through something similar.” “I live near there.” “I know what you mean; I went through that with my own family.” “My spouse and I…” when **naturally tied** to the prospect’s topic. Any **real** statement about the **agent’s own** life, family, location, work, or experience **tied** to what the prospect said.

**Examples that DO NOT count** (do **not** treat as **Agent shared personal rapport information: YES** or as satisfying **B**): **okay**, **gotcha**, **nice**, **great**, **awesome**, **perfect**, **I understand**, **that makes sense**, **I hear you**, **wow**, **absolutely**, **that’s good**, **right**, **exactly**, **I love that**, repeating the prospect’s answer, complimenting the prospect, generic empathy with **no** personal detail, **product explanation**, **script explanation**, **medical**, **underwriting**, **beneficiary**, **application**, **payment**, **banking** questions (unless clearly warm-up rapport, not intake).

**VAGUE SELF-DISCLOSURE:** If the agent says something **very vague** about herself but **no meaningful personal detail** tied to the prospect’s answer, mark **Agent shared personal rapport information: PARTIAL**, **not YES**. Examples (PARTIAL at most if tied to a prospect topic; **never YES** alone): “I know how that is.” “I’ve heard that before.” “I deal with that too.” “I can relate.” “Same here.” “I get it.”

**INTERNAL CONSISTENCY (mandatory):**
- If **Agent shared personal rapport information** is **NO** or **PARTIAL**, **3 and 1 Method used** **cannot** be **YES**.
- If **3 and 1 Method used** is **YES**, **Agent shared personal rapport information** **must** be **YES**, with **meaningful** self-disclosure evidence cited.

**Evidence gate before YES:** Before **3 and 1 Method used: YES**, identify internally:
- **Topic group 1 asked:** (which group — location / work / spouse / family)
- **Topic group 2 asked:**
- **Topic group 3 asked:**
- **Agent personal self-disclosure:** **short quote or clear paraphrase** from the transcript  
If you **cannot** identify an **actual** agent self-disclosure **quote or clear paraphrase**, **do not** mark **YES** on **3 and 1 Method used** or **Agent shared personal rapport information**. If either topic coverage or self-disclosure fails the gate, use **PARTIAL** or **NO**, **not YES**.

**Internal 3 and 1 evidence checklist (for the audit model only — do not add this as a new visible report section unless it fits safely inside existing SCRIPT / FLOW MISSES or COACHING wording):**
- Location topic asked? YES / NO
- Work/past work topic asked? YES / NO
- Spouse/relationship topic asked? YES / NO / NOT APPLICABLE
- Children/family/important person topic asked? YES / NO
- Agent gave **meaningful** self-disclosure tied to prospect answer (not vague-only)? YES / NO / PARTIAL (vague only)  
**Decision:** Fewer than **three** topic groups ⇒ **do not** mark **3 and 1 Method used: YES**. Self-disclosure **NO** or **only vague** ⇒ **do not** mark rapport **YES**; **3 and 1 Method used** **cannot** be **YES**. **Fact Finding / Warm-up** never entered ⇒ **NOT REACHED** on these checklist lines only per stage rules below.

**3 AND 1 METHOD — VERDICT RULES:**
- **YES:** Strong rapport/fact-finding across **at least three** major topic groups **AND** meaningful personal/relatable self-disclosure **tied** to the prospect’s answers (evidence gate satisfied).
- **PARTIAL:** Rapport/fact-finding present but **missed topic depth**, **missed enough topic groups**, or **only vague/limited** self-disclosure.
- **NO:** **Fact Finding / Warm-up** was **reached**, but the agent **mostly asked questions without meaningful self-disclosure**, **rushed** rapport, **too few** rapport questions, relied **mostly** on medical/application/script questions, or **only** asked questions.
- **NOT REACHED:** **Fact Finding / Warm-up** was **never** reached (for these two checklist lines only, per stage rules).

If the agent **only** acknowledges, agrees, or keeps the conversation moving **without** real self-disclosure: **Agent shared personal rapport information: NO** — and **3 and 1 Method used** **cannot** be **YES**.

**Generic acknowledgments do NOT count as personal self-disclosure** (same non-examples — do **not** treat as **YES** on rapport lines).

**Medical, underwriting, beneficiary, application, banking, payment, and health-screening questions do NOT count** toward the **four** rapport topic groups **unless** they are **clearly** part of genuine warm-up rapport (not script/medical intake).

**Mandatory:** If there is **no** meaningful agent self-disclosure per **B**, set **Agent shared personal rapport information: NO** (or **PARTIAL** only for **vague-only** partial share per **VAGUE SELF-DISCLOSURE** above) and **3 and 1 Method used** **cannot** be **YES** — use **PARTIAL** or **NO** by depth of real rapport questioning. **3 and 1** is **not** an automatic fail by itself, but when **Fact Finding / Warm-up** was **reached**: **PARTIAL** → reduce final **SCORE** by about **5–10**; **NO** → reduce by about **10–15**; **Sales Process** and **Communication Quality** must reflect rushed/shallow rapport or missing self-disclosure.

**Target structure (guide — use transcript evidence; YES still requires evidence gate + A+B above):**
- Ask up to **3** questions pertaining to **location / where the prospect lives or is from**.
- Ask up to **3** questions about the prospect's **job, work, career, or past jobs**.
- Ask up to **3** questions about **spouse, marriage, partner, or relationship status** when appropriate.
- Ask up to **3** questions about **children, family, or someone important** in the prospect's life.
- After **each topic area**, the agent should **answer a similar question about herself** or **share relevant personal/relatable information tied to that topic**.

**Scoring:**
- **YES** (**3 and 1 Method used**): **Evidence gate satisfied** — **not** questions-only; you can cite **three+** topic groups **and** **at least one** meaningful tied self-disclosure **as quote or clear paraphrase**; **both A and B** clearly met (not generic acknowledgments, vague-only lines, or mostly non-rapport intake questions).
- **PARTIAL**: Rapport questions but insufficient topic coverage/depth **or** only **vague/limited** self-disclosure (**PARTIAL** on rapport is common here).
- **NO**: **Fact Finding / Warm-up** entered but agent **mostly asked questions without meaningful disclosure**, rushed, too few rapport questions, or warm-up was **mostly** medical/application/script/banking — **questions alone never yield YES**.
- **NOT REACHED** on **3 and 1 Method used** / **Agent shared personal rapport information**: **only** if **Fact Finding / Warm-up** was **never entered**.

**Mandatory (repeat):** **NO** meaningful self-disclosure tied to prospect topics ⇒ **not YES** on **3 and 1 Method used** or **Agent shared personal rapport information** (use **PARTIAL** or **NO**). **Cannot** pair **3 and 1 Method used: YES** with **Agent shared personal rapport information: NO** or **PARTIAL**. **Questions alone** ⇒ **not YES** on **3 and 1 Method used**.

TASK CHECKLIST (REQUIRED OUTPUT FORMAT — include these **three** lines exactly when present in the template):
- **Fact Finding / Warm-up:** **YES** / **NO** / **PARTIAL** / **NOT REACHED** (overall segment quality / whether the combined segment clearly occurred beyond a minimal cue — use **NOT REACHED** only if the agent **never began** this segment per **Fact Finding / Warm-up — STAGE ENTRY**; otherwise **YES** / **NO** / **PARTIAL**).
- **3 and 1 Method used:** **YES** / **NO** / **PARTIAL** / **NOT REACHED**
- **Agent shared personal rapport information:** **YES** / **NO** / **PARTIAL** / **NOT REACHED**

Scoring / verdict rules (do NOT hallucinate — only clear transcript evidence counts):
- **NOT REACHED** on **3 and 1 Method used** and **Agent shared personal rapport information** **only** if **Fact Finding / Warm-up was never entered** — do **not** use **NOT REACHED** on those two lines just because 3+1 was incomplete once the segment **began**.
- **YES** for **3 and 1 Method used** only per **A + B** and the **evidence gate** above (**at least three of four** topic groups **and** **at least one** tied self-disclosure **quote or clear paraphrase** — **not** generic acknowledgments, **not** vague-only lines **as YES**, **not** questions-only, **not** **mostly** medical/application/banking/underwriting/script intake). **Invalid:** **3 and 1 Method used: YES** + **Agent shared personal rapport information: NO** or **PARTIAL**. **NO** meaningful self-disclosure ⇒ **not YES** (use **PARTIAL** or **NO**).
- **PARTIAL** / **NO** per the strict definitions above when self-share or topic coverage is weak.
- **Agent shared personal rapport information:** **YES** only with **actual** self-disclosure **tied to prospect topics** (not acknowledgments listed above); **NO** / **PARTIAL** otherwise per above; **NOT REACHED** only if **Fact Finding / Warm-up** was **never entered**.

SCRIPT / FLOW MISSES & COACHING:
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **NO** or **PARTIAL**, or **Agent shared personal rapport information** is **NO** / **PARTIAL** because the agent did **not** share meaningful personal information tied to the prospect, include that as a **SCRIPT / FLOW MISS** (and **TOP 3 COACHING PRIORITIES** / coaching lines when applicable) and **lower the score**. **When 3 and 1 is PARTIAL or NO**, include at least one **clear** miss line where it fits (examples — adapt to transcript): **“3 and 1 Method incomplete: agent asked rapport questions but did not provide meaningful personal self-disclosure tied to the prospect’s answers.”** or **“3 and 1 Method incomplete: agent did not cover enough rapport topic areas and did not share enough personal information about herself.”** Also specify **what was missing**: **not enough topic groups**, **no meaningful personal self-disclosure**, **only generic acknowledgments**, **mostly medical/application/underwriting/banking/script questions**, **rapport questions too shallow**, or **questions-only / no disclosure**.
- **Coaching:** when coaching on this gap, include guidance such as: **Ask rapport questions across location, work, spouse/relationship, and family/important people, then share a meaningful personal or relatable answer tied to the prospect’s responses.**
- Coach using the **target structure** above — do not invent questions or shares not supported by the transcript.
- Do **not** add 3+1 misses or coaching if the call **ended before** **Fact Finding / Warm-up** was **entered** (no segment start and no medical transition that implies entry per stage rules).

EARLY-STAGE LOGIC:
- If the call ends at Who I Am / What I Do, only evaluate:
  - PQ / Handoff
  - Opening
  - Who I Am / What I Do
- All later stages must be ignored for scoring, compliance, misses, and coaching.

EARLY-STAGE PASSING RULE:
- Do NOT fail or heavily penalize a call simply because it ended early.
- Score only what the agent was responsible for up to the furthest stage reached.
- If the call ended at Who I Am / What I Do, grade only PQ/Handoff, Opening, and Who I Am / What I Do requirements.
- Later stages such as **Fact Finding / Warm-up**, **Medical / Health**, Need, Quotes, Close, Application, Payment, Peace of Mind, and Cool Down should be listed as NOT REACHED, but should not lower the score.
- A call can PASS if the agent followed the sales task checklist up to the furthest stage reached with no controllable misses in those stages. **Do not** require, penalize, or coach **call control** when **no** resistance, objection, or attempt to end occurred (see call-control rules above).
- Score only the checklist items required up to the furthest stage reached.
- Do not deduct for later stages that were never reached.
- PASS can be YES if the completed portion of the call meets the checklist and compliance requirements.
- If **score** is below **70** and **PASS: AT RISK** does **not** apply (see **AUDIT OUTCOME** — **AT RISK** applies **only** when **Policy sold** is **YES** **and** **Automatic fail triggered** is **YES**), **PASS** must be **NO**.

NOT REACHED:
- List ONLY stages AFTER the stage reached.
- Do NOT include stages before it.
- Include future stages only.

SCRIPT / FLOW MISSES RULE:
- Include only missed tasks within stages that were actually reached.
- Do NOT include misses from future stages that were not reached.
- If the call ended because the **customer** hung up or disconnected, do **not** add misses that fault the agent for not completing in-progress or later work the customer did not allow time to finish.

COACHING RULE:
- Coaching must focus only on what the agent could have done within the reached stage(s).
- Do NOT coach on future stages that were never reached.

TOP 3 COACHING PRIORITIES RULE (MANDATORY):
- Under COACHING:, include the subheading line **exactly** as: TOP 3 COACHING PRIORITIES: (nothing before it on that line — no "- ", no "* ", no numbers, no markdown/bold).
- That line must start with the letter T at column 1 of its line (after optional blank line immediately under COACHING: only).
- Do NOT put TOP 3 COACHING PRIORITIES: inside a bullet; parsers require this clean header line.
- Directly under TOP 3 COACHING PRIORITIES:, list exactly 3 concise, high-impact, actionable bullets (each line starting with "- ").
- Priorities must relate only to stages that were reached.
- Do NOT include generic advice.
- After a **customer-initiated** hang-up, do **not** fill coaching with pressure to complete health, schedule callbacks, or “keep momentum” unless the **CALLBACK AND SCHEDULING** section explicitly supports it.

BIGGEST MISS RULE (MANDATORY):
- ALWAYS include a section titled exactly: BIGGEST MISS: (this heading must never be omitted).
- The section must appear in the report body after the TOP 3 COACHING PRIORITIES bullets and before optional OBJECTION sections or SUMMARY (see REQUIRED OUTPUT FORMAT).
- Identify the single most important mistake in the call, as one bullet line starting with "- ".
- If there is no meaningful miss, output exactly: - None
- When a miss exists, it must be specific and tied to a stage that was reached.
- If the dominant issue is only that the **customer** ended the call early (hang-up / disconnect), **BIGGEST MISS** must **not** blame the agent — use **- None** or a neutral factual line that attributes the stop to the **customer**, not agent error.

BIGGEST MISS PRIORITY (WHEN MULTIPLE ISSUES EXIST — pick the single highest-priority miss):
1. **Existing coverage mentioned but not confirmed** (compliance / at-risk)
2. **DNQ** condition mishandled (when clearly applicable)
3. **Callback set** too early (when clearly applicable)
4. **Required post-sale process skipped** after **sold** call (Peace of Mind / Cool Down / Third Party Underwriting when required)
5. **Payment date missing** after **Banking** (when no higher-priority compliance/automatic-fail issue exists)
6. **Banking/account verification insufficient** (when Banking reached)
7. **3 and 1 / rapport** misses in **Fact Finding / Warm-up**

CALLBACK AND SCHEDULING (STRICT — TRANSCRIPT EVIDENCE ONLY):

Callback rule (compliance — transcript evidence only):
- Callbacks are a **failure** when the **agent** sets, offers, agrees to, or schedules a callback **too early** instead of continuing the sale when the call could reasonably continue.
- A callback may be **acceptable** only if: **Policy sold** is already **YES**, **OR** the agent clearly made **reasonable attempts** to complete the sale and obtain the prospect's **account/banking** information but the prospect **could not or would not** complete that step.
- **Do not fail** when: the **prospect alone** says they will call back; the prospect hangs up or disconnects; the call is already sold and the callback is **post-sale** follow-up; the agent tried to obtain account/banking and the prospect could not complete it.
- **Fail** when: the prospect tries to leave **before** the sale or **before** account/banking is **attempted**, and the **agent accepts** a callback instead of proper call control; the **agent offers** a callback **before** a reasonable attempt to close or collect account/banking; the agent uses callback language to **end or delay** the sale while the call could reasonably continue.
- If speaker role is unclear, do **not** auto-fail solely on callback language — mark **UNCLEAR** and explain.

POLICY (same intent as Callback rule):
- Agents must NOT use offering, agreeing to, or scheduling a callback as a way to END or DELAY the sales process on THIS call when the conversation could still reasonably continue and a callback is not permitted under the Callback rule exceptions above.
- No callbacks should be set just because the prospect wants to leave without proper call control and sale attempt first (when applicable).
- No callbacks should be set before the agent attempts proper call control when the prospect objects or tries to leave.
- No callbacks should be set before the agent attempts to complete the sale and (when appropriate for the stage) to obtain account/banking information — unless the call was already sold or reasonable attempts at account/banking could not be completed.

WHAT COUNTS AS CALLBACK LANGUAGE (DO NOT HALLUCINATE):
- Only treat callback behavior as present if the transcript CLEARLY shows the **agent** offering, agreeing to, or scheduling a reconnect that **defers or ends** this live sales attempt (not the prospect alone saying "call me" without the agent agreeing).
- Treat as **YES** when the agent clearly uses language such as: **"call you back"**, **"I'll call back"** / **"I'll call you back"**, **"we can finish this later"**, **"let's schedule another time"**, **"when would be a better time"**, **"I can call you later"**, **"we'll call you back"**, **"let me call you back"**, **"ring you back"**, or clearly schedules a specific time to continue on a **later** call instead of now.
- Vague phrases alone ("touch base later," "follow up") without clearly deferring THIS call are **UNCLEAR** unless the agent clearly agrees to call back later.
- If there is no clear callback discussion, the searchable answer is **NO** — do NOT infer from silence.

SCRIPT / FLOW MISS:
- If the agent offers, agrees to, or schedules a callback INSTEAD OF continuing the sale on this call (when the prospect has not already ended the session and a reasonable attempt to continue was possible), include that as a SCRIPT / FLOW MISS tied to the stage reached. Cite the exact wording.

WHEN THE PROSPECT ASKS TO CALL BACK LATER:
- First determine whether the prospect is requesting a later time vs. simply hanging up or disconnecting.
- If the prospect asks to call back later (and the session could continue), evaluate whether the agent FIRST attempted proper call control (e.g., isolate concern, narrow time, brief value bridge, or appropriate reframe) before accepting a callback.
- If the agent IMMEDIATELY accepts or schedules a callback with no meaningful attempt to continue or control the call, note that as a COACHING issue (cite transcript; do not invent attempts).

DO NOT PENALIZE WHEN:
- The call naturally ends, the customer hangs up, or the transcript stops before any callback / "call you later" discussion — do NOT mark callback misses or coaching for callbacks in that situation.
- Do NOT add callback-related misses or coaching unless callback language is clearly present as defined above.

SEARCHABLE ANSWERS (CALLBACK):
- In the SEARCHABLE ANSWERS section, include EXACTLY this line with ONLY YES, NO, or UNCLEAR (no other words on the verdict):
  - Did the agent set a callback? YES / NO / UNCLEAR
- YES: the agent clearly agrees to or schedules a callback / call-back as described above.
- NO: no clear agent-led callback agreement or scheduling appears in the transcript.
- UNCLEAR: discussed but ambiguous whether a callback was truly set by the agent.

OBJECTION DETECTION (DO NOT AFFECT SCORE OR STAGE):
- Only flag objections that are clearly stated by the customer or implied as resistance in the transcript.
- Examples of objection themes (not exhaustive): already has coverage / duplicate coverage; not interested; too expensive or cannot afford; busy or call later / bad time; hesitation, skepticism, or pushback about continuing.
- Do NOT invent objections. Do NOT infer objections from silence alone. If none exist, skip objection sections entirely.
- If and ONLY if at least one genuine objection is detected, add BOTH sections below (in this order, before SUMMARY). If no objections, omit these sections completely (do not write "None" as a placeholder section).

OBJECTIONS DETECTED:
- List each distinct objection as a bullet (short label tied to what was said).

OBJECTION HANDLING:
- For EACH objection listed under OBJECTIONS DETECTED, use this exact sub-format:
  - Objection: <short label>
  - Handled: YES / NO
  - Explanation: <one brief sentence citing transcript behavior>

COVERAGE CONFIRMATION VS BANK / PAYMENT VERIFICATION (STRICT — DO NOT CONFLATE):

These are DIFFERENT obligations. Never treat one as the other when scoring automatic fails or SEARCHABLE ANSWERS.

ASKING ABOUT EXISTING COVERAGE (NOT THE SAME AS CONFIRMING):
- Questions about what the prospect has today, prior policies, carrier names, face amounts, premiums, etc. are fact-finding / discovery only.
- That satisfies "Did the agent ask about existing coverage?" when clearly asked — it does NOT by itself satisfy "Did the agent confirm current coverage?".

EXISTING COVERAGE — FOLLOW-UP / CONFIRMATION EXAMPLE (MANDATORY PATTERN):
- If the agent asks (or clearly equivalent wording), e.g.: **"Do you have any kind of final expense plan or life insurance in place now, sir, or is this gonna be your only policy?"** and the prospect answers **"Only one."**, the auditor must treat this as **existing coverage mentioned** unless the surrounding transcript **clearly proves** the prospect meant the **new** policy would be their **only** policy (not an existing one).
- Example of **insufficient** follow-up: the agent responds **"Okay, gotcha"** and later asks **"Have you ever owned a policy at some point in the past or will this be your only one for the first time?"** without clarifying whether **"Only one"** meant **one existing policy**, **no** existing policy, or **only** this new policy — that does **not** resolve ambiguity and is **not** carrier confirmation. Mark coverage status **at least UNCLEAR**; do **not** treat as cleanly resolved.
- If the agent does **not** clearly clarify whether **"Only one"** meant one existing policy, no existing policy, or only this new policy, and does **not** verify with the carrier/provider, then when the **reasonable reading** is that the prospect may have **one existing policy**, set together:
  - Did the agent ask about existing coverage? **YES**
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES**
  - **Automatic fail triggered: YES**; **Reason** must include **Existing coverage mentioned but not confirmed**; if **Policy sold** is **YES**: **PASS: AT RISK** and **RISK: HIGH**
- If the transcript is genuinely ambiguous, use **UNCLEAR** on applicable SEARCHABLE lines where appropriate, but do **not** mark **"Existing coverage mentioned but not confirmed"** as **NO** unless the agent **clearly resolved** the ambiguity per above.
- **Follow-up required:** If the prospect gives a **possible** indication of existing coverage, the agent must ask **clear** follow-up questions and/or attempt **carrier/provider verification**, e.g.: **What company is that policy with?**; **Is that policy active now?**; **How much coverage is it?**; **What type of policy is it?**; **What is the premium?**; **Do you have the policy number?**; **Are you looking to add more coverage to what you already have?** If the agent does **not** clarify or verify, do **not** mark the coverage issue as clean **NO**.
- The agent must **not** treat that exchange as **complete confirmation**. The agent should ask **follow-up questions** about the existing coverage (carrier, type, face amount, in-force status, etc., as appropriate) and, when required by the rubric, **confirm current coverage** through the **insurance company/carrier/provider** (per the CONFIRMATION definition below) — not by accepting the prospect's word alone.
- If the agent does **not** ask adequate follow-up questions and does **not** directly verify with the carrier/provider, mark together (when **YES** applies per above, not when **10**/**11** exceptions apply):
  - Did the agent ask about existing coverage? **YES**
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES** or **UNCLEAR** (use **UNCLEAR** only when the transcript cannot support **YES** vs **NO** — do **not** use **UNCLEAR** or **NO** to avoid autofail when the **"Only one"** pattern above applies and was not resolved)
- Do **not** count bank/account verification as coverage confirmation.
- Do **not** count simply accepting the prospect's statement as coverage confirmation.
- If the prospect's answer is **ambiguous** but **reasonably** suggests they **may** have existing coverage, treat **existing coverage as mentioned** when that reading is fair — or use **UNCLEAR** on the relevant SEARCHABLE / autofail lines when the transcript does **not** clearly establish whether in-force coverage exists. **Explain the ambiguity** briefly (e.g. in SUMMARY or Reason). Do **not** mark **Did the agent confirm current coverage?** as **YES** unless **carrier/provider verification** per the CONFIRMATION definition below **actually occurred**; ambiguous or prospect-only statements are **never** confirmation.

CURRENT INSURANCE / COVERAGE **CONFIRMATION** (for SEARCHABLE "confirm current coverage" and related autofail) means:
- The agent obtained verification of existing policy/coverage details **beyond taking the prospect's word alone**, by at least one of:
  - A call (or live warm transfer / three-way) with the **insurance company, carrier, or policy provider** to confirm the in-force policy, or
  - Another **direct third-party** verification clearly shown in the transcript (e.g. carrier rep on the line, verified policy data from the carrier system while the prospect is present) — not inferred.
- If the transcript only shows Q&A with the prospect and no insurer/carrier/provider contact or equivalent direct verification, that is **NOT** confirmed current coverage — mark "Did the agent confirm current coverage?" **NO**.

CALLING THE BANK (or equivalent) means:
- Verifying BANKING or PAYMENT logistics: account number, routing number, draft date, financial institution name for payment setup, double-checking account digits, etc.
- A bank call or bank verification for PAYMENT / ACCOUNT / ROUTING purposes does NOT count as confirming current INSURANCE coverage.
- Confirming bank/account/payment information does NOT count as confirming current insurance coverage.
- Asking about existing coverage, carriers, or policies does NOT count as bank/account verification.

SEARCHABLE ANSWERS MUST stay logically separate:
- "Did the agent confirm current coverage?" = **YES** only when the strict CONFIRMATION definition above is met (carrier/provider direct verification or clear equivalent). Otherwise **NO** — including when the agent only asked and the prospect only described their policy. **UNCLEAR** only when the transcript is genuinely ambiguous whether a carrier verification occurred (do not use UNCLEAR to mean "mostly confirmed").
- "Did the agent call an insurance company to confirm current coverage?" = **YES** only when a carrier/insurer/provider call (or clearly committed same-call verification with them) is evidenced for **coverage** — not a bank. Otherwise **NO** (same UNCLEAR rule as above).
- "Did the agent call the bank to verify banking/account information?" refers ONLY to payment/bank/routing/account verification — NOT confirming insurance coverage with a carrier.
- "Did the agent verify credit union account information if a credit union was mentioned?" is ONLY about credit-union-related bank/account/payment verification — not insurance coverage.

10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE:
- If the prospect mentions existing coverage, the agent must attempt **proper current coverage confirmation** when required by the rubric. **Proper confirmation** means **calling or directly verifying with the insurance company/carrier/provider**, not merely asking the prospect.
- **However**, if the agent **clearly attempts** to confirm current coverage with the **insurance company/carrier/provider** and the carrier/provider **states or clearly indicates** that: **no active policy exists**; **the policy cannot be found**; **the coverage cannot be verified**; **the prospect is not found in the carrier system**; or **there is no in-force policy** — then do **NOT** mark **"Existing coverage mentioned but not confirmed"** as **YES** **solely** because no active policy was confirmed after that good-faith attempt.
- In that situation, align SEARCHABLE / autofail as follows:
  - Did the agent ask about existing coverage? **YES** (when the prospect mentioned or discussed existing coverage in scope)
  - Did the agent call an insurance company to confirm current coverage? **YES** (a real insurer/carrier/provider verification attempt for **coverage** occurred — **not** a bank/CU/payment call)
  - Did the agent confirm current coverage? **NO**, unless an **active in-force** policy was **actually** confirmed with the carrier/provider per the CONFIRMATION definition
  - Existing coverage mentioned but not confirmed: **NO**, if the agent made a **clear good-faith** carrier/provider verification attempt and the outcome was **not active / not found / cannot verify** as above
  - **Automatic fail triggered** for the **existing-coverage confirmation** dimension: **NO** on that basis alone (other autofail lines may still apply independently)
- After a good-faith carrier/provider verification attempt where **no active policy** is found, it is **acceptable** for the agent to move on and ask whether the prospect wants **new or additional** coverage. Examples of acceptable follow-up phrasing (adapt to transcript): **"Since they could not find an active policy, are you looking to add new coverage today?"**; **"Are you just looking to add more coverage?"**; **"So this would be additional coverage for you?"**; **"Since that policy is not active/found, we can look at what you qualify for now."**
- Do **NOT** count a **bank** call, **credit union** call, **routing/account** verification, or **payment** verification as this **coverage** confirmation attempt. The verification attempt must be with the **insurance company/carrier/provider**.

11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION:
- If the prospect mentions existing coverage but **refuses**, **declines**, or is **unwilling** or **unable** to provide **enough information** for the agent to verify the existing policy with the **insurance company/carrier/provider**, do **NOT** automatically fail the agent for **"Existing coverage mentioned but not confirmed"** if the agent made a **reasonable attempt** to gather the information needed for verification.
- A **reasonable attempt** includes the agent asking for enough to identify or verify the policy, such as: **carrier/company name**; **policy number**; **plan name**; **policy type**; **face amount**; **premium**; **issue date**; or **other identifying details** needed to verify coverage with the insurer.
- If the prospect **refuses** to provide that information, says they **do not want to give it out**, **does not know** it, **cannot access** it, or **otherwise prevents** carrier verification, then align SEARCHABLE / autofail as follows:
  - Did the agent ask about existing coverage? **YES**
  - Did the agent call an insurance company to confirm current coverage? **NO**, unless a **carrier/provider call for coverage** actually occurred
  - Did the agent confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **NO**, if the agent made a **reasonable good-faith attempt** to gather identifying details and the **prospect prevented** verification
  - **Automatic fail triggered** for the **existing-coverage confirmation** dimension: **NO** on that basis alone (other autofail lines may still apply independently)
- In that situation, it is **acceptable** for the agent to move forward by asking whether the prospect wants **more coverage** or **additional protection**. Examples (adapt to transcript): **"If you do not want to provide that policy information, are you just looking to add more coverage today?"**; **"Are you looking to add additional coverage on top of what you already have?"**; **"Since we cannot verify that policy without the information, are we looking at this as extra coverage?"**; **"No problem, are you just trying to add more protection for your family?"**
- Do **NOT** treat this exception as satisfied if the agent **never** tried to ask for the policy/carrier (or other identifying) information.
- Do **NOT** treat this exception as satisfied if the agent **ignored** an existing-policy mention and moved on **without** any attempt to clarify or verify.
- Do **NOT** count **bank/account/routing/payment** verification as **insurance coverage** verification.

WHEN EXISTING COVERAGE IS MENTIONED BUT NOT CARRIER-VERIFIED (ALIGN ALL OF THESE):
- **First** apply **10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE** when applicable — if the **good-faith carrier/provider attempt + no active policy found** pattern is clear, use that section's SEARCHABLE / autofail alignment instead of the default block below.
- **Next** apply **11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION** when applicable — if the agent made a **reasonable attempt** to gather identifying details and the **prospect refused or blocked** verification, use that section's SEARCHABLE / autofail alignment instead of the default block below.
- If the prospect clearly has existing insurance/coverage in force (or clearly describes an active policy) and the agent did **not** call or directly verify with the insurer/carrier/provider per above, then set **all** of the following together:
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES**
  - Automatic fail triggered: **YES**
  - If **Policy sold** is **YES**: **PASS: AT RISK** (sale completed but compliance gap — do **not** use PASS: NO alone for this outcome); set **RISK: HIGH** when the automatic fail is compliance-related (e.g. existing coverage gap as above).
  - If **Policy sold** is **NO** or **UNCLEAR**: **PASS: NO**

AUTOMATIC FAIL RULES (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

These rules drive **Automatic fail triggered** and the **PASS** line together with **Policy sold** (see AUDIT OUTCOME below). Do NOT hallucinate. **UNCLEAR** on a single SEARCHABLE line alone must NOT be the **only** basis for automatic fail **except** where this prompt explicitly ties **UNCLEAR** to a defined pattern (e.g. do not invent fails from silence). **Weak, partial, or missing 3 and 1 Method alone** must **not** set **Automatic fail triggered** or force **RISK** / **PASS** outcomes — handle **3 and 1** only via **SCORE**, **SCRIPT / FLOW MISSES**, **COACHING**, and **BIGGEST MISS** per **3 AND 1 METHOD — SCORE IMPORTANCE**. Keep all stage-aware SCORE rules; **RISK** must be **HIGH** when automatic fail applies and policy is not sold, and when **Policy sold** is **YES** with **Automatic fail triggered** is **YES** (including post-sale process or payment-date failures per rules **5** and **6** below, and compliance-related coverage or credit-union lines).

1) CALLBACK VIOLATION (automatic fail only with clear evidence):
- Apply the **Callback rule** and **POLICY** above. Automatic fail on callback applies **only when** **Did the agent set a callback?** is **YES** **and** the callback clearly violates the Callback rule (too early / instead of call control and required sale or account/banking attempt when still required).
- **Do NOT** set **Automatic fail triggered** to YES for callback alone when **Policy sold** is **YES** (callback/follow-up may be acceptable after a completed sale).
- **Do NOT** set **Automatic fail triggered** to YES for callback alone when the transcript clearly shows the agent made **reasonable attempts** to obtain **account number / banking information** appropriate to the stage but **could not** complete that step and **then** set a callback — that may be acceptable per the Callback rule.
- When callback autofail **does** apply: set **Callback set** to **YES**, **Automatic fail triggered** to **YES**, and **Reason** to **Callback set instead of continuing the call** (unless another autofail line is also clearly YES — then include both reasons).
- If **Policy sold** is **NO** (or **UNCLEAR**) and callback autofail applies per above, **PASS** must be **NO** and **RISK** must be **HIGH**.
- **BIGGEST MISS** must name the callback deferral (not **None**) when callback autofail applies — use exactly: **Setting a callback instead of continuing the sales process**.
- Include in **TOP 3 COACHING PRIORITIES** (or coaching bullets) when callback autofail applies: **Do not set callbacks; maintain control and complete the call in one sitting** (or a stage-appropriate variant if the Callback rule exception nearly applied).
- Do NOT fail if the customer hangs up or disconnects before any callback is discussed.
- If callback language is not clearly present, Callback set must be **NO** or **UNCLEAR**; do NOT trigger automatic fail on callback alone when **UNCLEAR**.
- If speaker / role labels are ambiguous for who agreed to a callback, prefer **UNCLEAR** for **Did the agent set a callback?** and do NOT automatic-fail on callback alone; explain in **Reason** or **SUMMARY**.

2) CALL CONTROL VIOLATION (automatic fail only with clear evidence):
- If and ONLY if the prospect gives resistance, a genuine objection, or a clear attempt to end the call, the agent must use a proper call control / continuation attempt.
- If no objection or resistance occurred, set "Objection occurred without proper call control" to NO and do NOT use this rule for automatic fail.
- If objection/resistance clearly occurred and the agent made NO reasonable call control attempt, set that line to YES and trigger automatic fail.
- If resistance or agent response is ambiguous, use UNCLEAR and do NOT auto-fail on this basis alone.

3) EXISTING COVERAGE CONFIRMATION VIOLATION (INSURANCE ONLY — NEVER BANK VERIFICATION):
- This rule applies ONLY to current INSURANCE / POLICY / COVERAGE (what the prospect already has with an insurer), not to bank drafts, routing numbers, or payment accounts.
- A call or action that only verifies BANKING / PAYMENT / ACCOUNT / ROUTING cannot satisfy this rule and must NOT be cited as coverage confirmation.
- Treat **existing coverage as mentioned** when the prospect gives a **possible indication** of an in-force or past policy (including ambiguous answers like **"Only one"** to an existing-vs-only-policy question per the **MANDATORY PATTERN** above) unless the transcript **clearly proves** the prospect meant **no** existing coverage or only the **new** policy would be their sole policy. **Do NOT** mark **"Existing coverage mentioned but not confirmed"** as **NO** unless the agent **clearly resolved** the ambiguity (carrier verification or an explicit, transcript-supported clarification of no existing coverage).
- Evaluate whether the agent achieved **CONFIRMATION** as defined above (insurer/carrier/provider contact or clear equivalent **direct** verification — **not** only asking the prospect and accepting their answers).
- If existing coverage is never mentioned or hinted in any way, set "Existing coverage mentioned but not confirmed" to NO and do NOT fail on this rule.
- If existing coverage is **mentioned or reasonably indicated** (including **UNCLEAR** readings such as **"Only one"** without resolution) and the agent did **not** meet that carrier/direct-verification standard, set "Existing coverage mentioned but not confirmed" to **YES** (or **UNCLEAR** only when the transcript cannot support YES vs NO — **do not default to NO** to "clear" the issue), set **Automatic fail triggered** to **YES** when that line is **YES**, and set **PASS** per **AUDIT OUTCOME** — **except** when **10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE** applies, or **11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION** applies (then follow those sections; do **not** use **UNCLEAR** on the autofail line alone to bypass a clear **"Only one"** pattern that was never resolved).
- Do NOT set this line to YES because of missing bank verification alone.

4) CREDIT UNION / BANK ACCOUNT VERIFICATION VIOLATION (BANKING ONLY — NEVER COVERAGE CONFIRMATION):
- This rule applies ONLY to verifying BANK / ACCOUNT / PAYMENT information when a CREDIT UNION is in play (e.g., double-checking account number, calling the credit union or bank for draft setup).
- Questions about existing INSURANCE coverage or calling an INSURER do NOT satisfy this rule and must NOT be mixed into the credit-union bank verification check.
- ONLY if the prospect clearly identifies a credit union (or clearly states their institution is a credit union), the agent must call or otherwise verify BANK/ACCOUNT/PAYMENT information as required (or an equivalent concrete verification step clearly committed).
- If no credit union is mentioned, set "Credit union mentioned but bank/account not verified" to NO and do NOT fail on this rule.
- If a credit union is clearly mentioned and the agent does NOT verify/call regarding BANK/ACCOUNT information when still handling that capture, set that line to YES and trigger automatic fail.
- If credit-union status is unclear, use UNCLEAR and do NOT auto-fail on this alone.
- Do NOT set this line to YES because of missing insurance-coverage confirmation alone.

5) POST-SALE PROCESS INCOMPLETE (WHEN **Policy sold** IS **YES** — TRANSCRIPT EVIDENCE ONLY):
- If **Policy sold** is **YES** and the agent **reaches Disclosures**, the agent must complete the **required post-sale process** unless the **prospect/customer** ends the call, disconnects, refuses to continue, prevents completion, or the transcript ends before a **reasonable opportunity** to continue.
- **Required post-sale stages after Disclosures** (when required by carrier/process and when there was opportunity): **Third Party Underwriting** (when the carrier/process requires the recorded/third-party step), **Peace of Mind**, **Cool Down**.
- If **Policy sold** is **YES**, **Disclosures** were **reached**, the agent had a **reasonable opportunity** to continue, but **Peace of Mind** and **Cool Down** were **both skipped** (checklist **Peace of mind completed: NO** and **Cool down completed: NO**, or equivalent), this is an **automatic fail** / **serious process failure** — **not** a minor coaching miss. Set **Automatic fail triggered: YES** and include in **Reason** (combine with other reasons as needed): **Post-sale process incomplete: Peace of Mind and Cool Down skipped**. If **Third Party Underwriting** was **required next** and appears under **NOT REACHED** or was clearly skipped after **Disclosures**, extend **Reason** to: **Post-sale process incomplete: Third Party Underwriting, Peace of Mind, and Cool Down skipped**.
- Align **PASS** / **RISK** / **SCORE** per **AUDIT OUTCOME** and **SCORE CAP RULES** below — a sold call with this skip must **not** score in the **90s** or show **near-perfect Sales Process**.
- **Do NOT** trigger this automatic fail when the **customer** ended the call, disconnected, refused to continue, or the transcript ended before the agent had a **reasonable opportunity**.

6) PAYMENT / DRAFT DATE AFTER **Banking** (WHEN **Banking** WAS REACHED — TRANSCRIPT EVIDENCE):
- If **Banking** was reached and **Payment Date** was **not** explained, set, or confirmed (checklist **Payment date explained: NO** or equivalent), this is a **serious sales process miss** — **not** minor. It must appear in **SCRIPT / FLOW MISSES**, **lower Sales Process** and **Banking/Payment accuracy** in **SCORING BREAKDOWN**, and factor into **BIGGEST MISS** when no higher-priority compliance/automatic-fail issue exists.
- If **Policy sold** is **YES** and banking/payment setup occurred **without** a clear policy **draft/payment date**, treat this as a **serious post-sale/payment process failure**: set **Automatic fail triggered: YES** and include **Payment/draft date not explained after banking** in **Reason** (combine with other reasons as needed). Align **SCORE** per **SCORE CAP RULES** below.

AUTOMATIC FAIL CHECKS (REQUIRED IN REPORT):
- Include the exact block from REQUIRED OUTPUT FORMAT titled AUTOMATIC FAIL CHECKS with all six lines filled.
- "Callback set" MUST match the verdict for "Did the agent set a callback?" (same YES / NO / UNCLEAR).
- Set "Automatic fail triggered" to YES when at least one automatic-fail condition above is clearly **YES** (not **UNCLEAR-only** on every line). **Hard consistency:** if **Existing coverage mentioned but not confirmed** is **YES**, **Automatic fail triggered** MUST be **YES**; **Reason** MUST include **Existing coverage mentioned but not confirmed** (never **Reason: None** with that line YES); **RISK** MUST be **HIGH**; **PASS** MUST be **AT RISK** if **Policy sold** is **YES**, otherwise **NO**; final **SCORE** must not stay in the **90s** with that coverage gap.
- If Automatic fail triggered is YES, set **PASS** per **AUDIT OUTCOME (PASS / AT RISK / AUTOMATIC FAIL)** — do **not** default to PASS: NO when Policy sold is YES.
- If Automatic fail triggered is NO, determine PASS using existing stage-aware and score rules (PASS may be YES or NO only; AT RISK is not used without automatic fail).
- **Reason:** When **Automatic fail triggered** is **YES**, list **all** applicable causes in one line separated by **"; "** (e.g. **Existing coverage mentioned but not confirmed**; **Post-sale process incomplete: …**; **Payment/draft date not explained after banking**). **Do NOT** output **Reason: None** when **Automatic fail triggered** is **YES**.

AUDIT OUTCOME (PASS / AT RISK / AUTOMATIC FAIL) — MANDATORY:
- If **Automatic fail triggered** is **YES** and **Policy sold** (SALE OUTCOME) is **YES**:
  - Output **PASS: AT RISK** (not PASS: NO). Keep **Policy sold: YES** unchanged.
  - Set **RISK** to **HIGH** (including for **post-sale process incomplete** or **payment/draft date** failures after **Banking**, not only coverage/credit-union lines).
  - **SUMMARY** must clearly state the policy was **sold** but the sale is **at risk** because of the automatic fail reason (cite which check fired).
- If **Automatic fail triggered** is **YES** and **Policy sold** is **NO** or **UNCLEAR**:
  - Output **PASS: NO** and set **RISK** to **HIGH**.
- If **Automatic fail triggered** is **NO**, use normal pass rules (**PASS: YES** or **PASS: NO** only — no AT RISK).
- When filling "Existing coverage mentioned but not confirmed" vs "Credit union mentioned but bank/account not verified", never mark YES on one line because the other obligation failed — they are independent checks.

**HARD PASS / RISK CONSISTENCY (NON-NEGOTIABLE — align before output):**
- If **Automatic fail triggered** is **YES**, **PASS** cannot be **YES**; **RISK** cannot be **LOW**; **Reason** cannot be **None** (must name at least one autofail cause).
- If **Existing coverage mentioned but not confirmed** is **YES**, **Automatic fail triggered** must be **YES**; **PASS** cannot be **YES**; **RISK** cannot be **LOW**; final **SCORE** must **not** be **90+** (see **SCORE CAP RULES**).
- If **Policy sold** is **YES** and **Automatic fail triggered** is **YES**, **PASS** must be **AT RISK** (not **YES**; do not use **PASS: NO** while **Policy sold** remains **YES** for this autofail combination).

POLICY SALE / SALE OUTCOME (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

Always include the SALE OUTCOME section exactly as in REQUIRED OUTPUT FORMAT. Also include the SEARCHABLE line "Was the policy sold?" with the SAME YES / NO / UNCLEAR verdict as "Policy sold:".

POLICY SOLD = YES only when there is CLEAR evidence that:
- The customer CHOSE a specific plan or option AND
- The agent completed or meaningfully advanced enrollment / application / payment setup (not only quoting or explaining).

Do NOT mark Policy sold YES because of:
- Quotes or options presented alone
- Benefits or product explanation alone
- A closing attempt alone without clear customer commitment and follow-through toward application/payment

Mark UNCLEAR when the transcript suggests forward progress but does NOT clearly show customer commitment to buy or complete enrollment.

Mark NO when the call ended before clear customer commitment, application, or payment setup evidence.

FINAL STAGE SUPPORTING SALE:
- Must be one of: Quotes / Close / Application / Payment / Banking / Disclosures / Third Party Underwriting / Peace of Mind / Cool Down / None
- Use None when Policy sold is NO or UNCLEAR unless the transcript clearly shows the furthest post-commitment stage reached anyway.
- When Policy sold is YES, Final stage supporting sale should follow the post-sale order when evidenced: Application Information, Payment Date, Banking, Disclosures, Third Party Underwriting, Peace of Mind, Cool Down — use the **furthest clearly evidenced** label (often Application, Payment, Banking, Disclosures, or Third Party Underwriting on enrollment calls; Peace of Mind / Cool Down only when clearly performed).

EVIDENCE line: one short phrase tied to what was said or done in the transcript (or "None" if Policy sold is NO and there is nothing to cite). You **may** mention **voice signature completed**, **recorded verification completed**, **American Amicable recording system**, **app ID**, **pound sign**, or **recorded line** **only** when that language appears in the transcript **and** **CALL STAGE REACHED** / **NOT REACHED** are **consistent** with **Third Party Underwriting** reached (per **HARD TRIGGER** above — do **not** claim those in Evidence while listing **Third Party Underwriting** only under **NOT REACHED**); otherwise do **not** invent carrier-recorded details.

TONE & DELIVERY + COMMUNICATION ANALYSIS (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

Always include BOTH sections exactly as in REQUIRED OUTPUT FORMAT, with headings "TONE & DELIVERY:" and "COMMUNICATION ANALYSIS:". These sections do NOT change SCORE, RISK, or PASS by themselves.

BASE JUDGMENTS ONLY ON TRANSCRIPT TEXT CUES such as:
- Agent filler words (um, uh, er, hmm, you know, like, etc.) when frequent or clustered enough to suggest uncertainty
- Broken, restarted, or incomplete sentences; repeated phrases or stammering patterns in the text
- Abrupt one-word or very short prospect replies, repeated minimal answers, or clear disengagement cues in wording
- Obvious hesitation markers in what is spoken (as reflected in the transcript), not invented backstory

DO NOT:
- Hallucinate tone or confidence if the transcript does not support it
- Use audio or assumptions beyond the written transcript
- Write long prose — one line per bullet; optional one short clause after each label if needed (keep practical and brief)

LABEL RULES:
- Agent Tone must be EXACTLY one of: Confident / Neutral / Uncertain (pick Neutral or Uncertain when evidence is weak; use Uncertain when fillers/broken sentences clearly dominate).
- Prospect Tone must be EXACTLY one of: Engaged / Neutral / Disengaged (pick Neutral when thin evidence; Disengaged only with clear short/flat/withdrawn patterns in text).

COMMUNICATION ANALYSIS (each line YES or NO only):
- Answer YES only with clear supporting transcript cues for that question.
- Answer NO when the transcript clearly contradicts the question OR when affirmative evidence is absent (do not guess YES without cues).
- If the transcript is too thin to judge, answer NO (do not invent YES).

SALES TASK CHECKLIST PRIORITY (MANDATORY):
- Use the **SALES TASK CHECKLIST** below as the main process-flow guide for what the agent completed, what was required at the **furthest stage reached**, what was **NOT REACHED**, and what counts as a miss.
- **Automatic-fail / compliance rules** override normal checklist scoring when they apply.
- **Do not** penalize or coach **future-stage** checklist items the call **never reached**; do **not** coach future-stage items unless the agent incorrectly **skipped ahead** in **CALL STAGE REACHED**.
- **Sold** / **Policy sold** is an **outcome**, not a **call stage** — do **not** add **SOLD** as a stage and do **not** infer **call stages** from a sale alone.

SPEAKER-LABELED TRANSCRIPT (MANDATORY):
- The transcript may include generated labels such as **Agent**, **Prospect**, or **Unknown**. Use them as an **audit aid** (who asked, who objected, callbacks, banking/account/payment).
- Do **not** treat labels as absolute proof when the surrounding text is ambiguous; if callback, objection, or banking responsibility depends on unclear speaker identity, mark **UNCLEAR** rather than auto-failing.

SALES TASK CHECKLIST:
{checklist}

SCORING RUBRIC:
{rubric}

REQUIRED OUTPUT FORMAT:
{output_format}
{role_block}TRANSCRIPT:
{transcript}

REPORT CONSISTENCY SELF-CHECK (MANDATORY BEFORE FINAL OUTPUT):
Before you finish, scan for contradictions — **especially 3 and 1 / rapport** — and fix them:
- **Automatic fail triggered: YES** + **PASS: YES** — invalid; set **PASS** per **AUDIT OUTCOME** (**AT RISK** if **Policy sold YES**, else **NO**).
- **Automatic fail triggered: YES** + **RISK: LOW** (or **MEDIUM** when compliance autofail applies) — invalid; **RISK** must be **HIGH**.
- **Automatic fail triggered: YES** + **Reason: None** — invalid; **Reason** must name the autofail cause(s).
- **Existing coverage mentioned but not confirmed: YES** + **Automatic fail triggered: NO** — invalid; set **Automatic fail triggered: YES** and align **Reason** / **RISK** / **PASS** / **SCORE** per **SCORE CAP RULES**.
- **Existing coverage mentioned but not confirmed: YES** + **PASS: YES** or **RISK: LOW** or **SCORE** **90+** — invalid; cap and align per **SCORE CAP RULES**.
- **Policy sold: YES** + **Automatic fail triggered: YES** + **PASS** not **AT RISK** — invalid; **PASS** must be **AT RISK**.
- **Did the agent confirm current coverage? NO** + **Did the agent call an insurance company to confirm current coverage? NO** + existing coverage clearly mentioned or the **Only one** ambiguity pattern applies ⇒ **Existing coverage mentioned but not confirmed** must be **YES** or **UNCLEAR** (if **UNCLEAR**, explain the ambiguity — do not use **NO** to clear a fair **Only one** reading); if **YES**, automatic fail must trigger as above.
- **3 and 1 Method used: YES** without clear **A + B** evidence (three-of-four topic groups **and** tied self-disclosure, not generic acknowledgments) — invalid; correct to **PARTIAL** or **NO** and align **Agent shared personal rapport information**.
- **3 and 1 Method used: YES** + **Agent shared personal rapport information: NO** — **invalid**; cannot pair **YES** on 3 and 1 with **NO** on rapport — correct both lines per **Fact Finding / Warm-up — TASK CHECKLIST**.
- **3 and 1 Method used: YES** + **Agent shared personal rapport information: PARTIAL** — **invalid**; **YES** on 3 and 1 requires rapport **YES** with **meaningful** disclosure — correct both lines.
- **3 and 1 Method used: YES** when **no** meaningful self-disclosure is present — **invalid**; **YES** requires tied self-disclosure — use **PARTIAL** or **NO**.
- **3 and 1 Method used: YES** when the agent **only asked questions** (no meaningful self-disclosure) — **invalid**.
- **Agent shared personal rapport information: YES** when the transcript **only** shows generic acknowledgments / empathy with **no** real personal detail — **invalid**; correct to **NO** or **PARTIAL**.
- **Agent shared personal rapport information: YES** when the agent **only** used vague lines such as **“I can relate,”** **“I understand,”** **“same here,”** **“I get it”** without **meaningful** personal detail — **invalid**; use **PARTIAL** or **NO**.
- **3 and 1 Method used: YES** when the agent did **not** ask across **at least three** topic groups — **invalid**.
- **3 and 1 Method used: YES** when questions are **mostly** medical, application, banking, or underwriting (not rapport topic groups) — **invalid**.
- **SALE OUTCOME** Evidence claims **voice signature** / **recorded verification** / **American Amicable recording system** but **Third Party Underwriting** is only under **NOT REACHED** — invalid; align **CALL STAGE REACHED** / **NOT REACHED** or remove unsupported Evidence phrases.
- Transcript includes **Welcome to the American Amicable Group recording system**, **app ID** / **pound sign** IVR, **voice signature**, or **recorded verification** in enrollment context but **Third Party Underwriting** only under **NOT REACHED** — invalid; set furthest stage per **HARD TRIGGER**.
- **Banking**-context wording or placeholders (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, **[NUMBER]** with routing/account/bank cues) but **Banking** only under **NOT REACHED** — invalid unless a **later** stage clearly supersedes.
- Transcript includes **Welcome to the American Amicable Group recording system** (or equivalent app ID / pound-sign IVR) ⇒ **Third Party Underwriting** reached; do **not** list **Third Party Underwriting** only under **NOT REACHED** while claiming an earlier furthest stage.
- **Peace of mind completed: NO** + **Cool down completed: NO** after a **sold** call with opportunity + **SCORE** in the **90s** is invalid — lower **Sales Process** / **SCORE** and align **PASS** / **RISK** per **SCORE CAP RULES**.
- **Payment date explained: NO** after **Banking** + **Sales Process** or **Closing** scored near-perfect is invalid — lower those categories and cap the final **SCORE** per payment-date rules.
- **3 and 1 Method** weak / **PARTIAL** / **NO** (with **Fact Finding / Warm-up** reached) cannot be the **sole** reason for **Automatic fail triggered: YES**, **RISK: HIGH**, **PASS: NO**, or **PASS: AT RISK** — use **SCORE**, **SCRIPT / FLOW MISSES**, **COACHING**, and optional **BIGGEST MISS** per **3 AND 1 METHOD — SCORE IMPORTANCE**.

FINAL INSTRUCTION:
Start exactly like this:

SCORE: <number>
RISK: <LOW/MEDIUM/HIGH>
PASS: <YES/NO/AT RISK>
"""


# DOB context on a line → optional redaction on the following line only.
_DOB_LINE_CONTEXT = re.compile(
    r"(?i)(?:date\s+of\s+birth|birth\s+date|\bd\.?\s*o\.?\s*b\.?\b|\bbirthday\b|\bborn\b)"
)
_DOB_NEXT_LINE_PATTERNS = (
    re.compile(r"\b\d{5,8}\b"),
    re.compile(r"\b\d{1,2}\s+\d{1,2}\s+(?:\d{2}|\d{4})\b"),
    re.compile(r"\b[0-1]?\d[/\-][0-3]?\d[/\-](?:\d{2}|\d{4})\b", re.I),
)

# Spoken month/day/year after explicit DOB / birth cues (transcript redaction only).
_SPOKEN_DOB_CUE = (
    r"(?:my\s+)?date\s+of\s+birth(?:\s+is)?|d\.?o\.?b\.?(?:\s+is)?|"
    r"birthday(?:\s+is)?|birth\s+date(?:\s+is)?|born(?:\s+on)?|"
    r"what\s+is\s+your\s+date\s+of\s+birth|when\s+were\s+you\s+born|"
    r"what(?:'s|\s+is)\s+your\s+birthday"
)
_SPOKEN_DOB_WORD = (
    r"zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_SPOKEN_DOB_DECADE = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_SPOKEN_DOB_UNITS = r"(?:one|two|three|four|five|six|seven|eight|nine)"
_SPOKEN_DOB_COMPOUND = rf"(?:{_SPOKEN_DOB_DECADE}\s*[-]?\s*{_SPOKEN_DOB_UNITS})\b"
_SPOKEN_DOB_LINK = rf"(?:{_SPOKEN_DOB_COMPOUND}|\b(?:{_SPOKEN_DOB_WORD})\b)"
_SPOKEN_DOB_SEP = r"(?:\s*,\s*|\s+and\s+|\s+)"
_SPOKEN_DOB_MONTH_PAIR = (
    r"(?:\b(?:zero|oh)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)"
)
_SPOKEN_DOB_CHAIN = (
    rf"(?:{_SPOKEN_DOB_MONTH_PAIR}{_SPOKEN_DOB_SEP})?"
    rf"{_SPOKEN_DOB_LINK}(?:{_SPOKEN_DOB_SEP}{_SPOKEN_DOB_LINK}){{3,}}"
)
_SPOKEN_DOB_MONTH_NAME = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
)
_RE_SPOKEN_DOB_SAME_LINE = re.compile(
    rf"(?P<cue>{_SPOKEN_DOB_CUE})\s+(?P<chain>{_SPOKEN_DOB_CHAIN})\b",
    re.IGNORECASE,
)
_RE_SPOKEN_DOB_MONTH_LINE = re.compile(
    rf"(?P<cue>{_SPOKEN_DOB_CUE})\s+(?P<mon>{_SPOKEN_DOB_MONTH_NAME})\s+"
    rf"(?P<day>{_SPOKEN_DOB_LINK})\s+(?P<yr>{_SPOKEN_DOB_LINK}(?:\s+{_SPOKEN_DOB_LINK})?)\b",
    re.IGNORECASE,
)
_RE_NEXT_LINE_SPOKEN_DOB = re.compile(
    rf"^(?P<lead>\s*)"
    rf"(?P<prefix>(?:agent|prospect|unknown)\s*:\s*)?"
    rf"(?P<chain>{_SPOKEN_DOB_CHAIN})(?P<tail>\s*[.!?]*)$",
    re.IGNORECASE,
)
# Continuation-only: line ends on a DOB/birth cue (no spoken date on same line). Allows ? ! . after cue.
_NEXT_LINE_SPOKEN_DOB_CTX = re.compile(
    r"(?:.*\b)?(?:date\s+of\s+birth|d\.?\s*o\.?\s*b\.?|birthday|birth\s+date)"
    r"\s*(?:is\b\s*)?\s*:?\s*\??\s*[.!]*\s*$"
    r"|(?:.*\b)?born(?:\s+on)?\s*\??\s*[.!]*\s*$",
    re.IGNORECASE,
)


def _redact_spoken_dob_phrases(text):
    """Redact spelled-out or spoken month/day/year DOB after explicit birth/DOB cues."""
    if not text:
        return text
    out = _RE_SPOKEN_DOB_MONTH_LINE.sub(r"\g<cue> [DOB]", text)
    out = _RE_SPOKEN_DOB_SAME_LINE.sub(r"\g<cue> [DOB]", out)
    return out


def _redact_next_line_spoken_dob_after_cue(text):
    """If the previous line ended on a DOB cue (continuation), replace a spoken-number-only line with [DOB]."""
    if not text:
        return text
    lines = text.split("\n")
    out = []
    prev_ctx = False
    for line in lines:
        if prev_ctx:
            m = _RE_NEXT_LINE_SPOKEN_DOB.match(line)
            if m:
                tail = m.group("tail") or ""
                line = f"{m.group('lead')}{m.group('prefix') or ''}[DOB]{tail}"
            prev_ctx = False
        out.append(line)
        prev_ctx = bool(_NEXT_LINE_SPOKEN_DOB_CTX.match(line.strip()))
    return "\n".join(out)


def _redact_next_line_after_dob_context(text):
    """When a line mentions DOB/birth, replace DOB-shaped tokens on the next line with [DOB]."""
    lines = text.split("\n")
    out = []
    dob_context = False
    for line in lines:
        if dob_context:
            for pat in _DOB_NEXT_LINE_PATTERNS:
                line = pat.sub("[DOB]", line)
            dob_context = False
        out.append(line)
        if _DOB_LINE_CONTEXT.search(line):
            dob_context = True
    return "\n".join(out)


# Spoken digit words for phone / banking redaction (transcript only).
_SPOKEN_PHONE_DIGIT = (
    r"zero|oh|one|two|three|four|five|six|seven|eight|nine"
)
_SPOKEN_PHONE_SEP = r"(?:[\s,;\-\u2013]+)"
_SPOKEN_PHONE_CHAIN = (
    rf"\b(?:{_SPOKEN_PHONE_DIGIT})\b"
    rf"(?:{_SPOKEN_PHONE_SEP}+\b(?:{_SPOKEN_PHONE_DIGIT})\b){{6,}}"
)


def _spoken_digit_chain_min(n_tail_after_first):
    """Spoken digit words with separators; at least (1 + n_tail_after_first) words total."""
    return (
        rf"\b(?:{_SPOKEN_PHONE_DIGIT})\b"
        rf"(?:{_SPOKEN_PHONE_SEP}+\b(?:{_SPOKEN_PHONE_DIGIT})\b){{{n_tail_after_first},}}"
    )


# --- Spoken account / routing / bank (run before spoken phone) ---
_SPOKEN_ROUTING_HEAD = (
    r"(?:"
    r"routing\s+number|your\s+routing|the\s+routing\b|pull\s+the\s+routing|routing\s+on\s+my\s+end|"
    r"have\s+your\s+routing|verify\s+what\s+i\s+have"
    r")"
)
_RE_SPOKEN_ROUTING_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_ROUTING_HEAD})(?P<gap>[\s\S]{{0,2000}}?)(?P<nums>{_spoken_digit_chain_min(6)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_SPOKEN_ACCOUNT_HEAD = (
    r"(?:"
    r"i\s+just\s+need\s+your\s+account|need\s+your\s+account|your\s+account|the\s+account\b|"
    r"account\s+number\s+is|account\s+number|bank\s+account|checking\s+account|savings\s+account|"
    r"payment\s+account|draft\s+account|account\s+line|"
    r"which\s+number\??\s+the\s+account"
    r")"
)
_RE_SPOKEN_ACCOUNT_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_ACCOUNT_HEAD})(?P<gap>[\s\S]{{0,1200}}?)(?P<nums>{_spoken_digit_chain_min(4)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_RE_SPOKEN_ACCOUNT_CONT = re.compile(
    rf"(?P<pre>(?:\bI\s+got\b|\bwhat\s+was\s+the\s+last\s+part\?))(?P<gap>\s*)(?P<nums>{_spoken_digit_chain_min(2)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_SPOKEN_BANK_GENERAL_HEAD = (
    r"(?:"
    r"\bbank\b|credit\s+union|financial\s+institution|check\s+number|checkbook"
    r")"
)
_RE_SPOKEN_BANK_GENERAL_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_BANK_GENERAL_HEAD})(?P<gap>[\s\S]{{0,1200}}?)(?P<nums>{_spoken_digit_chain_min(6)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)


def _redact_spoken_banking_numbers(text):
    """Spoken digit runs in banking context → [ROUTING_NUMBER] / [ACCOUNT_NUMBER] / [BANK_NUMBER] (before phone pass)."""
    if not text:
        return text
    out = text

    def _rr(m):
        return f"{m.group('head')}{m.group('gap')}[ROUTING_NUMBER]{m.group('tail')}"

    def _ra(m):
        return f"{m.group('head')}{m.group('gap')}[ACCOUNT_NUMBER]{m.group('tail')}"

    def _rb(m):
        return f"{m.group('head')}{m.group('gap')}[BANK_NUMBER]{m.group('tail')}"

    def _rc(m):
        return f"{m.group('pre')}{m.group('gap')}[ACCOUNT_NUMBER]{m.group('tail')}"

    # Account-related passes first so earlier "routing on my end" chatter does not steal account digits.
    out = _RE_SPOKEN_ACCOUNT_SPOKEN.sub(_ra, out)
    out = _RE_SPOKEN_ACCOUNT_CONT.sub(_rc, out)
    out = _RE_SPOKEN_BANK_GENERAL_SPOKEN.sub(_rb, out)
    out = _RE_SPOKEN_ROUTING_SPOKEN.sub(_rr, out)
    return out


_BANKING_NEAR_PHONE_GUARD = re.compile(
    r"\b(?:account|routing|accounts?|routing\s+number|bank|credit\s+union|financial\s+institution|"
    r"checking|savings|payment|draft)\b",
    re.IGNORECASE,
)
_SPOKEN_PHONE_HEAD = (
    r"(?:"
    r"best\s+number\s+to\s+reach\s+you\s+at|"
    r"phone\s+number|callback\s+number|contact\s+number|"
    r"verify\s+your\s+number|confirm\s+your\s+number|what\s+is\s+your\s+number|"
    r"is\s+this\s+a\s+good\s+number|"
    r"call\s+you\s+at|reach\s+you\s+at|text\s+you\s+at|"
    r"(?:it\s*'s|it\s+is)\s+going\s+to\s+be|"
    r"cellphone|"
    r"\bcell\b|\bmobile\b|\btelephone\b|"
    r"best\s+number|"
    r"(?:phone|callback|contact|your|the)\s+number\s+is"
    r")"
)
_RE_SPOKEN_PHONE_BLOCK = re.compile(
    rf"(?P<head>{_SPOKEN_PHONE_HEAD})(?P<gap>[\s\S]{{0,1500}}?)(?P<nums>{_SPOKEN_PHONE_CHAIN})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)


def _redact_spoken_phone_numbers(text):
    """Replace 7+ spoken digit-words with [PHONE] only after phone/contact cues (not generic audit numbers)."""
    if not text:
        return text

    def _repl(m):
        lookback = m.string[max(0, m.start() - 500) : m.start()]
        if _BANKING_NEAR_PHONE_GUARD.search(lookback):
            return m.group(0)
        return f"{m.group('head')}{m.group('gap')}[PHONE]{m.group('tail')}"

    return _RE_SPOKEN_PHONE_BLOCK.sub(_repl, text)


# Four or more digits with only short separators (read-out / account / routing style).
# Separators exclude "." to avoid redacting common decimals like 3.14 in transcript.
_SPACED_DIGIT_RUN = re.compile(
    r"(?<![\d])(?:\d[\s,\-/]+){3,}\d(?![\d])",
    re.UNICODE,
)

# Stage / carrier / plan tokens preserved from aggressive digit + name passes (longest first).
_PROTECT_TRANSCRIPT_TOKENS = (
    r"(?i)\bAmerican\s+Amicable\s+Group\b",
    r"(?i)\bAmerican\s+Amicable\b",
    r"(?i)\bThird\s+Party\s+Underwriting\b",
    r"(?i)\bPeace\s+of\s+Mind\b",
    r"(?i)\bCool\s+Down\b",
    r"(?i)\bFact\s+Finding\s*/\s*Warm-?up\b",
    r"(?i)\bSocial\s+Security\b",
    r"(?i)\bCredit\s+Union\b",
    r"(?i)\bImmediate\s+plan\b",
    r"(?i)\bGraded\s+plan\b",
    r"(?i)\bROP\s+plan\b",
    r"(?i)\bImmediate\b",
    r"(?i)\bGraded\b",
    r"(?i)\bROP\b",
)
def _transcript_protect_token(index):
    """Single non-digit placeholder so downstream \\d+ passes cannot corrupt vault swaps."""
    return chr(0xE000 + index)


def _transcript_protect_phrases(text):
    """Temporarily hides audit-critical phrases so number/name passes cannot mangle them."""
    protected = []
    out = text or ""

    for rx in _PROTECT_TRANSCRIPT_TOKENS:

        def _repl(m):
            protected.append(m.group(0))
            return _transcript_protect_token(len(protected) - 1)

        out = re.sub(rx, _repl, out)
    return out, protected


def _transcript_restore_phrases(text, protected):
    out = text
    for i, orig in enumerate(protected):
        out = out.replace(_transcript_protect_token(i), orig)
    return out


_SPELLED_DECADE = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_SPELLED_UNIT = r"(?:one|two|three|four|five|six|seven|eight|nine)"
# (pattern, replacement) — keep cue prefix (e.g. "I am ") for readability.
_SPELLED_AGE_REPL = (
    (
        re.compile(
            rf"(?i)\b((?:i\s*am|i'm|age|aged|turning)\s+)({_SPELLED_DECADE}(?:[\s-]+{_SPELLED_UNIT})?)\b"
        ),
        r"\1[NUMBER]",
    ),
    (
        re.compile(
            r"(?i)\b((?:i\s*am|i'm)\s+)(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b"
        ),
        r"\1[NUMBER]",
    ),
    (
        re.compile(rf"(?i)\b((?:i\s*am|i'm)\s+)({_SPELLED_UNIT})\b"),
        r"\1[NUMBER]",
    ),
)


def _redact_spelled_age_phrases(text):
    if not text:
        return text
    out = text
    for pat, repl in _SPELLED_AGE_REPL:
        out = pat.sub(repl, out)
    return out


def _redact_numeric_tokens(text):
    """Replace phones, money, dates, times, spaced/long digit runs, and other digits with typed placeholders."""
    if not text:
        return text
    t = text
    t = re.sub(
        r"\b(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}\b",
        "[PHONE]",
        t,
    )
    t = re.sub(r"(?<![\d])\d{10}(?![\d])", "[PHONE]", t)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", t)
    t = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
        "[TIME]",
        t,
    )
    t = re.sub(r"\$\s*\d+(?:[,.]\d+)*\b", "[MONEY]", t)
    t = re.sub(r"\b\d+(?:\.\d+)?%", "[NUMBER]", t)
    t = _SPACED_DIGIT_RUN.sub("[BANK_NUMBER]", t)
    t = re.sub(r"(?<![\d])\d+(?:st|nd|rd|th)\b", "[NUMBER]", t, flags=re.I)
    t = re.sub(r"(?<!\d)\d{12,}(?!\d)", "[BANK_NUMBER]", t)
    t = re.sub(r"(?<![\d])\d+(?:\.\d+)?(?![\d])", "[NUMBER]", t)
    return t


def _redact_spaced_and_long_digit_sequences(text):
    """Backward-compatible shim: full numeric pass (used where callers expect this hook)."""
    return _redact_numeric_tokens(text)


def _spoken_banking_redaction_selftest():
    """Fake samples — spoken banking digits before phone classification."""
    s1 = "I just need your account. Seven, six, five, four, three, two, one."
    o1 = redact_sensitive_transcript(s1)
    if "[ACCOUNT_NUMBER]" not in o1 or "[PHONE]" in o1:
        raise RuntimeError(f"banking account spoken failed: {o1!r}")

    s2 = "Do you have your routing number? Zero, six, four, two, one, nine, six, eight, one."
    o2 = redact_sensitive_transcript(s2)
    if "[ROUTING_NUMBER]" not in o2:
        raise RuntimeError(f"routing spoken failed: {o2!r}")

    s3 = "Phone number is seven six five four three four three five three."
    o3 = redact_sensitive_transcript(s3)
    if "[PHONE]" not in o3:
        raise RuntimeError(f"phone still required: {o3!r}")

    s4 = "Account number is seven six five four three four three five three."
    o4 = redact_sensitive_transcript(s4)
    if "[ACCOUNT_NUMBER]" not in o4 or "[PHONE]" in o4:
        raise RuntimeError(f"account number is spoken failed: {o4!r}")


def _spoken_phone_redaction_selftest():
    """Fake samples — spoken digits redacted only after phone cues."""
    a = "It's going to be seven, six, five Four, three, Four, three, five, three."
    out_a = redact_sensitive_transcript(a)
    if "[PHONE]" not in out_a:
        raise RuntimeError(f"spoken phone redaction missing [PHONE]: {out_a!r}")
    if re.search(r"\bseven\s*,\s*six\b", out_a, re.I):
        raise RuntimeError(f"spoken phone digits leaked: {out_a!r}")

    b = "Phone number is seven six five four three four three five three."
    if "[PHONE]" not in redact_sensitive_transcript(b):
        raise RuntimeError(f"spoken phone (phone number is) failed: {redact_sensitive_transcript(b)!r}")


def _spoken_dob_next_line_selftest():
    """Fake only — next-line spoken DOB after cue ending in ? (and same-line chain)."""
    a = "What is your date of birth?\nEight, twelve, nineteen, sixty-eight."
    expected_a = "What is your date of birth?\n[DOB]."
    if redact_sensitive_transcript(a) != expected_a:
        raise RuntimeError(f"next-line spoken DOB failed: {redact_sensitive_transcript(a)!r}")

    b = "My date of birth is eight, twelve, nineteen, sixty-eight."
    expected_b = "My date of birth is [DOB]."
    if redact_sensitive_transcript(b) != expected_b:
        raise RuntimeError(f"same-line spoken DOB failed: {redact_sensitive_transcript(b)!r}")


def _redaction_smoke_assertions():
    """Synthetic sample only — no real customer data; raises on failure."""
    _spoken_banking_redaction_selftest()
    _spoken_phone_redaction_selftest()
    _spoken_dob_next_line_selftest()
    sample = (
        "My name is John Smith, I was born 12/31/1950, my phone is 555-123-4567, "
        "I am 74, my account is 123456789, and I pay $62.50 on the 3rd. "
        "American Amicable Immediate plan graded ROP. I am seventy four. "
        "My date of birth is eight, twelve, nineteen, sixty-eight."
    )
    out = redact_sensitive_transcript(sample)
    if "[NAME]" not in out or "John" in out or "Smith" in out:
        raise RuntimeError(f"name redaction failed: {out!r}")
    if "[DOB]" not in out or "[PHONE]" not in out or "[MONEY]" not in out:
        raise RuntimeError(f"missing typed placeholders: {out!r}")
    if "[ACCOUNT_NUMBER]" not in out:
        raise RuntimeError(f"expected account placeholder: {out!r}")
    if out.count("[NUMBER]") < 2:
        raise RuntimeError(f"expected some [NUMBER] ordinals/ages, got: {out!r}")
    if re.search(r"\beight\s*,\s*twelve", out, re.I):
        raise RuntimeError(f"spelled DOB not redacted: {out!r}")
    if "American Amicable" not in out:
        raise RuntimeError(f"lost protected carrier: {out!r}")
    low = out.lower()
    for keep in ("immediate", "rop", "graded"):
        if keep not in low:
            raise RuntimeError(f"lost protected plan token {keep!r}: {out!r}")
    if re.search(r"(?i)\b(?:seventy|sixty|eighty)\s+(?:one|two|three|four|five)\b", out):
        raise RuntimeError(f"spelled age not redacted: {out!r}")


def redact_sensitive_transcript(transcript):
    redacted, vault = _transcript_protect_phrases(transcript or "")
    redacted = _redact_spoken_dob_phrases(redacted)
    redacted = _redact_next_line_spoken_dob_after_cue(redacted)

    _title_word = r"(?-i:[A-Z][a-z]+)"
    _title_name = rf"{_title_word}(?:\s+{_title_word}){{0,2}}"
    patterns = [
        # Full names after identity cues — Title Case tokens only (so "seventy four" is not [NAME]).
        (
            rf"(?i)\b(my name is|this is|i am|i'm|speaking with|call me)\s+({_title_name})\b",
            r"\1 [NAME]",
        ),
        (
            rf"(?i)\b(?:full\s+name|name)\s*[:\-]\s*({_title_name})\b",
            "name: [NAME]",
        ),
        (
            r"(?i)\b(mr|mrs|ms|miss)\.?\s+((?-i:[A-Z][a-z]+)(?:\s+(?-i:[A-Z][a-z]+))+)\b",
            r"\1 [NAME]",
        ),
        (r"(?i)\b(my\s+account\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(bank\s+account\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(account\s+number\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(routing\s+number\s+is\s+)([\d\-]+)\b", r"\1[ROUTING_NUMBER]"),
        (r"(?i)\b(routing\s+is\s+)([\d\-]+)\b", r"\1[ROUTING_NUMBER]"),
        (
            r"(?i)\b(?:dob|d\.o\.b\.|date of birth|birth date|born(?:\s+on)?)\s*[:\-]?\s*(?:[0-1]?\d[\/\-][0-3]?\d[\/\-](?:\d{2}|\d{4})|(?:\d{4}[\/\-][0-1]?\d[\/\-][0-3]?\d)|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2},?\s+\d{4})\b",
            "DOB: [DOB]",
        ),
        (r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        (r"(?i)\b(?:ssn|social security)\s*[:\-]?\s*\d{9}\b", "SSN: [SSN]"),
        (r"\b(?:\d[ -]?){12,19}\b", "[BANK_NUMBER]"),
        (
            r"(?i)\b(routing\s*(?:number|#)?\s*[:\-]\s*)([\d\s\-]{6,})\b",
            r"\1[ROUTING_NUMBER]",
        ),
        (
            r"(?i)\b(account\s*(?:number|#)?\s*[:\-]\s*)([\d\s\-]{6,})\b",
            r"\1[ACCOUNT_NUMBER]",
        ),
        (
            r"(?i)\b(?:card\s+number|debit\s+card|credit\s+card)\s*[:\-]\s*[\d\w\-]{4,}\b",
            "card number: [BANK_NUMBER]",
        ),
    ]

    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)

    redacted = _redact_next_line_after_dob_context(redacted)
    redacted = _redact_spelled_age_phrases(redacted)
    redacted = _redact_spoken_banking_numbers(redacted)
    redacted = _redact_spoken_phone_numbers(redacted)
    redacted = _redact_numeric_tokens(redacted)
    redacted = _transcript_restore_phrases(redacted, vault)
    return redacted


ROLE_LABEL_TRANSCRIPT_NOTE = (
    "TRANSCRIPT NOTE (MANDATORY): The following transcript may include Agent:, Prospect:, or Unknown: "
    "role labels generated only from the redacted transcript. Use labeled turns to follow the conversation; "
    "they are an aid and are not absolute proof of who spoke if the source text was ambiguous. "
    "Do not auto-fail based solely on callback wording when speaker role is uncertain — explain uncertainty."
)


def create_role_labeled_transcript(transcript_text):
    """
    Ask the model to rewrite an already-redacted transcript into Agent:/Prospect:/Unknown: turns.
    Raises on failure; caller handles logging and fallback.
    """
    text = (transcript_text or "").strip()
    if not text:
        raise ValueError("Empty transcript for role labeling")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    instructions = """You are preparing a redacted transcript for sales compliance auditing.

Rewrite the transcript into clear speaker-role turns using only:
Agent:
Prospect:
Unknown:

Rules:
- Do not add facts.
- Do not remove compliance-relevant details.
- Do not unredact or guess redacted information.
- If speaker identity is unclear, use Unknown.
- Prefer Agent for sales rep/script/control/payment/banking/close language.
- Prefer Prospect for answers, objections, personal details, hesitations, hangup language, or refusal.
- Keep callback language exact enough to determine who initiated the callback.
- Keep banking/account-number/payment-date language exact enough to audit.
- Keep current coverage/provider/carrier language exact enough to audit.
- Keep important phrases related to: coverage; provider/carrier; Social Security deposit timing; payment/draft date; banking/account/routing/payment info; objections; callbacks; hangups; sale/close; account number; call control; medical questions; warm-up/fact finding.
- Keep the text concise but complete enough for audit.
- Output only the role-labeled transcript."""

    user_block = f"{instructions}\n\n---\n\n{text}"
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=user_block,
        temperature=0,
    )
    out = (response.output_text or "").strip()
    if not out:
        raise RuntimeError("Empty model output for role labeling")
    return out


def try_save_role_labeled_transcript(call_name, redacted_transcript_text):
    """Role-label redacted text, save under transcripts_role_labeled/. Returns labeled text or None."""
    print(f"Creating role-labeled transcript for {call_name}", flush=True)
    out_path = os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{call_name}.txt")
    try:
        labeled = create_role_labeled_transcript(redacted_transcript_text)
        write_text(out_path, labeled)
        print(f"Saved role-labeled transcript for {call_name}", flush=True)
        return labeled
    except Exception as e:
        print(f"Role labeling failed for {call_name}: {e}", flush=True)
        return None


def estimate_openai_cost(prompt_text, output_text):
    input_tokens_est = max(1, (len(prompt_text) + 3) // 4)
    output_tokens_est = max(1, (len(output_text) + 3) // 4)
    input_cost = (input_tokens_est / 1000) * OPENAI_INPUT_COST_PER_1K_TOKENS
    output_cost = (output_tokens_est / 1000) * OPENAI_OUTPUT_COST_PER_1K_TOKENS
    total_cost = input_cost + output_cost
    return {
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def append_openai_cost_footer(report, cost):
    return (
        f"{report}\n\n"
        f"OPENAI COST ESTIMATE:\n"
        f"- Model: {OPENAI_MODEL}\n"
        f"- Input tokens (est): {cost['input_tokens_est']}\n"
        f"- Output tokens (est): {cost['output_tokens_est']}\n"
        f"- Estimated cost (USD): ${cost['total_cost']:.6f}\n"
    )


def parse_json_object(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty JSON response")

    # Handle fenced JSON responses safely.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def _normalize_yes_no_unclear(val, default="UNCLEAR"):
    if val is True:
        return "YES"
    if val is False:
        return "NO"
    s = str(val or "").strip().upper()
    return s if s in {"YES", "NO", "UNCLEAR"} else default


def _normalize_yes_no(val, default="NO"):
    s = str(val or "").strip().upper()
    return s if s in {"YES", "NO"} else default


def _normalize_pass_outcome(val):
    """Structured / report PASS line: YES, NO, or AT RISK."""
    s = str(val or "").strip().upper()
    if re.match(r"^AT\s+RISK$", s) or s.replace(" ", "") == "ATRISK":
        return "AT RISK"
    if s == "YES":
        return "YES"
    if s == "NO":
        return "NO"
    return "NO"


_AGENT_TONE_LABELS = {"confident": "Confident", "neutral": "Neutral", "uncertain": "Uncertain"}
_PROSPECT_TONE_LABELS = {"engaged": "Engaged", "neutral": "Neutral", "disengaged": "Disengaged"}


def _normalize_agent_tone(val):
    s = str(val or "").strip().lower()
    return _AGENT_TONE_LABELS.get(s, "Neutral")


def _normalize_prospect_tone(val):
    s = str(val or "").strip().lower()
    return _PROSPECT_TONE_LABELS.get(s, "Neutral")


def _normalize_yes_no_strict(val):
    """YES/NO only; anything else maps to NO (no hallucinated YES)."""
    return "YES" if str(val or "").strip().upper() == "YES" else "NO"


def _normalize_sale_final_stage(val):
    """Canonical stage label for SALE OUTCOME (must match prompt list)."""
    s = (str(val or "").strip() or "None").lower()
    s = re.sub(r"[/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s == "none":
        return "None"
    if "peace" in s and "mind" in s:
        return "Peace of Mind"
    if "cool" in s and "down" in s:
        return "Cool Down"
    if "third party" in s or "third-party" in s or (
        "underwriting" in s and ("record" in s or "amicable" in s)
    ):
        return "Third Party Underwriting"
    if "disclosur" in s:
        return "Disclosures"
    if "bank" in s:
        return "Banking"
    if "payment" in s:
        return "Payment"
    if "application" in s:
        return "Application"
    if s == "close" or s.startswith("close ") or " closing" in f" {s} ":
        return "Close"
    if "quote" in s:
        return "Quotes"
    return "None"


def detect_agent_callback_from_transcript(transcript):
    """
    True only when transcript clearly shows the agent offering, agreeing to, or scheduling
    a callback / deferring the live sale — conservative (no silence / vague inference).
    """
    if not transcript or not str(transcript).strip():
        return False
    tl = str(transcript).lower()
    tl = tl.replace("\u2019", "'").replace("\u2018", "'")

    patterns = (
        r"i'?ll\s+call\s+(?:you\s+)?back\b",
        r"i\s+will\s+call\s+(?:you\s+)?back\b",
        r"i'?ll\s+call\s+back\b",
        r"\bcall\s+you\s+back\b",
        r"\bgive\s+you\s+a\s+call\s+back\b",
        r"\bcall\s+you\s+later\b",
        r"i\s+can\s+call\s+(?:you\s+)?later\b",
        r"we\s+can\s+finish\s+this\s+later\b",
        r"let'?s\s+schedule\s+another\s+time\b",
        r"when\s+would\s+be\s+(?:a\s+)?better\s+time\b",
        r"we'?ll\s+call\s+(?:you\s+)?back\b",
        r"let\s+me\s+call\s+(?:you\s+)?(?:back|later)\b",
        r"let\s+me\s+give\s+you\s+a\s+call\s+back\b",
        r"ring\s+you\s+(?:back|later|at)\b",
        r"schedule\s+(?:a\s+)?(?:time|call)\s+to\s+(?:call|connect|finish)\b",
        r"pick\s+up\s+(?:where\s+we\s+left\s+off\s+)?(?:tomorrow|later|another\s+day)\b",
        r"continue\s+(?:this\s+)?(?:tomorrow|later|another\s+time)\b",
    )

    neg_before = re.compile(
        r"(?:don'?t|do\s+not|never|no\s+need\s+to|not\s+gonna|won'?t)\s+(?:call|ring|bother)\b",
        re.I,
    )

    for pat in patterns:
        for m in re.finditer(pat, tl):
            window_start = max(0, m.start() - 60)
            before = tl[window_start : m.start()]
            if neg_before.search(before):
                continue
            return True
    return False


def normalize_objections(raw):
    """Parse optional structured objections; drop incomplete entries."""
    if not raw or not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        objection = str(item.get("objection", "")).strip()
        handled = str(item.get("handled", "")).strip().upper()
        explanation = str(item.get("explanation", "")).strip()
        if not objection or handled not in {"YES", "NO"}:
            continue
        if not explanation:
            explanation = "See transcript."
        out.append(
            {"objection": objection, "handled": handled, "explanation": explanation}
        )
    return out


# Canonical call stage order (furthest = latest clearly performed; NOT REACHED = all later stages).
CALL_STAGE_ORDER = (
    "PQ / Handoff",
    "Opening",
    "Who I Am / What I Do",
    "Fact Finding / Warm-up",
    "Medical / Health",
    "Need",
    "Features / Benefits",
    "Change Up",
    "Pre-Close",
    "Quotes",
    "Close",
    "Application Information",
    "Payment Date",
    "Banking",
    "Disclosures",
    "Third Party Underwriting",
    "Peace of Mind",
    "Cool Down",
)


def _call_stage_canonical_index(label):
    """Map model or shorthand stage label to CALL_STAGE_ORDER index, or None if unknown."""
    if label is None:
        return None
    s = " ".join(str(label).strip().split())
    if not s:
        return None
    sl = s.lower().replace("–", "-")
    for i, canon in enumerate(CALL_STAGE_ORDER):
        if sl == canon.lower():
            return i
    if sl.startswith("pq") or "handoff" in sl:
        return 0
    if sl == "application" or "application information" in sl:
        return CALL_STAGE_ORDER.index("Application Information")
    if "payment date" in sl or sl == "payment":
        return CALL_STAGE_ORDER.index("Payment Date")
    if sl == "banking" or sl.startswith("banking "):
        return CALL_STAGE_ORDER.index("Banking")
    if "third party" in sl or "third-party" in sl:
        return CALL_STAGE_ORDER.index("Third Party Underwriting")
    if "peace of mind" in sl or sl in ("pom", "peace-of-mind"):
        return CALL_STAGE_ORDER.index("Peace of Mind")
    if "cool down" in sl or sl == "cooldown" or sl == "cool-down":
        return CALL_STAGE_ORDER.index("Cool Down")
    if (
        "fact finding" in sl
        or "warm-up" in sl
        or "warm up" in sl
        or "rapport building" in sl
    ):
        return CALL_STAGE_ORDER.index("Fact Finding / Warm-up")
    if "who i am" in sl or ("what i do" in sl and "who" in sl):
        return CALL_STAGE_ORDER.index("Who I Am / What I Do")
    if re.fullmatch(r"quotes?", sl):
        return CALL_STAGE_ORDER.index("Quotes")
    if re.fullmatch(r"close", sl):
        return CALL_STAGE_ORDER.index("Close")
    for i, canon in enumerate(CALL_STAGE_ORDER):
        cl = canon.lower()
        if len(sl) >= 8 and (sl in cl or cl in sl):
            return i
    return None


def _sale_final_stage_to_order_index(sale_final_stage_raw):
    """When policy sold, sale_final_stage implies at least that point in the pipeline."""
    norm = _normalize_sale_final_stage(sale_final_stage_raw)
    mp = {
        "Quotes": "Quotes",
        "Close": "Close",
        "Application": "Application Information",
        "Payment": "Payment Date",
        "Banking": "Banking",
        "Disclosures": "Disclosures",
        "Third Party Underwriting": "Third Party Underwriting",
        "Peace of Mind": "Peace of Mind",
        "Cool Down": "Cool Down",
    }
    canon = mp.get(norm)
    if not canon:
        return None
    return CALL_STAGE_ORDER.index(canon)


def _stage_refinement_text_blob(result, transcript):
    parts = [
        transcript or "",
        str(result.get("summary") or ""),
        " ".join(str(x) for x in result.get("checklist_results") or []),
        str(result.get("sale_outcome_evidence") or ""),
    ]
    return "\n".join(parts).lower()


def _transcript_suggests_banking_collection(blob):
    """Routing/account/payment-setup banking — not insurer coverage calls."""
    if re.search(r"\brouting\s*(?:number|#)?\b", blob) and re.search(
        r"\b(account|checking|savings)\b", blob
    ):
        return True
    if re.search(r"\baccount\s*(?:number|#)?\b", blob) and re.search(
        r"\b(bank|credit union|routing)\b", blob
    ):
        return True
    if re.search(r"\bfor\s+(?:the\s+)?(?:draft|payment|premium|policy)\b", blob) and re.search(
        r"\b(bank|routing|checking|savings)\b", blob
    ):
        return True
    if re.search(r"\b(called|call)\s+(?:your\s+|the\s+)?bank\b", blob):
        return True
    if re.search(r"\bverify\s+(?:with\s+)?(?:your\s+|the\s+)?(?:bank|credit union)\b", blob):
        return True
    if re.search(r"\bnine\s*digit\b", blob) and re.search(r"\b(bank|routing|account)\b", blob):
        return True
    return False


def _payment_date_stage_evidence(blob):
    """
    Payment Date only when draft/payment date for the policy is set or explained — not SS deposit timing alone.
    """
    if re.search(
        r"\b(first\s+)?draft\s+(?:date|day|is|will|scheduled|comes|pulls)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(
        r"\b(premium|policy)\s+(?:draft|payment|withdrawal|eft)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(
        r"\b(set|scheduled|confirm(?:ed)?|choose|picked)\b.{0,120}\b(draft|withdrawal|eft|premium)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(r"\bpayment\s+date\b.{0,80}\b(yes|set|confirmed)\b", blob, re.I):
        return True
    if re.search(
        r"\bsocial security\b.{0,120}\b(deposit|check|hits)\b",
        blob,
        re.I,
    ) and not re.search(r"\b(draft|premium|policy payment|withdrawal|eft)\b", blob, re.I):
        return False
    return False


def _disclosures_stage_evidence(blob):
    return bool(
        re.search(r"\b(read|reading|went over|covered)\b.{0,60}\bdisclosur", blob, re.I)
        or re.search(r"\bdisclosur[a-z]*\b.{0,40}\b(read|given|provided|complete)\b", blob, re.I)
        or re.search(r"\bhipaa\b.{0,40}\b(read|authorization|notice)\b", blob, re.I)
    )


def _third_party_underwriting_evidence(blob):
    """
    True only when transcript/summary suggests the post-disclosure recorded third-party
    underwriting step (e.g. American Amicable recorded line), not generic underwriting talk.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    bl = b.lower()
    # Strong IVR / carrier recorded-line cues (American Amicable and similar).
    if re.search(
        r"welcome\s+to\s+the\s+american\s+amicable\s+group\s+recording\s+system",
        bl,
    ):
        return True
    if re.search(r"\bamerican\s+amicable\s+group\s+recording\s+system\b", bl):
        return True
    if "american amicable" in bl and "recording system" in bl:
        return True
    if re.search(r"\bamerican\s+amicable\s+recording\s+system\b", bl):
        return True
    if re.search(
        r"\bfor\s+the\s+app\s+id\b.{0,160}"
        r"(?:pound\s+sign|pound\s+key|followed\s+by\s+the\s+pound|#\s*(?:sign|key)?\b)",
        bl,
    ):
        return True
    if "app id" in bl and (
        "pound sign" in bl
        or "pound key" in bl
        or "followed by the pound" in bl
        or re.search(r"enter\s+the\s+app\s+id", bl)
    ):
        return True
    if re.search(r"\bamerican\s+amicable\b", b, re.I) and re.search(
        r"\b(record|recorded|recording|line|dial|press|pound|connect|enter)\b", b, re.I
    ):
        return True
    if re.search(
        r"\b(?:call(?:ing|s)?\s+into|dial(?:ing)?)\s+(?:the\s+)?(?:american\s+amicable\s+)?recorded\s+line\b|"
        r"\b(?:start|starting|begin|beginning)\s+(?:the\s+)?(?:american\s+amicable\s+)?record"
        r"(?:ing|\b)|"
        r"\brecorded\s+(?:third|3rd)[\s-]?party\s+(?:underwriting|verification|line)\b|"
        r"\bthird[\s-]?party\s+(?:recorded|recording)\s+(?:underwriting|verification)\b|"
        r"\b(?:after|following)\s+(?:the\s+)?disclosur.{0,100}\b(?:recorded|recording|american\s+amicable|"
        r"third[\s-]?party)\b",
        b,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:voice\s+signature|e-?signature|electronic\s+signature)\b.{0,40}\b(?:record|recorded|"
            r"american\s+amicable|third[\s-]?party)\b",
            b,
            re.I,
        )
    )


def _peace_of_mind_stage_evidence(blob):
    """
    Post-sale reassurance section — conservative: generic beneficiary/mail mentions alone are not enough.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    if re.search(r"\bpeace\s+of\s+mind\b", b, re.I):
        return True
    if re.search(
        r"\b(?:rest\s+easy|sleep\s+(?:better|at\s+night)|feel\s+good\s+about\s+(?:this|your|the)\s+decision)\b",
        b,
        re.I,
    ):
        return True
    if re.search(r"\b(?:not|ain'?t)\s+going\s+to\s+forget\s+about\s+you\b", b, re.I):
        return True
    if re.search(
        r"\b(?:welcome|policy)\s+letter\b.{0,120}\b(?:mail|tomorrow|send|sent|going\s+to\s+mail)\b",
        b,
        re.I,
    ) or re.search(r"\bmail(?:ing|ed)?\s+(?:the\s+)?(?:welcome|policy)\s+letter\b", b, re.I):
        return True
    if re.search(
        r"\b(?:all\s+)?my\s+personal\s+information\b.{0,100}\b(?:company|qualified|carrier|policy)\b",
        b,
        re.I,
    ):
        return True
    if re.search(
        r"\byou'?re\s+good\b.{0,120}\b(?:forget|here|with\s+you|company|qualified)\b",
        b,
        re.I,
    ):
        return True
    if re.search(
        r"\bprotect(?:ed|ing)\s+your\s+family\b.{0,80}\b(?:beneficiary|coverage|policy|approved|qualified)\b",
        b,
        re.I,
    ):
        return True
    if re.search(r"\bcoverage\s+is\s+in\s+place\b", b, re.I):
        return True
    if re.search(
        r"\breassur(?:e|ing)\b.{0,120}\b(?:approved|qualified|family|beneficiary|decision|policy|mail|letter)\b",
        b,
        re.I,
    ):
        return True
    if re.search(
        r"\bpolicy\s+(?:in\s+the\s+)?mail\b.{0,80}\b(?:tomorrow|few\s+days|welcome|letter)\b",
        b,
        re.I,
    ):
        return True
    return False


def _cool_down_stage_evidence(blob):
    """
    Casual non-insurance wind-down after the sale — do not match checklist headings like 'Cool down completed'.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    if re.search(
        r"\b(?:small\s+talk|nothing\s+to\s+do\s+with\s+(?:the\s+)?(?:insurance|policy)|"
        r"off\s+(?:the\s+)?(?:insurance|script)|besides\s+(?:the\s+)?insurance)\b",
        b,
        re.I,
    ) and re.search(
        r"\b(weather|sports|football|baseball|basketball|hunt(?:ing)?|fishing|grandkids?|grandchildren|"
        r"pets?|dog|cat|vacation|weekend|hobby|hobbies|where\s+you\s+(?:live|from)|"
        r"how\s+long\s+have\s+you\s+lived|kids\s+(?:are\s+)?grow|retirement|"
        r"what\s+do\s+you\s+do|work\s+at)\b",
        b,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:before\s+(?:we|i)\s+(?:go|let\s+you\s+go)|anything\s+else\s+for\s+you|"
        r"while\s+we'?re\s+(?:still\s+)?on\s+the\s+line|just\s+curious)\b"
        r".{0,280}\b(weather|sports|football|baseball|basketball|hunt(?:ing)?|fishing|"
        r"grandkids?|grandchildren|pets?|dog|cat|vacation|weekend|hobby|hobbies|"
        r"where\s+you\s+(?:live|from)|how\s+long\s+have\s+you\s+lived|"
        r"what\s+do\s+you\s+do|work\s+at|retirement)\b",
        b,
        re.I,
    ):
        return True
    return False


def _warmup_entry_evidence(blob):
    """
    True when the transcript suggests the **Fact Finding / Warm-up** call stage was entered
    (warm-up, rapport building, fact-finding before medical — even 1–2 questions). Not full 3+1
    checklist completion. Conservative — common FE discovery phrasing only.
    """
    return bool(
        re.search(
            r"\b(?:how\s+long\s+have\s+you\s+lived|where\s+are\s+you\s+calling\s+from|"
            r"what\s+do\s+you\s+do\s+for\s+(?:work|a\s+living)|tell\s+me\s+(?:a\s+)?(?:little|bit)\s+about|"
            r"married|any\s+kids|children|grandchildren|grandkids|"
            r"what\s+.*\s+(?:like|enjoy)\s+to\s+do|favorite\b|where\s+.*\s+from|"
            r"day[-\s]?to[-\s]?day|family\s+live|who\s+.*\s+depend|"
            r"how\s+.*\s+feel\s+about|comfortable\s+sharing)\b",
            blob,
            re.I,
        )
    )


def _medical_health_entry_evidence(blob):
    """True when underwriting / health questioning has clearly begun (implies Fact Finding / Warm-up was passed)."""
    return bool(
        re.search(
            r"\b(?:height|weight|tobacco|nicotine|cigarettes?|smoke|smoking|"
            r"prescription|medications?|meds\b|health\s+questions|underwriting\s+health|"
            r"any\s+health|hospitalized|diagnosed|health\s+conditions?|"
            r"insulin|oxygen|wheelchair|cancer|heart\s+attack|stroke|"
            r"doctor|physician|medical\s+history)\b",
            blob,
            re.I,
        )
    )


def apply_refined_call_stage(result, transcript):
    """
    Raise CALL STAGE REACHED to the furthest stage supported by transcript + structured fields
    when the model undershoots (sold calls stopping at Application Information; early calls
    skipping Fact Finding / Warm-up despite rapport or medical entry evidence).
    Rebuilds NOT REACHED as only stages after that index. Does not alter policy_sold or scoring inputs.
    """
    if not isinstance(result, dict):
        return
    model_idx = _call_stage_canonical_index(result.get("stage_reached"))
    if model_idx is None:
        return

    blob = _stage_refinement_text_blob(result, transcript)
    transcript_blob = (transcript or "").lower()
    floors = [model_idx]

    idx_warm = CALL_STAGE_ORDER.index("Fact Finding / Warm-up")
    idx_med = CALL_STAGE_ORDER.index("Medical / Health")
    # Entry-based floors: CALL STAGE progression ≠ TASK CHECKLIST completion (3+1 can be PARTIAL while Fact Finding / Warm-up is entered).
    if _medical_health_entry_evidence(blob):
        floors.append(idx_med)
        floors.append(idx_warm)
    elif _warmup_entry_evidence(blob):
        floors.append(idx_warm)

    if result.get("policy_sold") == "YES":
        sidx = _sale_final_stage_to_order_index(result.get("sale_final_stage"))
        if sidx is not None:
            idx_tpu = CALL_STAGE_ORDER.index("Third Party Underwriting")
            idx_pom = CALL_STAGE_ORDER.index("Peace of Mind")
            idx_cd = CALL_STAGE_ORDER.index("Cool Down")
            # Sale outcome alone must not push past strict post-disclosure stages without transcript proof.
            if sidx >= idx_cd and not _cool_down_stage_evidence(transcript_blob):
                sidx = idx_cd - 1
            if sidx >= idx_pom and not _peace_of_mind_stage_evidence(transcript_blob):
                sidx = idx_pom - 1
            if sidx >= idx_tpu and not _third_party_underwriting_evidence(transcript_blob):
                sidx = idx_tpu - 1
            floors.append(sidx)

    if result.get("searchable_call_bank_banking") == "YES":
        floors.append(CALL_STAGE_ORDER.index("Banking"))
    if result.get("searchable_verify_cu_if_mentioned") == "YES":
        floors.append(CALL_STAGE_ORDER.index("Banking"))
    if _transcript_suggests_banking_collection(blob):
        floors.append(CALL_STAGE_ORDER.index("Banking"))

    if _payment_date_stage_evidence(blob):
        floors.append(CALL_STAGE_ORDER.index("Payment Date"))

    if _disclosures_stage_evidence(blob):
        floors.append(CALL_STAGE_ORDER.index("Disclosures"))

    if _third_party_underwriting_evidence(transcript_blob):
        floors.append(CALL_STAGE_ORDER.index("Third Party Underwriting"))

    if _peace_of_mind_stage_evidence(transcript_blob):
        floors.append(CALL_STAGE_ORDER.index("Peace of Mind"))

    if _cool_down_stage_evidence(transcript_blob):
        floors.append(CALL_STAGE_ORDER.index("Cool Down"))

    furthest = min(max(floors), len(CALL_STAGE_ORDER) - 1)
    result["stage_reached"] = CALL_STAGE_ORDER[furthest]
    result["not_reached"] = list(CALL_STAGE_ORDER[furthest + 1 :])

    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    if furthest >= cd_idx:
        result["early_end"] = "NO"
    else:
        result["early_end"] = "YES"


def _checklist_line_has_verdict_no(checklist_results, must_contain_all):
    """True if some checklist line contains every fragment (case-insensitive), is not NOT REACHED, and has : NO."""
    frags = [s.lower() for s in must_contain_all]
    for raw in checklist_results or []:
        line = str(raw).strip().lower()
        if not all(f in line for f in frags):
            continue
        if "not reached" in line:
            continue
        if re.search(r":\s*no\b", line):
            return True
    return False


def _checklist_autofail_coverage_yes(checklist_results):
    """Recover YES when the model mirrors the autofail line inside checklist_results only."""
    for raw in checklist_results or []:
        line = str(raw).strip().lower()
        if "existing coverage mentioned but not confirmed" in line and re.search(
            r":\s*yes\b", line
        ):
            return True
    return False


def _transcript_only_one_coverage_ambiguity(transcript):
    """
    Prospect answered 'only one' to an existing-vs-only-policy question, without carrier verification.
    Corrects under-flagged existing-coverage autofail when the model marks that line NO.
    """
    t = (transcript or "").lower()
    if not re.search(r"\bonly\s+one\b", t):
        return False
    if not re.search(
        r"\b(only\s+policy|your\s+only|final\s+expense|life\s+insurance|insurance\s+in\s+place)\b",
        t,
    ):
        return False
    if re.search(
        r"\b(insurance\s+compan|carrier|underwrit|three[\s-]?way|warm\s+transfer|on\s+the\s+line).{0,120}\b(confirm|verif|in[\s-]?force|policy\s+number)\b",
        t,
    ):
        return False
    return True


def _post_sale_incomplete_autofail(policy_sold, stage_reached, checklist_results):
    if policy_sold != "YES":
        return False
    si = _call_stage_canonical_index(stage_reached)
    if si is None or si < CALL_STAGE_ORDER.index("Disclosures"):
        return False
    if not _checklist_line_has_verdict_no(
        checklist_results, ("peace", "mind", "completed")
    ):
        return False
    if not _checklist_line_has_verdict_no(
        checklist_results, ("cool", "down", "completed")
    ):
        return False
    return True


def _post_sale_autofail_reason(not_reached):
    joined = " ".join(str(x).lower() for x in (not_reached or []))
    if "third party" in joined:
        return "Post-sale process incomplete: Third Party Underwriting, Peace of Mind, and Cool Down skipped"
    return "Post-sale process incomplete: Peace of Mind and Cool Down skipped"


def _payment_date_miss_after_banking(stage_reached, checklist_results):
    si = _call_stage_canonical_index(stage_reached)
    if si is None or si < CALL_STAGE_ORDER.index("Banking"):
        return False
    return _checklist_line_has_verdict_no(
        checklist_results, ("payment", "date", "explained")
    )


def validate_structured_audit(data, transcript=None):
    required_fields = [
        "score",
        "risk",
        "pass",
        "stage_reached",
        "early_end",
        "not_reached",
        "checklist_results",
        "coaching",
        "summary",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing structured audit fields: {', '.join(missing)}")

    score = int(data["score"])
    risk = str(data["risk"]).strip().upper()
    pass_value = _normalize_pass_outcome(data.get("pass"))
    stage_reached = str(data["stage_reached"]).strip()
    early_end = str(data["early_end"]).strip().upper()
    not_reached = normalize_list(data["not_reached"])
    checklist_results = normalize_list(data["checklist_results"])
    coaching = normalize_list(data["coaching"])
    summary = str(data["summary"]).strip()
    biggest_miss = str(data.get("biggest_miss", "")).strip()
    objections = normalize_objections(data.get("objections"))
    agent_set_callback = str(data.get("agent_set_callback", "")).strip().upper()
    if agent_set_callback not in {"YES", "NO", "UNCLEAR"}:
        agent_set_callback = "UNCLEAR"
    if detect_agent_callback_from_transcript(transcript):
        agent_set_callback = "YES"

    autofail_objection_no_call_control = _normalize_yes_no_unclear(
        data.get("autofail_objection_no_call_control")
    )
    autofail_coverage_not_confirmed = _normalize_yes_no_unclear(
        data.get("autofail_coverage_not_confirmed")
    )
    autofail_credit_union_not_verified = _normalize_yes_no_unclear(
        data.get("autofail_credit_union_not_verified")
    )
    automatic_fail_triggered = _normalize_yes_no(data.get("automatic_fail_triggered"), "NO")
    automatic_fail_reason = str(data.get("automatic_fail_reason", "") or "").strip() or "None"

    agent_tone = _normalize_agent_tone(data.get("agent_tone"))
    prospect_tone = _normalize_prospect_tone(data.get("prospect_tone"))
    comm_agent_confident = _normalize_yes_no_strict(data.get("comm_agent_confident"))
    comm_agent_control = _normalize_yes_no_strict(data.get("comm_agent_control"))
    comm_prospect_engaged = _normalize_yes_no_strict(data.get("comm_prospect_engaged"))
    comm_hesitation_detected = _normalize_yes_no_strict(data.get("comm_hesitation_detected"))

    searchable_confirm_current_coverage = _normalize_yes_no_unclear(
        data.get("searchable_confirm_current_coverage")
    )
    searchable_call_insurer_coverage = _normalize_yes_no_unclear(
        data.get("searchable_call_insurer_coverage")
    )
    searchable_call_bank_banking = _normalize_yes_no_unclear(
        data.get("searchable_call_bank_banking")
    )
    searchable_verify_cu_if_mentioned = _normalize_yes_no_unclear(
        data.get("searchable_verify_cu_if_mentioned")
    )
    searchable_ask_existing_coverage = _normalize_yes_no_unclear(
        data.get("searchable_ask_existing_coverage")
    )

    policy_sold = _normalize_yes_no_unclear(data.get("policy_sold"))

    if _checklist_autofail_coverage_yes(checklist_results):
        autofail_coverage_not_confirmed = "YES"

    if (
        _transcript_only_one_coverage_ambiguity(transcript)
        and searchable_ask_existing_coverage == "YES"
    ):
        autofail_coverage_not_confirmed = "YES"

    # Coverage autofail YES implies no carrier confirmation and overall automatic fail.
    if autofail_coverage_not_confirmed == "YES":
        searchable_confirm_current_coverage = "NO"
        searchable_call_insurer_coverage = "NO"
        automatic_fail_triggered = "YES"

    if autofail_credit_union_not_verified == "YES":
        automatic_fail_triggered = "YES"

    post_sale_skip = _post_sale_incomplete_autofail(
        policy_sold, stage_reached, checklist_results
    )
    pay_miss_after_banking = _payment_date_miss_after_banking(
        stage_reached, checklist_results
    )
    pay_autofail = policy_sold == "YES" and pay_miss_after_banking

    if post_sale_skip:
        automatic_fail_triggered = "YES"
    if pay_autofail:
        automatic_fail_triggered = "YES"

    if automatic_fail_triggered == "YES":
        reason_parts = []
        br = str(data.get("automatic_fail_reason", "") or "").strip()
        if br and br.lower() != "none":
            reason_parts.append(br)
        if autofail_coverage_not_confirmed == "YES":
            reason_parts.append("Existing coverage mentioned but not confirmed")
        if autofail_credit_union_not_verified == "YES":
            reason_parts.append("Credit union mentioned but bank/account not verified")
        if post_sale_skip:
            reason_parts.append(_post_sale_autofail_reason(not_reached))
        if pay_autofail:
            reason_parts.append("Payment/draft date not explained after banking")
        deduped = []
        seen_r = set()
        for p in reason_parts:
            k = p.lower()
            if k not in seen_r:
                seen_r.add(k)
                deduped.append(p)
        automatic_fail_reason = (
            "; ".join(deduped) if deduped else "Automatic fail conditions met"
        )

    if automatic_fail_triggered == "YES":
        if policy_sold == "YES":
            pass_value = "AT RISK"
            risk = "HIGH"
        else:
            pass_value = "NO"
            risk = "HIGH"

    issue_cov = autofail_coverage_not_confirmed == "YES"
    issue_cu = autofail_credit_union_not_verified == "YES"
    issue_post = post_sale_skip
    issue_pay = pay_miss_after_banking
    pom_no = _checklist_line_has_verdict_no(
        checklist_results, ("peace", "mind", "completed")
    )
    cd_no = _checklist_line_has_verdict_no(
        checklist_results, ("cool", "down", "completed")
    )
    payment_date_no = _checklist_line_has_verdict_no(
        checklist_results, ("payment", "date", "explained")
    )
    stack_cov_pay_pom_cd = (
        issue_cov and payment_date_no and pom_no and cd_no
    )
    if automatic_fail_triggered == "YES":
        score = min(score, 85)
    if issue_cov or issue_cu:
        score = min(score, 80)
    if issue_post:
        score = min(score, 80)
    if issue_pay:
        score = min(score, 88)
    if issue_cov and issue_pay:
        score = min(score, 75)
    if stack_cov_pay_pom_cd:
        score = min(score, 72)
    combo = int(issue_cov) + int(issue_cu) + int(issue_post) + int(issue_pay)
    if combo >= 3:
        score = min(score, 62)
    elif combo >= 2:
        score = min(score, 70)
    score = max(0, min(100, int(score)))

    sale_outcome_evidence = str(data.get("sale_outcome_evidence", "") or "").strip() or "None"
    sale_outcome_evidence = re.sub(r"[\r\n]+", " ", sale_outcome_evidence)
    if len(sale_outcome_evidence) > 600:
        sale_outcome_evidence = sale_outcome_evidence[:597] + "..."
    sale_final_stage = _normalize_sale_final_stage(data.get("sale_final_stage"))

    # Final invariant (coverage autofail line YES can never coexist with aggregate NO / None reason / wrong pass-risk).
    if autofail_coverage_not_confirmed == "YES":
        automatic_fail_triggered = "YES"
        searchable_confirm_current_coverage = "NO"
        searchable_call_insurer_coverage = "NO"
        reason_parts = []
        br = str(automatic_fail_reason or "").strip()
        if br and br.lower() != "none":
            reason_parts.append(br)
        reason_parts.append("Existing coverage mentioned but not confirmed")
        if autofail_credit_union_not_verified == "YES":
            reason_parts.append("Credit union mentioned but bank/account not verified")
        if post_sale_skip:
            reason_parts.append(_post_sale_autofail_reason(not_reached))
        if pay_autofail:
            reason_parts.append("Payment/draft date not explained after banking")
        deduped = []
        seen_r = set()
        for p in reason_parts:
            k = p.lower()
            if k not in seen_r:
                seen_r.add(k)
                deduped.append(p)
        automatic_fail_reason = (
            "; ".join(deduped) if deduped else "Existing coverage mentioned but not confirmed"
        )
        if policy_sold == "YES":
            pass_value = "AT RISK"
        else:
            pass_value = "NO"
        risk = "HIGH"
        score = min(score, 80)

    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("risk must be LOW, MEDIUM, or HIGH")
    if pass_value not in {"YES", "NO", "AT RISK"}:
        raise ValueError("pass must be YES, NO, or AT RISK")
    if early_end not in {"YES", "NO"}:
        raise ValueError("early_end must be YES or NO")
    if not stage_reached:
        raise ValueError("stage_reached is required")
    if not summary:
        raise ValueError("summary is required")

    if (
        pass_value == "AT RISK"
        and automatic_fail_triggered == "YES"
        and policy_sold == "YES"
    ):
        low = summary.lower()
        if "at risk" not in low or "sold" not in low:
            tail = (
                " Policy was sold (SALE OUTCOME: YES) but the sale is at risk due to "
                f"{automatic_fail_reason}."
            )
            summary = (summary + tail).strip()
            if len(summary) > 4500:
                summary = summary[:4497] + "..."

    result = {
        "score": score,
        "risk": risk,
        "pass": pass_value,
        "stage_reached": stage_reached,
        "early_end": early_end,
        "not_reached": not_reached,
        "checklist_results": checklist_results,
        "coaching": coaching,
        "summary": summary,
        "biggest_miss": biggest_miss,
        "objections": objections,
        "agent_set_callback": agent_set_callback,
        "autofail_objection_no_call_control": autofail_objection_no_call_control,
        "autofail_coverage_not_confirmed": autofail_coverage_not_confirmed,
        "autofail_credit_union_not_verified": autofail_credit_union_not_verified,
        "automatic_fail_triggered": automatic_fail_triggered,
        "automatic_fail_reason": automatic_fail_reason,
        "agent_tone": agent_tone,
        "prospect_tone": prospect_tone,
        "comm_agent_confident": comm_agent_confident,
        "comm_agent_control": comm_agent_control,
        "comm_prospect_engaged": comm_prospect_engaged,
        "comm_hesitation_detected": comm_hesitation_detected,
        "searchable_confirm_current_coverage": searchable_confirm_current_coverage,
        "searchable_call_insurer_coverage": searchable_call_insurer_coverage,
        "searchable_call_bank_banking": searchable_call_bank_banking,
        "searchable_verify_cu_if_mentioned": searchable_verify_cu_if_mentioned,
        "searchable_ask_existing_coverage": searchable_ask_existing_coverage,
        "policy_sold": policy_sold,
        "sale_outcome_evidence": sale_outcome_evidence,
        "sale_final_stage": sale_final_stage,
    }
    apply_refined_call_stage(result, transcript)
    return result


def build_structured_audit_prompt(base_prompt):
    return f"""{base_prompt}

INTERNAL MODE:
Return a single valid JSON object only. No markdown, no code fences, no commentary.

Required JSON fields:
- score (integer)
- risk ("LOW" | "MEDIUM" | "HIGH")
- pass ("YES" | "NO" | "AT RISK")
- stage_reached (string) — furthest **entered** stage per CALL STAGE rules (Fact Finding / Warm-up counts as reached on **entry**, e.g. 1–2 rapport questions or before medical; **not** full 3+1 completion).
- early_end ("YES" | "NO")
- not_reached (array of strings) — only stages **after** stage_reached; never list Fact Finding / Warm-up as not reached if medical/rapport entry shows it was entered.
- checklist_results (array of strings) — mirror TASK CHECKLIST; must include these **three** lines (exact labels) when that section is present: **Fact Finding / Warm-up:** YES|NO|PARTIAL|NOT REACHED; **3 and 1 Method used:** …; **Agent shared personal rapport information:** … — use **NOT REACHED** on the last two only if **Fact Finding / Warm-up** was **never entered**; if the segment began but 3+1/rapport share is incomplete use **PARTIAL** or **NO**, not NOT REACHED. Include **Product benefits explained:** YES|NO|PARTIAL per main prompt **PRODUCT BENEFITS EXPLAINED — DETECTION** (Immediate / ROP / Graded). When **Banking** was reached, also include exactly these four lines with verdicts per audit prompt: **Banking/account information requested or verified 3 times:**; **Banking/account information verified at least 2 times:**; **Agent read banking/account information back to prospect:**; **Prospect confirmed banking/account read-back:** — use **NOT REACHED** on all four only if **Banking** was not reached.
- coaching (array of strings)
- summary (string)

Optional JSON fields:
- biggest_miss (string; always include — use a short miss description, or the literal None / empty if no meaningful miss)
- objections (array of objects; include only when at least one real customer objection exists in the transcript)
  Each object: { "objection": string, "handled": "YES"|"NO", "explanation": string (one brief sentence) }
  If no objections, omit the field or use an empty array.
- agent_set_callback ("YES" | "NO" | "UNCLEAR") — REQUIRED when possible: follow CALLBACK AND SCHEDULING rules; transcript evidence only. YES if the agent clearly agrees to or schedules a callback; NO if not; UNCLEAR if ambiguous. If omitted, downstream text will treat as UNCLEAR.
- autofail_objection_no_call_control ("YES" | "NO" | "UNCLEAR") — YES only if real objection/resistance AND no proper call control attempt; NO if no objection; UNCLEAR if ambiguous.
- autofail_coverage_not_confirmed ("YES" | "NO" | "UNCLEAR") — YES when existing coverage is **mentioned or reasonably indicated** (including **"Only one"**-type ambiguity per prompt) and the agent did NOT meet carrier/provider confirmation; follow prompt **3)** and **EXISTING COVERAGE — FOLLOW-UP** (do NOT mark NO to "clear" unresolved **Only one**). NO if coverage never mentioned or hinted. NEVER YES solely because bank/payment verification was missing.
- autofail_credit_union_not_verified ("YES" | "NO" | "UNCLEAR") — YES only for CREDIT UNION + BANK/ACCOUNT/PAYMENT verification gap. NO if no credit union. NEVER YES solely because insurance coverage was not confirmed. UNCLEAR if ambiguous.
- searchable_confirm_current_coverage ("YES" | "NO" | "UNCLEAR") — YES only with insurer/carrier/provider direct verification (or clear equivalent) of existing coverage, NOT prospect Q&A alone; NOT bank calls. NO if only asked/accepted prospect description.
- searchable_call_insurer_coverage ("YES" | "NO" | "UNCLEAR") — YES only when transcript shows insurer/carrier/provider contacted or clearly committed for coverage verification; NOT bank. Otherwise NO.
- searchable_call_bank_banking ("YES" | "NO" | "UNCLEAR") — bank call for banking/account/routing/payment verification only (not coverage confirmation).
- searchable_verify_cu_if_mentioned ("YES" | "NO" | "UNCLEAR") — credit-union account/bank verification only.
- searchable_ask_existing_coverage ("YES" | "NO" | "UNCLEAR") — agent asked about existing coverage (insurance) generally.
- policy_sold ("YES" | "NO" | "UNCLEAR") — YES only with clear customer plan choice plus meaningful enrollment/application/payment progress; NO if ended before commitment; UNCLEAR if suggestive but not clear. Same value as SEARCHABLE "Was the policy sold?"
- sale_outcome_evidence (string) — brief transcript-backed phrase, or "None".
- sale_final_stage (string) — one of: Quotes, Close, Application, Payment, Banking, Disclosures, Third Party Underwriting, Peace of Mind, Cool Down, None (see POLICY SALE rules in prompt).
- automatic_fail_triggered ("YES" | "NO") — YES if any rule in **AUTOMATIC FAIL RULES** (sections **1–6**) clearly applies (callback, call control, coverage, credit union, **post-sale process incomplete**, **payment/draft date after Banking**); otherwise NO. **Never** YES **solely** because **3 and 1 Method used** is weak, **PARTIAL**, or **NO** (rapport gaps use **score** / checklist / coaching only — see **3 AND 1 METHOD — SCORE IMPORTANCE** in main prompt). **Hard rule:** if **autofail_coverage_not_confirmed** is **YES**, **automatic_fail_triggered** MUST be **YES** (never NO); **automatic_fail_reason** MUST mention **Existing coverage mentioned but not confirmed**; **risk** MUST be **HIGH**; **pass** MUST be **AT RISK** if **policy_sold** is **YES**, else **NO**; keep **final SCORE** at or below **80** when that coverage line is YES.
- automatic_fail_reason (string) — short explanation, or "None" if automatic_fail_triggered is NO. When multiple rules apply, join with **"; "** (e.g. coverage + post-sale + payment). Never "None" when automatic_fail_triggered is YES.
- pass rules with automatic_fail_triggered and policy_sold (SALE OUTCOME): If policy_sold is YES and automatic_fail_triggered is YES, pass must be "AT RISK" (not NO) and risk must be "HIGH" (including post-sale or payment-date automatic fails, not only coverage/credit-union). If policy_sold is NO or UNCLEAR and automatic_fail_triggered is YES, pass must be "NO" and risk should be HIGH. If automatic_fail_triggered is NO, use normal YES/NO pass rules.
- If pass is "AT RISK", the summary must clearly state the policy was sold but the sale is at risk due to the automatic fail reason.
- agent_tone ("Confident" | "Neutral" | "Uncertain") — transcript evidence only; default Neutral if omitted.
- prospect_tone ("Engaged" | "Neutral" | "Disengaged") — transcript evidence only; default Neutral if omitted.
- comm_agent_confident ("YES" | "NO") — YES only if transcript cues clearly show confidence; otherwise NO.
- comm_agent_control ("YES" | "NO") — YES only if agent clearly steers the conversation; otherwise NO.
- comm_prospect_engaged ("YES" | "NO") — YES only if prospect shows engagement in text; otherwise NO.
- comm_hesitation_detected ("YES" | "NO") — YES only if fillers, broken sentences, or repeated phrases clearly indicate hesitation/uncertainty; otherwise NO.

REPORT CONSISTENCY SELF-CHECK (MANDATORY BEFORE YOU OUTPUT JSON):
Re-read your verdicts and fix any contradiction:
- **automatic_fail_triggered: YES** + **pass: YES** — invalid; **pass** must be **AT RISK** if **policy_sold** is **YES**, else **NO**; **risk** must be **HIGH**; **automatic_fail_reason** must not be **None**.
- **autofail_coverage_not_confirmed: YES** + **automatic_fail_triggered: NO** — invalid. **autofail_coverage_not_confirmed: YES** cannot pair with **pass: YES**, **risk: LOW**, or **score** **90+** — align with main prompt **SCORE CAP RULES**.
- **policy_sold: YES** + **automatic_fail_triggered: YES** + **pass** not **AT RISK** — invalid.
- **Did the agent confirm current coverage? NO** + **Did the agent call an insurance company to confirm current coverage? NO** + clear existing-coverage mention (including **Only one** ambiguity) cannot pair with **autofail_coverage_not_confirmed: NO**.
- **3 and 1 Method used** cannot be **YES** without **evidence gate** + checklist **A + B** evidence (**quote or clear paraphrase** for self-disclosure); cannot pair **3 and 1 Method used: YES** with **Agent shared personal rapport information: NO** or **PARTIAL**; **questions-only** ⇒ **3 and 1 Method used** not **YES**; **Agent shared personal rapport information** cannot be **YES** on generic acknowledgments or vague-only lines alone — fix **checklist_results** strings before output.
- **sale_outcome_evidence** mentioning **voice signature** / **recorded verification** / **American Amicable recording** while **not_reached** lists **Third Party Underwriting** only — invalid; align **stage_reached** / **not_reached** or trim Evidence.
- Transcript includes **Welcome to the American Amicable Group recording system** (or equivalent app ID / pound-sign IVR) ⇒ **Third Party Underwriting** reached; do not leave it only under **not_reached** while claiming an earlier stage as furthest.
- **Policy sold YES** + skipped Peace of Mind / Cool Down with opportunity + **Payment date explained: NO** after Banking + coverage gap ⇒ **score** not **90+**, **pass** not **YES**, **risk** not **LOW**.
- Weak / **PARTIAL** / **NO** on **3 and 1 Method used** (with **Fact Finding / Warm-up** entered) must **not** alone force **automatic_fail_triggered**, **risk: HIGH**, or **pass** NO/AT RISK — apply **score** / **SCRIPT / FLOW MISSES** / **coaching** per **3 AND 1 METHOD — SCORE IMPORTANCE** in the main prompt.
"""


def format_bullet_lines(items, empty_fallback="None"):
    if not items:
        return f"- {empty_fallback}"
    return "\n".join(f"- {item}" for item in items)


def get_top_three_coaching_priorities(coaching_items, checklist_results):
    priorities = []

    for item in (coaching_items or []) + (checklist_results or []):
        text = str(item).strip()
        if text and text not in priorities:
            priorities.append(text)
        if len(priorities) == 3:
            break

    while len(priorities) < 3:
        priorities.append("No additional reached-stage coaching priority identified.")

    return priorities


def format_objection_sections(objections):
    if not objections:
        return ""
    detected = ["OBJECTIONS DETECTED:"]
    for row in objections:
        detected.append(f"- {row['objection']}")
    handling = ["OBJECTION HANDLING:"]
    for row in objections:
        handling.append(f"- Objection: {row['objection']}")
        handling.append(f"  Handled: {row['handled']}")
        handling.append(f"  Explanation: {row['explanation']}")
    return "\n".join(detected) + "\n\n" + "\n".join(handling) + "\n"


def render_text_report_from_structured(data):
    top_three_priorities = get_top_three_coaching_priorities(
        data.get("coaching"),
        data.get("checklist_results")
    )

    bm = str(data.get("biggest_miss", "") or "").strip()
    biggest_miss_bullet = bm if bm else "None"
    biggest_miss_section = f"BIGGEST MISS:\n- {biggest_miss_bullet}\n"
    objection_sections = format_objection_sections(data.get("objections") or [])
    cb = str(data.get("agent_set_callback", "UNCLEAR")).strip().upper()
    if cb not in {"YES", "NO", "UNCLEAR"}:
        cb = "UNCLEAR"

    return f"""SCORE: {data['score']}
RISK: {data['risk']}
PASS: {data['pass']}

CALL STAGE REACHED: {data['stage_reached']}
EARLY END: {data['early_end']}
NOT REACHED:
{format_bullet_lines(data['not_reached'], "None")}

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
{format_bullet_lines(data['checklist_results'], "None")}

TASK CHECKLIST:
{format_bullet_lines(data['checklist_results'], "None")}

COACHING:
TOP 3 COACHING PRIORITIES:
{format_bullet_lines(top_three_priorities, "None")}

{biggest_miss_section}{objection_sections}SEARCHABLE ANSWERS:
- Did the agent set a callback? {cb}
- Did the agent confirm current coverage? {data["searchable_confirm_current_coverage"]}
- Did the agent call an insurance company to confirm current coverage? {data["searchable_call_insurer_coverage"]}
- Did the agent call the bank to verify banking/account information? {data["searchable_call_bank_banking"]}
- Did the agent verify credit union account information if a credit union was mentioned? {data["searchable_verify_cu_if_mentioned"]}
- Did the agent ask about existing coverage? {data["searchable_ask_existing_coverage"]}
- Was the policy sold? {data["policy_sold"]}

AUTOMATIC FAIL CHECKS:
- Callback set: {cb}
- Objection occurred without proper call control: {data["autofail_objection_no_call_control"]}
- Existing coverage mentioned but not confirmed: {data["autofail_coverage_not_confirmed"]}
- Credit union mentioned but bank/account not verified: {data["autofail_credit_union_not_verified"]}
- Automatic fail triggered: {data["automatic_fail_triggered"]}
- Reason: {data["automatic_fail_reason"]}

SALE OUTCOME:
- Policy sold: {data["policy_sold"]}
- Evidence: {data["sale_outcome_evidence"]}
- Final stage supporting sale: {data["sale_final_stage"]}

TONE & DELIVERY:
- Agent Tone: {data["agent_tone"]}
- Prospect Tone: {data["prospect_tone"]}

COMMUNICATION ANALYSIS:
- Did the agent sound confident? {data["comm_agent_confident"]}
- Did the agent maintain control of the conversation? {data["comm_agent_control"]}
- Was the prospect engaged? {data["comm_prospect_engaged"]}
- Any hesitation or uncertainty detected? {data["comm_hesitation_detected"]}

SUMMARY:
{data['summary']}"""


def run_structured_audit_model(prompt, redacted_transcript=None):
    if not os.getenv("OPENAI_API_KEY"):
        return None, None

    try:
        print(f"[audit] Using OpenAI structured JSON mode: {OPENAI_MODEL}")
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0
        )
        raw_output = response.output_text.strip()
        cost = estimate_openai_cost(prompt, raw_output)
        print(
            "[audit] OpenAI estimated cost "
            f"(input={cost['input_tokens_est']} tok, output={cost['output_tokens_est']} tok): "
            f"${cost['total_cost']:.6f}"
        )
        structured = validate_structured_audit(
            parse_json_object(raw_output), redacted_transcript
        )
        return render_text_report_from_structured(structured), cost
    except Exception as e:
        print(f"[audit] Structured JSON mode failed, using text flow: {e}")
        return None, None


def run_audit_model(openai_prompt, fallback_prompt):
    if os.getenv("OPENAI_API_KEY"):
        try:
            print(f"[audit] Using OpenAI API model: {OPENAI_MODEL}")
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                input=openai_prompt,
                temperature=0
            )
            report = response.output_text.strip()
            cost = estimate_openai_cost(openai_prompt, report)
            print(
                "[audit] OpenAI estimated cost "
                f"(input={cost['input_tokens_est']} tok, output={cost['output_tokens_est']} tok): "
                f"${cost['total_cost']:.6f}"
            )
            return report, cost
        except Exception as e:
            print(f"[audit] OpenAI failed, falling back to Ollama: {e}")
            return run_ollama(fallback_prompt).strip(), None

    print("[audit] Using local Ollama")
    return run_ollama(fallback_prompt).strip(), None


def normalize_top3_coaching_header_line(report):
    """
    Dashboard extract_top3_coaching requires ^TOP 3 COACHING PRIORITIES: at line start.
    Fix common model mistake: '- TOP 3 COACHING PRIORITIES:' on its own line.
    """
    if not report:
        return report
    return re.sub(
        r"(?im)^(\s*)-\s*(TOP\s*3\s+COACHING\s+PRIORITIES\s*:)\s*$",
        r"\1\2",
        report,
        count=1,
    )


def trim_to_score_and_remove_unwanted_sections(report):
    score_index = report.upper().find("SCORE:")
    if score_index > 0:
        report = report[score_index:].strip()

    unwanted_markers = [
        "**CONVERSATION QUALITY",
        "CONVERSATION QUALITY:",
        "**RELEVANCE",
        "RELEVANCE:",
        "**NOTABLE POINTS",
        "NOTABLE POINTS:",
        "**IMPROVEMENT AREAS",
        "IMPROVEMENT AREAS:",
        "OVERALL,",
    ]

    upper_report = report.upper()
    cut_positions = []

    for marker in unwanted_markers:
        pos = upper_report.find(marker)
        if pos != -1:
            cut_positions.append(pos)

    if cut_positions:
        report = report[:min(cut_positions)].strip()

    return report


def generate_audit_report(prompt, openai_prompt, redacted_transcript=None):
    report = None
    openai_cost = None

    if USE_STRUCTURED_AUDIT:
        structured_prompt = build_structured_audit_prompt(openai_prompt)
        report, openai_cost = run_structured_audit_model(
            structured_prompt, redacted_transcript
        )

    if report is None:
        report, openai_cost = run_audit_model(openai_prompt, prompt)

    return report, openai_cost


def _text_enforce_tpu_stage_report(report, transcript):
    """
    When transcript shows carrier recorded third-party underwriting (e.g. American Amicable IVR)
    but CALL STAGE / NOT REACHED still omit or mis-list Third Party Underwriting, rebuild that block.
    """
    if not report or not transcript:
        return report
    tb = transcript.strip().lower()
    if not _third_party_underwriting_evidence(tb):
        return report
    m_block = re.search(
        r"(?is)^CALL STAGE REACHED:\s*(.+?)\s*\n"
        r"^EARLY END:\s*(YES|NO)\s*\n"
        r"^NOT REACHED:\s*\n"
        r"((?:- [^\n]*\n)+)",
        report,
        re.MULTILINE,
    )
    if not m_block:
        return report
    current_stage = m_block.group(1).strip()
    not_reached_raw = m_block.group(3)
    bullets = []
    for line in not_reached_raw.strip().split("\n"):
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    has_tpu_in_nr = any("third party" in b.lower() for b in bullets)
    idx_cur = _call_stage_canonical_index(current_stage)
    if idx_cur is None:
        idx_cur = 0
    idx_tpu = CALL_STAGE_ORDER.index("Third Party Underwriting")
    if not has_tpu_in_nr and idx_cur >= idx_tpu:
        return report
    floors = [idx_cur, idx_tpu]
    if _peace_of_mind_stage_evidence(tb):
        floors.append(CALL_STAGE_ORDER.index("Peace of Mind"))
    if _cool_down_stage_evidence(tb):
        floors.append(CALL_STAGE_ORDER.index("Cool Down"))
    furthest = min(max(floors), len(CALL_STAGE_ORDER) - 1)
    new_stage = CALL_STAGE_ORDER[furthest]
    tail = list(CALL_STAGE_ORDER[furthest + 1 :])
    nr_lines = "\n".join(f"- {s}" for s in tail) if tail else "- None"
    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    early_end = "NO" if furthest >= cd_idx else "YES"
    new_section = (
        f"CALL STAGE REACHED: {new_stage}\n"
        f"EARLY END: {early_end}\n"
        f"NOT REACHED:\n{nr_lines}\n"
    )
    return report[: m_block.start()] + new_section + report[m_block.end() :]


def enforce_final_audit_consistency(report, transcript=None):
    """
    Post-process free-text audits (and harden any path) so invalid autofail / stage combinations
    cannot appear in the rendered report.
    """
    if not report:
        return report
    if transcript:
        report = _text_enforce_tpu_stage_report(report, transcript)
        if _transcript_only_one_coverage_ambiguity(transcript):
            ask_yes = bool(
                re.search(
                    r"(?im)^- Did the agent ask about existing coverage\?\s*YES\b",
                    report,
                )
            )
            cov_line_no = bool(
                re.search(
                    r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b",
                    report,
                )
            )
            confirm_no = bool(
                re.search(
                    r"(?im)^- Did the agent confirm current coverage\?\s*NO\b",
                    report,
                )
            )
            insurer_no = bool(
                re.search(
                    r"(?im)^- Did the agent call an insurance company to confirm current coverage\?\s*NO\b",
                    report,
                )
            )
            if ask_yes and cov_line_no and confirm_no and insurer_no:
                report = re.sub(
                    r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b",
                    "- Existing coverage mentioned but not confirmed: YES",
                    report,
                    count=1,
                )

    cov_yes = bool(
        re.search(
            r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b",
            report,
        )
    )
    trig_no = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*NO\b", report)
    )
    if cov_yes and trig_no:
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*NO\b",
            "- Automatic fail triggered: YES",
            report,
            count=1,
        )

        def _merge_cov_reason(m):
            body = (m.group(1) or "").strip()
            frag = "Existing coverage mentioned but not confirmed"
            if not body or body.lower() == "none":
                return f"- Reason: {frag}"
            if frag.lower() in body.lower():
                return m.group(0)
            return f"- Reason: {body}; {frag}"

        report = re.sub(
            r"(?im)^- Reason:\s*(.*)$", _merge_cov_reason, report, count=1
        )
        sm = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm:
            capped = min(int(sm.group(1)), 80)
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b", f"SCORE: {capped}", report, count=1
            )
        report = re.sub(r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1)

    if cov_yes:

        def _cap_compliance(m):
            v = int(m.group(1))
            return f"- Compliance: {min(v, 72)}"

        report = re.sub(
            r"(?im)^- Compliance:\s*(\d+)\b", _cap_compliance, report, count=1
        )

        def _cap_sales_proc(m):
            v = int(m.group(1))
            return f"- Sales Process: {min(v, 72)}"

        report = re.sub(
            r"(?im)^- Sales Process:\s*(\d+)\b", _cap_sales_proc, report, count=1
        )

        sm_cov = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm_cov and int(sm_cov.group(1)) > 80:
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b",
                "SCORE: 80",
                report,
                count=1,
            )
        if re.search(r"(?im)^RISK:\s*(?:LOW|MEDIUM)\s*$", report):
            report = re.sub(
                r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1
            )
        if _report_policy_sold_yes(report):
            report = re.sub(
                r"(?im)^PASS:\s*YES\s*$", "PASS: AT RISK", report, count=1
            )
            report = re.sub(
                r"(?im)^PASS:\s*NO\s*$", "PASS: AT RISK", report, count=1
            )
        else:
            report = re.sub(
                r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report, count=1
            )

    auto_yes = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    if auto_yes and not cov_yes:
        sm_auto = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm_auto and int(sm_auto.group(1)) > 85:
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b",
                "SCORE: 85",
                report,
                count=1,
            )
    if auto_yes and re.search(r"(?im)^- Reason:\s*None\s*$", report):
        report = re.sub(
            r"(?im)^- Reason:\s*None\s*$",
            "- Reason: Automatic fail triggered (see AUTOMATIC FAIL CHECKS)",
            report,
            count=1,
        )

    return report


def _report_policy_sold_yes(report):
    if not report:
        return False
    if re.search(r"(?im)^- Policy sold:\s*YES\b", report):
        return True
    return bool(re.search(r"(?im)^- Was the policy sold\?\s*YES\b", report))


def enforce_pass_logic(report):
    score_match = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
    score_value = int(score_match.group(1)) if score_match else None
    has_autofail = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    sold_yes = _report_policy_sold_yes(report)

    if has_autofail and sold_yes:
        report = re.sub(
            r"(?im)^PASS:\s*\S.*$",
            "PASS: AT RISK",
            report,
            count=1,
        )
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: AT RISK", report, count=1)
    elif has_autofail:
        report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report)
        report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: NO", report, count=1)

    if score_value is not None and score_value < 70:
        if not (has_autofail and sold_yes):
            report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report)

    no_compliance_failures = re.search(
        r"(?im)^COMPLIANCE FAILURES:\s*\n\s*-?\s*None",
        report,
    )
    if (
        score_value is not None
        and score_value >= 70
        and no_compliance_failures
        and not has_autofail
    ):
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report)

    return report


def enforce_risk_for_automatic_fail(report):
    """Align RISK with automatic-fail rules for text reports (structured path uses validate_structured_audit)."""
    if not report:
        return report
    has_autofail = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    if not has_autofail:
        return report
    return re.sub(r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1)


def transcribe(file_path, call_name, filename):
    set_processing_state(call_name, filename, "transcribing", 5, "Transcribing audio")

    # beam_size=5: wider search than greedy (beam 1); slower per chunk but fewer word errors on
    #    noisy phone audio — main lever to keep accuracy while other knobs favor throughput.
    # vad_filter + min_silence_duration_ms: skip decoding long dead air (hallways, hold, pauses)
    #    so wall-clock time drops a lot on long calls without changing spoken content.
    # condition_on_previous_text=False: avoids error propagation down long files and skips extra
    #    prefix attention work → faster + less drift on repetitive insurance scripts.
    # temperature=0: deterministic sampling (default path); no sampling randomness, tiny speed win.
    segments, info = get_model().transcribe(
        file_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        temperature=0,
    )

    duration = float(getattr(info, "duration", 0) or 0)
    transcript_parts = []
    last_progress = 5

    for segment in segments:
        transcript_parts.append(segment.text)

        if duration > 0:
            end_time = float(getattr(segment, "end", 0) or 0)
            ratio = max(0, min(1, end_time / duration))
            progress = 5 + int(ratio * 70)

            if progress >= last_progress + 3:
                set_processing_state(call_name, filename, "transcribing", progress, "Transcribing audio")
                last_progress = progress

    set_processing_state(call_name, filename, "transcribing", 75, "Transcription complete")

    return "\n".join(transcript_parts).strip()


def audit(transcript, progress_callback=None, call_name=None):
    if progress_callback:
        progress_callback(AI_START_PROGRESS, "Running AI audit")

    checklist = read_text("training/sales_task_checklist.txt")
    rubric = read_text("training/scoring_rubric.txt")
    output_format = read_text("training/audit_output_format.txt")

    prompt = build_audit_prompt(transcript, checklist, rubric, output_format)
    redacted_transcript = redact_sensitive_transcript(transcript)
    transcript_for_openai = redacted_transcript
    role_label_note = None
    if call_name:
        labeled = try_save_role_labeled_transcript(call_name, redacted_transcript)
        if labeled:
            transcript_for_openai = labeled
            role_label_note = ROLE_LABEL_TRANSCRIPT_NOTE
    openai_prompt = build_audit_prompt(
        transcript_for_openai,
        checklist,
        rubric,
        output_format,
        role_label_note=role_label_note,
    )
    report, openai_cost = generate_audit_report(
        prompt, openai_prompt, transcript_for_openai
    )
    report = trim_to_score_and_remove_unwanted_sections(report)
    report = normalize_top3_coaching_header_line(report)

    report = enforce_final_audit_consistency(report, transcript)
    report = enforce_pass_logic(report)
    report = enforce_risk_for_automatic_fail(report)
    if openai_cost:
        report = append_openai_cost_footer(report, openai_cost)

    if progress_callback:
        progress_callback(AI_DONE_PROGRESS, "Saving audit report")

    return report


def wait_until_file_ready(file_path, checks=3, delay=1):
    last_size = -1

    for _ in range(checks):
        if not os.path.exists(file_path):
            return False

        size = os.path.getsize(file_path)

        if size == last_size and size > 0:
            return True

        last_size = size
        time.sleep(delay)

    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def move_to_processed_calls(file_path):
    if not os.path.exists(file_path):
        return None

    os.makedirs(PROCESSED_CALLS_FOLDER, exist_ok=True)

    filename = os.path.basename(file_path)
    destination = os.path.join(PROCESSED_CALLS_FOLDER, filename)

    if os.path.exists(destination):
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        destination = os.path.join(PROCESSED_CALLS_FOLDER, f"{name}_{timestamp}{ext}")

        while os.path.exists(destination):
            timestamp += 1
            destination = os.path.join(PROCESSED_CALLS_FOLDER, f"{name}_{timestamp}{ext}")

    shutil.move(file_path, destination)
    return destination


def move_to_processed_transcript(file_path):
    """Move a processed .txt upload out of transcript_uploads/ (same collision rules as audio)."""
    if not os.path.exists(file_path):
        return None

    os.makedirs(PROCESSED_TRANSCRIPTS_FOLDER, exist_ok=True)

    filename = os.path.basename(file_path)
    destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, filename)

    if os.path.exists(destination):
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, f"{name}_{timestamp}{ext}")

        while os.path.exists(destination):
            timestamp += 1
            destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, f"{name}_{timestamp}{ext}")

    shutil.move(file_path, destination)
    return destination


def process_file(file_path):
    filename = os.path.basename(file_path)

    if not filename.lower().endswith(AUDIO_EXTENSIONS):
        return

    call_name = os.path.splitext(filename)[0]

    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

    if os.path.exists(report_path):
        set_processing_state(call_name, filename, "complete", 100, "Complete")
        return

    if not wait_until_file_ready(file_path):
        return

    try:
        print(f"Processing: {call_name}", flush=True)

        set_processing_state(call_name, filename, "processing", 1, "Starting processing")

        if os.path.exists(transcript_path):
            transcript = read_text(transcript_path)
            if not STORE_RAW_TRANSCRIPTS:
                transcript = redact_sensitive_transcript(transcript)
                write_text(transcript_path, transcript)
            print(f"Resuming from transcript: {call_name}", flush=True)
        else:
            raw_transcript = transcribe(file_path, call_name, filename)
            transcript = raw_transcript if STORE_RAW_TRANSCRIPTS else redact_sensitive_transcript(raw_transcript)
            write_text(transcript_path, transcript)

        set_processing_state(call_name, filename, "analyzing", 80, "Running AI audit")

        report = audit(
            transcript,
            lambda progress, message: set_processing_state(
                call_name,
                filename,
                "analyzing",
                progress,
                message
            ),
            call_name=call_name,
        )

        write_text(report_path, report)

        score, risk = parse_report(report)

        save_to_db(call_name, transcript, report, score, risk)
        move_to_processed_calls(file_path)

        set_processing_state(call_name, filename, "complete", 100, "Complete")

        print(f"Done: {call_name} | Score: {score} | Risk: {risk}", flush=True)

    except Exception as e:
        error = str(e)
        print(f"ERROR processing {call_name}: {error}", flush=True)

        transcript = read_text(transcript_path)

        failure_report = f"""SCORE: 0
RISK: HIGH
PASS: NO

CALL STAGE REACHED: Processing failed
EARLY END: YES
NOT REACHED:
- Unable to evaluate

COMPLIANCE FAILURES:
- Processing failed

SCRIPT / FLOW MISSES:
- Unable to complete audit

TASK CHECKLIST:
- Unable to evaluate

COACHING:
TOP 3 COACHING PRIORITIES:
- Coaching should focus only on missed items within stages that were reached.
- Do not coach on later stages that were never reached unless the agent clearly skipped ahead or mishandled the flow.
- Retry this audit or delete and re-upload the call.

BIGGEST MISS:
- Audit did not complete

SUMMARY:
The system could not complete the audit because: {error}

TRANSCRIPT:
{transcript}
"""

        write_text(report_path, failure_report)
        save_to_db(call_name, transcript, failure_report, 0, "HIGH")
        set_processing_state(call_name, filename, "failed", 100, "Processing failed", error)


def process_transcript_upload(file_path):
    """Process a .txt dropped in transcript_uploads/: redact, audit, DB + reports (no Whisper)."""
    filename = os.path.basename(file_path)

    if not filename.lower().endswith(".txt"):
        return

    call_name = os.path.splitext(filename)[0]
    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

    if os.path.exists(report_path):
        set_processing_state(call_name, filename, "complete", 100, "Complete")
        move_to_processed_transcript(file_path)
        return

    if not wait_until_file_ready(file_path):
        return

    try:
        print(f"Processing transcript upload: {call_name}", flush=True)

        set_processing_state(call_name, filename, "processing", 1, "Reading transcript")

        raw_transcript = read_text(file_path)
        transcript = (
            raw_transcript if STORE_RAW_TRANSCRIPTS else redact_sensitive_transcript(raw_transcript)
        )
        write_text(transcript_path, transcript)

        set_processing_state(call_name, filename, "analyzing", 80, "Running AI audit")

        report = audit(
            transcript,
            lambda progress, message: set_processing_state(
                call_name,
                filename,
                "analyzing",
                progress,
                message,
            ),
            call_name=call_name,
        )

        write_text(report_path, report)

        score, risk = parse_report(report)

        save_to_db(call_name, transcript, report, score, risk)
        move_to_processed_transcript(file_path)

        set_processing_state(call_name, filename, "complete", 100, "Complete")

        print(f"Done (transcript upload): {call_name} | Score: {score} | Risk: {risk}", flush=True)

    except Exception as e:
        error = str(e)
        print(f"ERROR processing transcript upload {call_name}: {error}", flush=True)

        transcript = read_text(transcript_path)

        failure_report = f"""SCORE: 0
RISK: HIGH
PASS: NO

CALL STAGE REACHED: Processing failed
EARLY END: YES
NOT REACHED:
- Unable to evaluate

COMPLIANCE FAILURES:
- Processing failed

SCRIPT / FLOW MISSES:
- Unable to complete audit

TASK CHECKLIST:
- Unable to evaluate

COACHING:
TOP 3 COACHING PRIORITIES:
- Coaching should focus only on missed items within stages that were reached.
- Do not coach on later stages that were never reached unless the agent clearly skipped ahead or mishandled the flow.
- Retry this audit or delete and re-upload the call.

BIGGEST MISS:
- Audit did not complete

SUMMARY:
The system could not complete the audit because: {error}

TRANSCRIPT:
{transcript}
"""

        write_text(report_path, failure_report)
        save_to_db(call_name, transcript, failure_report, 0, "HIGH")
        set_processing_state(call_name, filename, "failed", 100, "Processing failed", error)


def recover_interrupted_work():
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        UPDATE processing_state
        SET status='retry',
            progress=0,
            message='Recovered after restart',
            error='Recovered after watcher restart',
            updated_at=CURRENT_TIMESTAMP
        WHERE status IN ('processing', 'transcribing', 'analyzing')
    """)

    conn.commit()
    conn.close()


def start():
    os.makedirs(CALLS_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPT_UPLOADS_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_CALLS_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    ensure_db()
    recover_interrupted_work()

    print("QA SYSTEM RUNNING...", flush=True)
    print(f"Scanning: {CALLS_FOLDER} and {TRANSCRIPT_UPLOADS_FOLDER}", flush=True)
    print(f"Every {SCAN_INTERVAL_SECONDS} seconds", flush=True)
    print("Press CTRL+C to stop.", flush=True)

    while True:
        try:
            for filename in sorted(os.listdir(TRANSCRIPT_UPLOADS_FOLDER)):
                if not filename.lower().endswith(".txt"):
                    continue
                tp = os.path.join(TRANSCRIPT_UPLOADS_FOLDER, filename)
                if os.path.isfile(tp):
                    process_transcript_upload(tp)

            files = sorted(os.listdir(CALLS_FOLDER))

            for filename in files:
                if not filename.lower().endswith(AUDIO_EXTENSIONS):
                    continue

                file_path = os.path.join(CALLS_FOLDER, filename)

                if os.path.isfile(file_path):
                    process_file(file_path)

            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Watcher stopped.", flush=True)
            break

        except Exception as e:
            print(f"WATCHER ERROR: {e}", flush=True)
            time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-redaction":
        _redaction_smoke_assertions()
        print("redaction smoke: OK", flush=True)
        sys.exit(0)
    start()
