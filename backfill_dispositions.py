import re
import sqlite3

DB = "calls.db"

VALID = {"SOLD", "U90", "LCR", "BOOTC", "LEAD", "AGE"}

def clean_text(s):
    return (s or "").lower()

def report_says_sold(report):
    return bool(re.search(r"(?im)^- Policy sold:\s*YES\b|^- Was the policy sold\?\s*YES\b", report or ""))

def detect_disposition(call_name, transcript, report, duration_seconds=None):
    text = clean_text((transcript or "") + "\n" + (report or ""))

    if report_says_sold(report):
        return "SOLD", "Report indicates policy sold."

    if re.search(r"\b(over\s*80|older than\s*80|too old|outside (?:the )?age range|age limit|cannot qualify due to age)\b", text):
        return "AGE", "Age-related disqualification detected."

    # Health LCR requires an actual disqualification outcome, not the agent reading
    # health-screening questions containing DNQ terms.
    health_agent_dq = bool(re.search(
        r"(?is)"
        r"(unfortunately|sorry|based on that|because of that|with that condition|due to that|that means|"
        r"after reviewing|from those answers).{0,220}"
        r"(do(?:es)? not qualify|won't qualify|would not qualify|can't qualify|cannot qualify|"
        r"not able to qualify|unable to qualify|can't help you|cannot help you|not eligible|declined|knockout)",
        text,
    ))

    # Do not trust stale report wording alone for health LCR.
    # Old reports may already contain "Prospect had a disqualifying health condition"
    # from the former broad detector. Require transcript-supported agent DNQ language.
    if health_agent_dq:
        return "LCR", "Health-related disqualification language detected."

    if re.search(r"\b(no income|don't have any income|do not have any income|not at all.*income|working on my disability|take food off your table|can't afford it|cannot afford it)\b", text):
        return "LCR", "No-income / affordability disqualification language detected."

    # BOOTC = very early drop during PQ/handoff or immediately after,
    # before the selling agent meaningfully starts the call.
    opening_only = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*Opening / Handoff\b", report or ""))

    meaningful_agent_start = bool(re.search(
        r"(?is)"
        r"(call (?:may|will) be recorded|recorded for quality|"
        r"state licensed|license number|field underwriter|"
        r"fact finding|warm-up|warm up|3 and 1|"
        r"were you born|are you still working|beneficiary|"
        r"health questions|medications|height|weight|"
        r"product benefits|three options|application)",
        text,
    ))

    # Look only at the first chunk of the call so later disconnects do not become BOOTC.
    early_text = text[:1800]
    pq_or_handoff_early = "pq:" in early_text or "handoff" in early_text or "transfer" in early_text
    early_hangup = bool(re.search(
        r"\b(hung up|hang up|disconnected|stopped responding|are you there|can you hear me|bye-bye|bye)\b",
        early_text,
    ))

    if opening_only and pq_or_handoff_early and early_hangup and not meaningful_agent_start:
        return "BOOTC", "Prospect disconnected during PQ/handoff before the selling agent meaningfully started."

    if duration_seconds is not None:
        try:
            if int(duration_seconds) < 110:
                return "U90", "Call duration was under 110 seconds."
        except Exception:
            pass

    return "LEAD", "No sold, age, health-disqualification, BOOTC, or U90 indicator detected."

def main():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    rows = c.execute(
        """
        SELECT id, call_name, transcript, report, duration_seconds, manual_disposition
        FROM calls
        ORDER BY id
        """
    ).fetchall()

    changed = 0
    for row_id, call_name, transcript, report, duration_seconds, manual in rows:
        auto, reason = detect_disposition(call_name, transcript, report, duration_seconds)
        manual = (manual or "").strip().upper()
        final = manual if manual in VALID else auto

        c.execute(
            """
            UPDATE calls
            SET auto_disposition=?, final_disposition=?, disposition_reason=?
            WHERE id=?
            """,
            (auto, final, reason, row_id),
        )
        changed += c.rowcount

    conn.commit()
    conn.close()
    print(f"Disposition rows updated: {changed}")

if __name__ == "__main__":
    main()
