import os
import subprocess

TRANSCRIPTS_FOLDER = "transcripts"
REPORTS_FOLDER = "reports"

def run_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

# Loop through transcripts
for file in os.listdir(TRANSCRIPTS_FOLDER):
    if file.endswith(".txt"):
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, file)

        with open(transcript_path) as f:
            transcript = f.read()

        prompt = f"""
You are a quality assurance auditor reviewing a life insurance sales call.

Do NOT judge the agent based on strict script adherence. Instead, evaluate the overall effectiveness, professionalism, and compliance of the conversation.

=== EVALUATION CRITERIA ===

1. Conversation Quality
- Did the agent sound natural, confident, and professional?
- Did they build rapport with the client?

2. Relevance & Focus
- Did the agent stay on topic (life insurance, client needs)?
- Did they avoid rambling or unrelated conversation?

3. Fact-Finding
- Did the agent ask meaningful questions about:
  - Age
  - Dependents
  - Financial goals
  - Coverage needs

4. Clarity
- Did the agent clearly explain products or concepts?

5. Compliance Safety
- Any misleading statements?
- Any unrealistic promises or guarantees?
- Any potential compliance concerns?

=== SCORING ===

Rate each category from 1 to 10:
- conversation_quality
- relevance
- fact_finding
- clarity
- compliance_safety

=== OUTPUT FORMAT (STRICT JSON) ===
{{
  "conversation_quality": 1-10,
  "relevance": 1-10,
  "fact_finding": 1-10,
  "clarity": 1-10,
  "compliance_safety": 1-10,
  "issues_found": [],
  "summary": "Brief explanation of strengths and areas for improvement"
}}

=== CALL TRANSCRIPT ===
{transcript}
"""

        result = run_ollama(prompt)

        report_file = file.replace(".txt", "_report.txt")
        report_path = os.path.join(REPORTS_FOLDER, report_file)

        with open(report_path, "w") as f:
            f.write(result)

        print(f"Processed: {file} → {report_file}")
