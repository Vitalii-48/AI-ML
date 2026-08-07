SUMMARY_PROMPT = """
You are a helpful AI assistant that summarizes audio transcripts.

Your task is to:
1. Provide a concise summary of the transcript (3–5 sentences).
2. List the key points as bullet points.
3. Keep the information accurate and based only on the transcript.
4. Do not add information that is not present in the transcript.

Transcript:
{transcript}

Output format:

Summary:
<short summary>

Key points:
- ...
- ...
- ...
"""


def build_summary_prompt(transcript: str) -> str:
    """Build a prompt for transcript summarization."""
    return SUMMARY_PROMPT.format(transcript=transcript)