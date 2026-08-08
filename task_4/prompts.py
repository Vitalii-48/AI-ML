# ==========================================
# PROMPTS
# ==========================================

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


EXTRACT_KEYWORDS_PROMPT = """
You are a helpful AI assistant that extracts key terms from audio transcripts.

Your task is to:
1. Identify 5 to 10 most relevant keywords, entities, or key concepts from the transcript.
2. Keep the keywords concise and accurate to the original text.
3. Do not include terms that are not mentioned or relevant to the context.

Transcript:
{transcript}

Output format:

Keywords:
- <keyword 1>
- <keyword 2>
- ...
"""

GENERATE_TITLE_PROMPT = """
You are a helpful AI assistant that creates engaging titles for audio transcripts.

Your task is to:
1. Generate 3 short, clear, and relevant title options based on the content.
2. Ensure each title accurately reflects the main topic.
3. Keep each title under 10 words.

Transcript:
{transcript}

Output format:

Suggested Titles:
1. <Title 1>
2. <Title 2>
3. <Title 3>
"""

QNA_PROMPT = """
You are a helpful AI assistant that creates comprehension questions from audio transcripts.

Your task is to:
1. Generate 3 to 5 question-and-answer pairs based on the transcript.
2. Each question should test understanding of a key point from the content.
3. Answer each question accurately using ONLY the information from the transcript.

Transcript:
{transcript}

Output format:

Q1: <question>
A1: <answer>

Q2: <question>
A2: <answer>
...
"""



# ==========================================
# BUILDER FUNCTIONS
# ==========================================

def build_summary_prompt(transcript: str) -> str:
    """Build a prompt for transcript summarization."""
    return SUMMARY_PROMPT.format(transcript=transcript)


def build_keywords_prompt(transcript: str) -> str:
    """Build a prompt for keywords extraction."""
    return EXTRACT_KEYWORDS_PROMPT.format(transcript=transcript)


def build_title_prompt(transcript: str) -> str:
    """Build a prompt for title generation."""
    return GENERATE_TITLE_PROMPT.format(transcript=transcript)


def build_qna_prompt(transcript: str) -> str:
    """Build a prompt for answering questions based on transcript."""
    return QNA_PROMPT.format(transcript=transcript)