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

Return the result as valid JSON using exactly this structure:

{{
    "summary": "A concise summary of the transcript.",
    "key_points": [
        "First key point",
        "Second key point",
        "Third key point"
    ]
}}
"""


EXTRACT_KEYWORDS_PROMPT = """
You are a helpful AI assistant that extracts key terms from audio transcripts.

Your task is to:
1. Identify 5 to 10 most relevant keywords, entities, or key concepts from the transcript.
2. Keep the keywords concise and accurate to the original text.
3. Do not include terms that are not mentioned or relevant to the context.

Transcript:
{transcript}

Return the result as valid JSON using exactly this structure:

{{
    "keywords": [
        "keyword 1",
        "keyword 2",
        "keyword 3",
        "keyword 4",
        "keyword 5"
    ]
}}
"""

GENERATE_TITLE_PROMPT = """
You are a helpful AI assistant that creates engaging titles for audio transcripts.

Your task is to:
1. Generate 3 short, clear, and relevant title options based on the content.
2. Ensure each title accurately reflects the main topic.
3. Keep each title under 10 words.

Transcript:
{transcript}

Return the result as valid JSON using exactly this structure:

{{
    "titles": [
        "Title 1",
        "Title 2",
        "Title 3"
    ]
}}
"""

QNA_PROMPT = """
You are a helpful AI assistant that creates comprehension questions from audio transcripts.

Your task is to:
1. Generate 3 to 5 question-and-answer pairs based on the transcript.
2. Each question should test understanding of a key point from the content.
3. Answer each question accurately using ONLY the information from the transcript.

Transcript:
{transcript}

Return the result as valid JSON using exactly this structure:

{{
    "questions": [
        {{
            "question": "Question 1",
            "answer": "Answer 1"
        }},
        {{
            "question": "Question 2",
            "answer": "Answer 2"
        }},
        {{
            "question": "Question 3",
            "answer": "Answer 3"
        }}
    ]
}}
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
    """Build a prompt for generating questions and answers."""
    return QNA_PROMPT.format(transcript=transcript)