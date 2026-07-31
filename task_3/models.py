from typing import TypedDict


class Document(TypedDict):
    """A single paragraph (chunk) from the knowledge base along with source metadata."""

    text: str
    source: str
    chunk: int