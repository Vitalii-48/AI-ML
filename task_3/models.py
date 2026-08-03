from pydantic import BaseModel


class Document(BaseModel):
    """A single paragraph (chunk) from a document."""

    text: str
    source: str
    chunk: int