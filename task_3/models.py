from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single paragraph (chunk) from a document."""

    text: str = Field(..., description="Content of the document paragraph")
    source: str = Field(..., description="Source file name")
    chunk: int = Field(..., description="Paragraph index/number")