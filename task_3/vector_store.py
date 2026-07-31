# task_3\vector_store.py
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_MODEL_NAME, SUPPORTED_FILE_EXTENSIONS
from models import Document


class VectorStore:
    """Stores documents and their embeddings, and provides semantic search."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    @staticmethod
    def load_documents(folder: str) -> list[Document]:
        """Load all text files from the knowledge folder."""
        folder_path = Path(folder)
        documents: list[Document] = []

        if not folder_path.exists():
            print(f"[WARNING] Folder '{folder_path}' does not exist. Creating it.")
            folder_path.mkdir(parents=True, exist_ok=True)
            return documents

        for extension in SUPPORTED_FILE_EXTENSIONS:
            for file_path in folder_path.glob(extension):
                text = file_path.read_text(encoding="utf-8")
                paragraphs = text.split("\n\n")

                for paragraph_number, paragraph in enumerate(paragraphs, start=1):
                    paragraph = paragraph.strip()

                    if paragraph:
                        documents.append(
                            {
                                "text": paragraph,
                                "source": file_path.stem,
                                "chunk": paragraph_number,
                            }
                        )

        return documents

    def add_documents(self, folder: str) -> None:
        """Load documents and create normalized embeddings."""
        self.documents = self.load_documents(folder)

        embeddings = self.model.encode(
            [doc["text"] for doc in self.documents]
        )

        # Normalize embeddings once
        self.embeddings = embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

    def search(
            self,
            query: str,
            top_n: int = 3,
    ) -> list[tuple[int, float]]:
        """Search for the most relevant documents using cosine similarity."""
        if self.embeddings is None or not self.documents:
            return []

        query_embedding = self.model.encode(query)
        query_embedding /= np.linalg.norm(query_embedding)

        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_n]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]

    def get_by_id(self, doc_id: int) -> Document:
        """Return a document by its index."""
        return self.documents[doc_id]