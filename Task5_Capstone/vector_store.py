# vector_store.py
import  numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store and embedding model."""
        self.model = SentenceTransformer(model_name)
        self.texts: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.metadata: list[dict] = []

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Add a text document to the vector store."""
        vector: np.ndarray = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

        if metadata is None:
            metadata = {"source": "text"}

        self.texts.append(text)
        self.embeddings.append(vector)
        self.metadata.append(metadata)

    def search(self, query: str, top_k: int = 1) -> list[tuple[str, float, dict]]:
        """Return the most similar stored texts for the given query."""
        if not self.texts:
            return []

        query_vector = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = np.dot(self.embeddings, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.texts[int(i)], float(scores[i]), self.metadata[int(i)])
            for i in top_indices
        ]

    def to_dict(self) -> dict:
        """Serialize texts and metadata without embeddings for saving."""
        return {
            "texts": self.texts,
            "metadata": self.metadata,
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore the knowledge base and recompute embeddings."""
        self.texts = []
        self.embeddings = []
        self.metadata = []

        for text, meta in zip(data.get("texts", []), data.get("metadata", [])):
            self.add(text, metadata=meta)

    def is_empty(self) -> bool:
        """Return True if the knowledge base contains no entries."""
        return len(self.texts) == 0

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self.texts)

