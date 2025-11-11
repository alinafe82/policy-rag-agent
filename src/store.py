"""Document storage and retrieval with async support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """Document chunk with ID and text content."""

    id: str
    text: str

    def __hash__(self) -> int:
        """Make chunk hashable for caching."""
        return hash((self.id, self.text))


SAMPLE_POLICIES = [
    Chunk("HR-001", "Employees must complete security training annually."),
    Chunk("HR-002", "Remote work requires VPN connection and encrypted devices."),
    Chunk(
        "SEC-007",
        "Secrets must not be committed to git. Use a vault like HashiCorp Vault or AWS Secrets Manager.",
    ),
    Chunk("SEC-008", "All API endpoints must use HTTPS with TLS 1.2 or higher."),
    Chunk(
        "SEC-009",
        "Password requirements: minimum 12 characters, complexity required, 90-day rotation.",
    ),
    Chunk("IT-101", "MFA is required for administrative access to all systems."),
    Chunk(
        "IT-102",
        "Software updates must be applied within 48 hours of release for critical patches.",
    ),
    Chunk(
        "IT-103",
        "Data backup verification must occur weekly with documented test restores.",
    ),
    Chunk(
        "PRIV-010",
        "PII should be masked in logs and backups. Use tokenization for sensitive data.",
    ),
    Chunk("PRIV-011", "GDPR data subject requests must be fulfilled within 30 days."),
    Chunk(
        "PRIV-012", "Data retention: customer data kept for 7 years, logs for 1 year."
    ),
    Chunk(
        "COMP-001",
        "SOC 2 Type II compliance required for all customer-facing services.",
    ),
    Chunk(
        "COMP-002",
        "All production changes require documented change management approval.",
    ),
]


class DocumentStore(ABC):
    """Abstract interface for document storage and retrieval."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """Search for relevant documents."""
        pass

    @abstractmethod
    async def add_chunk(self, chunk: Chunk) -> None:
        """Add a document chunk to the store."""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a specific chunk by ID."""
        pass


class InMemoryStore(DocumentStore):
    """In-memory document store with keyword-based search."""

    def __init__(self, chunks: list[Chunk] | None = None):
        """Initialize store with optional chunks."""
        self.chunks = chunks or []
        logger.info(f"Initialized InMemoryStore with {len(self.chunks)} chunks")

    @classmethod
    def from_samples(cls) -> "InMemoryStore":
        """Create store with sample policy documents."""
        return cls(SAMPLE_POLICIES.copy())

    def _score(self, query: str, text: str) -> float:
        """Score relevance using keyword matching.

        In production, replace with embedding-based similarity using:
        - sentence-transformers (all-MiniLM-L6-v2)
        - OpenAI embeddings (text-embedding-ada-002)
        - Cohere embeddings
        """
        query_lower = query.lower()
        text_lower = text.lower()

        # Exact phrase match gets highest score
        if query_lower in text_lower:
            return 1.0

        # Word overlap scoring
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())

        if not query_words or not text_words:
            return 0.0

        intersection = len(query_words & text_words)
        union = len(query_words | text_words)

        # Jaccard similarity with boost for coverage
        jaccard = intersection / union if union > 0 else 0.0
        coverage = intersection / len(query_words) if query_words else 0.0

        return (jaccard * 0.5) + (coverage * 0.5)

    async def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """Search for relevant documents using keyword matching."""
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        logger.debug(f"Searching for: {query[:100]} (top_k={top_k})")

        scored = [(self._score(query, chunk.text), chunk) for chunk in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = [chunk for score, chunk in scored[:top_k] if score > 0.0]

        logger.info(
            f"Search returned {len(results)} results",
            extra={"query_length": len(query), "top_k": top_k},
        )

        return results

    async def add_chunk(self, chunk: Chunk) -> None:
        """Add a document chunk to the store."""
        if any(c.id == chunk.id for c in self.chunks):
            logger.warning(f"Chunk {chunk.id} already exists, skipping")
            return

        self.chunks.append(chunk)
        logger.info(f"Added chunk {chunk.id}")

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a specific chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
