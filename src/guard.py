"""Citation validation and safety guard for RAG responses."""

from dataclasses import dataclass

from .config import get_settings
from .logger import setup_logger
from .store import Chunk

logger = setup_logger(__name__)


@dataclass
class Decision:
    """Guard decision with detailed reasoning."""

    allowed: bool
    answer: str
    reason: str
    confidence: float
    citations_found: int
    citations_expected: int


class GuardRules:
    """Validation rules for RAG responses."""

    def __init__(self, min_confidence: float = 0.25):
        """Initialize guard with minimum confidence threshold."""
        self.min_confidence = min_confidence

    def validate_citations(
        self, draft: str, docs: list[Chunk]
    ) -> tuple[bool, str, int]:
        """Validate that response includes proper citations.

        Returns:
            (is_valid, reason, citation_count)
        """
        if not docs:
            return False, "No source documents provided", 0

        if not draft or not draft.strip():
            return False, "Empty response from LLM", 0

        # Check for citation format [DOC-ID]
        cited_count = sum(1 for doc in docs if f"[{doc.id}]" in draft)

        if cited_count == 0:
            return (
                False,
                "Missing citations - response must reference source documents",
                0,
            )

        return True, "", cited_count

    def validate_content(self, draft: str) -> tuple[bool, str]:
        """Validate response content for safety and quality.

        Future enhancements:
        - Toxicity detection
        - PII detection
        - Hallucination detection
        - Relevance scoring
        """
        if len(draft) < 10:
            return False, "Response too short"

        if len(draft) > 5000:
            return False, "Response too long"

        # Check for common hallucination patterns
        hallucination_indicators = [
            "i think",
            "in my opinion",
            "i believe",
            "personally",
            "i would say",
        ]

        draft_lower = draft.lower()
        for indicator in hallucination_indicators:
            if indicator in draft_lower:
                return False, f"Response contains uncertain language: '{indicator}'"

        return True, ""

    def calculate_confidence(
        self, citations_found: int, citations_expected: int, draft: str
    ) -> float:
        """Calculate confidence score based on citation coverage.

        Factors:
        - Citation coverage (primary)
        - Response length (secondary)
        - Citation density (tertiary)
        """
        if citations_expected == 0:
            return 0.0

        # Primary: Citation coverage
        coverage = citations_found / citations_expected

        # Secondary: Response length factor (penalize very short/long)
        length_factor = 1.0
        if len(draft) < 50:
            length_factor = 0.8
        elif len(draft) > 2000:
            length_factor = 0.9

        # Tertiary: Citation density (citations per 100 chars)
        density = (citations_found / max(len(draft), 1)) * 100
        density_factor = min(1.0, density / 5.0)  # Optimal ~5 citations per 100 chars

        confidence = (coverage * 0.7) + (length_factor * 0.2) + (density_factor * 0.1)

        return min(1.0, confidence)


async def approve_or_refuse(query: str, draft: str, docs: list[Chunk]) -> Decision:
    """Validate and approve or refuse a RAG response.

    Args:
        query: Original user query
        draft: Generated response from LLM
        docs: Source documents used for context

    Returns:
        Decision with approval status and reasoning
    """
    settings = get_settings()
    guard = GuardRules(min_confidence=settings.rag_min_confidence)

    logger.debug(f"Validating response for query: {query[:100]}")

    # Validate citations
    has_citations, citation_reason, cited_count = guard.validate_citations(draft, docs)

    if not has_citations:
        logger.warning(f"Citation validation failed: {citation_reason}")
        return Decision(
            allowed=False,
            answer="",
            reason=citation_reason,
            confidence=0.0,
            citations_found=cited_count,
            citations_expected=len(docs),
        )

    # Validate content
    content_valid, content_reason = guard.validate_content(draft)

    if not content_valid:
        logger.warning(f"Content validation failed: {content_reason}")
        return Decision(
            allowed=False,
            answer="",
            reason=content_reason,
            confidence=0.0,
            citations_found=cited_count,
            citations_expected=len(docs),
        )

    # Calculate confidence
    confidence = guard.calculate_confidence(cited_count, len(docs), draft)

    if confidence < guard.min_confidence:
        reason = f"Low confidence: {confidence:.2f} < {guard.min_confidence}"
        logger.warning(reason)
        return Decision(
            allowed=False,
            answer="",
            reason=reason,
            confidence=confidence,
            citations_found=cited_count,
            citations_expected=len(docs),
        )

    logger.info(
        "Response approved",
        extra={
            "confidence": confidence,
            "citations_found": cited_count,
            "citations_expected": len(docs),
        },
    )

    return Decision(
        allowed=True,
        answer=draft,
        reason="",
        confidence=confidence,
        citations_found=cited_count,
        citations_expected=len(docs),
    )
