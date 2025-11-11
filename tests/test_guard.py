"""Tests for guard validation functionality."""

import pytest

from src.guard import Decision, GuardRules, approve_or_refuse
from src.store import Chunk


class TestGuardRules:
    """Test GuardRules validation logic."""

    @pytest.fixture
    def guard(self):
        """Create guard with default settings."""
        return GuardRules(min_confidence=0.25)

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            Chunk("DOC-001", "Test content one"),
            Chunk("DOC-002", "Test content two"),
            Chunk("DOC-003", "Test content three"),
        ]

    def test_validate_citations_success(self, guard, sample_docs):
        """Test successful citation validation."""
        draft = "This is content from [DOC-001] and [DOC-002]."
        is_valid, reason, count = guard.validate_citations(draft, sample_docs)
        assert is_valid is True
        assert reason == ""
        assert count == 2

    def test_validate_citations_missing(self, guard, sample_docs):
        """Test validation fails with missing citations."""
        draft = "This is content with no citations."
        is_valid, reason, count = guard.validate_citations(draft, sample_docs)
        assert is_valid is False
        assert "Missing citations" in reason
        assert count == 0

    def test_validate_citations_empty_response(self, guard, sample_docs):
        """Test validation fails with empty response."""
        is_valid, reason, count = guard.validate_citations("", sample_docs)
        assert is_valid is False
        assert "Empty response" in reason

    def test_validate_citations_no_docs(self, guard):
        """Test validation fails with no documents."""
        draft = "Some content [DOC-001]."
        is_valid, reason, count = guard.validate_citations(draft, [])
        assert is_valid is False
        assert "No source documents" in reason

    def test_validate_content_success(self, guard):
        """Test successful content validation."""
        draft = "This is a good response with proper content."
        is_valid, reason = guard.validate_content(draft)
        assert is_valid is True
        assert reason == ""

    def test_validate_content_too_short(self, guard):
        """Test validation fails with too short content."""
        is_valid, reason = guard.validate_content("Short")
        assert is_valid is False
        assert "too short" in reason

    def test_validate_content_too_long(self, guard):
        """Test validation fails with too long content."""
        draft = "x" * 6000
        is_valid, reason = guard.validate_content(draft)
        assert is_valid is False
        assert "too long" in reason

    def test_validate_content_hallucination_indicators(self, guard):
        """Test validation fails with hallucination indicators."""
        drafts = [
            "I think this is correct.",
            "In my opinion, the answer is yes.",
            "I believe this is true.",
            "Personally, I would say yes.",
        ]
        for draft in drafts:
            is_valid, reason = guard.validate_content(draft)
            assert is_valid is False
            assert "uncertain language" in reason

    def test_calculate_confidence_full_coverage(self, guard):
        """Test confidence with full citation coverage."""
        confidence = guard.calculate_confidence(
            3, 3, "Good response with citations [DOC-001] [DOC-002] [DOC-003]."
        )
        assert confidence > 0.5

    def test_calculate_confidence_partial_coverage(self, guard):
        """Test confidence with partial citation coverage."""
        confidence = guard.calculate_confidence(
            2, 4, "Response with some citations [DOC-001] [DOC-002]."
        )
        assert 0.3 < confidence < 0.8

    def test_calculate_confidence_no_expected(self, guard):
        """Test confidence with no expected citations."""
        confidence = guard.calculate_confidence(0, 0, "Some response.")
        assert confidence == 0.0


class TestApproveOrRefuse:
    """Test approve_or_refuse function."""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            Chunk("DOC-001", "Security policy requires MFA."),
            Chunk("DOC-002", "Passwords must be 12+ characters."),
        ]

    async def test_approve_valid_response(self, sample_docs):
        """Test approving a valid response."""
        query = "What is the MFA policy?"
        draft = "Security policy requires MFA for all users [DOC-001]."
        decision = await approve_or_refuse(query, draft, sample_docs)

        assert decision.allowed is True
        assert decision.answer == draft
        assert decision.confidence > 0.0
        assert decision.citations_found >= 1

    async def test_refuse_missing_citations(self, sample_docs):
        """Test refusing response with missing citations."""
        query = "What is the policy?"
        draft = "The policy says you need strong authentication."
        decision = await approve_or_refuse(query, draft, sample_docs)

        assert decision.allowed is False
        assert "citation" in decision.reason.lower()
        assert decision.confidence == 0.0

    async def test_refuse_low_confidence(self):
        """Test refusing response with very low confidence threshold."""
        from unittest.mock import patch

        from src.config import Settings

        # Test with higher confidence threshold
        test_settings = Settings()
        test_settings.rag_min_confidence = 0.5  # Require 50% confidence

        docs = [
            Chunk("DOC-001", "Policy one."),
            Chunk("DOC-002", "Policy two."),
            Chunk("DOC-003", "Policy three."),
            Chunk("DOC-004", "Policy four."),
        ]
        query = "What is the policy?"
        draft = "The policy requires something [DOC-001]."  # Only 1/4 = 25% coverage

        with patch("src.guard.get_settings", return_value=test_settings):
            decision = await approve_or_refuse(query, draft, docs)

        # With 50% threshold and only 25% citation coverage, should be rejected
        assert decision.allowed is False
        assert "confidence" in decision.reason.lower()

    async def test_refuse_empty_response(self, sample_docs):
        """Test refusing empty response."""
        query = "What is the policy?"
        draft = ""
        decision = await approve_or_refuse(query, draft, sample_docs)

        assert decision.allowed is False
        assert decision.confidence == 0.0

    async def test_refuse_hallucination(self, sample_docs):
        """Test refusing response with hallucination indicators."""
        query = "What is the policy?"
        draft = "I think the policy requires MFA [DOC-001]."
        decision = await approve_or_refuse(query, draft, sample_docs)

        assert decision.allowed is False
        assert "uncertain language" in decision.reason

    async def test_decision_metadata(self, sample_docs):
        """Test decision includes proper metadata."""
        query = "What is the MFA policy?"
        draft = "Security policy requires MFA [DOC-001] and strong passwords [DOC-002]."
        decision = await approve_or_refuse(query, draft, sample_docs)

        assert isinstance(decision, Decision)
        assert decision.citations_found == 2
        assert decision.citations_expected == len(sample_docs)
        assert 0.0 <= decision.confidence <= 1.0
