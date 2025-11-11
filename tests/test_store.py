"""Tests for document store functionality."""

import pytest

from src.store import Chunk, InMemoryStore


class TestChunk:
    """Test Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(id="TEST-001", text="Test content")
        assert chunk.id == "TEST-001"
        assert chunk.text == "Test content"

    def test_chunk_hashable(self):
        """Test that chunks are hashable."""
        chunk1 = Chunk(id="TEST-001", text="Test content")
        chunk2 = Chunk(id="TEST-001", text="Test content")
        assert hash(chunk1) == hash(chunk2)
        assert {chunk1, chunk2} == {chunk1}


class TestInMemoryStore:
    """Test InMemoryStore functionality."""

    @pytest.fixture
    def store(self):
        """Create a store with sample data."""
        return InMemoryStore.from_samples()

    @pytest.fixture
    def empty_store(self):
        """Create an empty store."""
        return InMemoryStore([])

    async def test_from_samples(self, store):
        """Test creating store from samples."""
        assert len(store.chunks) > 0
        assert all(isinstance(c, Chunk) for c in store.chunks)

    async def test_search_exact_match(self, store):
        """Test search with exact phrase match."""
        results = await store.search("MFA is required", top_k=4)
        assert len(results) > 0
        assert any("MFA" in chunk.text for chunk in results)

    async def test_search_keyword_match(self, store):
        """Test search with keyword matching."""
        results = await store.search("security training", top_k=4)
        assert len(results) > 0
        # Should find HR-001 which mentions security training
        ids = [c.id for c in results]
        assert "HR-001" in ids

    async def test_search_empty_query(self, store):
        """Test search with empty query."""
        results = await store.search("", top_k=4)
        assert results == []

        results = await store.search("   ", top_k=4)
        assert results == []

    async def test_search_no_results(self, store):
        """Test search with query that matches nothing."""
        results = await store.search("xyzabc123nonexistent", top_k=4)
        assert results == []

    async def test_search_top_k(self, store):
        """Test search respects top_k limit."""
        results = await store.search("policy", top_k=2)
        assert len(results) <= 2

    async def test_add_chunk(self, empty_store):
        """Test adding a chunk."""
        chunk = Chunk(id="NEW-001", text="New content")
        await empty_store.add_chunk(chunk)
        assert len(empty_store.chunks) == 1
        assert empty_store.chunks[0] == chunk

    async def test_add_duplicate_chunk(self, store):
        """Test adding duplicate chunk is skipped."""
        initial_count = len(store.chunks)
        existing_chunk = store.chunks[0]
        await store.add_chunk(existing_chunk)
        assert len(store.chunks) == initial_count

    async def test_get_chunk(self, store):
        """Test retrieving a chunk by ID."""
        chunk = await store.get_chunk("HR-001")
        assert chunk is not None
        assert chunk.id == "HR-001"

    async def test_get_nonexistent_chunk(self, store):
        """Test retrieving non-existent chunk returns None."""
        chunk = await store.get_chunk("NONEXISTENT")
        assert chunk is None

    def test_score_exact_match(self, store):
        """Test scoring with exact phrase match."""
        score = store._score("MFA required", "MFA is required for admin")
        assert score >= 0.5  # Should have high score for phrase match

    def test_score_no_match(self, store):
        """Test scoring with no match."""
        score = store._score("xyz", "abc def")
        assert score == 0.0

    def test_score_partial_match(self, store):
        """Test scoring with partial match."""
        score = store._score("security training", "annual security training required")
        assert 0.3 < score <= 1.0  # Should have decent score for word overlap
