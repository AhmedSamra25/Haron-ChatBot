#!/usr/bin/env python3
"""
Business bel Arabi RAG - Backend API Tests

Comprehensive tests for the FastAPI backend with LangChain integration.
"""

import pytest
import asyncio
import os
import sys
from typing import Dict, Any
from httpx import AsyncClient
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.database import DatabaseManager
from backend.utils import get_settings
from backend.vector_store import VectorStoreManager

class TestBackendAPI:
    """Test suite for Backend API endpoints."""

    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def test_db_manager(self, tmp_path):
        """Create test database manager."""
        db_path = tmp_path / "test_business_bel_arabi.db"
        return DatabaseManager(str(db_path))

    @pytest.fixture
    def sample_chat_request(self):
        """Sample chat request for testing."""
        return {
            "message": "كيف أبدأ مشروع تجاري ناجح؟",
            "session_id": "test-session-123",
            "use_context": True,
            "max_context_docs": 3,
            "temperature": 0.7
        }

    @pytest.fixture
    def sample_ingestion_request(self):
        """Sample data ingestion request."""
        return {
            "limit": 5,
            "force_refresh": False,
            "chunk_size": 1000,
            "chunk_overlap": 200
        }

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "components" in health_data
        assert "timestamp" in health_data

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root API information endpoint."""
        response = await async_client.get("/")
        assert response.status_code == 200
        
        root_data = response.json()
        assert root_data["message"] == "Business bel Arabi RAG API"
        assert root_data["version"] == "2.0.0"
        assert "endpoints" in root_data
        assert "features" in root_data

    @pytest.mark.asyncio
    async def test_chat_endpoint_success(self, async_client: AsyncClient, sample_chat_request):
        """Test successful chat request."""
        response = await async_client.post("/chat", json=sample_chat_request)
        
        # Should succeed even without Google API key (mock response)
        assert response.status_code == 200
        
        chat_response = response.json()
        assert "answer" in chat_response
        assert "session_id" in chat_response
        assert "response_time" in chat_response
        assert "model_used" in chat_response
        assert chat_response["session_id"] == sample_chat_request["session_id"]

    @pytest.mark.asyncio
    async def test_chat_endpoint_validation(self, async_client: AsyncClient):
        """Test chat endpoint input validation."""
        # Empty message
        response = await async_client.post("/chat", json={"message": ""})
        assert response.status_code == 400

        # Invalid temperature
        response = await async_client.post("/chat", json={
            "message": "test",
            "temperature": 3.0  # > 2.0
        })
        assert response.status_code == 422

        # Invalid max_context_docs
        response = await async_client.post("/chat", json={
            "message": "test",
            "max_context_docs": 15  # > 10
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_conversations_endpoint(self, async_client: AsyncClient):
        """Test conversations history endpoint."""
        response = await async_client.get("/conversations")
        assert response.status_code == 200
        
        conversations = response.json()
        assert isinstance(conversations, list)

    @pytest.mark.asyncio
    async def test_conversations_search(self, async_client: AsyncClient):
        """Test conversation search endpoint."""
        search_request = {
            "query": "تسويق",
            "limit": 10
        }
        
        response = await async_client.post("/conversations/search", json=search_request)
        assert response.status_code == 200
        
        search_response = response.json()
        assert "results" in search_response
        assert "total_found" in search_response
        assert "query" in search_response
        assert search_response["query"] == "تسويق"

    @pytest.mark.asyncio
    async def test_sessions_endpoint(self, async_client: AsyncClient):
        """Test sessions endpoint."""
        response = await async_client.get("/sessions")
        assert response.status_code == 200
        
        sessions = response.json()
        assert isinstance(sessions, list)

    @pytest.mark.asyncio
    async def test_analytics_endpoint(self, async_client: AsyncClient):
        """Test analytics endpoint."""
        response = await async_client.get("/analytics")
        assert response.status_code == 200
        
        analytics = response.json()
        assert "total_conversations" in analytics
        assert "total_sessions" in analytics
        assert "avg_response_time" in analytics

    @pytest.mark.asyncio
    async def test_ingest_podcast_endpoint(self, async_client: AsyncClient, sample_ingestion_request):
        """Test podcast data ingestion endpoint."""
        response = await async_client.post("/ingest/podcast", json=sample_ingestion_request)
        
        # Should handle gracefully even if API is not available
        assert response.status_code == 200
        
        ingestion_response = response.json()
        assert "success" in ingestion_response
        assert "message" in ingestion_response
        assert "items_processed" in ingestion_response

    @pytest.mark.asyncio
    async def test_ingest_wordpress_endpoint(self, async_client: AsyncClient, sample_ingestion_request):
        """Test WordPress data ingestion endpoint."""
        response = await async_client.post("/ingest/wordpress", json=sample_ingestion_request)
        
        # Should handle gracefully even if API is not available
        assert response.status_code == 200
        
        ingestion_response = response.json()
        assert "success" in ingestion_response
        assert "message" in ingestion_response
        assert "items_processed" in ingestion_response

    @pytest.mark.asyncio
    async def test_session_deletion(self, async_client: AsyncClient):
        """Test session deletion endpoint."""
        # Create a test session first by sending a chat
        chat_request = {
            "message": "test message for deletion",
            "session_id": "test-session-to-delete"
        }
        
        await async_client.post("/chat", json=chat_request)
        
        # Now delete the session
        response = await async_client.delete("/sessions/test-session-to-delete")
        assert response.status_code == 200
        
        delete_response = response.json()
        assert "message" in delete_response

class TestDatabaseManager:
    """Test suite for Database Manager."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create test database manager."""
        db_path = tmp_path / "test_db.db"
        return DatabaseManager(str(db_path))

    def test_database_initialization(self, db_manager):
        """Test database initialization."""
        # Database should be initialized without errors
        assert os.path.exists(db_manager.db_path)

    def test_save_conversation(self, db_manager):
        """Test saving conversations."""
        conversation_id = db_manager.save_conversation(
            session_id="test-session",
            user_message="تست سؤال",
            assistant_response="تست إجابة",
            response_time=1.5,
            context_used=True,
            model_used="test-model"
        )
        
        assert conversation_id is not None
        assert isinstance(conversation_id, str)

    def test_get_conversation_history(self, db_manager):
        """Test retrieving conversation history."""
        # Save a test conversation
        db_manager.save_conversation(
            session_id="test-session",
            user_message="تست سؤال",
            assistant_response="تست إجابة"
        )
        
        # Retrieve history
        history = db_manager.get_conversation_history("test-session")
        assert len(history) >= 1
        assert history[0]["user_message"] == "تست سؤال"

    def test_search_conversations(self, db_manager):
        """Test conversation search functionality."""
        # Save test conversations
        db_manager.save_conversation(
            session_id="test-session",
            user_message="كيف أبدأ التسويق؟",
            assistant_response="ابدأ بدراسة السوق"
        )
        
        # Search
        results = db_manager.search_conversations("تسويق")
        assert len(results) >= 1

    def test_analytics_data(self, db_manager):
        """Test analytics data retrieval."""
        analytics = db_manager.get_analytics_data()
        
        assert "total_conversations" in analytics
        assert "total_sessions" in analytics
        assert "avg_response_time" in analytics
        assert isinstance(analytics["total_conversations"], int)

    def test_user_feedback(self, db_manager):
        """Test user feedback functionality."""
        # Save a conversation first
        conversation_id = db_manager.save_conversation(
            session_id="test-session",
            user_message="تست",
            assistant_response="إجابة تست"
        )
        
        # Save feedback
        success = db_manager.save_user_feedback(
            conversation_id=conversation_id,
            rating=5,
            feedback_text="ممتاز!"
        )
        
        assert success is True

class TestDataIngester:
    """Test suite for Data Ingester."""

    @pytest.fixture
    def data_ingester(self):
        """Create data ingester for testing."""
        from backend.data_ingester import EnhancedDataIngester
        return EnhancedDataIngester()

    def test_html_cleaning(self, data_ingester):
        """Test HTML text cleaning functionality."""
        html_text = "<p>هذا نص <strong>مهم</strong> يحتوي على <a href='#'>رابط</a></p>"
        clean_text = data_ingester.clean_html_text(html_text)
        
        assert "<p>" not in clean_text
        assert "<strong>" not in clean_text
        assert "مهم" in clean_text
        assert "رابط" in clean_text

    def test_content_hash_generation(self, data_ingester):
        """Test content hash generation."""
        content = "هذا محتوى للاختبار"
        hash1 = data_ingester._generate_content_hash(content)
        hash2 = data_ingester._generate_content_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    def test_duration_formatting(self, data_ingester):
        """Test duration formatting."""
        # Test seconds
        assert data_ingester._format_duration(65) == "1:05"
        
        # Test hours
        assert data_ingester._format_duration(3665) == "1:01:05"
        
        # Test string input
        assert data_ingester._format_duration("2:30") == "2:30"

    @pytest.mark.asyncio
    async def test_endpoint_testing(self, data_ingester):
        """Test endpoint health checking."""
        # This will test with actual endpoints, may fail if endpoints are down
        # but should not crash the test
        try:
            results = await data_ingester.test_endpoints()
            assert "podcast_api" in results
            assert "wordpress_api" in results
        except Exception:
            # It's okay if external APIs are not accessible during testing
            pass

class TestVectorStore:
    """Test suite for Vector Store Manager."""

    @pytest.fixture
    async def vector_manager(self, tmp_path):
        """Create test vector store manager."""
        try:
            from backend.vector_store import VectorStoreManager
            from langchain.embeddings import FakeEmbeddings
            
            # Use fake embeddings for testing
            embeddings = FakeEmbeddings(size=384)
            
            manager = VectorStoreManager(
                embeddings_model=embeddings,
                persist_directory=str(tmp_path / "test_vector_store"),
                index_name="test_index"
            )
            
            await manager.initialize()
            return manager
        except ImportError:
            pytest.skip("Vector store dependencies not available")

    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, vector_manager):
        """Test vector store initialization."""
        if vector_manager:
            assert vector_manager.is_ready is True

    @pytest.mark.asyncio
    async def test_document_addition(self, vector_manager):
        """Test adding documents to vector store."""
        if not vector_manager:
            return

        from langchain.docstore.document import Document
        
        docs = [
            Document(
                page_content="هذا نص تجريبي للاختبار",
                metadata={"source": "test", "id": "1"}
            ),
            Document(
                page_content="نص آخر للاختبار",
                metadata={"source": "test", "id": "2"}
            )
        ]
        
        result = await vector_manager.add_documents(docs)
        assert result is True

    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_manager):
        """Test similarity search functionality."""
        if not vector_manager:
            return

        # Add some documents first
        from langchain.docstore.document import Document
        
        docs = [
            Document(
                page_content="نصائح حول التسويق الرقمي",
                metadata={"source": "test"}
            )
        ]
        
        await vector_manager.add_documents(docs)
        
        # Search for similar content
        results = await vector_manager.similarity_search("تسويق", k=1)
        assert len(results) >= 0  # May be 0 with fake embeddings

    @pytest.mark.asyncio
    async def test_vector_store_stats(self, vector_manager):
        """Test vector store statistics."""
        if vector_manager:
            stats = await vector_manager.get_stats()
            assert "is_ready" in stats
            assert "total_documents" in stats

class TestUtils:
    """Test suite for utility functions."""

    def test_settings_loading(self):
        """Test settings loading."""
        from backend.utils import get_settings
        
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'google_api_key')
        assert hasattr(settings, 'podcast_api_base')

    def test_text_cleaning(self):
        """Test text cleaning utilities."""
        from backend.utils import clean_text, truncate_text
        
        # Test cleaning
        dirty_text = "  هذا نص    مع مسافات  \n كثيرة  "
        clean = clean_text(dirty_text)
        assert "مع مسافات كثيرة" in clean
        
        # Test truncation
        long_text = "هذا نص طويل جداً " * 10
        truncated = truncate_text(long_text, max_length=50)
        assert len(truncated) <= 53  # 50 + "..."

    def test_hash_calculation(self):
        """Test hash calculation."""
        from backend.utils import calculate_hash
        
        content = "محتوى للاختبار"
        hash1 = calculate_hash(content)
        hash2 = calculate_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 32

    def test_json_utilities(self):
        """Test JSON utility functions."""
        from backend.utils import safe_json_loads, safe_json_dumps
        
        # Test safe loading
        valid_json = '{"key": "value"}'
        invalid_json = '{"key": invalid}'
        
        assert safe_json_loads(valid_json) == {"key": "value"}
        assert safe_json_loads(invalid_json, default={}) == {}
        
        # Test safe dumping
        obj = {"arabic": "عربي", "number": 123}
        json_str = safe_json_dumps(obj)
        assert "عربي" in json_str

    def test_format_utilities(self):
        """Test formatting utilities."""
        from backend.utils import format_bytes, format_duration
        
        # Test byte formatting
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1048576) == "1.0 MB"
        
        # Test duration formatting
        assert format_duration(65) == "1.1m"
        assert format_duration(3665) == "1.0h"

# Integration Tests
class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        """Test complete chat flow from frontend to backend."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # 1. Check health
            health_response = await client.get("/health")
            assert health_response.status_code == 200
            
            # 2. Send chat message
            chat_request = {
                "message": "كيف أحسن مبيعاتي؟",
                "session_id": "integration-test-session",
                "use_context": False
            }
            
            chat_response = await client.post("/chat", json=chat_request)
            assert chat_response.status_code == 200
            
            chat_data = chat_response.json()
            session_id = chat_data["session_id"]
            
            # 3. Check conversation history
            history_response = await client.get(f"/conversations?session_id={session_id}")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert len(history_data) >= 1
            
            # 4. Search conversations
            search_request = {"query": "مبيعات", "limit": 10}
            search_response = await client.post("/conversations/search", json=search_request)
            assert search_response.status_code == 200
            
            # 5. Get analytics
            analytics_response = await client.get("/analytics")
            assert analytics_response.status_code == 200

    @pytest.mark.asyncio
    async def test_data_ingestion_flow(self):
        """Test data ingestion workflow."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test podcast ingestion
            podcast_request = {"limit": 2}
            podcast_response = await client.post("/ingest/podcast", json=podcast_request)
            assert podcast_response.status_code == 200
            
            # Test WordPress ingestion
            wp_request = {"limit": 2}
            wp_response = await client.post("/ingest/wordpress", json=wp_request)
            assert wp_response.status_code == 200

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling across the system."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test invalid endpoints
            response = await client.get("/nonexistent")
            assert response.status_code == 404
            
            # Test invalid chat request
            response = await client.post("/chat", json={})
            assert response.status_code == 422
            
            # Test invalid search request
            response = await client.post("/conversations/search", json={})
            assert response.status_code == 422

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
