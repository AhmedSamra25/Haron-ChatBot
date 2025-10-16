"""Tests for chat functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ChatRequest, ChatMessage, Conversation
from app.models.types import MessageRole
from app.services.llm import GeminiService
from app.services.memory import ConversationMemoryService
from app.services.retriever import RetrievalService


class TestGeminiService:
    """Tests for Gemini LLM service."""
    
    @pytest.fixture
    def gemini_service(self):
        with patch('app.services.llm.genai.configure'):
            return GeminiService()
    
    @pytest.mark.asyncio
    async def test_generate_chat_response(self, gemini_service):
        """Test chat response generation."""
        with patch('app.services.llm.asyncio.to_thread') as mock_thread:
            # Mock Gemini response
            mock_response = Mock()
            mock_response.text = "مرحباً! كيف يمكنني مساعدتك؟"
            mock_thread.return_value = mock_response
            
            response = await gemini_service.generate_chat_response(
                user_message="مرحبا",
                context="Some context",
                conversation_history=[]
            )
            
            assert response == "مرحباً! كيف يمكنني مساعدتك؟"
            mock_thread.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, gemini_service):
        """Test content summarization."""
        with patch('app.services.llm.asyncio.to_thread') as mock_thread:
            mock_response = Mock()
            mock_response.text = "ملخص المحتوى"
            mock_thread.return_value = mock_response
            
            summary = await gemini_service.generate_summary("Long content here...")
            
            assert summary == "ملخص المحتوى"
    
    @pytest.mark.asyncio
    async def test_extract_keywords(self, gemini_service):
        """Test keyword extraction."""
        with patch('app.services.llm.asyncio.to_thread') as mock_thread:
            mock_response = Mock()
            mock_response.text = "أعمال, ريادة, استثمار"
            mock_thread.return_value = mock_response
            
            keywords = await gemini_service.extract_keywords("Business content here...")
            
            assert len(keywords) == 3
            assert "أعمال" in keywords


class TestConversationMemoryService:
    """Tests for conversation memory service."""
    
    @pytest.fixture
    def memory_service(self):
        with patch('app.services.memory.redis.from_url'):
            return ConversationMemoryService()
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, memory_service):
        """Test conversation creation."""
        with patch.object(memory_service, '_store_conversation') as mock_store, \
             patch.object(memory_service, '_add_to_user_conversations') as mock_add:
            
            conversation_id = await memory_service.create_conversation(user_id="user123")
            
            assert isinstance(conversation_id, str)
            assert len(conversation_id) > 0
            mock_store.assert_called_once()
            mock_add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_message(self, memory_service):
        """Test adding message to conversation."""
        # Mock existing conversation
        mock_conversation = Conversation(
            id="conv123",
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with patch.object(memory_service, 'get_conversation', return_value=mock_conversation), \
             patch.object(memory_service, '_store_conversation') as mock_store:
            
            message = ChatMessage(role=MessageRole.USER, content="Test message")
            success = await memory_service.add_message("conv123", message)
            
            assert success is True
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, memory_service):
        """Test retrieving conversation history."""
        mock_messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        mock_conversation = Conversation(
            id="conv123",
            messages=mock_messages,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        with patch.object(memory_service, 'get_conversation', return_value=mock_conversation):
            history = await memory_service.get_conversation_history("conv123")
            
            assert len(history) == 2
            assert history[0].content == "Hello"
            assert history[1].content == "Hi there!"


class TestRetrievalService:
    """Tests for retrieval service."""
    
    @pytest.fixture
    def retrieval_service(self):
        return RetrievalService()
    
    @pytest.mark.asyncio
    async def test_retrieve_context(self, retrieval_service):
        """Test context retrieval."""
        mock_search_results = [
            {
                "id": "chunk1",
                "content_id": "episode1",
                "content_type": "podcast",
                "text": "Test content about business",
                "similarity_score": 0.8,
                "metadata": {"title": "Business Tips"}
            }
        ]
        
        with patch('app.services.retriever.vector_store') as mock_store:
            mock_store.search_similar.return_value = mock_search_results
            
            result = await retrieval_service.retrieve_context(
                query="business advice",
                max_results=5
            )
            
            assert "context" in result
            assert "sources" in result
            assert result["total_found"] == 1
            assert len(result["sources"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_trending_content(self, retrieval_service):
        """Test trending content retrieval."""
        mock_results = [
            {
                "content_id": "episode1",
                "content_type": "podcast",
                "similarity_score": 0.7,
                "metadata": {"title": "Trending Episode"}
            }
        ]
        
        with patch('app.services.retriever.vector_store') as mock_store:
            mock_store.search_similar.return_value = mock_results
            
            trending = await retrieval_service.get_trending_content(limit=5)
            
            assert len(trending) > 0
            assert trending[0]["content_type"] == "podcast"


class TestChatAPI:
    """Tests for chat API endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer local-dev-key"}
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        """Test chat endpoint."""
        with patch('app.services.llm.gemini_service') as mock_llm, \
             patch('app.services.memory.memory_service') as mock_memory, \
             patch('app.services.retriever.retrieval_service') as mock_retriever:
            
            # Setup mocks
            mock_retriever.retrieve_context.return_value = {
                "context": "Test context",
                "sources": [],
                "total_found": 1,
                "search_time_ms": 100.0
            }
            mock_llm.generate_chat_response.return_value = "مرحباً! كيف يمكنني مساعدتك؟"
            mock_memory.get_conversation_history.return_value = []
            mock_memory.add_message.return_value = True
            
            # Make request
            response = self.client.post(
                "/chat/",
                json={
                    "message": "مرحبا",
                    "use_context": True,
                    "max_context_results": 5
                },
                headers=self.headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "conversation_id" in data
            assert "sources" in data
    
    def test_chat_without_api_key(self):
        """Test chat endpoint without API key."""
        response = self.client.post(
            "/chat/",
            json={"message": "test"}
        )
        
        assert response.status_code == 401
    
    def test_chat_with_invalid_request(self):
        """Test chat endpoint with invalid request."""
        response = self.client.post(
            "/chat/",
            json={"message": ""},  # Empty message
            headers=self.headers
        )
        
        assert response.status_code == 422  # Validation error
