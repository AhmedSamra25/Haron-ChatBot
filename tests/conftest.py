#!/usr/bin/env python3
"""
Business bel Arabi RAG - Test Configuration

Pytest configuration and fixtures for the test suite.
"""

import pytest
import asyncio
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_google_api_key():
    """Mock Google API key for tests."""
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-api-key-for-testing'}):
        yield 'test-api-key-for-testing'

@pytest.fixture
def mock_external_apis():
    """Mock external API responses."""
    podcast_response = {
        "data": [
            {
                "id": 1,
                "title": "حلقة تجريبية",
                "description": "وصف تجريبي",
                "transcript": "نص الحلقة التجريبي",
                "duration": "15:30",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
    }
    
    wordpress_response = [
        {
            "id": 1,
            "title": {"rendered": "مقال تجريبي"},
            "content": {"rendered": "محتوى المقال التجريبي"},
            "excerpt": {"rendered": "مقتطف من المقال"},
            "date": "2024-01-01T00:00:00Z"
        }
    ]
    
    with patch('requests.get') as mock_get:
        def side_effect(*args, **kwargs):
            url = args[0]
            mock_response = Mock()
            mock_response.status_code = 200
            
            if 'podcast' in url:
                mock_response.json.return_value = podcast_response
            elif 'wp-json' in url:
                mock_response.json.return_value = wordpress_response
            else:
                mock_response.json.return_value = {}
            
            return mock_response
        
        mock_get.side_effect = side_effect
        yield {
            'podcast': podcast_response,
            'wordpress': wordpress_response
        }

@pytest.fixture
def sample_documents():
    """Sample documents for vector store testing."""
    from langchain.docstore.document import Document
    
    return [
        Document(
            page_content="كيفية بدء مشروع تجاري ناجح في السوق العربي",
            metadata={
                "source": "test",
                "title": "ريادة الأعمال",
                "id": "1"
            }
        ),
        Document(
            page_content="استراتيجيات التسويق الرقمي الفعالة للشركات الناشئة",
            metadata={
                "source": "test", 
                "title": "التسويق الرقمي",
                "id": "2"
            }
        ),
        Document(
            page_content="إدارة الفرق والموظفين في الشركات الصغيرة والمتوسطة",
            metadata={
                "source": "test",
                "title": "إدارة الأعمال", 
                "id": "3"
            }
        )
    ]

@pytest.fixture
def sample_conversations():
    """Sample conversation data for testing."""
    return [
        {
            "user_message": "كيف أبدأ مشروع تجاري؟",
            "assistant_response": "لبدء مشروع تجاري ناجح، تحتاج إلى دراسة السوق أولاً...",
            "session_id": "test-session-1"
        },
        {
            "user_message": "ما هي أفضل استراتيجيات التسويق؟",
            "assistant_response": "التسويق الرقمي يعتبر من أهم الاستراتيجيات الحديثة...",
            "session_id": "test-session-1"
        },
        {
            "user_message": "كيف أدير فريق العمل؟",
            "assistant_response": "إدارة الفريق تتطلب التواصل الفعال والتحفيز...",
            "session_id": "test-session-2"
        }
    ]

@pytest.fixture
def mock_langchain_embeddings():
    """Mock embeddings for testing."""
    class MockEmbeddings:
        def embed_documents(self, texts):
            # Return fake embeddings
            return [[0.1, 0.2, 0.3] for _ in texts]
        
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]
    
    return MockEmbeddings()

@pytest.fixture
def mock_langchain_llm():
    """Mock LLM for testing."""
    class MockLLM:
        async def ainvoke(self, prompt):
            class MockResponse:
                content = "هذه إجابة تجريبية من النموذج المحاكي للاختبار."
            return MockResponse()
        
        def invoke(self, prompt):
            return self.ainvoke(prompt)
    
    return MockLLM()

@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables before each test."""
    # Store original values
    original_env = dict(os.environ)
    
    # Set test environment
    test_env = {
        'DEBUG': 'true',
        'LOG_LEVEL': 'ERROR',  # Reduce noise in tests
        'DATABASE_PATH': ':memory:',  # Use in-memory database for tests
        'VECTOR_STORE_PATH': 'test_vector_store'
    }
    
    os.environ.update(test_env)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "external_api: marks tests that require external APIs"
    )

# Test collection modifications
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add external_api marker to tests that use external APIs
        if "external" in item.name or "api" in item.name:
            item.add_marker(pytest.mark.external_api)

# Async test support
@pytest.fixture(scope="session")
def anyio_backend():
    """Select asyncio as the backend for anyio."""
    return "asyncio"
