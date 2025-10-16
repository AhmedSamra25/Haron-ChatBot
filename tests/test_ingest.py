"""Tests for ingestion functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.services.podcast_client import PodcastAPIClient
from app.services.wp_scraper import WordPressScraper
from app.services.chunker import ContentChunker
from app.services.tasks import IngestionTasks
from app.models.schemas import PodcastEpisode, BlogPost


class TestPodcastAPIClient:
    """Tests for podcast API client."""
    
    @pytest.fixture
    def client(self):
        return PodcastAPIClient()
    
    @pytest.fixture
    def mock_episode_data(self):
        return {
            "id": "123",
            "title": "Test Episode",
            "description": "Test description",
            "content": "Test content",
            "published_date": "2024-01-01T00:00:00Z",
            "duration": "30:00",
            "author": "Test Author",
            "tags": ["business", "podcast"]
        }
    
    @pytest.mark.asyncio
    async def test_discover_pagination(self, client):
        """Test pagination discovery."""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                "total_pages": 5,
                "episodes": [{"id": "1", "title": "Test"}]
            }
            
            result = await client.discover_pagination()
            
            assert result["has_pagination"] is True
            assert result["page_param"] == "page"
    
    @pytest.mark.asyncio
    async def test_parse_episode(self, client, mock_episode_data):
        """Test episode parsing."""
        episode = client._parse_episode(mock_episode_data)
        
        assert episode is not None
        assert episode.id == "123"
        assert episode.title == "Test Episode"
        assert episode.author == "Test Author"
        assert len(episode.tags) == 2
    
    @pytest.mark.asyncio
    async def test_fetch_all_episodes(self, client, mock_episode_data):
        """Test fetching all episodes."""
        with patch.object(client, 'discover_pagination') as mock_pagination, \
             patch.object(client, '_make_request') as mock_request:
            
            mock_pagination.return_value = {
                "has_pagination": False,
                "data": {"episodes": [mock_episode_data]}
            }
            
            episodes = []
            async for episode in client.fetch_all_episodes():
                episodes.append(episode)
            
            assert len(episodes) == 1
            assert episodes[0].title == "Test Episode"


class TestWordPressScraper:
    """Tests for WordPress scraper."""
    
    @pytest.fixture
    def scraper(self):
        return WordPressScraper()
    
    @pytest.fixture
    def mock_html(self):
        return """
        <html>
            <head><title>Test Post</title></head>
            <body>
                <article>
                    <h1 class="entry-title">Test Blog Post</h1>
                    <div class="entry-content">
                        <p>This is test content for the blog post.</p>
                        <p>It contains multiple paragraphs.</p>
                    </div>
                    <div class="author">Test Author</div>
                    <time datetime="2024-01-01T00:00:00Z">January 1, 2024</time>
                </article>
            </body>
        </html>
        """
    
    def test_extract_text_content(self, scraper, mock_html):
        """Test text content extraction."""
        text = scraper._extract_text_content(mock_html)
        
        assert "Test Blog Post" in text
        assert "This is test content" in text
        assert "multiple paragraphs" in text
    
    def test_parse_blog_post(self, scraper, mock_html):
        """Test blog post parsing."""
        url = "https://example.com/test-post"
        post = scraper._parse_blog_post(url, mock_html)
        
        assert post is not None
        assert post.title == "Test Blog Post"
        assert "test content" in post.content.lower()
        assert post.url == url
    
    def test_should_scrape_url(self, scraper):
        """Test URL filtering."""
        base_url = "https://businessbelarabi.com"
        scraper.base_url = base_url
        
        # Valid URLs
        assert scraper._should_scrape_url(f"{base_url}/blog/test-post")
        assert scraper._should_scrape_url(f"{base_url}/article")
        
        # Invalid URLs
        assert not scraper._should_scrape_url("https://other-domain.com/post")
        assert not scraper._should_scrape_url(f"{base_url}/wp-admin/")
        assert not scraper._should_scrape_url(f"{base_url}/image.jpg")


class TestContentChunker:
    """Tests for content chunker."""
    
    @pytest.fixture
    def chunker(self):
        return ContentChunker()
    
    @pytest.fixture
    def mock_episode(self):
        return PodcastEpisode(
            id="episode_123",
            title="Test Episode",
            description="Test description",
            content="This is a long piece of content that should be chunked into smaller pieces. " * 50,
            published_date=datetime.now(),
            author="Test Author",
            tags=["business"]
        )
    
    @pytest.fixture
    def mock_blog_post(self):
        return BlogPost(
            id="post_123",
            title="Test Post",
            content="This is a long blog post content that should be chunked appropriately. " * 50,
            published_date=datetime.now(),
            author="Test Author",
            tags=["business"]
        )
    
    def test_split_by_sentences(self, chunker):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = chunker._split_by_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
    
    def test_chunk_podcast_episode(self, chunker, mock_episode):
        """Test podcast episode chunking."""
        chunks = chunker.chunk_podcast_episode(mock_episode)
        
        assert len(chunks) > 0
        assert all(chunk.content_type == "podcast" for chunk in chunks)
        assert all(chunk.content_id == "episode_123" for chunk in chunks)
        assert all(chunk.metadata.get("title") == "Test Episode" for chunk in chunks)
    
    def test_chunk_blog_post(self, chunker, mock_blog_post):
        """Test blog post chunking."""
        chunks = chunker.chunk_blog_post(mock_blog_post)
        
        assert len(chunks) > 0
        assert all(chunk.content_type == "blog" for chunk in chunks)
        assert all(chunk.content_id == "post_123" for chunk in chunks)
        assert all(chunk.metadata.get("title") == "Test Post" for chunk in chunks)
    
    def test_get_chunk_stats(self, chunker, mock_episode):
        """Test chunk statistics."""
        chunks = chunker.chunk_podcast_episode(mock_episode)
        stats = chunker.get_chunk_stats(chunks)
        
        assert stats["count"] == len(chunks)
        assert stats["avg_length"] > 0
        assert stats["total_length"] > 0


class TestIngestionTasks:
    """Tests for background ingestion tasks."""
    
    @pytest.fixture
    def ingestion_tasks(self):
        with patch('app.services.tasks.redis.from_url'):
            return IngestionTasks()
    
    @pytest.mark.asyncio
    async def test_ingest_podcasts(self, ingestion_tasks):
        """Test podcast ingestion."""
        mock_episode = PodcastEpisode(
            id="test_123",
            title="Test Episode",
            content="Test content",
            published_date=datetime.now()
        )
        
        with patch('app.services.tasks.podcast_client') as mock_client, \
             patch('app.services.tasks.content_chunker') as mock_chunker, \
             patch('app.services.tasks.vector_store') as mock_store, \
             patch('app.services.tasks.get_current_job') as mock_job:
            
            # Setup mocks
            mock_job.return_value = Mock(id="test_job")
            mock_client.fetch_all_episodes.return_value = iter([mock_episode])
            mock_chunker.chunk_podcast_episode.return_value = [Mock()]
            mock_store.add_chunks.return_value = True
            
            # Run ingestion
            result = await ingestion_tasks.ingest_podcasts()
            
            # Verify results
            assert result["episodes_processed"] == 1
            assert result["episodes_failed"] == 0
            assert "started_at" in result
            assert "completed_at" in result
    
    @pytest.mark.asyncio
    async def test_ingest_blog_posts(self, ingestion_tasks):
        """Test blog post ingestion."""
        mock_post = BlogPost(
            id="post_123",
            title="Test Post",
            content="Test content",
            published_date=datetime.now()
        )
        
        with patch('app.services.tasks.wp_scraper') as mock_scraper, \
             patch('app.services.tasks.content_chunker') as mock_chunker, \
             patch('app.services.tasks.vector_store') as mock_store, \
             patch('app.services.tasks.get_current_job') as mock_job:
            
            # Setup mocks
            mock_job.return_value = Mock(id="test_job")
            mock_scraper.scrape_all_posts.return_value = iter([mock_post])
            mock_chunker.chunk_blog_post.return_value = [Mock()]
            mock_store.add_chunks.return_value = True
            
            # Run ingestion
            result = await ingestion_tasks.ingest_blog_posts()
            
            # Verify results
            assert result["posts_processed"] == 1
            assert result["posts_failed"] == 0
            assert "started_at" in result
            assert "completed_at" in result
    
    @pytest.mark.asyncio
    async def test_ingest_all_content(self, ingestion_tasks):
        """Test full content ingestion."""
        with patch.object(ingestion_tasks, 'ingest_podcasts') as mock_podcasts, \
             patch.object(ingestion_tasks, 'ingest_blog_posts') as mock_blogs, \
             patch('app.services.tasks.get_current_job') as mock_job:
            
            # Setup mocks
            mock_job.return_value = Mock(id="test_job")
            mock_podcasts.return_value = {"episodes_processed": 5, "episodes_failed": 0, "chunks_created": 20}
            mock_blogs.return_value = {"posts_processed": 10, "posts_failed": 1, "chunks_created": 35}
            
            # Run full ingestion
            result = await ingestion_tasks.ingest_all_content()
            
            # Verify results
            assert result["total_items_processed"] == 15
            assert result["total_items_failed"] == 1
            assert result["total_chunks_created"] == 55
            assert "podcast_stats" in result
            assert "blog_stats" in result
