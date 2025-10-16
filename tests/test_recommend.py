"""Tests for recommendation functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.services.recommender import RecommendationService
from app.models.schemas import RecommendationRequest, RecommendedContent


class TestRecommendationService:
    """Tests for recommendation service."""
    
    @pytest.fixture
    def recommendation_service(self):
        return RecommendationService()
    
    @pytest.fixture
    def mock_sources(self):
        return [
            {
                "content_id": "episode1",
                "content_type": "podcast",
                "similarity_score": 0.8,
                "text": "Business advice content",
                "metadata": {
                    "title": "Business Tips Episode",
                    "author": "Business Expert",
                    "published_date": "2024-01-01T00:00:00Z",
                    "tags": ["business", "tips"]
                }
            },
            {
                "content_id": "post1",
                "content_type": "blog",
                "similarity_score": 0.7,
                "text": "Investment strategies content",
                "metadata": {
                    "title": "Investment Guide",
                    "author": "Financial Advisor",
                    "published_date": "2024-01-02T00:00:00Z",
                    "tags": ["investment", "finance"]
                }
            }
        ]
    
    @pytest.mark.asyncio
    async def test_get_query_based_recommendations(self, recommendation_service, mock_sources):
        """Test query-based recommendations."""
        request = RecommendationRequest(
            query="business advice",
            content_type="both",
            limit=5
        )
        
        with patch('app.services.recommender.retrieval_service') as mock_retrieval:
            mock_retrieval.retrieve_context.return_value = {
                "sources": mock_sources,
                "total_found": 2,
                "search_time_ms": 150.0
            }
            
            response = await recommendation_service.get_recommendations(request)
            
            assert len(response.recommendations) == 2
            assert response.query_used == "business advice"
            assert response.total_found == 2
            assert response.recommendations[0].content_type == "podcast"
            assert response.recommendations[1].content_type == "blog"
    
    @pytest.mark.asyncio
    async def test_get_trending_recommendations(self, recommendation_service, mock_sources):
        """Test trending content recommendations."""
        request = RecommendationRequest(
            content_type="podcast",
            limit=3
        )
        
        with patch('app.services.recommender.retrieval_service') as mock_retrieval:
            mock_retrieval.get_trending_content.return_value = mock_sources[:1]  # Only podcast
            
            response = await recommendation_service.get_recommendations(request)
            
            assert len(response.recommendations) == 1
            assert response.recommendations[0].content_type == "podcast"
            assert response.query_used is None
    
    def test_apply_recency_boost(self, recommendation_service, mock_sources):
        """Test recency boost application."""
        # Test with recent content
        recent_sources = mock_sources.copy()
        recent_sources[0]["published_date"] = datetime.now().isoformat()
        
        enhanced_sources = recommendation_service._apply_recency_boost(recent_sources)
        
        # Recent content should have higher score
        assert enhanced_sources[0]["similarity_score"] >= mock_sources[0]["similarity_score"]
        assert enhanced_sources[0].get("recency_boost_applied") is True
    
    def test_apply_diversity_filter(self, recommendation_service):
        """Test diversity filtering."""
        similar_sources = [
            {
                "content_id": "1",
                "text": "Business advice for entrepreneurs",
                "similarity_score": 0.9
            },
            {
                "content_id": "2", 
                "text": "Business advice for startups",  # Very similar
                "similarity_score": 0.8
            },
            {
                "content_id": "3",
                "text": "Cooking recipes and tips",  # Different topic
                "similarity_score": 0.7
            }
        ]
        
        diverse_results = recommendation_service._apply_diversity_filter(similar_sources, limit=3)
        
        # Should keep diverse content and filter out similar
        assert len(diverse_results) >= 2
        assert any("cooking" in result["text"].lower() for result in diverse_results)
    
    def test_convert_to_recommended_content(self, recommendation_service, mock_sources):
        """Test conversion to RecommendedContent objects."""
        source = mock_sources[0]
        
        recommended = recommendation_service._convert_to_recommended_content(source)
        
        assert recommended is not None
        assert recommended.id == "episode1"
        assert recommended.title == "Business Tips Episode"
        assert recommended.content_type == "podcast"
        assert recommended.similarity_score == 0.8
        assert "Business Expert" == recommended.author
        assert len(recommended.tags) == 2
    
    @pytest.mark.asyncio
    async def test_get_similar_content(self, recommendation_service):
        """Test getting similar content."""
        mock_related_data = {
            "related": [
                {
                    "content_id": "related1",
                    "content_type": "podcast",
                    "similarity_score": 0.75,
                    "text": "Related content",
                    "metadata": {"title": "Related Episode"}
                }
            ],
            "total_found": 1
        }
        
        with patch('app.services.recommender.retrieval_service') as mock_retrieval:
            mock_retrieval.retrieve_related_content.return_value = mock_related_data
            
            similar = await recommendation_service.get_similar_content("episode1", limit=5)
            
            assert len(similar) == 1
            assert similar[0].id == "related1"
            assert similar[0].content_type == "podcast"
    
    @pytest.mark.asyncio
    async def test_get_personalized_recommendations(self, recommendation_service):
        """Test personalized recommendations."""
        user_history = ["business", "investment", "entrepreneurship"]
        
        # Mock the fallback to regular recommendations
        mock_request = RecommendationRequest(
            query="business investment entrepreneurship",
            limit=10,
            content_type="both"
        )
        
        with patch.object(recommendation_service, 'get_recommendations') as mock_get_recs:
            mock_response = Mock()
            mock_response.recommendations = [
                Mock(id="rec1", title="Personalized Rec", content_type="podcast")
            ]
            mock_get_recs.return_value = mock_response
            
            recommendations = await recommendation_service.get_personalized_recommendations(
                user_history=user_history,
                limit=10
            )
            
            assert len(recommendations) == 1
            mock_get_recs.assert_called_once()
    
    def test_calculate_text_similarity(self, recommendation_service):
        """Test text similarity calculation."""
        text1 = "business advice for entrepreneurs"
        text2 = "business tips for entrepreneurs"
        text3 = "cooking recipes and kitchen tips"
        
        # Similar texts should have high similarity
        similarity_high = recommendation_service._calculate_text_similarity(text1, text2)
        assert similarity_high > 0.5
        
        # Different texts should have low similarity
        similarity_low = recommendation_service._calculate_text_similarity(text1, text3)
        assert similarity_low < 0.3
        
        # Empty texts should have zero similarity
        similarity_empty = recommendation_service._calculate_text_similarity("", text1)
        assert similarity_empty == 0.0
