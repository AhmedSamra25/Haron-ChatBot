#!/usr/bin/env python3
"""
Business bel Arabi RAG - API Models

Pydantic models for request/response validation and serialization.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Message role enumeration."""
    user = "user"
    assistant = "assistant"
    system = "system"

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_context: bool = Field(True, description="Whether to use RAG context from vector store")
    max_context_docs: int = Field(3, ge=1, le=10, description="Maximum number of context documents")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "كيف يمكنني تحسين استراتيجية التسويق لشركتي؟",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "use_context": True,
                "max_context_docs": 3,
                "temperature": 0.7
            }
        }
    )

class RecommendationMetadata(BaseModel):
    """Model for recommendation metadata."""
    title: str = Field(..., description="Title of the recommended content")
    url: str = Field(..., description="URL to the recommended content")
    source_type: str = Field(..., description="Type of source (podcast, blog, article)")
    description: Optional[str] = Field(None, description="Brief description of the content")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    duration: Optional[str] = Field(None, description="Duration for audio/video content")
    published_date: Optional[str] = Field(None, description="Publication date")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session ID")
    conversation_id: Optional[str] = Field(None, description="Unique conversation ID")
    response_time: float = Field(..., description="Response time in seconds")
    model_used: str = Field(..., description="AI model used for generation")
    context_used: bool = Field(False, description="Whether context was used")
    sources_count: int = Field(0, description="Number of context sources used")
    timestamp: str = Field(..., description="Response timestamp in ISO format")
    context_source: str = Field("llm", description="Source of the response: 'llm' or 'rag'")
    recommendations: List[RecommendationMetadata] = Field(default_factory=list, description="List of recommended content")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Fix for Pydantic model_ field warnings
        json_schema_extra={
            "example": {
                "answer": "حسب رأينا، لتحسين استراتيجية التسويق، أنصح بالتركيز على...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "conversation_id": "conv_1234567890",
                "response_time": 2.5,
                "model_used": "gemini-langchain",
                "context_used": True,
                "sources_count": 2,
                "timestamp": "2024-01-01T12:00:00",
                "context_source": "rag",
                "recommendations": [
                    {
                        "title": "CEO Cleo Pharmaceuticals",
                        "url": "https://appapi.businessbelarabi.com/758528/episodes/17266210-ceo-cleo-pharmaceuticals.mp3",
                        "source_type": "podcast",
                        "description": "مقابلة مع الرئيس التنفيذي",
                        "categories": ["قيادة", "أدوية"],
                        "duration": "45:30"
                    }
                ]
            }
        }
    )

class ConversationHistory(BaseModel):
    """Model for conversation history entries."""
    id: str = Field(..., description="Conversation ID")
    session_id: str = Field(..., description="Session ID")
    user_message: str = Field(..., description="User message")
    assistant_response: str = Field(..., description="Assistant response")
    timestamp: str = Field(..., description="Timestamp")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    model_used: Optional[str] = Field(None, description="Model used")
    context_used: bool = Field(False, description="Whether context was used")
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="User rating (1-5)")
    
    model_config = ConfigDict(protected_namespaces=())  # Fix for Pydantic warnings

class SearchRequest(BaseModel):
    """Request model for conversation search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    limit: Optional[int] = Field(50, ge=1, le=200, description="Maximum results")
    include_context: bool = Field(False, description="Include context in search")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "تسويق رقمي",
                "session_id": None,
                "limit": 50,
                "include_context": False
            }
        }
    )

class SearchResponse(BaseModel):
    """Response model for conversation search."""
    results: List[ConversationHistory] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")
    search_time: Optional[float] = Field(None, description="Search time in seconds")

class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    total_messages: int = Field(..., description="Total messages in session")
    session_name: Optional[str] = Field(None, description="Custom session name")
    avg_response_time: Optional[float] = Field(None, description="Average response time")

class SessionRequest(BaseModel):
    """Request model for session operations."""
    session_name: Optional[str] = Field(None, max_length=100, description="Custom session name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DataIngestionRequest(BaseModel):
    """Request model for data ingestion."""
    limit: Optional[int] = Field(10, ge=1, le=100, description="Number of items to ingest")
    force_refresh: bool = Field(False, description="Force refresh existing data")
    chunk_size: Optional[int] = Field(1000, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500, description="Text chunk overlap")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "limit": 10,
                "force_refresh": False,
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }
    )

class DataIngestionResponse(BaseModel):
    """Response model for data ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    items_processed: int = Field(..., description="Number of items processed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    items: Optional[List[Dict[str, Any]]] = Field(None, description="Preview of processed items")
    errors: Optional[List[str]] = Field(None, description="Any errors encountered")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    memory_stats: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    version: str = Field("2.0.0", description="API version")

class AnalyticsResponse(BaseModel):
    """Response model for analytics data."""
    total_conversations: int = Field(..., description="Total conversations")
    total_sessions: int = Field(..., description="Total sessions")
    avg_response_time: float = Field(..., description="Average response time")
    today_conversations: int = Field(..., description="Today's conversations")
    week_conversations: int = Field(..., description="This week's conversations")
    most_active_session: Optional[tuple] = Field(None, description="Most active session info")
    daily_stats: List[tuple] = Field(..., description="Daily conversation statistics")
    memory_stats: Optional[Dict[str, Any]] = Field(None, description="Memory statistics")
    vector_stats: Optional[Dict[str, Any]] = Field(None, description="Vector store statistics")
    model_usage: Optional[Dict[str, int]] = Field(None, description="Model usage statistics")
    
    model_config = ConfigDict(protected_namespaces=())  # Fix for Pydantic warnings

class VectorStoreStats(BaseModel):
    """Vector store statistics model."""
    total_documents: int = Field(0, description="Total documents in store")
    total_chunks: int = Field(0, description="Total text chunks")
    index_size: Optional[int] = Field(None, description="Index size in bytes")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    sources: Dict[str, int] = Field(default_factory=dict, description="Document sources breakdown")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class BatchChatRequest(BaseModel):
    """Request model for batch chat processing."""
    messages: List[ChatRequest] = Field(..., min_length=1, max_length=10, description="Batch of chat requests")
    parallel: bool = Field(True, description="Process messages in parallel")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {
                        "message": "ما هي أفضل استراتيجيات التسويق؟",
                        "session_id": "session_1",
                        "use_context": True
                    },
                    {
                        "message": "كيف أدير فريق عمل؟",
                        "session_id": "session_2",
                        "use_context": True
                    }
                ],
                "parallel": True
            }
        }
    )

class BatchChatResponse(BaseModel):
    """Response model for batch chat processing."""
    responses: List[ChatResponse] = Field(..., description="Batch chat responses")
    total_time: float = Field(..., description="Total processing time")
    successful_count: int = Field(..., description="Number of successful responses")
    error_count: int = Field(..., description="Number of errors")
    errors: Optional[List[ErrorResponse]] = Field(None, description="Any errors encountered")

class RatingRequest(BaseModel):
    """Request model for rating conversations."""
    conversation_id: str = Field(..., description="Conversation ID to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback: Optional[str] = Field(None, max_length=500, description="Optional feedback")

class ExportRequest(BaseModel):
    """Request model for data export."""
    export_type: str = Field(..., pattern="^(conversations|sessions|analytics)$", description="Type of data to export")
    format: str = Field("json", pattern="^(json|csv|xlsx)$", description="Export format")
    date_from: Optional[str] = Field(None, description="Start date (ISO format)")
    date_to: Optional[str] = Field(None, description="End date (ISO format)")
    session_ids: Optional[List[str]] = Field(None, description="Specific session IDs to export")
    include_metadata: bool = Field(True, description="Include metadata in export")

class ExportResponse(BaseModel):
    """Response model for data export."""
    success: bool = Field(..., description="Export success status")
    download_url: Optional[str] = Field(None, description="Download URL for exported file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    records_count: int = Field(..., description="Number of records exported")
    expires_at: Optional[str] = Field(None, description="Download link expiration")

class WebhookRequest(BaseModel):
    """Request model for webhook configuration."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to trigger webhook")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    active: bool = Field(True, description="Whether webhook is active")

class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, description="Disk usage percentage")
    active_connections: int = Field(0, description="Number of active connections")
    requests_per_minute: Optional[float] = Field(None, description="Requests per minute")
    average_response_time: Optional[float] = Field(None, description="Average response time")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")

class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    key: str = Field(..., description="Configuration key")
    value: Union[str, int, float, bool, Dict, List] = Field(..., description="Configuration value")
    scope: str = Field("global", description="Configuration scope")
    description: Optional[str] = Field(None, description="Configuration description")

# Validation functions
def validate_arabic_text(text: str) -> bool:
    """Validate if text contains Arabic characters."""
    import re
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    )
    return bool(uuid_pattern.match(session_id))

# Response helpers
class ResponseBuilder:
    """Helper class for building consistent API responses."""
    
    @staticmethod
    def success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Build success response."""
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error(message: str, error_type: str = "error", details: Optional[Dict] = None) -> Dict[str, Any]:
        """Build error response."""
        return {
            "success": False,
            "error": error_type,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def paginated(data: List[Any], total: int, page: int = 1, per_page: int = 50) -> Dict[str, Any]:
        """Build paginated response."""
        return {
            "data": data,
            "pagination": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page
            },
            "timestamp": datetime.now().isoformat()
        }
