#!/usr/bin/env python3
"""
Business bel Arabi RAG - Utilities

Utility functions for configuration, logging, and common operations.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import sys
from pathlib import Path
import re
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App Info
    app_name: str = Field(default="business-bel-arabi-api", env="APP_NAME")
    app_env: str = Field(default="dev", env="APP_ENV")
    
    # API Keys
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    api_keys_allowed: str = Field(default="local-dev-key", env="API_KEYS_ALLOWED")
    
    # External APIs
    podcast_api_base: str = Field(
        default="https://businessbelarabi.360marketingtec.com/api/podcast",
        env="PODCAST_API_BASE"
    )
    wp_base: str = Field(
        default="https://businessbelarabi.com",
        env="WP_BASE"
    )
    
    # AI Model Settings
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    embed_model: str = Field(default="text-embedding-004", env="EMBED_MODEL")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    
    # Vector Store Settings
    vector_db: str = Field(default="qdrant", env="VECTOR_DB")
    vector_store_path: str = Field(default="vector_store", env="VECTOR_STORE_PATH")
    embedding_model: str = Field(default="models/embedding-001", env="EMBEDDING_MODEL")
    qdrant_collection: str = Field(default="bbelarabi", env="QDRANT_COLLECTION")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Database Settings
    database_path: str = Field(default="business_bel_arabi.db", env="DATABASE_PATH")
    
    # Crawling Settings
    crawl_max_pages: int = Field(default=200, env="CRAWL_MAX_PAGES")
    crawl_concurrency: int = Field(default=5, env="CRAWL_CONCURRENCY")
    
    # Scheduler Settings
    ingest_interval_days: int = Field(default=3, env="INGEST_INTERVAL_DAYS")
    
    # Retrieval Settings
    max_retrieval_results: int = Field(default=10, env="MAX_RETRIEVAL_RESULTS")
    
    # Performance Settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    max_threads: int = Field(default=4, env="MAX_THREADS")
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    memory_window_size: int = Field(default=10, env="MEMORY_WINDOW_SIZE")
    async_pool_size: int = Field(default=20, env="ASYNC_POOL_SIZE")
    background_worker_count: int = Field(default=2, env="BACKGROUND_WORKER_COUNT")
    concurrent_requests: int = Field(default=100, env="CONCURRENT_REQUESTS")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Frontend Settings
    frontend_enabled: bool = Field(default=True, env="FRONTEND_ENABLED")
    frontend_port: int = Field(default=8501, env="FRONTEND_PORT")
    frontend_title: str = Field(default="CEO Expert - Business bel Arabi", env="FRONTEND_TITLE")
    
    # External Service URLs
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    redis_url: str = Field(default="", env="REDIS_URL")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        env="ALLOWED_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment
    
    @property
    def allowed_origins_list(self) -> list:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    def get_database_url(self) -> str:
        """Get full database URL path."""
        return os.path.abspath(self.database_path)
    
    def get_vector_store_path(self) -> str:
        """Get full vector store path."""
        return os.path.abspath(self.vector_store_path)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def setup_logging(
    level: str = None,
    format_string: str = None,
    log_file: str = None
) -> logging.Logger:
    """Set up comprehensive logging configuration."""
    
    settings = get_settings()
    log_level = level or settings.log_level
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default format
    if not format_string:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file or log_dir / f"business_bel_arabi_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            )
        ]
    )
    
    # Configure specific loggers
    logger = logging.getLogger("business_bel_arabi")
    
    # Reduce verbosity of external libraries
    external_loggers = [
        "aiohttp",
        "urllib3",
        "httpcore",
        "httpx",
        "faiss",
        "sqlite3"
    ]
    
    for ext_logger in external_loggers:
        logging.getLogger(ext_logger).setLevel(logging.WARNING)
    
    # Special handling for development
    if settings.debug:
        logger.setLevel(logging.DEBUG)
        # Add more detailed formatting for debug
        debug_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
        for handler in logger.handlers:
            handler.setFormatter(debug_formatter)
    
    logger.info("🚀 Logging system initialized")
    return logger

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def ensure_directory(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_version() -> str:
    """Get application version from various sources."""
    try:
        # Try to read from version file
        version_file = get_project_root() / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
    except Exception:
        pass
    
    try:
        # Try to get from git
        import subprocess
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            cwd=get_project_root()
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return "2.0.0"  # Fallback version

def format_bytes(bytes_count: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely load JSON string with default fallback."""
    try:
        import json
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string with default fallback."""
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default

def calculate_hash(content: str) -> str:
    """Calculate MD5 hash of content."""
    import hashlib
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    import platform
    import psutil
    import sys
    
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            },
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'path': sys.path[:3],  # First 3 paths only
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'percent': psutil.cpu_percent(interval=1)
            }
        }
    except ImportError:
        # Fallback if psutil is not available
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
            },
            'python': {
                'version': sys.version,
                'executable': sys.executable,
            }
        }

async def async_retry(
    func,
    args=None,
    kwargs=None,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async retry wrapper with exponential backoff."""
    args = args or []
    kwargs = kwargs or {}
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                raise
            
            wait_time = delay * (backoff ** attempt)
            await asyncio.sleep(wait_time)
    
    raise last_exception

def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key or len(api_key) < 10:
        return False
    
    # Basic validation - adjust based on actual API key formats
    return True

def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: set = None) -> Dict[str, Any]:
    """Mask sensitive data in dictionary."""
    if sensitive_keys is None:
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'api_key',
            'google_api_key', 'auth', 'authorization'
        }
    
    def mask_value(key: str, value: Any) -> Any:
        if isinstance(key, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
            if isinstance(value, str) and len(value) > 8:
                return f"{'*' * 4}...{value[-4:]}"
            else:
                return "***"
        elif isinstance(value, dict):
            return {k: mask_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [mask_value(str(i), item) for i, item in enumerate(value)]
        return value
    
    return {k: mask_value(k, v) for k, v in data.items()}

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration * 1000

# Health check utilities
def check_database_health(db_path: str) -> Dict[str, Any]:
    """Check database health status."""
    try:
        import sqlite3
        
        if not os.path.exists(db_path):
            return {"status": "missing", "error": "Database file not found"}
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
        
        file_size = os.path.getsize(db_path)
        
        return {
            "status": "healthy",
            "file_size": file_size,
            "file_size_formatted": format_bytes(file_size)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def check_vector_store_health(vector_store_path: str) -> Dict[str, Any]:
    """Check vector store health status."""
    try:
        if not os.path.exists(vector_store_path):
            return {"status": "missing", "error": "Vector store directory not found"}
        
        files = list(Path(vector_store_path).glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        return {
            "status": "healthy",
            "files_count": len(files),
            "total_size": total_size,
            "total_size_formatted": format_bytes(total_size)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# URL Transformation utilities
def transform_podcast_url(original_url: str) -> str:
    """
    Transform podcast URL from buzzsprout.com to appapi.businessbelarabi.com format.
    
    Example:
    Input: "https://www.buzzsprout.com/758528/episodes/17266210-ceo-cleo-pharmaceuticals.mp3"
    Output: "https://appapi.businessbelarabi.com/758528/episodes/17266210-ceo-cleo-pharmaceuticals.mp3"
    """
    if not original_url:
        return original_url
    
    try:
        # Check if it's a buzzsprout URL
        if "buzzsprout.com" in original_url:
            # Extract the path after buzzsprout.com
            pattern = r'https?://(?:www\.)?buzzsprout\.com(/.*?)(?:\?.*)?$'
            match = re.search(pattern, original_url)
            
            if match:
                path = match.group(1)
                return f"https://appapi.businessbelarabi.com{path}"
        
        # Return original URL if not a buzzsprout URL
        return original_url
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error transforming podcast URL: {e}")
        return original_url

def get_content_recommendations(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate content recommendations from source documents.
    
    Args:
        sources: List of source documents with metadata
    
    Returns:
        List of recommendation objects with transformed URLs
    """
    recommendations = []
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing {len(sources)} sources for recommendations")
    
    for i, source in enumerate(sources):
        try:
            metadata = source.get('metadata', {})
            logger.info(f"Source {i+1} metadata: {metadata}")
            
            source_type = metadata.get('source', 'unknown')
            title = metadata.get('title', '')
            original_url = metadata.get('url', '')
            
            # Skip if essential fields are missing or malformed
            if (not title or title == 'Untitled' or source_type == 'init' or 
                source_type == 'dummy' or metadata.get('type') == 'dummy'):
                logger.warning(f"Skipping malformed/dummy source {i+1}: title='{title}', source_type='{source_type}', metadata={metadata}")
                continue
                
            if not original_url:
                logger.warning(f"Skipping source {i+1} with empty URL: {title}")
                continue
            
            # Transform URL based on source type
            if source_type == 'podcast':
                transformed_url = transform_podcast_url(original_url)
            else:
                # For WordPress/blog posts, keep original URL
                transformed_url = original_url
            
            # Get description from page content
            page_content = source.get('page_content', '')
            if page_content and len(page_content) > 200:
                description = page_content[:200] + '...'
            elif page_content:
                description = page_content
            else:
                # Fallback description
                description = f"محتوى ذو صلة من {source_type}"
            
            # Validate URL before creating recommendation
            final_url = transformed_url
            try:
                # Quick validation for non-main-site URLs
                if transformed_url != 'https://businessbelarabi.com' and not validate_url_sync(transformed_url, timeout=2.0):
                    logger.warning(f"Invalid URL detected in recommendation: {transformed_url}. Using fallback.")
                    final_url = 'https://businessbelarabi.com'
            except Exception as e:
                logger.warning(f"URL validation error for {transformed_url}: {e}. Using fallback.")
                final_url = 'https://businessbelarabi.com'
            
            recommendation = {
                'title': title,
                'url': final_url,  # Use validated URL
                'source_type': source_type,
                'description': description,
                'categories': metadata.get('categories', []),
                'duration': metadata.get('duration', None) if source_type == 'podcast' else None,
                'published_date': metadata.get('published_date') or metadata.get('date')
            }
            
            recommendations.append(recommendation)
            logger.info(f"Added recommendation: {title} ({source_type})")
            
        except Exception as e:
            logger.warning(f"Error processing source {i+1} for recommendations: {e}")
            continue
    
    logger.info(f"Generated {len(recommendations)} valid recommendations")
    return recommendations

async def validate_url(url: str, timeout: float = 5.0) -> bool:
    """
    Validate if a URL is accessible and returns a successful response.
    
    Args:
        url: The URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if URL is accessible, False otherwise
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                return 200 <= response.status < 400
    except Exception as e:
        logging.getLogger(__name__).warning(f"URL validation failed for {url}: {e}")
        return False

def validate_url_sync(url: str, timeout: float = 5.0) -> bool:
    """
    Synchronous version of URL validation.
    
    Args:
        url: The URL to validate
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if URL is accessible, False otherwise
    """
    try:
        import requests
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return 200 <= response.status_code < 400
    except Exception as e:
        logging.getLogger(__name__).warning(f"URL validation failed for {url}: {e}")
        return False

def add_context_indicators(answer: str, context_source: str) -> str:
    """
    Add context indicators to the answer based on the source.
    
    Args:
        answer: The original answer text
        context_source: 'llm' for direct LLM response, 'rag' for RAG-based response
    
    Returns:
        Answer with appropriate context indicator
    """
    if not answer:
        return answer
    
    try:
        # Define context prefixes based on source
        if context_source == 'rag':
            # For RAG responses (retrieved from content)
            context_phrases = [
                "بناءً على بحثي في المحتوى المتاح،",
                "حسب ما وجدته في المصادر،",
                "استناداً إلى المعلومات المتوفرة،"
            ]
        else:
            # For direct LLM responses
            context_phrases = [
                "حسب رأينا،",
                "من وجهة نظرنا،",
                "بناءً على خبرتنا،"
            ]
        
        # Use the first phrase for simplicity (could be randomized)
        context_prefix = context_phrases[0]
        
        # Add context indicator if not already present
        if not any(phrase in answer for phrase in context_phrases):
            # Check if answer starts with Arabic text
            if re.match(r'^[\u0600-\u06FF]', answer.strip()):
                answer = f"{context_prefix} {answer}"
            else:
                # If it doesn't start with Arabic, add after first Arabic text found
                arabic_match = re.search(r'[\u0600-\u06FF]', answer)
                if arabic_match:
                    # Insert context indicator before first Arabic text
                    insert_pos = arabic_match.start()
                    answer = answer[:insert_pos] + f"{context_prefix} " + answer[insert_pos:]
                else:
                    # Fallback: add at the beginning
                    answer = f"{context_prefix} {answer}"
        
        return answer
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error adding context indicators: {e}")
        return answer

# Export commonly used items
__all__ = [
    'Settings',
    'get_settings',
    'setup_logging',
    'get_project_root',
    'ensure_directory',
    'get_version',
    'format_bytes',
    'format_duration',
    'clean_text',
    'truncate_text',
    'safe_json_loads',
    'safe_json_dumps',
    'calculate_hash',
    'get_system_info',
    'async_retry',
    'validate_api_key',
    'mask_sensitive_data',
    'Timer',
    'check_database_health',
    'check_vector_store_health',
    'transform_podcast_url',
    'get_content_recommendations',
    'add_context_indicators',
    'validate_url',
    'validate_url_sync'
]
