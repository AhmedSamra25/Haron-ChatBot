#!/usr/bin/env python3
"""
Business bel Arabi RAG - Backend Startup Script

Startup script for the FastAPI backend server with proper configuration.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment before importing other modules
os.environ.setdefault("PYTHONPATH", str(project_root))

from backend.utils import get_settings, setup_logging
from backend.main import app

def main():
    """Main startup function."""
    try:
        # Initialize settings
        settings = get_settings()
        
        # Setup logging
        logger = setup_logging(level=settings.log_level)
        logger.info("🚀 Starting Business bel Arabi RAG Backend Server")
        
        # Log configuration
        logger.info(f"📊 Configuration:")
        logger.info(f"  - Host: {settings.host}")
        logger.info(f"  - Port: {settings.port}")
        logger.info(f"  - Debug: {settings.debug}")
        logger.info(f"  - Log Level: {settings.log_level}")
        logger.info(f"  - Database: {settings.database_path}")
        logger.info(f"  - Vector Store: {settings.vector_store_path}")
        logger.info(f"  - Google AI: {'✅ Configured' if settings.google_api_key else '❌ Not configured'}")
        logger.info(f"  - Podcast API: {'✅ Configured' if settings.podcast_api_base else '❌ Not configured'}")
        logger.info(f"  - WordPress API: {'✅ Configured' if settings.wp_base else '❌ Not configured'}")
        
        # Import and run uvicorn
        import uvicorn
        
        # Server configuration
        server_config = {
            "app": "backend.main:app",
            "host": settings.host,
            "port": settings.port,
            "reload": settings.debug,
            "log_level": settings.log_level.lower(),
            "access_log": True,
            "use_colors": True,
            "loop": "asyncio",
        }
        
        # Additional production settings
        if not settings.debug:
            server_config.update({
                "workers": settings.max_threads,
                "backlog": 2048,
                "timeout_keep_alive": 5,
                "timeout_graceful_shutdown": 30,
            })
        
        logger.info("🎉 Starting server with uvicorn...")
        uvicorn.run(**server_config)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
