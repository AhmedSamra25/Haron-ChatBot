#!/usr/bin/env python3
"""
Business bel Arabi RAG - Backend Package

FastAPI backend with LangChain integration for Arabic business consulting.
"""

__version__ = "2.0.0"
__author__ = "Business bel Arabi Team"
__email__ = "support@businessbelarabi.com"
__description__ = "Advanced Arabic Business Consulting RAG System with LangChain"

# Import key components for easy access
from .utils import get_settings, setup_logging
from .database import DatabaseManager
from .vector_store import VectorStoreManager
from .data_ingester import EnhancedDataIngester

__all__ = [
    "get_settings",
    "setup_logging", 
    "DatabaseManager",
    "VectorStoreManager",
    "EnhancedDataIngester"
]
