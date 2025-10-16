#!/usr/bin/env python3
"""
Business bel Arabi RAG - Enhanced Database Manager

Advanced database management with async support and comprehensive features.
"""

import sqlite3
import asyncio
import aiosqlite
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import hashlib
import json
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with async support and advanced features."""
    
    def __init__(self, db_path: str = "business_bel_arabi.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with comprehensive schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_rating INTEGER DEFAULT NULL CHECK (user_rating BETWEEN 1 AND 5),
                    response_time REAL DEFAULT NULL,
                    context_used BOOLEAN DEFAULT FALSE,
                    model_used TEXT DEFAULT NULL,
                    message_hash TEXT NOT NULL,
                    sources_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT NULL,
                    metadata TEXT DEFAULT '{}',
                    UNIQUE(message_hash)
                )
            """)
            
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    session_name TEXT DEFAULT NULL,
                    metadata TEXT DEFAULT '{}',
                    is_archived BOOLEAN DEFAULT FALSE
                )
            """)
            
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    category TEXT DEFAULT 'general'
                )
            """)
            
            # User feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    feedback_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # Data ingestion log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    items_count INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Vector store metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_store_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    title TEXT,
                    content_hash TEXT,
                    chunk_count INTEGER DEFAULT 1,
                    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_model_used ON conversations(model_used)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_category ON system_metrics(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_conversation_id ON user_feedback(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_timestamp ON ingestion_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_store_docs_source_type ON vector_store_docs(source_type)")
            
            # Full-text search for conversations
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                    conversation_id UNINDEXED,
                    user_message,
                    assistant_response,
                    content='conversations',
                    content_rowid='rowid'
                )
            """)
            
            # Trigger to keep FTS table in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_insert AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(conversation_id, user_message, assistant_response)
                    VALUES (NEW.id, NEW.user_message, NEW.assistant_response);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_update AFTER UPDATE ON conversations BEGIN
                    UPDATE conversations_fts SET
                        user_message = NEW.user_message,
                        assistant_response = NEW.assistant_response
                    WHERE conversation_id = NEW.id;
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_delete AFTER DELETE ON conversations BEGIN
                    DELETE FROM conversations_fts WHERE conversation_id = OLD.id;
                END
            """)
            
            logger.info("✅ Database schema initialized successfully")
    
    def save_conversation(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str,
        response_time: float = None,
        context_used: bool = False,
        model_used: str = "unknown",
        sources_count: int = 0,
        token_count: int = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Save conversation to database with comprehensive tracking."""
        conversation_id = str(uuid.uuid4())
        message_hash = hashlib.md5(
            f"{session_id}_{user_message}_{assistant_response}".encode()
        ).hexdigest()
        
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            # Save conversation
            conn.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, session_id, user_message, assistant_response, response_time, 
                 context_used, model_used, message_hash, sources_count, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, session_id, user_message, assistant_response,
                response_time, context_used, model_used, message_hash,
                sources_count, token_count, metadata_json
            ))
            
            # Update or create session
            conn.execute("""
                INSERT OR REPLACE INTO sessions (session_id, last_activity, total_messages)
                VALUES (?, CURRENT_TIMESTAMP, 
                        COALESCE((SELECT total_messages FROM sessions WHERE session_id = ?), 0) + 1)
            """, (session_id, session_id))
        
        return conversation_id
    
    def get_conversation_history(
        self, 
        session_id: str = None, 
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = False
    ) -> List[Dict]:
        """Get conversation history with advanced filtering."""
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                query = """
                    SELECT id, session_id, user_message, assistant_response, timestamp, 
                           user_rating, response_time, context_used, model_used, 
                           sources_count, token_count, metadata
                    FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
                cursor = conn.execute(query, (session_id, limit, offset))
            else:
                query = """
                    SELECT id, session_id, user_message, assistant_response, timestamp, 
                           user_rating, response_time, context_used, model_used,
                           sources_count, token_count, metadata
                    FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """
                cursor = conn.execute(query, (limit, offset))
            
            columns = [desc[0] for desc in cursor.description]
            conversations = []
            
            for row in cursor.fetchall():
                conv = dict(zip(columns, row))
                if include_metadata and conv.get('metadata'):
                    try:
                        conv['metadata'] = json.loads(conv['metadata'])
                    except:
                        conv['metadata'] = {}
                conversations.append(conv)
            
            return conversations
    
    def search_conversations(
        self, 
        query: str, 
        limit: int = 50,
        session_id: str = None
    ) -> List[Dict]:
        """Advanced full-text search in conversations."""
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                search_query = """
                    SELECT c.id, c.session_id, c.user_message, c.assistant_response, 
                           c.timestamp, c.user_rating, c.response_time, c.context_used, 
                           c.model_used, c.sources_count, c.token_count
                    FROM conversations_fts fts
                    JOIN conversations c ON fts.conversation_id = c.id
                    WHERE fts MATCH ? AND c.session_id = ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                """
                cursor = conn.execute(search_query, (query, session_id, limit))
            else:
                search_query = """
                    SELECT c.id, c.session_id, c.user_message, c.assistant_response, 
                           c.timestamp, c.user_rating, c.response_time, c.context_used, 
                           c.model_used, c.sources_count, c.token_count
                    FROM conversations_fts fts
                    JOIN conversations c ON fts.conversation_id = c.id
                    WHERE fts MATCH ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                """
                cursor = conn.execute(search_query, (query, limit))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_sessions(self, limit: int = 50, include_archived: bool = False) -> List[Dict]:
        """Get all sessions with comprehensive statistics."""
        with sqlite3.connect(self.db_path) as conn:
            where_clause = "" if include_archived else "WHERE s.is_archived = FALSE"
            
            query = f"""
                SELECT s.session_id, s.created_at, s.last_activity, s.total_messages,
                       COALESCE(s.session_name, 'Session ' || SUBSTR(s.session_id, 1, 8)) as session_name,
                       AVG(c.response_time) as avg_response_time,
                       AVG(c.user_rating) as avg_rating,
                       COUNT(DISTINCT c.model_used) as models_used,
                       SUM(c.sources_count) as total_sources_used,
                       s.metadata, s.is_archived
                FROM sessions s
                LEFT JOIN conversations c ON s.session_id = c.session_id
                {where_clause}
                GROUP BY s.session_id
                ORDER BY s.last_activity DESC
                LIMIT ?
            """
            
            cursor = conn.execute(query, (limit,))
            columns = [desc[0] for desc in cursor.description]
            sessions = []
            
            for row in cursor.fetchall():
                session = dict(zip(columns, row))
                if session.get('metadata'):
                    try:
                        session['metadata'] = json.loads(session['metadata'])
                    except:
                        session['metadata'] = {}
                sessions.append(session)
            
            return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its conversations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete conversations first (due to foreign key constraints)
                conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                
                # Delete the session
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def archive_session(self, session_id: str) -> bool:
        """Archive a session instead of deleting it."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET is_archived = TRUE WHERE session_id = ?",
                    (session_id,)
                )
                return True
        except Exception as e:
            logger.error(f"Error archiving session {session_id}: {e}")
            return False
    
    def save_user_feedback(self, conversation_id: str, rating: int, feedback_text: str = None) -> bool:
        """Save user feedback for a conversation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update conversation rating
                conn.execute(
                    "UPDATE conversations SET user_rating = ? WHERE id = ?",
                    (rating, conversation_id)
                )
                
                # Save detailed feedback
                conn.execute("""
                    INSERT INTO user_feedback (conversation_id, rating, feedback_text)
                    VALUES (?, ?, ?)
                """, (conversation_id, rating, feedback_text))
                
                return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False
    
    def log_data_ingestion(
        self,
        source_type: str,
        source_url: str = None,
        items_count: int = 0,
        success: bool = False,
        error_message: str = None,
        processing_time: float = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """Log data ingestion operations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO ingestion_log 
                (source_type, source_url, items_count, success, error_message, processing_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source_type, source_url, items_count, success, 
                error_message, processing_time, json.dumps(metadata or {})
            ))
            
            return cursor.lastrowid
    
    def save_vector_document(
        self,
        doc_id: str,
        source_type: str,
        source_id: str = None,
        title: str = None,
        content_hash: str = None,
        chunk_count: int = 1,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Track documents added to vector store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vector_store_docs 
                    (doc_id, source_type, source_id, title, content_hash, chunk_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, source_type, source_id, title, 
                    content_hash, chunk_count, json.dumps(metadata or {})
                ))
                return True
        except Exception as e:
            logger.error(f"Error saving vector document: {e}")
            return False
    
    def get_analytics_data(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            total_conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE is_archived = FALSE").fetchone()[0]
            
            # Average response time
            avg_response_time = conn.execute(
                "SELECT AVG(response_time) FROM conversations WHERE response_time IS NOT NULL"
            ).fetchone()[0] or 0
            
            # Today's conversations
            today_conversations = conn.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE date(timestamp) = date('now')
            """).fetchone()[0]
            
            # This week's conversations
            week_conversations = conn.execute("""
                SELECT COUNT(*) FROM conversations 
                WHERE timestamp >= date('now', '-7 days')
            """).fetchone()[0]
            
            # Most active session
            most_active_session = conn.execute("""
                SELECT session_id, COUNT(*) as message_count 
                FROM conversations 
                GROUP BY session_id 
                ORDER BY message_count DESC 
                LIMIT 1
            """).fetchone()
            
            # Daily conversation counts (last N days)
            daily_stats = conn.execute(f"""
                SELECT date(timestamp) as date, COUNT(*) as count
                FROM conversations
                WHERE timestamp >= date('now', '-{days_back} days')
                GROUP BY date(timestamp)
                ORDER BY date
            """).fetchall()
            
            # Model usage statistics
            model_usage = conn.execute("""
                SELECT model_used, COUNT(*) as count
                FROM conversations
                WHERE model_used IS NOT NULL
                GROUP BY model_used
                ORDER BY count DESC
            """).fetchall()
            
            # Context usage statistics
            context_stats = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN context_used = 1 THEN 1 END) as with_context,
                    COUNT(CASE WHEN context_used = 0 THEN 1 END) as without_context,
                    AVG(sources_count) as avg_sources
                FROM conversations
            """).fetchone()
            
            # User ratings statistics
            rating_stats = conn.execute("""
                SELECT 
                    AVG(user_rating) as avg_rating,
                    COUNT(user_rating) as total_ratings,
                    COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as positive_ratings
                FROM conversations
                WHERE user_rating IS NOT NULL
            """).fetchone()
            
            # Recent ingestion statistics
            ingestion_stats = conn.execute(f"""
                SELECT 
                    source_type,
                    COUNT(*) as attempts,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(items_count) as total_items
                FROM ingestion_log
                WHERE timestamp >= date('now', '-{days_back} days')
                GROUP BY source_type
            """).fetchall()
            
            return {
                'total_conversations': total_conversations,
                'total_sessions': total_sessions,
                'avg_response_time': round(avg_response_time, 2),
                'today_conversations': today_conversations,
                'week_conversations': week_conversations,
                'most_active_session': most_active_session,
                'daily_stats': daily_stats,
                'model_usage': dict(model_usage),
                'context_stats': {
                    'with_context': context_stats[0] or 0,
                    'without_context': context_stats[1] or 0,
                    'avg_sources': round(context_stats[2] or 0, 2)
                },
                'rating_stats': {
                    'avg_rating': round(rating_stats[0] or 0, 2),
                    'total_ratings': rating_stats[1] or 0,
                    'positive_ratings': rating_stats[2] or 0
                },
                'ingestion_stats': {
                    source_type: {
                        'attempts': attempts,
                        'successful': successful,
                        'total_items': total_items
                    }
                    for source_type, attempts, successful, total_items in ingestion_stats
                }
            }
    
    def get_system_metrics(self, metric_name: str = None, limit: int = 100) -> List[Dict]:
        """Get system metrics."""
        with sqlite3.connect(self.db_path) as conn:
            if metric_name:
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, timestamp, category
                    FROM system_metrics
                    WHERE metric_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (metric_name, limit))
            else:
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, timestamp, category
                    FROM system_metrics
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_system_metric(
        self, 
        metric_name: str, 
        metric_value: str, 
        category: str = "general"
    ) -> bool:
        """Save a system metric."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics (metric_name, metric_value, category)
                    VALUES (?, ?, ?)
                """, (metric_name, metric_value, category))
                return True
        except Exception as e:
            logger.error(f"Error saving metric: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data to maintain database performance."""
        cleanup_results = {
            'conversations_deleted': 0,
            'metrics_deleted': 0,
            'ingestion_logs_deleted': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean old system metrics
                cursor = conn.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                cleanup_results['metrics_deleted'] = cursor.rowcount
                
                # Clean old ingestion logs
                cursor = conn.execute("""
                    DELETE FROM ingestion_log 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                cleanup_results['ingestion_logs_deleted'] = cursor.rowcount
                
                # Optionally clean very old conversations (be careful with this)
                # This is commented out as conversations are usually valuable
                # cursor = conn.execute("""
                #     DELETE FROM conversations 
                #     WHERE timestamp < ? AND user_rating IS NULL
                # """, (cutoff_date,))
                # cleanup_results['conversations_deleted'] = cursor.rowcount
                
                logger.info(f"Database cleanup completed: {cleanup_results}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleanup_results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Table sizes
            tables = ['conversations', 'sessions', 'system_metrics', 'user_feedback', 
                     'ingestion_log', 'vector_store_docs']
            
            stats = {}
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Database file size
            if os.path.exists(self.db_path):
                stats['database_size_bytes'] = os.path.getsize(self.db_path)
                stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)
            
            return stats

# Async version of the database manager
class AsyncDatabaseManager:
    """Async version of the database manager for better performance."""
    
    def __init__(self, db_path: str = "business_bel_arabi.db"):
        self.db_path = db_path
        # Initialize sync version to set up schema
        sync_manager = DatabaseManager(db_path)
    
    async def save_conversation_async(self, *args, **kwargs) -> str:
        """Async version of save_conversation."""
        # Use thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: DatabaseManager(self.db_path).save_conversation(*args, **kwargs)
        )
    
    async def get_analytics_data_async(self) -> Dict[str, Any]:
        """Async version of get_analytics_data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: DatabaseManager(self.db_path).get_analytics_data()
        )
