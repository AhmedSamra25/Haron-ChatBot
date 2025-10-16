#!/usr/bin/env python3
"""
Business bel Arabi RAG - Complete Data Ingestion Service

Comprehensive service that fetches ALL data from APIs, processes it, tokenizes it,
and stores it in the vector database for RAG functionality.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .data_ingester import EnhancedDataIngester
from .vector_store import VectorStoreManager
from .database import DatabaseManager
from .utils import get_settings

logger = logging.getLogger(__name__)

class CompleteIngestionService:
    """
    Complete data ingestion service that processes all content from APIs
    and stores it in the vector database for RAG functionality.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.data_ingester = EnhancedDataIngester()
        self.db_manager = DatabaseManager()
        self.vector_manager: Optional[VectorStoreManager] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " ", ""]
        )
        
        # Statistics tracking
        self.stats = {
            "total_podcast_episodes": 0,
            "total_wordpress_posts": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "processing_time": 0,
            "errors": [],
            "duplicates_removed": 0,
            "posts_skipped_limit": 0
        }
        
        # Duplication tracking
        self.processed_hashes = set()  # Track content hashes to avoid duplicates
        self.max_wordpress_posts = 10000  # Limit WordPress posts to 10,000
    
    async def initialize_vector_store(self) -> bool:
        """Initialize the vector store with Google embeddings."""
        try:
            if not self.settings.google_api_key:
                logger.error("❌ Google API key not configured!")
                return False
            
            logger.info("🔧 Initializing vector store with Google embeddings...")
            
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.settings.google_api_key
            )
            
            # Create vector store manager
            self.vector_manager = VectorStoreManager(
                embeddings_model=embeddings,
                persist_directory="vector_store"
            )
            
            # Initialize the vector store
            await self.vector_manager.initialize()
            
            if self.vector_manager.is_ready:
                logger.info("✅ Vector store initialized successfully!")
                return True
            else:
                logger.error("❌ Vector store failed to initialize")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            return False
    
    async def fetch_all_podcast_episodes(self) -> List[Dict]:
        """Fetch ALL podcast episodes from the API with proper pagination."""
        logger.info("📻 Fetching ALL podcast episodes...")
        
        all_episodes = []
        page = 1
        page_size = 20  # Smaller batches for more reliable API calls
        
        while True:
            try:
                # Use the enhanced fetch method with pagination support
                episodes = await self._fetch_podcast_page(page, page_size)
                
                if not episodes:
                    logger.info(f"   📝 No more episodes found at page {page}")
                    break
                
                all_episodes.extend(episodes)
                logger.info(f"   📥 Fetched page {page}: {len(episodes)} episodes (Total: {len(all_episodes)})")
                
                # If we got fewer than page_size, we've reached the end
                if len(episodes) < page_size:
                    logger.info(f"   🏁 Reached end of pagination (got {len(episodes)} < {page_size})")
                    break
                
                page += 1
                
                # Add a small delay to be respectful to the API
                await asyncio.sleep(0.5)
                
                # Safety limit to prevent infinite loops
                if page > 100:  # Max 2000 episodes (20 * 100)
                    logger.warning(f"⚠️ Reached safety limit of {page-1} pages")
                    break
                
            except Exception as e:
                logger.error(f"❌ Error fetching podcast episodes page {page}: {e}")
                break
        
        logger.info(f"✅ Fetched total of {len(all_episodes)} podcast episodes across {page-1} pages")
        self.stats["total_podcast_episodes"] = len(all_episodes)
        return all_episodes
    
    async def _fetch_podcast_page(self, page: int, page_size: int) -> List[Dict]:
        """Fetch a specific page of podcast episodes."""
        import aiohttp
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'BusinessBelArabi-RAG/2.0 (+https://businessbelarabi.com)',
                'Accept': 'application/json',
                'Accept-Language': 'ar,en;q=0.9'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                params = {"per_page": page_size, "page": page}
                
                async with session.get(self.data_ingester.podcast_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        episodes = self.data_ingester._parse_podcast_response(data, page_size)
                        return episodes
                    elif response.status == 404:
                        # No more pages
                        return []
                    else:
                        logger.error(f"API returned status {response.status} for page {page}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            return []
    
    async def fetch_all_wordpress_posts(self) -> List[Dict]:
        """Fetch WordPress posts with 10,000 limit and duplication checking."""
        logger.info(f"📰 Fetching WordPress posts (limit: {self.max_wordpress_posts:,})...")
        
        all_posts = []
        unique_posts = []
        page = 1
        page_size = 50  # Fetch in batches
        posts_processed = 0
        duplicates_found = 0
        
        while posts_processed < self.max_wordpress_posts:
            try:
                # Calculate remaining posts needed
                remaining_needed = self.max_wordpress_posts - posts_processed
                current_batch_size = min(page_size, remaining_needed)
                
                # Fetch posts using paginated WordPress API
                posts = await self._fetch_wordpress_page(page, current_batch_size)
                
                if not posts:
                    logger.info(f"   📝 No more posts found at page {page}")
                    break
                
                # Check for duplicates by content hash
                for post in posts:
                    content_hash = self._calculate_post_hash(post)
                    
                    if content_hash in self.processed_hashes:
                        duplicates_found += 1
                        logger.debug(f"   🔄 Duplicate found: post {post.get('id', 'unknown')}")
                        continue
                    
                    self.processed_hashes.add(content_hash)
                    unique_posts.append(post)
                    posts_processed += 1
                    
                    # Stop if we've reached the limit
                    if posts_processed >= self.max_wordpress_posts:
                        break
                
                all_posts.extend(posts)
                logger.info(f"   📥 Page {page}: {len(posts)} fetched, {len(unique_posts) - (len(unique_posts) - len(posts) + duplicates_found)} unique (Total unique: {len(unique_posts)})")
                
                # If we got fewer than requested, we've reached the end
                if len(posts) < current_batch_size:
                    logger.info(f"   🏁 Reached end of available posts")
                    break
                
                page += 1
                
                # Add a small delay to be respectful to the API
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error fetching WordPress posts page {page}: {e}")
                break
        
        # Update stats
        self.stats["total_wordpress_posts"] = len(unique_posts)
        self.stats["duplicates_removed"] = duplicates_found
        
        if posts_processed >= self.max_wordpress_posts:
            self.stats["posts_skipped_limit"] = len(all_posts) - len(unique_posts)
            logger.info(f"⚠️ Reached WordPress posts limit: {self.max_wordpress_posts:,}")
        
        logger.info(f"✅ WordPress ingestion summary:")
        logger.info(f"   • Total fetched: {len(all_posts):,} posts")
        logger.info(f"   • Unique posts: {len(unique_posts):,} posts")
        logger.info(f"   • Duplicates removed: {duplicates_found:,}")
        
        return unique_posts
    
    async def _fetch_wordpress_page(self, page: int, page_size: int) -> List[Dict]:
        """Fetch a specific page of WordPress posts using direct WordPress API."""
        import aiohttp
        from urllib.parse import urljoin
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'BusinessBelArabi-RAG/2.0 (+https://businessbelarabi.com)',
                'Accept': 'application/json',
                'Accept-Language': 'ar,en;q=0.9'
            }
            
            # Use WordPress REST API directly
            wp_api_url = urljoin(self.data_ingester.wp_base, '/wp-json/wp/v2/posts')
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                params = {
                    "per_page": page_size,
                    "page": page,
                    "_embed": 1,
                    "status": "publish",
                    "orderby": "date",
                    "order": "desc"
                }
                
                async with session.get(wp_api_url, params=params) as response:
                    if response.status == 200:
                        raw_posts = await response.json()
                        processed_posts = []
                        
                        for post in raw_posts:
                            processed_post = self.data_ingester._process_wordpress_post(post)
                            if processed_post:
                                processed_posts.append(processed_post)
                        
                        return processed_posts
                    elif response.status == 404:
                        # No more pages
                        return []
                    elif response.status == 400:
                        # Bad request - possibly page out of range
                        logger.warning(f"Bad request for WordPress page {page}")
                        return []
                    else:
                        logger.error(f"WordPress API returned status {response.status} for page {page}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching WordPress page {page}: {e}")
            return []
    
    def _calculate_post_hash(self, post: Dict) -> str:
        """Calculate a hash for a post to detect duplicates."""
        try:
            # Use title + content + excerpt for duplicate detection
            title = post.get('title', '')
            content = post.get('content', '')
            excerpt = post.get('excerpt', '')
            
            # Create a combined string for hashing
            combined_content = f"{title}||{content}||{excerpt}"
            return hashlib.md5(combined_content.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating post hash: {e}")
            # Fallback to post ID if available
            return str(post.get('id', f'unknown_{hash(str(post))}'))
    
    def process_podcast_episode(self, episode: Dict) -> List[Document]:
        """Process a podcast episode into LangChain documents."""
        documents = []
        
        try:
            episode_id = str(episode.get('id', 'unknown'))
            title = episode.get('title', 'Untitled Episode')
            description = episode.get('description', '')
            transcript = episode.get('transcript', '')
            
            # Create comprehensive content
            content_parts = []
            
            if title:
                content_parts.append(f"العنوان: {title}")
            
            if description:
                content_parts.append(f"الوصف: {description}")
            
            if transcript:
                content_parts.append(f"النص الكامل: {transcript}")
            
            full_content = "\n\n".join(content_parts)
            
            if not full_content.strip():
                logger.warning(f"⚠️ Empty content for podcast episode {episode_id}")
                return documents
            
            # Create base metadata
            metadata = {
                "source": "podcast",
                "source_id": episode_id,
                "title": title,
                "description": description,
                "episode_id": episode_id,
                "audio_url": episode.get('audio_url', ''),
                "published_date": episode.get('published_date', ''),
                "duration": episode.get('duration', ''),
                "content_type": "podcast_episode",
                "language": "ar",
                "content_hash": hashlib.md5(full_content.encode()).hexdigest()
            }
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(full_content)
            
            # Create documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{episode_id}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            logger.debug(f"📄 Processed episode '{title[:50]}...' into {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error processing podcast episode {episode.get('id', 'unknown')}: {e}")
            self.stats["errors"].append(f"Podcast episode {episode.get('id', 'unknown')}: {e}")
        
        return documents
    
    def process_wordpress_post(self, post: Dict) -> List[Document]:
        """Process a WordPress post into LangChain documents."""
        documents = []
        
        try:
            post_id = str(post.get('id', 'unknown'))
            title = post.get('title', 'Untitled Post')
            content = post.get('content', '')
            excerpt = post.get('excerpt', '')
            
            # Create comprehensive content
            content_parts = []
            
            if title:
                content_parts.append(f"العنوان: {title}")
            
            if excerpt:
                content_parts.append(f"الملخص: {excerpt}")
            
            if content:
                content_parts.append(f"المحتوى: {content}")
            
            full_content = "\n\n".join(content_parts)
            
            if not full_content.strip():
                logger.warning(f"⚠️ Empty content for WordPress post {post_id}")
                return documents
            
            # Create base metadata
            metadata = {
                "source": "wordpress",
                "source_id": post_id,
                "title": title,
                "excerpt": excerpt,
                "post_id": post_id,
                "link": post.get('link', ''),
                "author": post.get('author', 'Unknown'),
                "date": post.get('date', ''),
                "categories": post.get('categories', []),
                "tags": post.get('tags', []),
                "content_type": "wordpress_post",
                "language": "ar",
                "content_hash": hashlib.md5(full_content.encode()).hexdigest()
            }
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(full_content)
            
            # Create documents for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{post_id}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            logger.debug(f"📄 Processed post '{title[:50]}...' into {len(documents)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error processing WordPress post {post.get('id', 'unknown')}: {e}")
            self.stats["errors"].append(f"WordPress post {post.get('id', 'unknown')}: {e}")
        
        return documents
    
    async def store_documents_in_vector_db(self, documents: List[Document]) -> bool:
        """Store all documents in the vector database."""
        if not self.vector_manager or not self.vector_manager.is_ready:
            logger.error("❌ Vector store not ready!")
            return False
        
        if not documents:
            logger.warning("⚠️ No documents to store")
            return True
        
        try:
            logger.info(f"🗄️ Storing {len(documents)} document chunks in vector database...")
            
            # Store documents in batches to avoid memory issues
            batch_size = 100
            total_stored = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                success = await self.vector_manager.add_documents(batch)
                if success:
                    total_stored += len(batch)
                    logger.info(f"   ✅ Stored batch {i//batch_size + 1}: {len(batch)} documents (Total: {total_stored})")
                else:
                    logger.error(f"   ❌ Failed to store batch {i//batch_size + 1}")
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"✅ Successfully stored {total_stored} document chunks in vector database")
            self.stats["total_chunks"] = total_stored
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing documents in vector database: {e}")
            return False
    
    async def log_ingestion_results(self):
        """Log the ingestion results to database."""
        try:
            # Log podcast ingestion
            if self.stats["total_podcast_episodes"] > 0:
                self.db_manager.log_data_ingestion(
                    source_type="podcast",
                    source_url=self.data_ingester.podcast_api,
                    items_count=self.stats["total_podcast_episodes"],
                    success=True,
                    processing_time=self.stats["processing_time"],
                    metadata={
                        "total_chunks": self.stats["total_chunks"],
                        "chunk_size": self.settings.chunk_size,
                        "chunk_overlap": self.settings.chunk_overlap
                    }
                )
            
            # Log WordPress ingestion
            if self.stats["total_wordpress_posts"] > 0:
                self.db_manager.log_data_ingestion(
                    source_type="wordpress",
                    source_url=self.data_ingester.wp_base,
                    items_count=self.stats["total_wordpress_posts"],
                    success=True,
                    processing_time=self.stats["processing_time"],
                    metadata={
                        "total_chunks": self.stats["total_chunks"],
                        "chunk_size": self.settings.chunk_size,
                        "chunk_overlap": self.settings.chunk_overlap
                    }
                )
            
            logger.info("✅ Ingestion results logged to database")
            
        except Exception as e:
            logger.error(f"❌ Error logging ingestion results: {e}")
    
    async def run_complete_ingestion(self) -> Dict[str, Any]:
        """
        Run the complete data ingestion process:
        1. Initialize vector store
        2. Fetch all data from APIs
        3. Process and tokenize content
        4. Store in vector database
        """
        start_time = datetime.now()
        logger.info("🚀 STARTING COMPLETE DATA INGESTION PROCESS")
        logger.info("=" * 60)
        
        try:
            # Step 1: Initialize vector store
            logger.info("1️⃣ Initializing vector store...")
            if not await self.initialize_vector_store():
                return {
                    "success": False,
                    "error": "Failed to initialize vector store",
                    "stats": self.stats
                }
            
            # Step 2: Fetch all data
            logger.info("2️⃣ Fetching all data from APIs...")
            
            # Fetch podcast episodes and WordPress posts concurrently
            podcast_episodes, wordpress_posts = await asyncio.gather(
                self.fetch_all_podcast_episodes(),
                self.fetch_all_wordpress_posts(),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(podcast_episodes, Exception):
                logger.error(f"❌ Failed to fetch podcast episodes: {podcast_episodes}")
                podcast_episodes = []
            
            if isinstance(wordpress_posts, Exception):
                logger.error(f"❌ Failed to fetch WordPress posts: {wordpress_posts}")
                wordpress_posts = []
            
            if not podcast_episodes and not wordpress_posts:
                return {
                    "success": False,
                    "error": "No data fetched from either API",
                    "stats": self.stats
                }
            
            # Step 3: Process and tokenize content
            logger.info("3️⃣ Processing and tokenizing content...")
            all_documents = []
            
            # Process podcast episodes
            if podcast_episodes:
                logger.info(f"   📻 Processing {len(podcast_episodes)} podcast episodes...")
                for episode in podcast_episodes:
                    docs = self.process_podcast_episode(episode)
                    all_documents.extend(docs)
                logger.info(f"   ✅ Created {len([d for d in all_documents if d.metadata.get('source') == 'podcast'])} podcast document chunks")
            
            # Process WordPress posts
            if wordpress_posts:
                logger.info(f"   📰 Processing {len(wordpress_posts)} WordPress posts...")
                for post in wordpress_posts:
                    docs = self.process_wordpress_post(post)
                    all_documents.extend(docs)
                logger.info(f"   ✅ Created {len([d for d in all_documents if d.metadata.get('source') == 'wordpress'])} WordPress document chunks")
            
            self.stats["total_documents"] = len(all_documents)
            logger.info(f"📊 Total document chunks created: {len(all_documents)}")
            
            # Step 4: Store in vector database
            logger.info("4️⃣ Storing all content in vector database...")
            success = await self.store_documents_in_vector_db(all_documents)
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to store documents in vector database",
                    "stats": self.stats
                }
            
            # Step 5: Log results
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.info("5️⃣ Logging ingestion results...")
            await self.log_ingestion_results()
            
            # Final summary
            logger.info("=" * 60)
            logger.info("🎉 COMPLETE DATA INGESTION FINISHED SUCCESSFULLY!")
            logger.info(f"📊 Statistics:")
            logger.info(f"   • Podcast Episodes: {self.stats['total_podcast_episodes']}")
            logger.info(f"   • WordPress Posts: {self.stats['total_wordpress_posts']:,} (limit: {self.max_wordpress_posts:,})")
            logger.info(f"   • Total Document Chunks: {self.stats['total_chunks']:,}")
            logger.info(f"   • Duplicates Removed: {self.stats['duplicates_removed']:,}")
            logger.info(f"   • Processing Time: {self.stats['processing_time']:.2f} seconds")
            logger.info(f"   • Errors: {len(self.stats['errors'])}")
            if self.stats['posts_skipped_limit'] > 0:
                logger.info(f"   ⚠️ Posts skipped (over limit): {self.stats['posts_skipped_limit']:,}")
            logger.info("✅ Vector database is ready for RAG queries!")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "message": "Complete data ingestion finished successfully",
                "stats": self.stats
            }
            
        except Exception as e:
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.error("=" * 60)
            logger.error("❌ COMPLETE DATA INGESTION FAILED!")
            logger.error(f"Error: {e}")
            logger.error("=" * 60)
            
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }

# Global instance
complete_ingester = CompleteIngestionService()
