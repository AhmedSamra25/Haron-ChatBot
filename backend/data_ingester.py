#!/usr/bin/env python3
"""
Business bel Arabi RAG - Enhanced Data Ingester

Advanced data ingestion from podcast API and WordPress with async support.
"""

import os
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re
import hashlib
from urllib.parse import urljoin, urlparse
import time

from backend.utils import get_settings

logger = logging.getLogger(__name__)

class EnhancedDataIngester:
    """Enhanced data ingester with async support and better error handling."""
    
    def __init__(self):
        self.settings = get_settings()
        self.podcast_api = "https://appapi.businessbelarabi.com/api/podcast"
        self.articles_api = "https://appapi.businessbelarabi.com/api/articles"
        self.wp_base = self.settings.wp_base  # Keep for backward compatibility
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'BusinessBelArabi-RAG/2.0 (+https://businessbelarabi.com)',
            'Accept': 'application/json',
            'Accept-Language': 'ar,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate'
        }
    
    async def fetch_podcast_episodes_async(self, limit: int = 10) -> List[Dict]:
        """Async version of podcast episodes fetching."""
        if not self.podcast_api:
            logger.error("Podcast API base URL not configured")
            return []
        
        async with aiohttp.ClientSession(
            timeout=self.session_timeout,
            headers=self.headers
        ) as session:
            return await self._fetch_podcast_with_retry(session, limit)
    
    async def _fetch_podcast_with_retry(self, session: aiohttp.ClientSession, limit: int) -> List[Dict]:
        """Fetch podcast episodes with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                params = {"per_page": limit, "page": 1}
                
                async with session.get(self.podcast_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        episodes = self._parse_podcast_response(data, limit)
                        
                        logger.info(f"✅ Successfully fetched {len(episodes)} podcast episodes")
                        return episodes
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Podcast API returned {response.status}: {error_text}")
                        return []
            
            except asyncio.TimeoutError:
                last_error = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Timeout fetching podcast data (attempt {attempt + 1})")
            
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                logger.warning(f"Client error on attempt {attempt + 1}: {e}")
            
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {str(e)}"
                logger.error(f"Invalid JSON response from podcast API: {e}")
                return []
            
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error fetching podcast data: {e}")
            
            if attempt < self.retry_attempts - 1:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        logger.error(f"Failed to fetch podcast data after {self.retry_attempts} attempts: {last_error}")
        return []
    
    def fetch_podcast_episodes(self, limit: int = 10) -> List[Dict]:
        """Sync wrapper for podcast episodes fetching."""
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, there's a running loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.fetch_podcast_episodes_async(limit))
                    return future.result()
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(self.fetch_podcast_episodes_async(limit))
        except Exception as e:
            logger.error(f"Error in sync podcast fetch: {e}")
            return []
    
    def _parse_podcast_response(self, data: Any, limit: int) -> List[Dict]:
        """Parse podcast API response with enhanced error handling."""
        try:
            episodes = []
            
            # Handle different response structures
            if isinstance(data, dict):
                if 'data' in data:
                    episodes_data = data['data']
                    if isinstance(episodes_data, dict) and 'data' in episodes_data:
                        # Paginated response: data.data.data
                        raw_episodes = episodes_data['data']
                    elif isinstance(episodes_data, list):
                        # Direct list: data.data
                        raw_episodes = episodes_data
                    else:
                        # Single episode: data.data
                        raw_episodes = [episodes_data]
                elif 'episodes' in data:
                    # Episodes key: data.episodes
                    raw_episodes = data['episodes']
                else:
                    # Root level data
                    raw_episodes = [data] if not isinstance(data, list) else data
            elif isinstance(data, list):
                # Direct list response
                raw_episodes = data
            else:
                logger.error(f"Unexpected podcast response structure: {type(data)}")
                return []
            
            # Process episodes
            for i, episode in enumerate(raw_episodes[:limit]):
                if not isinstance(episode, dict):
                    logger.warning(f"Skipping invalid episode data: {episode}")
                    continue
                
                processed_episode = self._process_episode_data(episode, i)
                if processed_episode:
                    episodes.append(processed_episode)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Error parsing podcast response: {e}")
            return []
    
    def _process_episode_data(self, episode: Dict, index: int) -> Optional[Dict]:
        """Process individual episode data."""
        try:
            # Extract title
            title = self._extract_title_from_trans(episode)
            if not title:
                title = f"Episode {episode.get('id', index + 1)}"
            
            # Extract description
            description = self._extract_description(episode)
            
            # Extract transcript with enhanced Arabic content handling
            transcript = episode.get('buzzsprout_transcript', '')
            if not transcript:
                transcript = episode.get('transcript', '')
            
            # Extract AI summary if available
            ai_summary = episode.get('ai_summary', '')
            
            # Clean and validate transcript with Arabic-specific processing
            if transcript:
                transcript = self.clean_transcript_content(transcript)
            
            # Combine transcript and AI summary for complete content
            combined_content = ''
            if ai_summary:
                combined_content = f"الملخص الذكي: {ai_summary}\n\n"
            if transcript:
                combined_content += f"النص الكامل: {transcript}"
            elif not ai_summary:  # If no ai_summary and no transcript
                combined_content = transcript
            
            # Build processed episode
            processed_episode = {
                'id': episode.get('id'),
                'uuid': episode.get('uuid', ''),
                'title': title,
                'description': description,
                'transcript': combined_content,
                'ai_summary': ai_summary,
                'audio_url': f"https://appapi.businessbelarabi.com/podcast_details/{episode.get('id')}" if episode.get('id') else '',
                'cover_image': episode.get('cover_image', episode.get('main_image', '')),
                'duration': self._format_duration(episode.get('duration')),
                'scheduled_date': episode.get('scheduled_date_time', ''),
                'published_date': episode.get('published_at', episode.get('created_at', '')),
                'views_count': int(episode.get('views_count', 0)),
                'likes_count': int(episode.get('likes_count', 0)),
                'comments_count': int(episode.get('comments_count', 0)),
                'categories': episode.get('categories', []),
                'tags': episode.get('tags', []),
                'language': episode.get('language', 'ar'),
                'status': episode.get('status', 'published'),
                'metadata': {
                    'source': 'podcast',
                    'api_version': '2.0',
                    'fetched_at': datetime.now().isoformat(),
                    'original_id': episode.get('id'),
                    'content_hash': self._generate_content_hash(title + description + transcript)
                }
            }
            
            return processed_episode
            
        except Exception as e:
            logger.error(f"Error processing episode data: {e}")
            return None
    
    def _extract_title_from_trans(self, episode: Dict) -> str:
        """Extract title from trans array with fallbacks."""
        # Try trans array first for 'name' field (as per new API structure)
        trans = episode.get('trans', [])
        if trans and isinstance(trans, list) and len(trans) > 0:
            name = trans[0].get('name', '')
            if name and name != 'null':
                return self.clean_html_text(name)
            
            title = trans[0].get('title', '')
            if title:
                return self.clean_html_text(title)
        
        # Try direct title field
        if 'title' in episode:
            title = episode['title']
            if isinstance(title, dict):
                title = title.get('rendered', title.get('raw', ''))
            return self.clean_html_text(str(title))
        
        # Try name field
        if 'name' in episode:
            return self.clean_html_text(str(episode['name']))
        
        return ''
    
    def _extract_description(self, episode: Dict) -> str:
        """Extract description with multiple fallback strategies."""
        # Try trans array first
        trans = episode.get('trans', [])
        if trans and isinstance(trans, list) and len(trans) > 0:
            desc = trans[0].get('description', '')
            if desc:
                return self.clean_html_text(desc)
        
        # Try direct description field
        if 'description' in episode:
            desc = episode['description']
            if isinstance(desc, dict):
                desc = desc.get('rendered', desc.get('raw', ''))
            if desc:
                return self.clean_html_text(str(desc))
        
        # Try excerpt field
        if 'excerpt' in episode:
            excerpt = episode['excerpt']
            if isinstance(excerpt, dict):
                excerpt = excerpt.get('rendered', excerpt.get('raw', ''))
            if excerpt:
                return self.clean_html_text(str(excerpt))
        
        # Fallback to transcript excerpt
        transcript = episode.get('buzzsprout_transcript', episode.get('transcript', ''))
        if transcript:
            clean_transcript = self.clean_html_text(transcript)
            # Extract first few sentences
            sentences = clean_transcript.split('.')[:3]
            if len(sentences) > 1:
                return '. '.join(sentences) + '...'
            elif len(clean_transcript) > 200:
                return clean_transcript[:200] + '...'
            else:
                return clean_transcript
        
        return f"Business episode from {episode.get('created_at', 'unknown date')}"
    
    def _format_duration(self, duration: Any) -> str:
        """Format duration to consistent string format."""
        if not duration:
            return "Unknown"
        
        if isinstance(duration, (int, float)):
            # Assume seconds
            minutes, seconds = divmod(int(duration), 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes}:{seconds:02d}"
        
        return str(duration)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for duplicate detection."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def fetch_articles_async(self, limit: int = None, page: int = None) -> List[Dict]:
        """Async version of articles fetching from new API."""
        if not self.articles_api:
            logger.error("Articles API base URL not configured")
            return []
        
        async with aiohttp.ClientSession(
            timeout=self.session_timeout,
            headers=self.headers
        ) as session:
            return await self._fetch_articles_with_retry(session, limit, page)
    
    async def _fetch_articles_with_retry(self, session: aiohttp.ClientSession, limit: int = None, page: int = None) -> List[Dict]:
        """Fetch articles with retry logic and pagination support."""
        articles = []
        current_page = page or 1
        last_page = None
        
        while True:
            try:
                params = {"page": current_page}
                
                async with session.get(self.articles_api, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') and 'data' in data:
                            page_data = data['data']
                            articles_data = page_data.get('data', [])
                            
                            # Process articles from this page
                            for article in articles_data:
                                processed_article = self._process_article_data(article)
                                if processed_article:
                                    articles.append(processed_article)
                            
                            # Check pagination info
                            last_page = page_data.get('last_page', current_page)
                            
                            logger.info(f"✅ Fetched {len(articles_data)} articles from page {current_page}/{last_page}")
                            
                            # If limit is specified, check if we've reached it
                            if limit and len(articles) >= limit:
                                articles = articles[:limit]
                                break
                                
                            # If we specified a specific page, only fetch that page
                            if page is not None:
                                break
                                
                            # If we've reached the last page, stop
                            if current_page >= last_page:
                                break
                                
                            current_page += 1
                            
                            # Small delay between page requests
                            await asyncio.sleep(0.1)
                        else:
                            logger.error(f"Invalid articles API response structure: {data}")
                            break
                            
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Articles API rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Articles API returned {response.status}: {error_text}")
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching articles page {current_page}")
                break
            except aiohttp.ClientError as e:
                logger.warning(f"Client error fetching articles page {current_page}: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from articles API: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching articles: {e}")
                break
        
        logger.info(f"✅ Total articles fetched: {len(articles)} from {current_page - (page or 1) + 1} pages")
        return articles
    
    def _process_article_data(self, article: Dict) -> Optional[Dict]:
        """Process individual article data."""
        try:
            # Extract basic info
            article_id = article.get('id')
            if not article_id:
                return None
            
            # Extract title and description from trans array
            title = "مقال من Business bel Arabi"
            description = "محتوى تعليمي في مجال الأعمال"
            
            trans = article.get('trans', [])
            if trans and len(trans) > 0:
                trans_data = trans[0]
                if trans_data.get('name') and trans_data['name'] != 'null':
                    title = trans_data['name']
                if trans_data.get('desc') and trans_data['desc'] != 'null':
                    description = trans_data['desc']
            
            # Extract content from sections where section_name is paragraph
            content_parts = []
            sections = article.get('sections', [])
            
            if sections:
                for section in sections:
                    if isinstance(section, dict):
                        section_name = section.get('section_name', '')
                        if section_name == 'paragraph':
                            section_content = section.get('content', '')
                            
                            # Handle JSON content format
                            if isinstance(section_content, str):
                                try:
                                    import json
                                    content_json = json.loads(section_content)
                                    if 'paragraph_text' in content_json:
                                        content_parts.append(self.clean_html_text(content_json['paragraph_text']))
                                    elif content_json and isinstance(content_json, dict):
                                        # Handle other potential JSON structures
                                        for key, value in content_json.items():
                                            if isinstance(value, str) and value:
                                                content_parts.append(self.clean_html_text(value))
                                except (json.JSONDecodeError, TypeError):
                                    # If not JSON, treat as regular content
                                    if section_content:
                                        content_parts.append(self.clean_html_text(section_content))
                            elif section_content:
                                content_parts.append(self.clean_html_text(str(section_content)))
            
            # Combine all paragraph contents
            full_content = '\n\n'.join(content_parts) if content_parts else description
            
            # If still no good title, try to create one from metadata
            if title == "مقال من Business bel Arabi":
                # Extract meaningful info from available data
                created_date = article.get('created_at', '')
                if created_date:
                    try:
                        date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                        title = f"مقال {article_id} - {date_obj.strftime('%Y-%m-%d')}"
                    except:
                        title = f"مقال {article_id}"
            
            # Extract categories
            categories = []
            for cat in article.get('categories', []):
                if isinstance(cat, dict):
                    cat_trans = cat.get('trans', [])
                    if cat_trans:
                        cat_name = cat_trans[0].get('name', 'عام')
                        if cat_name and cat_name != 'null':
                            categories.append(cat_name)
            
            if not categories:
                categories = ['أعمال', 'تجارة']
            
            # Extract author info
            author_info = article.get('author', {})
            author_name = "فريق Business bel Arabi"
            if author_info:
                author_trans = author_info.get('trans', [])
                if author_trans and author_trans[0].get('name'):
                    author_name = author_trans[0]['name']
            
            processed_article = {
                'id': article_id,
                'title': title,
                'description': description,
                'content': full_content,  # Use extracted content from sections
                'article_url': f"https://appapi.businessbelarabi.com/article_details/{article_id}" if article_id else '',
                'cover_image': article.get('cover', ''),
                'main_image': article.get('image', ''),
                'author': author_name,
                'categories': categories,
                'created_at': article.get('created_at'),
                'updated_at': article.get('updated_at'),
                'likes_count': article.get('likes_count', 0),
                'comments_count': article.get('comments_count', 0),
                'views_count': article.get('views_count', 0),
                'metadata': {
                    'source': 'article',
                    'api_version': '2.0',
                    'fetched_at': datetime.now().isoformat(),
                    'original_id': article_id,
                    'content_hash': self._generate_content_hash(title + description)
                }
            }
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article data: {e}")
            return None
    
    def fetch_articles(self, limit: int = None, page: int = None) -> List[Dict]:
        """Sync wrapper for articles fetching."""
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, there's a running loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.fetch_articles_async(limit, page))
                    return future.result()
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(self.fetch_articles_async(limit, page))
        except Exception as e:
            logger.error(f"Error in sync articles fetch: {e}")
            return []
    
    async def fetch_wordpress_posts_async(self, limit: int = 10) -> List[Dict]:
        """Async version of WordPress posts fetching."""
        if not self.wp_base:
            logger.error("WordPress base URL not configured")
            return []
        
        async with aiohttp.ClientSession(
            timeout=self.session_timeout,
            headers=self.headers
        ) as session:
            return await self._fetch_wordpress_with_retry(session, limit)
    
    async def _fetch_wordpress_with_retry(self, session: aiohttp.ClientSession, limit: int) -> List[Dict]:
        """Fetch WordPress posts with retry logic."""
        last_error = None
        wp_api_url = urljoin(self.wp_base, '/wp-json/wp/v2/posts')
        
        for attempt in range(self.retry_attempts):
            try:
                params = {
                    "per_page": limit,
                    "_embed": 1,
                    "status": "publish",
                    "orderby": "date",
                    "order": "desc"
                }
                
                async with session.get(wp_api_url, params=params) as response:
                    if response.status == 200:
                        posts = await response.json()
                        processed_posts = []
                        
                        for post in posts:
                            processed_post = self._process_wordpress_post(post)
                            if processed_post:
                                processed_posts.append(processed_post)
                        
                        logger.info(f"✅ Successfully fetched {len(processed_posts)} WordPress posts")
                        return processed_posts
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"WordPress API rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"WordPress API returned {response.status}: {error_text}")
                        return []
            
            except asyncio.TimeoutError:
                last_error = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Timeout fetching WordPress data (attempt {attempt + 1})")
            
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                logger.warning(f"Client error on attempt {attempt + 1}: {e}")
            
            except json.JSONDecodeError as e:
                last_error = f"JSON decode error: {str(e)}"
                logger.error(f"Invalid JSON response from WordPress API: {e}")
                return []
            
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error fetching WordPress data: {e}")
            
            if attempt < self.retry_attempts - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying WordPress fetch in {delay}s...")
                await asyncio.sleep(delay)
        
        logger.error(f"Failed to fetch WordPress data after {self.retry_attempts} attempts: {last_error}")
        return []
    
    def fetch_wordpress_posts(self, limit: int = 10) -> List[Dict]:
        """Sync wrapper for WordPress posts fetching."""
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, there's a running loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.fetch_wordpress_posts_async(limit))
                    return future.result()
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                return asyncio.run(self.fetch_wordpress_posts_async(limit))
        except Exception as e:
            logger.error(f"Error in sync WordPress fetch: {e}")
            return []
    
    def _process_wordpress_post(self, post: Dict) -> Optional[Dict]:
        """Process WordPress post data."""
        try:
            # Extract content
            title = self._extract_rendered_content(post.get('title', {}))
            content = self._extract_rendered_content(post.get('content', {}))
            excerpt = self._extract_rendered_content(post.get('excerpt', {}))
            
            # Clean content
            content = self.clean_html_text(content)
            excerpt = self.clean_html_text(excerpt)
            
            # Extract featured image
            featured_image = ''
            if post.get('_embedded', {}).get('wp:featuredmedia'):
                featured_media = post['_embedded']['wp:featuredmedia'][0]
                featured_image = featured_media.get('source_url', '')
            
            # Extract author info
            author_name = 'Unknown'
            if post.get('_embedded', {}).get('author'):
                author = post['_embedded']['author'][0]
                author_name = author.get('name', 'Unknown')
            
            # Extract categories and tags
            categories = []
            if post.get('_embedded', {}).get('wp:term'):
                for term_group in post['_embedded']['wp:term']:
                    for term in term_group:
                        if term.get('taxonomy') == 'category':
                            categories.append(term.get('name', ''))
            
            tags = []
            if post.get('_embedded', {}).get('wp:term'):
                for term_group in post['_embedded']['wp:term']:
                    for term in term_group:
                        if term.get('taxonomy') == 'post_tag':
                            tags.append(term.get('name', ''))
            
            processed_post = {
                'id': post.get('id'),
                'title': title,
                'content': content,
                'excerpt': excerpt,
                'date': post.get('date'),
                'modified': post.get('modified'),
                'author': author_name,
                'link': post.get('link'),
                'featured_image': featured_image,
                'categories': categories,
                'tags': tags,
                'status': post.get('status', 'publish'),
                'comment_status': post.get('comment_status', 'closed'),
                'metadata': {
                    'source': 'wordpress',
                    'api_version': '2.0',
                    'fetched_at': datetime.now().isoformat(),
                    'original_id': post.get('id'),
                    'content_hash': self._generate_content_hash(title + content + excerpt)
                }
            }
            
            return processed_post
            
        except Exception as e:
            logger.error(f"Error processing WordPress post: {e}")
            return None
    
    def _extract_rendered_content(self, content_obj: Any) -> str:
        """Extract rendered content from WordPress API response."""
        if isinstance(content_obj, dict):
            rendered = content_obj.get('rendered', '')
            if rendered:
                return rendered
            # Fallback to raw content
            return content_obj.get('raw', '')
        return str(content_obj) if content_obj else ''
    
    def clean_transcript_content(self, transcript: str) -> str:
        """Clean transcript content with Arabic podcast-specific processing."""
        if not transcript:
            return ""
        
        try:
            # First do basic HTML cleaning
            text = self.clean_html_text(transcript)
            
            # Remove citation tags specific to podcast transcripts
            text = re.sub(r'<cite>[^<]*</cite>', '', text)
            
            # Remove time markers like <time>0:19</time>
            text = re.sub(r'<time>[^<]*</time>', '', text)
            
            # Clean up remaining HTML tags
            text = re.sub(r'</?[^>]+>', '', text)
            
            # Fix common Arabic text issues in transcripts
            # Remove speaker labels that might interfere with content
            text = re.sub(r'Speaker \d+:', 'المتحدث:', text)
            
            # Clean multiple spaces and normalize Arabic text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'([.!?])\s*([.!?])', r'\1 \2', text)  # Fix multiple punctuation
            
            # Remove empty lines and excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Ensure proper Arabic sentence structure
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning transcript content: {e}")
            return self.clean_html_text(transcript)  # Fallback to basic cleaning
    
    def clean_html_text(self, html_text: str) -> str:
        """Enhanced HTML cleaning with better text processing."""
        if not html_text:
            return ""
        
        try:
            # Convert to string if not already
            text = str(html_text)
            
            # Remove script and style elements
            text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            
            # Remove HTML tags but preserve some formatting
            text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)  # Convert <br> to newlines
            text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)    # Convert </p> to double newlines
            text = re.sub(r'</?[^>]+>', '', text)  # Remove all other HTML tags
            
            # Clean up HTML entities
            html_entities = {
                '&nbsp;': ' ',
                '&amp;': '&',
                '&lt;': '<',
                '&gt;': '>',
                '&quot;': '"',
                '&apos;': "'",
                '&mdash;': '—',
                '&ndash;': '–',
                '&hellip;': '...',
                '&laquo;': '«',
                '&raquo;': '»',
                '&rsquo;': "'",
                '&lsquo;': "'",
                '&rdquo;': '"',
                '&ldquo;': '"'
            }
            
            for entity, replacement in html_entities.items():
                text = text.replace(entity, replacement)
            
            # Remove extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
            text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs to single
            text = re.sub(r'\n ', '\n', text)        # Remove spaces at line start
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML text: {e}")
            return str(html_text)  # Return original if cleaning fails
    
    async def test_endpoints(self) -> Dict[str, Any]:
        """Test all configured endpoints and return status."""
        results = {
            'podcast_api': {
                'url': self.podcast_api,
                'status': 'not_configured' if not self.podcast_api else 'testing',
                'response_time': None,
                'error': None
            },
            'articles_api': {
                'url': self.articles_api,
                'status': 'not_configured' if not self.articles_api else 'testing',
                'response_time': None,
                'error': None
            },
            'wordpress_api': {
                'url': self.wp_base,
                'status': 'deprecated',
                'response_time': None,
                'error': 'Replaced by articles API'
            }
        }
        
        # Test Podcast API
        if self.podcast_api:
            start_time = time.time()
            try:
                episodes = await self.fetch_podcast_episodes_async(1)
                response_time = time.time() - start_time
                results['podcast_api'].update({
                    'status': 'healthy' if episodes else 'no_data',
                    'response_time': round(response_time, 2),
                    'sample_count': len(episodes)
                })
            except Exception as e:
                results['podcast_api'].update({
                    'status': 'error',
                    'response_time': time.time() - start_time,
                    'error': str(e)
                })
        
        # Test Articles API
        if self.articles_api:
            start_time = time.time()
            try:
                articles = await self.fetch_articles_async(limit=1, page=1)
                response_time = time.time() - start_time
                results['articles_api'].update({
                    'status': 'healthy' if articles else 'no_data',
                    'response_time': round(response_time, 2),
                    'sample_count': len(articles)
                })
            except Exception as e:
                results['articles_api'].update({
                    'status': 'error',
                    'response_time': time.time() - start_time,
                    'error': str(e)
                })
        
        # Test WordPress API (deprecated)
        if self.wp_base:
            start_time = time.time()
            try:
                posts = await self.fetch_wordpress_posts_async(1)
                response_time = time.time() - start_time
                results['wordpress_api'].update({
                    'status': 'healthy' if posts else 'no_data',
                    'response_time': round(response_time, 2),
                    'sample_count': len(posts)
                })
            except Exception as e:
                results['wordpress_api'].update({
                    'status': 'error',
                    'response_time': time.time() - start_time,
                    'error': str(e)
                })
        
        return results
    
    def get_ingester_stats(self) -> Dict[str, Any]:
        """Get ingester statistics and configuration."""
        return {
            'podcast_api_configured': bool(self.podcast_api),
            'articles_api_configured': bool(self.articles_api),
            'wordpress_configured': bool(self.wp_base),  # Deprecated
            'retry_attempts': self.retry_attempts,
            'timeout_seconds': self.session_timeout.total,
            'endpoints': {
                'podcast': self.podcast_api,
                'articles': self.articles_api,
                'wordpress': f"{self.wp_base}/wp-json/wp/v2/posts" if self.wp_base else "deprecated"
            }
        }
