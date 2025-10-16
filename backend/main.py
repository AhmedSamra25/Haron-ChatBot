#!/usr/bin/env python3
"""
Business bel Arabi RAG - FastAPI Backend with LangChain Integration

Advanced backend server with proper AI model handling, memory management,
and vector database integration using LangChain.

Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import os
import sys
import json
import asyncio
import logging
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
# Updated imports for new LangChain memory approach
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    ConversationBufferWindowMemory = None
# Remove duplicate import - already imported above
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema.runnable import Runnable
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever

# Local imports
from backend.models import (
    ChatRequest, ChatResponse, ConversationHistory, HealthResponse,
    DataIngestionRequest, DataIngestionResponse, SessionRequest, SessionResponse,
    SearchRequest, SearchResponse, AnalyticsResponse, RecommendationMetadata
)
from backend.database import DatabaseManager
from backend.data_ingester import EnhancedDataIngester
from backend.utils import (
    setup_logging, get_settings, get_content_recommendations, add_context_indicators,
    validate_url_sync
)
from backend.vector_store import VectorStoreManager
from backend.scheduler import get_scheduler
from backend.complete_ingester import complete_ingester

# Setup logging
logger = setup_logging()

# Global variables for managers
db_manager = None
vector_manager = None
data_ingester = None
llm = None
memory_store = {}  # Store memories by session_id

class BusinessArabicChain(Runnable):
    """Custom LangChain chain for Arabic business consulting with RAG."""
    
    def __init__(self, llm, retriever: Optional[BaseRetriever] = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self):
        """Create Arabic business consulting prompt template."""
        # Updated to avoid system message deprecation warning
        human_message = """أنت خبير أعمال متخصص باللغة العربية. مهمتك مساعدة رواد الأعمال والمديرين في:
- ريادة الأعمال والشركات الناشئة
- التسويق والمبيعات  
- الإدارة والقيادة
- التخطيط الاستراتيجي
- الإدارة المالية

### إرشادات الإجابة:
1. اكتب بالعربية بأسلوب واضح ومهني
2. قدم نصائح عملية وقابلة للتطبيق
3. استخدم أمثلة من السوق العربي عندما أمكن
4. اربط النصائح بالسياق المحلي والثقافي
5. كن مختصراً ولكن شاملاً في الإجابة
6. إذا كنت تستخدم معلومات من مقالات، ابدأ الإجابة بـ "وفقاً لمقال [عنوان المقال]... يمكننا القول أن"
7. إذا كنت تستخدم معلومات من البودكاست، اذكر اسم الحلقة أو المتحدث إن أمكن

{context}

### تاريخ المحادثة:
{chat_history}

### السؤال الحالي:
{question}

الرجاء الإجابة بالعربية:"""

        return ChatPromptTemplate.from_template(human_message)
    
    async def ainvoke(self, inputs: Dict[str, Any], config=None) -> Dict[str, Any]:
        """Async invoke method for the chain."""
        try:
            question = inputs.get("question", "")
            chat_history = inputs.get("chat_history", "")
            
            # Get context from retriever if available
            context = ""
            source_docs = []
            if self.retriever and question:
                try:
                    docs = await self.retriever.aget_relevant_documents(question)
                    source_docs = docs[:3]  # Store source documents
                    context = "\n\n### معلومات من البودكاست والمقالات:\n"
                    
                    article_sources = []
                    podcast_sources = []
                    
                    for doc in source_docs:
                        source_type = doc.metadata.get('source', 'unknown')
                        if source_type == 'article' or source_type == 'wordpress':
                            article_sources.append(doc)
                        elif source_type == 'podcast':
                            podcast_sources.append(doc)
                    
                    # Format articles first
                    if article_sources:
                        context += "**المقالات المرجعية:**\n"
                        for i, doc in enumerate(article_sources, 1):
                            title = doc.metadata.get('title', f'مقال {i}')
                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            context += f"{i}. مقال: \"{title}\"\n{content}\n\n"
                    
                    # Format podcasts second
                    if podcast_sources:
                        context += "**حلقات البودكاست المرجعية:**\n"
                        for i, doc in enumerate(podcast_sources, 1):
                            title = doc.metadata.get('title', f'حلقة {i}')
                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            context += f"{i}. بودكاست: \"{title}\"\n{content}\n\n"
                
                except Exception as e:
                    logger.warning(f"Error retrieving context: {e}")
                    context = ""
                    source_docs = []
            
            # Format prompt
            formatted_prompt = self.prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke(formatted_prompt)
            
            return {
                "answer": response.content,
                "context_used": bool(context),
                "sources": len(source_docs),
                "source_documents": source_docs  # Include source documents for recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in BusinessArabicChain: {e}")
            return {
                "answer": "عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى.",
                "context_used": False,
                "sources": 0,
                "source_documents": []
            }
    
    def invoke(self, inputs: Dict[str, Any], config=None) -> Dict[str, str]:
        """Sync invoke method."""
        return asyncio.run(self.ainvoke(inputs, config))

class SimpleChatHistory(BaseChatMessageHistory):
    """Simple in-memory chat history implementation."""
    
    def __init__(self, window_size: int = 10):
        self.messages: List[BaseMessage] = []
        self.window_size = window_size
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
        # Keep only the last window_size messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    
    def clear(self) -> None:
        """Clear the message history."""
        self.messages.clear()

class MemoryManager:
    """Enhanced memory management for conversations using modern LangChain approach."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.chat_histories: Dict[str, SimpleChatHistory] = {}
    
    def get_chat_history(self, session_id: str) -> SimpleChatHistory:
        """Get or create chat history for a session."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = SimpleChatHistory(self.window_size)
        return self.chat_histories[session_id]
    
    def add_message(self, session_id: str, human_message: str, ai_message: str):
        """Add messages to session history."""
        chat_history = self.get_chat_history(session_id)
        chat_history.add_message(HumanMessage(content=human_message))
        chat_history.add_message(AIMessage(content=ai_message))
    
    def get_history_string(self, session_id: str) -> str:
        """Get formatted chat history string."""
        chat_history = self.get_chat_history(session_id)
        messages = chat_history.messages
        
        history_parts = []
        for msg in messages[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                history_parts.append(f"المستخدم: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"الخبير: {msg.content}")
        
        return "\n".join(history_parts)
    
    def clear_memory(self, session_id: str):
        """Clear memory for a session."""
        if session_id in self.chat_histories:
            self.chat_histories[session_id].clear()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "active_sessions": len(self.chat_histories),
            "total_messages": sum(
                len(chat_history.messages) 
                for chat_history in self.chat_histories.values()
            )
        }

# Global memory manager
memory_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global db_manager, vector_manager, data_ingester, llm, memory_manager
    
    logger.info("🚀 Starting Business bel Arabi RAG Backend...")
    
    try:
        # Initialize settings
        settings = get_settings()
        
        # Initialize database
        db_manager = DatabaseManager()
        logger.info("✅ Database manager initialized")
        
        # Initialize LLM with Google Gemini
        if settings.google_api_key:
            try:
                # Try different model names for compatibility (updated for 2024/2025 models)
                model_names = [
                    "gemini-2.0-flash",           # Latest fast model (recommended)
                    "gemini-2.5-flash",           # Alternative fast model
                    "gemini-2.0-flash-001",       # Stable version
                    "gemini-2.5-pro",             # Pro version
                    "gemini-flash-latest",        # Latest alias
                    "gemini-pro-latest",          # Pro alias
                    "gemini-1.5-flash",           # Legacy fallback
                    "gemini-pro",                 # Legacy fallback
                ]
                
                llm = None
                for model_name in model_names:
                    # Try with convert_system_message_to_human first
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=settings.google_api_key,
                            temperature=0.7,
                            max_output_tokens=1024,
                            convert_system_message_to_human=True
                        )
                        # Test the model with a simple call
                        test_response = llm.invoke("Test")
                        logger.info(f"✅ Successfully initialized {model_name} with convert_system_message_to_human=True")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Model {model_name} with convert_system_message_to_human=True failed: {e}")
                        
                        # Try without convert_system_message_to_human
                        try:
                            llm = ChatGoogleGenerativeAI(
                                model=model_name,
                                google_api_key=settings.google_api_key,
                                temperature=0.7,
                                max_output_tokens=1024
                            )
                            # Test the model with a simple call
                            test_response = llm.invoke("Test")
                            logger.info(f"✅ Successfully initialized {model_name} without convert_system_message_to_human")
                            break
                        except Exception as e2:
                            logger.warning(f"⚠️ Model {model_name} without convert_system_message_to_human also failed: {e2}")
                            continue
                
                if not llm:
                    raise Exception("All Gemini models failed to initialize")
                logger.info("✅ Google Gemini Flash initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google LLM: {e}")
                logger.warning("⚠️ Falling back to mock responses")
                llm = None
        else:
            logger.warning("⚠️ No Google API key found - using mock responses")
            llm = None
        
        # Initialize memory manager
        memory_manager = MemoryManager(window_size=10)
        logger.info("✅ Memory manager initialized")
        
        # Initialize vector store with Google embeddings (if available)
        if settings.google_api_key:
            try:
                vector_manager = VectorStoreManager(
                    embeddings_model=GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=settings.google_api_key
                    )
                )
                await vector_manager.initialize()
                logger.info("✅ Vector store initialized with Google embeddings")
            except Exception as e:
                logger.error(f"❌ Vector store initialization failed: {e}")
                logger.warning("⚠️ Vector store will run without embeddings")
                vector_manager = None
        else:
            logger.warning("⚠️ No Google API key - vector store disabled")
            vector_manager = None
        
        # Initialize data ingester
        data_ingester = EnhancedDataIngester()
        logger.info("✅ Data ingester initialized")
        
        # Start scheduler service
        scheduler = get_scheduler()
        scheduler.run_scheduler()
        logger.info("✅ Scheduler service started")
        
        # Skip automatic ingestion on startup - vector store already has data
        logger.info("⚠️ Skipping automatic data ingestion on startup - data already available")
        logger.info("💡 Use /ingestion/trigger endpoint to manually trigger data ingestion if needed")
        
        logger.info("🎉 Backend startup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize backend: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down backend...")
    if vector_manager:
        await vector_manager.close()

# Create FastAPI app
app = FastAPI(
    title="Business bel Arabi RAG API",
    description="Advanced Arabic Business Consulting RAG System with LangChain",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    global db_manager, vector_manager, data_ingester, llm, memory_manager
    
    try:
        # Check database
        db_status = "healthy" if db_manager else "not_initialized"
        if db_manager:
            try:
                analytics = db_manager.get_analytics_data()
                db_status = "healthy"
            except Exception:
                db_status = "error"
        
        # Check LLM
        llm_status = "healthy" if llm else "not_configured"
        
        # Check vector store
        vector_status = "healthy" if vector_manager and vector_manager.is_ready else "not_ready"
        
        # Check memory
        memory_stats = memory_manager.get_session_stats() if memory_manager else {}
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            components={
                "database": db_status,
                "llm": llm_status,
                "vector_store": vector_status,
                "memory": "healthy" if memory_manager else "not_initialized"
            },
            memory_stats=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with LangChain integration."""
    global db_manager, vector_manager, llm, memory_manager
    
    try:
        start_time = datetime.now()
        
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get chat history from memory
        chat_history = ""
        if memory_manager:
            chat_history = memory_manager.get_history_string(session_id)
        
        # Create business chain
        retriever = None
        if vector_manager and vector_manager.is_ready and request.use_context:
            retriever = vector_manager.get_retriever()
        
        if llm:
            chain = BusinessArabicChain(llm, retriever)
            
            # Get response
            result = await chain.ainvoke({
                "question": request.message,
                "chat_history": chat_history
            })
            
            answer = result["answer"]
            context_used = result["context_used"]
            sources_count = result["sources"]
            source_documents = result.get("source_documents", [])
            model_used = "gemini-langchain"
            
            # Determine context source
            context_source = "rag" if context_used and source_documents else "llm"
            
            # Add context indicators to the answer
            answer = add_context_indicators(answer, context_source)
            
            # Generate recommendations from source documents or fallback
            recommendations = []
            
            if source_documents:
                logger.info(f"Trying to generate recommendations from {len(source_documents)} source documents")
                
                try:
                    raw_recommendations = get_content_recommendations([
                        {
                            'metadata': doc.metadata,
                            'page_content': doc.page_content
                        }
                        for doc in source_documents
                    ])
                    
                    # Convert to RecommendationMetadata objects if we have valid recommendations
                    if raw_recommendations:
                        for rec in raw_recommendations:
                            try:
                                recommendations.append(RecommendationMetadata(
                                    title=rec['title'],
                                    url=rec['url'],
                                    source_type=rec['source_type'],
                                    description=rec.get('description'),
                                    categories=rec.get('categories', []),
                                    duration=rec.get('duration'),
                                    published_date=rec.get('published_date')
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to create recommendation from source: {e}")
                                continue
                        
                        logger.info(f"Generated {len(recommendations)} recommendations from source documents")
                    else:
                        logger.warning("Source documents yielded no valid recommendations, falling back to static recommendations")
                
                except Exception as e:
                    logger.error(f"Error generating recommendations from source documents: {e}")
                    logger.info("Falling back to static recommendations due to error")
            
            # If we still don't have recommendations, try to get real ones from vector store
            if not recommendations:
                logger.info("Trying to get real recommendations from vector store")
                recommendations = _get_real_recommendations_from_vector_store(vector_manager, limit=5)
                
                # If still no recommendations, use safe fallback
                if not recommendations:
                    logger.info("Using safe fallback recommendations")
                    recommendations = _get_fallback_recommendations(request.message)
                    logger.info(f"Generated {len(recommendations)} fallback recommendations")
                else:
                    logger.info(f"Generated {len(recommendations)} real recommendations from vector store")
            
        else:
            # Mock response when no API key
            answer = _get_mock_response(request.message)
            context_used = False
            sources_count = 0
            model_used = "mock"
            context_source = "llm"
            # Still provide recommendations even in mock mode
            recommendations = _get_fallback_recommendations(request.message)
            logger.info(f"Generated {len(recommendations)} fallback recommendations (mock mode)")
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Update memory
        if memory_manager:
            memory_manager.add_message(session_id, request.message, answer)
        
        # Save to database
        conversation_id = None
        if db_manager:
            try:
                conversation_id = db_manager.save_conversation(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=answer,
                    response_time=response_time,
                    context_used=context_used,
                    model_used=model_used
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            conversation_id=conversation_id,
            response_time=response_time,
            model_used=model_used,
            context_used=context_used,
            sources_count=sources_count,
            timestamp=datetime.now().isoformat(),
            context_source=context_source,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def _get_real_recommendations_from_vector_store(vector_manager, limit: int = 5) -> List[RecommendationMetadata]:
    """
    Fetch real recommendations from the vector store instead of using hardcoded ones.
    This retrieves actual content URLs that exist in the system.
    """
    if not vector_manager or not vector_manager.is_ready:
        return []
    
    try:
        # Use a general business query to get diverse content
        retriever = vector_manager.get_retriever()
        
        # Try multiple general queries to get diverse results
        queries = ["أعمال", "تسويق", "قيادة", "إدارة", "شركة"]
        all_docs = []
        
        for query in queries:
            try:
                # Use sync version or convert if needed
                if hasattr(retriever, 'get_relevant_documents'):
                    docs = retriever.get_relevant_documents(query)
                else:
                    # If only async available, we'll skip for now
                    continue
                
                # Add documents that have valid URLs
                for doc in docs[:2]:  # Limit per query
                    metadata = doc.metadata
                    url = metadata.get('url', '')
                    if url and url not in [d.metadata.get('url', '') for d in all_docs]:
                        all_docs.append(doc)
                
                if len(all_docs) >= limit * 2:  # Get extra for filtering
                    break
                    
            except Exception as e:
                logger.warning(f"Error retrieving docs for query '{query}': {e}")
                continue
        
        # Convert to recommendations
        recommendations = []
        for doc in all_docs[:limit]:
            try:
                metadata = doc.metadata
                title = metadata.get('title', 'محتوى ذو صلة')
                url = metadata.get('url', '')
                source_type = metadata.get('source', 'blog')
                
                if not url or not title or title in ['Untitled', 'init', 'dummy']:
                    continue
                
                # Use description from page content
                description = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                recommendations.append(RecommendationMetadata(
                    title=title,
                    url=url,
                    source_type=source_type,
                    description=description,
                    categories=metadata.get('categories', []),
                    duration=metadata.get('duration') if source_type == 'podcast' else None,
                    published_date=metadata.get('published_date') or metadata.get('date')
                ))
                
            except Exception as e:
                logger.warning(f"Error creating recommendation from vector doc: {e}")
                continue
        
        logger.info(f"Retrieved {len(recommendations)} real recommendations from vector store")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting real recommendations from vector store: {e}")
        return []

def _get_real_content_recommendations() -> List[RecommendationMetadata]:
    """Get real content recommendations from both articles and podcasts."""
    try:
        from backend.data_ingester import EnhancedDataIngester
        
        ingester = EnhancedDataIngester()
        recommendations = []
        
        # Get both articles and podcasts
        try:
            # Fetch 3 articles
            articles = ingester.fetch_articles(limit=3)
            for article in articles:
                    # Use the new API format for article URLs
                    article_id = article.get('id')
                    article_url = f"https://appapi.businessbelarabi.com/article_details/{article_id}" if article_id else '#'
                    
                    recommendations.append(RecommendationMetadata(
                        title=article.get('title', 'مقال'),
                        url=article_url,
                    source_type='article',
                    description=article.get('description', '')[:200] + '...' if len(article.get('description', '')) > 200 else article.get('description', ''),
                    categories=article.get('categories', ['أعمال']),
                    duration=None,
                    published_date=article.get('created_at', '')
                ))
            
            logger.info(f"Added {len(articles)} article recommendations")
        except Exception as e:
            logger.warning(f"Could not fetch articles: {e}")
        
        # Get podcasts to fill remaining slots
        remaining_slots = 5 - len(recommendations)
        if remaining_slots > 0:
            try:
                episodes = ingester.fetch_podcast_episodes(remaining_slots + 2)  # Get extra for filtering
                
                for episode in episodes:
                    if len(recommendations) >= 5:
                        break
                        
                    title = episode.get('title', '')
                    episode_id = episode.get('id', '')
                    description = episode.get('description', '')
                    
                    # Skip invalid episodes
                    if not title or not episode_id:
                        continue
                    
                    # If title is generic like "Episode 360", try to make it more descriptive
                    if title.startswith('Episode ') and description:
                        # Extract first meaningful part of description for better title
                        desc_lines = description.split('\n')[:3]
                        meaningful_content = ' '.join(desc_lines).strip()
                        if meaningful_content and len(meaningful_content) > 20:
                            title = f"حلقة {title.split(' ')[1]} - {meaningful_content[:50]}..."
                        else:
                            title = f"حلقة {title.split(' ')[1]} من بودكاست Business bel Arabi"
                    
                    # Use new API format for podcast URLs
                    podcast_url = f"https://appapi.businessbelarabi.com/podcast_details/{episode_id}" if episode_id else '#'
                    
                    # Process categories
                    categories = episode.get('categories', [])
                    processed_categories = []
                    for cat in categories:
                        if isinstance(cat, dict):
                            cat_name = cat.get('name', cat.get('title', cat.get('category_name', 'عام')))
                            processed_categories.append(str(cat_name))
                        elif isinstance(cat, str):
                            processed_categories.append(cat)
                        else:
                            processed_categories.append('عام')
                    
                    if not processed_categories:
                        processed_categories = ['بودكاست', 'أعمال']
                    
                    # Create recommendation
                    recommendations.append(RecommendationMetadata(
                        title=title,
                        url=podcast_url,
                        source_type='podcast',
                        description=description[:200] + '...' if len(description) > 200 else description,
                        categories=processed_categories,
                        duration=episode.get('duration'),
                        published_date=episode.get('published_date', episode.get('scheduled_date', ''))
                    ))
                
                logger.info(f"Added {len(recommendations) - len(articles)} podcast recommendations")
                
            except Exception as e:
                logger.warning(f"Could not fetch podcasts: {e}")
        
        logger.info(f"Generated {len(recommendations)} total real content recommendations")
        return recommendations[:5]  # Ensure we return exactly 5
        
    except Exception as e:
        logger.error(f"Error getting real content recommendations: {e}")
        return []

def _get_fallback_recommendations(message: str) -> List[RecommendationMetadata]:
    """Generate fallback recommendations using real podcast content when possible."""
    
    # First try to get real content recommendations (both articles and podcasts)
    real_recommendations = _get_real_content_recommendations()
    if real_recommendations:
        return real_recommendations[:5]  # Return up to 5
    
    # Fallback to safe static recommendations only if we can't get real ones
    logger.warning("Could not fetch real podcast recommendations, using safe fallbacks")
    fallback_recs = [
        {
            'title': 'بودكاست Business bel Arabi - حلقات عن الأعمال',
            'url': 'https://businessbelarabi.com',
            'source_type': 'podcast',
            'description': 'استمع إلى بودكاست Business bel Arabi للحصول على نصائح في ريادة الأعمال والتسويق',
            'categories': ['بودكاست', 'أعمال', 'تسويق'],
            'published_date': '2024-01-15'
        }
    ]
    
    # Convert to RecommendationMetadata objects
    recommendations = []
    for rec_data in fallback_recs:
        recommendations.append(RecommendationMetadata(
            title=rec_data['title'],
            url=rec_data['url'],
            source_type=rec_data['source_type'],
            description=rec_data['description'],
            categories=rec_data['categories'],
            duration=rec_data.get('duration'),
            published_date=rec_data['published_date']
        ))
    
    return recommendations
    
    # This function now calls _get_real_podcast_recommendations() first
    # which provides real podcast URLs. The old keyword matching logic
    # has been replaced with real content fetching.
    pass
    
    return recommendations

def _get_mock_response(message: str) -> str:
    """Generate mock response when no API key is available."""
    return f"""🎙️ **مرحباً بك في Business bel Arabi!**

بخصوص سؤالك: "{message}"

### 🔑 **لاستخدام الميزات الكاملة:**
1. أضف مفتاح Google AI في متغيرات البيئة
2. أعد تشغيل الخادم للحصول على:
   - ردود ذكية مخصصة باستخدام LangChain
   - بحث في محتوى البودكاست والمقالات
   - ذاكرة محادثة محسنة

### 📊 **الميزات المتاحة الآن:**
- حفظ المحادثات وإدارة الجلسات
- واجهة API محسنة
- نظام إدارة الذاكرة

**جرب إعداد المفاتيح للحصول على تجربة كاملة!**"""

# Conversation history endpoints
@app.get("/conversations", response_model=List[ConversationHistory])
async def get_conversations(
    session_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
):
    """Get conversation history with pagination."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        conversations = db_manager.get_conversation_history(
            session_id=session_id,
            limit=limit,
            offset=skip
        )
        
        return [
            ConversationHistory(
                id=conv["id"],
                session_id=conv["session_id"],
                user_message=conv["user_message"],
                assistant_response=conv["assistant_response"],
                timestamp=conv["timestamp"],
                response_time=conv["response_time"],
                model_used=conv["model_used"],
                context_used=conv["context_used"]
            )
            for conv in conversations
        ]
        
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/search", response_model=SearchResponse)
async def search_conversations(request: SearchRequest):
    """Search conversations by content."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        conversations = db_manager.search_conversations(
            query=request.query,
            limit=request.limit or 50
        )
        
        return SearchResponse(
            results=[
                ConversationHistory(
                    id=conv["id"],
                    session_id=conv["session_id"],
                    user_message=conv["user_message"],
                    assistant_response=conv["assistant_response"],
                    timestamp=conv["timestamp"],
                    response_time=conv["response_time"],
                    model_used=conv["model_used"],
                    context_used=conv["context_used"]
                )
                for conv in conversations
            ],
            total_found=len(conversations),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session management endpoints
@app.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(limit: int = 50):
    """Get all sessions with statistics."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        sessions = db_manager.get_sessions(limit=limit)
        
        return [
            SessionResponse(
                session_id=session["session_id"],
                created_at=session["created_at"],
                last_activity=session["last_activity"],
                total_messages=session["total_messages"],
                session_name=session.get("session_name"),
                avg_response_time=session.get("avg_response_time")
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversations."""
    global memory_manager
    
    try:
        # Clear from memory
        if memory_manager:
            memory_manager.clear_memory(session_id)
        
        # Clear from database
        if db_manager:
            db_manager.delete_session(session_id)
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data ingestion endpoints
@app.post("/ingest/podcast", response_model=DataIngestionResponse)
async def ingest_podcast_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest podcast data into vector store."""
    global data_ingester, vector_manager
    
    if not data_ingester:
        raise HTTPException(status_code=503, detail="Data ingester not available")
    
    try:
        # Fetch podcast episodes
        episodes = data_ingester.fetch_podcast_episodes(request.limit or 10)
        
        if not episodes:
            return DataIngestionResponse(
                success=False,
                message="No podcast episodes found",
                items_processed=0
            )
        
        # Process in background if vector store available
        if vector_manager and vector_manager.is_ready:
            background_tasks.add_task(_ingest_episodes_to_vector_store, episodes)
            message = f"Started processing {len(episodes)} episodes in background"
        else:
            message = f"Fetched {len(episodes)} episodes (vector store not available for indexing)"
        
        return DataIngestionResponse(
            success=True,
            message=message,
            items_processed=len(episodes),
            items=episodes[:5]  # Return first 5 as preview
        )
        
    except Exception as e:
        logger.error(f"Podcast ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/articles", response_model=DataIngestionResponse)
async def ingest_articles_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest articles data into vector store."""
    global data_ingester, vector_manager
    
    if not data_ingester:
        raise HTTPException(status_code=503, detail="Data ingester not available")
    
    try:
        # Fetch articles (all pages if no limit, or specified limit)
        limit = request.limit if request.limit else None
        articles = data_ingester.fetch_articles(limit=limit)
        
        if not articles:
            return DataIngestionResponse(
                success=False,
                message="No articles found",
                items_processed=0
            )
        
        # Process in background if vector store available
        if vector_manager and vector_manager.is_ready:
            background_tasks.add_task(_ingest_articles_to_vector_store, articles)
            message = f"Started processing {len(articles)} articles in background"
        else:
            message = f"Fetched {len(articles)} articles (vector store not available for indexing)"
        
        return DataIngestionResponse(
            success=True,
            message=message,
            items_processed=len(articles),
            items=articles[:5]  # Return first 5 as preview
        )
        
    except Exception as e:
        logger.error(f"Articles ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/wordpress", response_model=DataIngestionResponse)
async def ingest_wordpress_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest WordPress data into vector store."""
    global data_ingester, vector_manager
    
    if not data_ingester:
        raise HTTPException(status_code=503, detail="Data ingester not available")
    
    try:
        # Fetch WordPress posts
        posts = data_ingester.fetch_wordpress_posts(request.limit or 10)
        
        if not posts:
            return DataIngestionResponse(
                success=False,
                message="No WordPress posts found",
                items_processed=0
            )
        
        # Process in background if vector store available
        if vector_manager and vector_manager.is_ready:
            background_tasks.add_task(_ingest_posts_to_vector_store, posts)
            message = f"Started processing {len(posts)} posts in background"
        else:
            message = f"Fetched {len(posts)} posts (vector store not available for indexing)"
        
        return DataIngestionResponse(
            success=True,
            message=message,
            items_processed=len(posts),
            items=posts[:5]  # Return first 5 as preview
        )
        
    except Exception as e:
        logger.error(f"WordPress ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/complete")
async def trigger_complete_ingestion(background_tasks: BackgroundTasks):
    """Trigger complete data ingestion manually."""
    global vector_manager
    
    try:
        settings = get_settings()
        
        if not settings.google_api_key:
            raise HTTPException(
                status_code=400, 
                detail="Google API key required for complete ingestion"
            )
        
        # Run complete ingestion in background
        background_tasks.add_task(_run_complete_ingestion)
        
        return {
            "message": "Complete data ingestion started in background",
            "description": "This will fetch ALL data from APIs, process it, and store in vector database",
            "timestamp": datetime.now().isoformat(),
            "estimated_time": "5-15 minutes depending on data volume"
        }
        
    except Exception as e:
        logger.error(f"Complete ingestion trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _run_complete_ingestion():
    """Background task to run complete data ingestion."""
    global vector_manager
    
    try:
        logger.info("🚀 Manual complete data ingestion triggered")
        result = await complete_ingester.run_complete_ingestion()
        
        if result["success"]:
            logger.info("✅ Manual complete data ingestion finished successfully!")
            
            # Update global vector manager if needed
            if complete_ingester.vector_manager and complete_ingester.vector_manager.is_ready:
                vector_manager = complete_ingester.vector_manager
                logger.info("✅ Updated global vector manager with new data")
        else:
            logger.error(f"❌ Manual complete data ingestion failed: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"❌ Manual complete ingestion background task failed: {e}")

async def _ingest_episodes_to_vector_store(episodes: List[Dict]):
    """Background task to ingest episodes into vector store."""
    global vector_manager
    
    try:
        if not vector_manager or not vector_manager.is_ready:
            logger.warning("Vector store not ready for ingestion")
            return
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛"]
        )
        
        for episode in episodes:
            # Create document from episode
            content = f"{episode.get('title', '')}\n\n{episode.get('description', '')}\n\n{episode.get('transcript', '')}"
            
            if content.strip():
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "podcast",
                            "episode_id": episode.get("id"),
                            "title": episode.get("title", ""),
                            "chunk_id": i,
                            "url": episode.get("audio_url", ""),
                            "categories": episode.get("categories", []),
                            "duration": episode.get("duration", ""),
                            "published_date": episode.get("published_date", ""),
                            "description": episode.get("description", "")
                        }
                    )
                    documents.append(doc)
        
        if documents:
            await vector_manager.add_documents(documents)
            logger.info(f"Successfully ingested {len(documents)} podcast chunks into vector store")
        
    except Exception as e:
        logger.error(f"Error ingesting episodes: {e}")

async def _ingest_articles_to_vector_store(articles: List[Dict]):
    """Background task to ingest articles into vector store."""
    global vector_manager
    
    try:
        if not vector_manager or not vector_manager.is_ready:
            logger.warning("Vector store not ready for ingestion")
            return
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛"]
        )
        
        for article in articles:
            # Create document from article
            content = f"{article.get('title', '')}\n\n{article.get('description', '')}\n\n{article.get('content', '')}"
            
            if content.strip():
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "article",
                            "article_id": article.get("id"),
                            "title": article.get("title", ""),
                            "chunk_id": i,
                            "url": article.get("article_url", ""),
                            "categories": article.get("categories", []),
                            "author": article.get("author", ""),
                            "created_at": article.get("created_at", ""),
                            "description": article.get("description", "")
                        }
                    )
                    documents.append(doc)
        
        if documents:
            await vector_manager.add_documents(documents)
            logger.info(f"Successfully ingested {len(documents)} article chunks into vector store")
        
    except Exception as e:
        logger.error(f"Error ingesting articles: {e}")

async def _ingest_posts_to_vector_store(posts: List[Dict]):
    """Background task to ingest WordPress posts into vector store."""
    global vector_manager
    
    try:
        if not vector_manager or not vector_manager.is_ready:
            logger.warning("Vector store not ready for ingestion")
            return
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", "،", "؛"]
        )
        
        for post in posts:
            # Create document from post
            content = f"{post.get('title', '')}\n\n{post.get('excerpt', '')}\n\n{post.get('content', '')}"
            
            if content.strip():
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": "wordpress",
                            "post_id": post.get("id"),
                            "title": post.get("title", ""),
                            "chunk_id": i,
                            "url": post.get("link", ""),
                            "categories": post.get("categories", []),
                            "date": post.get("date", ""),
                            "author": post.get("author", ""),
                            "excerpt": post.get("excerpt", "")
                        }
                    )
                    documents.append(doc)
        
        if documents:
            await vector_manager.add_documents(documents)
            logger.info(f"Successfully ingested {len(documents)} WordPress chunks into vector store")
        
    except Exception as e:
        logger.error(f"Error ingesting posts: {e}")

# Analytics endpoint
@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get comprehensive analytics data."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        analytics = db_manager.get_analytics_data()
        
        # Add memory stats
        memory_stats = {}
        if memory_manager:
            memory_stats = memory_manager.get_session_stats()
        
        # Add vector store stats
        vector_stats = {}
        if vector_manager and vector_manager.is_ready:
            vector_stats = await vector_manager.get_stats()
        
        return AnalyticsResponse(
            **analytics,
            memory_stats=memory_stats,
            vector_stats=vector_stats
        )
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scheduler endpoints
@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and configuration."""
    try:
        scheduler = get_scheduler()
        status = scheduler.get_status()
        return status
    except Exception as e:
        logger.error(f"Scheduler status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/force-update")
async def force_scheduler_update(background_tasks: BackgroundTasks):
    """Force an immediate scheduler update."""
    try:
        scheduler = get_scheduler()
        
        # Run in background to avoid timeout
        background_tasks.add_task(_run_force_update, scheduler)
        
        return {
            "message": "Force update initiated in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Force update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/start")
async def start_scheduler():
    """Start the scheduler service."""
    try:
        scheduler = get_scheduler()
        scheduler.run_scheduler()
        return {
            "message": "Scheduler started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Start scheduler error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the scheduler service."""
    try:
        scheduler = get_scheduler()
        scheduler.stop_scheduler()
        return {
            "message": "Scheduler stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stop scheduler error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _run_force_update(scheduler):
    """Background task to run force update."""
    try:
        result = await scheduler.force_update()
        logger.info(f"Force update completed: {result}")
    except Exception as e:
        logger.error(f"Force update background task failed: {e}")

# Debug endpoint
@app.get("/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to check vector store status and sample data."""
    global vector_manager, data_ingester
    
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "vector_manager_status": {
            "exists": vector_manager is not None,
            "ready": vector_manager.is_ready if vector_manager else False,
        },
        "sample_retrieval": None,
        "ingester_status": None
    }
    
    # Test retrieval with a sample query
    if vector_manager and vector_manager.is_ready:
        try:
            retriever = vector_manager.get_retriever()
            sample_docs = await retriever.aget_relevant_documents("تسويق شركة")
            
            debug_info["sample_retrieval"] = {
                "query": "تسويق شركة",
                "documents_found": len(sample_docs),
                "sample_docs": [
                    {
                        "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in sample_docs[:2]
                ]
            }
        except Exception as e:
            debug_info["sample_retrieval"] = {"error": str(e)}
    
    # Get ingester status
    if data_ingester:
        try:
            debug_info["ingester_status"] = data_ingester.get_ingester_stats()
        except Exception as e:
            debug_info["ingester_status"] = {"error": str(e)}
    
    return debug_info

@app.get("/debug/fallback-recommendations")
async def debug_fallback_recommendations(message: str = "تسويق وقيادة"):
    """Debug endpoint to test fallback recommendations."""
    recommendations = _get_fallback_recommendations(message)
    
    return {
        "message": message,
        "recommendations_count": len(recommendations),
        "recommendations": [
            {
                "title": rec.title,
                "url": rec.url,
                "source_type": rec.source_type,
                "categories": rec.categories,
                "description": rec.description
            }
            for rec in recommendations
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Business bel Arabi RAG API",
        "version": "2.0.0",
        "description": "Advanced Arabic Business Consulting RAG System with LangChain",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "conversations": "/conversations",
            "sessions": "/sessions",
            "ingest_podcast": "/ingest/podcast",
            "ingest_articles": "/ingest/articles",
            "ingest_wordpress": "/ingest/wordpress (deprecated)",
            "analytics": "/analytics",
            "debug_vector_store": "/debug/vector-store",
            "debug_recommendations": "/debug/fallback-recommendations",
            "docs": "/docs"
        },
        "features": [
            "LangChain integration with Google Generative AI",
            "Conversation memory management",
            "Vector store RAG with FAISS",
            "Real-time data ingestion from Articles & Podcast APIs",
            "Mixed content recommendations (Articles + Podcasts)",
            "Proper URL routing to appapi.businessbelarabi.com",
            "Comprehensive analytics",
            "Arabic business expertise",
            "Context-aware response indicators",
            "Automatic pagination handling for complete data ingestion"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
