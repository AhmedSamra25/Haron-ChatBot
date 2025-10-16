#!/usr/bin/env python3
"""
Business bel Arabi RAG - Streamlit Frontend with FastAPI Backend Integration

Enhanced Streamlit application that communicates with the FastAPI backend.
Run with: streamlit run app_with_backend.py
"""

import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime
import os
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Page configuration
st.set_page_config(
    page_title="Business bel Arabi - CEO Expert",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS styling (same as before)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .ltr-text {
        direction: ltr;
        text-align: left;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        position: relative;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .recommendation-container {
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    .recommendation-item {
        margin: 0.5rem 0;
        padding: 0.75rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .recommendation-item:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-link {
        background: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.8em;
        transition: background 0.2s ease;
    }
    
    .recommendation-link:hover {
        background: #218838;
        text-decoration: none;
        color: white;
    }
    
    .category-tag {
        background: #007bff;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.7em;
        margin-right: 2px;
        display: inline-block;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

class BackendClient:
    """Client for communicating with FastAPI backend."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = API_TIMEOUT
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to backend."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = requests.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"🔗 Error connecting to backend: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            response = self._make_request("GET", "/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def chat(self, message: str, session_id: str = None, use_context: bool = True) -> Dict[str, Any]:
        """Send chat message to backend."""
        try:
            payload = {
                "message": message,
                "session_id": session_id,
                "use_context": use_context,
                "max_context_docs": 3,
                "temperature": 0.7
            }
            
            response = self._make_request("POST", "/chat", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else f"HTTP {response.status_code}"
                return {"error": error_detail}
        except Exception as e:
            return {"error": str(e)}
    
    def get_conversations(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get conversation history."""
        try:
            params = {"limit": limit}
            if session_id:
                params["session_id"] = session_id
            
            response = self._make_request("GET", "/conversations", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            st.error(f"Error fetching conversations: {e}")
            return []
    
    def search_conversations(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search conversations."""
        try:
            payload = {"query": query, "limit": limit}
            response = self._make_request("POST", "/conversations/search", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"results": [], "total_found": 0}
        except Exception as e:
            st.error(f"Error searching conversations: {e}")
            return {"results": [], "total_found": 0}
    
    def get_sessions(self) -> List[Dict]:
        """Get all sessions."""
        try:
            response = self._make_request("GET", "/sessions")
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            st.error(f"Error fetching sessions: {e}")
            return []
    
    def ingest_podcast_data(self, limit: int = 10) -> Dict[str, Any]:
        """Ingest podcast data."""
        try:
            payload = {"limit": limit}
            response = self._make_request("POST", "/ingest/podcast", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else f"HTTP {response.status_code}"
                return {"success": False, "message": error_detail}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def ingest_wordpress_data(self, limit: int = 10) -> Dict[str, Any]:
        """Ingest WordPress data."""
        try:
            payload = {"limit": limit}
            response = self._make_request("POST", "/ingest/wordpress", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.content else f"HTTP {response.status_code}"
                return {"success": False, "message": error_detail}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data."""
        try:
            response = self._make_request("GET", "/analytics")
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception as e:
            st.error(f"Error fetching analytics: {e}")
            return {}

def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "backend_client" not in st.session_state:
        st.session_state.backend_client = BackendClient(BACKEND_URL)

def check_backend_connection():
    """Check if backend is accessible."""
    try:
        health = st.session_state.backend_client.health_check()
        return health.get("status") == "healthy"
    except:
        return False

def render_sidebar():
    """Render enhanced sidebar."""
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h2>🎙️ Business bel Arabi</h2>
            <p style="margin: 0; opacity: 0.9;">CEO Expert Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page = st.selectbox(
            "📍 Choose Page:",
            [
                "💬 Chat",
                "📜 Chat History", 
                "📊 Data Analysis",
                "⚙️ System Status",
                "📚 Help"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Backend status
        st.markdown("### 🔧 **Server Status**")
        
        backend_healthy = check_backend_connection()
        if backend_healthy:
            st.markdown('<span class="status-badge status-success">✅ Backend Connected</span>', 
                       unsafe_allow_html=True)
            
            # Get health details
            try:
                health = st.session_state.backend_client.health_check()
                components = health.get("components", {})
                
                for component, status in components.items():
                    if status == "healthy":
                        st.markdown(f'<span class="status-badge status-success">✅ {component}</span>', 
                                   unsafe_allow_html=True)
                    elif status == "not_configured":
                        st.markdown(f'<span class="status-badge status-warning">⚠️ {component}</span>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="status-badge status-error">❌ {component}</span>', 
                                   unsafe_allow_html=True)
            except:
                pass
        else:
            st.markdown('<span class="status-badge status-error">❌ Backend Disconnected</span>', 
                       unsafe_allow_html=True)
            st.error(f"🔗 Unable to connect to server: {BACKEND_URL}")
            st.info("Make sure to start the server using: `python start_backend.py`")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ **Quick Actions**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.session_id = str(__import__('uuid').uuid4())
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        return page

def render_chat_page():
    """Enhanced chat page using backend."""
    st.title("💬 Chat with Business Expert")
    st.markdown("*Ask any question related to business, entrepreneurship, marketing, or management!*")
    
    if not check_backend_connection():
        st.error("🔗 **Cannot connect to server** - Please make sure Backend is running first")
        st.info("Run the command: `python start_backend.py`")
        return
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message ltr-text">
                    <strong>🧑‍💼 You:</strong><br>{message["content"]}
                    <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                        {message.get('timestamp', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message ltr-text">
                    <strong>🤖 CEO Expert:</strong><br>{message["content"]}
                    <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                        ⚡ {message.get('response_time', 0):.2f}s | 
                        🎯 {message.get('model_used', 'backend')}
                        {' | 🔍 Context used' if message.get('context_used') else ''}
                        {f' | 📚 {message.get("sources_count", 0)} sources' if message.get('sources_count', 0) > 0 else ''}
                        {f' | 🧠 {message.get("context_source", "llm")} response' if message.get('context_source') else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display recommendations if available
                recommendations = message.get('recommendations', [])
                if recommendations:
                    st.markdown("""
                    <div class="recommendation-container">
                        <h4 style="color: #28a745; margin-bottom: 0.5rem;">📚 محتوى مُوصى به:</h4>
                    """, unsafe_allow_html=True)
                    
                    for i, rec in enumerate(recommendations, 1):
                        # Determine icon based on source type
                        icon = "🎙️" if rec.get('source_type') == 'podcast' else "📝" if rec.get('source_type') == 'blog' else "📄"
                        
                        # Format duration if available
                        duration_text = f" ({rec.get('duration')})" if rec.get('duration') else ""
                        
                        # Format categories
                        categories = rec.get('categories', [])
                        category_tags = ' '.join([f"<span class='category-tag'>{cat}</span>" for cat in categories[:3]])
                        
                        # Format description
                        description = rec.get('description', '')
                        description_text = f"<div style='color: #6c757d; font-size: 0.9em; margin-bottom: 0.5rem;'>{description[:150]}...</div>" if description else ""
                        
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <div class="recommendation-header">
                                <strong style="color: #495057;">{icon} {rec.get('title', 'Untitled')}{duration_text}</strong>
                                <a href="{rec.get('url', '#')}" target="_blank" class="recommendation-link">عرض</a>
                            </div>
                            {description_text}
                            <div style="margin-top: 0.5rem;">
                                {category_tags}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Debug information (you can remove this later)
                    with st.expander("🔍 Debug: Recommendations Data", expanded=False):
                        st.json(recommendations)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input form
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Write your question here:",
                height=100,
                placeholder="Example: How can I improve my company's marketing strategy?"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(
                "Send 📤", 
                type="primary",
                use_container_width=True
            )
            
            with st.expander("⚙️ Advanced Options"):
                use_context = st.checkbox("🔍 Use context from podcast", value=True)
    
    # Process message
    if submit_button and user_input.strip():
        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
        st.session_state.chat_history.append(user_message)
        
        # Send to backend
        with st.spinner("🤔 CEO Expert is thinking..."):
            try:
                response = st.session_state.backend_client.chat(
                    message=user_input,
                    session_id=st.session_state.session_id,
                    use_context=use_context
                )
                
                if "error" in response:
                    error_msg = f"❌ Error occurred: {response['error']}"
                    st.error(error_msg)
                    assistant_message = {
                        "role": "assistant",
                        "content": "Sorry, I encountered a technical issue. Please try again.",
                        "timestamp": datetime.now().strftime('%H:%M:%S'),
                        "response_time": 0,
                        "model_used": "error"
                    }
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.get("answer", "No response"),
                        "timestamp": datetime.now().strftime('%H:%M:%S'),
                        "response_time": response.get("response_time", 0),
                        "model_used": response.get("model_used", "backend"),
                        "context_used": response.get("context_used", False),
                        "sources_count": response.get("sources_count", 0),
                        "context_source": response.get("context_source", "llm"),
                        "recommendations": response.get("recommendations", [])
                    }
                
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
        
        st.rerun()
    
    # Chat statistics
    if st.session_state.chat_history:
        assistant_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant"]
        if assistant_messages:
            total_recommendations = sum(len(msg.get('recommendations', [])) for msg in assistant_messages)
            if total_recommendations > 0:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                            padding: 0.5rem 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
                    📚 <strong>Total Recommendations Generated:</strong> {total_recommendations}
                </div>
                """, unsafe_allow_html=True)
    
    # Quick suggestions
    st.markdown("### 💡 **Suggested Questions:**")
    
    suggestions = [
        "كيف أبدأ مشروعاً ناجحاً؟",
        "ما هي أفضل استراتيجيات التسويق الرقمي؟", 
        "كيف يمكنني إدارة فريقي بفعالية؟",
        "ما هي طرق زيادة المبيعات؟"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": suggestion,
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                })
                st.rerun()

def render_history_page():
    """Enhanced history page using backend."""
    st.title("📜 تاريخ المحادثات")
    st.markdown("*مراجعة وإدارة جميع محادثاتك السابقة*")
    
    if not check_backend_connection():
        st.error("🔗 تعذر الاتصال بالخادم لجلب البيانات")
        return
    
    # Get analytics
    try:
        analytics = st.session_state.backend_client.get_analytics()
    except:
        analytics = {}
    
    # Overview metrics
    st.markdown("### 📊 **نظرة عامة**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_conversations = analytics.get('total_conversations', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>💬 إجمالي المحادثات</h3>
            <h2 style="color: #667eea;">{total_conversations}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_sessions = analytics.get('total_sessions', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔗 الجلسات</h3>
            <h2 style="color: #764ba2;">{total_sessions}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        today_conversations = analytics.get('today_conversations', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 اليوم</h3>
            <h2 style="color: #28a745;">{today_conversations}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_response_time = analytics.get('avg_response_time', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ متوسط الاستجابة</h3>
            <h2 style="color: #ffc107;">{avg_response_time:.2f}s</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search and filter
    st.markdown("### 🔍 **البحث والتصفية**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "🔎 البحث في المحادثات:",
            placeholder="ابحث في المحادثات السابقة..."
        )
    
    with col2:
        limit = st.selectbox("📊 عدد النتائج:", [20, 50, 100], index=1)
    
    # Get conversations
    try:
        if search_query:
            search_result = st.session_state.backend_client.search_conversations(search_query, limit)
            conversations = search_result.get("results", [])
            st.info(f"🔍 عُثر على {search_result.get('total_found', 0)} نتيجة")
        else:
            conversations = st.session_state.backend_client.get_conversations(limit=limit)
    except:
        conversations = []
        st.error("❌ تعذر جلب المحادثات من الخادم")
    
    # Display conversations
    if conversations:
        st.markdown(f"### 💬 **المحادثات ({len(conversations)} محادثة)**")
        
        for conv in conversations:
            with st.expander(f"💬 {conv['user_message'][:80]}... | ⏰ {conv['timestamp']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**🧑‍💼 سؤالك:**")
                    st.markdown(f"<div class='rtl-text'>{conv['user_message']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("**🤖 الإجابة:**")
                    response_preview = conv['assistant_response'][:500]
                    if len(conv['assistant_response']) > 500:
                        response_preview += "..."
                    st.markdown(f"<div class='rtl-text'>{response_preview}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📊 تفاصيل:**")
                    st.text(f"🕒 {conv['timestamp']}")
                    st.text(f"⚡ {conv.get('response_time', 0):.2f}s")
                    st.text(f"🎯 {conv.get('model_used', 'N/A')}")
                    
                    if conv.get('context_used'):
                        st.markdown("🔍 **السياق مستخدم**")
                    
                    if conv.get('user_rating'):
                        st.text(f"⭐ التقييم: {conv['user_rating']}/5")
    else:
        st.info("📭 لا توجد محادثات متاحة بعد.")

def render_data_analysis_page():
    """Enhanced data analysis page using backend."""
    st.title("📊 تحليل البيانات وإدارة المحتوى")
    st.markdown("*استيراد وتحليل محتوى البودكاست والمقالات*")
    
    if not check_backend_connection():
        st.error("🔗 تعذر الاتصال بالخادم")
        return
    
    # Data ingestion
    st.markdown("### 📥 **استيراد البيانات**")
    
    tab1, tab2 = st.tabs(["🎙️ بودكاست", "📝 مقالات WordPress"])
    
    with tab1:
        st.markdown("#### 🎙️ **محتوى البودكاست**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            podcast_limit = st.slider("عدد الحلقات للاستيراد:", 1, 50, 10)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_podcast_btn = st.button("🚀 استيراد حلقات البودكاست", type="primary")
        
        if fetch_podcast_btn:
            with st.spinner("🎙️ جاري استيراد حلقات البودكاست..."):
                try:
                    result = st.session_state.backend_client.ingest_podcast_data(podcast_limit)
                    
                    if result.get("success"):
                        st.success(f"✅ {result.get('message', 'تم الاستيراد بنجاح')}")
                        
                        # Display preview if available
                        if result.get("items"):
                            st.markdown("### 🎧 **معاينة الحلقات:**")
                            for item in result["items"][:3]:
                                with st.expander(f"🎙️ {item.get('title', 'بدون عنوان')}"):
                                    if item.get('description'):
                                        st.markdown(f"**الوصف:** {item['description'][:200]}...")
                                    if item.get('duration'):
                                        st.text(f"المدة: {item['duration']}")
                    else:
                        st.error(f"❌ {result.get('message', 'فشل في الاستيراد')}")
                        
                except Exception as e:
                    st.error(f"❌ خطأ: {str(e)}")
    
    with tab2:
        st.markdown("#### 📝 **مقالات WordPress**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            wp_limit = st.slider("عدد المقالات للاستيراد:", 1, 50, 10)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_wp_btn = st.button("🚀 استيراد مقالات WordPress", type="primary")
        
        if fetch_wp_btn:
            with st.spinner("📝 جاري استيراد مقالات WordPress..."):
                try:
                    result = st.session_state.backend_client.ingest_wordpress_data(wp_limit)
                    
                    if result.get("success"):
                        st.success(f"✅ {result.get('message', 'تم الاستيراد بنجاح')}")
                        
                        if result.get("items"):
                            st.markdown("### 📰 **معاينة المقالات:**")
                            for item in result["items"][:3]:
                                with st.expander(f"📝 {item.get('title', 'بدون عنوان')}"):
                                    if item.get('excerpt'):
                                        st.markdown(f"**المقتطف:** {item['excerpt'][:200]}...")
                    else:
                        st.error(f"❌ {result.get('message', 'فشل في الاستيراد')}")
                        
                except Exception as e:
                    st.error(f"❌ خطأ: {str(e)}")
    
    # Analytics
    st.markdown("---")
    st.markdown("### 📈 **إحصائيات**")
    
    try:
        analytics = st.session_state.backend_client.get_analytics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 **إحصائيات المحادثات**")
            
            # Create simple metrics display
            metrics = [
                ("💬", "إجمالي المحادثات", analytics.get('total_conversations', 0)),
                ("🔗", "إجمالي الجلسات", analytics.get('total_sessions', 0)),
                ("📅", "محادثات اليوم", analytics.get('today_conversations', 0)),
                ("📈", "محادثات الأسبوع", analytics.get('week_conversations', 0)),
            ]
            
            for icon, label, value in metrics:
                st.markdown(f"**{icon} {label}:** {value}")
        
        with col2:
            st.markdown("#### 🎯 **استخدام النماذج**")
            
            model_usage = analytics.get('model_usage', {})
            if model_usage:
                for model, count in model_usage.items():
                    st.markdown(f"**{model}:** {count}")
            else:
                st.info("لا توجد بيانات متاحة")
    
    except Exception as e:
        st.error(f"❌ تعذر جلب الإحصائيات: {str(e)}")

def render_system_status_page():
    """System status page using backend."""
    st.title("⚙️ حالة النظام")
    st.markdown("*مراقبة حالة النظام والخوادم*")
    
    # Backend connection status
    st.markdown("### 🔗 **حالة الاتصال**")
    
    backend_healthy = check_backend_connection()
    
    if backend_healthy:
        st.success("✅ **الخادم متصل ويعمل بشكل طبيعي**")
        
        try:
            health = st.session_state.backend_client.health_check()
            
            st.markdown("### 📊 **تفاصيل حالة المكونات**")
            components = health.get("components", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### قواعد البيانات والتخزين")
                for component in ["database", "vector_store", "memory"]:
                    status = components.get(component, "unknown")
                    if status == "healthy":
                        st.success(f"✅ {component}: صحي")
                    elif status == "not_configured":
                        st.warning(f"⚠️ {component}: غير مُعد")
                    else:
                        st.error(f"❌ {component}: خطأ")
            
            with col2:
                st.markdown("#### نماذج الذكاء الاصطناعي")
                for component in ["llm"]:
                    status = components.get(component, "unknown")
                    if status == "healthy":
                        st.success(f"✅ {component}: صحي")
                    elif status == "not_configured":
                        st.warning(f"⚠️ {component}: غير مُعد")
                    else:
                        st.error(f"❌ {component}: خطأ")
            
            # Memory stats
            memory_stats = health.get("memory_stats", {})
            if memory_stats:
                st.markdown("### 💾 **إحصائيات الذاكرة**")
                st.text(f"الجلسات النشطة: {memory_stats.get('active_sessions', 0)}")
                st.text(f"إجمالي الرسائل: {memory_stats.get('total_messages', 0)}")
                
        except Exception as e:
            st.error(f"❌ تعذر الحصول على تفاصيل الحالة: {str(e)}")
    else:
        st.error("❌ **تعذر الاتصال بالخادم**")
        st.info(f"رابط الخادم: {BACKEND_URL}")
        st.info("تأكد من تشغيل الخادم: `python start_backend.py`")

def render_help_page():
    """Help page with backend integration info."""
    st.title("📚 المساعدة والدليل")
    st.markdown("*دليل الاستخدام مع تكامل الخادم الخلفي*")
    
    tab1, tab2 = st.tabs(["🚀 البدء السريع", "🔧 إعداد النظام"])
    
    with tab1:
        st.markdown("""
        ## 🚀 **دليل البدء السريع**
        
        ### 1. **تشغيل الخادم الخلفي (Backend)**
        ```bash
        # تثبيت المتطلبات
        pip install -r requirements.txt
        
        # تشغيل الخادم
        python start_backend.py
        ```
        
        ### 2. **تشغيل الواجهة الأمامية (Frontend)**
        ```bash
        # في نافذة طرفية جديدة
        streamlit run app_with_backend.py
        ```
        
        ### 3. **إعداد المتغيرات البيئية**
        أنشئ ملف `.env` وأضف:
        ```
        GOOGLE_API_KEY=your_google_api_key
        PODCAST_API_BASE=https://businessbelarabi.360marketingtec.com/api/podcast
        WP_BASE=https://businessbelarabi.com
        BACKEND_URL=http://localhost:8000
        ```
        """)
    
    with tab2:
        st.markdown("""
        ## 🔧 **إعداد النظام المتقدم**
        
        ### **هندسة النظام:**
        - **Frontend**: Streamlit (Port 8501)
        - **Backend**: FastAPI (Port 8000)
        - **Database**: SQLite + FAISS Vector Store
        - **AI**: Google Generative AI + LangChain
        
        ### **الميزات المتقدمة:**
        - ✅ إدارة الذاكرة مع LangChain
        - ✅ استرجاع المعلومات (RAG) المحسن
        - ✅ API متقدم مع FastAPI
        - ✅ تخزين المحادثات والتحليلات
        - ✅ استيراد البيانات غير المتزامن
        
        ### **استكشاف الأخطاء:**
        - تأكد من تشغيل Backend قبل Frontend
        - تحقق من صحة المتغيرات البيئية
        - راجع لوجات النظام في مجلد `logs/`
        """)

def main():
    """Main application function."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar and get current page
        current_page = render_sidebar()
        
        # Route to appropriate page
        if current_page == "💬 Chat":
            render_chat_page()
        elif current_page == "📜 Chat History":
            render_history_page()
        elif current_page == "📊 Data Analysis":
            render_data_analysis_page()
        elif current_page == "⚙️ System Status":
            render_system_status_page()
        elif current_page == "📚 Help":
            render_help_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.7; padding: 1rem;">
            🎙️ <strong>Business bel Arabi CEO Expert</strong> | 
            Frontend + Backend v2.0 | 
            Made with ❤️ for Arab Entrepreneurs
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error("❌ حدث خطأ في التطبيق")
        st.error(f"تفاصيل الخطأ: {str(e)}")

if __name__ == "__main__":
    main()
