# Business bel Arabi RAG - Advanced Arabic Business Consulting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.350%2B-yellow.svg)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)

A cutting-edge Arabic business consulting system that combines FastAPI, LangChain, and Google Generative AI to provide intelligent business advice in Arabic. Features a modern Streamlit frontend, robust FastAPI backend, and advanced RAG capabilities.

## ✨ Key Features

- 🤖 **Advanced AI Chat** - LangChain-powered conversations with Google Generative AI
- 🧠 **Smart Memory** - Context-aware conversations with memory management
- 📚 **RAG Integration** - Real-time content retrieval from podcast transcripts and blog posts
- 🎙️ **Podcast Integration** - Automatic ingestion of Business bel Arabi podcast content
- 📝 **WordPress Integration** - Dynamic content from businessbelarabi.com
- 💬 **Conversation History** - Persistent storage with search and analytics
- 🔍 **Vector Search** - FAISS-powered semantic search
- 📊 **Analytics Dashboard** - Comprehensive insights and metrics
- 🌐 **Modern UI** - Beautiful Streamlit interface with RTL Arabic support
- 🚀 **FastAPI Backend** - High-performance REST API with automatic documentation

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Business bel Arabi RAG                   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────┐    HTTP/REST    ┌──────────────────┐
│   Streamlit     │ ◄─────────────► │   FastAPI        │
│   Frontend      │                 │   Backend        │
│   (Port 8501)   │                 │   (Port 8000)    │
└─────────────────┘                 └──────────────────┘
                                             │
                    ┌────────────────────────┼──────────────────────┐
                    │                        │                      │
           ┌────────▼───────┐      ┌────────▼──────┐    ┌─────────▼──────┐
           │   SQLite DB    │      │ FAISS Vector │    │  Google AI +   │
           │  + Analytics   │      │    Store      │    │   LangChain    │
           └────────────────┘      └───────────────┘    └────────────────┘
```

## 🚀 **Quick Start**

### 1. **Prerequisites**
```bash
# System requirements
Python 3.8+
Git
```

### 2. **Installation**
```bash
# Clone and setup
git clone <your-repo-url>
cd business-bel-arabi-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Configuration**
Create a `.env` file:
```env
# Required: Google AI API Key
GOOGLE_API_KEY=your_google_ai_api_key

# Optional: External APIs
PODCAST_API_BASE=https://businessbelarabi.360marketingtec.com/api/podcast
WP_BASE=https://businessbelarabi.com

# Server Configuration
BACKEND_URL=http://localhost:8000
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

🔑 **Get Google AI API Key**: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. **Run the Application**

**Terminal 1: Backend**
```bash
python start_backend.py
```

**Terminal 2: Frontend** 
```bash
streamlit run app_with_backend.py
```

### 5. **Access the System**
- 🌐 **Frontend**: http://localhost:8501
- 🔧 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

## 📱 **Alternative: Standalone Streamlit App**

For a simpler setup without the FastAPI backend:

```bash
# Run the standalone version
streamlit run app_enhanced.py
```

## 🛠️ **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | System health check |
| `POST` | `/chat` | Main chat interface |
| `GET` | `/conversations` | Get conversation history |
| `POST` | `/conversations/search` | Search conversations |
| `GET` | `/sessions` | Get all sessions |
| `DELETE` | `/sessions/{id}` | Delete session |
| `POST` | `/ingest/podcast` | Ingest podcast data |
| `POST` | `/ingest/wordpress` | Ingest WordPress posts |
| `GET` | `/analytics` | System analytics |

### **Example API Usage**
```bash
# Health check
curl http://localhost:8000/health

# Chat with AI
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "كيف أبدأ مشروع تجاري؟", "use_context": true}'

# Get analytics
curl http://localhost:8000/analytics
```

## 🧠 **AI & RAG Features**

### **LangChain Integration**
- 🔗 **Google Generative AI** (Gemini) integration
- 🧠 **Conversation Memory** with window management
- 📝 **Custom Prompts** for Arabic business consulting
- ⚙️ **Chain Management** for complex workflows

### **RAG Capabilities**
- 📚 **Vector Search** using FAISS
- 🎙️ **Podcast Transcripts** as knowledge base
- 📰 **Blog Articles** integration
- 🔍 **Semantic Search** with Google embeddings
- 📊 **Context Ranking** and relevance scoring

### **Arabic Language Support**
- 🌐 **RTL Support** in UI
- 📝 **Arabic Prompts** optimized for business
- 🎯 **Topic Detection** in Arabic
- 💬 **Natural Conversations** in Arabic

## 📊 **Analytics & Monitoring**

### **Conversation Analytics**
- 📈 **Response Times** tracking
- 💬 **Message Counts** per session
- 🎯 **Topic Distribution** analysis
- ⭐ **User Ratings** aggregation

### **System Monitoring**
- 🔧 **Component Health** checks
- 💾 **Database Statistics**
- 🧠 **Memory Usage** tracking
- 📊 **API Performance** metrics

### **Business Insights**
- 📅 **Daily/Weekly Trends**
- 🏆 **Most Active Sessions**
- 🎯 **Popular Topics**
- 📈 **Growth Metrics**

## 🧪 **Testing**

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m "not integration" -v  # Unit tests only
pytest tests/ -m "integration" -v      # Integration tests only

# With coverage
pip install pytest-cov
pytest tests/ --cov=backend --cov-report=html
```

## 🚀 **Deployment**

### **Production Setup**
```bash
# Production backend
pip install gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Production frontend
streamlit run app_with_backend.py --server.port 8501 --server.address 0.0.0.0
```

### **Docker Deployment**
```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -t business-bel-arabi-rag .
docker run -p 8000:8000 -p 8501:8501 business-bel-arabi-rag
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

## 🔧 **Building the Model (Data Ingestion)**

### 📋 Prerequisites

Before the AI can give intelligent answers, you need to feed it content:

1. **Get a Google AI API Key**: 
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

2. **Configure your data sources** in `.env`:
   ```bash
   # Your podcast API endpoint
   PODCAST_API_BASE=https://your-podcast-api.com/api
   
   # Your WordPress blog URL  
   WP_BASE=https://your-business-blog.com
   ```

### 🏗️ Building the Knowledge Base

Once the app is running:

1. **Open the Admin Panel** (📊 Admin Panel in sidebar)

2. **Ingest Podcast Content:**
   - Enter your podcast API URL
   - Click "Start Podcast Ingestion"
   - Wait for completion (shows progress)

3. **Ingest Blog Content:**
   - Enter your WordPress URL
   - Click "Start WordPress Ingestion" 
   - Wait for completion

4. **Verify the Build:**
   - Go to "System Status" (⚙️ System Status)
   - Check that collections have data
   - Test search functionality

### 🎯 What Happens During Build

The system will:
- 📥 **Fetch Content**: Download podcasts and articles from your sources
- ✂️ **Process Text**: Break content into chunks for better search
- 🧠 **Create Embeddings**: Convert text to vectors using Google AI
- 💾 **Store Vectors**: Save to Qdrant database for fast semantic search
- 🔗 **Index Relations**: Link related content for recommendations

**⏱️ Build Time**: Typically 10-30 minutes depending on content volume

## 🎥 What You Get

After building, your AI assistant can:

### 💬 Smart Business Chat
- Answer questions in Arabic about business, entrepreneurship, management
- Remember your conversation history for contextual responses
- Provide specific advice based on your industry or situation

### 🎙️ Podcast Recommendations  
- Suggest relevant podcast episodes based on your questions
- Find content matching your current business challenges
- Discover new topics related to your interests

### 🔍 Intelligent Search
- Search through all your content semantically (meaning-based, not just keywords)
- Find relevant information even if you don't use exact terms
- Get ranked results by relevance to your query

## 🏗️ How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │   AI Services    │    │  External Data  │
│                 │    │                  │    │                 │
│ • Chat Interface│    │ • Google Gemini  │    │ • Podcast API   │
│ • Admin Panel   │    │ • Vector Search  │    │ • WordPress     │
│ • Status Monitor│    │ • Text Processing│    │ • Content Data  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌┴──────────────────┐
         │   Qdrant DB     │    │   Background      │
         │   (Optional)    │    │   Processing      │
         │ • Vector Store  │    │                   │
         │ • Fast Search   │    │ • Content Ingest │
         │ • Embeddings    │    │ • Multi-threading │
         └─────────────────┘    └───────────────────┘
```

## ⚙️ Configuration

### 🔧 Required Environment Variables

Edit your `.env` file with these settings:

```bash
# Google AI API Key (REQUIRED)
GOOGLE_API_KEY=your_google_api_key_here

# Data Sources (REQUIRED for content ingestion)
PODCAST_API_BASE=https://your-podcast-api.com/api
WP_BASE=https://your-wordpress-site.com

# Vector Database (Optional - auto-configured)
QDRANT_URL=http://localhost:6333

# Performance (Optional)
MAX_THREADS=4
BATCH_SIZE=10
VECTOR_DIMENSIONS=768
```

## 🗺 **Project Structure**

```
business-bel-arabi-rag/
├── 📁 backend/                 # FastAPI backend
│   ├── main.py                 # FastAPI application
│   ├── models.py              # Pydantic models
│   ├── database.py            # Database management
│   ├── vector_store.py        # Vector store with FAISS
│   ├── data_ingester.py       # Content ingestion
│   └── utils.py               # Utilities and settings
├── 📁 tests/                   # Comprehensive test suite
│   ├── test_backend.py        # Backend API tests
│   └── conftest.py            # Test configuration
├── 📄 app_enhanced.py          # Standalone Streamlit app
├── 📄 app_with_backend.py      # Backend-integrated app
├── 📄 start_backend.py         # Backend startup script
├── 📦 requirements.txt         # Dependencies
├── 📄 DEPLOYMENT.md           # Deployment guide
├── 📄 README.md               # This file
└── 📄 .env.example            # Environment template
```

## 🧪 Testing Your Setup

### 🎯 Test the AI Assistant

1. **Start the app**: `streamlit run app.py`
2. **Open your browser**: Go to http://localhost:8501
3. **Go to Chat page** and ask: "ما هي أفضل النصائح لبدء مشروع تجاري؟"
4. **Check you get**: A detailed response in Arabic

### 🔍 Test Search & Admin

1. **Go to Admin Panel** in the sidebar
2. **Check System Status** - see current system state
3. **Configure your data sources** when ready
4. **Ingest content** and test search functionality

## 📋 Troubleshooting

### 🔍 Common Issues

**"Failed to install dependencies"**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install -r requirements.txt
```

**"Port 8501 is already in use"**
```bash
# Kill existing streamlit processes
pkill -f streamlit
# Then start again
streamlit run app.py
```

**"Google API Key not working"**
- Verify your key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Make sure you've enabled the Gemini API
- Check your API quota limits

**"No search results after ingestion"**
- Check System Status to see if data was actually ingested
- Verify your data source URLs are accessible
- Try a different search query

## 🆆 **Support**

### **Get Help**
- 📧 **Email**: support@businessbelarabi.com
- 🌐 **Website**: https://businessbelarabi.com
- 🎙️ **Podcast**: Business bel Arabi Podcast
- 📚 **Documentation**: [DEPLOYMENT.md](./DEPLOYMENT.md)

### **Common Issues**
- **Backend not connecting**: Check if `python start_backend.py` is running
- **Google AI not working**: Verify your API key in `.env` file
- **Vector store errors**: Ensure disk space and FAISS installation

## 🗺️ **Roadmap**

### **Version 2.0.0** ✅
- FastAPI backend with LangChain
- Enhanced Streamlit frontend
- Vector store RAG integration
- Comprehensive testing suite

### **Version 2.1.0** (Coming Soon)
- 🔄 Qdrant integration
- 🌍 Multi-language support
- 📱 Mobile-responsive design
- 🔐 User authentication

### **Version 2.2.0** (Future)
- 🤖 Multiple AI model support
- 📊 Advanced analytics
- 🌐 Multi-tenant architecture
- 📈 Business intelligence features

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black isort flake8 pytest

# Format code
black .
isort .

# Lint code
flake8 .

# Run tests
pytest tests/ -v
```

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 **Acknowledgments**

- **LangChain** team for the amazing framework
- **FastAPI** for the high-performance web framework
- **Streamlit** for the beautiful UI framework
- **Google AI** for the powerful language models
- **Business bel Arabi** community for inspiration and feedback

---

<div align="center">

**🎙️ Made with ❤️ for Arab Entrepreneurs**

[Website](https://businessbelarabi.com) • [Podcast](https://businessbelarabi.com/podcast) • [Support](mailto:support@businessbelarabi.com)

*Empowering Arabic business knowledge through AI*

</div>

## 🎯 **Quick Start Summary**

### **Full System (Recommended)**
```bash
# 1. Clone and setup
git clone <your-repo>
cd business-bel-arabi-rag
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Google AI API key

# 3. Start backend (Terminal 1)
python start_backend.py

# 4. Start frontend (Terminal 2)
streamlit run app_with_backend.py

# 5. Access system
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### **Standalone App (Simple)**
```bash
# Quick single-app setup
streamlit run app_enhanced.py
```
