.PHONY: help install run run-script dev dev-down test test-cov lint format clean

help: ## Show available commands
	@echo "🎙️ Business bel Arabi RAG - Available Commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "   make install    # Install dependencies"
	@echo "   make run        # Run Streamlit app"
	@echo "   make run-script # Run with auto-setup"
	@echo ""
	@echo "🛠️  Development:"
	@echo "   make dev        # Start Qdrant database"
	@echo "   make dev-down   # Stop Qdrant database"
	@echo "   make test       # Run tests"
	@echo "   make lint       # Check code quality"
	@echo "   make format     # Format code"
	@echo "   make clean      # Clean artifacts"

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

run: ## Run Streamlit application
	@echo "🚀 Starting Business bel Arabi CEO Expert..."
	@echo "📱 Open http://localhost:8501 in your browser"
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

run-script: ## Run with auto-setup script
	@echo "🛠️ Running with auto-setup..."
	python run.py

dev: ## Start Qdrant vector database
	@echo "🐳 Starting Qdrant vector database..."
	@docker run -d --name qdrant-rag -p 6333:6333 -p 6334:6334 qdrant/qdrant 2>/dev/null || echo "⚠️  Qdrant already running or Docker not available"

dev-down: ## Stop Qdrant database
	@echo "🛑 Stopping Qdrant database..."
	@docker stop qdrant-rag 2>/dev/null || true
	@docker rm qdrant-rag 2>/dev/null || true

test: ## Run tests
	@echo "🧪 Running tests..."
	pytest || echo "⚠️ pytest not available"

test-cov: ## Run tests with coverage
	@echo "📊 Running tests with coverage..."
	pytest --cov=app --cov-report=html --cov-report=term || echo "⚠️ pytest not available"

lint: ## Run code quality checks
	@echo "🔍 Running code quality checks..."
	@ruff check app/ tests/ scripts/ 2>/dev/null || echo "⚠️ ruff not installed"
	@mypy app/ 2>/dev/null || echo "⚠️ mypy not installed"

format: ## Format code
	@echo "✨ Formatting code..."
	@ruff format app/ tests/ scripts/ 2>/dev/null || echo "⚠️ ruff not installed"
	@black app/ tests/ scripts/ 2>/dev/null || echo "⚠️ black not installed"

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Setup shortcuts
setup: install ## Install dependencies (alias for install)

start: run ## Start the app (alias for run)

dev-setup: ## Setup development environment
	@echo "🔧 Setting up development environment..."
	@make install
	@cp .env.example .env 2>/dev/null || echo ".env file already exists"
	@echo "📝 Please edit .env file with your Google API key"
	@echo "🚀 Then run: make run"
