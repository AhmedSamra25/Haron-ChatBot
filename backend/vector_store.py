#!/usr/bin/env python3
"""
Business bel Arabi RAG - Vector Store Manager

Advanced vector store management with LangChain, FAISS, and async support.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle
import json

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Enhanced vector store manager with async support and persistence."""
    
    def __init__(
        self, 
        embeddings_model: Embeddings,
        persist_directory: str = "vector_store",
        index_name: str = "business_bel_arabi"
    ):
        self.embeddings = embeddings_model
        self.persist_directory = persist_directory
        self.index_name = index_name
        self.vectorstore: Optional[FAISS] = None
        self.is_ready = False
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_updated": None,
            "sources": {}
        }
        
        # Create persist directory if not exists
        os.makedirs(persist_directory, exist_ok=True)
    
    async def initialize(self):
        """Initialize the vector store."""
        try:
            # Try to load existing index
            if await self._index_exists():
                await self._load_index()
                logger.info("✅ Loaded existing vector store index")
            else:
                # Create new empty index
                await self._create_empty_index()
                logger.info("✅ Created new vector store index")
            
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            self.is_ready = False
            raise
    
    async def _index_exists(self) -> bool:
        """Check if vector store index exists."""
        index_path = os.path.join(self.persist_directory, f"{self.index_name}.faiss")
        pkl_path = os.path.join(self.persist_directory, f"{self.index_name}.pkl")
        return os.path.exists(index_path) and os.path.exists(pkl_path)
    
    async def _load_index(self):
        """Load existing FAISS index."""
        try:
            index_path = os.path.join(self.persist_directory, self.index_name)
            self.vectorstore = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            await self._load_metadata()
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Fallback to creating empty index
            await self._create_empty_index()
    
    async def _create_empty_index(self):
        """Create an empty FAISS index."""
        # Create a dummy document to initialize the index
        dummy_doc = Document(
            page_content="مرحباً بكم في Business bel Arabi",
            metadata={"source": "init", "type": "dummy"}
        )
        
        self.vectorstore = FAISS.from_documents(
            documents=[dummy_doc],
            embedding=self.embeddings
        )
        
        # Save immediately
        await self._save_index()
        
        # Remove dummy document
        # Note: FAISS doesn't have a direct way to remove documents
        # This is a limitation we'll work around by tracking in metadata
    
    async def _save_index(self):
        """Save FAISS index to disk."""
        try:
            if self.vectorstore:
                index_path = os.path.join(self.persist_directory, self.index_name)
                self.vectorstore.save_local(index_path)
                await self._save_metadata()
                logger.debug("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    async def _load_metadata(self):
        """Load metadata about the vector store."""
        metadata_path = os.path.join(self.persist_directory, f"{self.index_name}_metadata.json")
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
    
    async def _save_metadata(self):
        """Save metadata about the vector store."""
        metadata_path = os.path.join(self.persist_directory, f"{self.index_name}_metadata.json")
        try:
            self.stats["last_updated"] = datetime.now().isoformat()
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        if not self.is_ready or not self.vectorstore:
            logger.error("Vector store not ready")
            return False
        
        try:
            if not documents:
                return False
            
            # Add documents to vector store
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.add_documents,
                documents
            )
            
            # Update statistics
            self.stats["total_documents"] += len(documents)
            self.stats["total_chunks"] += len(documents)
            
            # Update source statistics
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
            
            # Save index and metadata
            await self._save_index()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def add_text_chunks(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> bool:
        """Add text chunks with automatic splitting."""
        if not texts:
            return False
        
        try:
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", "،", "؛", " "]
            )
            
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                # Split text into chunks
                chunks = text_splitter.split_text(text)
                
                for j, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata["chunk_id"] = j
                    doc_metadata["total_chunks"] = len(chunks)
                    
                    doc = Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    )
                    documents.append(doc)
            
            return await self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error adding text chunks: {e}")
            return False
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Perform similarity search."""
        if not self.is_ready or not self.vectorstore:
            return []
        
        try:
            # Perform similarity search
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.similarity_search,
                query,
                k
            )
            
            # Apply metadata filtering if provided
            if filter_metadata:
                filtered_docs = []
                for doc in docs:
                    match = True
                    for key, value in filter_metadata.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                docs = filtered_docs
            
            return docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        score_threshold: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        if not self.is_ready or not self.vectorstore:
            return []
        
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.similarity_search_with_score,
                query,
                k
            )
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_retriever(
        self, 
        search_type: str = "similarity",
        search_kwargs: Dict[str, Any] = None
    ) -> 'AsyncVectorStoreRetriever':
        """Get a retriever for the vector store."""
        if not self.is_ready or not self.vectorstore:
            raise ValueError("Vector store not ready")
        
        return AsyncVectorStoreRetriever(self)
    
    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """Delete documents by metadata filter (not directly supported by FAISS)."""
        # This is a limitation of FAISS - it doesn't support deletion
        # We would need to rebuild the index without the filtered documents
        logger.warning("FAISS does not support document deletion. Consider rebuilding the index.")
        return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = self.stats.copy()
        
        if self.vectorstore:
            # Add current index stats
            try:
                stats["index_size"] = self.vectorstore.index.ntotal
            except:
                pass
        
        stats["is_ready"] = self.is_ready
        return stats
    
    async def rebuild_index(self, documents: List[Document]) -> bool:
        """Rebuild the entire vector store index."""
        try:
            logger.info("Rebuilding vector store index...")
            
            if documents:
                # Create new vector store
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # Update statistics
                self.stats = {
                    "total_documents": len(documents),
                    "total_chunks": len(documents),
                    "last_updated": datetime.now().isoformat(),
                    "sources": {}
                }
                
                # Count sources
                for doc in documents:
                    source = doc.metadata.get("source", "unknown")
                    self.stats["sources"][source] = self.stats["sources"].get(source, 0) + 1
            else:
                await self._create_empty_index()
            
            # Save the rebuilt index
            await self._save_index()
            
            self.is_ready = True
            logger.info("✅ Vector store index rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error rebuilding index: {e}")
            self.is_ready = False
            return False
    
    async def backup_index(self, backup_path: str = None) -> str:
        """Create a backup of the vector store."""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                self.persist_directory, 
                f"{self.index_name}_backup_{timestamp}"
            )
        
        try:
            if self.vectorstore:
                self.vectorstore.save_local(backup_path)
                
                # Copy metadata
                import shutil
                metadata_source = os.path.join(
                    self.persist_directory, 
                    f"{self.index_name}_metadata.json"
                )
                metadata_dest = f"{backup_path}_metadata.json"
                
                if os.path.exists(metadata_source):
                    shutil.copy2(metadata_source, metadata_dest)
                
                logger.info(f"Vector store backed up to: {backup_path}")
                return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def restore_from_backup(self, backup_path: str) -> bool:
        """Restore vector store from backup."""
        try:
            self.vectorstore = FAISS.load_local(
                backup_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Restore metadata
            metadata_path = f"{backup_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            
            # Save as current index
            await self._save_index()
            
            self.is_ready = True
            logger.info(f"✅ Restored vector store from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error restoring from backup: {e}")
            return False
    
    async def optimize_index(self):
        """Optimize the vector store index (placeholder for future optimizations)."""
        # FAISS optimization could include things like:
        # - Index training with more data
        # - Using different index types (IVF, HNSW, etc.)
        # - Quantization
        logger.info("Vector store optimization not implemented for FAISS")
    
    async def close(self):
        """Clean up resources."""
        try:
            if self.vectorstore:
                await self._save_index()
            logger.info("Vector store closed successfully")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")

class AsyncVectorStoreRetriever:
    """Async wrapper for vector store retriever."""
    
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
    
    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Get relevant documents asynchronously."""
        k = kwargs.get('k', 4)
        return await self.vector_manager.similarity_search(query, k=k)
    
    async def aget_relevant_documents_with_score(
        self, 
        query: str, 
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Get relevant documents with scores asynchronously."""
        k = kwargs.get('k', 4)
        score_threshold = kwargs.get('score_threshold', 0.5)
        return await self.vector_manager.similarity_search_with_score(
            query, k=k, score_threshold=score_threshold
        )
