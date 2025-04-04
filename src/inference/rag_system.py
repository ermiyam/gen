import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage import StorageContext
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

class DocumentType(Enum):
    BRAND = "brand"
    SCRIPT = "script"
    CAPTION = "caption"
    CAMPAIGN = "campaign"
    ANALYTICS = "analytics"

@dataclass
class Document:
    type: DocumentType
    content: str
    metadata: Dict[str, Any]
    timestamp: str

class RAGSystem:
    """RAG system for Gen to access brand knowledge and past content."""
    
    def __init__(
        self,
        docs_dir: str = "data/brand_docs",
        vector_store_dir: str = "data/vector_store"
    ):
        self.docs_dir = Path(docs_dir)
        self.vector_store_dir = Path(vector_store_dir)
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_rag()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"rag_system_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_rag(self):
        """Initialize the RAG system with vector store."""
        try:
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            chroma_collection = chroma_client.get_or_create_collection("brand_docs")
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Initialize LLM predictor
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto"
            )
            
            llm_predictor = LLMPredictor(
                llm=model,
                tokenizer=tokenizer
            )
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor,
                storage_context=storage_context
            )
            
            # Create index
            self.index = GPTVectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context
            )
            
            self.logger.info("RAG system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def add_document(
        self,
        doc_type: DocumentType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a new document to the RAG system."""
        try:
            # Create document
            doc = Document(
                type=doc_type,
                content=content,
                metadata=metadata or {},
                timestamp=datetime.now().isoformat()
            )
            
            # Save document
            doc_file = self.docs_dir / f"{doc_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc.__dict__, f, ensure_ascii=False, indent=2)
            
            # Add to index
            self.index.insert(
                text=content,
                metadata={
                    "type": doc_type.value,
                    "timestamp": doc.timestamp,
                    **(metadata or {})
                }
            )
            
            self.logger.info(f"Added document of type {doc_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return False
    
    def query(
        self,
        query_text: str,
        doc_type: Optional[DocumentType] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Query the RAG system for relevant information."""
        try:
            # Prepare query
            if doc_type:
                query_text = f"Find information about {doc_type.value}: {query_text}"
            
            # Query index
            query_engine = self.index.as_query_engine()
            response = query_engine.query(query_text)
            
            # Process results
            results = []
            for node in response.source_nodes[:top_k]:
                results.append({
                    "content": node.text,
                    "metadata": node.metadata,
                    "score": node.score
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {str(e)}")
            return []
    
    def get_relevant_context(
        self,
        prompt: str,
        doc_type: Optional[DocumentType] = None
    ) -> str:
        """Get relevant context for a prompt."""
        results = self.query(prompt, doc_type)
        
        if not results:
            return ""
        
        # Format context
        context = "Relevant information:\n"
        for i, result in enumerate(results, 1):
            context += f"\n{i}. {result['content']}\n"
            if result['metadata']:
                context += f"   Source: {result['metadata'].get('type', 'Unknown')}\n"
        
        return context 