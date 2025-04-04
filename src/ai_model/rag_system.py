from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from datetime import datetime

class RAGSystem:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/vector_store",
        knowledge_base_path: str = "data/knowledge_base"
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Create directories if they don't exist
        os.makedirs(index_path, exist_ok=True)
        os.makedirs(knowledge_base_path, exist_ok=True)
        
        # Initialize or load the FAISS index
        self.index = self._initialize_index()
        
        # Load or create knowledge base
        self.knowledge_base = self._load_knowledge_base()
    
    def _initialize_index(self) -> faiss.Index:
        """Initialize or load the FAISS index."""
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file):
            return faiss.read_index(index_file)
        return faiss.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load or create the knowledge base."""
        kb_file = os.path.join(self.knowledge_base_path, "knowledge_base.json")
        if os.path.exists(kb_file):
            with open(kb_file, 'r') as f:
                return json.load(f)
        return {"documents": [], "metadata": {}}
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a new document to the knowledge base and index."""
        # Generate embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Add to knowledge base
        doc_id = str(len(self.knowledge_base["documents"]))
        self.knowledge_base["documents"].append({
            "id": doc_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Save updates
        self._save_updates()
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Retrieve documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base["documents"]):
                doc = self.knowledge_base["documents"][idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(1 / (1 + distance)),
                    "timestamp": doc["timestamp"]
                })
        
        return results
    
    def _save_updates(self):
        """Save updates to disk."""
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        
        # Save knowledge base
        with open(os.path.join(self.knowledge_base_path, "knowledge_base.json"), 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def get_relevant_context(self, query: str, max_tokens: int = 1000) -> str:
        """Get relevant context for a query, respecting token limits."""
        results = self.search(query)
        context = []
        current_tokens = 0
        
        for result in results:
            # Rough estimate of tokens (can be improved with actual tokenizer)
            tokens = len(result["text"].split()) * 1.3
            if current_tokens + tokens > max_tokens:
                break
                
            context.append(result["text"])
            current_tokens += tokens
        
        return "\n\n".join(context) 