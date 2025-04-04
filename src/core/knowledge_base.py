from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class KnowledgeBase:
    def __init__(self, storage_path: str = "data/knowledge"):
        self.storage_path = storage_path
        self.memories: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        self.load_memories()
    
    def add_memory(self, 
                  text: str, 
                  category: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add new knowledge with metadata."""
        memory = {
            "text": text,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            "metadata": metadata or {},
            "id": len(self.memories)
        }
        self.memories.append(memory)
        self._update_vectors()
        self._save_memories()
    
    def _update_vectors(self) -> None:
        """Update vector representations of memories."""
        if self.memories:
            texts = [m["text"] for m in self.memories]
            self.vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, 
              query: str, 
              top_k: int = 3, 
              category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories by relevance with optional category filter."""
        if not self.memories:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        # Apply category filter if specified
        if category:
            category_mask = [m["category"] == category for m in self.memories]
            similarities = similarities * np.array(category_mask)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                self.memories[idx]["access_count"] += 1
                results.append(self.memories[idx])
        
        self._save_memories()  # Save updated access counts
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        categories = {}
        total_access = 0
        
        for memory in self.memories:
            cat = memory["category"]
            categories[cat] = categories.get(cat, 0) + 1
            total_access += memory["access_count"]
        
        return {
            "total_memories": len(self.memories),
            "categories": categories,
            "total_access": total_access,
            "avg_access": total_access / len(self.memories) if self.memories else 0
        }
    
    def _save_memories(self) -> None:
        """Save memories to disk."""
        memories_file = os.path.join(self.storage_path, "memories.json")
        with open(memories_file, 'w') as f:
            # Convert datetime objects to ISO format strings
            json.dump(self.memories, f, indent=2)
    
    def load_memories(self) -> None:
        """Load memories from disk."""
        memories_file = os.path.join(self.storage_path, "memories.json")
        if os.path.exists(memories_file):
            with open(memories_file, 'r') as f:
                self.memories = json.load(f)
            self._update_vectors()
    
    def clear_category(self, category: str) -> int:
        """Clear all memories of a specific category. Returns number of memories cleared."""
        initial_count = len(self.memories)
        self.memories = [m for m in self.memories if m["category"] != category]
        self._update_vectors()
        self._save_memories()
        return initial_count - len(self.memories)
    
    def prune_old_memories(self, days_threshold: int = 30) -> int:
        """Remove memories older than threshold days. Returns number of memories removed."""
        cutoff_date = datetime.now() - datetime.timedelta(days=days_threshold)
        initial_count = len(self.memories)
        
        self.memories = [
            m for m in self.memories 
            if datetime.fromisoformat(m["timestamp"]) > cutoff_date
        ]
        
        self._update_vectors()
        self._save_memories()
        return initial_count - len(self.memories) 