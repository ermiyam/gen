import faiss
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import List, Dict, Any

class MarketingKnowledgeStore:
    def __init__(self, dimension: int = 768):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []  # Store additional marketing context
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self.model = AutoModelForQuestionAnswering.from_pretrained("facebook/bart-large")

    def add_knowledge(self, text: str, metadata: Dict[str, Any] = None):
        """Add marketing knowledge with metadata."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        embeddings = self.model(**inputs).logits.detach().numpy().mean(axis=1)
        self.index.add(embeddings)
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def search_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant marketing knowledge with context."""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        query_embedding = self.model(**inputs).logits.detach().numpy().mean(axis=1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):  # Ensure valid index
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'relevance_score': float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                })
        return results

    def add_marketing_data(self, data: List[Dict[str, Any]]):
        """Add multiple marketing data points at once."""
        for item in data:
            text = item.get('content', '')
            metadata = {
                'source': item.get('source', ''),
                'date': item.get('date', ''),
                'category': item.get('category', ''),
                'metrics': item.get('metrics', {})
            }
            self.add_knowledge(text, metadata)

    def get_marketing_insights(self, category: str = None) -> List[Dict[str, Any]]:
        """Get insights for a specific marketing category."""
        if not category:
            return [{'text': text, 'metadata': meta} 
                   for text, meta in zip(self.texts, self.metadata)]
        
        return [{'text': text, 'metadata': meta} 
                for text, meta in zip(self.texts, self.metadata)
                if meta.get('category') == category]

# Example usage
if __name__ == "__main__":
    store = MarketingKnowledgeStore()
    
    # Add some marketing data
    marketing_data = [
        {
            'content': 'Product launch campaign achieved 150% of target engagement',
            'source': 'campaign_analytics',
            'date': '2024-03-15',
            'category': 'campaign_performance',
            'metrics': {'engagement_rate': 0.15, 'conversion_rate': 0.08}
        },
        {
            'content': 'Customer feedback shows high satisfaction with new features',
            'source': 'customer_surveys',
            'date': '2024-03-16',
            'category': 'customer_feedback',
            'metrics': {'satisfaction_score': 4.5, 'nps': 45}
        }
    ]
    
    store.add_marketing_data(marketing_data)
    
    # Search for insights
    results = store.search_knowledge("What was the campaign performance?")
    print("Search Results:", results)
    
    # Get category-specific insights
    campaign_insights = store.get_marketing_insights("campaign_performance")
    print("Campaign Insights:", campaign_insights) 