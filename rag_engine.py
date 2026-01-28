import os
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class RagEngine:
    """
    Offline RAG engine using SentenceTransformers.
    Optimized for Raspberry Pi (uses 'all-MiniLM-L6-v2').
    """
    def __init__(self, data_path="data/syllabus.txt"):
        self.data_path = data_path
        print("📚 Loading RAG Model (MiniLM-L6-v2)...")
        # This model is small and fast enough for Pi 4
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = []
        
        if os.path.exists(data_path):
            self.load_and_index(data_path)
        else:
            print(f"⚠️ Syllabus file not found at {data_path}. RAG disabled.")

    def load_and_index(self, path):
        """Reads text file and indexes it into chunks."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple chunking by paragraphs
        raw_chunks = text.split('\n\n')
        self.chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 50]
        
        print(f"🔢 Indexing {len(self.chunks)} text chunks...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        print("✅ Indexing complete.")

    def retrieve_context(self, query, top_k=2):
        """Finds the most relevant text chunks for the query."""
        if not self.chunks:
            return ""
            
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarity
        similarities = [1 - cosine(query_embedding, emb) for emb in self.embeddings]
        
        # Get top indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        context_parts = [self.chunks[i] for i in top_indices]
        return "\n\n".join(context_parts)