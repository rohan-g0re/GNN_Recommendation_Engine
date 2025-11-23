import faiss
import numpy as np
from typing import List, Tuple, Dict
import pickle
import os


class AnnIndex:
    """
    Wrapper around Faiss for ANN search.
    """
    
    def __init__(self, dimension: int, metric: str = 'cosine'):
        """
        Args:
            dimension: Embedding dimension
            metric: 'cosine' or 'l2'
        """
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.ids = []  # Maps index position -> actual ID
    
    def build(self, embeddings: np.ndarray, ids: List[int]):
        """
        Build index from embeddings.
        
        Args:
            embeddings: (N, D) array
            ids: List of N IDs corresponding to embeddings
        """
        assert len(embeddings) == len(ids)
        self.ids = ids
        
        # Normalize if cosine
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: (D,) query vector
            top_k: Number of results
        
        Returns:
            List of (id, distance) tuples
        """
        if self.index is None:
            return []
        
        query = query.reshape(1, -1).astype(np.float32)
        
        if self.metric == 'cosine':
            faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.ids):
                results.append((self.ids[idx], float(dist)))
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'metric': self.metric,
                'ids': self.ids,
                'index': faiss.serialize_index(self.index)
            }, f)
    
    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.metric = data['metric']
        self.ids = data['ids']
        self.index = faiss.deserialize_index(data['index'])


class CityAnnIndexManager:
    """
    Manages separate ANN indices per city.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.city_indices: Dict[int, AnnIndex] = {}
    
    def build_city_index(
        self,
        city_id: int,
        embeddings: np.ndarray,
        ids: List[int]
    ):
        """Build index for a specific city."""
        index = AnnIndex(self.dimension, metric='cosine')
        index.build(embeddings, ids)
        self.city_indices[city_id] = index
    
    def search(
        self,
        city_id: int,
        query: np.ndarray,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Search within a city's index."""
        if city_id not in self.city_indices:
            return []
        return self.city_indices[city_id].search(query, top_k)
    
    def save(self, output_dir: str, prefix: str):
        """Save all city indices."""
        os.makedirs(output_dir, exist_ok=True)
        for city_id, index in self.city_indices.items():
            index.save(f"{output_dir}/{prefix}_city_{city_id}.idx")
    
    def load(self, input_dir: str, prefix: str, city_ids: List[int]):
        """Load city indices."""
        for city_id in city_ids:
            path = f"{input_dir}/{prefix}_city_{city_id}.idx"
            try:
                index = AnnIndex(self.dimension)
                index.load(path)
                self.city_indices[city_id] = index
            except FileNotFoundError:
                print(f"Warning: Index for city {city_id} not found at {path}")

