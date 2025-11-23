import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from recsys.serving.ann_index import CityAnnIndexManager
from recsys.data.schemas import UserSchema, PlaceSchema
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.config.model_config import ModelConfig


class EmbeddingStore:
    """
    In-memory storage for embeddings.
    """
    
    def __init__(self):
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.place_embeddings: Dict[int, np.ndarray] = {}
    
    def load_from_parquet(self, user_path: str, place_path: str):
        """Load embeddings from parquet files."""
        import pandas as pd
        
        user_df = pd.read_parquet(user_path)
        for _, row in user_df.iterrows():
            self.user_embeddings[row['user_id']] = np.array(row['embedding'])
        
        place_df = pd.read_parquet(place_path)
        for _, row in place_df.iterrows():
            self.place_embeddings[row['place_id']] = np.array(row['embedding'])
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        return self.user_embeddings.get(user_id)
    
    def get_place_embedding(self, place_id: int) -> Optional[np.ndarray]:
        return self.place_embeddings.get(place_id)


class PlaceRecommender:
    """
    Core logic for place recommendations.
    """
    
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        place_ann_manager: CityAnnIndexManager,
        place_head: PlaceHead,
        ctx_encoder: ContextEncoder,
        user_repo,
        place_repo,
        explanation_service,
        config: ModelConfig
    ):
        self.embedding_store = embedding_store
        self.place_ann = place_ann_manager
        self.place_head = place_head
        self.ctx_encoder = ctx_encoder
        self.user_repo = user_repo
        self.place_repo = place_repo
        self.explanation_service = explanation_service
        self.config = config
        
        self.place_head.eval()
    
    def recommend(
        self,
        user_id: int,
        city_id: Optional[int] = None,
        time_slot: Optional[int] = None,
        desired_categories: Optional[List[int]] = None,
        top_k: int = 10,
        top_m_candidates: int = 200
    ) -> List[Dict]:
        """
        Generate place recommendations.
        
        Returns:
            List of dicts with place_id, score, explanations
        """
        # 1. Get user embedding
        z_u = self.embedding_store.get_user_embedding(user_id)
        if z_u is None:
            return []
        
        # 2. Determine city
        if city_id is None:
            user = self.user_repo.get_user(user_id)
            if user is None:
                return []
            city_id = user.home_city_id
        
        # 3. ANN candidate retrieval
        candidates = self.place_ann.search(city_id, z_u, top_m_candidates)
        if not candidates:
            return []
        
        # 4. Prepare context
        if time_slot is None:
            time_slot = 3  # Default: evening
        
        if desired_categories is None:
            desired_categories = [0.0] * 6
        else:
            # Convert to multi-hot
            cat_vec = [0.0] * 6
            for cat_idx in desired_categories:
                if 0 <= cat_idx < 6:
                    cat_vec[cat_idx] = 1.0
            desired_categories = cat_vec
        
        # 5. Score with head
        scored_candidates = []
        
        with torch.no_grad():
            z_u_torch = torch.tensor(z_u, dtype=torch.float32).unsqueeze(0)
            
            for place_id, ann_score in candidates:
                z_p = self.embedding_store.get_place_embedding(place_id)
                if z_p is None:
                    continue
                
                z_p_torch = torch.tensor(z_p, dtype=torch.float32).unsqueeze(0)
                
                # Build context
                city_tensor = torch.tensor([city_id], dtype=torch.long)
                time_tensor = torch.tensor([time_slot], dtype=torch.long)
                cat_tensor = torch.tensor([desired_categories], dtype=torch.float32)
                
                ctx = self.ctx_encoder(city_tensor, time_tensor, cat_tensor)
                
                # Score
                score = self.place_head(z_u_torch, z_p_torch, ctx).item()
                
                scored_candidates.append((place_id, score))
        
        # 6. Sort and top-K
        scored_candidates.sort(key=lambda x: -x[1])
        top_candidates = scored_candidates[:top_k]
        
        # 7. Generate explanations
        results = []
        user = self.user_repo.get_user(user_id)
        if user is None:
            return []
        
        for place_id, score in top_candidates:
            place = self.place_repo.get_place(place_id)
            if place is None:
                continue
            explanations = self.explanation_service.explain_place(user, place)
            
            results.append({
                'place_id': place_id,
                'score': score,
                'explanations': explanations
            })
        
        return results


class PeopleRecommender:
    """
    Core logic for people recommendations.
    """
    
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        user_ann_manager: CityAnnIndexManager,
        friend_head: FriendHead,
        ctx_encoder: ContextEncoder,
        user_repo,
        explanation_service,
        config: ModelConfig
    ):
        self.embedding_store = embedding_store
        self.user_ann = user_ann_manager
        self.friend_head = friend_head
        self.ctx_encoder = ctx_encoder
        self.user_repo = user_repo
        self.explanation_service = explanation_service
        self.config = config
        
        self.friend_head.eval()
    
    def recommend(
        self,
        user_id: int,
        city_id: Optional[int] = None,
        target_place_id: Optional[int] = None,
        activity_tags: Optional[List[int]] = None,
        top_k: int = 10,
        top_m_candidates: int = 200,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Generate people recommendations.
        
        Returns:
            List of dicts with user_id, compat_score, attend_prob, combined_score, explanations
        """
        # 1. Get query user embedding
        z_u = self.embedding_store.get_user_embedding(user_id)
        if z_u is None:
            return []
        
        # 2. Determine city
        if city_id is None:
            user = self.user_repo.get_user(user_id)
            if user is None:
                return []
            city_id = user.home_city_id
        
        # 3. ANN retrieval
        candidates = self.user_ann.search(city_id, z_u, top_m_candidates)
        # Filter out self
        candidates = [(uid, score) for uid, score in candidates if uid != user_id]
        
        if not candidates:
            return []
        
        # 4. Prepare context (dummy for now)
        # Can be enhanced with target_place_id and activity_tags
        
        # 5. Score with friend head
        scored_candidates = []
        
        with torch.no_grad():
            z_u_torch = torch.tensor(z_u, dtype=torch.float32).unsqueeze(0)
            
            for candidate_uid, ann_score in candidates:
                z_v = self.embedding_store.get_user_embedding(candidate_uid)
                if z_v is None:
                    continue
                
                z_v_torch = torch.tensor(z_v, dtype=torch.float32).unsqueeze(0)
                
                # Dummy context
                ctx = torch.zeros(1, self.config.D_CTX_FRIEND)
                
                # Score
                compat_logit, attend_prob = self.friend_head(z_u_torch, z_v_torch, ctx)
                
                compat_score = torch.sigmoid(compat_logit).item()
                attend_prob = attend_prob.item()
                
                combined_score = alpha * compat_score + (1 - alpha) * attend_prob
                
                scored_candidates.append((
                    candidate_uid, compat_score, attend_prob, combined_score
                ))
        
        # 6. Sort and top-K
        scored_candidates.sort(key=lambda x: -x[3])
        top_candidates = scored_candidates[:top_k]
        
        # 7. Generate explanations
        results = []
        query_user = self.user_repo.get_user(user_id)
        if query_user is None:
            return []
        
        for candidate_uid, compat_score, attend_prob, combined_score in top_candidates:
            candidate_user = self.user_repo.get_user(candidate_uid)
            if candidate_user is None:
                continue
            explanations = self.explanation_service.explain_people(query_user, candidate_user)
            
            results.append({
                'user_id': candidate_uid,
                'compat_score': compat_score,
                'attend_prob': attend_prob,
                'combined_score': combined_score,
                'explanations': explanations
            })
        
        return results

