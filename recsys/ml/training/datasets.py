import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
import random
from recsys.data.schemas import InteractionSchema, PlaceSchema, FriendLabelSchema


class PlaceRecommendationDataset(Dataset):
    """
    Dataset for place recommendation task with BPR sampling.
    
    For each user-place positive interaction, samples a negative place.
    """
    
    def __init__(
        self,
        interactions: List[InteractionSchema],
        user_id_to_index: Dict[int, int],
        place_id_to_index: Dict[int, int],
        places: List[PlaceSchema],
        rating_threshold: float = 3.5,
        negatives_per_positive: int = 1
    ):
        """
        Args:
            interactions: All user-place interactions
            user_id_to_index: Mapping user_id -> graph index
            place_id_to_index: Mapping place_id -> graph index
            places: List of all places
            rating_threshold: Min implicit_rating to consider positive
            negatives_per_positive: How many negatives to sample per positive
        """
        self.user_id_to_index = user_id_to_index
        self.place_id_to_index = place_id_to_index
        
        # Filter positive interactions
        self.positives = [
            (inter.user_id, inter.place_id)
            for inter in interactions
            if inter.implicit_rating >= rating_threshold
        ]
        
        # Build user -> positive places mapping
        self.user_to_pos_places = {}
        for user_id, place_id in self.positives:
            if user_id not in self.user_to_pos_places:
                self.user_to_pos_places[user_id] = set()
            self.user_to_pos_places[user_id].add(place_id)
        
        # Build city -> places mapping for negative sampling
        self.city_to_places = {}
        for place in places:
            if place.city_id not in self.city_to_places:
                self.city_to_places[place.city_id] = []
            self.city_to_places[place.city_id].append(place.place_id)
        
        # Store user home cities
        self.user_to_city = {}  # To be filled externally
        
        self.negatives_per_positive = negatives_per_positive
    
    def __len__(self):
        return len(self.positives) * self.negatives_per_positive
    
    def __getitem__(self, idx):
        # Get positive sample
        pos_idx = idx // self.negatives_per_positive
        user_id, pos_place_id = self.positives[pos_idx]
        
        # Sample negative place from same city
        city_id = self.user_to_city.get(user_id, 0)  # Default to city 0
        candidate_places = self.city_to_places.get(city_id, list(self.place_id_to_index.keys()))
        
        # Exclude positives
        pos_set = self.user_to_pos_places.get(user_id, set())
        neg_candidates = [p for p in candidate_places if p not in pos_set]
        
        if len(neg_candidates) == 0:
            # Fallback to any place
            neg_candidates = [p for p in self.place_id_to_index.keys() if p not in pos_set]
        
        neg_place_id = random.choice(neg_candidates)
        
        # Convert to graph indices
        user_idx = self.user_id_to_index[user_id]
        pos_place_idx = self.place_id_to_index[pos_place_id]
        neg_place_idx = self.place_id_to_index[neg_place_id]
        
        return {
            'user_idx': user_idx,
            'pos_place_idx': pos_place_idx,
            'neg_place_idx': neg_place_idx
        }


class FriendCompatibilityDataset(Dataset):
    """
    Dataset for friend compatibility task.
    """
    
    def __init__(
        self,
        friend_labels: List[FriendLabelSchema],
        user_id_to_index: Dict[int, int]
    ):
        self.friend_labels = friend_labels
        self.user_id_to_index = user_id_to_index
    
    def __len__(self):
        return len(self.friend_labels)
    
    def __getitem__(self, idx):
        label = self.friend_labels[idx]
        
        user_u_idx = self.user_id_to_index[label.user_u]
        user_v_idx = self.user_id_to_index[label.user_v]
        
        return {
            'user_u_idx': user_u_idx,
            'user_v_idx': user_v_idx,
            'label_compat': label.label_compat,
            'label_attend': label.label_attend
        }

