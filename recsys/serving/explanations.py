from typing import List
from recsys.data.schemas import UserSchema, PlaceSchema
from recsys.config.constants import FINE_TAGS, COARSE_CATEGORIES, VIBE_TAGS
import numpy as np


class ExplanationService:
    """
    Generates human-readable explanations for recommendations.
    """
    
    def explain_place(
        self,
        user: UserSchema,
        place: PlaceSchema,
        top_k_tags: int = 2
    ) -> List[str]:
        """
        Generate explanations for why a place was recommended to a user.
        
        Args:
            user: UserSchema
            place: PlaceSchema
            top_k_tags: Number of tag overlaps to mention
        
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # 1. Find overlapping fine tags
        user_fine = np.array(user.fine_pref)
        place_fine = np.array(place.fine_tag_vector)
        
        # Element-wise product to find mutual high-weight tags
        overlap_scores = user_fine * place_fine
        top_indices = np.argsort(-overlap_scores)[:top_k_tags]
        
        top_tags = [FINE_TAGS[idx] for idx in top_indices if overlap_scores[idx] > 0.01]
        
        if len(top_tags) >= 2:
            explanations.append(
                f"Matches your interest in {top_tags[0]} and {top_tags[1]}."
            )
        elif len(top_tags) == 1:
            explanations.append(
                f"Matches your interest in {top_tags[0]}."
            )
        
        # 2. Check coarse category alignment
        user_cat = np.array(user.cat_pref)
        place_cat = np.array(place.category_one_hot)
        
        top_cat_idx = np.argmax(user_cat * place_cat)
        if user_cat[top_cat_idx] > 0.15 and place_cat[top_cat_idx] > 0:
            category_name = COARSE_CATEGORIES[top_cat_idx]
            explanations.append(
                f"You enjoy {category_name} spots."
            )
        
        # 3. Check neighborhood proximity
        if place.neighborhood_id in user.area_freqs:
            explanations.append(
                f"You often go out in this neighborhood."
            )
        
        # If no strong explanations, add a generic one
        if not explanations:
            explanations.append("Recommended based on your activity history.")
        
        return explanations[:3]  # Max 3 explanations
    
    def explain_people(
        self,
        user_u: UserSchema,
        user_v: UserSchema,
        top_k_tags: int = 2
    ) -> List[str]:
        """
        Generate explanations for why two users are compatible.
        
        Args:
            user_u: Query user
            user_v: Candidate user
            top_k_tags: Number of overlaps to mention
        
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # 1. Find overlapping vibe/personality tags
        vibe_u = np.array(user_u.vibe_pref)
        vibe_v = np.array(user_v.vibe_pref)
        
        vibe_overlap = vibe_u * vibe_v
        top_vibe_indices = np.argsort(-vibe_overlap)[:top_k_tags]
        
        top_vibe_tags = [
            VIBE_TAGS[idx] for idx in top_vibe_indices
            if vibe_overlap[idx] > 0.01
        ]
        
        if len(top_vibe_tags) >= 2:
            explanations.append(
                f"You both are {top_vibe_tags[0]} and {top_vibe_tags[1]}."
            )
        elif len(top_vibe_tags) == 1:
            explanations.append(
                f"You both are {top_vibe_tags[0]}."
            )
        
        # 2. Find overlapping fine interests
        fine_u = np.array(user_u.fine_pref)
        fine_v = np.array(user_v.fine_pref)
        
        fine_overlap = fine_u * fine_v
        top_fine_indices = np.argsort(-fine_overlap)[:top_k_tags]
        
        top_fine_tags = [
            FINE_TAGS[idx] for idx in top_fine_indices
            if fine_overlap[idx] > 0.01
        ]
        
        if len(top_fine_tags) >= 2:
            explanations.append(
                f"You both like {top_fine_tags[0]} and {top_fine_tags[1]}."
            )
        elif len(top_fine_tags) == 1:
            explanations.append(
                f"You both like {top_fine_tags[0]}."
            )
        
        # 3. Check shared neighborhoods
        shared_neighborhoods = set(user_u.area_freqs.keys()) & set(user_v.area_freqs.keys())
        if shared_neighborhoods:
            explanations.append(
                "You both often go out in the same neighborhoods."
            )
        
        # Fallback
        if not explanations:
            explanations.append("You have similar interests and activity patterns.")
        
        return explanations[:3]

