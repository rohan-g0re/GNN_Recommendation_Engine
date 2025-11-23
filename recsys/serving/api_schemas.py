from pydantic import BaseModel
from typing import List, Optional


class PlaceRecommendationRequest(BaseModel):
    """Request for place recommendations."""
    user_id: int
    city_id: Optional[int] = None
    time_slot: Optional[int] = None  # 0-5
    desired_categories: Optional[List[int]] = None  # Indices into C_COARSE
    top_k: int = 10


class PlaceRecommendation(BaseModel):
    """Single place recommendation."""
    place_id: int
    score: float
    explanations: List[str]


class PlaceRecommendationResponse(BaseModel):
    """Response with place recommendations."""
    recommendations: List[PlaceRecommendation]


class PeopleRecommendationRequest(BaseModel):
    """Request for people recommendations."""
    user_id: int
    city_id: Optional[int] = None
    target_place_id: Optional[int] = None
    activity_tags: Optional[List[int]] = None
    top_k: int = 10


class PeopleRecommendation(BaseModel):
    """Single person recommendation."""
    user_id: int
    compat_score: float
    attend_prob: float
    combined_score: float
    explanations: List[str]


class PeopleRecommendationResponse(BaseModel):
    """Response with people recommendations."""
    recommendations: List[PeopleRecommendation]

