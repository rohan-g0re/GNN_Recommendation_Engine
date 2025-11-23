from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


def compute_implicit_rating(
    dwell_time: float,
    num_likes: int,
    num_saves: int,
    num_shares: int,
    attended: bool
) -> float:
    """
    Maps engagement signals to 1-5 rating scale.
    
    Logic:
    - Base score from normalized dwell time (0-2 points)
    - +0.5 per like (up to 1.5)
    - +1.0 per save (up to 2.0)
    - +0.5 per share (up to 1.0)
    - +2.0 if attended
    
    Capped at 5.0, floored at 1.0
    """
    score = 1.0
    
    # Dwell contribution (normalize assuming max ~300 seconds = 5 min)
    score += min(dwell_time / 150.0, 2.0)
    
    # Action contributions
    score += min(num_likes * 0.5, 1.5)
    score += min(num_saves * 1.0, 2.0)
    score += min(num_shares * 0.5, 1.0)
    
    if attended:
        score += 2.0
    
    return min(score, 5.0)


@dataclass
class UserSchema:
    """
    Raw user data schema.
    Maps 1:1 to storage (CSV/Parquet/DB).
    """
    # Identity
    user_id: int  # Primary key, 0 to N_USERS-1
    
    # Location attributes (categorical IDs)
    home_city_id: int  # 0 to N_CITIES-1
    home_neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1 within city
    
    # Preference vectors (pre-computed, normalized)
    cat_pref: List[float]  # Length C_COARSE=6, sums to 1.0
    fine_pref: List[float]  # Length C_FINE=100, sums to 1.0
    vibe_pref: List[float]  # Length C_VIBE=30, sums to 1.0
    
    # Location behavior (sparse or dense)
    area_freqs: Dict[int, float]  # Keys: neighborhood_ids, values sum to 1.0
    
    # Behavioral statistics (continuous)
    avg_sessions_per_week: float  # e.g., 2.5
    avg_views_per_session: float  # e.g., 25.0
    avg_likes_per_session: float  # e.g., 3.0
    avg_saves_per_session: float  # e.g., 1.0
    avg_attends_per_month: float  # e.g., 4.0
    
    # Optional demographics (for future extension)
    age_group: Optional[int] = None  # e.g., 0-5 for age buckets
    gender: Optional[int] = None  # e.g., 0/1/2


@dataclass
class PlaceSchema:
    """
    Raw place data schema.
    """
    # Identity
    place_id: int  # Primary key, 0 to N_PLACES-1
    
    # Location
    city_id: int  # 0 to N_CITIES-1
    neighborhood_id: int  # 0 to N_NEIGHBORHOODS_PER_CITY-1
    
    # Category (can be multi-label)
    category_ids: List[int]  # List of category indices, e.g., [0, 2] for entertainment+clubs
    category_one_hot: List[float]  # Length C_COARSE=6, binary or weighted
    
    # Fine-grained tags
    fine_tag_vector: List[float]  # Length C_FINE=100, normalized weights
    
    # Operational attributes
    price_band: int  # 0-4
    typical_time_slot: int  # Index into TIME_SLOTS (0-5)
    
    # Popularity metrics
    base_popularity: float  # Sampled from log-normal, e.g., 0.1 to 10.0
    avg_daily_visits: float  # Derived from base_popularity
    conversion_rate: float  # Fraction of views that lead to attendance
    novelty_score: float  # 1.0 / (1 + log(base_popularity))


@dataclass
class InteractionSchema:
    """
    User-place interaction event.
    Represents one view/engagement session.
    """
    # Foreign keys
    user_id: int  # References UserSchema.user_id
    place_id: int  # References PlaceSchema.place_id
    
    # Engagement metrics
    dwell_time: float  # Seconds spent viewing posts about this place
    num_likes: int
    num_saves: int
    num_shares: int
    attended: bool  # True if user attended event at this place
    
    # Derived implicit rating
    implicit_rating: float  # 1.0-5.0, derived from above metrics
    
    # Temporal context
    timestamp: datetime  # When interaction occurred
    time_of_day_bucket: int  # 0-3 (morning/afternoon/evening/night)
    day_of_week_bucket: int  # 0-1 (weekday/weekend)


@dataclass
class UserUserEdgeSchema:
    """
    Soft social edge between two users.
    Stored as undirected (single row per pair, user_u < user_v).
    """
    user_u: int  # First user ID (smaller ID)
    user_v: int  # Second user ID (larger ID)
    
    # Edge strength features
    interest_overlap_score: float  # Cosine similarity of fine_pref, [0, 1]
    co_attendance_count: int  # Number of places both attended
    same_neighborhood_freq: float  # Jaccard or overlap of area_freqs, [0, 1]


@dataclass
class FriendLabelSchema:
    """
    Labeled user pairs for training friend compatibility head.
    """
    user_u: int
    user_v: int
    
    # Binary labels
    label_compat: int  # 0 (negative) or 1 (positive) for compatibility
    label_attend: int  # 0 (won't attend) or 1 (will attend) if invited together

