"""
Generate synthetic user-place interactions.
"""

import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from recsys.data.schemas import UserSchema, PlaceSchema, InteractionSchema, compute_implicit_rating
from recsys.synthetic.generator_config import SyntheticConfig


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_preference_score(
    user: UserSchema,
    place: PlaceSchema,
    config: SyntheticConfig
) -> float:
    """
    Compute preference score for user-place pair.
    
    Formula:
    score = w1 * sim_interest + w2 * sim_cat + w3 * loc_factor + w4 * popularity + noise
    """
    # Interest similarity (cosine of fine prefs)
    sim_interest = cosine_similarity(
        np.array(user.fine_pref),
        np.array(place.fine_tag_vector)
    )
    
    # Category alignment (dot product)
    sim_cat = np.dot(
        np.array(user.cat_pref),
        np.array(place.category_one_hot)
    )
    
    # Location factor (higher if place in frequent neighborhoods)
    loc_factor = user.area_freqs.get(place.neighborhood_id, 0.0)
    
    # Popularity factor (log-normalize)
    popularity = np.log1p(place.base_popularity) / 10.0
    
    # Combine with weights
    score = (
        config.W_INTEREST * sim_interest +
        config.W_CATEGORY * sim_cat +
        config.W_LOCATION * loc_factor +
        config.W_POPULARITY * popularity
    )
    
    return max(0.0, score)  # Ensure non-negative


def sample_interactions_for_user(
    user: UserSchema,
    places: List[PlaceSchema],
    config: SyntheticConfig,
    rng: np.random.Generator,
    start_date: datetime
) -> List[InteractionSchema]:
    """
    Sample interactions for a single user.
    """
    # 1. Determine number of interactions
    n_interactions_raw = int(
        user.avg_sessions_per_week *
        config.TIME_HORIZON_WEEKS *
        user.avg_views_per_session / 10.0  # Scale down
    )
    n_interactions = np.clip(
        n_interactions_raw,
        *config.INTERACTIONS_PER_USER_RANGE
    )
    
    # 2. Get candidate places (same city, occasionally others)
    same_city_places = [p for p in places if p.city_id == user.home_city_id]
    
    if len(same_city_places) < 20:
        # If not enough places in city, use all places
        candidate_places = places
    else:
        # 90% same city, 10% any city (travel)
        if rng.random() < 0.9:
            candidate_places = same_city_places
        else:
            candidate_places = places
    
    if len(candidate_places) == 0:
        return []
    
    # 3. Compute preference scores for all candidates
    scores = np.array([
        compute_preference_score(user, place, config)
        for place in candidate_places
    ])
    
    # Add noise
    scores = scores + rng.normal(0, 0.1, size=len(scores))
    scores = np.maximum(scores, 0.0)
    
    # 4. Convert to probabilities via softmax
    scores_exp = np.exp(scores - scores.max())  # Numerical stability
    probs = scores_exp / scores_exp.sum()
    
    # 5. Sample places according to probabilities (with replacement)
    sampled_indices = rng.choice(
        len(candidate_places),
        size=n_interactions,
        replace=True,
        p=probs
    )
    
    # 6. Generate interaction details for each sampled place
    interactions = []
    
    for idx in sampled_indices:
        place = candidate_places[idx]
        score = scores[idx]
        
        # Dwell time (higher score → more time)
        base_dwell = 30.0  # seconds
        dwell_time = base_dwell + score * 200.0 + rng.exponential(50.0)
        dwell_time = max(5.0, min(600.0, dwell_time))  # Cap at 10 minutes
        
        # Actions (higher score → more actions)
        action_prob = min(0.95, score * 2.0)
        
        num_likes = int(rng.random() < action_prob * 0.7)
        num_saves = int(rng.random() < action_prob * 0.4)
        num_shares = int(rng.random() < action_prob * 0.2)
        attended = bool(rng.random() < action_prob * 0.1)  # Attending is rare
        
        # Implicit rating
        implicit_rating = compute_implicit_rating(
            dwell_time, num_likes, num_saves, num_shares, attended
        )
        
        # Timestamp (random within time horizon)
        days_offset = rng.integers(0, config.TIME_HORIZON_WEEKS * 7)
        hours_offset = rng.integers(0, 24)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)
        
        # Time buckets
        hour = timestamp.hour
        if hour < 6:
            time_of_day_bucket = 3  # night
        elif hour < 12:
            time_of_day_bucket = 0  # morning
        elif hour < 18:
            time_of_day_bucket = 1  # afternoon
        else:
            time_of_day_bucket = 2  # evening
        
        day_of_week_bucket = 0 if timestamp.weekday() < 5 else 1  # weekday/weekend
        
        interaction = InteractionSchema(
            user_id=user.user_id,
            place_id=place.place_id,
            dwell_time=float(dwell_time),
            num_likes=num_likes,
            num_saves=num_saves,
            num_shares=num_shares,
            attended=attended,
            implicit_rating=float(implicit_rating),
            timestamp=timestamp,
            time_of_day_bucket=time_of_day_bucket,
            day_of_week_bucket=day_of_week_bucket
        )
        
        interactions.append(interaction)
    
    return interactions


def generate_all_interactions(
    users: List[UserSchema],
    places: List[PlaceSchema],
    config: SyntheticConfig,
    start_date: datetime = None
) -> List[InteractionSchema]:
    """
    Generate all user-place interactions.
    
    Args:
        users: List of users
        places: List of places
        config: Generator configuration
        start_date: Start date for timestamps
    
    Returns:
        List of InteractionSchema
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    rng = np.random.default_rng(config.RANDOM_SEED + 2)
    
    all_interactions = []
    print(f"Generating interactions for {len(users)} users...")
    
    for i, user in enumerate(users):
        if (i + 1) % 1000 == 0:
            print(f"  Generated interactions for {i + 1}/{len(users)} users")
        
        user_interactions = sample_interactions_for_user(
            user, places, config, rng, start_date
        )
        all_interactions.extend(user_interactions)
    
    print(f"✅ Generated {len(all_interactions)} interactions")
    return all_interactions

