"""
Generate synthetic users.
"""

import numpy as np
from typing import List, Dict
from recsys.data.schemas import UserSchema
from recsys.synthetic.generator_config import SyntheticConfig


def generate_user(user_id: int, config: SyntheticConfig, rng: np.random.Generator) -> UserSchema:
    """
    Generate a single synthetic user.
    
    Args:
        user_id: User ID (0 to N_USERS-1)
        config: Generator configuration
        rng: Random number generator
    
    Returns:
        UserSchema
    """
    # 1. Home location (sample city according to weights)
    home_city_id = rng.choice(config.N_CITIES, p=config.CITY_WEIGHTS)
    home_neighborhood_id = rng.integers(0, config.N_NEIGHBORHOODS_PER_CITY)
    
    # 2. Interest over coarse categories (Dirichlet for diversity)
    cat_pref_raw = rng.dirichlet([config.DIRICHLET_ALPHA_CAT] * config.C_COARSE)
    cat_pref = cat_pref_raw / cat_pref_raw.sum()  # Ensure normalization
    
    # 3. Fine-tag interest (bias towards tags in preferred categories)
    # Build biased alpha for fine tags
    alpha_fine = np.ones(config.C_FINE) * 0.1  # Base small value
    
    # Boost tags in preferred categories
    for cat_idx, cat_weight in enumerate(cat_pref):
        if cat_weight > 0.1:  # Only boost significant categories
            for tag_idx in config.CATEGORY_TO_TAGS[cat_idx]:
                alpha_fine[tag_idx] += cat_weight * config.DIRICHLET_ALPHA_FINE * 5
    
    fine_pref_raw = rng.dirichlet(alpha_fine)
    fine_pref = fine_pref_raw / fine_pref_raw.sum()
    
    # 4. Vibe / personality profile (choose 3-10 core traits)
    num_vibe_tags = rng.integers(3, 11)
    selected_vibe_indices = rng.choice(config.C_VIBE, size=num_vibe_tags, replace=False)
    
    vibe_weights = rng.exponential(scale=1.0, size=num_vibe_tags)
    vibe_pref = np.zeros(config.C_VIBE)
    vibe_pref[selected_vibe_indices] = vibe_weights
    vibe_pref = vibe_pref / vibe_pref.sum()
    
    # 5. Behavior statistics (sample from configured ranges)
    avg_sessions_per_week = float(rng.uniform(*config.SESSIONS_PER_WEEK_RANGE))
    avg_views_per_session = float(rng.uniform(*config.VIEWS_PER_SESSION_RANGE))
    avg_likes_per_session = float(rng.uniform(*config.LIKES_PER_SESSION_RANGE))
    avg_saves_per_session = float(rng.uniform(*config.SAVES_PER_SESSION_RANGE))
    avg_attends_per_month = float(rng.uniform(*config.ATTENDS_PER_MONTH_RANGE))
    
    # 6. Location behavior (concentrate on 1-3 neighborhoods)
    num_frequent_neighborhoods = rng.integers(1, 4)
    frequent_neighborhoods = rng.choice(
        config.N_NEIGHBORHOODS_PER_CITY,
        size=num_frequent_neighborhoods,
        replace=False
    )
    
    # Dirichlet over frequent neighborhoods
    area_weights = rng.dirichlet([config.DIRICHLET_ALPHA_AREA] * num_frequent_neighborhoods)
    area_freqs: Dict[int, float] = {}
    for neigh_id, weight in zip(frequent_neighborhoods, area_weights):
        area_freqs[int(neigh_id)] = float(weight)
    
    return UserSchema(
        user_id=user_id,
        home_city_id=int(home_city_id),
        home_neighborhood_id=int(home_neighborhood_id),
        cat_pref=cat_pref.tolist(),
        fine_pref=fine_pref.tolist(),
        vibe_pref=vibe_pref.tolist(),
        area_freqs=area_freqs,
        avg_sessions_per_week=avg_sessions_per_week,
        avg_views_per_session=avg_views_per_session,
        avg_likes_per_session=avg_likes_per_session,
        avg_saves_per_session=avg_saves_per_session,
        avg_attends_per_month=avg_attends_per_month
    )


def generate_all_users(config: SyntheticConfig) -> List[UserSchema]:
    """
    Generate all synthetic users.
    
    Args:
        config: Generator configuration
    
    Returns:
        List of UserSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 1)  # Different seed from places
    
    users = []
    print(f"Generating {config.N_USERS} users...")
    
    for user_id in range(config.N_USERS):
        if (user_id + 1) % 1000 == 0:
            print(f"  Generated {user_id + 1}/{config.N_USERS} users")
        
        user = generate_user(user_id, config, rng)
        users.append(user)
    
    print(f"✅ Generated {len(users)} users")
    return users

