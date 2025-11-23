"""
Generate synthetic places.
"""

import numpy as np
from typing import List
from recsys.data.schemas import PlaceSchema
from recsys.synthetic.generator_config import SyntheticConfig


def generate_place(place_id: int, config: SyntheticConfig, rng: np.random.Generator) -> PlaceSchema:
    """
    Generate a single synthetic place.
    
    Args:
        place_id: Place ID (0 to N_PLACES-1)
        config: Generator configuration
        rng: Random number generator
    
    Returns:
        PlaceSchema
    """
    # 1. Location
    city_id = rng.choice(config.N_CITIES, p=config.CITY_WEIGHTS)
    neighborhood_id = rng.integers(0, config.N_NEIGHBORHOODS_PER_CITY)
    
    # 2. Categories (sample 1-2 coarse categories)
    num_categories = rng.choice([1, 2], p=[0.7, 0.3])  # 70% single category
    category_ids = rng.choice(
        config.C_COARSE,
        size=num_categories,
        replace=False
    ).tolist()
    
    # Create multi-hot encoding
    category_one_hot = [0.0] * config.C_COARSE
    for cat_id in category_ids:
        category_one_hot[cat_id] = 1.0 / num_categories  # Equal weight if multi-category
    
    # 3. Fine tags (sample from tags associated with selected categories)
    candidate_tags = []
    for cat_id in category_ids:
        candidate_tags.extend(config.CATEGORY_TO_TAGS[cat_id])
    
    # Sample 3-7 fine tags
    num_tags = rng.integers(3, 8)
    if len(candidate_tags) < num_tags:
        # If not enough candidate tags, add random tags
        other_tags = [t for t in range(config.C_FINE) if t not in candidate_tags]
        candidate_tags.extend(rng.choice(other_tags, size=num_tags - len(candidate_tags), replace=False))
    
    selected_tags = rng.choice(candidate_tags, size=min(num_tags, len(candidate_tags)), replace=False)
    
    # Assign weights and normalize
    tag_weights = rng.exponential(scale=1.0, size=len(selected_tags))
    fine_tag_vector = np.zeros(config.C_FINE)
    fine_tag_vector[selected_tags] = tag_weights
    fine_tag_vector = fine_tag_vector / fine_tag_vector.sum()  # Normalize
    
    # 4. Operational attributes
    # Price band: higher in certain cities (simulate expensive cities)
    expensive_cities = [0, 1]  # First two cities are expensive
    if city_id in expensive_cities:
        price_band = rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
    else:
        price_band = rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
    
    # Typical time slot (0-5)
    typical_time_slot = rng.integers(0, 6)
    
    # 5. Popularity (log-normal distribution)
    base_popularity = float(rng.lognormal(mean=1.0, sigma=1.0))
    
    # Derive related metrics with noise
    avg_daily_visits = base_popularity * rng.uniform(0.8, 1.2) * 10
    conversion_rate = min(0.95, max(0.05, rng.beta(2, 5) * (base_popularity / 10)))
    novelty_score = 1.0 / (1.0 + np.log1p(base_popularity))
    
    return PlaceSchema(
        place_id=place_id,
        city_id=int(city_id),
        neighborhood_id=int(neighborhood_id),
        category_ids=category_ids,
        category_one_hot=category_one_hot,
        fine_tag_vector=fine_tag_vector.tolist(),
        price_band=int(price_band),
        typical_time_slot=int(typical_time_slot),
        base_popularity=float(base_popularity),
        avg_daily_visits=float(avg_daily_visits),
        conversion_rate=float(conversion_rate),
        novelty_score=float(novelty_score)
    )


def generate_all_places(config: SyntheticConfig) -> List[PlaceSchema]:
    """
    Generate all synthetic places.
    
    Args:
        config: Generator configuration
    
    Returns:
        List of PlaceSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    places = []
    print(f"Generating {config.N_PLACES} places...")
    
    for place_id in range(config.N_PLACES):
        if (place_id + 1) % 1000 == 0:
            print(f"  Generated {place_id + 1}/{config.N_PLACES} places")
        
        place = generate_place(place_id, config, rng)
        places.append(place)
    
    print(f"✅ Generated {len(places)} places")
    return places

