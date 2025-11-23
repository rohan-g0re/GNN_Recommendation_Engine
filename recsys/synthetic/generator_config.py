"""
Configuration for synthetic data generation.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    # Scale
    N_USERS: int = 10_000
    N_PLACES: int = 10_000
    N_CITIES: int = 8
    N_NEIGHBORHOODS_PER_CITY: int = 15
    
    # Taxonomy dimensions (must match constants.py)
    C_COARSE: int = 6
    C_FINE: int = 100
    C_VIBE: int = 30
    
    # Category-to-tag mapping (which fine tags belong to which coarse categories)
    CATEGORY_TO_TAGS: Dict[int, List[int]] = None
    
    # Distribution parameters
    DIRICHLET_ALPHA_CAT: float = 2.0  # For category preferences
    DIRICHLET_ALPHA_FINE: float = 1.0  # For fine-tag preferences
    DIRICHLET_ALPHA_VIBE: float = 1.5  # For vibe preferences
    DIRICHLET_ALPHA_AREA: float = 3.0  # For area frequencies
    
    # City distribution (weights for sampling cities, sums to 1.0)
    CITY_WEIGHTS: List[float] = None
    
    # User behavior ranges
    SESSIONS_PER_WEEK_RANGE: tuple = (1.0, 8.0)
    VIEWS_PER_SESSION_RANGE: tuple = (10.0, 80.0)
    LIKES_PER_SESSION_RANGE: tuple = (0.5, 5.0)
    SAVES_PER_SESSION_RANGE: tuple = (0.2, 3.0)
    ATTENDS_PER_MONTH_RANGE: tuple = (1.0, 12.0)
    
    # Interaction parameters
    TIME_HORIZON_WEEKS: int = 12  # Simulate 12 weeks of data
    INTERACTIONS_PER_USER_RANGE: tuple = (50, 500)
    
    # Preference scoring weights
    W_INTEREST: float = 0.4
    W_CATEGORY: float = 0.3
    W_LOCATION: float = 0.2
    W_POPULARITY: float = 0.1
    
    # Social edge parameters
    SIMILARITY_THRESHOLD: float = 0.3  # Min cosine similarity for social edge
    K_SIMILAR_USERS: int = 50  # Find top-K similar users per user
    CO_ATTENDANCE_PROB: float = 0.1  # Probability of co-attendance for similar pairs
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    def __post_init__(self):
        """Initialize derived parameters."""
        if self.CITY_WEIGHTS is None:
            # Skewed distribution: some cities larger than others
            rng = np.random.default_rng(self.RANDOM_SEED)
            weights = rng.dirichlet([2.0] * self.N_CITIES)
            self.CITY_WEIGHTS = weights.tolist()
        
        if self.CATEGORY_TO_TAGS is None:
            # Define which fine tags belong to which coarse categories
            tags_per_category = self.C_FINE // self.C_COARSE
            self.CATEGORY_TO_TAGS = {}
            for cat_idx in range(self.C_COARSE):
                start = cat_idx * tags_per_category
                end = start + tags_per_category
                if cat_idx == self.C_COARSE - 1:  # Last category gets remaining tags
                    end = self.C_FINE
                self.CATEGORY_TO_TAGS[cat_idx] = list(range(start, end))


def get_default_config() -> SyntheticConfig:
    """Get default configuration."""
    return SyntheticConfig()

