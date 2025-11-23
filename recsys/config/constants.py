# File: recsys/config/constants.py

# Scale parameters
N_USERS = 10_000
N_PLACES = 10_000
N_CITIES = 8
N_NEIGHBORHOODS_PER_CITY = 15

# Feature dimensions
C_COARSE = 6  # Number of coarse categories
C_FINE = 100  # Number of fine-grained tags
C_VIBE = 30   # Number of vibe/personality tags

# Coarse categories (index-based, 0-indexed)
COARSE_CATEGORIES = [
    "entertainment",  # 0
    "sports",         # 1
    "clubs",          # 2
    "dining",         # 3
    "outdoors",       # 4
    "culture"         # 5
]

# Fine tags (0-indexed, first 20 shown as example)
FINE_TAGS = [
    "fishing", "bouldering", "techno", "live_music", "board_games",
    "rooftop", "brunch", "karaoke", "jazz", "trivia",
    "wine_tasting", "comedy", "art_gallery", "theater", "dance",
    "pottery", "yoga", "crossfit", "rock_climbing", "cycling",
    # ... 80 more tags to total 100
] + [f"tag_{i}" for i in range(20, 100)]

# Vibe tags (0-indexed, first 15 shown)
VIBE_TAGS = [
    "introvert", "extrovert", "party", "fitness", "artsy",
    "night_owl", "early_bird", "foodie", "adventurous", "chill",
    "intellectual", "spontaneous", "planner", "social", "independent",
    # ... 15 more to total 30
] + [f"vibe_{i}" for i in range(15, 30)]

# Time slot buckets
TIME_SLOTS = ["morning", "afternoon", "evening", "night", "weekend_day", "weekend_night"]
N_TIME_SLOTS = len(TIME_SLOTS)

# Time of day buckets (for interactions)
TIME_OF_DAY_BUCKETS = ["morning", "afternoon", "evening", "night"]
N_TIME_OF_DAY = 4

# Day of week buckets
DAY_OF_WEEK_BUCKETS = ["weekday", "weekend"]
N_DAY_OF_WEEK = 2

# Price bands
N_PRICE_BANDS = 5  # 0-4

# Neighborhood representation
MAX_NEIGHBORHOODS_PER_USER = 5  # Fixed-size vector for area_freqs

# Computed feature dimensions (MUST MATCH INTEGRATION_CONTRACTS.md)
D_USER_RAW = 148  # 2 + 6 + 100 + 30 + 5 + 5
D_PLACE_RAW = 114  # 2 + 6 + 100 + 2 + 4
D_EDGE_UP = 12  # User-place edge features
D_EDGE_UU = 3   # User-user edge features
D_MODEL = 128  # Output dimension of encoders and GNN

