"""
Generate synthetic user-user edges and friend labels.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from recsys.data.schemas import UserSchema, InteractionSchema, UserUserEdgeSchema, FriendLabelSchema
from recsys.synthetic.generator_config import SyntheticConfig


def cosine_similarity_users(user_a: UserSchema, user_b: UserSchema) -> float:
    """Compute cosine similarity between two users based on fine preferences."""
    fine_a = np.array(user_a.fine_pref)
    fine_b = np.array(user_b.fine_pref)
    
    dot = np.dot(fine_a, fine_b)
    norm_a = np.linalg.norm(fine_a)
    norm_b = np.linalg.norm(fine_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))


def compute_neighborhood_overlap(user_a: UserSchema, user_b: UserSchema) -> float:
    """
    Compute overlap in area_freqs (Jaccard-like similarity).
    """
    neighborhoods_a = set(user_a.area_freqs.keys())
    neighborhoods_b = set(user_b.area_freqs.keys())
    
    if len(neighborhoods_a) == 0 or len(neighborhoods_b) == 0:
        return 0.0
    
    intersection = neighborhoods_a & neighborhoods_b
    union = neighborhoods_a | neighborhoods_b
    
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def build_co_attendance_map(interactions: List[InteractionSchema]) -> Dict[Tuple[int, int], int]:
    """
    Build map of co-attendance counts.
    
    Returns:
        Dict[(user_u, user_v)] -> count (where user_u < user_v)
    """
    # Group interactions by place and time window (day)
    place_day_users: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    
    for interaction in interactions:
        if not interaction.attended:
            continue
        
        place_id = interaction.place_id
        day = interaction.timestamp.date()
        place_day_users[(place_id, day)].add(interaction.user_id)
    
    # Count co-attendances
    co_attendance: Dict[Tuple[int, int], int] = defaultdict(int)
    
    for users_at_place in place_day_users.values():
        users_list = list(users_at_place)
        # Create pairs
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                user_u = min(users_list[i], users_list[j])
                user_v = max(users_list[i], users_list[j])
                co_attendance[(user_u, user_v)] += 1
    
    return dict(co_attendance)


def generate_social_edges(
    users: List[UserSchema],
    interactions: List[InteractionSchema],
    config: SyntheticConfig
) -> List[UserUserEdgeSchema]:
    """
    Generate user-user social edges.
    
    Args:
        users: List of users
        interactions: List of interactions (for co-attendance)
        config: Generator configuration
    
    Returns:
        List of UserUserEdgeSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 3)
    
    print("Building co-attendance map...")
    co_attendance_map = build_co_attendance_map(interactions)
    
    print(f"Finding similar users for {len(users)} users...")
    
    # Build user index for fast lookup
    user_dict = {user.user_id: user for user in users}
    
    edges = []
    processed_pairs = set()
    
    for i, user_a in enumerate(users):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(users)} users")
        
        # Find similar users in same city
        candidates = [
            user_b for user_b in users
            if user_b.user_id != user_a.user_id and
            user_b.home_city_id == user_a.home_city_id
        ]
        
        if len(candidates) == 0:
            continue
        
        # Compute similarities
        similarities = [
            (user_b, cosine_similarity_users(user_a, user_b))
            for user_b in candidates
        ]
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        
        # Take top-K similar users
        top_similar = similarities[:config.K_SIMILAR_USERS]
        
        for user_b, sim_score in top_similar:
            if sim_score < config.SIMILARITY_THRESHOLD:
                continue
            
            # Ensure consistent ordering (smaller ID first)
            user_u_id = min(user_a.user_id, user_b.user_id)
            user_v_id = max(user_a.user_id, user_b.user_id)
            
            # Skip if already processed
            if (user_u_id, user_v_id) in processed_pairs:
                continue
            
            processed_pairs.add((user_u_id, user_v_id))
            
            user_u = user_dict[user_u_id]
            user_v = user_dict[user_v_id]
            
            # Compute edge features
            interest_overlap_score = sim_score
            co_attendance_count = co_attendance_map.get((user_u_id, user_v_id), 0)
            same_neighborhood_freq = compute_neighborhood_overlap(user_u, user_v)
            
            edge = UserUserEdgeSchema(
                user_u=user_u_id,
                user_v=user_v_id,
                interest_overlap_score=float(interest_overlap_score),
                co_attendance_count=int(co_attendance_count),
                same_neighborhood_freq=float(same_neighborhood_freq)
            )
            
            edges.append(edge)
    
    print(f"✅ Generated {len(edges)} social edges")
    return edges


def generate_friend_labels(
    edges: List[UserUserEdgeSchema],
    users: List[UserSchema],
    config: SyntheticConfig
) -> List[FriendLabelSchema]:
    """
    Generate friend compatibility labels from edges.
    
    Args:
        edges: Social edges
        users: List of users
        config: Generator configuration
    
    Returns:
        List of FriendLabelSchema
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 4)
    
    print("Generating friend labels...")
    
    labels = []
    
    # Positive labels from edges
    for edge in edges:
        # Compatibility: high similarity or co-attendance
        is_compatible = (
            edge.interest_overlap_score >= 0.5 or
            edge.co_attendance_count >= 2
        )
        
        label_compat = 1 if is_compatible else 0
        
        # Attendance: probability based on similarity and co-attendance
        attend_prob = min(
            0.9,
            edge.interest_overlap_score * 0.5 +
            min(edge.co_attendance_count / 10.0, 0.4) +
            edge.same_neighborhood_freq * 0.1
        )
        
        label_attend = 1 if rng.random() < attend_prob else 0
        
        labels.append(FriendLabelSchema(
            user_u=edge.user_u,
            user_v=edge.user_v,
            label_compat=label_compat,
            label_attend=label_attend
        ))
    
    # Negative labels (sample random pairs with no edge)
    # Sample ~20% of positive labels as negatives
    num_negatives = len(labels) // 5
    
    user_ids = [u.user_id for u in users]
    edge_set = {(edge.user_u, edge.user_v) for edge in edges}
    
    attempts = 0
    max_attempts = num_negatives * 10
    
    while len(labels) - len(edges) < num_negatives and attempts < max_attempts:
        attempts += 1
        
        # Sample two random users
        user_u_id, user_v_id = rng.choice(user_ids, size=2, replace=False)
        user_u_id, user_v_id = min(user_u_id, user_v_id), max(user_u_id, user_v_id)
        
        # Skip if edge exists
        if (user_u_id, user_v_id) in edge_set:
            continue
        
        # Skip if already added as negative
        if any(l.user_u == user_u_id and l.user_v == user_v_id for l in labels[len(edges):]):
            continue
        
        # Negative label
        labels.append(FriendLabelSchema(
            user_u=user_u_id,
            user_v=user_v_id,
            label_compat=0,
            label_attend=0
        ))
    
    print(f"✅ Generated {len(labels)} friend labels ({len(edges)} positive, {len(labels) - len(edges)} negative)")
    return labels

