from recsys.data.schemas import UserSchema, PlaceSchema
from recsys.config.constants import (
    N_USERS, N_CITIES, N_NEIGHBORHOODS_PER_CITY,
    C_COARSE, C_FINE, C_VIBE, N_PRICE_BANDS, N_TIME_SLOTS
)


def validate_user_schema(user: UserSchema) -> None:
    """Validate user data integrity."""
    assert 0 <= user.user_id < N_USERS
    assert 0 <= user.home_city_id < N_CITIES
    assert 0 <= user.home_neighborhood_id < N_NEIGHBORHOODS_PER_CITY
    
    assert len(user.cat_pref) == C_COARSE
    assert abs(sum(user.cat_pref) - 1.0) < 1e-5, "cat_pref must sum to 1"
    
    assert len(user.fine_pref) == C_FINE
    assert abs(sum(user.fine_pref) - 1.0) < 1e-5, "fine_pref must sum to 1"
    
    assert len(user.vibe_pref) == C_VIBE
    assert abs(sum(user.vibe_pref) - 1.0) < 1e-5, "vibe_pref must sum to 1"
    
    assert all(v >= 0 for v in user.area_freqs.values())
    assert abs(sum(user.area_freqs.values()) - 1.0) < 1e-5
    
    assert user.avg_sessions_per_week > 0
    assert user.avg_views_per_session > 0


def validate_place_schema(place: PlaceSchema) -> None:
    """Validate place data integrity."""
    assert 0 <= place.place_id < N_PLACES
    assert 0 <= place.city_id < N_CITIES
    assert 0 <= place.neighborhood_id < N_NEIGHBORHOODS_PER_CITY
    
    assert len(place.category_one_hot) == C_COARSE
    assert all(0 <= v <= 1 for v in place.category_one_hot)
    
    assert len(place.fine_tag_vector) == C_FINE
    assert abs(sum(place.fine_tag_vector) - 1.0) < 1e-5
    
    assert 0 <= place.price_band < N_PRICE_BANDS
    assert 0 <= place.typical_time_slot < N_TIME_SLOTS

