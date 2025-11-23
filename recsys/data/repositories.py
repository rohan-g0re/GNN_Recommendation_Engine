"""
Data repositories for loading parquet files.
"""

import pandas as pd
from typing import Iterator, List
from recsys.data.schemas import (
    UserSchema, PlaceSchema, InteractionSchema,
    UserUserEdgeSchema, FriendLabelSchema
)
from datetime import datetime


class UserRepository:
    """Repository for loading users from parquet."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
        self._user_cache = {}  # Cache for individual lookups
    
    def _load(self):
        if self.df is None:
            try:
                self.df = pd.read_parquet(f"{self.data_dir}/users.parquet")
            except FileNotFoundError:
                # If file doesn't exist, create empty dataframe
                self.df = pd.DataFrame()
    
    def _row_to_user(self, row) -> UserSchema:
        """Convert a dataframe row to UserSchema."""
        return UserSchema(
            user_id=int(row['user_id']),
            home_city_id=int(row['home_city_id']),
            home_neighborhood_id=int(row['home_neighborhood_id']),
            cat_pref=row['cat_pref'] if isinstance(row['cat_pref'], list) else eval(row['cat_pref']),
            fine_pref=row['fine_pref'] if isinstance(row['fine_pref'], list) else eval(row['fine_pref']),
            vibe_pref=row['vibe_pref'] if isinstance(row['vibe_pref'], list) else eval(row['vibe_pref']),
            area_freqs=row['area_freqs'] if isinstance(row['area_freqs'], dict) else eval(str(row['area_freqs'])),
            avg_sessions_per_week=float(row['avg_sessions_per_week']),
            avg_views_per_session=float(row['avg_views_per_session']),
            avg_likes_per_session=float(row['avg_likes_per_session']),
            avg_saves_per_session=float(row['avg_saves_per_session']),
            avg_attends_per_month=float(row['avg_attends_per_month'])
        )
    
    def get_user(self, user_id: int) -> UserSchema:
        """Get a single user by ID."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        
        self._load()
        if self.df.empty:
            return None
        
        user_rows = self.df[self.df['user_id'] == user_id]
        if len(user_rows) == 0:
            return None
        
        user = self._row_to_user(user_rows.iloc[0])
        self._user_cache[user_id] = user
        return user
    
    def get_users_by_city(self, city_id: int) -> List[UserSchema]:
        """Get all users in a city."""
        self._load()
        if self.df.empty:
            return []
        
        city_users = self.df[self.df['home_city_id'] == city_id]
        return [self._row_to_user(row) for _, row in city_users.iterrows()]
    
    def get_all_users(self) -> Iterator[UserSchema]:
        """Load all users."""
        self._load()
        for _, row in self.df.iterrows():
            yield self._row_to_user(row)


class PlaceRepository:
    """Repository for loading places from parquet."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
        self._place_cache = {}  # Cache for individual lookups
    
    def _load(self):
        if self.df is None:
            try:
                self.df = pd.read_parquet(f"{self.data_dir}/places.parquet")
            except FileNotFoundError:
                # If file doesn't exist, create empty dataframe
                self.df = pd.DataFrame()
    
    def _row_to_place(self, row) -> PlaceSchema:
        """Convert a dataframe row to PlaceSchema."""
        return PlaceSchema(
            place_id=int(row['place_id']),
            city_id=int(row['city_id']),
            neighborhood_id=int(row['neighborhood_id']),
            category_ids=row['category_ids'] if isinstance(row['category_ids'], list) else eval(row['category_ids']),
            category_one_hot=row['category_one_hot'] if isinstance(row['category_one_hot'], list) else eval(row['category_one_hot']),
            fine_tag_vector=row['fine_tag_vector'] if isinstance(row['fine_tag_vector'], list) else eval(row['fine_tag_vector']),
            price_band=int(row['price_band']),
            typical_time_slot=int(row['typical_time_slot']),
            base_popularity=float(row['base_popularity']),
            avg_daily_visits=float(row['avg_daily_visits']),
            conversion_rate=float(row['conversion_rate']),
            novelty_score=float(row['novelty_score'])
        )
    
    def get_place(self, place_id: int) -> PlaceSchema:
        """Get a single place by ID."""
        if place_id in self._place_cache:
            return self._place_cache[place_id]
        
        self._load()
        if self.df.empty:
            return None
        
        place_rows = self.df[self.df['place_id'] == place_id]
        if len(place_rows) == 0:
            return None
        
        place = self._row_to_place(place_rows.iloc[0])
        self._place_cache[place_id] = place
        return place
    
    def get_places_by_city(self, city_id: int) -> List[PlaceSchema]:
        """Get all places in a city."""
        self._load()
        if self.df.empty:
            return []
        
        city_places = self.df[self.df['city_id'] == city_id]
        return [self._row_to_place(row) for _, row in city_places.iterrows()]
    
    def get_all_places(self) -> Iterator[PlaceSchema]:
        """Load all places."""
        self._load()
        for _, row in self.df.iterrows():
            yield self._row_to_place(row)


class InteractionRepository:
    """Repository for loading interactions from parquet."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
    
    def _load(self):
        if self.df is None:
            self.df = pd.read_parquet(f"{self.data_dir}/interactions.parquet")
    
    def get_all_interactions(self) -> Iterator[InteractionSchema]:
        """Load all interactions."""
        self._load()
        for _, row in self.df.iterrows():
            timestamp = row['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif pd.isna(timestamp):
                timestamp = datetime.now()
            
            yield InteractionSchema(
                user_id=int(row['user_id']),
                place_id=int(row['place_id']),
                dwell_time=float(row['dwell_time']),
                num_likes=int(row['num_likes']),
                num_saves=int(row['num_saves']),
                num_shares=int(row['num_shares']),
                attended=bool(row['attended']),
                implicit_rating=float(row['implicit_rating']),
                timestamp=timestamp,
                time_of_day_bucket=int(row['time_of_day_bucket']),
                day_of_week_bucket=int(row['day_of_week_bucket'])
            )


class UserUserEdgeRepository:
    """Repository for loading user-user edges from parquet."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
    
    def _load(self):
        if self.df is None:
            self.df = pd.read_parquet(f"{self.data_dir}/user_user_edges.parquet")
    
    def get_all_edges(self) -> Iterator[UserUserEdgeSchema]:
        """Load all edges."""
        self._load()
        for _, row in self.df.iterrows():
            yield UserUserEdgeSchema(
                user_u=int(row['user_u']),
                user_v=int(row['user_v']),
                interest_overlap_score=float(row['interest_overlap_score']),
                co_attendance_count=int(row['co_attendance_count']),
                same_neighborhood_freq=float(row['same_neighborhood_freq'])
            )


class FriendLabelRepository:
    """Repository for loading friend labels from parquet."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.df = None
    
    def _load(self):
        if self.df is None:
            self.df = pd.read_parquet(f"{self.data_dir}/friend_labels.parquet")
    
    def get_all_labels(self) -> Iterator[FriendLabelSchema]:
        """Load all labels."""
        self._load()
        for _, row in self.df.iterrows():
            yield FriendLabelSchema(
                user_u=int(row['user_u']),
                user_v=int(row['user_v']),
                label_compat=int(row['label_compat']),
                label_attend=int(row['label_attend'])
            )

