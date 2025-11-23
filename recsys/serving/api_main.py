from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

from recsys.serving.api_schemas import (
    PlaceRecommendationRequest,
    PlaceRecommendationResponse,
    PlaceRecommendation,
    PeopleRecommendationRequest,
    PeopleRecommendationResponse,
    PeopleRecommendation
)
from recsys.serving.recommender_core import PlaceRecommender, PeopleRecommender, EmbeddingStore
from recsys.serving.ann_index import CityAnnIndexManager
from recsys.serving.explanations import ExplanationService
from recsys.ml.models.heads import PlaceHead, FriendHead, ContextEncoder
from recsys.config.model_config import ModelConfig
from recsys.config.constants import N_CITIES

from recsys.data.repositories import UserRepository, PlaceRepository

# Initialize FastAPI app
app = FastAPI(
    title="Social Outing Recommender API",
    description="GNN-powered place and people recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialized on startup)
place_recommender = None
people_recommender = None


@app.on_event("startup")
async def startup_event():
    """
    Load models, embeddings, and indices on startup.
    """
    global place_recommender, people_recommender
    
    print("Loading configuration...")
    config = ModelConfig()
    
    # Load embeddings
    print("Loading embeddings...")
    embedding_store = EmbeddingStore()
    embedding_store.load_from_parquet(
        user_path="data/embeddings/user_embeddings.parquet",
        place_path="data/embeddings/place_embeddings.parquet"
    )
    
    # Load ANN indices
    print("Loading ANN indices...")
    place_ann = CityAnnIndexManager(dimension=config.D_MODEL)
    user_ann = CityAnnIndexManager(dimension=config.D_MODEL)
    
    city_ids = list(range(N_CITIES))
    place_ann.load("data/indices", "place", city_ids)
    user_ann.load("data/indices", "user", city_ids)
    
    # Load trained model heads
    print("Loading model heads...")
    checkpoint = torch.load("data/models/final_model.pt", map_location='cpu')
    
    place_head = PlaceHead(config)
    friend_head = FriendHead(config)
    place_ctx_encoder = ContextEncoder(config.D_CTX_PLACE)
    friend_ctx_encoder = ContextEncoder(config.D_CTX_FRIEND)
    
    place_head.load_state_dict(checkpoint['place_head'])
    friend_head.load_state_dict(checkpoint['friend_head'])
    place_ctx_encoder.load_state_dict(checkpoint['place_ctx_encoder'])
    friend_ctx_encoder.load_state_dict(checkpoint['friend_ctx_encoder'])
    
    place_head.eval()
    friend_head.eval()
    
    # Initialize repositories
    user_repo = UserRepository("data")
    place_repo = PlaceRepository("data")
    
    # Initialize explanation service
    explanation_service = ExplanationService()
    
    # Initialize recommenders
    place_recommender = PlaceRecommender(
        embedding_store=embedding_store,
        place_ann_manager=place_ann,
        place_head=place_head,
        ctx_encoder=place_ctx_encoder,
        user_repo=user_repo,
        place_repo=place_repo,
        explanation_service=explanation_service,
        config=config
    )
    
    people_recommender = PeopleRecommender(
        embedding_store=embedding_store,
        user_ann_manager=user_ann,
        friend_head=friend_head,
        ctx_encoder=friend_ctx_encoder,
        user_repo=user_repo,
        explanation_service=explanation_service,
        config=config
    )
    
    print("Server ready!")


@app.get("/")
async def root():
    return {"message": "Social Outing Recommender API", "status": "online"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/recommend/places", response_model=PlaceRecommendationResponse)
async def recommend_places(request: PlaceRecommendationRequest):
    """
    Get place recommendations for a user.
    
    Example request:
    ```
    {
      "user_id": 42,
      "city_id": 2,
      "time_slot": 3,
      "desired_categories": [0, 2],
      "top_k": 10
    }
    ```
    """
    if place_recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = place_recommender.recommend(
            user_id=request.user_id,
            city_id=request.city_id,
            time_slot=request.time_slot,
            desired_categories=request.desired_categories,
            top_k=request.top_k
        )
        
        recommendations = [
            PlaceRecommendation(**result) for result in results
        ]
        
        return PlaceRecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/people", response_model=PeopleRecommendationResponse)
async def recommend_people(request: PeopleRecommendationRequest):
    """
    Get people recommendations for a user.
    
    Example request:
    ```
    {
      "user_id": 42,
      "city_id": 2,
      "target_place_id": 1234,
      "top_k": 10
    }
    ```
    """
    if people_recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = people_recommender.recommend(
            user_id=request.user_id,
            city_id=request.city_id,
            target_place_id=request.target_place_id,
            activity_tags=request.activity_tags,
            top_k=request.top_k
        )
        
        recommendations = [
            PeopleRecommendation(**result) for result in results
        ]
        
        return PeopleRecommendationResponse(recommendations=recommendations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "recsys.serving.api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

