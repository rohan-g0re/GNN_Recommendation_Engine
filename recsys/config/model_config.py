from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters for GNN model.
    MUST be consistent between training and serving.
    """
    # Embedding dimension
    D_MODEL: int = 128  # Output dimension of encoders and GNN layers
    
    # Encoder architecture
    CITY_EMBED_DIM: int = 16
    NEIGHBORHOOD_EMBED_DIM: int = 32
    PRICE_EMBED_DIM: int = 8
    TIME_SLOT_EMBED_DIM: int = 8
    
    # GNN backbone
    NUM_GNN_LAYERS: int = 2  # Number of message passing layers
    GNN_HIDDEN_DIM: int = 128  # Hidden dimension in conv layers
    GNN_DROPOUT: float = 0.1
    GNN_AGGR: str = 'mean'  # Aggregation: 'mean', 'sum', or 'attention'
    
    # Task heads
    PLACE_HEAD_HIDDEN: list = None  # Hidden layers for place head MLP
    FRIEND_HEAD_HIDDEN: list = None  # Hidden layers for friend head MLP
    HEAD_DROPOUT: float = 0.2
    
    # Context vector dimensions
    D_CTX_PLACE: int = 16  # Dimension of place context vector
    D_CTX_FRIEND: int = 16  # Dimension of friend context vector
    
    # Training
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    BATCH_SIZE_PLACE: int = 512
    BATCH_SIZE_FRIEND: int = 512
    
    # Loss weights
    LAMBDA_PLACE: float = 1.0
    LAMBDA_FRIEND: float = 0.5
    LAMBDA_ATTEND: float = 0.3
    
    def __post_init__(self):
        if self.PLACE_HEAD_HIDDEN is None:
            self.PLACE_HEAD_HIDDEN = [256, 128]
        if self.FRIEND_HEAD_HIDDEN is None:
            self.FRIEND_HEAD_HIDDEN = [256, 128]

