"""
Mock settings for tests
"""

class Config:
    def __init__(self):
        self.model_path = "test_model.pt"
        self.user_emb = "test_user_emb.pt"
        self.item_emb = "test_item_emb.pt"
        self.faiss_index = "test_faiss.index"
        self.movie_ids = "test_movie_ids.npy"
        self.redis_url = "redis://localhost:6379/0"

cfg = Config()
