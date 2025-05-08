import os

class cfg:
    faiss_index = "/data/recall.index"
    movie_ids   = "/data/movie_ids.npy"
    user_emb    = "/data/user_emb.pt"
    redis_url   = os.getenv("REDIS_URL", "redis://redis:6379/0")