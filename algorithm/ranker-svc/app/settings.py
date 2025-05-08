import os
class cfg:
    model_path = os.getenv("MODEL_PATH", "/data/ranker.pt")
    user_emb   = "/data/user_emb.pt"
    item_emb   = "/data/item_emb.pt"
