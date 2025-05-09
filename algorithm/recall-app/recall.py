import faiss, torch, json, numpy as np
from functools import lru_cache
from settings import cfg
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
try:
    INDEX = faiss.read_index(cfg.faiss_index)
    ITEM_ID = np.load(cfg.movie_ids).astype("int32")
    USER_EMB = torch.load(cfg.user_emb).weight.detach().cpu().numpy().astype("float32")

except Exception as e:
    print(f"Error loading model/data files: {e}")
    print("Please ensure dummy files are created or actual paths in `cfg` are correct.")
    # Exit if essential data isn't loaded, or handle more gracefully
    import sys
    sys.exit(1)


async def recall(user_id: int, topn: int):

    if user_id < 0 or user_id >= USER_EMB.shape[0]:
        # Handle invalid user_id gracefully, e.g., return empty list or raise error
        # Raising an error might be better handled in the FastAPI endpoint.
        print(f"User ID {user_id} is out of bounds for USER_EMB (size {USER_EMB.shape[0]})")
        return []

    vec = USER_EMB[user_id:user_id+1].copy() # .copy() is good practice if normalization is in-place
    faiss.normalize_L2(vec) # Normalize the query vector, assuming index vectors are also normalized
    
    # D: distances (lower is better for L2), I: indices in the FAISS index
    D, I = INDEX.search(vec, topn) 
    
    # D is shape (1, topn), I is shape (1, topn)
    
    results_list = []
    if I.size > 0 and D.size > 0: # Check if search returned any results
        indices_in_faiss = I[0]
        distances = D[0]
        
        for j, idx_in_faiss in enumerate(indices_in_faiss):
            if 0 <= idx_in_faiss < len(ITEM_ID): # Boundary check for safety
                movie_id = int(ITEM_ID[idx_in_faiss])
                score = float(distances[j])
                results_list.append({"movie_id": movie_id, "score": score})
            else:
                print(f"Warning: FAISS returned out-of-bounds index {idx_in_faiss} for ITEM_ID array.")

    # No caching in serverless mode
    return results_list