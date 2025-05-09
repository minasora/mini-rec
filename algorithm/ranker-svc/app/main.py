from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from settings import cfg
import torch
import numpy as np
MODEL_PATH = cfg.model_path
USER_EMB_PATH = cfg.user_emb
ITEM_EMB_PATH = cfg.item_emb
from typing import List
# Load models and embeddings globally
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])

try:
    MODEL = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
    # These are the full matrices of embeddings
    # Assuming user_emb.pt and item_emb.pt were saved as nn.Embedding modules
    UE_MATRIX = torch.load(USER_EMB_PATH, map_location="cpu").weight.detach().cpu().numpy()
    IE_MATRIX = torch.load(ITEM_EMB_PATH, map_location="cpu").weight.detach().cpu().numpy()
    print(f"Ranker: Loaded USER_EMB_PATH: {USER_EMB_PATH}, shape: {UE_MATRIX.shape}")
    print(f"Ranker: Loaded ITEM_EMB_PATH: {ITEM_EMB_PATH}, shape: {IE_MATRIX.shape}")
    print(f"Ranker: Loaded MODEL_PATH: {MODEL_PATH}")

except Exception as e:
    print(f"Error loading ranker models/embeddings: {e}")
    print("Please ensure dummy files are created or actual paths in `cfg` are correct.")
    import sys
    sys.exit(1)


async def user_vec(user_id: int): 
    """Fetches user vector from UE_MATRIX."""
    # Ensure user_id is valid for UE_MATRIX
    if not (isinstance(user_id, int) and 0 <= user_id < UE_MATRIX.shape[0]):
        print(f"Warning: User ID {user_id} is out of bounds for UE_MATRIX (size {UE_MATRIX.shape[0]}). Returning zero vector.")
        return np.zeros(UE_MATRIX.shape[1], dtype=np.float32)
    
    # Direct lookup from preloaded UE_MATRIX
    # This assumes USER_EMB_PATH contains embeddings indexed by original user_id
    return UE_MATRIX[user_id].astype(np.float32)


async def rank_items_logic(user_id: int, item_ids: list[int], k: int):
    """
    Core ranking logic.
    user_id: The ID of the user.
    item_ids: A list of candidate item IDs to rank. These are ORIGINAL item IDs.
    k: The number of top items to return.
    """
    if not item_ids:
        return []

    # 1. Get user embedding
    user_embedding_np = await user_vec(user_id) # user_id is original user_id
    if np.all(user_embedding_np == 0): # Check if user_vec returned a zero vector (e.g. user not found)
        print(f"Warning: User embedding for user_id {user_id} is zero. Ranking might be suboptimal.")
    
    ue_t = torch.tensor(user_embedding_np, dtype=torch.float32)

    # 2. Get item embeddings for the given item_ids
    # IMPORTANT: The item_ids received here are ORIGINAL movie IDs.
    # IE_MATRIX is indexed by INNER item IDs (0 to N_unique_items-1) if you used remapping.
    # We need to map original_item_ids to inner_item_ids to index IE_MATRIX.

    # Load the mapping from original_movie_id to inner_movie_id
    # This map is the reverse of what recall's movie_ids.npy (inner_id -> original_id) is.
    # We need to create this map during training or from movie_ids.npy.
    try:
        # recall's movie_ids.npy: index is inner_id, value is original_id
        inner_id_to_original_id_map = np.load("/data/movie_ids.npy")
        # Create the reverse: original_id -> inner_id
        original_id_to_inner_id_map = {orig_id: inner_id for inner_id, orig_id in enumerate(inner_id_to_original_id_map)}
    except FileNotFoundError:
        print("Error: movie_ids.npy (for mapping original_id to inner_id) not found. Cannot rank items.")
        print("Please ensure movie_ids.npy from recall training is available at ../artifacts/movie_ids.npy")
        return []


    valid_inner_indices_for_ie = []
    original_item_ids_for_valid_indices = []

    for original_item_id in item_ids:
        inner_id = original_id_to_inner_id_map.get(original_item_id)
        if inner_id is not None and 0 <= inner_id < IE_MATRIX.shape[0]:
            valid_inner_indices_for_ie.append(inner_id)
            original_item_ids_for_valid_indices.append(original_item_id) # Keep track of original ID
        # else:
        #     print(f"Warning: Original item ID {original_item_id} not found in mapping or inner_id out of bounds for IE_MATRIX. Skipping.")
            
    if not valid_inner_indices_for_ie:
        # print(f"No valid item IDs (after mapping to inner IDs) found from input: {item_ids}")
        return []

    # Index IE_MATRIX using the list of valid_inner_indices_for_ie
    item_embeddings_np = IE_MATRIX[valid_inner_indices_for_ie]
    ie_t = torch.tensor(item_embeddings_np, dtype=torch.float32)

    if ie_t.ndim == 1 and len(valid_inner_indices_for_ie) == 1:
        ie_t = ie_t.unsqueeze(0)
    elif ie_t.ndim == 1 and len(valid_inner_indices_for_ie) > 1:
        print(f"Warning: Item embeddings tensor is 1D but multiple valid items. Shape: {ie_t.shape}.")
        if ie_t.shape[0] == UE_MATRIX.shape[1] * len(valid_inner_indices_for_ie):
             ie_t = ie_t.reshape(len(valid_inner_indices_for_ie), UE_MATRIX.shape[1])
        else:
            print("Error: Cannot reshape item embeddings tensor correctly. Returning empty.")
            return []

    num_items_to_rank = ie_t.shape[0]
    if num_items_to_rank == 0:
        return []

    ue_expanded_t = ue_t.unsqueeze(0).expand(num_items_to_rank, -1)
    feats_t = torch.cat([ue_expanded_t, ie_t], dim=1)

    with torch.no_grad():
        scores_t = MODEL(feats_t).squeeze(-1)

    scores_np = scores_t.cpu().numpy()
    if scores_np.ndim == 0:
        scores_np = np.array([scores_np.item()])

    actual_k = min(k, num_items_to_rank)
    if actual_k <= 0:
        return []

    top_k_indices_in_scores_array = np.argsort(scores_np)[-actual_k:][::-1]

    ranked_results = []
    for idx_in_scores in top_k_indices_in_scores_array:
        # Use original_item_ids_for_valid_indices to get the original item ID
        original_item_id_val = original_item_ids_for_valid_indices[idx_in_scores]
        score_val = float(scores_np[idx_in_scores])
        # The desired output format for FastAPI
        ranked_results.append({"movie_id": int(original_item_id_val), "score": score_val})
        
    return ranked_results


# --- FastAPI App Definition ---
class RankReq(BaseModel):
    user: int
    items: list[int] # List of original movie_ids from recall
    k: int = 10

class RankedItem(BaseModel): # Response model for a single item
    movie_id: int
    score: float

app = FastAPI(title="RankerSvc")

@app.post("/rank", response_model=List[RankedItem]) # Use the response model
async def post_rank(req: RankReq):
    return await rank_items_logic(req.user, req.items, req.k)

# To run (example):
# pip install fastapi uvicorn torch numpy "redis[hiredis]" pydantic
# Ensure ../artifacts/ has ranker_model.pt, user_emb.pt, item_emb.pt, movie_ids.npy (from recall)
# uvicorn rank:app --reload --port 8001 (assuming rank.py)