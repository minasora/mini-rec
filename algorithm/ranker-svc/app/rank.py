import torch
import numpy as np
import json
from settings import cfg # Make sure cfg provides correct model paths

# Ensure model paths are correct, e.g., /app/model.pt or /data/model.pt
# Fallback paths are for example only, replace with your actual defaults if cfg is not set
MODEL_PATH = getattr(cfg, 'model_path', '/app/model.pt')
USER_EMB_PATH = getattr(cfg, 'user_emb', '/app/user_emb.pt')
ITEM_EMB_PATH = getattr(cfg, 'item_emb', '/app/item_emb.pt')

# Load models and embeddings globally
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
MODEL = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
# These are the full matrices of embeddings
UE_MATRIX = torch.load(USER_EMB_PATH).weight.detach().cpu().numpy()
IE_MATRIX = torch.load(ITEM_EMB_PATH).weight.detach().cpu().numpy()

async def user_vec(user_id: int):
    """Fetches user vector from UE_MATRIX."""
    if isinstance(user_id, int) and 0 <= user_id < len(UE_MATRIX):
        return UE_MATRIX[user_id].astype(np.float32) # Ensure correct dtype
    
    # For invalid or out-of-range user_id, return zero vector
    print(f"Warning: User ID {user_id} not found in UE_MATRIX. Returning zero vector.")
    return np.zeros(UE_MATRIX.shape[1], dtype="float32") # Return zero vector of correct dimension

async def rank(user_id: int, items: list[int], k: int):
    """Wrapper function to maintain compatibility with older code."""
    return await rank_items_logic(user_id, items, k)

async def rank_items_logic(user_id: int, item_ids: list[int], k: int):
    """
    Core ranking logic.
    user_id: The ID of the user.
    item_ids: A list of candidate item IDs to rank.
    k: The number of top items to return.
    """
    if not item_ids:
        return []

    # 1. Get user embedding
    user_embedding_np = await user_vec(user_id)
    ue_t = torch.tensor(user_embedding_np, dtype=torch.float32)  # Shape: (D,) e.g., (64,)

    # 2. Get item embeddings for the given item_ids
    # Filter for valid item_ids that are within the bounds of IE_MATRIX
    valid_indices_for_ie = []
    original_item_ids_for_valid_indices = []

    for item_id in item_ids:
        if isinstance(item_id, int) and 0 <= item_id < len(IE_MATRIX):
            valid_indices_for_ie.append(item_id)
            original_item_ids_for_valid_indices.append(item_id)
        # else:
        #     print(f"Warning: Item ID {item_id} is invalid or out of bounds. Skipping.")
            
    if not valid_indices_for_ie:
        # print(f"No valid item IDs found in the input list: {item_ids}")
        return []

    item_embeddings_np = IE_MATRIX[valid_indices_for_ie]
    ie_t = torch.tensor(item_embeddings_np, dtype=torch.float32) # Shape: (N, D) e.g. (num_valid_items, 64)

    # Handle case where only one valid item results in ie_t being (D,) instead of (1, D)
    if ie_t.ndim == 1 and len(valid_indices_for_ie) == 1:
        ie_t = ie_t.unsqueeze(0) # Reshape from (D,) to (1, D)
    elif ie_t.ndim == 1 and len(valid_indices_for_ie) > 1:
        # This shouldn't typically happen with numpy advanced indexing IE_MATRIX[list_of_indices]
        # but as a safeguard:
        print(f"Warning: Item embeddings tensor is 1D but multiple valid items. Shape: {ie_t.shape}. Attempting to reshape.")
        if ie_t.shape[0] == UE_MATRIX.shape[1] * len(valid_indices_for_ie): # if it's a flat (N*D,)
             ie_t = ie_t.reshape(len(valid_indices_for_ie), UE_MATRIX.shape[1])
        else:
            print("Error: Cannot reshape item embeddings tensor correctly. Returning empty.")
            return []


    num_items_to_rank = ie_t.shape[0]
    if num_items_to_rank == 0: # Should be caught by "if not valid_indices_for_ie" but double check
        return []

    # 3. Prepare features for the model
    # Expand ue_t to match the number of items: (D,) -> (1, D) -> (N, D)
    ue_expanded_t = ue_t.unsqueeze(0).expand(num_items_to_rank, -1)

    # Concatenate: (N, D) and (N, D) along dim 1 -> (N, 2D)
    # THIS IS THE CRITICAL FIX:
    feats_t = torch.cat([ue_expanded_t, ie_t], dim=1)

    # 4. Get scores from the model
    with torch.no_grad(): # Ensure no gradient calculations
        scores_t = MODEL(feats_t).squeeze(-1) # Output (N, 1) -> (N,)

    scores_np = scores_t.cpu().numpy()

    # Ensure scores_np is 1D array for np.argsort
    if scores_np.ndim == 0: # If only one item, squeeze might make it a scalar
        scores_np = np.array([scores_np.item()])

    # 5. Select top-k items
    # Adjust k if it's larger than the number of scored items
    actual_k = min(k, num_items_to_rank)
    if actual_k <= 0: # If k is 0 or num_items_to_rank is 0
        return []

    # np.argsort sorts in ascending order. We want descending scores.
    # Get indices that would sort scores_np in ascending order:
    # Then take the last 'actual_k' (largest scores) and reverse their order.
    # Example: scores [10, 50, 20], k=2. argsort -> [0, 2, 1]. [-2:] -> [2,1]. [::-1] -> [1,2]
    # Item at index 1 (score 50), then item at index 2 (score 20)
    top_k_indices_in_scores_array = np.argsort(scores_np)[-actual_k:][::-1]

    # 6. Format results
    ranked_results = []
    for idx_in_scores in top_k_indices_in_scores_array:
        original_item_id = original_item_ids_for_valid_indices[idx_in_scores]
        score = float(scores_np[idx_in_scores])
        ranked_results.append((int(original_item_id), score))
        
    return ranked_results