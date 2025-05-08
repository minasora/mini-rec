import faiss, torch, numpy as np
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])


IE_MATRIX_GATEWAY = None
FAISS_INDEX_GATEWAY = None
INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY = None
ORIGINAL_TMDBID_TO_INNER_ID_GATEWAY = None
_FAISS_INDEX_GATEWAY = None



ITEM_EMB_FOR_GATEWAY_PATH = "/data/item_emb.pt"
INNER_ID_TO_ORIGINAL_TMDBID_PATH = "/data/movie_ids.npy"

loaded_item_emb = torch.load(ITEM_EMB_FOR_GATEWAY_PATH, map_location="cpu")
_INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY = np.load(INNER_ID_TO_ORIGINAL_TMDBID_PATH)
_ORIGINAL_TMDBID_TO_INNER_ID_GATEWAY  = {
            orig_id: inner_id for inner_id, orig_id in enumerate(_INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY)
        }
loaded_item_emb = torch.load(ITEM_EMB_FOR_GATEWAY_PATH, map_location="cpu")
_IE_MATRIX_GATEWAY = loaded_item_emb.weight.detach().cpu().numpy().astype(np.float32)




async def compute_provisional_user_embedding(rated_items_data: list[dict]) -> np.ndarray | None:
    if _IE_MATRIX_GATEWAY is None or _ORIGINAL_TMDBID_TO_INNER_ID_GATEWAY is None:
        print("Gateway: Item data not loaded for provisional embedding computation.")
        return None

    item_embeddings_for_user = []
    for rated_item in rated_items_data:
        internal_movie_id = rated_item.get("internal_movie_id")

        if internal_movie_id is not None:
            inner_item_id = _ORIGINAL_TMDBID_TO_INNER_ID_GATEWAY.get(internal_movie_id)
            if inner_item_id is not None and 0 <= inner_item_id < _IE_MATRIX_GATEWAY.shape[0]:
                item_emb = _IE_MATRIX_GATEWAY[inner_item_id] # Already normalized
                item_embeddings_for_user.append(item_emb)

    if not item_embeddings_for_user:
        return None

    provisional_embedding = np.mean(item_embeddings_for_user, axis=0).astype(np.float32)
    # Normalize the provisional embedding
    faiss.normalize_L2(provisional_embedding.reshape(1, -1))
    return provisional_embedding



async def get_recommendations_with_provisional_emb(
    provisional_emb: np.ndarray,
    k: int,
    rated_tmdb_ids: set[int]
) -> list[dict]:
    """
    Gets recommendations using the provisional embedding directly against IE_MATRIX.
    """
    if _IE_MATRIX_GATEWAY is None or _INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY is None:
        print("Gateway: Item data not available for provisional recommendation.")
        return []

    if _FAISS_INDEX_GATEWAY:
        # Use FAISS index for faster search
        query_vector = provisional_emb.reshape(1, -1) # FAISS expects (nqueries, dim)
        distances, inner_indices = _FAISS_INDEX_GATEWAY.search(query_vector, k + len(rated_tmdb_ids) + 20) # Get more to filter
        
        results = []
        if inner_indices.size > 0:
            for i in range(inner_indices.shape[1]):
                inner_idx = inner_indices[0, i]
                if inner_idx == -1 : continue # Should not happen with IndexFlatIP unless k > ntotal
                original_tmdb_id = int(_INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY[inner_idx])
                if original_tmdb_id not in rated_tmdb_ids:
                    results.append({
                        "movie_id": original_tmdb_id,
                        "score": float(distances[0, i]) # Dot product score (higher is better)
                    })
                if len(results) >= k:
                    break
        return results
    else:

        scores = np.dot(_IE_MATRIX_GATEWAY, provisional_emb) # Shape: (num_items,)
        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for inner_idx in sorted_indices:
            original_tmdb_id = int(_INNER_ID_TO_ORIGINAL_TMDBID_GATEWAY[inner_idx])
            if original_tmdb_id not in rated_tmdb_ids:
                results.append({
                    "movie_id": original_tmdb_id,
                    "score": float(scores[inner_idx])
                })
            if len(results) >= k:
                break
        return results




async def get_popular_items(k: int, rated_tmdb_ids: set[int]) -> list:
    if k > 8:
        k = 8
    popular = [ # Example TMDB IDs
        {"tmdb_id": 807, "score": 0.99}, {"tmdb_id": 10193, "score": 0.98},
        {"tmdb_id": 278, "score": 0.97}, {"tmdb_id": 1593, "score": 0.96},
        {"tmdb_id": 44912, "score": 0.95}, {"tmdb_id": 12405, "score": 0.94},
        {"tmdb_id": 800, "score": 0.93}, {"tmdb_id": 6977, "score": 0.92},
    ]
    filtered_popular = [item for item in popular if item['tmdb_id'] not in rated_tmdb_ids]
    return filtered_popular[:k]