from fastapi import FastAPI, HTTPException, Query
from typing import List

from models import (
    PublicRecommendedItem,
    ProvisionalRecRequest,
    ItemInputWithTmdbId,
)
from services import (
    fetch_recall_candidates,
    fetch_ranked_items,
    _to_public,
)
from config import MIN_RATED_ITEMS_FOR_PROVISIONAL
import id_mapper
from utils import (  # 依旧用你原 utils 中的逻辑
    get_popular_items,
    get_recommendations_with_provisional_emb,
    compute_provisional_user_embedding,
)

app = FastAPI(title="GatewaySvc")


# ------------------------ 启动加载映射 -------------------------
@app.on_event("startup")
async def _load_mapping_on_startup():
    id_mapper.load_mappings()



@app.get(
    "/recommend/{user_id}",
    response_model=List[PublicRecommendedItem],
)
async def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    try:
        recall_raw = await fetch_recall_candidates(user_id, n=k * 20)
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(503, f"Recall service error: {e}") from e

    if not recall_raw:
        return []

    item_ids = [it["movie_id"] for it in recall_raw]
    try:
        ranked = await fetch_ranked_items(user_id, item_ids, k)
    except Exception as e:
        raise HTTPException(503, f"Ranker service error: {e}") from e

    return _to_public(ranked)



@app.post(
    "/recommend/provisional",
    response_model=List[PublicRecommendedItem],
)
async def recommend_provisional(req: ProvisionalRecRequest):
    rated_tmdb_ids: set[int] = {ri.tmdb_id for ri in req.rated_items}

    if len(req.rated_items) < MIN_RATED_ITEMS_FOR_PROVISIONAL:
        return await get_popular_items(req.k, rated_tmdb_ids)



    valid_internal = [
        {"internal_movie_id": ri.internal_movie_id, "rating": ri.rating}
        for ri in req.rated_items
        if ri.internal_movie_id is not None
    ]

    if not valid_internal:
        return await get_popular_items(req.k, rated_tmdb_ids)
    provisional_emb = await compute_provisional_user_embedding(valid_internal)
    if provisional_emb is None:
        return await get_popular_items(req.k, rated_tmdb_ids)

    recs_internal = await get_recommendations_with_provisional_emb(
        provisional_emb,
        req.k,
        {vi["internal_movie_id"] for vi in valid_internal},
    )
    return _to_public(recs_internal)
