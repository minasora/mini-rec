import httpx
from typing import List, Dict, Any, Iterable, Set
from config import REC_URL, RANK_URL, DEFAULT_TIMEOUT
from models import PublicRecommendedItem
import id_mapper
from utils import (               
    get_popular_items,
    get_recommendations_with_provisional_emb,
    compute_provisional_user_embedding,
)

# -----------------------------#
# 下游 HTTP 调用薄封装
# -----------------------------#

async def fetch_recall_candidates(
    user_id: int, n: int
) -> List[Dict[str, Any]]:
    url = f"{REC_URL}/recall/{user_id}?n={n}"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as cli:
        r = await cli.get(url)
        r.raise_for_status()
        return r.json()  

async def fetch_ranked_items(
    user_id: int, items: List[int], k: int
) -> List[Dict[str, Any]]:
    url = f"{RANK_URL}/rank"
    payload = {"user": user_id, "items": items, "k": k}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as cli:
        r = await cli.post(url, json=payload)
        r.raise_for_status()
        return r.json()  





def _to_public(items: Iterable[Dict[str, Any]]) -> List[PublicRecommendedItem]:
    out: list[PublicRecommendedItem] = []
    for rec in items:
        tmdb = id_mapper.get_tmdbid_from_movieid(rec["movie_id"])
        if tmdb:
            out.append(PublicRecommendedItem(tmdb_id=tmdb, score=rec["score"]))
    return out
