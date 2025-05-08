from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from .recall import get_similar, TOP_K

from .mapping import MOVIE2TMDB, TMDB2MOVIE 

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Movie Recall API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 你的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SimilarResponse(BaseModel):
    tmdbId: int
    score: float



@app.get(
    "/similar/{tmdb_id}",
    response_model=list[SimilarResponse],
    summary="Get similar movies by TMDB ID",
)
async def similar_items_by_tmdb(
    tmdb_id: int,
    k: int = Query(10, ge=1, le=TOP_K, description="number of neighbours"),
):
    # 1) TMDB → MovieLens
    if tmdb_id not in TMDB2MOVIE:
        raise HTTPException(404, detail=f"tmdbId {tmdb_id} not in MovieLens links.csv")
    movie_id = TMDB2MOVIE[tmdb_id]

    # 2) Faiss 检索（内部 movieId）
    sims = await get_similar(movie_id, k)
    if not sims:
        raise HTTPException(404, detail=f"No neighbours found for tmdbId {tmdb_id}")

    # 3) MovieLens → TMDB  （只保留映射存在的）
    results = [
        {"tmdbId": MOVIE2TMDB[mid], "score": score}
        for mid, score in sims
        if mid in MOVIE2TMDB
    ][:k]

    return results