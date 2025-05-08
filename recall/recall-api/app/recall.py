# recall-api/app/recall.py
import os, json
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import faiss
from fastapi import HTTPException
import redis.asyncio as aioredis

EMB_PATH    = os.getenv("EMB_PATH", "/embeddings")
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379")
TOP_K       = int(os.getenv("TOP_K", "20"))
HNSW_M      = int(os.getenv("HNSW_M", "32"))     # HNSW connectivity
HNSW_EF_C   = int(os.getenv("HNSW_EF_C", "200")) # construction complexity
HNSW_EF_S   = int(os.getenv("HNSW_EF_S", "50"))  # search complexity

def _load_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads `item_embeddings.parquet` which must have columns:
      - `id`    : int
      - `norm`  : either list[float] or VectorUDT
    Returns (ids, vectors) both as numpy arrays.
    """
    table = pq.read_table(os.path.join(EMB_PATH, "item_embeddings.parquet"),
                          columns=["id", "norm"])
    df    = table.to_pandas()
    ids   = df["id"].astype(np.int32).to_numpy()
    # If pyarrow gives list-of-floats:
    vecs  = np.vstack(df["norm"].apply(
                  lambda x: np.array(x["values"], dtype="float32")
              ).to_numpy())
    return ids, vecs

# Load on import
IDS, VECTORS = _load_embeddings()
DIM = VECTORS.shape[1]

def _build_index() -> faiss.Index:
    """
    Build an HNSW index on the L2-normalized vectors for cosine-similarity search.
    """
    # Ensure unit-norm
    faiss.normalize_L2(VECTORS)

    index = faiss.IndexHNSWFlat(DIM, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_C
    index.hnsw.efSearch       = HNSW_EF_S
    index.add(VECTORS)
    return index

INDEX = _build_index()

@lru_cache(maxsize=1)
def get_redis():
    return aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

async def get_similar(movie_id: int, k: int = 10) -> List[Tuple[int, float]]:
    redis     = get_redis()
    cache_key = f"sim:{movie_id}:{k}"

    # 1) try Redis cache
    if await redis.exists(cache_key):
        return json.loads(await redis.get(cache_key))

    # 2) find position in our ID array
    positions = np.where(IDS == movie_id)[0]
    if positions.size == 0:
        raise HTTPException(status_code=404, detail="movieId not found")
    pos = int(positions[0])

    # 3) HNSW search (k+1 so we can filter out the query itself)
    D, I = INDEX.search(VECTORS[pos : pos+1], k + 1)
    sims = []
    for dist, idx in zip(D[0], I[0]):
        mid = int(IDS[idx])
        if mid != movie_id:
            sims.append((mid, float(dist)))
        if len(sims) >= k:
            break

    # 4) cache and return
    await redis.set(cache_key, json.dumps(sims), ex=6 * 3600)
    return sims
