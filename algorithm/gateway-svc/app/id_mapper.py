import os
import pandas as pd
from typing import Dict, Optional

# 链接 CSV 文件路径，可通过环境变量覆盖
LINKS_CSV = os.getenv("LINKS_CSV", "/data/links.csv")

# 全局映射字典
MOVIE2TMDB: Dict[int, int] = {}
TMDB2MOVIE: Dict[int, int] = {}

def load_mappings() -> None:
    df = (
        pd.read_csv(
            LINKS_CSV,
            usecols=["movieId", "tmdbId"],
            dtype={"movieId": "Int64", "tmdbId": "Int64"},
        )
        .dropna()
        .query("tmdbId > 0")         
    )

    movie_ids = df["movieId"].astype(int)
    tmdb_ids  = df["tmdbId"].astype(int)

    global MOVIE2TMDB, TMDB2MOVIE
    MOVIE2TMDB = dict(zip(movie_ids, tmdb_ids))
    TMDB2MOVIE = dict(zip(tmdb_ids, movie_ids))
    print(f"id_mapper 载入 {len(MOVIE2TMDB)} 条映射")


# 在模块加载时执行映射加载
load_mappings()

def get_movieid_from_tmdbid(tmdb_id: int) -> Optional[int]:
    return TMDB2MOVIE.get(tmdb_id)


def get_tmdbid_from_movieid(internal_id: int) -> Optional[int]:
    return MOVIE2TMDB.get(internal_id)
