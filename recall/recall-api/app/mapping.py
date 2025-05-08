import csv, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINKS_CSV = os.getenv("LINKS_CSV", os.path.join(BASE_DIR, "links.csv"))

MOVIE2TMDB: dict[int, int] = {}
TMDB2MOVIE: dict[int, int] = {}

with open(LINKS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, fieldnames=["movieId", "imdbId", "tmdbId"])
    next(reader)  # 跳过表头
    for row in reader:
        mid_str = row["movieId"].strip()
        tid_str = row["tmdbId"].strip()
        if not mid_str or not tid_str:
            continue

        mid = int(mid_str)
        tid = int(tid_str)
        MOVIE2TMDB[mid] = tid
        TMDB2MOVIE[tid] = mid
