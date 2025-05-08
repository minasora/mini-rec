import os
import httpx
from typing import Final


REC_URL:  Final[str] = os.getenv("REC_URL",  "http://rec-user-svc:8000")
RANK_URL: Final[str] = os.getenv("RANK_URL", "http://ranker-svc:8000")

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


MIN_RATED_ITEMS_FOR_PROVISIONAL: Final[int] = int(
    os.getenv("MIN_RATED_ITEMS_FOR_PROVISIONAL", 5)
)
