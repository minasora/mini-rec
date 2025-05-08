from typing import List, Optional, Dict, Any
import id_mapper     

from pydantic import BaseModel, Field, field_validator, FieldValidationInfo, model_validator

class ItemInputWithTmdbId(BaseModel):
    tmdb_id: int
    rating: float = Field(..., ge=0.5, le=5.0) # Example validation
    internal_movie_id: Optional[int] = None

    @model_validator(mode='after')
    def populate_internal_id(self) -> 'ItemInputWithTmdbId':
        if self.tmdb_id is not None and self.internal_movie_id is None:
            if not id_mapper.TMDB2MOVIE:
                print("Warning: TMDB2MOVIE mapping is not loaded. Cannot populate internal_movie_id.")
            else:
                self.internal_movie_id = id_mapper.get_movieid_from_tmdbid(self.tmdb_id)
        return self


class ProvisionalRecRequest(BaseModel):
    firebase_user_id: str
    rated_items: List[ItemInputWithTmdbId] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)  



class PublicRecommendedItem(BaseModel):
    """
    对前端公开的推荐条目：只暴露 tmdb_id + 评分
    """
    tmdb_id: int
    score: float