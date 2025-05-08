# main.py (FastAPI application file)
from fastapi import FastAPI, Query, HTTPException
from typing import List
from pydantic import BaseModel

# Assuming recall.py is in the same directory or accessible via PYTHONPATH
from recall import recall, USER_EMB # Import USER_EMB for validation if needed

app = FastAPI(title="RecallSvc")

# Define a Pydantic model for a single recall item
class RecallItem(BaseModel):
    movie_id: int
    score: float # Note: This score is the FAISS distance. Lower is "better"/closer.
                 # If you want a similarity score (higher is better), you might need to transform it.

# Define the response model for the endpoint as a list of RecallItem
@app.get("/recall/{user_id}", response_model=List[RecallItem])
async def get_candidates(user_id: int, n: int = Query(100, ge=1, le=1000)):
    # Optional: Add validation for user_id based on your USER_EMB size
    # This check is also in `recall` function now, but good for early exit.
    if user_id < 0 or user_id >= USER_EMB.shape[0]:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found or out of bounds.")
    
    # The `recall` function now returns a list of dictionaries,
    # which FastAPI will validate against `List[RecallItem]`
    # and serialize into the desired JSON format.
    candidates = await recall(user_id, n)
    if not candidates and (0 <= user_id < USER_EMB.shape[0]):
        # User exists, but no candidates found (e.g., FAISS search returned empty)
        # This is fine, an empty list will be returned, matching the response_model
        pass
    return candidates