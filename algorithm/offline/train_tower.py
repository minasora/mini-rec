# train_tower.py (Modified for remapping)
import os, torch, pandas as pd, numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

CSV      = "data/ratings.csv" # Make sure this path is correct
EPOCHS   = 1000 # Reduced for quicker testing
BATCH    = 1 << 14
DIM      = 64
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

def main():


    df = pd.read_csv(CSV, usecols=["userId", "movieId", "rating"])
    print(f"Loaded {len(df)} ratings.")
    df = df[df.rating >= 3.5]
    print(f"Filtered to {len(df)} ratings with rating >= 3.5.")

    if df.empty:
        print("No data after filtering. Exiting.")
        return

    # --- ID Remapping ---
    # User IDs
    unique_user_ids = df.userId.unique()
    user_id_to_inner = {original_id: inner_id for inner_id, original_id in enumerate(unique_user_ids)}
    # inner_user_id_to_original = {v: k for k, v in user_id_to_inner.items()} # If needed later
    df['inner_userId'] = df['userId'].map(user_id_to_inner)
    num_users = len(unique_user_ids)
    print(f"Number of unique users: {num_users}")

    # Movie IDs
    unique_movie_ids = df.movieId.unique()
    movie_id_to_inner = {original_id: inner_id for inner_id, original_id in enumerate(unique_movie_ids)}
    inner_movie_id_to_original_map = np.array(unique_movie_ids, dtype=np.int32) # This will be our movie_ids.npy
    df['inner_movieId'] = df['movieId'].map(movie_id_to_inner)
    num_items = len(unique_movie_ids) # Number of unique movies with rating >= 3.5
    print(f"Number of unique movies (rating >= 3.5): {num_items}")

    users_inner = torch.tensor(df.inner_userId.values, dtype=torch.long)
    items_inner = torch.tensor(df.inner_movieId.values, dtype=torch.long)
    # --- End ID Remapping ---


    ds = TensorDataset(users_inner, items_inner)
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0, # Safer for Windows: 0 workers if issues persist
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True if (os.name != 'nt' and torch.__version__ >= "1.7") else False # Check PyTorch version for persistent_workers
    )

    # num_users and num_items are now based on unique counts after filtering
    # No +1 needed as inner IDs are 0 to N-1

    # ---------- Model ----------
    class TwoTower(nn.Module):
        def __init__(self, n_user, n_item, dim):
            super().__init__()
            # Embeddings are now for inner IDs
            self.user_emb = nn.Embedding(n_user, dim)
            self.item_emb = nn.Embedding(n_item, dim)
        def forward(self, u_inner, i_inner):
            return (self.user_emb(u_inner) * self.item_emb(i_inner)).sum(-1)

    model = TwoTower(num_users, num_items, DIM).to(DEVICE) # Use remapped counts
    opt   = torch.optim.Adam(model.parameters(), 5e-3)

    # ---------- Train ----------
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        for u_inner_batch, i_inner_batch in dl: # Use inner IDs
            u_inner_batch, i_inner_batch = u_inner_batch.to(DEVICE), i_inner_batch.to(DEVICE)
            uvec, ivec = model.user_emb(u_inner_batch), model.item_emb(i_inner_batch)
            logits = uvec @ ivec.T
            labels = torch.arange(len(u_inner_batch), device=DEVICE)
            loss   = nn.functional.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            num_batches += 1
        if num_batches > 0:
            print(f"epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/num_batches:.4f}")
        else:
            print(f"epoch {epoch+1}/{EPOCHS}  no batches processed (dataloader might be empty)")


    print("Training complete. Saving artifacts...")
    os.makedirs("../artifacts", exist_ok=True)

    # Save user embedding module (embeddings are for inner_user_ids)
    # You might also want to save user_id_to_inner or unique_user_ids if you need to map back
    torch.save(model.user_emb.cpu(), "../artifacts/user_emb.pt")
    np.save("../artifacts/user_id_map.npy", unique_user_ids) # Maps inner user ID to original user ID

    # Save item embedding module (embeddings are for inner_item_ids)
    item_embedding_module = model.item_emb.cpu()
    torch.save(item_embedding_module, "../artifacts/item_emb.pt")

    # Save the mapping from inner_movie_id to original_movie_id
    # This array's index is the inner_movie_id, and the value is the original movieId
    np.save("../artifacts/movie_ids.npy", inner_movie_id_to_original_map)
    print(f"Saved movie_ids.npy with {len(inner_movie_id_to_original_map)} entries (mapping inner IDs to original movie IDs).")
    print(f"Sample original movie IDs from mapping: {inner_movie_id_to_original_map[:5]} ... {inner_movie_id_to_original_map[-5:]}")


if __name__ == "__main__":
    if os.name == 'nt': # More robust check for Windows
        mp.set_start_method("spawn", force=True)
    main()