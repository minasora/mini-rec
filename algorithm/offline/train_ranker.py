import torch
import torch.nn as nn
import pandas as pd
import numpy as np # Added for loading numpy arrays
from torch.utils.data import DataLoader, Dataset
import os # Added for path joining

# --------------------------------------------------
# 配置
# --------------------------------------------------
EMB_DIM    = 64                  # 单个 embedding 的维度
INPUT_DIM  = EMB_DIM * 2         # 拼接后输入维度
BATCH_SIZE = 1 << 14
EPOCHS     = 100 # Define number of epochs
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = "../artifacts" # Define artifacts directory

print(f"Using device: {DEVICE}")
print(f"Embedding dimension: {EMB_DIM}")
print(f"Input dimension for Ranker: {INPUT_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")

# --------------------------------------------------
# 加载召回模型保存的ID映射
# --------------------------------------------------
try:
    tower_unique_user_ids_original = np.load(os.path.join(ARTIFACTS_DIR, "user_id_map.npy"))
    user_original_to_inner_map = {original_id: inner_id for inner_id, original_id in enumerate(tower_unique_user_ids_original)}
    print(f"Loaded user_id_map.npy with {len(user_original_to_inner_map)} user mappings.")

    tower_unique_movie_ids_original = np.load(os.path.join(ARTIFACTS_DIR, "movie_ids.npy"))
    movie_original_to_inner_map = {original_id: inner_id for inner_id, original_id in enumerate(tower_unique_movie_ids_original)}
    print(f"Loaded movie_ids.npy with {len(movie_original_to_inner_map)} movie mappings.")
except FileNotFoundError:
    print("Error: ID mapping files (user_id_map.npy or movie_ids.npy) not found in artifacts directory.")
    print("Please ensure train_tower.py has been run successfully and artifacts are saved.")
    exit()

# --------------------------------------------------
# 读取评分数据并构造标签
# --------------------------------------------------
df = pd.read_csv("data/ratings.csv", usecols=["userId","movieId","rating"])
print(f"Loaded ratings.csv with {len(df)} entries.")
df["label"] = (df.rating >= 3.5).astype("float32")

# --- 过滤掉在召回阶段没有对应Embedding的user/movie ---
print(f"Original df length: {len(df)}")
df = df[df['userId'].isin(user_original_to_inner_map.keys())]
df = df[df['movieId'].isin(movie_original_to_inner_map.keys())]
print(f"Filtered df length (users/movies with embeddings): {len(df)}")

if df.empty:
    print("No data left after filtering for users/movies with embeddings. Exiting.")
    exit()

# --- 将原始ID转换为内部ID ---
df["inner_userId"] = df["userId"].map(user_original_to_inner_map)
df["inner_movieId"] = df["movieId"].map(movie_original_to_inner_map)

# --- 使用内部ID创建张量 ---
user_inner_ids = torch.tensor(df.inner_userId.values, dtype=torch.long)
item_inner_ids = torch.tensor(df.inner_movieId.values, dtype=torch.long)
labels   = torch.tensor(df.label.values,  dtype=torch.float32)

print(f"Number of samples for ranker training: {len(labels)}")

# --------------------------------------------------
# 自定义 Dataset，传入的是内部 id 和 label
# --------------------------------------------------
class RatingDataset(Dataset):
    def __init__(self, u_inner_ids, i_inner_ids, labs):
        self.u_inner_ids = u_inner_ids
        self.i_inner_ids = i_inner_ids
        self.labs        = labs

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return self.u_inner_ids[idx], self.i_inner_ids[idx], self.labs[idx]

dataset = RatingDataset(user_inner_ids, item_inner_ids, labels)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True, # Important if batch size doesn't evenly divide dataset size
    num_workers=4 if os.name != 'nt' else 0,
    pin_memory=(DEVICE == "cuda")
)
print(f"DataLoader created with {len(loader)} batches.")

# --------------------------------------------------
# 加载预训练的 Embedding module
# --------------------------------------------------
torch.serialization.add_safe_globals([torch.nn.modules.sparse.Embedding])
try:
    ue_path = os.path.join(ARTIFACTS_DIR, "user_emb.pt")
    ie_path = os.path.join(ARTIFACTS_DIR, "item_emb.pt")

    ue: nn.Embedding = torch.load(ue_path, map_location=DEVICE, weights_only=False) # Use map_location
    print(f"Loaded user embeddings from {ue_path}. Shape: {ue.weight.shape}")
    ie: nn.Embedding = torch.load(ie_path, map_location=DEVICE, weights_only=False) # Use map_location
    print(f"Loaded item embeddings from {ie_path}. Shape: {ie.weight.shape}")
except FileNotFoundError:
    print(f"Error: Embedding files (user_emb.pt or item_emb.pt) not found in {ARTIFACTS_DIR}.")
    print("Please ensure train_tower.py has been run successfully.")
    exit()
except Exception as e:
    print(f"Error loading embedding files: {e}")
    exit()


# 确保加载的Embedding层维度与配置一致
if ue.embedding_dim != EMB_DIM:
    print(f"Warning: Loaded user embedding dimension ({ue.embedding_dim}) does not match EMB_DIM ({EMB_DIM}).")
if ie.embedding_dim != EMB_DIM:
    print(f"Warning: Loaded item embedding dimension ({ie.embedding_dim}) does not match EMB_DIM ({EMB_DIM}).")

# 如果不想在精排阶段继续更新这两个 embedding，就 freeze 它们：
FREEZE_EMBEDDINGS = True # Set to False if you want to fine-tune embeddings
if FREEZE_EMBEDDINGS:
    for p in ue.parameters(): p.requires_grad = False
    for p in ie.parameters(): p.requires_grad = False
    print("Pre-trained embeddings are FROZEN.")
else:
    print("Pre-trained embeddings will be FINE-TUNED.")


# --------------------------------------------------
# 定义精排模型（点式二分类 MLP）
# --------------------------------------------------
class Ranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[128, 32]): # Allow configurable hidden layers
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.3)) # Optional dropout
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

model = Ranker(INPUT_DIM).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
bce   = nn.BCEWithLogitsLoss()

print("Ranker model structure:")
print(model)

# --------------------------------------------------
# 训练循环
# --------------------------------------------------
print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):
    model.train() # Set model to training mode
    epoch_loss = 0.0
    num_batches = 0
    for batch_u_inner, batch_i_inner, batch_y in loader:
        # 1) 移动设备 (already done by DataLoader if pin_memory=True and DEVICE="cuda")
        # batch_u_inner = batch_u_inner.to(DEVICE)
        # batch_i_inner = batch_i_inner.to(DEVICE)
        # batch_y = batch_y.to(DEVICE)

        # Ensure IDs are on the correct device if not handled by DataLoader
        batch_u_inner = batch_u_inner.to(DEVICE, non_blocking=True)
        batch_i_inner = batch_i_inner.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)


        # 2) 动态取 embedding (using inner IDs)
        feat_u = ue(batch_u_inner)            # [B, EMB_DIM]
        feat_i = ie(batch_i_inner)            # [B, EMB_DIM]

        # 3) 拼接输入
        x = torch.cat([feat_u, feat_i], dim=1)  # [B, INPUT_DIM]

        # 4) 前向 + 损失
        logits = model(x)              # [B]
        loss   = bce(logits, batch_y)

        # 5) 反向 & 更新
        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        num_batches += 1

    if num_batches > 0:
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}/{EPOCHS}  Loss={avg_epoch_loss:.4f}")
    else:
        print(f"Epoch {epoch}/{EPOCHS}  No batches processed (dataloader might be empty after filtering or BATCH_SIZE too large for filtered data).")
        if len(dataset) > 0 and len(dataset) < BATCH_SIZE and drop_last:
            print(f"  Note: Dataset size ({len(dataset)}) is less than BATCH_SIZE ({BATCH_SIZE}) and drop_last=True.")


# --------------------------------------------------
# 保存模型
# --------------------------------------------------
print("\nTraining complete. Saving ranker model...")
os.makedirs(ARTIFACTS_DIR, exist_ok=True) # Ensure artifacts directory exists
ranker_save_path = os.path.join(ARTIFACTS_DIR, "ranker.pt")
try:
    # Move model to CPU before scripting and saving for wider compatibility
    model_cpu = model.cpu()
    scripted_model = torch.jit.script(model_cpu)
    scripted_model.save(ranker_save_path)
    print(f"Ranker model saved to {ranker_save_path}")
except Exception as e:
    print(f"Error saving scripted model: {e}")
    # Fallback to saving state_dict if scripting fails
    fallback_path = os.path.join(ARTIFACTS_DIR, "ranker_statedict.pt")
    torch.save(model_cpu.state_dict(), fallback_path)
    print(f"Scripting failed. Saved model state_dict to {fallback_path}")