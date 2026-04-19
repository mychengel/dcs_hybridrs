"""
Layer 1 – Neural Collaborative Filtering (Candidate Generation).

Architecture: NeuMF (He et al., 2017) combining:
  • GMF branch  — element-wise product of embeddings
  • MLP branch  — concatenated embeddings through dense layers
  • NeuMF output — sigmoid over concatenated GMF+MLP representations

Training uses session-item implicit feedback (1 = item in cart, 0 = sampled negative).
Each transaction is one "session"; there are no persistent user IDs.

Inference: given cart item IDs, a virtual-user vector is computed as the mean of the
corresponding item embeddings. Every product is then scored via a forward pass that
substitutes this virtual vector for the session embedding.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _PairDataset(Dataset):
    def __init__(self, sessions: np.ndarray, items: np.ndarray, labels: np.ndarray):
        self.sessions = torch.tensor(sessions, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sessions[idx], self.items[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuralCF(nn.Module):
    """NeuMF: generalised matrix factorisation + MLP combined."""

    GMF_DIM = 16
    MLP_DIM = 16
    MLP_HIDDEN = [64, 32, 16]   # last dim must equal GMF_DIM for concat output

    def __init__(self, n_sessions: int, n_items: int):
        super().__init__()
        g, m = self.GMF_DIM, self.MLP_DIM

        # GMF embeddings
        self.sess_emb_gmf = nn.Embedding(n_sessions, g)
        self.item_emb_gmf = nn.Embedding(n_items, g)

        # MLP embeddings
        self.sess_emb_mlp = nn.Embedding(n_sessions, m)
        self.item_emb_mlp = nn.Embedding(n_items, m)

        # MLP tower
        layers: list[nn.Module] = []
        in_dim = m * 2
        for out_dim in self.MLP_HIDDEN:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # NeuMF output: GMF_DIM + last MLP hidden dim → 1
        self.out = nn.Linear(g + self.MLP_HIDDEN[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=0.01)

    # ---- training forward ----
    def forward(self, sess_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.sess_emb_gmf(sess_ids) * self.item_emb_gmf(item_ids)
        mlp_in = torch.cat(
            [self.sess_emb_mlp(sess_ids), self.item_emb_mlp(item_ids)], dim=1
        )
        return torch.sigmoid(self.out(torch.cat([gmf, self.mlp(mlp_in)], dim=1))).squeeze()

    # ---- inference (no session ID) ----
    @torch.no_grad()
    def score_with_virtual_user(self, cart_item_indices: list[int]) -> np.ndarray:
        """
        Score all items by substituting the virtual-user vector (mean of cart
        item embeddings) in place of a real session embedding.

        Returns a float32 ndarray of shape [n_items] with values in (0, 1).
        """
        self.eval()
        cart_t = torch.tensor(cart_item_indices, dtype=torch.long)
        n_items = self.item_emb_gmf.num_embeddings
        all_t = torch.arange(n_items)

        # Virtual user = mean of cart item representations
        virt_gmf = self.item_emb_gmf(cart_t).mean(0, keepdim=True)   # [1, g]
        virt_mlp = self.item_emb_mlp(cart_t).mean(0, keepdim=True)   # [1, m]

        all_gmf = self.item_emb_gmf(all_t)   # [n_items, g]
        all_mlp = self.item_emb_mlp(all_t)   # [n_items, m]

        gmf_out = virt_gmf.expand(n_items, -1) * all_gmf              # [n_items, g]
        mlp_in  = torch.cat(
            [virt_mlp.expand(n_items, -1), all_mlp], dim=1
        )                                                               # [n_items, 2m]
        mlp_out = self.mlp(mlp_in)                                     # [n_items, 16]

        combined = torch.cat([gmf_out, mlp_out], dim=1)                # [n_items, g+16]
        scores = torch.sigmoid(self.out(combined)).squeeze()

        return scores.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_pairs(
    session_item_df: pd.DataFrame,
    n_items: int,
    neg_ratio: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build positive + negative (session, item) pairs for BPR-style training."""
    if rng is None:
        rng = np.random.default_rng(42)

    all_items = np.arange(n_items)
    sess_list, item_list, label_list = [], [], []

    for sess_idx, group in session_item_df.groupby("session_idx"):
        pos_items = set(group["item_idx"].tolist())

        for item in pos_items:
            sess_list.append(sess_idx)
            item_list.append(item)
            label_list.append(1.0)

        neg_pool = np.setdiff1d(all_items, list(pos_items))
        n_neg = min(len(pos_items) * neg_ratio, len(neg_pool))
        negs = rng.choice(neg_pool, size=n_neg, replace=False)
        for item in negs:
            sess_list.append(sess_idx)
            item_list.append(int(item))
            label_list.append(0.0)

    return (
        np.array(sess_list, dtype=np.int64),
        np.array(item_list, dtype=np.int64),
        np.array(label_list, dtype=np.float32),
    )


def train_ncf(
    session_item_df: pd.DataFrame,
    n_sessions: int,
    n_items: int,
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 1e-3,
    neg_ratio: int = 4,
) -> "NeuralCF":
    """Train NeuralCF and persist model weights + config."""
    print("  Building training pairs …")
    sessions, items, labels = _build_pairs(session_item_df, n_items, neg_ratio)
    print(f"  Pairs: {len(labels):,}  (pos={labels.sum():,.0f}  neg={(~labels.astype(bool)).sum():,.0f})")

    dataset = _PairDataset(sessions, items, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCF(n_sessions, n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    print(f"  Device: {device}  |  Sessions: {n_sessions:,}  |  Items: {n_items}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for s_b, i_b, l_b in loader:
            s_b, i_b, l_b = s_b.to(device), i_b.to(device), l_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(s_b, i_b), l_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch:02d}/{epochs}  loss={avg:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "ncf_model.pt")
    joblib.dump({"n_sessions": n_sessions, "n_items": n_items}, MODELS_DIR / "ncf_config.pkl")
    print("  NCF saved.")
    return model.cpu()


def load_ncf_model() -> NeuralCF:
    cfg = joblib.load(MODELS_DIR / "ncf_config.pkl")
    model = NeuralCF(cfg["n_sessions"], cfg["n_items"])
    model.load_state_dict(
        torch.load(MODELS_DIR / "ncf_model.pt", map_location="cpu")
    )
    model.eval()
    return model
