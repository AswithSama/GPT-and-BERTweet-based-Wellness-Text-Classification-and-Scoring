# src/training.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer


MODEL_NAME_DEFAULT = "vinai/bertweet-base"

USER_RE = re.compile(r"@\w+")
URL_RE  = re.compile(r"http\S+|www\.\S+")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_tweet(t: str) -> str:
    t = USER_RE.sub("<user>", t)
    t = URL_RE.sub("<url>", t)
    return t


class LabeledTextDataset(Dataset):
    """
    Stores raw text + multi-label targets. Tokenization happens in collate.
    """
    def __init__(self, df: pd.DataFrame, text_col: str, label_cols: List[str]):
        self.texts  = df[text_col].astype(str).tolist()
        self.labels = df[label_cols].astype("float32").values

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | str]:
        return {
            "text": normalize_tweet(self.texts[i]),
            "labels": torch.from_numpy(self.labels[i])
        }


@dataclass
class CollateBatch:
    """
    Top-level collator (picklable) so num_workers > 0 works.
    """
    tokenizer: AutoTokenizer
    max_len: int = 128

    def __call__(self, batch):
        texts  = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc


class BertweetClassifier(nn.Module):
    """
    BERTweet encoder + mean pooling + linear head for multi-label logits.
    """
    def __init__(self, model_name: str, num_labels: int, unfreeze_top_k: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze all parameters first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze top-k transformer layers (partial fine-tuning)
        total_layers = self.encoder.config.num_hidden_layers
        start = max(0, total_layers - unfreeze_top_k)
        for idx in range(start, total_layers):
            for p in self.encoder.encoder.layer[idx].parameters():
                p.requires_grad = True

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        x = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        x = self.dropout(x)
        return self.classifier(x)  # logits [B, C]


def compute_pos_weight(df: pd.DataFrame, label_cols: List[str], device: str) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: neg_count/pos_count per label.
    """
    Y = df[label_cols].astype("float32").values
    Y_t = torch.tensor(Y, dtype=torch.float32)
    N = Y_t.shape[0]
    pos = Y_t.sum(dim=0)
    neg = N - pos
    pos_weight = (neg / (pos + 1e-8)).to(device).float()
    return pos_weight


def train_model(
    processed_df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_cols: Optional[List[str]] = None,
    model_name: str = MODEL_NAME_DEFAULT,
    max_len: int = 128,
    batch_size: int = 16,
    val_frac: float = 0.2,
    epochs: int = 25,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    patience: int = 5,
    unfreeze_top_k: int = 3,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[nn.Module, AutoTokenizer, Dict[str, List[float]]]:
    """
    Trains classifier from processed_df. Returns (model, tokenizer, history).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # Infer label columns if not provided
    if label_cols is None:
        label_cols = processed_df.drop(columns=[text_col]).columns.tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dataset = LabeledTextDataset(processed_df, text_col=text_col, label_cols=label_cols)

    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate = CollateBatch(tokenizer=tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )

    model = BertweetClassifier(model_name=model_name, num_labels=len(label_cols), unfreeze_top_k=unfreeze_top_k).to(device)

    pos_weight = compute_pos_weight(processed_df, label_cols, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss-based scheduler (correct for val_loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)

    best_val = float("inf")
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running = 0.0

        for batch in train_loader:
            yb = batch["labels"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            opt.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                yb = batch["labels"].to(device)
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                logits = model(**inputs)
                vloss += criterion(logits, yb).item()

        val_loss = vloss / max(1, len(val_loader))

        # scheduler step (correct usage)
        scheduler.step(val_loss)

        cur_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        # ---- Early stopping ----
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience and best_state is not None:
                model.load_state_dict(best_state)
                break

        print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.4f} - val: {val_loss:.4f} - lr: {cur_lr:.2e}")

    return model, tokenizer, history
