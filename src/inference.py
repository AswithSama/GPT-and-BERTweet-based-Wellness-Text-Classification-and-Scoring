# src/inference.py

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import AutoTokenizer


USER_RE = re.compile(r"@\w+")
URL_RE  = re.compile(r"http\S+|www\.\S+")


def get_device(prefer: Optional[str] = None) -> str:
    """
    prefer can be 'cuda', 'mps', or 'cpu'. If None, auto-pick best available.
    """
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_tweet(t: str) -> str:
    """
    Match training-time normalization:
    @user -> <user>, URLs -> <url>
    """
    t = USER_RE.sub("<user>", t)
    t = URL_RE.sub("<url>", t)
    return t


def predict(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    label_cols: List[str],
    max_len: int = 128,
    threshold: float = 0.5,
    device: Optional[str] = None,
    return_probabilities: bool = True,
    only_above_threshold: bool = True,
) -> List[Dict[str, Any]]:
    """
    Multi-label inference.

    Returns a list of dicts:
      {
        "text": <original text>,
        "predicted_labels": [...],
        "probabilities": {label: prob, ...}   # optional
      }
    """
    device = get_device(device)

    model.eval()
    model.to(device)

    # normalize input text the same way as dataset
    norm_texts = [normalize_tweet(str(t)) for t in texts]

    with torch.no_grad():
        enc = tokenizer(
            norm_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits)  # [B, C]

    results: List[Dict[str, Any]] = []
    for original_text, row_probs in zip(texts, probs):
        row_probs_list = row_probs.detach().cpu().tolist()

        if only_above_threshold:
            pred_labels = [
                label_cols[i] for i, p in enumerate(row_probs_list) if p > threshold
            ]
        else:
            pred_labels = [
                label_cols[i] for i, p in enumerate(row_probs_list)
            ]

        out: Dict[str, Any] = {
            "text": original_text,
            "predicted_labels": pred_labels,
        }

        if return_probabilities:
            out["probabilities"] = {
                label_cols[i]: float(row_probs_list[i]) for i in range(len(label_cols))
            }

        results.append(out)

    return results


def pretty_print_predictions(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
    show_only_above_threshold: bool = True,
) -> None:
    """
    Helper for notebook/demo printing.
    """
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Predicted labels: {pred['predicted_labels']}")

        probs = pred.get("probabilities", {})
        if probs:
            print("Probabilities:")
            for lbl, score in probs.items():
                if (not show_only_above_threshold) or (score > threshold):
                    print(f"  {lbl}: {score:.3f}")
        print()
