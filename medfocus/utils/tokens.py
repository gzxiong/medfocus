"""Tokenizer helpers shared across attribution methods."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def normalize_piece(t: str) -> str:
    """Strip BPE/SentencePiece markers and lowercase a single token piece."""
    return t.replace("▁", "").replace("Ġ", "").strip().lower()


def locate_target_span(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    answer_text: str,
) -> tuple[int, int]:
    """Find the [start, end) token indices in `input_ids[0]` that cover `answer_text`.

    Used to define the attribution objective: we sum log-probs over this span.
    Returns (start, end) where `end` is exclusive. Falls back to the last
    `len(answer_ids)` tokens if no contiguous match is found.
    """
    ids = input_ids[0].tolist()
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        return len(ids), len(ids)

    target_pieces = [normalize_piece(tokenizer.decode([t])) for t in answer_ids]

    for start in range(len(ids) - len(answer_ids), -1, -1):
        window = [normalize_piece(tokenizer.decode([t])) for t in ids[start:start + len(answer_ids)]]
        if window == target_pieces:
            return start, start + len(answer_ids)

    # Fallback: assume the answer is the last `len(answer_ids)` tokens.
    return len(ids) - len(answer_ids), len(ids)
