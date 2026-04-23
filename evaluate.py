"""
evaluate.py  –  Recall@K and NDCG@K evaluation metrics
========================================================

These are the two standard top-K ranking metrics used throughout the
recommender systems literature and in the DCMPT paper.

Why Recall@K and NDCG@K?
-------------------------
•  **Recall@K** measures *coverage*: what fraction of the user's truly
   relevant items appear in the model's top-K recommendation list?
   It answers: "Did we find the right items?"

•  **NDCG@K** (Normalised Discounted Cumulative Gain) measures *ranking
   quality*: are the relevant items placed near the top of the list?
   It answers: "Did we rank the right items highly?"

Together they give a balanced picture of recommendation quality.
"""

import math

import torch
import numpy as np
from collections import defaultdict


# =========================================================================
# 1.  RECALL@K
# =========================================================================

def recall_at_k(predicted_items: list, ground_truth: set, k: int) -> float:
    """
    Recall@K for a single user.

    Formula
    -------
        Recall@K = |{recommended ∩ relevant}| / |{relevant}|

    Parameters
    ----------
    predicted_items : list of int
        Top-K item IDs recommended by the model (already sorted by score).
    ground_truth : set of int
        The user's true relevant (held-out) item IDs.
    k : int
        Cut-off length.

    Returns
    -------
    recall : float   in [0, 1]

    Notes
    -----
    If the user has no ground truth items we return 0.0 (cannot recall
    anything that doesn't exist).
    """
    if len(ground_truth) == 0:
        return 0.0

    # Take only the first k predictions
    topk = predicted_items[:k]

    # Count how many of the top-k items are in the ground truth
    hits = len(set(topk) & ground_truth)

    return hits / len(ground_truth)


# =========================================================================
# 2.  NDCG@K
# =========================================================================

def ndcg_at_k(predicted_items: list, ground_truth: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K for a single user.

    Formula
    -------
        DCG@K  = Σ_{i=1}^{K}  rel_i / log₂(i + 1)
        IDCG@K = best possible DCG  (all relevant items packed at the top)
        NDCG@K = DCG@K / IDCG@K

    where rel_i = 1 if item at rank i is relevant, else 0.

    Parameters
    ----------
    predicted_items : list of int
    ground_truth : set of int
    k : int

    Returns
    -------
    ndcg : float   in [0, 1]

    Why log₂(i+1)?
    ---------------
    This "discounting" means a relevant item at position 1 contributes
    1/log₂(2) = 1.0, but a relevant item at position 10 contributes only
    1/log₂(11) ≈ 0.29.  This reflects the real-world fact that users are
    much more likely to see and click on top-ranked items.
    """
    if len(ground_truth) == 0:
        return 0.0

    topk = predicted_items[:k]

    # --- Compute DCG ---
    dcg = 0.0
    for rank, item in enumerate(topk, start=1):
        if item in ground_truth:
            # Relevance is binary: 1 if item is in ground truth
            dcg += 1.0 / math.log2(rank + 1)

    # --- Compute IDCG  (the ideal / best-case DCG) ---
    # If we had |ground_truth| relevant items packed into the first positions:
    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_relevant + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =========================================================================
# 3.  FULL EVALUATION LOOP
# =========================================================================

def evaluate_model(model, adj, test_dict, n_items, k_list,
                   train_interactions=None, prompt_module=None,
                   batch_size=256, device="cpu"):
    """
    Evaluate a recommendation model on held-out test data.

    Steps
    -----
    1.  Run the model forward pass to get all user/item embeddings.
    2.  For each test user, compute scores against ALL items.
    3.  Mask out training items (so we don't "cheat" by recommending
        items the user already interacted with during training).
    4.  Take top-K items and compute Recall@K and NDCG@K.
    5.  Average over all test users.

    Parameters
    ----------
    model : LightGCN
        The backbone model (should be in eval mode).
    adj : torch.sparse.FloatTensor
        The adjacency matrix to use for message passing.
    test_dict : dict[int, list[int]]
        user_id → list of held-out item IDs (ground truth).
    n_items : int
        Total number of items.
    k_list : list of int
        List of K values to evaluate, e.g. [10, 20, 50].
    train_interactions : list of (user, item), optional
        If provided, these items are masked out during evaluation to
        prevent information leakage.
    prompt_module : PromptModule, optional
        If provided, prompts are injected into embeddings (for student eval).
    batch_size : int
        Number of users to score at once (memory management).
    device : str
        "cpu" or "cuda".

    Returns
    -------
    metrics : dict
        Keys like "Recall@10", "NDCG@20", etc.  Values are floats.
    """
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    with torch.no_grad():
        # Step 1: get all embeddings from the model
        user_emb, item_emb = model(adj.to(device))

        # Inject prompts if we're evaluating the student
        if prompt_module is not None:
            user_emb, item_emb = prompt_module(user_emb, item_emb)

    # Build set of training items per user (for masking)
    user_train_items = defaultdict(set)
    if train_interactions is not None:
        for u, i in train_interactions:
            user_train_items[u].add(i)

    # Prepare test users
    test_users = list(test_dict.keys())

    # Accumulators for metrics
    results = {f"Recall@{k}": [] for k in k_list}
    results.update({f"NDCG@{k}": [] for k in k_list})

    # Step 2–4: batch-wise scoring
    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start : start + batch_size]
        user_ids = torch.LongTensor(batch_users).to(device)

        # (batch, dim)
        batch_user_emb = user_emb[user_ids]

        # Scores against ALL items:  (batch, n_items)
        scores = model.get_scores(batch_user_emb, item_emb)

        # Mask out training items by setting their scores to -inf
        for idx, u in enumerate(batch_users):
            if u in user_train_items:
                train_items = list(user_train_items[u])
                scores[idx, train_items] = float("-inf")

        # Get top-max(k_list) items
        max_k = max(k_list)
        _, topk_indices = scores.topk(max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # Compute metrics per user
        for idx, u in enumerate(batch_users):
            gt = set(test_dict[u])
            preds = topk_indices[idx].tolist()

            for k in k_list:
                results[f"Recall@{k}"].append(recall_at_k(preds, gt, k))
                results[f"NDCG@{k}"].append(ndcg_at_k(preds, gt, k))

    # Step 5: average over all test users
    metrics = {key: float(np.mean(vals)) for key, vals in results.items()}

    return metrics