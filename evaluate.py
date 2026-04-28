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
import random

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
                   train_interactions=None, val_dict=None,
                   prompt_module=None,
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
    val_dict : dict[int, list[int]], optional
        Validation ground truth - also masked during TEST evaluation
        to prevent information leakage (standard practice).
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
        if prompt_module is not None:
            # ---------------------------------------------------------
            # Paper-correct prompt injection: BEFORE GCN propagation
            #
            #   X'_T = X_T + ψ(X_T)           ← add prompts to initials
            #   E_final = GCN(A_T, X'_T)       ← propagate through GNN
            #
            # This matches the training procedure in StudentTrainer.
            # ---------------------------------------------------------
            # Get initial (layer-0) embeddings
            init_user = model.user_embedding.weight
            init_item = model.item_embedding.weight

            # Compute attention-based prompts from initials
            user_prompts, item_prompts = prompt_module.compute_prompts(
                init_user, init_item
            )

            # Create prompted initial embeddings
            prompted_init = torch.cat([
                init_user + user_prompts,
                init_item + item_prompts
            ], dim=0)

            # Run GCN propagation with prompted embeddings
            user_emb, item_emb = model.propagate(
                prompted_init, adj.to(device)
            )
        else:
            # Step 1: get all embeddings from the model (no prompts)
            user_emb, item_emb = model(adj.to(device))

    # Build set of known items per user (for masking: train + val)
    user_known_items = defaultdict(set)
    if train_interactions is not None:
        for u, i in train_interactions:
            user_known_items[u].add(i)
    if val_dict is not None:
        for u, items in val_dict.items():
            for i in items:
                user_known_items[u].add(i)

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

        # Mask out known items (train + val) by setting scores to -inf
        for idx, u in enumerate(batch_users):
            if u in user_known_items:
                known_items = list(user_known_items[u])
                scores[idx, known_items] = float("-inf")

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


# =========================================================================
# 4.  SAMPLED EVALUATION  (paper-compatible: 1 positive + 99 negatives)
# =========================================================================

def evaluate_model_sampled(model, adj, test_dict, n_items, k_list,
                           train_interactions=None, val_dict=None, prompt_module=None,
                           n_neg=99, batch_size=256, device="cpu",
                           seed=42):
    """
    Sampled evaluation matching FOREC / DCMPT protocol.

    For each test user:
        1. Take ONE positive test item
        2. Sample `n_neg` random items the user hasn't interacted with
        3. Score only these (1 + n_neg) candidate items
        4. Rank and compute Recall@K / NDCG@K

    This is MUCH easier than full-ranking (100 items vs 24K), so
    metric values are naturally ~2x higher and directly comparable
    to Table 1 in the DCMPT paper.

    Parameters
    ----------
    n_neg : int
        Number of negative samples per user (default: 99, standard in CMR).
    seed : int
        Random seed for reproducible negative sampling.
    (other params identical to evaluate_model)

    Returns
    -------
    metrics : dict
        Keys like "Recall@10", "NDCG@10", etc.
    """
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    rng = random.Random(seed)

    with torch.no_grad():
        if prompt_module is not None:
            # Paper-correct prompt injection: BEFORE GCN propagation
            init_user = model.user_embedding.weight
            init_item = model.item_embedding.weight
            user_prompts, item_prompts = prompt_module.compute_prompts(
                init_user, init_item
            )
            prompted_init = torch.cat([
                init_user + user_prompts,
                init_item + item_prompts
            ], dim=0)
            user_emb, item_emb = model.propagate(
                prompted_init, adj.to(device)
            )
        else:
            user_emb, item_emb = model(adj.to(device))

    # Build set of ALL interacted items per user (train + val + test)
    user_all_items = defaultdict(set)
    if train_interactions is not None:
        for u, i in train_interactions:
            user_all_items[u].add(i)
    if val_dict is not None:
        for u, items in val_dict.items():
            for i in items:
                user_all_items[u].add(i)
    # Also add test items to the "known" set for negative sampling
    for u, items in test_dict.items():
        for i in items:
            user_all_items[u].add(i)

    # Prepare result accumulators
    results = {f"Recall@{k}": [] for k in k_list}
    results.update({f"NDCG@{k}": [] for k in k_list})

    all_items_set = set(range(n_items))

    for u, pos_items in test_dict.items():
        if len(pos_items) == 0:
            continue

        # Take ONE positive item (first one in the list)
        pos_item = pos_items[0]

        # Sample n_neg negatives (items user hasn't interacted with)
        neg_pool = list(all_items_set - user_all_items[u])
        if len(neg_pool) < n_neg:
            neg_items = neg_pool
        else:
            neg_items = rng.sample(neg_pool, n_neg)

        # Candidate set: 1 positive + n_neg negatives
        candidates = [pos_item] + neg_items
        candidate_ids = torch.LongTensor(candidates).to(device)

        # Score only the candidates
        u_emb = user_emb[u].unsqueeze(0)  # (1, dim)
        c_emb = item_emb[candidate_ids]   # (n_candidates, dim)
        scores = (u_emb * c_emb).sum(dim=-1)  # (n_candidates,)

        # Rank candidates by score (descending)
        _, sorted_indices = scores.sort(descending=True)
        ranked_items = candidate_ids[sorted_indices].cpu().tolist()

        # Ground truth: the positive item
        gt = {pos_item}

        for k in k_list:
            results[f"Recall@{k}"].append(recall_at_k(ranked_items, gt, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, gt, k))

    metrics = {key: float(np.mean(vals)) for key, vals in results.items()}
    return metrics


# =========================================================================
# 5.  DUAL EVALUATION  (both full-ranking and sampled)
# =========================================================================

def evaluate_model_both(model, adj, test_dict, n_items, k_list,
                        train_interactions=None, val_dict=None, prompt_module=None,
                        n_neg=99, batch_size=256, device="cpu"):
    """
    Run both evaluation protocols and return combined results.

    Returns
    -------
    dict with keys like:
        "full/Recall@10", "full/NDCG@10",
        "sampled/Recall@10", "sampled/NDCG@10"
    """
    full_metrics = evaluate_model(
        model, adj, test_dict, n_items, k_list,
        train_interactions=train_interactions, val_dict=val_dict,
        prompt_module=prompt_module,
        batch_size=batch_size, device=device,
    )
    sampled_metrics = evaluate_model_sampled(
        model, adj, test_dict, n_items, k_list,
        train_interactions=train_interactions, val_dict=val_dict,
        prompt_module=prompt_module,
        n_neg=n_neg, batch_size=batch_size, device=device,
    )

    combined = {}
    for k, v in full_metrics.items():
        combined[f"full/{k}"] = v
    for k, v in sampled_metrics.items():
        combined[f"sampled/{k}"] = v

    return combined