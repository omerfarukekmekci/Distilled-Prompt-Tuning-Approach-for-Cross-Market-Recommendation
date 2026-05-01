"""
losses.py  –  BPR, WRD, and AMRDD loss functions
==================================================

DCMPT's total training loss is a weighted combination of three terms:

    L_total = α · L_BPR  +  β · L_WRD  +  γ · L_AMRDD

Each term captures a different learning signal:

1. **L_BPR** (Bayesian Personalised Ranking)
   Standard pairwise ranking loss:  push observed items above unobserved ones.
   This gives the student a direct supervision signal from the sparse target
   market data.

2. **L_WRD** (Weighted Ranking Distillation)
   Transfers *global ranking patterns* from the teacher to the student.
   It focuses on the teacher's top-K ranked items and weights their
   importance by (a) rank position and (b) how much the student disagrees
   with the teacher's ranking.

3. **L_AMRDD** (Adaptive Market-aware Ranking Decoupled Distillation)
   Provides *fine-grained, market-specific* guidance by comparing the
   student's and teacher's preference distributions via KL divergence.
   It "decouples" the loss into observed and unobserved item sets and
   dynamically weights them based on the teacher's confidence.

Combined, these three losses let the student learn from:
•  sparse real labels (BPR),
•  the teacher's overall item ordering (WRD),
•  the teacher's nuanced preference distribution (AMRDD).
"""

import torch
import torch.nn.functional as F


# =========================================================================
# 1.  BPR LOSS
# =========================================================================

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.

    Formula
    -------
        L_BPR = -mean( log σ(s_pos - s_neg) )

    where σ is the sigmoid function.

    Parameters
    ----------
    pos_scores : Tensor (batch,)
        Predicted scores for positive (observed) items.
    neg_scores : Tensor (batch,)
        Predicted scores for negative (sampled) items.

    Returns
    -------
    loss : scalar Tensor

    Intuition
    ---------
    We want  s_pos > s_neg  ⟹  (s_pos - s_neg) > 0  ⟹  σ(…) > 0.5
    ⟹  log σ(…) > log 0.5  ⟹  minimising the negative of that pushes
    positive items higher.
    """
    # Difference between positive and negative scores
    diff = pos_scores - neg_scores

    # -log σ(diff) = log(1 + exp(-diff))  which is numerically more stable
    # when computed via F.softplus.  softplus(x) = log(1 + exp(x)), so
    # -log σ(diff) = softplus(-diff).
    loss = F.softplus(-diff).mean()

    return loss


# =========================================================================
# 2.  WRD LOSS  (Weighted Ranking Distillation)
# =========================================================================

def wrd_loss(student_scores: torch.Tensor,
             teacher_scores: torch.Tensor,
             K: int = 50,
             lam: float = 1.0,
             mu: float = 1.0) -> torch.Tensor:
    """
    Weighted Ranking Distillation loss.

    High-level idea
    ---------------
    Look at the K items the *teacher* ranks highest for each user.
    Encourage the student to also rank them highly, but weight the
    encouragement by:
      (a)  rank position  – top-ranked items matter more, and
      (b)  rank deviation – items where student disagrees more get more weight.

    Formula
    -------
        L_WRD = -(1/|U|) Σ_u  Σ_{r=1}^{K}  w_r · log σ(s_S(u, i_r))

    where:
        w_r   = w_pos(r) · w_dev(r)
        w_pos = (1 - exp(-(K-r+1)·λ)) / (1 - exp(-λ))     (position weight)
        w_dev = σ( μ · |r - r̂_student(i_r)| )             (deviation weight)

    Parameters
    ----------
    student_scores : Tensor (batch_users, n_items)
        Student's predicted scores for all items.
    teacher_scores : Tensor (batch_users, n_items)
        Teacher's predicted scores for all items.
    K : int
        Number of top teacher-ranked items to consider.
    lam : float  (λ)
        Temperature for position weight decay.  Higher → flatter weights.
    mu : float  (μ)
        Sensitivity of deviation weight.  Higher → sharper penalty for
        disagreement.

    Returns
    -------
    loss : scalar Tensor
    """
    batch_size, n_items = student_scores.shape

    # Clamp K to available items
    K = min(K, n_items)

    # ---- Teacher's top-K items per user ----
    # teacher_topk_indices: (batch, K) – item indices ranked by teacher
    _, teacher_topk_indices = teacher_scores.topk(K, dim=1)

    # Teacher ranks: 1-indexed positions in teacher's top-K.  Shape: (K,)
    teacher_ranks = torch.arange(1, K + 1, dtype=torch.float32,
                                 device=student_scores.device)

    # ---- Position weight  w_pos  (Paper Eq. 7) ----
    # w_pos(r) = exp(-r / λ) / Σ_{k=1}^{K} exp(-k / λ)
    # Softmax-style: higher-ranked items (smaller r) get more weight.
    w_pos = torch.exp(-teacher_ranks / lam)          # (K,)
    w_pos = w_pos / w_pos.sum()                       # normalise to sum=1

    # ---- Student's ranks for the teacher's top-K items ----
    # argsort of argsort gives ranks (1-based).
    student_rank_all = student_scores.argsort(dim=1, descending=True).argsort(dim=1) + 1

    # Gather student ranks for teacher's top-K items  → (batch, K)
    student_ranks_topk = student_rank_all.gather(1, teacher_topk_indices).float()

    # ---- Deviation weight  w_dev  (Paper Eq. 8) ----
    # w_dev(r) = σ( μ · (r̂_{i_r} - r) )
    # Signed difference: penalises more when student ranks item LOWER
    # than the teacher (r̂ > r → positive → larger sigmoid).
    rank_deviation = student_ranks_topk - teacher_ranks.unsqueeze(0)  # (batch, K)
    w_dev = torch.sigmoid(mu * rank_deviation)  # (batch, K)

    # ---- Combined weight  (Paper Eq. 9) ----
    # w_r = (w_pos_r · w_dev_r) / Σ_j (w_pos_j · w_dev_j)
    w = w_pos.unsqueeze(0) * w_dev                    # (batch, K)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-10)  # normalise per user

    # ---- Student scores for teacher's top-K items ----
    student_topk_scores = student_scores.gather(1, teacher_topk_indices)  # (batch, K)

    # ---- WRD loss  (Paper Eq. 5) ----
    # L_WRD = -Σ w_r · log σ(ŷ_{i_r})
    log_sigmoid = F.logsigmoid(student_topk_scores)  # (batch, K)
    loss = -(w * log_sigmoid).sum(dim=1).mean()

    return loss


# =========================================================================
# 3.  AMRDD LOSS  (Adaptive Market-aware Ranking Decoupled Distillation)
#     Paper Equations 10–15
# =========================================================================

def amrdd_loss(student_scores: torch.Tensor,
               teacher_scores: torch.Tensor,
               pos_mask: torch.Tensor,
               K: int = 50,
               temperature: float = 1.0):
    """
    Adaptive Market-aware Ranking Decoupled Distillation loss.

    **Paper-faithful implementation (Eq. 10–15).**

    For each user u, we construct a ranked list of K items:
        i = [i⁺₁, i⁻₂, …, i⁻_K]
    containing 1 positive (observed) item and K−1 randomly sampled
    negative (unobserved) items.

    Returns
    -------
    loss : scalar Tensor
    d_pos_S_mean : scalar Tensor  – mean student positive probability
        (used for α_m computation in multi-market AMRDD)
    """
    batch_size, n_items = student_scores.shape
    K = min(K, n_items)
    device = student_scores.device
    eps = 1e-10

    # Mask for users that have at least 1 positive item and K-1 negative items
    valid_mask = (pos_mask.sum(dim=1) > 0) & ((~pos_mask).sum(dim=1) >= K - 1)
    if not valid_mask.any():
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero.detach()

    # Filter to only valid users
    valid_student_scores = student_scores[valid_mask]
    valid_teacher_scores = teacher_scores[valid_mask]
    valid_pos_mask = pos_mask[valid_mask]

    # ---- Sample 1 positive item per user ----
    pos_probs = valid_pos_mask.float()
    pos_indices = torch.multinomial(pos_probs, num_samples=1)  # (N, 1)

    # ---- Sample K-1 negative items per user ----
    neg_probs = (~valid_pos_mask).float()
    neg_indices = torch.multinomial(neg_probs, num_samples=K - 1, replacement=False)  # (N, K-1)

    # ---- Construct item list: [pos, neg₁, neg₂, …, neg_{K-1}] ----
    item_list = torch.cat([pos_indices, neg_indices], dim=1)  # (N, K)

    # ---- Logit matrices for teacher and student (N, K) ----
    y_T = valid_teacher_scores.gather(1, item_list) / temperature
    y_S = valid_student_scores.gather(1, item_list) / temperature

    # ==============================================================
    # a) Global Preference Alignment  (Paper Eq. 10–11)
    # ==============================================================
    log_denom_T = torch.logsumexp(y_T, dim=1, keepdim=True)
    d_pos_T = torch.exp(y_T[:, 0:1] - log_denom_T)
    d_neg_T = torch.exp(torch.logsumexp(y_T[:, 1:], dim=1, keepdim=True) - log_denom_T)

    log_denom_S = torch.logsumexp(y_S, dim=1, keepdim=True)
    d_pos_S = torch.exp(y_S[:, 0:1] - log_denom_S)
    d_neg_S = torch.exp(torch.logsumexp(y_S[:, 1:], dim=1, keepdim=True) - log_denom_S)

    kl_global = (d_pos_T * torch.log((d_pos_T + eps) / (d_pos_S + eps)) +
                 d_neg_T * torch.log((d_neg_T + eps) / (d_neg_S + eps))).squeeze(1)  # (N,)

    # ==============================================================
    # b) Fine-grained Preference Alignment  (Paper Eq. 12–13)
    # ==============================================================
    z_T = F.softmax(y_T[:, 1:], dim=1)  # (N, K-1)
    z_S = F.softmax(y_S[:, 1:], dim=1)  # (N, K-1)

    kl_fine = (z_T * torch.log((z_T + eps) / (z_S + eps))).sum(dim=1)  # (N,)

    # ==============================================================
    # Combined loss  (Paper Eq. 15, single-market component)
    # ==============================================================
    user_loss = kl_global + d_neg_T.squeeze(1) * kl_fine

    # d_pos_S_mean for α_m computation (Eq. 14)
    d_pos_S_mean = d_pos_S.mean().detach()

    return user_loss.mean(), d_pos_S_mean


def amrdd_loss_multi_market(batch_student_user: torch.Tensor,
                            student_item_emb: torch.Tensor,
                            batch_teacher_user: torch.Tensor,
                            teacher_item_emb: torch.Tensor,
                            batch_users: list,
                            source_market_data: dict,
                            target_user_pos: dict,
                            K: int = 50,
                            temperature: float = 1.0,
                            device: str = "cpu"):
    """
    Multi-market AMRDD with adaptive α_m weighting (Paper Eq. 14–15).

    **Correct implementation:** Uses TARGET market users (already propagated
    through the target adjacency) and treats each SOURCE market's item sets
    as the positive reference for those users.

    The key insight: α_m measures which source market's item distribution is
    most similar to the target market (via the teacher). We use target users
    so their embeddings are properly computed with neighbor aggregation.

    Parameters
    ----------
    batch_student_user : (B, dim) – student embeddings for current batch
    student_item_emb   : (n_items, dim) – student item embeddings
    batch_teacher_user : (B, dim) – teacher embeddings for current batch
    teacher_item_emb   : (n_items, dim) – teacher item embeddings
    batch_users        : list[int] – global user IDs for this batch
    source_market_data : dict[str, dict] with keys 'user_pos'
        Source market positive item sets (global item IDs)
    target_user_pos    : dict[int, set] – target market positive sets
    K                  : int – items per sampled list (1 pos + K-1 neg)
    temperature        : float
    device             : str

    Returns
    -------
    loss : scalar Tensor
    """
    n_items = student_item_emb.shape[0]
    eps = 1e-10

    # Pre-compute full score matrices for the batch (B, n_items)
    s_scores = batch_student_user @ student_item_emb.t()
    t_scores = (batch_teacher_user @ teacher_item_emb.t()).detach()

    market_losses = {}
    market_d_pos_S = {}

    for m, mdata in source_market_data.items():
        source_user_pos = mdata['user_pos']

        # For each target user in the batch, use SOURCE market m's positive
        # items as the "positive" reference set. If a target user has no
        # interactions in source market m, fall back to their target positives.
        pos_mask = torch.zeros(len(batch_users), n_items,
                               dtype=torch.bool, device=device)
        valid_count = 0
        for idx, u in enumerate(batch_users):
            # Use source market m's positives if available, else target
            src_pos = source_user_pos.get(u, set())
            tgt_pos = target_user_pos.get(u, set())
            pos_items = list(src_pos) if len(src_pos) > 0 else list(tgt_pos)
            if len(pos_items) > 0:
                pos_mask[idx, pos_items] = True
                valid_count += 1

        if valid_count == 0:
            continue

        loss_m, d_pos_S_m = amrdd_loss(
            s_scores, t_scores, pos_mask,
            K=K, temperature=temperature
        )
        market_losses[m] = loss_m
        market_d_pos_S[m] = d_pos_S_m

    if len(market_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # α_m: compute teacher's d_pos_T using TARGET positives as reference
    target_pos_mask = torch.zeros(len(batch_users), n_items,
                                  dtype=torch.bool, device=device)
    for idx, u in enumerate(batch_users):
        tgt_pos = list(target_user_pos.get(u, set()))
        if len(tgt_pos) > 0:
            target_pos_mask[idx, tgt_pos] = True

    _, d_pos_T = amrdd_loss(t_scores, t_scores, target_pos_mask,
                            K=K, temperature=temperature)

    # α_m = exp(-|d_m^{+S} - d^{+T}|) / Σ exp(-|d_k^{+S} - d^{+T}|)
    alpha_raw = {m: torch.exp(-torch.abs(d - d_pos_T))
                 for m, d in market_d_pos_S.items()}
    alpha_sum = sum(alpha_raw.values()) + eps

    total = sum((alpha_raw[m] / alpha_sum) * market_losses[m]
                for m in market_losses)
    return total





# =========================================================================
# 4.  TOTAL LOSS
# =========================================================================

def total_loss(l_bpr: torch.Tensor, l_wrd: torch.Tensor,
               l_amrdd: torch.Tensor,
               alpha: float = 1.0, beta: float = 0.5,
               gamma: float = 0.5) -> torch.Tensor:
    """
    Combine the three loss components with hyperparameter weights.

        L_total = α · L_BPR  +  β · L_WRD  +  γ · L_AMRDD

    Parameters
    ----------
    l_bpr   : scalar Tensor
    l_wrd   : scalar Tensor
    l_amrdd : scalar Tensor
    alpha, beta, gamma : float
        Balancing hyperparameters.  Typical starting values:
        α = 1.0, β = 0.5, γ = 0.5  (from paper defaults).

    Returns
    -------
    loss : scalar Tensor

    Tuning guidance
    ---------------
    •  α (BPR) grounds the model in real target-market labels.
    •  β (WRD) controls how much global ranking knowledge flows from
       the teacher.  Set higher if the teacher is strong and trustworthy.
    •  γ (AMRDD) controls fine-grained distributional alignment.  Set
       higher when target market has distinctive preferences.
    """
    return alpha * l_bpr + beta * l_wrd + gamma * l_amrdd