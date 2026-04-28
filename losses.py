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
               temperature: float = 1.0) -> torch.Tensor:
    """
    Adaptive Market-aware Ranking Decoupled Distillation loss.

    **Paper-faithful implementation (Eq. 10–15).**

    For each user u, we construct a ranked list of K items:
        i = [i⁺₁, i⁻₂, …, i⁻_K]
    containing 1 positive (observed) item and K−1 randomly sampled
    negative (unobserved) items.

    The loss decouples the alignment into two parts:

    a) **Global Preference Alignment** (Eq. 10–11)
       d = [d⁺, d⁻] is a 2-element distribution capturing the
       aggregated probability of the positive vs all negatives.
       We minimise KL(d_T ∥ d_S).

    b) **Fine-grained Preference Alignment** (Eq. 12–13)
       z = softmax over only the negative items' logits.
       We minimise KL(z_T ∥ z_S).

    The final objective (Eq. 15, simplified without per-market αm):
       L = KL(d_T ∥ d_S) + d⁻_T · KL(z_T ∥ z_S)

    The fine-grained loss is weighted by d⁻_T (teacher's confidence
    in the unobserved set).  When the teacher is very confident about
    the positive item (d⁺_T ≈ 1), the negative ranking matters less.

    Parameters
    ----------
    student_scores : (batch, n_items)  – student logits for all items
    teacher_scores : (batch, n_items)  – teacher logits (detached)
    pos_mask       : (batch, n_items)  – bool, True for observed items
    K              : int  – total items in sampled list (1 pos + K-1 neg)
    temperature    : float – softmax temperature

    Returns
    -------
    loss : scalar Tensor
    """
    batch_size, n_items = student_scores.shape
    K = min(K, n_items)
    device = student_scores.device
    eps = 1e-10

    # Mask for users that have at least 1 positive item and K-1 negative items
    valid_mask = (pos_mask.sum(dim=1) > 0) & ((~pos_mask).sum(dim=1) >= K - 1)
    if not valid_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

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
    # Combined loss  (Paper Eq. 15, simplified)
    # ==============================================================
    user_loss = kl_global + d_neg_T.squeeze(1) * kl_fine

    return user_loss.mean()


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