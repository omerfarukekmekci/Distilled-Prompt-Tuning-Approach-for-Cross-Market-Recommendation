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

    # Teacher ranks: for item at position r in teacher_topk, its rank is r+1
    # (1‐indexed).  Shape: (K,)
    teacher_ranks = torch.arange(1, K + 1, dtype=torch.float32,
                                 device=student_scores.device)

    # ---- Position weight  w_pos ----
    # Higher-ranked items (smaller r) get more weight.
    # w_pos(r) = (1 - exp(-(K-r+1)·λ)) / (1 - exp(-λ))
    exponents = -(K - teacher_ranks + 1) * lam
    w_pos = (1.0 - torch.exp(exponents)) / (1.0 - torch.exp(torch.tensor(-lam,
              device=student_scores.device)))  # (K,)

    # ---- Student's ranks for the teacher's top-K items ----
    # We compute the student's rank of each item by sorting student_scores
    # descendingly and finding where the teacher's chosen items end up.
    # argsort of argsort gives ranks.
    student_rank_all = student_scores.argsort(dim=1, descending=True).argsort(dim=1) + 1
    # student_rank_all is (batch, n_items) with 1-based ranks

    # Gather student ranks for teacher's top-K items  → (batch, K)
    student_ranks_topk = student_rank_all.gather(1, teacher_topk_indices).float()

    # ---- Deviation weight  w_dev ----
    # w_dev(r) = σ( μ · | teacher_rank - student_rank | )
    rank_deviation = (teacher_ranks.unsqueeze(0) - student_ranks_topk).abs()  # (batch, K)
    w_dev = torch.sigmoid(mu * rank_deviation)  # (batch, K)

    # ---- Combined weight ----
    w = w_pos.unsqueeze(0) * w_dev  # (batch, K)

    # ---- Student scores for teacher's top-K items ----
    student_topk_scores = student_scores.gather(1, teacher_topk_indices)  # (batch, K)

    # ---- WRD loss ----
    # -w · log σ(student_score)
    log_sigmoid = F.logsigmoid(student_topk_scores)  # (batch, K)
    loss = -(w * log_sigmoid).sum(dim=1).mean()

    return loss


# =========================================================================
# 3.  AMRDD LOSS  (Adaptive Market-aware Ranking Decoupled Distillation)
# =========================================================================

def _safe_kl_divergence(p_target: torch.Tensor, p_input: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per-user KL( p_target || p_input ) only over masked items.

    This avoids the NaN that arises from  0 * log(0)  or  0 * (-inf)
    when using masked_fill + F.kl_div on sparse positive sets.

    Parameters
    ----------
    p_target : (batch, n_items)  – teacher softmax probabilities (masked items only)
    p_input  : (batch, n_items)  – student softmax probabilities (masked items only)
    mask     : (batch, n_items)  – bool, True for items in the active set

    Returns
    -------
    kl_per_user : (batch,)
    """
    eps = 1e-10
    # Only compute on masked positions → avoid 0·log(0) = NaN
    # KL = Σ p_T · (log p_T - log p_S)   over masked items
    log_p_t = (p_target + eps).log()
    log_p_s = (p_input  + eps).log()
    kl = p_target * (log_p_t - log_p_s)
    # Zero-out contributions from items NOT in the set
    kl = kl * mask.float()
    return kl.sum(dim=-1)  # (batch,)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Softmax over items where mask is True, zero elsewhere.
    Numerically stable: subtracts max before exp.
    """
    # Set masked-out items to -inf so they become 0 after softmax
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def amrdd_loss(student_scores: torch.Tensor,
               teacher_scores: torch.Tensor,
               pos_mask: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    """
    Adaptive Market-aware Ranking Decoupled Distillation loss.

    Decouples items into observed/unobserved sets per user, then minimises
    KL divergence between teacher and student softmax distributions in each
    set.  The unobserved-set term is down-weighted when the teacher is
    uncertain (high entropy → noisy signal).

        L_AMRDD = mean( KL_obs + w_neg · KL_unobs )

    Parameters
    ----------
    student_scores : (batch, n_items)
    teacher_scores : (batch, n_items)
    pos_mask       : (batch, n_items) bool – True for observed items
    temperature    : float – softmax temperature

    Returns
    -------
    loss : scalar Tensor
    """
    batch_size, n_items = student_scores.shape
    neg_mask = ~pos_mask

    # Scale by temperature
    t_sc = teacher_scores / temperature
    s_sc = student_scores / temperature

    # --- Skip users with < 2 positive items (can't form a distribution) ---
    n_pos = pos_mask.sum(dim=1)  # (batch,)
    valid = n_pos >= 2           # users we can compute KL_obs for

    if valid.sum() == 0:
        # Edge case: no valid users at all → return 0 loss
        return torch.tensor(0.0, device=student_scores.device, requires_grad=True)

    # ------------------------------------------------------------------
    # OBSERVED SET  KL(p_T^+ || p_S^+)
    # ------------------------------------------------------------------
    p_T_pos = _masked_softmax(t_sc, pos_mask)   # (batch, n_items)
    p_S_pos = _masked_softmax(s_sc, pos_mask)
    kl_obs  = _safe_kl_divergence(p_T_pos, p_S_pos, pos_mask)  # (batch,)

    # ------------------------------------------------------------------
    # UNOBSERVED SET  KL(p_T^- || p_S^-)
    # ------------------------------------------------------------------
    p_T_neg = _masked_softmax(t_sc, neg_mask)
    p_S_neg = _masked_softmax(s_sc, neg_mask)
    kl_unobs = _safe_kl_divergence(p_T_neg, p_S_neg, neg_mask)  # (batch,)

    # ------------------------------------------------------------------
    # TEACHER CONFIDENCE WEIGHT  (per user)
    # ------------------------------------------------------------------
    n_neg = neg_mask.sum(dim=1).float().clamp(min=2)
    eps = 1e-10
    entropy = -(p_T_neg.clamp(min=eps) * p_T_neg.clamp(min=eps).log())
    entropy = (entropy * neg_mask.float()).sum(dim=1)           # (batch,)
    max_entropy = torch.log(n_neg)
    confidence = (1.0 - entropy / max_entropy).clamp(min=0.0)  # (batch,)

    # ------------------------------------------------------------------
    # COMBINE per-user, then average over valid users
    # ------------------------------------------------------------------
    per_user = kl_obs + confidence * kl_unobs   # (batch,)
    loss = per_user[valid].mean()

    return loss


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