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
   dynamically weights each source market's contribution using adaptive
   weights αm (Eq. 14).

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
    Weighted Ranking Distillation loss  (Paper Eq. 5–9).

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
        w_pos(r) = exp(-r/λ) / Σ_{k=1}^{K} exp(-k/λ)   (Eq. 7)
        w_dev(r) = σ( μ · (r̂_{i_r} - r) )               (Eq. 8)
        w_r = (w_pos · w_dev) / Σ (w_pos · w_dev)         (Eq. 9)

    IMPORTANT FIX: The student rank r̂_{i_r} is computed WITHIN the
    teacher's top-K items only, not across all items globally. This
    ensures rank deviation is meaningful (both ranks in [1, K]).

    Parameters
    ----------
    student_scores : Tensor (batch_users, n_items)
    teacher_scores : Tensor (batch_users, n_items)
    K : int
    lam : float  (λ)
    mu : float  (μ)

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
    w_pos = torch.exp(-teacher_ranks / lam)          # (K,)
    w_pos = w_pos / w_pos.sum()                       # normalise to sum=1

    # ---- Student's ranks for the teacher's top-K items ----
    # FIX: Compute ranks WITHIN the teacher's top-K items only.
    # Get student scores for teacher's top-K items
    student_topk_scores = student_scores.gather(1, teacher_topk_indices)  # (batch, K)

    # Rank within top-K: argsort of argsort gives ranks (0-based), +1 for 1-based
    student_ranks_topk = (student_topk_scores.argsort(dim=1, descending=True)
                          .argsort(dim=1).float() + 1)  # (batch, K)

    # ---- Deviation weight  w_dev  (Paper Eq. 8) ----
    # w_dev(r) = σ( μ · (r̂_{i_r} - r) )
    # When student ranks item LOWER than teacher (r̂ > r): positive → larger sigmoid
    # When student agrees with teacher (r̂ ≈ r): sigmoid ≈ 0.5
    rank_deviation = student_ranks_topk - teacher_ranks.unsqueeze(0)  # (batch, K)
    w_dev = torch.sigmoid(mu * rank_deviation)  # (batch, K)

    # ---- Combined weight  (Paper Eq. 9) ----
    w = w_pos.unsqueeze(0) * w_dev                    # (batch, K)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-10)  # normalise per user

    # ---- WRD loss  (Paper Eq. 5) ----
    # L_WRD = -Σ w_r · log σ(ŷ_{i_r})
    log_sigmoid = F.logsigmoid(student_topk_scores)  # (batch, K)
    loss = -(w * log_sigmoid).sum(dim=1).mean()

    return loss


# =========================================================================
# 3.  AMRDD LOSS  (Adaptive Market-aware Ranking Decoupled Distillation)
#     Paper Equations 10–15
# =========================================================================

def _compute_amrdd_components(logits_T: torch.Tensor,
                               logits_S: torch.Tensor,
                               eps: float = 1e-10):
    """
    Compute the decoupled KL components for one batch.

    Parameters
    ----------
    logits_T : (N, K) – teacher logits [y+, y-_2, ..., y-_K]
    logits_S : (N, K) – student logits

    Returns
    -------
    kl_global : (N,)  – KL(d^T || d^S) per user
    kl_fine   : (N,)  – KL(z^T || z^S) per user
    d_pos_T   : (N, 1) – teacher's positive preference
    d_neg_T   : (N, 1) – teacher's negative preference
    d_pos_S   : (N, 1) – student's positive preference (for αm)
    """
    # --- Global Preference Distribution d = [d+, d-]  (Eq. 10) ---
    log_denom_T = torch.logsumexp(logits_T, dim=1, keepdim=True)
    d_pos_T = torch.exp(logits_T[:, 0:1] - log_denom_T)
    d_neg_T = torch.exp(torch.logsumexp(logits_T[:, 1:], dim=1, keepdim=True) - log_denom_T)

    log_denom_S = torch.logsumexp(logits_S, dim=1, keepdim=True)
    d_pos_S = torch.exp(logits_S[:, 0:1] - log_denom_S)
    d_neg_S = torch.exp(torch.logsumexp(logits_S[:, 1:], dim=1, keepdim=True) - log_denom_S)

    # --- Global KL  (Eq. 11) ---
    kl_global = (d_pos_T * torch.log((d_pos_T + eps) / (d_pos_S + eps)) +
                 d_neg_T * torch.log((d_neg_T + eps) / (d_neg_S + eps))).squeeze(1)

    # --- Fine-grained Preference Distribution z  (Eq. 12) ---
    z_T = F.softmax(logits_T[:, 1:], dim=1)
    z_S = F.softmax(logits_S[:, 1:], dim=1)

    # --- Fine-grained KL  (Eq. 13) ---
    kl_fine = (z_T * torch.log((z_T + eps) / (z_S + eps))).sum(dim=1)

    return kl_global, kl_fine, d_pos_T, d_neg_T, d_pos_S


def amrdd_loss(student_user_emb: torch.Tensor,
               student_item_emb: torch.Tensor,
               teacher_user_emb: torch.Tensor,
               teacher_item_emb: torch.Tensor,
               source_market_batches: dict,
               target_user_pos: dict,
               n_items: int,
               K: int = 50,
               temperature: float = 1.0) -> torch.Tensor:
    """
    Adaptive Market-aware Ranking Decoupled Distillation loss.

    **Paper-faithful implementation (Eq. 10–15).**

    For each source market m, we:
      1. Sample a batch of users from market m
      2. For each user, construct item list [pos, neg_1, ..., neg_{K-1}]
      3. Compute student & teacher logits
      4. Compute decoupled KL components
      5. Compute adaptive weight αm (Eq. 14)
      6. Weight the per-market loss

    The final loss (Eq. 15):
        L_AMRDD = Σ_{m∈Ms} αm · [KL(d^T ∥ d^S_m) + d^{-T} · KL(z^T ∥ z^S_m)]

    Parameters
    ----------
    student_user_emb : (n_users, dim) – student user embeddings
    student_item_emb : (n_items, dim) – student item embeddings
    teacher_user_emb : (n_users, dim) – teacher user embeddings
    teacher_item_emb : (n_items, dim) – teacher item embeddings
    source_market_batches : dict
        {market_name: list of user_ids} – pre-sampled user batches per market
    target_user_pos : dict
        {user_id: set of positive item ids} – for all markets combined
    n_items : int
    K : int – total items in sampled list (1 pos + K-1 neg)
    temperature : float

    Returns
    -------
    loss : scalar Tensor
    """
    device = student_user_emb.device
    eps = 1e-10

    if len(source_market_batches) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- Step 1: Compute teacher's global preference on TARGET users ---
    # We need d^{+T} for αm computation (Eq. 14).
    # Use the target users that appear in at least one source market batch.
    # For efficiency, compute a single reference d^{+T} from the target.

    # --- Step 2: Per-market loss computation ---
    market_losses = []       # per-market weighted loss
    market_d_pos_S = {}      # d^{+S}_m for αm computation

    for market_name, batch_user_ids in source_market_batches.items():
        if len(batch_user_ids) == 0:
            continue

        user_ids_tensor = torch.LongTensor(batch_user_ids).to(device)

        # Get embeddings for this batch
        s_user = student_user_emb[user_ids_tensor]  # (B, dim)
        t_user = teacher_user_emb[user_ids_tensor]  # (B, dim)

        # Compute scores against all items
        s_scores = s_user @ student_item_emb.t()  # (B, n_items)
        t_scores = t_user @ teacher_item_emb.t()  # (B, n_items)

        # For each user, sample 1 positive + K-1 negatives
        valid_users = []
        item_lists = []

        for idx, uid in enumerate(batch_user_ids):
            pos_items = target_user_pos.get(uid, set())
            if len(pos_items) == 0:
                continue

            # Sample 1 positive
            pos_item = next(iter(pos_items)) if len(pos_items) == 1 else \
                list(pos_items)[torch.randint(len(pos_items), (1,)).item()]

            # Sample K-1 negatives
            neg_items = []
            attempts = 0
            while len(neg_items) < K - 1 and attempts < K * 10:
                neg = torch.randint(0, n_items, (1,)).item()
                if neg not in pos_items:
                    neg_items.append(neg)
                attempts += 1

            if len(neg_items) < K - 1:
                continue

            valid_users.append(idx)
            item_lists.append([pos_item] + neg_items)

        if len(valid_users) == 0:
            continue

        valid_indices = torch.LongTensor(valid_users).to(device)
        item_indices = torch.LongTensor(item_lists).to(device)  # (N_valid, K)

        # Gather logits
        logits_T = t_scores[valid_indices].gather(1, item_indices) / temperature  # (N, K)
        logits_S = s_scores[valid_indices].gather(1, item_indices) / temperature  # (N, K)

        # Compute decoupled KL components
        kl_global, kl_fine, d_pos_T, d_neg_T, d_pos_S = _compute_amrdd_components(
            logits_T, logits_S
        )

        # Per-user loss for this market (Eq. 15 inner part)
        per_user_loss = kl_global + d_neg_T.squeeze(1) * kl_fine  # (N,)
        market_loss = per_user_loss.mean()
        market_losses.append(market_loss)

        # Store mean d^{+S}_m for αm computation (Eq. 14)
        market_d_pos_S[market_name] = d_pos_S.mean().detach()

    if len(market_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- Step 3: Compute teacher's d^{+T} for αm ---
    # Use a representative set of target users (from all batches combined)
    all_batch_users = []
    for users in source_market_batches.values():
        all_batch_users.extend(users)
    all_batch_users = list(set(all_batch_users))

    # Filter to users with positive items
    target_users_with_pos = [u for u in all_batch_users if len(target_user_pos.get(u, set())) > 0]
    if len(target_users_with_pos) > 0:
        sample_size = min(64, len(target_users_with_pos))
        sample_users = target_users_with_pos[:sample_size]
        sample_ids = torch.LongTensor(sample_users).to(device)

        t_user_sample = teacher_user_emb[sample_ids]
        t_scores_sample = t_user_sample @ teacher_item_emb.t()

        # Build item lists for teacher d^{+T} estimation
        valid_t = []
        item_lists_t = []
        for idx, uid in enumerate(sample_users):
            pos_items = target_user_pos.get(uid, set())
            if len(pos_items) == 0:
                continue
            pos_item = list(pos_items)[0]
            neg_items = []
            attempts = 0
            while len(neg_items) < K - 1 and attempts < K * 10:
                neg = torch.randint(0, n_items, (1,)).item()
                if neg not in pos_items:
                    neg_items.append(neg)
                attempts += 1
            if len(neg_items) < K - 1:
                continue
            valid_t.append(idx)
            item_lists_t.append([pos_item] + neg_items)

        if len(valid_t) > 0:
            valid_t_idx = torch.LongTensor(valid_t).to(device)
            item_idx_t = torch.LongTensor(item_lists_t).to(device)
            logits_T_ref = t_scores_sample[valid_t_idx].gather(1, item_idx_t) / temperature
            log_denom = torch.logsumexp(logits_T_ref, dim=1, keepdim=True)
            d_pos_T_ref = torch.exp(logits_T_ref[:, 0:1] - log_denom).mean().detach()
        else:
            d_pos_T_ref = torch.tensor(0.5, device=device)
    else:
        d_pos_T_ref = torch.tensor(0.5, device=device)

    # --- Step 4: Compute adaptive weights αm (Eq. 14) ---
    # αm = exp(-|d^{+S}_m - d^{+T}|) / Σ_k exp(-|d^{+S}_k - d^{+T}|)
    market_names = list(market_d_pos_S.keys())
    if len(market_names) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    alpha_scores = []
    for m_name in market_names:
        diff = torch.abs(market_d_pos_S[m_name] - d_pos_T_ref)
        alpha_scores.append(-diff)

    alpha_scores = torch.stack(alpha_scores)
    alphas = F.softmax(alpha_scores, dim=0)  # (n_markets,)

    # --- Step 5: Weighted sum (Eq. 15) ---
    total = torch.tensor(0.0, device=device, requires_grad=True)
    for i, market_name in enumerate(market_names):
        total = total + alphas[i] * market_losses[i]

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
        α = 0.5, β = 1.0, γ = 1.0  (from paper defaults).

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