"""
prompt.py  –  Attention-Based Prompt Module
=============================================

This is the heart of the DCMPT adaptation strategy.  During the student
training phase, the entire LightGCN backbone is FROZEN and these prompts
are the **only trainable parameters**.

What are "prompts" in this context?
------------------------------------
Prompts are small learnable vectors (same dimensionality as the backbone
embeddings) that are *added* to the frozen user/item embeddings to steer
them toward the target market's preferences.  Think of them as lightweight
"corrections" applied on top of a strong pretrained representation.

Why prompts instead of fine-tuning the whole model?
----------------------------------------------------
1. **Parameter efficiency** – instead of updating millions of backbone
   parameters, we only train a handful of prompt vectors.
2. **Preserves general knowledge** – the frozen backbone retains universal
   collaborative filtering patterns learned during pre-training.
3. **Avoids negative transfer** – full fine-tuning on sparse target data
   can overwrite useful source-market knowledge (catastrophic forgetting).
   Prompts minimise this risk.

Why attention?
--------------
Not every user/item needs the same amount of adaptation.  An attention layer
learns to *weight* the prompt contribution per-node:
•  A user whose behaviour already matches the target market gets a small
   prompt addition.
•  A user who differs significantly from the target distribution gets a
   larger prompt contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptModule(nn.Module):
    """
    Attention-based prompt injection module.

    Parameters
    ----------
    n_users : int
        Number of users in the global ID space.
    n_items : int
        Number of items in the global ID space.
    embed_dim : int
        Must match the backbone's embedding dimensionality.
    n_prompts : int
        Number of independent prompt vectors per entity type (user / item).
        More prompts = more expressivity but also more parameters.
        The paper typically uses a small number (e.g. 4–8).

    Architecture
    ------------
    For users:
        1.  Maintain a bank of `n_prompts` user prompt vectors, each of
            size `embed_dim`.
        2.  For a given frozen user embedding e_u, compute attention weights
            over the prompt bank:
                α_k = softmax( (W_q · e_u)^T · (W_k · p_k) / √d )
        3.  The final prompt for this user is a weighted sum:
                prompt_u = Σ_k  α_k · p_k
        4.  Add it to the frozen embedding:
                ẽ_u = e_u + prompt_u

    Items follow the exact same procedure with their own prompt bank and
    attention parameters.
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int, n_prompts: int = 4):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_prompts = n_prompts

        # ------------------------------------------------------------------
        # PROMPT BANKS
        # ------------------------------------------------------------------
        # Each bank is a learnable (n_prompts, embed_dim) matrix.
        # We initialise with moderate random values (0.1) so that prompts
        # are large enough to actually influence the dot-product ranking
        # from the start.  Too small (0.01) and the prompts are invisible
        # relative to backbone embeddings of magnitude ~1.0.
        # ------------------------------------------------------------------
        self.user_prompt_bank = nn.Parameter(
            torch.randn(n_prompts, embed_dim) * 0.1
        )
        self.item_prompt_bank = nn.Parameter(
            torch.randn(n_prompts, embed_dim) * 0.1
        )

        # ------------------------------------------------------------------
        # ATTENTION VECTORS
        # ------------------------------------------------------------------
        # a_k projects the node's input feature vector to compute attention
        # weights. There is one vector per basis prompt.
        # ------------------------------------------------------------------
        # For users
        self.user_attention_vecs = nn.Parameter(
            torch.randn(n_prompts, embed_dim) * 0.1
        )

        # For items
        self.item_attention_vecs = nn.Parameter(
            torch.randn(n_prompts, embed_dim) * 0.1
        )

    # ------------------------------------------------------------------
    # INTERNAL:  compute attention-weighted prompt for one entity type
    # ------------------------------------------------------------------
    def _attention_prompt(self, frozen_emb: torch.Tensor,
                          prompt_bank: nn.Parameter,
                          attention_vecs: nn.Parameter) -> torch.Tensor:
        """
        Parameters
        ----------
        frozen_emb : Tensor (batch, dim)
            The frozen backbone embeddings for a batch of entities.
        prompt_bank : Parameter (n_prompts, dim)
        attention_vecs : Parameter (n_prompts, dim)

        Returns
        -------
        prompt_vectors : Tensor (batch, dim)
            The attention-weighted prompt to ADD to the frozen embedding.
        """
        # Attention scores: (batch, dim) @ (dim, n_prompts) = (batch, n_prompts)
        # α_{n,k} = exp(a_k^T · e_n) / Σ_{l} exp(a_l^T · e_n)
        attn_scores = frozen_emb @ attention_vecs.t()

        # Softmax over the prompt dimension → attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_prompts)

        # Weighted sum of prompt vectors:
        #   (batch, n_prompts) × (n_prompts, dim) = (batch, dim)
        prompt_vectors = attn_weights @ prompt_bank

        return prompt_vectors

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, user_emb: torch.Tensor,
                item_emb: torch.Tensor):
        """
        Inject prompts into frozen user/item embeddings.

        Parameters
        ----------
        user_emb : Tensor (n_users, dim)  –  frozen backbone user embeddings
        item_emb : Tensor (n_items, dim)  –  frozen backbone item embeddings

        Returns
        -------
        user_emb_prompted : Tensor (n_users, dim)
        item_emb_prompted : Tensor (n_items, dim)

        These are the adapted embeddings that the student model will use
        for scoring and loss computation.
        """
        # Compute attention-weighted prompts for every user
        user_prompts = self._attention_prompt(
            user_emb, self.user_prompt_bank, self.user_attention_vecs
        )
        # Compute attention-weighted prompts for every item
        item_prompts = self._attention_prompt(
            item_emb, self.item_prompt_bank, self.item_attention_vecs
        )

        # Add prompts to frozen embeddings (the "injection" step)
        user_emb_prompted = user_emb + user_prompts
        item_emb_prompted = item_emb + item_prompts

        return user_emb_prompted, item_emb_prompted

    # ------------------------------------------------------------------
    # COMPUTE PROMPTS ONLY  (for paper-correct pre-GCN injection)
    # ------------------------------------------------------------------
    def compute_prompts(self, user_emb: torch.Tensor,
                        item_emb: torch.Tensor):
        """
        Compute prompt vectors WITHOUT adding them to embeddings.

        This is used for the paper's prompt injection strategy where
        prompts are added to initial (layer-0) embeddings BEFORE GCN
        propagation:

            X'_T = X_T + ψ(X_T)         ← prompts added here
            E_final = GCN(A_T, X'_T)     ← then propagated through GNN

        Parameters
        ----------
        user_emb : Tensor (n_users, dim) – initial (layer-0) user embeddings
        item_emb : Tensor (n_items, dim) – initial (layer-0) item embeddings

        Returns
        -------
        user_prompts : Tensor (n_users, dim) – prompt vectors for users
        item_prompts : Tensor (n_items, dim) – prompt vectors for items
        """
        user_prompts = self._attention_prompt(
            user_emb, self.user_prompt_bank, self.user_attention_vecs
        )
        item_prompts = self._attention_prompt(
            item_emb, self.item_prompt_bank, self.item_attention_vecs
        )
        return user_prompts, item_prompts