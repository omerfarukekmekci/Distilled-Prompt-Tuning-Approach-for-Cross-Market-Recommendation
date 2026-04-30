"""
lightgcn.py  –  LightGCN backbone model
========================================

LightGCN is the core recommendation backbone used by DCMPT.  Both the teacher
and the student share this architecture; the only difference is what data they
are trained on and whether their parameters are frozen.

Key idea of LightGCN (He et al., 2020)
---------------------------------------
Standard GCNs apply a learnable weight matrix W and a non-linear activation
at each layer.  LightGCN removes BOTH — the only operation per layer is
neighbourhood aggregation through the normalised adjacency matrix:

    E^{(l+1)} = Â · E^{(l)}

where  Â = D^{-1/2} A D^{-1/2}  is the symmetrically normalised bipartite
adjacency matrix (built in data_utils.py).

The final embedding for each node is the **mean** of its embeddings across
all layers (including the initial layer-0 embedding):

    E_final = (1 / (L+1)) · Σ_{l=0}^{L}  E^{(l)}

This simple design is surprisingly effective because:
•  It preserves the collaborative filtering signal without distortion from
   unnecessary non-linearities.
•  Averaging across layers implicitly combines local and higher-order
   neighbourhood information, acting as a form of self-ensemble.
"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    LightGCN model.

    Parameters
    ----------
    n_users : int
        Number of users in the global ID space.
    n_items : int
        Number of items in the global ID space.
    embed_dim : int
        Dimensionality of the user/item embeddings.  Typical values: 64, 128.
    n_layers : int
        Number of GCN propagation layers.  The paper uses 3.

    Attributes
    ----------
    user_embedding : nn.Embedding  – learnable user embedding table  (layer-0)
    item_embedding : nn.Embedding  – learnable item embedding table  (layer-0)
    """

    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = 64, n_layers: int = 3):
        super().__init__()

        self.n_users   = n_users
        self.n_items   = n_items
        self.embed_dim = embed_dim
        self.n_layers  = n_layers

        # -----------------------------------------------------------------
        # Embedding tables (layer-0 embeddings).
        # We use Normal(0, 0.1) as recommended by the LightGCN authors,
        # which is crucial for stable inner-product scoring.
        # -----------------------------------------------------------------
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    # -----------------------------------------------------------------
    # FORWARD PASS
    # -----------------------------------------------------------------
    def forward(self, adj: torch.sparse.FloatTensor):
        """
        Run L layers of LightGCN propagation.

        Parameters
        ----------
        adj : torch.sparse.FloatTensor
            Normalised adjacency matrix  Â  of shape (n_users+n_items, n_users+n_items).

        Returns
        -------
        user_emb_final : Tensor (n_users, embed_dim)
        item_emb_final : Tensor (n_items, embed_dim)
        """
        # Concatenate user and item embeddings into one big matrix
        # so we can do a single sparse-matmul per layer.
        #
        # Layout:  [ user embeddings ]   ← rows 0 .. n_users-1
        #          [ item embeddings ]   ← rows n_users .. n_users+n_items-1
        all_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # shape: (n_users + n_items, embed_dim)

        # We accumulate every layer's embeddings for the final average.
        layer_embeddings = [all_emb]

        current_emb = all_emb
        for _ in range(self.n_layers):
            # One LightGCN layer:  E^{(l+1)} = Â · E^{(l)}
            # torch.sparse.mm performs sparse × dense matrix multiplication.
            current_emb = torch.sparse.mm(adj, current_emb)
            layer_embeddings.append(current_emb)

        # Mean-pool across all layers  (Eq. from LightGCN paper)
        # Stack into (L+1, n_nodes, dim) then mean along dim=0
        stacked = torch.stack(layer_embeddings, dim=0)   # (L+1, N, D)
        final_emb = stacked.mean(dim=0)                   # (N, D)

        # Split back into user and item parts
        user_emb_final = final_emb[:self.n_users]
        item_emb_final = final_emb[self.n_users:]

        return user_emb_final, item_emb_final

    # -----------------------------------------------------------------
    # PROPAGATE  (for student: runs GCN on external initial embeddings)
    # -----------------------------------------------------------------
    def propagate(self, initial_emb: torch.Tensor,
                  adj: torch.sparse.FloatTensor):
        """
        Run GCN propagation layers on externally provided initial embeddings.

        This is used by the student model to propagate prompt-modified
        embeddings through the GNN.  The paper injects prompts BEFORE
        GCN propagation:

            X'_T = X_T + ψ(X_T)
            E_final = GCN(A_T, X'_T)

        Since LightGCN has no learnable weight matrices or activations,
        the propagation is a **linear** operation.  This means:

            GCN(X + ψ) = GCN(X) + GCN(ψ)

        So we can propagate JUST the prompts and add to the precomputed
        base embeddings, enabling efficient per-epoch computation.

        Parameters
        ----------
        initial_emb : Tensor (n_users + n_items, embed_dim)
            The initial embeddings (e.g. prompt-only vectors, or
            full prompted embeddings X + ψ).
        adj : torch.sparse.FloatTensor
            Normalised adjacency matrix.

        Returns
        -------
        user_emb_final : Tensor (n_users, embed_dim)
        item_emb_final : Tensor (n_items, embed_dim)
        """
        layer_embeddings = [initial_emb]

        current_emb = initial_emb
        for _ in range(self.n_layers):
            current_emb = torch.sparse.mm(adj, current_emb)
            layer_embeddings.append(current_emb)

        stacked = torch.stack(layer_embeddings, dim=0)
        final_emb = stacked.mean(dim=0)

        user_emb_final = final_emb[:self.n_users]
        item_emb_final = final_emb[self.n_users:]

        return user_emb_final, item_emb_final

    # -----------------------------------------------------------------
    # SCORING
    # -----------------------------------------------------------------
    def get_scores(self, user_emb: torch.Tensor,
                   item_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product scores between user and item embeddings.

        Parameters
        ----------
        user_emb : Tensor (batch, dim)
        item_emb : Tensor (batch, dim)   or  (n_items, dim)

        Returns
        -------
        scores : Tensor
            If item_emb has the same batch dimension as user_emb:
                returns element-wise dot products → shape (batch,)
            If item_emb is the full item table:
                returns user × item^T → shape (batch, n_items)

        Why dot product?
        ----------------
        LightGCN uses inner product as the scoring function.  It is
        computationally cheap and, combined with BPR loss, learns a
        meaningful metric space where closer embeddings = higher affinity.
        """
        if user_emb.dim() == 2 and item_emb.dim() == 2:
            if user_emb.shape[0] == item_emb.shape[0]:
                # Element-wise dot product for matched pairs
                return (user_emb * item_emb).sum(dim=-1)
            else:
                # Full score matrix (used during evaluation)
                return user_emb @ item_emb.t()
        # Fallback: generic matmul
        return user_emb @ item_emb.t()

    # -----------------------------------------------------------------
    # REGULARISATION LOSS
    # -----------------------------------------------------------------
    def reg_loss(self, user_ids: torch.Tensor,
                 pos_item_ids: torch.Tensor,
                 neg_item_ids: torch.Tensor) -> torch.Tensor:
        """
        L2 regularisation on the layer-0 embeddings of the active users/items.

        Parameters
        ----------
        user_ids : Tensor (batch,)
        pos_item_ids : Tensor (batch,)
        neg_item_ids : Tensor (batch,)

        Returns
        -------
        loss : scalar Tensor

        Why only layer-0?
        -----------------
        The higher-layer embeddings are deterministic functions of layer-0
        embeddings (no extra parameters), so regularising layer-0 is
        sufficient and avoids double-counting.
        """
        u_emb = self.user_embedding(user_ids)
        pos_emb = self.item_embedding(pos_item_ids)
        neg_emb = self.item_embedding(neg_item_ids)
        # Sum of squared norms divided by batch size and 2
        return (1/2) * (u_emb.norm(2).pow(2) + 
                        pos_emb.norm(2).pow(2) + 
                        neg_emb.norm(2).pow(2)) / float(len(user_ids))