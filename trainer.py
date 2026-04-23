"""
trainer.py  –  PreTrainer, TeacherTrainer, StudentTrainer
==========================================================

The DCMPT pipeline has THREE distinct training phases, each handled by
its own trainer class:

┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: PRE-TRAINING  (PreTrainer)                            │
│  Train LightGCN on the COMBINED graph (all markets) with BPR.   │
│  Goal: learn universal collaborative filtering signals.          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: TEACHER TRAINING  (TeacherTrainer)                    │
│  Train a SEPARATE LightGCN on the TARGET market only with BPR.  │
│  Goal: capture dense, target-market-specific preferences.        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: STUDENT TRAINING  (StudentTrainer)                    │
│  FREEZE the pre-trained backbone.  Train ONLY the PromptModule  │
│  using  L_total = α·L_BPR + β·L_WRD + γ·L_AMRDD.              │
│  Goal: adapt the backbone to the target market via prompts,      │
│  guided by the teacher's distillation signal.                    │
└─────────────────────────────────────────────────────────────────┘

Why three separate phases?
--------------------------
•  Pre-training gives us a strong, general backbone.
•  The teacher learns what the target market actually wants (dense signal).
•  The student takes the best of both worlds: the backbone's general
   knowledge + the teacher's target-specific guidance, fused through
   lightweight prompts.  This avoids full fine-tuning (expensive, prone
   to catastrophic forgetting) while still adapting to the target market.
"""

import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# Local modules
from lightgcn import LightGCN
from prompt import PromptModule
from losses import bpr_loss, wrd_loss, amrdd_loss, total_loss
from evaluate import evaluate_model
from data_utils import build_bpr_triplets


# =========================================================================
# 1.  PRE-TRAINER  –  Phase 1
# =========================================================================

class PreTrainer:
    """
    Train LightGCN on the combined graph with BPR loss.

    Parameters
    ----------
    model : LightGCN
    combined_adj : torch.sparse.FloatTensor
    combined_interactions : list of (user, item)
    n_items : int
    lr : float
    weight_decay : float
        L2 regularisation coefficient applied via the optimiser.
    device : str
    """

    def __init__(self, model: LightGCN,
                 combined_adj, combined_interactions,
                 n_items: int,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = "cpu"):

        self.model = model.to(device)
        self.adj = combined_adj.to(device)
        self.interactions = combined_interactions
        self.n_items = n_items
        self.device = device

        # Adam is the standard optimiser used in recommendation GNN papers.
        # weight_decay adds implicit L2 regularisation on all parameters.
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Pre-compute positive items per user for faster negative sampling
        self.user_pos = defaultdict(set)
        for u, i in combined_interactions:
            self.user_pos[u].add(i)

    def train_epoch(self, batch_size: int = 1024):
        """
        Train for one epoch over all (user, pos, neg) triplets.

        Returns
        -------
        avg_loss : float
        """
        self.model.train()

        # Generate fresh negative samples every epoch
        triplets = build_bpr_triplets(
            self.interactions, self.n_items, self.user_pos
        )

        # Shuffle triplets
        random.shuffle(triplets)

        total_loss_val = 0.0
        n_batches = 0

        # -------------------------------------------------------------
        # PERFORMANCE FIX: Compute full-graph embeddings ONCE per epoch.
        # Computing sparse matmuls inside the batch loop makes training
        # pathologically slow. We compute once, and use retain_graph=True
        # when backpropagating. We also enforce a large batch size on
        # CPU to minimize the number of backward sparse-matmuls.
        # -------------------------------------------------------------
        batch_size = max(batch_size, 262144)
        user_emb, item_emb = self.model(self.adj)

        for start in range(0, len(triplets), batch_size):
            batch = triplets[start : start + batch_size]
            users = torch.LongTensor([t[0] for t in batch]).to(self.device)
            pos_items = torch.LongTensor([t[1] for t in batch]).to(self.device)
            neg_items = torch.LongTensor([t[2] for t in batch]).to(self.device)

            # Look up embeddings for this batch
            u_emb = user_emb[users]           # (batch, dim)
            pos_emb = item_emb[pos_items]     # (batch, dim)
            neg_emb = item_emb[neg_items]     # (batch, dim)

            # Scores = dot product
            pos_scores = (u_emb * pos_emb).sum(dim=-1)
            neg_scores = (u_emb * neg_emb).sum(dim=-1)

            # BPR loss + L2 regularisation on active base embeddings
            loss = bpr_loss(pos_scores, neg_scores)
            reg = self.model.reg_loss(users, pos_items) * 1e-4

            total = loss + reg

            self.optimizer.zero_grad()
            
            # Since multiple batches backpropagate through the same
            # single forward graph, we must retain the graph until
            # the final batch of the epoch.
            retain = (start + batch_size < len(triplets))
            total.backward(retain_graph=retain)
            
            self.optimizer.step()

            total_loss_val += total.item()
            n_batches += 1

        return total_loss_val / max(n_batches, 1)

    def train(self, n_epochs: int = 100, batch_size: int = 1024,
              verbose: bool = True):
        """
        Full pre-training loop.

        Parameters
        ----------
        n_epochs : int
        batch_size : int
        verbose : bool   – print loss every 5 epochs
        """
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            avg_loss = self.train_epoch(batch_size)
            epoch_time = time.time() - t0

            if verbose and (epoch % 2 == 0):
                print(f"  [PreTrain] Epoch {epoch:3d}/{n_epochs}  "
                      f"Loss: {avg_loss:.4f}  "
                      f"({epoch_time:.1f}s/epoch)")


# =========================================================================
# 2.  TEACHER TRAINER  –  Phase 2
# =========================================================================

class TeacherTrainer:
    """
    Train a separate LightGCN on the target-market training graph only.

    The teacher provides dense supervision to the student later.
    It is trained with plain BPR loss — same as pre-training, but on a
    different (smaller, market-specific) graph.

    Parameters
    ----------
    model : LightGCN
        A FRESH model (not the pre-trained one).
    target_adj : torch.sparse.FloatTensor
        Adjacency of the target market's training interactions.
    target_interactions : list of (user, item)
    n_items : int
    lr, weight_decay, device : as above
    """

    def __init__(self, model: LightGCN,
                 target_adj, target_interactions,
                 n_items: int,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = "cpu"):

        self.model = model.to(device)
        self.adj = target_adj.to(device)
        self.interactions = target_interactions
        self.n_items = n_items
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.user_pos = defaultdict(set)
        for u, i in target_interactions:
            self.user_pos[u].add(i)

    def train_epoch(self, batch_size: int = 1024):
        """One epoch of BPR training on target market data."""
        self.model.train()

        triplets = build_bpr_triplets(
            self.interactions, self.n_items, self.user_pos
        )

        random.shuffle(triplets)

        total_loss_val = 0.0
        n_batches = 0

        # PERFORMANCE FIX: Compute full-graph embeddings ONCE per epoch
        batch_size = max(batch_size, 262144)
        user_emb, item_emb = self.model(self.adj)

        for start in range(0, len(triplets), batch_size):
            batch = triplets[start : start + batch_size]
            users = torch.LongTensor([t[0] for t in batch]).to(self.device)
            pos_items = torch.LongTensor([t[1] for t in batch]).to(self.device)
            neg_items = torch.LongTensor([t[2] for t in batch]).to(self.device)

            u_emb = user_emb[users]
            pos_emb = item_emb[pos_items]
            neg_emb = item_emb[neg_items]

            pos_scores = (u_emb * pos_emb).sum(dim=-1)
            neg_scores = (u_emb * neg_emb).sum(dim=-1)

            loss = bpr_loss(pos_scores, neg_scores)
            reg = self.model.reg_loss(users, pos_items) * 1e-4

            total = loss + reg

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            total_loss_val += total.item()
            n_batches += 1

        return total_loss_val / max(n_batches, 1)

    def train(self, n_epochs: int = 100, batch_size: int = 1024,
              verbose: bool = True):
        """Full teacher training loop."""
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            avg_loss = self.train_epoch(batch_size)
            epoch_time = time.time() - t0

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  [Teacher]  Epoch {epoch:3d}/{n_epochs}  "
                      f"Loss: {avg_loss:.4f}  "
                      f"({epoch_time:.1f}s/epoch)")


# =========================================================================
# 3.  STUDENT TRAINER  –  Phase 3
# =========================================================================

class StudentTrainer:
    """
    Freeze the pre-trained backbone and train ONLY the PromptModule.

    The student's loss combines three signals:
        L_total = α · L_BPR + β · L_WRD + γ · L_AMRDD

    •  L_BPR uses the student's own (prompted) embeddings against target
       market interaction data.
    •  L_WRD and L_AMRDD compare the student's scores to the teacher's
       scores, transferring the teacher's target-market knowledge.

    Parameters
    ----------
    backbone : LightGCN
        The pre-trained model — its parameters are FROZEN.
    prompt_module : PromptModule
        The learnable prompts — the ONLY trainable component.
    teacher_model : LightGCN
        The trained teacher — also FROZEN (used for distillation only).
    target_adj : torch.sparse.FloatTensor
    target_interactions : list of (user, item)
    n_items : int
    alpha, beta, gamma : float
        Loss balancing hyperparameters.
    K : int
        Top-K for WRD.
    wrd_lambda, wrd_mu : float
        WRD hyperparameters.
    amrdd_temperature : float
        Temperature for AMRDD softmax.
    lr, device : as above
    """

    def __init__(self, backbone: LightGCN,
                 prompt_module: PromptModule,
                 teacher_model: LightGCN,
                 target_adj,
                 target_interactions,
                 n_items: int,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.5,
                 K: int = 50,
                 wrd_lambda: float = 1.0,
                 wrd_mu: float = 1.0,
                 amrdd_temperature: float = 1.0,
                 lr: float = 1e-3,
                 device: str = "cpu"):

        # ---- Freeze backbone ----
        # This is the key step: we do NOT want gradients flowing into
        # the pre-trained backbone.  Only prompts get updated.
        self.backbone = backbone.to(device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---- Freeze teacher ----
        # The teacher is used inference-only to produce distillation targets.
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # ---- Trainable prompt module ----
        self.prompt_module = prompt_module.to(device)

        self.adj = target_adj.to(device)
        self.interactions = target_interactions
        self.n_items = n_items
        self.device = device

        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.K = K
        self.wrd_lambda = wrd_lambda
        self.wrd_mu = wrd_mu
        self.amrdd_temperature = amrdd_temperature

        # Optimiser over prompt parameters ONLY
        self.optimizer = optim.Adam(
            prompt_module.parameters(), lr=lr
        )

        # Pre-compute user positive sets
        self.user_pos = defaultdict(set)
        for u, i in target_interactions:
            self.user_pos[u].add(i)

        # Get the unique users who appear in the target market training set,
        # used for sampling batches during distillation.
        self.train_users = list(self.user_pos.keys())

    def train_epoch(self, batch_size: int = 1024):
        """
        Train the PromptModule (frozen backbone, frozen teacher).
        """
        self.prompt_module.train()

        random.shuffle(self.train_users)

        total_loss_val = 0.0
        n_batches = 0

        # -------------------------------------------------------------
        # PERFORMANCE FIX: Backbone and Teacher are completely frozen!
        # Do not recalculate the graph propagation inside the batch loop.
        # We compute them exactly once per epoch.
        # -------------------------------------------------------------
        with torch.no_grad():
            base_user_emb, base_item_emb = self.backbone(self.adj)
            teacher_user_emb, teacher_item_emb = self.teacher(self.adj)

        # Batch size can be moderate here because we evaluate Full WRD
        batch_size = max(batch_size, 1024)

        for start in range(0, len(self.train_users), batch_size):
            batch_users = self.train_users[start : start + batch_size]
            user_ids = torch.LongTensor(batch_users).to(self.device)

            # 1. Base batch users
            batch_base_user = base_user_emb[user_ids]

            # 2. Inject prompts
            # We ONLY compute user prompts for this batch's users! Saves
            # allocating/processing 3M+ user prompts needlessly.
            # We do need prompts for ALL items for the WRD/AMRDD distillation matrices.
            batch_student_user, student_item_emb = self.prompt_module(
                batch_base_user, base_item_emb
            )

            # 3. Teacher batch users
            batch_teacher_user = teacher_user_emb[user_ids]

            # ----------------------------------------------------------
            # BPR Loss (Iterating locally helps indexing on Python side)
            # ----------------------------------------------------------
            bpr_pos_scores = []
            bpr_neg_scores = []
            for i, u in enumerate(batch_users):
                pos_items = list(self.user_pos[u])
                if len(pos_items) == 0:
                    continue
                pos_i = random.choice(pos_items)
                neg_i = pos_i
                while neg_i in self.user_pos[u]:
                    neg_i = random.randint(0, self.n_items - 1)

                u_emb = batch_student_user[i]
                pos_emb = student_item_emb[pos_i]
                neg_emb = student_item_emb[neg_i]

                bpr_pos_scores.append((u_emb * pos_emb).sum())
                bpr_neg_scores.append((u_emb * neg_emb).sum())

            if len(bpr_pos_scores) == 0:
                continue

            l_bpr = bpr_loss(
                torch.stack(bpr_pos_scores),
                torch.stack(bpr_neg_scores)
            )

            # ----------------------------------------------------------
            # WRD Loss   (student vs teacher full score matrices)
            # ----------------------------------------------------------
            student_scores = batch_student_user @ student_item_emb.t()  
            teacher_scores = batch_teacher_user @ teacher_item_emb.t()

            l_wrd = wrd_loss(
                student_scores, teacher_scores,
                K=self.K, lam=self.wrd_lambda, mu=self.wrd_mu
            )

            # ----------------------------------------------------------
            # AMRDD Loss
            # ----------------------------------------------------------
            pos_mask = torch.zeros(
                len(batch_users), self.n_items,
                dtype=torch.bool, device=self.device
            )
            for idx, u in enumerate(batch_users):
                pos_items_list = list(self.user_pos[u])
                if len(pos_items_list) > 0:
                    pos_mask[idx, pos_items_list] = True

            l_amrdd = amrdd_loss(
                student_scores, teacher_scores.detach(),
                pos_mask, temperature=self.amrdd_temperature
            )

            # ----------------------------------------------------------
            # Combine & Backprop (Prompt parameters only)
            # ----------------------------------------------------------
            loss = total_loss(l_bpr, l_wrd, l_amrdd,
                              self.alpha, self.beta, self.gamma)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss_val += loss.item()
            n_batches += 1

        return total_loss_val / max(n_batches, 1)

    def train(self, n_epochs: int = 50, batch_size: int = 256,
              val_dict: dict = None, test_dict: dict = None,
              train_interactions: list = None,
              k_list: list = None,
              eval_every: int = 10,
              verbose: bool = True):
        """
        Full student training loop with optional periodic evaluation.

        Parameters
        ----------
        n_epochs : int
        batch_size : int
        val_dict : dict   – validation ground truth (user → items)
        test_dict : dict  – test ground truth
        train_interactions : list – for masking during eval
        k_list : list of int – K values for Recall/NDCG
        eval_every : int – evaluate every N epochs
        verbose : bool

        Returns
        -------
        final_metrics : dict or None
        """
        if k_list is None:
            k_list = [10, 20]

        best_metrics = None

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            avg_loss = self.train_epoch(batch_size)
            epoch_time = time.time() - t0

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  [Student]  Epoch {epoch:3d}/{n_epochs}  "
                      f"Loss: {avg_loss:.4f}  "
                      f"({epoch_time:.1f}s/epoch)")

            # Periodic validation
            if val_dict and epoch % eval_every == 0:
                metrics = evaluate_model(
                    self.backbone, self.adj, val_dict, self.n_items,
                    k_list, train_interactions=train_interactions,
                    prompt_module=self.prompt_module,
                    device=self.device
                )
                if verbose:
                    metrics_str = "  ".join(
                        f"{k}: {v:.4f}" for k, v in metrics.items()
                    )
                    print(f"    [Val] {metrics_str}")
                best_metrics = metrics

        # Final test evaluation
        if test_dict:
            final_metrics = evaluate_model(
                self.backbone, self.adj, test_dict, self.n_items,
                k_list, train_interactions=train_interactions,
                prompt_module=self.prompt_module,
                device=self.device
            )
            if verbose:
                print("\n  === FINAL TEST RESULTS ===")
                for k, v in final_metrics.items():
                    print(f"    {k}: {v:.4f}")
            return final_metrics

        return best_metrics