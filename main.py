"""
main.py  –  Full DCMPT Pipeline Orchestration
===============================================

This is the entry point that ties everything together.  It runs the
complete DCMPT (Distilled Cross-Market Prompt-Tuning) pipeline:

    1.  Parse CLI arguments
    2.  Load and prepare data     (data_utils.py)
    3.  Pre-train LightGCN        (trainer.py → PreTrainer)
    4.  Train the teacher          (trainer.py → TeacherTrainer)
    5.  Train the student          (trainer.py → StudentTrainer)
    6.  Evaluate on test set       (evaluate.py)

Usage
-----
    python main.py --data_dir xmrec/ --source_markets us uk --target_market de --category Electronics

Data layout expected in --data_dir  (XMRec structure):
    xmrec/
      us/raw/Electronics/ratings_us_Electronics.txt.gz
      uk/raw/Electronics/ratings_uk_Electronics.txt.gz
      de/raw/Electronics/ratings_de_Electronics.txt.gz
      ...

Each ratings file is space-separated with columns:
    userId  itemId  rating  date

Checkpoint system
-----------------
Trained models are saved to --checkpoint_dir (default: checkpoints/).
On subsequent runs, if a checkpoint exists for a given phase, that
phase is SKIPPED and the saved model is loaded instead.  This avoids
re-training from scratch every time you want to tweak the student
hyperparameters or re-run evaluation.

To force re-training, either delete the checkpoint files or pass
--force_retrain.
"""

import argparse
import os
import torch
import time

# Local modules
from data_utils import load_all_markets
from lightgcn import LightGCN
from prompt import PromptModule
from trainer import PreTrainer, TeacherTrainer, StudentTrainer


# =====================================================================
# EASY-ACCESS DEFAULTS  (modify these to avoid typing CLI args)
# =====================================================================
DEFAULT_PRETRAIN_EPOCHS  = 10 # all of these were 10 times higher
DEFAULT_TEACHER_EPOCHS   = 10
DEFAULT_STUDENT_EPOCHS   = 5
DEFAULT_EVAL_EVERY       = 1


# =====================================================================
# CHECKPOINT HELPERS
# =====================================================================

def _checkpoint_path(checkpoint_dir: str, name: str) -> str:
    """Build the full path for a named checkpoint file."""
    return os.path.join(checkpoint_dir, f"{name}.pt")


def save_checkpoint(model, checkpoint_dir: str, name: str):
    """
    Save a model's state_dict to disk.

    Parameters
    ----------
    model : nn.Module
    checkpoint_dir : str
    name : str   – e.g. "backbone_pretrained", "teacher"
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, name)
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(model, checkpoint_dir: str, name: str, device: str = "cpu"):
    """
    Load a model's state_dict from disk.

    Returns
    -------
    True if loaded successfully, False if checkpoint doesn't exist.
    """
    path = _checkpoint_path(checkpoint_dir, name)
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"  ✓ Loaded checkpoint ← {path}")
    return True


def checkpoint_exists(checkpoint_dir: str, name: str) -> bool:
    """Check whether a named checkpoint file exists on disk."""
    return os.path.exists(_checkpoint_path(checkpoint_dir, name))


# =====================================================================
# FORMATTING HELPERS
# =====================================================================

def format_duration(seconds: float) -> str:
    """Pretty-print a duration in h/m/s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


# =====================================================================
# CLI ARGUMENTS
# =====================================================================

def parse_args():
    """
    Parse command-line arguments.

    All hyperparameters have sensible defaults from the DCMPT paper,
    so you can run with just the data and market flags.
    """
    parser = argparse.ArgumentParser(
        description="DCMPT: Distilled Cross-Market Prompt-Tuning"
    )

    # ---- Data ----
    parser.add_argument("--data_dir", type=str, default="xmrec/",
                        help="Root data directory (XMRec folder structure)")
    parser.add_argument("--source_markets", type=str, nargs="+",
                        default=["us"],
                        help="Source market codes, e.g. us uk")
    parser.add_argument("--target_market", type=str, default="de",
                        help="Target market code")
    parser.add_argument("--category", type=str, default="Electronics",
                        help="Product category to load, e.g. Electronics. "
                             "The paper evaluates per-category.")

    # ---- Model architecture ----
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimensionality for LightGCN")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of LightGCN propagation layers")
    parser.add_argument("--n_prompts", type=int, default=4,
                        help="Number of prompt vectors per entity type "
                             "(user/item)")

    # ---- Pre-training ----
    parser.add_argument("--pretrain_epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS,
                        help="Epochs for pre-training on combined graph")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                        help="Learning rate for pre-training")
    parser.add_argument("--pretrain_batch", type=int, default=1024,
                        help="Batch size for pre-training")

    # ---- Teacher training ----
    parser.add_argument("--teacher_epochs", type=int, default=DEFAULT_TEACHER_EPOCHS,
                        help="Epochs for teacher on target market")
    parser.add_argument("--teacher_lr", type=float, default=1e-3,
                        help="Learning rate for teacher")
    parser.add_argument("--teacher_batch", type=int, default=1024,
                        help="Batch size for teacher")

    # ---- Student training ----
    parser.add_argument("--student_epochs", type=int, default=DEFAULT_STUDENT_EPOCHS,
                        help="Epochs for student prompt training")
    parser.add_argument("--student_lr", type=float, default=1e-3,
                        help="Learning rate for student (prompts only)")
    parser.add_argument("--student_batch", type=int, default=256,
                        help="Batch size for student (smaller because "
                             "we compute full score matrices)")

    # ---- Loss weights  (L_total = α·BPR + β·WRD + γ·AMRDD) ----
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for BPR loss")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight for WRD loss")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Weight for AMRDD loss")

    # ---- WRD hyperparameters ----
    parser.add_argument("--wrd_K", type=int, default=50,
                        help="Top-K items for WRD distillation")
    parser.add_argument("--wrd_lambda", type=float, default=1.0,
                        help="Position weight temperature (λ)")
    parser.add_argument("--wrd_mu", type=float, default=1.0,
                        help="Deviation weight sensitivity (μ)")

    # ---- AMRDD hyperparameters ----
    parser.add_argument("--amrdd_temp", type=float, default=1.0,
                        help="Softmax temperature for AMRDD")

    # ---- Evaluation ----
    parser.add_argument("--k_list", type=int, nargs="+", default=[10, 20],
                        help="K values for Recall@K and NDCG@K")
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY,
                        help="Evaluate student every N epochs")

    # ---- Checkpointing ----
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/",
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Ignore existing checkpoints and retrain "
                             "everything from scratch")

    # ---- General ----
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularisation for pre-train and teacher")

    return parser.parse_args()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    args = parse_args()

    # ---- Device selection ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)

    # ---- Checkpoint naming ----
    # Include markets + category so different configs don't overwrite
    # each other's checkpoints.
    src_tag = "_".join(sorted(args.source_markets))
    ckpt_prefix = f"{src_tag}_to_{args.target_market}_{args.category}"
    backbone_ckpt = f"backbone_{ckpt_prefix}"
    teacher_ckpt  = f"teacher_{ckpt_prefix}"
    student_ckpt  = f"student_prompts_{ckpt_prefix}"

    pipeline_start = time.time()

    # ==================================================================
    # STEP 1:  LOAD DATA
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 1:  Loading data")
    print("=" * 60)

    t0 = time.time()
    data = load_all_markets(
        args.data_dir, args.source_markets, args.target_market,
        category=args.category
    )
    print(f"  Data loading done in {format_duration(time.time() - t0)}")

    n_users = data["n_users"]
    n_items = data["n_items"]

    # ==================================================================
    # STEP 2:  PRE-TRAIN LIGHTGCN ON COMBINED GRAPH
    # ==================================================================
    # This learns universal collaborative filtering signals from all
    # markets.  The resulting model captures general item-item and
    # user-item patterns that are not specific to any single market.
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2:  Pre-training LightGCN on combined graph")
    print("=" * 60)

    backbone = LightGCN(n_users, n_items,
                        embed_dim=args.embed_dim,
                        n_layers=args.n_layers)

    # Check for existing checkpoint
    if not args.force_retrain and load_checkpoint(
        backbone, args.checkpoint_dir, backbone_ckpt, device
    ):
        print("  Skipping pre-training (loaded from checkpoint)")
        backbone = backbone.to(device)
    else:
        pre_trainer = PreTrainer(
            model=backbone,
            combined_adj=data["combined_adj"],
            combined_interactions=data["combined_interactions"],
            n_items=n_items,
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        t0 = time.time()
        pre_trainer.train(
            n_epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch,
        )
        elapsed = time.time() - t0
        print(f"  Pre-training done in {format_duration(elapsed)}")

        # Save checkpoint
        save_checkpoint(backbone, args.checkpoint_dir, backbone_ckpt)

    # ==================================================================
    # STEP 3:  TRAIN THE TEACHER ON TARGET MARKET ONLY
    # ==================================================================
    # The teacher is a SEPARATE LightGCN (fresh random init) trained
    # exclusively on the target market.  It becomes an expert on the
    # target market's preferences and will later guide the student.
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3:  Training teacher on target market")
    print("=" * 60)

    teacher = LightGCN(n_users, n_items,
                       embed_dim=args.embed_dim,
                       n_layers=args.n_layers)

    # Check for existing checkpoint
    if not args.force_retrain and load_checkpoint(
        teacher, args.checkpoint_dir, teacher_ckpt, device
    ):
        print("  Skipping teacher training (loaded from checkpoint)")
        teacher = teacher.to(device)
    else:
        teacher_trainer = TeacherTrainer(
            model=teacher,
            target_adj=data["target_adj_train"],
            target_interactions=data["target_train_interactions"],
            n_items=n_items,
            lr=args.teacher_lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        t0 = time.time()
        teacher_trainer.train(
            n_epochs=args.teacher_epochs,
            batch_size=args.teacher_batch,
        )
        elapsed = time.time() - t0
        print(f"  Teacher training done in {format_duration(elapsed)}")

        # Save checkpoint
        save_checkpoint(teacher, args.checkpoint_dir, teacher_ckpt)

    # ==================================================================
    # STEP 4:  TRAIN THE STUDENT  (prompts only, backbone frozen)
    # ==================================================================
    # The pre-trained backbone is FROZEN.  We create a PromptModule
    # (lightweight attention-based prompts) and train ONLY those
    # parameters.  The loss combines:
    #   •  BPR   –  direct supervision from target market labels
    #   •  WRD   –  global ranking distillation from the teacher
    #   •  AMRDD –  fine-grained distributional distillation
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4:  Training student (prompts) with distillation")
    print("=" * 60)

    prompt_module = PromptModule(
        n_users=n_users,
        n_items=n_items,
        embed_dim=args.embed_dim,
        n_prompts=args.n_prompts,
    )

    # Check for existing prompt checkpoint
    if not args.force_retrain and load_checkpoint(
        prompt_module, args.checkpoint_dir, student_ckpt, device
    ):
        print("  Loaded student prompts from checkpoint")
        prompt_module = prompt_module.to(device)
        # Still run evaluation even if loaded from checkpoint
        from evaluate import evaluate_model
        final_metrics = evaluate_model(
            backbone, data["target_adj_train"].to(device),
            data["target_test"], n_items,
            args.k_list,
            train_interactions=data["target_train_interactions"],
            prompt_module=prompt_module,
            device=device,
        )
    else:
        student_trainer = StudentTrainer(
            backbone=backbone,
            prompt_module=prompt_module,
            teacher_model=teacher,
            target_adj=data["target_adj_train"],
            target_interactions=data["target_train_interactions"],
            n_items=n_items,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            K=args.wrd_K,
            wrd_lambda=args.wrd_lambda,
            wrd_mu=args.wrd_mu,
            amrdd_temperature=args.amrdd_temp,
            lr=args.student_lr,
            device=device,
        )

        t0 = time.time()
        final_metrics = student_trainer.train(
            n_epochs=args.student_epochs,
            batch_size=args.student_batch,
            val_dict=data["target_val"],
            test_dict=data["target_test"],
            train_interactions=data["target_train_interactions"],
            k_list=args.k_list,
            eval_every=args.eval_every,
        )
        elapsed = time.time() - t0
        print(f"  Student training done in {format_duration(elapsed)}")

        # Save prompt checkpoint
        save_checkpoint(prompt_module, args.checkpoint_dir, student_ckpt)

    # ==================================================================
    # STEP 5:  PRINT FINAL RESULTS
    # ==================================================================
    total_elapsed = time.time() - pipeline_start

    if final_metrics:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION RESULTS")
        print("=" * 60)
        for metric_name, value in final_metrics.items():
            print(f"  {metric_name:15s} = {value:.4f}")
    else:
        print("\n  No test data provided — skipping final evaluation.")

    print(f"\nTotal pipeline time: {format_duration(total_elapsed)}")
    print("Done!")


if __name__ == "__main__":
    main()