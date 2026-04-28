"""
quick_eval.py  –  Re-evaluate saved checkpoint with baseline comparison
========================================================================
Run this to quickly evaluate a trained model WITHOUT re-training.
Shows:
  - Baseline (backbone only, no prompts) performance
  - DCMPT (backbone + prompts) performance
  - Paper comparison table

Usage:
    python quick_eval.py --data_dir path/to/xmrec --source_markets us uk fr mx ca --target_market de
"""

import argparse
import os
import time
import torch

from data_utils import load_all_markets
from lightgcn import LightGCN
from prompt import PromptModule
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation with baseline comparison")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--source_markets", nargs="+", default=["us", "uk", "fr", "mx", "ca"])
    parser.add_argument("--target_market", type=str, default="de")
    parser.add_argument("--category", type=str, default="Electronics")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--k_list", type=int, nargs="+", default=[10, 20])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\n[1/3] Loading data...")
    data = load_all_markets(
        args.data_dir, args.source_markets, args.target_market, args.category
    )
    n_users = data["n_users"]
    n_items = data["n_items"]
    print(f"  {n_users:,} users, {n_items:,} items")

    # Build checkpoint names
    src_tag = "_".join(sorted(args.source_markets))
    backbone_ckpt = f"backbone_{src_tag}_to_{args.target_market}_{args.category}.pt"
    student_ckpt = f"student_prompts_{src_tag}_to_{args.target_market}_{args.category}.pt"

    # Load backbone
    print("\n[2/3] Loading models from checkpoints...")
    backbone = LightGCN(n_users, n_items, args.embed_dim, args.n_layers).to(device)
    bb_path = os.path.join(args.checkpoint_dir, backbone_ckpt)
    if os.path.exists(bb_path):
        backbone.load_state_dict(torch.load(bb_path, map_location=device, weights_only=True))
        print(f"  Backbone: {backbone_ckpt}")
    else:
        print(f"  Backbone not found: {bb_path}")
        return

    # Load prompt module
    prompt_module = PromptModule(n_users, n_items, args.embed_dim, args.n_prompts).to(device)
    pm_path = os.path.join(args.checkpoint_dir, student_ckpt)
    has_prompts = False
    if os.path.exists(pm_path):
        prompt_module.load_state_dict(torch.load(pm_path, map_location=device, weights_only=True))
        print(f"  Prompts:  {student_ckpt}")
        has_prompts = True
    else:
        print(f"  Prompts not found: {pm_path} (will evaluate baseline only)")

    backbone.eval()
    prompt_module.eval()

    adj = data["target_adj_train"].to(device)
    common_args = dict(
        adj=adj,
        test_dict=data["target_test"],
        n_items=n_items,
        k_list=args.k_list,
        train_interactions=data["target_train_interactions"],
        val_dict=data["target_val"],
        device=device,
    )

    # ---- Baseline evaluation ----
    print("\n[3/3] Running evaluations...")
    print("\n  Baseline (backbone only, no prompts)...")
    t0 = time.time()
    baseline_metrics = evaluate_model(model=backbone, prompt_module=None, **common_args)
    for k, v in baseline_metrics.items():
        print(f"    {k:15s} = {v:.4f}")
    print(f"    (took {time.time()-t0:.1f}s)")

    # ---- DCMPT evaluation ----
    if has_prompts:
        print("\n  DCMPT (backbone + prompts)...")
        t0 = time.time()
        dcmpt_metrics = evaluate_model(model=backbone, prompt_module=prompt_module, **common_args)
        for k, v in dcmpt_metrics.items():
            print(f"    {k:15s} = {v:.4f}")
        print(f"    (took {time.time()-t0:.1f}s)")
    else:
        dcmpt_metrics = None

    # ---- Comparison table ----
    print("\n" + "=" * 60)
    print("COMPARISON TABLE  (full ranking)")
    print("=" * 60)

    paper = {"NDCG@10": 0.3898, "Recall@10": 0.5365}
    paper_base = {"NDCG@10": 0.1389, "Recall@10": 0.1958}  # single LightGCN from paper

    print(f"\n  {'Metric':<15} {'Baseline':>10} {'DCMPT':>10} {'Delta':>10} {'Paper':>10} {'PaperBase':>10}")
    print(f"  {'-'*65}")

    for metric in ["NDCG@10", "Recall@10", "NDCG@20", "Recall@20"]:
        base_val = baseline_metrics.get(metric, 0.0)
        dcmpt_val = dcmpt_metrics.get(metric, 0.0) if dcmpt_metrics else 0.0
        delta = dcmpt_val - base_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        paper_val = paper.get(metric, None)
        paper_str = f"{paper_val:.4f}" if paper_val else "---"
        paper_b = paper_base.get(metric, None)
        paper_b_str = f"{paper_b:.4f}" if paper_b else "---"
        print(f"  {metric:<15} {base_val:>10.4f} {dcmpt_val:>10.4f} "
              f"{delta_str:>10} {paper_str:>10} {paper_b_str:>10}")

    if dcmpt_metrics:
        base_n = baseline_metrics.get("NDCG@10", 0)
        dcmpt_n = dcmpt_metrics.get("NDCG@10", 0)
        if base_n > 0:
            print(f"\n  Prompt improvement: NDCG@10 +{(dcmpt_n-base_n)/base_n*100:.1f}%")
            print(f"  Paper improvement:  NDCG@10 +{(0.3898-0.1389)/0.1389*100:.1f}%")


if __name__ == "__main__":
    main()
