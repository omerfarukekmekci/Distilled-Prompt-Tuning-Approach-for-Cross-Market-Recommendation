"""
data_utils.py  –  Data loading, graph building, node alignment, negative sampling
=================================================================================

This module handles everything related to getting raw XMRec data files into the
format that LightGCN expects:

1.  **Loading** – XMRec provides space-separated text files with columns:
        userId  itemId  rating  date
    We read these with pandas.

2.  **Global ID alignment** – Different markets share some items (same ASIN)
    but have independent user pools.  We remap every user and item to a single
    global integer index so we can build one big combined graph for pre-training.

3.  **Graph construction** – LightGCN operates on a bipartite user-item graph.
    We store it as a sparse adjacency matrix with symmetric normalisation:
        Â = D^{-1/2} · A · D^{-1/2}
    This is the standard GCN normalisation that prevents embeddings from
    diverging during message passing.

4.  **Train / val / test splitting** – For the target market we need held-out
    sets to evaluate.  We use leave-N-out per user.

5.  **Negative sampling** – BPR loss requires (user, pos_item, neg_item)
    triplets.  For every observed edge we uniformly sample one item the user
    has NOT interacted with.
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# 1.  LOAD RAW MARKET DATA
# ---------------------------------------------------------------------------

def k_core_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Recursively filter users and items with strictly less than `k` interactions.
    This shrinks massive, ultra-sparse networks down to their informative core
    and dramatically speeds up training.
    """
    if k <= 1:
        return df

    while True:
        start_len = len(df)
        user_counts = df["userId"].value_counts()
        item_counts = df["itemId"].value_counts()

        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        df = df[df["userId"].isin(valid_users) & df["itemId"].isin(valid_items)]

        if len(df) == start_len:
            break

    return df


def load_market_data(path: str, k_core: int = 5) -> pd.DataFrame:
    """
    Read one XMRec ratings file and apply k-core filtering.

    Parameters
    ----------
    path : str
        Path to a ratings file.  Can be plain .txt or gzipped .txt.gz.
    k_core : int  – Minimum interactions per user/item (default: 5).

    Returns
    -------
    pd.DataFrame
        Columns: userId (str), itemId (str), rating (float), timestamp (float).

    Why string IDs?
    ----------------
    XMRec user/item identifiers are alphanumeric (e.g. Amazon reviewer IDs
    and ASINs).  We keep them as strings here and remap them to integers in
    `build_global_id_maps`.
    """
    df = pd.read_csv(
        path, sep=r'\s+', names=["userId", "itemId", "rating", "timestamp"],
        dtype={"userId": str, "itemId": str, "rating": float, "timestamp": str}
    )
    df = k_core_filter(df, k=k_core)
    return df


# ---------------------------------------------------------------------------
# 2.  GLOBAL ID ALIGNMENT
# ---------------------------------------------------------------------------

def build_global_id_maps(market_dfs: dict):
    """
    Create two mappings  (user → int)  and  (item → int)  that span *all*
    markets.

    Parameters
    ----------
    market_dfs : dict[str, pd.DataFrame]
        key   = market name (e.g. "us", "de")
        value = DataFrame returned by `load_market_data`

    Returns
    -------
    user2id : dict[str, int]
        Maps raw userId strings to global integer IDs.
    item2id : dict[str, int]
        Maps raw itemId strings to global integer IDs.

    Design note
    -----------
    Items can overlap across markets (same ASIN sold in the US and DE),
    so they naturally share the same global ID.  Users, in practice,
    rarely overlap, but we handle it the same way just in case.
    """
    # Collect all unique user IDs and item IDs across every market
    all_users = set()
    all_items = set()
    for df in market_dfs.values():
        all_users.update(df["userId"].unique())
        all_items.update(df["itemId"].unique())

    # Sort for reproducibility, then assign sequential integers starting at 0
    user2id = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    item2id = {iid: idx for idx, iid in enumerate(sorted(all_items))}

    return user2id, item2id


# ---------------------------------------------------------------------------
# 3.  BUILD SPARSE ADJACENCY MATRIX
# ---------------------------------------------------------------------------

def build_adj_matrix(interactions: list, n_users: int, n_items: int):
    """
    Construct the symmetrically normalised adjacency matrix used by LightGCN.

    Parameters
    ----------
    interactions : list of (int, int)
        Each tuple is (global_user_id, global_item_id).
    n_users : int
        Total number of users in the global ID space.
    n_items : int
        Total number of items in the global ID space.

    Returns
    -------
    adj_norm : torch.sparse.FloatTensor
        Shape (n_users + n_items, n_users + n_items).
        This is Â = D^{-1/2} A D^{-1/2}  where A is the bipartite adj matrix
        laid out as:
            [ 0      R ]
            [ R^T    0 ]
        with R being the user-item interaction matrix.

    Why symmetric normalisation?
    ----------------------------
    LightGCN aggregates neighbour embeddings via:
        e_u^{(l+1)} = Σ_i  (1 / √(|N_u| · |N_i|)) · e_i^{(l)}
    This is exactly the operation encoded by Â · E, where E is the full
    embedding matrix.  So multiplying by Â once == one GCN layer, and we
    can simply do  Â^L · E  to get L layers of propagation.
    """
    # ---- Step A: build the raw (un-normalised) adjacency matrix ----
    # We work in the space of (n_users + n_items) nodes.
    # Users occupy indices  [0, n_users)
    # Items occupy indices  [n_users, n_users + n_items)

    user_ids = [u for u, _ in interactions]
    item_ids = [i + n_users for _, i in interactions]   # shift items

    # Symmetric edges:  user→item  AND  item→user
    rows = user_ids + item_ids
    cols = item_ids + user_ids
    vals = np.ones(len(rows), dtype=np.float32)

    n_nodes = n_users + n_items
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

    # ---- Step B: symmetric normalisation  D^{-1/2} A D^{-1/2} ----
    # Compute degree of each node
    degrees = np.array(adj.sum(axis=1)).flatten()  # shape (n_nodes,)
    # Inverse square root;  avoid division by zero for isolated nodes.
    # We compute only on nonzero degrees to avoid the RuntimeWarning
    # that np.power produces on zeros (np.where still evaluates both branches).
    d_inv_sqrt = np.zeros_like(degrees)
    nonzero_mask = degrees > 0
    d_inv_sqrt[nonzero_mask] = np.power(degrees[nonzero_mask], -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # Â
    adj_norm = adj_norm.tocoo()                # ensure COO format for conversion

    # ---- Step C: convert to PyTorch sparse tensor ----
    indices = torch.LongTensor(np.stack([adj_norm.row, adj_norm.col]))
    values  = torch.FloatTensor(adj_norm.data)
    adj_tensor = torch.sparse_coo_tensor(
        indices, values, torch.Size([n_nodes, n_nodes])
    ).coalesce()  # Critical for matrix multiplication performance

    return adj_tensor


# ---------------------------------------------------------------------------
# 4.  BUILD COMBINED GRAPH  (for pre-training)
# ---------------------------------------------------------------------------

def build_combined_graph(market_dfs: dict, user2id: dict, item2id: dict):
    """
    Merge interactions from ALL markets into a single bipartite graph.

    Parameters
    ----------
    market_dfs : dict[str, pd.DataFrame]
    user2id, item2id : global ID mappings

    Returns
    -------
    adj : torch.sparse.FloatTensor
        The combined normalised adjacency matrix.
    interactions : list of (int, int)
        All (user, item) pairs used (useful for negative sampling later).

    Why combine?
    ------------
    The pre-training phase wants the backbone to see as many collaborative
    signals as possible so it learns *universal* user-item patterns that
    generalise across markets.  Combining everything into one graph is the
    standard approach in DCMPT.
    """
    n_users = len(user2id)
    n_items = len(item2id)

    all_interactions = []
    for df in market_dfs.values():
        for _, row in df.iterrows():
            uid = user2id[row["userId"]]
            iid = item2id[row["itemId"]]
            all_interactions.append((uid, iid))

    # Deduplicate – a user may have rated the same item in multiple markets
    all_interactions = list(set(all_interactions))

    adj = build_adj_matrix(all_interactions, n_users, n_items)
    return adj, all_interactions


# ---------------------------------------------------------------------------
# 5.  BUILD SINGLE-MARKET GRAPH  (for teacher or evaluation)
# ---------------------------------------------------------------------------

def build_market_graph(df: pd.DataFrame, user2id: dict, item2id: dict):
    """
    Build a graph from ONE market's interactions.

    This is used for:
    •  The teacher model  → trained exclusively on target market data
    •  Evaluation         → we test on target market edges

    Parameters
    ----------
    df : pd.DataFrame
        One market's ratings.
    user2id, item2id : global ID mappings

    Returns
    -------
    adj : torch.sparse.FloatTensor
    interactions : list of (int, int)
    """
    n_users = len(user2id)
    n_items = len(item2id)

    interactions = []
    for _, row in df.iterrows():
        uid = user2id[row["userId"]]
        iid = item2id[row["itemId"]]
        interactions.append((uid, iid))

    interactions = list(set(interactions))
    adj = build_adj_matrix(interactions, n_users, n_items)
    return adj, interactions


# ---------------------------------------------------------------------------
# 6.  TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------

def train_val_test_split(interactions: list, val_ratio: float = 0.1,
                         test_ratio: float = 0.1, seed: int = 42):
    """
    Split interactions into train / val / test per user.

    Strategy: for each user, hold out `test_ratio` of their items for testing
    and `val_ratio` for validation.  The rest go to training.  If a user has
    fewer than 3 interactions, all go to training (to avoid empty sets).

    Parameters
    ----------
    interactions : list of (user, item)
    val_ratio, test_ratio : float
    seed : int

    Returns
    -------
    train_interactions : list of (user, item)
    val_dict  : dict[int, list[int]]   user → list of held-out items
    test_dict : dict[int, list[int]]   user → list of held-out items

    Why per-user?
    -------------
    Random global splitting can leave some users with NO training edges,
    giving the model nothing to learn from for those users.  Per-user
    splitting guarantees every user retains some training signal.
    """
    rng = random.Random(seed)

    # Group interactions by user
    user_items = defaultdict(list)
    for u, i in interactions:
        user_items[u].append(i)

    train_interactions = []
    val_dict  = defaultdict(list)
    test_dict = defaultdict(list)

    for u, items in user_items.items():
        rng.shuffle(items)
        n = len(items)

        if n < 3:
            # Too few interactions – keep all for training
            for i in items:
                train_interactions.append((u, i))
            continue

        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))

        test_items = items[:n_test]
        val_items  = items[n_test : n_test + n_val]
        train_items = items[n_test + n_val:]

        for i in train_items:
            train_interactions.append((u, i))
        val_dict[u]  = val_items
        test_dict[u] = test_items

    return train_interactions, dict(val_dict), dict(test_dict)


# ---------------------------------------------------------------------------
# 7.  NEGATIVE SAMPLING
# ---------------------------------------------------------------------------

def sample_negative(user: int, n_items: int, positive_set: set) -> int:
    """
    Uniformly sample one item that `user` has NOT interacted with.

    Parameters
    ----------
    user : int          (unused – kept for API clarity)
    n_items : int       Total number of items in the global space.
    positive_set : set  Set of item IDs the user HAS interacted with.

    Returns
    -------
    neg_item : int

    Why uniform sampling?
    ---------------------
    BPR's theoretical guarantee holds under uniform negative sampling.
    More sophisticated strategies (popularity-biased, hard negatives) can
    help in practice but add complexity.  The paper uses uniform sampling.
    """
    while True:
        neg = random.randint(0, n_items - 1)
        if neg not in positive_set:
            return neg


def build_bpr_triplets(interactions: list, n_items: int,
                       user_pos_items: dict = None):
    """
    Generate (user, pos_item, neg_item) triplets for one epoch of BPR training.

    Parameters
    ----------
    interactions : list of (user, item)
    n_items : int
    user_pos_items : dict[int, set]  –  pre-computed positive sets per user.
        If None, it is built on the fly.

    Returns
    -------
    triplets : list of (int, int, int)
    """
    if user_pos_items is None:
        user_pos_items = defaultdict(set)
        for u, i in interactions:
            user_pos_items[u].add(i)

    triplets = []
    for u, pos_i in interactions:
        neg_i = sample_negative(u, n_items, user_pos_items[u])
        triplets.append((u, pos_i, neg_i))

    return triplets


# ---------------------------------------------------------------------------
# 8.  CONVENIENCE: LOAD EVERYTHING AT ONCE
# ---------------------------------------------------------------------------

def _find_ratings_file(data_dir: str, market: str, category: str) -> str:
    """
    Locate the ratings file for a given market and category.

    Supports two directory layouts:

    Layout A  (your XMRec folder):
        xmrec/<market>/raw/<category>/ratings_<market>_<category>.txt.gz

    Layout B  (flat folder, legacy):
        data/ratings_<market>.txt   or   data/ratings_<market>.txt.gz

    Parameters
    ----------
    data_dir : str   – root data directory (e.g. "xmrec/")
    market   : str   – e.g. "us", "de"
    category : str   – e.g. "Electronics"  (ignored for Layout B)

    Returns
    -------
    fpath : str   – absolute path to the ratings file

    Raises
    ------
    FileNotFoundError if no matching file is found.
    """
    # --- Layout A: xmrec/<market>/raw/<category>/ratings_<market>_<category>.<ext> ---
    for ext in [".txt.gz", ".txt"]:
        fpath = os.path.join(
            data_dir, market, "raw", category,
            f"ratings_{market}_{category}{ext}"
        )
        if os.path.exists(fpath):
            return fpath

    # --- Layout B (legacy flat): data/ratings_<market>.<ext> ---
    for ext in [".txt.gz", ".txt"]:
        fpath = os.path.join(data_dir, f"ratings_{market}{ext}")
        if os.path.exists(fpath):
            return fpath

    raise FileNotFoundError(
        f"Could not find ratings file for market='{market}', "
        f"category='{category}' in {data_dir}.\n"
        f"  Tried:  {data_dir}/{market}/raw/{category}/ratings_{market}_{category}.txt.gz\n"
        f"  Also:   {data_dir}/ratings_{market}.txt(.gz)"
    )


def load_all_markets(data_dir: str, source_markets: list, target_market: str,
                     category: str = "Electronics"):
    """
    End-to-end data loading used by main.py.

    Parameters
    ----------
    data_dir : str
        Root data directory.  For the XMRec layout this should be the
        path to the ``xmrec/`` folder, e.g.::

            xmrec/
              us/raw/Electronics/ratings_us_Electronics.txt.gz
              de/raw/Electronics/ratings_de_Electronics.txt.gz
              ...

    source_markets : list of str
        E.g. ["us", "uk"]
    target_market : str
        E.g. "de"
    category : str
        Product category to load, e.g. "Electronics".
        The DCMPT paper evaluates per-category because:
        •  Items (ASINs) are shared across markets within a category,
           enabling meaningful cross-market transfer.
        •  Mixing all categories would create an enormous graph and
           dilute the cross-market signal with cross-domain noise.

    Returns
    -------
    A dict with keys:
        "user2id", "item2id",
        "n_users", "n_items",
        "combined_adj", "combined_interactions",
        "target_adj_train", "target_train_interactions",
        "target_val", "target_test",
        "target_full_interactions",
        "market_dfs"
    """
    print(f"  Category: {category}")

    # ---- Load each market file ----
    market_dfs = {}
    all_market_names = source_markets + [target_market]

    for m in all_market_names:
        fpath = _find_ratings_file(data_dir, m, category)
        market_dfs[m] = load_market_data(fpath)
        print(f"  Loaded {m}: {len(market_dfs[m]):,} interactions from {fpath}")

    # ---- Build global ID maps ----
    user2id, item2id = build_global_id_maps(market_dfs)
    n_users = len(user2id)
    n_items = len(item2id)
    print(f"  Global space: {n_users:,} users, {n_items:,} items")

    # ---- Combined graph for pre-training ----
    combined_adj, combined_interactions = build_combined_graph(
        market_dfs, user2id, item2id
    )
    print(f"  Combined graph: {len(combined_interactions):,} edges")

    # ---- Target market: split and build training graph ----
    target_df = market_dfs[target_market]
    _, target_full_interactions = build_market_graph(target_df, user2id, item2id)

    train_inter, val_dict, test_dict = train_val_test_split(
        target_full_interactions
    )
    target_adj_train = build_adj_matrix(train_inter, n_users, n_items)
    print(f"  Target train: {len(train_inter):,} edges, "
          f"val users: {len(val_dict):,}, test users: {len(test_dict):,}")

    return {
        "user2id": user2id,
        "item2id": item2id,
        "n_users": n_users,
        "n_items": n_items,
        "combined_adj": combined_adj,
        "combined_interactions": combined_interactions,
        "target_adj_train": target_adj_train,
        "target_train_interactions": train_inter,
        "target_val": val_dict,
        "target_test": test_dict,
        "target_full_interactions": target_full_interactions,
        "market_dfs": market_dfs,
    }