#!/usr/bin/env python3
"""
Link Prediction Utility 


Use:
    python utility_robust.py \
        --baseline-graph /path/to/original.tsv \
        --defended-graph /path/to/defended.tsv \
        --test /path/to/test.tsv \
        --target-relation has_careplan \
        --output result.json
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# FIXED SEEDS
# ═══════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_triples(path, has_header=False):
    """Load triples from TSV"""
    if has_header:
        df = pd.read_csv(path, sep="\t", header=0, names=["h", "r", "t"], dtype=str)
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=["h", "r", "t"], dtype=str)
    
    df = df.dropna()
    df["h"] = df["h"].astype(str)
    df["r"] = df["r"].astype(str)
    df["t"] = df["t"].astype(str)
    
    return [(row.h, row.r, row.t) for row in df.itertuples(index=False)]


def build_fixed_vocab(all_triples_list):
    """
    Build FIXED vocabulary from ALL graphs (baseline + defended + test)
    
    CRITICAL: Le vocabulaire DOIT être le même pour baseline et défenses
    sinon les embeddings ne sont pas comparables!
    """
    entities = set()
    relations = set()
    
    for triples in all_triples_list:
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
    
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    
    return ent2id, rel2id


def triples_to_ids(triples, ent2id, rel2id, skip_unknown=True):
    """Convert triples to IDs"""
    id_triples = []
    skipped = 0
    
    for h, r, t in triples:
        if skip_unknown and (h not in ent2id or r not in rel2id or t not in ent2id):
            skipped += 1
            continue
        
        h_id = ent2id.get(h, -1)
        r_id = rel2id.get(r, -1)
        t_id = ent2id.get(t, -1)
        
        if h_id != -1 and r_id != -1 and t_id != -1:
            id_triples.append((h_id, r_id, t_id))
    
    if skipped > 0:
        print(f"  Skipped {skipped:,} triples with unknown entities/relations")
    
    return id_triples


# ═══════════════════════════════════════════════════════════════
# MODEL - TransE with PROPER NORMALIZATION
# ═══════════════════════════════════════════════════════════════
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = torch.nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def forward(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
    
    def normalize_embeddings(self):
        """Normalize entity embeddings after each epoch"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = torch.nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )


class TripleDataset(Dataset):
    def __init__(self, triples, num_entities, seed=42):
        self.triples = triples
        self.num_entities = num_entities
        self.rng = np.random.RandomState(seed)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Corrupt head or tail with 50% probability
        if self.rng.rand() < 0.5:
            t_neg = self.rng.randint(0, self.num_entities)
            return h, r, t, h, r, t_neg
        else:
            h_neg = self.rng.randint(0, self.num_entities)
            return h, r, t, h_neg, r, t


# ═══════════════════════════════════════════════════════════════
# TRAINING with EARLY STOPPING
# ═══════════════════════════════════════════════════════════════
def train_transe(
    train_triples,
    val_triples,
    num_entities,
    num_relations,
    embedding_dim=100,
    margin=1.0,
    epochs=200,
    batch_size=1024,
    lr=0.01,
    patience=10,
):
    """
    Train TransE with early stopping
    
    Args:
        patience: Number of epochs to wait before stopping if no improvement
    """
    model = TransE(num_entities, num_relations, embedding_dim, margin).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TripleDataset(train_triples, num_entities, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TripleDataset(val_triples, num_entities, seed=SEED+1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training TransE:")
    print(f"    Train: {len(train_triples):,} triples")
    print(f"    Val:   {len(val_triples):,} triples")
    print(f"    Entities: {num_entities:,}, Relations: {num_relations}")
    print(f"    Embedding dim: {embedding_dim}, Epochs: {epochs}, Batch: {batch_size}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            h, r, t, h_neg, r_neg, t_neg = [b.to(DEVICE) for b in batch]
            
            pos_score = model(h, r, t)
            neg_score = model(h_neg, r_neg, t_neg)
            
            loss = torch.mean(torch.relu(margin + pos_score - neg_score))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            model.normalize_embeddings()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / max(len(train_loader), 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                h, r, t, h_neg, r_neg, t_neg = [b.to(DEVICE) for b in batch]
                pos_score = model(h, r, t)
                neg_score = model(h_neg, r_neg, t_neg)
                loss = torch.mean(torch.relu(margin + pos_score - neg_score))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Patience={patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"    Loaded best model (val_loss={best_val_loss:.4f})")
    
    return model


# ═══════════════════════════════════════════════════════════════
# EVALUATION - FILTERED SETTING
# ═══════════════════════════════════════════════════════════════
def evaluate_link_prediction(
    model,
    test_triples,
    train_val_triples_set,
    num_entities,
    batch_size=1000,
):
    """
    Evaluate link prediction in FILTERED setting
    
    CRITICAL: Remove all train+val triples from candidates during ranking
    """
    model.eval()
    
    ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    
    print(f"  Evaluating on {len(test_triples):,} test triples (filtered)...")
    
    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples):
            if (i + 1) % 500 == 0:
                print(f"    Progress: {i+1}/{len(test_triples)}")
            
            # Score all entities as tail
            h_tensor = torch.full((num_entities,), h, dtype=torch.long, device=DEVICE)
            r_tensor = torch.full((num_entities,), r, dtype=torch.long, device=DEVICE)
            t_candidates = torch.arange(num_entities, device=DEVICE)
            
            scores = model(h_tensor, r_tensor, t_candidates).cpu().numpy()
            
            # Filter out train+val triples
            for j in range(num_entities):
                if j != t and (h, r, j) in train_val_triples_set:
                    scores[j] = np.inf
            
            # Rank
            sorted_indices = np.argsort(scores)
            rank = np.where(sorted_indices == t)[0][0] + 1
            
            ranks.append(rank)
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 10:
                hits_at_10 += 1
    
    num_test = len(test_triples)
    
    metrics = {
        "hits@1": hits_at_1 / num_test if num_test else 0.0,
        "hits@3": hits_at_3 / num_test if num_test else 0.0,
        "hits@10": hits_at_10 / num_test if num_test else 0.0,
        "mrr": float(np.mean(1.0 / np.array(ranks))) if ranks else 0.0,
        "mr": float(np.mean(ranks)) if ranks else 0.0,
        "num_test_triples": num_test,
    }
    
    print(f"\n  Results:")
    print(f"    Hits@1:  {metrics['hits@1']:.4f}")
    print(f"    Hits@3:  {metrics['hits@3']:.4f}")
    print(f"    Hits@10: {metrics['hits@10']:.4f}")
    print(f"    MRR:     {metrics['mrr']:.4f}")
    print(f"    MR:      {metrics['mr']:.1f}")
    
    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════
def evaluate_utility(
    baseline_graph_path,
    defended_graph_path,
    test_path,
    target_relation,
    embedding_dim=100,
    epochs=200,
    batch_size=1024,
    lr=0.01,
    patience=10,
    val_fraction=0.1,
    has_header=False,
):

    
    print("\n" + "="*70)
    print("  ROBUST UTILITY EVALUATION")
    print("="*70)
    
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════
    # Step 1: Load all data
    # ═══════════════════════════════════════════════════════════
    print("\n[1/6] Loading data...")
    baseline_triples = load_triples(baseline_graph_path, has_header)
    defended_triples = load_triples(defended_graph_path, has_header)
    test_triples_all = load_triples(test_path, has_header)
    
    print(f"  Baseline:  {len(baseline_triples):,} triples")
    print(f"  Defended:  {len(defended_triples):,} triples")
    print(f"  Test (all): {len(test_triples_all):,} triples")
    
    # Filter test to target relation only
    test_triples = [tr for tr in test_triples_all if tr[1] == target_relation]
    print(f"  Test (target '{target_relation}'): {len(test_triples):,} triples")
    
    if len(test_triples) == 0:
        raise ValueError(f"No test triples for target relation: {target_relation}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 2: Build FIXED vocabulary
    # ═══════════════════════════════════════════════════════════
    print("\n[2/6] Building FIXED vocabulary...")
    ent2id, rel2id = build_fixed_vocab([baseline_triples, defended_triples, test_triples])
    
    print(f"  Entities:  {len(ent2id):,}")
    print(f"  Relations: {len(rel2id)}")
    
    if target_relation not in rel2id:
        raise ValueError(f"Target relation not in vocab: {target_relation}")
    
    # Convert to IDs
    baseline_ids = triples_to_ids(baseline_triples, ent2id, rel2id)
    defended_ids = triples_to_ids(defended_triples, ent2id, rel2id)
    test_ids = triples_to_ids(test_triples, ent2id, rel2id, skip_unknown=True)
    
    # ═══════════════════════════════════════════════════════════
    # Step 3: Split baseline into train/val
    # ═══════════════════════════════════════════════════════════
    print("\n[3/6] Splitting baseline into train/val...")
    n_val = int(len(baseline_ids) * val_fraction)
    
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(baseline_ids))
    
    baseline_train_ids = [baseline_ids[i] for i in perm[n_val:]]
    baseline_val_ids = [baseline_ids[i] for i in perm[:n_val]]
    
    print(f"  Baseline train: {len(baseline_train_ids):,}")
    print(f"  Baseline val:   {len(baseline_val_ids):,}")
    
    # Similarly for defended
    n_val_def = int(len(defended_ids) * val_fraction)
    perm_def = rng.permutation(len(defended_ids))
    
    defended_train_ids = [defended_ids[i] for i in perm_def[n_val_def:]]
    defended_val_ids = [defended_ids[i] for i in perm_def[:n_val_def]]
    
    print(f"  Defended train: {len(defended_train_ids):,}")
    print(f"  Defended val:   {len(defended_val_ids):,}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 4: Train on BASELINE
    # ═══════════════════════════════════════════════════════════
    print("\n[4/6] Training on BASELINE...")
    baseline_model = train_transe(
        train_triples=baseline_train_ids,
        val_triples=baseline_val_ids,
        num_entities=len(ent2id),
        num_relations=len(rel2id),
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    
    # ═══════════════════════════════════════════════════════════
    # Step 5: Evaluate BASELINE
    # ═══════════════════════════════════════════════════════════
    print("\n[5/6] Evaluating BASELINE...")
    baseline_train_val_set = set(baseline_train_ids + baseline_val_ids)
    baseline_metrics = evaluate_link_prediction(
        model=baseline_model,
        test_triples=test_ids,
        train_val_triples_set=baseline_train_val_set,
        num_entities=len(ent2id),
    )
    
    # ═══════════════════════════════════════════════════════════
    # Step 6: Train on DEFENDED
    # ═══════════════════════════════════════════════════════════
    print("\n[6/6] Training on DEFENDED...")
    defended_model = train_transe(
        train_triples=defended_train_ids,
        val_triples=defended_val_ids,
        num_entities=len(ent2id),
        num_relations=len(rel2id),
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    
    # ═══════════════════════════════════════════════════════════
    # Step 7: Evaluate DEFENDED
    # ═══════════════════════════════════════════════════════════
    print("\n[7/7] Evaluating DEFENDED...")
    defended_train_val_set = set(defended_train_ids + defended_val_ids)
    defended_metrics = evaluate_link_prediction(
        model=defended_model,
        test_triples=test_ids,
        train_val_triples_set=defended_train_val_set,
        num_entities=len(ent2id),
    )
    
    # ═══════════════════════════════════════════════════════════
    # Compare results
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("  COMPARISON")
    print("="*70)
    print(f"\n  BASELINE:")
    print(f"    Hits@10: {baseline_metrics['hits@10']:.4f}")
    print(f"    MRR:     {baseline_metrics['mrr']:.4f}")
    
    print(f"\n  DEFENDED:")
    print(f"    Hits@10: {defended_metrics['hits@10']:.4f}")
    print(f"    MRR:     {defended_metrics['mrr']:.4f}")
    
    mrr_ratio = defended_metrics['mrr'] / max(baseline_metrics['mrr'], 1e-10)
    h10_ratio = defended_metrics['hits@10'] / max(baseline_metrics['hits@10'], 1e-10)
    
    print(f"\n  RATIOS (Defended / Baseline):")
    print(f"    MRR ratio:     {mrr_ratio:.4f}")
    print(f"    Hits@10 ratio: {h10_ratio:.4f}")
    
    if mrr_ratio > 1.05:
        print(f"\n  ⚠️  WARNING: Utility INCREASED by {(mrr_ratio-1)*100:.1f}%!")
        print(f"     This suggests the defense is removing noise/low-quality edges.")
    elif mrr_ratio < 0.95:
        print(f"\n  ✅ Utility DECREASED by {(1-mrr_ratio)*100:.1f}% as expected.")
    else:
        print(f"\n  ≈  Utility roughly unchanged (±5%)")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    print("="*70 + "\n")
    
    return {
        "baseline": baseline_metrics,
        "defended": defended_metrics,
        "mrr_ratio": float(mrr_ratio),
        "hits10_ratio": float(h10_ratio),
        "execution_time_sec": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Robust utility evaluation")
    
    parser.add_argument("--baseline-graph", required=True, help="Baseline (original) graph")
    parser.add_argument("--defended-graph", required=True, help="Defended graph")
    parser.add_argument("--test", required=True, help="Test set")
    parser.add_argument("--target-relation", required=True, help="Target relation to evaluate")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--has-header", action="store_true")
    
    args = parser.parse_args()
    
    results = evaluate_utility(
        baseline_graph_path=args.baseline_graph,
        defended_graph_path=args.defended_graph,
        test_path=args.test,
        target_relation=args.target_relation,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_fraction=args.val_fraction,
        has_header=args.has_header,
    )
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()