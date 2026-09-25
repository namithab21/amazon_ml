import os
import sys
import csv
import time
import random
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    MODEL_PATH, MAX_CANDIDATES_PER_ENTITY, MAX_KEY_BUCKET_SIZE
)
from src.preprocess import (
    clean_name, clean_address, extract_numbers, get_combined_blocking_keys
)
from src.features import extract_pair_features, FEATURE_NAMES

def compute_macro_f05(gt_mapping: dict, pred_mapping: dict) -> float:
    """
    Compute competition Macro F_0.5 score:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - true empty & pred empty -> 1.0
      - true empty & pred non-empty -> 0.0
    """
    scores = []
    for s1_id, true_set in gt_mapping.items():
        pred_set = pred_mapping.get(s1_id, set())
        
        # Singleton case
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
            
        if not pred_set:
            scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
            
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        
        denom = (0.25 * precision + recall)
        if denom > 0:
            f05 = (1.25 * precision * recall) / denom
        else:
            f05 = 0.0
        scores.append(f05)
        
    return float(np.mean(scores))

def train_pipeline(num_s1_samples: int = 30000):
    print(f"=== Starting Model Training Pipeline with {num_s1_samples} S1 entities ===")
    
    # 1. Load Ground Truth sample
    print("Step 1: Reading ground truth...")
    gt = {}
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= num_s1_samples:
                break
            s1_id = row[0]
            matches = set(row[1].strip().split(',')) if row[1].strip() else set()
            gt[s1_id] = matches
            
    print(f"Loaded {len(gt)} S1 records from ground truth.")
    all_needed_matches = set()
    for m in gt.values():
        all_needed_matches.update(m)
        
    # Split S1 into Train (80%) and Validation (20%)
    s1_ids = list(gt.keys())
    train_s1_ids, val_s1_ids = train_test_split(s1_ids, test_size=0.20, random_state=42)
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    print(f"Train split: {len(train_s1_set)} S1 entities | Val split: {len(val_s1_set)} S1 entities")

    # 2. Load corresponding S1 records
    print("Step 2: Loading S1 records and preprocessing...")
    s1_data = {}
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            s1_id = row[0]
            if s1_id in gt:
                cn = clean_name(row[1])
                ca = clean_address(row[2])
                s1_data[s1_id] = {
                    'name': cn,
                    'addr': ca,
                    'nums': extract_numbers(ca),
                    'country': row[3]
                }
                if len(s1_data) == len(gt):
                    break

    # 3. Load S2 + S3 candidates & build inverted index
    print("Step 3: Indexing S2 + S3 records...")
    t0 = time.time()
    cand_data = {}
    index = defaultdict(lambda: defaultdict(list))
    
    # We load all ground-truth matches + additional candidate pool to simulate realistic noise
    for s_file in [TRAIN_S2, TRAIN_S3]:
        with open(s_file, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for i, row in enumerate(r):
                cid = row[0]
                # Include true matches or background records up to 150,000 per file
                if cid in all_needed_matches or i < 150000:
                    cn = clean_name(row[1])
                    ca = clean_address(row[2])
                    country = row[3]
                    cand_data[cid] = {
                        'name': cn,
                        'addr': ca,
                        'nums': extract_numbers(ca),
                        'country': country
                    }
                    keys = get_combined_blocking_keys(cn, ca)
                    for k in keys:
                        index[country][k].append(cid)

    print(f"Index built with {len(cand_data)} records in {time.time()-t0:.2f}s.")

    # 4. Generate Training Pairs & Features
    print("Step 4: Generating feature vectors for training pairs...")
    X_train, y_train = [], []
    
    for s1_id in train_s1_set:
        s1 = s1_data[s1_id]
        country = s1['country']
        keys = get_combined_blocking_keys(s1['name'], s1['addr'])
        c_index = index[country]
        
        # Collect candidate hits
        cand_hits = defaultdict(int)
        for k in keys:
            if k in c_index:
                bucket = c_index[k]
                if len(bucket) <= MAX_KEY_BUCKET_SIZE:
                    for cid in bucket:
                        cand_hits[cid] += 1
                        
        true_matches = gt[s1_id]
        
        # Add all true matches as positive examples
        for tid in true_matches:
            if tid in cand_data:
                c = cand_data[tid]
                hits = cand_hits.get(tid, 1)
                feats = extract_pair_features(
                    s1['name'], s1['addr'], s1['nums'],
                    tid, c['name'], c['addr'], c['nums'],
                    hits
                )
                X_train.append(feats)
                y_train.append(1)
                
        # Add hard negatives (candidates with high hits that are NOT true matches)
        neg_candidates = [cid for cid in cand_hits if cid not in true_matches]
        # Sort negatives by hits to pick the hardest negatives
        neg_candidates.sort(key=lambda cid: cand_hits[cid], reverse=True)
        # Sample up to 3 hard negatives per entity
        for nid in neg_candidates[:3]:
            c = cand_data[nid]
            hits = cand_hits[nid]
            feats = extract_pair_features(
                s1['name'], s1['addr'], s1['nums'],
                nid, c['name'], c['addr'], c['nums'],
                hits
            )
            X_train.append(feats)
            y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Training dataset: {len(X_train)} pairs (Positives: {np.sum(y_train)}, Negatives: {len(y_train)-np.sum(y_train)})")

    # 5. Train LightGBM Model
    print("Step 5: Training LightGBM classifier...")
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=250)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Feature Importance
    importances = model.feature_importance(importance_type='gain')
    print("\nFeature Importances (gain):")
    for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"  {fname:20s}: {imp:.2f}")

    # 6. Evaluate and Calibrate F_0.5 Threshold on Validation Set
    print("\nStep 6: Running validation and tuning F_0.5 threshold...")
    val_candidates = {}
    val_pair_features = []
    val_pair_keys = [] # list of (s1_id, cid)
    
    for s1_id in val_s1_set:
        s1 = s1_data[s1_id]
        country = s1['country']
        keys = get_combined_blocking_keys(s1['name'], s1['addr'])
        c_index = index[country]
        
        cand_hits = defaultdict(int)
        for k in keys:
            if k in c_index:
                bucket = c_index[k]
                if len(bucket) <= MAX_KEY_BUCKET_SIZE:
                    for cid in bucket:
                        cand_hits[cid] += 1
                        
        if not cand_hits:
            continue
            
        # Select top candidates
        sorted_cands = sorted(cand_hits.keys(), key=lambda cid: cand_hits[cid], reverse=True)[:MAX_CANDIDATES_PER_ENTITY]
        for cid in sorted_cands:
            if cid in cand_data:
                c = cand_data[cid]
                feats = extract_pair_features(
                    s1['name'], s1['addr'], s1['nums'],
                    cid, c['name'], c['addr'], c['nums'],
                    cand_hits[cid]
                )
                val_pair_features.append(feats)
                val_pair_keys.append((s1_id, cid))

    if val_pair_features:
        X_val = np.array(val_pair_features, dtype=np.float32)
        val_preds = model.predict(X_val)
        
        # Organize predictions by s1_id -> list of (cid, prob)
        preds_by_s1 = defaultdict(list)
        for (s1_id, cid), prob in zip(val_pair_keys, val_preds):
            preds_by_s1[s1_id].append((cid, prob))
            
        val_gt = {s1_id: gt[s1_id] for s1_id in val_s1_set}
        
        # Sweep threshold to maximize Macro F_0.5
        best_thresh = 0.50
        best_f05 = -1.0
        print("\nThreshold Sweep on Holdout Validation Set:")
        for thresh in np.arange(0.40, 0.95, 0.05):
            pred_mapping = {}
            for s1_id in val_s1_set:
                matched = {cid for cid, prob in preds_by_s1[s1_id] if prob >= thresh}
                pred_mapping[s1_id] = matched
            score = compute_macro_f05(val_gt, pred_mapping)
            print(f"  Threshold {thresh:.2f} -> Macro F_0.5 = {score:.4f}")
            if score > best_f05:
                best_f05 = score
                best_thresh = thresh
                
        print(f"\n>>> OPTIMAL THRESHOLD: {best_thresh:.2f} with Macro F_0.5 = {best_f05:.4f} <<<")
        return best_thresh, best_f05
    else:
        print("No validation candidates generated.")
        return 0.65, 0.0

if __name__ == '__main__':
    train_pipeline(num_s1_samples=30000)
