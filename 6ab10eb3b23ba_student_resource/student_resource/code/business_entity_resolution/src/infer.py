import os
import sys
import csv
import time
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    OUT_MATCHING, OUT_CANDIDATES,
    MODEL_PATH, MAX_CANDIDATES_PER_ENTITY, MAX_KEY_BUCKET_SIZE,
    DEFAULT_THRESHOLD
)
from src.preprocess import (
    clean_name, clean_address, extract_numbers, get_combined_blocking_keys
)
from src.features import extract_pair_features

def run_inference(threshold: float = DEFAULT_THRESHOLD, limit_test: int = None):
    print(f"=== Starting Inference Pipeline (Threshold={threshold:.2f}) ===")
    
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        
    print("Loading LightGBM model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    # Ensure output directory exists
    out_dir = os.path.dirname(OUT_MATCHING)
    os.makedirs(out_dir, exist_ok=True)
    
    # Step 1: Detect unique countries in test_source1
    print("Step 1: Discovering countries in test set...")
    countries = set()
    s1_count = 0
    with open(TEST_S1, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            countries.add(row[3].strip())
            s1_count += 1
            if limit_test and s1_count >= limit_test:
                break
                
    print(f"Discovered {len(countries)} countries: {countries} ({s1_count} S1 entities)")
    
    # We open the output files for streaming
    f_match = open(OUT_MATCHING, 'w', encoding='utf-8', newline='')
    f_cand = open(OUT_CANDIDATES, 'w', encoding='utf-8', newline='')
    
    match_writer = csv.writer(f_match, delimiter='\t')
    cand_writer = csv.writer(f_cand, delimiter='\t')
    
    match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
    
    total_processed = 0
    total_matched = 0
    
    # Step 2: Process Country by Country to keep memory usage minimal
    for country in sorted(countries):
        print(f"\n--- Processing Country: {country} ---")
        t0 = time.time()
        
        # Load and index S2 + S3 candidates for THIS country only
        index = defaultdict(list)
        cand_data = {}
        
        for s_file in [TEST_S2, TEST_S3]:
            fname = os.path.basename(s_file)
            print(f"Indexing {fname} for country {country}...")
            with open(s_file, 'r', encoding='utf-8') as f:
                r = csv.reader(f, delimiter='\t')
                next(r)
                for row in r:
                    if row[3].strip() == country:
                        cid = row[0]
                        cn = clean_name(row[1])
                        ca = clean_address(row[2])
                        cand_data[cid] = {
                            'name': cn,
                            'addr': ca,
                            'nums': extract_numbers(ca)
                        }
                        keys = get_combined_blocking_keys(cn, ca)
                        for k in keys:
                            index[k].append(cid)
                            
        print(f"Indexed {len(cand_data)} records for {country} in {time.time()-t0:.2f}s.")
        
        # Now stream S1 entities for THIS country
        print(f"Running candidate generation and matching for {country}...")
        batch_size = 5000
        batch_s1_ids = []
        batch_cands = [] # list of lists of cand_ids
        batch_features = [] # flat list of features
        batch_slices = [] # (start_idx, end_idx) in batch_features
        
        def process_batch():
            nonlocal total_processed, total_matched
            if not batch_s1_ids:
                return
                
            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                preds = model.predict(X_batch)
            else:
                preds = np.array([])
                
            for i, s1_id in enumerate(batch_s1_ids):
                cands_for_s1 = batch_cands[i]
                start_idx, end_idx = batch_slices[i]
                
                if start_idx == end_idx:
                    matched_ids = []
                else:
                    cand_probs = preds[start_idx:end_idx]
                    matched_ids = [cands_for_s1[j] for j, p in enumerate(cand_probs) if p >= threshold]
                    
                # Format output rows (no quoting, tab delimited, comma-separated IDs)
                match_str = ",".join(matched_ids)
                cand_str = ",".join(cands_for_s1)
                
                match_writer.writerow([s1_id, match_str])
                cand_writer.writerow([s1_id, cand_str])
                
                total_processed += 1
                if matched_ids:
                    total_matched += 1
                    
            batch_s1_ids.clear()
            batch_cands.clear()
            batch_features.clear()
            batch_slices.clear()

        with open(TEST_S1, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for row in r:
                if row[3].strip() != country:
                    continue
                if limit_test and total_processed >= limit_test:
                    break
                    
                s1_id = row[0]
                cn = clean_name(row[1])
                ca = clean_address(row[2])
                s1_nums = extract_numbers(ca)
                
                keys = get_combined_blocking_keys(cn, ca)
                
                cand_hits = defaultdict(int)
                for k in keys:
                    if k in index:
                        bucket = index[k]
                        if len(bucket) <= MAX_KEY_BUCKET_SIZE:
                            for cid in bucket:
                                cand_hits[cid] += 1
                                
                if not cand_hits:
                    batch_s1_ids.append(s1_id)
                    batch_cands.append([])
                    batch_slices.append((len(batch_features), len(batch_features)))
                else:
                    sorted_cands = sorted(cand_hits.keys(), key=lambda cid: cand_hits[cid], reverse=True)[:MAX_CANDIDATES_PER_ENTITY]
                    start_pos = len(batch_features)
                    valid_cands = []
                    for cid in sorted_cands:
                        if cid in cand_data:
                            c = cand_data[cid]
                            feats = extract_pair_features(
                                cn, ca, s1_nums,
                                cid, c['name'], c['addr'], c['nums'],
                                cand_hits[cid]
                            )
                            batch_features.append(feats)
                            valid_cands.append(cid)
                            
                    end_pos = len(batch_features)
                    batch_s1_ids.append(s1_id)
                    batch_cands.append(valid_cands)
                    batch_slices.append((start_pos, end_pos))
                    
                if len(batch_s1_ids) >= batch_size:
                    process_batch()
                    if total_processed % 20000 == 0:
                        print(f"  Processed {total_processed} S1 entities... ({total_matched} with matches)")
                        
        process_batch()
        # Free memory of this country's index before loading next country
        del index
        del cand_data

    f_match.close()
    f_cand.close()
    
    print("\n=== Inference Complete ===")
    print(f"Total S1 Entities Processed: {total_processed}")
    print(f"Total Entities with Matches: {total_matched}")
    print(f"Output files saved:")
    print(f"  - {OUT_MATCHING}")
    print(f"  - {OUT_CANDIDATES}")

if __name__ == '__main__':
    run_inference()
