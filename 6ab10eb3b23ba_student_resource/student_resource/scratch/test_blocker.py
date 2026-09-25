import csv
import sys
import time
from collections import defaultdict

sys.path.append('code/business_entity_resolution')
from src.preprocess import (
    clean_name, clean_address,
    get_combined_blocking_keys
)

def benchmark_blocking():
    print("Loading 10,000 ground truth rows...")
    gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 10000:
                break
            if row[1].strip():
                gt[row[0]] = set(row[1].strip().split(','))

    print(f"Loaded {len(gt)} non-empty ground truth S1 entities to evaluate.")
    needed_s1 = set(gt.keys())
    needed_matches = set()
    for m in gt.values():
        needed_matches.update(m)

    print("Loading corresponding S1 records...")
    s1_records = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in needed_s1:
                s1_records[row[0]] = {
                    'name': clean_name(row[1]),
                    'addr': clean_address(row[2]),
                    'country': row[3]
                }
                if len(s1_records) == len(needed_s1):
                    break

    print("Building inverted index over sample S2 + S3 records...")
    # Index: country -> key -> list of cand_ids
    index = defaultdict(lambda: defaultdict(list))
    cand_records = {}

    t0 = time.time()
    for s_file in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(s_file, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            # To simulate a realistic search space, load 100,000 records from S2 and S3
            # plus all true matches
            for i, row in enumerate(r):
                cid = row[0]
                if i < 100000 or cid in needed_matches:
                    cn = clean_name(row[1])
                    ca = clean_address(row[2])
                    country = row[3]
                    cand_records[cid] = {'name': cn, 'addr': ca, 'country': country}
                    
                    keys = get_combined_blocking_keys(cn, ca)
                    for k in keys:
                        index[country][k].append(cid)

    print(f"Index built in {time.time()-t0:.2f}s with {len(cand_records)} candidate records.")

    print("Evaluating recall ceiling on S1 entities...")
    total_true_links = sum(len(m) for m in gt.values())
    found_true_links = 0
    candidate_counts = []

    for s1_id, data in s1_records.items():
        country = data['country']
        keys = get_combined_blocking_keys(data['name'], data['addr'])
        
        # Gather all candidates matching ANY key
        cands = set()
        c_index = index[country]
        for k in keys:
            if k in c_index:
                # Cap bucket size if key is too generic (> 500 items)
                bucket = c_index[k]
                if len(bucket) <= 500:
                    cands.update(bucket)
                    
        candidate_counts.append(len(cands))
        true_for_s1 = gt[s1_id]
        found_true_links += len(cands.intersection(true_for_s1))

    recall = found_true_links / total_true_links * 100
    avg_cands = sum(candidate_counts) / len(candidate_counts)
    print(f"--- BLOCKING BENCHMARK RESULTS ---")
    print(f"Total True Links: {total_true_links}")
    print(f"Found True Links: {found_true_links}")
    print(f"Recall Ceiling: {recall:.2f}%")
    print(f"Average Candidates per Entity: {avg_cands:.2f}")

if __name__ == '__main__':
    benchmark_blocking()
