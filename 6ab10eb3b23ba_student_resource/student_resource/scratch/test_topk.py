import csv
import sys
import time
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.path.append('code/business_entity_resolution')
from src.preprocess import (
    clean_name, clean_address,
    get_combined_blocking_keys
)

def benchmark_top_k():
    print("Loading 2,000 ground truth rows for Top-K ranking test...")
    gt = {}
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for i, row in enumerate(r):
            if i >= 2000: break
            if row[1].strip():
                gt[row[0]] = set(row[1].strip().split(','))

    needed_s1 = set(gt.keys())
    needed_matches = set()
    for m in gt.values():
        needed_matches.update(m)

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
                if len(s1_records) == len(needed_s1): break

    index = defaultdict(lambda: defaultdict(list))
    cand_records = {}
    for s_file in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        with open(s_file, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for i, row in enumerate(r):
                cid = row[0]
                if i < 50000 or cid in needed_matches:
                    cn = clean_name(row[1])
                    ca = clean_address(row[2])
                    country = row[3]
                    cand_records[cid] = {'name': cn, 'addr': ca, 'country': country}
                    keys = get_combined_blocking_keys(cn, ca)
                    for k in keys:
                        index[country][k].append(cid)

    for K in [5, 10, 15, 20]:
        total_true = sum(len(m) for m in gt.values())
        found_true = 0
        for s1_id, data in s1_records.items():
            country = data['country']
            keys = get_combined_blocking_keys(data['name'], data['addr'])
            c_index = index[country]
            
            # Count key overlaps
            cand_hits = defaultdict(int)
            for k in keys:
                if k in c_index:
                    bucket = c_index[k]
                    if len(bucket) <= 500:
                        for cid in bucket:
                            cand_hits[cid] += 1
                            
            if not cand_hits:
                continue
                
            # If candidates <= K, keep all
            if len(cand_hits) <= K:
                top_cands = set(cand_hits.keys())
            else:
                # Fast score top items by key overlap + name similarity
                scored = []
                for cid, hits in cand_hits.items():
                    c_name = cand_records[cid]['name']
                    # Rapid name ratio
                    sim = fuzz.token_sort_ratio(data['name'], c_name)
                    score = hits * 100 + sim
                    scored.append((score, cid))
                scored.sort(reverse=True)
                top_cands = {cid for _, cid in scored[:K]}
                
            found_true += len(top_cands.intersection(gt[s1_id]))
            
        print(f"Top-{K}: Recall = {found_true / total_true * 100:.2f}% ({found_true}/{total_true})")

if __name__ == '__main__':
    benchmark_top_k()
