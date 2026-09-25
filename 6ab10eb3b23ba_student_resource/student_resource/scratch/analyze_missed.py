import csv
import sys
from collections import defaultdict

sys.path.append('code/business_entity_resolution')
from src.preprocess import (
    clean_name, clean_address,
    get_name_blocking_keys, get_address_blocking_keys
)

gt = {}
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for i, row in enumerate(r):
        if i >= 1000: break
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
            s1_records[row[0]] = row
            if len(s1_records) == len(needed_s1): break

cand_records = {}
index = defaultdict(lambda: defaultdict(list))
for s_file in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(s_file, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            cid = row[0]
            if cid in needed_matches:
                cand_records[cid] = row
                cn = clean_name(row[1])
                ca = clean_address(row[2])
                country = row[3]
                keys = get_name_blocking_keys(cn) + get_address_blocking_keys(ca)
                for k in keys:
                    index[country][k].append(cid)
                if len(cand_records) == len(needed_matches): break
    if len(cand_records) == len(needed_matches): break

missed = []
for s1_id, row in s1_records.items():
    country = row[3]
    cn = clean_name(row[1])
    ca = clean_address(row[2])
    keys = get_name_blocking_keys(cn) + get_address_blocking_keys(ca)
    cands = set()
    for k in keys:
        if k in index[country]:
            cands.update(index[country][k])
    for true_id in gt[s1_id]:
        if true_id in cand_records and true_id not in cands:
            missed.append((row, cand_records[true_id]))

print(f"Total missed found in sample: {len(missed)}")
for i, (s1_row, c_row) in enumerate(missed[:10]):
    print(f"\n--- Missed #{i+1} ---")
    print(f"S1: {s1_row[0]} | Name: {s1_row[1]!r} | Addr: {s1_row[2]!r}")
    print(f"Cand: {c_row[0]} | Name: {c_row[1]!r} | Addr: {c_row[2]!r}")
    print("S1 Clean:", clean_name(s1_row[1]), "|", clean_address(s1_row[2]))
    print("Cand Clean:", clean_name(c_row[1]), "|", clean_address(c_row[2]))
    print("S1 Keys:", get_name_blocking_keys(clean_name(s1_row[1])) + get_address_blocking_keys(clean_address(s1_row[2])))
    print("Cand Keys:", get_name_blocking_keys(clean_name(c_row[1])) + get_address_blocking_keys(clean_address(c_row[2])))
