import math
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    'name_jw',
    'name_sort_ratio',
    'name_set_ratio',
    'name_partial_ratio',
    'name_len_diff',
    'first_word_match',
    'is_addr_empty_s1',
    'is_addr_empty_c',
    'addr_token_sort',
    'addr_jaccard',
    'num_common',
    'num_exact_first',
    'blocking_hits',
    'source_is_s3'
]

def extract_pair_features(s1_name: str, s1_addr: str, s1_nums: list[str],
                          c_id: str, c_name: str, c_addr: str, c_nums: list[str],
                          blocking_hits: int = 1) -> list[float]:
    """Extract fast SIMD similarity features between S1 and candidate."""
    # 1. Name features
    jw = distance.JaroWinkler.similarity(s1_name, c_name)
    sort_ratio = fuzz.token_sort_ratio(s1_name, c_name)
    set_ratio = fuzz.token_set_ratio(s1_name, c_name)
    partial_ratio = fuzz.partial_ratio(s1_name, c_name)
    
    len_s1 = len(s1_name)
    len_c = len(c_name)
    max_len = max(len_s1, len_c, 1)
    len_diff = abs(len_s1 - len_c) / max_len
    
    s1_w0 = s1_name.split()[0] if s1_name else ""
    c_w0 = c_name.split()[0] if c_name else ""
    first_w_match = 1.0 if (s1_w0 and s1_w0 == c_w0) else 0.0
    
    # 2. Address features
    addr_empty_s1 = 1.0 if not s1_addr else 0.0
    addr_empty_c = 1.0 if not c_addr else 0.0
    
    if addr_empty_s1 or addr_empty_c:
        addr_sort = 0.0
        addr_jaccard = 0.0
        num_common = 0.0
        num_exact = 0.0
    else:
        addr_sort = fuzz.token_sort_ratio(s1_addr, c_addr)
        
        s1_tokens = set(s1_addr.split())
        c_tokens = set(c_addr.split())
        union_len = len(s1_tokens | c_tokens)
        addr_jaccard = (len(s1_tokens & c_tokens) / union_len) if union_len > 0 else 0.0
        
        # Number matching
        s1_num_set = set(s1_nums)
        c_num_set = set(c_nums)
        common_nums = s1_num_set & c_num_set
        num_common = float(len(common_nums))
        
        num_exact = 1.0 if (s1_nums and c_nums and s1_nums[0] == c_nums[0]) else 0.0

    source_is_s3 = 1.0 if c_id.startswith("S3-") else 0.0
    hits_val = float(blocking_hits)
    
    return [
        jw,
        sort_ratio,
        set_ratio,
        partial_ratio,
        len_diff,
        first_w_match,
        addr_empty_s1,
        addr_empty_c,
        addr_sort,
        addr_jaccard,
        num_common,
        num_exact,
        hits_val,
        source_is_s3
    ]
