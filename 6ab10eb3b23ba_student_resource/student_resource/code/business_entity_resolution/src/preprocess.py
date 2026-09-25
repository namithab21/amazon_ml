import re
import unicodedata

# Compiled regexes for fast processing
RE_ACCENTS = re.compile(r'[\u0300-\u036f]')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_URL = re.compile(r'(?:https?:\/\/)?(?:www\.)?([a-z0-9\-]+)(?:\.[a-z]{2,})+(?:\/[^\s]*)?', re.IGNORECASE)

# DBA / Trade name splits
RE_DBA = re.compile(r'\b(?:d\s*/\s*b\s*/\s*a|d\s*b\s*a|trading as|t\s*/\s*a|c\s*/\s*o|operating as)\b', re.IGNORECASE)

# Legal terms across US, India, and France
LEGAL_TERMS = [
    # Multi-word
    'private limited', 'pvt ltd', 'pvt limited',
    'holding company', 'enterprises ltd', 'solutions inc',
    'societe civile immobiliere',
    # France
    'sarl', 'sci', 'sasu', 'sas', 'eurl', 'snc', 'gie', 'sca', 'scs',
    # US & General
    'incorporated', 'inc', 'corporation', 'corp',
    'limited liability company', 'limited liability partnership',
    'pllc', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'pte', 'co', 'company',
    'group', 'holdings', 'enterprises',
    'services', 'solutions', 'technologies', 'consulting'
]

LEGAL_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(LEGAL_TERMS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

# Common leading stopwords to skip in name tokens
STOP_WORDS = {'the', 'a', 'an', 'and', 'of', 'in', 'at', 'xx', 'de', 'du', 'le', 'la', 'des'}

# Address abbreviations mapping
ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'bvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'hwy': 'highway', 'pkwy': 'parkway', 'ste': 'suite', 'apt': 'apartment',
    'bldg': 'building', 'fl': 'floor', 'rte': 'route', 'r': 'rue', 'all': 'allee', 'imp': 'impasse',
    'po box': 'pobox', 'p o box': 'pobox'
}

def remove_accents(text: str) -> str:
    """Normalize unicode and strip diacritics/accents (e.g. é -> e, à -> a)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return RE_ACCENTS.sub('', nfkd)

def clean_name(name: str) -> str:
    """
    Clean business name:
    1. Strip accents
    2. Lowercase
    3. Extract domain roots if URL (e.g. example.com -> example)
    4. Remove legal prefixes and suffixes
    5. Clean non-alphanumeric chars
    6. Normalize whitespace
    """
    if not name:
        return ""
    
    text = remove_accents(name).lower()
    
    # Check if name contains URL
    match_url = RE_URL.search(text)
    if match_url:
        domain_root = match_url.group(1)
        text = text + " " + domain_root
    
    # Handle d/b/a or trading as: replace with space to keep both parts
    text = RE_DBA.sub(' ', text)
    
    # Remove punctuation
    text = RE_NON_ALPHANUM.sub(' ', text)
    
    # Strip legal terms
    text = LEGAL_PATTERN.sub(' ', text)
    
    text = RE_SPACES.sub(' ', text).strip()
    return text

def clean_address(address: str) -> str:
    """Clean business address with abbreviation expansions."""
    if not address:
        return ""
    
    text = remove_accents(address).lower()
    text = RE_NON_ALPHANUM.sub(' ', text)
    tokens = text.split()
    
    standardized = [ADDRESS_ABBR.get(t, t) for t in tokens]
    return ' '.join(standardized)

def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens (building numbers, PIN/zip codes)."""
    if not text:
        return []
    return RE_NUMBERS.findall(text)

def get_name_blocking_keys(clean_name_str: str) -> list[str]:
    """Generate candidate blocking keys for business name."""
    keys = []
    if not clean_name_str:
        return keys
    
    no_space = clean_name_str.replace(" ", "")
    if len(no_space) >= 3:
        keys.append("NAME_EXACT:" + no_space[:14])
        keys.append("NAME_PREF:" + no_space[:6])
    
    # Significant tokens (ignoring stopwords)
    tokens = [t for t in clean_name_str.split() if len(t) >= 3 and t not in STOP_WORDS]
    if tokens:
        keys.append("NAME_TOK0:" + tokens[0])
        if len(tokens) > 1:
            keys.append("NAME_TOK1:" + tokens[1])
            # Sorted first 2 tokens to handle transpositions (Apex Nippon vs Nippon Apex)
            sorted_pair = "_".join(sorted([tokens[0], tokens[1]]))
            keys.append("NAME_SORT2:" + sorted_pair)
        if len(tokens) > 2:
            keys.append("NAME_TOK2:" + tokens[2])
            
    return keys

def get_address_blocking_keys(clean_addr_str: str) -> list[str]:
    """Generate candidate blocking keys for address."""
    keys = []
    if not clean_addr_str:
        return keys
        
    nums = extract_numbers(clean_addr_str)
    # Significant tokens in address
    skip_addr_words = {'door', 'no', 'unit', 'near', 'flat', 'plot', 'co', 'c', 'o', 'road', 'street', 'avenue'}
    tokens = [t for t in clean_addr_str.split() if not t.isdigit() and len(t) >= 3 and t not in skip_addr_words]
    
    if nums and tokens:
        # Pair primary number with top 2 street/locality tokens
        for t in tokens[:2]:
            keys.append(f"ADDR_NUM_TOK:{nums[0]}_{t}")
            if len(t) >= 3:
                keys.append(f"ADDR_NUM_3P:{nums[0]}_{t[:3]}")
                
        if len(nums) > 1:
            keys.append(f"ADDR_NUM2_TOK:{nums[-1]}_{tokens[0]}")
            
    return keys

def get_combined_blocking_keys(clean_name_str: str, clean_addr_str: str) -> list[str]:
    """Generate all blocking keys combining name and address."""
    keys = get_name_blocking_keys(clean_name_str) + get_address_blocking_keys(clean_addr_str)
    
    nums = extract_numbers(clean_addr_str)
    no_space = clean_name_str.replace(" ", "")
    if len(no_space) >= 3 and nums:
        pref3 = no_space[:3]
        # Number + Name prefix (3 chars)
        keys.append(f"NUM_NAME3:{nums[0]}_{pref3}")
        # Postal / PIN code (5 or 6 digits) + Name prefix
        for n in nums:
            if len(n) in (5, 6):
                keys.append(f"ZIP_NAME3:{n}_{pref3}")
                
    return keys

