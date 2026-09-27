import string
import re
from typing import List, Set, Optional

# --- Fast Accent & Punctuation Translation Table ---
accent_map = {
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'à': 'a', 'â': 'a', 'ä': 'a',
    'î': 'i', 'ï': 'i', 'ô': 'o', 'ö': 'o', 'ù': 'u', 'û': 'u', 'ü': 'u',
    'ç': 'c', 'ñ': 'n'
}
punct_chars = string.punctuation + '#@*[](){}<>_~^"\'`|=+/'
trans_dict = {c: ' ' for c in punct_chars}
trans_dict.update(accent_map)
CHAR_TABLE = str.maketrans(trans_dict)

MULTI_SUFFIXES = [
    'private limited', 'pvt limited', 'pvt ltd', 'private ltd', 'p limited', 'p ltd',
    'holding company', 'holdings co', 'holdings inc', 'holdings ltd',
    'societe anonyme', 'societe a responsabilite limitee', 'प्राइवेट लिमिटेड'
]
SINGLE_SUFFIXES = {
    'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited',
    'pllc', 'llp', 'lp', 'pvt', 'sarl', 'sas', 'sasu', 'sci', 'eurl', 'snc',
    'sca', 'sa', 'association', 'societe', 'société', 'लिमिटेड', 'एलएलपी', 'कंपनी'
}
DOMAIN_PATTERN = re.compile(r'\.(com|net|org|co\.uk|in|fr|io|biz|info|gov|edu)$', re.IGNORECASE)

ADDR_WORD_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard',
    'bd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'cir': 'circle',
    'hwy': 'highway', 'pkwy': 'parkway', 'pl': 'place', 'sq': 'square', 'fl': 'floor',
    'flr': 'floor', 'apt': 'apartment', 'ste': 'suite', 'no': '', 'r': 'rue',
    'imp': 'impasse', 'ch': 'chemin', 'all': 'allee',
    # Indian state mappings
    'mh': 'maharashtra', 'rj': 'rajasthan', 'ka': 'karnataka', 'tn': 'tamil nadu',
    'up': 'uttar pradesh', 'dl': 'delhi', 'hr': 'haryana', 'wb': 'west bengal',
    'gj': 'gujarat', 'ap': 'andhra pradesh', 'ts': 'telangana', 'mp': 'madhya pradesh',
    'pb': 'punjab', 'kl': 'kerala', 'or': 'odisha', 'jh': 'jharkhand', 'br': 'bihar',
    'ct': 'chhattisgarh', 'cg': 'chhattisgarh',
    # Indic words
    'राजस्थान': 'rajasthan', 'महाराष्ट्र': 'maharashtra', 'उत्तर': 'uttar', 'प्रदेश': 'pradesh',
    'मध्य': 'madhya', 'पश्चिम': 'west', 'बंगाल': 'bengal', 'कर्नाटक': 'karnataka',
    'ಕರ್ನಾಟಕ': 'karnataka', 'तमिलनाडु': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu',
    'गुजरात': 'gujarat', 'ગુજરાત': 'gujarat', 'तेलंगाना': 'telangana', 'తెలంగాణ': 'telangana',
    'हरियाणा': 'haryana', 'पंजाब': 'punjab', 'दिल्ली': 'delhi', 'केरल': 'kerala'
}


def normalize_name(text: Optional[str]) -> str:
    """
    Standardize business name at C speed:
    - strip Latin accents (preserves Indic)
    - strip domain suffixes (.com, .net, etc.)
    - replace & with 'and'
    - strip legal suffixes from prefix and suffix positions
    - clean punctuation and whitespace
    """
    if not text:
        return ''
    t = str(text).strip()
    if t.lower() in ('nan', 'none', 'null', 'n/a', ''):
        return ''
    
    t = DOMAIN_PATTERN.sub('', t.lower())
    t = t.replace('&', ' and ').translate(CHAR_TABLE)
    
    for ms in MULTI_SUFFIXES:
        if t.endswith(ms):
            t = t[:-len(ms)]
            break
            
    words = t.split()
    while words and words[-1] in SINGLE_SUFFIXES:
        words.pop()
    while words and words[0] in SINGLE_SUFFIXES:
        words.pop(0)
        
    return ' '.join(words)


def normalize_address(text: Optional[str]) -> str:
    """
    Standardize business address at C speed:
    - strip Latin accents (preserves Indic)
    - map abbreviations & Indic state names to standard tokens
    - clean punctuation and whitespace
    """
    if not text:
        return ''
    t = str(text).strip()
    if t.lower() in ('nan', 'none', 'null', 'n/a', ''):
        return ''
    
    t = t.lower().translate(CHAR_TABLE)
    words = [ADDR_WORD_MAP.get(w, w) for w in t.split()]
    return ' '.join(w for w in words if w)


def extract_tokens(text: str) -> List[str]:
    """Returns list of lowercase tokens of length >= 2."""
    if not text:
        return []
    return [tok for tok in text.lower().split() if len(tok) >= 2]


def extract_numeric_tokens(text: str) -> Set[str]:
    """Returns set of numeric substrings."""
    if not text:
        return set()
    return set(re.findall(r'\d+', str(text)))
