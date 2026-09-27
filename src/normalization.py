import string
import re
import anyascii
from typing import List, Set, Optional

# --- Fast Accent & Punctuation Translation Table ---
accent_map = {
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'à': 'a', 'â': 'a', 'ä': 'a', 'á': 'a',
    'î': 'i', 'ï': 'i', 'í': 'i', 'ô': 'o', 'ö': 'o', 'ó': 'o', 'ù': 'u', 'û': 'u', 'ü': 'u', 'ú': 'u',
    'ç': 'c', 'ñ': 'n'
}
punct_chars = string.punctuation + '#@*[](){}<>_~^"\'`|=+/'
trans_dict = {c: ' ' for c in punct_chars}
trans_dict.update(accent_map)
CHAR_TABLE = str.maketrans(trans_dict)

MULTI_SUFFIXES = [
    'private limited', 'pvt limited', 'pvt ltd', 'private ltd', 'p limited', 'p ltd',
    'holding company', 'holdings co', 'holdings inc', 'holdings ltd',
    'societe anonyme', 'societe a responsabilite limitee', 'praivet limited',
    'piraivet limitet', 'praivet limitet'
]

SINGLE_SUFFIXES = {
    'llc', 'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited',
    'pllc', 'llp', 'lp', 'pvt', 'private', 'sarl', 'sas', 'sasu', 'sci', 'eurl', 'snc',
    'sca', 'sa', 'association', 'societe', 'société', 'holdings', 'holding',
    'elelpi', 'praivet', 'gmbh', 'ag', 'bv', 'nv', 'spa', 'srl',
    'limitet', 'piraivet', 'pra', 'li'
}

HONORIFICS = {'smt', 'shri', 'sri', 'shree', 'mr', 'mrs', 'dr', 'prof', 'ms', 'm/s'}

DBA_PATTERN = re.compile(r'\b(d/?b/?a|t/?a|trading as|c/?o|operating as)\b', re.IGNORECASE)
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
    'gj': 'gujarat', 'ap': 'andhra pradesh', 'ts': 'telangana', 'tg': 'telangana',
    'mp': 'madhya pradesh', 'pb': 'punjab', 'kl': 'kerala', 'or': 'odisha',
    'od': 'odisha', 'jh': 'jharkhand', 'br': 'bihar', 'ct': 'chhattisgarh', 'cg': 'chhattisgarh'
}


LEET_TRANS = str.maketrans('0134578', 'oieastb')

INDUSTRY_WORDS = {
    'services', 'service', 'solutions', 'solution', 'enterprises', 'enterprise',
    'consultants', 'consultant', 'consulting', 'management', 'center', 'centre',
    'group', 'systems', 'system', 'technologies', 'technology', 'associates',
    'industries', 'industry', 'network', 'holdings', 'holding', 'international',
    'global', 'national'
}


def uncorrupt_leetspeak(text: Optional[str]) -> str:
    """Replaces numeric character substitutions in words (e.g. 5heth -> sheth, H0lloway -> holloway)."""
    if not text:
        return ''
    words = str(text).split()
    new_words = []
    for w in words:
        if (any(c.isdigit() for c in w) and any(c.isalpha() for c in w)) or (len(w) == 2 and w[0].isdigit() and w[1].isalpha()):
            new_words.append(w.translate(LEET_TRANS))
        else:
            new_words.append(w)
    return ' '.join(new_words)


def clean_core_name(text: Optional[str]) -> str:
    """Strips generic industry words (services, solutions, enterprises, etc.)."""
    if not text:
        return ''
    words = [w for w in str(text).split() if w not in INDUSTRY_WORDS]
    return ' '.join(words) if words else str(text)


def normalize_name(text: Optional[str]) -> str:
    """
    Standardize business name:
    - Uncorrupts leetspeak digits (5heth -> sheth)
    - Transliterates Indic, Cyrillic, accented scripts to ASCII with anyascii
    - Strips domain suffixes (.com, .net, etc.)
    - Replaces & with 'and'
    - Strips honorifics (Smt, Shri, etc.) and legal suffixes from any position
    - Cleans punctuation and whitespace
    """
    if not text:
        return ''
    t = str(text).strip()
    if t.lower() in ('nan', 'none', 'null', 'n/a', ''):
        return ''
    
    t = uncorrupt_leetspeak(t)
    # 1. Phonetic transliteration to ASCII
    t = anyascii.anyascii(t).lower()
    
    # 2. Strip domain extensions
    t = DOMAIN_PATTERN.sub('', t)
    t = t.replace('&', ' and ').translate(CHAR_TABLE)
    
    # 3. Clean legal suffixes and honorifics
    words = [w for w in t.split() if w not in SINGLE_SUFFIXES and w not in HONORIFICS and len(w) >= 2]
    return ' '.join(words)


def extract_sub_names(text: Optional[str]) -> List[str]:
    """
    Extracts all plausible sub-names:
    - DBA / trade name parts (e.g. 'Beloavi d/b/a Novent Owl' -> ['beloavi', 'novent owl'])
    - De-leeted and core names (without industry words)
    """
    if not text:
        return []
    t = anyascii.anyascii(str(text)).lower()
    t = DOMAIN_PATTERN.sub('', t)
    
    parts = DBA_PATTERN.split(t)
    results = []
    seen = set()
    
    for p in parts:
        cleaned = normalize_name(p)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)
        # Also add core name without generic industry words
        core = clean_core_name(cleaned)
        if core and len(core) >= 3 and core not in seen:
            seen.add(core)
            results.append(core)
            
    return results if results else [normalize_name(text)]


def normalize_address(text: Optional[str]) -> str:
    """
    Standardize business address:
    - Transliterates Indic to Latin
    - Maps abbreviations & state names to standard tokens
    - Cleans punctuation and whitespace
    """
    if not text:
        return ''
    t = str(text).strip()
    if t.lower() in ('nan', 'none', 'null', 'n/a', ''):
        return ''
    
    t = anyascii.anyascii(t).lower().translate(CHAR_TABLE)
    words = [ADDR_WORD_MAP.get(w, w) for w in t.split()]
    return ' '.join(w for w in words if w)


def extract_tokens(text: str) -> List[str]:
    """Returns list of lowercase tokens of length >= 2."""
    if not text:
        return []
    return [tok for tok in anyascii.anyascii(str(text)).lower().split() if len(tok) >= 2]


def extract_numeric_tokens(text: str) -> Set[str]:
    """Returns set of numeric substrings."""
    if not text:
        return set()
    return set(re.findall(r'\d+', str(text)))
