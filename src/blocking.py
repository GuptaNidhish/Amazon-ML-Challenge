import os
import time
import sqlite3
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Iterable
import pandas as pd
from normalization import (
    normalize_name, normalize_address, extract_sub_names,
    CHAR_TABLE, SINGLE_SUFFIXES, HONORIFICS
)

# Common street suffixes and noise words to ignore when extracting street keywords
STREET_STOPWORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'lane', 'court', 'circle', 
    'highway', 'way', 'place', 'floor', 'room', 'block', 'unit', 'building', 
    'near', 'opp', 'opposite', 'behind', 'beside', 'phase', 'sector', 'plot',
    'rue', 'impasse', 'chemin', 'allee', 'suite', 'ste', 'apt', 'apartment',
    'flat', 'shop', 'bldg', 'fl', 'flr', 'door', 'house', 'no', 'number', 'lot'
}

# Very frequent business name words that shouldn't index alone as first word
NAME_STOPWORDS = {
    'the', 'and', 'new', 'general', 'global', 'national', 'first', 'premier',
    'royal', 'star', 'super', 'best', 'top', 'om', 'shree', 'sri', 'shri',
    'sai', 'jai', 'association', 'societe', 'group'
}


def eid_to_int(eid: str) -> int:
    """Converts S2-12345 -> +12345 and S3-12345 -> -12345 for 4-byte integer storage."""
    num = int(eid[3:])
    return num if eid[1] == '2' else -num


def int_to_eid(eid_int: int) -> str:
    """Converts +12345 -> S2-12345 and -12345 -> S3-12345."""
    if eid_int > 0:
        return f"S2-{eid_int}"
    else:
        return f"S3-{-eid_int}"


def soundex(name: str) -> str:
    """Computes standard 4-character Soundex phonetic code."""
    if not name:
        return ''
    name = name.upper()
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first = name[0]
    res = [first]
    prev = codes.get(first, '')
    for char in name[1:]:
        c = codes.get(char, '')
        if c != '' and c != prev:
            res.append(c)
        prev = c
    return (''.join(res) + '000')[:4]


def extract_blocking_keys(raw_name: str, raw_addr: str) -> List[str]:
    """Generates prioritized multi-channel blocking keys for an entity."""
    keys = []
    sub_names = extract_sub_names(raw_name)
    for cn in sub_names:
        if not cn:
            continue
        keys.append(f"1_{cn}")
        compact = cn.replace(' ', '')
        if len(compact) >= 5 and compact != cn:
            keys.append(f"1_{compact}")
            
        w = cn.split()
        if len(w) >= 2:
            keys.append(f"2_{w[0]}_{w[1]}")
            if len(w[0]) + len(w[1]) >= 4:
                keys.append(f"1_{w[0]}{w[1]}")
            s1, s2 = soundex(w[0]), soundex(w[1])
            if s1 and s2:
                keys.append(f"sndx_{s1}_{s2}")
        if len(w) >= 3:
            keys.append(f"2_{w[1]}_{w[2]}")
            
    norm_a = normalize_address(raw_addr)
    if norm_a:
        aw = norm_a.split()
        raw_nums = re.findall(r'\d+', norm_a)
        nums = []
        for x in raw_nums:
            clean_n = x.lstrip('0')
            if not clean_n:
                clean_n = '0'
            if clean_n not in nums:
                nums.append(clean_n)
                
        streets = [x for x in aw if not re.search(r'\d', x) and len(x) >= 3 and x not in STREET_STOPWORDS]
        salient_streets = sorted(streets, key=len, reverse=True)[:4]
        
        p4 = sub_names[0][:4] if (sub_names and len(sub_names[0]) >= 4) else ''
        
        for n in nums[:2]:
            for s in salient_streets[:3]:
                keys.append(f"3_{s}_{n}")
            if p4:
                keys.append(f"4_{p4}_{n}")
                
        if len(nums) >= 2:
            keys.append(f"num_{nums[0]}_{nums[1]}")
            
        for i in range(min(len(streets) - 1, 3)):
            keys.append(f"5_{streets[i]}_{streets[i+1]}")
            
    return list(dict.fromkeys(keys))


class SqliteCountryBlocker:
    """
    High-performance, disk-backed, zero-RAM-spill blocking engine.
    Stores target normalized entities and candidate keys in a lightweight SQLite index,
    allowing 4+ million entities to be indexed in seconds with under 150 MB RAM.
    """
    def __init__(self, country: str, db_path: str = "output/blocker.db"):
        self.country = country
        self.db_path = db_path
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.c.execute('PRAGMA synchronous = OFF')
        self.c.execute('PRAGMA journal_mode = OFF')
        self.c.execute('PRAGMA temp_store = MEMORY')
        self.c.execute('PRAGMA cache_size = -256000')  # 256 MB cache (500 MB RAM budget)
        
        self.c.execute('CREATE TABLE targets (eid_int INT PRIMARY KEY, norm_n TEXT, norm_a TEXT)')
        self.c.execute('CREATE TABLE cands (k TEXT, eid_int INT)')

    def fit_stream(self, record_stream: Iterable[Tuple[str, str, str]], batch_size: int = 50000) -> int:
        """Stream S2 + S3 target records, insert into SQLite, and build b-tree index."""
        targets = []
        cand_rows = []
        count = 0
        
        for eid, name, addr in record_stream:
            eid_int = eid_to_int(eid)
            norm_n = normalize_name(name)
            norm_a = normalize_address(addr)
            targets.append((eid_int, norm_n, norm_a))
            
            keys = extract_blocking_keys(name, addr)
            for k in keys:
                cand_rows.append((k, eid_int))
                
            count += 1
            if len(targets) >= batch_size:
                self.c.executemany('INSERT INTO targets VALUES (?, ?, ?)', targets)
                self.c.executemany('INSERT INTO cands VALUES (?, ?)', cand_rows)
                targets.clear()
                cand_rows.clear()
                
        if targets:
            self.c.executemany('INSERT INTO targets VALUES (?, ?, ?)', targets)
            self.c.executemany('INSERT INTO cands VALUES (?, ?)', cand_rows)
            targets.clear()
            cand_rows.clear()
            
        self.conn.commit()
        self.c.execute('CREATE INDEX idx_cands_k ON cands (k)')
        self.conn.commit()
        return count

    def query_candidates(self, raw_name: str, raw_addr: str, max_cands: int = 35) -> List[Tuple[int, str, str]]:
        """Queries candidates and returns list of (eid_int, norm_n2, norm_a2)."""
        keys = extract_blocking_keys(raw_name, raw_addr)
        if not keys:
            return []
            
        sql = f'SELECT DISTINCT c.eid_int, t.norm_n, t.norm_a FROM cands c JOIN targets t ON c.eid_int = t.eid_int WHERE c.k IN ({",".join(["?"]*len(keys))}) LIMIT ?'
        params = list(keys) + [max_cands]
        return self.c.execute(sql, params).fetchall()

    def close(self):
        """Close connection and remove temp database file."""
        try:
            self.conn.close()
        except Exception:
            pass
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass


class CountryBlocker:
    """In-memory blocker for training / validation sampling."""
    def __init__(self, country: str):
        self.country = country
        self.idx = defaultdict(list)

    def fit(self, target_ids: List[str], target_names: List[str], target_addrs: List[str]):
        for eid, name, addr in zip(target_ids, target_names, target_addrs):
            keys = extract_blocking_keys(name, addr)
            for k in keys:
                self.idx[k].append(eid)

    def query(self, s1_ids: List[str], s1_names: List[str], s1_addrs: List[str], max_candidates: int = 35) -> Dict[str, List[str]]:
        results = {}
        for s1_id, name, addr in zip(s1_ids, s1_names, s1_addrs):
            keys = extract_blocking_keys(name, addr)
            cands = []
            seen = set()
            for k in keys:
                if k in self.idx:
                    for cid in self.idx[k][:50]:
                        if cid not in seen:
                            seen.add(cid)
                            cands.append(cid)
                            if len(cands) >= max_candidates:
                                break
                if len(cands) >= max_candidates:
                    break
            results[s1_id] = cands
        return results
