import math
import json
from collections import Counter, defaultdict

# Measures char randomness
def entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in counts.values()
    )

# Measures ratio of numbers to letters
def digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isdigit() for c in s) / len(s)

# Measures ratio of vowels to characters
def vowel_ratio(s: str) -> float:
    if not s:
        return 0.0
    vowels = set("aeiou")
    return sum(c in vowels for c in s.lower()) / len(s)

# Measures ratio of unique characterrs
def unique_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(set(s)) / len(s)

# Measures length of hostname
def hostname_length(s: str) -> int:
    return len(s)

# Measures length of longest consecutive consonant sequence
def longest_consonant_run(s: str) -> int:
    consonants = set("bcdfghjklmnpqrstvwxyz")
    max_run = 0
    current = 0
    for c in s.lower():
        if c in consonants:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run

# Measures number of dots
def dot_count(domain: str) -> int:
    return domain.count(".")

# Detects the presence of hyphens
def has_hyphen(s: str) -> int:
    return int("-" in s)

# Extracts the top-level domain
def tld(domain: str) -> str:
    parts = domain.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else ""

# N-gram frequency table from a list of legitimate domains
def build_ngram_table(domains: list[str], n: int = 3) -> dict:
    counts = defaultdict(int)
    for domain in domains:
        hostname = domain.rsplit(".", 1)[0].lower()
        for i in range(len(hostname) - n + 1):
            ngram = hostname[i:i+n]
            counts[ngram] += 1

    # Laplace smoothing
    total = sum(counts.values()) + len(counts)
    probs = {ng: (count + 1) / total for ng, count in counts.items()}
    return probs

# N-gram scoring
def ngram_score(hostname: str, table: dict, n: int = 3) -> float:
    hostname = hostname.lower()
    ngrams = [hostname[i:i+n] for i in range(len(hostname) - n + 1)]
    if not ngrams:
        return 0.0
    
    floor = 1e-10
    log_probs = [math.log(table.get(ng, floor)) for ng in ngrams]
    return sum(log_probs) / len(log_probs)

def save_ngram_table(table: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(table, f)

def load_ngram_table(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def extract_tld(domain: str) -> str:
    parts = domain.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else ""

def extract_hostname(domain: str) -> str:
    parts = domain.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else domain

# Extract all features from a domain string
def extract_features(domain: str, ngram_table: dict) -> dict:
    hostname = domain.rsplit(".", 1)[0].lower()
    domain_tld = tld(domain)

    return {
        "length":               hostname_length(hostname),
        "entropy":              entropy(hostname),
        "digit_ratio":          digit_ratio(hostname),
        "vowel_ratio":          vowel_ratio(hostname),
        "unique_char_ratio":    unique_char_ratio(hostname),
        "longest_consonant_run": longest_consonant_run(hostname),
        "dot_count":            dot_count(domain),
        "has_hyphen":           has_hyphen(hostname),
        "ngram_score":          ngram_score(hostname, ngram_table),
        "tld":                  domain_tld,
    }