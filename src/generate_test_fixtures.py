import json
import sys
sys.path.insert(0, ".")
from features import (
    entropy, digit_ratio, vowel_ratio, unique_char_ratio,
    longest_consonant_run, dot_count, has_hyphen,
    extract_tld, extract_hostname, ngram_score, load_ngram_table
)

table = load_ngram_table("../models/ngram_table.json")

test_domains = [
    "google.com",
    "ymvpdgmplptkaommi.tw",
    "horsepressurecalculatedeliver.com",
    "cmfggfkte.com.sv",
    "fmspuxrs7i8sfi8osd.biz",
    "wideworld-sports.me",
    "otleawfaufex.ddns.net",
]

fixtures = []
for domain in test_domains:
    hostname = domain.rsplit(".", 1)[0].lower()
    fixtures.append({
        "domain": domain,
        "hostname": hostname,
        "entropy": entropy(hostname),
        "digit_ratio": digit_ratio(hostname),
        "vowel_ratio": vowel_ratio(hostname),
        "unique_char_ratio": unique_char_ratio(hostname),
        "longest_consonant_run": longest_consonant_run(hostname),
        "dot_count": dot_count(domain),
        "has_hyphen": has_hyphen(hostname),
        "tld": extract_tld(domain) if hasattr(__import__("features"), "extract_tld") else domain.rsplit(".", 1)[-1],
        "ngram_score": ngram_score(hostname, table),
    })

with open("../dns-templar-rs/tests/fixtures.json", "w") as f:
    json.dump(fixtures, f, indent=2)

print(f"Generated {len(fixtures)} fixtures")
for f in fixtures:
    print(f"  {f['domain']}: entropy={f['entropy']:.6f}, ngram={f['ngram_score']:.6f}")