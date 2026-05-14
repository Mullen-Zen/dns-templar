# DNS Templar

[![CI](https://github.com/Mullen-Zen/dns-templar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Mullen-Zen/dns-templar-rs/actions/workflows/ci.yml)
[![Release](https://github.com/Mullen-Zen/dns-templar-rs/actions/workflows/release.yml/badge.svg)](https://github.com/Mullen-Zen/dns-templar-rs/releases/latest)

A DNS server that classifies domains in real time with an XGBoost model trained on detecting Domain Generation Algorithm (DGA) traffic. DGA domains are generated hostnames used by malware to communicate with command-and-control (C2) infrastructure. DNS Templar intercepts DNS queries, classifies each domain before resolution, and returns NXDOMAIN for suspected DGA traffic, preventing a successful connection/response.

Developed as a full pipeline, from Python training and feature construction, to an ONNX model export, to an async Rust inference server.

---

## How It Works
 
```
Client DNS query
     │
     ▼
dns-templar (port 53)
     │
     ├─ Check against blacklist  →  NXDOMAIN (known DGA domains)
     ├─ Check against whitelist  →  forward immediately
     └─ Classifier               →  NXDOMAIN if DGA, else forward
                                         │
                                         ▼
                               Upstream DNS (e.g. AdGuard)
```

Each query is classed into one of five tiers:
 
| Tier | Meaning |
|---|---|
| `BLACKLISTED` | Match in known DGA domain list |
| `DGA LIKELY` | Model confidence above threshold |
| `DGA SUSPECTED` | Model confidence in warning range, passed to DNS but logged |
| `WHITELISTED` | Matched in Tranco top-1m allowlist (first 50k entries) |
| `CLEAN` | Below threshold/forwarded normally |
 
---

## Feature Engineering
 
The classifier operates on linguistic and structural features extracted from the domain name itself:
 
- **Shannon entropy**
- **N-gram score** (trained on legitimate domains, scores likelihood of hostname)
- **Vowel ratio**
- **Digit ratio**
- **Unique character ratio**
- **Longest consonant run**
- **TLD frequency** (some top-level domains are more often malicious than others)
- **Dot count / hyphen presence**
SHAP analysis confirmed ngram_score as the dominant feature (mean |SHAP| = 3.45), with domain length and TLD frequency as secondary signals. Results visualized below.

[IMAGE 1]
[IMAGE 2]
 
---

## Model
 
XGBoost classifier trained on ~675,000 labeled domains (Bambenek DGA feeds + Tranco legitimate domains), exported to ONNX. The ONNX export is validated in CI tests against a suite of fixtures that assert feature parity between Python training and Rust inference.
 
---
 
## Known Limitations
 
Dictionary-word DGA families are the current primary weakness. Some malware families (e.g. Suppobox) generate domains from concatenated dictionary words (`horsepressurecalculatedeliver.com`) which have entropy and n-gram distributions that look legitimate to the current feature set. Future releases will aim to catch these domains as frequently as others.
 
The blacklist (337,500 known DGA domains from training data) catches many of these families by exact match. Currently, however, novel wordlist-DGA domains often pass through.
 
---
 
## Installation
 
Download the latest release for your platform from the [releases page](https://github.com/Mullen-Zen/dns-templar-rs/releases/latest):
 
```bash
# Linux x86_64
wget https://github.com/Mullen-Zen/dns-templar-rs/releases/latest/download/dns-templar-x86_64-unknown-linux-musl.tar.gz
wget https://github.com/Mullen-Zen/dns-templar-rs/releases/latest/download/models.tar.gz
 
tar -xzf dns-templar-x86_64-unknown-linux-musl.tar.gz -C /usr/local/bin/
tar -xzf models.tar.gz -C /opt/dns-templar/
chmod +x /usr/local/bin/dns-templar
```
 
---
 
## Configuration
 
Copy `config.example.toml` and edit paths for your environment:
 
```bash
cp dns-templar-rs/config.example.toml /etc/dns-templar/config.toml
```
 
```toml
[model]
classifier  = "/opt/dns-templar/models/classifier.onnx"
threshold   = "/opt/dns-templar/models/threshold.json"
ngram_table = "/opt/dns-templar/models/ngram_table.json"
tld_freq    = "/opt/dns-templar/models/tld_freq.json"
whitelist   = "/opt/dns-templar/models/whitelist.txt"
blacklist   = "/opt/dns-templar/models/blacklist.txt"
 
[server]
listen   = "0.0.0.0:53"
upstream = "127.0.0.1:5353"
 
[logging]
dir = "/var/log/dns-templar"
 
# optional, uncomment to set global model threshold
# [classification]
# threshold_override = 0.8
```
 
---
 
## Usage
 
```bash
# classify a single domain
dns-templar check google.com
dns-templar check --explain xkqjznmbvw.ru
 
# classify a file of domains (one per line)
dns-templar batch domains.txt
 
# run as a DNS server
dns-templar serve
```
 
---
 
## Deployment with AdGuard Home
 
DNS Templar is not designed to be a standalone, full-featured DNS filtering system, as it is built to only catch and filter one type of malicious domains. It is designed to sit in front of AdGuard Home (or similar software), forwarding clean traffic through for ad/tracker filtering:
 
1. Move AdGuard off port 53 — edit `/opt/AdGuardHome/AdGuardHome.yaml`:
   ```yaml
   dns:
     port: 5353
   ```
2. Restart AdGuard: `systemctl restart AdGuardHome`
3. Set `upstream = "127.0.0.1:5353"` in DNS Templar config
4. Run DNS Templar on port 53
To run as a system service:
 
```bash
# /etc/systemd/system/dns-templar.service
[Unit]
Description=DNS Templar DGA DNS filter
After=network.target AdGuardHome.service
Wants=AdGuardHome.service
 
[Service]
ExecStart=/usr/local/bin/dns-templar serve
Restart=on-failure
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
 
[Install]
WantedBy=multi-user.target
```
 
```bash
systemctl daemon-reload
systemctl enable --now dns-templar
```
 
---
 
## Training Your Own Model
 
Dependencies:
```bash
pip install -r requirements.txt
```
 
```bash
cd src
python fetch_data.py
python train.py
python export.py
python generate_test_fixtures.py
```
 
---
