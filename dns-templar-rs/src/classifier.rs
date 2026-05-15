use crate::features::{
    entropy, digit_ratio, vowel_ratio, unique_char_ratio,
    longest_consonant_run, dot_count, has_hyphen,
    extract_tld, extract_hostname, ngram_score,
    load_ngram_table, load_tld_freq, NgramTable, TldFreqMap,
};
use crate::model::Classifier;
use std::collections::HashSet;
use crate::blacklist::Blacklist;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;

pub struct DnsTemplar {
    classifier: Classifier,
    ngram_table: NgramTable,
    tld_freq: TldFreqMap,
    whitelist: HashSet<String>,
    blacklist: Blacklist,
    cache: Mutex<LruCache<String, Verdict>>,
}

#[derive(Debug, Clone)]
pub enum Tier {
    Whitelisted,
    Blacklisted,
    HighConfidence,
    Suspicious,
    Clean,
}

#[derive(Clone)]
pub struct Verdict {
    pub domain: String,
    pub probability: f32,
    pub is_dga: bool,
    pub whitelisted: bool,
    pub tier: Tier,
    pub features: Vec<(&'static str, f64)>,
}

impl DnsTemplar {
    pub fn load(
        model_path: &str,
        threshold_path: &str,
        ngram_path: &str,
        tld_freq_path: &str,
        whitelist_path: &str,
        blacklist_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let classifier = Classifier::load(model_path, threshold_path)?;
        let ngram_table = load_ngram_table(ngram_path)?;
        let tld_freq = load_tld_freq(tld_freq_path)?;

        let whitelist_raw = std::fs::read_to_string(whitelist_path)?;
        let whitelist: HashSet<String> = whitelist_raw
            .lines()
            .map(|l| l.trim().to_lowercase())
            .filter(|l| !l.is_empty())
            .collect();
        println!("Loaded {} whitelisted domains", whitelist.len());

        let blacklist = Blacklist::load(blacklist_path)?;

        let cache = Mutex::new(LruCache::new(NonZeroUsize::new(65536).unwrap()));

        Ok(Self { classifier, ngram_table, tld_freq, whitelist, blacklist, cache })
    }

    pub fn classify(&self, domain: &str, threshold_override: Option<f32>) -> Result<Verdict, Box<dyn std::error::Error>> {
        let domain_lower = domain.trim_end_matches('.').to_lowercase();
        
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&domain_lower) {
                tracing::info!(
                    domain = %domain_lower,
                    "cache hit!"
                );
                return Ok(Verdict {
                    domain: domain.to_string(),
                    ..cached.clone()
                });
            }
        }

        if self.whitelist.contains(&domain_lower) {
            return Ok(Verdict {
                domain: domain.to_string(),
                probability: 0.0,
                is_dga: false,
                whitelisted: true,
                tier: Tier::Whitelisted,
                features: vec![],
            });
        }

        if self.blacklist.contains(&domain_lower) {
            return Ok(Verdict {
                domain: domain.to_string(),
                probability: 1.0,
                is_dga: true,
                whitelisted: false,
                tier: Tier::Blacklisted,
                features: vec![],
            });
        }
        
        let hostname = extract_hostname(&domain_lower).to_string();
        let tld = extract_tld(&domain_lower).to_string();
        let tld_freq_val = self.tld_freq.get(&tld).copied().unwrap_or(0.0);

        let features: Vec<(&'static str, f64)> = vec![
            ("length",                  hostname.len() as f64),
            ("entropy",                 entropy(&hostname)),
            ("digit_ratio",             digit_ratio(&hostname)),
            ("vowel_ratio",             vowel_ratio(&hostname)),
            ("unique_char_ratio",       unique_char_ratio(&hostname)),
            ("longest_consonant_run",   longest_consonant_run(&hostname) as f64),
            ("dot_count",               dot_count(&domain_lower) as f64),
            ("has_hyphen",              has_hyphen(&hostname)),
            ("ngram_score",             ngram_score(&hostname, &self.ngram_table, 3)),
            ("tld_freq",                tld_freq_val),
        ];

        let feature_vec: Vec<f32> = features
            .iter()
            .map(|(_, v)| *v as f32)
            .collect();

        let (probability, _) = self.classifier.predict(&feature_vec)?;
        let threshold = threshold_override.unwrap_or(self.classifier.threshold);
        let is_dga = probability >= threshold;
    
        let tier = if is_dga && probability >= 0.85 {
            Tier::HighConfidence
        } else if is_dga {
            Tier::Suspicious
        } else {
            Tier::Clean
        };

        let verdict = Verdict {
            domain: domain.to_string(),
            probability,
            is_dga,
            whitelisted: false,
            tier,
            features,
        };

        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(domain_lower, verdict.clone());
        }

        Ok(verdict)
    }
}