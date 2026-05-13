use crate::features::{
    entropy, digit_ratio, vowel_ratio, unique_char_ratio,
    longest_consonant_run, dot_count, has_hyphen,
    extract_tld, extract_hostname, ngram_score,
    load_ngram_table, load_tld_freq, NgramTable, TldFreqMap,
};
use crate::model::Classifier;

pub struct DnsTemplar {
    classifier: Classifier,
    ngram_table: NgramTable,
    tld_freq: TldFreqMap,
}

pub struct Verdict {
    pub domain: String,
    pub probability: f32,
    pub is_dga: bool,
    pub features: Vec<(&'static str, f64)>,
}

impl DnsTemplar {
    pub fn load(
        model_path: &str,
        threshold_path: &str,
        ngram_path: &str,
        tld_freq_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let classifier = Classifier::load(model_path, threshold_path)?;
        let ngram_table = load_ngram_table(ngram_path)?;
        let tld_freq = load_tld_freq(tld_freq_path)?;

        Ok(Self { classifier, ngram_table, tld_freq })
    }

    pub fn classify(&self, domain: &str) -> Result<Verdict, Box<dyn std::error::Error>> {
        let domain_lower = domain.to_lowercase();
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

        let (probability, is_dga) = self.classifier.predict(&feature_vec)?;

        Ok(Verdict {
            domain: domain.to_string(),
            probability,
            is_dga,
            features,
        })
    }
}