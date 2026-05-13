use std::collections::HashMap;
use std::fs;

pub type NgramTable = HashMap<String, f64>;
pub type TldFreqMap = HashMap<String, f64>;

pub fn entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }

    let len = s.len() as f64;
    counts.values().fold(0.0, |acc, &count| {
        let p = count as f64 / len;
        acc - p * p.log2()
    })
}

pub fn digit_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let digits = s.chars().filter(|c| c.is_ascii_digit()).count();
    digits as f64 / s.len() as f64
}

pub fn vowel_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let vowels = "aeiou";
    let count = s.chars()
        .filter(|c| vowels.contains(*c))
        .count();
    count as f64 / s.len() as f64
}

pub fn unique_char_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let unique: std::collections::HashSet<char> = s.chars().collect();
    unique.len() as f64 / s.len() as f64
}

pub fn longest_consonant_run(s: &str) -> usize {
    let consonants = "bcdfghjklmnpqrstvwxyz";
    let mut max_run = 0usize;
    let mut current = 0usize;
    for c in s.chars() {
        if consonants.contains(c) {
            current += 1;
            if current > max_run { max_run = current; }
        } else {
            current = 0;
        }
    }
    max_run
}

pub fn dot_count(domain: &str) -> usize {
    domain.chars().filter(|&c| c == '.').count()
}

pub fn has_hyphen(s: &str) -> f64 {
    if s.contains('-') { 1.0 } else { 0.0 }
}

pub fn extract_tld(domain: &str) -> &str {
    match domain.rfind('.') {
        Some(pos) => &domain[pos + 1..],
        None => "",
    }
}

pub fn extract_hostname(domain: &str) -> &str {
    match domain.rfind('.') {
        Some(pos) => &domain[..pos],
        None => domain,
    }
}

pub fn load_ngram_table(path: &str) -> Result<NgramTable, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let table: NgramTable = serde_json::from_str(&contents)?;
    Ok(table)
}

pub fn load_tld_freq(path: &str) -> Result<TldFreqMap, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let map: TldFreqMap = serde_json::from_str(&contents)?;
    Ok(map)
}

pub fn ngram_score(hostname: &str, table: &NgramTable, n: usize) -> f64 {
    let hostname = hostname.to_lowercase();
    let chars: Vec<char> = hostname.chars().collect();

    if chars.len() < n {
        return 0.0;
    }

    let floor = 1e-10_f64;
    let ngrams: Vec<String> = chars
        .windows(n)
        .map(|w| w.iter().collect::<String>())
        .collect();

    let log_prob_sum: f64 = ngrams
        .iter()
        .map(|ng| {
            let p = table.get(ng).copied().unwrap_or(floor);
            p.ln()
        })
        .sum();

    log_prob_sum / ngrams.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[derive(serde::Deserialize)]
    struct Fixture {
        domain: String,
        hostname: String,
        entropy: f64,
        digit_ratio: f64,
        vowel_ratio: f64,
        unique_char_ratio: f64,
        longest_consonant_run: usize,
        dot_count: usize,
        has_hyphen: f64,
        tld: String,
        ngram_score: f64,
    }

    fn load_fixtures() -> Vec<Fixture> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures.json");
        let contents = fs::read_to_string(path).expect("fixtures.json not found");
        serde_json::from_str(&contents).expect("failed to parse fixtures")
    }

    #[test]
    fn test_entropy() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = entropy(&f.hostname);
            assert!(
                (result - f.entropy).abs() < 1e-10,
                "entropy mismatch for {}: got {}, expected {}",
                f.domain, result, f.entropy
            );
        }
    }

    #[test]
    fn test_digit_ratio() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = digit_ratio(&f.hostname);
            assert!(
                (result - f.digit_ratio).abs() < 1e-10,
                "digit_ratio mismatch for {}: got {}, expected {}",
                f.domain, result, f.digit_ratio
            );
        }
    }

    #[test]
    fn test_vowel_ratio() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = vowel_ratio(&f.hostname);
            assert!(
                (result - f.vowel_ratio).abs() < 1e-10,
                "vowel_ratio mismatch for {}: got {}, expected {}",
                f.domain, result, f.vowel_ratio
            );
        }
    }

    #[test]
    fn test_unique_char_ratio() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = unique_char_ratio(&f.hostname);
            assert!(
                (result - f.unique_char_ratio).abs() < 1e-10,
                "unique_char_ratio mismatch for {}: got {}, expected {}",
                f.domain, result, f.unique_char_ratio
            );
        }
    }

    #[test]
    fn test_longest_consonant_run() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = longest_consonant_run(&f.hostname);
            assert_eq!(
                result, f.longest_consonant_run,
                "longest_consonant_run mismatch for {}", f.domain
            );
        }
    }

    #[test]
    fn test_dot_count() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = dot_count(&f.domain);
            assert_eq!(
                result, f.dot_count,
                "dot_count mismatch for {}", f.domain
            );
        }
    }

    #[test]
    fn test_has_hyphen() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = has_hyphen(&f.hostname);
            assert_eq!(
                result, f.has_hyphen,
                "has_hyphen mismatch for {}", f.domain
            );
        }
    }

    #[test]
    fn test_extract_tld() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = extract_tld(&f.domain);
            assert_eq!(
                result, f.tld,
                "tld mismatch for {}", f.domain
            );
        }
    }

    #[test]
    fn test_extract_hostname() {
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = extract_hostname(&f.domain);
            assert_eq!(
                result, f.hostname,
                "hostname mismatch for {}", f.domain
            );
        }
    }

    #[test]
    fn test_ngram_score() {
        let table_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/ngram_table.json");
        let table = load_ngram_table(table_path).expect("failed to load ngram table");
        let fixtures = load_fixtures();
        for f in &fixtures {
            let result = ngram_score(&f.hostname, &table, 3);
            assert!(
                (result - f.ngram_score).abs() < 1e-6,
                "ngram_score mismatch for {}: got {}, expected {}",
                f.domain, result, f.ngram_score
            );
        }
    }
}