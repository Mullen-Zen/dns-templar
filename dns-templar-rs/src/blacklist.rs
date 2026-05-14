use std::collections::HashSet;

pub struct Blacklist {
    exact: HashSet<String>,
    wildcards: Vec<String>,
}

impl Blacklist {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut exact = HashSet::new();
        let mut wildcards = Vec::new();

        for line in std::fs::read_to_string(path)?.lines() {
            let entry = line.trim();
            if entry.is_empty() || entry.starts_with('#') {
                continue;
            }
            if let Some(suffix) = entry.strip_prefix("*.") {
                wildcards.push(suffix.to_lowercase());
            } else {
                exact.insert(entry.to_lowercase());
            }
        }

        tracing::info!("loaded {} blacklisted domains", exact.len());
        Ok(Self { exact, wildcards })
    }

    pub fn contains(&self, domain: &str) -> bool {
        let d = domain.trim_end_matches('.').to_lowercase();
        if self.exact.contains(&d) {
            return true;
        }
        self.wildcards.iter().any(|w| d == *w || d.ends_with(&format!(".{w}")))
    }
}