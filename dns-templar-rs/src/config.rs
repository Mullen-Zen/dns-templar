use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
    pub classification: Option<ClassificationConfig>,
}

#[derive(Deserialize)]
pub struct ModelConfig {
    pub classifier: PathBuf,
    pub threshold: PathBuf,
    pub ngram_table: PathBuf,
    pub tld_freq: PathBuf,
    pub whitelist: PathBuf,
    pub blacklist: PathBuf,
}

#[derive(Deserialize)]
pub struct ServerConfig {
    pub listen: String,
    pub upstream: String,
}

#[derive(Deserialize)]
pub struct LoggingConfig {
    pub dir: PathBuf,
}

#[derive(Deserialize)]
pub struct ClassificationConfig {
    pub threshold_override: Option<f32>,
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}