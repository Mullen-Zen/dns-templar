mod features;
mod model;
mod classifier;
mod server;
mod blacklist;
mod config;

use std::sync::Arc;
use classifier::{DnsTemplar, Tier};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_appender::rolling;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use config::Config;

#[derive(Parser)]
#[command(name = "dns-templar")]
#[command(about = "ML-powered DNS classifier for DGA domain detection")]
struct Cli {
    #[arg(long, default_value = "/etc/dns-templar/config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Check {
        domain: String,
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        threshold: Option<f32>,
    },
    Batch {
        file: PathBuf,
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        threshold: Option<f32>,
    },
    Serve {
        #[arg(long)]
        threshold: Option<f32>,
    },
}

fn init_logging(log_dir: &str) -> tracing_appender::non_blocking::WorkerGuard {
    let file_appender = rolling::daily(log_dir, "dns-templar.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(fmt::layer().json().with_writer(non_blocking))
        .init();

    guard
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;

    let _guard = init_logging(cfg.logging.dir.to_str().unwrap());

    let templar = DnsTemplar::load(
        cfg.model.classifier.to_str().unwrap(),
        cfg.model.threshold.to_str().unwrap(),
        cfg.model.ngram_table.to_str().unwrap(),
        cfg.model.tld_freq.to_str().unwrap(),
        cfg.model.whitelist.to_str().unwrap(),
        cfg.model.blacklist.to_str().unwrap(),
    )?;

    let config_threshold = cfg.classification
        .as_ref()
        .and_then(|c| c.threshold_override);

    let _ = templar.classify("warmup.internal", None);
    tracing::info!("model warmed up");

    match cli.command {
        Command::Check { domain, explain, threshold } => {
            let t = threshold.or(config_threshold);
            classify_and_print(&templar, &domain, explain, t)?;
        }
        Command::Batch { file, explain, threshold } => {
            let t = threshold.or(config_threshold);
            let contents = std::fs::read_to_string(&file)?;
            for line in contents.lines() {
                let domain = line.trim();
                if !domain.is_empty() {
                    classify_and_print(&templar, domain, explain, t)?;
                }
            }
        }
        Command::Serve { threshold } => {
            let t = threshold.or(config_threshold);
            let templar = Arc::new(templar);
            server::serve(&cfg.server.listen, &cfg.server.upstream, templar, t).await?;
        }
    }

    Ok(())
}

fn classify_and_print(
    templar: &DnsTemplar,
    domain: &str,
    explain: bool,
    threshold_override: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use colored::Colorize;

    let verdict = templar.classify(domain, threshold_override)?;
    
    tracing::info!(
        domain = %verdict.domain,
        probability = verdict.probability,
        is_dga = verdict.is_dga,
        whitelisted = verdict.whitelisted,
        tier = ?verdict.tier,
        "classified"
    );
    
    let prob_str = if verdict.whitelisted {
        " -- ".to_string()
    } else {
        format!("{:.1}%", verdict.probability * 100.0)
    };

    let line = match verdict.tier {
        Tier::Blacklisted =>
            format!("[BLACKLISTED]      {prob_str}  {}", verdict.domain).red().bold().to_string(),
        Tier::HighConfidence => 
            format!("[DGA LIKELY]       {prob_str}  {}", verdict.domain).red().to_string(),
        Tier::Suspicious => 
            format!("[DGA SUSPECTED]    {prob_str}  {}", verdict.domain).yellow().to_string(),
        Tier::Whitelisted =>
            format!("[WHITELISTED]      {prob_str}  {}", verdict.domain).green().to_string(),
        Tier::Clean =>
            format!("[CLEAN]            {prob_str}  {}", verdict.domain).green().to_string(),
    };

    println!("{line}");

    if explain && !verdict.whitelisted {
        println!("Feature contributions:");
        for (name, value) in &verdict.features {
            println!("      {:<25} {:.6}", name, value);
        }
    }

    Ok(())
}