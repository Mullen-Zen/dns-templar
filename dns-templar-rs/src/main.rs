mod features;
mod model;
mod classifier;
mod server;
mod blacklist;

use std::sync::Arc;
use classifier::{DnsTemplar, Tier};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_appender::rolling;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Parser)]
#[command(name = "dns-templar")]
#[command(about = "ML-powered DNS classifier for DGA domain detection")]
struct Cli {
    #[arg(long, default_value = "../models/classifier.onnx")]
    model: PathBuf,
    #[arg(long, default_value = "../models/threshold.json")]
    threshold: PathBuf,
    #[arg(long, default_value = "../models/ngram_table.json")]
    ngram_table: PathBuf,
    #[arg(long, default_value = "../models/tld_freq.json")]
    tld_freq: PathBuf,
    #[arg(long, default_value = "../models/whitelist.txt")]
    whitelist: PathBuf,
    #[arg(long, default_value = "../logs")]
    log_dir: PathBuf,
    #[arg(long, default_value = "../models/blacklist.txt")]
    blacklist: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Check {
        domain: String,
        #[arg(long)]
        explain: bool,
        #[arg(long, help = "Override classification threshold (0.0-1.0)")]
        threshold: Option<f32>,
    },
    Batch {
        file: PathBuf,
        #[arg(long)]
        explain: bool,
        #[arg(long, help = "Override classification threshold (0.0-1.0)")]
        threshold: Option<f32>,
    },
    Serve {
        #[arg(long, default_value = "0.0.0.0:53")]
        listen: String,
        #[arg(long, default_value = "127.0.0.1:5353")]
        upstream: String,
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
    let _guard = init_logging(cli.log_dir.to_str().unwrap());

    let templar = DnsTemplar::load(
        cli.model.to_str().unwrap(),
        cli.threshold.to_str().unwrap(),
        cli.ngram_table.to_str().unwrap(),
        cli.tld_freq.to_str().unwrap(),
        cli.whitelist.to_str().unwrap(),
        cli.blacklist.to_str().unwrap(),
    )?;

    let _ = templar.classify("warmup.internal", None);
    tracing::info!("model warmed up");

    match cli.command {
        Command::Check { domain, explain, threshold } => {
            classify_and_print(&templar, &domain, explain, threshold)?;
        }
        Command::Batch { file, explain, threshold } => {
            let contents = std::fs::read_to_string(&file)?;
            for line in contents.lines() {
                let domain = line.trim();
                if !domain.is_empty() {
                    classify_and_print(&templar, domain, explain, threshold)?;
                }
            }
        }
        Command::Serve { listen, upstream, threshold } => {
            let templar = Arc::new(templar);
            server::serve(&listen, &upstream, templar, threshold).await?;
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