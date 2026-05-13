mod features;
mod model;
mod classifier;

use classifier::DnsTemplar;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let templar = DnsTemplar::load(
        cli.model.to_str().unwrap(),
        cli.threshold.to_str().unwrap(),
        cli.ngram_table.to_str().unwrap(),
        cli.tld_freq.to_str().unwrap(),
        cli.whitelist.to_str().unwrap(),
    )?;

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
    }

    Ok(())
}

fn classify_and_print(
    templar: &DnsTemplar,
    domain: &str,
    explain: bool,
    threshold_override: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let verdict = templar.classify(domain, threshold_override)?;
    
    let label = if verdict.whitelisted {
        "SAFE "
    } else if verdict.is_dga {
        "DGA"
    } else {
        "LEGIT"
    };
    
    let prob_str = if verdict.whitelisted {
        " -- ".to_string()
    } else {
        format!("{:.1}%", verdict.probability * 100.0)
    };

    println!("[{label}] {prob_str}  {}", verdict.domain);

    if explain && !verdict.whitelisted {
        println!("  Feature contributions:");
        for (name, value) in &verdict.features {
            println!("      {:<25} {:.6}", name, value);
        }
    }

    Ok(())
}