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

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Check {
        domain: String,
        #[arg(long)]
        explain: bool,
    },
    Batch {
        file: PathBuf,
        #[arg(long)]
        explain: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let templar = DnsTemplar::load(
        cli.model.to_str().unwrap(),
        cli.threshold.to_str().unwrap(),
        cli.ngram_table.to_str().unwrap(),
        cli.tld_freq.to_str().unwrap(),
    )?;

    match cli.command {
        Command::Check { domain, explain } => {
            classify_and_print(&templar, &domain, explain)?;
        }
        Command::Batch { file, explain } => {
            let contents = std::fs::read_to_string(&file)?;
            for line in contents.lines() {
                let domain = line.trim();
                if !domain.is_empty() {
                    classify_and_print(&templar, domain, explain)?;
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
) -> Result<(), Box<dyn std::error::Error>> {
    let verdict = templar.classify(domain)?;
    let label = if verdict.is_dga { "DGA  " } else { "LEGIT" };
    let prob = verdict.probability * 100.0;

    println!("[{label}] {:.1}%  {}", prob, verdict.domain);

    if explain {
        println!("  Feature contributions:");
        for (name, value) in &verdict.features {
            println!("      {:<25} {:.6}", name, value);
        }
    }

    Ok(())
}