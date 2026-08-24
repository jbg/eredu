use std::{fs, path::PathBuf, process::ExitCode};

use clap::Parser;
use eredu_evaluation::{compare_checkpoint_artifacts, CheckpointParityOptions, LogitTolerance};

#[derive(Debug, Parser)]
#[command(about = "Compare portable Eredu/backend reference evidence")]
struct Args {
    #[arg(long)]
    actual: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.02)]
    relative_l2_max: f64,
    #[arg(long, default_value_t = 0.999)]
    cosine_similarity_min: f64,
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    #[arg(long, default_value_t = 4)]
    top_k_overlap_min: usize,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    require_unambiguous_argmax_match: bool,
    #[arg(long, default_value_t = 0.0)]
    argmax_margin_min: f32,
    #[arg(long)]
    overwrite: bool,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Args) -> Result<bool, Box<dyn std::error::Error>> {
    if args.output.exists() && !args.overwrite {
        return Err(format!(
            "refusing to replace {}; pass --overwrite",
            args.output.display()
        )
        .into());
    }
    let report = compare_checkpoint_artifacts(
        &args.actual,
        &args.reference,
        CheckpointParityOptions {
            logits: LogitTolerance {
                relative_l2_max: args.relative_l2_max,
                cosine_similarity_min: args.cosine_similarity_min,
                top_k: args.top_k,
                top_k_overlap_min: args.top_k_overlap_min,
                require_unambiguous_argmax_match: args.require_unambiguous_argmax_match,
                argmax_margin_min: args.argmax_margin_min,
            },
        },
    )?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "comparison={} passed={}",
        args.output.display(),
        report.passed
    );
    Ok(report.passed)
}
