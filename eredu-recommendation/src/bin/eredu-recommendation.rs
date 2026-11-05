use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use eredu_recommendation::{
    builder::{write_json, CatalogueBuilder},
    catalogue::Catalogue,
    forecast::DomainBudget,
    quality::BenchmarkIndex,
    transport::Hub,
    worker::{self, WorkerLimits},
    RecommendationPolicy,
};
use std::{path::PathBuf, time::Duration};

#[derive(Parser)]
#[command(about = "Discover checkpoints and assess revision-pinned Eredu recommendations")]
struct Cli {
    #[arg(long)]
    hf_token_file: Option<PathBuf>,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Action,
}

#[derive(Args)]
struct Planning {
    /// JSON text workload, backend, placement, and allocator assumptions.
    #[arg(long)]
    profile: PathBuf,
    /// Exact engine build identifier to record with compatibility evidence.
    #[arg(long)]
    engine_revision: String,
    #[arg(long, default_value_t = 60)]
    timeout_seconds: u64,
    #[arg(long,default_value_t=2 * 1024 * 1024 * 1024)]
    max_memory_bytes: u64,
}

#[derive(Subcommand)]
enum Action {
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Inspect {
        repo: String,
        #[arg(long)]
        revision: Option<String>,
        #[command(flatten)]
        planning: Planning,
    },
    Build {
        /// Explicit benchmark index, or provide an Artificial Analysis key file.
        #[arg(
            long,
            conflicts_with = "aa_key_file",
            required_unless_present = "aa_key_file"
        )]
        index: Option<PathBuf>,
        #[arg(long)]
        aa_key_file: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[command(flatten)]
        planning: Planning,
    },
    Refresh {
        catalogue: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[command(flatten)]
        planning: Planning,
    },
    Check {
        catalogue: PathBuf,
        #[arg(long)]
        engine_revision: String,
    },
    Rank {
        catalogue: PathBuf,
        #[arg(long)]
        engine_revision: String,
        /// JSON array of physical domains and their current memory budgets.
        #[arg(long)]
        budgets: PathBuf,
        #[arg(long)]
        allow_unbenchmarked: bool,
        #[arg(long)]
        allow_nonconversational: bool,
    },
}

fn read<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    serde_json::from_reader(std::fs::File::open(path)?).context("Invalid JSON input")
}

fn builder(hub: Hub, planning: Planning) -> Result<CatalogueBuilder> {
    Ok(CatalogueBuilder {
        hub,
        executable: std::env::current_exe()?,
        profile: read(&planning.profile)?,
        engine_revision: planning.engine_revision,
        limits: WorkerLimits {
            timeout: Duration::from_secs(planning.timeout_seconds),
            max_memory_bytes: planning.max_memory_bytes,
            ..Default::default()
        },
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        return worker::run();
    }
    let cli = Cli::parse();
    // Offline operations do not initialize an HTTP client or resolve credentials.
    match cli.command {
        Action::Check {
            catalogue,
            engine_revision,
        } => {
            let catalogue: Catalogue = read(&catalogue)?;
            catalogue.validate(&engine_revision)?;
            println!("Catalogue valid: {} repositories", catalogue.models.len());
        }
        Action::Rank {
            catalogue,
            engine_revision,
            budgets,
            allow_unbenchmarked,
            allow_nonconversational,
        } => {
            let catalogue: Catalogue = read(&catalogue)?;
            let budgets: Vec<DomainBudget> = read(&budgets)?;
            let candidates = catalogue.assess(&engine_revision, &budgets)?;
            let policy = RecommendationPolicy {
                require_benchmark: !allow_unbenchmarked,
                require_conversational: !allow_nonconversational,
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&policy.rank(&candidates))?
            );
        }
        command => {
            let token = cli
                .hf_token_file
                .map(std::fs::read_to_string)
                .transpose()?
                .map(|s| s.trim().to_owned());
            let hub = Hub::new(token, cli.cache_dir)?;
            match command {
                Action::Search { query, limit } => println!(
                    "{}",
                    serde_json::to_string_pretty(&hub.search(&query, limit).await?)?
                ),
                Action::Inspect {
                    repo,
                    revision,
                    planning,
                } => {
                    let builder = builder(hub, planning)?;
                    let (repository, manifest) =
                        builder.hub.repository(&repo, revision.as_deref()).await?;
                    for variant in &repository.variants {
                        let metadata = builder
                            .hub
                            .metadata(&repository, variant, &manifest)
                            .await?;
                        let inspection = match metadata {
                            Ok(metadata) => worker::inspect(
                                &builder.executable,
                                &builder.profile,
                                metadata,
                                builder.limits,
                            )?,
                            Err(reasons) => eredu_recommendation::forecast::Inspection {
                                provenance: None,
                                compatibility: eredu_recommendation::Compatibility::Incompatible,
                                reasons,
                                forecast: None,
                            },
                        };
                        println!(
                            "{}",
                            serde_json::to_string(
                                &serde_json::json!({"engineRevision":builder.engine_revision,"profile":builder.profile,"repo":repo,"revision":repository.revision,"variant":variant,"inspection":inspection})
                            )?
                        );
                    }
                }
                Action::Build {
                    index,
                    aa_key_file,
                    output,
                    report,
                    planning,
                } => {
                    let index = match index {
                        Some(path) => read(&path)?,
                        None => {
                            BenchmarkIndex::artificial_analysis(&std::fs::read_to_string(
                                aa_key_file.context("Missing benchmark credential")?,
                            )?)
                            .await?
                        }
                    };
                    let builder = builder(hub, planning)?;
                    let (catalogue, diagnostics) = builder.build(&index).await?;
                    write_json(&report, &diagnostics)?;
                    write_json(&output, &catalogue)?;
                }
                Action::Refresh {
                    catalogue,
                    report,
                    planning,
                } => {
                    let original: Catalogue = read(&catalogue)?;
                    let builder = builder(hub, planning)?;
                    let (refreshed, diagnostics) = builder.refresh(&original).await?;
                    write_json(&report, &diagnostics)?;
                    write_json(&catalogue, &refreshed)?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}
