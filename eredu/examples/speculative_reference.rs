//! Compare ordinary and assisted literal-text generation through the public facade.
//!
//! EREDU_REFERENCE_TP/PP/EP select the topology; Ring uses MLX_RANK/MLX_HOSTFILE.
//! EREDU_REFERENCE_TARGET_DEVICE_GIB/HOST_GIB and DRAFT_DEVICE_GIB/HOST_GIB
//! (each with the EREDU_REFERENCE_ prefix) bound the per-rank parameter pools.
//! EREDU_REFERENCE_CONTROLLED=1 selects explicit stepping; SNAPSHOTS=1 with the
//! same prefix also checks snapshot restore and unadvanced child exchange.
//! SNAPSHOT_BOUNDARIES (same prefix, default 1) selects settled round trips.
//! EREDU_REFERENCE_FORECAST=1 observes settled zero- and three-token outlooks.
//! BASELINE_CACHE_PARITY=1 (same prefix) permits only managed cache-budget
//! differences from a retained ordinary baseline, recording both target plans.
//! ALLOCATOR_CACHE_GIB (same prefix) explicitly bounds unused native buffers.
use eredu::api::{
    GenerationControlOptions, GenerationRequest, LoadedModel, OutputMode,
    PreparedChatGenerationSettings, PreparedChatInput,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::decoding::{DecodingMode, SpeculationOptionsRequest};
use eredu_core::{
    CompletionCancellationMode, DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan,
    GenerationCancellationToken, GenerationConfigOverrides, ParallelRankTopology, ParallelTopology,
    ResidencyPlan,
};
use serde_json::{json, Value};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use eredu_evaluation::checkpoint_reference::{
    sha256, validator_sha256, verify_metadata, verify_payload,
};

fn environment_usize(name: &str, default: usize) -> anyhow::Result<usize> {
    Ok(std::env::var(name)
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(default))
}

fn budget(name: &str, default_gib: usize) -> anyhow::Result<u64> {
    u64::try_from(environment_usize(name, default_gib)?)?
        .checked_mul(1 << 30)
        .ok_or_else(|| anyhow::anyhow!("{name} overflows bytes"))
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!((6..=9).contains(&args.len()),
        "usage: speculative_reference TARGET ASSISTANT TARGET_REFERENCE_DIR ASSISTANT_REFERENCE_JSON INVENTORY OUTPUT [MAX_TOKENS=4] [PROPOSALS=1] [BASELINE_EVIDENCE_JSON]");
    let target = PathBuf::from(&args[0]);
    let assistant = PathBuf::from(&args[1]);
    let reference_dir = PathBuf::from(&args[2]);
    let topology = ParallelTopology::new(
        environment_usize("EREDU_REFERENCE_TP", 1)?,
        environment_usize("EREDU_REFERENCE_PP", 1)?,
        environment_usize("EREDU_REFERENCE_EP", 1)?,
        1,
    )?;
    let rank = ParallelRankTopology::new(topology, environment_usize("MLX_RANK", 0)?)?;
    let mut output = PathBuf::from(&args[5]);
    if topology.world_size() > 1 {
        output = output.join(format!("rank-{}", rank.global_rank()));
    }
    let baseline_cache_parity = environment_usize("EREDU_REFERENCE_BASELINE_CACHE_PARITY", 0)? != 0;
    let controlled = environment_usize("EREDU_REFERENCE_CONTROLLED", 0)? != 0;
    let snapshots = environment_usize("EREDU_REFERENCE_SNAPSHOTS", 0)? != 0;
    let snapshot_boundaries = environment_usize("EREDU_REFERENCE_SNAPSHOT_BOUNDARIES", 1)?;
    anyhow::ensure!(
        !snapshots || snapshot_boundaries > 0,
        "snapshot boundary count must be positive"
    );
    let forecast = environment_usize("EREDU_REFERENCE_FORECAST", 0)? != 0;
    anyhow::ensure!(
        !forecast || (controlled && snapshots),
        "forecast checks require controlled snapshot boundaries"
    );
    anyhow::ensure!(
        !snapshots || controlled,
        "snapshot checks require controlled execution"
    );
    let maximum: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let proposals = NonZeroUsize::new(args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(1))
        .ok_or_else(|| anyhow::anyhow!("proposal capacity must be positive"))?;
    anyhow::ensure!(maximum >= 3, "at least three output tokens are required");
    let reference_bytes = std::fs::read(reference_dir.join("reference-evidence.json"))?;
    let assistant_bytes = std::fs::read(&args[3])?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    let assistant_reference: Value = serde_json::from_slice(&assistant_bytes)?;
    let inventory: Value = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let identity = |r: &Value, key| {
        r[key]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("reference omits {key}"))
    };
    let target_repository = identity(&reference, "repository")?;
    let target_revision = identity(&reference, "revision")?;
    let assistant_repository = identity(&assistant_reference, "repository")?;
    let assistant_revision = identity(&assistant_reference, "revision")?;
    let checked = |path: &Path, r: &Value, repo: &str, revision: &str| {
        verify_payload(path, r, &inventory, repo, revision)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    };
    let target_proof = checked(&target, &reference, &target_repository, &target_revision)?;
    let assistant_proof = checked(
        &assistant,
        &assistant_reference,
        &assistant_repository,
        &assistant_revision,
    )?;
    let target_metadata =
        verify_metadata(&target, &inventory, &target_repository, &target_revision)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let assistant_metadata = verify_metadata(
        &assistant,
        &inventory,
        &assistant_repository,
        &assistant_revision,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let prefill = &reference["steps"][0];
    let ids = serde_json::from_value::<Vec<u32>>(prefill["input_ids"].clone())?;
    anyhow::ensure!(!ids.is_empty(), "reference prefill is empty");
    let file = identity(prefill, "file")?;
    anyhow::ensure!(
        Path::new(&file).file_name().and_then(|p| p.to_str()) == Some(file.as_str()),
        "reference score file must be a direct child"
    );
    let scores = std::fs::read(reference_dir.join(file))?;
    let vocabulary = prefill["shape"][2]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing vocabulary"))? as usize;
    anyhow::ensure!(
        vocabulary > 0
            && prefill["sha256"] == sha256(&scores)
            && scores.len()
                == ids
                    .len()
                    .checked_mul(vocabulary)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| anyhow::anyhow!("score shape overflow"))?,
        "reference score hash or shape differs"
    );
    let row = &scores[scores.len() - vocabulary * 4..];
    let mut best = (0, f32::NEG_INFINITY);
    for (id, bytes) in row.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(bytes.try_into()?);
        anyhow::ensure!(value.is_finite(), "nonfinite reference score");
        if value > best.1 {
            best = (id, value);
        }
    }
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0")?)
        .with_residency(ResidencyPlan::DenseDiskStream {
            device_budget_bytes: budget("EREDU_REFERENCE_TARGET_DEVICE_GIB", 64)?,
            host_budget_bytes: budget("EREDU_REFERENCE_TARGET_HOST_GIB", 64)?,
            host_lookahead: 1,
            background_queue: 1,
        })
        .with_topology(topology)
        .with_max_cached_shards(12)
        .with_drafting(DraftingPlan::External {
            model: assistant.display().to_string(),
            placement: DraftPlacementPlan::Target,
            residency: ResidencyPlan::DenseDiskStream {
                device_budget_bytes: budget("EREDU_REFERENCE_DRAFT_DEVICE_GIB", 8)?,
                host_budget_bytes: budget("EREDU_REFERENCE_DRAFT_HOST_GIB", 8)?,
                host_lookahead: 1,
                background_queue: 1,
            },
            max_draft_tokens: proposals.get(),
            lookahead: false,
            adaptive_lookahead: false,
        });
    if std::env::var_os("EREDU_REFERENCE_ALLOCATOR_CACHE_GIB").is_some() {
        let bytes = budget("EREDU_REFERENCE_ALLOCATOR_CACHE_GIB", 0)?;
        eredu_backend_mlx::set_allocator_cache_limit(usize::try_from(bytes)?)?;
    }
    let mut report = json!({
        "target": {"repository":target_repository,"revision":target_revision,"payload_proof_sha256":target_proof},
        "assistant": {"repository":assistant_repository,"revision":assistant_revision,"payload_proof_sha256":assistant_proof},
        "target_metadata":target_metadata,"assistant_metadata":assistant_metadata,
        "reference_evidence_sha256":sha256(&reference_bytes),
        "assistant_reference_evidence_sha256":sha256(&assistant_bytes),
        "runner_sha256":sha256(include_bytes!("speculative_reference.rs")),
        "payload_validator_sha256":validator_sha256(),
        "plan":plan,"input_ids":ids,"maximum_tokens":maximum,"proposal_capacity":proposals,
        "global_rank":rank.global_rank(),"controlled":controlled,"snapshot_checks_requested":snapshots,
        "forecast_checks_requested":forecast,"snapshot_boundaries_requested":snapshot_boundaries,
        "baseline_cache_budget_parity_permitted":baseline_cache_parity,
        "expected_first_token":best.0,"comparison_verified":false,
        "scope":"ordinary versus external-assisted facade literal-text generation from exact token IDs; prepared chat supplies tokenizer and termination policy",
    });
    std::fs::create_dir_all(&output)?;
    let evidence = output.join("comparison-evidence.json");
    let save = |report: &Value| -> anyhow::Result<()> {
        std::fs::write(&evidence, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    let started = Instant::now();
    println!("Loading one bounded target and its selected external assistant through the facade");
    let mut factory = MlxBackendFactory::default();
    if topology.world_size() > 1 {
        let parallel = eredu_runtime::ParallelLoadRequest::new(
            rank,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            32,
            eredu_runtime::CommunicationCompletionPolicy::new(
                Duration::from_secs(300),
                CompletionCancellationMode::QuarantineUntilComplete,
            )?,
        )?;
        factory = factory.with_parallel_execution(parallel, "ring".parse()?);
    }
    let mut planned = LoadedModel::load_execution_plan(&factory, &target, &plan)?;
    anyhow::ensure!(planned.drafting().is_enabled(), "assistant is not enabled");
    report["load_seconds"] = json!(started.elapsed().as_secs_f64());
    report["allocator_cache_policy"] = json!(eredu_backend_mlx::allocator_cache_policy()?);
    report["tokenizer_fingerprint"] = json!(planned.model().tokenizer_fingerprint());
    save(&report)?;
    let prepared = planned.model_mut().prepare_chat(ChatTemplateRequest {
        messages: vec![json!({"role":"user","content":"reference"})],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let settings = || PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(maximum),
            ..Default::default()
        },
        ..Default::default()
    };
    let ordinary_ids = if let Some(path) = args.get(8) {
        let bytes = std::fs::read(path)?;
        let baseline: Value = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            baseline["baseline_generation_completed"] == true,
            "ordinary baseline is incomplete"
        );
        for key in [
            "target",
            "target_metadata",
            "input_ids",
            "maximum_tokens",
            "reference_evidence_sha256",
            "tokenizer_fingerprint",
        ] {
            anyhow::ensure!(baseline[key] == report[key], "baseline differs in {key}");
        }
        let mut baseline_plan = baseline["plan"].clone();
        let mut target_plan = report["plan"].clone();
        baseline_plan
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("missing baseline plan"))?
            .remove("drafting");
        target_plan.as_object_mut().unwrap().remove("drafting");
        let different_budgets = baseline_plan != target_plan;
        report["ordinary_baseline_target_plan"] = baseline_plan.clone();
        if baseline_cache_parity && different_budgets {
            for value in [&mut baseline_plan, &mut target_plan] {
                let residency = value["residency"]
                    .as_object_mut()
                    .ok_or_else(|| anyhow::anyhow!("baseline has no residency policy"))?;
                anyhow::ensure!(
                    residency.get("mode").and_then(Value::as_str) == Some("dense_disk_stream"),
                    "cache-budget parity requires disk-streamed policies on both sides"
                );
                for key in ["device_budget_bytes", "host_budget_bytes"] {
                    anyhow::ensure!(
                        residency
                            .get(key)
                            .and_then(Value::as_u64)
                            .is_some_and(|n| n > 0),
                        "missing positive cache budget {key}"
                    );
                    residency.remove(key);
                }
            }
        }
        anyhow::ensure!(
            baseline_plan == target_plan,
            "baseline differs beyond the admitted target execution policy"
        );
        report["ordinary_baseline_cache_budget_parity"] = json!(different_budgets);
        let values = serde_json::from_value::<Vec<u32>>(baseline["ordinary_ids"].clone())?;
        anyhow::ensure!(
            !values.is_empty() && values.len() <= maximum,
            "invalid baseline token count"
        );
        report["ordinary_baseline_sha256"] = json!(sha256(&bytes));
        println!("Reusing the verified ordinary baseline for the same target and inputs; cache-budget parity={different_budgets}");
        values
    } else {
        println!("Starting ordinary facade generation");
        let ordinary_started = Instant::now();
        let ordinary = planned.model_mut().session().generate(GenerationRequest {
            output_mode: OutputMode::Text,
            input: PreparedChatInput::token_ids(&prepared, ids.clone()),
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: GenerationCancellationToken::new(),
            control_handle: None,
            on_event: |event| eprintln!("ordinary rank={} event={event:?}", rank.global_rank()),
        })?;
        report["ordinary_generation_seconds"] = json!(ordinary_started.elapsed().as_secs_f64());
        ordinary.token_ids().to_vec()
    };
    report["ordinary_ids"] = json!(ordinary_ids);
    report["ordinary_completed_seconds"] = json!(started.elapsed().as_secs_f64());
    save(&report)?;
    anyhow::ensure!(
        ordinary_ids.first().copied() == Some(u32::try_from(best.0)?),
        "ordinary first token differs from independent prefill reference"
    );
    planned.model().synchronize()?;
    planned.model_mut().reset()?;
    report["baseline_generation_completed"] = json!(true);
    save(&report)?;
    println!("Starting coupled external-assistant facade generation");
    let mut assisted_settings = settings();
    assisted_settings.decoding_mode = Some(DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: proposals,
            ..Default::default()
        }
        .try_into()?,
    ));
    let request = GenerationRequest {
        output_mode: OutputMode::Text,
        input: PreparedChatInput::token_ids(&prepared, ids),
        settings: assisted_settings,
        caller_stop_sequences: &[],
        cancellation: GenerationCancellationToken::new(),
        control_handle: None,
        on_event: |event| eprintln!("assisted rank={} event={event:?}", rank.global_rank()),
    };
    let mut control_steps = Vec::new();
    let mut lifecycle = Vec::new();
    let mut forecasts = Vec::new();
    let assisted = if controlled {
        planned.with_controlled(
            request,
            GenerationControlOptions {
                speculative_snapshots: snapshots.then_some(eredu_core::execution_control::SnapshotLimits {
                    max_snapshots: 2,
                    max_branches: 2,
                    retained_bytes: 1 << 30,
                    cumulative_copy_bytes: 8 << 30,
                }),
                ..Default::default()
            },
            |control| {
                let session = control.speculative().expect("assisted request selects speculation");
                while let Some(step) = session.step()? {
                    eprintln!("controlled rank={} sequence={} committed={:?}", rank.global_rank(), step.sequence, step.committed_token_ids);
                    control_steps.push(json!({"sequence":step.sequence,"epoch":step.epoch,
                        "committed_token_ids":step.committed_token_ids,"verification":step.verification}));
                    if snapshots && lifecycle.len() < snapshot_boundaries && session.can_snapshot() {
                        let prefix = session.token_ids().to_vec();
                        let saved = session.snapshot()?;
                        let saved_usage = session.snapshot_usage();
                        session.restore(&saved)?;
                        let child = session.fork(&saved)?;
                        session.exchange(&child)?;
                        session.exchange(&child)?;
                        assert_eq!(session.token_ids(), prefix, "snapshot/fork round trip changes the committed prefix");
                        session.release_branch(&child)?;
                        session.release_snapshot(&saved)?;
                        let released_usage = session.snapshot_usage();
                        assert!(released_usage.cumulative_copy_bytes > saved_usage.cumulative_copy_bytes);
                        assert_eq!(released_usage.snapshots, 0);
                        assert_eq!(released_usage.branches, 0);
                        lifecycle.push(json!({"prefix":prefix,"saved_usage":saved_usage,
                            "released_usage":released_usage,"epoch":session.epoch(),
                            "scope":"snapshot restore and unadvanced child exchange round trip; subsequent generation checks continuation against the ordinary baseline"}));
                    }
                    if forecast && forecasts.len() < lifecycle.len() && session.can_snapshot() {
                        let prefix = session.token_ids().to_vec();
                        let usage = session.snapshot_usage();
                        let epoch = session.epoch();
                        let mut horizons = Vec::new();
                        for additional in [0, 3] {
                            let outlook = session.forecast_remaining_generation(additional, &Default::default()).map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
                            assert!(outlook.speculative.conditioned.is_some(), "missing retained conditioned topology");
                            horizons.push(json!({"additional_tokens":additional,"outlook":outlook}));
                        }
                        forecasts.push(json!({"prefix":prefix,"epoch":epoch,"horizons":horizons}));
                        assert_eq!(session.token_ids(), prefix);
                        assert_eq!(session.snapshot_usage(), usage, "forecast consumes snapshot/copy budget");
                        assert_eq!(session.epoch(), epoch, "forecast changes installed state epoch");
                    }
                    // Retain completed checks even if a later native invocation
                    // fails or the host terminates this validation process.
                    report["controlled_steps"] = json!(control_steps);
                    report["snapshot_round_trips"] = json!(lifecycle);
                    report["continuation_forecasts"] = json!(forecasts);
                    let bytes = serde_json::to_vec_pretty(&report)
                        .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
                    std::fs::write(&evidence, bytes)
                        .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
                }
                Ok(())
            },
        )?
    } else {
        planned.generate(request)?
    };
    report["controlled_steps"] = json!(control_steps);
    report["snapshot_round_trips"] = json!(lifecycle);
    report["continuation_forecasts"] = json!(forecasts);
    planned.model().synchronize()?;
    let stats = assisted.stats().drafting.as_ref().expect("assisted generation retains draft statistics");
    let equal = assisted.token_ids() == ordinary_ids
        && (!snapshots || lifecycle.len() == snapshot_boundaries)
        && (!forecast || forecasts.len() == snapshot_boundaries);
    report["assisted_ids"] = json!(assisted.token_ids());
    report["speculation"] = json!({"draft_tokens":stats.draft_tokens(), "accepted_tokens":stats.accepted_tokens(),
        "rounds":stats.rounds(), "accept_lens":stats.accept_lens(), "target_tokens":stats.target_tokens()});
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["comparison_verified"] = json!(equal && stats.draft_tokens() > 0 && stats.rounds() > 0);
    save(&report)?;
    anyhow::ensure!(
        report["comparison_verified"] == true,
        "facade comparison differs or performs no assistant verification; evidence retained"
    );
    println!(
        "Ordinary and assisted token IDs agree; {} proposals, {} accepted, {} verification rounds",
        stats.draft_tokens(),
        stats.accepted_tokens(),
        stats.rounds()
    );
    Ok(())
}
