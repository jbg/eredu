//! Application memory advice; estimation never changes backend validity checks.
use super::*;
use eredu::api::{
    inspected_generation_memory_request, AttentionWorkspace, CacheUpdateWorkspace,
    GenerationMemoryEstimate, GenerationMemoryOptions, GenerationMemoryPlacement, LogitsWorkspace,
    MemoryBudget, MemoryBytes, MemoryDomain, MemoryFit,
};
use eredu_runtime::memory_estimation::{estimate_generation_memory, GenerationMemoryRequest};

#[derive(Serialize)]
pub(super) struct Report {
    pub estimate: GenerationMemoryEstimate,
    pub requested_chunk_tokens: u64,
    pub effective_chunk_tokens: u64,
    pub reserve_bytes: u64,
    pub capacity_comparisons: Vec<(MemoryDomain, MemoryBudget)>,
    pub recommendations: Vec<String>,
}

pub(super) fn requested(args: &Cli) -> bool {
    args.memory_report.is_some() || args.memory_budget_bytes.is_some()
}

pub(super) fn report(
    args: &Cli,
    path: &Path,
    plan: &ExecutionPlan,
    input_tokens: u64,
    output_tokens: Option<u64>,
    already_resident: Option<u64>,
) -> Result<Report> {
    let inspection = eredu_backend_mlx::native::inspect_model_preparation(
        path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(
            MlxBackendFactory::default().load_request_for_plan(plan)?,
        ),
    )?;
    let hardware = discover_local_hardware();
    let placement = if args.device == CliDevice::Cpu {
        GenerationMemoryPlacement::Host
    } else {
        match hardware.physical_memory_semantics {
            HardwareMemorySemantics::Unified => GenerationMemoryPlacement::Unified,
            HardwareMemorySemantics::SeparateTiers => GenerationMemoryPlacement::Separate {
                device: args.device.to_string(),
            },
            _ => GenerationMemoryPlacement::Unknown,
        }
    };
    let input = eredu_core::InputTokenCount::text(input_tokens);
    let mut options = if let Some(cache_limit) = args
        .mlx_cache_limit_bytes
        .filter(|_| already_resident.is_none())
    {
        // Cold explicit overrides describe the proposed policy without changing
        // process-global allocator state. Loaded reports sample the actual policy.
        let mut options = GenerationMemoryOptions::new(input, placement);
        options.backend_overhead = MemoryBytes::estimated(0,
            cache_limit.checked_add(64 * 1024 * 1024).context("memory overhead arithmetic overflow")?,
            "proposed MLX allocator-cache limit plus 64 MiB graph/driver planning allowance; uncalibrated outside the documented matrix");
        options
    } else {
        GenerationMemoryOptions::for_local_backend(input, placement)
    };
    options.max_output_tokens = output_tokens;
    options.chunked_prefill_supported = true;
    options.prefill_chunk_tokens = if args.prefill_chunk_size == 0 {
        input_tokens
    } else {
        args.prefill_chunk_size as u64
    };
    // Conservative fallback covers materialized attention as well as kernels that
    // avoid those matrices. These are planning assumptions, not allocator facts.
    options.attention = AttentionWorkspace::ScoreMatrixUpperBound;
    options.cache_update = CacheUpdateWorkspace::CopyState;
    options.logits = LogitsWorkspace::FinalPosition;
    let host_available = observed_u64(&hardware.available_memory_bytes);
    options.budget = MemoryBudget {
        application_limit_bytes: args.memory_budget_bytes,
        available_bytes: if args.device == CliDevice::Cpu
            || hardware.physical_memory_semantics == HardwareMemorySemantics::Unified
        {
            host_available
        } else {
            selected_device_available_memory(&hardware, plan.device())
        },
        reserve_bytes: args.memory_reserve_bytes,
    };
    options.host_budget = MemoryBudget {
        available_bytes: host_available,
        reserve_bytes: args.memory_reserve_bytes,
        ..MemoryBudget::default()
    };
    options.already_resident_bytes = already_resident.unwrap_or(0);
    let mut request = inspected_generation_memory_request(&inspection, &options)?;
    apply_execution_assumptions(&mut request, already_resident.is_some())?;
    let mut estimate = estimate_generation_memory(&request)?;
    estimate.assumptions.push("Forecast covers model/request payloads and declared backend allowances, not executable/shared-library pages, unrelated requests, or total process footprint. Reserve these separately.".into());
    if host_available.is_none() {
        estimate.assumptions.push("Point-in-time available host/unified memory is unavailable; a likely fit against an application budget does not establish current system headroom.".into());
    }
    estimate.assumptions.push("CLI uses full float32 score/probability matrices as a conservative attention fallback even when the selected MLX kernel avoids them; shortfall advice can be pessimistic.".into());
    estimate.assumptions.push(format!("CLI reserve: {} bytes per physical pool; {}.", args.memory_reserve_bytes, options.backend_overhead.detail));
    if already_resident.is_some() {
        estimate.assumptions.push("Model already loaded: loading peak is excluded; availability is compared with additional request memory.".into());
    }
    let mut recommendations = Vec::new();
    for domain in &estimate.domains {
        if let Some(phase) = domain.phases.iter().max_by_key(|p| p.total.lower_bytes) {
            let contributions = [
                ("parameters", phase.parameters.lower_bytes),
                ("persistent state", phase.persistent_state.lower_bytes),
                ("retained input", phase.retained_input.lower_bytes),
                ("execution workspace", phase.workspace.lower_bytes),
                ("materialization/loading staging", phase.staging.lower_bytes),
            ];
            if let Some((name, bytes)) = contributions.into_iter().max_by_key(|(_, n)| *n) {
                recommendations.push(format!(
                    "{:?}: dominant modeled contributor is {name} ({bytes} bytes) in {:?}.",
                    domain.domain, phase.phase
                ));
            }
        }
    }
    if request.prefill_chunk_tokens == options.prefill_chunk_tokens {
        for chunk in [256, 128] {
            if chunk < request.prefill_chunk_tokens {
                let mut candidate_options = options.clone();
                candidate_options.prefill_chunk_tokens = chunk;
                let mut candidate =
                    inspected_generation_memory_request(&inspection, &candidate_options)?;
                apply_execution_assumptions(&mut candidate, already_resident.is_some())?;
                add_candidate_advice(
                    &estimate,
                    &candidate,
                    &format!("--prefill-chunk-size {chunk}"),
                    &mut recommendations,
                )?;
            }
        }
    } else {
        recommendations.push("Selected execution preserves a full prefill pass; changing chunk size provides no estimated saving on this path.".into());
    }
    if let Some(output) = output_tokens.filter(|n| *n > 1) {
        let mut candidate = request.clone();
        candidate.max_output_tokens = Some(output / 2);
        add_candidate_advice(
            &estimate,
            &candidate,
            &format!("--max-tokens {}", output / 2),
            &mut recommendations,
        )?;
    }
    if !matches!(estimate.fit, MemoryFit::LikelyFit) {
        recommendations.push("When parameters dominate, inspect a supported quantization or residency plan separately; conversion and transfer costs can change the peak.".into());
    }
    let report = Report {
        estimate,
        requested_chunk_tokens: options.prefill_chunk_tokens,
        effective_chunk_tokens: request.prefill_chunk_tokens,
        reserve_bytes: args.memory_reserve_bytes,
        capacity_comparisons: request
            .domains
            .iter()
            .map(|d| (d.domain.clone(), d.budget.clone()))
            .collect(),
        recommendations,
    };
    if let Some(path) = &args.memory_report {
        fs::write(path, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("failed to write memory report {}", path.display()))?;
    }
    Ok(report)
}

fn apply_execution_assumptions(request: &mut GenerationMemoryRequest, loaded: bool) -> Result<()> {
    for domain in &mut request.domains {
        if loaded {
            domain.loading_peak = MemoryBytes::exact(0);
        }
        for execution in &mut domain.executions {
            let layers = u64::try_from(execution.state_layout.layer_layout().len())?;
            let upper = layers
                .checked_add(layers.div_ceil(4))
                .context("memory workspace overlap arithmetic overflow")?
                .max(1);
            execution.workspace_overlap = eredu::api::WorkspaceOverlap {
                upper_live_copies: Some(upper),
                detail: "CLI lazy-graph planning envelope: one linear activation set per local state layer plus 25% scratch margin; calibrated on SmolLM-135M Metal original/4-bit, not a universal bound".into(),
            };
        }
    }
    Ok(())
}

fn add_candidate_advice(
    original: &GenerationMemoryEstimate,
    candidate: &GenerationMemoryRequest,
    label: &str,
    advice: &mut Vec<String>,
) -> Result<()> {
    let estimate = estimate_generation_memory(candidate)?;
    for (before, after) in original.domains.iter().zip(&estimate.domains) {
        if let (Some(before), Some(after_peak)) = (
            before.generation_peak.upper_bytes,
            after.generation_peak.upper_bytes,
        ) {
            if after_peak < before {
                advice.push(format!("{label}: recomputed {:?} generation peak {after_peak} bytes ({} fewer); fit {:?}.", after.domain, before - after_peak, after.fit));
            }
        }
    }
    Ok(())
}

pub(super) fn advise(args: &Cli, report: &Report) -> Result<()> {
    eprintln!(
        "memory forecast: {:?}; planning result, not a total-process allocation guarantee",
        report.estimate.fit
    );
    for domain in &report.estimate.domains {
        eprintln!(
            "  {:?}: generation {}; additional {}; lifecycle {}",
            domain.domain,
            range(&domain.generation_peak),
            range(&domain.additional_generation_peak),
            range(&domain.overall_peak)
        );
    }
    for advice in &report.recommendations {
        eprintln!("  {advice}");
    }
    if matches!(args.memory_policy, MemoryPolicy::Refuse)
        && report.estimate.fit == MemoryFit::LikelyShortfall
    {
        bail!("memory policy refused a predicted shortfall; inspect the memory report or choose a smaller supported request/plan");
    }
    Ok(())
}

fn range(bytes: &MemoryBytes) -> String {
    match bytes.upper_bytes {
        Some(upper) if upper == bytes.lower_bytes => format!("{} bytes", upper),
        Some(upper) => format!("{}..{} bytes", bytes.lower_bytes, upper),
        None => format!("{} known bytes plus unestimated costs", bytes.lower_bytes),
    }
}
