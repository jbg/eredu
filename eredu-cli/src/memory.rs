//! Application memory advice; estimation never changes backend validity checks.
use super::*;
use eredu::api::{
    forecast_inspected_generation, ForecastCalibration, GenerationForecast,
    GenerationForecastOptions, GenerationMemoryEstimate, GenerationMemoryOptions, MemoryBudget,
    MemoryBytes, MemoryDomain, MemoryFit, SpeculativeForecastBackend,
};

#[derive(Serialize)]
pub(super) struct Report {
    pub estimate: GenerationMemoryEstimate,
    pub execution: eredu::api::ForecastExecutionContract,
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
) -> Result<Report> {
    let inspection = eredu_backend_mlx::native::inspect_model_preparation(
        path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(
            MlxBackendFactory::default().load_request_for_plan(plan)?,
        ),
    )?;
    let input = eredu_core::InputTokenCount::text(input_tokens);
    let mut options = GenerationMemoryOptions::for_local_device(input, plan.device())?;
    if let Some(cache_limit) = args.mlx_cache_limit_bytes {
        // Cold explicit overrides describe the proposed policy without changing
        // process-global allocator state. Loaded reports sample the actual policy.
        options.backend_overhead =
            ForecastCalibration::default().allocator_overhead(cache_limit, 0);
    }
    options.max_output_tokens = output_tokens;
    options.prefill_chunk_tokens = if args.prefill_chunk_size == 0 {
        input_tokens
    } else {
        args.prefill_chunk_size as u64
    };
    options.budget.application_limit_bytes = args.memory_budget_bytes;
    options.budget.reserve_bytes = args.memory_reserve_bytes;
    options.host_budget.reserve_bytes = args.memory_reserve_bytes;
    let mut forecast =
        forecast_inspected_generation(&inspection, &options, &ForecastCalibration::default())?;
    if !matches!(plan.drafting(), DraftingPlan::Disabled) {
        eredu::api::mark_speculative_forecast(&mut forecast)?;
    }
    finish_report(args, forecast)
}

pub(super) fn report_loaded<B: SpeculativeForecastBackend<D>, D>(
    args: &Cli,
    model: &LoadedModel<B>,
    tokens: &[u32],
    settings: PreparedChatGenerationSettings,
    drafting: Option<&eredu_core::SpeculativeDraft<'_, D>>,
    speculative: PreparedChatSpeculativeGenerationOptions,
) -> Result<Report> {
    let options = GenerationForecastOptions {
        budget: MemoryBudget {
            application_limit_bytes: args.memory_budget_bytes,
            reserve_bytes: args.memory_reserve_bytes,
            ..MemoryBudget::default()
        },
        host_budget: MemoryBudget {
            reserve_bytes: args.memory_reserve_bytes,
            ..MemoryBudget::default()
        },
        ..GenerationForecastOptions::default()
    };
    let forecast = match drafting {
        Some(drafting) => model.forecast_speculative_token_ids(
            tokens,
            settings,
            drafting,
            speculative,
            &options,
        )?,
        None => model.forecast_token_ids(tokens, settings, &options)?,
    };
    finish_report(args, forecast)
}

fn finish_report(args: &Cli, forecast: GenerationForecast) -> Result<Report> {
    let request = &forecast.request;
    let estimate = &forecast.estimate;
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
    if forecast.execution.full_pass_reason.is_none() {
        for chunk in [256, 128] {
            if chunk < request.prefill_chunk_tokens {
                let candidate = forecast.with_prefill_chunk(chunk)?;
                add_candidate_advice(
                    estimate,
                    &candidate.estimate,
                    &format!("--prefill-chunk-size {chunk}"),
                    &mut recommendations,
                )?;
            }
        }
    } else if let Some(reason) = &forecast.execution.full_pass_reason {
        recommendations.push(format!(
            "Full prefill pass: {reason}; smaller chunks provide no estimated saving."
        ));
    }
    if let Some(output) = request.max_output_tokens.filter(|n| *n > 1) {
        let candidate = forecast.with_max_output_tokens(output / 2)?;
        add_candidate_advice(
            estimate,
            &candidate.estimate,
            &format!("--max-tokens {}", output / 2),
            &mut recommendations,
        )?;
    }
    if !matches!(estimate.fit, MemoryFit::LikelyFit) {
        recommendations.push("When parameters dominate, inspect a supported quantization or residency plan separately; conversion and transfer costs can change the peak.".into());
    }
    let report = Report {
        estimate: estimate.clone(),
        execution: forecast.execution.clone(),
        requested_chunk_tokens: forecast.requested_chunk_tokens,
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

fn add_candidate_advice(
    original: &GenerationMemoryEstimate,
    estimate: &GenerationMemoryEstimate,
    label: &str,
    advice: &mut Vec<String>,
) -> Result<()> {
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
