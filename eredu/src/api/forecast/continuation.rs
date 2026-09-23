use super::*;
use eredu_core::{ModelRuntime, PendingTextInput};
use eredu_runtime::execution_control::{TextSnapshotBackend, TraceLimits};
pub use eredu_runtime::memory_forecast::{
    estimate_continuation_memory, ContinuationForecastBackend, ContinuationMemoryPlan,
};

/// A read-only, horizon-specific continuation observation. Request a new forecast
/// after advancement, restoration, or a horizon change; native capacity bounds
/// cannot be extrapolated by changing `request.max_output_tokens` alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuationForecast {
    /// Horizon-specific capture geometry and charged history, when instrumented.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture: Option<CaptureMemoryPlan>,
    /// Total and additional continuation peaks, with no loading/prefill phases.
    pub estimate: GenerationMemoryEstimate,
    /// Physical pools, budgets and calibrated decode geometry used by the estimate.
    pub request: GenerationMemoryRequest,
    /// Actual cache frontier and native retained-state bounds for this horizon.
    pub continuation: ContinuationMemoryPlan,
}

impl<B: ContinuationForecastBackend + TextSnapshotBackend> super::super::TextGeneration<'_, B> {
    /// Forecasts `additional_tokens` further predictions at a settled decode
    /// boundary, without consuming a token or changing this iterator's limit.
    /// Initial prefill, terminal, and in-flight states are rejected. If native
    /// work is still in flight, call `synchronize()` on this iterator
    /// first; this method never waits for or submits execution.
    /// External decoders, application copies and unrelated runs are excluded.
    pub fn forecast_remaining_generation(
        &self,
        additional_tokens: u64,
        options: &GenerationForecastOptions,
    ) -> Result<ContinuationForecast, GenerationForecastError> {
        let (runtime, state, pending) = self.inner.observation_parts();
        continuation_forecast::<B>(runtime, state, pending, additional_tokens, options, None)
    }
}

pub(crate) fn continuation_forecast<B: ContinuationForecastBackend + TextSnapshotBackend>(
    runtime: &ModelRuntime<B>,
    state: &B::TextGenerationState,
    pending: Option<PendingTextInput<&B::Prompt, &B::Token>>,
    additional_tokens: u64,
    options: &GenerationForecastOptions,
    trace: Option<TraceLimits>,
) -> Result<ContinuationForecast, GenerationForecastError> {
    if !matches!(pending, Some(PendingTextInput::Decode(_))) {
        return Err(GenerationForecastError::UnsupportedContinuation(
            "requires an advanced ordinary continuation with a pending decode token".into(),
        ));
    }
    let inputs = B::continuation_input_tokens(
        pending.as_ref().map(|p| match p {
            PendingTextInput::Prefill(p) => PendingTextInput::Prefill(*p),
            PendingTextInput::Decode(t) => PendingTextInput::Decode(*t),
        }),
        additional_tokens,
    )
    .filter(|n| *n == additional_tokens)
    .ok_or_else(|| {
        GenerationForecastError::UnsupportedContinuation(
            "backend cannot bound the pending decode input".into(),
        )
    })?;
    let mut profile = B::continuation_memory_profile(runtime, inputs)?.ok_or_else(|| {
        GenerationForecastError::UnsupportedContinuation(
            "selected executable has no settled continuation state projection".into(),
        )
    })?;
    if profile.plan.additional_input_tokens != inputs {
        return Err(failure(
            "backend continuation projection has a different horizon",
        ));
    }
    let assumptions = profile.loaded.geometry.assumptions.clone();
    let mut forecast = loaded_forecast(
        profile.loaded,
        InputTokenCount::text(1),
        0,
        eredu_core::PrefillChunkPolicy::Unchunked,
        ForecastExecutionContract {
            full_pass_reason: None,
            logits: LogitsWorkspace::FinalPosition,
        },
        options,
    )?;
    forecast.request.input = InputTokenCount::text(profile.plan.current_positions);
    forecast.request.max_output_tokens = Some(additional_tokens);
    forecast.request.forecast_output_tokens = additional_tokens;
    profile.plan = profile.plan.with_logical_state_bounds(&forecast.request)?;
    // The completed prompt no longer belongs to a raw token iterator. Controlled
    // facade retention, including its prompt and semantic buffers, is added below.
    for pool in &mut forecast.request.domains {
        pool.retained_input = MemoryBytes::exact(0);
    }
    let sampling = B::sampling_state(state);
    let current = B::estimate_sampling_state(runtime, sampling)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let growth = B::estimate_sampling_growth(runtime, sampling, additional_tokens)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let pending_bytes = B::estimate_pending_input(runtime, pending)
        .map_err(eredu_core::BackendFailure::from_error)?;
    let bound = current
        .and_then(|n| n.retained_bytes.checked_add(growth?))
        .and_then(|n| n.checked_add(pending_bytes?.retained_bytes));
    // Sampler mechanisms may retain both host history and native RNG/input arrays.
    // Without finer attribution, cover the allowance in each separate pool.
    add_retention(&mut forecast.request, bound,
        "current sampler/pending input plus future sampling storage; conservative per-pool allowance", false)?;
    let mut capture_plan = None;
    if let Some(capture) = B::capture_run(state) {
        let first = B::sampling_prediction(sampling);
        let projection = B::capture_memory_projection(
            runtime,
            capture.plan(),
            capture.intervention_plan(),
            first,
        )?;
        let mut plan = CaptureMemoryPlan::new(
            &forecast.request,
            capture.plan(),
            capture.intervention_plan(),
            trace.unwrap_or(TraceLimits {
                per_record_bytes: 0,
                total_bytes: 0,
            }),
            B::capture_transforms_complete_per_step(runtime),
            projection,
        );
        plan.first_prediction = first;
        plan.inherited_usage = capture.cumulative_usage();
        plan.apply(&mut forecast.request, additional_tokens)?;
        capture_plan = Some(plan);
    } else if let Some(trace) = trace {
        add_retention(
            &mut forecast.request,
            Some(trace.total_bytes),
            "retained compact JSON trace allowance",
            true,
        )?;
    }
    let mut result = ContinuationForecast {
        capture: capture_plan,
        estimate: estimate_continuation_memory(&forecast.request, &profile.plan)?,
        request: forecast.request,
        continuation: profile.plan,
    };
    result.estimate.assumptions.extend(assumptions);
    Ok(result)
}

pub(crate) fn add_retention(
    request: &mut GenerationMemoryRequest,
    bound: Option<u64>,
    detail: &str,
    host_only: bool,
) -> Result<(), GenerationForecastError> {
    for pool in &mut request.domains {
        if host_only && !matches!(pool.domain, MemoryDomain::Host | MemoryDomain::Unified) {
            continue;
        }
        pool.retained_input.upper_bytes = match (pool.retained_input.upper_bytes, bound) {
            (Some(a), Some(b)) => Some(a.checked_add(b).ok_or(
                CapabilityError::ArithmeticOverflow {
                    operation: "continuation retained storage",
                },
            )?),
            _ => None,
        };
        pool.retained_input.kind = eredu_core::ObservationKind::Estimated;
        pool.retained_input.detail.push_str("; ");
        pool.retained_input.detail.push_str(detail);
    }
    Ok(())
}
