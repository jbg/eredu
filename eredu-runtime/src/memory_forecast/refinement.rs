//! Full-canvas refinement and causal finalization over one parameter owner.
use super::*;

/// Selected block geometry remains complete when the output allowance is short.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementMemoryPlan {
    /// Trained invocation width, including seeded and unresolved positions.
    pub canvas_positions: std::num::NonZeroU32,
    /// Canonical token domain used by full position-score rows.
    pub vocabulary_size: std::num::NonZeroU32,
    /// Native live score, feedback, noise and processing storage per position/entry.
    pub bytes_per_vocabulary_entry: MemoryBytes,
    /// Retained preceding token canvases used by entropy stability checks.
    pub stability_history: u32,
}

/// Projects transient block state separately from the canonical causal frontier.
pub fn estimate_refinement_memory(
    request: &GenerationMemoryRequest,
    plan: &RefinementMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    estimate_refinement_inner(request, plan, None)
}

/// Uses installed native cache capacity without charging completed prompt execution.
pub fn estimate_refinement_continuation_memory(
    request: &GenerationMemoryRequest,
    plan: &RefinementMemoryPlan,
    continuation: &ContinuationMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let continuation = continuation.with_logical_state_bounds(request)?;
    estimate_refinement_inner(request, plan, Some(&continuation))
}

fn estimate_refinement_inner(
    request: &GenerationMemoryRequest,
    plan: &RefinementMemoryPlan,
    continuation: Option<&ContinuationMemoryPlan>,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    plan.bytes_per_vocabulary_entry.validate()?;
    let mut estimate = match continuation {
        Some(native) => estimate_continuation_memory(request, native)?,
        None => estimate_generation_memory(request)?,
    };
    let output = request
        .max_output_tokens
        .unwrap_or(request.forecast_output_tokens);
    if output == 0 {
        return Ok(estimate);
    }
    let checked_add = |a: u64, b: u64| {
        a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
            operation: "refinement frontier",
        })
    };
    let checked_mul = |a: u64, b: u64| {
        a.checked_mul(b).ok_or(CapabilityError::ArithmeticOverflow {
            operation: "refinement numerical storage",
        })
    };
    let canvas = u64::from(plan.canvas_positions.get());
    let frontier = match continuation {
        Some(native) => checked_add(native.current_positions, native.additional_input_tokens)?,
        None => checked_add(request.input.model_positions, output)?,
    };
    let numerical_entries = checked_mul(
        checked_mul(canvas, u64::from(plan.vocabulary_size.get()))?,
        request.batch_size,
    )?;
    let numerical = MemoryBytes {
        lower_bytes: 0,
        upper_bytes: plan.bytes_per_vocabulary_entry.upper_bytes.map(|n| checked_mul(n, numerical_entries)).transpose()?,
        kind: eredu_core::ObservationKind::Estimated,
        detail: "complete refinement position scores, feedback, noise and processing envelope; independent of shortened output allowance".into(),
    };
    // Canvas tokens, selected-token retention, validity and confidence coordinates.
    let canvas_storage = MemoryBytes::estimated(
        0,
        checked_mul(
            checked_mul(canvas, request.batch_size)?,
            checked_add(64, checked_mul(8, u64::from(plan.stability_history))?)?,
        )?,
        "canvas token, selection, validity and confidence storage envelope",
    );
    for (domain, pool) in estimate.domains.iter_mut().zip(&request.domains) {
        domain.phases.retain(|p| p.phase != MemoryPhase::Decode);
        let mut selected = pool.clone();
        for execution in &mut selected.executions {
            execution.logits = LogitsWorkspace::EveryPosition;
        }
        let mut refinement = phase(
            &selected,
            request,
            MemoryPhase::Refinement,
            checked_add(frontier, canvas)?,
            canvas,
        )?;
        let canonical =
            phase(&selected, request, MemoryPhase::Refinement, frontier, 0)?.persistent_state;
        let transient = MemoryBytes::estimated(
            0,
            refinement
                .persistent_state
                .upper_bytes
                .unwrap_or(u64::MAX)
                .saturating_sub(canonical.lower_bytes),
            "transient block K/V; canonical prefix remains unchanged",
        );
        let transient = if refinement.persistent_state.upper_bytes.is_none() {
            MemoryBytes::unknown(transient.detail)
        } else {
            transient
        };
        refinement.persistent_state = continuation
            .filter(|_| !pool.executions.is_empty())
            .map_or(canonical, |n| n.peak_state.clone());
        refinement.workspace = refinement
            .workspace
            .add(&transient)?
            .add(&numerical)?
            .add(&canvas_storage)?;
        refinement.total = phase_total(&refinement)?;
        let mut commit = phase(
            &selected,
            request,
            MemoryPhase::RefinementCommit,
            frontier,
            if continuation.is_some() {
                canvas
            } else {
                canvas.min(output)
            },
        )?;
        if let Some(native) = continuation.filter(|_| !pool.executions.is_empty()) {
            commit.persistent_state = native.peak_state.clone();
            // Native capacity can exceed the logical payload used by ordinary scratch sizing.
            let mut copy = native.peak_state.clone();
            copy.lower_bytes = 0;
            copy.detail =
                "native causal-prefix replacement envelope during refinement finalization".into();
            commit.workspace = commit.workspace.add(&copy)?;
            refinement.workspace = refinement.workspace.add(&copy)?;
            refinement.total = phase_total(&refinement)?;
        }
        commit.retained_input = commit
            .retained_input
            .add(&numerical)?
            .add(&canvas_storage)?;
        commit.total = phase_total(&commit)?;
        domain.phases.extend([refinement, commit]);
        domain.generation_peak = domain
            .phases
            .iter()
            .filter(|p| p.phase != MemoryPhase::Loading)
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        domain.overall_peak = domain
            .phases
            .iter()
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        domain.additional_generation_peak = domain
            .generation_peak
            .additional(pool.already_resident_bytes);
        domain.generation_fit = fit(
            &domain.generation_peak,
            pool.already_resident_bytes,
            &pool.budget,
        )?;
        domain.fit = fit(
            &domain.overall_peak,
            pool.already_resident_bytes,
            &pool.budget,
        )?;
    }
    estimate.fit = if estimate
        .domains
        .iter()
        .any(|d| d.fit == MemoryFit::LikelyShortfall)
    {
        MemoryFit::LikelyShortfall
    } else if estimate
        .domains
        .iter()
        .all(|d| d.fit == MemoryFit::LikelyFit)
    {
        MemoryFit::LikelyFit
    } else {
        MemoryFit::InsufficientInformation
    };
    estimate.assumptions.push("Refinement reads the canonical causal prefix and retains complete invocation geometry; finalized tokens receive a causal append. Every phase reuses the same resident parameter catalog.".into());
    Ok(estimate)
}

fn phase_total(p: &PhaseMemoryEstimate) -> Result<MemoryBytes, CapabilityError> {
    p.parameters
        .add(&p.persistent_state)?
        .add(&p.retained_input)?
        .add(&p.workspace)?
        .add(&p.staging)?
        .add(&p.backend_overhead)
}
