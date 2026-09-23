//! Read-only projection from installed ordinary decode state.
use super::*;

/// Native facts for one settled continuation. The state allowances cover all
/// installed tensors, backing capacity and interior growth peaks, not just the
/// logical payload at the final position. They are not observed allocations.
pub struct ContinuationMemoryProfile {
    /// Loaded geometry with the actual continuation workspace available.
    pub loaded: LoadedMemoryProfile,
    /// Native state bounds for this exact horizon.
    pub plan: ContinuationMemoryPlan,
}

/// Horizon-specific state observations. Re-observe the owner for another horizon;
/// changing the token count alone cannot extrapolate native capacity facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuationMemoryPlan {
    /// Positions already consumed by the model, excluding a pending emitted token.
    pub current_positions: u64,
    /// Inputs submitted for the requested additional predictions, counted once.
    pub additional_input_tokens: u64,
    /// Installed logical payload floor through native retained-capacity allowance.
    pub current_state: MemoryBytes,
    /// Persistent-state envelope throughout the horizon, including the start.
    pub peak_state: MemoryBytes,
}

impl ContinuationMemoryPlan {
    /// Adds architecture-declared logical payload floors at the installed and
    /// requested frontiers. Native capacity upper bounds are preserved, including
    /// unknown bounds. This observes no allocations and grants no residency credit.
    pub fn with_logical_state_bounds(
        &self,
        request: &GenerationMemoryRequest,
    ) -> Result<Self, CapabilityError> {
        self.current_state.validate()?;
        self.peak_state.validate()?;
        if self
            .current_state
            .upper_bytes
            .zip(self.peak_state.upper_bytes)
            .is_some_and(|(current, peak)| peak < current)
        {
            return Err(CapabilityError::InvalidConfiguration {
                field: "continuation peak_state",
                detail: "horizon envelope must include installed state".into(),
            });
        }
        if request.input.model_positions != self.current_positions
            || request.batch_size != 1
            || request
                .domains
                .iter()
                .map(|d| d.executions.len())
                .sum::<usize>()
                != 1
            || request.max_output_tokens != Some(self.additional_input_tokens)
        {
            return Err(CapabilityError::InvalidConfiguration {
                field: "continuation forecast",
                detail: "requires one ordinary decode lane and a matching finite input horizon"
                    .into(),
            });
        }
        let frontier = self
            .current_positions
            .checked_add(self.additional_input_tokens)
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "continuation frontier",
            })?;

        let execution = request
            .domains
            .iter()
            .flat_map(|p| &p.executions)
            .next()
            .expect("validated single continuation execution");
        let payload = |positions| {
            let mut input = request.input;
            input.model_positions = positions;
            eredu_core::estimate_runtime_state_payload_lower_bound(
                &execution.state_layout,
                input,
                request.batch_size,
                request.scalar_bytes,
            )
        };
        let mut result = self.clone();
        let current = payload(self.current_positions)?;
        let endpoint = payload(frontier)?;
        raise_floor(&mut result.current_state, current)?;
        // Remainder-shaped and sliding state can shrink: the horizon still
        // includes the starting state. Interior capacity stays in the upper end.
        let peak_floor = result.current_state.lower_bytes.max(endpoint);
        raise_floor(&mut result.peak_state, peak_floor)?;
        Ok(result)
    }
}

fn raise_floor(bytes: &mut MemoryBytes, logical: u64) -> Result<(), CapabilityError> {
    if logical > bytes.lower_bytes {
        bytes.lower_bytes = logical;
        bytes.kind = eredu_core::ObservationKind::Estimated;
        bytes.detail.push_str("; lower bound includes required logical state payload, excluding optional tensors and allocation rounding");
    }
    bytes.validate()
}

/// Backend continuation observations never submit, settle or copy model state.
pub trait ContinuationForecastBackend: GenerationForecastBackend {
    /// Returns no projection for unsupported executables. An in-flight or poisoned
    /// owner must return an error rather than sampling inconsistent state.
    fn continuation_memory_profile(
        _runtime: &ModelRuntime<Self>,
        _additional_input_tokens: u64,
    ) -> Result<Option<ContinuationMemoryProfile>, GenerationForecastError> {
        Ok(None)
    }
}

/// Projects a settled single-lane decode continuation. No completed loading or
/// prefill phase is charged. Only declared parameter residency is deducted from
/// available capacity: native state allowances and snapshot reservations are not
/// measurements of distinct resident backing.
pub fn estimate_continuation_memory(
    request: &GenerationMemoryRequest,
    plan: &ContinuationMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let plan = plan.with_logical_state_bounds(request)?;
    let frontier = plan
        .current_positions
        .checked_add(plan.additional_input_tokens)
        .expect("validated continuation frontier");
    // Reuse ordinary validation/uncertainty reporting on a minimal request, then
    // replace every phase. It must not evaluate a fictitious full-context prefill.
    let mut validation = request.clone();
    validation.input = eredu_core::InputTokenCount::text(1);
    validation.max_output_tokens = Some(0);
    validation.prefill_chunk_tokens = 1;
    let mut result = estimate_generation_memory(&validation)?;
    result.uncertainties = memory_uncertainties(request, true);
    result.requested_positions = frontier;
    result.is_forecast = true;
    result.assumptions = vec![
        "Settled ordinary continuation: pending decode input counted once; completed prefill/loading excluded; horizon does not change generation limits.".into(),
        "Native storage and retained snapshot/branch quotas are upper allowances, not allocation observations. Only declared parameter backing is deducted from current available capacity; additional peak is conservative.".into(),
    ];
    result.uncertainties.push(plan.current_state.detail.clone());
    result.uncertainties.push(plan.peak_state.detail.clone());
    for (pool, domain) in request.domains.iter().zip(&mut result.domains) {
        let executes = !pool.executions.is_empty();
        let mut without_copy = pool.clone();
        for execution in &mut without_copy.executions {
            execution.cache_update = CacheUpdateWorkspace::InPlace;
        }
        let mut start = phase(
            &without_copy,
            request,
            MemoryPhase::ContinuationStart,
            plan.current_positions,
            0,
        )?;
        start.workspace = MemoryBytes::exact(0);
        start.persistent_state = if executes {
            plan.current_state.clone()
        } else {
            MemoryBytes::exact(0)
        };
        start.total = total(&start)?;
        let mut phases = vec![start];
        if plan.additional_input_tokens > 0 {
            let mut decode = phase(&without_copy, request, MemoryPhase::Decode, frontier, 1)?;
            if executes {
                // The normalized plan includes logical endpoint/start floors;
                // its native upper end covers capacity and interior peaks.
                decode.persistent_state = plan.peak_state.clone();
                let copy = match pool.executions[0].cache_update {
                    CacheUpdateWorkspace::InPlace => MemoryBytes::exact(0),
                    CacheUpdateWorkspace::CopyState => {
                        let mut bytes = plan.peak_state.clone();
                        bytes.lower_bytes = 0;
                        bytes.detail = "old/replacement native-state envelope overlap".into();
                        bytes
                    }
                    CacheUpdateWorkspace::Unknown => {
                        MemoryBytes::unknown("cache update overlap unavailable")
                    }
                };
                decode.workspace = decode.workspace.add(&copy)?;
            }
            decode.total = total(&decode)?;
            phases.push(decode);
        }
        let peak = phases
            .iter()
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        let verdict = fit(&peak, pool.already_resident_bytes, &pool.budget)?;
        domain.phases = phases;
        domain.additional_generation_peak = peak.additional(pool.already_resident_bytes);
        domain.generation_peak = peak.clone();
        domain.overall_peak = peak;
        domain.generation_fit = verdict;
        domain.fit = verdict;
    }
    result.fit = if result
        .domains
        .iter()
        .any(|d| d.fit == MemoryFit::LikelyShortfall)
    {
        MemoryFit::LikelyShortfall
    } else if result.domains.iter().all(|d| d.fit == MemoryFit::LikelyFit) {
        MemoryFit::LikelyFit
    } else {
        MemoryFit::InsufficientInformation
    };
    Ok(result)
}

fn total(p: &PhaseMemoryEstimate) -> Result<MemoryBytes, CapabilityError> {
    p.parameters
        .add(&p.persistent_state)?
        .add(&p.retained_input)?
        .add(&p.workspace)?
        .add(&p.staging)?
        .add(&p.backend_overhead)
}
