//! Active and queued decoding phases share one canonical parameter/cache owner.
use super::*;

/// Read-only policies and retained mechanism facts at a settled continuation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingContinuationMemoryPlan {
    /// Queued commands remain visible until their settled handoff.
    pub state: eredu_core::decoding::DecodingState,
    /// Committed inputs appended by the admitted pending handoff, even without further output.
    #[serde(default)]
    pub settlement_input_tokens: u64,
    /// Active and pending full-canvas policies, including differing history bounds.
    pub refinement: Vec<RefinementMemoryPlan>,
    /// Active and pending shared-target speculative transaction envelopes.
    pub speculative: Vec<SpeculativeMemoryPlan>,
}

impl DecodingContinuationMemoryPlan {
    /// Cache horizon includes a committed block awaiting append and verification overshoot.
    pub fn input_horizon(
        &self,
        tokens: u64,
        current_mode_inputs: u64,
    ) -> Result<u64, CapabilityError> {
        if tokens == 0 {
            return Ok(self.settlement_input_tokens);
        }
        let mut horizon = current_mode_inputs.max(self.settlement_input_tokens);
        for refinement in &self.refinement {
            horizon = horizon.max(
                tokens
                    .checked_add(u64::from(refinement.canvas_positions.get()))
                    .ok_or(CapabilityError::ArithmeticOverflow {
                        operation: "decoding transition append horizon",
                    })?,
            );
        }
        for speculation in &self.speculative {
            horizon = horizon.max(speculative_continuation_positions(
                tokens,
                speculation.max_draft_tokens,
                speculation.scheduler,
            )?);
        }
        Ok(horizon)
    }
}

/// Composes phase peaks across an admitted transition without duplicating base residency.
pub fn estimate_decoding_continuation_memory(
    request: &GenerationMemoryRequest,
    native: &ContinuationMemoryPlan,
    decoding: &DecodingContinuationMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let mut result = estimate_continuation_memory(request, native)?;
    let mut refinement_request = request.clone();
    if decoding.settlement_input_tokens > 0
        && matches!(
            decoding.state.active,
            eredu_core::decoding::DecodingMode::Diffusion { .. }
        )
    {
        refinement_request.forecast_output_tokens =
            refinement_request.forecast_output_tokens.max(1);
        refinement_request.max_output_tokens =
            Some(refinement_request.max_output_tokens.unwrap_or(0).max(1));
    }
    for refinement in &decoding.refinement {
        merge_decoding_phase_estimate(
            &mut result,
            estimate_refinement_continuation_memory(&refinement_request, refinement, native)?,
            request,
        )?;
    }
    for speculative in &decoding.speculative {
        let continuation = SpeculativeContinuationMemoryPlan {
            additional_tokens: request.forecast_output_tokens,
            settlement_input_tokens: 0,
            target: native.clone(),
            draft: None,
            embedded: None,
            seed: MemoryBytes::exact(0),
            host_retention: MemoryBytes::exact(0),
            retained_snapshots: MemoryBytes::exact(0),
        };
        merge_decoding_phase_estimate(
            &mut result,
            estimate_speculative_continuation_memory(request, speculative, &continuation)?,
            request,
        )?;
    }
    if decoding.settlement_input_tokens > 0 {
        let mut append_request = request.clone();
        append_request.forecast_output_tokens = 0;
        append_request.max_output_tokens = Some(0);
        let mut target = native.clone();
        target.additional_input_tokens = decoding.settlement_input_tokens;
        let append = SpeculativeContinuationMemoryPlan {
            additional_tokens: 0,
            settlement_input_tokens: decoding.settlement_input_tokens,
            target,
            draft: None,
            embedded: None,
            seed: MemoryBytes::exact(0),
            host_retention: MemoryBytes::exact(0),
            retained_snapshots: MemoryBytes::exact(0),
        };
        let causal = SpeculativeMemoryPlan {
            conditioned: None,
            shared_backbone: None,
            embedded: None,
            draft: None,
            auxiliary_bytes_per_position: MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry: MemoryBytes::exact(0),
            max_draft_tokens: 0,
            scheduler: Default::default(),
            shared_allocator: true,
        };
        merge_decoding_phase_estimate(
            &mut result,
            estimate_speculative_continuation_memory(&append_request, &causal, &append)?,
            request,
        )?;
    }
    result.assumptions.push("Active and admitted pending decoding policies share one parameter catalog and canonical causal cache. Phase peaks cover settled append, full tentative canvases and verification overshoot; current sampler, pending suffix and snapshot retention remain charged across the transition.".into());
    Ok(result)
}

pub(crate) fn merge_decoding_phase_estimate(
    result: &mut GenerationMemoryEstimate,
    candidate: GenerationMemoryEstimate,
    request: &GenerationMemoryRequest,
) -> Result<(), CapabilityError> {
    for extra in candidate.domains {
        let domain = result
            .domains
            .iter_mut()
            .find(|domain| domain.domain == extra.domain)
            .ok_or_else(|| CapabilityError::InvalidConfiguration {
                field: "decoding phase domain",
                detail: "phase belongs to an undeclared source pool".into(),
            })?;
        let pool = request
            .domains
            .iter()
            .find(|pool| pool.domain == extra.domain)
            .ok_or_else(|| CapabilityError::InvalidConfiguration {
                field: "decoding phase domain",
                detail: "phase has no request pool".into(),
            })?;
        domain.phases.extend(
            extra
                .phases
                .into_iter()
                .filter(|p| p.phase != MemoryPhase::ContinuationStart),
        );
        domain.generation_peak = domain.generation_peak.maximum(&extra.generation_peak);
        domain.overall_peak = domain
            .overall_peak
            .maximum(&extra.overall_peak)
            .maximum(&domain.generation_peak);
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
    for assumption in candidate.assumptions {
        if !result.assumptions.contains(&assumption) {
            result.assumptions.push(assumption);
        }
    }
    for uncertainty in candidate.uncertainties {
        if !result.uncertainties.contains(&uncertainty) {
            result.uncertainties.push(uncertainty);
        }
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
    Ok(())
}
