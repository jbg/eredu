//! Phase accounting for one speculative lane. This never allocates model state.
use super::*;
mod embedded;
mod shared_backbone;
mod conditioned;
pub use conditioned::{ConditionedAssistantMemoryPlan, ConditionedAssistantMemoryProfile,
    ConditionedBlockConvolution, ConditionedPredictionBody, ConditionedPredictionScoring,
    ConditionedPredictionTopology};
pub use shared_backbone::SharedBackboneMemoryPlan;
pub use embedded::{apply_embedded_parameter_conversion_credit, EmbeddedPredictionMemoryPlan};
use eredu_core::{generation::SpeculativeSchedulerOptions, ObservationKind, SpeculativeDraft};

/// Native/architecture facts about the selected speculative mechanism.
pub struct SpeculativeMemoryProfile {
    /// Separately parameterized target-conditioned context and fixed-block execution.
    pub conditioned: Option<ConditionedAssistantMemoryProfile>,
    /// Complete retained ordinary refinement and native numerical workspace.
    pub refinement: Option<RefinementMemoryPlan>,
    /// Complete masked invocation geometry over the target's parameter catalog.
    pub shared_backbone: Option<SharedBackboneMemoryPlan>,
    /// Exact selected embedded invocation, state and retained-feature contracts.
    pub embedded: Option<crate::prediction_resources::EmbeddedPredictionTopology>,
    /// Exact existing conversion allocations and their authoritative retaining bindings.
    /// Already included in target parameter residency; used only to credit future conversion work.
    pub parameter_conversions: Option<Vec<crate::ResidentParameterConversion>>,
    /// Separately owned drafter geometry and resident parameters. Embedded heads
    /// are already included in target residency and must not be charged again.
    pub draft: Option<LoadedMemoryProfile>,
    /// Additional prediction state/features per context position. Unknown
    /// components stay explicit rather than poisoning unrelated contributions.
    pub auxiliary_bytes_per_position: MemoryBytes,
    /// Retained distribution plus processing scratch per vocabulary entry and
    /// proposal row, supplied by the selected native sampling implementation.
    pub sampling_bytes_per_vocabulary_entry: MemoryBytes,
    /// Maximum width admitted by the retained selection.
    pub proposal_capacity: u64,
    /// True when cache/driver overhead in a shared physical pool has one owner.
    pub shared_allocator: bool,
}

/// Backend projection borrowed from the exact target and selected draft. A report
/// must not load artifacts, allocate caches, submit work or consume the drafter.
pub trait SpeculativeForecastBackend<D>: GenerationForecastBackend {
    /// Target selection for an isolated speculative lane. Backends with separate
    /// lane caches override this to avoid treating unrelated installed ordinary
    /// state as this lane's state. This never resets or creates a cache.
    fn speculative_target_memory_profile(
        runtime: &ModelRuntime<Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError> {
        Self::loaded_memory_profile(runtime)
    }

    /// Retains unknown coverage for backends without speculative resource facts.
    fn speculative_memory_profile(
        _runtime: &ModelRuntime<Self>,
        _draft: &SpeculativeDraft<'_, D>,
    ) -> Result<Option<SpeculativeMemoryProfile>, GenerationForecastError> {
        Ok(None)
    }
}

/// Recomputable single-lane speculative costs. Scheduler limits are ceilings,
/// never predictions of acceptance rate or adaptive lookahead behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeMemoryPlan {
    /// Conditioned invocation over the paired parameter-only draft request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditioned: Option<ConditionedAssistantMemoryPlan>,
    /// Fixed masked canvas and optional additive draft work; no second parameter owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_backbone: Option<SharedBackboneMemoryPlan>,
    /// Embedded invocation costs; target residency already includes its parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded: Option<EmbeddedPredictionMemoryPlan>,
    /// Ordinary geometry for a separately resident draft; absent for embedded heads.
    pub draft: Option<GenerationMemoryRequest>,
    /// Extra selected prediction state/features per context position.
    pub auxiliary_bytes_per_position: MemoryBytes,
    /// Native distribution/sampling envelope, including normalization scratch.
    pub sampling_bytes_per_vocabulary_entry: MemoryBytes,
    /// Resolved positive maximum proposal width.
    pub max_draft_tokens: u64,
    /// Configured scheduler resource ceilings (one lane is forecast here).
    pub scheduler: SpeculativeSchedulerOptions,
    /// Whether a shared pool has one cache/driver allowance.
    pub shared_allocator: bool,
}

/// Projects a retained source and ordinary refinement over the same target.
/// Parameters are shared between mutually exclusive numerical phases. Separate
/// source state remains resident; its full transaction allowance conservatively
/// bounds retained state and scratch when exact idle-source storage is unavailable.
pub fn estimate_source_generation_memory(
    request: &GenerationMemoryRequest,
    source: &SpeculativeMemoryPlan,
    refinement: Option<&RefinementMemoryPlan>,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let mut estimate = estimate_speculative_memory(request, source)?;
    let Some(refinement) = refinement else { return Ok(estimate) };
    let mut refined = estimate_refinement_memory(request, refinement)?;
    if source.draft.is_some() || source.embedded.is_some() {
        let allowance = |bytes: &MemoryBytes| match bytes.upper_bytes {
            Some(upper) => MemoryBytes::estimated(0, upper, "retained source transaction storage allowance; shared target state may alias"),
            None => MemoryBytes::unknown("retained source transaction storage allowance is unavailable"),
        };
        for domain in &mut refined.domains {
            let Some(owner) = estimate.domains.iter().find(|owner| owner.domain == domain.domain) else { continue };
            let phases = owner.phases.iter().filter(|phase| phase.phase != MemoryPhase::Loading).collect::<Vec<_>>();
            let maximum = |value: fn(&PhaseMemoryEstimate) -> &MemoryBytes| phases.iter()
                .fold(MemoryBytes::exact(0), |bytes, phase| bytes.maximum(value(phase)));
            let parameters = maximum(|phase| &phase.parameters);
            let state = allowance(&maximum(|phase| &phase.persistent_state));
            let workspace = allowance(&maximum(|phase| &phase.workspace));
            let staging = allowance(&maximum(|phase| &phase.staging));
            for phase in &mut domain.phases {
                if matches!(phase.phase, MemoryPhase::Refinement | MemoryPhase::RefinementCommit) {
                    phase.parameters = phase.parameters.maximum(&parameters);
                    phase.persistent_state = phase.persistent_state.add(&state)?;
                    phase.workspace = phase.workspace.add(&workspace)?;
                    phase.staging = phase.staging.add(&staging)?;
                    phase.total = phase.parameters.add(&phase.persistent_state)?
                        .add(&phase.retained_input)?.add(&phase.workspace)?
                        .add(&phase.staging)?.add(&phase.backend_overhead)?;
                }
            }
            domain.generation_peak = domain.phases.iter().filter(|phase| phase.phase != MemoryPhase::Loading)
                .fold(MemoryBytes::exact(0), |bytes, phase| bytes.maximum(&phase.total));
            domain.overall_peak = domain.overall_peak.maximum(&domain.generation_peak);
        }
        refined.assumptions.push("Ordinary refinement retains the selected source's parameters once and conservatively includes its full transaction state/scratch upper allowance; this is not measured distinct allocation.".into());
    }
    super::merge_decoding_phase_estimate(&mut estimate, refined, request)?;
    Ok(estimate)
}

/// Installed embedded state with native capacity and retained capture observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddedContinuationMemoryPlan {
    /// Native layer frontiers in the ordinary prediction-state order.
    pub layer_positions: Vec<u64>,
    /// Native envelope horizon, including configured speculative overshoot.
    pub additional_input_tokens: u64,
    /// Current logical payload floor through the native capacity allowance.
    pub current_state: MemoryBytes,
    /// Native capacity envelope through the requested horizon.
    pub peak_state: MemoryBytes,
    /// Captured target features, including potentially pinned prefix backing.
    pub retained_features: MemoryBytes,
}

/// Settled, horizon-specific observations for a speculative lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeContinuationMemoryPlan {
    /// Requested additional committed decisions; this does not change the run limit.
    pub additional_tokens: u64,
    /// Already committed input awaiting append for an admitted mode change.
    /// This work can occur without any additional output decision.
    #[serde(default)]
    pub settlement_input_tokens: u64,
    /// Actual target state, including speculative overshoot in its peak horizon.
    pub target: ContinuationMemoryPlan,
    /// Actual draft state, which need not have the same installed frontier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft: Option<ContinuationMemoryPlan>,
    /// Installed prediction owner for embedded speculation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded: Option<EmbeddedContinuationMemoryPlan>,
    /// Current retained assistant seed; future copies use the draft-state envelope.
    pub seed: MemoryBytes,
    /// Current and future sampler, random, history and semantic storage allowance.
    /// Includes unattributed native RNG/capture storage, so charged to every pool.
    pub host_retention: MemoryBytes,
    /// Live user snapshot and inactive-branch reservations; not resident credit.
    pub retained_snapshots: MemoryBytes,
}

/// Reproducible settled speculative outlook. Reobserve after advancing/restoring
/// or changing the horizon; native capacity cannot be extrapolated from this record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeContinuationForecast {
    /// Active and admitted pending policy at this read-only observation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoding: Option<eredu_core::decoding::DecodingState>,
    /// Total/additional peaks without completed loading or prefill.
    pub estimate: GenerationMemoryEstimate,
    /// Target physical pools and calibrated geometry.
    pub request: GenerationMemoryRequest,
    /// Separate draft request and selected transaction ceilings.
    pub speculative: SpeculativeMemoryPlan,
    /// Native state and controlled-retention observations for this horizon.
    pub continuation: SpeculativeContinuationMemoryPlan,
}

/// Projects settled speculative continuation from actual installed state.
pub fn estimate_speculative_continuation_memory(
    target: &GenerationMemoryRequest,
    plan: &SpeculativeMemoryPlan,
    continuation: &SpeculativeContinuationMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    if plan.conditioned.is_some() && continuation.draft.as_ref().is_none_or(|state|
        state.current_positions != continuation.target.current_positions
    ) {
        return Err(CapabilityError::InvalidConfiguration { field: "conditioned assistant continuation", detail: "installed projected context and pending capture observations must cover the target frontier".into() });
    }
    let mut continuation = continuation.clone();
    continuation.target = continuation.target.with_logical_state_bounds(target)?;
    let extra = speculative_continuation_positions(
        continuation.additional_tokens,
        plan.max_draft_tokens,
        plan.scheduler,
    )?.max(continuation.settlement_input_tokens);
    if continuation.target.additional_input_tokens < extra {
        return Err(CapabilityError::InvalidConfiguration { field: "speculative continuation horizon", detail: "native state observations must include the requested horizon and configured speculative overshoot".into() });
    }
    match (&plan.draft, &plan.embedded, &mut continuation.draft, &mut continuation.embedded) {
        (Some(draft), None, Some(state), None) if state.additional_input_tokens == extra => {
            if plan.conditioned.is_some() {
                state.current_state.validate()?;
                state.peak_state.validate()?;
                if state.query_tokens == 0 {
                    return Err(CapabilityError::InvalidConfiguration {
                        field: "conditioned assistant continuation",
                        detail: "conditioned invocation query geometry is empty".into(),
                    });
                }
            } else {
                *state = state.with_logical_state_bounds(draft)?;
            }
        }
        (None, Some(prediction), None, Some(state)) if state.additional_input_tokens == extra => {
            prediction.validate_continuation(target, state)?;
        }
        (None, None, None, None) if plan.shared_backbone.is_some() || plan.max_draft_tokens == 0 => {}
        _ => return Err(CapabilityError::InvalidConfiguration {
            field: "speculative continuation", detail: "requires matching settled observations and horizon; nonzero proposals require one selected draft mechanism".into(),
        }),
    }
    estimate_speculative_inner(target, plan, Some(&continuation))
}

/// Additional cache positions required by a what-if horizon and selected lookahead.
pub fn speculative_continuation_positions(
    tokens: u64,
    width: u64,
    scheduler: SpeculativeSchedulerOptions,
) -> Result<u64, CapabilityError> {
    if tokens == 0 {
        return Ok(0);
    }
    add(
        tokens,
        mul(
            add(width, 1)?,
            1 + u64::from(
                width > 0 && scheduler.lookahead_blocks != 0 && scheduler.max_optimistic_branches != 0,
            ),
        )?,
    )
}

fn add(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "speculative memory sum",
    })
}
fn mul(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_mul(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "speculative memory product",
    })
}
fn scale(bytes: &MemoryBytes, copies: u64) -> Result<MemoryBytes, CapabilityError> {
    bytes.validate()?;
    Ok(MemoryBytes {
        lower_bytes: mul(bytes.lower_bytes, copies)?,
        upper_bytes: bytes.upper_bytes.map(|n| mul(n, copies)).transpose()?,
        kind: bytes.kind,
        detail: format!("{copies} copies/positions of {}", bytes.detail),
    })
}
fn interval(
    bytes: &MemoryBytes,
    copies: u64,
    detail: &str,
) -> Result<MemoryBytes, CapabilityError> {
    let mut bound = scale(bytes, copies)?;
    bound.lower_bytes = 0;
    bound.kind = eredu_core::ObservationKind::Estimated;
    bound.detail = detail.into();
    Ok(bound)
}
fn total(p: &PhaseMemoryEstimate) -> Result<MemoryBytes, CapabilityError> {
    p.parameters
        .add(&p.persistent_state)?
        .add(&p.retained_input)?
        .add(&p.workspace)?
        .add(&p.staging)?
        .add(&p.backend_overhead)
}

/// Projects prefill, proposal, verification and replay peaks. Parameter backing
/// stays resident across phases; state copies and retained distributions overlap
/// only within each phase. This is a conservative logical allocation envelope,
/// with the ordinary estimator's explicit kernel/allocator calibration.
pub fn estimate_speculative_memory(
    target: &GenerationMemoryRequest,
    plan: &SpeculativeMemoryPlan,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    estimate_speculative_inner(target, plan, None)
}

fn estimate_speculative_inner(
    target: &GenerationMemoryRequest,
    plan: &SpeculativeMemoryPlan,
    continuation: Option<&SpeculativeContinuationMemoryPlan>,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    // These are transaction contracts, not caller-tunable savings: both
    // prefills run in full and verification/replay can project every row.
    let normalize = |r: &mut GenerationMemoryRequest| {
        r.prefill_chunk_tokens = r.input.model_positions.max(1);
        for pool in &mut r.domains {
            for execution in &mut pool.executions {
                execution.logits = LogitsWorkspace::EveryPosition;
            }
        }
    };
    let mut target = target.clone();
    normalize(&mut target);
    let target = &target;
    let mut normalized = plan.clone();
    if let Some(draft) = &mut normalized.draft {
        normalize(draft);
    }
    let plan = &normalized;
    if let Some(conditioned) = &plan.conditioned {
        conditioned.validate(plan.draft.as_ref(), plan.max_draft_tokens)?;
        if plan.embedded.is_some() || plan.shared_backbone.is_some() {
            return Err(CapabilityError::InvalidConfiguration {
                field: "conditioned assistant memory", detail: "conditioned execution cannot share another prediction mechanism".into(),
            });
        }
    }
    if let Some(shared) = &plan.shared_backbone { shared.validate(plan.max_draft_tokens)?; }
    if [plan.embedded.is_some(), plan.draft.is_some(), plan.shared_backbone.is_some()].into_iter().filter(|selected| *selected).count() > 1 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "speculative forecast",
            detail: "exactly one draft mechanism owns speculative invocation geometry".into(),
        });
    }
    plan.auxiliary_bytes_per_position.validate()?;
    plan.sampling_bytes_per_vocabulary_entry.validate()?;
    let ordinary = |request: &GenerationMemoryRequest| {
        let mut validation = request.clone();
        if continuation.is_some() {
            validation.input = eredu_core::InputTokenCount::text(1);
            validation.max_output_tokens = Some(0);
            validation.prefill_chunk_tokens = 1;
            for pool in &mut validation.domains {
                pool.loading_peak = MemoryBytes::exact(0);
            }
        }
        estimate_generation_memory(&validation)
    };
    let mut result = ordinary(target)?;
    if target.batch_size != 1 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "speculative forecast",
            detail: "requires one lane".into(),
        });
    }
    plan.scheduler
        .validate()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let draft_estimate = plan.draft.as_ref().map(ordinary).transpose()?;
    let output = continuation.map_or_else(
        || {
            target
                .max_output_tokens
                .unwrap_or(target.forecast_output_tokens)
        },
        |c| c.additional_tokens,
    );
    let canonical = add(target.input.model_positions, output)?;
    let settlement_only = output == 0 && continuation.is_some_and(|c| c.settlement_input_tokens != 0);
    let width = if settlement_only { continuation.unwrap().settlement_input_tokens } else { add(plan.max_draft_tokens, 1)? };
    // A verified block and one optimistic block may extend beyond the canonical
    // frontier. Use configured maxima even if the run ends or disables lookahead.
    let lookahead = u64::from(
        !settlement_only && plan.max_draft_tokens > 0 && plan.scheduler.lookahead_blocks != 0 && plan.scheduler.max_optimistic_branches != 0,
    );
    let frontier = add(canonical, mul(width, add(1, lookahead)?)?)?;
    let mut pools = target.domains.clone();
    if let Some(draft) = &plan.draft {
        if (continuation.is_none() && draft.input != target.input)
            || draft.max_output_tokens != target.max_output_tokens
            || draft.forecast_output_tokens != target.forecast_output_tokens
            || draft.batch_size != 1
        {
            return Err(CapabilityError::InvalidConfiguration {
                field: "speculative draft",
                detail: "target and draft must describe the same single-lane request".into(),
            });
        }
        for pool in &draft.domains {
            if !pools.iter().any(|p| p.domain == pool.domain) {
                pools.push(pool.clone());
            }
        }
        if pools.len() > 1 && pools.iter().any(|p| p.domain == MemoryDomain::Unified) {
            return Err(CapabilityError::InvalidConfiguration {
                field: "speculative pools",
                detail: "shared backing must use one unified domain for both models".into(),
            });
        }
    }
    if !pools
        .iter()
        .any(|p| matches!(p.domain, MemoryDomain::Unified | MemoryDomain::Host))
    {
        return Err(CapabilityError::InvalidConfiguration { field: "speculative pools", detail: "include a host pool for token histories and transaction records on discrete devices".into() });
    }
    let target_domains = std::mem::take(&mut result.domains);
    for pool in &pools {
        let t = target.domains.iter().find(|p| p.domain == pool.domain);
        let d = plan.draft.as_ref().and_then(|r| {
            r.domains
                .iter()
                .find(|p| p.domain == pool.domain)
                .map(|p| (r, p))
        });
        let shared_conversions = plan
            .draft
            .as_ref()
            .map(|draft| super::retention::shared_conversion_payload(target, draft, &pool.domain))
            .transpose()?
            .unwrap_or(0);
        let resident = add(
            t.map_or(0, |p| p.already_resident_bytes),
            d.map_or(0, |(_, p)| p.already_resident_bytes),
        )?
        .checked_sub(shared_conversions)
        .ok_or_else(|| {
            CapabilityError::Observation(
                "shared conversion credit exceeds resident baseline".into(),
            )
        })?;
        let budget = match (t, d) {
            (Some(t), Some((_, d))) => MemoryBudget {
                application_limit_bytes: stricter(
                    t.budget.application_limit_bytes,
                    d.budget.application_limit_bytes,
                ),
                available_bytes: stricter(t.budget.available_bytes, d.budget.available_bytes),
                reserve_bytes: t.budget.reserve_bytes.max(d.budget.reserve_bytes),
            },
            _ => pool.budget.clone(),
        };
        let mut phases = Vec::new();
        let stages = if continuation.is_some() {
            let mut stages = vec![MemoryPhase::ContinuationStart];
            if output != 0 {
                stages.extend([
                    MemoryPhase::SpeculativeDraft,
                    MemoryPhase::SpeculativeVerification,
                    MemoryPhase::SpeculativeCommit,
                ]);
            } else if settlement_only {
                stages.extend([MemoryPhase::SpeculativeVerification, MemoryPhase::SpeculativeCommit]);
            }
            stages
        } else if output == 0 {
            vec![MemoryPhase::Prefill]
        } else {
            vec![
                MemoryPhase::Prefill,
                MemoryPhase::SpeculativeDraft,
                MemoryPhase::SpeculativeVerification,
                MemoryPhase::SpeculativeCommit,
            ]
        };
        for stage in stages.into_iter().filter(|stage| plan.max_draft_tokens > 0 || *stage != MemoryPhase::SpeculativeDraft) {
            let prefill = stage == MemoryPhase::Prefill;
            let start = stage == MemoryPhase::ContinuationStart;
            let positions = if prefill {
                target.input.model_positions
            } else {
                frontier
            };
            let shared_draft = !prefill && !start && stage == MemoryPhase::SpeculativeDraft && plan.shared_backbone.is_some();
            let query = if prefill { positions } else if shared_draft { plan.shared_backbone.as_ref().unwrap().canvas_positions } else { width };
            let positions = if shared_draft { positions.max(add(canonical, query)?) } else { positions };
            let get = |r: &GenerationMemoryRequest,
                       p: &DomainMemoryPlan,
                       native: Option<&ContinuationMemoryPlan>|
             -> Result<PhaseMemoryEstimate, CapabilityError> {
                let mut positions = positions;
                let mut pool = p.clone();
                if let Some(native) = native {
                    positions = if start {
                        native.current_positions
                    } else {
                        add(native.current_positions, native.additional_input_tokens)?
                    };
                    if shared_draft { positions = add(positions, query)?; }
                    for e in &mut pool.executions {
                        e.cache_update = CacheUpdateWorkspace::InPlace;
                    }
                }
                let mut result = phase(&pool, r, stage, positions, if start { 0 } else { query })?;
                if shared_draft && native.is_none() {
                    // Bidirectional block K/V is workspace; it does not extend the causal frontier.
                    result.persistent_state = phase(&pool, r, stage, canonical, 1)?.persistent_state;
                }
                if let Some(native) = native {
                    if !pool.executions.is_empty() {
                        result.persistent_state = if start {
                            native.current_state.clone()
                        } else {
                            native.peak_state.clone()
                        };
                        if !start {
                            let copy = match p.executions[0].cache_update {
                                CacheUpdateWorkspace::InPlace => MemoryBytes::exact(0),
                                CacheUpdateWorkspace::CopyState => {
                                    interval(&native.peak_state, 1, "native cache-update overlap")?
                                }
                                CacheUpdateWorkspace::Unknown => {
                                    MemoryBytes::unknown("native cache update overlap unavailable")
                                }
                            };
                            result.workspace = result.workspace.add(&copy)?;
                        }
                    }
                    if start {
                        result.workspace = MemoryBytes::exact(0);
                    }
                }
                Ok(result)
            };
            let tp = t
                .map(|p| get(target, p, continuation.map(|c| &c.target)))
                .transpose()?;
            let dp = d
                .map(|(r, p)| get(r, p, continuation.and_then(|c| c.draft.as_ref())))
                .transpose()?;
            let mut p = tp
                .clone()
                .or_else(|| dp.clone())
                .expect("pool has an owner");
            if let (Some(t), Some(d)) = (&tp, &dp) {
                p.parameters = t.parameters.add(&d.parameters)?;
                p.parameters.lower_bytes = p
                    .parameters
                    .lower_bytes
                    .checked_sub(shared_conversions)
                    .ok_or_else(|| {
                        CapabilityError::Observation(
                            "shared conversion credit exceeds parameter baseline".into(),
                        )
                    })?;
                p.parameters.upper_bytes = p.parameters.upper_bytes.map(|n| n - shared_conversions);
                p.persistent_state = t.persistent_state.add(&d.persistent_state)?;
                p.retained_input = t.retained_input.add(&d.retained_input)?;
                p.retained_input.lower_bytes = t
                    .retained_input
                    .lower_bytes
                    .max(d.retained_input.lower_bytes);
                p.retained_input.kind = ObservationKind::Estimated;
                // Prefills and commit replays are serial, but their graphs and
                // outputs can survive until settlement. Keep both workspace
                // envelopes live; optimistic verification genuinely overlaps.
                p.workspace = t.workspace.add(&d.workspace)?;
                p.workspace.lower_bytes = t.workspace.lower_bytes.max(d.workspace.lower_bytes);
                p.workspace.kind = ObservationKind::Estimated;
                p.staging = t.staging.add(&d.staging)?;
                p.backend_overhead = if plan.shared_allocator {
                    t.backend_overhead.maximum(&d.backend_overhead)
                } else {
                    t.backend_overhead.add(&d.backend_overhead)?
                };
            }
            if !start && matches!(pool.domain, MemoryDomain::Host | MemoryDomain::Unified) {
                p.staging = p.staging.add(&MemoryBytes::estimated(0,
                    mul(mul(positions, 4)?, add(6, mul(3,lookahead)?)?)?,
                    "single-lane token/history copies through rollback, replay and optimistic advancement"))?;
            }
            let conditioned_pool = plan.conditioned.as_ref()
                .filter(|assistant| assistant.execution_domain == pool.domain);
            let execution_pool = t.is_some_and(|p| !p.executions.is_empty())
                || d.is_some_and(|(_, p)| !p.executions.is_empty()) || conditioned_pool.is_some();
            if let Some(assistant) = conditioned_pool {
                if assistant.native_workspace.upper_bytes.is_none() {
                    result.uncertainties.push(assistant.native_workspace.detail.clone());
                }
                let vocabulary = t.into_iter().flat_map(|p| &p.executions)
                    .filter_map(|e| e.execution_topology.as_ref().map(|t| t.vocabulary_size)
                        .or_else(|| e.workspace.as_ref().map(|w| w.vocabulary_size))).max();
                let installed = continuation.and_then(|c| c.draft.as_ref());
                let context_positions = if start {
                    installed.map_or(positions, |state| state.current_positions)
                } else { positions };
                let (logical, workspace, features) = assistant.costs(context_positions, if prefill { positions } else { width }, vocabulary)?;
                let state = match installed {
                    Some(state) if start => state.current_state.maximum(&logical),
                    Some(state) => state.peak_state.add(&logical)?,
                    None => logical,
                };
                p.persistent_state = p.persistent_state.add(&state)?;
                p.retained_input = p.retained_input.add(&features)?;
                if !start {
                    p.workspace = p.workspace.add(&interval(&workspace, add(1, lookahead)?,
                        "conditioned full-block invocation and optimistic graph overlap")?)?;
                    let copies = if prefill { 2 } else { add(5, mul(3, lookahead)?)? };
                    p.staging = p.staging.add(&interval(&state, copies,
                        "conditioned seed, context replacement, rollback and optimistic context copies")?)?
                        .add(&interval(&features, copies, "conditioned target capture views/copies through commitment and rollback")?)?;
                }
            }
            // Pending context belongs to the target state. A separate assistant
            // pool's transfer envelope cannot pay for the source backing or its
            // rollback copies. Shared pools already include these charges above.
            if let Some(assistant) = plan.conditioned.as_ref().filter(|_| {
                conditioned_pool.is_none() && t.is_some_and(|p| !p.executions.is_empty())
            }) {
                let capture_positions = if start {
                    continuation.map_or(positions, |c| c.target.current_positions)
                } else { positions };
                let features = assistant.captured_features(capture_positions)?;
                p.retained_input = p.retained_input.add(&features)?;
                if !start {
                    let copies = if prefill { 2 } else { add(5, mul(3, lookahead)?)? };
                    p.staging = p.staging.add(&interval(&features, copies,
                        "target-owned conditioned capture backing through commitment and rollback")?)?;
                }
            }
            if execution_pool {
                if let Some(embedded) = &plan.embedded {
                    let native = continuation.and_then(|c| c.embedded.as_ref());
                    let positions = if start {
                        continuation.unwrap().target.current_positions
                    } else {
                        positions
                    };
                    let selected_state = native.map(|n| {
                        if start {
                            &n.current_state
                        } else {
                            &n.peak_state
                        }
                    });
                    let (state, workspace, mut features) = embedded.costs_with_state(
                        target,
                        positions,
                        if start { 0 } else { query },
                        prefill,
                        selected_state,
                    )?;
                    if let Some(native) = native {
                        features = if start {
                            native.retained_features.clone()
                        } else {
                            features.add(&native.retained_features)?
                        };
                    }
                    p.persistent_state = p.persistent_state.add(&state)?;
                    if !start {
                        p.workspace = p.workspace.add(&interval(
                            &workspace,
                            if prefill { 1 } else { add(1, lookahead)? },
                            &format!(
                                "embedded prediction invocation and optimistic graph overlap: {}",
                                workspace.detail
                            ),
                        )?)?;
                    }
                    p.retained_input = p.retained_input.add(&features)?;
                    if !start {
                        p.staging = p.staging.add(&interval(&state,
                        if prefill { 2 } else { add(5, mul(3, lookahead)?)? },
                        "embedded seed, proposal restore/replacement, rollback/replay and optimistic prediction state copies",
                    )?)?;
                        p.staging = p.staging.add(&interval(&features,
                        if prefill { 2 } else { add(5, mul(3, lookahead)?)? },
                        "retained target feature views/copies through prediction, verification, rollback and lookahead",
                    )?)?;
                    }
                    if !start && workspace.upper_bytes.is_none() {
                        result.uncertainties.push(workspace.detail);
                    }
                    if features.upper_bytes.is_none() {
                        result.uncertainties.push(features.detail);
                    }
                }
            }
            if execution_pool && shared_draft {
                let shared = plan.shared_backbone.as_ref().unwrap();
                p.workspace = p.workspace.add(&shared.adapter_workspace(add(1, lookahead)?)?)?;
            }
            if execution_pool && !start {
                // The ordinary workspace already includes cache-update overlap.
                // These *additional* copies envelope durable checkpoints, seed,
                // proposal restore/replacement, rollback and optimistic branches.
                let target_copies = if prefill {
                    1
                } else if plan.max_draft_tokens == 0 && plan.draft.is_none()
                    && plan.embedded.is_none() && plan.shared_backbone.is_none() {
                    // A causal-only append retains its isolated lane and rollback
                    // checkpoint, without an assistant seed checkpoint.
                    2
                } else {
                    add(3, mul(2, lookahead)?)?
                };
                let draft_copies = if prefill {
                    2
                } else {
                    add(5, mul(3, lookahead)?)?
                };
                if let Some(t) = &tp {
                    p.staging = p.staging.add(&interval(
                        &t.persistent_state,
                        target_copies,
                        "target rollback/replay and optimistic state-copy envelope",
                    )?)?;
                }
                if let Some(d) = &dp {
                    p.staging = p.staging.add(&interval(&d.persistent_state,draft_copies,"draft seed, proposal restore/replacement and optimistic state-copy envelope")?)?;
                }
                p.staging = p.staging.add(&interval(
                    &plan.auxiliary_bytes_per_position,
                    mul(positions, add(6, mul(3, lookahead)?)?)?,
                    "selected prediction state/features with transaction-copy overlap",
                )?)?;
                // Dense processed distributions can be retained for every proposal,
                // verification row, residual sample and optimistic proposal block.
                let mut vocab = Some(0u64);
                for owner in [t, d.map(|(_, p)| p)].into_iter().flatten() {
                    for e in &owner.executions {
                        vocab = vocab
                            .zip(
                                e.execution_topology
                                    .as_ref()
                                    .map(|topology| topology.vocabulary_size)
                                    .or_else(|| e.workspace.as_ref().map(|w| w.vocabulary_size)),
                            )
                            .map(|(a, b)| a.max(b));
                    }
                }
                let distributions = if settlement_only { MemoryBytes::exact(0) } else { match vocab {
                    Some(v) => interval(&plan.sampling_bytes_per_vocabulary_entry,
                        mul(v, if let Some(shared) = plan.shared_backbone.as_ref().filter(|_| plan.max_draft_tokens > 0) {
                            add(mul(shared.canvas_positions, add(1, lookahead)?)?, mul(width, add(2,lookahead)?)?)?
                        } else { mul(width, add(2,lookahead)?)? })?,
                        &format!("proposal, verification and residual distributions including optimistic overlap: {}", plan.sampling_bytes_per_vocabulary_entry.detail))?,
                    None => MemoryBytes::unknown("speculative sampling vocabulary geometry unavailable"),
                }};
                p.staging = p.staging.add(&distributions)?;
            }
            // The frontier includes uncommitted positions and worst-case replay.
            // Do not promote that upper envelope to an inevitable allocation.
            if !prefill && !start {
                p.persistent_state.lower_bytes = 0;
                p.persistent_state.kind = ObservationKind::Estimated;
                p.workspace.lower_bytes = 0;
                p.workspace.kind = ObservationKind::Estimated;
            }
            if let Some(c) = continuation {
                if execution_pool {
                    p.staging = p.staging.add(&c.seed)?;
                }
                p.staging = p.staging.add(&c.retained_snapshots)?;
                // Sampler/RNG and instrumented retention may own native arrays;
                // without exact placement, conservatively charge every pool.
                p.retained_input = p.retained_input.add(&c.host_retention)?;
            }
            p.total = total(&p)?;
            phases.push(p);
        }
        let peak = phases
            .iter()
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        let mut overall = peak.clone();
        if continuation.is_none() {
            let loading = t
                .map(|p| p.loading_peak.clone())
                .unwrap_or(MemoryBytes::exact(0))
                .add(
                    &d.map(|(_, p)| p.loading_peak.clone())
                        .unwrap_or(MemoryBytes::exact(0)),
                )?
                .add(&phases[0].backend_overhead)?;
            overall = peak.maximum(&loading);
            phases.push(PhaseMemoryEstimate {
                phase: MemoryPhase::Loading,
                positions: 0,
                query_positions: 0,
                parameters: MemoryBytes::exact(0),
                persistent_state: MemoryBytes::exact(0),
                retained_input: MemoryBytes::exact(0),
                workspace: MemoryBytes::exact(0),
                staging: t
                    .map(|p| p.loading_peak.clone())
                    .unwrap_or(MemoryBytes::exact(0))
                    .add(
                        &d.map(|(_, p)| p.loading_peak.clone())
                            .unwrap_or(MemoryBytes::exact(0)),
                    )?,
                backend_overhead: phases[0].backend_overhead.clone(),
                total: loading,
            });
        }
        let growth_in = |domains: &[DomainMemoryEstimate]| {
            domains
                .iter()
                .find(|d| d.domain == pool.domain)
                .map_or(0, |d| d.state_growth_bytes_per_position)
        };
        let embedded_growth = if t.is_some_and(|p| !p.executions.is_empty()) {
            plan.embedded
                .as_ref()
                .map(|p| p.state_growth(target))
                .transpose()?
                .unwrap_or(0)
        } else {
            0
        };
        // Conditioned context compacts at its declared window. Its native
        // floating width may be narrower, so retain a labeled FP32 growth envelope.
        let conditioned_growth = match plan.conditioned.as_ref()
            .filter(|p| p.execution_domain == pool.domain && target.input.model_positions < p.topology.body.context_window) {
            Some(p) => mul(4, add(p.topology.body.hidden,
                mul(mul(2, p.topology.body.layers)?, p.topology.body.key_value_width)?)?)?,
            None => 0,
        };
        let growth = add(add(growth_in(&target_domains), embedded_growth)?, conditioned_growth)?
            .checked_add(draft_estimate.as_ref().map_or(0, |r| growth_in(&r.domains)))
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "speculative state growth",
            })?;
        result.domains.push(DomainMemoryEstimate {
            domain: pool.domain.clone(),
            phases,
            additional_generation_peak: peak.additional(resident),
            generation_fit: fit(&peak, resident, &budget)?,
            fit: fit(&overall, resident, &budget)?,
            generation_peak: peak,
            overall_peak: overall,
            state_growth_bytes_per_position: growth,
        });
    }
    if let Some(draft) = draft_estimate {
        result.uncertainties.extend(
            draft
                .uncertainties
                .into_iter()
                .map(|s| format!("draft: {s}")),
        );
    }
    if plan.auxiliary_bytes_per_position.upper_bytes != Some(0) {
        result
            .uncertainties
            .push(plan.auxiliary_bytes_per_position.detail.clone());
    }
    result
        .uncertainties
        .push(plan.sampling_bytes_per_vocabulary_entry.detail.clone());
    result.assumptions.push("Speculative single-lane envelope: full-pass prefill; target and draft residency, rollback, proposal copies, verification and replay; configured lookahead is retained without assuming acceptance or adaptive disabling. User snapshots/branches are additional allocations.".into());
    if let Some(embedded) = &plan.embedded {
        result.assumptions.push(format!("Embedded {:?} execution: prediction parameters remain in target residency; ordinary prediction components, retained target features, invocation workspace, state-copy and lookahead envelopes are composed without charging shared target weights twice.", embedded.mode));
        result.uncertainties.extend(
            embedded
                .missing
                .iter()
                .map(|reason| format!("embedded prediction: {reason}")),
        );
    }
    if plan.conditioned.is_some() {
        result.assumptions.push("Target-conditioned assistant: separate immutable parameter owner, compact encoded context/KV, full trained proposal block, raw/scattered scores and captured full-prefix backing; logical tensors and state growth use a conservative FP32 envelope. Capture backing and rollback allowances cover target execution pools and the assistant transfer pool, once when shared. Native scratch and bounded loading coverage remain separately labeled; this is not a total-process bound.".into());
    }
    result.uncertainties.sort();
    result.uncertainties.dedup();
    if let Some(c) = continuation {
        result.requested_positions = add(c.target.current_positions, c.additional_tokens)?;
        result.is_forecast = true;
        result
            .assumptions
            .retain(|s| !s.starts_with("Speculative single-lane envelope:"));
        result.assumptions.push("Settled speculative continuation: actual target/draft or prediction frontiers and native capacity; no completed loading/prefill; configured proposal and lookahead ceilings, with no assumption of acceptance. Snapshot/branch and state bounds are allowances, never resident credit. Horizon does not change generation limits.".into());
        result.uncertainties.extend([
            c.target.current_state.detail.clone(),
            c.target.peak_state.detail.clone(),
            c.host_retention.detail.clone(),
        ]);
        if let Some(draft) = &c.draft {
            result.uncertainties.extend([
                draft.current_state.detail.clone(),
                draft.peak_state.detail.clone(),
            ]);
        }
        if let Some(embedded) = &c.embedded {
            result.uncertainties.extend([
                embedded.current_state.detail.clone(),
                embedded.peak_state.detail.clone(),
                embedded.retained_features.detail.clone(),
            ]);
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
    Ok(result)
}
fn stricter(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
#[cfg(test)]
mod tests;
