//! Phase accounting for one speculative lane. This never allocates model state.
use super::*;
use eredu_core::{generation::SpeculativeSchedulerOptions, ObservationKind, SpeculativeDraft};

/// Native/architecture facts about the selected speculative mechanism.
pub struct SpeculativeMemoryProfile {
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
    plan.auxiliary_bytes_per_position.validate()?;
    plan.sampling_bytes_per_vocabulary_entry.validate()?;
    let mut result = estimate_generation_memory(target)?;
    if target.batch_size != 1 || plan.max_draft_tokens == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "speculative forecast",
            detail: "requires one lane and a positive proposal width".into(),
        });
    }
    plan.scheduler
        .validate()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let draft_estimate = plan
        .draft
        .as_ref()
        .map(estimate_generation_memory)
        .transpose()?;
    let output = target
        .max_output_tokens
        .unwrap_or(target.forecast_output_tokens);
    let canonical = add(target.input.model_positions, output)?;
    let width = add(plan.max_draft_tokens, 1)?;
    // A verified block and one optimistic block may extend beyond the canonical
    // frontier. Use configured maxima even if the run ends or disables lookahead.
    let lookahead = u64::from(
        plan.scheduler.lookahead_blocks != 0 && plan.scheduler.max_optimistic_branches != 0,
    );
    let frontier = add(canonical, mul(width, add(1, lookahead)?)?)?;
    let mut pools = target.domains.clone();
    if let Some(draft) = &plan.draft {
        if draft.input != target.input
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
        let resident = add(
            t.map_or(0, |p| p.already_resident_bytes),
            d.map_or(0, |(_, p)| p.already_resident_bytes),
        )?;
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
        let stages = if output == 0 {
            vec![MemoryPhase::Prefill]
        } else {
            vec![
                MemoryPhase::Prefill,
                MemoryPhase::SpeculativeDraft,
                MemoryPhase::SpeculativeVerification,
                MemoryPhase::SpeculativeCommit,
            ]
        };
        for stage in stages {
            let prefill = stage == MemoryPhase::Prefill;
            let positions = if prefill {
                target.input.model_positions
            } else {
                frontier
            };
            let query = if prefill { positions } else { width };
            let get = |r: &GenerationMemoryRequest, p: &DomainMemoryPlan| {
                phase(p, r, stage, positions, query)
            };
            let tp = t.map(|p| get(target, p)).transpose()?;
            let dp = d.map(|(r, p)| get(r, p)).transpose()?;
            let mut p = tp
                .clone()
                .or_else(|| dp.clone())
                .expect("pool has an owner");
            if let (Some(t), Some(d)) = (&tp, &dp) {
                p.parameters = t.parameters.add(&d.parameters)?;
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
            if matches!(pool.domain, MemoryDomain::Host | MemoryDomain::Unified) {
                p.staging = p.staging.add(&MemoryBytes::estimated(0,
                    mul(mul(positions, 4)?, add(6, mul(3,lookahead)?)?)?,
                    "single-lane token/history copies through rollback, replay and optimistic advancement"))?;
            }
            let execution_pool = t.is_some_and(|p| !p.executions.is_empty())
                || d.is_some_and(|(_, p)| !p.executions.is_empty());
            if execution_pool {
                // The ordinary workspace already includes cache-update overlap.
                // These *additional* copies envelope durable checkpoints, seed,
                // proposal restore/replacement, rollback and optimistic branches.
                let target_copies = if prefill {
                    1
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
                            .zip(e.workspace.as_ref().map(|w| w.vocabulary_size))
                            .map(|(a, b)| a.max(b));
                    }
                }
                let distributions = match vocab {
                    Some(v) => interval(&plan.sampling_bytes_per_vocabulary_entry,
                        mul(v, mul(width, add(2,lookahead)?)?)?,
                        &format!("proposal, verification and residual distributions including optimistic overlap: {}", plan.sampling_bytes_per_vocabulary_entry.detail))?,
                    None => MemoryBytes::unknown("speculative sampling vocabulary geometry unavailable"),
                };
                p.staging = p.staging.add(&distributions)?;
            }
            // The frontier includes uncommitted positions and worst-case replay.
            // Do not promote that upper envelope to an inevitable allocation.
            if !prefill {
                p.persistent_state.lower_bytes = 0;
                p.persistent_state.kind = ObservationKind::Estimated;
                p.workspace.lower_bytes = 0;
                p.workspace.kind = ObservationKind::Estimated;
            }
            p.total = total(&p)?;
            phases.push(p);
        }
        let peak = phases
            .iter()
            .fold(MemoryBytes::exact(0), |a, p| a.maximum(&p.total));
        let loading = t
            .map(|p| p.loading_peak.clone())
            .unwrap_or(MemoryBytes::exact(0))
            .add(
                &d.map(|(_, p)| p.loading_peak.clone())
                    .unwrap_or(MemoryBytes::exact(0)),
            )?
            .add(&phases[0].backend_overhead)?;
        let overall = peak.maximum(&loading);
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
        let growth_in = |domains: &[DomainMemoryEstimate]| {
            domains
                .iter()
                .find(|d| d.domain == pool.domain)
                .map_or(0, |d| d.state_growth_bytes_per_position)
        };
        let growth = growth_in(&target_domains)
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
