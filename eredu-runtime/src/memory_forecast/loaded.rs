//! Portable composition of native residency facts and request calibration.
use super::*;
use eredu_core::{InputTokenCount, Observed, PhysicalMemorySemantics};
fn observed(value: &Observed<u64>) -> Option<u64> {
    value.value().copied()
}
fn failure(error: impl std::fmt::Display) -> GenerationForecastError {
    CapabilityError::Observation(error.to_string()).into()
}
fn chunk_tokens(policy: eredu_core::PrefillChunkPolicy, positions: u64) -> u64 {
    match policy {
        eredu_core::PrefillChunkPolicy::Unchunked => positions.max(1),
        eredu_core::PrefillChunkPolicy::Bounded(n) => n.get() as u64,
    }
}
/// Application comparisons and optional calibration overrides. Missing available
/// bytes are filled from backend observations, never from installed capacity.
#[derive(Debug, Clone, Default)]
pub struct GenerationForecastOptions {
    /// Execution-pool budget and reserve.
    pub budget: MemoryBudget,
    /// Independent host budget for a discrete accelerator.
    pub host_budget: MemoryBudget,
    /// Overrides the observed allocator plus graph/driver allowance.
    pub backend_overhead: Option<MemoryBytes>,
    /// Shared, labeled planning assumptions.
    pub calibration: ForecastCalibration,
}

/// Composes loaded request geometry and native residency without allocation or execution.
pub fn loaded_generation_request(
    mut profile: LoadedMemoryProfile,
    input: InputTokenCount,
    output: u64,
    prefill: eredu_core::PrefillChunkPolicy,
    contract: ForecastExecutionContract,
    options: &GenerationForecastOptions,
) -> Result<(GenerationMemoryRequest, Vec<String>), GenerationForecastError> {
    let unified = profile.parameters.physical_semantics == PhysicalMemorySemantics::Unified;
    if profile.host_execution {
        profile.parameters.physical_semantics = PhysicalMemorySemantics::Unified;
    }
    let placements =
        crate::memory_estimation::static_parameter_placement(&profile.parameters, None)?;
    let mut geometry = profile.geometry;
    if let Some(cached) = observed(&profile.parameters.current_device_parameter_conversion_bytes) {
        if observed(&profile.parameters.current_device_resident_bytes)
            .is_some_and(|resident| cached > resident)
        {
            return Err(failure(
                "parameter conversions exceed the declared device residency",
            ));
        }
        if let Some(potential) = geometry
            .execution_topology
            .as_mut()
            .and_then(|g| g.selected_parameter_promotion_bytes.as_mut())
        {
            *potential = potential.saturating_sub(cached);
        }
        if let Some(potential) = geometry
            .workspace
            .as_mut()
            .and_then(|g| g.mixed_precision_parameter_bytes.as_mut())
        {
            // These retained conversions are already in the parameter baseline.
            // Keep Some(0) when complete: mixed-width activation/state promotion
            // still applies even when no further parameter conversion is needed.
            let reused = (*potential).min(cached);
            *potential -= reused;
            if reused != 0 {
                geometry.assumptions.push(format!("{reused} bytes of resident F32 parameter conversions are reused; only uncached conversions remain in workspace."));
            }
        }
    }
    let overhead = options.backend_overhead.clone().unwrap_or_else(|| {
        match (
            observed(&profile.allocator_cache_limit),
            observed(&profile.parameters.backend_allocator_cache_bytes),
        ) {
            (Some(limit), Some(retained)) => {
                options.calibration.allocator_overhead(limit, retained)
            }
            _ => MemoryBytes::unknown("current allocator cache limit or retention unavailable"),
        }
    });
    let mut domains = Vec::new();
    let separate = profile.parameters.physical_semantics == PhysicalMemorySemantics::SeparateTiers;
    for (mut domain, mut parameters) in placements {
        let executes = !separate || matches!(domain, MemoryDomain::Device(_));
        if profile.host_execution && !unified {
            domain = MemoryDomain::Host;
        }
        let already_resident_bytes = parameters.lower_bytes;
        if !geometry.fully_resident {
            parameters.upper_bytes = observed(&profile.parameters.logical_parameter_bytes)
                .and_then(|bytes| bytes.checked_mul(2))
                .map(|n| n.max(parameters.lower_bytes));
            parameters.detail =
                "current residency through possible host/device parameter copies".into();
        }
        let mut budget = if executes {
            options.budget.clone()
        } else {
            options.host_budget.clone()
        };
        if budget.available_bytes.is_none() && (!separate || !executes) {
            budget.available_bytes = observed(&profile.available.available_memory_bytes);
        }
        let retained_input = if input.model_positions != input.text_tokens {
            MemoryBytes::unknown(
                "prepared media retention and transformations are not fully projected",
            )
        } else {
            MemoryBytes::exact(input.text_tokens.checked_mul(4).ok_or(
                CapabilityError::ArithmeticOverflow {
                    operation: "retained prompt bytes",
                },
            )?)
        };
        domains.push(DomainMemoryPlan {
            domain,
            resident_parameters: parameters,
            already_resident_bytes,
            retained_input,
            staging: if geometry.fully_resident {
                MemoryBytes::exact(0)
            } else {
                MemoryBytes::unknown("bounded transfer and materialization overlap")
            },
            backend_overhead: if executes {
                overhead.clone()
            } else {
                MemoryBytes::exact(0)
            },
            loading_peak: MemoryBytes::exact(0),
            executions: if executes {
                vec![ExecutionMemoryPlan {
                    execution_topology: geometry.execution_topology.clone(),
                    input_score_attention_mechanism: geometry.input_score_attention_mechanism,
                    state_layout: geometry.state_layout.clone(),
                    workspace: geometry.workspace.clone(),
                    attention: AttentionWorkspace::Unknown,
                    cache_update: CacheUpdateWorkspace::Unknown,
                    logits: contract.logits,
                    workspace_overlap: WorkspaceOverlap::unknown(),
                }]
            } else {
                vec![]
            },
            budget,
        });
    }
    let requested_chunk_tokens = chunk_tokens(prefill, input.model_positions);
    let mut request = GenerationMemoryRequest {
        input,
        max_output_tokens: Some(output),
        forecast_output_tokens: output,
        batch_size: 1,
        prefill_chunk_tokens: if contract.full_pass_reason.is_some() {
            input.model_positions.max(1)
        } else {
            requested_chunk_tokens
        },
        scalar_bytes: geometry.scalar_bytes,
        domains,
    };
    options.calibration.apply(&mut request)?;
    Ok((request, geometry.assumptions))
}
