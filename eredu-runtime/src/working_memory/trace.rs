//! Composition of equation execution and separately owned workspace bounds.

use eredu_core::{
    CapabilityError, ExecutionWorkspaceEstimate, RuntimeStateEstimate, WorkspaceBound,
};
use eredu_nn::workspace::WorkspaceTraceReport;

/// Extends the existing state/admission report with a completed equation trace.
///
/// `outside_trace` must price every required operation absent from the trace:
/// preparation, state mechanisms, materialization, sampling, observations and
/// retained snapshots, as applicable. Unknown components remain unknown. Its
/// activation component must exclude the trace being added here.
///
/// The allocation graph can share storage across attention, projection and
/// cache operations. Its combined tensor and disjoint managed-host transient
/// bound is therefore assigned once to
/// `activations`, with that aggregation recorded in the assumptions. Separately
/// retained roots are already included in the persistent-state estimate and
/// must fit that estimate. Each trace conservatively retains all allocations
/// until completion; callers pricing several safely separated spans take their
/// peak rather than summing their graphs.
pub fn with_equation_workspace(
    mut state: RuntimeStateEstimate,
    mut outside_trace: ExecutionWorkspaceEstimate,
    trace: &WorkspaceTraceReport,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let metadata = super::WorkspaceReportMetadata::ordinary();
    if let Some(domains) = &trace.physical_domains {
        if let (Some(transient), Some(retained)) =
            (&domains.state_transient, &domains.retained_state)
        {
            if let Some(outside) = &mut outside_trace.physical_domains {
                outside.activations = metadata
                    .combine_domain_requirements(&outside.activations, transient, true)
                    .map_err(super::WorkspaceReportError::into_capability)?;
            }
            if let Some(existing) = &mut state.physical_domains {
                existing.decoder_state = metadata
                    .clone_domain_requirements(retained)
                    .map_err(super::WorkspaceReportError::into_capability)?;
            } else if state.multimodal_embedding_bytes == 0
                && state.media_execution_workspace_bytes == 0
            {
                state.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
                    geometry: outside_trace.geometry,
                    decoder_state: metadata
                        .clone_domain_requirements(retained)
                        .map_err(super::WorkspaceReportError::into_capability)?,
                    media_embeddings: metadata
                        .empty_domain_requirements(retained)
                        .map_err(super::WorkspaceReportError::into_capability)?,
                    media_workspace: metadata
                        .empty_domain_requirements(retained)
                        .map_err(super::WorkspaceReportError::into_capability)?,
                });
            }
        } else {
            outside_trace.physical_domains = None;
        }
    } else {
        outside_trace.physical_domains = None;
    }
    let transient = match trace.inference_transient_bytes() {
        Some(bytes) => WorkspaceBound::bounded(
            bytes,
            format!(
                "equation trace aggregates unique transient tensor outputs, displaced opening state and scratch plus disjoint managed-host workspace, including attention, projections and state copies, until completion; closing state roots excluded once; {}",
                trace.assumptions.join("; ")
            ),
        ),
        None if trace.physical_domains.is_some() => WorkspaceBound::PerDomain {
            assumptions: "complete physical-domain equation demand has no aggregate u64 diagnostic"
                .into(),
        },
        None => WorkspaceBound::Unknown {
            reason: format!(
                "equation trace has no complete transient/state-overlap bound; opening state supplied: {}; unpriced tensor-buffer operation indices: {:?}; unpriced managed-host operation indices: {:?}",
                trace.state.is_some(),
                trace.unpriced_operations,
                trace.unpriced_host_operations
            ),
        },
    };
    with_transient_workspace(
        state,
        outside_trace,
        transient,
        trace.state.as_ref().and_then(|state| state.retained_bytes),
    )
}

pub(super) fn with_transient_workspace(
    state: RuntimeStateEstimate,
    outside_trace: ExecutionWorkspaceEstimate,
    transient: WorkspaceBound,
    retained_bytes: Option<u64>,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    with_transient_workspace_metadata(
        state,
        outside_trace,
        transient,
        retained_bytes,
        super::WorkspaceReportMetadata::ordinary(),
    )
    .map_err(super::WorkspaceReportError::into_capability)
}

pub(super) fn with_transient_workspace_metadata(
    state: RuntimeStateEstimate,
    mut outside_trace: ExecutionWorkspaceEstimate,
    transient: WorkspaceBound,
    retained_bytes: Option<u64>,
    metadata: super::WorkspaceReportMetadata<'_>,
) -> Result<RuntimeStateEstimate, super::WorkspaceReportError> {
    let attributed = state.physical_domains.is_some() && outside_trace.physical_domains.is_some();
    if !attributed && retained_bytes.is_some_and(|bytes| bytes > state.requested_state_bytes) {
        return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
            field: "equation_workspace",
            detail: "traced retained storage exceeds the persistent-state estimate",
        }
        .into());
    }
    let transient = match retained_bytes {
        Some(_) => transient,
        None if attributed => metadata.per_domain(format_args!(
            "complete per-domain retained-state backing has no aggregate u64 diagnostic"
        ))?,
        None => metadata.unknown(format_args!(
            "equation trace has no complete retained-state backing bound"
        ))?,
    };
    outside_trace.activations = match (outside_trace.activations, transient) {
        (
            WorkspaceBound::Bounded { bytes, assumptions },
            WorkspaceBound::Bounded {
                bytes: traced,
                assumptions: trace_assumptions,
            },
        ) => {
            let bytes = bytes.checked_add(traced);
            if attributed && bytes.is_none() {
                metadata.per_domain(format_args!(
                    "aggregate diagnostic exceeds u64; physical-domain requirements remain complete"
                ))?
            } else {
                let bytes = bytes.ok_or(eredu_core::AdmissionPolicyError::ArithmeticOverflow {
                    operation: "equation and external workspace",
                })?;
                metadata.bounded(bytes, format_args!("{assumptions}; {trace_assumptions}"))?
            }
        }
        (bound @ WorkspaceBound::Unknown { .. }, _) => bound,
        (_, unknown @ WorkspaceBound::Unknown { .. }) => unknown,
        (bound @ WorkspaceBound::PerDomain { .. }, _)
        | (_, bound @ WorkspaceBound::PerDomain { .. }) => bound,
    };
    metadata.admit::<RuntimeStateEstimate>()?;
    Ok(state.with_execution_workspace_fixed(outside_trace)?)
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
