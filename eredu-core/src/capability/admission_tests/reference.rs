//! Preserved pre-refactor policy/geometry control, independent of borrowed worker.
use super::*;
pub(super) fn reference_policy(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    incremental: Option<&WorkspaceBound>,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
    if let Some(rejection) = reference_context(capabilities, request)? {
        return Ok(AdmissionResult::Rejected(rejection));
    }
    reference_selected(&state)?;
    let requested_positions = checked_add(
        request.input.model_positions,
        request.max_output_tokens,
        "admission prompt plus output",
    )?;
    if state.assumptions.batch_size != request.batch_size
        || state.assumptions.requested_positions != requested_positions
    {
        return Err(CapabilityError::InvalidConfiguration {
            field: "admission_estimate",
            detail: "state estimate does not describe the admitted request".into(),
        });
    }
    if let Some(workspace) = &state.execution_workspace {
        let geometry = workspace.geometry;
        if geometry.batch_size != request.batch_size
            || geometry
                .cached_positions
                .checked_add(geometry.input_positions)
                != Some(request.input.model_positions)
            || geometry.max_output_tokens != request.max_output_tokens
        {
            return Err(CapabilityError::InvalidConfiguration {
                field: "admission_workspace",
                detail: "workspace estimate does not describe the admitted request".into(),
            });
        }
    }
    let workspace_bytes = state
        .execution_workspace
        .as_ref()
        .map(reference_workspace)
        .transpose()?
        .flatten();
    let full_required_bytes = checked_add(
        state.requested_state_bytes,
        workspace_bytes.unwrap_or(0),
        "state plus execution workspace",
    )?;
    let required_bytes = match incremental {
        Some(WorkspaceBound::Bounded { bytes, .. }) => *bytes,
        Some(WorkspaceBound::Unknown { reason }) => {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::EstimationUnsupported {
                    reason: format!("incremental execution bound is unknown: {reason}"),
                },
            ));
        }
        None => full_required_bytes,
    };
    let incremental_required_bytes = checked_add(
        required_bytes,
        request.safety_reserve_bytes,
        "state plus safety reserve",
    )?;
    if let Some(budget_bytes) = request.application_memory_budget_bytes {
        if incremental_required_bytes > budget_bytes {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::MemoryBudgetExceeded {
                    required_bytes: incremental_required_bytes,
                    budget_bytes,
                },
            ));
        }
    }
    if (incremental.is_some()
        || request.require_complete_estimate
        || request.application_memory_budget_bytes.is_some())
        && (workspace_bytes.is_none()
            || state
                .selected_state_backing
                .as_ref()
                .is_some_and(|backing| backing.bytes().is_none())
            || state.persistent_state_completeness == EstimationCompleteness::PersistentStateOnly
            || state.completeness == EstimationCompleteness::PersistentStateOnly)
    {
        return Ok(AdmissionResult::Rejected(
            AdmissionRejection::EstimationUnsupported {
                reason: format!(
                    "architecture estimator coverage is {:?}",
                    state.completeness
                ),
            },
        ));
    }
    let available_memory_bytes = match available {
        Some(report) => match &report.available_memory_bytes {
            Observed::Available { value, .. } => Some(*value),
            Observed::Unsupported { reason } | Observed::Unavailable { reason } => {
                return Ok(AdmissionResult::Rejected(
                    AdmissionRejection::AvailableMemoryUnavailable {
                        reason: reason.clone(),
                    },
                ));
            }
        },
        None => None,
    };
    if let Some(available_bytes) = available_memory_bytes {
        if incremental_required_bytes > available_bytes {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::InsufficientAvailableMemory {
                    required_bytes: incremental_required_bytes,
                    available_bytes,
                },
            ));
        }
    }
    Ok(AdmissionResult::Admitted(Admission {
        requested_positions,
        state,
        incremental_required_bytes,
        available_memory_bytes,
    }))
}
fn reference_context(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
) -> Result<Option<AdmissionRejection>, CapabilityError> {
    let maximum = match &capabilities.effective_max_context {
        Observed::Available { value, .. } => *value,
        Observed::Unsupported { reason } | Observed::Unavailable { reason } => {
            return Ok(Some(AdmissionRejection::EstimationUnsupported {
                reason: reason.clone(),
            }));
        }
    };
    if request.input.model_positions > maximum {
        return Ok(Some(AdmissionRejection::PromptExceedsContext {
            prompt_positions: request.input.model_positions,
            maximum_positions: maximum,
        }));
    }
    let requested_positions = checked_add(
        request.input.model_positions,
        request.max_output_tokens,
        "admission prompt plus output",
    )?;
    if requested_positions > maximum {
        return Ok(Some(AdmissionRejection::OutputHeadroomExceedsContext {
            prompt_positions: request.input.model_positions,
            output_tokens: request.max_output_tokens,
            maximum_positions: maximum,
        }));
    }
    Ok(None)
}
fn reference_selected(state: &RuntimeStateEstimate) -> Result<(), CapabilityError> {
    if let Some(backing) = &state.selected_state_backing {
        let geometry = backing.geometry;
        reference_geometry(geometry)?;
        if geometry.batch_size != state.assumptions.batch_size
            || geometry.cached_positions + geometry.input_positions + geometry.max_output_tokens
                != state.assumptions.requested_positions
        {
            return Err(CapabilityError::InvalidConfiguration {
                field: "selected_state_backing",
                detail: "selected backing and persistent-state geometry differ".into(),
            });
        }
    }
    if let (Some(backing), Some(workspace)) =
        (&state.selected_state_backing, &state.execution_workspace)
    {
        if backing.geometry != workspace.geometry {
            return Err(CapabilityError::InvalidConfiguration {
                field: "selected_state_backing",
                detail:
                    "selected backing and execution workspace require identical request geometry"
                        .into(),
            });
        }
    }
    Ok(())
}
fn reference_geometry(value: InferenceGeometry) -> Result<(), crate::CapabilityError> {
    if value.batch_size == 0
        || value.input_positions == 0
        || value.prefill_chunk_positions == 0
        || value.prefill_chunk_positions > value.input_positions
    {
        return Err(crate::CapabilityError::InvalidConfiguration {
            field: "inference_geometry",
            detail: "batch, input and chunk must be positive; chunk cannot exceed input".into(),
        });
    }
    value
        .cached_positions
        .checked_add(value.input_positions)
        .and_then(|n| n.checked_add(value.max_output_tokens))
        .ok_or(crate::CapabilityError::ArithmeticOverflow {
            operation: "inference frontier",
        })?;
    Ok(())
}
fn reference_workspace(value: &ExecutionWorkspaceEstimate) -> Result<Option<u64>, CapabilityError> {
    reference_geometry(value.geometry)?;
    let mut total = 0u64;
    for bound in [
        &value.activations,
        &value.attention,
        &value.vocabulary,
        &value.state_update,
        &value.materialization,
        &value.retained,
    ] {
        let Some(bytes) = bound.bytes() else {
            return Ok(None);
        };
        total = total
            .checked_add(bytes)
            .ok_or(CapabilityError::ArithmeticOverflow {
                operation: "simultaneous execution workspace",
            })?;
    }
    Ok(Some(total))
}
