//! Borrowed reporting requirements for the same admission policy worker.
//!
//! These values perform policy arithmetic only. They neither authenticate a
//! source nor reserve capacity; runtime must bind its actual source recipe and
//! use the existing atomic account commit before constructing original owners.

use super::{
    Admission, AdmissionRejection, AdmissionRequest, AdmissionResult, CapabilityError,
    EstimationCompleteness, ExecutionWorkspaceEstimate, ModelCapabilities, RuntimeStateEstimate,
    WorkspaceBound,
};
use crate::{InferenceGeometry, Observed};

/// An allocation-free error from checked admission geometry or arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionPolicyError {
    /// A request or estimate has incompatible geometry.
    #[error("invalid model capability field {field}: {detail}")]
    InvalidConfiguration {
        /// Stable field name.
        field: &'static str,
        /// Stable geometry diagnostic.
        detail: &'static str,
    },
    /// Checked byte or position arithmetic overflowed.
    #[error("capability arithmetic overflow while computing {operation}")]
    ArithmeticOverflow {
        /// Stable operation name.
        operation: &'static str,
    },
}

impl From<AdmissionPolicyError> for CapabilityError {
    fn from(value: AdmissionPolicyError) -> Self {
        match value {
            AdmissionPolicyError::InvalidConfiguration { field, detail } => {
                Self::InvalidConfiguration {
                    field,
                    detail: detail.into(),
                }
            }
            AdmissionPolicyError::ArithmeticOverflow { operation } => {
                Self::ArithmeticOverflow { operation }
            }
        }
    }
}

/// Borrowed scalar observation, with its existing missing-bound explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionObservation<'a> {
    /// A known reporting value. This is not source or allocation authority.
    Available(u64),
    /// The selected producer has no available value.
    Unavailable(&'a str),
}

impl<'a> From<&'a Observed<u64>> for AdmissionObservation<'a> {
    fn from(value: &'a Observed<u64>) -> Self {
        match value {
            Observed::Available { value, .. } => Self::Available(*value),
            Observed::Unsupported { reason } | Observed::Unavailable { reason } => {
                Self::Unavailable(reason)
            }
        }
    }
}

/// The six ordered workspace terms, without allocating descriptive strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionWorkspaceRequirements {
    /// Exact request geometry.
    pub geometry: InferenceGeometry,
    /// Activations, attention, vocabulary, state update, materialization and
    /// retained output, in the existing estimate's checked summation order.
    pub components: [Option<u64>; 6],
}

impl From<&ExecutionWorkspaceEstimate> for ExecutionWorkspaceRequirements {
    fn from(value: &ExecutionWorkspaceEstimate) -> Self {
        Self {
            geometry: value.geometry,
            components: [
                value.activations.bytes(),
                value.attention.bytes(),
                value.vocabulary.bytes(),
                value.state_update.bytes(),
                value.materialization.bytes(),
                value.retained.bytes(),
            ],
        }
    }
}

impl ExecutionWorkspaceRequirements {
    /// Performs the same checked geometry and ordered sum as owned reporting.
    pub fn peak_bytes(self) -> Result<Option<u64>, AdmissionPolicyError> {
        self.geometry.validate_fixed()?;
        let mut total = 0u64;
        for value in self.components {
            let Some(bytes) = value else {
                return Ok(None);
            };
            total = add(total, bytes, "simultaneous execution workspace")?;
        }
        Ok(Some(total))
    }
}

/// Scalar selected-state backing evidence used only for policy reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedStateRequirements {
    /// Exact selected request geometry.
    pub geometry: InferenceGeometry,
    /// Complete reported backing capacity, or an explicit missing bound.
    pub bytes: Option<u64>,
}

/// Borrowed-report projection of the state fields consumed by admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionStateRequirements {
    /// Complete physical-domain attribution is available independently of any
    /// aggregate diagnostic. This pure observation grants no execution authority.
    pub physical_domains: bool,
    /// Actual state plus distinct retained-media and media-workspace terms.
    pub requested_state_bytes: u64,
    /// State estimator batch assumption.
    pub batch_size: u64,
    /// State estimator total position assumption.
    pub requested_positions: u64,
    /// State and media coverage before execution workspace.
    pub persistent_state_completeness: EstimationCompleteness,
    /// Complete state and execution coverage.
    pub completeness: EstimationCompleteness,
    /// Selected backing geometry and availability, when supplied.
    pub selected_state_backing: Option<SelectedStateRequirements>,
    /// Ordered execution workspace, when supplied.
    pub execution_workspace: Option<ExecutionWorkspaceRequirements>,
}

impl From<&RuntimeStateEstimate> for AdmissionStateRequirements {
    fn from(value: &RuntimeStateEstimate) -> Self {
        Self {
            physical_domains: value.physical_domains.is_some()
                && value.selected_state_backing.as_ref().is_none_or(|backing| {
                    !matches!(backing.bound, super::WorkspaceBound::Unknown { .. })
                })
                && value.execution_workspace.as_ref().is_some_and(|w| {
                    w.physical_domains.is_some()
                        && [
                            &w.activations,
                            &w.attention,
                            &w.vocabulary,
                            &w.state_update,
                            &w.materialization,
                            &w.retained,
                        ]
                        .into_iter()
                        .all(|bound| !matches!(bound, super::WorkspaceBound::Unknown { .. }))
                }),
            requested_state_bytes: value.requested_state_bytes,
            batch_size: value.assumptions.batch_size,
            requested_positions: value.assumptions.requested_positions,
            persistent_state_completeness: value.persistent_state_completeness,
            completeness: value.completeness,
            selected_state_backing: value.selected_state_backing.as_ref().map(|v| {
                SelectedStateRequirements {
                    geometry: v.geometry,
                    bytes: v.bytes(),
                }
            }),
            execution_workspace: value.execution_workspace.as_ref().map(Into::into),
        }
    }
}

impl AdmissionStateRequirements {
    pub(super) fn validate_selected_geometry(self) -> Result<(), AdmissionPolicyError> {
        if let Some(backing) = self.selected_state_backing {
            let geometry = backing.geometry;
            geometry.validate_fixed()?;
            if geometry.batch_size != self.batch_size
                || geometry.cached_positions + geometry.input_positions + geometry.max_output_tokens
                    != self.requested_positions
            {
                return Err(AdmissionPolicyError::InvalidConfiguration {
                    field: "selected_state_backing",
                    detail: "selected backing and persistent-state geometry differ",
                });
            }
        }
        if let (Some(backing), Some(workspace)) =
            (self.selected_state_backing, self.execution_workspace)
        {
            if backing.geometry != workspace.geometry {
                return Err(AdmissionPolicyError::InvalidConfiguration {
                    field: "selected_state_backing",
                    detail: "selected backing and execution workspace require identical request geometry",
                });
            }
        }
        Ok(())
    }
}

/// Allocation-free inputs to the existing reporting and policy comparison.
///
/// A successful result does not reserve memory, authenticate source ownership or
/// grant execution authority. A runtime original producer must provide its own
/// closed complete source recipe and use the existing account commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionRequirements<'a> {
    /// Effective maximum model context.
    pub maximum_context: AdmissionObservation<'a>,
    /// Scalar state and workspace requirements.
    pub state: AdmissionStateRequirements,
    /// Separately proved incremental requirement, when supplied.
    pub incremental: Option<AdmissionObservation<'a>>,
    /// The complete incremental charge is reported independently in each
    /// physical domain and has no aggregate byte diagnostic.
    pub incremental_per_domain: bool,
}

impl<'a> AdmissionRequirements<'a> {
    /// Borrows existing descriptions and copies scalar fields without allocation.
    pub fn from_reports(
        capabilities: &'a ModelCapabilities,
        state: &RuntimeStateEstimate,
        incremental: Option<&'a WorkspaceBound>,
    ) -> Self {
        Self {
            maximum_context: (&capabilities.effective_max_context).into(),
            state: state.into(),
            incremental: incremental.and_then(|v| match v {
                WorkspaceBound::Bounded { bytes, .. } => {
                    Some(AdmissionObservation::Available(*bytes))
                }
                WorkspaceBound::Unknown { reason } => {
                    Some(AdmissionObservation::Unavailable(reason))
                }
                WorkspaceBound::PerDomain { .. } => None,
            }),
            incremental_per_domain: matches!(incremental, Some(WorkspaceBound::PerDomain { .. })),
        }
    }
}

/// A borrowed rejection retains diagnostics without constructing an error box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowedAdmissionRejection<'a> {
    /// Prompt alone exceeds the context.
    PromptExceedsContext {
        /// Actual decoder extent.
        prompt_positions: u64,
        /// Actual context maximum.
        maximum_positions: u64,
    },
    /// Generated allowance exceeds the remaining context.
    OutputHeadroomExceedsContext {
        /// Actual decoder extent.
        prompt_positions: u64,
        /// Requested output allowance.
        output_tokens: u64,
        /// Actual context maximum.
        maximum_positions: u64,
    },
    /// Context estimation is unavailable.
    EstimationUnsupported(&'a str),
    /// Incremental workspace is unknown.
    IncrementalUnsupported(&'a str),
    /// A required state or execution component is incomplete.
    IncompleteCoverage(EstimationCompleteness),
}

impl BorrowedAdmissionRejection<'_> {
    /// Builds the legacy owned diagnostic; callers must account for allocation.
    pub fn into_owned(self) -> AdmissionRejection {
        match self.try_into_owned_with(|text| Ok::<_, std::convert::Infallible>(text.to_string())) {
            Ok(rejection) => rejection,
            Err(never) => match never {},
        }
    }

    /// Constructs the same owned diagnostic through a supplied text destination.
    /// The destination receives the complete legacy diagnostic once, without an
    /// intermediate String. This performs no admission or storage reservation.
    pub fn try_into_owned_with<E>(
        self,
        mut text: impl FnMut(std::fmt::Arguments<'_>) -> Result<String, E>,
    ) -> Result<AdmissionRejection, E> {
        Ok(match self {
            Self::PromptExceedsContext {
                prompt_positions,
                maximum_positions,
            } => AdmissionRejection::PromptExceedsContext {
                prompt_positions,
                maximum_positions,
            },
            Self::OutputHeadroomExceedsContext {
                prompt_positions,
                output_tokens,
                maximum_positions,
            } => AdmissionRejection::OutputHeadroomExceedsContext {
                prompt_positions,
                output_tokens,
                maximum_positions,
            },
            Self::EstimationUnsupported(reason) => AdmissionRejection::EstimationUnsupported {
                reason: text(format_args!("{reason}"))?,
            },
            Self::IncrementalUnsupported(reason) => AdmissionRejection::EstimationUnsupported {
                reason: text(format_args!(
                    "incremental execution bound is unknown: {reason}"
                ))?,
            },
            Self::IncompleteCoverage(coverage) => AdmissionRejection::EstimationUnsupported {
                reason: text(format_args!(
                    "architecture estimator coverage is {coverage:?}"
                ))?,
            },
        })
    }
}

/// Scalar reporting outcome; not allocation or submission permission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionPolicyDecision {
    /// Prompt plus output allowance.
    pub requested_positions: u64,
    /// Complete incremental requirement including safety reserve.
    pub incremental_required_bytes: Option<u64>,
}

/// Allocation-free policy result, with borrowed rejection diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowedAdmissionResult<'a> {
    /// The policy comparison accepted these requirements only.
    Admitted(AdmissionPolicyDecision),
    /// The policy comparison rejected these requirements.
    Rejected(BorrowedAdmissionRejection<'a>),
}

fn add(a: u64, b: u64, operation: &'static str) -> Result<u64, AdmissionPolicyError> {
    a.checked_add(b)
        .ok_or(AdmissionPolicyError::ArithmeticOverflow { operation })
}

/// Checks the exact context policy before any candidate construction.
/// This borrowed reporting check creates no storage or submission authority.
pub fn check_admission_context_borrowed<'a>(
    maximum: AdmissionObservation<'a>,
    request: AdmissionRequest,
) -> Result<Option<BorrowedAdmissionRejection<'a>>, AdmissionPolicyError> {
    let maximum = match maximum {
        AdmissionObservation::Available(value) => value,
        AdmissionObservation::Unavailable(reason) => {
            return Ok(Some(BorrowedAdmissionRejection::EstimationUnsupported(
                reason,
            )));
        }
    };
    if request.input.model_positions > maximum {
        return Ok(Some(BorrowedAdmissionRejection::PromptExceedsContext {
            prompt_positions: request.input.model_positions,
            maximum_positions: maximum,
        }));
    }
    let positions = add(
        request.input.model_positions,
        request.max_output_tokens,
        "admission prompt plus output",
    )?;
    if positions > maximum {
        return Ok(Some(
            BorrowedAdmissionRejection::OutputHeadroomExceedsContext {
                prompt_positions: request.input.model_positions,
                output_tokens: request.max_output_tokens,
                maximum_positions: maximum,
            },
        ));
    }
    Ok(None)
}

/// Runs the same core policy used by owned admission reporting, without allocation.
/// Source authentication and the sole atomic reservation remain runtime duties.
pub fn apply_admission_requirements<'a>(
    request: AdmissionRequest,
    requirements: AdmissionRequirements<'a>,
) -> Result<BorrowedAdmissionResult<'a>, AdmissionPolicyError> {
    use BorrowedAdmissionRejection as R;
    use BorrowedAdmissionResult::{Admitted, Rejected};
    if let Some(rejection) =
        check_admission_context_borrowed(requirements.maximum_context, request.clone())?
    {
        return Ok(Rejected(rejection));
    }
    let state = requirements.state;
    state.validate_selected_geometry()?;
    let requested_positions = add(
        request.input.model_positions,
        request.max_output_tokens,
        "admission prompt plus output",
    )?;
    if state.batch_size != request.batch_size || state.requested_positions != requested_positions {
        return Err(AdmissionPolicyError::InvalidConfiguration {
            field: "admission_estimate",
            detail: "state estimate does not describe the admitted request",
        });
    }
    if let Some(workspace) = state.execution_workspace {
        let geometry = workspace.geometry;
        if geometry.batch_size != request.batch_size
            || geometry
                .cached_positions
                .checked_add(geometry.input_positions)
                != Some(request.input.model_positions)
            || geometry.max_output_tokens != request.max_output_tokens
        {
            return Err(AdmissionPolicyError::InvalidConfiguration {
                field: "admission_workspace",
                detail: "workspace estimate does not describe the admitted request",
            });
        }
    }
    let workspace_bytes = match state
        .execution_workspace
        .map(ExecutionWorkspaceRequirements::peak_bytes)
        .transpose()
    {
        Ok(value) => value.flatten(),
        Err(_) if state.physical_domains => None,
        Err(error) => return Err(error),
    };
    let full_required = match state
        .requested_state_bytes
        .checked_add(workspace_bytes.unwrap_or(0))
    {
        Some(bytes) if workspace_bytes.is_some() || !state.physical_domains => Some(bytes),
        _ if state.physical_domains => None,
        _ => {
            return Err(AdmissionPolicyError::ArithmeticOverflow {
                operation: "state plus execution workspace",
            })
        }
    };
    let incremental_required_bytes = match requirements.incremental {
        Some(AdmissionObservation::Available(bytes)) => Some(bytes),
        Some(AdmissionObservation::Unavailable(reason)) => {
            return Ok(Rejected(R::IncrementalUnsupported(reason)))
        }
        None if requirements.incremental_per_domain && state.physical_domains => None,
        None if requirements.incremental_per_domain => {
            return Ok(Rejected(R::IncrementalUnsupported(
                "physical-domain requirements are incomplete",
            )))
        }
        None => full_required,
    };
    if (!state.physical_domains
        && (workspace_bytes.is_none()
            || state
                .selected_state_backing
                .is_some_and(|backing| backing.bytes.is_none())))
        || state.persistent_state_completeness == EstimationCompleteness::PersistentStateOnly
        || state.completeness == EstimationCompleteness::PersistentStateOnly
    {
        return Ok(Rejected(R::IncompleteCoverage(state.completeness)));
    }
    Ok(Admitted(AdmissionPolicyDecision {
        requested_positions,
        incremental_required_bytes,
    }))
}

pub(super) fn owned(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    incremental: Option<&WorkspaceBound>,
) -> Result<AdmissionResult, CapabilityError> {
    if let (Some(persistent), Some(workspace)) = (
        state.physical_domains.as_ref(),
        state
            .execution_workspace
            .as_ref()
            .and_then(|w| w.physical_domains.as_ref()),
    ) {
        if persistent.geometry != workspace.geometry {
            return Err(CapabilityError::InvalidConfiguration {
                field: "physical_domains",
                detail: "state and workspace geometries differ".into(),
            });
        }
        for (domain, initial) in persistent.decoder_state.iter() {
            let mut charge = initial;
            for component in [
                &persistent.media_embeddings,
                &persistent.media_workspace,
                &workspace.activations,
                &workspace.attention,
                &workspace.vocabulary,
                &workspace.state_update,
                &workspace.materialization,
                &workspace.retained,
            ] {
                charge = charge
                    .checked_add(component.get(domain).map_err(|_| {
                        CapabilityError::InvalidConfiguration {
                            field: "physical_domains",
                            detail: "component topology differs".into(),
                        }
                    })?)
                    .map_err(|_| CapabilityError::ArithmeticOverflow {
                        operation: "physical-domain requirement",
                    })?;
            }
        }
    }
    match apply_admission_requirements(
        request.clone(),
        AdmissionRequirements::from_reports(capabilities, &state, incremental),
    )? {
        BorrowedAdmissionResult::Rejected(rejection) => {
            Ok(AdmissionResult::Rejected(rejection.into_owned()))
        }
        BorrowedAdmissionResult::Admitted(decision) => Ok(AdmissionResult::Admitted(Admission {
            memory_limits: request.memory_limits,
            additional_headroom: request.additional_headroom,
            requested_positions: decision.requested_positions,
            state,
            incremental_required_bytes: decision.incremental_required_bytes,
        })),
    }
}
