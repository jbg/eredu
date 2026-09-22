//! Initial reservation metadata, paid by the already retained planning account.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceMetadataError};
use std::{
    error::Error as StdError,
    fmt,
    mem::{size_of, size_of_val},
};

struct Failure {
    cause: eredu_nn::Error,
    // The closed outer Arc and exact cause retire before this account alias.
    _funding: HostMetadataFunding,
}

/// An owning metadata-construction failure retaining the original typed cause.
/// Clones share its closed owner without cloning or formatting that cause.
/// Equality denotes the same retained failure, not diagnostic text equality.
pub struct WorkspaceReservationMetadataError(Option<Arc<Failure>>);
impl WorkspaceReservationMetadataError {
    /// Borrow a planning refusal enclosed by this exact metadata producer.
    /// Retaining its payer must not change candidate-selection policy.
    pub(super) fn planning_error(&self) -> Option<&PrefillPlanningError> {
        self.0
            .as_ref()
            .expect("live metadata failure")
            .cause
            .source()
            .and_then(|source| source.downcast_ref())
    }
}
impl Clone for WorkspaceReservationMetadataError {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl PartialEq for WorkspaceReservationMetadataError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live metadata failure"),
            other.0.as_ref().expect("live metadata failure"),
        )
    }
}
impl Eq for WorkspaceReservationMetadataError {}
impl Drop for WorkspaceReservationMetadataError {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for WorkspaceReservationMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WorkspaceReservationMetadataError")
            .field(&self.0.as_ref().expect("live metadata failure").cause)
            .finish()
    }
}
impl fmt::Display for WorkspaceReservationMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.as_ref().expect("live metadata failure").cause, f)
    }
}
impl StdError for WorkspaceReservationMetadataError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.0.as_ref().expect("live metadata failure").cause)
    }
}

pub(super) fn funding_error(cause: HostMetadataFundingError) -> WorkingMemoryError {
    match cause {
        HostMetadataFundingError::Domain(cause) => WorkingMemoryError::Domain(cause),
        HostMetadataFundingError::DomainAllowance {
            domain,
            required,
            available,
        } => WorkingMemoryError::DomainAllowanceExceeded {
            domain,
            required_bytes: required,
            available_bytes: available,
        },
        HostMetadataFundingError::Overflow => WorkingMemoryError::Overflow,
        HostMetadataFundingError::Poisoned => WorkingMemoryError::Poisoned,
        cause => WorkingMemoryError::MetadataConstruction(WorkspaceMetadataError::Funding(cause)),
    }
}

pub(super) fn neural_error(
    cause: eredu_nn::Error,
    funding: &HostMetadataFunding,
) -> WorkingMemoryError {
    // Fixed admission errors own no source payload. Preserve the shared
    // smaller-chunk classification without wrapping them into an owning error.
    if let Some(fixed) = cause
        .source()
        .and_then(|source| source.downcast_ref::<WorkspaceMetadataError>())
        .copied()
    {
        drop(cause);
        return match fixed {
            WorkspaceMetadataError::Funding(cause) => funding_error(cause),
            WorkspaceMetadataError::Overflow => WorkingMemoryError::Overflow,
            cause => WorkingMemoryError::MetadataConstruction(cause),
        };
    }
    let failure = Failure {
        cause,
        _funding: funding.clone(),
    };
    let controls = [
        size_of::<Failure>(),
        size_of::<WorkspaceReservationMetadataError>(),
        size_of::<Option<Arc<Failure>>>(),
        size_of::<Option<Failure>>(),
        size_of::<WorkingMemoryError>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
    ];
    let bytes = qualified_storage::shared_bytes::<Failure>()
        .and_then(|value| usize::try_from(value).map_err(|_| WorkingMemoryError::Overflow))
        .and_then(|value| {
            value
                .checked_add(size_of_val(&controls))
                .ok_or(WorkingMemoryError::Overflow)
        })
        .and_then(|value| {
            controls
                .into_iter()
                .try_fold(value, usize::checked_add)
                .ok_or(WorkingMemoryError::Overflow)
        });
    let result = bytes.and_then(|bytes| funding.reserve_metadata(bytes).map_err(funding_error));
    if let Err(cause) = result {
        drop(failure);
        return cause;
    }
    WorkingMemoryError::ReservationMetadata(WorkspaceReservationMetadataError(Some(Arc::new(
        failure,
    ))))
}

pub(super) fn constructor_bytes(
    topology: &eredu_core::MemoryTopology,
    requirements: &eredu_core::DomainMemoryRequirements,
    limits: &eredu_core::MemoryLimits,
) -> Result<usize, WorkingMemoryError> {
    constructor_bytes_from_backing(
        topology,
        requirements.clone_backing_bytes()?,
        limits.backing_bytes()?,
    )
}

pub(super) fn constructor_bytes_from_backing(
    topology: &eredu_core::MemoryTopology,
    requirements_backing: u64,
    limits_backing: u64,
) -> Result<usize, WorkingMemoryError> {
    control_mutex::require_known_layout()?;
    let shared = usize::try_from(qualified_storage::shared_bytes::<Reservation>()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
    let controls = [
        shared,
        // Actual Box<AccountNode> and its by-value construction/commit states.
        size_of::<funding::AccountNode>(),
        size_of::<funding::AccountNode>(),
        size_of::<Box<funding::AccountNode>>(),
        size_of::<funding::FundingState>(),
        size_of::<Reservation>(),
        size_of::<ReservationOwner>(),
        size_of::<WorkingMemoryReservation>(),
        size_of::<Option<Reservation>>(),
        size_of::<Result<WorkingMemoryReservation, WorkingMemoryError>>(),
        size_of::<PreparedAccountCommit<'_>>(),
        size_of::<Result<PreparedAccountCommit<'_>, WorkingMemoryError>>(),
        size_of::<std::sync::MutexGuard<'_, Usage>>(),
        size_of::<
            Result<
                std::sync::MutexGuard<'_, Usage>,
                std::sync::PoisonError<std::sync::MutexGuard<'_, Usage>>,
            >,
        >(),
        size_of::<WorkspaceReportMetadata<'_>>(),
        size_of::<Option<HostMetadataFunding>>(),
        size_of::<funding::PendingAccount>(),
        size_of::<funding::AccountTicket>(),
        size_of::<transaction_buffers::TransactionBuffers>(),
        size_of::<ControlMutex<text_preparation::RequestStart>>(),
        size_of::<(u64, u64, u64, Option<u64>, InferenceGeometry)>(),
    ];
    let fixed = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
    // The retained requirement and both live limit vectors belong to this
    // metadata owner. FundingState owns one dense balance for every domain.
    let vectors = requirements_backing
        .checked_add(
            limits_backing
                .checked_mul(2)
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .and_then(|bytes| {
            bytes.checked_add(
                u64::try_from(
                    topology
                        .len()
                        .checked_mul(size_of::<DomainMemoryCharge>())?,
                )
                .ok()?,
            )
        })
        .and_then(|bytes| bytes.checked_add(funding::domain_balance_bytes(topology).ok()?))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    fixed
        .checked_add(vectors)
        .ok_or(WorkingMemoryError::Overflow)
}

/// Backing retained by an ordinary admitted report. Shared declaration owners
/// are borrowed from the request; their producer retains that host custody.
pub(super) fn admission_backing_bytes(admission: &Admission) -> Result<u64, WorkingMemoryError> {
    fn bound(value: &eredu_core::WorkspaceBound) -> Result<u64, WorkingMemoryError> {
        let text = match value {
            eredu_core::WorkspaceBound::Bounded { assumptions, .. }
            | eredu_core::WorkspaceBound::PerDomain { assumptions } => assumptions,
            eredu_core::WorkspaceBound::Unknown { reason } => reason,
        };
        u64::try_from(text.len()).map_err(|_| WorkingMemoryError::Overflow)
    }
    let state = &admission.state;
    let mut bytes = u64::try_from(
        state
            .assumptions
            .sliding_window_bounds
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or(WorkingMemoryError::Overflow)?,
    )
    .map_err(|_| WorkingMemoryError::Overflow)?;
    if let Some(backing) = &state.selected_state_backing {
        bytes = bytes
            .checked_add(bound(&backing.bound)?)
            .ok_or(WorkingMemoryError::Overflow)?;
    }
    if let Some(state) = &state.physical_domains {
        for requirement in [
            &state.decoder_state,
            &state.media_embeddings,
            &state.media_workspace,
        ] {
            bytes = bytes
                .checked_add(requirement.clone_backing_bytes()?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
    }
    if let Some(workspace) = &state.execution_workspace {
        for value in [
            &workspace.activations,
            &workspace.attention,
            &workspace.vocabulary,
            &workspace.state_update,
            &workspace.materialization,
            &workspace.retained,
        ] {
            bytes = bytes
                .checked_add(bound(value)?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        if let Some(workspace) = &workspace.physical_domains {
            for requirement in [
                &workspace.activations,
                &workspace.attention,
                &workspace.vocabulary,
                &workspace.state_update,
                &workspace.materialization,
                &workspace.retained,
            ] {
                bytes = bytes
                    .checked_add(requirement.clone_backing_bytes()?)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
    }
    Ok(bytes)
}
