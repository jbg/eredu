//! Paid payload retention around the existing independent draft driver.
//! Each native phase supplies its own scope, admission and completion. This
//! enclosing scope only retains the actual drafter through the existing recovery
//! chain; it creates no numerical work, observer or completion certificate.
use super::*;
use crate::backend::submission_recovery::PreparedRecovery;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

type RetainedSlot = RefCell<Option<Rc<OrdinaryRetirement<DrafterPayload>>>>;

pub(super) struct FundedRetention {
    retained: DrafterRetention,
    // Recovery frees its Box and native scope before this payload; its retained
    // Rc shell and any payload alias retire before the last metadata account.
    funding: WorkspaceMetadataFunding,
}
impl Retention for FundedRetention {
    fn observe(&self, status: Status) {
        self.retained.observe(status);
    }
}

pub(super) enum DrafterRecovery {
    Ordinary(Recovery<DrafterRetention>),
    Funded(Recovery<FundedRetention>),
}
impl DrafterRecovery {
    pub(super) fn funding(&self) -> Option<&WorkspaceMetadataFunding> {
        match self {
            Self::Ordinary(_) => None,
            Self::Funded(recovery) => Some(&recovery.retention().funding),
        }
    }
    pub(super) fn seal(&mut self) {
        match self {
            Self::Ordinary(recovery) => recovery.seal(),
            Self::Funded(recovery) => recovery.seal(),
        }
    }
    pub(super) fn progress(&self) -> Status {
        match self {
            Self::Ordinary(recovery) => recovery.progress(),
            Self::Funded(recovery) => recovery.progress(),
        }
    }
    pub(super) fn wait(&self) -> Status {
        match self {
            Self::Ordinary(recovery) => recovery.wait(),
            Self::Funded(recovery) => recovery.wait(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("native drafter scope preparation failed: {0}")]
    Scope(#[source] safemlx::SubmissionScopeOwnerCause),
    #[error("native drafter work failed or remains unresolved")]
    Unresolved(#[source] Option<Error>),
    #[error(transparent)]
    Operation(Error),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    funding: WorkspaceMetadataFunding,
}

/// Exactly one returned failure shell was reserved before the operation began.
/// No !Send payload/native owner is placed in the public error; recovery keeps it.
pub(super) fn failure(cause: Cause, funding: WorkspaceMetadataFunding) -> Error {
    let kind = match &cause {
        Cause::Operation(error) | Cause::Unresolved(Some(error)) => error
            .retained_backend_failure_kind()
            .unwrap_or(eredu_core::BackendFailureKind::Other),
        _ => eredu_core::BackendFailureKind::Other,
    };
    // The enclosing speculative transaction has no complete rollback proof.
    Error::with_original_control_source(
        eredu_core::BackendFailure::new(kind, Failure { cause, funding }),
        false,
    )
}

fn control_bytes<T>() -> Option<usize> {
    let rc = std::alloc::Layout::new::<[Cell<usize>; 2]>()
        .extend(std::alloc::Layout::new::<RetainedSlot>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let controls = [
        rc,
        size_of::<RetainedSlot>(),
        size_of::<Rc<RetainedSlot>>(),
        size_of::<DrafterRetention>(),
        size_of::<FundedRetention>(),
        size_of::<DrafterRecovery>(),
        size_of::<DrafterOperation<'_>>(),
        size_of::<&SelectedSpeculativeRealization>(),
        size_of::<Result<DrafterOperation<'_>, Error>>(),
        size_of::<Result<T, Error>>(),
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<WorkspaceMetadataFunding>(),
        size_of::<eredu_core::BackendFailureKind>(),
        size_of::<eredu_core::BackendFailure>(),
        size_of::<Option<&WorkspaceMetadataFunding>>(),
        size_of::<Result<(), WorkspaceMetadataFundingError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<&mut DrafterExecution>(),
        size_of::<(&mut MlxDrafter, WorkspaceMetadataFunding)>(),
        size_of::<Result<&crate::composition::mlx::Executable, WorkingMemoryError>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
        usize::try_from(
            PreparedRecovery::<FundedRetention, WorkspaceMetadataFunding>::control_bytes()?,
        )
        .ok()?,
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}

impl MlxDrafter {
    /// Borrows only this payload's selected backend. A raw stream or equivalent
    /// device cannot supply the admitted source/worker/allocator identities.
    /// The caller must prepare a separate owned loan before mutable visitation.
    pub(crate) fn placement_environment(
        &self,
    ) -> Result<Option<crate::backend::OriginalCopyEnvironment<'_>>, crate::backend::OriginalCopyEnvironmentError> {
        self.validate_funded_availability()?;
        self.payload.backend.as_ref()
            .map(crate::backend::MlxBackend::original_copy_environment)
            .transpose()
    }

    /// Lends the actual independently materialized source before mutable work.
    /// It neither creates a recovery scope nor copies/reconstructs selection.
    /// The caller's paired-source preparation authenticates its epoch and role.
    pub(crate) fn autoregressive_source(
        &self,
    ) -> Result<&crate::composition::mlx::Executable, WorkingMemoryError> {
        self.validate_funded_availability()?;
        match &self.payload.execution {
            DrafterExecution::Autoregressive { model, .. } => Ok(model.executable()),
            DrafterExecution::Assistant(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }

    /// Uses the same mutation/poison/recovery worker with constructor metadata
    /// paid by the actual paired request. Call before entering any phase arena;
    /// child phases own their exact quotas and native completion independently.
    pub(crate) fn with_autoregressive_funded<T>(
        &mut self,
        funding: WorkspaceMetadataFunding,
        run: impl FnOnce(
            &mut crate::backend::MlxModel,
            &SelectedSpeculativeRealization,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.autoregressive_source()
            .map_err(Error::PrefillControl)?;
        self.with_funded(funding, |execution| match execution {
            DrafterExecution::Autoregressive {
                model, selected, ..
            } => run(model, selected),
            DrafterExecution::Assistant(_) => unreachable!("validated independent draft"),
        })
    }

    /// Same typed architecture visitor under the funded payload guard. This
    /// grants no model role or numerical work: an external phase must still
    /// quote and admit its exact source-backed invocation before executing it.
    pub(crate) fn visit_funded<W, T>(
        &mut self,
        funding: WorkspaceMetadataFunding,
        visitor: W,
    ) -> Result<T, Error>
    where
        W: eredu_architectures::MaterializedExternalAssistantVisitor<
                MlxAssistantPreparationVisitor,
                Output = Result<T, Error>,
            >,
    {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.validate_funded_availability()
            .map_err(Error::PrefillControl)?;
        if !matches!(&self.payload.execution, DrafterExecution::Assistant(_)) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        self.with_funded(funding, |execution| execution.visit(visitor))
    }

    /// Source-aware selected visitor. The exact selection/capture stay borrowed
    /// inside the same guarded assistant operation and are never cloned here.
    pub(crate) fn visit_selected_funded<W, T>(
        &mut self,
        funding: WorkspaceMetadataFunding,
        visitor: W,
    ) -> Result<T, Error>
    where
        W: eredu_architectures::SelectedExternalAssistantVisitor<
            MlxAssistantPreparationVisitor, Output = Result<T, Error>,
        >,
    {
        crate::backend::submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.validate_funded_availability().map_err(Error::PrefillControl)?;
        if !matches!(&self.payload.execution, DrafterExecution::Assistant(_)) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let controls = eredu_architectures::MaterializedExternalAssistantExecution::<
            MlxAssistantPreparationVisitor,
        >::selected_visit_control_bytes::<W>()
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
        funding.reserve_metadata(controls).map_err(Error::WorkspacePlanning)?;
        self.with_funded(funding, |execution| match execution {
            DrafterExecution::Assistant(assistant) => assistant.visit_selected(visitor),
            DrafterExecution::Autoregressive { .. } => unreachable!("validated external assistant"),
        })
    }

    fn validate_funded_availability(&self) -> Result<(), WorkingMemoryError> {
        if self.poisoned.get() {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if Rc::strong_count(&self.payload) != 1 || Rc::weak_count(&self.payload) != 0 {
            return Err(WorkingMemoryError::ReservedWorkActive);
        }
        Ok(())
    }

    fn with_funded<T>(
        &mut self,
        funding: WorkspaceMetadataFunding,
        run: impl FnOnce(&mut DrafterExecution) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let controls = control_bytes::<T>()
            .and_then(|bytes| bytes.checked_add(size_of_val(&run)))
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let retained = Rc::new(RefCell::new(None));
        let retention = FundedRetention {
            retained: DrafterRetention {
                _payload: Rc::clone(&retained),
                poisoned: Rc::clone(&self.poisoned),
                backend: None,
            },
            funding: funding.clone(),
        };
        let pending = PreparedRecovery::new(retention, funding.clone())
            .map_err(|error| failure(Cause::Scope(error.cause), funding.clone()))?;
        let recovery = pending
            .try_begin()
            .map_err(|error| failure(Cause::Scope(error.cause), funding.clone()))?;
        let mut operation = DrafterOperation {
            drafter: self,
            retained,
            recovery: DrafterRecovery::Funded(recovery),
            completed: false,
        };
        let result = run(&mut operation.payload().execution);
        operation.finish(result)
    }
}
