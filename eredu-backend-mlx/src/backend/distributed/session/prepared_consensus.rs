//! Scheduler and parameter exchanges share the original finite-word source and completion.
use super::*;
use crate::backend::runtime::distributed::{
    completion::{CompletedCommunicationU32Words, OriginalCommunicationU32Words},
    topology::{original_source::PreparationCompletion, OriginalCommunicationOwner},
};
use eredu_core::{
    consensus::{BoundedConsensusTransport, ConsensusTransport},
    BackendFailure, BackendFailureKind, BoundedCompletion, BoundedCompletionOutcome,
    BoundedCompletionWait, Completion, SharedBackendFailure,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    RetainedCommunicationSource,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

#[derive(Debug, thiserror::Error)]
#[error("original consensus: {cause}")]
struct Failure {
    #[source]
    cause: Error,
    _source: RetainedCommunicationSource,
    _funding: HostMetadataFunding,
}
#[derive(Debug)]
struct FailureState {
    first: RefCell<Option<SharedBackendFailure>>,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
/// One prepaid first-cause cell shared with exact outstanding completions.
#[derive(Clone, Debug)]
struct Failures(Option<Rc<FailureState>>);
impl Drop for Failures {
    fn drop(&mut self) {
        if let Some(state) = self.0.take() {
            drop(Rc::into_inner(state));
        }
    }
}
impl Failures {
    fn state(&self) -> &FailureState {
        self.0.as_deref().expect("live consensus failure source")
    }
    fn record(&self, cause: Error) -> Error {
        if let Some(first) = self.state().first.borrow().as_ref() {
            return Error::retained_original(first.retained(), false);
        }
        // The one possible shared diagnostic allocation was included in setup,
        // before native work. A later failure never replaces the first cause.
        let first = SharedBackendFailure::new(
            BackendFailureKind::Other,
            Failure {
                cause,
                _source: self.state().source.clone(),
                _funding: self.state().funding.clone(),
            },
        );
        *self.state().first.borrow_mut() = Some(first.retained());
        Error::retained_original(first, false)
    }
    fn take(&self) -> Option<BackendFailure> {
        self.state()
            .first
            .borrow()
            .as_ref()
            .map(|first| first.retained().into_failure())
    }
    fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
}
/// Exact loaded communicator/source owner, independently admitted before any
/// scheduler branch or parameter exchange. It supplies no frame execution authority.
pub(crate) struct PreparedConsensusTransport {
    owner: OriginalCommunicationOwner,
    pool: MemoryLedger,
    execution: InferenceExecutionIdentity,
    capacity: eredu_core::MemoryLimits,
    failures: Failures,
    funding: HostMetadataFunding,
}
/// Output stays on its actual paid Host destination through neutral validation.
pub(crate) struct PreparedConsensusOutput {
    words: OriginalCommunicationU32Words,
    expected: usize,
    failures: Failures,
    funding: HostMetadataFunding,
}
#[derive(Debug)]
pub(crate) struct PreparedConsensusCompletion {
    inner: PreparationCompletion,
    failures: Failures,
}
impl PreparedConsensusTransport {
    pub(crate) fn prepare(
        session: &MlxDistributedSession,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, Error> {
        let funding = pool
            .prepare_workspace_metadata(execution, capacity.clone())
            .map_err(Error::WorkspacePlanning)?;
        funding
            .reserve_metadata(
                usize::try_from(capacity.backing_bytes().map_err(|_| overflow())?)
                    .map_err(|_| overflow())?,
            )
            .map_err(Error::WorkspacePlanning)?;
        reserve(
            &funding,
            &[
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<Failures>(),
                size_of::<FailureState>(),
                size_of::<Failure>(),
                size_of::<Error>(),
                size_of::<Option<SharedBackendFailure>>(),
                size_of::<Option<BackendFailure>>(),
                size_of::<std::cell::Ref<'_, Option<SharedBackendFailure>>>(),
                size_of::<std::cell::RefMut<'_, Option<SharedBackendFailure>>>(),
                size_of::<(
                    &MlxDistributedSession,
                    &MemoryLedger,
                    &InferenceExecutionIdentity,
                    eredu_core::MemoryLimits,
                )>(),
                WorkspaceContext::metadata_rc_bytes::<FailureState>().ok_or_else(overflow)?,
                SharedBackendFailure::control_bytes::<Failure>().ok_or_else(overflow)?,
            ],
        )?;
        let owner = session
            .original_communication_owner(&session.manifest, session.native_world(), &funding)?
            .pin_registered_buffers(pool)?;
        let failures = Failures(Some(Rc::new(FailureState {
            first: RefCell::new(None),
            source: owner.source().clone(),
            funding: funding.clone(),
        })));
        let inspected = (|| {
            let source = owner.borrow()?;
            if source.world().retained_transport_stream().is_none()
                || source.world().has_original_parallel()
                || source.world().has_original_control()
                || source.world().original_control_request().is_some()
            {
                return Err(identity());
            }
            source.world_persistent()?;
            Ok(())
        })();
        inspected.map_err(|cause| failures.record(cause))?;
        Ok(Self {
            owner,
            pool: pool.clone(),
            execution: execution.clone(),
            capacity,
            failures,
            funding,
        })
    }
    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    /// Returns an allocation-free alias; the cell stays retained while any
    /// completion can report, so cleanup cannot replace the first native cause.
    pub(crate) fn take_failure(&self) -> Option<BackendFailure> {
        self.failures.take()
    }
    fn submit(
        &self,
        local: &[u32],
    ) -> Result<Submission<PreparedConsensusOutput, PreparedConsensusCompletion>, Error> {
        let funding = self
            .pool
            .prepare_workspace_metadata(&self.execution, self.capacity.clone())
            .map_err(Error::WorkspacePlanning)?;
        reserve(
            &funding,
            &[
                size_of::<PreparedConsensusOutput>(),
                size_of::<PreparedConsensusCompletion>(),
                size_of::<Submission<PreparedConsensusOutput, PreparedConsensusCompletion>>(),
                size_of::<
                    Result<Submission<PreparedConsensusOutput, PreparedConsensusCompletion>, Error>,
                >(),
                size_of::<(&Self, &[u32])>(),
                size_of::<Failures>(),
                size_of::<usize>(),
            ],
        )?;
        let expected = local
            .len()
            .checked_mul(self.participant_count())
            .ok_or_else(overflow)?;
        let source = self.owner.borrow_funded(&funding)?;
        let frame = source.prepare_preparation_frame(local, &self.pool, None)?;
        let operation = frame.prepare(&source)?;
        let stream = source
            .world()
            .retained_transport_stream()
            .ok_or_else(identity)?;
        let Submission { output, completion } =
            operation.start(&source, local, stream, &self.pool, None)?;
        Ok(Submission {
            output: PreparedConsensusOutput {
                words: output,
                expected,
                failures: self.failures.clone(),
                funding,
            },
            completion: PreparedConsensusCompletion {
                inner: completion,
                failures: self.failures.clone(),
            },
        })
    }
}
impl ConsensusTransport for PreparedConsensusTransport {
    type Error = Error;
    fn participant_count(&self) -> usize {
        self.owner.source().manifest().world_size()
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Error> {
        Err(self.failures.record(Error::InvalidOperation(
            "original consensus requires bounded borrowed resolution",
        )))
    }
}
impl BoundedConsensusTransport for PreparedConsensusTransport {
    type Completion = PreparedConsensusCompletion;
    type GatherOutput = PreparedConsensusOutput;
    fn metadata_funding(&self) -> Option<&eredu_core::HostMetadataFunding> {
        Some(&self.funding)
    }
    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Error> {
        self.submit(local)
            .map_err(|cause| self.failures.record(cause))
    }
    fn resolve_all_gather_words(&self, _: Self::GatherOutput) -> Result<Vec<u32>, Error> {
        Err(self.failures.record(Error::InvalidOperation(
            "original consensus requires borrowed Host custody",
        )))
    }
    fn with_resolved_all_gather_words<T, E, F>(
        &self,
        output: Self::GatherOutput,
        validate: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(&[u32]) -> Result<T, E>,
    {
        let result = (|| {
            reserve(
                &output.funding,
                &[
                    size_of::<Self::GatherOutput>(),
                    size_of::<CompletedCommunicationU32Words>(),
                    size_of::<T>(),
                    size_of::<E>(),
                    size_of::<F>(),
                    size_of::<Result<T, E>>(),
                    size_of::<Result<Result<T, E>, Error>>(),
                    size_of::<(&Self, &[u32])>(),
                ],
            )?;
            if !self.failures.same(&output.failures) {
                return Err(identity());
            }
            let completed = output.words.resolve()?;
            if completed.as_slice().len() != output.expected {
                return Err(identity());
            }
            Ok(validate(completed.as_slice()))
        })();
        result.map_err(|cause| self.failures.record(cause))
    }
}
impl Completion for PreparedConsensusCompletion {
    type Error = Error;
    fn resources_releasable(&self) -> bool {
        self.inner.resources_releasable()
    }
    fn is_complete(&self) -> Result<bool, Error> {
        self.inner
            .is_complete()
            .map_err(|cause| self.failures.record(cause))
    }
    fn wait(&self) -> Result<(), Error> {
        self.inner
            .wait()
            .map_err(|cause| self.failures.record(cause))
    }
}
impl BoundedCompletion for PreparedConsensusCompletion {
    fn wait_bounded(
        self,
        policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Error> {
        let Self { inner, failures } = self;
        inner
            .wait_bounded(policy)
            .map_err(|cause| failures.record(cause))
    }
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn identity() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)
}
fn reserve(funding: &HostMetadataFunding, parts: &[usize]) -> Result<(), Error> {
    funding
        .reserve_metadata(
            parts
                .iter()
                .copied()
                .try_fold(size_of_val(parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)
}
