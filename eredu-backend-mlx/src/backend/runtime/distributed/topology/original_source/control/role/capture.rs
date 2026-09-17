//! The shared capture protocol's words, under the actual request control owner.
use super::*;
mod hook;
use crate::backend::distributed::MlxDistributedSession;
use crate::backend::runtime::distributed::completion::{
    MlxNeuralCommunicationCompletion, OriginalCommunicationU32Words, PreparedCommunicationU32Words,
};
use eredu_core::{
    capture::{CaptureError, CaptureUsage},
    consensus::{BoundedConsensusTransport, ConsensusTransport},
    BoundedCompletionWait, Submission,
};
use eredu_runtime::capture::partition::{
    PartitionCaptureBuffer, PartitionCaptureExchangeError, PartitionCaptureFrame,
    PartitionCaptureFrameKind, PartitionCaptureStorageError, PartitionCaptureTransport,
};
use eredu_runtime::{DistributedExecutionPhase, PartitionExecutionError};
use safemlx::{distributed::GroupWorkerOperation, Array, PreparedInputPlan};
use std::result::Result;

/// Explicit request owner, including the ordinary transport's exact authority.
/// Construction authenticates the loaded transport once; its retained request
/// already owns the communicator table and source. No model payload borrow is
/// held while the shared observer mutably executes that model, and no native
/// handle/table clone or ambient latest source replaces this same authority.
pub(crate) struct OriginalCaptureTransport {
    authority: PartitionCommunicationAuthority,
    attempt: u64,
    // Existing immutable setup allocation; remote descriptors are facts only.
    groups: std::sync::Arc<[CommunicationGroupDescriptor]>,
    owner: OriginalParallelControlOwner,
}
impl std::fmt::Debug for OriginalCaptureTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalCaptureTransport")
            .field("attempt", &self.attempt)
            .finish_non_exhaustive()
    }
}
#[derive(Debug, thiserror::Error)]
enum CaptureCause {
    #[error("capture transport differs from its exact request, frame or completed destination")]
    Identity,
    #[error(transparent)]
    Storage(PartitionCaptureStorageError),
    #[error(transparent)]
    Protocol(PartitionCaptureExchangeError),
    #[error(transparent)]
    Communication(PartitionExecutionError),
    #[error(transparent)]
    Backend(eredu_core::BackendFailure),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CaptureFailure {
    #[source]
    cause: CaptureCause,
    custody: Custody,
}
fn fail(cause: CaptureCause, custody: &Custody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::new(
            eredu_core::BackendFailureKind::Other,
            CaptureFailure {
                cause,
                custody: custody.clone(),
            },
        ),
        false,
    )
}
fn controls(custody: &Custody) -> Result<(), Error> {
    reserve(
        &custody.funding,
        &[
            size_of::<CaptureCause>(),
            size_of::<CaptureFailure>(),
            size_of::<Custody>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<Error>()
                .ok_or_else(overflow)?,
            size_of::<Result<(), Error>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<CaptureFailure>()
                .ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ],
    )
}
impl OriginalParallelControlOwner {
    pub(crate) fn capture_transport(
        &self,
        base: &MlxDistributedSession,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<OriginalCaptureTransport, Error> {
        let owner = self.owner();
        let c = &owner.custody;
        controls(c)?;
        reserve(
            &c.funding,
            &[
                size_of::<OriginalCaptureTransport>(),
                size_of::<Result<OriginalCaptureTransport, Error>>(),
                size_of::<(
                    &Self,
                    &MlxDistributedSession,
                    &eredu_runtime::working_memory::InferenceTextStep,
                )>(),
                size_of::<Result<(), eredu_runtime::working_memory::WorkingMemoryError>>(),
            ],
        )?;
        owner
            .execution
            .validate_same_request(step.request())
            .map_err(|cause| {
                control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding)
            })?;
        let source = owner.request.source.communication_source()?;
        source.validate()?;
        if !base.matches_capture_source(source.source(), source.authority)
            || source.authority.completion_policy().is_none()
            || source.world().retained_transport_stream().is_none()
            || owner.failed.get()
        {
            return Err(fail(CaptureCause::Identity, c));
        }
        Ok(OriginalCaptureTransport {
            authority: source.authority.clone(),
            attempt: step.attempt(),
            groups: source.actual.global_descriptors.clone(),
            owner: OriginalParallelControlOwner(self.0.clone()),
        })
    }
}
fn phase(kind: PartitionCaptureFrameKind) -> DistributedExecutionPhase {
    match kind {
        PartitionCaptureFrameKind::Coordination | PartitionCaptureFrameKind::Source => {
            DistributedExecutionPhase::ObservationCoordination
        }
        PartitionCaptureFrameKind::Preparation
        | PartitionCaptureFrameKind::Payload
        | PartitionCaptureFrameKind::Delivery
        | PartitionCaptureFrameKind::InterventionReceipt => DistributedExecutionPhase::ObservationDelivery,
    }
}
impl OriginalCaptureTransport {
    /// Reserve one producer's fixed failure destinations before lending the
    /// same retained request metadata account. This is no native grant.
    pub(crate) fn prepare_capture_metadata(&self) -> Result<&WorkspaceMetadataFunding, Error> {
        let custody = &self.owner.owner().custody;
        controls(custody)?;
        Ok(&custody.funding)
    }
    /// The preceding prepare_capture_metadata call paid this closed cause shell.
    /// It retains source/raw account custody even if frame construction refuses.
    pub(crate) fn retain_capture_preparation_error(&self, cause: Error) -> Error {
        fail(CaptureCause::Backend(cause.into_backend_failure()), &self.owner.owner().custody)
    }
    fn destination<T>(&self, capacity: usize) -> Result<PartitionCaptureBuffer<T>, Error> {
        let c = &self.owner.owner().custody;
        controls(c)?;
        PartitionCaptureBuffer::funded(capacity, &c.funding)
            .map_err(|cause| fail(CaptureCause::Storage(cause), c))
    }
    fn gather(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        wait: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, Error> {
        let owner = self.owner.owner();
        let c = &owner.custody;
        controls(c)?;
        reserve(
            &c.funding,
            &[
                size_of::<Invocation>(),
                size_of::<ParallelControlClaim>(),
                size_of::<
                    Result<ParallelControlClaim, eredu_runtime::working_memory::WorkingMemoryError>,
                >(),
                size_of::<PreparedInputPlan<'_>>(),
                size_of::<[usize; 1]>(),
                size_of::<[Array; 1]>(),
                size_of::<Result<PreparedInputPlan<'_>, safemlx::PreparedInputCause>>(),
                size_of::<Result<PartitionCaptureBuffer<u32>, Error>>(),
                size_of::<RefCell<Option<PartitionCaptureBuffer<u32>>>>(),
                size_of::<(&Self, &PartitionCaptureFrame<'_>, BoundedCompletionWait)>(),
                size_of::<
                    Result<Result<PartitionCaptureBuffer<u32>, Error>, eredu_core::BackendFailure>,
                >(),
            ],
        )?;
        // Attempt spending belongs to the live request, outside all saved state.
        let event = ParallelControlEvent::Phase(phase(frame.kind()));
        let claim = owner
            .request
            .cursor
            .try_borrow_mut()
            .map_err(|_| fail(CaptureCause::Identity, c))?
            .claim(event)
            .map_err(|cause| {
                control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding)
            })?;
        if owner.failed.get() || owner.running.replace(true) {
            owner.failed.set(true);
            return Err(fail(CaptureCause::Identity, c));
        }
        let _running = Running {
            running: &owner.running,
            failed: &owner.failed,
        };
        let result = (|| {
            let source = owner.request.source.communication_source()?;
            if frame.rank() != source.source().manifest().rank()
                || frame.participants() != source.source().manifest().world_size()
                || !self.authority.same_authority(source.authority)
                || !c.source.same_source(source.source())
                || source
                    .authority
                    .completion_policy()
                    .is_none_or(|policy| policy.bounded_wait() != wait)
            {
                return Err(fail(CaptureCause::Identity, c));
            }
            let inputs = owner
                .request
                .source
                .agreement_inputs()
                .ok_or_else(|| fail(CaptureCause::Identity, c))?;
            let shape = [frame.words().len()];
            let plan = inputs
                .runtime()
                .u32(frame.words(), &shape)
                .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
            let [input] = super::super::super::inputs::construct(&source, [plan])?;
            let capacity = capacity(&source, &input, frame.gathered_words(), owner)?;
            let destination = self.destination(frame.gathered_words())?;
            drop(source);
            let invocation = Invocation {
                input,
                claim,
                event,
                capacity,
                expected: frame.gathered_words(),
                destination: RefCell::new(Some(destination)),
                owner: OriginalParallelControlOwner(self.owner.0.clone()),
            };
            run_native_role(
                invocation,
                capacity,
                &owner.bank,
                &owner.controls,
                c,
                |value, observer| Ok(value.run(observer)),
            )
            .map_err(|cause| Error::with_original_control_source(cause, false))?
        })();
        if result.is_err() {
            owner.failed.set(true);
        }
        result
    }
}
struct Invocation {
    input: Array,
    claim: ParallelControlClaim,
    event: ParallelControlEvent,
    capacity: AgreementCapacity,
    expected: usize,
    destination: RefCell<Option<PartitionCaptureBuffer<u32>>>,
    owner: OriginalParallelControlOwner,
}
fn capacity(
    source: &OriginalCommunicationSource<'_>,
    input: &Array,
    expected: usize,
    owner: &Owner,
) -> Result<AgreementCapacity, Error> {
    let c = &owner.custody;
    controls(c)?;
    reserve(
        &c.funding,
        &[
            size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity, Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>, &Array, usize, &Owner)>(),
            size_of::<super::super::super::OriginalCommunicationCompletedOperation<'_>>(),
        ],
    )?;
    let operation = source.world_cpu_operation_storage(input, GroupWorkerOperation::Gather)?;
    if input.dtype() != safemlx::Dtype::Uint32
        || input.ndim() != 1
        || operation.native().constructor().output_geometry() != (1, expected)
        || operation.native().constructor().output_dtype() != safemlx::Dtype::Uint32
    {
        return Err(fail(CaptureCause::Identity, c));
    }
    let inputs = owner
        .request
        .source
        .agreement_inputs()
        .ok_or_else(|| fail(CaptureCause::Identity, c))?;
    let backing = operation.backing_storage(inputs.runtime())?.capacity();
    let operation = operation.with_completion()?;
    Ok(AgreementCapacity {
        graph: operation.graph_capacity(),
        records: operation.record_capacity(),
        backing,
    })
}
impl Invocation {
    fn run(&self, observer: &OriginalScopeObserver) -> Result<PartitionCaptureBuffer<u32>, Error> {
        let owner = self.owner.owner();
        let c = &owner.custody;
        controls(c)?;
        reserve(
            &c.funding,
            &[
                size_of::<Self>(),
                size_of::<PartitionExecutionError>(),
                size_of::<OriginalCommunicationU32Words>(),
                size_of::<PreparedCommunicationU32Words>(),
                size_of::<Result<PartitionCaptureBuffer<u32>, Error>>(),
                size_of::<Result<(), PartitionCaptureExchangeError>>(),
                size_of::<std::cell::RefMut<'_, Option<PartitionCaptureBuffer<u32>>>>(),
            ],
        )?;
        let source = owner.request.source.communication_source()?;
        if self.claim.identity() != owner.request.cursor.borrow().identity()
            || self.claim.event() != self.event
            || !self
                .capacity
                .covers(capacity(&source, &self.input, self.expected, owner)?)
        {
            return Err(fail(CaptureCause::Identity, c));
        }
        let operation = source
            .world_cpu_operation_storage(&self.input, GroupWorkerOperation::Gather)?
            .with_completion()?;
        let words = PreparedCommunicationU32Words::prepare_operation(&source, &operation)?;
        let ready = operation.prepare_resources(&source, None)?;
        let stream = source
            .world()
            .retained_transport_stream()
            .ok_or_else(|| fail(CaptureCause::Identity, c))?;
        let event_phase = match self.event {
            ParallelControlEvent::Phase(value) => value,
            _ => return Err(fail(CaptureCause::Identity, c)),
        };
        let submitted = operation
            .construct_accepted(&source, observer, stream)
            .and_then(|accepted| words.submit_accepted(accepted, ready));
        let (output, completion) = submitted.map_err(|cause| {
            source.authority.fence_protocol_failure(
                CommunicationOperation::AllGatherEven,
                event_phase,
                None,
            );
            cause
        })?;
        let output = source
            .authority
            .wait_with_error(
                Submission { output, completion },
                CommunicationOperation::AllGatherEven,
                event_phase,
                None,
                |cause| PartitionExecutionError::PreparedCommunication {
                    operation: CommunicationOperation::AllGatherEven,
                    phase: event_phase,
                    completion: true,
                    source: failure(Cause::Native(cause), &c.source, &c.funding)
                        .into_backend_failure(),
                },
            )
            .map_err(|cause| fail(CaptureCause::Communication(cause), c))?;
        let completed = output.resolve()?;
        if completed.as_slice().len() != self.expected {
            return Err(fail(CaptureCause::Identity, c));
        }
        let mut destination = self
            .destination
            .borrow_mut()
            .take()
            .ok_or_else(|| fail(CaptureCause::Identity, c))?;
        destination
            .extend_from_slice(completed.as_slice())
            .map_err(|cause| fail(CaptureCause::Protocol(cause), c))?;
        Ok(destination)
    }
}
impl ConsensusTransport for OriginalCaptureTransport {
    type Error = Error;
    fn participant_count(&self) -> usize {
        self.owner.owner().custody.source.manifest().world_size()
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Error> {
        controls(&self.owner.owner().custody)?;
        Err(fail(CaptureCause::Identity, &self.owner.owner().custody))
    }
}
impl BoundedConsensusTransport for OriginalCaptureTransport {
    type Completion = MlxNeuralCommunicationCompletion;
    type GatherOutput = OriginalCommunicationU32Words;
    fn submit_all_gather_words(
        &self,
        _: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Error> {
        controls(&self.owner.owner().custody)?;
        Err(fail(CaptureCause::Identity, &self.owner.owner().custody))
    }
    fn resolve_all_gather_words(&self, _: Self::GatherOutput) -> Result<Vec<u32>, Error> {
        controls(&self.owner.owner().custody)?;
        Err(fail(CaptureCause::Identity, &self.owner.owner().custody))
    }
}
impl PartitionCaptureTransport for OriginalCaptureTransport {
    fn capture_rank(&self) -> usize {
        self.owner.owner().custody.source.manifest().rank()
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        self.authority
            .completion_policy()
            .map(|policy| policy.bounded_wait())
            .ok_or(CaptureError::Overflow)
    }
    fn ensure_capture_active(&self) -> Result<(), eredu_core::BackendFailure> {
        let c = &self.owner.owner().custody;
        controls(c).map_err(Error::into_backend_failure)?;
        if self.owner.owner().failed.get() {
            return Err(fail(CaptureCause::Identity, c).into_backend_failure());
        }
        self.authority.ensure_active()
            .map_err(|cause| fail(CaptureCause::Communication(cause), c).into_backend_failure())
    }
    fn estimate_capture_gather(&self, words: usize) -> Result<CaptureUsage, CaptureError> {
        MlxDistributedSession::capture_gather_usage(self.participant_count(),words)
    }
    fn fail_capture_exchange(&self, error: &PartitionCaptureExchangeError) {
        let _ = error;
        self.owner.owner().failed.set(true);
        self.authority.fence_protocol_failure(
            CommunicationOperation::AllGatherEven,
            DistributedExecutionPhase::ObservationDelivery,
            None,
        );
    }
    fn capture_word_destination(
        &self,
        capacity: usize,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.destination(capacity)
            .map_err(|cause| PartitionCaptureExchangeError::Backend(cause.into_backend_failure()))
    }
    fn capture_byte_destination(
        &self,
        capacity: usize,
    ) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        self.destination(capacity)
            .map_err(|cause| PartitionCaptureExchangeError::Backend(cause.into_backend_failure()))
    }
    fn gather_capture_frame(
        &self,
        frame: &PartitionCaptureFrame<'_>,
        wait: BoundedCompletionWait,
    ) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        self.gather(frame, wait)
            .map_err(|cause| PartitionCaptureExchangeError::Backend(cause.into_backend_failure()))
    }
}
