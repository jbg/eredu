use super::*;

mod block_refinement;
mod memory_forecast;
pub(super) mod partition_capture;
pub(super) mod shared_speculation;
use eredu_core::TextGenerationBackend;
use eredu_core::{
    backend::{TextSubmissionScope, TextTransactionError, TextTransactionMechanisms},
    BackendFailure,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use super::recovery::{Recovery, ReleaseSubmission, ScopeRetention, Status, SubmissionResources};
use crate::backend::ordinary_retirement::{self, OrdinaryRetirement};

#[path = "parameters.rs"]
mod parameters;

#[doc(hidden)]
pub struct MlxRetainedSession {
    target: crate::backend::MlxPreparedTarget,
    distributed: Option<MlxDistributedSession>,
    _cache_cleanup: crate::backend::ordinary_retirement::TrimCacheOnRetirement,
}

struct OperationRecovery {
    failure: Rc<RefCell<Option<String>>>,
    owner: Rc<SubmissionResources>,
    recovery: Option<Recovery<ScopeRetention>>,
    handed_off: bool,
}

pub(super) fn restored_error_permits_recovery(status: Status, error: &Error) -> bool {
    !status.failed && !status.blocked && error.model_state_preserved()
}

pub(super) fn complete_model_operation<T>(
    value: T,
    owner: Rc<SubmissionResources>,
    recovery: Recovery<ScopeRetention>,
) -> Result<T, Error> {
    owner.request_release();
    let status = recovery.finish();
    if !status.settled || status.failed || status.blocked {
        owner.reject_unresolved();
        return Err(Error::ArchitectureModel(
            "native operation failed or returned without proven completion; session is poisoned"
                .into(),
        ));
    }
    Ok(value)
}

impl OperationRecovery {
    fn finish<T>(
        mut self,
        root: Rc<dyn std::any::Any>,
        result: Result<T, Error>,
        allow_preservation: bool,
    ) -> Result<(T, Rc<SubmissionResources>, Recovery<ScopeRetention>), Error> {
        self.owner.retain_until_release(root);
        let recovery = self.recovery.as_mut().expect("live operation scope");
        recovery.seal();
        let status = recovery.progress();
        if status.failed || status.blocked {
            if let Err(error) = result {
                record_first_failure(&self.failure, &error);
                return Err(error);
            }
            return Err(Error::ArchitectureModel(
                "native session execution failed or became unobservable; session is poisoned"
                    .into(),
            ));
        }
        let value = match result {
            Err(error) if allow_preservation && restored_error_permits_recovery(status, &error) => {
                // Only a direct model call can supply this evidence. A larger
                // speculative/cache operation may have other mutated state.
                // Pending cleanup keeps the restored executable and its lease
                // in recovery without permanently poisoning the session. The
                // lease still excludes reuse until every scope retires, and a
                // later failed/blocked observation still poisons its owner.
                self.handed_off = true;
                self.owner.request_release();
                self.recovery.take();
                return Err(error);
            }
            Err(error) => {
                record_first_failure(&self.failure, &error);
                return Err(error);
            }
            Ok(value) => value,
        };
        self.handed_off = true;
        Ok((value, Rc::clone(&self.owner), self.recovery.take().unwrap()))
    }

    fn abandon(self, root: Rc<dyn std::any::Any>) {
        self.owner.retain_until_release(root);
    }
}

impl Drop for OperationRecovery {
    fn drop(&mut self) {
        if self.handed_off {
            return;
        }
        self.owner.reject_unresolved();
        if let Some(recovery) = self.recovery.as_mut() {
            recovery.seal();
            let status = recovery.progress();
            if !status.settled || status.failed || status.blocked || std::thread::panicking() {
                self.owner.reject_unresolved();
            }
        }
        self.owner.request_release();
        // The preallocated node either releases the scope ticket or retains it
        // together with the executable and lease in nonblocking quarantine.
        self.recovery.take();
    }
}

#[doc(hidden)]
pub struct MlxSessionMechanisms {
    poison: Rc<Cell<bool>>,
    failure: Rc<RefCell<Option<String>>>,
    _retirement: ordinary_retirement::ReclaimOnDrop,
}

impl MlxSessionMechanisms {
    fn begin_recovery(
        &self,
        lease: eredu_core::SubmissionLease,
    ) -> Result<OperationRecovery, Error> {
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        let recovery = owner.recovery()?;
        Ok(OperationRecovery {
            failure: self.failure.clone(),
            owner,
            recovery: Some(recovery),
            handed_off: false,
        })
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        if self.poison.get() {
            let message = self.failure.borrow().as_ref().map_or_else(
                || "native session is fenced by unresolved or failed work".to_owned(),
                |cause| format!("native session is fenced after prior operation failure: {cause}"),
            );
            return Err(Error::ArchitectureModel(message));
        }
        Ok(())
    }

    fn with_shared_operation<T>(
        &self,
        lease: eredu_core::SubmissionLease,
        root: Rc<dyn std::any::Any>,
        operation: impl FnOnce() -> Result<T, Error>,
    ) -> Result<T, Error> {
        let owner = SubmissionResources::new(lease, Rc::clone(&self.poison));
        owner.retain_until_release(root);
        let mut recovery = owner.recovery()?;
        let _release = ReleaseSubmission(Rc::clone(&owner));
        let result = operation();
        if let Err(error) = &result {
            self.record_failure(error);
        }
        recovery.seal();
        // Successful host reads can leave completion bookkeeping pending. Wait
        // for that work before reusing the session; errors remain nonblocking.
        let status = if result.is_ok() {
            recovery.finish()
        } else {
            recovery.progress()
        };
        if !status.settled || status.failed || status.blocked {
            owner.reject_unresolved();
            return Err(result.err().unwrap_or_else(|| {
                Error::ArchitectureModel(
                    "native observation or sampling failed or remains unresolved".into(),
                )
            }));
        }
        if result.is_err() {
            owner.reject_unresolved();
        }
        result
    }

    fn record_failure(&self, error: &Error) {
        record_first_failure(&self.failure, error);
    }
}

impl eredu_core::session_authority::SessionRetirement for MlxSessionMechanisms {
    type Error = Error;

    fn refresh_native_session(&self) -> Result<(), Error> {
        super::recovery::reap();
        ordinary_retirement::reclaim();
        self.ensure_healthy()
    }

    fn submission_authority_error(error: eredu_core::SessionAuthorityError) -> Error {
        Error::Other(Box::new(error))
    }

    fn native_session_fenced(&self) -> bool {
        self.poison.get()
    }
}

fn record_first_failure(failure: &RefCell<Option<String>>, error: &Error) {
    let mut failure = failure.borrow_mut();
    if failure.is_none() {
        *failure = Some(error.to_string());
    }
}

/// Native submission state and token-validation roots for a portable invocation.
#[doc(hidden)]
pub struct MlxTextSubmissionScope {
    operation: OperationRecovery,
    validation: Option<TokenValidationScope>,
}

impl TextTransactionMechanisms for MlxSessionMechanisms {
    type Scores = Array;
    type Scope<'a> = MlxTextSubmissionScope;

    fn open_transaction(
        &self,
        lease: eredu_core::SubmissionLease,
    ) -> Result<Self::Scope<'_>, Error> {
        Ok(MlxTextSubmissionScope {
            operation: self.begin_recovery(lease)?,
            validation: None,
        })
    }

    fn transaction_error(error: TextTransactionError) -> Error {
        Error::ArchitectureModel(error.to_string())
    }
}

impl TextSubmissionScope<Array, Error> for MlxTextSubmissionScope {
    type Completion = MlxSessionCompletion;

    fn abandon(self, root: Rc<dyn std::any::Any>) {
        drop(self.validation);
        self.operation.abandon(root);
    }

    fn finish(
        self,
        root: Rc<dyn std::any::Any>,
        output: Result<Array, Error>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let (output, owner, recovery) = self.operation.finish(root, output, true)?;
        Ok(model_array_submission(
            output,
            self.validation.expect("token submission scope").finish(),
            owner,
            recovery,
        ))
    }

    fn validate_tokens(&mut self) -> Result<(), Error> {
        if let Some(validation) = self.validation.take() {
            let validation = validation.finish();
            safemlx::transforms::eval(validation.arrays())?;
            validation.validate_completed()?;
        }
        Ok(())
    }

    fn settle<T>(self, root: Rc<dyn std::any::Any>, result: Result<T, Error>) -> Result<T, Error> {
        drop(self.validation);
        let (value, owner, recovery) = self.operation.finish(root, result, false)?;
        complete_model_operation(value, owner, recovery)
    }
}

impl<Dispatch> eredu_core::input::PlainTextTokens<Array> for MlxBackend<'_, Dispatch> {
    fn materialize_tokens(&self, ids: &[u32]) -> Result<Array, Error> {
        super::recovery::detached(Vec::new(), || {
            Array::from(ids)
                .try_index_device(NewAxis, self.stream())
                .map_err(Into::into)
        })
    }

    fn token_identity(
        tokens: &Array,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::input::PreparedInputInspector::identity(&input::MlxInputInspector, tokens)
    }

    fn token_count(tokens: &Array) -> Option<usize> {
        let [_, sequence] = tokens.shape() else {
            return None;
        };
        usize::try_from(*sequence).ok()
    }

    fn split_tokens(
        &self,
        tokens: &Array,
        at: std::num::NonZeroUsize,
    ) -> Result<(Array, Array), Error> {
        let split = i32::try_from(at.get()).map_err(Error::before_model_mutation)?;
        super::recovery::detached(Vec::new(), || {
            let stream = self.stream();
            let prefix = tokens.try_index_device((.., ..split), stream)?;
            let suffix = tokens.try_index_device((.., split..), stream)?;
            Ok((prefix, suffix))
        })
    }
}

impl<'a, Dispatch> eredu_core::backend::TextSessionNative<MlxBackend<'a, Dispatch>>
    for MlxSessionMechanisms
{
    type Generation = eredu_core::backend::AdvancedTextGeneration;
    type Target = crate::backend::MlxPreparedTarget;
    type Communication = MlxDistributedSession;
    type LocalCollector<'s> = super::bounded_capture::NativeCapture<'s>;

    fn check_native_health(&self) -> Result<(), Error> {
        self.ensure_healthy()
    }

    fn validate_target(
        backend: &MlxBackend<'_, Dispatch>,
        target: &Self::Target,
    ) -> Result<(), Error> {
        backend.validate_prepared_target(target)
    }

    fn attach_native(
        target: Self::Target,
        communication: Option<Self::Communication>,
    ) -> Result<(MlxRetainedSession, MlxSessionMechanisms), Error> {
        ordinary_retirement::reclaim();
        Ok((
            MlxRetainedSession {
                target,
                distributed: communication,
                _cache_cleanup: crate::backend::ordinary_retirement::TrimCacheOnRetirement,
            },
            MlxSessionMechanisms {
                poison: Rc::new(Cell::new(false)),
                failure: Rc::new(RefCell::new(None)),
                _retirement: ordinary_retirement::ReclaimOnDrop,
            },
        ))
    }

    type TextValues = crate::composition::mlx::replicated_text::MlxTextValues;
    type Retained = MlxRetainedSession;
    type ParameterValue = MlxTensor;
    type Retired<R: 'static> = OrdinaryRetirement<R>;
    type Output = MlxModelOutput;
    type Completion = MlxSessionCompletion;
    type Scope<'s> = MlxTextSubmissionScope;

    fn retain_resources<R: 'static>(resources: R) -> Self::Retired<R> {
        // Native recovery can release the final reference while holding the
        // runtime lock. Retire model resources outside that lock.
        OrdinaryRetirement::new(resources)
    }

    fn begin(
        &self,
        lease: eredu_core::SubmissionLease,
        token_validation: bool,
    ) -> Result<Self::Scope<'_>, Error> {
        let validation = token_validation
            .then(TokenValidationScope::begin)
            .transpose()?;
        let operation = self.begin_recovery(lease)?;
        Ok(MlxTextSubmissionScope {
            operation,
            validation,
        })
    }

    fn decode_token_id(token: u32, backend: &MlxBackend<'a, Dispatch>) -> Result<Array, Error> {
        Array::from([token].as_slice())
            .try_index_device(NewAxis, backend.stream())
            .map_err(Into::into)
    }

    type SampledDecode = Array;

    fn decode_sample(value: Array, backend: &MlxBackend<'a, Dispatch>) -> Result<Array, Error> {
        value
            .try_index_device((.., NewAxis), backend.stream())
            .map_err(Into::into)
    }

    fn busy() -> Error {
        Error::ArchitectureModel(
            "text executable is still retained by an outstanding native scope".into(),
        )
    }

    fn context<'s>(backend: &'s MlxBackend<'a, Dispatch>) -> &'s Stream {
        backend.stream()
    }

    fn publish_scores(scores: Array, publish: bool) -> MlxModelOutput {
        MlxModelOutput::new(publish.then(|| MlxTensor::from_array(scores)))
    }

    fn validate_native_target(
        &self,
        retained: &MlxRetainedSession,
        backend: &MlxBackend<'a, Dispatch>,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        backend.validate_prepared_target(&retained.target)
    }

    fn check_transport_active(retained: &MlxRetainedSession) -> Result<(), BackendFailure> {
        if let Some(transport) = &retained.distributed {
            eredu_core::run_preparation::TextPreparationTransport::ensure_preparation_active(
                transport,
            )?;
        }
        Ok(())
    }

    type ControlTransport = MlxDistributedSession;

    fn control_transport(retained: &MlxRetainedSession) -> Option<&Self::ControlTransport> {
        retained.distributed.as_ref()
    }

    fn control_transport_identity(
        transport: &MlxDistributedSession,
    ) -> eredu_core::communication::CommunicationSessionIdentity {
        transport.session_identity()
    }

    fn parameter_control_supported(transport: &Self::ControlTransport) -> bool {
        transport.has_bounded_control()
    }

    fn preparation_admission(
        transport: &Self::ControlTransport,
        words: usize,
    ) -> Result<Option<eredu_core::run_preparation::TextPreparationAdmission>, Error> {
        transport.text_preparation_admission(words)
    }

    fn synchronize_queues(backend: &MlxBackend<'a, Dispatch>) -> Result<(), Error> {
        backend.weights_stream().synchronize()?;
        backend.synchronize()
    }

    fn output_scores(output: &Self::Output) -> Option<&MlxTensor> {
        output.logits()
    }

    fn read_observation_tensors(
        &self,
        lease: eredu_core::SubmissionLease,
        root: Rc<dyn std::any::Any>,
        backend: &MlxBackend<'a, Dispatch>,
        tensors: Vec<MlxTensor>,
    ) -> Result<Vec<TensorObservation>, Error> {
        let retained = Rc::new((root, tensors));
        self.with_shared_operation(lease, retained.clone(), || {
            retained
                .1
                .iter()
                .map(|tensor| observe_tensor(tensor, backend.stream()))
                .collect()
        })
    }
}

impl<'a, Dispatch> eredu_core::backend::TextSamplingNative<MlxBackend<'a, Dispatch>>
    for MlxSessionMechanisms
{
    type SamplingCompletion = MlxTextCompletion;

    type ScoreRow = MlxTensor;
    type Sample = MlxTensor;
    type SamplingScope = super::recovery::ResourceOperation;

    fn begin_sampling(completion: &MlxSessionCompletion) -> Result<Self::SamplingScope, Error> {
        super::recovery::ResourceOperation::begin(completion.owner())
    }

    fn last_score_row(
        output: &MlxModelOutput,
        stream: &Stream,
    ) -> Result<Option<MlxTensor>, Error> {
        super::generation::last_score_row(output, stream)
    }

    fn sampling_group_coordinates(
        group: &crate::backend::runtime::distributed::Group,
    ) -> (usize, usize) {
        (group.rank(), group.size())
    }

    fn exchange_sampled_token(
        token: Option<MlxTensor>,
        rank: usize,
        group: &crate::backend::runtime::distributed::Group,
        authority: &eredu_core::communication::CommunicationAuthority,
        stream: &Stream,
    ) -> Result<MlxTensor, Error> {
        super::generation::exchange_sampled_token(token, rank, group, authority, stream)
    }

    fn finish_sampling(
        scope: Self::SamplingScope,
        completion: MlxSessionCompletion,
        token: Result<MlxTensor, Error>,
        stream: &Stream,
    ) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
        super::generation::finish_sampling(scope, completion, token, stream)
    }

    fn completion_cancellation_mode() -> Option<eredu_core::CompletionCancellationMode> {
        Some(eredu_core::CompletionCancellationMode::QuarantineUntilComplete)
    }

    type Token = MlxTextToken;

    fn sampled_token_value(token: MlxTextToken) -> Array {
        token.value
    }
}

impl<'a, Dispatch> eredu_core::backend::DeviceTextSampling<MlxBackend<'a, Dispatch>>
    for MlxSessionMechanisms
{
    type Sampling = MlxSamplingBackend;
    type SamplingError = Exception;
}

impl<Dispatch> eredu_core::sampling::SeededRandomKey for MlxBackend<'_, Dispatch> {
    type Key = RandomState;

    fn seeded_random(seed: u64) -> Result<RandomState, Error> {
        super::recovery::detached(Vec::new(), || Ok(RandomState::with_seed(seed)?))
    }
}

impl<'a, Dispatch> TextGenerationBackend for MlxBackend<'a, Dispatch> {
    type InputTensor = Array;
    type Token = MlxTextToken;
    type TextCompletion = MlxTextCompletion;
}

fn model_array_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    owner: Rc<SubmissionResources>,
    recovery: Recovery<ScopeRetention>,
) -> Submission<Array, MlxSessionCompletion> {
    let mut retained = Vec::with_capacity(1 + token_validations.arrays().count());
    retained.push(output.clone());
    retained.extend(token_validations.arrays().cloned());
    Submission {
        output,
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model {
                token_validations,
                _retained: retained,
                owner,
                recovery: RefCell::new(Some(recovery)),
                observation: RefCell::new(super::output_completion::ModelObservation::Unobserved),
            },
        },
    }
}

#[cfg(any(feature = "image", feature = "audio"))]
impl<Dispatch> eredu_core::backend::TextMediaNative<MlxBackend<'_, Dispatch>>
    for MlxSessionMechanisms
{
    type ProcessorScope = crate::composition::mlx::processor::MlxProcessorScope;

    fn begin_processor(&self) -> Result<Self::ProcessorScope, Error> {
        Self::ProcessorScope::begin()
    }
}
