//! A saved native key/scalar component, independently funded before copying.
//!
//! This does not construct a continuation, copy source execution permission, or
//! establish complete native/facade snapshot coverage.

use super::*;
pub(in crate::composition::mlx::session) mod capture;
mod collector;
pub(in crate::composition::mlx::session) mod decoder;
pub(in crate::composition::mlx::session) mod pending_input;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    nn::workspace::ExistingArrayProjection,
    runtime::residency::storage::{RetainedStorage, StorageIdentity},
};
use collector::RootCollectorPlan;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_runtime::working_memory::{
    AdmittedWorkspaceCopy, BorrowedFundedSampler, FundedSamplerCopy, InferenceExecutionIdentity,
    InferenceStateRevision, RegisteredSamplingCopy, RegisteredWorkspaceCopy,
    RegisteredWorkspaceStorage, WorkingMemoryError, WorkspaceCopyCustody, WorkspaceCopyLimits,
};

fn memory(error: WorkingMemoryError) -> Error {
    Error::Other(Box::new(error))
}
fn mismatch() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}
fn unknown() -> Error {
    memory(WorkingMemoryError::UnknownBound)
}

/// Payload-free provenance. It neither retains the model nor issues a step.
#[derive(Clone)]
struct TextArraySource {
    session: Rc<Cell<bool>>,
    execution: InferenceExecutionIdentity,
    revision: InferenceStateRevision,
    parameter_epoch: u64,
    next_prediction: u64,
    // Checked absolute decoder frontier; unlike prediction/history length this
    // also preserves the position of an explicitly stateless saved branch.
    frontier: u64,
    // Closed immutable checkpoint shares the actual cumulative source, not its
    // issuance bank or native permission. It survives every independent copy.
    capture: Option<capture::SavedCaptureCheckpoint>,
    sampling_input: Option<eredu_runtime::working_memory::SamplingWorkspaceInputPlan>,
    // Accepted complete host-copy custody permits this paired source to be
    // revalidated after its original sequence bank has moved to the facade.
    host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl TextArraySource {
    fn inspect(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::generation::MlxTextSamplingState,
        pending: Option<&MlxTextToken>,
    ) -> Result<Self, Error> {
        Self::inspect_with_host(runtime, sampling, pending, None)
    }

    fn inspect_with_host(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::generation::MlxTextSamplingState,
        pending: Option<&MlxTextToken>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Self, Error> {
        Self::inspect_with_capture(runtime, sampling, pending, host, None)
    }

    fn inspect_with_capture(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::generation::MlxTextSamplingState,
        pending: Option<&MlxTextToken>,
        host: Option<&eredu_core::HostPreparationAuthority>,
        capture: Option<&capture::SavedCaptureCheckpoint>,
    ) -> Result<Self, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        if !sampling.sampler.is_funded() {
            return Err(unknown());
        }
        let quote = sampling.quote.as_ref().ok_or_else(unknown)?;
        // Only the complete paired worker has admitted and copied the facade's
        // sequence/controller state. Standalone native component copies retain
        // the stricter guard and cannot omit an original sequence or capture.
        if host.is_some() {
            quote
                .validate_paired_capture_copy_with_source(
                    capture.map(|capture| capture.checkpoint().source()),
                )
                .map_err(memory)?;
            if capture.is_some_and(|capture| {
                capture.checkpoint().next_prediction() != sampling.next_prediction
            }) {
                return Err(mismatch());
            }
        } else {
            quote.validate_capture_copy()?;
        }
        let local_prediction = quote.local_prediction(sampling.next_prediction)?;
        let frontier = quote.validate_frontier(runtime, sampling.next_prediction)?;
        let retained = session
            .payload
            .model
            .erased()
            .retained_inference_authority()?;
        if local_prediction == 0 {
            quote.validate_opening(&retained)?;
        } else {
            let admission = retained.admission().ok_or_else(mismatch)?;
            admission
                .request()
                .validate_same_request(quote.request())
                .map_err(memory)?;
            if admission.position() != frontier {
                return Err(memory(WorkingMemoryError::StateFrontierMismatch {
                    expected: frontier,
                    actual: admission.position(),
                }));
            }
        }
        if let Some(token) = pending {
            token.owner.ensure_healthy()?;
            if !token.owner.resources_releasable() {
                return Err(memory(WorkingMemoryError::ExecutionFenced));
            }
            retained
                .validate_revision(token.state_revision().ok_or_else(mismatch)?)
                .map_err(memory)?;
            token
                .step_receipt()
                .ok_or_else(mismatch)?
                .validate_completed_source(quote.request(), local_prediction)
                .map_err(memory)?;
            Self::validate_pending_descriptor(&token.value).map_err(|cause| match cause {
                decoder::cold_source::PendingDescriptorCause::Native(cause) => {
                    Error::Other(Box::new(cause))
                }
                decoder::cold_source::PendingDescriptorCause::Geometry => mismatch(),
            })?;
        }
        let mut parameter_epoch = sampling.parameter_epoch;
        session.validate_parameter_epoch(&mut parameter_epoch)?;
        Ok(Self {
            session: Rc::clone(&session.poison),
            execution: session
                .payload
                .model
                .erased()
                .inference_execution_identity()
                .clone(),
            revision: retained.revision().clone(),
            parameter_epoch: parameter_epoch.expect("validated parameter epoch"),
            next_prediction: sampling.next_prediction,
            frontier,
            capture: capture.cloned(),
            sampling_input: quote.saved_sampling_input(),
            host_preparation: host.cloned(),
        })
    }

    fn validate_pending_descriptor(
        array: &Array,
    ) -> Result<(), decoder::cold_source::PendingDescriptorCause> {
        let descriptor = array.try_descriptor()?;
        if descriptor.shape().iter().try_fold(1_u64, |n, &d| {
            u64::try_from(d).ok().and_then(|d| n.checked_mul(d))
        }) != Some(1)
            || !matches!(descriptor.facts().dtype(), Dtype::Int32 | Dtype::Uint32)
        {
            return Err(decoder::cold_source::PendingDescriptorCause::Geometry);
        }
        Ok(())
    }

    fn validate(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &super::super::generation::MlxTextSamplingState,
        pending: Option<&MlxTextToken>,
    ) -> Result<(), Error> {
        let actual = Self::inspect_with_capture(
            runtime,
            sampling,
            pending,
            self.host_preparation.as_ref(),
            self.capture.as_ref(),
        )?;
        if !Rc::ptr_eq(&self.session, &actual.session)
            || self.revision != actual.revision
            || self.parameter_epoch != actual.parameter_epoch
            || self.next_prediction != actual.next_prediction
            || self.frontier != actual.frontier
        {
            return Err(mismatch());
        }
        // The exact original request validates the execution identity; this
        // strong handle is only saved provenance and cannot create a request.
        sampling
            .quote
            .as_ref()
            .ok_or_else(mismatch)?
            .request()
            .validate(
                &self.execution,
                sampling.quote.as_ref().unwrap().request().geometry(),
            )
            .map_err(memory)
    }
}

/// Exact immutable slots from either the installed live run or independently
/// funded saved data. Saved data does not replay a live predecessor receipt.
enum TextArrayBinding<'a> {
    Live {
        sampling: &'a super::super::generation::MlxTextSamplingState,
        pending: Option<&'a MlxTextToken>,
        source: TextArraySource,
    },
    LivePrefill {
        sampling: &'a super::super::generation::MlxTextSamplingState,
        prompt: pending_input::PromptCopySource<'a>,
        source: TextArraySource,
    },
    Saved(&'a CopiedTextSampling),
}

impl TextArrayBinding<'_> {
    fn validate(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        match self {
            Self::Live {
                sampling,
                pending,
                source,
            } => source.validate(runtime, sampling, *pending),
            Self::LivePrefill {
                sampling,
                prompt,
                source,
            } => {
                prompt
                    .validate(sampling)
                    .map_err(|cause| Error::Other(Box::new(cause)))?;
                source.validate(runtime, sampling, None)
            }
            Self::Saved(saved) => {
                runtime.session().validate_backend(runtime.backend())?;
                if !runtime
                    .backend()
                    .memory_pool()
                    .same_domain(saved.arrays.custody.pool())
                {
                    return Err(mismatch());
                }
                // The immutable host proof and exact physical origins are
                // revalidated together by aggregate admission. The old live
                // frontier/receipt is intentionally not a copy requirement.
                Ok(())
            }
        }
    }

    fn key(&self) -> Option<&Array> {
        match self {
            Self::Live { sampling, .. } | Self::LivePrefill { sampling, .. } => {
                sampling.prng.as_ref().map(|key| key.as_array())
            }
            Self::Saved(saved) => saved.arrays.key.as_ref(),
        }
    }

    fn pending(&self) -> Option<&Array> {
        match self {
            Self::Live { pending, .. } => pending.map(|token| &token.value),
            Self::LivePrefill { prompt, .. } => prompt.tokens(),
            Self::Saved(saved) => saved.arrays.pending.as_ref(),
        }
    }

    fn pending_metadata(&self) -> Option<pending_input::SavedPendingInput> {
        match self {
            Self::Live { pending, .. } => pending.map(|_| pending_input::SavedPendingInput::Decode),
            Self::LivePrefill { prompt, .. } => Some(prompt.metadata()),
            Self::Saved(saved) => saved.arrays.pending_metadata.clone(),
        }
    }

    fn sampler(&self) -> &eredu_runtime::generation::ConfiguredTextSampler {
        match self {
            Self::Live { sampling, .. } | Self::LivePrefill { sampling, .. } => {
                sampling.sampler.as_sampler()
            }
            Self::Saved(saved) => saved.sampler.as_sampler(),
        }
    }

    fn borrow_funded(&self) -> Result<BorrowedFundedSampler<'_>, Error> {
        match self {
            Self::Live { sampling, .. } | Self::LivePrefill { sampling, .. } => {
                sampling.sampler.borrow_funded().map_err(memory)
            }
            Self::Saved(saved) => Ok(saved.sampler.borrow_funded()),
        }
    }

    fn metadata(&self) -> (f32, u64, Option<u64>) {
        match self {
            Self::Live {
                sampling, source, ..
            }
            | Self::LivePrefill {
                sampling, source, ..
            } => (
                sampling.temperature,
                sampling.next_prediction,
                // Capture already authenticated this exact epoch, even before
                // the first prefill binds the live sampler's optional field.
                Some(source.parameter_epoch),
            ),
            Self::Saved(saved) => (
                saved.temperature,
                saved.next_prediction,
                saved.parameter_epoch,
            ),
        }
    }

    fn provenance(&self) -> TextArraySource {
        match self {
            Self::Live { source, .. } | Self::LivePrefill { source, .. } => source.clone(),
            Self::Saved(saved) => saved.arrays.source.clone(),
        }
    }
}

/// The lease belongs to this exact destination session. Saved source data may
/// be copied into another session in the same pool, but that requires a fresh
/// preparation and its own destination lease. This identity owns no payload.
struct TextCopyDestination {
    session: Rc<Cell<bool>>,
    lease: SubmissionLease,
}

impl TextCopyDestination {
    fn validate(&self, runtime: &ModelRuntime<MlxBackend<'_>>) -> Result<(), Error> {
        if !Rc::ptr_eq(&self.session, &runtime.session().poison) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    }
}

/// The same session lease excludes mutation through cold preparation, admission
/// and native execution. Sources are borrowed, never replaced by scalar quotes.
pub(in crate::composition::mlx::session) struct PreparedTextArrayCopy<'a> {
    source: TextArrayBinding<'a>,
    proof: RegisteredWorkspaceCopy<StorageIdentity>,
    required_bytes: u64,
    collector: RootCollectorPlan,
    destination: TextCopyDestination,
}

impl<'a> PreparedTextArrayCopy<'a> {
    pub(in crate::composition::mlx::session) fn prepare(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        sampling: &'a super::super::generation::MlxTextSamplingState,
        pending: Option<&'a MlxTextToken>,
    ) -> Result<Self, Error> {
        runtime.session().validate_backend(runtime.backend())?;
        runtime
            .session()
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source = TextArraySource::inspect(runtime, sampling, pending)?;
        Self::prepare_source(
            runtime,
            TextArrayBinding::Live {
                sampling,
                pending,
                source,
            },
        )
    }

    pub(in crate::composition::mlx::session) fn prepare_saved(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        saved: &'a CopiedTextSampling,
    ) -> Result<Self, Error> {
        Self::prepare_source(runtime, TextArrayBinding::Saved(saved))
    }

    fn prepare_source(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: TextArrayBinding<'a>,
    ) -> Result<Self, Error> {
        let session = runtime.session();
        session.validate_backend(runtime.backend())?;
        session
            .authority
            .borrow()
            .require_idle()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        source.validate(runtime)?;
        let lease = session
            .authority
            .borrow_mut()
            .begin_submission()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let context = WorkspaceContext::new(
            session
                .payload
                .model
                .resident_workspace_mechanisms()
                .ok_or_else(unknown)?,
        );
        let projection_sources =
            usize::from(source.key().is_some()) + usize::from(source.pending().is_some());
        let mut projection =
            ExistingArrayProjection::with_source_count(&context, projection_sources)
                .map_err(|error| Error::Other(Box::new(error)))?;
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(projection_sources)
            .map_err(|error| Error::Other(Box::new(error)))?;
        for array in source.key().into_iter().chain(source.pending()) {
            inputs.push(
                projection
                    .project(array)
                    .map_err(|error| Error::Other(Box::new(error)))?,
            );
        }
        let collector = RootCollectorPlan::new(inputs.len(), inputs.len())?;
        let native = projection.into_storage();
        if !native.is_complete() {
            return Err(unknown());
        }
        let registered = RegisteredWorkspaceStorage::bind(
            runtime.backend().memory_pool(),
            &context,
            native
                .iter()
                .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
        )
        .map_err(memory)?;
        let plan =
            WorkspaceIsolatedCopyPlan::prepare(&context, registered.borrowed_storage(), &inputs)
                .map_err(|error| Error::Other(Box::new(error)))?;
        let copy_controls = collector.copy_control_bytes()?;
        let required_bytes = plan
            .incremental_bytes()
            .ok_or_else(unknown)?
            .checked_add(copy_controls)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let proof = RegisteredWorkspaceCopy::bind(plan, registered)
            .map_err(|error| Error::Other(Box::new(error)))?;
        // Borrowed source slots outlive this temporary physical witness inventory.
        Ok(Self {
            source,
            proof,
            required_bytes,
            collector,
            destination: TextCopyDestination {
                session: Rc::clone(&session.poison),
                lease,
            },
        })
    }

    pub(in crate::composition::mlx::session) fn required_bytes(&self) -> u64 {
        self.required_bytes
    }

    pub(in crate::composition::mlx::session) fn copy(
        self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        limits: WorkspaceCopyLimits,
    ) -> Result<CopiedTextArrays, Error> {
        self.destination.validate(runtime)?;
        self.source.validate(runtime)?;
        let account = runtime
            .backend()
            .memory_pool()
            .admit_workspace_copy(self.proof, self.collector.limits(limits)?)
            .map_err(|error| Error::Other(Box::new(error)))?;
        AdmittedTextArrayCopy {
            source: self.source,
            destination: self.destination,
            account,
            collector: self.collector,
        }
        .execute(runtime)
    }

    /// Diagnostic only. Actual admission reconstructs the closed aggregate plan
    /// from these same borrowed sampler and array slots.
    pub(in crate::composition::mlx::session) fn required_sampling_bytes(
        &self,
    ) -> Result<u64, Error> {
        let host = self
            .source
            .sampler()
            .prepare_copy()
            .map_err(|error| Error::Other(Box::new(error)))?
            .retained_bytes();
        host.checked_add(self.required_bytes)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))
    }

    pub(in crate::composition::mlx::session) fn copy_sampling(
        self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        limits: WorkspaceCopyLimits,
    ) -> Result<CopiedTextSampling, Error> {
        self.destination.validate(runtime)?;
        self.source.validate(runtime)?;
        let (temperature, next_prediction, parameter_epoch) = self.source.metadata();
        let plan = RegisteredSamplingCopy::prepare(self.source.borrow_funded()?, self.proof)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let (sampler, account) = runtime
            .backend()
            .memory_pool()
            .copy_sampling_components(plan, self.collector.limits(limits)?)
            .map_err(|error| Error::Other(Box::new(error)))?;
        // History is already copied and protected in this same account. The
        // native worker consumes its admitted scope without reserving again.
        let arrays = AdmittedTextArrayCopy {
            source: self.source,
            destination: self.destination,
            account,
            collector: self.collector,
        }
        .execute(runtime)?;
        Ok(CopiedTextSampling {
            sampler,
            arrays,
            temperature,
            next_prediction,
            parameter_epoch,
        })
    }
}

/// Independently saved controls, history, key and optional pending scalar. No
/// live quote, source receipt or sampling permit can be obtained from this data.
pub(in crate::composition::mlx::session) struct CopiedTextSampling {
    sampler: FundedSamplerCopy,
    arrays: CopiedTextArrays,
    temperature: f32,
    next_prediction: u64,
    parameter_epoch: Option<u64>,
}

impl CopiedTextSampling {
    pub(in crate::composition::mlx::session) fn sampling_state_facts(&self) -> eredu_core::SamplingStateFacts {
        eredu_core::SamplingStateFacts {
            temperature: self.temperature,
            requires_positive_temperature: matches!(self.sampler.as_sampler(), eredu_runtime::ConfiguredTextSampler::MirostatV2(_)),
            has_rng: self.arrays.key.is_some(),
        }
    }
    /// Saved numerical position, independent of absolute prediction ordinal.
    /// This is immutable provenance, never permission to start another request.
    pub(in crate::composition::mlx::session) fn frontier(&self) -> u64 {
        self.arrays.source.frontier
    }

    pub(in crate::composition::mlx::session) fn next_prediction(&self) -> u64 {
        self.next_prediction
    }
    pub(in crate::composition::mlx::session) fn input_tokens(
        &self,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        let pending = self.arrays.pending_metadata.as_ref()?;
        if !matches!(pending, pending_input::SavedPendingInput::Media(_)) {
            self.arrays.pending.as_ref()?;
        }
        pending.positions().checked_add(predictions - 1)
    }
    fn pending_decode(&self) -> Option<&Array> {
        matches!(
            self.arrays.pending_metadata.as_ref(),
            Some(pending_input::SavedPendingInput::Decode)
        )
        .then_some(self.arrays.pending.as_ref())
        .flatten()
    }
    pub(in crate::composition::mlx::session) fn bytes(&self) -> u64 {
        self.arrays.custody.bytes()
    }
}

struct AdmittedTextArrayCopy<'a> {
    source: TextArrayBinding<'a>,
    destination: TextCopyDestination,
    account: AdmittedWorkspaceCopy,
    collector: RootCollectorPlan,
}

/// An immutable saved numerical component, with no pending-input receipt or
/// generation quote. Arrays retire before the remaining destination envelope.
pub(in crate::composition::mlx::session) struct CopiedTextArrays {
    key: Option<Array>,
    pending: Option<Array>,
    pending_metadata: Option<pending_input::SavedPendingInput>,
    source: TextArraySource,
    custody: WorkspaceCopyCustody,
}

impl AdmittedTextArrayCopy<'_> {
    fn execute(
        self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ) -> Result<CopiedTextArrays, Error> {
        let (custody, scope) = self.account.into_parts();
        if let Err(error) = self
            .destination
            .validate(runtime)
            .and_then(|()| self.source.validate(runtime))
        {
            // This scope has not entered native recovery or performed work.
            let _ = scope.certify();
            return Err(error);
        }
        let roots = match self.collector.construct() {
            Ok(roots) => roots,
            Err(cause) => return Err(collector::allocation_failure(cause, custody, scope)),
        };
        let stream = runtime.backend().stream().clone();
        let key_source = self.source.key().cloned();
        let pending_source = self.source.pending().cloned();
        roots.borrow_mut().extend(key_source);
        roots.borrow_mut().extend(pending_source);
        let session = runtime.session_mut();
        let owner = SubmissionResources::with_purpose(
            self.destination.lease,
            Rc::clone(&session.poison),
            SubmissionPurpose::SavedArrayCopy(Rc::clone(&roots)),
        );
        let funding = text_funding::FundedWork::new(scope);
        owner.funding.replace(Some(funding.clone()));
        let recovery = owner.recovery()?;
        let operation = SessionOperation {
            session,
            owner: owner.clone(),
            recovery: Some(recovery),
            handed_off: false,
            token_validations: Default::default(),
        };
        // The worker is fixed: key, then pending scalar, each using the sealed
        // contiguous/deep-copy program. All partial roots stay in recovery.
        let copied = (|| {
            let one = |array: &Array| -> Result<Array, Error> {
                let copy = IsolatedArrayCopy::new(array).copy_retained(&stream, &roots)?;
                funding.retain(&copy);
                copy.evaluated()?;
                Ok(copy)
            };
            let key = self.source.key().map(one).transpose()?;
            #[cfg(all(
                test,
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))]
            tests::after_first_copy()?;
            let pending = self.source.pending().map(one).transpose()?;
            Ok::<_, Error>((key, pending))
        })();
        // A settled read-only copy failure does not poison an unchanged model.
        // Native recovery failure still follows the ordinary session fencing.
        let (key, pending) = finish_saved_copy(operation, copied)?;
        // Retire temporary/source handles after proven settlement and outside a
        // RefCell borrow. They are not part of destination storage publication.
        if let SubmissionPurpose::SavedArrayCopy(roots) = &owner.purpose {
            let retired = std::mem::take(&mut *roots.borrow_mut());
            drop(retired);
        }
        // Publication errors must reach this caller; never infer successful
        // completion from the ordinary model owner's best-effort publication.
        funding.publish(RetainedStorage::default())?;
        funding.certify()?;
        Ok(CopiedTextArrays {
            key,
            pending,
            pending_metadata: self.source.pending_metadata(),
            source: self.source.provenance(),
            custody,
        })
    }
}

// Keep the original worker error outside the nested result handed to the
// generic session engine. Both its early progress check and final completion
// can fail; those failures must fence recovery without replacing this cause.
fn finish_saved_copy<T, P: Probe>(
    operation: SessionOperation<'_, P>,
    copied: Result<T, Error>,
) -> Result<T, Error> {
    let (value, original_error) = match copied {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(error)),
    };
    let settled = operation
        .finish(Ok(value))
        .and_then(|(value, owner, recovery)| complete_model_operation(value, owner, recovery));
    match (settled, original_error) {
        (_, Some(error)) | (Err(error), None) => Err(error),
        (Ok(Some(value)), None) => Ok(value),
        (Ok(None), None) => unreachable!("successful copy retains its result"),
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
