//! Immutable sampling/input values owned by one admitted decoder/sampler copy.

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
    BorrowedFundedSampler, FundedSamplerCopy, InferenceExecutionIdentity, InferenceStateRevision,
    RegisteredSamplingCopy, RegisteredWorkspaceCopy, RegisteredWorkspaceStorage,
    WorkingMemoryError, WorkspaceCopyCustody, WorkspaceCopyLimits,
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
                    .memory_ledger()
                    .same_ledger(saved.arrays.custody.pool())
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
    pub(in crate::composition::mlx::session) fn sampling_state_facts(
        &self,
    ) -> eredu_core::SamplingStateFacts {
        eredu_core::SamplingStateFacts {
            temperature: self.temperature,
            requires_positive_temperature: matches!(
                self.sampler.as_sampler(),
                eredu_runtime::ConfiguredTextSampler::MirostatV2(_)
            ),
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
