//! Ordinary prepared-input carrier; no source constructor is enabled under R.
use super::*;
use eredu_core::{
    BackendFailure, HostPreparationAuthority, PreparedControlInput, PreparedControlInputBackend,
    PreparedControlInputError as Rejection, PreparedPromptAttribution, SharedPromptAttribution,
};

/// A one-use ordinary input, authenticated against its exact selected session.
/// Native prompt handles and their recovery ownership retire before host custody.
pub struct MlxControlInput {
    prompt: MlxModelInput,
    attribution: SharedPromptAttribution,
    binding: crate::composition::mlx::replicated_text::PreparedControlBinding,
    memory: NativeMemoryOwner,
}
impl PreparedControlInput for MlxControlInput {
    type Prompt = MlxModelInput;
    fn attribution(&self) -> &PreparedPromptAttribution {
        self.attribution.attribution()
    }
    fn shared_attribution(&self) -> &SharedPromptAttribution {
        &self.attribution
    }
}

#[derive(Debug, thiserror::Error)]
#[error("ordinary prepared control input: {cause}")]
struct PreparationFailure {
    #[source]
    cause: Error,
    // BackendFailure destroys its enclosing Box before this source value.
    _host: HostPreparationAuthority,
}
fn failed(cause: Error, host: HostPreparationAuthority) -> BackendFailure {
    BackendFailure::from_error(PreparationFailure { cause, _host: host })
}

impl PreparedControlInputBackend for MlxBackend<'_> {
    type ControlInput = MlxControlInput;

    fn prepare_control_input(
        runtime: &ModelRuntime<Self>,
        prompt: MlxModelInput,
    ) -> Result<MlxControlInput, BackendFailure> {
        // Preserve original authority. Never strip a quote/request into this route.
        if prompt.quote.is_some()
            || prompt
                .inference_request
                .as_ref()
                .is_some_and(|r| r.memory_reservation().is_some())
        {
            return Err(Rejection::UnknownBound.into_backend_failure());
        }
        let memory = NativeMemoryOwner::acquire(runtime.backend().memory_pool())
            .map_err(BackendFailure::from_error)?;
        let host = HostPreparationAuthority::retain(
            memory
                .unquoted_lease()
                .map_err(BackendFailure::from_error)?,
        );
        let local = (|| {
            let session = runtime.session();
            session.validate_backend(runtime.backend())?;
            session
                .authority
                .borrow()
                .require_idle()
                .map_err(|e| Error::Other(Box::new(e)))?;
            let binding = session.payload.model.erased().prepared_control_binding()?;
            // Metadata/token reads may evaluate native arrays. Existing detached
            // retention owns all ordinary completion, failure and unwind work.
            let attribution = super::super::recovery::detached_retained(memory.clone(), || {
                prompt.with_borrowed(|input| {
                    session
                        .payload
                        .model
                        .erased()
                        .prepared_control_attribution(input)
                })
            })?;
            session
                .payload
                .model
                .erased()
                .validate_prepared_control_binding(&binding)?;
            if attribution.opening_position != binding.frontier {
                return Err(Error::Other(Box::new(Rejection::SourceMismatch)));
            }
            Ok((binding, attribution))
        })();
        let (binding, attribution) = local.map_err(|e| failed(e, host.clone()))?;
        let mut prompt = prompt.with_memory_owner(memory.clone());
        if prompt.cache_identity().is_none() {
            prompt = prompt
                .with_semantic_content_fingerprint(attribution.semantic_content_identity.clone())
                .map_err(|e| failed(e, host.clone()))?;
        }
        let attribution = SharedPromptAttribution::from_prepared(attribution, host.clone())
            .map_err(|e| failed(Error::Other(Box::new(e)), host.clone()))?;
        Ok(MlxControlInput {
            prompt,
            attribution,
            binding,
            memory,
        })
    }

    fn bind_control_input_capture(
        runtime: &ModelRuntime<Self>,
        mut input: MlxControlInput,
        config: eredu_core::TextGenerationConfig,
        source: eredu_core::capture::SharedCapturePlan,
    ) -> Result<MlxControlInput, BackendFailure> {
        let policy = config.inference_policy();
        policy
            .validate(config.sampling().max_new_tokens)
            .map_err(BackendFailure::from_error)?;
        if input.prompt.has_original_input_custody()
            || policy.managed_memory_capacity_bytes.is_some()
        {
            return Err(Rejection::UnknownBound.into_backend_failure());
        }
        let host = HostPreparationAuthority::retain(
            input
                .memory
                .unquoted_lease()
                .map_err(BackendFailure::from_error)?,
        );
        let result = (|| {
            let session = runtime.session();
            session.validate_backend(runtime.backend())?;
            session
                .authority
                .borrow()
                .require_idle()
                .map_err(|e| Error::Other(Box::new(e)))?;
            let model = session.payload.model.erased();
            model.validate_prepared_control_binding(&input.binding)?;
            if session
                .loaded_partition_capture()
                .map_err(Error::observation)?
                .is_some()
            {
                return Err(Error::Other(Box::new(
                    Rejection::InstrumentationUnavailable,
                )));
            }
            let paths = model
                .shared_observation_paths()
                .ok_or_else(|| Error::Other(Box::new(Rejection::InstrumentationUnavailable)))?;
            model.validate_prepared_observation_paths(paths)?;
            let selection = paths
                .prepare_media_capture_selection(&source)
                .map_err(|e| Error::Other(Box::new(e)))?;
            let attribution = input.attribution.attribution();
            let maximum = config
                .sampling()
                .max_new_tokens
                .ok_or_else(|| Error::Other(Box::new(Rejection::InvalidAttribution)))?;
            let selected_chunk = config
                .inference_policy()
                .prefill_chunk_positions
                .or(input.prompt.prefill_chunk_positions);
            let geometry = eredu_core::InferenceGeometry {
                batch_size: attribution.batch,
                cached_positions: attribution.opening_position,
                input_positions: attribution.decoder_positions,
                max_output_tokens: maximum as u64,
                prefill_chunk_positions: selected_chunk
                    .map_or(attribution.decoder_positions, |n| {
                        n.get().min(attribution.decoder_positions)
                    }),
                output: selection.physical_output(eredu_core::OutputDemand::LastPosition),
            };
            let capture = eredu_runtime::capture::OrdinaryPrefillCapture::new(selection, geometry)
                .map_err(Error::observation)?;
            input
                .prompt
                .with_borrowed(|view| model.validate_media_capture_input(view, geometry))?;
            // No accepted request may be silently re-quoted or have its geometry changed.
            let request = match input.prompt.inference_request.as_ref() {
                Some(request) if request.geometry() == geometry => {
                    request
                        .validate(model.inference_execution_identity(), geometry)
                        .map_err(|error| Error::Other(Box::new(error)))?;
                    request.clone()
                }
                Some(_) => return Err(Error::Other(Box::new(Rejection::SourceMismatch))),
                None => eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
                    model.inference_execution_identity(),
                    geometry,
                )
                .map_err(|e| Error::Other(Box::new(e)))?,
            };
            Ok((capture, request))
        })();
        let (capture, request) = result.map_err(|e| failed(e, host.clone()))?;
        capture
            .source()
            .retain_host_preparation(&host)
            .map_err(|e| failed(Error::observation(e), host))?;
        input.prompt = input.prompt.with_inference_request(request);
        input.prompt.prepared_capture = Some(capture);
        Ok(input)
    }

    fn consume_control_input(
        runtime: &ModelRuntime<Self>,
        input: MlxControlInput,
    ) -> Result<(MlxModelInput, SharedPromptAttribution), BackendFailure> {
        let host = HostPreparationAuthority::retain(
            input
                .memory
                .unquoted_lease()
                .map_err(BackendFailure::from_error)?,
        );
        let valid = (|| {
            let session = runtime.session();
            session.validate_backend(runtime.backend())?;
            session
                .authority
                .borrow()
                .require_idle()
                .map_err(|e| Error::Other(Box::new(e)))?;
            session
                .payload
                .model
                .erased()
                .validate_prepared_control_binding(&input.binding)
        })();
        valid.map_err(|e| failed(e, host))?;
        let MlxControlInput {
            mut prompt,
            attribution,
            binding,
            memory,
        } = input;
        prompt.controlled_attribution = Some(attribution.clone());
        drop(binding);
        drop(memory);
        Ok((prompt, attribution))
    }
}

/// The core installation hook supplies the actual pending prompt after its
/// source/session consume check. No separate sampler or preparation copy exists.
pub(super) fn install_capture(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &mut MlxTextGenerationState,
    prompt: &MlxModelInput,
    source: eredu_core::capture::SharedCapturePlan,
) -> Result<(), BackendFailure> {
    if prompt.has_original_input_custody() {
        return Err(Rejection::UnknownBound.into_backend_failure());
    }
    let memory = prompt
        .memory_owner
        .as_ref()
        .ok_or_else(|| Rejection::SourceMismatch.into_backend_failure())?;
    let host = HostPreparationAuthority::retain(
        memory
            .unquoted_lease()
            .map_err(BackendFailure::from_error)?,
    );
    let result = (|| {
        text_step::validate_instrumentation(state).map_err(Error::observation)?;
        if state.capture.is_some() || state.funded_capture.is_some() {
            return Err(Error::Other(Box::new(Rejection::SourceMismatch)));
        }
        let binding = prompt
            .prepared_capture
            .as_ref()
            .ok_or_else(|| Error::Other(Box::new(Rejection::SourceMismatch)))?;
        let request = prompt
            .inference_request
            .as_ref()
            .ok_or_else(|| Error::Other(Box::new(Rejection::SourceMismatch)))?;
        let model = runtime.session().payload.model.erased();
        let paths = model
            .shared_observation_paths()
            .ok_or_else(|| Error::Other(Box::new(Rejection::SourceMismatch)))?;
        model.validate_prepared_observation_paths(paths)?;
        request
            .validate(model.inference_execution_identity(), binding.geometry())
            .map_err(|error| Error::Other(Box::new(error)))?;
        binding
            .validate(&source, paths, request.geometry())
            .map_err(Error::observation)?;
        let attribution = prompt
            .controlled_attribution
            .as_ref()
            .ok_or_else(|| Error::Other(Box::new(Rejection::SourceMismatch)))?;
        if attribution.attribution().opening_position != binding.geometry().cached_positions {
            return Err(Error::Other(Box::new(Rejection::SourceMismatch)));
        }
        model.validate_text_frontier(binding.geometry().cached_positions)?;
        eredu_runtime::capture::CaptureSession::with_ordinary_prefill(binding.clone(), &host)
            .map_err(Error::observation)
    })();
    let capture = result.map_err(|e| failed(e, host))?;
    state.capture = Some(capture);
    Ok(())
}

pub(super) fn rebind_error_bytes() -> u64 {
    // The new ordinary lease Arc and closed returned error shell. Arbitrary
    // native diagnostic payloads keep the existing snapshot error classification.
    let header = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>();
    let (lease, _) = header
        .extend(std::alloc::Layout::new::<
            eredu_runtime::working_memory::WorkingMemoryUnquotedLease,
        >())
        .expect("fixed host control");
    (lease.pad_to_align().size()
        + std::mem::size_of::<PreparationFailure>()
        + std::mem::size_of::<Rejection>()) as u64
}
impl MlxModelInput {
    /// Called only by the shared post-copy snapshot hook for restore and fork.
    /// The saved checkpoint and actual destination run are borrowed through the
    /// whole validation; no equal plan text or digest establishes this relation.
    pub(in crate::composition::mlx::session) fn rebind_ordinary_capture(
        &mut self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        saved: Option<&eredu_runtime::capture::CaptureCheckpoint>,
        capture: Option<&eredu_runtime::capture::CaptureSession>,
    ) -> Result<(), Error> {
        let expected_attachment = saved
            .and_then(|s| s.ordinary_prefill_source_binding())
            .is_some()
            || capture
                .and_then(|s| s.ordinary_prefill_source_binding())
                .is_some();
        if self.prepared_capture.is_none() && !expected_attachment {
            // Unrelated original/legacy token inputs retain their exact path.
            return Ok(());
        }
        if self.has_original_input_custody() {
            // No accepted constructor combines original custody with this
            // ordinary attachment. Reject the contradictory private call inline;
            // it cannot acquire a new ordinary lease or allocate an unowned box.
            return Err(Error::PreparedCaptureRebind(Rejection::UnknownBound));
        }
        let memory = self
            .memory_owner
            .as_ref()
            .ok_or(Error::PreparedCaptureRebind(Rejection::SourceMismatch))?;
        let host = HostPreparationAuthority::retain(memory.unquoted_lease()?);
        let result = (|| {
            let mismatch = || Error::Other(Box::new(Rejection::SourceMismatch));
            let original = self.prepared_capture.as_ref().ok_or_else(mismatch)?;
            let saved = saved.ok_or_else(mismatch)?;
            let run = capture.ok_or_else(mismatch)?;
            saved
                .validate_pending_capture_destination(run)
                .map_err(Error::observation)?;
            let source = saved
                .ordinary_prefill_source_binding()
                .ok_or_else(mismatch)?;
            let destination = capture
                .and_then(|s| s.ordinary_prefill_source_binding())
                .ok_or_else(mismatch)?;
            let current = self.inference_request.as_ref().ok_or_else(mismatch)?;
            let model = runtime.session().payload.model.erased();
            let paths = model.shared_observation_paths().ok_or_else(mismatch)?;
            current
                .validate(model.inference_execution_identity(), original.geometry())
                .map_err(|error| Error::Other(Box::new(error)))?;
            original
                .validate(source.source(), paths, current.geometry())
                .map_err(Error::observation)?;
            model.validate_prepared_observation_paths(paths)?;
            let mut expected = original.geometry();
            expected.max_output_tokens = destination.geometry().max_output_tokens;
            if expected != destination.geometry()
                || !capture
                    .and_then(|c| c.shared_plan_source())
                    .is_some_and(|s| s.same_storage(destination.source()))
            {
                return Err(mismatch());
            }
            destination
                .validate(destination.source(), paths, expected)
                .map_err(Error::observation)?;
            let request = eredu_runtime::working_memory::InferenceRequest::without_memory_budget(
                model.inference_execution_identity(),
                expected,
            )
            .map_err(|e| Error::Other(Box::new(e)))?;
            Ok((request, destination.clone()))
        })();
        let (request, capture) =
            result.map_err(|error| Error::StorageSource(failed(error, host)))?;
        self.inference_request = Some(request);
        self.prepared_capture = Some(capture);
        Ok(())
    }
}

#[cfg(test)]
mod rebind_error_tests;
