//! Existing speculative capture delivery with original source and host custody.
use super::*;
use crate::composition::mlx::speculative::autoregressive::MlxAutoregressiveMechanisms;
use eredu_core::{
    HostPreparationAuthority, SpeculativeBuffer,
    capture::*,
    speculative::{SpeculativeCaptureRole, SpeculativePredictionCapture},
};
use eredu_runtime::{
    speculative::autoregressive::AutoregressiveMechanisms,
    working_memory::{
        OriginalCaptureSource, OriginalSpeculativeSourceIdentity, WorkingMemoryError,
    },
};
use std::{
    alloc::Layout,
    cell::RefCell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

struct Owner {
    source: OriginalCaptureSource,
    identity: OriginalSpeculativeSourceIdentity,
    records: SpeculativeBuffer<SpeculativePredictionCapture>,
    _host: HostPreparationAuthority,
}
pub(super) struct OriginalCapture(Option<Rc<RefCell<Owner>>>);
impl Clone for OriginalCapture {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live capture").clone()))
    }
}
impl Drop for OriginalCapture {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
fn wrong() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
impl OriginalCapture {
    pub(super) fn prepare(
        plan: &AdmittedCapturePlan,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<Self, Error> {
        let (sources, environment) = context.original_numerical().ok_or_else(wrong)?;
        sources.validate_environment(environment)?;
        let frames = [
            size_of::<Self>(),
            size_of::<Owner>(),
            size_of::<Option<Owner>>(),
            size_of::<std::cell::RefMut<'_, Owner>>(),
            size_of::<SpeculativeBuffer<SpeculativePredictionCapture>>(),
            PreparedCapturePlanCopy::inspection_control_bytes().ok_or_else(wrong)?,
            size_of::<Result<PreparedCapturePlanCopy<'_>, CapturePlanCopyError>>(),
            size_of::<
                Result<
                    OriginalCaptureSource,
                    eredu_runtime::working_memory::OriginalCaptureSourceError,
                >,
            >(),
            size_of::<Result<Self, Error>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<Error>().ok_or_else(wrong)?,
        ];
        let shell = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<RefCell<Owner>>())
            .map_err(|_| wrong())?
            .0
            .pad_to_align()
            .size();
        let bytes = frames.into_iter().try_fold(
            shell.checked_add(size_of_val(&frames)).ok_or_else(wrong)?,
            usize::checked_add,
        );
        let host = MlxAutoregressiveMechanisms::driver_host_metadata(bytes, context)?;
        // Native qualification remains explicit; copying an admitted declaration
        // never certifies an unavailable transformation or activation callback.
        if plan.request().batch != 1
            || plan.request().prompt_tokens != 1
            || plan.invocation_bounds().is_some()
            || plan.plan().selections.iter().any(|s| {
                s.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH
                    || !matches!(
                        s.transform,
                        CaptureTransform::FullTensor
                            | CaptureTransform::Slice
                            | CaptureTransform::Preview { .. }
                            | CaptureTransform::TokenScores { .. }
                            | CaptureTransform::TopCandidates { .. }
                            | CaptureTransform::Summary
                            | CaptureTransform::Histogram { .. }
                    )
            })
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        let plan = PreparedCapturePlanCopy::inspect(plan)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let source = environment
            .pool()
            .compile_capture_source(plan)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        Ok(Self(Some(Rc::new(RefCell::new(Owner {
            source,
            identity: sources.request().source_identity(),
            records: Default::default(),
            _host: host,
        })))))
    }
    pub(super) fn preflight_intervention(
        &self,
        plan: &eredu_runtime::working_memory::OriginalInterventionSource,
        scratch: &mut eredu_runtime::intervention::StaticInterventionPreflight,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<(), Error> {
        let (sources, environment) = context.original_numerical().ok_or_else(wrong)?;
        sources.validate_environment(environment)?;
        let owner = self
            .0
            .as_ref()
            .expect("live capture")
            .try_borrow()
            .map_err(|_| {
                sources.retain_startup_error(WorkingMemoryError::AccountConstructionBusy)
            })?;
        if !owner.identity.belongs_to_request(sources.request()) {
            return Err(sources.retain_error(wrong()));
        }
        owner
            .source
            .validate_pool(environment.pool())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        plan.validate_pool(environment.pool())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        scratch
            .run_with_source(
                owner.source.plan().admission(),
                plan.plan(),
                &crate::composition::mlx::session::intervention::NativeInterventionEstimator,
            )
            .map_err(|cause| sources.retain_startup_error(cause))
    }
    pub(super) fn preflight_control_bytes() -> Option<usize> {
        Some(
            size_of::<std::cell::Ref<'_, Owner>>()
                + size_of::<(
                    &Self,
                    &eredu_runtime::working_memory::OriginalInterventionSource,
                    &mut eredu_runtime::intervention::StaticInterventionPreflight,
                    SpeculativeExecutionStreams<'_>,
                )>(),
        )
    }
    pub(super) fn process<S: SpeculativeSampler<MlxSamplingBackend>>(
        &self,
        policy: &S,
        logits: &numerical::OriginalNumericalValue,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<numerical::OriginalNumericalValue, Error> {
        self.process_with_interventions(
            policy,
            logits,
            temperature,
            history,
            placement,
            context,
            None,
        )
    }
    pub(super) fn process_with_interventions<S: SpeculativeSampler<MlxSamplingBackend>>(
        &self,
        policy: &S,
        logits: &numerical::OriginalNumericalValue,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: SpeculativeExecutionStreams<'_>,
        interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
    ) -> Result<numerical::OriginalNumericalValue, Error> {
        let (sources, environment) = context.original_numerical_for(placement).ok_or_else(wrong)?;
        sources.validate_environment(environment)?;
        let frames = [
            size_of::<std::cell::RefMut<'_, Owner>>(),
            size_of::<SpeculativePredictionCapture>(),
            size_of::<Option<SharedCapturedStep>>(),
            size_of::<(
                Option<SharedCapturedStep>,
                Result<numerical::OriginalNumericalValue, Error>,
            )>(),
            size_of::<Result<numerical::OriginalNumericalValue, Error>>(),
            size_of::<SpeculativeCaptureRole>(),
            size_of::<SpeculativeBuffer<SpeculativePredictionCapture>>(),
            size_of::<Result<(), eredu_core::GenerationError>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add);
        let _host = MlxAutoregressiveMechanisms::driver_host_metadata(bytes, context)?;
        let mut owner = self
            .0
            .as_ref()
            .expect("live capture")
            .try_borrow_mut()
            .map_err(|_| {
                sources.retain_startup_error(WorkingMemoryError::AccountConstructionBusy)
            })?;
        if !owner.identity.belongs_to_request(sources.request()) {
            return Err(sources.retain_error(wrong()));
        }
        let role = match placement {
            SamplingPlacement::Target => SpeculativeCaptureRole::Target,
            SamplingPlacement::Draft => SpeculativeCaptureRole::Draft,
            _ => return Err(sources.retain_error(wrong())),
        };
        // Prepare actual delivery capacity before the native invocation. Growing
        // chooses and pays a fresh exact destination; existing ordinary buffers
        // cannot be adopted. Moves preserve each record's independent custody.
        if owner.records.len() == owner.records.capacity() {
            let capacity = if owner.records.is_empty() {
                1
            } else {
                owner.records.capacity().checked_mul(2).ok_or_else(wrong)?
            };
            let mut next = MlxAutoregressiveMechanisms::driver_buffer(capacity, context)?;
            next.try_extend(owner.records.drain())
                .map_err(|cause| sources.retain_startup_error(cause))?;
            owner.records = next;
        }
        let capture =
            numerical::CaptureSource::original(&owner.source, sources, environment.pool())?;
        let capture = match interventions {
            Some(source) => capture.with_interventions(source),
            None => capture,
        };
        let mut failed_capture = None;
        let result = numerical::process_policy_with_capture_at(
            policy,
            logits,
            temperature,
            history,
            placement,
            context,
            capture,
            &mut failed_capture,
        );
        let (frame, result) = match result {
            Ok((value, frame)) => (Some(frame), Ok(value)),
            Err(cause) => (failed_capture, Err(cause)),
        };
        if let Some(frame) = frame {
            let recorded = owner.records.try_push(SpeculativePredictionCapture {
                role,
                position: history.len() as u64,
                capture: CapturedStepDelivery::Shared(frame),
            });
            if let Err(cause) = recorded {
                // Capacity was prepared before entry. Preserve any primary native
                // error even if a future destination change rejects this handoff.
                return match result {
                    Err(first) => Err(first),
                    Ok(_) => Err(sources.retain_startup_error(cause)),
                };
            }
        }
        result
    }
    pub(super) fn take(&self) -> SpeculativeBuffer<SpeculativePredictionCapture> {
        std::mem::take(&mut self.0.as_ref().expect("live capture").borrow_mut().records)
    }
}
