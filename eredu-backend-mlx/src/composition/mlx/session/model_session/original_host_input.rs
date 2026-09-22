//! Ordinary uploads from an original host source; never a managed media grant.
use super::*;
mod construction;
mod native;
pub(super) use construction::prepare_model_input;
mod text_domain;
use crate::backend::submission_recovery::{self, Retention, Status};
use eredu_architectures::processor_execution::{
    lower_original_prepared_host_input, ProcessorExecutionError, ProcessorMechanisms,
};
use eredu_runtime::{
    input::host::HostInputPartView,
    working_memory::{MemoryLedger, OriginalPreparedHostInput, WorkingMemoryError},
    PreparedInputInspector,
};
pub(crate) use native::CompletedOriginalModelInput;
pub(super) use native::CompletedOriginalTextInput;
pub use native::{
    MlxOriginalPreparedModelInput, MlxOriginalPreparedNativeInput, MlxPreparedInputMaterializer,
    MlxPreparedModelInputBindError, MlxPreparedModelInputError, MlxPreparedModelInputPlan,
    MlxPreparedNativeInputError, MlxPreparedNativeInputPlan,
};
pub(super) use native::{RetiredMlxPreparedModelInputBindError, RetiredMlxPreparedModelInputError};
use std::{cell::RefCell, convert::Infallible, rc::Rc};
pub(super) use text_domain::validate_text_domain;

/// Ordinary upload failure. Admission rejection is inline and precedes owned
/// diagnostics/native work. Started failures retain original source and actual
/// ordinary custody in the closed neutral error source.
#[derive(Debug, thiserror::Error)]
pub enum MlxHostInputUploadError {
    /// No ordinary upload was started.
    #[error("prepared host upload admission: {0}")]
    Admission(#[source] WorkingMemoryError),
    /// Complete originally compiled body retained on a cold source rejection.
    #[error("original semantic source: {0}")]
    Semantics(#[source] eredu_runtime::working_memory::OriginalCompositeSemanticStorageError),
    /// Pre-work rejection retaining the genuine completed B source and A body.
    #[error("native prepared input admission: {cause}")]
    Native {
        /// Exact pre-work rejection.
        #[source]
        cause: WorkingMemoryError,
        /// The unchanged original completed native source.
        native_input: MlxOriginalPreparedNativeInput,
        /// Existing semantic body retained through rejection.
        semantics: Option<eredu_runtime::working_memory::OriginalCompositeSemanticStorageError>,
    },
    /// Actual started operation failed with its retained neutral source.
    #[error("prepared host upload failed: {0}")]
    Operation(#[source] BackendFailure),
}
impl MlxModelInput {
    /// Uploads this exact original host source using ordinary native ownership.
    /// The source must belong to the runtime's real pool. Successful arrays are
    /// independent copies and retain ordinary exclusion through their aliases;
    /// they cannot be adopted into original managed media admission afterward.
    /// This is upload custody only, under both finite and unlimited limits.
    /// For execution, use `MlxPreparedInputMaterializer::model_input_plan`, then
    /// materialize and bind the complete prepared model-input source.
    pub fn from_original_host_input(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: &OriginalPreparedHostInput,
    ) -> Result<Self, MlxHostInputUploadError> {
        upload(runtime.backend().memory_ledger(), source)
    }
    /// Consumes a cold original semantic body before ordinary native upload.
    /// The selected graph, processor and actual empty session are authenticated;
    /// this authenticates upload semantics and custody, not execution admission
    /// under finite or unlimited limits. Execution uses the complete source from
    /// `MlxPreparedInputMaterializer::model_input_plan`, materialized and bound.
    pub fn from_original_host_input_with_semantics(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    ) -> Result<Self, MlxHostInputUploadError> {
        upload_authenticated(runtime, original, None)
    }
    /// Uses the exact completed original leaves with the originally compiled
    /// semantic source. Prompt/maps/cache/handle copies remain ordinary; native
    /// leaf values are not uploaded, evaluated or adopted a second time.
    /// These leaf-only controls do not supply execution admission under finite
    /// or unlimited limits; a complete model-input plan supplies that producer.
    pub fn from_original_native_input_with_semantics(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        native_input: MlxOriginalPreparedNativeInput,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    ) -> Result<Self, MlxHostInputUploadError> {
        upload_authenticated(runtime, original, Some(native_input))
    }
}
fn upload_authenticated(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    native_input: Option<MlxOriginalPreparedNativeInput>,
) -> Result<MlxModelInput, MlxHostInputUploadError> {
    let source = original.source().clone();
    let boundary = if source
        .validate_pool(runtime.backend().memory_ledger())
        .is_err()
        || native_input.as_ref().is_some_and(|n| {
            !n.source().same_source(&source)
                || n.validate_pool(runtime.backend().memory_ledger()).is_err()
        }) {
        Some(WorkingMemoryError::IdentityMismatch)
    } else {
        let session = runtime.session();
        if session.validate_backend(runtime.backend()).is_err()
            || session.authority.borrow().require_idle().is_err()
            || !session
                .payload
                .model
                .inference_blueprint()
                .is_some_and(|b| original.matches_destination(b))
        {
            Some(WorkingMemoryError::UnknownBound)
        } else {
            None
        }
    };
    if let Some(error) = boundary {
        return Err(match native_input {
            Some(native_input) => MlxHostInputUploadError::Native {
                semantics: Some(original.reject_boundary(error.clone())),
                cause: error,
                native_input,
            },
            None => MlxHostInputUploadError::Semantics(original.reject_boundary(error)),
        });
    }
    upload_with_native(
        runtime.backend().memory_ledger(),
        &source,
        Some((runtime, original)),
        native_input,
    )
}

fn upload(
    pool: &MemoryLedger,
    source: &OriginalPreparedHostInput,
) -> Result<MlxModelInput, MlxHostInputUploadError> {
    upload_with_semantics(pool, source, None)
}
fn upload_with_semantics(
    pool: &MemoryLedger,
    source: &OriginalPreparedHostInput,
    original: Option<(
        &ModelRuntime<MlxBackend<'_>>,
        eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    )>,
) -> Result<MlxModelInput, MlxHostInputUploadError> {
    upload_with_native(pool, source, original, None)
}
fn upload_with_native(
    pool: &MemoryLedger,
    source: &OriginalPreparedHostInput,
    original: Option<(
        &ModelRuntime<MlxBackend<'_>>,
        eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    )>,
    mut native_input: Option<MlxOriginalPreparedNativeInput>,
) -> Result<MlxModelInput, MlxHostInputUploadError> {
    let mut original = original;
    let reject = |original: &mut Option<(
        &ModelRuntime<MlxBackend<'_>>,
        eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
    )>,
                  native_input: &mut Option<MlxOriginalPreparedNativeInput>,
                  error: WorkingMemoryError| {
        if let Some(native_input) = native_input.take() {
            return MlxHostInputUploadError::Native {
                semantics: original
                    .take()
                    .map(|(_, a)| a.reject_boundary(error.clone())),
                cause: error,
                native_input,
            };
        }
        match original.take() {
            Some((_, original)) => {
                MlxHostInputUploadError::Semantics(original.reject_boundary(error))
            }
            None => MlxHostInputUploadError::Admission(error),
        }
    };
    source
        .validate_pool(pool)
        .map_err(|e| reject(&mut original, &mut native_input, e))?;
    // All host-only native-dimension checks precede the ordinary owner and any
    // arrays/shape vectors; no source Vec or callback is constructed here.
    for part in source.parts() {
        for value in std::iter::once(part.payload()).chain(part.metadata().map(|(_, v)| v)) {
            if i32::try_from(value.shape.len()).is_err()
                || value.shape.iter().any(|d| i32::try_from(*d).is_err())
            {
                return Err(reject(
                    &mut original,
                    &mut native_input,
                    WorkingMemoryError::Overflow,
                ));
            }
        }
    }
    let owner = NativeMemoryOwner::acquire_typed(pool)
        .map_err(|e| reject(&mut original, &mut native_input, e))?;
    let roots = Rc::new(RefCell::new(Vec::with_capacity(source.slot_count())));
    let retained = UploadRetention {
        _roots: Rc::clone(&roots),
        _source: source.clone(),
        _memory: owner.clone(),
    };
    let result = submission_recovery::detached_retained(retained, || {
        // Real ordinary custody precedes any lazy state identity or boxed error.
        let semantics = match original {
            Some((runtime, original)) => {
                let session = runtime.session();
                Some(
                    session
                        .payload
                        .model
                        .erased()
                        .bind_original_media_semantics(
                            original,
                            session
                                .payload
                                .model
                                .inference_blueprint()
                                .expect("validated blueprint"),
                            source,
                        )
                        .map_err(|error| Error::Other(Box::new(error)))?,
                )
            }
            None => None,
        };
        let mut mechanisms = UploadMechanisms {
            owner: &owner,
            roots: &roots,
            native_input: native_input.as_ref(),
            next_slot: 0,
        };
        let lowered = lower_original_prepared_host_input::<_, Infallible>(source, &mut mechanisms)
            .map_err(|e| match e {
                ProcessorExecutionError::Mechanism(e) => e,
                ProcessorExecutionError::Prepared(e) => Error::Other(Box::new(e)),
                ProcessorExecutionError::Plan(e) => Error::ArchitectureModel(e),
                ProcessorExecutionError::Text(e) => match e {},
            })?;
        // Existing source and every already returned native slot stay in the
        // recovery owner. Copy source values are no longer needed after this
        // exact completion; partial errors/unwind preserve pending roots too.
        if let Some(native) = native_input.as_ref() {
            if mechanisms.next_slot != native.slot_count() {
                return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
            }
            // Closed B construction copied every actual slot synchronously.
            // No native graph/evaluation/receipt exists to settle here.
        } else {
            let all = roots.borrow().clone();
            safemlx::transforms::async_eval_with_event(all.iter())?.synchronize()?;
        }
        drop(mechanisms);
        let prepared = lowered.prepared();
        let fingerprint = source
            .content_digest()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let identity = eredu_runtime::SharedPreparedInputCacheIdentity::new(
            prepared
                .cache_identity(fingerprint)
                .map_err(|e| Error::ArchitectureModel(e.to_string()))?,
        );
        owner.retain_metadata(&eredu_runtime::SharedHostMetadata::Input(identity.clone()))?;
        // Descriptor/handle copies are ordinary upload controls. They preserve
        // every ordered slot; no native backing is copied a second time.
        let parts = prepared
            .parts()
            .iter()
            .map(|part| {
                let value = part.payload().value().as_array().clone();
                let payload = match part.payload() {
                    eredu_runtime::PreparedInputPayload::TokenIds(_) => {
                        input::InputPayload::TokenIds(value)
                    }
                    eredu_runtime::PreparedInputPayload::Tensor(_) => {
                        input::InputPayload::Tensor(value)
                    }
                    eredu_runtime::PreparedInputPayload::Embeddings(_) => {
                        input::InputPayload::Embeddings(value)
                    }
                    _ => {
                        return Err(Error::ArchitectureModel(
                            "unrecognized original upload payload".into(),
                        ))
                    }
                };
                input::InputPart::new_with_extents(
                    part.modality(),
                    payload,
                    part.metadata()
                        .iter()
                        .map(|(key, value)| (*key, value.as_array().clone())),
                    part.extents().iter().copied(),
                )
                .map_err(|error| Error::Other(Box::new(error)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut prompt = MlxModelInput::from(
            input::ModelInput::with_shared_cache_identity(&parts, &identity)
                .with_memory_owner(&owner),
        );
        if let Some(semantics) = semantics {
            // The completion above covers all slots, including future-span-only
            // metadata and payloads. Only this path can publish a compiled packet.
            prompt.original_media = Some(input::OriginalMediaPacket::completed(
                lowered,
                semantics,
                owner.clone(),
                native_input,
            ));
        }
        Ok(prompt)
    });
    result.map_err(|cause| {
        MlxHostInputUploadError::Operation(BackendFailure::from_error(UploadFailure {
            cause,
            _source: source.clone(),
            _memory: owner,
        }))
    })
}
struct UploadMechanisms<'a> {
    owner: &'a NativeMemoryOwner,
    roots: &'a RefCell<Vec<Array>>,
    native_input: Option<&'a MlxOriginalPreparedNativeInput>,
    next_slot: usize,
}
impl UploadMechanisms<'_> {
    fn tensor(
        &mut self,
        shape: &[usize],
        values: eredu_runtime::input::host::HostTensorValues<'_>,
        make: impl FnOnce(&[i32]) -> Result<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        if let Some(source) = self.native_input {
            let value = source.clone_slot(self.next_slot, values, shape)?;
            self.roots.borrow_mut().push(value.clone());
            self.owner.retain_array(&value)?;
            self.next_slot += 1;
            return Ok(value);
        }
        let shape = shape
            .iter()
            .map(|d| {
                i32::try_from(*d).map_err(|_| Error::Other(Box::new(WorkingMemoryError::Overflow)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = make(&shape)?;
        self.roots.borrow_mut().push(value.clone());
        value.evaluated()?;
        self.owner.retain_array(&value)?;
        #[cfg(all(test, unix))]
        tests::after_slot(self.roots.borrow().len(), &value)?;
        Ok(value)
    }
}
impl PreparedInputInspector<crate::MlxTensor> for UploadMechanisms<'_> {
    fn identity(
        &self,
        value: &crate::MlxTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        input::MlxTensorInputInspector.identity(value)
    }
    fn i32_values(
        &self,
        value: &crate::MlxTensor,
    ) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        input::MlxTensorInputInspector.i32_values(value)
    }
    fn bool_values(
        &self,
        value: &crate::MlxTensor,
    ) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        input::MlxTensorInputInspector.bool_values(value)
    }
}
impl ProcessorMechanisms for UploadMechanisms<'_> {
    type Tensor = crate::MlxTensor;
    type Error = Error;
    fn tensor_u32(&mut self, v: &[u32], shape: &[usize]) -> Result<crate::MlxTensor, Error> {
        self.tensor(
            shape,
            eredu_runtime::input::host::HostTensorValues::U32(v),
            |shape| Array::try_from_slice(v, shape),
        )
        .map(crate::MlxTensor::from_array)
    }
    fn tensor_i32(&mut self, v: &[i32], shape: &[usize]) -> Result<crate::MlxTensor, Error> {
        self.tensor(
            shape,
            eredu_runtime::input::host::HostTensorValues::I32(v),
            |shape| Array::try_from_slice(v, shape),
        )
        .map(crate::MlxTensor::from_array)
    }
    fn tensor_f32(&mut self, v: &[f32], shape: &[usize]) -> Result<crate::MlxTensor, Error> {
        self.tensor(
            shape,
            eredu_runtime::input::host::HostTensorValues::F32(v),
            |shape| Array::try_from_slice(v, shape),
        )
        .map(crate::MlxTensor::from_array)
    }
}
struct UploadRetention {
    _roots: Rc<RefCell<Vec<Array>>>,
    _source: OriginalPreparedHostInput,
    _memory: NativeMemoryOwner,
}
impl Retention for UploadRetention {
    fn observe(&self, _: Status) {}
}
#[derive(Debug, thiserror::Error)]
#[error("ordinary original-host-source upload failed: {cause}")]
struct UploadFailure {
    #[source]
    cause: Error,
    _source: OriginalPreparedHostInput,
    _memory: NativeMemoryOwner,
}
#[cfg(all(test, unix))]
mod tests;
