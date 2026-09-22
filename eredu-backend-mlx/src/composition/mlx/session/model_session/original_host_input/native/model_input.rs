//! Full B source adaptation; no encoder or request admission is installed here.
use super::*;
use crate::composition::mlx::replicated_text::CompletedMediaBindingError;
use eredu_architectures::{
    media_plan::{
        BoundPreparedMediaSemantics, OriginalPreparedMediaSemantics, PreparedMediaEncoderTablePlan,
    },
    processor_execution::OriginalHostLowering,
};
use eredu_runtime::{
    input::{
        PreparedModelInputSource, PreparedModelInputSourceError, PreparedModelInputSourcePlan,
    },
    working_memory::OriginalCompositeSemanticStorageError,
};

type Body = PreparedModelInputSource<crate::MlxTensor, Array, NativeCells>;
mod prediction;
mod text;
pub(in crate::composition::mlx::session::model_session) use text::CompletedOriginalTextInput;
type HostPlan<'a> =
    PreparedModelInputSourcePlan<'a, crate::MlxTensor, Array, NativeCells, MlxNativeInputCause>;
type CompileCause = PreparedModelInputSourceError<MlxNativeInputCause>;

/// Recipe chosen before the one original B comparison. It borrows A's actual I
/// source; a settled B1-only result cannot be upgraded into this population.
pub struct MlxPreparedModelInputPlan<'a> {
    leaf: MlxPreparedNativeInputPlan<'a>,
    host: HostPlan<'a>,
    encoder: Option<PreparedMediaEncoderTablePlan<'a>>,
    required: usize,
}
impl MlxPreparedInputMaterializer {
    /// Plans both native leaves and complete prepared controls from the same
    /// original semantic source. No model/session, Vec, Box or Arc is constructed.
    pub fn model_input_plan<'a>(
        &'a self,
        original: &'a OriginalPreparedMediaSemantics<'_>,
    ) -> Result<MlxPreparedModelInputPlan<'a>, WorkingMemoryError> {
        let source = original.source();
        let encoder = original
            .optional_encoder_table_plan()
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        if encoder.is_some_and(|encoder| !encoder.source().same_source(source)) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let FullCompiler {
            leaf,
            host,
            required,
            ..
        } = self.full_source_plan(source, encoder)?;
        Ok(MlxPreparedModelInputPlan {
            leaf,
            host,
            encoder,
            required,
        })
    }
    fn full_source_plan<'a>(
        &'a self,
        source: &'a OriginalPreparedHostInput,
        encoder: Option<PreparedMediaEncoderTablePlan<'a>>,
    ) -> Result<FullCompiler<'a>, WorkingMemoryError> {
        let host = HostPlan::new(source)?;
        let host = match encoder {
            Some(encoder) => host.with_encoder_tables(encoder.layout())?,
            None => host,
        };
        let handle =
            PreparedInputLeaf::array_layout().map_err(|_| WorkingMemoryError::UnknownBound)?;
        let typed_handles = handle
            .metadata_bytes()
            .checked_mul(source.slot_count())
            .ok_or(WorkingMemoryError::Overflow)?;
        let array_handles = handle
            .metadata_bytes()
            .checked_mul(source.slot_count())
            .ok_or(WorkingMemoryError::Overflow)?;
        let (arena, native) = leaf_recipe(
            &self.0,
            source,
            typed_handles
                .checked_add(array_handles)
                .ok_or(WorkingMemoryError::Overflow)?,
        )?;
        let required = [
            native,
            host.required_storage_bytes(),
            handle.control_bytes(),
            Stream::device_comparison_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Option<&safemlx::distributed::Group>>(),
            size_of::<Result<(), eredu_core::SessionAuthorityError>>(),
            text::control_bytes(),
            size_of::<MlxPreparedModelInputPlan<'_>>(),
            size_of::<MlxOriginalPreparedModelInput>(),
            size_of::<MlxPreparedModelInputError>(),
            size_of::<MlxPreparedModelInputBindError>(),
            size_of::<CompletedOriginalModelInput>(),
            size_of::<input::OriginalMediaPacket>(),
            size_of::<Option<input::OriginalMediaPacket>>(),
            size_of::<MlxModelInput>(),
            size_of::<Result<MlxModelInput, MlxPreparedModelInputBindError>>(),
            size_of::<CompletedMediaBindingError>(),
            size_of::<eredu_runtime::replicated_session::MediaSemanticBindingError<Error>>(),
            size_of::<std::num::TryFromIntError>(),
            size_of::<Box<std::num::TryFromIntError>>(),
            size_of::<Error>(),
            size_of::<Option<Error>>(),
            size_of::<BoundPreparedMediaSemantics>(),
            size_of::<OriginalCompositeSemanticStorageError>(),
            size_of::<eredu_runtime::working_memory::MediaSessionBinding>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        Ok(FullCompiler {
            leaf: MlxPreparedNativeInputPlan {
                runtime: &self.0,
                source,
                arena,
                required: native,
            },
            host,
            encoder,
            required,
        })
    }
}
impl MlxPreparedModelInputPlan<'_> {
    /// Full original B bytes, including the one generic account/output/error shell.
    pub fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        MemoryLedger::prepared_native_input_required_bytes(&FullCompiler {
            leaf: self.leaf.borrowed(),
            host: match self.encoder {
                Some(encoder) => {
                    HostPlan::new(self.leaf.source)?.with_encoder_tables(encoder.layout())?
                }
                None => HostPlan::new(self.leaf.source)?,
            },
            encoder: self.encoder,
            required: self.required,
        })
    }
    /// Performs one comparison, then the shared B1 initializer and both exact
    /// source views. No ordinary owner, general guard or native submission enters.
    pub fn materialize(
        self,
        pool: &MemoryLedger,
    ) -> Result<MlxOriginalPreparedModelInput, MlxPreparedModelInputError> {
        pool.compile_prepared_native_input(FullCompiler {
            leaf: self.leaf,
            host: self.host,
            encoder: self.encoder,
            required: self.required,
        })
        .map(MlxOriginalPreparedModelInput)
        .map_err(MlxPreparedModelInputError)
    }
}
struct FullCompiler<'a> {
    leaf: MlxPreparedNativeInputPlan<'a>,
    host: HostPlan<'a>,
    encoder: Option<PreparedMediaEncoderTablePlan<'a>>,
    required: usize,
}
impl PreparedNativeInputCompiler for FullCompiler<'_> {
    type Output = Body;
    type Error = CompileCause;
    fn source(&self) -> &OriginalPreparedHostInput {
        self.leaf.source
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        Ok(self.required)
    }
    fn required_storage_requirements(
        &self,
        topology: &eredu_core::MemoryTopology,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError> {
        if self.leaf.runtime.allocation_placement() != safemlx::AllocationPlacement::Host {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let mut requirements = eredu_core::DomainMemoryRequirements::zero(topology);
        requirements.add_allocation(
            u64::try_from(self.required_storage_bytes()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            &eredu_core::MemoryPlacement::fixed(topology, topology.host_domain())?,
        )?;
        Ok(requirements)
    }
    fn compile(self, custody: OriginalPreparedInputCustody) -> Result<Body, (Body, CompileCause)> {
        let source = self.leaf.source;
        self.host.construct_with_encoder(
            custody,
            |owner| Compiler { plan: self.leaf }.compile(owner),
            |native, index, actual| {
                let expected = source
                    .slot(index)
                    .ok_or(MlxNativeInputCause::Leaf(PreparedInputCause::Invalid))?;
                if !same_slot(expected, actual) {
                    return Err(MlxNativeInputCause::Leaf(PreparedInputCause::Invalid));
                }
                let leaf = native
                    .leaves
                    .get(index)
                    .ok_or(MlxNativeInputCause::Leaf(PreparedInputCause::Invalid))?;
                let typed = leaf.try_source_array().map_err(MlxNativeInputCause::Leaf)?;
                let raw = leaf.try_source_array().map_err(MlxNativeInputCause::Leaf)?;
                Ok((crate::MlxTensor::from_array(typed), raw))
            },
            |integers, floats| match self.encoder {
                Some(encoder) => encoder.fill(integers, floats),
                None => Err(eredu_nn::sequence_layout::PatchEncoderTableError::SourceGeometry),
            },
        )
    }
}
fn same_slot(a: HostTensorView<'_>, b: HostTensorView<'_>) -> bool {
    a.shape == b.shape
        && match (a.values, b.values) {
            (HostTensorValues::U32(a), HostTensorValues::U32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (HostTensorValues::I32(a), HostTensorValues::I32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (HostTensorValues::F32(a), HostTensorValues::F32(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            (HostTensorValues::Bool(a), HostTensorValues::Bool(b)) => {
                a.len() == b.len() && std::ptr::eq(a.as_ptr(), b.as_ptr())
            }
            _ => false,
        }
}
/// One completed source with exactly one live-session bind attempt. Immutable
/// aliases cannot recreate this capability or repeat its prepaid error population.
/// ```compile_fail
/// use eredu_backend_mlx::native::MlxOriginalPreparedModelInput;
/// fn no_retry(source:MlxOriginalPreparedModelInput) { let _=source.clone(); }
/// ```
#[derive(Debug)]
pub struct MlxOriginalPreparedModelInput(OriginalPreparedInputMaterialization<Body>);
impl MlxOriginalPreparedModelInput {
    /// Exact original I; borrowing it does not export a bind capability.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.0.source()
    }
    /// The one originally accepted full B charge.
    pub fn original_bytes(&self) -> u64 {
        self.0
            .original_bytes()
            .expect("prepared mechanism has one fixed host domain")
    }
    /// Binds once to an already initialized actual session. Every rejection owns
    /// B/A and the actual cause; success only shares already allocated storage.
    pub fn bind(
        self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        original: OriginalPreparedMediaSemantics<'_>,
    ) -> Result<MlxModelInput, MlxPreparedModelInputBindError> {
        let session = runtime.session();
        let boundary = if self
            .0
            .validate_pool(runtime.backend().memory_ledger())
            .is_err()
            || !self.source().same_source(original.source())
        {
            Some(WorkingMemoryError::IdentityMismatch)
        } else if session.poison.get() || session.authority.borrow().require_idle().is_err() {
            Some(WorkingMemoryError::ExecutionFenced)
        } else if !runtime
            .backend()
            .matches_prepared_target(&session.payload.target)
            || !session
                .payload
                .model
                .inference_blueprint()
                .is_some_and(|b| original.matches_destination(b))
        {
            Some(WorkingMemoryError::IdentityMismatch)
        } else {
            None
        };
        if let Some(cause) = boundary {
            return Err(MlxPreparedModelInputBindError {
                cause: CompletedMediaBindingError::boundary(original, cause),
                input: self,
            });
        }
        let semantics = match session
            .payload
            .model
            .erased()
            .bind_completed_original_media_semantics(
                original,
                session
                    .payload
                    .model
                    .inference_blueprint()
                    .expect("validated blueprint"),
                self.source(),
            ) {
            Ok(v) => v,
            Err(cause) => return Err(MlxPreparedModelInputBindError { cause, input: self }),
        };
        let packet = CompletedOriginalModelInput {
            body: self.0.storage().clone(),
            semantics,
        };
        let prompt = packet.prompt(None, None);
        drop(self); // outer Box retires; body/native/cache pins keep the same B
        Ok(prompt)
    }
}
/// Inline original packet. No Arc allocation occurs after session binding.
pub(crate) struct CompletedOriginalModelInput {
    semantics: BoundPreparedMediaSemantics,
    body: Body,
}
impl CompletedOriginalModelInput {
    // Same canonical B parts/cache/semantics for initial binding and pending
    // resume. Only the fresh request/chunk differs; no encoder or native copy.
    fn prompt(
        &self,
        request: Option<eredu_runtime::working_memory::InferenceRequest>,
        chunk: Option<std::num::NonZeroU64>,
    ) -> MlxModelInput {
        MlxModelInput {
            parts: super::super::super::pending_prompt::ModelInputParts::Original(
                self.body.parts().expect("complete B parts").clone(),
            ),
            controlled_attribution: None,
            original_media: Some(input::OriginalMediaPacket::original(self.clone())),
            placement_semantics: None,
            cache_identity: Some(self.body.cache().expect("complete B cache").clone()),
            prefill_chunk_positions: chunk,
            inference_request: request,
            memory_owner: None,
            quote: None,
        }
    }
    pub(in crate::composition::mlx::session) fn pending_prompt(
        &self,
        request: eredu_runtime::working_memory::InferenceRequest,
        chunk: std::num::NonZeroU64,
    ) -> MlxModelInput {
        self.prompt(Some(request), Some(chunk))
    }
    pub(in crate::composition::mlx::session) fn pending_parts(
        &self,
    ) -> (
        &[input::InputPart],
        &eredu_runtime::SharedPreparedInputCacheIdentity,
    ) {
        (
            self.body.parts().expect("complete B parts").as_ref(),
            self.body.cache().expect("complete B cache"),
        )
    }
    pub(crate) fn matches_workspace_source(
        &self,
        source: &eredu_runtime::input::OriginalPreparedWorkspaceSource,
    ) -> bool {
        self.body
            .prepared()
            .is_some_and(|prepared| source.matches_prepared_owner(prepared))
    }
    // Borrows the genuine body and canonical views. Content/digest equality is
    // never used as permission, and no alias gains another one-shot B bind.
    pub(crate) fn validate_request_source(
        &self,
        pool: &MemoryLedger,
        parts: &[input::InputPart],
        cache: &eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Result<
        &eredu_runtime::working_memory::MediaSessionBinding,
        eredu_core::PreparedRequestRejection,
    > {
        use eredu_core::PreparedRequestRejection as R;
        self.body
            .source()
            .validate_pool(pool)
            .map_err(|_| R::IdentityMismatch)?;
        if !self.body.source().same_source(self.semantics.source())
            || !self
                .body
                .prepared()
                .and_then(|p| p.original_source())
                .is_some_and(|s| s.same_source(self.body.source()))
            || !self
                .body
                .parts()
                .is_some_and(|p| std::ptr::eq(p.as_ref(), parts))
            || !self.body.cache().is_some_and(|c| c.same_storage(cache))
        {
            return Err(R::IdentityMismatch);
        }
        Ok(self.semantics.binding())
    }
    pub(crate) fn shape(&self) -> [u64; 2] {
        [1, self.semantics.decoder_positions() as u64]
    }
    pub(crate) fn clone_lowered(&self) -> OriginalHostLowering<crate::MlxTensor> {
        OriginalHostLowering::from_original_owner(
            self.body.prepared().expect("complete B prepared").clone(),
        )
        .unwrap_or_else(|_| unreachable!("B worker installs the genuine source"))
    }
    // No alias allocation, source rebind or semantic reconstruction.
    pub(crate) fn borrowed_semantics(&self) -> &BoundPreparedMediaSemantics {
        &self.semantics
    }
    pub(crate) fn semantics(&self) -> BoundPreparedMediaSemantics {
        self.semantics.clone()
    }
}
impl Clone for CompletedOriginalModelInput {
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            semantics: self.semantics.clone(),
        }
    }
}
/// Owning construction failure retaining its input prefix and original B charge.
#[derive(Debug)]
pub struct MlxPreparedModelInputError(
    OriginalPreparedInputMaterializationError<Body, CompileCause>,
);
impl MlxPreparedModelInputError {
    pub(in crate::composition::mlx::session::model_session) fn retirement_control_bytes(
    ) -> Option<usize> {
        OriginalPreparedInputMaterializationError::<Body, CompileCause>::retirement_control_bytes()
    }
    pub(in crate::composition::mlx::session::model_session) fn retire_storage(
        self,
    ) -> RetiredMlxPreparedModelInputError {
        RetiredMlxPreparedModelInputError(self.0.retire_storage())
    }
    /// Returns the original accounting failure, when present.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.0.accounting_failure()
    }
    /// Returns the original charge retained by this construction failure.
    pub fn retained_bytes(&self) -> u64 {
        self.0
            .retained_bytes()
            .expect("prepared mechanism has one fixed host domain")
    }
}
impl std::fmt::Display for MlxPreparedModelInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for MlxPreparedModelInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub(in crate::composition::mlx::session::model_session) struct RetiredMlxPreparedModelInputError(
    eredu_runtime::working_memory::RetiredPreparedInputMaterializationError<CompileCause>,
);

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub(in crate::composition::mlx::session::model_session) struct RetiredMlxPreparedModelInputBindError(
    eredu_runtime::working_memory::RetiredPreparedInputMaterializationError<
        CompletedMediaBindingError,
    >,
);

/// Terminal owning failure. Cause retires before input and its original B pins.
#[derive(Debug)]
pub struct MlxPreparedModelInputBindError {
    cause: CompletedMediaBindingError,
    input: MlxOriginalPreparedModelInput,
}
impl MlxPreparedModelInputBindError {
    pub(in crate::composition::mlx::session::model_session) fn retirement_control_bytes(
    ) -> Option<usize> {
        OriginalPreparedInputMaterializationError::<Body, CompletedMediaBindingError>::retirement_control_bytes()
    }
    pub(in crate::composition::mlx::session::model_session) fn retire_storage(
        self,
    ) -> RetiredMlxPreparedModelInputBindError {
        RetiredMlxPreparedModelInputBindError(self.input.0.retire_rejected(self.cause))
    }
    /// Returns the original B charge retained by this failed binding.
    pub fn retained_bytes(&self) -> u64 {
        self.input.original_bytes()
    }
}
impl std::fmt::Display for MlxPreparedModelInputBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for MlxPreparedModelInputBindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(all(test, unix, target_vendor = "apple"))]
mod encoder_tests;
#[cfg(all(test, unix, target_vendor = "apple"))]
mod tests;

mod workspace;

#[cfg(all(test, unix, target_vendor = "apple"))]
mod request_tests;

#[cfg(test)]
impl MlxPreparedInputMaterializer {
    /// Test loan from the same full B compiler, including signed source leaves
    /// which the public plain-U32 policy does not select on its own.
    pub(crate) fn with_test_original_copy_source<R>(
        &self,
        source: &OriginalPreparedHostInput,
        pool: &MemoryLedger,
        run: impl for<'source> FnOnce(
            crate::backend::array_copy::OriginalPreparedArrayCopySource<'source>,
        ) -> R,
    ) -> R {
        let full = pool
            .compile_prepared_native_input(self.full_source_plan(source, None).unwrap())
            .unwrap();
        let source =
            crate::backend::array_copy::OriginalPreparedArrayCopySource::from_input(full.storage())
                .unwrap();
        run(source)
    }
}
