//! Exact native prompt source copied once, then borrowed by each admitted span.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{OriginalPendingTokenCopy, PreparedPendingTokenInput},
    nn::workspace::MlxMetalWorkspaceMechanisms,
};
use eredu_nn::{
    Index, Tensor,
    workspace::{
        HostMetadataFunding, WorkspaceContext, WorkspaceDtype, WorkspaceTensor,
        WorkspaceTraceReport,
    },
};
use eredu_runtime::{prefill::PrefillChunk, working_memory::WorkingMemoryError};
use safemlx::{OriginalScopeObserver, PrefillRootsRuntime};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU64,
};

pub(crate) struct PreparedPrefillInput {
    body: Body,
    positions: u64,
}
enum Body {
    Tokens(OriginalPendingTokenCopy),
    Media {
        packet: crate::backend::runtime::media::input::OriginalMediaPacket,
        semantics: eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        source: std::cell::RefCell<
            Option<crate::composition::mlx::replicated_text::OriginalAutoregressiveMediaPrefill>,
        >,
    },
}
impl PreparedPrefillInput {
    /// The actual retained ingress can lend an earlier span's backing to a
    /// later decoder state. Keep its completed account facts with that state.
    pub(crate) fn visit_retained_media_roots(
        &self,
        visitor: &mut dyn FnMut(&Array),
        funding: &HostMetadataFunding,
    ) -> Result<(), Error> {
        use crate::composition::mlx::replicated_text::OriginalAutoregressiveMediaPrefill;
        funding
            .reserve_metadata(size_of::<(
                &Self,
                &mut dyn FnMut(&Array),
                &HostMetadataFunding,
                std::cell::Ref<'_, Option<OriginalAutoregressiveMediaPrefill>>,
                &OriginalAutoregressiveMediaPrefill,
                &dyn std::any::Any,
                &mut dyn FnMut(&Array),
                &crate::MlxTensor,
                Result<(), Error>,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        if let Body::Media { source, .. } = &self.body {
            let source = source
                .try_borrow()
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            source
                .as_ref()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
                .visit_retained_roots(visitor);
        }
        Ok(())
    }
    pub(crate) fn media_packet(
        &self,
    ) -> Option<&crate::backend::runtime::media::input::OriginalMediaPacket> {
        match &self.body {
            Body::Media { packet, .. } => Some(packet),
            Body::Tokens(_) => None,
        }
    }
    pub(crate) fn media_semantics(
        &self,
    ) -> Option<&eredu_architectures::media_plan::BoundPreparedMediaSemantics> {
        match &self.body {
            Body::Media { semantics, .. } => Some(semantics),
            Body::Tokens(_) => None,
        }
    }
    pub(crate) fn prepare_media_source(
        &self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        geometry: eredu_core::InferenceGeometry,
        role: &eredu_runtime::working_memory::OriginalSpeculativeRole,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        let Body::Media {
            packet,
            semantics,
            source,
        } = &self.body
        else {
            return Ok(());
        };
        let mut source = source
            .try_borrow_mut()
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        if source.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::AlreadyStarted));
        }
        *source = Some(model.erased_mut().prepare_autoregressive_media_prefill(
            packet,
            semantics,
            &mut state.native,
            geometry,
            role,
            context,
            &state.stream,
        )?);
        Ok(())
    }
    pub(crate) fn media_span(
        &self,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        completion:&mut dyn crate::composition::mlx::replicated_text::AutoregressiveSequenceCompletion,
        observer: &mut dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>,
    ) -> Result<Option<Array>, Error> {
        let Body::Media { source, .. } = &self.body else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        let mut source = source
            .try_borrow_mut()
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        model
            .erased_mut()
            .autoregressive_media_prefill_span_with_completion(
                source
                    .as_mut()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,
                &mut state.native,
                span,
                &state.stream,
                completion,
                observer,
            )
    }

    /// The actual source array is never downloaded or replaced by invented IDs.
    /// Existing registered-copy admission and original completion finish the
    /// complete [1,N] unsigned matrix before this owner can escape.
    pub(crate) fn prepare_copy(
        input: &MlxModelInput,
        model: &mut Executable,
        state: &mut MlxAutoregressiveState,
        sources: &AutoregressiveSourcePair,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &HostMetadataFunding,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, Error> {
        let controls = [
            crate::backend::array_copy::OriginalPreparedArrayCopySource::control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            size_of::<Self>(),
            size_of::<eredu_architectures::media_plan::BoundPreparedMediaSemantics>(),
            size_of::<Result<eredu_architectures::media_plan::BoundPreparedMediaSemantics, Error>>(
            ),
            size_of::<Option<crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>>>(),
            size_of::<
                Result<
                    crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>,
                    WorkingMemoryError,
                >,
            >(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<&Array>>(),
            size_of::<NonZeroU64>(),
            size_of::<
                Result<
                    PreparedPendingTokenInput<'_>,
                    crate::backend::array_copy::PendingTokenSourceCause,
                >,
            >(),
            Array::descriptor_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if let Some(packet) = input.with_borrowed(|input| input.original_media().cloned()) {
            let crate::backend::runtime::media::input::OriginalMediaPacket::Original(_) = &packet
            else {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            };
            packet
                .semantics()
                .source()
                .validate_pool(environment.pool())
                .map_err(Error::PrefillControl)?;
            let [batch, positions] = packet.shape();
            if batch != 1 || positions == 0 {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let original = packet.semantics();
            sources.validate_media_input(&original)?;
            let semantics = model.prepare_autoregressive_media_semantics(
                original.source(),
                &mut state.native,
                environment.pool(),
                funding,
                &state.stream,
            )?;
            if u64::try_from(semantics.decoder_positions()).ok() != Some(positions)
                || !state.native.matches_media_binding(
                    model.erased().inference_execution_identity(),
                    semantics.binding(),
                )
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            return Ok(Self {
                body: Body::Media {
                    packet,
                    semantics,
                    source: std::cell::RefCell::new(None),
                },
                positions,
            });
        }
        let tokens = input
            .plain_token_array()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let descriptor = tokens
            .try_descriptor()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let [1, positions] = descriptor.shape() else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        let positions = u64::try_from(*positions)
            .ok()
            .and_then(NonZeroU64::new)
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        drop(descriptor);
        let plan = PreparedPendingTokenInput::new_prefill_fixed(tokens, positions)
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let original = input
            .original_text_copy_source()
            .transpose()
            .map_err(Error::PrefillControl)?;
        let copied = match original.as_ref() {
            Some(source) => plan.copy_registered_with_prepared(
                Some(source),
                environment,
                initialized,
                mechanisms,
                funding,
                &capacity,
            )?,
            None => {
                plan.copy_registered(environment, initialized, mechanisms, funding, &capacity)?
            }
        };
        Ok(Self {
            body: Body::Tokens(copied),
            positions: positions.get(),
        })
    }
    pub(crate) fn positions(&self) -> u64 {
        self.positions
    }
    pub(crate) fn dtype(&self) -> WorkspaceDtype {
        WorkspaceDtype::Uint32
    }

    /// Only a static slice of the copied native source is appended to the
    /// equation receipt. This source view creates no Initialize or input upload.
    pub(crate) fn trace_span(
        &self,
        chunk: &PrefillChunk,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTraceReport, Error> {
        let view = SpanView::inspect(chunk, self.positions)?;
        let source = WorkspaceTensor::existing(
            context.layout(&[1, self.positions as i32], self.dtype())?,
            context,
        )?;
        context.begin_span();
        let output = view.apply(WorkspaceView {
            source: &source,
            context,
        })?;
        Ok(context.finish_report(&[output])?)
    }
    /// The caller has entered this exact span Scope and begun its prepared
    /// construction bank. The shared slice worker creates the view there; its
    /// backing already retains the independently admitted whole-copy account.
    pub(crate) fn view_span(
        &self,
        chunk: &PrefillChunk,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if !observer.same_scope(&OriginalScopeObserver::require_current()?) {
            return Err(observer.domain_error().into());
        }
        let Body::Tokens(copied) = &self.body else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        };
        SpanView::inspect(chunk, self.positions)?.apply(NativeView {
            source: copied.array(),
            stream,
        })
    }
    /// Actual shared view caller controls, to be included in each span's H
    /// before the bank/Scope is created. Native descriptors come from the trace.
    pub(crate) fn view_control_bytes() -> Option<usize> {
        [
            safemlx::ops::indexing::inline_basic_index_control_bytes()?,
            size_of::<SpanView>(),
            size_of::<Result<SpanView, Error>>(),
            size_of::<NativeView<'_>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<Array>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<Result<OriginalScopeObserver, safemlx::error::Exception>>(),
            size_of::<[Index; 2]>(),
            size_of::<(std::ops::RangeFull, std::ops::Range<i32>)>(),
            size_of::<Error>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

#[derive(Clone, Copy)]
struct SpanView {
    start: i32,
    end: i32,
}
impl SpanView {
    fn inspect(chunk: &PrefillChunk, positions: u64) -> Result<Self, Error> {
        if chunk.input.start >= chunk.input.end || chunk.input.end > positions {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(Self {
            start: i32::try_from(chunk.input.start)
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
            end: i32::try_from(chunk.input.end)
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
        })
    }
    fn apply<M: ViewMechanism>(self, mechanism: M) -> Result<M::Value, Error> {
        mechanism.slice(self.start, self.end)
    }
}
trait ViewMechanism {
    type Value;
    fn slice(self, start: i32, end: i32) -> Result<Self::Value, Error>;
}
struct NativeView<'a> {
    source: &'a Array,
    stream: &'a Stream,
}
impl ViewMechanism for NativeView<'_> {
    type Value = Array;
    fn slice(self, start: i32, end: i32) -> Result<Array, Error> {
        Ok(self
            .source
            .try_index_device((.., start..end), self.stream)?)
    }
}
struct WorkspaceView<'a> {
    source: &'a WorkspaceTensor,
    context: &'a WorkspaceContext,
}
impl ViewMechanism for WorkspaceView<'_> {
    type Value = WorkspaceTensor;
    fn slice(self, start: i32, end: i32) -> Result<WorkspaceTensor, Error> {
        Ok(self
            .source
            .index(&[Index::Full, Index::Range(start, end)], self.context)?)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
