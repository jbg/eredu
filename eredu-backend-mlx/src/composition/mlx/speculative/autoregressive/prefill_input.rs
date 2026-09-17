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
        WorkspaceContext, WorkspaceDtype, WorkspaceMetadataFunding, WorkspaceTensor,
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
    copied: OriginalPendingTokenCopy,
    positions: u64,
}
impl PreparedPrefillInput {
    /// The actual source array is never downloaded or replaced by invented IDs.
    /// Existing registered-copy admission and original completion finish the
    /// complete [1,N] unsigned matrix before this owner can escape.
    pub(crate) fn prepare_copy(
        input: &MlxModelInput,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<Self, Error> {
        let controls = [
            crate::backend::array_copy::OriginalPreparedArrayCopySource::control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            size_of::<Self>(),
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
                capacity,
            )?,
            None => {
                plan.copy_registered(environment, initialized, mechanisms, funding, capacity)?
            }
        };
        Ok(Self {
            copied,
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
        SpanView::inspect(chunk, self.positions)?.apply(NativeView {
            source: self.copied.array(),
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
