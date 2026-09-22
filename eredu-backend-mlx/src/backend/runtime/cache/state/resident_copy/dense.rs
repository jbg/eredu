//! Exact native representation dispatch; no conversion or extra prompt claim.
use super::super::{
    hybrid::{
        InitializedHybridDenseGroup, PreparedDenseHybridGroupedState,
        PublishedDenseHybridGroupedState,
    },
    key_value::{
        InitializedPagedDenseCopy, PreparedDenseResidentKvState, PublishedDenseResidentKvState,
    },
    pooling::{PreparedDenseResidentPoolingState, PublishedDenseResidentPoolingState},
};
use super::*;
use crate::backend::nn::workspace::ProjectedNativeStorage;
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTraceReport};
use eredu_runtime::{
    working_memory::{
        InferencePreparationStage, InferencePromptCompletion, InitializedDenseDecoderSlots,
        WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkspaceResidentLayerState,
    },
    DeviceState, HostSlotAttachmentError, HostSlotMetadata,
};
use std::num::NonZeroU32;
fn other(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(e))
}

pub(crate) enum PreparedResidentDenseCopy<'a> {
    Paged(PreparedPagedKvCopy<'a>),
    KeyValue(PreparedResidentKvCopy<'a>),
    HybridGrouped(PreparedHybridGroupedCopy<'a>),
    Pooling(PreparedResidentPoolingCopy<'a>),
}
impl<'a> PreparedResidentDenseCopy<'a> {
    pub(crate) fn is_paged(&self) -> bool {
        matches!(self, Self::Paged(_))
            || matches!(self,Self::HybridGrouped(plan) if plan.is_paged())
    }
    /// Logical copied-state allowance from this exact immutable source. The
    /// dense constructor carries its selected independent manager and no old
    /// request owners; installation retains one newly admitted request. Native physical
    /// storage and future growth remain in that request's independent quote.
    pub(crate) fn logical_snapshot_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        use crate::backend::runtime::cache::state::snapshot_estimate;
        let state_bytes = match self {
            Self::KeyValue(_) | Self::Paged(_) => std::mem::size_of::<MlxKeyValueState>(),
            Self::HybridGrouped(_) => std::mem::size_of::<MlxHybridState>(),
            Self::Pooling(_) => std::mem::size_of::<MlxPoolingAttentionState>(),
        };
        let mut total = snapshot_estimate::state_metadata(
            self.shared_layout().layout(),
            state_bytes,
            std::mem::size_of::<eredu_runtime::working_memory::InferenceRequest>() as u64,
        );
        let manager = match self {
            Self::Paged(plan) => plan.logical_snapshot_manager_bytes(),
            Self::HybridGrouped(plan) => plan.logical_snapshot_manager_bytes(),
            _ => Some(0),
        };
        total = total.and_then(|bytes| bytes.checked_add(manager?));
        // Preserve the shared logical snapshot treatment of stored arrays and
        // views. Compact-source backing may exceed its copied logical view;
        // this remains conservative without treating physical Q as copy work.
        self.visit_snapshot_operands(&mut |operand| {
            total = total.and_then(|bytes| {
                let (logical, rank) = match operand {
                    SnapshotOperand::Array(array) => {
                        let descriptor = array.try_descriptor().ok()?;
                        (
                            descriptor.facts().logical_bytes(),
                            descriptor.facts().rank(),
                        )
                    }
                    SnapshotOperand::Host(host) => {
                        let descriptor = host.try_fixed_descriptor::<4>().ok()?;
                        (descriptor.nbytes(), descriptor.shape().len())
                    }
                };
                bytes.checked_add(snapshot_estimate::array_bytes(logical, rank)?)
            });
            Ok(())
        })
        .ok()?;
        let bytes = total?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        match self {
            Self::Paged(p) => p.shared_layout(),
            Self::KeyValue(p) => p.shared_layout(),
            Self::HybridGrouped(p) => p.shared_layout(),
            Self::Pooling(p) => p.shared_layout(),
        }
    }
    /// Actual saved physical source kinds under the same lexical source loan.
    /// This grants no copy, promotion or prepared execution authority.
    pub(crate) fn visit_snapshot_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        if let Self::Paged(source) = self {
            return source.snapshot_source().visit_operands(visitor);
        }
        if let Self::HybridGrouped(source) = self {
            if source.is_paged() {
                return SnapshotArraySources::visit_operands(source, visitor);
            }
        }
        let mut failure = None;
        self.visit_retained_arrays(&mut |array| {
            if failure.is_none() {
                failure = visitor(SnapshotOperand::Array(array)).err();
            }
        })?;
        failure.map_or(Ok(()), Err)
    }
    pub(crate) fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        match self {
            Self::Paged(p) => {
                return p.snapshot_source().visit_arrays(&mut |array| {
                    visitor(array);
                    Ok(())
                });
            }
            Self::KeyValue(p) => p.visit_operands(visitor),
            Self::HybridGrouped(p) => {
                if p.is_paged() {
                    return p.visit_paged_arrays(visitor);
                }
                p.visit_operands(visitor)
            }
            Self::Pooling(p) => p.visit_operands(visitor),
        };
        Ok(())
    }
    pub(crate) fn visit_retained_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        match self {
            Self::Paged(p) => {
                return p.snapshot_source().visit_arrays(&mut |array| {
                    visitor(array);
                    Ok(())
                });
            }
            Self::KeyValue(p) => p.visit_operands(visitor),
            Self::HybridGrouped(p) => {
                if p.is_paged() {
                    return p.visit_paged_arrays(visitor);
                }
                p.visit_retained_arrays(visitor)
            }
            Self::Pooling(p) => p.visit_retained_arrays(visitor),
        };
        Ok(())
    }
    pub(crate) fn initialization_peak_bytes(&self, pool: &MemoryLedger) -> Result<u64, Error> {
        match self {
            Self::Paged(p) => p
                .dense_initialization_peak_bytes_fixed()
                .map_err(ResidentDecoderPreparationError::into_error),
            Self::KeyValue(p) => Ok(p
                .dense_host_copy(pool)
                .map_err(other)?
                .initialization_peak_bytes()),
            Self::HybridGrouped(p) => Ok(p.dense_host_copy(pool)?.initialization_peak_bytes()),
            Self::Pooling(p) => Ok(p
                .dense_host_copy(pool)
                .map_err(pooling_error)?
                .initialization_peak_bytes()),
        }
    }
    pub(crate) fn initialization_peak_bytes_fixed(
        &self,
    ) -> Result<u64, ResidentDecoderPreparationError> {
        match self {
            Self::Paged(p) => p.dense_initialization_peak_bytes_fixed(),
            Self::KeyValue(p) => p.dense_initialization_peak_bytes_fixed(),
            Self::HybridGrouped(p) => p.dense_initialization_peak_bytes_fixed(),
            Self::Pooling(p) => p.dense_initialization_peak_bytes_fixed(),
        }
    }
    /// Host constructors only. Dense payload is already in the fresh prompt P;
    /// native work/collectors and run controls remain in its complete Q recipe.
    pub(crate) fn host_preparation_control_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let registered = match self {
            Self::Paged(p) => p.registered_source_tables(),
            Self::KeyValue(p) => p.registered_source_tables(),
            Self::HybridGrouped(p) => p.registered_source_tables().map_err(|e| match e {
                WorkingMemoryError::Overflow => E::Overflow,
                _ => E::UnknownBound,
            })?,
            Self::Pooling(p) => p.registered_source_tables(),
        };
        let tables = match self {
            Self::Paged(p) => p.dense_host_preparation_bytes()?,
            Self::KeyValue(p) => p.dense_host_preparation_bytes()?,
            Self::HybridGrouped(p) => p.dense_host_preparation_bytes()?,
            Self::Pooling(p) => p.dense_host_preparation_bytes()?,
        };
        let count = registered.checked_add(1).ok_or(E::Overflow)?;
        let pins = WorkingMemoryStorage::<StorageIdentity>::copy_source_wrapper_bytes(0, count)
            .map_err(|e| match e {
                WorkingMemoryError::Overflow => E::Overflow,
                _ => E::UnknownBound,
            })?;
        let pair = WorkingMemoryStorage::<StorageIdentity>::copy_source_pair_bytes().map_err(
            |e| match e {
                WorkingMemoryError::Overflow => E::Overflow,
                _ => E::UnknownBound,
            },
        )?;
        [
            tables,
            pins,
            pair,
            // Actual boxed source of the selected host binding/attachment error,
            // separate from the return value's inline transport below.
            std::mem::size_of::<eredu_runtime::working_memory::DecoderCopyAdmissionError>()
                .max(std::mem::size_of::<
                    HostSlotAttachmentError<WorkingMemoryError>,
                >())
                .max(std::mem::size_of::<WorkingMemoryError>())
                .max(std::mem::size_of::<
                    crate::backend::runtime::cache::state::key_value::ResidentKvCopyError,
                >())
                .max(std::mem::size_of::<
                    crate::backend::runtime::cache::state::pooling::ResidentPoolingCopyError,
                >()),
            std::mem::size_of::<Box<dyn std::error::Error + Send + Sync>>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<
                Result<(InitializedDenseResidentCopy<'_>, WorkingMemoryFundingScope), Error>,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(E::Overflow)
    }

    pub(crate) fn construct(
        &self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete: WorkingMemoryStorage<StorageIdentity>,
        pool: &MemoryLedger,
    ) -> Result<(InitializedDenseResidentCopy<'a>, WorkingMemoryFundingScope), Error> {
        self.construct_with_preparation(stage, funding, complete, pool, None)
    }
    pub(crate) fn construct_prepared(
        &self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete: WorkingMemoryStorage<StorageIdentity>,
        pool: &MemoryLedger,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<(InitializedDenseResidentCopy<'a>, WorkingMemoryFundingScope), Error> {
        self.construct_with_preparation(stage, funding, complete, pool, Some(host))
    }
    fn construct_with_preparation(
        &self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete: WorkingMemoryStorage<StorageIdentity>,
        pool: &MemoryLedger,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<(InitializedDenseResidentCopy<'a>, WorkingMemoryFundingScope), Error> {
        match self {
            Self::Paged(p) => {
                let (slots, native) =
                    p.construct_dense(stage, funding, complete, pool, host.ok_or_else(unknown)?)?;
                Ok((InitializedDenseResidentCopy::Paged(slots), native))
            }
            Self::KeyValue(p) => {
                let (slots, native) = stage
                    .construct_dense_decoder(
                        p.dense_host_copy_with_preparation(pool, host)
                            .map_err(other)?,
                        funding,
                        complete,
                    )
                    .map_err(other)?;
                Ok((InitializedDenseResidentCopy::KeyValue(slots), native))
            }
            Self::Pooling(p) => {
                let (slots, native) = stage
                    .construct_dense_decoder(
                        p.dense_host_copy_with_preparation(pool, host)
                            .map_err(pooling_error)?,
                        funding,
                        complete,
                    )
                    .map_err(other)?;
                Ok((InitializedDenseResidentCopy::Pooling(slots), native))
            }
            Self::HybridGrouped(p) => p
                .dense_host_copy_with_preparation(pool, host)?
                .construct(stage, funding, complete)
                .map(|(s, n)| (InitializedDenseResidentCopy::HybridGrouped(s), n)),
        }
    }
    pub(crate) fn copy_dense_retained(
        self,
        slots: InitializedDenseResidentCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<PreparedDenseResidentState<'a>, Error> {
        self.copy_dense_retained_observed(slots, stream, roots, &mut |_| Ok(()))
    }
    pub(crate) fn copy_dense_retained_observed(
        self,
        slots: InitializedDenseResidentCopy<'a>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<PreparedDenseResidentState<'a>, Error> {
        match (self, slots) {
            (Self::Paged(p), InitializedDenseResidentCopy::Paged(s)) => p
                .copy_dense_retained_with(s, stream, roots, observe)
                .map(PreparedDenseResidentState::Paged),
            (Self::KeyValue(p), InitializedDenseResidentCopy::KeyValue(s)) => p
                .copy_dense_retained(s, stream, roots)
                .map(PreparedDenseResidentState::KeyValue)
                .map_err(other),
            (Self::HybridGrouped(p), InitializedDenseResidentCopy::HybridGrouped(s)) => p
                .copy_dense_retained_with(s, stream, roots, observe)
                .map(PreparedDenseResidentState::HybridGrouped),
            (Self::Pooling(p), InitializedDenseResidentCopy::Pooling(s)) => p
                .copy_dense_retained(s, stream, roots)
                .map(PreparedDenseResidentState::Pooling)
                .map_err(pooling_error),
            _ => Err(mismatch()),
        }
    }
    pub(crate) fn project_dense_workspace(
        &self,
        batch: NonZeroU32,
        context: &'a WorkspaceContext,
    ) -> Result<ProjectedDenseResidentCopy, Error> {
        match self {
            Self::Paged(p) => {
                let p = p.project_dense_workspace(batch, context)?;
                Ok(ProjectedDenseResidentCopy {
                    state: p.state,
                    source_storage: p.source_storage,
                    copy: p.copy,
                })
            }
            Self::KeyValue(p) => {
                let p = p.project_dense_workspace(batch, context).map_err(other)?;
                Ok(ProjectedDenseResidentCopy {
                    state: p.state,
                    source_storage: p.source_storage,
                    copy: p.copy,
                })
            }
            Self::Pooling(p) => {
                let p = p.project_dense_workspace(batch, context).map_err(other)?;
                Ok(ProjectedDenseResidentCopy {
                    state: p.state,
                    source_storage: p.source_storage,
                    copy: p.copy,
                })
            }
            Self::HybridGrouped(p) => {
                let p = p.project_dense_workspace(batch, context)?;
                Ok(ProjectedDenseResidentCopy {
                    state: p.state,
                    source_storage: p.source_storage,
                    copy: p.copy,
                })
            }
        }
    }
}
pub(crate) struct ProjectedDenseResidentCopy {
    pub(crate) state: DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    pub(crate) source_storage: ProjectedNativeStorage,
    pub(crate) copy: WorkspaceTraceReport,
}
pub(crate) enum InitializedDenseResidentCopy<'a> {
    Paged(InitializedPagedDenseCopy<'a>),
    Pooling(
        InitializedDenseDecoderSlots<
            'a,
            MlxPoolingAttentionCache,
            MlxPoolingAttentionCache,
            StorageIdentity,
        >,
    ),
    HybridGrouped(InitializedHybridDenseGroup<'a>),
    KeyValue(
        InitializedDenseDecoderSlots<
            'a,
            MlxKeyValueLayerState,
            MlxKeyValueLayerState,
            StorageIdentity,
        >,
    ),
}
pub(crate) enum PreparedDenseResidentState<'a> {
    Paged(PreparedDenseResidentKvState<'a>),
    KeyValue(PreparedDenseResidentKvState<'a>),
    HybridGrouped(PreparedDenseHybridGroupedState<'a>),
    Pooling(PreparedDenseResidentPoolingState<'a>),
}
impl PreparedDenseResidentState<'_> {
    pub(crate) fn slot_metadata(&self) -> &HostSlotMetadata {
        match self {
            Self::Paged(p) | Self::KeyValue(p) => p.slot_metadata(),
            Self::HybridGrouped(p) => p.slot_metadata(),
            Self::Pooling(p) => p.slot_metadata(),
        }
    }
    pub(crate) fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        match self {
            Self::Paged(p) => return p.visit_paged_operands(visitor),
            Self::KeyValue(p) => p.visit_operands(visitor),
            Self::HybridGrouped(p) => return p.visit_complete_operands(visitor),
            Self::Pooling(p) => p.visit_operands(visitor),
        }
        Ok(())
    }
}
impl<'a> PreparedDenseResidentState<'a> {
    pub(crate) fn publish_for_control(
        self,
    ) -> Result<
        (PublishedResidentDecoderState, InferencePromptCompletion),
        DenseResidentPublishError<'a>,
    > {
        match self {
            Self::Paged(p) => p
                .publish_for_control()
                .map(|(s, c)| (PublishedResidentDecoderState::KeyValue(s), c))
                .map_err(|e| {
                    let (p, error) = e.into_parts();
                    DenseResidentPublishError {
                        owner: Self::Paged(p),
                        error,
                    }
                }),
            Self::KeyValue(p) => p
                .publish_for_control()
                .map(|(s, c)| (PublishedResidentDecoderState::KeyValue(s), c))
                .map_err(|e| {
                    let (p, error) = e.into_parts();
                    DenseResidentPublishError {
                        owner: Self::KeyValue(p),
                        error,
                    }
                }),
            Self::Pooling(p) => p
                .publish_for_control()
                .map(|(s, c)| (PublishedResidentDecoderState::Pooling(s), c))
                .map_err(|e| {
                    let (p, error) = e.into_parts();
                    DenseResidentPublishError {
                        owner: Self::Pooling(p),
                        error,
                    }
                }),
            Self::HybridGrouped(p) => p
                .publish_for_control()
                .map(|(s, c)| (PublishedResidentDecoderState::HybridGrouped(s), c))
                .map_err(|e| {
                    let (p, error) = e.into_parts();
                    DenseResidentPublishError {
                        owner: Self::HybridGrouped(p),
                        error,
                    }
                }),
        }
    }
}
/// Every variant contains its exact already-published actual native table.
/// This grants neither settled native values nor runnable session authority.
pub(crate) enum PublishedResidentDecoderState {
    KeyValue(PublishedDenseResidentKvState),
    HybridGrouped(PublishedDenseHybridGroupedState),
    Pooling(PublishedDenseResidentPoolingState),
}
impl PublishedResidentDecoderState {
    pub(crate) fn visit_registered_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match self {
            Self::HybridGrouped(state) => state.visit_registered_child_metadata(visitor),
            Self::KeyValue(_) | Self::Pooling(_) => Ok(()),
        }
    }
}
pub(crate) struct DenseResidentPublishError<'a> {
    owner: PreparedDenseResidentState<'a>,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}
impl<'a> DenseResidentPublishError<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedDenseResidentState<'a>,
        HostSlotAttachmentError<WorkingMemoryError>,
    ) {
        (self.owner, self.error)
    }
}
