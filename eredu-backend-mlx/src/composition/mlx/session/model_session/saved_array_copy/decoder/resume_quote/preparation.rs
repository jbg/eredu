//! Finite source registration and isolated-copy metadata for saved resume.
use super::super::{preparation as projection, storage, *};
use crate::backend::nn::workspace::{
    ResidentExecutionMechanisms, MlxWorkspaceFactError, ProjectedNativeStorage,
    ProjectionSourceLayout,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::{
    WorkspaceCopyPreparationError, WorkspaceCopyPreparationLayout,
    WorkspaceCopyPreparationLayoutBuilder, WorkspaceIsolatedCopyPreparation,
};
use eredu_runtime::working_memory::{
    RegisteredWorkspaceStorageLayout, WorkingMemoryPool, WorkspaceCopyAdmissionError,
};
use std::mem::{size_of, size_of_val};

type CopyError = WorkspaceCopyPreparationError<MlxWorkspaceFactError>;

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::session) enum ResumeSourceCause {
    #[error(transparent)]
    Pending(#[from] crate::composition::mlx::session::model_session::pending_prompt::PendingPromptPreparationCause),
    #[error(transparent)]
    DenseHost(#[from] eredu_runtime::working_memory::DecoderHostPreparationError),
    #[error(transparent)]
    Decoder(#[from] crate::backend::runtime::cache::state::ResidentDecoderPreparationError),
    #[error(transparent)]
    Storage(#[from] storage::StoragePreparationCause),
    #[error(transparent)]
    Projection(#[from] projection::SnapshotProjectionCause),
    #[error(transparent)]
    Copy(#[from] CopyError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Context(#[from] eredu_nn::workspace::WorkspaceContextMetadataError<MlxWorkspaceFactError>),
    #[error(transparent)]
    PendingMetadata(#[from] crate::backend::array_copy::PendingTokenMetadataError<eredu_nn::workspace::WorkspaceContextMetadataError<MlxWorkspaceFactError>>),
}

impl ResumeSourceCause {
    /// Fixed pregrant classification. Preserve arithmetic failures and existing
    /// memory refusals without formatting or erasing any source into a Box.
    pub(in crate::composition::mlx::session) fn into_memory(self) -> WorkingMemoryError {
        use crate::backend::{
            array_copy::{PendingTokenMetadataError, PendingTokenSourceCause},
            nn::workspace::ProjectionSourceError,
            runtime::{
                cache::state::{
                    ResidentDecoderPreparationError as Decoder, ResidentKvPreparationError as Kv,
                    ResidentPoolingPreparationError as Pooling,
                },
                residency::{manager::ResidencyError, storage::SnapshotStorageInspectionError},
            },
        };
        use crate::composition::mlx::session::model_session::pending_prompt::PendingPromptPreparationCause as Pending;
        use eredu_nn::workspace::{WorkspaceCopyError, WorkspaceReportError};
        use eredu_runtime::{
            HostSlotInitializationError, working_memory::DecoderHostPreparationError,
        };
        match self {
            Self::Memory(cause)
            | Self::Pending(Pending::Host(cause))
            | Self::Decoder(Decoder::Memory(cause))
            | Self::Decoder(Decoder::KeyValue(Kv::Memory(cause)))
            | Self::Decoder(Decoder::Pooling(Pooling::Memory(cause)))
            | Self::Storage(storage::StoragePreparationCause::Memory(cause))
            | Self::Storage(storage::StoragePreparationCause::Inspection(
                SnapshotStorageInspectionError::Residency(
                    ResidencyError::OriginalInventory(cause) | ResidencyError::OriginalCache(cause),
                ),
            )) => cause,
            Self::DenseHost(DecoderHostPreparationError::Overflow)
            | Self::Decoder(Decoder::Initialization(HostSlotInitializationError::Overflow {
                ..
            }))
            | Self::Decoder(Decoder::KeyValue(Kv::Initialization(
                HostSlotInitializationError::Overflow { .. },
            )))
            | Self::Decoder(Decoder::Pooling(Pooling::Initialization(
                HostSlotInitializationError::Overflow { .. },
            )))
            | Self::Storage(storage::StoragePreparationCause::Inspection(
                SnapshotStorageInspectionError::Residency(ResidencyError::ArithmeticOverflow {
                    ..
                }),
            ))
            | Self::Projection(projection::SnapshotProjectionCause::Overflow)
            | Self::Projection(projection::SnapshotProjectionCause::Source(
                ProjectionSourceError::Overflow,
            ))
            | Self::Copy(CopyError::Overflow)
            | Self::Copy(CopyError::Source(WorkspaceCopyError::Overflow { .. }))
            | Self::Copy(CopyError::Report(WorkspaceReportError::Overflow)) => {
                WorkingMemoryError::Overflow
            }
            Self::Pending(Pending::Source(PendingTokenSourceCause::Descriptor(cause)))
            | Self::PendingMetadata(PendingTokenMetadataError::Source(
                PendingTokenSourceCause::Descriptor(cause),
            ))
            | Self::Projection(projection::SnapshotProjectionCause::Source(
                ProjectionSourceError::Descriptor(cause),
            )) => match cause {
                safemlx::ArrayDescriptorError::LogicalBytesOverflow => WorkingMemoryError::Overflow,
                _ => WorkingMemoryError::UnknownBound,
            },
            Self::Projection(projection::SnapshotProjectionCause::Source(
                ProjectionSourceError::Layout(cause),
            ))
            | Self::Copy(CopyError::Geometry(cause))
            | Self::PendingMetadata(PendingTokenMetadataError::Geometry(cause)) => {
                layout_memory(cause)
            }
            Self::Copy(CopyError::Mechanism(cause)) => fact_memory(cause),
            Self::Context(cause)
            | Self::PendingMetadata(PendingTokenMetadataError::Visitor(cause)) => {
                context_memory(cause)
            }
            // Other cold refusals prove no complete bound. Construction-only
            // variants retain their ordinary typed causes on the admitted path;
            // this query never invokes those allocating workers.
            _ => WorkingMemoryError::UnknownBound,
        }
    }
}

fn layout_memory(cause: eredu_nn::workspace::WorkspaceLayoutError) -> WorkingMemoryError {
    use eredu_nn::workspace::WorkspaceLayoutError;
    match cause {
        WorkspaceLayoutError::ElementCountOverflow | WorkspaceLayoutError::ByteCountOverflow => {
            WorkingMemoryError::Overflow
        }
        WorkspaceLayoutError::NegativeExtent => WorkingMemoryError::UnknownBound,
    }
}

fn fact_memory(cause: MlxWorkspaceFactError) -> WorkingMemoryError {
    use crate::backend::nn::workspace::MlxWorkspaceFactCause;
    match *cause.cause() {
        MlxWorkspaceFactCause::PopulationOverflow | MlxWorkspaceFactCause::Integer(_) => {
            WorkingMemoryError::Overflow
        }
        MlxWorkspaceFactCause::Layout(cause) => layout_memory(cause),
        _ => WorkingMemoryError::UnknownBound,
    }
}

fn context_memory(
    cause: eredu_nn::workspace::WorkspaceContextMetadataError<MlxWorkspaceFactError>,
) -> WorkingMemoryError {
    use eredu_nn::workspace::{
        WorkspaceContextMetadataError, WorkspaceMetadataError, HostMetadataFundingError,
        WorkspaceReportError,
    };
    match cause {
        WorkspaceContextMetadataError::Mechanism(cause) => fact_memory(cause),
        WorkspaceContextMetadataError::Geometry(cause) => layout_memory(cause),
        WorkspaceContextMetadataError::Metadata(
            WorkspaceMetadataError::Overflow
            | WorkspaceMetadataError::Funding(HostMetadataFundingError::Overflow)
            | WorkspaceMetadataError::Report(WorkspaceReportError::Overflow),
        ) => WorkingMemoryError::Overflow,
        _ => WorkingMemoryError::UnknownBound,
    }
}

/// The source census includes pending input, but the credited program copies
/// only decoder operands and the random key. The later pending-input program
/// remains a full contribution, exactly as in the ordinary resume equation.
pub(super) struct ResumeSourcePreparation<'a> {
    source: storage::DecoderStoragePlan<'a>,
    projection: projection::SnapshotProjectionPlan<'a>,
    registration: RegisteredWorkspaceStorageLayout<StorageIdentity>,
    copy: WorkspaceCopyPreparationLayout,
    pool: &'a WorkingMemoryPool,
    mechanisms: ResidentExecutionMechanisms,
    bytes: usize,
}

pub(super) struct PreparedResumeSource {
    pub(super) copy_preparation: RegisteredWorkspaceCopy<StorageIdentity>,
    // Owns the accepted preparation through every preceding metadata field.
    // The enclosing quote moves this into its last registered-source field.
    pub(super) registered_source: WorkingMemoryStorage<StorageIdentity>,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PreparationFailure {
    #[source]
    cause: ResumeSourceCause,
    _host: HostPreparationAuthority,
}

impl<'a> ResumeSourcePreparation<'a> {
    /// Borrows settled inputs. No native handle, shape, context, inventory or
    /// diagnostic string is allocated before the enclosing host admission.
    pub(super) fn inspect(
        owner: &'a DecoderCopyOwner,
        decoder: &'a PreparedResidentDecoderCopy<'a>,
        key: Option<&'a Array>,
        pending: Option<&'a Array>,
        pool: &'a WorkingMemoryPool,
        mechanisms: ResidentExecutionMechanisms,
    ) -> Result<Self, ResumeSourceCause> {
        let source = owner.storage_plan_with_sampling(decoder, pool, key, pending)?;
        let projection = projection::SnapshotProjectionPlan::inspect(decoder, key, None)?;
        let registration = RegisteredWorkspaceStorageLayout::new(projection.array_operand_count())?;
        let mut builder = WorkspaceCopyPreparationLayoutBuilder::new();
        let mut failure = None;
        projection.visit_operands(&mut |operand| {
            let crate::backend::runtime::cache::state::SnapshotOperand::Array(array) = operand
            else {
                return Ok(());
            };
            if failure.is_none() {
                failure = ProjectionSourceLayout::with_layout(array, |layout| {
                    builder.push_source(layout, &mechanisms)
                })
                .map_err(projection::SnapshotProjectionCause::Source)
                .map_err(ResumeSourceCause::Projection)
                .and_then(|result| result.map_err(ResumeSourceCause::Copy))
                .err();
            }
            Ok(())
        })?;
        if let Some(cause) = failure {
            return Err(cause);
        }
        let copy = builder.finish(projection.array_operand_count())?;
        let parts = [
            source.known_control_bytes(),
            projection.requested_bytes(),
            registration.requested_bytes(),
            copy.total_bytes(),
            ProjectedNativeStorage::iteration_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<PreparedResumeSource>(),
            size_of::<ResumeSourceCause>(),
            size_of::<Result<Self, ResumeSourceCause>>(),
            size_of::<Result<PreparedResumeSource, Error>>(),
            size_of::<Result<(), CopyError>>(),
            size_of::<Option<ResumeSourceCause>>(),
            size_of::<WorkspaceIsolatedCopyPreparation<'_, ResidentExecutionMechanisms>>(),
            size_of::<WorkspaceCopyAdmissionError>(),
            size_of::<Result<RegisteredWorkspaceCopy<StorageIdentity>, WorkspaceCopyAdmissionError>>(
            ),
            size_of::<(
                &mut WorkspaceCopyPreparationLayoutBuilder<MlxWorkspaceFactError>,
                &ResidentExecutionMechanisms,
            )>(),
            size_of::<(
                safemlx::AllocationIdentity,
                u64,
                eredu_nn::workspace::WorkspaceExistingStorage,
            )>(),
            size_of::<
                Result<Result<(), CopyError>, crate::backend::nn::workspace::ProjectionSourceError>,
            >(),
            BackendFailure::source_retention_peak_bytes::<PreparationFailure>()
                .ok_or(WorkingMemoryError::Overflow)?,
            cold_source::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            source,
            projection,
            registration,
            copy,
            pool,
            mechanisms,
            bytes,
        })
    }

    /// Source/copy metadata only. This array-only sealed program never builds
    /// a role table; the dense destination state and future equation Context
    /// account for their own projection constructors.
    pub(super) fn known_component_bytes(&self) -> usize {
        self.bytes
    }

    pub(super) fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<PreparedResumeSource, Error> {
        let result = (|| {
            let source = self.source.construct(host)?;
            let registered_source = source.pin_registered_with_host(self.pool, host)?;
            let projected = self
                .projection
                .construct_resume_arrays(self.mechanisms, host)?;
            let registered = self
                .registration
                .construct(
                    self.pool,
                    &projected.context,
                    projected
                        .native
                        .iter()
                        .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
                )
                .map_err(memory)?;
            let program = WorkspaceIsolatedCopyPlan::prepare_finite_with_layout(
                &projected.context,
                registered.borrowed_storage(),
                &projected.inputs,
                &self.mechanisms,
                self.copy,
            )
            .map_err(|cause| Error::Other(Box::new(cause)))?
            .construct()
            .map_err(|cause| Error::Other(Box::new(cause)))?;
            let copy_preparation = RegisteredWorkspaceCopy::bind(program, registered)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            Ok(PreparedResumeSource {
                copy_preparation,
                registered_source,
            })
        })();
        result.map_err(|cause| cold_source::retain_operation_failure(cause, host))
    }
}

pub(super) fn retain_failure(cause: ResumeSourceCause, host: &HostPreparationAuthority) -> Error {
    Error::StorageSource(BackendFailure::from_error(PreparationFailure {
        cause,
        _host: host.clone(),
    }))
}
