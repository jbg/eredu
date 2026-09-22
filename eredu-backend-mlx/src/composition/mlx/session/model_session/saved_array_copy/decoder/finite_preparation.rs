//! Consumed finite metadata components of the shared saved-copy preparation.
use super::*;
use crate::backend::nn::workspace::{
    MlxMetalWorkspaceMechanisms, MlxWorkspacePreparationError, ProjectedNativeStorage,
    ProjectionSourceLayout,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::{
    WorkspaceCopyPreparationError, WorkspaceCopyPreparationLayout,
    WorkspaceCopyPreparationLayoutBuilder, WorkspaceIsolatedCopyPreparation,
};
use eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout;
use std::mem::{size_of, size_of_val};

type CopyError = WorkspaceCopyPreparationError<MlxWorkspacePreparationError>;

#[derive(Debug, thiserror::Error)]
pub(super) enum FinitePreparationCause {
    #[error(transparent)]
    Projection(#[from] preparation::SnapshotProjectionCause),
    #[error(transparent)]
    Storage(#[from] storage::StoragePreparationCause),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Copy(#[from] CopyError),
    #[error(transparent)]
    DecoderHost(#[from] eredu_runtime::working_memory::DecoderHostPreparationError),
    #[error(transparent)]
    NativeCopy(#[from] original::CopyPreparationCause),
    #[error("snapshot finite preparation component layout overflows")]
    Overflow,
}

/// Source-bound composition of the separate owning constructor plans. The
/// immutable source loans survive count and construction; the public native
/// snapshot hook remains gated until the complete joined worker is qualified.
pub(super) struct SnapshotFinitePreparation<'a> {
    storage: storage::DecoderStoragePlan<'a>,
    projection: preparation::SnapshotProjectionPlan<'a>,
    registration: RegisteredWorkspaceStorageLayout<StorageIdentity>,
    copy: WorkspaceCopyPreparationLayout,
    pool: &'a eredu_runtime::working_memory::MemoryLedger,
    mechanisms: MlxMetalWorkspaceMechanisms,
    bytes: usize,
    copy_requirements: original::CopyRequirements,
}

pub(super) struct PreparedSnapshotMetadata {
    pub(super) source_native: ProjectedNativeStorage,
    pub(super) program: WorkspaceIsolatedCopyPlan,
    pub(super) registered: RegisteredWorkspaceStorage<StorageIdentity>,
    // The retained source carrier holds the preparation authority through the
    // preceding metadata's destruction, including an abandoned successful plan.
    pub(super) complete_source: WorkingMemoryStorage<StorageIdentity>,
    pub(super) operand_count: usize,
    pub(super) copy_requirements: original::CopyRequirements,
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PreparationFailure {
    #[source]
    cause: FinitePreparationCause,
    // BackendFailure destroys its source allocation before this last token.
    _host: HostPreparationAuthority,
}

impl<'a> SnapshotFinitePreparation<'a> {
    /// Borrows the same prepared decoder and optional sampling operands used by
    /// construction. No projection, context, retained descriptor, shape or pin
    /// population is allocated by this query.
    pub(super) fn inspect(
        owner: &'a DecoderCopyOwner,
        decoder: &'a PreparedResidentDecoderCopy<'a>,
        key: Option<&'a Array>,
        pending: Option<&'a Array>,
        pool: &'a eredu_runtime::working_memory::MemoryLedger,
        mechanisms: MlxMetalWorkspaceMechanisms,
        backend: &MlxBackend<'_>,
    ) -> Result<Self, FinitePreparationCause> {
        let storage = owner.storage_plan(decoder, pool)?;
        let projection = preparation::SnapshotProjectionPlan::inspect(decoder, key, pending)?;
        let registration = RegisteredWorkspaceStorageLayout::new(projection.operand_count())?;
        let mut builder = WorkspaceCopyPreparationLayoutBuilder::new();
        let mut failure = None;
        let mut host_layout_controls = Some(0usize);
        projection.visit_operands(&mut |source| {
            if failure.is_none() {
                let result = match source {
                    crate::backend::runtime::cache::state::SnapshotOperand::Array(array) => {
                        ProjectionSourceLayout::with_layout(array, |layout| {
                            builder.push_source(layout, &mechanisms)
                        })
                    }
                    crate::backend::runtime::cache::state::SnapshotOperand::Host(host) => {
                        host_layout_controls = host_layout_controls.and_then(|n| {
                            n.checked_add(ProjectionSourceLayout::host_layout_control_bytes()?)
                        });
                        ProjectionSourceLayout::with_host_layout(host, |layout| {
                            builder.push_source(layout, &mechanisms)
                        })
                    }
                }
                .map_err(preparation::SnapshotProjectionCause::Source)
                .map_err(FinitePreparationCause::Projection)
                .and_then(|result| result.map_err(FinitePreparationCause::Copy));
                failure = result.err();
            }
            Ok(())
        })?;
        if let Some(cause) = failure {
            return Err(cause);
        }
        // Aliased operands remain separate destination requests. Distinct roots
        // can only reduce this bound; no source identity is inferred from count.
        let copy = builder.finish(projection.operand_count())?;
        if !pool.same_ledger(backend.memory_ledger()) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let copy_requirements = original::CopyRequirements::inspect(
            decoder,
            key,
            pending,
            backend,
            projection.operand_count(),
        )?;
        let parts = [
            usize::try_from(
                eredu_core::DomainMemoryRequirements::construction_backing_bytes(
                    pool.topology(),
                    0,
                )
                .map_err(WorkingMemoryError::from)?,
            )
            .map_err(|_| FinitePreparationCause::Overflow)?
            .checked_mul(2)
            .ok_or(FinitePreparationCause::Overflow)?,
            size_of::<eredu_core::DomainMemoryRequirements>()
                .checked_mul(2)
                .ok_or(FinitePreparationCause::Overflow)?,
            std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<
                    eredu_core::DomainMemoryRequirements,
                >())
                .map_err(|_| FinitePreparationCause::Overflow)?
                .0
                .pad_to_align()
                .size(),
            storage.known_control_bytes(),
            host_layout_controls.ok_or(FinitePreparationCause::Overflow)?,
            projection.requested_bytes(),
            registration.requested_bytes(),
            copy.total_bytes(),
            decoder.host_copy_preparation_bytes()?,
            cold_source::control_bytes().ok_or(FinitePreparationCause::Overflow)?,
            super::super::pending_input::control_bytes().ok_or(FinitePreparationCause::Overflow)?,
            owner::returned_control_bytes().ok_or(FinitePreparationCause::Overflow)?,
            copy_requirements.control_bytes(),
            size_of::<PreparedTextComponentsCopy<'_>>(),
            ProjectedNativeStorage::iteration_control_bytes()
                .ok_or(FinitePreparationCause::Overflow)?,
            size_of::<Self>(),
            size_of::<PreparedSnapshotMetadata>(),
            size_of::<FinitePreparationCause>(),
            size_of::<Result<Self, FinitePreparationCause>>(),
            size_of::<Result<PreparedSnapshotMetadata, FinitePreparationCause>>(),
            size_of::<Result<PreparedSnapshotMetadata, Error>>(),
            size_of::<Option<FinitePreparationCause>>(),
            size_of::<Result<(), CopyError>>(),
            size_of::<
                Result<Result<(), CopyError>, crate::backend::nn::workspace::ProjectionSourceError>,
            >(),
            size_of::<WorkspaceIsolatedCopyPreparation<'_, MlxMetalWorkspaceMechanisms>>(),
            size_of::<MlxMetalWorkspaceMechanisms>(),
            size_of::<(
                &mut WorkspaceCopyPreparationLayoutBuilder<MlxWorkspacePreparationError>,
                &MlxMetalWorkspaceMechanisms,
            )>(),
            size_of::<(
                safemlx::AllocationIdentity,
                u64,
                eredu_nn::workspace::WorkspaceExistingStorage,
            )>(),
            size_of::<HostPreparationAuthority>(),
            BackendFailure::source_retention_peak_bytes::<PreparationFailure>()
                .ok_or(FinitePreparationCause::Overflow)?,
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(FinitePreparationCause::Overflow)?;
        Ok(Self {
            storage,
            projection,
            registration,
            copy,
            pool,
            mechanisms,
            bytes,
            copy_requirements,
        })
    }

    /// Known metadata contribution only. The enclosing cold planner must add
    /// every remaining constructor before admitting an original snapshot.
    pub(super) fn known_component_bytes(&self) -> usize {
        self.bytes
    }

    /// Consumes the queried producers under the accepted enclosing host owner.
    /// Source identity is authenticated by the actual source pin and projection;
    /// the finite copy worker validates its inputs against this supplied layout.
    pub(super) fn construct(
        self,
        host: &HostPreparationAuthority,
    ) -> Result<PreparedSnapshotMetadata, Error> {
        let result = (|| -> Result<PreparedSnapshotMetadata, Error> {
            let source = self.storage.construct(host)?;
            let complete_source = source.pin_registered_with_host(self.pool, host)?;
            let projected = self.projection.construct(self.mechanisms, host)?;
            let operand_count = projected.inputs.len();
            let registered = self
                .registration
                .construct(
                    self.pool,
                    &projected.context,
                    projected.native.iter().map(|(id, _, root)| {
                        crate::backend::nn::workspace::registered_storage_row(id, root)
                    }),
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
            Ok(PreparedSnapshotMetadata {
                source_native: projected.native,
                program,
                registered,
                complete_source,
                operand_count,
                copy_requirements: self.copy_requirements,
            })
        })();
        result.map_err(|cause| cold_source::retain_operation_failure(cause, host))
    }
}

pub(super) fn retain_failure(
    cause: FinitePreparationCause,
    host: &HostPreparationAuthority,
) -> Error {
    Error::StorageSource(BackendFailure::from_error(PreparationFailure {
        cause,
        _host: host.clone(),
    }))
}
