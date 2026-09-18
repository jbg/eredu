//! Bind actual source accounts once before selecting their complete quote union.
use crate::backend::{
    error::Error,
    nn::workspace::ProjectedNativeStorage,
    runtime::cache::state::{
        bind_unselected_completed_resident_source_priors, CompletedResidentSource,
        OriginalResidentSourceBinding,
    },
    OriginalCopyEnvironment,
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceContext, HostMetadataFunding};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

/// Every exact account/pin survives the quote, execution and completion owner.
/// Selecting metadata confers no native authority and never changes registration.
pub(in crate::composition::mlx::replicated_text) struct SourceBindings {
    _bindings: Vec<OriginalResidentSourceBinding>,
    _selection: WorkspaceBorrowedStorage,
    _prepared: Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>>,
    _host: HostPreparationAuthority,
}
impl SourceBindings {
    pub(in crate::composition::mlx::replicated_text) fn prepare(
        context: &WorkspaceContext,
        native: &[&ProjectedNativeStorage],
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
        completed: &[&CompletedResidentSource],
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
        host: HostPreparationAuthority,
    ) -> Result<Self, Error> {
        let frames = [
            size_of::<Self>(),
            size_of::<(
                Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
                eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout<()>,
                Result<eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout<()>, WorkingMemoryError>,
                Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>>,
                Result<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>, WorkingMemoryError>,
            )>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Vec<OriginalResidentSourceBinding>>(),
            size_of::<(
                &WorkspaceContext,
                &[&ProjectedNativeStorage],
                &[&CompletedResidentSource],
                &OriginalCopyEnvironment<'_>,
                &HostMetadataFunding,
                HostPreparationAuthority,
            )>(),
            size_of::<usize>(),
            size_of::<
                Result<
                    WorkspaceBorrowedStorage,
                    eredu_nn::workspace::WorkspaceBorrowedStorageError,
                >,
            >(),
            size_of::<std::slice::Iter<'_, &ProjectedNativeStorage>>(),
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let mut bindings = funding
            .metadata_vec(native.len())
            .map_err(Error::Neural)?;
        for storage in native {
            bindings.push(bind_unselected_completed_resident_source_priors(
                context,
                storage,
                completed,
                environment,
                funding,
                &host,
            )?);
        }
        let prepared = prepared.map(|source| {
            let layout = eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout::<()>::new_with_prepared_source(0, source)
                .map_err(Error::PrefillControl)?;
            funding.reserve_metadata(layout.requested_bytes()).map_err(Error::WorkspacePlanning)?;
            layout.construct_unselected_with_prepared_source(environment.pool(), context,
                std::iter::empty(), source.clone()).map_err(Error::PrefillControl)
        }).transpose()?;
        let prepared_roots = prepared.as_ref().map(|binding| binding.borrowed_storage().roots()).unwrap_or(&[]);
        let roots = bindings
            .iter()
            .try_fold(prepared_roots.len(), |n, binding| {
                n.checked_add(binding.borrowed().roots().len())
            })
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let iter = bindings
            .iter()
            .flat_map(|binding| binding.borrowed().roots()).chain(prepared_roots);
        funding
            .reserve_metadata(
                WorkspaceBorrowedStorage::construction_bytes(roots)
                    .and_then(|n| n.checked_add(size_of_val(&iter)))
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let selection = WorkspaceBorrowedStorage::new_finite(context, iter, roots)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        context
            .set_borrowed_storage_checked(selection.clone())
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        Ok(Self {
            _bindings: bindings,
            _prepared: prepared,
            _selection: selection,
            _host: host,
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
