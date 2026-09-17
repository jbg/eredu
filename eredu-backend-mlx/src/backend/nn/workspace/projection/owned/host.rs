//! Immutable host roots use the same sorted native-identity table as arrays.
use super::*;
use safemlx::{HostTransferDescriptor, ImmutableHostTransferBuffer};
use std::sync::Arc;

impl OwnedArrayProjection<'_> {
    /// Exact fixed rank/source import controls. The actual shape/root/row
    /// construction still uses the existing paid metadata constructors.
    pub(crate) fn host_import_control_bytes<const N: usize>() -> Option<usize> {
        Self::host_frame_bytes::<N>()?.checked_add(preparation::storage_control_bytes::<
            ProjectionNativeSource,
        >(size_of::<(
            &Arc<ImmutableHostTransferBuffer>,
            &Arc<ImmutableHostTransferBuffer>,
        )>())?)
    }
    fn host_frame_bytes<const N: usize>() -> Option<usize> {
        let frames = [
            size_of::<(&mut Self, &Arc<ImmutableHostTransferBuffer>)>(),
            size_of::<HostTransferDescriptor<N>>(),
            size_of::<Result<HostTransferDescriptor<N>, safemlx::HostTransferMetadataError>>(),
            size_of::<Result<WorkspaceTensor, ProjectionSourceError>>(),
            size_of::<Result<WorkspaceLayoutView<'_>, eredu_nn::workspace::WorkspaceLayoutError>>(),
            size_of::<Result<WorkspaceDtype, ProjectionSourceError>>(),
            size_of::<Result<WorkspaceExistingStorage, ProjectionSourceError>>(),
            HostTransferDescriptor::<N>::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Same exact source-side rank/root census as array projection. This does
    /// not import a value or create an owner, and duplicate roots remain shared.
    pub(crate) fn host_source_bytes<const N: usize>(
        host: &ImmutableHostTransferBuffer,
    ) -> Result<usize, ProjectionSourceError> {
        let descriptor = host.try_fixed_descriptor::<N>()?;
        WorkspaceLayoutView::new(descriptor.shape(), projected_dtype(descriptor.dtype())?)?;
        u64::try_from(descriptor.allocation().bytes())
            .map_err(|_| ProjectionSourceError::Overflow)?;
        Self::host_import_control_bytes::<N>()
            .and_then(|n| n.checked_add(WorkspaceExistingStorage::construction_bytes()?))
            .and_then(|n| {
                n.checked_add(WorkspaceTensor::imported_construction_bytes(
                    descriptor.shape().len(),
                )?)
            })
            .ok_or(ProjectionSourceError::Overflow)
    }
    /// Imports an actual retained host descriptor without materializing an
    /// array, copying payload or authorizing execution. A symbolic host value
    /// requires its explicit transfer consumer before a native equation uses it.
    pub(crate) fn project_host<const N: usize>(
        &mut self,
        host: &Arc<ImmutableHostTransferBuffer>,
    ) -> Result<WorkspaceTensor, ProjectionSourceError> {
        self.context
            .charge_metadata(Self::host_frame_bytes::<N>().ok_or(ProjectionSourceError::Overflow)?)
            .map_err(Error::from)?;
        let descriptor = host.try_fixed_descriptor::<N>()?;
        let dtype = projected_dtype(descriptor.dtype())?;
        let layout = WorkspaceLayoutView::new(descriptor.shape(), dtype)?
            .with_representation(preparation::projected_representation(descriptor.dtype()));
        let storage = preparation::storage_for(
            &mut self.storage.rows,
            &mut self.storage.known,
            Some(self.maximum),
            Some(descriptor.allocation()),
            self.context,
            || Ok(ProjectionNativeSource::host(Arc::clone(host))),
            |source| {
                if source.host.is_none() {
                    source.host = Some(Arc::clone(host));
                }
                Ok(())
            },
        )?;
        Ok(WorkspaceTensor::import_existing(
            layout,
            &storage,
            self.context,
        )?)
    }
}
