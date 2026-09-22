//! Requested storage for the original branch of the shared snapshot builder.
use super::*;
use eredu_runtime::working_memory::OriginalHostMetadataCustody;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

impl HostCopyWorkspace {
    /// Inspect exact selected definitions and their retained native physical
    /// shapes before the manager source account constructs host buffers. Aliases
    /// supply their canonical owner's same shape; no provider or native query is
    /// called here. Later construction must use the same selected definitions.
    ///
    /// Geometry/layout/name identity storage is separately quoted by
    /// HostCopyIdentity::requested_bytes and is not included a second time.
    pub(crate) fn constructor_storage_bytes<'a>(
        definitions: &[OffloadUnit],
        shape_for: impl Fn(&OffloadUnitId, &WeightBinding) -> Option<&'a [i32]>,
    ) -> Result<u64, HostCopyWorkspaceError> {
        let overflow = || HostCopyWorkspaceError::Storage(WorkingMemoryError::Overflow);
        let layout = || {
            let copies = definitions.iter().try_fold(0usize, |count, definition| {
                count.checked_add(definition.bindings().len())
            })?;
            let mut payload = Layout::array::<HostCopyUnit>(definitions.len())
                .ok()?
                .size()
                .checked_add(Layout::array::<HostCopyBinding>(copies).ok()?.size())?;
            for definition in definitions {
                payload = payload.checked_add(
                    super::super::operation_source::unit_clone_payload_bytes(definition)?,
                )?;
                for binding in definition.bindings() {
                    payload = payload.checked_add(
                        super::super::operation_source::binding_clone_payload_bytes(binding)?,
                    )?;
                }
            }
            Some(payload)
        };
        let mut payload = layout().ok_or_else(overflow)?;
        // Shape loans come only from the actual prepared native output metadata.
        // The complete clone request uses length, while the source owns any
        // spare capacity of its independently retained metadata shape vector.
        for definition in definitions {
            for binding in definition.bindings() {
                let shape = shape_for(definition.id(), binding).ok_or_else(|| {
                    HostCopyWorkspaceError::unknown("host snapshot has no prepared output shape")
                })?;
                let shape_bytes = Layout::array::<i32>(shape.len())
                    .map_err(|_| overflow())?
                    .size();
                payload = payload.checked_add(shape_bytes).ok_or_else(overflow)?;
            }
        }
        let shared = OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<
            HostCopyWorkspaceData,
        >())
        .map_err(HostCopyWorkspaceError::Storage)?;
        let controls = [
            size_of::<(&ResidencyManager, &ManagerState, &OffloadUnitId)>(),
            size_of::<Option<&ResidentHostOwner>>(),
            std::mem::size_of_val(&shape_for),
            size_of::<&[OffloadUnit]>(),
            size_of::<HostCopyWorkspace>(),
            size_of::<HostCopyWorkspaceData>(),
            size_of::<HostCopyWorkspaceError>(),
            safemlx::StreamCopyPlan::<()>::capture_control_bytes()
                .map_err(HostCopyWorkspaceError::StreamCopy)?,
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<(
                safemlx::DeviceType,
                &[i32],
                std::slice::Iter<'_, i32>,
                Option<i32>,
                bool,
            )>(),
            size_of::<Result<HostCopyWorkspace, HostCopyWorkspaceError>>(),
            size_of::<HostCopyUnit>(),
            size_of::<HostCopyBinding>(),
            size_of::<OffloadUnit>(),
            size_of::<WeightBinding>(),
            size_of::<HostTransferMetadataSnapshot>(),
            size_of::<Vec<i32>>(),
            size_of::<RetainedHostBuffer>(),
            size_of::<super::super::ManagerWeak>(),
            size_of::<super::super::ManagerCustody>(),
            size_of::<Option<super::super::ManagerCustody>>(),
            size_of::<Arc<HostCopyWorkspaceData>>(),
            size_of::<Vec<HostCopyUnit>>(),
            size_of::<Vec<HostCopyBinding>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::slice::Iter<'_, OffloadUnit>>(),
            size_of::<std::slice::Iter<'_, WeightBinding>>(),
            size_of::<std::slice::Iter<'_, i32>>(),
            size_of::<Option<&[i32]>>(),
            size_of::<Option<&eredu_checkpoint::recipe::RecipeMetadata>>(),
            size_of::<Option<&HostTransferMetadataSnapshot>>(),
            size_of::<Option<(&OffloadUnitId, &WeightBinding)>>(),
            size_of::<crate::backend::nn::workspace::MlxWorkspaceFactError>(),
            size_of::<Result<u64, crate::backend::nn::workspace::MlxWorkspaceFactError>>(),
            size_of::<Result<(u64, u64), HostCopyWorkspaceError>>(),
            size_of::<AllocationInfo>(),
            size_of::<Option<(safemlx::AllocationIdentity, u64)>>(),
        ];
        let bytes = controls
            .iter()
            .try_fold(payload, |bytes, control| bytes.checked_add(*control))
            .ok_or_else(overflow)?;
        shared
            .checked_add(u64::try_from(bytes).map_err(|_| overflow())?)
            .ok_or_else(overflow)
    }
}
