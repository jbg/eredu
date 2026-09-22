//! Shared metadata-only identity and geometry for one admitted host snapshot.
use super::*;
use eredu_core::WorkspaceBound;
use eredu_runtime::{working_memory::completed_layerwise_window_bytes, ExecutionUnitLayout};
use std::{alloc::Layout, num::NonZeroUsize};

pub(crate) type HostSourceIdentities =
    Vec<(OffloadUnit, Vec<(String, safemlx::AllocationIdentity, u64)>)>;
const ASSUMPTIONS: &str = "maximum completed group-bounded host-copy window; preceding consumers and transfers settle before replacement; existing sources and constructor/equation storage are separate";

struct IdentityData {
    layout: ExecutionUnitLayout,
    depth: usize,
    sources: HostSourceIdentities,
    materialization: WorkspaceBound,
}

/// Contains no buffers, source providers or native manager handle. Legacy
/// identity-only quotes can retain this metadata without retaining host payload.
#[derive(Clone)]
pub(crate) struct HostCopyIdentity {
    value: Arc<IdentityData>,
    _custody: super::super::ManagerCustody,
}
impl std::fmt::Debug for HostCopyIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCopyIdentity")
            .field("layout", self.layout())
            .field("depth", &self.depth())
            .field("sources", self.sources())
            .finish()
    }
}
impl PartialEq for HostCopyIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
            || (self.layout() == other.layout()
                && self.depth() == other.depth()
                && self.sources() == other.sources())
    }
}
impl Eq for HostCopyIdentity {}
impl HostCopyIdentity {
    pub(crate) fn layout(&self) -> &ExecutionUnitLayout {
        &self.value.layout
    }
    pub(crate) fn depth(&self) -> usize {
        self.value.depth
    }
    pub(crate) fn sources(&self) -> &HostSourceIdentities {
        &self.value.sources
    }
    pub(crate) fn materialization(&self) -> &WorkspaceBound {
        &self.value.materialization
    }

    /// Exact requested immutable identity backing. The source constructor also
    /// includes the host-copy data/metadata producer and its error controls.
    pub(crate) fn requested_bytes<'a>(
        layout: &ExecutionUnitLayout,
        definitions: impl ExactSizeIterator<Item = &'a OffloadUnit>,
    ) -> Option<usize> {
        use eredu_runtime::working_memory::OriginalHostMetadataCustody;
        let mut bytes = usize::try_from(
            OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<IdentityData>())
                .ok()?,
        )
        .ok()?
        .checked_add(layout.cloned_payload_bytes()?)?
        .checked_add(ASSUMPTIONS.len())?
        .checked_add(
            Layout::array::<(OffloadUnit, Vec<(String, safemlx::AllocationIdentity, u64)>)>(
                definitions.len(),
            )
            .ok()?
            .size(),
        )?;
        for definition in definitions {
            bytes = bytes
                .checked_add(HostCopyWorkspace::cloned_definition_payload_bytes(
                    definition,
                )?)?
                .checked_add(
                    Layout::array::<(String, safemlx::AllocationIdentity, u64)>(
                        definition.bindings().len(),
                    )
                    .ok()?
                    .size(),
                )?;
            for binding in definition.bindings() {
                bytes = bytes.checked_add(binding.name().len())?;
            }
        }
        // Named identity constructor, return, initialization and fixed error
        // transports are separate from the host-copy producer's controls.
        for control in [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<IdentityData>(),
            std::mem::size_of::<HostSourceIdentities>(),
            std::mem::size_of::<Vec<(String, safemlx::AllocationIdentity, u64)>>(),
            std::mem::size_of::<OffloadUnit>(),
            std::mem::size_of::<String>(),
            std::mem::size_of::<WorkspaceBound>(),
            std::mem::size_of::<std::collections::TryReserveError>(),
            std::mem::size_of::<Result<Self, HostCopyWorkspaceError>>(),
            std::mem::size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
            std::mem::size_of::<Option<super::super::ManagerCustody>>(),
            std::mem::size_of::<HostCopyWorkspace>(),
            std::mem::size_of::<Result<(), HostCopyWorkspaceError>>(),
            std::mem::size_of::<Result<(), HostCopyWorkspace>>(),
            std::mem::size_of::<(
                &ResidencyManager,
                &[OffloadUnitId],
                &ExecutionUnitLayout,
                usize,
                NativeAllocationFacts,
            )>(),
        ] {
            bytes = bytes.checked_add(control)?;
        }
        Some(bytes)
    }

    fn construct(
        copies: &HostCopyWorkspace,
        layout: &ExecutionUnitLayout,
        depth: usize,
        custody: super::super::ManagerCustody,
    ) -> Result<Self, HostCopyWorkspaceError> {
        let reserve = |cause| {
            HostCopyWorkspaceError::Storage(WorkingMemoryError::ControlStorageReserve(cause))
        };
        let depth_value = NonZeroUsize::new(depth)
            .ok_or_else(|| HostCopyWorkspaceError::mismatch("zero host-copy window depth"))?;
        let bytes =
            completed_layerwise_window_bytes(layout, depth_value, copies.units().len(), |i| {
                Some(copies.units()[i].fresh_capacity_bytes())
            })
            .map_err(|_| HostCopyWorkspaceError::Storage(WorkingMemoryError::Overflow))?
            .ok_or_else(|| {
                HostCopyWorkspaceError::unknown("host-copy window has an unknown member")
            })?;
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(copies.units().len())
            .map_err(reserve)?;
        for unit in copies.units() {
            let mut rows = Vec::new();
            rows.try_reserve_exact(copies.copies(unit).len())
                .map_err(reserve)?;
            for copy in copies.copies(unit) {
                let allocation = copy.source_allocation();
                rows.push((
                    copy.name().to_owned(),
                    allocation.identity(),
                    u64::try_from(allocation.bytes()).map_err(|_| {
                        HostCopyWorkspaceError::Storage(WorkingMemoryError::Overflow)
                    })?,
                ));
            }
            sources.push((unit.definition().clone(), rows));
        }
        Ok(Self {
            value: Arc::new(IdentityData {
                layout: layout.clone(),
                depth,
                sources,
                materialization: WorkspaceBound::bounded(bytes, ASSUMPTIONS),
            }),
            _custody: custody,
        })
    }
}

impl ResidencyManager {
    /// Called exactly once while the source-accounted manager is unpublished.
    /// Its actual source initializer prepays both snapshot and identity layouts.
    /// Later lookup only validates locked owner rows and shares these objects.
    pub(in crate::backend::runtime::residency::manager) fn initialize_host_workspace(
        &mut self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
        allocation: NativeAllocationFacts,
    ) -> Result<(), HostCopyWorkspaceError> {
        if self.inner.host_workspace.get().is_some() {
            return Err(HostCopyWorkspaceError::mismatch(
                "host snapshot already initialized",
            ));
        }
        let copies = self.build_original_host_workspace(ids, layout, depth, allocation)?;
        self.inner
            .host_workspace
            .set(copies)
            .map_err(|_| HostCopyWorkspaceError::mismatch("host snapshot initialization raced"))
    }

    pub(in crate::backend::runtime::residency::manager) fn build_original_host_workspace(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
        allocation: NativeAllocationFacts,
    ) -> Result<HostCopyWorkspace, HostCopyWorkspaceError> {
        if ids.len() != layout.len() {
            return Err(HostCopyWorkspaceError::mismatch(
                "host snapshot layout differs from selected units",
            ));
        }
        let custody = self
            .original_source_custody()
            .ok_or_else(|| HostCopyWorkspaceError::unknown("host snapshot lacks source custody"))?;
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| HostCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        let mut copies = self.host_copy_workspace_locked(&state, ids, allocation)?;
        copies.geometry = Some(HostCopyIdentity::construct(
            &copies,
            layout,
            depth,
            custody.clone(),
        )?);
        copies.custody = Some(custody);
        Ok(copies)
    }
}
