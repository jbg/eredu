//! Cold scalar source descriptors for the actual selected lookahead windows.
//! These describe populations; dynamic owning map/recipe/native storage is not
//! certified by the source count and cannot activate original execution.
use super::*;
mod declarations;
mod foreground;
mod supplementary;
use declarations::DeclarationCloneShape;
use eredu_runtime::{
    execution::ExecutionUnitLayout,
    residency::{ResidencyClosureError, ResidencyClosureSlot},
};
pub(crate) use foreground::ForegroundDiskIdentity;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of, num::NonZeroUsize};
pub(crate) use supplementary::SupplementaryResidencySource;

// Owning clone worker shared with retained ready-host snapshots. No map/node,
// manager or native backing is inferred from these declaration requests.
pub(super) fn binding_clone_payload_bytes(binding: &WeightBinding) -> Option<usize> {
    DeclarationCloneShape::binding_payload(binding)
}
pub(super) fn unit_clone_payload_bytes(unit: &OffloadUnit) -> Option<usize> {
    DeclarationCloneShape::bindings(unit.bindings())?
        .payload_bytes
        .checked_add(Layout::array::<u8>(unit.id().as_str().len()).ok()?.size())
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum OperationSourceFailure {
    #[error("residency source manager is busy")]
    Busy,
    #[error("residency source manager lock is poisoned")]
    Poisoned,
    #[error("residency source layout does not match the selected units")]
    Layout,
    #[error("residency source layout overflow")]
    Overflow,
    #[error("residency source storage allocation failed: {0}")]
    Reserve(#[source] TryReserveError),
    #[error("residency source canonical closure: {0:?}")]
    Closure(ResidencyClosureError),
    #[error("original source allocator: {0}")]
    Allocation(#[source] safemlx::OriginalBufferCause),
    #[error("original source storage: {0}")]
    Memory(#[source] eredu_runtime::working_memory::WorkingMemoryError),
}

/// Scalar extents from one actual window and its finite canonical closure.
/// Byte fields describe source strings/values, not allocator/container charges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WindowPopulation {
    /// Exact selected request slice, retained from ExecutionUnitLayout::window_range.
    pub(crate) named: super::named_arrays::NamedStorageLayout,
    pub(crate) request_start: usize,
    pub(crate) request_end: usize,
    pub(crate) requested: usize,
    pub(crate) controller_units: usize,
    pub(crate) maximum_id_bytes: usize,
    pub(crate) requested_id_bytes: usize,
    pub(crate) units: usize,
    pub(crate) unit_id_bytes: usize,
    pub(crate) bindings: usize,
    pub(crate) physical_bindings: usize,
    pub(crate) aliases: usize,
    pub(crate) local_aliases: usize,
    pub(crate) binding_name_bytes: usize,
    pub(crate) alias_owner_name_bytes: usize,
    pub(crate) physical_bytes: u64,
    pub(crate) recipe_pending: usize,
    pub(crate) recipe_materializations: usize,
    pub(crate) recipe_depth: usize,
    pub(crate) declaration_clone_bytes: usize,
    pub(crate) declaration_clone_allocations: usize,
}

/// Scalar window rows whose final shared backing retires before source custody.
/// No raw Arc/Weak escapes. Ordinary rows retain their caller-owned behavior.
#[derive(Clone, Debug)]
pub(crate) struct OperationWindows {
    values: Arc<Vec<WindowPopulation>>,
    custody: Option<ManagerCustody>,
}
impl std::ops::Deref for OperationWindows {
    type Target = Vec<WindowPopulation>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
#[derive(Debug, Clone)]
enum OperationSelection {
    Host(HostCopyIdentity),
    Foreground(ForegroundDiskIdentity),
}
#[derive(Debug, Clone)]
pub(crate) struct OriginalResidencySource {
    pub(crate) controller_units: usize,
    named: super::named_arrays::NamedStorageLayout,
    windows: OperationWindows,
    selection: Option<OperationSelection>,
    manager_owner: Option<ManagerWeak>,
}
impl OriginalResidencySource {
    pub(crate) fn named_layout(&self) -> super::named_arrays::NamedStorageLayout {
        self.named
    }
    pub(crate) fn foreground_identity(&self) -> Option<&ForegroundDiskIdentity> {
        match self.selection.as_ref()? {
            OperationSelection::Foreground(value) => Some(value),
            OperationSelection::Host(_) => None,
        }
    }
    pub(crate) fn windows(&self) -> &[WindowPopulation] {
        &self.windows
    }
    pub(crate) fn window_owner(&self) -> OperationWindows {
        self.windows.clone()
    }
    pub(crate) fn manager_owner(&self) -> Option<ManagerWeak> {
        self.manager_owner.clone()
    }
    fn window_owner_bytes() -> Option<usize> {
        usize::try_from(
            eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<Vec<WindowPopulation>>(),
            )
            .ok()?,
        )
        .ok()
    }
    /// Pre-grant source construction, using actual controller and selected-window
    /// counts. Scratch, output backing and shared owner are each included once;
    /// this conservative constructor sum takes no scratch-retirement credit.
    pub(crate) fn constructor_storage_bytes(
        controller_units: usize,
        windows: usize,
    ) -> Option<u64> {
        let mut bytes = ResidencyClosureSlot::layout(controller_units)?
            .size()
            .checked_add(Layout::array::<WindowPopulation>(windows).ok()?.size())?
            .checked_add(Self::window_owner_bytes()?)?;
        for control in [
            size_of::<Self>(),
            size_of::<OperationWindows>(),
            size_of::<Option<OperationSelection>>(),
            size_of::<Option<ManagerWeak>>(),
            size_of::<Option<ManagerCustody>>(),
            size_of::<Vec<ResidencyClosureSlot>>(),
            size_of::<Vec<WindowPopulation>>(),
            size_of::<WindowPopulation>(),
            size_of::<OperationSourceFailure>(),
            size_of::<Result<Self, OperationSourceFailure>>(),
            size_of::<Result<(), OperationSourceFailure>>(),
            size_of::<Result<(), Self>>(),
            size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::slice::Iter<'static, OffloadUnitId>>(),
            size_of::<(
                &ResidencyManager,
                &[OffloadUnitId],
                &ExecutionUnitLayout,
                NonZeroUsize,
                &HostCopyWorkspace,
            )>(),
        ] {
            bytes = bytes.checked_add(control)?;
        }
        u64::try_from(bytes).ok()
    }
    pub(crate) fn retained_control_bytes(&self) -> Option<u64> {
        // The original manager source already owns these allocations. Every
        // window clone independently retains that account, without new backing.
        if self.windows.custody.is_some() {
            return Some(0);
        }
        let bytes = Layout::array::<WindowPopulation>(self.windows.capacity())
            .ok()?
            .size()
            .checked_add(Self::window_owner_bytes()?)?
            .checked_add(size_of::<Result<Self, OperationSourceFailure>>())?;
        u64::try_from(bytes).ok()
    }
}

impl WindowPopulation {
    pub(super) fn collect(
        controller: &ResidencyController,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<Self, OperationSourceFailure> {
        let closure = controller
            .operation_closure(roots, scratch)
            .map_err(OperationSourceFailure::Closure)?;
        let mut value = Self {
            requested: roots.len(),
            controller_units: controller.ledger().plan().units().len(),
            maximum_id_bytes: controller
                .ledger()
                .plan()
                .units()
                .iter()
                .map(|unit| unit.id().as_str().len())
                .max()
                .unwrap_or(0),
            units: closure.len(),
            named: super::named_arrays::NamedStorageLayout::from_units(closure.units())
                .map_err(|_| OperationSourceFailure::Overflow)?,
            ..Self::default()
        };
        for id in roots {
            value.requested_id_bytes = value
                .requested_id_bytes
                .checked_add(id.as_str().len())
                .ok_or(OperationSourceFailure::Overflow)?;
        }
        for unit in closure.units() {
            value.unit_id_bytes = value
                .unit_id_bytes
                .checked_add(unit.id().as_str().len())
                .ok_or(OperationSourceFailure::Overflow)?;
            let declarations = DeclarationCloneShape::bindings(unit.bindings())
                .ok_or(OperationSourceFailure::Overflow)?;
            value.declaration_clone_bytes = value
                .declaration_clone_bytes
                .checked_add(declarations.payload_bytes)
                .ok_or(OperationSourceFailure::Overflow)?;
            value.declaration_clone_allocations = value
                .declaration_clone_allocations
                .checked_add(declarations.allocations)
                .ok_or(OperationSourceFailure::Overflow)?;
            let recipes =
                super::operation_population::MaterializationPopulation::bindings(unit.bindings())
                    .ok_or(OperationSourceFailure::Overflow)?;
            value.physical_bindings = value
                .physical_bindings
                .checked_add(recipes.bindings)
                .ok_or(OperationSourceFailure::Overflow)?;
            value.recipe_pending = value
                .recipe_pending
                .checked_add(recipes.recipe_pending_weight_bound)
                .ok_or(OperationSourceFailure::Overflow)?;
            value.recipe_materializations = value
                .recipe_materializations
                .checked_add(recipes.recipe_materialization_bound)
                .ok_or(OperationSourceFailure::Overflow)?;
            value.recipe_depth = value.recipe_depth.max(recipes.recipe_depth);
            for binding in unit.bindings() {
                value.bindings = value
                    .bindings
                    .checked_add(1)
                    .ok_or(OperationSourceFailure::Overflow)?;
                value.binding_name_bytes = value
                    .binding_name_bytes
                    .checked_add(binding.name().len())
                    .ok_or(OperationSourceFailure::Overflow)?;
                if binding.is_alias() {
                    value.aliases = value
                        .aliases
                        .checked_add(1)
                        .ok_or(OperationSourceFailure::Overflow)?;
                    let (owner_id, owner) = controller
                        .binding_owner_borrowed(unit.id(), binding)
                        .ok_or(OperationSourceFailure::Closure(
                        ResidencyClosureError::InvalidOwner,
                    ))?;
                    value.alias_owner_name_bytes = value
                        .alias_owner_name_bytes
                        .checked_add(owner.name().len())
                        .ok_or(OperationSourceFailure::Overflow)?;
                    if owner_id == unit.id() {
                        value.local_aliases = value
                            .local_aliases
                            .checked_add(1)
                            .ok_or(OperationSourceFailure::Overflow)?;
                    }
                } else {
                    value.physical_bytes = value
                        .physical_bytes
                        .checked_add(binding.expected_bytes())
                        .ok_or(OperationSourceFailure::Overflow)?;
                }
            }
        }
        Ok(value)
    }
}
impl ResidencyManager {
    /// Read-only identity of the immutable manager/source declaration owner.
    pub(crate) fn same_source_manager(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }
    /// ManagerInner owns immutable unit declarations and exact source dispatch.
    /// The retained Weak pins that header, preventing address reuse. Comparing
    /// it requires no lock, source callback, recount, native work or allocation.
    pub(crate) fn matches_operation_source(&self, owner: &ManagerWeak) -> bool {
        std::ptr::eq(self.inner.as_ptr(), owner.as_ptr())
    }

    /// The finite window rows must be the actual source-accounted constructor
    /// result retained by this manager, not equal ordinary diagnostic rows.
    pub(super) fn owns_original_operation_windows(&self, windows: &OperationWindows) -> bool {
        self.original_sources()
            .any(|source| self.owns_source_windows(source, windows))
    }

    fn original_sources(&self) -> impl Iterator<Item = &OriginalResidencySource> {
        self.inner
            .original_operation_source
            .get()
            .into_iter()
            .chain(self.inner.background_operation_source.get())
            .chain(
                self.inner
                    .supplementary_source
                    .get()
                    .map(|source| source.source()),
            )
    }
    fn owns_source_windows(
        &self,
        source: &OriginalResidencySource,
        windows: &OperationWindows,
    ) -> bool {
        windows.custody.is_some()
            && source.windows.custody.is_some()
            && source.selection.is_some()
            && source
                .manager_owner
                .as_ref()
                .is_some_and(|owner| self.matches_operation_source(owner))
            && Arc::ptr_eq(&source.windows.values, &windows.values)
    }

    /// Borrow the actual source-owned Host lookahead declarations. Its depth
    /// was selected by the original dense manager constructor, independently
    /// of the existing Device-window source. No request reconstruction occurs.
    pub(crate) fn background_operation_source(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> Result<&OriginalResidencySource, OperationSourceFailure> {
        let depth = NonZeroUsize::new(depth).ok_or(OperationSourceFailure::Layout)?;
        let source = self.inner.background_operation_source.get().ok_or(OperationSourceFailure::Layout)?;
        if !source.matches_selection(ids, layout, depth)
            || !self.owns_source_windows(source, &source.windows) {
            return Err(OperationSourceFailure::Layout);
        }
        Ok(source)
    }

    /// Runs during ordinary policy construction, before any request inspection.
    /// Retains only scalar rows. No recipe/key/selection/native/source is cloned.
    /// A failed optional description is retained by the policy; it does not
    /// replace an ordinary construction/materialization error or change its order.
    pub(crate) fn prepare_original_operation_source(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> Result<OriginalResidencySource, OperationSourceFailure> {
        let depth = NonZeroUsize::new(depth).ok_or(OperationSourceFailure::Layout)?;
        if ids.len() != layout.len() {
            return Err(OperationSourceFailure::Layout);
        }
        if let Some(source) = self.inner.original_operation_source.get() {
            if !source.matches_selection(ids, layout, depth) {
                return Err(OperationSourceFailure::Layout);
            }
            return Ok(source.clone());
        }
        if self.original_source_custody().is_some() {
            // Original policy construction only borrows the already funded
            // source. It cannot fall back to ordinary preparation allocations.
            return Err(OperationSourceFailure::Layout);
        }
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(OperationSourceFailure::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(OperationSourceFailure::Poisoned);
            }
        };
        let mut source = OriginalResidencySource::prepare(&state.control, ids, layout, depth)?;
        source.manager_owner = Some(self.inner.downgrade());
        Ok(source)
    }

    /// Called once by the admitted source initializer, after its host snapshot
    /// exists and before publishing the manager. The caller's source account
    /// surrounds all partial construction and any escaped error.
    pub(in crate::backend::runtime::residency::manager) fn initialize_original_operation_source(
        &mut self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> Result<(), OperationSourceFailure> {
        let depth = NonZeroUsize::new(depth).ok_or(OperationSourceFailure::Layout)?;
        if self.inner.original_operation_source.get().is_some() {
            return Err(OperationSourceFailure::Layout);
        }
        let snapshot = self
            .inner
            .host_workspace
            .get()
            .ok_or(OperationSourceFailure::Layout)?;
        let source = self.build_original_host_operation_source(ids, layout, depth, snapshot)?;
        self.inner
            .original_operation_source
            .set(source)
            .map_err(|_| OperationSourceFailure::Layout)
    }

    pub(in crate::backend::runtime::residency::manager) fn build_original_host_operation_source(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
        snapshot: &HostCopyWorkspace,
    ) -> Result<OriginalResidencySource, OperationSourceFailure> {
        let custody = self
            .original_source_custody()
            .ok_or(OperationSourceFailure::Layout)?;
        let selection = snapshot
            .prepared_identity()
            .ok_or(OperationSourceFailure::Layout)?;
        if selection.layout() != layout
            || selection.depth() != depth.get()
            || snapshot.units().len() != ids.len()
            || snapshot
                .units()
                .iter()
                .zip(ids)
                .any(|(unit, id)| unit.id() != id)
        {
            return Err(OperationSourceFailure::Layout);
        }
        let state = match self.inner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(OperationSourceFailure::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(OperationSourceFailure::Poisoned);
            }
        };
        let mut source = OriginalResidencySource::prepare_with_custody(
            &state.control,
            ids,
            layout,
            depth,
            Some(custody),
        )?;
        source.manager_owner = Some(self.inner.downgrade());
        source.selection = Some(OperationSelection::Host(selection.clone()));
        Ok(source)
    }
}

impl OriginalResidencySource {
    fn matches_selection(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
    ) -> bool {
        let Some(selection) = &self.selection else {
            return false;
        };
        match selection {
            OperationSelection::Host(selection) => {
                selection.layout() == layout
                    && selection.depth() == depth.get()
                    && selection.sources().len() == ids.len()
                    && selection
                        .sources()
                        .iter()
                        .zip(ids)
                        .all(|((unit, _), id)| unit.id() == id)
            }
            OperationSelection::Foreground(selection) => {
                selection.matches_selection(ids, layout, depth.get())
            }
        }
    }
    fn prepare(
        controller: &ResidencyController,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
    ) -> Result<Self, OperationSourceFailure> {
        Self::prepare_with_custody(controller, ids, layout, depth, None)
    }
    fn prepare_with_custody(
        controller: &ResidencyController,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
        custody: Option<ManagerCustody>,
    ) -> Result<Self, OperationSourceFailure> {
        if ids.len() != layout.len() {
            return Err(OperationSourceFailure::Layout);
        }
        let n = controller.units().len();
        ResidencyClosureSlot::layout(n).ok_or(OperationSourceFailure::Overflow)?;
        Layout::array::<WindowPopulation>(ids.len())
            .map_err(|_| OperationSourceFailure::Overflow)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(n)
            .map_err(OperationSourceFailure::Reserve)?;
        scratch.resize(n, ResidencyClosureSlot::default());
        let all = controller
            .operation_closure(ids, &mut scratch)
            .map_err(OperationSourceFailure::Closure)?;
        let named = super::named_arrays::NamedStorageLayout::from_units(all.units())
            .map_err(|_| OperationSourceFailure::Overflow)?;
        let mut windows = Vec::new();
        windows
            .try_reserve_exact(ids.len())
            .map_err(OperationSourceFailure::Reserve)?;
        for ordinal in 0..ids.len() {
            let range = layout
                .window_range(ordinal, depth)
                .ok_or(OperationSourceFailure::Layout)?;
            let mut population =
                WindowPopulation::collect(controller, &ids[range.clone()], &mut scratch)?;
            population.request_start = range.start;
            population.request_end = range.end;
            windows.push(population);
        }
        // The accepted manager source pays the pre-grant constructor sum,
        // including this scratch. Later policy clones retain only shared rows.
        drop(scratch);
        Ok(OriginalResidencySource {
            controller_units: n,
            named,
            windows: OperationWindows {
                values: Arc::new(windows),
                custody,
            },
            selection: None,
            manager_owner: None, // The owning manager binds the completed cold source.
        })
    }
}

#[cfg(test)]
mod tests;
