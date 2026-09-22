//! Source-owned ordinal metadata for the existing foreground unit worker.
use super::*;
use eredu_core::WorkspaceBound;
use eredu_runtime::residency::ResidencyClosure;
use eredu_runtime::working_memory::{
    MemoryLedger, WorkingMemoryError, WorkspaceParameterLifetime, WorkspaceParameterOwner,
};
use std::mem::size_of_val;
use std::ops::Range;

const ASSUMPTIONS: &str = "completed foreground windows with exact canonical output owners; cross-unit alias owner units remain separately covered; source backing and request constructors are separate";
struct Unit {
    source: usize,
    rows: Range<usize>,
    capacity: u64,
    persistent: bool,
}
struct Row {
    owner_unit: usize,
    owner_row: usize,
    bytes: u64,
    capacity: u64,
}
struct Data {
    units: Vec<Unit>,
    rows: Vec<Row>,
    requested: Vec<usize>,
    layout: ExecutionUnitLayout,
    depth: usize,
    materialization: WorkspaceBound,
    source: ForegroundDiskDescriptors,
    domain: eredu_core::SharedStorageAccountingId,
    destination: safemlx::StreamCopyPlan<()>,
}
/// Shares exact source metadata, not an active disk receipt or source grant.
#[derive(Clone)]
pub(crate) struct ForegroundDiskIdentity {
    value: Arc<Data>,
    _custody: ManagerCustody,
}
impl std::fmt::Debug for ForegroundDiskIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForegroundDiskIdentity")
            .field("layout", self.layout())
            .field("depth", &self.depth())
            .finish_non_exhaustive()
    }
}
impl PartialEq for ForegroundDiskIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}
impl Eq for ForegroundDiskIdentity {}
impl ForegroundDiskIdentity {
    pub(crate) fn destination_device_index(&self) -> i32 {
        self.value.destination.device_index()
    }
    pub(crate) fn destination_device_type(&self) -> safemlx::DeviceType {
        self.value.destination.device_type()
    }
    pub(crate) fn layout(&self) -> &ExecutionUnitLayout {
        &self.value.layout
    }
    pub(crate) fn depth(&self) -> usize {
        self.value.depth
    }
    pub(crate) fn source(&self) -> &ForegroundDiskDescriptors {
        &self.value.source
    }
    pub(crate) fn validate_operation_custody(
        &self,
        controls: &eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Result<(), WorkingMemoryError> {
        if controls.matches_accounting_owner(&self.value.domain) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(crate) fn validate_request_custody(
        &self,
        controls: &eredu_runtime::working_memory::OriginalTextControlGuard,
    ) -> Result<(), WorkingMemoryError> {
        if controls
            .metadata_custody()
            .matches_accounting_owner(&self.value.domain)
        {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(crate) fn materialization(&self) -> &WorkspaceBound {
        &self.value.materialization
    }
    pub(crate) fn unit_count(&self) -> usize {
        self.value.units.len()
    }
    pub(crate) fn unit(&self, index: usize) -> Option<&OffloadUnit> {
        self.source()
            .units()
            .nth(self.value.units.get(index)?.source)
    }
    pub(crate) fn unit_capacity(&self, index: usize) -> Option<u64> {
        Some(self.value.units.get(index)?.capacity)
    }
    pub(crate) fn requested_unit(&self, ordinal: usize) -> Option<usize> {
        self.value.requested.get(ordinal).copied()
    }
    pub(crate) fn matches_selection(
        &self,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> bool {
        self.layout() == layout
            && self.depth() == depth
            && ids.len() == self.value.requested.len()
            && ids.iter().enumerate().all(|(ordinal, id)| {
                self.requested_unit(ordinal)
                    .and_then(|unit| self.unit(unit))
                    .is_some_and(|unit| unit.id() == id)
            })
    }
    pub(crate) fn persistent_units(&self) -> impl Iterator<Item = &OffloadUnitId> + Clone {
        self.value
            .units
            .iter()
            .enumerate()
            .filter(|(_, unit)| unit.persistent)
            .map(|(index, _)| self.unit(index).expect("validated source unit").id())
    }
    pub(crate) fn qualifier_control_bytes() -> Option<usize> {
        [
            size_of::<(&Self, &ResidencyManager, &OperationWindows)>(),
            size_of::<(&ResidencyManager, &Self, &OperationWindows)>(),
            size_of::<Option<&Self>>(),
            size_of::<Option<&ForegroundDiskDescriptors>>(),
            size_of::<Option<&OriginalResidencySource>>(),
            size_of::<Option<&OperationSelection>>(),
            size_of::<(&ForegroundDiskDescriptors, &ForegroundDiskDescriptors)>(),
            size_of::<bool>(),
            size_of::<Option<usize>>(),
            size_of::<(
                &Self,
                &eredu_runtime::working_memory::OriginalTextControlGuard,
            )>(),
            size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub(crate) fn row(
        &self,
        unit: usize,
        row: usize,
    ) -> Option<(
        &WeightBinding,
        &[i32],
        safemlx::Dtype,
        safemlx::HostTransferArrayLayout,
        u64,
        u64,
        WorkspaceParameterOwner,
    )> {
        let selected = self.value.units.get(unit)?;
        if row >= selected.rows.len() {
            return None;
        }
        let definition = self.unit(unit)?;
        let binding = definition.bindings().get(row)?;
        let data = self.value.rows.get(selected.rows.start.checked_add(row)?)?;
        let (shape, dtype, output_layout) = self
            .source()
            .native_copy_output(definition.id(), binding.name())?;
        Some((
            binding,
            shape,
            dtype,
            output_layout,
            data.bytes,
            data.capacity,
            WorkspaceParameterOwner {
                unit: data.owner_unit,
                row: data.owner_row,
                lifetime: if self.value.units.get(data.owner_unit)?.persistent {
                    WorkspaceParameterLifetime::Trace
                } else {
                    WorkspaceParameterLifetime::Invocation
                },
            },
        ))
    }

    fn construct(
        control: &ResidencyController,
        destination: safemlx::StreamCopyPlan<()>,
        source: &ForegroundDiskDescriptors,
        runtime: &safemlx::PreparedInputRuntime,
        domain: &eredu_core::SharedStorageAccountingId,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
        custody: ManagerCustody,
    ) -> Result<Self, OperationSourceFailure> {
        let bad = || OperationSourceFailure::Layout;
        let overflow = || OperationSourceFailure::Overflow;
        if ids.len() != layout.len() || !control.ledger().initialized() {
            return Err(bad());
        }
        let count = control.units().len();
        let rows = control
            .units()
            .try_fold(0usize, |sum, unit| sum.checked_add(unit.bindings().len()))
            .ok_or_else(overflow)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(count)
            .map_err(OperationSourceFailure::Reserve)?;
        scratch.resize(count, ResidencyClosureSlot::default());
        let closure = control
            .operation_closure(ids, &mut scratch)
            .map_err(OperationSourceFailure::Closure)?;
        let mut value = Data {
            units: Vec::new(),
            rows: Vec::new(),
            requested: Vec::new(),
            layout: layout.clone(),
            depth: depth.get(),
            materialization: WorkspaceBound::bounded(0, ASSUMPTIONS),
            source: source.clone(),
            domain: domain.clone(),
            destination,
        };
        value
            .units
            .try_reserve_exact(count)
            .map_err(OperationSourceFailure::Reserve)?;
        value
            .rows
            .try_reserve_exact(rows)
            .map_err(OperationSourceFailure::Reserve)?;
        value
            .requested
            .try_reserve_exact(ids.len())
            .map_err(OperationSourceFailure::Reserve)?;
        let mut row_start = 0usize;
        for unit in closure.units() {
            let source_ordinal = source
                .units()
                .position(|retained| retained == unit)
                .ok_or_else(bad)?;
            let end = row_start
                .checked_add(unit.bindings().len())
                .ok_or_else(overflow)?;
            value.units.push(Unit {
                source: source_ordinal,
                rows: row_start..end,
                capacity: 0,
                persistent: false,
            });
            row_start = end;
        }
        for id in ids {
            value.requested.push(
                value
                    .units
                    .iter()
                    .position(|unit| {
                        source
                            .units()
                            .nth(unit.source)
                            .is_some_and(|unit| unit.id() == id)
                    })
                    .ok_or_else(bad)?,
            );
        }
        for ordinal in 0..value.units.len() {
            let definition = source
                .units()
                .nth(value.units[ordinal].source)
                .ok_or_else(bad)?;
            for binding in definition.bindings() {
                let (owner_id, owner) = if binding.is_alias() {
                    control
                        .binding_owner_borrowed(definition.id(), binding)
                        .ok_or_else(bad)?
                } else {
                    (definition.id(), binding)
                };
                let owner_unit = value
                    .units
                    .iter()
                    .position(|unit| {
                        source
                            .units()
                            .nth(unit.source)
                            .is_some_and(|unit| unit.id() == owner_id)
                    })
                    .ok_or_else(bad)?;
                let owner_definition = source
                    .units()
                    .nth(value.units[owner_unit].source)
                    .ok_or_else(bad)?;
                let owner_row = owner_definition
                    .bindings()
                    .iter()
                    .position(|binding| binding.name() == owner.name())
                    .ok_or_else(bad)?;
                let metadata = source
                    .output(definition.id(), binding.name())
                    .ok_or_else(bad)?;
                let (shape, _) = source
                    .native_output(definition.id(), binding.name())
                    .ok_or_else(bad)?;
                if !super::super::host_workspace::copy_shape_is_supported(
                    destination.device_type(),
                    shape,
                ) || metadata.byte_len() != binding.expected_bytes()
                    || shape.len() != metadata.shape().len()
                    || !shape
                        .iter()
                        .zip(metadata.shape())
                        .all(|(&a, &b)| usize::try_from(a).ok() == Some(b))
                {
                    return Err(bad());
                }
                let bytes = metadata.byte_len();
                let physical = safemlx::OriginalBufferBudget::request_layout(
                    runtime,
                    usize::try_from(bytes).map_err(|_| overflow())?,
                )
                .map_err(OperationSourceFailure::Allocation)?;
                let capacity = u64::try_from(physical.capacity()).map_err(|_| overflow())?;
                if owner_unit == ordinal && owner.name() == binding.name() {
                    value.units[ordinal].capacity = value.units[ordinal]
                        .capacity
                        .checked_add(capacity)
                        .ok_or_else(overflow)?;
                }
                if owner_unit != ordinal {
                    value.units[owner_unit].persistent = true;
                }
                value.rows.push(Row {
                    owner_unit,
                    owner_row,
                    bytes,
                    capacity,
                });
            }
        }
        // These copies were published from the admitted source before this
        // constructor. Pinned Device units cannot be evicted by the shared
        // ledger, so their backing is existing residency, not a future copy.
        // Keep row capacities for parameter geometry and exclude only the
        // additional materialization amount. Disk alias owners remain covered.
        for unit in &mut value.units {
            let definition = source.units().nth(unit.source).ok_or_else(bad)?;
            let spec = control
                .ledger()
                .plan()
                .unit(definition.id())
                .ok_or_else(bad)?;
            if spec.tier() == MemoryTier::Device
                && spec.policy() == eredu_core::residency::ResidencyPolicy::Pinned
            {
                if !control
                    .ledger()
                    .is_resident(definition.id(), MemoryTier::Device)
                    .map_err(|_| bad())?
                {
                    return Err(bad());
                }
                unit.capacity = 0;
                unit.persistent = true;
            }
        }
        let persistent = value
            .units
            .iter()
            .filter(|unit| unit.persistent)
            .try_fold(0u64, |sum, unit| sum.checked_add(unit.capacity))
            .ok_or_else(overflow)?;
        let window = eredu_runtime::working_memory::completed_layerwise_window_bytes(
            layout,
            depth,
            ids.len(),
            |ordinal| {
                let unit = value.units.get(*value.requested.get(ordinal)?)?;
                Some(if unit.persistent { 0 } else { unit.capacity })
            },
        )
        .map_err(|_| overflow())?
        .ok_or_else(bad)?;
        match &mut value.materialization {
            WorkspaceBound::Bounded { bytes, .. } => {
                *bytes = window.checked_add(persistent).ok_or_else(overflow)?
            }
            _ => unreachable!("bounded source construction"),
        }
        Ok(Self {
            value: Arc::new(value),
            _custody: custody,
        })
    }
}
impl ResidencyManager {
    /// Exact source constructor payload and fixed transports. No query allocates,
    /// reads payload, initializes an allocator or grants request/native authority.
    pub(crate) fn original_foreground_operation_source_bytes(
        pool: &MemoryLedger,
        layout: &ExecutionUnitLayout,
        units: &[OffloadUnit],
        selected_count: usize,
    ) -> Result<u64, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        if selected_count != layout.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let runtime = crate::backend::managed_memory::input_allocator::borrow_admitted(pool)?;
        let query = safemlx::OriginalBufferBudget::request_layout(&runtime, 0)
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let rows = units
            .iter()
            .try_fold(0usize, |sum, unit| sum.checked_add(unit.bindings().len()))
            .ok_or_else(overflow)?;
        let arrays = Layout::array::<Unit>(units.len())
            .map_err(|_| overflow())?
            .size()
            .checked_add(Layout::array::<Row>(rows).map_err(|_| overflow())?.size())
            .and_then(|n| n.checked_add(Layout::array::<usize>(selected_count).ok()?.size()))
            .and_then(|n| n.checked_add(ResidencyClosureSlot::layout(units.len())?.size()))
            .and_then(|n| n.checked_add(layout.cloned_payload_bytes()?))
            .and_then(|n| n.checked_add(ASSUMPTIONS.len()))
            .and_then(|n| n.checked_add(query.control_bytes().checked_mul(rows.checked_add(1)?)?))
            .ok_or_else(overflow)?;
        let shared =
            eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<Data>(),
            )?;
        let fixed = [
            size_of::<Data>(),
            safemlx::StreamCopyPlan::<()>::capture_control_bytes()
                .map_err(|_| WorkingMemoryError::UnknownBound)?,
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<(
                safemlx::DeviceType,
                &[i32],
                std::slice::Iter<'_, i32>,
                Option<i32>,
                bool,
            )>(),
            size_of::<ForegroundDiskIdentity>(),
            size_of::<Unit>(),
            size_of::<Row>(),
            size_of::<OperationSelection>(),
            size_of::<Option<OperationSelection>>(),
            size_of::<Vec<ResidencyClosureSlot>>(),
            size_of::<ResidencyClosure<'_>>(),
            size_of::<safemlx::PreparedInputRuntime>(),
            size_of::<WorkingMemoryError>(),
            size_of::<Result<ForegroundDiskIdentity, OperationSourceFailure>>(),
            size_of::<Option<ForegroundDiskIdentity>>(),
            size_of::<Result<(), OperationSourceFailure>>(),
            size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
            size_of::<ManagerCustody>(),
            size_of::<WorkspaceBound>(),
            size_of::<Option<u64>>(),
            size_of::<Range<usize>>(),
            size_of::<eredu_core::SharedStorageAccountingId>(),
            size_of::<(
                &ResidencyManager,
                &MemoryLedger,
                &[OffloadUnitId],
                &ExecutionUnitLayout,
                NonZeroUsize,
            )>(),
        ];
        let fixed = fixed.into_iter().try_fold(
            arrays
                .checked_add(size_of_val(&fixed))
                .ok_or_else(overflow)?,
            |sum, n| sum.checked_add(n).ok_or_else(overflow),
        )?;
        OriginalResidencySource::constructor_storage_bytes(units.len(), layout.len())
            .ok_or_else(overflow)?
            .checked_add(shared)
            .and_then(|n| n.checked_add(u64::try_from(fixed).ok()?))
            .ok_or_else(overflow)
    }
    pub(crate) fn initialize_original_foreground_operation_source(
        &mut self,
        pool: &MemoryLedger,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> Result<(), OperationSourceFailure> {
        let depth = NonZeroUsize::new(depth).ok_or(OperationSourceFailure::Layout)?;
        if self.inner.original_operation_source.get().is_some() {
            return Err(OperationSourceFailure::Layout);
        }
        let source = self.build_original_foreground_operation_source(pool, ids, layout, depth)?;
        self.inner
            .original_operation_source
            .set(source)
            .map_err(|_| OperationSourceFailure::Layout)
    }
    /// Same source/geometry constructor, selected once for Host lookahead.
    /// Its population is separately included in the actual manager load source.
    pub(in crate::backend::runtime::residency::manager) fn initialize_original_background_operation_source(
        &mut self,
        pool: &MemoryLedger,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: usize,
    ) -> Result<(), OperationSourceFailure> {
        let depth = NonZeroUsize::new(depth).ok_or(OperationSourceFailure::Layout)?;
        if self.inner.background_operation_source.get().is_some() {
            return Err(OperationSourceFailure::Layout);
        }
        let source = self.build_original_foreground_operation_source(pool, ids, layout, depth)?;
        self.inner
            .background_operation_source
            .set(source)
            .map_err(|_| OperationSourceFailure::Layout)
    }

    pub(in crate::backend::runtime::residency::manager) fn build_original_foreground_operation_source(
        &self,
        pool: &MemoryLedger,
        ids: &[OffloadUnitId],
        layout: &ExecutionUnitLayout,
        depth: NonZeroUsize,
    ) -> Result<OriginalResidencySource, OperationSourceFailure> {
        let custody = self
            .original_source_custody()
            .ok_or(OperationSourceFailure::Layout)?;
        let descriptor = self
            .original_foreground_disk_descriptors()
            .ok_or(OperationSourceFailure::Layout)?;
        custody
            .validate_pool(pool)
            .map_err(OperationSourceFailure::Memory)?;
        let runtime = crate::backend::managed_memory::input_allocator::borrow_admitted(pool)
            .map_err(OperationSourceFailure::Memory)?;
        let state = self.inner.state.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => OperationSourceFailure::Busy,
            std::sync::TryLockError::Poisoned(_) => OperationSourceFailure::Poisoned,
        })?;
        // This manager has completed initial source publication but has not
        // escaped the accepted initializer. The only native call in construct
        // is the pure physical-population query: it reads retained allocator
        // facts and neither invokes runtime callbacks nor constructs storage.
        let identity = ForegroundDiskIdentity::construct(
            &state.control,
            safemlx::StreamCopyPlan::<()>::capture(&state.device_stream)
                .map_err(|_| OperationSourceFailure::Layout)?,
            descriptor,
            &runtime,
            pool.shared_storage_accounting_id(),
            ids,
            layout,
            depth,
            custody.clone(),
        )?;
        let mut source = OriginalResidencySource::prepare_with_custody(
            &state.control,
            ids,
            layout,
            depth,
            Some(custody),
        )?;
        source.manager_owner = Some(self.inner.downgrade());
        source.selection = Some(OperationSelection::Foreground(identity));
        Ok(source)
    }
    pub(crate) fn original_foreground_workspace(&self) -> Option<&ForegroundDiskIdentity> {
        let source = self.inner.original_operation_source.get()?;
        let OperationSelection::Foreground(identity) = source.selection.as_ref()? else {
            return None;
        };
        self.original_foreground_disk_descriptors()?
            .same_source(identity.source())
            .then_some(identity)
    }
    pub(crate) fn original_foreground_persistent_units(
        &self,
    ) -> Option<impl Iterator<Item = &OffloadUnitId> + Clone> {
        Some(self.original_foreground_workspace()?.persistent_units())
    }
    pub(crate) fn owns_original_foreground_workspace(
        &self,
        identity: &ForegroundDiskIdentity,
        windows: &OperationWindows,
    ) -> bool {
        self.original_sources().any(|source| {
            self.owns_source_windows(source, windows)
                && matches!(source.selection.as_ref(), Some(OperationSelection::Foreground(actual))
                    if actual == identity && self.original_foreground_disk_descriptors()
                        .is_some_and(|descriptors| descriptors.same_source(actual.source())))
        })
    }
}
