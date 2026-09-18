//! Final named value destinations prepared from the actual retained controller.
//! Names are shared once per request; canonical cells are shared by alias rows.
//! No shared owner contains a thread-affine native observer.
use super::{ManagerOwner, ManagerWeak, ResidencyManager, ResidentArrays};
use eredu_core::residency::OffloadUnitId;
use eredu_runtime::{
    residency::{OffloadUnit, ResidencyClosureSlot},
    working_memory::OriginalOperationMetadataCustody,
};
use safemlx::Array;
use std::{
    alloc::Layout,
    collections::{btree_map, BTreeMap, TryReserveError},
    ops::{Deref, Index, Range},
    sync::{atomic::AtomicUsize, Arc, OnceLock, TryLockError, Weak},
};

/// Fixed failures of a source-bound prepared named-array destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedArrayError {
    /// The actual manager is borrowed by another operation.
    ManagerBusy,
    /// The actual manager lock is poisoned.
    ManagerPoisoned,
    /// The destination belongs to another retained manager.
    ForeignManager,
    /// The retained declaration or incomplete source destination is invalid.
    InvalidSource,
    /// The declared unit has no unused prepared destination.
    UnknownUnit,
    /// The borrowed output name is absent from the retained declaration.
    UnknownName,
    /// The destination already contains a value for this name.
    DuplicateValue,
    /// A declared canonical or alias row has not been filled.
    MissingValue,
    /// The unpublished destination has an escaped shared owner.
    SharedDestination,
    /// A checked source layout cannot be represented.
    Overflow,
}

impl std::fmt::Display for NamedArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "named residency destination: {self:?}")
    }
}
impl std::error::Error for NamedArrayError {}

/// Requested heap layouts for actual retained names and final value cells.
/// Catalog storage is shared once per request; destination storage is per real
/// window transfer slot. Allocator rounding and native Array payloads remain
/// separate. These descriptive facts convey no activation authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NamedStorageLayout {
    pub(crate) catalog_requested_bytes: usize,
    pub(crate) catalog_allocations: usize,
    pub(crate) destination_requested_bytes: usize,
    pub(crate) destination_allocations: usize,
    pub(crate) units: usize,
    pub(crate) rows: usize,
    pub(crate) physical_cells: usize,
}

// Pinned Rust 1.98 alloc::sync::ArcInner: repr(C, align(2)), strong and weak
// Atomic<usize>, followed by T. No allocator-private size or BTree ABI inferred.
fn arc_layout<T>() -> Result<Layout, NamedArrayError> {
    Layout::new::<[AtomicUsize; 2]>()
        .align_to(2)
        .and_then(|counts| {
            counts
                .extend(Layout::new::<T>())
                .map(|(layout, _)| layout.pad_to_align())
        })
        .map_err(|_| NamedArrayError::Overflow)
}
fn add_layout(
    bytes: &mut usize,
    allocations: &mut usize,
    layout: Layout,
) -> Result<(), NamedArrayError> {
    *bytes = bytes
        .checked_add(layout.size())
        .ok_or(NamedArrayError::Overflow)?;
    if layout.size() != 0 {
        *allocations = allocations
            .checked_add(1)
            .ok_or(NamedArrayError::Overflow)?;
    }
    Ok(())
}
fn array_layout<T>(len: usize) -> Result<Layout, NamedArrayError> {
    Layout::array::<T>(len).map_err(|_| NamedArrayError::Overflow)
}

impl NamedStorageLayout {
    pub(super) fn from_units<'a>(
        units: impl Iterator<Item = &'a OffloadUnit>,
    ) -> Result<Self, NamedArrayError> {
        let mut facts = Self::default();
        add_layout(
            &mut facts.catalog_requested_bytes,
            &mut facts.catalog_allocations,
            arc_layout::<NameCatalog>()?,
        )?;
        for unit in units {
            facts.units = facts
                .units
                .checked_add(1)
                .ok_or(NamedArrayError::Overflow)?;
            facts.rows = facts
                .rows
                .checked_add(unit.bindings().len())
                .ok_or(NamedArrayError::Overflow)?;
            add_layout(
                &mut facts.catalog_requested_bytes,
                &mut facts.catalog_allocations,
                array_layout::<u8>(unit.id().as_str().len())?,
            )?;
            add_layout(
                &mut facts.destination_requested_bytes,
                &mut facts.destination_allocations,
                arc_layout::<ResidentArrays>()?,
            )?;
            add_layout(
                &mut facts.destination_requested_bytes,
                &mut facts.destination_allocations,
                array_layout::<NamedSlot>(unit.bindings().len())?,
            )?;
            for binding in unit.bindings() {
                add_layout(
                    &mut facts.catalog_requested_bytes,
                    &mut facts.catalog_allocations,
                    array_layout::<u8>(binding.name().len())?,
                )?;
                if !binding.is_alias() {
                    facts.physical_cells = facts
                        .physical_cells
                        .checked_add(1)
                        .ok_or(NamedArrayError::Overflow)?;
                    add_layout(
                        &mut facts.destination_requested_bytes,
                        &mut facts.destination_allocations,
                        arc_layout::<CanonicalArray>()?,
                    )?;
                }
            }
        }
        add_layout(
            &mut facts.catalog_requested_bytes,
            &mut facts.catalog_allocations,
            array_layout::<CatalogUnit>(facts.units)?,
        )?;
        add_layout(
            &mut facts.catalog_requested_bytes,
            &mut facts.catalog_allocations,
            array_layout::<CatalogName>(facts.rows)?,
        )?;
        add_layout(
            &mut facts.destination_requested_bytes,
            &mut facts.destination_allocations,
            array_layout::<Option<PreparedNamedArrays>>(facts.units)?,
        )?;
        Ok(facts)
    }
}

struct CatalogUnit {
    id: String,
    names: Range<usize>,
}
struct CatalogName {
    name: String,
    alias: bool,
    canonical: usize,
}
struct NameCatalog {
    manager: ManagerWeak,
    units: Vec<CatalogUnit>,
    names: Vec<CatalogName>,
    complete: bool,
}

/// Each alias retains custody after its Arc, including the allocation's final
/// deallocation. Custody inside the Arc's payload would drop too early.
#[derive(Clone)]
pub(crate) struct NameCatalogOwner {
    value: Arc<NameCatalog>,
    custody: OriginalOperationMetadataCustody,
}

pub(crate) enum NamePreparationCause {
    Source(NamedArrayError),
    Reserve(TryReserveError),
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum NamedPreparationSource {
    #[error(transparent)]
    Source(#[from] NamedArrayError),
    #[error(transparent)]
    Reserve(#[from] TryReserveError),
}
/// The actual partial final allocation and every initialized name stay owned.
/// No source error conversion may discard this prefix before retaining custody.
pub(crate) struct NamePreparationError {
    pub(crate) cause: NamePreparationCause,
    pub(crate) prefix: NameCatalogOwner,
}

impl NamePreparationError {
    pub(crate) fn into_source(self) -> NamedPreparationSource {
        match self.cause {
            NamePreparationCause::Source(cause) => cause.into(),
            NamePreparationCause::Reserve(cause) => cause.into(),
        }
    }
}

impl ResidencyManager {
    /// Same actual closure and retained name population as construction, with
    /// caller-owned topology scratch. No names/keys/maps/arrays are allocated.
    pub(crate) fn named_storage_layout(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<NamedStorageLayout, NamedArrayError> {
        let state = self.inner.state.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => NamedArrayError::ManagerBusy,
            TryLockError::Poisoned(_) => NamedArrayError::ManagerPoisoned,
        })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(|_| NamedArrayError::InvalidSource)?;
        NamedStorageLayout::from_units(closure.units())
    }

    /// Closed cold source loan: the manager itself supplies the immutable
    /// declarations and canonical closure. No tensor acquisition participates.
    pub(crate) fn prepare_name_catalog(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        custody: OriginalOperationMetadataCustody,
    ) -> Result<NameCatalogOwner, NamePreparationError> {
        let mut prefix = NameCatalogOwner {
            value: Arc::new(NameCatalog {
                manager: self.inner.downgrade(),
                units: Vec::new(),
                names: Vec::new(),
                complete: false,
            }),
            custody,
        };
        let result = (|| {
            let state = self.inner.state.try_lock().map_err(|error| {
                NamePreparationCause::Source(match error {
                    TryLockError::WouldBlock => NamedArrayError::ManagerBusy,
                    TryLockError::Poisoned(_) => NamedArrayError::ManagerPoisoned,
                })
            })?;
            let closure = state
                .control
                .operation_closure(roots, scratch)
                .map_err(|_| NamePreparationCause::Source(NamedArrayError::InvalidSource))?;
            let rows = closure.units().try_fold(0usize, |n, unit| {
                n.checked_add(unit.bindings().len())
                    .ok_or(NamePreparationCause::Source(NamedArrayError::InvalidSource))
            })?;
            let value =
                Arc::get_mut(&mut prefix.value).expect("catalog has not exposed another alias");
            value
                .units
                .try_reserve_exact(closure.len())
                .map_err(NamePreparationCause::Reserve)?;
            value
                .names
                .try_reserve_exact(rows)
                .map_err(NamePreparationCause::Reserve)?;
            for unit in closure.units() {
                let mut id = String::new();
                id.try_reserve_exact(unit.id().as_str().len())
                    .map_err(NamePreparationCause::Reserve)?;
                id.push_str(unit.id().as_str());
                let start = value.names.len();
                for binding in unit.bindings() {
                    let (owner, canonical) = if binding.is_alias() {
                        state
                            .control
                            .binding_owner_borrowed(unit.id(), binding)
                            .ok_or(NamePreparationCause::Source(NamedArrayError::InvalidSource))?
                    } else {
                        (unit.id(), binding)
                    };
                    let mut coordinate = 0usize;
                    let mut found = None;
                    for candidate in closure.units() {
                        if candidate.id() == owner {
                            let position = candidate
                                .bindings()
                                .iter()
                                .position(|row| row.name() == canonical.name() && !row.is_alias())
                                .ok_or(NamePreparationCause::Source(
                                    NamedArrayError::InvalidSource,
                                ))?;
                            found =
                                Some(coordinate.checked_add(position).ok_or(
                                    NamePreparationCause::Source(NamedArrayError::Overflow),
                                )?);
                            break;
                        }
                        coordinate = coordinate
                            .checked_add(candidate.bindings().len())
                            .ok_or(NamePreparationCause::Source(NamedArrayError::Overflow))?;
                    }
                    let mut name = String::new();
                    name.try_reserve_exact(binding.name().len())
                        .map_err(NamePreparationCause::Reserve)?;
                    name.push_str(binding.name());
                    value.names.push(CatalogName {
                        name,
                        alias: binding.is_alias(),
                        canonical: found
                            .ok_or(NamePreparationCause::Source(NamedArrayError::InvalidSource))?,
                    });
                }
                value.units.push(CatalogUnit {
                    id,
                    names: start..value.names.len(),
                });
            }
            value.complete = true;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(prefix),
            Err(cause) => Err(NamePreparationError { cause, prefix }),
        }
    }
}

impl NameCatalogOwner {
    fn coordinate(&self, unit: &OffloadUnitId, name: &str) -> Result<usize, NamedArrayError> {
        let unit = &self.value.units[self.unit_index(unit)?];
        let index = self.value.names[unit.names.clone()]
            .binary_search_by(|row| row.name.as_str().cmp(name))
            .map_err(|_| NamedArrayError::UnknownName)?;
        Ok(unit.names.start + index)
    }
    fn same_canonical(&self, coordinate: usize, other: &Self, other_coordinate: usize) -> bool {
        if !self.value.manager.ptr_eq(&other.value.manager) {
            return false;
        }
        let Some(row) = self.value.names.get(coordinate) else {
            return false;
        };
        let Some(other_row) = other.value.names.get(other_coordinate) else {
            return false;
        };
        if row.alias || other_row.alias || row.name != other_row.name {
            return false;
        }
        let unit = self
            .value
            .units
            .iter()
            .find(|unit| unit.names.contains(&coordinate));
        let other_unit = other
            .value
            .units
            .iter()
            .find(|unit| unit.names.contains(&other_coordinate));
        matches!((unit, other_unit), (Some(unit), Some(other)) if unit.id == other.id)
    }
    fn unit_index(&self, id: &OffloadUnitId) -> Result<usize, NamedArrayError> {
        if !self.value.complete {
            return Err(NamedArrayError::InvalidSource);
        }
        self.value
            .units
            .binary_search_by(|row| row.id.as_str().cmp(id.as_str()))
            .map_err(|_| NamedArrayError::UnknownUnit)
    }
    fn validate_unit(
        &self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<usize, NamedArrayError> {
        if !self.value.manager.ptr_eq(&manager.downgrade()) {
            return Err(NamedArrayError::ForeignManager);
        }
        let index = self.unit_index(unit.id())?;
        let names = &self.value.names[self.value.units[index].names.clone()];
        if names.len() != unit.bindings().len()
            || names
                .iter()
                .zip(unit.bindings())
                .any(|(row, binding)| row.name != binding.name() || row.alias != binding.is_alias())
        {
            return Err(NamedArrayError::InvalidSource);
        }
        Ok(index)
    }

    /// The caller supplies the genuine per-window unit loan from this same
    /// manager. The final unit Arc and physical cells are allocated here once.
    pub(super) fn prepare_unit(
        &self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<PreparedNamedArrays, PreparedNamedError> {
        let mut ready = PreparedNamedArrays {
            arrays: NamedArrays::Empty,
            owner: ResidentArraysOwner {
                arrays: Arc::new(ResidentArrays {
                    arrays: NamedArrays::Empty,
                }),
                custody: Some(self.custody.clone()),
                source_custody: super::ManagerCustody::default(),
            },
        };
        let result = (|| {
            let index = self
                .validate_unit(manager, unit)
                .map_err(PreparedNamedCause::Source)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(unit.bindings().len())
                .map_err(PreparedNamedCause::Reserve)?;
            // The vector is installed before any subsequent cell allocation.
            ready.arrays = NamedArrays::Prepared(PreparedNames {
                catalog: self.clone(),
                unit: index,
                values,
            });
            let NamedArrays::Prepared(table) = &mut ready.arrays else {
                unreachable!()
            };
            for (offset, binding) in unit.bindings().iter().enumerate() {
                table.values.push(if binding.is_alias() {
                    NamedSlot::Alias(None)
                } else {
                    NamedSlot::Canonical(CanonicalArrayOwner {
                        cell: Arc::new(CanonicalArray {
                            publication: OnceLock::new(),
                            value: OnceLock::new(),
                            host_source: OnceLock::new(),
                            coordinate: table.catalog.value.units[index].names.start + offset,
                            catalog: self.clone(),
                        }),
                        custody: self.custody.clone(),
                    })
                });
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(ready),
            Err(cause) => Err(PreparedNamedError {
                cause,
                prefix: ready,
            }),
        }
    }
}

pub(super) enum PreparedNamedCause {
    Source(NamedArrayError),
    Reserve(TryReserveError),
}
pub(super) struct PreparedNamedError {
    pub(super) cause: PreparedNamedCause,
    pub(super) prefix: PreparedNamedArrays,
}

/// One actual acquire-window closure, allocated before its transfer activates.
/// Entries are consumed once by source identity, independently of warm hits.
pub(crate) struct PreparedNamedWindow {
    units: Vec<Option<PreparedNamedArrays>>,
    catalog: NameCatalogOwner,
    complete: bool,
}
pub(crate) enum NamedWindowPreparationCause {
    Source(NamedArrayError),
    Reserve(TryReserveError),
    Unit(PreparedNamedError),
}
pub(crate) struct NamedWindowPreparationError {
    pub(crate) cause: NamedWindowPreparationCause,
    pub(crate) prefix: PreparedNamedWindow,
}
impl NamedWindowPreparationError {
    pub(crate) fn into_source(self) -> NamedPreparationSource {
        match self.cause {
            NamedWindowPreparationCause::Source(cause) => cause.into(),
            NamedWindowPreparationCause::Reserve(cause) => cause.into(),
            NamedWindowPreparationCause::Unit(error) => match error.cause {
                PreparedNamedCause::Source(cause) => cause.into(),
                PreparedNamedCause::Reserve(cause) => cause.into(),
            },
        }
    }
}
impl NameCatalogOwner {
    /// A closed manager loan supplies every declaration and the real canonical
    /// closure. No caller-populated name table can authorize these destinations.
    pub(crate) fn prepare_window(
        &self,
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
    ) -> Result<PreparedNamedWindow, NamedWindowPreparationError> {
        let mut prefix = PreparedNamedWindow {
            units: Vec::new(),
            catalog: self.clone(),
            complete: false,
        };
        let result = (|| {
            if !self.value.manager.ptr_eq(&manager.inner.downgrade()) {
                return Err(NamedWindowPreparationCause::Source(
                    NamedArrayError::ForeignManager,
                ));
            }
            let state = manager.inner.state.try_lock().map_err(|error| {
                NamedWindowPreparationCause::Source(match error {
                    TryLockError::WouldBlock => NamedArrayError::ManagerBusy,
                    TryLockError::Poisoned(_) => NamedArrayError::ManagerPoisoned,
                })
            })?;
            let closure = state
                .control
                .operation_closure(roots, scratch)
                .map_err(|_| NamedWindowPreparationCause::Source(NamedArrayError::InvalidSource))?;
            prefix
                .units
                .try_reserve_exact(closure.len())
                .map_err(NamedWindowPreparationCause::Reserve)?;
            for unit in closure.units() {
                let ready = self
                    .prepare_unit(&manager.inner, unit)
                    .map_err(NamedWindowPreparationCause::Unit)?;
                prefix.units.push(Some(ready));
            }
            prefix.complete = true;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(prefix),
            Err(cause) => Err(NamedWindowPreparationError { cause, prefix }),
        }
    }
}
impl PreparedNamedWindow {
    pub(super) fn take_unit(
        &mut self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<PreparedNamedArrays, NamedArrayError> {
        if !self.complete {
            return Err(NamedArrayError::InvalidSource);
        }
        let index = self.catalog.validate_unit(manager, unit)?;
        let slot = self
            .units
            .iter_mut()
            .find(|slot| {
                slot.as_ref().is_some_and(|ready| {
            matches!(&ready.arrays, NamedArrays::Prepared(table) if table.unit == index)
        })
            })
            .ok_or(NamedArrayError::UnknownUnit)?;
        // The catalogue and exact unit were checked before removing a slot.
        Ok(slot.take().expect("matched populated named destination"))
    }
}

pub(crate) struct CanonicalArray {
    publication: OnceLock<super::super::storage::PublishedAllocation>,
    value: OnceLock<Array>,
    // A source-backed device publication retains its actual host owner here,
    // not in Host-tier cache state. Canonical aliases inherit this same cell.
    // Array retirement precedes host source/custody retirement.
    host_source: OnceLock<super::RetainedHostBuffer>,
    coordinate: usize,
    catalog: NameCatalogOwner,
}
// Each physical-cell alias carries the ORIGINAL cell's custody after its Arc.
// A later request's table guard cannot fund deallocation of an older cell.
#[derive(Clone)]
pub(crate) struct CanonicalArrayOwner {
    cell: Arc<CanonicalArray>,
    custody: OriginalOperationMetadataCustody,
}
impl CanonicalArrayOwner {
    pub(crate) fn publication_control_bytes() -> Option<usize> {
        use super::super::storage::{
            PublishedAllocation, RetainedAllocationReceipt, RetainedStorageRef,
        };
        fn iterator_bytes<T>(_: impl FnOnce(&'static NamedArrays) -> T) -> usize {
            size_of::<T>()
        }
        let frames = [
            iterator_bytes(NamedArrays::retained_values),
            size_of::<&Self>(),
            size_of::<Self>(),
            size_of::<Arc<CanonicalArray>>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<Option<PublishedAllocation>>(),
            size_of::<PublishedAllocation>(),
            size_of::<&OnceLock<PublishedAllocation>>(),
            size_of::<Option<&PublishedAllocation>>(),
            size_of::<Result<(), PublishedAllocation>>(),
            size_of::<bool>(),
            size_of::<Option<RetainedAllocationReceipt<'_>>>(),
            size_of::<&NamedArrays>(),
            size_of::<NamedIter<'_>>(),
            size_of::<(&str, &Array)>(),
            size_of::<&PreparedNames>(),
            size_of::<&NamedSlot>(),
            size_of::<Option<&CanonicalArrayOwner>>(),
            size_of::<RetainedStorageRef<'_>>(),
            size_of::<Result<usize, NamedArrayError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn array(&self) -> &Array {
        self.value.get().expect("published canonical value")
    }
    pub(crate) fn custody(&self) -> &OriginalOperationMetadataCustody {
        &self.custody
    }
    pub(crate) fn proof(&self) -> Option<super::super::storage::PublishedAllocation> {
        self.publication.get().copied()
    }
    pub(crate) fn same_cell(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell) && self.custody.same_account(&other.custody)
    }
    pub(crate) fn record_attachment(
        &self,
        proof: super::super::storage::PublishedAllocation,
    ) -> bool {
        match self.publication.set(proof) {
            Ok(()) => true,
            Err(proof) => self.publication.get() == Some(&proof),
        }
    }
}
impl Deref for CanonicalArrayOwner {
    type Target = CanonicalArray;
    fn deref(&self) -> &Self::Target {
        &self.cell
    }
}
#[derive(Clone)]
pub(super) enum AliasArray {
    Canonical(CanonicalArrayOwner),
    // This variant is constructed only for an actual immutable ordinary map.
    // The owner, not an unbound integer, fixes the coordinate for its lifetime.
    Ordinary {
        owner: ResidentArraysOwner,
        index: usize,
        coordinate: usize,
        catalog: NameCatalogOwner,
    },
}
impl AliasArray {
    fn get(&self) -> Option<&Array> {
        match self {
            Self::Canonical(cell) => cell.value.get(),
            Self::Ordinary { owner, index, .. } => match &owner.arrays.arrays {
                NamedArrays::Ordinary(values) => values.values().nth(*index),
                NamedArrays::Source(values) => {
                    values.values().nth(*index).map(|value| &value.value)
                }
                _ => None,
            },
        }
    }
    fn host_source(&self) -> Option<&super::RetainedHostBuffer> {
        match self {
            Self::Canonical(cell) => cell.host_source.get(),
            Self::Ordinary { owner, index, .. } => match &owner.arrays.arrays {
                NamedArrays::Source(values) => {
                    values.values().nth(*index).map(|value| &value.source)
                }
                _ => None,
            },
        }
    }
    fn matches(&self, catalog: &NameCatalogOwner, coordinate: usize) -> bool {
        match self {
            Self::Canonical(cell) => {
                catalog.same_canonical(coordinate, &cell.catalog, cell.coordinate)
            }
            Self::Ordinary {
                coordinate: source,
                catalog: owner,
                ..
            } => catalog.same_canonical(coordinate, owner, *source),
        }
    }
}
enum NamedSlot {
    Canonical(CanonicalArrayOwner),
    Alias(Option<AliasArray>),
}
impl NamedSlot {
    fn get(&self) -> Option<&Array> {
        match self {
            Self::Canonical(cell) => cell.value.get(),
            Self::Alias(value) => value.as_ref()?.get(),
        }
    }
    fn host_source(&self) -> Option<&super::RetainedHostBuffer> {
        match self {
            Self::Canonical(cell) => cell.host_source.get(),
            Self::Alias(value) => value.as_ref()?.host_source(),
        }
    }
    fn alias(&self) -> Option<AliasArray> {
        self.get()?;
        Some(match self {
            Self::Canonical(cell) => AliasArray::Canonical(cell.clone()),
            Self::Alias(value) => value.as_ref()?.clone(),
        })
    }
}

pub(super) struct PreparedNames {
    catalog: NameCatalogOwner,
    unit: usize,
    values: Vec<NamedSlot>,
}
impl PreparedNames {
    fn names(&self) -> &[CatalogName] {
        &self.catalog.value.names[self.catalog.value.units[self.unit].names.clone()]
    }
    fn index(&self, name: &str) -> Result<usize, NamedArrayError> {
        self.names()
            .binary_search_by(|row| row.name.as_str().cmp(name))
            .map_err(|_| NamedArrayError::UnknownName)
    }
    fn empty_canonical(&mut self, name: &str) -> Result<&mut CanonicalArray, NamedArrayError> {
        let index = self.index(name)?;
        let Some(NamedSlot::Canonical(cell)) = self.values.get_mut(index) else {
            return Err(NamedArrayError::InvalidSource);
        };
        if cell.value.get().is_some() {
            return Err(NamedArrayError::DuplicateValue);
        }
        Arc::get_mut(&mut cell.cell).ok_or(NamedArrayError::SharedDestination)
    }
}

pub(super) enum NamedArrays {
    Empty,
    Ordinary(BTreeMap<String, Array>),
    Prepared(PreparedNames),
    Source(super::rows::Rows<String, SourceArray>),
}
pub(super) struct SourceArray {
    pub(super) value: Array,
    pub(super) source: super::RetainedHostBuffer,
}
impl NamedArrays {
    pub(super) fn is_prepared(&self) -> bool {
        matches!(self, Self::Prepared(_))
    }
    pub(super) fn retained_values(
        &self,
    ) -> impl Iterator<Item = super::super::storage::RetainedStorageRef<'_>> {
        self.iter().map(move |(name, array)| {
            if let Self::Prepared(table) = self {
                let row = &table.values[table.index(name).expect("retained name")];
                let cell = match row {
                    NamedSlot::Canonical(cell)
                    | NamedSlot::Alias(Some(AliasArray::Canonical(cell))) => Some(cell),
                    _ => None,
                };
                if let Some(cell) = cell {
                    return super::super::storage::RetainedStorageRef::CanonicalArray(cell);
                }
            }
            super::super::storage::RetainedStorageRef::Array(array)
        })
    }
    pub(super) fn host_sources(&self) -> NamedHostIter<'_> {
        match self {
            Self::Prepared(table) => NamedHostIter::Prepared(table.values.iter()),
            Self::Source(values) => NamedHostIter::Source(values.iter()),
            _ => NamedHostIter::Empty,
        }
    }
    pub(super) fn host_source_control_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<NamedHostIter<'_>>(),
            std::mem::size_of::<Option<&super::RetainedHostBuffer>>(),
            std::mem::size_of::<super::RetainedHostBuffer>(),
            std::mem::size_of::<Result<(), NamedArrayError>>(),
            std::mem::size_of::<Result<(), super::RetainedHostBuffer>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn validate_host_source(&mut self, name: &str) -> Result<(), NamedArrayError> {
        let Self::Prepared(table) = self else {
            return Err(NamedArrayError::InvalidSource);
        };
        if table.empty_canonical(name)?.host_source.get().is_some() {
            return Err(NamedArrayError::DuplicateValue);
        }
        Ok(())
    }
    pub(super) fn retain_host_source(
        &mut self,
        name: &str,
        source: &super::RetainedHostBuffer,
    ) -> Result<(), NamedArrayError> {
        self.validate_host_source(name)?;
        let Self::Prepared(table) = self else {
            unreachable!("validated prepared source slot");
        };
        table
            .empty_canonical(name)?
            .host_source
            .set(source.clone())
            .map_err(|_| NamedArrayError::DuplicateValue)
    }
    pub(super) fn catalog(&self) -> Option<&NameCatalogOwner> {
        match self {
            Self::Prepared(table) => Some(&table.catalog),
            _ => None,
        }
    }
    pub(super) fn get(&self, name: &str) -> Option<&Array> {
        match self {
            Self::Empty => None,
            Self::Ordinary(values) => values.get(name),
            Self::Source(values) => values.get(name).map(|value| &value.value),
            Self::Prepared(values) => values.values.get(values.index(name).ok()?)?.get(),
        }
    }
    pub(super) fn iter(&self) -> NamedIter<'_> {
        match self {
            Self::Empty => NamedIter::Empty,
            Self::Ordinary(values) => NamedIter::Ordinary(values.iter()),
            Self::Source(values) => NamedIter::Source(values.into_iter()),
            Self::Prepared(values) => NamedIter::Prepared { values, index: 0 },
        }
    }
    pub(super) fn values(&self) -> impl Iterator<Item = &Array> {
        self.iter().map(|(_, value)| value)
    }
    pub(super) fn keys(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|(name, _)| name)
    }
    pub(super) fn len(&self) -> usize {
        self.iter().count()
    }
    pub(super) fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Validate the entire declared destination before producing any canonical
    /// payload. The same cell predicate is used by put; aliases remain managed
    /// by the caller's existing canonical closure/publication worker.
    pub(super) fn validate_prepared_canonical_outputs(
        &mut self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<(), NamedArrayError> {
        let Self::Prepared(table) = self else {
            return Err(NamedArrayError::InvalidSource);
        };
        if table.catalog.validate_unit(manager, unit)? != table.unit
            || table.values.len() != unit.bindings().len()
        {
            return Err(NamedArrayError::InvalidSource);
        }
        for binding in unit.bindings().iter().filter(|binding| !binding.is_alias()) {
            if table
                .empty_canonical(binding.name())?
                .host_source
                .get()
                .is_some()
            {
                return Err(NamedArrayError::DuplicateValue);
            }
        }
        Ok(())
    }

    /// Strict destination validation happens before the value moves. On error
    /// the original input is returned; no row, name, or prior value is replaced.
    pub(super) fn put(&mut self, name: &str, value: Array) -> Result<(), (NamedArrayError, Array)> {
        match self {
            Self::Empty | Self::Source(_) => Err((NamedArrayError::InvalidSource, value)),
            Self::Ordinary(values) => {
                values.insert(name.to_owned(), value);
                Ok(())
            }
            Self::Prepared(table) => {
                let cell = match table.empty_canonical(name) {
                    Ok(cell) => cell,
                    Err(error) => return Err((error, value)),
                };
                cell.value
                    .set(value)
                    .map_err(|value| (NamedArrayError::DuplicateValue, value))
            }
        }
    }
    /// Retry reuses the same unpublished destination after retained source
    /// detachment. Preflight every cell before dropping any prior value. No
    /// alias may have escaped into another table at this canonical-only phase.
    pub(super) fn reset_unpublished(&mut self) -> Result<(), NamedArrayError> {
        match self {
            Self::Ordinary(values) => {
                values.clear();
                Ok(())
            }
            Self::Empty | Self::Source(_) => Err(NamedArrayError::InvalidSource),
            Self::Prepared(table) => {
                for row in &mut table.values {
                    match row {
                        NamedSlot::Canonical(cell) => {
                            if cell.publication.get().is_some() {
                                return Err(NamedArrayError::InvalidSource);
                            }
                            if Arc::get_mut(&mut cell.cell).is_none() {
                                return Err(NamedArrayError::SharedDestination);
                            }
                        }
                        NamedSlot::Alias(Some(_)) => {
                            return Err(NamedArrayError::SharedDestination);
                        }
                        NamedSlot::Alias(None) => {}
                    }
                }
                for row in &mut table.values {
                    if let NamedSlot::Canonical(cell) = row {
                        let cell = Arc::get_mut(&mut cell.cell).expect("preflight unique cell");
                        cell.value.take();
                        cell.host_source.take();
                    }
                }
                Ok(())
            }
        }
    }
    pub(super) fn alias_value(&self, name: &str) -> Option<AliasArray> {
        match self {
            Self::Prepared(table) => table.values.get(table.index(name).ok()?)?.alias(),
            _ => None,
        }
    }
    pub(super) fn put_alias(
        &mut self,
        name: &str,
        value: AliasArray,
    ) -> Result<(), NamedArrayError> {
        let Self::Prepared(table) = self else {
            return Err(NamedArrayError::InvalidSource);
        };
        let index = table.index(name)?;
        if !value.matches(&table.catalog, table.names()[index].canonical) {
            return Err(NamedArrayError::InvalidSource);
        }
        let Some(NamedSlot::Alias(slot)) = table.values.get_mut(index) else {
            return Err(NamedArrayError::InvalidSource);
        };
        if slot.is_some() {
            return Err(NamedArrayError::DuplicateValue);
        }
        if value.get().is_none() {
            return Err(NamedArrayError::MissingValue);
        }
        *slot = Some(value);
        Ok(())
    }
    pub(super) fn insert_ordinary(
        &mut self,
        name: String,
        value: Array,
    ) -> Result<Option<Array>, NamedArrayError> {
        match self {
            Self::Ordinary(values) => Ok(values.insert(name, value)),
            _ => Err(NamedArrayError::InvalidSource),
        }
    }
}
impl From<BTreeMap<String, Array>> for NamedArrays {
    fn from(value: BTreeMap<String, Array>) -> Self {
        Self::Ordinary(value)
    }
}
impl Index<&str> for NamedArrays {
    type Output = Array;
    fn index(&self, name: &str) -> &Array {
        self.get(name).expect("missing resident array")
    }
}
pub(super) enum NamedIter<'a> {
    Empty,
    Ordinary(btree_map::Iter<'a, String, Array>),
    Source(std::slice::Iter<'a, (String, SourceArray)>),
    Prepared {
        values: &'a PreparedNames,
        index: usize,
    },
}
impl<'a> Iterator for NamedIter<'a> {
    type Item = (&'a str, &'a Array);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Ordinary(values) => values.next().map(|(name, value)| (name.as_str(), value)),
            Self::Source(values) => values
                .next()
                .map(|(name, value)| (name.as_str(), &value.value)),
            Self::Prepared { values, index } => {
                while let Some(row) = values.values.get(*index) {
                    let position = *index;
                    *index += 1;
                    if let Some(value) = row.get() {
                        return Some((&values.names()[position].name, value));
                    }
                }
                None
            }
        }
    }
}

pub(super) enum NamedHostIter<'a> {
    Empty,
    Prepared(std::slice::Iter<'a, NamedSlot>),
    Source(std::slice::Iter<'a, (String, SourceArray)>),
}
impl<'a> Iterator for NamedHostIter<'a> {
    type Item = &'a super::RetainedHostBuffer;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Prepared(values) => values.find_map(NamedSlot::host_source),
            Self::Source(values) => values.next().map(|(_, value)| &value.source),
        }
    }
}

/// Storage alias with custody after the Arc, including when this is its final
/// strong alias and deallocates both the array table and its native values.
#[derive(Clone)]
pub struct ResidentArraysOwner {
    arrays: Arc<ResidentArrays>,
    custody: Option<OriginalOperationMetadataCustody>,
    source_custody: super::ManagerCustody,
}
impl Deref for ResidentArraysOwner {
    type Target = ResidentArrays;
    fn deref(&self) -> &Self::Target {
        &self.arrays
    }
}
impl From<Arc<ResidentArrays>> for ResidentArraysOwner {
    fn from(arrays: Arc<ResidentArrays>) -> Self {
        Self {
            arrays,
            custody: None,
            source_custody: super::ManagerCustody::default(),
        }
    }
}
impl ResidentArraysOwner {
    pub(super) fn from_source(
        values: super::rows::Rows<String, SourceArray>,
        custody: super::ManagerCustody,
    ) -> Self {
        Self {
            arrays: Arc::new(ResidentArrays {
                arrays: NamedArrays::Source(values),
            }),
            custody: None,
            source_custody: custody,
        }
    }
    pub(super) fn source_storage_bytes(
    ) -> Result<u64, eredu_runtime::working_memory::WorkingMemoryError> {
        eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
            Layout::new::<ResidentArrays>(),
        )
    }
    pub(super) fn alias_value(
        &self,
        name: &str,
        unit: &OffloadUnitId,
        catalog: &NameCatalogOwner,
    ) -> Option<AliasArray> {
        match &self.arrays.arrays {
            NamedArrays::Prepared(values) => values.values.get(values.index(name).ok()?)?.alias(),
            NamedArrays::Ordinary(values) => {
                let index = values.keys().position(|key| key == name)?;
                let coordinate = catalog.coordinate(unit, name).ok()?;
                if catalog.value.names[coordinate].alias {
                    return None;
                }
                Some(AliasArray::Ordinary {
                    owner: self.clone(),
                    index,
                    coordinate,
                    catalog: catalog.clone(),
                })
            }
            NamedArrays::Source(values) => {
                let index = values.keys().position(|key| key == name)?;
                let coordinate = catalog.coordinate(unit, name).ok()?;
                if catalog.value.names[coordinate].alias {
                    return None;
                }
                Some(AliasArray::Ordinary {
                    owner: self.clone(),
                    index,
                    coordinate,
                    catalog: catalog.clone(),
                })
            }
            NamedArrays::Empty => None,
        }
    }
}

/// The destination vector and the final Arc exist before native work. The
/// unique Arc receives that vector by move only after full source validation.
pub(super) struct PreparedNamedArrays {
    pub(super) arrays: NamedArrays,
    owner: ResidentArraysOwner,
}
impl PreparedNamedArrays {
    pub(super) fn validate_publication(
        &mut self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<(), NamedArrayError> {
        let NamedArrays::Prepared(table) = &self.arrays else {
            return Err(NamedArrayError::InvalidSource);
        };
        if table.catalog.validate_unit(manager, unit)? != table.unit {
            return Err(NamedArrayError::InvalidSource);
        }
        if table.values.len() != unit.bindings().len()
            || table.values.iter().any(|row| row.get().is_none())
        {
            return Err(NamedArrayError::MissingValue);
        }
        if Arc::get_mut(&mut self.owner.arrays).is_none() {
            return Err(NamedArrayError::SharedDestination);
        }
        Ok(())
    }
    pub(super) fn publish(
        mut self,
        manager: &ManagerOwner,
        unit: &OffloadUnit,
    ) -> Result<ResidentArraysOwner, (NamedArrayError, Self)> {
        let validate = self.validate_publication(manager, unit);
        if let Err(error) = validate {
            return Err((error, self));
        }
        Arc::get_mut(&mut self.owner.arrays)
            .expect("validated unique final Arc")
            .arrays = self.arrays;
        Ok(self.owner)
    }
}

#[cfg(test)]
pub(super) mod tests;

#[cfg(test)]
use eredu_runtime::working_memory::OriginalTextControlGuard;
