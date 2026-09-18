//! Borrowed parameter-source facts; these are not admission or readiness grants.
use crate::{ExecutionUnitAddress, ExecutionUnitLayout, WeightBinding};
use eredu_core::residency::OffloadUnitId;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceLayoutError, WorkspaceLayoutView, WorkspaceRepresentation};
use std::ops::Range;

/// Where a prospective metadata root may be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspaceParameterLifetime {
    /// A new root is required on each unit invocation, including revisits.
    #[default]
    Invocation,
    /// Reuse only within the same source and workspace trace.
    Trace,
}

/// Exact canonical row in this source's complete owner closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkspaceParameterOwner {
    /// Owner unit index, including units outside the requested sequence.
    pub unit: usize,
    /// Owner binding index within that unit.
    pub row: usize,
    /// Root reuse lifetime; this does not describe physical completion.
    pub lifetime: WorkspaceParameterLifetime,
}

/// One retained unit's borrowed declaration.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceParameterUnit<'a> {
    /// Actual immutable unit identity.
    pub id: &'a OffloadUnitId,
    /// Number of binding rows in source order.
    pub rows: usize,
    /// Whole-unit future materialization capacity, including companions.
    pub fresh_capacity_bytes: u64,
}

/// A borrowed output declaration from an actual retained parameter source.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceParameterRow<'a> {
    /// Exact local name, source selection, recipe and companion declarations.
    pub binding: &'a WeightBinding,
    /// Actual physical output shape; no copy is made here.
    pub shape: &'a [i32],
    /// Representation used by the ordinary workspace equations.
    pub dtype: WorkspaceDtype,
    /// Exact scalar and layout evidence from this same retained output source.
    pub representation: Option<WorkspaceRepresentation>,
    /// Actual physical output bytes, distinct from represented Float32 bytes.
    pub physical_bytes: u64,
    /// Conservative output backing envelope; never an existing-storage credit.
    pub capacity_bytes: u64,
    /// Canonical prospective root; host aliases can have independent owners.
    pub owner: WorkspaceParameterOwner,
}

/// Fixed failure before constructing owned parameter metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceParameterSourceError {
    /// The provider has no borrowed companion; legacy allocation is not attempted.
    #[error("borrowed parameter source is unavailable")]
    CompanionUnavailable,
    /// An indexed unit, row, requested address or window is absent.
    #[error("parameter source index is absent")]
    MissingIndex,
    /// Retained names, owner references, capacities or selected windows disagree.
    #[error("parameter source declaration mismatch")]
    SourceMismatch,
    /// The actual output dtype has no workspace representation.
    #[error("parameter source dtype at unit {unit}, row {row} has no workspace representation")]
    UnsupportedDtype {
        /// Complete owner-closure unit index.
        unit: usize,
        /// Binding index within the unit.
        row: usize,
    },
    /// The shared borrowed shape worker rejected the declaration.
    #[error("invalid parameter source layout at unit {unit}, row {row}: {source}")]
    Layout {
        /// Complete owner-closure unit index.
        unit: usize,
        /// Binding index within the unit.
        row: usize,
        /// Actual fixed failure from the shared geometry worker.
        source: WorkspaceLayoutError,
    },
    /// Count, range or destination layout arithmetic overflowed.
    #[error("parameter source destination size overflow")]
    Overflow,
    /// At least one caller-provided destination is too short.
    #[error("parameter source destination capacity is insufficient")]
    DestinationCapacity,
}
type Result<T> = std::result::Result<T, WorkspaceParameterSourceError>;

/// Immutable indexed source law for a borrowed parameter frame.
///
/// All membership, borrowed values and scalar facts must remain unchanged for
/// every outstanding loan. Implementations must read retained metadata only:
/// no allocation, native operation, lock, source read, callback or mutation.
/// This contract conveys facts, not source registration or accounting authority.
/// The native adapter implements it on the actual owning LayerwiseWorkspace.
pub trait WorkspaceParameterRows {
    /// Exact selected parameters populated by an independent owner, rather than
    /// this unit source. Missing rows must never be used to infer exclusions.
    fn excludes_parameter(&self, _name: &str) -> bool { false }
    /// Retained selected execution layout.
    fn layout(&self) -> &ExecutionUnitLayout;
    /// Architecture address of a local requested slot. Pipeline storage can use
    /// local ordinals while module identities retain their global addresses.
    fn execution_address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        self.layout().address(ordinal)
    }
    /// Complete owner-closure unit count.
    fn unit_count(&self) -> usize;
    /// Borrow one unit declaration.
    fn unit(&self, unit: usize) -> Result<WorkspaceParameterUnit<'_>>;
    /// Borrow one physical output declaration.
    fn row(&self, unit: usize, row: usize) -> Result<WorkspaceParameterRow<'_>>;
    /// Map one selected execution ordinal into the owner closure.
    fn requested_unit(&self, ordinal: usize) -> Result<usize>;
    /// Number of units in the actual permitted window for this ordinal.
    fn window_len(&self, ordinal: usize) -> Result<usize>;
    /// Resolve an actual window member into the owner closure.
    fn window_unit(&self, ordinal: usize, member: usize) -> Result<usize>;
}

/// Counts for exact caller-owned destinations, without allocator/header credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkspaceParameterCounts {
    /// Complete owner units.
    pub units: usize,
    /// Selected execution units and their windows.
    pub requested: usize,
    /// All binding rows, including aliases.
    pub rows: usize,
    /// Unit-name and binding-name UTF-8 bytes, without terminators.
    pub name_bytes: usize,
    /// Flattened i32 shape extents.
    pub shape_elements: usize,
    /// Canonical root prototypes, not constructed context roots.
    pub roots: usize,
    /// Flattened permitted-window unit indices.
    pub window_members: usize,
}
impl WorkspaceParameterCounts {
    /// Element storage only. The actual destination allocator and enclosing
    /// owner must separately account for headers/alignment and all wrappers.
    pub fn element_bytes(self) -> Result<usize> {
        [
            (
                self.units,
                std::mem::size_of::<WorkspaceParameterUnitRecord>(),
            ),
            (
                self.requested,
                std::mem::size_of::<WorkspaceParameterRequestRecord>(),
            ),
            (self.rows, std::mem::size_of::<WorkspaceParameterRecord>()),
            (self.name_bytes, 1),
            (self.shape_elements, std::mem::size_of::<i32>()),
            (
                self.roots,
                std::mem::size_of::<WorkspaceParameterRootRecord>(),
            ),
            (self.window_members, std::mem::size_of::<usize>()),
        ]
        .into_iter()
        .try_fold(0usize, |sum, (n, bytes)| {
            sum.checked_add(
                n.checked_mul(bytes)
                    .ok_or(WorkspaceParameterSourceError::Overflow)?,
            )
            .ok_or(WorkspaceParameterSourceError::Overflow)
        })
    }
}

/// Size and alignment of one concrete component representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceParameterTypeLayout {
    /// Exact size on the current target.
    pub bytes: usize,
    /// Required target alignment.
    pub alignment: usize,
}
impl WorkspaceParameterTypeLayout {
    const fn of<T>() -> Self {
        Self {
            bytes: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
        }
    }
}

/// Named concrete wrappers and transient results for this component only.
///
/// Entries describe individual representations, not an additive arena recipe:
/// nesting and overlap depend on the actual caller's owner and call lifetime.
/// The returned array and its own result representation are ordinary stack
/// values, not a hidden allocation. Destination headers and source owners are
/// intentionally absent from these component facts.
pub const fn workspace_parameter_control_layouts(
) -> [(&'static str, WorkspaceParameterTypeLayout); 19] {
    use WorkspaceParameterTypeLayout as L;
    [
        (
            "source loan",
            L::of::<WorkspaceParameterSourceLoan<'static>>(),
        ),
        (
            "counted source",
            L::of::<CountedWorkspaceParameterSource<'static>>(),
        ),
        (
            "destinations",
            L::of::<WorkspaceParameterDestinations<'static>>(),
        ),
        (
            "table",
            L::of::<WorkspaceParameterTable<'static, 'static>>(),
        ),
        ("counts", L::of::<WorkspaceParameterCounts>()),
        ("owner", L::of::<WorkspaceParameterOwner>()),
        ("borrowed unit", L::of::<WorkspaceParameterUnit<'static>>()),
        ("borrowed row", L::of::<WorkspaceParameterRow<'static>>()),
        ("error", L::of::<WorkspaceParameterSourceError>()),
        (
            "loan result",
            L::of::<Result<WorkspaceParameterSourceLoan<'static>>>(),
        ),
        (
            "count result",
            L::of::<Result<CountedWorkspaceParameterSource<'static>>>(),
        ),
        (
            "table result",
            L::of::<Result<WorkspaceParameterTable<'static, 'static>>>(),
        ),
        (
            "unit result",
            L::of::<Result<WorkspaceParameterUnit<'static>>>(),
        ),
        (
            "row result",
            L::of::<Result<WorkspaceParameterRow<'static>>>(),
        ),
        ("counts result", L::of::<Result<WorkspaceParameterCounts>>()),
        ("index result", L::of::<Result<usize>>()),
        ("validation result", L::of::<Result<()>>()),
        ("range", L::of::<Range<usize>>()),
        (
            "layout result",
            L::of::<std::result::Result<WorkspaceLayoutView<'static>, WorkspaceLayoutError>>(),
        ),
    ]
}

/// A fixed unit record. Its ranges are meaningful only through its source table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkspaceParameterUnitRecord {
    name: Range<usize>,
    rows: Range<usize>,
    fresh_capacity_bytes: u64,
}
/// A fixed selected-unit/window record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkspaceParameterRequestRecord {
    unit: usize,
    window: Range<usize>,
}
/// A fixed binding record. Logical identity alone is not a physical root.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkspaceParameterRecord {
    name: Range<usize>,
    shape: Range<usize>,
    dtype: Option<WorkspaceDtype>,
    representation: Option<WorkspaceRepresentation>,
    physical_bytes: u64,
    root: usize,
}
/// A fixed prospective root prototype; not an actual storage allocation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkspaceParameterRootRecord {
    owner: WorkspaceParameterOwner,
    capacity_bytes: u64,
}

/// Exact destinations supplied by the existing caller's storage owner.
pub struct WorkspaceParameterDestinations<'a> {
    /// Unit records.
    pub units: &'a mut [WorkspaceParameterUnitRecord],
    /// Selected-unit/window records.
    pub requested: &'a mut [WorkspaceParameterRequestRecord],
    /// Binding records.
    pub rows: &'a mut [WorkspaceParameterRecord],
    /// UTF-8 unit and binding names.
    pub names: &'a mut [u8],
    /// Shape extents.
    pub shapes: &'a mut [i32],
    /// Root prototypes.
    pub roots: &'a mut [WorkspaceParameterRootRecord],
    /// Window membership indices.
    pub window_members: &'a mut [usize],
}

/// Loan of one actual source. No temporary provider or owned source is created.
///
/// A counted loan cannot escape the actual source lifetime.
/// ```compile_fail
/// use eredu_runtime::working_memory::*;
/// fn escape(source: &dyn WorkspaceParameterRows) -> CountedWorkspaceParameterSource<'static> {
///     WorkspaceParameterSourceLoan::new(source).count().unwrap()
/// }
/// ```
/// The same source and caller destination can be used without another owner.
/// ```
/// use eredu_runtime::working_memory::*;
/// fn fill<'s, 'd>(source: &'s dyn WorkspaceParameterRows, destination: WorkspaceParameterDestinations<'d>)
///     -> Result<WorkspaceParameterTable<'s, 'd>, WorkspaceParameterSourceError> {
///     WorkspaceParameterSourceLoan::new(source).count()?.fill(destination)
/// }
/// ```
pub struct WorkspaceParameterSourceLoan<'a> {
    source: &'a dyn WorkspaceParameterRows,
}
impl<'a> WorkspaceParameterSourceLoan<'a> {
    /// Borrows a source satisfying the immutable indexed-source law.
    pub fn new(source: &'a dyn WorkspaceParameterRows) -> Self {
        Self { source }
    }
    /// Descriptive source-object address equality, independent of trait vtables.
    /// This is not an authentication or storage-credit grant. The concrete native
    /// workspace owns nonzero-sized immutable snapshot storage; arbitrary public
    /// zero-sized providers may share an address without sharing logical identity.
    pub fn same_source(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self.source, other.source)
    }
    /// Validate and count the actual complete source without owned metadata.
    pub fn count(self) -> Result<CountedWorkspaceParameterSource<'a>> {
        let counts = validate(self.source)?;
        counts.element_bytes()?;
        Ok(CountedWorkspaceParameterSource { loan: self, counts })
    }
}

/// Counted source and its original borrow. Copied counts cannot construct this.
/// ```compile_fail
/// use eredu_runtime::working_memory::*;
/// fn forge(source: &dyn WorkspaceParameterRows, counts: WorkspaceParameterCounts) {
///     let _ = CountedWorkspaceParameterSource { loan: WorkspaceParameterSourceLoan::new(source), counts };
/// }
/// ```
pub struct CountedWorkspaceParameterSource<'a> {
    loan: WorkspaceParameterSourceLoan<'a>,
    counts: WorkspaceParameterCounts,
}
impl<'a> CountedWorkspaceParameterSource<'a> {
    /// Exact element counts, excluding allocation/header and enclosing controls.
    pub fn counts(&self) -> WorkspaceParameterCounts {
        self.counts
    }
    /// Fill this same immutable source. All ordinary failures precede writes.
    ///
    /// A provider violating the documented immutable-source law is a programming
    /// error; its callbacks are not an allocation/completion authority. The actual
    /// native provider borrows immutable snapshot storage for this whole call.
    pub fn fill<'d>(
        self,
        mut destination: WorkspaceParameterDestinations<'d>,
    ) -> Result<WorkspaceParameterTable<'a, 'd>> {
        let c = self.counts;
        if destination.units.len() < c.units
            || destination.requested.len() < c.requested
            || destination.rows.len() < c.rows
            || destination.names.len() < c.name_bytes
            || destination.shapes.len() < c.shape_elements
            || destination.roots.len() < c.roots
            || destination.window_members.len() < c.window_members
        {
            return Err(WorkspaceParameterSourceError::DestinationCapacity);
        }
        if validate(self.loan.source)? != c {
            return Err(WorkspaceParameterSourceError::SourceMismatch);
        }
        write_validated(self.loan.source, c, &mut destination);
        Ok(WorkspaceParameterTable {
            source: self.loan.source,
            counts: c,
            destination,
        })
    }
}

/// Populated metadata borrowing both its exact source and destination owner.
/// ```compile_fail
/// use eredu_runtime::working_memory::*;
/// fn escape<'s>(counted: CountedWorkspaceParameterSource<'s>) -> WorkspaceParameterTable<'s, 'static> {
///     let mut names = [0u8; 1];
///     counted.fill(WorkspaceParameterDestinations { units: &mut [], requested: &mut [], rows: &mut [],
///         names: &mut names, shapes: &mut [], roots: &mut [], window_members: &mut [] }).unwrap()
/// }
/// ```
/// Borrowed views cannot survive destruction of the table that binds their source.
/// ```compile_fail
/// use eredu_runtime::working_memory::*;
/// fn escape_name<'s, 'd>(table: WorkspaceParameterTable<'s, 'd>) -> &'d str {
///     table.row_name(0).unwrap()
/// }
/// ```
pub struct WorkspaceParameterTable<'s, 'd> {
    source: &'s dyn WorkspaceParameterRows,
    counts: WorkspaceParameterCounts,
    destination: WorkspaceParameterDestinations<'d>,
}
impl WorkspaceParameterTable<'_, '_> {
    /// Exact initialized destination counts.
    pub fn counts(&self) -> WorkspaceParameterCounts {
        self.counts
    }
    /// The actual source's selected layout; no copied layout or policy is made.
    pub fn layout(&self) -> &ExecutionUnitLayout {
        self.source.layout()
    }
    /// Resolve the exact selected address, rejecting a wrong ordinal/address.
    pub fn requested_unit(&self, ordinal: usize, address: ExecutionUnitAddress) -> Result<usize> {
        if self.source.execution_address(ordinal) != Some(address) {
            return Err(WorkspaceParameterSourceError::SourceMismatch);
        }
        self.destination
            .requested
            .get(ordinal)
            .filter(|_| ordinal < self.counts.requested)
            .map(|r| r.unit)
            .ok_or(WorkspaceParameterSourceError::MissingIndex)
    }
    /// Borrow a unit's exact stable name.
    pub fn unit_name(&self, unit: usize) -> Option<&str> {
        let r = self
            .destination
            .units
            .get(unit)
            .filter(|_| unit < self.counts.units)?;
        std::str::from_utf8(&self.destination.names[r.name.clone()]).ok()
    }
    /// Borrow the flat binding range of a closure unit.
    pub fn unit_rows(&self, unit: usize) -> Option<Range<usize>> {
        self.destination
            .units
            .get(unit)
            .filter(|_| unit < self.counts.units)
            .map(|r| r.rows.clone())
    }
    /// Whole-unit future materialization, including all canonical companions.
    pub fn unit_fresh_capacity_bytes(&self, unit: usize) -> Option<u64> {
        self.destination
            .units
            .get(unit)
            .filter(|_| unit < self.counts.units)
            .map(|r| r.fresh_capacity_bytes)
    }
    /// Borrow a populated local binding name.
    pub fn row_name(&self, row: usize) -> Option<&str> {
        let r = self
            .destination
            .rows
            .get(row)
            .filter(|_| row < self.counts.rows)?;
        std::str::from_utf8(&self.destination.names[r.name.clone()]).ok()
    }
    /// Borrow populated represented geometry, distinct from physical bytes.
    pub fn row_layout(&self, row: usize) -> Option<WorkspaceLayoutView<'_>> {
        let r = self
            .destination
            .rows
            .get(row)
            .filter(|_| row < self.counts.rows)?;
        WorkspaceLayoutView::new(&self.destination.shapes[r.shape.clone()], r.dtype?).ok()
            .map(|layout| layout.with_representation(r.representation))
    }
    /// Actual selected physical bytes; these need not equal represented bytes.
    pub fn row_physical_bytes(&self, row: usize) -> Option<u64> {
        self.destination
            .rows
            .get(row)
            .filter(|_| row < self.counts.rows)
            .map(|r| r.physical_bytes)
    }
    /// Source-local root prototype index; this grants no actual storage identity.
    pub fn row_root(&self, row: usize) -> Option<usize> {
        self.destination
            .rows
            .get(row)
            .filter(|_| row < self.counts.rows)
            .map(|r| r.root)
    }
    /// Canonical source-local owner and required reuse lifetime.
    pub fn root_owner(&self, root: usize) -> Option<WorkspaceParameterOwner> {
        self.destination
            .roots
            .get(root)
            .filter(|_| root < self.counts.roots)
            .map(|r| r.owner)
    }
    /// Prospective backing capacity, not registered existing storage.
    pub fn root_capacity_bytes(&self, root: usize) -> Option<u64> {
        self.destination
            .roots
            .get(root)
            .filter(|_| root < self.counts.roots)
            .map(|r| r.capacity_bytes)
    }
    /// The actual permitted owner-closure window for one selected ordinal.
    pub fn window(&self, ordinal: usize) -> Option<&[usize]> {
        let r = self
            .destination
            .requested
            .get(ordinal)
            .filter(|_| ordinal < self.counts.requested)?;
        Some(&self.destination.window_members[r.window.clone()])
    }
    /// Borrow the retained selection/recipe declaration from the same source.
    pub fn binding(&self, unit: usize, row: usize) -> Result<&WeightBinding> {
        Ok(self.source.row(unit, row)?.binding)
    }
}

fn add(total: &mut usize, n: usize) -> Result<()> {
    *total = total
        .checked_add(n)
        .ok_or(WorkspaceParameterSourceError::Overflow)?;
    Ok(())
}
fn validate(source: &dyn WorkspaceParameterRows) -> Result<WorkspaceParameterCounts> {
    use WorkspaceParameterSourceError::{MissingIndex, SourceMismatch};
    let mut c = WorkspaceParameterCounts {
        units: source.unit_count(),
        requested: source.layout().len(),
        ..Default::default()
    };
    for u in 0..c.units {
        let unit = source.unit(u)?;
        for before in 0..u {
            if source.unit(before)?.id == unit.id {
                return Err(SourceMismatch);
            }
        }
        add(&mut c.name_bytes, unit.id.as_str().len())?;
        add(&mut c.rows, unit.rows)?;
        for i in 0..unit.rows {
            let row = source.row(u, i)?;
            if row.binding.name().trim().is_empty() {
                return Err(SourceMismatch);
            }
            WorkspaceLayoutView::new(row.shape, row.dtype).map_err(|source| {
                WorkspaceParameterSourceError::Layout {
                    unit: u,
                    row: i,
                    source,
                }
            })?;
            if row.representation.is_some() && row.dtype != WorkspaceDtype::Float32 {
                return Err(SourceMismatch);
            }
            if row.physical_bytes > row.capacity_bytes {
                return Err(SourceMismatch);
            }
            for before in 0..i {
                if source.row(u, before)?.binding.name() == row.binding.name() {
                    return Err(SourceMismatch);
                }
            }
            add(&mut c.name_bytes, row.binding.name().len())?;
            add(&mut c.shape_elements, row.shape.len())?;
            let owner = row.owner;
            if owner.unit >= c.units || owner.row >= source.unit(owner.unit)?.rows {
                return Err(MissingIndex);
            }
            let canonical = source.row(owner.unit, owner.row)?;
            if canonical.owner != owner || canonical.capacity_bytes != row.capacity_bytes {
                return Err(SourceMismatch);
            }
            if owner.unit == u && owner.row == i {
                add(&mut c.roots, 1)?;
            }
        }
    }
    for ordinal in 0..c.requested {
        let local_address = source.layout().address(ordinal).ok_or(MissingIndex)?;
        let execution_address = source.execution_address(ordinal).ok_or(MissingIndex)?;
        if execution_address.group() != local_address.group() {
            return Err(SourceMismatch);
        }
        for before in 0..ordinal {
            if source.execution_address(before) == Some(execution_address) {
                return Err(SourceMismatch);
            }
        }
        let unit = source.requested_unit(ordinal)?;
        if unit >= c.units {
            return Err(MissingIndex);
        }
        for before in 0..ordinal {
            if source.requested_unit(before)? == unit {
                return Err(SourceMismatch);
            }
        }
        let count = source.window_len(ordinal)?;
        if count == 0 || source.window_unit(ordinal, 0)? != unit {
            return Err(SourceMismatch);
        }
        let address = source.layout().address(ordinal).ok_or(MissingIndex)?;
        let group = source
            .layout()
            .group_range(address.group())
            .ok_or(MissingIndex)?;
        if count > group.end - ordinal {
            return Err(SourceMismatch);
        }
        for offset in 0..count {
            if source.window_unit(ordinal, offset)? != source.requested_unit(ordinal + offset)? {
                return Err(SourceMismatch);
            }
        }
        add(&mut c.window_members, count)?;
    }
    Ok(c)
}

fn write_validated(
    source: &dyn WorkspaceParameterRows,
    c: WorkspaceParameterCounts,
    d: &mut WorkspaceParameterDestinations<'_>,
) {
    let (mut names, mut shapes, mut rows, mut roots) = (0, 0, 0, 0);
    for u in 0..c.units {
        let unit = source.unit(u).expect("immutable validated parameter unit");
        let name = names..names + unit.id.as_str().len();
        d.names[name.clone()].copy_from_slice(unit.id.as_str().as_bytes());
        names = name.end;
        d.units[u] = WorkspaceParameterUnitRecord {
            name,
            rows: rows..rows + unit.rows,
            fresh_capacity_bytes: unit.fresh_capacity_bytes,
        };
        for i in 0..unit.rows {
            let row = source.row(u, i).expect("immutable validated parameter row");
            let name = names..names + row.binding.name().len();
            d.names[name.clone()].copy_from_slice(row.binding.name().as_bytes());
            names = name.end;
            let shape = shapes..shapes + row.shape.len();
            d.shapes[shape.clone()].copy_from_slice(row.shape);
            shapes = shape.end;
            // Canonical indices are assigned in source order. Aliases are
            // resolved only after every canonical coordinate has been written,
            // so forward owner references need no repeated provider search.
            let canonical = row.owner.unit == u && row.owner.row == i;
            let root = if canonical { roots } else { usize::MAX };
            d.rows[rows] = WorkspaceParameterRecord {
                name,
                shape,
                dtype: Some(row.dtype),
                representation: row.representation,
                physical_bytes: row.physical_bytes,
                root,
            };
            if canonical {
                d.roots[root] = WorkspaceParameterRootRecord {
                    owner: row.owner,
                    capacity_bytes: row.capacity_bytes,
                };
                roots += 1;
            }
            rows += 1;
        }
    }
    for u in 0..c.units {
        let range = d.units[u].rows.clone();
        for (i, index) in range.enumerate() {
            let row = source.row(u, i).expect("immutable validated parameter row");
            let owner_index = d.units[row.owner.unit].rows.start + row.owner.row;
            d.rows[index].root = d.rows[owner_index].root;
        }
    }
    let mut member = 0;
    for ordinal in 0..c.requested {
        let count = source
            .window_len(ordinal)
            .expect("immutable validated parameter window");
        d.requested[ordinal] = WorkspaceParameterRequestRecord {
            unit: source
                .requested_unit(ordinal)
                .expect("immutable validated requested unit"),
            window: member..member + count,
        };
        for offset in 0..count {
            d.window_members[member] = source
                .window_unit(ordinal, offset)
                .expect("immutable validated window member");
            member += 1;
        }
    }
}

#[cfg(test)]
mod tests;

mod projection;
pub use projection::WorkspaceParameterProjection;
