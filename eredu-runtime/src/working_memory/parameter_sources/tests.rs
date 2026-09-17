use super::*;
use crate::{ExecutionGraph, ExecutionGroupSpec};
use eredu_checkpoint::store::TensorSelection;

struct RowData {
    binding: WeightBinding,
    shape: Vec<i32>,
    dtype: WorkspaceDtype,
    representation: Option<WorkspaceRepresentation>,
    physical: u64,
    capacity: u64,
    owner: WorkspaceParameterOwner,
}
struct Source {
    layout: ExecutionUnitLayout,
    units: Vec<(OffloadUnitId, Vec<RowData>)>,
    requested: Vec<usize>,
    windows: Vec<Vec<usize>>,
    addresses: Option<Vec<ExecutionUnitAddress>>,
}
impl WorkspaceParameterRows for Source {
    fn layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }
    fn execution_address(&self, ordinal: usize) -> Option<ExecutionUnitAddress> {
        self.addresses.as_ref().map_or_else(|| self.layout.address(ordinal), |values| values.get(ordinal).copied())
    }
    fn unit_count(&self) -> usize {
        self.units.len()
    }
    fn unit(&self, unit: usize) -> Result<WorkspaceParameterUnit<'_>> {
        let (id, rows) = self
            .units
            .get(unit)
            .ok_or(WorkspaceParameterSourceError::MissingIndex)?;
        Ok(WorkspaceParameterUnit {
            id,
            rows: rows.len(),
            fresh_capacity_bytes: 48,
        })
    }
    fn row(&self, unit: usize, row: usize) -> Result<WorkspaceParameterRow<'_>> {
        let row = self
            .units
            .get(unit)
            .and_then(|u| u.1.get(row))
            .ok_or(WorkspaceParameterSourceError::MissingIndex)?;
        Ok(WorkspaceParameterRow {
            representation: row.representation,
            binding: &row.binding,
            shape: &row.shape,
            dtype: row.dtype,
            physical_bytes: row.physical,
            capacity_bytes: row.capacity,
            owner: row.owner,
        })
    }
    fn requested_unit(&self, ordinal: usize) -> Result<usize> {
        self.requested
            .get(ordinal)
            .copied()
            .ok_or(WorkspaceParameterSourceError::MissingIndex)
    }
    fn window_len(&self, ordinal: usize) -> Result<usize> {
        self.windows
            .get(ordinal)
            .map(Vec::len)
            .ok_or(WorkspaceParameterSourceError::MissingIndex)
    }
    fn window_unit(&self, ordinal: usize, member: usize) -> Result<usize> {
        self.windows
            .get(ordinal)
            .and_then(|w| w.get(member))
            .copied()
            .ok_or(WorkspaceParameterSourceError::MissingIndex)
    }
}
fn source() -> Source {
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("a"),
            ExecutionGroupSpec::with_dependencies("b", ["a"]),
        ],
        "b",
    )
    .unwrap();
    let row = |name, unit, index| RowData {
        binding: WeightBinding::new(name, "source", TensorSelection::Full, 4).unwrap(),
        shape: vec![2],
        dtype: WorkspaceDtype::Float32,
        representation: None,
        physical: 4,
        capacity: 4,
        owner: WorkspaceParameterOwner {
            unit,
            row: index,
            lifetime: WorkspaceParameterLifetime::Invocation,
        },
    };
    let mut units = vec![
        (
            OffloadUnitId::new("初").unwrap(),
            vec![row("owner", 0, 0), row("別", 0, 1)],
        ),
        (
            OffloadUnitId::new("alias").unwrap(),
            vec![row("tied", 0, 0)],
        ),
        (
            OffloadUnitId::new("last").unwrap(),
            vec![row("weight", 2, 0)],
        ),
    ];
    units[0].1[0].owner.lifetime = WorkspaceParameterLifetime::Trace;
    units[1].1[0].owner.lifetime = WorkspaceParameterLifetime::Trace;
    Source {
        layout: ExecutionUnitLayout::new(&graph, [2, 1]).unwrap(),
        units,
        requested: vec![0, 1, 2],
        windows: vec![vec![0, 1], vec![1], vec![2]],
        addresses: None,
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Buffers {
    units: Vec<WorkspaceParameterUnitRecord>,
    requested: Vec<WorkspaceParameterRequestRecord>,
    rows: Vec<WorkspaceParameterRecord>,
    names: Vec<u8>,
    shapes: Vec<i32>,
    roots: Vec<WorkspaceParameterRootRecord>,
    members: Vec<usize>,
}
impl Buffers {
    fn new(c: WorkspaceParameterCounts) -> Self {
        Self {
            units: vec![Default::default(); c.units],
            requested: vec![Default::default(); c.requested],
            rows: vec![Default::default(); c.rows],
            names: vec![0xaa; c.name_bytes],
            shapes: vec![-123; c.shape_elements],
            roots: vec![Default::default(); c.roots],
            members: vec![usize::MAX; c.window_members],
        }
    }
    fn lend(&mut self) -> WorkspaceParameterDestinations<'_> {
        WorkspaceParameterDestinations {
            units: &mut self.units,
            requested: &mut self.requested,
            rows: &mut self.rows,
            names: &mut self.names,
            shapes: &mut self.shapes,
            roots: &mut self.roots,
            window_members: &mut self.members,
        }
    }
}

#[test]
fn actual_rows_fill_names_shapes_aliases_and_nonuniform_windows() {
    let source = source();
    let counted = WorkspaceParameterSourceLoan::new(&source).count().unwrap();
    let c = counted.counts();
    assert_eq!(
        (
            c.units,
            c.requested,
            c.rows,
            c.roots,
            c.shape_elements,
            c.window_members
        ),
        (3, 3, 4, 3, 4, 4)
    );
    let mut buffers = Buffers::new(c);
    let table = counted.fill(buffers.lend()).unwrap();
    assert_eq!(table.unit_name(0), Some("初"));
    assert_eq!(table.row_name(1), Some("別"));
    assert_eq!(table.row_root(0), table.row_root(2));
    assert_ne!(table.row_root(0), table.row_root(1));
    assert_eq!(
        table.root_owner(0).unwrap().lifetime,
        WorkspaceParameterLifetime::Trace
    );
    assert_eq!(
        table.root_owner(1).unwrap().lifetime,
        WorkspaceParameterLifetime::Invocation
    );
    assert_eq!(table.window(0), Some(&[0, 1][..]));
    assert_eq!(table.window(1), Some(&[1][..]));
    assert_eq!(table.window(2), Some(&[2][..]));
    assert_eq!(table.row_layout(0).unwrap().bytes().unwrap(), 8);
    assert_eq!(table.row_physical_bytes(0), Some(4));
    assert_eq!(
        table.root_capacity_bytes(0),
        Some(4),
        "Float32 representation cannot enlarge the actual source capacity"
    );
    assert!(std::ptr::eq(
        table.binding(0, 0).unwrap(),
        &source.units[0].1[0].binding
    ));
    assert_eq!(
        table.requested_unit(0, source.layout.address(1).unwrap()),
        Err(WorkspaceParameterSourceError::SourceMismatch)
    );
}

#[test]
fn every_short_destination_rejects_before_any_write() {
    let source = source();
    for short in 0..7 {
        let counted = WorkspaceParameterSourceLoan::new(&source).count().unwrap();
        let mut buffers = Buffers::new(counted.counts());
        match short {
            0 => {
                buffers.units.pop();
            }
            1 => {
                buffers.requested.pop();
            }
            2 => {
                buffers.rows.pop();
            }
            3 => {
                buffers.names.pop();
            }
            4 => {
                buffers.shapes.pop();
            }
            5 => {
                buffers.roots.pop();
            }
            _ => {
                buffers.members.pop();
            }
        }
        let before = buffers.clone();
        assert!(matches!(
            counted.fill(buffers.lend()),
            Err(WorkspaceParameterSourceError::DestinationCapacity)
        ));
        assert_eq!(buffers, before);
    }
}

#[test]
fn host_style_same_content_names_remain_independent_and_source_is_not_content_identity() {
    let mut source = source();
    source.units[1].1[0].owner = WorkspaceParameterOwner {
        unit: 1,
        row: 0,
        lifetime: WorkspaceParameterLifetime::Invocation,
    };
    let first = WorkspaceParameterSourceLoan::new(&source);
    let same = WorkspaceParameterSourceLoan::new(&source);
    let other = self::source();
    assert!(first.same_source(&same));
    assert!(!first.same_source(&WorkspaceParameterSourceLoan::new(&other)));
    assert_eq!(first.count().unwrap().counts().roots, 4);
}

#[test]
fn malformed_owner_capacity_names_and_windows_never_publish_a_counted_source() {
    for kind in 0..6 {
        let mut source = source();
        match kind {
            0 => source.units[1].1[0].capacity = 9,
            1 => source.units[1].1[0].owner.unit = 99,
            2 => source.units[0].1[1].binding = source.units[0].1[0].binding.clone(),
            3 => source.windows[1].push(2),
            4 => source.units[0].1[0].owner = source.units[1].1[0].owner,
            _ => source.requested[1] = 0,
        }
        // Case 4 explicitly creates a canonical-owner cycle.
        if kind == 4 {
            source.units[0].1[0].owner.unit = 1;
        }
        assert!(
            WorkspaceParameterSourceLoan::new(&source).count().is_err(),
            "case {kind}"
        );
    }
}

#[test]
fn scalar_zero_negative_and_unbounded_rank_use_shared_geometry_worker() {
    let mut source = source();
    source.units[2].1[0].shape.clear();
    assert!(WorkspaceParameterSourceLoan::new(&source).count().is_ok());
    source.units[2].1[0].shape = vec![1; 129];
    assert_eq!(
        WorkspaceParameterSourceLoan::new(&source)
            .count()
            .unwrap()
            .counts()
            .shape_elements,
        132
    );
    source.units[2].1[0].shape = vec![0, 17];
    source.units[2].1[0].physical = 0;
    source.units[2].1[0].capacity = 0;
    assert!(WorkspaceParameterSourceLoan::new(&source).count().is_ok());
    source.units[2].1[0].shape = vec![0, -1];
    assert!(matches!(
        WorkspaceParameterSourceLoan::new(&source).count(),
        Err(WorkspaceParameterSourceError::Layout {
            unit: 2,
            row: 0,
            source: WorkspaceLayoutError::NegativeExtent
        })
    ));
    let counts = WorkspaceParameterCounts {
        rows: usize::MAX,
        ..Default::default()
    };
    assert_eq!(
        counts.element_bytes(),
        Err(WorkspaceParameterSourceError::Overflow)
    );
}

#[test]
fn owner_only_closure_unit_is_kept_without_becoming_an_execution_address() {
    let mut source = source();
    let id = OffloadUnitId::new("only-owner").unwrap();
    let row = RowData {
        binding: WeightBinding::new("canonical", "extra", TensorSelection::Full, 4).unwrap(),
        shape: vec![1],
        dtype: WorkspaceDtype::Int32,
        representation: None,
        physical: 4,
        capacity: 4,
        owner: WorkspaceParameterOwner {
            unit: 3,
            row: 0,
            lifetime: WorkspaceParameterLifetime::Trace,
        },
    };
    source.units.push((id, vec![row]));
    source.units[1].1[0].owner = source.units[3].1[0].owner;
    let counted = WorkspaceParameterSourceLoan::new(&source).count().unwrap();
    assert_eq!((counted.counts().units, counted.counts().requested), (4, 3));
    let mut buffers = Buffers::new(counted.counts());
    let table = counted.fill(buffers.lend()).unwrap();
    assert_eq!(table.unit_name(3), Some("only-owner"));
    assert_eq!(table.row_root(2), table.row_root(4));
    assert_eq!(table.unit_fresh_capacity_bytes(3), Some(48));
}

#[test]
fn fixed_control_layouts_include_actual_borrows_errors_and_nested_results() {
    let facts = workspace_parameter_control_layouts();
    let layout = |name| facts.iter().find(|entry| entry.0 == name).unwrap().1;
    assert_eq!(
        layout("source loan").bytes,
        std::mem::size_of::<&dyn WorkspaceParameterRows>()
    );
    assert_eq!(
        layout("destinations").bytes,
        7 * std::mem::size_of::<&mut [u8]>()
    );
    assert_eq!(
        layout("table result").bytes,
        std::mem::size_of::<Result<WorkspaceParameterTable<'_, '_>>>()
    );
    assert_eq!(
        layout("row result").alignment,
        std::mem::align_of::<Result<WorkspaceParameterRow<'_>>>()
    );
    assert!(layout("count result").bytes >= layout("counted source").bytes);
    let counts = WorkspaceParameterCounts {
        units: 1,
        requested: 1,
        rows: 1,
        name_bytes: 1,
        shape_elements: 1,
        roots: 1,
        window_members: 1,
    };
    assert_eq!(
        counts.element_bytes().unwrap(),
        std::mem::size_of::<WorkspaceParameterUnitRecord>()
            + std::mem::size_of::<WorkspaceParameterRequestRecord>()
            + std::mem::size_of::<WorkspaceParameterRecord>()
            + 1
            + std::mem::size_of::<i32>()
            + std::mem::size_of::<WorkspaceParameterRootRecord>()
            + std::mem::size_of::<usize>()
    );
}

#[path = "tests/projection.rs"]
mod projection;
