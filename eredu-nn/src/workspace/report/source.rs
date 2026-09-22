use super::*;
use std::cmp::Ordering;
pub(super) const OPEN: usize = 0;
pub(super) const NEW: usize = 1;
pub(super) const CLOSE: usize = 2;
pub(super) const BORROWED: usize = 3;
#[derive(Clone, Copy)]
pub(super) struct Facts {
    pub seeded: bool,
    pub residual: bool,
    pub scratch: Option<u64>,
    pub host: Option<u64>,
    pub complete: bool,
}
// Private, closed adapters: a caller cannot insert a allocating source callback.
pub(super) trait Graph {
    type Root: Clone;
    fn facts(&self) -> Facts;
    fn roots(&self, domain: usize) -> usize;
    fn root(&self, domain: usize, index: usize) -> Self::Root;
    fn compare(&self, a: &Self::Root, b: &Self::Root) -> Ordering;
    fn bytes(&self, root: &Self::Root) -> Option<u64>;
    fn placement<'a>(&'a self, _root: &'a Self::Root) -> Option<&'a eredu_core::MemoryPlacement> {
        None
    }
    fn host_control_bytes(&self, _root: &Self::Root) -> Option<u64> {
        Some(0)
    }
    fn scratch_host_control_bytes(&self) -> Result<Option<u64>, WorkspaceReportError> {
        Ok(Some(0))
    }
    fn maximum_allocations(&self, root: &Self::Root) -> usize {
        usize::from(self.bytes(root) != Some(0))
    }
    fn edges(&self, root: &Self::Root) -> usize;
    fn edge(&self, root: &Self::Root, index: usize) -> Self::Root;
}
pub(super) struct Ordinary<'a> {
    pub trace: &'a Trace,
    pub retained: &'a [WorkspaceTensor],
    pub borrowed: Option<&'a WorkspaceBorrowedStorage>,
}
impl Graph for Ordinary<'_> {
    type Root = Rc<Storage>;
    fn facts(&self) -> Facts {
        Facts {
            seeded: self.trace.opening_state.is_some(),
            residual: self.borrowed.is_some(),
            scratch: (!self.trace.scratch_overflow).then_some(self.trace.scratch),
            host: (!self.trace.host_staging_incomplete).then_some(self.trace.host_workspace),
            complete: self.trace.missing.is_empty(),
        }
    }
    fn roots(&self, d: usize) -> usize {
        match d {
            OPEN => self.trace.opening_state.as_ref().map_or(0, Vec::len),
            NEW => self.trace.allocations.len(),
            CLOSE => self.retained.len(),
            BORROWED => self.borrowed.map_or(0, |b| b.roots().len()),
            _ => unreachable!(),
        }
    }
    fn root(&self, d: usize, i: usize) -> Self::Root {
        match d {
            OPEN => self.trace.opening_state.as_ref().expect("seeded opening")[i].clone(),
            NEW => self.trace.allocations[i].clone(),
            CLOSE => self.retained[i].storage.clone(),
            BORROWED => self.borrowed.expect("borrowed roots").roots()[i]
                .storage
                .clone(),
            _ => unreachable!(),
        }
    }
    fn compare(&self, a: &Self::Root, b: &Self::Root) -> Ordering {
        Rc::as_ptr(a).cmp(&Rc::as_ptr(b))
    }
    fn bytes(&self, r: &Self::Root) -> Option<u64> {
        r.bytes
    }
    fn placement<'a>(&'a self, r: &'a Self::Root) -> Option<&'a eredu_core::MemoryPlacement> {
        r.placement.as_ref()
    }
    fn host_control_bytes(&self, r: &Self::Root) -> Option<u64> {
        r.host_control_bytes
    }
    fn scratch_host_control_bytes(&self) -> Result<Option<u64>, WorkspaceReportError> {
        let mut total = scratch_controls(&self.trace.placed_scratch)?;
        for source in self
            .trace
            .scratch_sources
            .iter()
            .filter(|source| source.domain_population().is_some())
        {
            total = match (total, source.host_control_bytes()) {
                (Some(a), Some(b)) => Some(a.checked_add(b).ok_or(WorkspaceReportError::Overflow)?),
                _ => None,
            };
        }
        Ok(total)
    }
    fn maximum_allocations(&self, r: &Self::Root) -> usize {
        r.maximum_allocations
    }
    fn edges(&self, r: &Self::Root) -> usize {
        r.possible_aliases.len()
    }
    fn edge(&self, r: &Self::Root, i: usize) -> Self::Root {
        r.possible_aliases[i].clone()
    }
}
pub(super) struct Flat<'a> {
    graph: WorkspaceReportGraph<'a>,
    input: WorkspaceReportInputs<'a>,
    placements: Option<&'a [Option<eredu_core::MemoryPlacement>]>,
    host_controls: Option<&'a [Option<u64>]>,
    scratch_controls: &'a [WorkspaceScratchAllocation],
}
impl<'a> Flat<'a> {
    pub(super) fn new(
        graph: WorkspaceReportGraph<'a>,
        input: WorkspaceReportInputs<'a>,
    ) -> Result<Self, WorkspaceReportError> {
        for roots in [
            input.opening.unwrap_or(&[]),
            input.allocations,
            input.closing,
            input.borrowed.unwrap_or(&[]),
        ] {
            if roots.iter().any(|i| *i >= graph.nodes.len()) {
                return Err(WorkspaceReportError::Source);
            }
        }
        // Actual allocation roots occur once in the shared trace. Public flat
        // metadata must preserve that producer invariant too.
        for (i, root) in input.allocations.iter().enumerate() {
            if input.allocations[..i].contains(root) {
                return Err(WorkspaceReportError::Source);
            }
        }
        Ok(Self {
            graph,
            input,
            placements: None,
            host_controls: None,
            scratch_controls: &[],
        })
    }
    pub(super) fn with_placements(
        mut self,
        placements: &'a [Option<eredu_core::MemoryPlacement>],
        host_controls: Option<&'a [Option<u64>]>,
    ) -> Result<Self, WorkspaceReportError> {
        if placements.len() != self.graph.nodes.len() {
            return Err(WorkspaceReportError::Source);
        }
        if host_controls.is_some_and(|values| values.len() != self.graph.nodes.len()) {
            return Err(WorkspaceReportError::Source);
        }
        self.host_controls = host_controls;
        self.placements = Some(placements);
        Ok(self)
    }
    pub(super) fn with_controls(
        mut self,
        controls: Option<&'a [Option<u64>]>,
        scratch: &'a [WorkspaceScratchAllocation],
    ) -> Result<Self, WorkspaceReportError> {
        if controls.is_some_and(|values| values.len() != self.graph.nodes.len()) {
            return Err(WorkspaceReportError::Source);
        }
        self.host_controls = controls;
        self.scratch_controls = scratch;
        Ok(self)
    }
    fn seeds(&self, d: usize) -> &[usize] {
        match d {
            OPEN => self.input.opening.unwrap_or(&[]),
            NEW => self.input.allocations,
            CLOSE => self.input.closing,
            BORROWED => self.input.borrowed.unwrap_or(&[]),
            _ => unreachable!(),
        }
    }
}
impl Graph for Flat<'_> {
    type Root = usize;
    fn facts(&self) -> Facts {
        Facts {
            seeded: self.input.opening.is_some(),
            residual: self.input.borrowed.is_some(),
            scratch: Some(self.input.scratch),
            host: self.input.host_workspace,
            complete: self.input.tensor_complete,
        }
    }
    fn roots(&self, d: usize) -> usize {
        self.seeds(d).len()
    }
    fn root(&self, d: usize, i: usize) -> usize {
        self.seeds(d)[i]
    }
    fn compare(&self, a: &usize, b: &usize) -> Ordering {
        a.cmp(b)
    }
    fn bytes(&self, r: &usize) -> Option<u64> {
        self.graph.nodes[*r].bytes
    }
    fn placement<'a>(&'a self, r: &'a usize) -> Option<&'a eredu_core::MemoryPlacement> {
        self.placements?.get(*r)?.as_ref()
    }
    fn host_control_bytes(&self, r: &usize) -> Option<u64> {
        self.host_controls.map_or(Some(0), |values| values[*r])
    }
    fn scratch_host_control_bytes(&self) -> Result<Option<u64>, WorkspaceReportError> {
        scratch_controls(self.scratch_controls)
    }
    fn edges(&self, r: &usize) -> usize {
        self.graph.nodes[*r].alias_count
    }
    fn edge(&self, r: &usize, i: usize) -> usize {
        self.graph.edges[self.graph.nodes[*r].alias_start + i]
    }
}

fn scratch_controls(
    values: &[WorkspaceScratchAllocation],
) -> Result<Option<u64>, WorkspaceReportError> {
    let mut total = Some(0u64);
    for value in values {
        total = match total.zip(value.host_control_bytes) {
            Some((a, b)) => Some(a.checked_add(b).ok_or(WorkspaceReportError::Overflow)?),
            None => None,
        };
    }
    Ok(total)
}
