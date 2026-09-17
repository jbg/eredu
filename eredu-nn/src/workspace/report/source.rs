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
    pub scratch: u64,
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
            scratch: self.trace.scratch,
            host: self
                .trace
                .missing_host
                .is_empty()
                .then_some(self.trace.host_workspace),
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
            BORROWED => self.borrowed.expect("borrowed roots").roots()[i].storage.clone(),
            _ => unreachable!(),
        }
    }
    fn compare(&self, a: &Self::Root, b: &Self::Root) -> Ordering {
        Rc::as_ptr(a).cmp(&Rc::as_ptr(b))
    }
    fn bytes(&self, r: &Self::Root) -> Option<u64> {
        r.bytes
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
        Ok(Self { graph, input })
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
            scratch: self.input.scratch,
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
    fn edges(&self, r: &usize) -> usize {
        self.graph.nodes[*r].alias_count
    }
    fn edge(&self, r: &usize, i: usize) -> usize {
        self.graph.edges[self.graph.nodes[*r].alias_start + i]
    }
}
