use super::*;
use source::{BORROWED, CLOSE, NEW, OPEN};
use std::mem::size_of;
pub(super) mod domains;
#[derive(Debug)]
pub(super) struct Node<R> {
    identity: R,
    bytes: Option<u64>,
    host_control_bytes: Option<u64>,
    maximum_allocations: usize,
    start: usize,
    len: usize,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    node: usize,
    next: usize,
}
#[derive(Debug)]
pub(super) struct Scratch<R> {
    nodes: Vec<Node<R>>,
    edges: Vec<usize>,
    work: Vec<Frame>,
    marks: Vec<u8>,
    // Stable node indices in sorted binary runs during insertion, then one
    // sorted index. Node/edge insertion and arithmetic observation order stay
    // unchanged. Merges reuse the otherwise idle DFS destination.
    index: Vec<usize>,
    index_sorted: bool,
    layout: WorkspaceReportLayout,
    diagnostic_overflow: bool,
}
impl<R> Scratch<R> {
    pub(super) fn new(
        layout: WorkspaceReportLayout,
        fail: Option<usize>,
    ) -> Result<Self, (Self, ConstructionCause)> {
        let mut s = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            work: Vec::new(),
            marks: Vec::new(),
            index: Vec::new(),
            index_sorted: false,
            layout,
            diagnostic_overflow: false,
        };
        if let Err(e) = layout.bytes_for::<R>() {
            return Err((s, ConstructionCause::Layout(e)));
        }
        macro_rules! reserve {
            ($index:expr,$field:ident,$count:expr) => {{
                // The failure seam selects this very destination and this same call.
                // Production callers always pass None.
                let count = if fail == Some($index) {
                    usize::MAX
                } else {
                    $count
                };
                if let Err(source) = s.$field.try_reserve_exact(count) {
                    return Err((
                        s,
                        ConstructionCause::Reserve {
                            destination: $index,
                            source,
                        },
                    ));
                }
            }};
        }
        reserve!(0, nodes, layout.nodes);
        reserve!(1, edges, layout.edges);
        reserve!(2, work, layout.nodes);
        reserve!(3, marks, layout.nodes);
        reserve!(4, index, layout.nodes);
        Ok(s)
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node<R>>()
            + self.edges.capacity() * size_of::<usize>()
            + self.work.capacity() * size_of::<Frame>()
            + self.marks.capacity()
            + self.index.capacity() * size_of::<usize>()
    }
}
fn add(
    a: Option<u64>,
    b: Option<u64>,
    diagnostic_overflow: bool,
) -> Result<Option<u64>, WorkspaceReportError> {
    a.zip(b)
        .map(|(a, b)| match a.checked_add(b) {
            Some(value) => Ok(Some(value)),
            None if diagnostic_overflow => Ok(None),
            None => Err(WorkspaceReportError::Overflow),
        })
        .transpose()
        .map(Option::flatten)
}
const C: u8 = 1;
const O: u8 = 2;
const B: u8 = 4;
impl<R: Clone> Scratch<R> {
    fn find<G: Graph<Root = R>>(&self, s: &G, r: &R) -> Option<usize> {
        let mut end = self.index.len();
        while end != 0 {
            let count = if self.index_sorted {
                end
            } else {
                1 << end.trailing_zeros()
            };
            let start = end - count;
            if let Ok(position) =
                self.index[start..end].binary_search_by(|&i| s.compare(&self.nodes[i].identity, r))
            {
                return Some(self.index[start + position]);
            }
            end = start;
        }
        None
    }
    fn merge_index<G: Graph<Root = R>>(&mut self, s: &G, start: usize, middle: usize, end: usize) {
        self.work.clear();
        let (mut left, mut right) = (start, middle);
        while left < middle || right < end {
            let take_left = right == end
                || (left < middle
                    && s.compare(
                        &self.nodes[self.index[left]].identity,
                        &self.nodes[self.index[right]].identity,
                    )
                    .is_le());
            let position = if take_left {
                let position = left;
                left += 1;
                position
            } else {
                let position = right;
                right += 1;
                position
            };
            // At most the N admitted nodes, before DFS has begun.
            self.work.push(Frame {
                node: self.index[position],
                next: 0,
            });
        }
        for (destination, source) in self.index[start..end].iter_mut().zip(&self.work) {
            *destination = source.node;
        }
        self.work.clear();
    }
    fn insert<G: Graph<Root = R>>(&mut self, s: &G, r: R) -> Result<usize, WorkspaceReportError> {
        if let Some(index) = self.find(s, &r) {
            return Ok(index);
        }
        if self.nodes.len() == self.layout.nodes {
            return Err(WorkspaceReportError::Capacity);
        }
        let i = self.nodes.len();
        let bytes = s.bytes(&r);
        self.nodes.push(Node {
            maximum_allocations: s.maximum_allocations(&r),
            host_control_bytes: s.host_control_bytes(&r),
            identity: r,
            bytes,
            start: 0,
            len: 0,
        });
        self.marks.push(0);
        self.index.push(i);
        let end = self.index.len();
        let merged = 1usize << end.trailing_zeros();
        let mut width = 1;
        while width < merged {
            self.merge_index(s, end - width * 2, end - width, end);
            width *= 2;
        }
        Ok(i)
    }
    fn fill<G: Graph<Root = R>>(&mut self, s: &G) -> Result<(), WorkspaceReportError> {
        self.nodes.clear();
        self.edges.clear();
        self.work.clear();
        self.marks.clear();
        self.index.clear();
        self.index_sorted = false;
        // Stream all seed lists. Repeated roots never enter the work buffer.
        for d in [CLOSE, NEW, OPEN] {
            for i in 0..s.roots(d) {
                self.insert(s, s.root(d, i))?;
            }
        }
        let mut i = 0;
        while i < self.nodes.len() {
            let root = self.nodes[i].identity.clone();
            let len = s.edges(&root);
            let start = self.edges.len();
            let end = start
                .checked_add(len)
                .ok_or(WorkspaceReportError::Overflow)?;
            if end > self.layout.edges {
                return Err(WorkspaceReportError::Capacity);
            }
            for j in 0..len {
                let child = self.insert(s, s.edge(&root, j))?;
                self.edges.push(child);
            }
            self.nodes[i].start = start;
            self.nodes[i].len = len;
            i += 1;
        }
        // Coalesce the remaining binary runs once. Every earlier insertion
        // merged only equal-sized runs, so no append shifts the full index.
        let end = self.index.len();
        if end != 0 {
            let mut middle = end - (1usize << end.trailing_zeros());
            while middle != 0 {
                let start = middle - (1usize << middle.trailing_zeros());
                self.merge_index(s, start, middle, end);
                middle = start;
            }
        }
        self.index_sorted = true;
        // Mark only borrowed identities already in the actual report union.
        // Unused borrowed roots never introduce storage or alias descendants.
        for i in 0..s.roots(BORROWED) {
            if let Some(index) = self.find(s, &s.root(BORROWED, i)) {
                self.marks[index] |= B;
            }
        }
        Ok(())
    }
    // Mark on entry. Reverse roots/children preserve the old LIFO visit order,
    // but at most N distinct frames are resident, including cyclic flat input.
    fn walk<G: Graph<Root = R>>(
        &mut self,
        s: &G,
        d: usize,
        bit: u8,
        tally: bool,
    ) -> Result<Option<u64>, WorkspaceReportError> {
        self.work.clear();
        let mut sum = Some(0);
        for r in (0..s.roots(d)).rev() {
            let root = s.root(d, r);
            let i = self.find(s, &root).ok_or(WorkspaceReportError::Source)?;
            if self.marks[i] & bit != 0 {
                continue;
            }
            self.enter(i, bit, &mut sum, tally)?;
            while let Some(frame) = self.work.last_mut() {
                if frame.next == 0 {
                    self.work.pop();
                    continue;
                }
                frame.next -= 1;
                let child = self.edges[self.nodes[frame.node].start + frame.next];
                if self.marks[child] & bit == 0 {
                    self.enter(child, bit, &mut sum, tally)?;
                }
            }
        }
        Ok(sum)
    }
    fn enter(
        &mut self,
        i: usize,
        bit: u8,
        sum: &mut Option<u64>,
        tally: bool,
    ) -> Result<(), WorkspaceReportError> {
        self.marks[i] |= bit;
        if tally && (bit == C || self.marks[i] & C == 0) {
            *sum = add(
                *sum,
                add(
                    self.nodes[i].bytes,
                    self.nodes[i].host_control_bytes,
                    self.diagnostic_overflow,
                )?,
                self.diagnostic_overflow,
            )?;
        }
        if self.work.len() == self.layout.nodes {
            return Err(WorkspaceReportError::Capacity);
        }
        self.work.push(Frame {
            node: i,
            next: self.nodes[i].len,
        });
        Ok(())
    }
    pub(super) fn report<G: Graph<Root = R>>(
        &mut self,
        s: &G,
    ) -> Result<WorkspaceReportScalars, WorkspaceReportError> {
        self.report_with_diagnostics(s, false)
    }
    pub(super) fn report_with_diagnostics<G: Graph<Root = R>>(
        &mut self,
        s: &G,
        diagnostic_overflow: bool,
    ) -> Result<WorkspaceReportScalars, WorkspaceReportError> {
        self.diagnostic_overflow = diagnostic_overflow;
        if self.nodes.capacity() < self.layout.nodes
            || self.edges.capacity() < self.layout.edges
            || self.work.capacity() < self.layout.nodes
            || self.marks.capacity() < self.layout.nodes
            || self.index.capacity() < self.layout.nodes
        {
            return Err(WorkspaceReportError::Capacity);
        }
        self.fill(s)?;
        let f = s.facts();
        let retained_state = self.walk(s, CLOSE, C, true)?;
        let maximum_allocations = self
            .nodes
            .iter()
            .zip(&self.marks)
            .filter(|(_, marks)| **marks & C != 0)
            .try_fold(0usize, |sum, (node, _)| {
                sum.checked_add(node.maximum_allocations)
            })
            .ok_or(WorkspaceReportError::Overflow)?;
        let closing_storage = WorkspaceStoragePopulation {
            bytes: retained_state,
            maximum_allocations,
        };
        let mut total = f.scratch;
        let mut persistent = Some(0);
        let mut controls = s.scratch_host_control_bytes()?;
        let mut retained_controls = Some(0);
        for i in 0..s.roots(NEW) {
            let n = self
                .find(s, &s.root(NEW, i))
                .ok_or(WorkspaceReportError::Source)?;
            total = add(total, self.nodes[n].bytes, self.diagnostic_overflow)?;
            controls = add(
                controls,
                self.nodes[n].host_control_bytes,
                self.diagnostic_overflow,
            )?;
            if self.marks[n] & C != 0 {
                persistent = add(persistent, self.nodes[n].bytes, self.diagnostic_overflow)?;
                retained_controls = add(
                    retained_controls,
                    self.nodes[n].host_control_bytes,
                    self.diagnostic_overflow,
                )?;
            }
        }
        if !f.complete {
            total = None;
        }
        let tensor_buffers = WorkspaceTensorBufferReport {
            total_bytes: total,
            retained_bytes: persistent,
            transient_bytes: total.zip(persistent).map(|(a, b)| a - b),
        };
        let total = add(
            add(total, controls, self.diagnostic_overflow)?,
            f.host,
            self.diagnostic_overflow,
        )?;
        let persistent = add(persistent, retained_controls, self.diagnostic_overflow)?;
        let transient = total.zip(persistent).map(|(a, b)| a - b);
        let state = if f.seeded {
            let displaced = self.walk(s, OPEN, O, true)?;
            Some(WorkspaceStateSpanReport {
                retained_bytes: retained_state,
                displaced_bytes: displaced,
                transient_bytes: add(transient, displaced, self.diagnostic_overflow)?,
            })
        } else {
            None
        };
        let opening_storage = if f.seeded {
            let mut bytes = Some(0u64);
            let mut maximum_allocations = 0usize;
            for (node, marks) in self.nodes.iter().zip(&self.marks) {
                if marks & O != 0 {
                    bytes = add(
                        bytes,
                        add(
                            node.bytes,
                            node.host_control_bytes,
                            self.diagnostic_overflow,
                        )?,
                        self.diagnostic_overflow,
                    )?;
                    maximum_allocations = maximum_allocations
                        .checked_add(node.maximum_allocations)
                        .ok_or(WorkspaceReportError::Overflow)?;
                }
            }
            Some(WorkspaceStoragePopulation {
                bytes,
                maximum_allocations,
            })
        } else {
            None
        };
        let residual = if f.residual {
            Some(if f.seeded {
                self.residual(s)?
            } else {
                WorkspaceReportResidual {
                    opening_storage: None,
                    closing_storage: None,
                    total_bytes: None,
                    retained_bytes: None,
                    displaced_bytes: None,
                    transient_bytes: None,
                }
            })
        } else {
            None
        };
        Ok(WorkspaceReportScalars {
            closing_storage,
            opening_storage,
            residual,
            state,
            total_bytes: total,
            retained_bytes: persistent,
            transient_bytes: transient,
            tensor_buffers,
            host_workspace_bytes: f.host,
        })
    }
    #[cfg(test)]
    pub(super) fn shorten_for_test(&mut self, destination: usize) {
        match destination {
            0 => self.nodes = Vec::with_capacity(self.layout.nodes - 1),
            1 => self.edges = Vec::with_capacity(self.layout.edges - 1),
            2 => self.work = Vec::with_capacity(self.layout.nodes - 1),
            3 => self.marks = Vec::with_capacity(self.layout.nodes - 1),
            4 => self.index = Vec::with_capacity(self.layout.nodes - 1),
            _ => panic!("bad test destination"),
        }
    }
    fn residual<G: Graph<Root = R>>(
        &mut self,
        s: &G,
    ) -> Result<WorkspaceReportResidual, WorkspaceReportError> {
        // The paid index already has the exact identity order used by the
        // legacy residual fold. No repeated sort or new destination is needed.
        let f = s.facts();
        let scratch = add(
            f.scratch,
            s.scratch_host_control_bytes()?,
            self.diagnostic_overflow,
        )?;
        let mut total = scratch;
        let mut transient = scratch;
        let mut retained = Some(0);
        let mut displaced = Some(0);
        let mut opening_storage = WorkspaceStoragePopulation::EMPTY;
        let mut closing_allocations = 0usize;
        for &i in &self.index {
            let n = &self.nodes[i];
            if self.marks[i] & B != 0 {
                continue;
            }
            let bytes = add(n.bytes, n.host_control_bytes, self.diagnostic_overflow)?;
            if self.marks[i] & O != 0 {
                opening_storage.bytes =
                    add(opening_storage.bytes, bytes, self.diagnostic_overflow)?;
                opening_storage.maximum_allocations = opening_storage
                    .maximum_allocations
                    .checked_add(n.maximum_allocations)
                    .ok_or(WorkspaceReportError::Overflow)?;
            }
            total = add(total, bytes, self.diagnostic_overflow)?;
            if self.marks[i] & C != 0 {
                retained = add(retained, bytes, self.diagnostic_overflow)?;
                closing_allocations = closing_allocations
                    .checked_add(n.maximum_allocations)
                    .ok_or(WorkspaceReportError::Overflow)?;
            } else {
                transient = add(transient, bytes, self.diagnostic_overflow)?;
                if self.marks[i] & O != 0 {
                    displaced = add(displaced, bytes, self.diagnostic_overflow)?;
                }
            }
        }
        if !f.complete {
            total = None;
            transient = None;
        }
        Ok(WorkspaceReportResidual {
            opening_storage: Some(opening_storage),
            closing_storage: Some(WorkspaceStoragePopulation {
                bytes: retained,
                maximum_allocations: closing_allocations,
            }),
            total_bytes: add(total, f.host, self.diagnostic_overflow)?,
            retained_bytes: retained,
            displaced_bytes: displaced,
            transient_bytes: add(transient, f.host, self.diagnostic_overflow)?,
        })
    }
}
