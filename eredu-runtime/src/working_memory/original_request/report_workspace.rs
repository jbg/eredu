//! Closed flat-report destination custody under the existing accepted account.
use super::*;
use eredu_nn::workspace::{
    WorkspaceReportConstructionError, WorkspaceReportError, WorkspaceReportGraph,
    WorkspaceReportInputs, WorkspaceReportLayout, WorkspaceReportNode, WorkspaceReportScalars,
    WorkspaceReportWorkspace,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::size_of,
    sync::{Arc, RwLock, RwLockWriteGuard, TryLockError},
};

// Only the private complete producer may contribute this exact population.
// Scalar counts never turn the native MissingInspectionStorage path complete.
#[derive(Clone, Copy, Debug)]
pub(super) struct ReportRecipe {
    pub nodes: usize,
    pub edges: usize,
    pub roots: usize,
}
fn arithmetic(_: WorkspaceReportError) -> WorkingMemoryError {
    WorkingMemoryError::Overflow
}
impl ReportRecipe {
    fn layout(self) -> Result<WorkspaceReportLayout, WorkingMemoryError> {
        WorkspaceReportLayout::new(self.nodes, self.edges).map_err(arithmetic)
    }
    pub(super) fn bytes(self) -> Result<usize, WorkingMemoryError> {
        // Pinned std's SOLID implementation lazily creates a kernel object.
        // Its exact producer is unfinished; the other selected implementations
        // have inline try-write/unlock paths. Reject before the single Q.
        if cfg!(target_os = "solid_asp3") {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let arc = Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(Layout::new::<ReportBody>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let mut n = self.layout()?.heap_bytes().map_err(arithmetic)?;
        for term in [
            Layout::array::<WorkspaceReportNode>(self.nodes)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            Layout::array::<usize>(self.edges)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            Layout::array::<usize>(self.roots)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            self.layout()?.control_bytes().map_err(arithmetic)?,
            arc,
            size_of::<ReportBody>(),
            size_of::<ReportOwner>(),
            size_of::<Storage>(),
            size_of::<ReportRecipe>(),
            size_of::<WorkspaceReportInputs<'_>>(),
            size_of::<WorkspaceReportGraph<'_>>(),
            size_of::<[usize; 5]>(),
            size_of::<[&[usize]; 4]>(),
            size_of::<ReportFailure>(),
            size_of::<ReportUseError>(),
            size_of::<RwLockWriteGuard<'_, Storage>>(),
            size_of::<TryLockError<RwLockWriteGuard<'_, Storage>>>(),
            size_of::<
                Result<RwLockWriteGuard<'_, Storage>, TryLockError<RwLockWriteGuard<'_, Storage>>>,
            >(),
            size_of::<Result<WorkspaceReportScalars, ReportUseError>>(),
            size_of::<Result<WorkspaceReportScalars, WorkspaceReportError>>(),
        ] {
            n = n.checked_add(term).ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(n)
    }
}
#[derive(Debug, Default)]
pub(super) struct Storage {
    nodes: Vec<WorkspaceReportNode>,
    edges: Vec<usize>,
    roots: Vec<usize>,
    workspace: Option<WorkspaceReportWorkspace>,
    recipe: Option<ReportRecipe>,
}
#[derive(Debug, thiserror::Error)]
pub(super) enum ReportFailure {
    #[error("original report destination {destination} reserve failed")]
    Reserve {
        destination: usize,
        #[source]
        source: TryReserveError,
    },
    #[error("{0}")]
    Workspace(#[source] WorkspaceReportConstructionError),
    #[error("{0}")]
    Accounting(#[source] WorkingMemoryError),
}
impl Storage {
    pub(super) fn construct(
        &mut self,
        recipe: ReportRecipe,
        fail: Option<usize>,
    ) -> Result<(), ReportFailure> {
        // Count/layout checks happened before Q; preserve a fixed retained cause
        // even if an internal caller violates that already checked invariant.
        let layout = recipe.layout().map_err(ReportFailure::Accounting)?;
        macro_rules! reserve {
            ($i:expr,$field:ident,$count:expr) => {{
                let count = if fail == Some($i) { usize::MAX } else { $count };
                self.$field
                    .try_reserve_exact(count)
                    .map_err(|source| ReportFailure::Reserve {
                        destination: $i,
                        source,
                    })?;
            }};
        }
        reserve!(8, nodes, recipe.nodes);
        reserve!(9, edges, recipe.edges);
        reserve!(10, roots, recipe.roots);
        self.workspace =
            Some(WorkspaceReportWorkspace::new(layout).map_err(ReportFailure::Workspace)?);
        self.recipe = Some(recipe);
        Ok(())
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<WorkspaceReportNode>()
            + self.edges.capacity() * size_of::<usize>()
            + self.roots.capacity() * size_of::<usize>()
            + self
                .workspace
                .as_ref()
                .map_or(0, WorkspaceReportWorkspace::retained_heap_bytes)
    }
    fn report(
        &mut self,
        nodes: &[WorkspaceReportNode],
        edges: &[usize],
        input: WorkspaceReportInputs<'_>,
    ) -> Result<WorkspaceReportScalars, WorkspaceReportError> {
        let recipe = self.recipe.ok_or(WorkspaceReportError::Source)?;
        let roots = [
            input.opening.unwrap_or(&[]),
            input.allocations,
            input.closing,
            input.borrowed.unwrap_or(&[]),
        ];
        let len = roots
            .iter()
            .try_fold(0usize, |n, v| n.checked_add(v.len()))
            .ok_or(WorkspaceReportError::Overflow)?;
        if nodes.len() > recipe.nodes || edges.len() > recipe.edges || len > recipe.roots {
            return Err(WorkspaceReportError::Capacity);
        }
        // Validate the complete source and all roots before modifying the prior
        // owned source image. No clone of model/source payload occurs here.
        WorkspaceReportGraph::new(nodes, edges)?;
        for list in roots {
            if list.iter().any(|i| *i >= nodes.len()) {
                return Err(WorkspaceReportError::Source);
            }
        }
        for (i, index) in input.allocations.iter().enumerate() {
            if input.allocations[..i].contains(index) {
                return Err(WorkspaceReportError::Source);
            }
        }
        self.nodes.clear();
        self.edges.clear();
        self.roots.clear();
        self.nodes.extend_from_slice(nodes);
        self.edges.extend_from_slice(edges);
        let mut ranges = [0usize; 5];
        for (i, list) in roots.into_iter().enumerate() {
            self.roots.extend_from_slice(list);
            ranges[i + 1] = self.roots.len();
        }
        let view = WorkspaceReportGraph::new(&self.nodes, &self.edges)?;
        let input = WorkspaceReportInputs {
            opening: input.opening.map(|_| &self.roots[ranges[0]..ranges[1]]),
            allocations: &self.roots[ranges[1]..ranges[2]],
            closing: &self.roots[ranges[2]..ranges[3]],
            borrowed: input.borrowed.map(|_| &self.roots[ranges[3]..ranges[4]]),
            ..input
        };
        self.workspace
            .as_mut()
            .ok_or(WorkspaceReportError::Source)?
            .report(view, input)
    }
}
struct ReportBody {
    storage: RwLock<Storage>,
    retained_bytes: usize,
    selected_model: OriginalPreparedHostInput,
    input: OriginalPreparedHostInput,
    // Last, after flat buffers, inline lock shell, and actual source aliases.
    _reservation: WorkingMemoryReservation,
}
impl std::fmt::Debug for ReportBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not expose Debug's general RwLock read path either.
        f.debug_struct("OriginalReportBody")
            .field("retained_bytes", &self.retained_bytes)
            .finish_non_exhaustive()
    }
}
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(super) enum ReportUseCause {
    #[error("original report destinations already have an active consumer")]
    Busy,
    #[error("original report destinations are poisoned")]
    Poisoned,
    #[error("{0}")]
    Boundary(#[source] WorkingMemoryError),
}
#[derive(Debug)]
pub(super) struct ReportUseError {
    cause: ReportUseCause,
    // Every failure retains the same buffers/source/Q, without a new allocation.
    _owner: ReportOwner,
}
impl std::fmt::Display for ReportUseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for ReportUseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl ReportUseError {
    pub(super) fn working_memory(&self) -> WorkingMemoryError {
        match &self.cause {
            // The closed caller runs inside one genuine prediction; a second
            // consumer is an active-step violation, never a blocking retry.
            ReportUseCause::Busy => WorkingMemoryError::TextStepActive,
            ReportUseCause::Poisoned => WorkingMemoryError::Poisoned,
            ReportUseCause::Boundary(error) => error.clone(),
        }
    }
    #[cfg(test)]
    pub(super) fn cause(&self) -> ReportUseCause {
        self.cause.clone()
    }
}
#[derive(Debug)]
pub(super) struct ReportOwner(Option<Arc<ReportBody>>);
impl Clone for ReportOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live report"))))
    }
}
impl Drop for ReportOwner {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            drop(Arc::into_inner(v));
        }
    }
}
impl ReportOwner {
    pub(super) fn new(
        storage: Storage,
        sources: &Sources,
        reservation: WorkingMemoryReservation,
    ) -> Self {
        let retained_bytes = storage.retained_bytes();
        Self(Some(Arc::new(ReportBody {
            storage: RwLock::new(storage),
            retained_bytes,
            selected_model: sources.selected_model.clone(),
            input: sources.input.clone(),
            _reservation: reservation,
        })))
    }
    fn error(&self, cause: ReportUseCause) -> ReportUseError {
        ReportUseError {
            cause,
            _owner: self.clone(),
        }
    }
    pub(super) fn report(
        &self,
        sources: &Sources,
        nodes: &[WorkspaceReportNode],
        edges: &[usize],
        input: WorkspaceReportInputs<'_>,
    ) -> Result<WorkspaceReportScalars, ReportUseError> {
        let owner = self.0.as_ref().expect("live report owner");
        if !owner.selected_model.same_source(&sources.selected_model)
            || !owner.input.same_source(&sources.input)
        {
            return Err(self.error(ReportUseCause::Boundary(
                WorkingMemoryError::IdentityMismatch,
            )));
        }
        // Closed production access: no read, blocking write, downgrade, mapped
        // guard or raw-lock export can enqueue a waiter or initialize Thread.
        let mut storage = owner.storage.try_write().map_err(|error| {
            self.error(match error {
                TryLockError::WouldBlock => ReportUseCause::Busy,
                TryLockError::Poisoned(_) => ReportUseCause::Poisoned,
            })
        })?;
        let result = storage.report(nodes, edges, input);
        drop(storage);
        result.map_err(|e| {
            self.error(ReportUseCause::Boundary(match e {
                WorkspaceReportError::Overflow => WorkingMemoryError::Overflow,
                _ => WorkingMemoryError::IdentityMismatch,
            }))
        })
    }
    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.0.as_ref().unwrap().retained_bytes
    }
    #[cfg(test)]
    pub(super) fn hold_for_test(&self) -> RwLockWriteGuard<'_, Storage> {
        self.0.as_ref().unwrap().storage.try_write().unwrap()
    }
}
