//! Physical module visits emitted by the actual shared equation worker.
use super::*;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// One actual physical-module entry and its eventual completion boundary.
/// This is quote metadata, not a native lease, source proof, or grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspacePredictionModuleCall {
    pub module: usize,
    /// Number of actual state/output handles retained by this invocation.
    pub retained_roots: Option<usize>,
    /// Zero-based order of completed module boundaries (nested owners close first).
    pub completion: Option<usize>,
}
struct Recorded {
    calls: RefCell<Vec<WorkspacePredictionModuleCall>>,
    completed: Cell<usize>,
    modules: usize,
    // The concrete metadata destinations retire before their paying context.
    context: WorkspaceContext,
}
/// Shared source-bound metadata sink. Bindings issue a handle only for an
/// authenticated dense module row; the shared invocation worker records use.
pub struct WorkspacePredictionInvocations(Rc<Recorded>);
/// An authenticated metadata module row in the above source inventory.
pub struct WorkspacePredictionInvocation {
    module: usize,
    recorded: Rc<Recorded>,
}
#[derive(Debug, thiserror::Error)]
enum InvocationError {
    #[error("prediction module invocation differs from its source quote")]
    Source,
    #[error("prediction module invocation did not complete")]
    Incomplete,
}
impl WorkspacePredictionInvocations {
    pub fn new(modules: usize, context: &WorkspaceContext) -> Result<Self, Error> {
        controls::<(Self, Recorded)>(context)?;
        let calls = context.metadata_vec(0)?;
        Ok(Self(context.metadata_rc(Recorded {
            calls: RefCell::new(calls),
            completed: Cell::new(0),
            modules,
            context: context.clone(),
        })?))
    }
    /// Aliases the paid recorder; no invocation or source permission is created.
    pub fn alias(&self) -> Result<Self, Error> {
        controls::<Self>(&self.0.context)?;
        Ok(Self(Rc::clone(&self.0)))
    }
    pub fn module(
        &self,
        module: usize,
        context: &WorkspaceContext,
    ) -> Result<WorkspacePredictionInvocation, Error> {
        controls::<WorkspacePredictionInvocation>(context)?;
        if !self.0.context.shares_trace(context) || module >= self.0.modules {
            return Err(context.metadata_source(InvocationError::Source));
        }
        Ok(WorkspacePredictionInvocation {
            module,
            recorded: Rc::clone(&self.0),
        })
    }
    /// Lends the complete ordered entry population after successful equations.
    /// An unfinished prefix remains retained but cannot become a complete quote.
    pub fn with_completed<T, F>(&self, inspect: F) -> Result<T, Error>
    where
        F: FnOnce(&[WorkspacePredictionModuleCall]) -> Result<T, Error>,
    {
        controls::<(
            F,
            std::cell::Ref<'_, Vec<WorkspacePredictionModuleCall>>,
            Result<T, Error>,
        )>(&self.0.context)?;
        let calls = self
            .0
            .calls
            .try_borrow()
            .map_err(|_| self.0.context.metadata_source(InvocationError::Source))?;
        if self.0.completed.get() != calls.len()
            || calls
                .iter()
                .any(|call| call.completion.is_none() || call.retained_roots.is_none())
        {
            return Err(self.0.context.metadata_source(InvocationError::Incomplete));
        }
        inspect(&calls)
    }
}
impl WorkspacePredictionInvocation {
    pub(super) fn begin(&self, context: &WorkspaceContext) -> Result<usize, Error> {
        controls::<(
            usize,
            WorkspacePredictionModuleCall,
            std::cell::RefMut<'_, Vec<WorkspacePredictionModuleCall>>,
        )>(context)?;
        if !self.recorded.context.shares_trace(context) {
            return Err(context.metadata_source(InvocationError::Source));
        }
        let mut calls = self
            .recorded
            .calls
            .try_borrow_mut()
            .map_err(|_| context.metadata_source(InvocationError::Source))?;
        context.reserve_metadata_vec(&mut calls, 1)?;
        let entry = calls.len();
        calls.push(WorkspacePredictionModuleCall {
            module: self.module,
            retained_roots: None,
            completion: None,
        });
        Ok(entry)
    }
    pub(super) fn complete(
        &self,
        entry: usize,
        roots: usize,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        controls::<(
            usize,
            usize,
            std::cell::RefMut<'_, Vec<WorkspacePredictionModuleCall>>,
        )>(context)?;
        if !self.recorded.context.shares_trace(context) {
            return Err(context.metadata_source(InvocationError::Source));
        }
        let next = self
            .recorded
            .completed
            .get()
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut calls = self
            .recorded
            .calls
            .try_borrow_mut()
            .map_err(|_| context.metadata_source(InvocationError::Source))?;
        let call = calls
            .get_mut(entry)
            .ok_or_else(|| context.metadata_source(InvocationError::Source))?;
        if call.module != self.module || call.completion.is_some() {
            return Err(context.metadata_source(InvocationError::Source));
        }
        call.retained_roots = Some(roots);
        call.completion = Some(self.recorded.completed.get());
        self.recorded.completed.set(next);
        Ok(())
    }
}
