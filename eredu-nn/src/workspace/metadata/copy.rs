//! Borrowed visitation of the shared isolated-copy program.
use super::*;
use crate::{IsolatedCopyMechanism, isolated_copy};

/// Visits the same contiguous/deep-copy program without constructing a context,
/// shape, value, or operation owner. The source loan remains with the caller;
/// the callback owns its own failure and no descriptor or reference is retained.
pub fn visit_isolated_copy_operations<E>(
    source: WorkspaceLayoutView<'_>,
    visit: &mut impl FnMut(WorkspaceOperationView<'_>) -> Result<(), E>,
) -> Result<(), E> {
    isolated_copy(Visitor {
        source,
        visit: RefCell::new(visit),
    })
}

struct Visitor<'a, 'b, F> {
    source: WorkspaceLayoutView<'a>,
    visit: RefCell<&'b mut F>,
}
impl<E, F: FnMut(WorkspaceOperationView<'_>) -> Result<(), E>> Visitor<'_, '_, F> {
    fn operation(&self, kind: WorkspaceOperationKindView<'_>) -> Result<(), E> {
        let layouts = [self.source];
        (self.visit.borrow_mut())(WorkspaceOperationView {
            kind,
            inputs: WorkspaceLayoutList::Views(&layouts),
            outputs: WorkspaceLayoutList::Views(&layouts),
        })
    }
}
impl<E, F: FnMut(WorkspaceOperationView<'_>) -> Result<(), E>> IsolatedCopyMechanism
    for Visitor<'_, '_, F>
{
    type Value = ();
    type Error = E;
    fn contiguous(&self) -> Result<(), E> {
        self.operation(WorkspaceOperationKindView::Contiguous)
    }
    fn deep_copy(&self, (): ()) -> Result<(), E> {
        self.operation(WorkspaceOperationKindView::DeepCopy)
    }
}
