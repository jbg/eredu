//! Read-only traversal of tensor roots retained by neural state and outputs.

/// Borrows every tensor root retained by an owner, including independent carry
/// that is not an ancestor of its visible output. Composites delegate to their
/// component owners so native completion never needs to reconstruct family state.
///
/// Traversal does not allocate or evaluate tensors, establish completion, or
/// transfer ownership. The caller must retain the owner through its submission.
/// Aliases and duplicate roots are permitted; this is not a backing-byte census.
pub trait RetainedTensorRoots<T> {
    /// Visits all retained tensor roots without reading their values.
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T));
}
