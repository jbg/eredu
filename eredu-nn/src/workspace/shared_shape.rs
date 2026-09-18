//! Immutable shape ownership shared by layout and tensor metadata clones.
use super::*;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    mem::{size_of, size_of_val},
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

/// The closed owner exposes no Weak or mutable/owning shape extraction.
/// Keeping Vec inside a sized shared block permits the final block to retire
/// before its axes, matching enclosing source/error custody drop ordering.
pub(super) struct SharedShape(Option<Arc<Vec<i32>>>);

impl SharedShape {
    fn new(shape: Vec<i32>) -> Self {
        Self(Some(Arc::new(shape)))
    }
}
impl Clone for SharedShape {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for SharedShape {
    fn drop(&mut self) {
        if let Some(shape) = self.0.take() {
            drop(Arc::into_inner(shape));
        }
    }
}
impl Deref for SharedShape {
    type Target = [i32];
    fn deref(&self) -> &[i32] {
        self.0.as_deref().expect("live workspace shape").as_slice()
    }
}
impl fmt::Debug for SharedShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}
impl PartialEq for SharedShape {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for SharedShape {}

impl WorkspaceLayout {
    /// Requested final axes, one actual shared owner, and fixed constructor
    /// controls. Cloning the resulting immutable layout creates no allocation.
    /// This is metadata geometry only; callers retain their own source/custody.
    pub fn construction_bytes(rank: usize) -> Option<usize> {
        Layout::array::<i32>(rank)
            .ok()?
            .size()
            .checked_add(Self::shape_owner_bytes()?)?
            .checked_add(Self::shape_control_bytes()?)
    }

    pub(super) fn shape_owner_bytes() -> Option<usize> {
        Some(
            Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<Vec<i32>>())
                .ok()?
                .0
                .pad_to_align()
                .size(),
        )
    }

    pub(super) fn shape_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<(WorkspaceDtype,Option<WorkspaceRepresentation>)>(),
            size_of::<Result<Self, Error>>(),
            size_of::<SharedShape>(),
            size_of::<Vec<i32>>(),
            size_of::<Arc<Vec<i32>>>(),
            size_of::<Option<Arc<Vec<i32>>>>(),
            size_of::<Option<Vec<i32>>>(),
            size_of::<WorkspaceLayoutView<'_>>(),
            size_of::<Result<WorkspaceLayoutView<'_>, WorkspaceLayoutError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<WorkspaceLayoutError>(),
            size_of::<TryReserveError>(),
            size_of::<(&[i32], &mut Vec<i32>, usize)>(),
            size_of::<Layout>(),
            Error::retained_source_construction_bytes::<WorkspaceLayoutError>()?
                .max(Error::retained_source_construction_bytes::<TryReserveError>()?),
            size_of::<Error>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    // Only checked borrowed layouts or the already validated ordinary worker
    // supply this final owner. No shape is copied while wrapping or cloning it.
    pub(super) fn from_owned_shape(shape: Vec<i32>, dtype: WorkspaceDtype) -> Self {
        Self {
            shape: SharedShape::new(shape),
            dtype,
            representation: None,
        }
    }
}

impl WorkspaceContext {
    /// Validates and constructs an immutable logical shape under this
    /// context's actual metadata census or finite destination allowance.
    /// This does not construct a tensor or authorize any numerical backing.
    pub fn layout(&self, shape: &[i32], dtype: WorkspaceDtype) -> Result<WorkspaceLayout, Error> {
        let view = WorkspaceLayoutView::new(shape, dtype);
        let bytes = if view.is_ok() {
            WorkspaceLayout::construction_bytes(shape.len())
        } else {
            // A failed geometry query has no shape buffers. Its exact retained
            // diagnostic still belongs to the same admitted metadata attempt.
            WorkspaceLayout::shape_control_bytes()
        }
        .ok_or_else(|| Error::from(WorkspaceMetadataError::Overflow))?;
        self.charge_metadata(bytes)?;
        let view = view.map_err(Error::backend_retained_source)?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(view.shape().len())
            .map_err(Error::backend_retained_source)?;
        shape.extend_from_slice(view.shape());
        Ok(WorkspaceLayout::from_owned_shape(shape, dtype))
    }
}
