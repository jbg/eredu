//! Owning layouts for a fresh context and its imported existing metadata.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

fn shared<T>() -> Option<usize> {
    Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<T>())
        .ok()
        .map(|(layout, _)| layout.pad_to_align().size())
}

impl WorkspaceContext {
    /// Requested six Rc allocations and fixed construction controls for the
    /// same `new` worker. Empty trace containers allocate no backing. Nested
    /// storage already owned by `M`, future equations and reports are separate.
    /// The caller must qualify the actual host allocator before admission.
    pub fn construction_bytes<M: WorkspaceMechanisms + 'static>() -> Option<usize> {
        let parts = [
            shared::<WorkspaceIdentity>()?,
            shared::<M>()?,
            shared::<RefCell<Trace>>()?,
            shared::<RefCell<Option<WorkspaceBorrowedStorage>>>()?,
            shared::<RefCell<Option<Vec<WorkspaceParameterRepresentation>>>>()?,
            Self::report_lifecycle_bytes().ok()?,
            size_of::<Self>(),
            size_of::<M>(),
            size_of::<WorkspaceIdentity>(),
            size_of::<Trace>(),
            size_of::<RefCell<Trace>>(),
            size_of::<Rc<M>>(),
            size_of::<Rc<dyn WorkspaceMechanisms>>(),
            size_of::<Rc<dyn facts::owned::FiniteWorkspaceFacts>>(),
            size_of::<fact_context::OperationFacts>(),
            size_of::<facts::owned::EmittedWorkspaceFacts>(),
            size_of::<facts::owned::FactEmissionFailure>(),
            size_of::<
                Result<
                    (
                        Option<fact_context::OperationFacts>,
                        Option<WorkspaceHostBound>,
                    ),
                    Error,
                >,
            >(),
            size_of::<WorkspaceMetadataEnvelope>(),
            size_of::<WorkspaceMetadataError>(),
            size_of::<Result<Self, WorkspaceMetadataError>>(),
            size_of::<Result<(), WorkspaceMetadataError>>(),
            size_of::<Option<WorkspaceMetadataFunding>>(),
            metadata_funding::reservation_control_bytes(),
            size_of::<RefCell<Option<WorkspaceBorrowedStorage>>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

impl WorkspaceExistingStorage {
    /// One actual imported Storage Rc. Its empty alias vector has no payload;
    /// existing context references are cloned without creating context owners.
    pub fn construction_bytes() -> Option<usize> {
        let parts = [
            shared::<Storage>()?,
            size_of::<Self>(),
            size_of::<Storage>(),
            size_of::<Rc<Storage>>(),
            size_of::<Vec<Rc<Storage>>>(),
            size_of::<Option<u64>>(),
            size_of::<&WorkspaceContext>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<usize>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

/// Fixed import refusal; no formatted diagnostic or source metadata is copied.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceImportError {
    /// The actual storage belongs to another metadata context.
    #[error("existing storage belongs to another workspace trace")]
    Context,
    /// The exact logical-shape destination could not reserve its storage.
    #[error("existing workspace shape reservation failed")]
    Reserve(#[source] TryReserveError),
    /// The enclosing context refused its finite constructor population.
    #[error(transparent)]
    Metadata(#[from] Error),
}
impl WorkspaceImportError {
    pub(super) fn ordinary(self) -> Error {
        match self {
            Self::Context => Error::backend("existing storage belongs to another workspace trace"),
            Self::Reserve(cause) => Error::backend_source(cause),
            Self::Metadata(cause) => cause,
        }
    }
}
impl WorkspaceTensor {
    /// One final logical shape plus fixed imported-handle construction controls.
    /// The shared backing root and context are separately constructed owners.
    pub fn imported_construction_bytes(rank: usize) -> Option<usize> {
        let parts = [
            WorkspaceLayout::construction_bytes(rank)?,
            size_of::<Self>(),
            size_of::<WorkspaceLayout>(),
            size_of::<WorkspaceLayoutView<'_>>(),
            size_of::<Vec<i32>>(),
            size_of::<WorkspaceImportError>(),
            size_of::<Result<Self, WorkspaceImportError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<(&WorkspaceExistingStorage, &WorkspaceContext)>(),
            size_of::<(&[i32], &mut Vec<i32>, usize)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Copies only the validated logical shape into its final allocation.
    /// Uses the same imported-handle constructor as `existing_with_storage`.
    pub fn import_existing(
        layout: WorkspaceLayoutView<'_>,
        storage: &WorkspaceExistingStorage,
        context: &WorkspaceContext,
    ) -> Result<Self, WorkspaceImportError> {
        if !Rc::ptr_eq(&storage.context, &context.identity) {
            return Err(WorkspaceImportError::Context);
        }
        context
            .charge_metadata(
                WorkspaceLayout::construction_bytes(layout.shape().len())
                    .ok_or_else(|| Error::from(WorkspaceMetadataError::Overflow))?,
            )
            .map_err(Error::from)?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(layout.shape().len())
            .map_err(WorkspaceImportError::Reserve)?;
        shape.extend_from_slice(layout.shape());
        Self::import_owned(
            WorkspaceLayout::from_owned_shape(shape, layout.dtype())
                .with_representation(layout.representation()),
            storage,
            context,
        )
    }
    pub(super) fn import_owned(
        layout: WorkspaceLayout,
        storage: &WorkspaceExistingStorage,
        context: &WorkspaceContext,
    ) -> Result<Self, WorkspaceImportError> {
        if !Rc::ptr_eq(&storage.context, &context.identity) {
            return Err(WorkspaceImportError::Context);
        }
        Ok(Self {
            layout,
            storage: storage.storage.clone(),
            context: context.identity.clone(),
            imported_existing: true,
        })
    }
}
