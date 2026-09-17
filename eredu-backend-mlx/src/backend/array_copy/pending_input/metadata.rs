//! Borrowed cold visitation of the same saved-input preparation driver.
use super::*;
use eredu_nn::workspace::{
    WorkspaceLayoutError, WorkspaceLayoutList, WorkspaceLayoutView, WorkspaceOperationKindView,
    WorkspaceOperationView, visit_isolated_copy_operations,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum PendingTokenMetadataError<E> {
    #[error(transparent)]
    Source(#[from] PendingTokenSourceCause),
    #[error(transparent)]
    Geometry(#[from] WorkspaceLayoutError),
    #[error("pending input metadata visitor failed: {0}")]
    Visitor(#[source] E),
}

impl PreparedPendingTokenInput<'_> {
    /// Visits complete operation geometry from this actual immutable source.
    /// No owned shape, operation, context, tensor or formatted error is created.
    /// This is a constructor census input, not memory or execution authority.
    pub(crate) fn visit_workspace_operations<E>(
        &self,
        visit: &mut impl FnMut(WorkspaceOperationView<'_>) -> Result<(), E>,
    ) -> Result<(), PendingTokenMetadataError<E>> {
        self.validate_fixed()?;
        let descriptor = self
            .source
            .try_descriptor()
            .map_err(PendingTokenSourceCause::Descriptor)?;
        let dtype = if self.observed.dtype() == Dtype::Int32 {
            WorkspaceDtype::Int32
        } else {
            WorkspaceDtype::Uint32
        };
        let source = WorkspaceLayoutView::new(descriptor.shape(), dtype)?;
        let shape = self.shape.target();
        prepare(
            &mut BorrowedPendingInput {
                source,
                shape: &shape,
                visit,
            },
            dtype == WorkspaceDtype::Int32,
        )?;
        Ok(())
    }
}

struct BorrowedPendingInput<'a, 'v, V> {
    source: WorkspaceLayoutView<'a>,
    shape: &'a [i32; 2],
    visit: &'v mut V,
}
impl<'a, E, V> PendingInputMechanism for BorrowedPendingInput<'a, '_, V>
where
    V: FnMut(WorkspaceOperationView<'_>) -> Result<(), E>,
{
    type Value = WorkspaceLayoutView<'a>;
    type Error = PendingTokenMetadataError<E>;

    fn isolate(&mut self) -> Result<Self::Value, Self::Error> {
        visit_isolated_copy_operations(self.source, self.visit)
            .map_err(PendingTokenMetadataError::Visitor)?;
        Ok(self.source)
    }
    fn reshape(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error> {
        let output = WorkspaceLayoutView::new(self.shape, value.dtype())?;
        (self.visit)(WorkspaceOperationView {
            kind: WorkspaceOperationKindView::View("reshape"),
            inputs: WorkspaceLayoutList::Views(&[value]),
            outputs: WorkspaceLayoutList::Views(&[output]),
        })
        .map_err(PendingTokenMetadataError::Visitor)?;
        Ok(output)
    }
    fn cast(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error> {
        let output = WorkspaceLayoutView::new(self.shape, WorkspaceDtype::Uint32)?;
        (self.visit)(WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Elementwise("cast_u32"),
            inputs: WorkspaceLayoutList::Views(&[value]),
            outputs: WorkspaceLayoutList::Views(&[output]),
        })
        .map_err(PendingTokenMetadataError::Visitor)?;
        Ok(output)
    }
}
