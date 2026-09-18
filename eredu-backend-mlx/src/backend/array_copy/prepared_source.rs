//! Exact one-payload original B loan; no raw tensor/account constructor.
use super::*;
use eredu_runtime::{
    input::{OriginalPreparedWorkspaceSource, PreparedModelInputOwner, PreparedModelInputSource},
    working_memory::WorkingMemoryError,
};

pub(crate) struct OriginalPreparedArrayCopySource<'a> {
    array: &'a Array,
    prepared: &'a PreparedModelInputOwner<crate::MlxTensor>,
    index: usize,
}
impl<'a> OriginalPreparedArrayCopySource<'a> {
    /// Both views were produced by the same original B compiler. Callers cannot
    /// substitute an arbitrary Array or manufacture its account/root identity.
    pub(crate) fn from_input<N>(
        input: &'a PreparedModelInputSource<crate::MlxTensor, Array, N>,
    ) -> Result<Self, WorkingMemoryError> {
        let prepared = input
            .prepared()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let parts = input.parts().ok_or(WorkingMemoryError::IdentityMismatch)?;
        let [part] = parts.as_ref() else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        if !part.metadata().is_empty() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            array: part.payload().value(),
            prepared,
            index: 0,
        })
    }
    /// Lends only a payload physically retained by this completed original
    /// owner. Equal descriptors or an unrelated tensor cannot select a source.
    pub(crate) fn from_payload(
        prepared: &'a PreparedModelInputOwner<crate::MlxTensor>,
        payload: &'a crate::MlxTensor,
    ) -> Result<Self, WorkingMemoryError> {
        if prepared.original_source().is_none() { return Err(WorkingMemoryError::IdentityMismatch); }
        let index = prepared.parts().iter().position(|part| std::ptr::eq(part.payload().value(), payload))
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(Self { array: payload.as_array(), prepared, index })
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<(
                &PreparedModelInputSource<crate::MlxTensor, Array, ()>,
                &PreparedModelInputOwner<crate::MlxTensor>,
                &[eredu_runtime::PreparedInputPart<Array>],
            )>(),
            size_of::<Option<&eredu_runtime::input::SharedPreparedInputParts<Array>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn array(&self) -> &Array {
        self.array
    }
    pub(super) fn project(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(WorkspaceTensor, OriginalPreparedWorkspaceSource), eredu_nn::Error> {
        self.prepared.project_payload_with_metadata(self.index, context)
    }
}
