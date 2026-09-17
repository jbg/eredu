//! Exact one-payload original B loan; no raw tensor/account constructor.
use super::*;
use eredu_runtime::{
    input::{OriginalPreparedWorkspaceSource, PreparedModelInputOwner, PreparedModelInputSource},
    working_memory::WorkingMemoryError,
};

pub(crate) struct OriginalPreparedArrayCopySource<'a> {
    array: &'a Array,
    prepared: &'a PreparedModelInputOwner<crate::MlxTensor>,
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
        })
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
        self.prepared.project_single_payload_with_metadata(context)
    }
}
