//! Retained Host-load prefix feeding the existing isolated Array-copy worker.
use super::*;
use crate::backend::error::Error;
use eredu_nn::workspace::{WorkspaceMetadataError, HostMetadataFunding};
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{
    HostTransferDescriptor, ImmutableHostTransferBuffer, OperationEvent, OriginalScopeObserver,
};
use std::{mem::size_of, sync::Arc};

/// The source caller authenticates this immutable buffer through its exact
/// canonical/registered inventory and includes push_host_operand in the same
/// accepted copy program. This preparation creates no native permission.
/// Keep this entire value outside the source loan and through enclosing failure
/// recovery: native outputs/events retire before the source buffer and its H.
pub(crate) struct PreparedHostArrayCopy {
    loaded: Option<Array>,
    completion: Option<OperationEvent>,
    source: Arc<ImmutableHostTransferBuffer>,
    descriptor: HostTransferDescriptor<4>,
    attempted: bool,
    _funding: HostMetadataFunding,
}
impl PreparedHostArrayCopy {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<(&Arc<ImmutableHostTransferBuffer>, &WorkspaceContext)>(),
            size_of::<(&mut Self, &Stream, &RefCell<Vec<Array>>)>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Result<(Array, OperationEvent), Exception>>(),
            size_of::<Result<(), Exception>>(),
            HostTransferDescriptor::<4>::control_bytes()?,
            Array::descriptor_control_bytes()?,
            size_of::<OriginalCopyLayoutBuilder>(),
            size_of::<Result<HostTransferDescriptor<4>, safemlx::HostTransferMetadataError>>(),
            WorkspaceContext::metadata_source_bytes::<OriginalCopyCause>()?,
            WorkspaceContext::metadata_source_bytes::<safemlx::HostTransferMetadataError>()?,
            WorkspaceContext::metadata_source_bytes::<safemlx::ArrayDescriptorError>()?,
            WorkspaceContext::metadata_source_bytes::<WorkspaceMetadataError>()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn prepare(
        source: &Arc<ImmutableHostTransferBuffer>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        context
            .charge_metadata(
                Self::control_bytes().ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        // Reuse the actual source census validation. It cannot create a Scope,
        // copy account, registration, or destination from these metadata facts.
        let mut check = OriginalCopyLayoutBuilder::new();
        check
            .push_host_operand(source)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let descriptor = source
            .try_fixed_descriptor::<4>()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        Ok(Self {
            loaded: None,
            completion: None,
            source: Arc::clone(source),
            descriptor,
            attempted: false,
            _funding: funding,
        })
    }
    /// Same transfer and isolated-copy bodies as ordinary page copying. The
    /// caller's accepted source/copy plan pays the original Scope and three
    /// extra/reused recovery roots before this one-use program runs.
    pub(crate) fn copy_retained(
        &mut self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<Array, Error> {
        if self.attempted {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let observer = OriginalScopeObserver::try_current()?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        {
            let roots = roots
                .try_borrow()
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            if roots.capacity().saturating_sub(roots.len()) < 3 {
                return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
            }
        }
        self.attempted = true;
        let (array, completion) = self
            .source
            .copy_to_array_in_original_scope(stream, &observer)?;
        self.loaded = Some(array);
        self.completion = Some(completion);
        let array = self.loaded.as_ref().expect("retained load prefix");
        roots.borrow_mut().push(array.try_clone_handle()?);
        self.completion
            .as_ref()
            .expect("retained completion prefix")
            .synchronize()?;
        // Host metadata is immutable. Authenticate the actual returned native
        // shape/type before feeding the existing isolated-copy worker.
        let actual = array
            .try_descriptor()
            .map_err(|cause| Error::Neural(self._funding.metadata_source(cause)))?;
        if actual.shape() != self.descriptor.shape()
            || actual.facts().dtype() != self.descriptor.dtype()
        {
            return Err(Error::Neural(
                self._funding
                    .metadata_source(WorkspaceMetadataError::Unqualified),
            ));
        }
        drop(actual);
        Ok(IsolatedArrayCopy::new(array).copy_retained(stream, roots)?)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
