//! Registered local-tail copy with actual canonical source/occupancy claims.
use super::*;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    error::Error,
    runtime::cache::{original_copy::Worker, residency::PreparedIndependentCacheManager},
};
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::WorkingMemoryError;
impl PagedKeyValueCache {
    pub(in crate::backend::runtime::cache) fn original_tail_operands(
        &self,
    ) -> Result<usize, WorkingMemoryError> {
        if self.retained_history.is_some() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        match (&self.tail_keys, &self.tail_values) {
            (None, None) if self.tail_start == self.offset => Ok(0),
            (Some(first), Some(second))
                if crate::backend::runtime::cache::residency::CacheBlockMetadata::floating_dtype_bytes(first.dtype()).is_some()
                    && crate::backend::runtime::cache::residency::CacheBlockMetadata::floating_dtype_bytes(second.dtype()).is_some() =>
            {
                Ok(2)
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(in crate::backend::runtime::cache) fn copy_registered_tail(
        &self,
        destination: &mut PreparedIndependentCacheManager,
        worker: &mut Worker<'_, '_>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        self.copy_tail_with(
            destination,
            &mut |array| worker.copy(IsolatedArrayCopy::new(array)),
            context,
        )
    }
    pub(in crate::backend::runtime::cache) fn copy_retained_tail(
        &self,
        destination: &mut PreparedIndependentCacheManager,
        stream: &safemlx::Stream,
        roots: &std::cell::RefCell<Vec<Array>>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        self.copy_retained_tail_with(destination, stream, roots, context, &mut |_| Ok(()))
    }
    pub(in crate::backend::runtime::cache) fn copy_retained_tail_with(
        &self,
        destination: &mut PreparedIndependentCacheManager,
        stream: &safemlx::Stream,
        roots: &std::cell::RefCell<Vec<Array>>,
        context: &WorkspaceContext,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        self.copy_tail_with(
            destination,
            &mut |array| {
                let copy = IsolatedArrayCopy::new(array).copy_retained(stream, roots)?;
                observe(&copy)?;
                Ok(copy)
            },
            context,
        )
    }
    fn copy_tail_with(
        &self,
        destination: &mut PreparedIndependentCacheManager,
        copy: &mut dyn FnMut(&Array) -> Result<Array, Error>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        self.original_tail_operands()
            .map_err(Error::PrefillControl)?;
        context
            .charge_metadata(
                Self::copy_tail_program_control_bytes()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let claim = self
            .with_workspace_source(context, |source| {
                destination.claim_copy_tail(source.manager_source(), context)
            })
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        // The source guard has returned; each real tail now uses the same
        // completed/registered copy binder as each sealed page in this worker.
        let first = self
            .tail_keys
            .as_ref()
            .map(|array| copy(array))
            .transpose()?;
        let second = self
            .tail_values
            .as_ref()
            .map(|array| copy(array))
            .transpose()?;
        let copy = self.copied_with_tails(destination.destination().clone(), first, second, None);
        destination
            .publish_copy_tail(
                claim,
                copy.tail_keys
                    .as_ref()
                    .zip(copy.tail_values.as_ref())
                    .map(|(first, second)| [first, second]),
                context,
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        Ok(copy)
    }
    pub(in crate::backend::runtime::cache) fn copy_tail_program_control_bytes() -> Option<usize> {
        let controls = [
            std::mem::size_of::<(
                &Self,
                &mut PreparedIndependentCacheManager,
                &mut dyn FnMut(&Array) -> Result<Array, Error>,
                &WorkspaceContext,
            )>(),
            std::mem::size_of::<(&safemlx::Stream, &std::cell::RefCell<Vec<Array>>)>(),
            std::mem::size_of::<&mut Worker<'_, '_>>(),
            std::mem::size_of::<&mut dyn FnMut(&Array) -> Result<(), Error>>(),
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<[Option<Array>; 2]>(),
            std::mem::size_of::<Option<[&Array; 2]>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
}
