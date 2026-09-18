//! One prepaid synchronous root worker shared by prediction and target completion.
use crate::{MlxTensor, backend::error::Error};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{
    OriginalOperationMetadataCustody, OriginalSpeculativeBudgetCustody, WorkingMemoryError,
};
use safemlx::{
    Array, OperationEvent, OriginalScopeObserver, PreparedArrayClone, PreparedNestedRoots, Stream,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};
type RootIter<'a> = std::iter::Map<std::slice::Iter<'a, MlxTensor>, fn(&MlxTensor) -> &Array>;
fn native(value: &MlxTensor) -> &Array {
    value.as_array()
}
/// Arrays and unresolved native records stay in the enclosing Q through final
/// recovery. The native inner wait preserves architecture completion ordering.
pub(crate) struct NestedRootCompletion {
    values: Vec<MlxTensor>,
    clones: Vec<PreparedArrayClone>,
    nested: Option<PreparedNestedRoots<OriginalOperationMetadataCustody>>,
    equation_roots: usize,
    validations: usize,
    used: bool,
    completed: bool,
    funding: HostMetadataFunding,
}
impl NestedRootCompletion {
    pub(crate) fn control_bytes(roots: usize, validations: usize) -> Option<usize> {
        let count = roots.checked_add(validations)?;
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<RefCell<Self>>(),
            size_of::<std::cell::RefMut<'_, Self>>(),
            size_of::<Result<OperationEvent, safemlx::error::Exception>>(),
            size_of::<&mut dyn Iterator<Item = &MlxTensor>>(),
            size_of::<&mut dyn FnMut(&MlxTensor)>(),
            size_of::<(&mut Self, &OriginalScopeObserver, &Stream)>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<usize>>(),
            size_of::<(usize, Result<Array, safemlx::error::Exception>)>(),
            std::alloc::Layout::array::<MlxTensor>(count).ok()?.size(),
            std::alloc::Layout::array::<PreparedArrayClone>(count)
                .ok()?
                .size(),
            count.checked_mul(
                PreparedArrayClone::control_bytes()?
                    .checked_add(Array::inspection_clone_handle_bytes())?,
            )?,
            if count == 0 {
                0
            } else {
                PreparedNestedRoots::<OriginalOperationMetadataCustody>::control_bytes(count)?
            },
            PreparedNestedRoots::<OriginalOperationMetadataCustody>::submission_control_bytes::<
                RootIter<'_>,
            >()?,
            OperationEvent::control_bytes()?,
            size_of::<Option<OperationEvent>>(),
            size_of::<Result<(), safemlx::error::Exception>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn prepare(
        roots: usize,
        validations: usize,
        custody: OriginalSpeculativeBudgetCustody,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::prepare_metadata(roots, validations, custody.into(), funding)
    }
    /// Same paid root destination for a child of an admitted text/control role.
    pub(crate) fn prepare_metadata(roots: usize, validations: usize,
        custody: OriginalOperationMetadataCustody, funding: &HostMetadataFunding) -> Result<Self, Error> {
        funding
            .reserve_metadata(Self::control_bytes(roots, validations).ok_or_else(|| {
                Error::from(funding.metadata_source(WorkingMemoryError::Overflow))
            })?)
            .map_err(Error::WorkspacePlanning)?;
        let count = roots
            .checked_add(validations)
            .ok_or_else(|| Error::from(funding.metadata_source(WorkingMemoryError::Overflow)))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|cause| Error::from(funding.metadata_source(cause)))?;
        let mut clones = Vec::new();
        clones
            .try_reserve_exact(count)
            .map_err(|cause| Error::from(funding.metadata_source(cause)))?;
        for _ in 0..count {
            clones.push(
                PreparedArrayClone::try_prepare_for_inspection()
                    .map_err(|cause| Error::from(funding.metadata_source(cause)))?,
            );
        }
        let nested = if count == 0 {
            None
        } else {
            Some(
                PreparedNestedRoots::try_new(count, custody).map_err(|failure| {
                    let (cause, _custody) = failure.into_parts();
                    Error::from(funding.metadata_source(cause))
                })?,
            )
        };
        Ok(Self {
            values,
            clones,
            nested,
            equation_roots: roots,
            validations,
            used: false,
            completed: false,
            funding: funding.clone(),
        })
    }
    pub(crate) fn complete(
        &mut self,
        visit: impl FnOnce(&mut dyn FnMut(&MlxTensor)) -> Result<(), Error>,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<(), Error> {
        if self.used {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        self.used = true;
        let mut result = Ok(());
        visit(&mut |value| {
            if result.is_ok() {
                result = (|| {
                    let at = self.values.len();
                    if at == self.equation_roots {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    let value =
                        self.clones[at].fill_in_original_scope(value.as_array(), observer)?;
                    self.values.push(MlxTensor::from_array(value));
                    Ok(())
                })();
            }
        })?;
        result?;
        if self.values.len() != self.equation_roots {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        crate::backend::nn::tensor::append_active_token_validation_copies(
            &mut self.clones[self.equation_roots..],
            &mut self.values,
            self.equation_roots
                .checked_add(self.validations)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            observer,
        )?;
        if let Some(nested) = &mut self.nested {
            // The exact recipe may include an admitted router CPU stream.
            // Keep the nested host bank until those callbacks have settled;
            // the asynchronous single-GPU helper cannot establish that fact.
            nested.complete(
                self.values.iter().map(native as fn(&MlxTensor) -> &Array),
                observer,
                stream,
            )?;
        }
        crate::backend::nn::tensor::validate_active_original_token_validations(observer)?;
        self.completed = true;
        Ok(())
    }
    pub(crate) fn validate_complete(&self) -> Result<(), Error> {
        if self.completed {
            Ok(())
        } else {
            Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
        }
    }
}

mod projection;
pub(crate) use projection::{
    NestedCompletionActivation, NestedCompletionOwner, NestedCompletionProjection,
};

use eredu_nn::workspace::WorkspaceMetadataAllocation;
