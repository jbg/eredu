//! Host checkpoint destinations bound to the actual retained state and Work.
use crate::backend::error::Error;
use eredu_core::{HostMetadataFunding, HostPreparationAuthority};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::working_memory::{
    InferenceRequest, InferenceRetention, InferenceSpanWorkspacePlan, InferenceTextStep,
    InferenceWorkspaceSpan, WorkingMemoryError,
};
use eredu_runtime::{HostSlotMetadata, HostSlotTable, SharedStateLayout};
use safemlx::error::Exception;
use std::{
    cell::Cell,
    mem::{size_of, size_of_val},
    ops::Range,
    rc::Rc,
};

#[derive(Debug)]
struct Source {
    layout: SharedStateLayout,
    outer: HostSlotMetadata,
    nested: Vec<HostSlotMetadata>,
    retention: InferenceRetention,
    plan: InferenceSpanWorkspacePlan,
    used: Vec<Cell<bool>>,
    prefill: usize,
    bytes: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct OrdinaryCheckpointProgram {
    source: Rc<Source>,
    funding: HostMetadataFunding,
}
#[derive(Clone, Debug)]
pub(crate) struct OrdinaryCheckpointWork {
    program: OrdinaryCheckpointProgram,
    request: InferenceRequest,
    rows: Range<usize>,
}
#[derive(Debug, thiserror::Error)]
#[error("ordinary state checkpoint: {cause}")]
struct Failure {
    #[source]
    cause: Cause,
    _host: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Source(#[from] WorkingMemoryError),
    #[error(transparent)]
    Metadata(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    #[error(transparent)]
    Native(#[from] Exception),
}
fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
fn failure(cause: impl Into<Cause>, host: &HostPreparationAuthority) -> Exception {
    Exception::from_retained_source(Failure {
        cause: cause.into(),
        _host: host.clone(),
    })
}
fn controls() -> Option<usize> {
    let parts = [
        size_of::<OrdinaryCheckpointWork>(),
        size_of::<Loan>(),
        size_of::<Failure>(),
        size_of::<(&SharedStateLayout, &HostSlotMetadata, &InferenceRetention)>(),
        size_of::<Result<Option<Loan>, Exception>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        Exception::retained_source_control_bytes::<Failure>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(crate) fn table_bytes<T>(count: usize) -> Option<usize> {
    WorkspaceContext::metadata_vec_bytes::<T>(count)?
        .checked_add(HostSlotTable::<T>::host_source_control_bytes()?)?
        .checked_add(size_of::<(
            &[T],
            &HostMetadataFunding,
            Vec<T>,
            Result<HostSlotTable<T>, Exception>,
        )>())
}
impl OrdinaryCheckpointProgram {
    pub(crate) fn prepare<'a, T, I>(
        layout: &SharedStateLayout,
        outer: &HostSlotMetadata,
        slots: &[T],
        nested: I,
        retention: &InferenceRetention,
        plan: &InferenceSpanWorkspacePlan,
        context: &WorkspaceContext,
    ) -> Result<Self, Error>
    where
        I: Iterator<Item = (&'a HostSlotMetadata, usize)> + Clone,
    {
        let funding = context.metadata_funding().ok_or_else(invalid)?;
        context
            .charge_metadata(size_of::<(Source, Self, I, Result<Self, Error>)>())
            .map_err(|e| Error::Neural(e.into()))?;
        let mut bytes = table_bytes::<T>(slots.len())
            .ok_or_else(overflow)?
            .checked_add(retention.checkpoint_clone_bytes().ok_or_else(overflow)?)
            .and_then(|n| n.checked_add(controls()?))
            .ok_or_else(overflow)?;
        let mut retained = context
            .metadata_vec(nested.clone().count())
            .map_err(|e| Error::Neural(e.into()))?;
        for (metadata, child) in nested {
            bytes = bytes.checked_add(child).ok_or_else(overflow)?;
            retained.push(metadata.clone());
        }
        let forwards = plan.generation_forward_count().ok_or_else(invalid)?;
        let prefill = plan
            .generation_records()
            .ok_or_else(invalid)?
            .take_while(|record| matches!(record.span(), InferenceWorkspaceSpan::Prefill(_)))
            .count();
        let mut used = context
            .metadata_vec(forwards)
            .map_err(|e| Error::Neural(e.into()))?;
        used.extend((0..forwards).map(|_| Cell::new(false)));
        let retention = retention
            .clone_with_host_source(&funding)
            .map_err(|e| Error::Neural(context.metadata_source(e)))?;
        let source = context
            .metadata_rc(Source {
                layout: layout.clone(),
                outer: outer.clone(),
                nested: retained,
                retention,
                plan: plan.clone(),
                used,
                prefill,
                bytes,
            })
            .map_err(|e| Error::Neural(e.into()))?;
        Ok(Self { source, funding })
    }
    pub(crate) fn metadata_bytes(&self) -> Option<u64> {
        u64::try_from(self.source.bytes.checked_mul(self.source.used.len())?).ok()
    }
    pub(crate) fn for_step(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<OrdinaryCheckpointWork, Error> {
        if step.request().geometry() != self.source.plan.geometry()
            || prefill != (step.attempt() == 0)
        {
            return Err(invalid());
        }
        let rows = if prefill {
            0..self.source.prefill
        } else {
            let index = usize::try_from(step.attempt().checked_sub(1).ok_or_else(invalid)?)
                .map_err(|_| invalid())?
                .checked_add(self.source.prefill)
                .filter(|n| *n < self.source.used.len())
                .ok_or_else(invalid)?;
            index..index.checked_add(1).ok_or_else(overflow)?
        };
        Ok(OrdinaryCheckpointWork {
            program: self.clone(),
            request: step.request().clone(),
            rows,
        })
    }
}
impl OrdinaryCheckpointWork {
    pub(crate) fn metadata_bytes(&self) -> Option<usize> {
        self.program.source.bytes.checked_mul(self.rows.len())
    }
}
/// Runtime loan retains the exact Work payer through all partial construction.
pub(crate) struct Loan {
    funding: HostMetadataFunding,
    host: HostPreparationAuthority,
}
impl Loan {
    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(crate) fn retain(&self, cause: Exception) -> Exception {
        failure(cause, &self.host)
    }
    pub(crate) fn retention(
        &self,
        source: &InferenceRetention,
    ) -> Result<InferenceRetention, Exception> {
        source
            .clone_with_host_source(&self.funding)
            .map_err(|e| failure(e, &self.host))
    }
    pub(crate) fn table<T>(
        &self,
        source: &[T],
        mut copy: impl FnMut(&T) -> Result<T, Exception>,
    ) -> Result<HostSlotTable<T>, Exception> {
        self.funding
            .reserve_metadata(size_of::<(
                &[T],
                &HostMetadataFunding,
                Vec<T>,
                Result<HostSlotTable<T>, Exception>,
            )>())
            .map_err(|e| failure(e, &self.host))?;
        let mut values = self
            .funding
            .metadata_vec(source.len())
            .map_err(|e| failure(e, &self.host))?;
        for value in source {
            values.push(copy(value).map_err(|e| self.retain(e))?);
        }
        HostSlotTable::from_boxed_with_host_source(values.into_boxed_slice(), &self.funding)
            .map_err(|e| failure(e, &self.host))
    }
}
/// Metadata tokens select the retained source; the caller must still borrow the
/// actual tables whose contents its existing checkpoint worker copies.
pub(crate) fn begin<'a>(
    layout: &SharedStateLayout,
    outer: &HostSlotMetadata,
    nested: impl Iterator<Item = &'a HostSlotMetadata>,
    retention: &InferenceRetention,
) -> Result<Option<Loan>, Exception> {
    let Some(owner) = crate::backend::nn::shared::current_ordinary_execution_owner()? else {
        return Ok(None);
    };
    let work = owner
        .checkpoint()
        .ok_or_else(|| failure(WorkingMemoryError::IdentityMismatch, owner.host()))?;
    let source = &work.program.source;
    if !source.layout.same_storage(layout) || !source.outer.same_storage(outer) {
        return Err(failure(WorkingMemoryError::IdentityMismatch, owner.host()));
    }
    let mut actual = nested;
    for expected in &source.nested {
        if actual.next().is_none_or(|v| !expected.same_storage(v)) {
            return Err(failure(WorkingMemoryError::IdentityMismatch, owner.host()));
        }
    }
    if actual.next().is_some() {
        return Err(failure(WorkingMemoryError::IdentityMismatch, owner.host()));
    }
    retention
        .validate_checkpoint_source(&source.retention, &work.request)
        .map_err(|e| failure(e, owner.host()))?;
    let funding = owner
        .metadata_funding()
        .ok_or_else(|| failure(WorkingMemoryError::IdentityMismatch, owner.host()))?
        .clone();
    let row = source.used[work.rows.clone()]
        .iter()
        .find(|row| !row.get())
        .ok_or_else(|| failure(WorkingMemoryError::ExecutionFenced, owner.host()))?;
    row.set(true);
    funding
        .reserve_metadata(
            controls().ok_or_else(|| failure(WorkingMemoryError::Overflow, owner.host()))?,
        )
        .map_err(|e| failure(e, owner.host()))?;
    Ok(Some(Loan {
        funding,
        host: owner.host().clone(),
    }))
}
