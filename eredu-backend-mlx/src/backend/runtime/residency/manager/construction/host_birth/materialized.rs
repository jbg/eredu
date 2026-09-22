//! Transformed outputs use genuine detached leaf reads and the shared native
//! recipe equation, retaining every asynchronous source until exact completion.
use super::*;
use crate::backend::runtime::checkpoint::{
    recipe::prepare_ordinary_recipe_from_funded_leaves,
    store::{MaterializationView, PendingWeightMaterialization},
};
use crate::backend::submission_recovery::{PreparedRecovery, Retention, Status};
use eredu_core::HostPreparationAuthority;
use safemlx::{ImmutableHostTransferBuffer, OperationEvent, ScopedPhysicalBackingObserver};
mod initialization;
pub(in crate::backend::runtime::residency::manager::construction) mod quotation;
pub(in crate::backend::runtime::residency::manager::construction) use initialization::{
    ConstructionPlan, Session,
};

pub(super) struct Resources {
    hosts: Vec<ImmutableHostTransferBuffer>,
    arrays: Vec<Array>,
    events: Vec<OperationEvent>,
    detached: Option<HostTransferBuffer>,
    pending: Vec<PendingWeightMaterialization>,
}
impl Retention for Resources {
    fn observe(&self, _: Status) {}
}

#[derive(Debug, thiserror::Error)]
pub(in crate::backend::runtime::residency::manager::construction) enum Failure {
    #[error(transparent)]
    Accounting(#[from] WorkingMemoryError),
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Backend(#[from] crate::backend::error::Error),
    #[error("materialized source leaf or output identity differs")]
    Identity,
    #[error(transparent)]
    Native(#[from] safemlx::error::Exception),
    #[error(transparent)]
    Read(#[from] eredu_checkpoint::store::DetachedReadFailure<ManagerCustody>),
    #[error(transparent)]
    Recipe(#[from] WeightRecipeError),
    #[error(transparent)]
    Reserve(#[from] std::collections::TryReserveError),
    #[error("materialized source recovery: {0:?}")]
    Scope(safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)]
    Retirement(#[from] crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause),
    #[error("materialized source completion is unresolved")]
    Completion,
}

pub(in crate::backend::runtime::residency::manager::construction) fn execute(
    plan: &read_source_plan::MaterializedReadPlan,
    source: &source::OriginalReadSources,
    leaves: std::ops::Range<usize>,
    output: &mut HostTransferBuffer,
    context: MaterializationView<'_>,
    observer: &ScopedPhysicalBackingObserver,
    host: HostPreparationAuthority,
) -> Result<(), Failure> {
    if leaves.len() != plan.leaves.len() {
        return Err(Failure::Identity);
    }
    let mut resources = Resources {
        hosts: Vec::new(),
        arrays: Vec::new(),
        events: Vec::new(),
        detached: None,
        pending: Vec::new(),
    };
    resources.hosts.try_reserve_exact(leaves.len())?;
    resources
        .arrays
        .try_reserve_exact(leaves.len().checked_add(2).ok_or(Failure::Identity)?)?;
    resources
        .events
        .try_reserve_exact(leaves.len().checked_add(2).ok_or(Failure::Identity)?)?;
    let prepared = PreparedRecovery::new(resources, host.clone())
        .map_err(|error| Failure::Scope(error.cause))?;
    let mut recovery = prepared
        .try_begin()
        .map_err(|error| Failure::Scope(error.cause))?;
    recovery.configure_scope(|scope| scope.bind_physical_observer(observer))?;
    let result = (|| {
        let resources = recovery.retention_mut();
        let mut ordinal = 0usize;
        let mut read_failure = None;
        let prepared = {
            let mut produce = |key: &str,
                               selection: &eredu_checkpoint::store::TensorSelection,
                               stream: &Stream| {
                let leaf = plan
                    .leaves
                    .get(ordinal)
                    .ok_or(WeightRecipeError::ShapeMismatch)?;
                if leaf.key != key || &leaf.selection != selection {
                    return Err(WeightRecipeError::ShapeMismatch);
                }
                let mut buffer = HostTransferBuffer::new(
                    leaf.read.shape(),
                    leaf.read.dtype(),
                    safemlx::HostTransferPolicy::Transfer,
                )?;
                let index = leaves
                    .start
                    .checked_add(ordinal)
                    .ok_or(WeightRecipeError::ShapeMismatch)?;
                let end = index
                    .checked_add(1)
                    .ok_or(WeightRecipeError::ShapeMismatch)?;
                let read = source
                    .read_slice(index..end)
                    .ok_or(WeightRecipeError::ShapeMismatch)?;
                if let Err(cause) = read.read_many_into(&mut [buffer.as_bytes_mut()?]) {
                    read_failure = Some(cause);
                    return Err(WeightRecipeError::ShapeMismatch);
                }
                resources.hosts.push(buffer.freeze());
                let (value, event) = resources
                    .hosts
                    .last()
                    .expect("inserted leaf")
                    .copy_to_array(stream)?
                    .into_parts();
                resources.arrays.push(value.try_clone_handle()?);
                resources.events.push(event.into());
                ordinal += 1;
                Ok(value)
            };
            prepare_ordinary_recipe_from_funded_leaves(&plan.recipe, context, &mut produce, &host)
        };
        if let Some(cause) = read_failure {
            return Err(Failure::Read(cause));
        }
        let (value, pending) = prepared?.into_parts();
        resources.pending = pending;
        if ordinal != plan.leaves.len() || !resources.pending.is_empty() {
            return Err(Failure::Identity);
        }
        resources.arrays.push(value);
        // This is the equation's one quoted final-root completion. Completing
        // it before detachment keeps the following transfer traversal limited
        // to its actual ready input and one CopyToHost producer.
        let event = safemlx::transforms::async_eval_with_operation_event(resources.arrays.last())?;
        resources.events.push(event.into());
        resources
            .events
            .last()
            .expect("retained equation completion")
            .synchronize()?;
        let (detached, event) = HostTransferBuffer::copy_from_array(
            resources.arrays.last().expect("retained recipe output"),
            safemlx::HostTransferPolicy::Transfer,
            context.source_stream(),
        )?
        .into_parts();
        resources.detached = Some(detached);
        resources.events.push(event.into());
        resources
            .events
            .last()
            .expect("retained host completion")
            .synchronize()?;
        let bytes = resources
            .detached
            .as_ref()
            .expect("retained host output")
            .as_bytes()?;
        let destination = output.as_bytes_mut()?;
        if bytes.len() != destination.len() {
            return Err(Failure::Identity);
        }
        destination.copy_from_slice(bytes);
        Ok(())
    })();
    recovery.seal();
    // Callback refusals are returned only after this exact recovery settles.
    // Native failure or an unobservable scope preserves its existing fencing.
    let status = recovery.finish()?;
    if !status.settled || status.failed || status.blocked {
        return Err(Failure::Completion);
    }
    result
}
