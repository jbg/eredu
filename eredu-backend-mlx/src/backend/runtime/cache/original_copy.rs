//! Exact registered-source leaf copies for independent prediction caches.
//! This constructs native representations only; the caller authenticates the
//! loaded source/header and preserves exclusive, settled access throughout.
use super::{
    kv::{CompressedLatentCache, PreparedCompressedCopy},
    state::{CompletedResidentSource, MlxPoolingAttentionCache, PreparedPoolingAttentionCopy},
};
use crate::backend::{
    array_copy::{IsolatedArrayCopy, RegisteredArrayCopy, RegisteredArrayCopyCustody},
    error::Error,
    nn::workspace::MlxMetalWorkspaceMechanisms,
    OriginalCopyEnvironment,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{Array, PrefillRootsRuntime, PreparedArrayClone};
use std::mem::{size_of, size_of_val};

/// No bare cache/Clone exit. Consumers must place the payload before this
/// account-only host authority in the enclosing prepared cache/lane owner.
#[derive(Debug)]
pub(crate) struct PreparedPredictionCacheCopy<C> {
    value: C,
    host: HostPreparationAuthority,
}
impl<C> PreparedPredictionCacheCopy<C> {
    pub(crate) fn into_parts(self) -> (C, HostPreparationAuthority) {
        (self.value, self.host)
    }
}
struct CopyCustody {
    _copies: Vec<RegisteredArrayCopyCustody>,
    // The vector backing, its account controls and prepared clone handles
    // retire before the H which paid for their construction.
    _funding: HostMetadataFunding,
}
#[derive(thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    // A counted host-slot error may own a completed native prefix. Its arrays
    // retire with the cause before their actual registered numerical receipts.
    _copies: Vec<RegisteredArrayCopyCustody>,
    _funding: HostMetadataFunding,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Failure")
            .field("cause", &self.cause)
            .field("completed_copies", &self._copies.len())
            .finish_non_exhaustive()
    }
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn reserve(funding: &HostMetadataFunding, bytes: Option<usize>) -> Result<(), Error> {
    funding
        .reserve_metadata(bytes.ok_or_else(|| memory(WorkingMemoryError::Overflow))?)
        .map_err(Error::WorkspacePlanning)
}
fn paid<E: std::error::Error + Send + Sync + 'static>(
    funding: &HostMetadataFunding,
    cause: E,
) -> Error {
    Error::Neural(funding.metadata_source(cause))
}
pub(super) struct Worker<'a, 'environment> {
    environment: &'a OriginalCopyEnvironment<'environment>,
    initialized: &'a PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &'a HostMetadataFunding,
    capacity: &'a eredu_core::MemoryLimits,
    completed: Option<&'a CompletedResidentSource>,
    copies: Vec<RegisteredArrayCopyCustody>,
    count: usize,
}
impl Worker<'_, '_> {
    pub(super) fn slots(&mut self, count: usize) -> Result<(), Error> {
        self.copies = self.funding.metadata_vec(count).map_err(Error::Neural)?;
        self.count = count;
        Ok(())
    }
    pub(super) fn copy(&mut self, source: IsolatedArrayCopy<'_>) -> Result<Array, Error> {
        if self.copies.len() == self.count {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        // The existing copy worker binds every actual complete backing to the
        // registered source and admits a fresh numerical account before work.
        // Its completion/recovery/publication must succeed before extraction.
        let completed = source.copy_completed(
            self.completed,
            self.environment,
            self.initialized,
            self.mechanisms,
            self.funding,
            self.capacity,
        )?;
        let (array, custody) = completed.into_parts();
        self.copies.push(custody);
        Ok(array)
    }
}
fn alias(funding: &HostMetadataFunding, source: &Array) -> Result<Array, Error> {
    // Native numerical copy is complete. Retain its descriptor with the exact
    // separately-paid handle slot, without ordinary hooks or another Scope.
    reserve(
        funding,
        PreparedArrayClone::control_bytes()
            .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes())),
    )?;
    let mut slot =
        PreparedArrayClone::try_prepare_for_inspection().map_err(|e| paid(funding, e))?;
    slot.fill_for_inspection(source)
        .map_err(|e| paid(funding, e))
}
pub(super) fn construct<C, F>(
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    capacity: &eredu_core::MemoryLimits,
    preparation_controls: Option<usize>,
    operation: F,
) -> Result<PreparedPredictionCacheCopy<C>, Error>
where
    F: FnOnce(&mut Worker<'_, '_>) -> Result<C, Error>,
{
    let frames = [
        preparation_controls.ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        size_of::<Worker<'_, '_>>(),
        size_of::<Option<&CompletedResidentSource>>(),
        size_of::<&mut Worker<'_, '_>>(),
        size_of::<IsolatedArrayCopy<'_>>(),
        size_of::<RegisteredArrayCopy>(),
        size_of::<Result<RegisteredArrayCopy, Error>>(),
        size_of::<&HostMetadataFunding>(),
        size_of::<Option<usize>>(),
        size_of::<Result<(), eredu_nn::workspace::HostMetadataFundingError>>(),
        size_of::<F>(),
        size_of::<C>(),
        size_of::<Result<C, Error>>(),
        size_of::<PreparedPredictionCacheCopy<C>>(),
        size_of::<Result<PreparedPredictionCacheCopy<C>, Error>>(),
        size_of::<(C, HostPreparationAuthority)>(),
        size_of::<(Array, RegisteredArrayCopyCustody)>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<CopyCustody>(),
        size_of::<Failure>(),
        HostPreparationAuthority::retention_bytes::<CopyCustody>()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        BackendFailure::source_retention_peak_bytes::<Failure>()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
    ];
    reserve(
        funding,
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add),
    )?;
    let mut worker = Worker {
        completed,
        environment,
        initialized,
        mechanisms,
        funding,
        capacity,
        copies: Vec::new(),
        count: 0,
    };
    let result = operation(&mut worker);
    match result {
        Ok(value) => {
            debug_assert_eq!(worker.copies.len(), worker.count);
            let host = HostPreparationAuthority::retain(CopyCustody {
                _copies: worker.copies,
                _funding: funding.clone(),
            });
            Ok(PreparedPredictionCacheCopy { value, host })
        }
        Err(cause) => Err(Error::StorageSource(BackendFailure::from_error(Failure {
            cause,
            _copies: worker.copies,
            _funding: funding.clone(),
        }))),
    }
}
fn operand_count<'a, F>(funding: &HostMetadataFunding, visit: F) -> Result<usize, Error>
where
    F: FnOnce(&mut dyn FnMut(&'a Array)),
{
    let frames = [
        size_of::<F>(),
        size_of::<Option<usize>>(),
        size_of::<&mut Option<usize>>(),
        size_of::<&mut dyn FnMut(&Array)>(),
        size_of::<Result<usize, Error>>(),
    ];
    reserve(
        funding,
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add),
    )?;
    let mut count = Some(0usize);
    visit(&mut |_| count = count.and_then(|n| n.checked_add(1)));
    count.ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

pub(crate) fn copy_original_compressed(
    source: &CompressedLatentCache,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    capacity: &eredu_core::MemoryLimits,
) -> Result<PreparedPredictionCacheCopy<CompressedLatentCache>, Error> {
    copy_completed_compressed(
        source,
        None,
        environment,
        initialized,
        mechanisms,
        funding,
        capacity,
    )
}

pub(crate) fn copy_completed_compressed(
    source: &CompressedLatentCache,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    capacity: &eredu_core::MemoryLimits,
) -> Result<PreparedPredictionCacheCopy<CompressedLatentCache>, Error> {
    construct(
        completed,
        environment,
        initialized,
        mechanisms,
        funding,
        capacity,
        PreparedCompressedCopy::preparation_control_bytes(),
        |worker| {
            let plan = source
                .prepare_isolated_copy_fixed()
                .map_err(|e| paid(funding, e))?;
            reserve(funding, plan.copy_control_bytes::<Error>())?;
            let count = operand_count(funding, |visitor| plan.visit_operands(visitor))?;
            worker.slots(count)?;
            plan.copy_with(&mut |source| worker.copy(source), &mut |source| {
                alias(funding, source)
            })
        },
    )
}
pub(crate) fn copy_original_pooling(
    source: &MlxPoolingAttentionCache,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    capacity: &eredu_core::MemoryLimits,
) -> Result<PreparedPredictionCacheCopy<MlxPoolingAttentionCache>, Error> {
    copy_completed_pooling(
        source,
        None,
        environment,
        initialized,
        mechanisms,
        funding,
        capacity,
    )
}

pub(crate) fn copy_completed_pooling(
    source: &MlxPoolingAttentionCache,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    initialized: &PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    funding: &HostMetadataFunding,
    capacity: &eredu_core::MemoryLimits,
) -> Result<PreparedPredictionCacheCopy<MlxPoolingAttentionCache>, Error> {
    construct(
        completed,
        environment,
        initialized,
        mechanisms,
        funding,
        capacity,
        PreparedPoolingAttentionCopy::preparation_control_bytes(),
        |worker| {
            let plan = source
                .prepare_isolated_copy_fixed()
                .map_err(|e| paid(funding, e))?;
            reserve(funding, plan.copy_control_bytes::<Error>())?;
            let count = operand_count(funding, |visitor| plan.visit_operands(visitor))?;
            worker.slots(count)?;
            plan.copy_with(&mut |source| worker.copy(source))
        },
    )
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
