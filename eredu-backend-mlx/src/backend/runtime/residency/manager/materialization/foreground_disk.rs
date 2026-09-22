//! One synchronous disk read followed by the ordinary prepared host-copy worker.
use super::*;
use crate::backend::runtime::residency::manager::{
    ForegroundDiskReadError, PreparedForegroundDiskRead, ReadForegroundDiskBatch,
};
use eredu_runtime::working_memory::{OriginalHostSourceCustody, WorkingMemoryReservation};
use safemlx::OriginalScopeObserver;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("foreground disk batch: {0}")]
    Read(#[from] ForegroundDiskReadError),
    #[error("foreground disk native transfer: {0}")]
    Native(#[from] safemlx::error::Exception),
    #[error("foreground disk named destination: {0}")]
    Named(#[from] NamedArrayError),
    #[error("foreground disk request custody: {0}")]
    Cache(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("foreground disk source or native role mismatch")]
    Domain,
    #[error(transparent)]
    Copy(#[from] OriginalHostCopyCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ForegroundDiskMaterializationError {
    #[source]
    cause: Cause,
    _custody: OriginalHostSourceCustody,
}
// Both routes retain the same exact source facts. A completed input merely
// moves the disk read before queue completion; native copies stay in this worker.
enum DiskInput {
    Deferred(PreparedForegroundDiskRead),
    Completed(ReadForegroundDiskBatch),
}
impl DiskInput {
    fn prepared(&self) -> &PreparedForegroundDiskRead {
        match self {
            Self::Deferred(value) => value,
            Self::Completed(value) => value.prepared(),
        }
    }
    fn into_host(
        self,
        context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
        slots: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
        loan: Option<super::super::OriginalMaterializedLoan<'_>>,
        observer: &OriginalScopeObserver,
    ) -> Result<ResidentHostOwner, ForegroundDiskReadError> {
        let batch = match self {
            Self::Deferred(value) => value.read()?,
            Self::Completed(value) => value,
        };
        batch
            .materialize(context, slots, loan, observer)?
            .into_host()
    }
}
pub(in crate::backend::runtime::residency::manager) fn control_bytes() -> Option<usize> {
    let controls = [
        original_host_copy_control_bytes()?,
        size_of::<ResidentHostOwner>(),
        size_of::<Result<ResidentHostOwner, ForegroundDiskReadError>>(),
        size_of::<ForegroundDiskMaterializationError>(),
        size_of::<Result<(), ForegroundDiskMaterializationError>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<OriginalHostSourceCustody>(),
        size_of::<eredu_runtime::working_memory::OriginalOperationMetadataCustody>(),
        size_of::<Option<&eredu_runtime::working_memory::OriginalOperationMetadataCustody>>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<ReadForegroundDiskBatch>(),
        size_of::<DiskInput>(),
        size_of::<&PreparedForegroundDiskRead>(),
        size_of::<Array>(),
        size_of::<Event>(),
        size_of::<Result<(Array, Event), safemlx::error::Exception>>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<(
            PreparedForegroundDiskRead,
            Option<&WorkingMemoryReservation>,
            &OriginalScopeObserver,
        )>(),
        size_of::<
            Option<(
                PreparedForegroundDiskRead,
                Option<&WorkingMemoryReservation>,
                &OriginalScopeObserver,
            )>,
        >(),
        size_of::<Result<(), ResidencyError>>(),
        size_of::<Result<TransferDirection, ResidencyError>>(),
        size_of::<Option<&WorkingMemoryReservation>>(),
        size_of::<&Stream>(),
        size_of::<(
            crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
            &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
            Option<super::super::OriginalMaterializedLoan<'_>>,
        )>(),
        size_of::<&ManagerOwner>(),
        size_of::<ManagerWeak>(),
        size_of::<&OffloadUnit>(),
        size_of::<&mut NamedArrays>(),
        size_of::<Result<(), NamedArrayError>>(),
        size_of::<&WeightBinding>(),
        size_of::<&safemlx::ImmutableHostTransferBuffer>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
/// The enclosing transfer already owns final named rows, output/event vectors,
/// original Eval banks and aggregate completion. This worker creates no recovery
/// engine or publication authority. A refusal never uses ordinary source reads.
/// The caller's registered text role lends its actual reservation; a speculative
/// role carries its exact admitted account. Matching that source custody and the
/// current registered observer preserves the request and native-role checks
/// without manufacturing a neutral funding scope.
pub(in crate::backend::runtime::residency::manager) fn prepare_foreground_disk_into(
    batch: PreparedForegroundDiskRead,
    manager: &ManagerOwner,
    reservation: Option<&WorkingMemoryReservation>,
    observer: &OriginalScopeObserver,
    stream: &Stream,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
    slots: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
    loan: Option<super::super::OriginalMaterializedLoan<'_>>,
) -> Result<(), ForegroundDiskMaterializationError> {
    prepare_disk_host_into(
        DiskInput::Deferred(batch),
        manager,
        reservation,
        observer,
        stream,
        arrays,
        retained,
        context,
        slots,
        loan,
    )
}
/// Consumes a successful background read on the calling thread. Source/custody,
/// exact current role and destination capacity checks are shared with foreground.
pub(in crate::backend::runtime::residency::manager) fn prepare_background_disk_into(
    batch: ReadForegroundDiskBatch,
    manager: &ManagerOwner,
    reservation: Option<&WorkingMemoryReservation>,
    observer: &OriginalScopeObserver,
    stream: &Stream,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
    slots: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
    loan: Option<super::super::OriginalMaterializedLoan<'_>>,
) -> Result<(), ForegroundDiskMaterializationError> {
    prepare_disk_host_into(
        DiskInput::Completed(batch),
        manager,
        reservation,
        observer,
        stream,
        arrays,
        retained,
        context,
        slots,
        loan,
    )
}
fn prepare_disk_host_into(
    batch: DiskInput,
    manager: &ManagerOwner,
    reservation: Option<&WorkingMemoryReservation>,
    observer: &OriginalScopeObserver,
    stream: &Stream,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
    slots: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
    loan: Option<super::super::OriginalMaterializedLoan<'_>>,
) -> Result<(), ForegroundDiskMaterializationError> {
    let custody = batch.prepared().control_custody().clone();
    let result = (|| -> Result<(), Cause> {
        custody.validate_account(reservation)?;
        if !retained
            .metadata_custody()
            .is_some_and(|actual| actual.same_account(&custody.metadata_custody()))
        {
            return Err(Cause::Domain);
        }
        let source = match &manager.sources {
            ResidencySources::Original(source) => source.foreground(),
            _ => None,
        }
        .ok_or(Cause::Domain)?;
        if !batch.prepared().matches_source(source) {
            return Err(Cause::Domain);
        }
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer)
            || retained.retained_host.len() == retained.retained_host.capacity()
            || retained.retained_arrays.capacity() - retained.retained_arrays.len()
                < batch.prepared().outputs()
            || retained.retained_events.capacity() - retained.retained_events.len()
                < batch.prepared().outputs()
        {
            return Err(Cause::Domain);
        }
        arrays.validate_prepared_canonical_outputs(manager, batch.prepared().unit())?;
        let unit = source
            .units()
            .find(|unit| unit.id() == batch.prepared().unit().id())
            .ok_or(Cause::Domain)?;
        let host = batch.into_host(context, slots, loan, observer)?;
        prepare_original_host_copy(unit.bindings(), host, stream, arrays, retained, observer)?;
        Ok(())
    })();
    result.map_err(|cause| ForegroundDiskMaterializationError {
        cause,
        _custody: custody,
    })
}
