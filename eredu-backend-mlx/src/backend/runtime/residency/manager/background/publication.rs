//! Final host rows consumed by the existing acquisition transaction.
use super::*;
use crate::backend::runtime::residency::dense_stream::{
    BackgroundHostReadService, BackgroundHostServiceError,
};
struct Unit {
    id: OffloadUnitId,
    names: Vec<String>,
    values: Vec<(String, RetainedHostBuffer)>,
    canonical: Option<ResidentHostOwner>,
    completed: Option<ResidentHostOwner>,
    logical: u64,
    capacity: u64,
    required_capacity: u64,
    selected: bool,
}
/// One final, paid canonical/alias publication destination.
pub(crate) struct PreparedHostPublication {
    units: Vec<Unit>,
    source: ForegroundDiskDescriptors,
    custody: OriginalHostSourceCustody,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum PublicationCause {
    #[error("background host result: {0}")]
    Read(#[from] BackgroundHostServiceError),
    #[error("background host source: {0}")]
    Source(#[from] ForegroundDiskReadError),
    #[error("background host publication: {0}")]
    Control(#[from] ResidencyError),
}
/// Failed publication retains its actual source and unfinished rows. The
/// existing acquisition transaction rolls back rows already in the manager.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub struct BackgroundHostPublicationFailure {
    #[source]
    cause: PublicationCause,
    retained: PreparedHostPublication,
}
impl std::fmt::Debug for BackgroundHostPublicationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundHostPublicationFailure")
            .field("cause", &self.cause)
            .field("units", &self.retained.units.len())
            .finish()
    }
}
impl PreparedHostPublication {
    pub(crate) fn host_bytes(plan: &ForegroundDiskWindowPlan) -> Option<usize> {
        let fixed = [
            Layout::array::<Unit>(plan.read_units().len()).ok()?.size(),
            size_of::<Self>(),
            size_of::<Unit>(),
            size_of::<Vec<Unit>>(),
            size_of::<Option<Self>>(),
            size_of::<&mut Option<Self>>(),
            size_of::<BackgroundHostPublicationFailure>(),
            size_of::<Box<BackgroundHostPublicationFailure>>(),
            Layout::new::<BackgroundHostPublicationFailure>().size(),
            size_of::<PublicationCause>(),
            size_of::<BackgroundSourceAttempt<'_>>(),
            size_of::<Result<BackgroundSourceAttempt<'_>, BackgroundHostServiceError>>(),
            size_of::<ResidencyError>(),
            size_of::<BackgroundHostReadFailure>(),
            size_of::<Result<Self, BackgroundHostReadFailure>>(),
            size_of::<Result<(), PublicationCause>>(),
            size_of::<Result<(), ResidencyError>>(),
            size_of::<Result<ReadForegroundDiskBatch, BackgroundHostServiceError>>(),
            size_of::<Result<ResidentHostOwner, ForegroundDiskReadError>>(),
            size_of::<ReadForegroundDiskBatch>(),
            size_of::<(
                &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
                Option<super::super::OriginalMaterializedLoan<'_>>,
                &safemlx::OriginalScopeObserver,
            )>(),
            size_of::<ResidentHostOwner>(),
            size_of::<RetainedHostBuffer>(),
            size_of::<String>(),
            size_of::<Result<RetainedHostBuffer, ResidencyError>>(),
            size_of::<Option<RetainedHostBuffer>>(),
            size_of::<(&ManagerState, &OffloadUnitId, &WeightBinding)>(),
            size_of::<(&Self, &OffloadUnitId)>(),
            size_of::<(
                &mut Self,
                &mut ManagerState,
                &ManagerOwner,
                &[OffloadUnitId],
                &[bool],
                &BackgroundHostReadService,
                Instant,
            )>(),
            size_of::<Vec<(String, RetainedHostBuffer)>>(),
            size_of::<Vec<String>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<(u64, u64)>(),
            size_of::<std::time::Duration>(),
            size_of::<eredu_runtime::working_memory::OriginalOperationMetadataCustody>(),
        ];
        let mut bytes = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)?;
        for id in plan.read_units() {
            let unit = plan.source().units().find(|unit| unit.id() == id)?;
            for n in [
                id.as_str().len(),
                Layout::array::<String>(unit.bindings().len()).ok()?.size(),
                Layout::array::<(String, RetainedHostBuffer)>(unit.bindings().len())
                    .ok()?
                    .size(),
                usize::try_from(ResidentHostOwner::storage_bytes().ok()?).ok()?,
            ] {
                bytes = bytes.checked_add(n)?;
            }
            for binding in unit.bindings() {
                bytes = bytes.checked_add(binding.name().len())?;
            }
        }
        Some(bytes)
    }
    pub(crate) fn prepare(
        manager: &ResidencyManager,
        plan: &ForegroundDiskWindowPlan,
        custody: OriginalHostSourceCustody,
        funding: HostMetadataFunding,
    ) -> Result<Self, BackgroundHostReadFailure> {
        let fail = |cause| BackgroundHostReadFailure::Source {
            cause,
            custody: custody.clone(),
            funding: funding.clone(),
        };
        if !plan.matches_manager(manager) {
            return Err(fail(Cause::Identity));
        }
        manager
            .inner
            .validate_operation_custody(&custody.metadata_custody())
            .map_err(|cause| fail(cause.into()))?;
        let bytes = Self::host_bytes(plan).ok_or_else(|| {
            fail(Cause::Funding(
                eredu_core::HostMetadataFundingError::Overflow,
            ))
        })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| fail(cause.into()))?;
        let mut units = Vec::new();
        units
            .try_reserve_exact(plan.read_units().len())
            .map_err(|cause| fail(cause.into()))?;
        for id in plan.read_units() {
            let unit = plan
                .source()
                .units()
                .find(|unit| unit.id() == id)
                .ok_or_else(|| fail(Cause::Identity))?;
            let mut names = Vec::new();
            let mut values = Vec::new();
            names
                .try_reserve_exact(unit.bindings().len())
                .map_err(|cause| fail(cause.into()))?;
            values
                .try_reserve_exact(unit.bindings().len())
                .map_err(|cause| fail(cause.into()))?;
            for binding in unit.bindings() {
                let mut name = String::new();
                name.try_reserve_exact(binding.name().len())
                    .map_err(|cause| fail(cause.into()))?;
                name.push_str(binding.name());
                names.push(name);
            }
            names.sort_unstable();
            units.push(Unit {
                id: id.clone(),
                names,
                values,
                canonical: None,
                completed: None,
                logical: 0,
                capacity: 0,
                required_capacity: u64::try_from(
                    plan.read_layout(id)
                        .ok_or_else(|| fail(Cause::Identity))?
                        .source_backing_bytes,
                )
                .map_err(|_| fail(Cause::Identity))?,
                selected: false,
            });
        }
        Ok(Self {
            units,
            source: plan.source().clone(),
            custody,
            funding,
        })
    }
    pub(in crate::backend::runtime::residency::manager) fn required_capacity(
        &self,
        id: &OffloadUnitId,
    ) -> Option<u64> {
        self.units
            .iter()
            .find(|row| &row.id == id)
            .map(|row| row.required_capacity)
    }
    pub(in crate::backend::runtime::residency::manager) fn validate(
        &self,
        manager: &ManagerOwner,
        state: &ManagerState,
        ids: &[OffloadUnitId],
        missing: &[bool],
    ) -> Result<(), ResidencyError> {
        manager
            .validate_operation_custody(&self.custody.metadata_custody())
            .map_err(ResidencyError::OriginalCache)?;
        if ids.len() != missing.len()
            || !manager
                .sources
                .foreground()
                .is_some_and(|source| self.source.same_source(source))
        {
            return Err(ResidencyError::OriginalOperationDomain);
        }
        for (id, absent) in ids.iter().zip(missing) {
            if !absent {
                continue;
            }
            let row = self
                .units
                .iter()
                .find(|row| &row.id == id)
                .ok_or(ResidencyError::OriginalOperationDomain)?;
            let unit = state
                .control
                .unit(id)
                .ok_or(ResidencyError::OriginalOperationDomain)?;
            if row.selected || !self.source.units().any(|source| source == unit) {
                return Err(ResidencyError::OriginalOperationDomain);
            }
        }
        Ok(())
    }
    pub(in crate::backend::runtime::residency::manager) fn publish(
        mut self,
        state: &mut ManagerState,
        manager: &ManagerOwner,
        ids: &[OffloadUnitId],
        missing: &[bool],
        reads: &BackgroundHostReadService,
        started: Instant,
        materialization: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
        loan: Option<super::super::OriginalMaterializedLoan<'_>>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), ResidencyError> {
        let attempt = match reads.attempt() {
            Ok(attempt) => attempt,
            Err(cause) => {
                return Err(ResidencyError::OriginalHostPublication(Box::new(
                    BackgroundHostPublicationFailure {
                        cause: cause.into(),
                        retained: self,
                    },
                )));
            }
        };
        match self.fill_and_publish(
            state,
            manager,
            ids,
            missing,
            reads,
            started,
            materialization,
            loan,
            observer,
        ) {
            Ok(()) => {
                attempt.succeed();
                Ok(())
            }
            Err(cause) => Err(ResidencyError::OriginalHostPublication(Box::new(
                BackgroundHostPublicationFailure {
                    cause,
                    retained: self,
                },
            ))),
        }
    }
    fn fill_and_publish(
        &mut self,
        state: &mut ManagerState,
        manager: &ManagerOwner,
        ids: &[OffloadUnitId],
        missing: &[bool],
        reads: &BackgroundHostReadService,
        started: Instant,
        materialization: &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
        loan: Option<super::super::OriginalMaterializedLoan<'_>>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), PublicationCause> {
        let domain = || ResidencyError::OriginalOperationDomain;
        self.validate(manager, state, ids, missing)?;
        for (id, absent) in ids.iter().zip(missing) {
            if !absent {
                continue;
            }
            let row = self
                .units
                .iter_mut()
                .find(|row| &row.id == id)
                .ok_or_else(domain)?;
            let unit = state.control.unit(id).ok_or_else(domain)?;
            if row.selected || !self.source.units().any(|source| source == unit) {
                return Err(domain().into());
            }
            let batch = reads.acquire(id)?;
            if !batch.prepared().matches_source(&self.source)
                || batch.prepared().unit().id() != id
                || !batch
                    .prepared()
                    .control_custody()
                    .metadata_custody()
                    .same_account(&self.custody.metadata_custody())
            {
                return Err(domain().into());
            }
            let batch = batch.materialize(
                state.materialization.view(),
                materialization,
                loan,
                observer,
            )?;
            let (logical, capacity) = batch.completed_bytes();
            let planned = state
                .control
                .ledger()
                .spec(id)
                .map_err(ResidencyError::from)?
                .bytes();
            if logical != planned || capacity > row.required_capacity {
                return Err(domain().into());
            }
            row.canonical = Some(batch.into_host()?);
            row.logical = logical;
            row.capacity = capacity;
            row.selected = true;
        }
        // Same ordinary canonical lookup, with direct physical buffer handles.
        // Every canonical owner exists before any alias is filled.
        for index in 0..self.units.len() {
            if !self.units[index].selected {
                continue;
            }
            while let Some(name) = self.units[index].names.pop() {
                let id = &self.units[index].id;
                let unit = state.control.unit(id).ok_or_else(domain)?;
                let binding = unit
                    .bindings()
                    .iter()
                    .find(|binding| binding.name() == name)
                    .ok_or_else(domain)?;
                let buffer = super::super::transfer::aliases::host_binding(
                    state,
                    id,
                    binding,
                    |owner, canonical| {
                        self.units
                            .iter()
                            .find(|row| &row.id == owner)
                            .and_then(|row| row.canonical.as_ref())
                            .and_then(|buffers| buffers.buffers.get(canonical))
                            .cloned()
                    },
                )?;
                if self.units[index].values.len() == self.units[index].values.capacity() {
                    return Err(domain().into());
                }
                self.units[index].values.push((name, buffer));
            }
            self.units[index].values.reverse();
            let buffers = ResidentHostBuffers {
                buffers: rows::Rows::from_sorted(std::mem::take(&mut self.units[index].values)),
            };
            self.units[index].completed = Some(ResidentHostOwner::request_with_metadata(
                buffers,
                self.custody.metadata_custody(),
                self.funding.clone(),
            ));
        }
        for (id, absent) in ids.iter().zip(missing) {
            if !absent {
                continue;
            }
            let row = self
                .units
                .iter_mut()
                .find(|row| &row.id == id)
                .ok_or_else(domain)?;
            state
                .control
                .publish_acquisition_copy(
                    &row.id,
                    MemoryTier::Host,
                    row.capacity,
                    row.logical,
                    None,
                    TransferDirection::DiskToHost,
                    started.elapsed(),
                )
                .map_err(ResidencyError::from)?;
            state.storage.get_mut(&row.id).ok_or_else(domain)?.host = row.completed.take();
        }
        Ok(())
    }
}
