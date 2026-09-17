//! Source-owned dense controller with no exported raw Arc/Weak.
use super::*;
use crate::backend::runtime::residency::manager::{ManagerCustody, ManagerWeak};
use eredu_runtime::{
    working_memory::{OriginalHostMetadataCustody, WorkingMemoryError},
    ExecutionUnitLayout,
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    ops::Deref,
};

/// Actual selected load declarations; these are measured and consumed by the
/// same original manager initializer as static source arrays and metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OriginalDenseControllerFacts {
    pub(crate) options: DenseDiskStreamLoadOptions,
    pub(crate) planned_layers: usize,
    pub(crate) planned_bytes: u64,
    pub(crate) maximum_host_bytes: u64,
    pub(crate) static_bytes: u64,
    pub(crate) stream_index: i32,
}
#[derive(Clone)]
pub(crate) struct PreparedDenseController {
    value: Arc<DenseStreamController>,
    custody: ManagerCustody,
}
#[derive(Clone)]
pub(crate) enum DenseControllerHandle {
    Ordinary(Arc<DenseStreamController>),
    Original(PreparedDenseController),
}
impl Deref for DenseControllerHandle {
    type Target = DenseStreamController;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(v) => v,
            Self::Original(v) => &v.value,
        }
    }
}
pub(super) struct PreparedOrigin {
    pub(super) manager: ManagerWeak,
    pub(super) facts: OriginalDenseControllerFacts,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum PreparedDenseControllerError {
    #[error("prepared dense controller source identity mismatch")]
    Identity,
    #[error(transparent)]
    Telemetry(#[from] eredu_runtime::DenseTelemetryPreparationError),
}
impl PreparedDenseController {
    pub(crate) fn storage_bytes(
        facts: OriginalDenseControllerFacts,
        layout: &ExecutionUnitLayout,
        ids: &[OffloadUnitId],
    ) -> Result<usize, WorkingMemoryError> {
        if facts.options.samples_backend_memory()
            || facts.options.samples_process_memory()
        {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let telemetry = DenseStreamTelemetry::prepare(
            layout,
            ids,
            facts.planned_layers,
            facts.planned_bytes,
            facts.maximum_host_bytes,
            facts.static_bytes,
            facts.stream_index,
        )
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let mut bytes = telemetry.required_storage_bytes();
        let arc = OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<
            DenseStreamController,
        >())?;
        let mutex = OriginalHostMetadataCustody::initialized_mutex_bytes()?;
        for n in [
            usize::try_from(arc).map_err(|_| WorkingMemoryError::Overflow)?,
            usize::try_from(mutex)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .checked_mul(DenseStreamTelemetry::prepared_mutex_count())
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<PreparedOrigin>(),
            size_of::<DenseControllerHandle>(),
            size_of::<PreparedDenseControllerError>(),
            size_of::<Result<Self, PreparedDenseControllerError>>(),
        ] {
            bytes = bytes.checked_add(n).ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(bytes)
    }
    pub(crate) fn construct(
        facts: OriginalDenseControllerFacts,
        layout: &ExecutionUnitLayout,
        ids: &[OffloadUnitId],
        manager: ManagerWeak,
        custody: ManagerCustody,
    ) -> Result<Self, PreparedDenseControllerError> {
        if facts.options.samples_backend_memory()
            || facts.options.samples_process_memory()
        {
            return Err(PreparedDenseControllerError::Identity);
        }
        let telemetry = DenseStreamTelemetry::prepare(
            layout,
            ids,
            facts.planned_layers,
            facts.planned_bytes,
            facts.maximum_host_bytes,
            facts.static_bytes,
            facts.stream_index,
        )?
        .construct()?;
        Ok(Self {
            value: Arc::new(DenseStreamController {
                options: facts.options,
                // Original requests own their admitted host-only worker. The
                // immutable manager never starts the ordinary manager callback.
                background: None,
                telemetry,
                prepared: Some(PreparedOrigin { manager, facts }),
            }),
            custody,
        })
    }
    pub(crate) fn validate(
        &self,
        facts: OriginalDenseControllerFacts,
        layout: &ExecutionUnitLayout,
        ids: &[OffloadUnitId],
    ) -> bool {
        self.value
            .prepared
            .as_ref()
            .is_some_and(|origin| origin.facts == facts)
            && self.value.telemetry.matches_prepared_geometry(layout, ids)
    }
    pub(crate) fn into_handle(self) -> DenseControllerHandle {
        DenseControllerHandle::Original(self)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let controls = [
            DenseStreamTelemetry::borrowed_operation_control_bytes()?,
            size_of::<DenseStreamForwardGuard>(),
            size_of::<DenseControllerHandle>(),
            size_of::<ResidencyManager>(),
            size_of::<ManagerWeak>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<DenseStreamForwardGuard, Error>>(),
            size_of::<[&str; 2]>(),
            size_of::<bool>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
}
impl DenseControllerHandle {
    pub(crate) fn is_source_prepared(&self) -> bool {
        matches!(self, Self::Original(_))
    }
    pub(crate) fn transfer_window(
        &self,
        manager: &ResidencyManager,
        group: impl Into<String>,
        units: &[OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow, Error> {
        DenseStreamController::transfer_window_owned(
            self.clone(),
            manager,
            group,
            units,
            indices,
            prefill,
        )
    }
    pub(crate) fn forward_guard(
        &self,
        prefill: bool,
        manager: &ResidencyManager,
    ) -> Result<DenseStreamForwardGuard, Error> {
        DenseStreamController::forward_guard_owned(self.clone(), prefill, manager)
    }
    pub(crate) fn group_guard(
        &self,
        manager: &ResidencyManager,
        group: &str,
    ) -> DenseStreamGroupGuard {
        DenseStreamController::group_guard_owned(self.clone(), manager, group)
    }
}
