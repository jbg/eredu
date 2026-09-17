//! Finite telemetry construction and borrowed observations of the same ledger.
use super::*;
use crate::ExecutionUnitLayout;
use eredu_core::residency::ResidencyLedger;
use std::{alloc::Layout, mem::size_of};

/// Exact borrowed group declarations and scalar telemetry facts. Construction
/// consumes this plan; callers retain their accepted owner through its result.
#[derive(Debug)]
pub struct DenseStreamTelemetryPlan<'a> {
    layout: &'a ExecutionUnitLayout,
    units: &'a [OffloadUnitId],
    planned_layer_count: usize,
    planned_layer_bytes: u64,
    maximum_host_layer_bytes: u64,
    pinned_static_device_bytes: u64,
    transfer_stream_index: i32,
    bytes: usize,
}
/// Fixed constructor refusals; no source names or metadata are copied on error.
#[derive(Debug, thiserror::Error)]
pub enum DenseTelemetryPreparationError {
    /// Group geometry and retained unit declarations differ.
    #[error("dense telemetry source geometry mismatch")]
    Geometry,
    /// A requested allocation extent is not representable.
    #[error("dense telemetry storage layout overflow")]
    Layout,
    /// An exact finite destination could not be reserved.
    #[error("dense telemetry storage reservation: {0}")]
    Reserve(#[from] std::collections::TryReserveError),
    /// A newly constructed synchronization object could not be initialized.
    #[error("dense telemetry synchronization initialization failed")]
    Synchronization,
}
impl DenseStreamTelemetry {
    /// Measures the actual selected group/unit clones before constructing them.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<'a>(
        layout: &'a ExecutionUnitLayout,
        units: &'a [OffloadUnitId],
        planned_layer_count: usize,
        planned_layer_bytes: u64,
        maximum_host_layer_bytes: u64,
        pinned_static_device_bytes: u64,
        transfer_stream_index: i32,
    ) -> Result<DenseStreamTelemetryPlan<'a>, DenseTelemetryPreparationError> {
        if layout.len() != units.len() || planned_layer_count != units.len() {
            return Err(DenseTelemetryPreparationError::Geometry);
        }
        let bytes = (|| {
            let mut bytes = Layout::array::<DenseExecutionGroupPlan>(layout.group_count())
                .ok()?
                .size()
                .checked_add(
                    Layout::array::<DenseExecutionGroupState>(layout.group_count())
                        .ok()?
                        .size(),
                )?;
            for group in 0..layout.group_count() {
                let ids = units.get(layout.group_range(group)?)?;
                bytes = bytes
                    .checked_add(layout.group_id(group)?.as_str().len())?
                    .checked_add(Layout::array::<OffloadUnitId>(ids.len()).ok()?.size())?;
                for id in ids {
                    bytes = bytes.checked_add(id.as_str().len())?;
                }
            }
            for n in [
                size_of::<Self>(),
                size_of::<DenseStreamTelemetryPlan<'_>>(),
                size_of::<Vec<DenseExecutionGroupPlan>>(),
                size_of::<Vec<DenseExecutionGroupState>>(),
                size_of::<std::sync::MutexGuard<'_, Vec<DenseExecutionGroupState>>>(),
                size_of::<std::sync::MutexGuard<'_, DensePassState>>(),
                size_of::<
                    std::sync::PoisonError<
                        std::sync::MutexGuard<'_, Vec<DenseExecutionGroupState>>,
                    >,
                >(),
                size_of::<std::sync::PoisonError<std::sync::MutexGuard<'_, DensePassState>>>(),
                size_of::<DenseTelemetryPreparationError>(),
                size_of::<Result<Self, DenseTelemetryPreparationError>>(),
            ] {
                bytes = bytes.checked_add(n)?;
            }
            (bytes <= isize::MAX as usize).then_some(bytes)
        })()
        .ok_or(DenseTelemetryPreparationError::Layout)?;
        Ok(DenseStreamTelemetryPlan {
            layout,
            units,
            planned_layer_count,
            planned_layer_bytes,
            maximum_host_layer_bytes,
            pinned_static_device_bytes,
            transfer_stream_index,
            bytes,
        })
    }
    /// Authenticates the selected ordered group/unit geometry without cloning it.
    pub fn matches_prepared_geometry(
        &self,
        layout: &ExecutionUnitLayout,
        units: &[OffloadUnitId],
    ) -> bool {
        self.prepared
            && self.planned_layer_count == units.len()
            && layout.len() == units.len()
            && layout.group_count() == self.groups.len()
            && self.groups.iter().enumerate().all(|(index, group)| {
                layout
                    .group_id(index)
                    .is_some_and(|id| id.as_str() == group.id)
                    && layout
                        .group_range(index)
                        .and_then(|range| units.get(range))
                        .is_some_and(|ids| ids == group.units)
            })
    }
    /// Actual mutex producers initialized by the finite constructor. The caller's
    /// qualified host-library layout accounts for any out-of-line PAL backing.
    pub const fn prepared_mutex_count() -> usize {
        2
    }
    /// Observes actual ledger rows without allocating unit reports or ID sets.
    pub fn observe_group_ledger(
        &self,
        group: &str,
        prefill: bool,
        ledger: &ResidencyLedger,
    ) -> Result<(), DenseStreamTelemetryError> {
        self.observe_usage(group, prefill, LedgerUsage { ledger, next: 0 })
    }
    /// Fixed controls of borrowed observation and pass/group updates. Retained
    /// arrays and platform mutex backing remain separate constructor facts.
    pub fn borrowed_operation_control_bytes() -> Option<usize> {
        let controls = [
            size_of::<LedgerUsage<'_>>(),
            size_of::<Usage<'_>>(),
            size_of::<Result<Usage<'_>, DenseStreamTelemetryError>>(),
            size_of::<std::sync::MutexGuard<'_, Vec<DenseExecutionGroupState>>>(),
            size_of::<std::sync::MutexGuard<'_, DensePassState>>(),
            size_of::<DensePassCounterSnapshot>(),
            size_of::<DensePassActivity>(),
            size_of::<DensePassReport>(),
            size_of::<BackgroundPrefetchReport>(),
            size_of::<(&DenseStreamTelemetry, BackgroundPrefetchReport)>(),
            size_of::<OffloadReport>(),
            size_of::<Result<(), DenseStreamTelemetryError>>(),
            size_of::<[usize; 4]>(),
            size_of::<[u64; 4]>(),
            size_of::<[&str; 2]>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
    }
}
impl DenseStreamTelemetryPlan<'_> {
    /// Requested retained buffers and fixed constructor controls.
    pub fn required_storage_bytes(&self) -> usize {
        self.bytes
    }
    /// Clones only the measured declarations and initializes both mutexes.
    pub fn construct(self) -> Result<DenseStreamTelemetry, DenseTelemetryPreparationError> {
        let mut groups = Vec::new();
        groups.try_reserve_exact(self.layout.group_count())?;
        for group in 0..self.layout.group_count() {
            let id = self
                .layout
                .group_id(group)
                .ok_or(DenseTelemetryPreparationError::Geometry)?;
            let range = self
                .layout
                .group_range(group)
                .ok_or(DenseTelemetryPreparationError::Geometry)?;
            let ids = self
                .units
                .get(range)
                .ok_or(DenseTelemetryPreparationError::Geometry)?;
            let mut units = Vec::new();
            units.try_reserve_exact(ids.len())?;
            units.extend(ids.iter().cloned());
            groups.push(DenseExecutionGroupPlan {
                id: id.as_str().to_owned(),
                units,
            });
        }
        let mut activity = Vec::new();
        activity.try_reserve_exact(groups.len())?;
        activity.resize(groups.len(), DenseExecutionGroupState::default());
        let value = DenseStreamTelemetry {
            planned_layer_count: self.planned_layer_count,
            planned_layer_bytes: self.planned_layer_bytes,
            maximum_host_layer_bytes: self.maximum_host_layer_bytes,
            pinned_static_device_bytes: self.pinned_static_device_bytes,
            transfer_stream_index: self.transfer_stream_index,
            groups,
            group_activity: Mutex::new(activity),
            prepared: true,
            pass: Mutex::new(DensePassState {
                active: None,
                prefill: DensePassReport::default(),
                decode: DensePassReport::default(),
                background: BackgroundPrefetchReport::default(),
            }),
        };
        drop(
            value
                .group_activity
                .lock()
                .map_err(|_| DenseTelemetryPreparationError::Synchronization)?,
        );
        drop(
            value
                .pass
                .lock()
                .map_err(|_| DenseTelemetryPreparationError::Synchronization)?,
        );
        Ok(value)
    }
}
#[derive(Clone, Copy)]
pub(super) struct Usage<'a> {
    pub(super) id: &'a OffloadUnitId,
    pub(super) host: Option<u64>,
    pub(super) device: Option<u64>,
}
#[derive(Clone)]
struct LedgerUsage<'a> {
    ledger: &'a ResidencyLedger,
    next: usize,
}
impl<'a> Iterator for LedgerUsage<'a> {
    type Item = Result<Usage<'a>, DenseStreamTelemetryError>;
    fn next(&mut self) -> Option<Self::Item> {
        let spec = self.ledger.plan().units().get(self.next)?;
        self.next += 1;
        Some((|| {
            let host = self
                .ledger
                .copy_status(spec.id(), MemoryTier::Host)
                .map_err(|_| DenseStreamTelemetryError::InvalidLedger)?;
            let device = self
                .ledger
                .copy_status(spec.id(), MemoryTier::Device)
                .map_err(|_| DenseStreamTelemetryError::InvalidLedger)?;
            Ok(Usage {
                id: spec.id(),
                host: host.map(|v| v.bytes()),
                device: device.map(|v| v.bytes()),
            })
        })())
    }
}
pub(super) fn occupancy<'a>(
    units: impl Iterator<Item = Result<Usage<'a>, DenseStreamTelemetryError>>,
    selected: impl Fn(&OffloadUnitId) -> bool,
) -> Result<(usize, u64, usize, u64), DenseStreamTelemetryError> {
    let (mut hosts, mut host_bytes, mut devices, mut device_bytes) = (0, 0, 0, 0);
    for unit in units {
        let unit = unit?;
        if !selected(unit.id) {
            continue;
        }
        if let Some(bytes) = unit.host {
            hosts += 1;
            host_bytes += bytes;
        }
        if let Some(bytes) = unit.device {
            devices += 1;
            device_bytes += bytes;
        }
    }
    Ok((hosts, host_bytes, devices, device_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionGraph, ExecutionGroupSpec};
    use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec};

    #[test]
    fn prepared_telemetry_observes_the_same_ledger_without_report_ownership() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("first"),
                ExecutionGroupSpec::with_dependencies("last", ["first"]),
            ],
            "last",
        )
        .unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [1, 1]).unwrap();
        let ids = [
            OffloadUnitId::new("a").unwrap(),
            OffloadUnitId::new("b").unwrap(),
        ];
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            ids.iter().map(|id| {
                OffloadUnitSpec::new(id.clone(), 32, ResidencyPolicy::Windowed, MemoryTier::Disk)
                    .unwrap()
            }),
        )
        .unwrap();
        let mut ledger = ResidencyLedger::new(plan);
        for (id, tier, bytes) in [
            (&ids[0], MemoryTier::Host, 20),
            (&ids[1], MemoryTier::Device, 28),
        ] {
            ledger
                .reserve_copy(id, tier, bytes, &BTreeSet::new())
                .unwrap();
            ledger.publish_reserved(id, tier, bytes, None).unwrap();
        }
        let prepared = DenseStreamTelemetry::prepare(&layout, &ids, 2, 64, 32, 0, 7).unwrap();
        assert!(prepared.required_storage_bytes() > size_of::<DenseStreamTelemetry>());
        let prepared = prepared.construct().unwrap();
        assert!(prepared.matches_prepared_geometry(&layout, &ids));
        assert!(!prepared.matches_prepared_geometry(&layout, &[ids[1].clone(), ids[0].clone()]));
        let ordinary = DenseStreamTelemetry::new(
            2,
            64,
            32,
            0,
            7,
            [
                ("first".into(), vec![ids[0].clone()]),
                ("last".into(), vec![ids[1].clone()]),
            ],
        );
        for telemetry in [&ordinary, &prepared] {
            telemetry.begin_forward(true, &ledger.telemetry()).unwrap();
        }
        for group in ["first", "last"] {
            ordinary
                .observe_group(group, true, &ledger.unit_reports())
                .unwrap();
            prepared.observe_group_ledger(group, true, &ledger).unwrap();
            for telemetry in [&ordinary, &prepared] {
                telemetry.record_group_execution(group).unwrap();
            }
        }
        for telemetry in [&ordinary, &prepared] {
            telemetry.commit_forward(&ledger.telemetry()).unwrap();
        }
        assert_eq!(
            ordinary.pass.lock().unwrap().prefill,
            prepared.pass.lock().unwrap().prefill
        );
        let activity = prepared.group_activity.lock().unwrap();
        assert_eq!(
            (
                activity[0].completed_executions,
                activity[0].peak_host_bytes
            ),
            (1, 20)
        );
        assert_eq!(
            (
                activity[1].completed_executions,
                activity[1].peak_device_bytes
            ),
            (1, 28)
        );
        drop(activity);
        assert_eq!(
            prepared.observe_group_ledger("foreign", true, &ledger),
            Err(DenseStreamTelemetryError::UnknownPreparedExecutionGroup)
        );
        for telemetry in [&ordinary, &prepared] {
            telemetry.begin_forward(false, &ledger.telemetry()).unwrap();
            telemetry.abort_forward();
            assert_eq!(
                telemetry.pass.lock().unwrap().decode,
                DensePassReport::default()
            );
        }
    }
}
