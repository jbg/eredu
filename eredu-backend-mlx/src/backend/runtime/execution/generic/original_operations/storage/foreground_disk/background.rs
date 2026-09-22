//! Actual Host-depth declaration and finite source-only forward preparation.
use super::*;
use crate::backend::runtime::residency::{
    dense_stream::{
        BackgroundHostCoordinator, BackgroundHostReadService, PreparedBackgroundForward,
    },
    manager::{OriginalResidencySource, PreparedBackgroundHostReads, PreparedBackgroundHostWindow},
};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::{
    DenseDiskStreamLoadOptions,
    residency::ResidencyClosureSlot,
    working_memory::{
        HostSourcePeakSelection, OriginalHostSourceCustody, WorkingMemoryReservation,
    },
};

/// Aliases only the original manager's actual selected Host-depth source.
#[derive(Clone)]
pub(in crate::backend::runtime::execution::generic::original_operations) struct BackgroundSelection
{
    source: OriginalResidencySource,
    options: DenseDiskStreamLoadOptions,
}
impl BackgroundSelection {
    pub(in crate::backend::runtime::execution::generic::original_operations) fn new(
        source: OriginalResidencySource,
        options: DenseDiskStreamLoadOptions,
    ) -> Self {
        Self { source, options }
    }
}
/// Cold construction metadata retires before the account which produced it.
pub(crate) struct BackgroundRequestPlan {
    selection: BackgroundSelection,
    host: Vec<ForegroundDiskWindowPlan>,
    promotions: Vec<ForegroundDiskWindowPlan>,
    direct: Vec<ForegroundDiskWindowPlan>,
    host_acquisitions: Vec<crate::backend::runtime::residency::manager::WindowPopulation>,
    reads: ForegroundDiskWindowPlan,
    forwards: usize,
    selected: Option<Vec<Vec<usize>>>,
    funding: HostMetadataFunding,
}
impl BackgroundRequestPlan {
    pub(super) fn new(
        selection: &BackgroundSelection,
        manager: &ResidencyManager,
        pool: &MemoryLedger,
        ids: &[OffloadUnitId],
        device: &[ForegroundDiskWindowPlan],
        device_sources: &[crate::backend::runtime::residency::manager::WindowPopulation],
        forwards: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<BackgroundSelection>(),
            size_of::<(usize, u64, usize, usize, [u128; 2], Option<usize>)>(),
            size_of::<(
                &BackgroundSelection,
                &ResidencyManager,
                &MemoryLedger,
                &[OffloadUnitId],
                &[ForegroundDiskWindowPlan],
                &[crate::backend::runtime::residency::manager::WindowPopulation],
                usize,
                &HostMetadataFunding,
            )>(),
            size_of::<crate::backend::runtime::residency::manager::WindowPopulation>(),
            size_of::<(usize, &ForegroundDiskWindowPlan)>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ForegroundDiskWindowPlan>>>(),
            size_of::<
                std::slice::Iter<'_, crate::backend::runtime::residency::manager::WindowPopulation>,
            >(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let declared = selection
            .source
            .foreground_identity()
            .ok_or_else(identity)?;
        let exact = manager
            .background_operation_source(ids, declared.layout(), selection.options.host_lookahead())
            .map_err(|_| identity())?;
        if exact.foreground_identity() != Some(declared)
            || device.len() != exact.windows().len()
            || device.len() != device_sources.len()
            || device.is_empty()
            || selection.options.host_budget_bytes() == 0
        {
            return Err(identity());
        }
        let mut scratch = funding.metadata_vec::<ResidencyClosureSlot>(exact.controller_units)?;
        scratch.resize(exact.controller_units, ResidencyClosureSlot::default());
        let mut host = funding.metadata_vec::<ForegroundDiskWindowPlan>(exact.windows().len())?;
        for window in exact.windows() {
            host.push(ForegroundDiskWindowPlan::new_with_metadata(
                manager,
                pool,
                *window,
                ids,
                &mut scratch,
                Some(funding),
            )?);
        }
        let mut promotions = funding.metadata_vec::<ForegroundDiskWindowPlan>(device.len())?;
        let mut direct = funding.metadata_vec::<ForegroundDiskWindowPlan>(device.len())?;
        let mut host_acquisitions = funding
            .metadata_vec::<crate::backend::runtime::residency::manager::WindowPopulation>(
            device.len(),
        )?;
        for (ordinal, device) in device.iter().enumerate() {
            let host_source = exact.windows().get(ordinal).ok_or_else(identity)?;
            let device_source = device_sources.get(ordinal).ok_or_else(identity)?;
            if host_source.request_start != device_source.request_start {
                return Err(identity());
            }
            // Both ranges come from the same declared group-major traversal.
            // Select its smaller existing prefix; no closure is inferred from
            // summed windows and configured Host lookahead is never widened.
            let selected = if host_source.request_end < device_source.request_end {
                *host_source
            } else {
                *device_source
            };
            let promotion = ForegroundDiskWindowPlan::new_with_metadata(
                manager,
                pool,
                selected,
                ids,
                &mut scratch,
                Some(funding),
            )?;
            direct.push(device.difference(&promotion, funding)?);
            promotions.push(promotion);
            host_acquisitions.push(selected);
        }
        let reads = ForegroundDiskWindowPlan::background_occurrences(&host, funding)?;
        Ok(Self {
            selection: selection.clone(),
            host,
            promotions,
            direct,
            host_acquisitions,
            reads,
            forwards,
            selected: None,
            funding: funding.clone(),
        })
    }
    /// The completed cold traversal supplies logical acquisition ordinals, not
    /// a new residency layout. Retain those exact visits while each selected
    /// window keeps its original Host lookahead and device promotion source.
    pub(super) fn select_execution_ordinals(&mut self, ordinals: &[usize]) -> Result<(), Error> {
        let controls = [
            size_of::<(&mut Self, &[usize])>(),
            size_of::<Option<Vec<Vec<usize>>>>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<std::slice::Windows<'_, usize>>(),
            size_of::<Result<(), Error>>(),
        ];
        self.funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if self.selected.is_some() {
            return Err(identity());
        }
        let mut selected = self.funding.metadata_vec(self.forwards)?;
        for _ in 0..self.forwards {
            selected.push(self.copy_execution_ordinals(ordinals)?);
        }
        self.selected = Some(selected);
        Ok(())
    }
    fn copy_execution_ordinals(&self, ordinals: &[usize]) -> Result<Vec<usize>, Error> {
        self.funding
            .reserve_metadata(size_of::<(
                &Self,
                &[usize],
                Vec<usize>,
                Result<Vec<usize>, Error>,
                std::slice::Iter<'_, usize>,
                std::slice::Windows<'_, usize>,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        if ordinals.iter().any(|ordinal| *ordinal >= self.host.len())
            || ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(identity());
        }
        let mut selected = self.funding.metadata_vec(ordinals.len())?;
        selected.extend_from_slice(ordinals);
        Ok(selected)
    }
    /// Each actual generation span retains its own shared-driver visits. The
    /// final conservative decode has no generation submission or worker.
    pub(super) fn select_execution_recipe(
        &mut self,
        recipe: &crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(), Error> {
        let controls = [
            size_of::<(
                &mut Self,
                &crate::backend::nn::workspace::ResidentNativeRecipe,
            )>(),
            size_of::<Option<Vec<Vec<usize>>>>(),
            size_of::<Vec<usize>>(),
            size_of::<Result<Vec<usize>, Error>>(),
            size_of::<Option<&[usize]>>(),
            size_of::<std::slice::Iter<'_, crate::backend::nn::workspace::ResidentSpanRecipe>>(),
            size_of::<Result<(), Error>>(),
        ];
        self.funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if self.selected.is_some()
            || recipe.plan().generation_forward_count() != Some(self.forwards)
        {
            return Err(identity());
        }
        let mut selected = self.funding.metadata_vec(self.forwards)?;
        for span in recipe.plan().generation_records().ok_or_else(identity)? {
            let row = recipe
                .records()
                .iter()
                .find(|row| row.span() == span.span())
                .ok_or_else(identity)?;
            selected.push(
                self.copy_execution_ordinals(row.layerwise_ordinals().ok_or_else(identity)?)?,
            );
        }
        self.selected = Some(selected);
        Ok(())
    }
    pub(super) fn population(&self) -> Option<ForegroundDiskPopulation> {
        self.direct
            .iter()
            .try_fold(self.reads.population()?, |sum, row| {
                sum.checked_add(row.population()?)
            })
    }
    pub(super) fn source_control_bytes(&self) -> Option<u64> {
        self.direct
            .iter()
            .try_fold(self.reads.source_control_bytes()?, |sum, row| {
                sum.checked_add(row.source_control_bytes()?)
            })
    }
    pub(crate) fn direct(&self, ordinal: usize) -> Option<&ForegroundDiskWindowPlan> {
        self.direct.get(ordinal)
    }
    pub(crate) fn host_acquisitions(
        &self,
    ) -> &[crate::backend::runtime::residency::manager::WindowPopulation] {
        &self.host_acquisitions
    }
    pub(super) fn backing_capacity(&self) -> Option<usize> {
        let total = self
            .population()?
            .checked_mul(self.forwards)?
            .source_backing_bytes;
        let ready = self.host.iter().try_fold(0usize, |peak, row| {
            Some(peak.max(row.single_read_population()?.source_backing_bytes))
        })?;
        let direct = self.direct.iter().try_fold(0usize, |peak, row| {
            Some(peak.max(row.population()?.source_backing_bytes))
        })?;
        let bounded = bounded_source_peak(
            total,
            self.selection.options.host_budget_bytes(),
            ready,
            direct,
        )?;
        // The bank requires an empty pending queue after actual unit wait,
        // transfer synchronization and payload release. The source service
        // then observes wait_idle and drops old ready buffers before reading.
        // Shared Host acquisition pins external canonical alias owners in the
        // Host ledger (aliases::pin_owners); their backing stays charged even
        // when an alias row contributes zero new bytes. Any failed publication
        // or retirement closes all later reads. Independently retained native
        // aliases still consume the shared counter until actual destruction.
        Some(bounded)
    }
    pub(super) fn peak_selection(&self, bytes: u64) -> HostSourcePeakSelection {
        self.reads.peak_selection(bytes)
    }

    /// Prepare final per-forward read/queue/window destinations before the
    /// first worker starts. All generations use the same native retirement
    /// counter and accepted host-source bank; there is no per-forward capacity.
    pub(crate) fn prepare(
        &self,
        manager: &ResidencyManager,
        pool: &MemoryLedger,
        ids: &[OffloadUnitId],
        device: &[ForegroundDiskWindowPlan],
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
        capacity: &ForegroundDiskSourceCapacity,
    ) -> Result<BackgroundHostCoordinator, Error> {
        let funding = &self.funding;
        let controls = [
            size_of::<Result<BackgroundHostCoordinator, Error>>(),
            size_of::<eredu_runtime::working_memory::HostThreadStartupPlan>(),
            size_of::<
                Result<eredu_runtime::working_memory::HostThreadStartupPlan, WorkingMemoryError>,
            >(),
            size_of::<(
                &Self,
                &ResidencyManager,
                &MemoryLedger,
                &[OffloadUnitId],
                &[ForegroundDiskWindowPlan],
                OriginalHostSourceCustody,
                Option<&WorkingMemoryReservation>,
                &ForegroundDiskSourceCapacity,
            )>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<(usize, &ForegroundDiskWindowPlan)>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ForegroundDiskWindowPlan>>>(),
            size_of::<String>(),
            size_of::<std::fmt::Arguments<'_>>(),
            size_of::<OriginalHostSourceCustody>(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if device.len() != self.host.len()
            || !capacity
                .custody()
                .metadata_custody()
                .same_account(&custody.metadata_custody())
        {
            return Err(identity());
        }
        let declared = self
            .selection
            .source
            .foreground_identity()
            .ok_or_else(identity)?;
        let exact = manager
            .background_operation_source(
                ids,
                declared.layout(),
                self.selection.options.host_lookahead(),
            )
            .map_err(|_| identity())?;
        if exact.foreground_identity() != Some(declared) {
            return Err(identity());
        }
        let startup = BackgroundHostReadService::thread_plan().map_err(Error::PrefillControl)?;
        let mut forwards =
            funding.metadata_vec::<Option<PreparedBackgroundForward>>(self.forwards)?;
        for forward in 0..self.forwards {
            let reads = PreparedBackgroundHostReads::prepare(
                &self.reads,
                manager,
                pool,
                custody.clone(),
                reservation,
                capacity,
                funding.clone(),
                self.selection.options.background_queue_capacity(),
            )
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
            let selected = self.selected.as_ref().map(|rows| rows[forward].as_slice());
            let count = selected.map_or(device.len(), <[usize]>::len);
            let mut windows =
                funding.metadata_vec::<(usize, Option<PreparedBackgroundHostWindow>)>(count)?;
            for position in 0..count {
                let ordinal = selected.map_or(position, |selected| selected[position]);
                let window = exact.windows().get(ordinal).ok_or_else(identity)?;
                let active = ids
                    .get(window.request_start..window.request_end)
                    .ok_or_else(identity)?;
                let address = declared.layout().address(ordinal).ok_or_else(identity)?;
                let group = declared
                    .layout()
                    .group_id(address.group())
                    .ok_or_else(identity)?;
                // Same ordinary dense protection name, with its real formatting
                // destination paid before construction. Child protection owns
                // its independently counted immutable copy after this retires.
                let group =
                    funding.metadata_string(format_args!("dense:{}:host", group.as_str()))?;
                windows.push((
                    ordinal,
                    Some(
                        PreparedBackgroundHostWindow::prepare(
                            manager,
                            &self.host[ordinal],
                            &self.promotions[ordinal],
                            &group,
                            active,
                            self.host_acquisitions[ordinal].requested,
                            custody.clone(),
                            funding.clone(),
                        )
                        .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                    ),
                ));
            }
            forwards.push(Some(PreparedBackgroundForward::from_prepared(
                reads, windows, startup,
            )));
        }
        BackgroundHostCoordinator::from_prepared(forwards, custody, funding.clone())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

/// The finite actual source population is an independent upper limit. A valid
/// larger configured cache ceiling must not overflow while computing a bound
/// that is already no greater than that actual population.
fn bounded_source_peak(
    total: usize,
    host_budget: u64,
    ready: usize,
    direct: usize,
) -> Option<usize> {
    let total = u128::try_from(total).ok()?;
    let resident = u128::from(host_budget)
        .checked_add(u128::try_from(ready).ok()?)?
        .checked_add(u128::try_from(direct).ok()?)?;
    usize::try_from(total.min(resident)).ok()
}

#[cfg(test)]
mod bounded_peak_tests {
    use super::bounded_source_peak;
    #[test]
    fn configured_maximum_keeps_the_finite_actual_source_peak() {
        assert_eq!(bounded_source_peak(4096, u64::MAX, 512, 1024), Some(4096));
        assert_eq!(bounded_source_peak(4096, 1024, 512, 256), Some(1792));
        assert_eq!(bounded_source_peak(4096, 4096, 512, 1024), Some(4096));
        assert_eq!(
            bounded_source_peak(0, u64::MAX, usize::MAX, usize::MAX),
            Some(0)
        );
        assert_eq!(
            bounded_source_peak(usize::MAX, u64::MAX, usize::MAX, usize::MAX),
            Some(usize::MAX)
        );
    }
}
