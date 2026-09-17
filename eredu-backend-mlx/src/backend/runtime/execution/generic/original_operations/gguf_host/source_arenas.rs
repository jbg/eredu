//! Actual selected output layouts; no native arena is created during planning.
use super::super::*;
use crate::backend::runtime::checkpoint::store::PreparedPendingWeight;
use crate::backend::runtime::checkpoint::store::PreparedSourceAcquisitions;
use crate::backend::runtime::residency::manager::acquisition_destinations::SelectedSourceOccurrence;
use crate::backend::runtime::residency::manager::WindowPopulation;
use eredu_runtime::working_memory::{HostDestinationFacts, HostSourceConstructionFacts};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceArenaWindow {
    pub(crate) acquisition_bytes: u64,
    pub(crate) bytes_per_pending: u64,
    pub(crate) attempts_per_pending: usize,
    pub(crate) pending: usize,
    pub(crate) destination_bytes: u64,
    pub(crate) destination_attempts: usize,
}
struct Window {
    bound: SourceArenaWindow,
    // Exact wrapper/source/request identity and actual physical descriptors.
    plans: Vec<SelectedSourceOccurrence>,
}
pub(crate) struct SourceArenaPlan {
    windows: Vec<Window>,
    facts: HostSourceConstructionFacts,
    host_facts: HostDestinationFacts,
    preparation_temporary_bytes: usize,
    // The actual shared owner; its fixed birth is charged once, never per
    // window. Ordinary origin remains ordinary and keeps complete fit unknown.
    cache: crate::backend::runtime::checkpoint::store::CacheHandle,
}
impl SourceArenaPlan {
    pub(crate) fn prepare(
        manager: &ResidencyManager,
        windows: &[WindowPopulation],
        ids: &[OffloadUnitId],
        population: ResidencyPopulation,
        runtime: &safemlx::PreparedInputRuntime,
    ) -> Result<Self, Error> {
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(population.controller_units)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        scratch.resize(
            population.controller_units,
            eredu_runtime::residency::ResidencyClosureSlot::default(),
        );
        let mut out = Vec::new();
        out.try_reserve_exact(windows.len())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let mut bytes = 0u64;
        let mut attempts = 0usize;
        let mut partitions = 0usize;
        let mut destination_bytes = 0u64;
        let mut destination_attempts = 0usize;
        let mut temporary =
            eredu_checkpoint::gguf_store::GgufConversionPlan::supplied_storage_control_bytes()
                .and_then(|n| n.checked_add(size_of::<eredu_gguf::StorageRequestBound>()))
                .and_then(|n| {
                    n.checked_add(size_of::<
                        Result<
                            eredu_gguf::StorageRequestBound,
                            crate::backend::runtime::checkpoint::store::GgufHostCopyCause,
                        >,
                    >())
                })
                .ok_or_else(overflow)?;
        for window in windows {
            let roots = ids
                .get(window.request_start..window.request_end)
                .ok_or_else(identity)?;
            if roots.len() != window.requested {
                return Err(identity());
            }
            let (plans, descriptor_bytes) = manager.source_conversion_plans(roots, &mut scratch)?;
            temporary = temporary.max(descriptor_bytes);
            let mut bound = SourceArenaWindow {
                pending: ResidencyPopulation::pending_for_window(*window)?,
                ..SourceArenaWindow::default()
            };
            let acquisition_count = plans
                .iter()
                .try_fold(0usize, |sum, row| sum.checked_add(row.multiplicity))
                .ok_or_else(overflow)?;
            if acquisition_count != bound.pending {
                return Err(identity());
            }
            bound.acquisition_bytes =
                PreparedSourceAcquisitions::selected_storage_bytes(&plans, acquisition_count)
                    .map_err(memory)?;
            for occurrence in &plans {
                let plan = &occurrence.plan;
                let destinations =
                    PreparedPendingWeight::host_destination_requests(plan.physical())
                        .map_err(|cause| Error::Other(Box::new(cause)))?;
                bound.destination_bytes = bound
                    .destination_bytes
                    .max(u64::try_from(destinations.bytes()).map_err(|_| overflow())?);
                bound.destination_attempts = bound.destination_attempts.max(destinations.calls());
                let (source_storage, shape_storage) =
                    PreparedPendingWeight::source_copy_storage_bytes(plan.physical(), runtime)
                        .map_err(|cause| Error::Other(Box::new(cause)))?;
                temporary = temporary.max(shape_storage);
                bound.bytes_per_pending = bound.bytes_per_pending.max(source_storage);
                bound.attempts_per_pending = bound.attempts_per_pending.max(
                    plan.physical()
                        .conversion()
                        .outputs()
                        .len()
                        .checked_add(1)
                        .ok_or_else(overflow)?,
                );
            }
            let n = bound
                .pending
                .checked_mul(population.forwards)
                .ok_or_else(overflow)?;
            bytes = bytes
                .checked_add(
                    bound
                        .bytes_per_pending
                        .checked_mul(u64::try_from(n).map_err(|_| overflow())?)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            attempts = attempts
                .checked_add(
                    bound
                        .attempts_per_pending
                        .checked_mul(n)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            destination_bytes = destination_bytes
                .checked_add(
                    bound
                        .destination_bytes
                        .checked_mul(u64::try_from(n).map_err(|_| overflow())?)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            destination_attempts = destination_attempts
                .checked_add(
                    bound
                        .destination_attempts
                        .checked_mul(n)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            // One additional paired child with zero Vec allowance owns the
            // actual G4 bank for each reached window attempt.
            bytes = bytes
                .checked_add(
                    bound
                        .acquisition_bytes
                        .checked_mul(u64::try_from(population.forwards).map_err(|_| overflow())?)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            attempts = attempts
                .checked_add(population.forwards)
                .ok_or_else(overflow)?;
            partitions = partitions
                .checked_add(n)
                .and_then(|p| p.checked_add(population.forwards))
                .ok_or_else(overflow)?;
            out.push(Window { bound, plans });
        }
        let preparation_temporary_bytes = Layout::array::<
            eredu_runtime::residency::ResidencyClosureSlot,
        >(scratch.capacity())
        .map_err(|_| overflow())?
        .size()
        .checked_add(temporary)
        // This plan retains the handle inline; its separate lookup and
        // install check reuse these concrete loan/result transports.
        .and_then(|bytes| {
            bytes
                .checked_add(ResidencyManager::gguf_cache_identity_control_bytes()?)?
                .checked_add(eredu_runtime::working_memory::WorkingMemoryReservation::gguf_catalog_validation_control_bytes()?)?
                .checked_add(size_of::<Result<(), WorkingMemoryError>>())?
                .checked_add(size_of::<std::slice::Iter<'_, Window>>())?
                .checked_add(size_of::<std::slice::Iter<'_, SelectedSourceOccurrence>>())
        })
        .ok_or_else(overflow)?;
        let facts =
            HostSourceConstructionFacts::new(bytes, attempts, partitions).map_err(memory)?;
        let host_facts = HostDestinationFacts::new(destination_bytes, destination_attempts)
            .and_then(|facts| facts.with_partitions(partitions))
            .and_then(|host| host.with_source_constructions(facts))
            .map_err(memory)?;
        Ok(Self {
            windows: out,
            facts,
            host_facts,
            preparation_temporary_bytes,
            cache: manager.gguf_cache_handle()?,
        })
    }
    /// Existing request/source validators run before this partial origin check.
    /// A retained ordinary handle never proves original cache construction.
    pub(crate) fn validate_cache_origin(
        &self,
        manager: &ResidencyManager,
        reservation: &eredu_runtime::working_memory::WorkingMemoryReservation,
    ) -> Result<(), Error> {
        manager.validate_gguf_cache(&self.cache)?;
        self.cache.validate_reservation(reservation).map_err(memory)
    }

    /// Each retained built-in physical plan borrows its actual immutable
    /// catalog. Source and built-in union constructors have separate original
    /// accounts; this catalog check does not certify their origin, outer erasure
    /// shells, future recipe entries or manager construction.
    pub(crate) fn validate_catalog_origins(
        &self,
        reservation: &eredu_runtime::working_memory::WorkingMemoryReservation,
    ) -> Result<(), Error> {
        for window in &self.windows {
            for occurrence in &window.plans {
                reservation
                    .validate_gguf_catalog_plan(occurrence.plan.physical())
                    .map_err(memory)?;
            }
        }
        Ok(())
    }

    pub(crate) fn facts(&self) -> HostSourceConstructionFacts {
        self.facts
    }
    pub(crate) fn host_facts(&self) -> HostDestinationFacts {
        self.host_facts
    }
    pub(crate) fn window(&self, ordinal: usize) -> Option<SourceArenaWindow> {
        self.windows.get(ordinal).map(|w| w.bound)
    }
    pub(crate) fn acquisition_plans(&self, ordinal: usize) -> Option<&[SelectedSourceOccurrence]> {
        self.windows.get(ordinal).map(|w| w.plans.as_slice())
    }
    pub(crate) fn retained_control_bytes(&self) -> Option<u64> {
        // Self is inline in OriginalOperationPlan's already counted enum.
        let mut bytes = Layout::array::<Window>(self.windows.capacity())
            .ok()?
            .size();
        for window in &self.windows {
            bytes = bytes.checked_add(
                Layout::array::<SelectedSourceOccurrence>(window.plans.capacity())
                    .ok()?
                    .size(),
            )?;
            for plan in &window.plans {
                bytes = bytes.checked_add(plan.plan.metadata_bytes()?.checked_sub(
                    size_of::<eredu_checkpoint::store::SelectedGgufConversionPlan>(),
                )?)?;
                let shared = plan.plan.acquisition_route_shared_layout()?;
                let allocated = eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(shared).ok()?;
                // Payload is already in metadata_bytes; each retained route's
                // qualified Arc header is counted once, never per ticket alias.
                bytes = bytes.checked_add(
                    usize::try_from(allocated)
                        .ok()?
                        .checked_sub(shared.size())?,
                )?;
            }
        }
        // The exact descriptor/scratch overlap is named, but arbitrary cold
        // source callbacks and cold allocation producers stay unqualified.
        bytes = bytes.checked_add(self.preparation_temporary_bytes)?;
        u64::try_from(bytes).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
    use eredu_core::residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy,
    };
    use eredu_runtime::residency::{OffloadUnit, WeightBinding};
    use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};

    #[test]
    fn real_source_windows_and_nested_retry_population_fund_only_finite_metadata_tickets() {
        let fixture = crate::backend::runtime::checkpoint::store::OriginalGgufMissFixture::new();
        let (source, runtime, stream) = fixture.source_planning_inputs();
        let ids = ["direct", "joined"]
            .map(|id| OffloadUnitId::new(id).unwrap())
            .to_vec();
        let direct =
            WeightBinding::new("weight", "bank.weight", TensorSelection::Full, 32).unwrap();
        let joined = WeightBinding::from_recipe(
            "joined",
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source("bank.weight", TensorSelection::Full),
                    DerivedWeightRecipe::source("bank.weight", TensorSelection::Full),
                ],
            },
            64,
        )
        .unwrap();
        let units = [
            OffloadUnit::new(ids[0].clone(), vec![direct]).unwrap(),
            OffloadUnit::new(ids[1].clone(), vec![joined]).unwrap(),
        ];
        let plan = OffloadPlan::new(
            OffloadConfig::default(),
            ids.iter().zip([32, 64]).map(|(id, bytes)| {
                OffloadUnitSpec::new(
                    id.clone(),
                    bytes,
                    ResidencyPolicy::Windowed,
                    MemoryTier::Disk,
                )
                .unwrap()
            }),
        )
        .unwrap();
        let manager =
            ResidencyManager::new_shared(source, plan, units, stream.clone(), stream).unwrap();
        let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("all")], "all").unwrap();
        let layout = eredu_runtime::execution::ExecutionUnitLayout::new(&graph, [2]).unwrap();
        let selected = manager
            .prepare_original_operation_source(&ids, &layout, 2)
            .unwrap();
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            input_positions: 5,
            cached_positions: 0,
            max_output_tokens: 3,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let population = ResidencyPopulation::from_windows(
            selected.controller_units,
            selected.windows(),
            geometry,
        )
        .unwrap();
        assert_eq!(population.forwards, 5);
        assert_eq!(
            selected
                .windows()
                .iter()
                .map(|w| w.recipe_pending)
                .collect::<Vec<_>>(),
            [5, 4]
        );
        let reads = fixture.source_physical_reads();
        let prepared =
            SourceArenaPlan::prepare(&manager, selected.windows(), &ids, population, &runtime)
                .unwrap();
        assert_eq!(fixture.source_physical_reads(), reads);
        assert_eq!(
            prepared
                .windows
                .iter()
                .map(|w| w.plans.len())
                .collect::<Vec<_>>(),
            [3, 2]
        );
        assert_eq!(
            prepared
                .windows
                .iter()
                .map(|w| w.bound.pending)
                .collect::<Vec<_>>(),
            [20, 16]
        );
        let controls = fixture.source_arena_layouts()[0];
        let bytes = fixture.source_storage_layouts()[0]
            + PreparedPendingWeight::cache_metadata_control_bytes().unwrap();
        assert!(bytes > controls);
        let physical = &prepared.windows[0].plans[0].plan;
        let (control_only, _) =
            PreparedPendingWeight::source_copy_control_bytes(physical.physical(), &runtime)
                .unwrap();
        assert_eq!(control_only, controls);
        assert!(bytes > 0);
        assert!(prepared
            .windows
            .iter()
            .all(|w| w.bound.bytes_per_pending == bytes && w.bound.attempts_per_pending == 2));
        let acquisitions = prepared
            .windows
            .iter()
            .map(|w| w.bound.acquisition_bytes)
            .sum::<u64>()
            * population.forwards as u64;
        assert!(acquisitions > 0);
        assert_eq!(
            prepared.facts().capacity_bytes(),
            180 * bytes + acquisitions
        );
        let destinations: u64 =
            fixture.admitted_dense_layouts().into_iter().sum::<u64>() + "bank.weight".len() as u64;
        assert!(prepared.windows.iter().all(
            |w| w.bound.destination_bytes == destinations && w.bound.destination_attempts == 10
        ));
        assert_eq!(
            (
                prepared.host_facts().capacity_bytes(),
                prepared.host_facts().maximum_attempts(),
                prepared.host_facts().maximum_partitions()
            ),
            (180 * destinations, 180 * 10, 190)
        );
        assert_eq!(
            prepared.host_facts().source_constructions(),
            Some(prepared.facts())
        );
        assert_eq!(
            (
                prepared.facts().maximum_attempts(),
                prepared.facts().maximum_partitions()
            ),
            (370, 190)
        );
        assert!(prepared.retained_control_bytes().unwrap() > 0);
        // This is an actual metadata query: no pending slot/arena or physical
        // materialization is constructed. Public readiness still has other gaps.
        assert!(population
            .prepared_payload_control_bytes(selected.windows())
            .is_none());
    }
}

#[cfg(test)]
#[path = "source_arenas/cache_context_tests.rs"]
mod cache_context_tests;
