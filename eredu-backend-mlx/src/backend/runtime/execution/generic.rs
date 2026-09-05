//! MLX realization of the backend-neutral layerwise unit policy.

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    DenseDiskStreamReport, ExecutionUnitAddress, ExecutionUnitLayout, LayerWeightResidency,
    LayerwiseModelMetadata, ResidentLayerGroup, ResidentLayerGroupReport,
};

use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

use eredu_nn::Parameterized;
use eredu_runtime::LayerwiseAcquireError;
use eredu_runtime::{
    LayeredArchitecture, LayerwisePolicy, OffloadUnit, RuntimeState, WeightBinding,
};
use safemlx::{transforms::async_eval_with_event, Event, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        checkpoint::binding::{
            binding_bytes, build_module_bindings, populate_module_from_lease,
            populate_module_from_lease_excluding,
        },
        execution::layerwise::{
            validate_device_budget, validate_host_budget, validate_unused, DensePreparedTransfer,
            DenseStreamController, DenseStreamForwardGuard, DenseStreamGroupGuard,
            DenseTransferWindow,
        },
        residency::manager::{
            host_capacity_upper_bound_for_bindings, ResidencyManager, ResidentTransfer,
            ResidentUnitLease,
        },
    },
};
use crate::MlxTensor;
use eredu_core::residency::{
    MemoryTier, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};

/// One populated MLX unit retained with its residency transfer.
pub struct MlxUnitLease<U> {
    unit: MlxModule<U>,
    _transfer: MlxUnitTransfer,
}

enum MlxUnitTransfer {
    Ordinary { _transfer: ResidentTransfer },
    Dense { _transfer: DensePreparedTransfer },
}

impl<U> std::ops::Deref for MlxUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.unit.inner
    }
}

impl<U> std::ops::DerefMut for MlxUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit.inner
    }
}

/// Exact-completion MLX policy over generic parameterized execution units.
pub struct MlxLayerwisePolicy<U, P = ()> {
    residency: ResidencyManager,
    store: SharedCheckpointSource,
    unit_ids: Vec<OffloadUnitId>,
    layout: ExecutionUnitLayout,
    window_depth: usize,
    populator: P,
    _static_leases: Vec<ResidentUnitLease>,
    pending: VecDeque<(Event, MlxUnitLease<U>)>,
    dense: Option<MlxDenseExecution>,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

struct MlxDenseExecution {
    controller: Arc<DenseStreamController>,
    windows: Vec<Option<DenseTransferWindow>>,
    forward: Option<DenseStreamForwardGuard>,
    groups: Vec<Option<DenseStreamGroupGuard>>,
    prefill: bool,
}

mod resident;
pub use resident::{MlxResidentPolicy, MlxResidentUnit};

/// Statically dispatched parameter population used by the MLX policy.
pub trait MlxUnitPopulator<U> {
    /// Populates the parameters owned by the execution-unit residency policy.
    ///
    /// Most units own every materialized parameter. Architectures with an
    /// independently managed addressable parameter class
    /// override this method with the same exclusion used while binding the
    /// unit. Keeping the ownership decision on the populator makes resident,
    /// bounded, and dense-stream execution follow one contract.
    fn populate(&mut self, unit: &mut MlxModule<U>, lease: &ResidentUnitLease) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        populate_module_from_lease(unit, lease)?;
        Ok(())
    }
}

impl<U> MlxUnitPopulator<U> for () {}

/// Exact parameter exclusions for units with independently stored members.
#[derive(Clone)]
pub struct MlxSelectiveUnitPopulator {
    excluded: Arc<BTreeSet<String>>,
}

impl MlxSelectiveUnitPopulator {
    /// Creates one generic logical-parameter exclusion set.
    pub fn new(excluded: BTreeSet<String>) -> Self {
        Self {
            excluded: Arc::new(excluded),
        }
    }
}

impl<U> MlxUnitPopulator<U> for MlxSelectiveUnitPopulator {
    fn populate(&mut self, unit: &mut MlxModule<U>, lease: &ResidentUnitLease) -> Result<(), Error>
    where
        U: Parameterized<MlxTensor>,
    {
        populate_module_from_lease_excluding(unit, lease, |name| self.excluded.contains(name))?;
        Ok(())
    }
}

/// Builds one unloaded unit through the neutral architecture's declared flat layout.
pub fn construct_architecture_unit<A, S>(
    architecture: &A,
    layout: &ExecutionUnitLayout,
    ordinal: usize,
    stream: &Stream,
    _state: std::marker::PhantomData<S>,
) -> Result<A::Unit, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let address = layout.address(ordinal).ok_or_else(|| {
        Error::ArchitectureModel(format!(
            "architecture unit ordinal {ordinal} is outside 0..{}",
            layout.len()
        ))
    })?;
    architecture
        .build_unit(address.group(), address.index(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl<U, P> MlxLayerwisePolicy<U, P> {
    /// Creates a bounded policy over validated ordered residency units.
    pub fn new(
        residency: ResidencyManager,
        store: SharedCheckpointSource,
        unit_ids: Vec<OffloadUnitId>,
        layout: ExecutionUnitLayout,
        window_depth: usize,
        populator: P,
        static_leases: Vec<ResidentUnitLease>,
        dense: Option<Arc<DenseStreamController>>,
        sample_mlx_memory: bool,
        sample_process_memory: bool,
    ) -> Result<Self, Error> {
        if unit_ids.is_empty() {
            return Err(Error::Parallel(
                "generic MLX layerwise policy requires at least one unit".into(),
            ));
        }
        if window_depth == 0 {
            return Err(Error::Parallel(
                "generic MLX layerwise policy window depth must be nonzero".into(),
            ));
        }
        Ok(Self {
            residency,
            store,
            unit_ids,
            layout: layout.clone(),
            window_depth,
            populator,
            _static_leases: static_leases,
            pending: VecDeque::new(),
            dense: dense.map(|controller| MlxDenseExecution {
                controller,
                windows: (0..layout.group_count()).map(|_| None).collect(),
                forward: None,
                groups: (0..layout.group_count()).map(|_| None).collect(),
                prefill: false,
            }),
            sample_mlx_memory,
            sample_process_memory,
        })
    }

    fn reap_completed(&mut self) -> Result<(), Error> {
        loop {
            let Some((event, _)) = self.pending.front() else {
                return Ok(());
            };
            if !event.is_complete()? {
                return Ok(());
            }
            self.pending.pop_front();
        }
    }

    fn drain_one(&mut self) -> Result<(), Error> {
        if let Some((event, lease)) = self.pending.pop_front() {
            event.synchronize()?;
            drop(lease);
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), Error> {
        while !self.pending.is_empty() {
            self.drain_one()?;
        }
        Ok(())
    }

    fn trim_device_window(
        &self,
        current: usize,
        address: ExecutionUnitAddress,
    ) -> Result<(), Error> {
        let range = self.layout.group_range(address.group()).ok_or_else(|| {
            Error::Parallel(format!("unknown execution group {}", address.group()))
        })?;
        let end = current.saturating_add(self.window_depth).min(range.end);
        for (index, id) in self.unit_ids.iter().enumerate() {
            if !range.contains(&index) || index < current || index >= end {
                self.residency.evict(id, MemoryTier::Device)?;
            }
        }
        Ok(())
    }

    /// Returns the checkpoint source backing this layerwise policy.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }

    /// Clones the shared checkpoint source backing this layerwise policy.
    pub fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        Arc::clone(&self.store)
    }

    /// Returns current weight-residency accounting.
    pub fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency.report().map_err(Into::into)
    }

    /// Returns dense disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense
            .as_ref()
            .map(|dense| dense.controller.report(&self.residency))
            .transpose()
    }

    /// Returns the number of pinned static-weight leases.
    pub fn static_lease_count(&self) -> usize {
        self._static_leases.len()
    }

    fn execution_group(&self, group: usize) -> Result<ResidentLayerGroup, Error> {
        let range = self
            .layout
            .group_range(group)
            .ok_or_else(|| Error::Parallel(format!("unknown execution group {group}")))?;
        let id = self
            .layout
            .group_id(group)
            .expect("validated layout names every execution group")
            .as_str();
        let depth = self.window_depth.min(range.len());
        ResidentLayerGroup::new(id, self.unit_ids[range].iter().cloned(), depth)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Returns residency reports for every semantic execution group.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        (0..self.layout.group_count())
            .map(|group| {
                self.execution_group(group)?
                    .report(&self.residency)
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Populates every unit once and converts the bounded loader into a
    /// permanently resident policy without changing the architecture loop.
    pub fn into_resident<A, S>(
        mut self,
        architecture: &A,
        stream: &Stream,
        _state: std::marker::PhantomData<S>,
    ) -> Result<MlxResidentPolicy<U>, Error>
    where
        U: Parameterized<MlxTensor>,
        P: MlxUnitPopulator<U>,
        A: LayeredArchitecture<MlxNeuralBackend, S, Unit = U>,
        S: RuntimeState<MlxNeuralBackend>,
        A::Error: std::fmt::Display,
    {
        self.drain()?;
        let requests = self
            .unit_ids
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        let transfer = self
            .residency
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let mut units = Vec::with_capacity(self.unit_ids.len());
        for (index, lease) in transfer.leases().iter().enumerate() {
            let address = self
                .layout
                .address(index)
                .expect("validated layout covers every resident unit");
            let mut unit = MlxModule::new(
                architecture
                    .build_unit(address.group(), address.index(), stream)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            );
            self.populator.populate(&mut unit, lease)?;
            units.push(Some(unit));
        }
        Ok(MlxResidentPolicy {
            units,
            residency: self.residency.clone(),
            store: Arc::clone(&self.store),
            unit_ids: self.unit_ids.clone(),
            layout: self.layout.clone(),
            window_depth: self.window_depth,
            _transfer: transfer,
        })
    }

    /// Populates architecture units already constructed by neutral
    /// orchestration and converts this loader into a resident policy.
    pub fn into_resident_units(
        mut self,
        units: Vec<U>,
        stream: &Stream,
    ) -> Result<MlxResidentPolicy<U>, Error>
    where
        U: Parameterized<MlxTensor>,
        P: MlxUnitPopulator<U>,
    {
        self.drain()?;
        if units.len() != self.unit_ids.len() {
            return Err(Error::ArchitectureModel(format!(
                "neutral resident construction supplied {} units for layout length {}",
                units.len(),
                self.unit_ids.len()
            )));
        }
        let requests = self
            .unit_ids
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        let transfer = self
            .residency
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let mut populated = Vec::with_capacity(units.len());
        for (unit, lease) in units.into_iter().zip(transfer.leases()) {
            let mut unit = MlxModule::new(unit);
            self.populator.populate(&mut unit, lease)?;
            populated.push(Some(unit));
        }
        Ok(MlxResidentPolicy {
            units: populated,
            residency: self.residency.clone(),
            store: Arc::clone(&self.store),
            unit_ids: self.unit_ids.clone(),
            layout: self.layout.clone(),
            window_depth: self.window_depth,
            _transfer: transfer,
        })
    }
}

fn populate_parameterized<M: Parameterized<MlxTensor>>(
    module: &mut M,
    lease: &ResidentUnitLease,
) -> Result<(), Error> {
    populate_module_from_lease(module, lease).map_err(Error::from)
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current = current
                .checked_add(*bytes)
                .ok_or_else(|| Error::Parallel("generic device window bytes overflowed".into()))?;
        }
        largest = largest.max(current);
    }
    Ok(largest)
}

/// Derives the canonical flat unit layout from a neutral architecture.
pub fn architecture_execution_layout<A, S>(architecture: &A) -> Result<ExecutionUnitLayout, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let counts = (0..graph.groups().len())
        .map(|group| {
            architecture
                .group_unit_count(group)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

/// Cold preparation for bounded and resident policies.
mod preparation;
pub use preparation::{
    prepare_layerwise_policy, prepare_layerwise_policy_from_bindings,
    prepare_layerwise_policy_with_bindings,
};

mod bounded;
