//! SafeMLX realization of the backend-neutral layerwise unit policy.

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{
    DenseDiskStreamReport, ExecutionUnitAddress, ExecutionUnitLayout, LayerWeightResidency,
    LayerwiseModelMetadata,
};

use std::{
    collections::{BTreeSet, VecDeque},
    sync::Arc,
};

use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};
use eredu_runtime::{LayerwisePolicy, OffloadUnit, WeightBinding};
use safemlx::{transforms::async_eval_with_event, Array, Event, Stream};

use crate::backend::mlx::{
    error::Error,
    nn::shared::{MlxBackend, MlxModule},
    runtime::{
        checkpoint::binding::{binding_bytes, build_module_bindings, populate_module_from_lease},
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
use crate::core::residency::{
    MemoryTier, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};

/// One populated MLX unit retained with its residency transfer.
pub(crate) struct MlxUnitLease<U> {
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
pub(crate) struct MlxLayerwisePolicy<U, F> {
    residency: ResidencyManager,
    store: SharedCheckpointSource,
    unit_ids: Vec<OffloadUnitId>,
    layout: ExecutionUnitLayout,
    window_depth: usize,
    build: F,
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

/// Permanently populated MLX units used by fully resident execution.
pub(crate) struct MlxResidentPolicy<U> {
    units: Vec<Option<MlxModule<U>>>,
    residency: ResidencyManager,
    store: SharedCheckpointSource,
    _transfer: ResidentTransfer,
}

/// Exclusive borrow-by-ownership of one permanently resident unit.
pub(crate) struct MlxResidentUnit<U> {
    index: usize,
    unit: MlxModule<U>,
}

impl<U> std::ops::Deref for MlxResidentUnit<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.unit.inner
    }
}

impl<U> std::ops::DerefMut for MlxResidentUnit<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit.inner
    }
}

/// Statically dispatched unloaded-unit construction used by the MLX policy.
pub(crate) trait MlxUnitFactory<U> {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<U, Error>;
}

impl<U, F> MlxUnitFactory<U> for F
where
    F: FnMut(usize, &Stream) -> Result<U, Error>,
{
    fn build(&mut self, index: usize, stream: &Stream) -> Result<U, Error> {
        self(index, stream)
    }
}

impl<U, F> MlxLayerwisePolicy<U, F> {
    /// Creates a bounded policy over validated ordered residency units.
    pub(crate) fn new(
        residency: ResidencyManager,
        store: SharedCheckpointSource,
        unit_ids: Vec<OffloadUnitId>,
        layout: ExecutionUnitLayout,
        window_depth: usize,
        build: F,
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
            build,
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

    pub(crate) fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }

    pub(crate) fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency.report().map_err(Into::into)
    }

    pub(crate) fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense
            .as_ref()
            .map(|dense| dense.controller.report(&self.residency))
            .transpose()
    }

    pub(crate) fn static_lease_count(&self) -> usize {
        self._static_leases.len()
    }

    /// Evicts every temporary execution-device unit after exact completion.
    pub(crate) fn clear_device_window(&self) -> Result<(), Error> {
        if let Some(dense) = &self.dense {
            dense.controller.clear_group(&self.residency, "model")?;
        }
        for id in &self.unit_ids {
            self.residency.evict(id, MemoryTier::Device)?;
        }
        Ok(())
    }

    /// Populates every unit once and converts the bounded loader into a
    /// permanently resident policy without changing the architecture loop.
    pub(crate) fn into_resident(mut self, stream: &Stream) -> Result<MlxResidentPolicy<U>, Error>
    where
        U: Parameterized<Array>,
        F: MlxUnitFactory<U>,
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
            let mut unit = MlxModule::new(self.build.build(index, stream)?);
            populate_module_from_lease(&mut unit, lease)?;
            units.push(Some(unit));
        }
        Ok(MlxResidentPolicy {
            units,
            residency: self.residency.clone(),
            store: Arc::clone(&self.store),
            _transfer: transfer,
        })
    }
}

impl<U> MlxResidentPolicy<U> {
    pub(crate) fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }

    pub(crate) fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency.report().map_err(Into::into)
    }

    pub(crate) fn static_lease_count(&self) -> usize {
        1
    }
}

impl<U> LayerwisePolicy<MlxBackend, U> for MlxResidentPolicy<U> {
    type Lease = MlxResidentUnit<U>;
    type Error = Error;

    fn begin(&mut self, _initial: &Array, _stream: &Stream) -> Result<(), Self::Error> {
        Ok(())
    }

    fn acquire(
        &mut self,
        index: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
        _stream: &Stream,
    ) -> Result<Self::Lease, Self::Error> {
        let count = self.units.len();
        let unit = self
            .units
            .get_mut(index)
            .ok_or_else(|| {
                Error::Parallel(format!("resident unit {index} is outside {count} units"))
            })?
            .take()
            .ok_or_else(|| Error::Parallel(format!("resident unit {index} is already acquired")))?;
        Ok(MlxResidentUnit { index, unit })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
        lease: Self::Lease,
        _output: &'a Array,
        _state_values: StateValues,
        _context_values: ContextValues,
        _stream: &Stream,
    ) -> Result<(), Self::Error>
    where
        Array: 'a,
        StateValues: Iterator<Item = &'a Array>,
        ContextValues: Iterator<Item = &'a Array>,
    {
        if lease.index != index {
            return Err(Error::Parallel(format!(
                "resident completion returned unit {} to slot {index}",
                lease.index
            )));
        }
        let slot = &mut self.units[index];
        if slot.replace(lease.unit).is_some() {
            return Err(Error::Parallel(format!(
                "resident unit slot {index} was unexpectedly occupied"
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _output: &Array, _stream: &Stream) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn populate_parameterized<M: Parameterized<Array>>(
    module: &mut M,
    lease: &ResidentUnitLease,
) -> Result<(), Error> {
    struct Binder<'a> {
        lease: &'a ResidentUnitLease,
        visited: BTreeSet<String>,
        error: Option<String>,
    }
    impl<'a, 'value> ParameterVisitorMut<'value, Array> for Binder<'a> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, parameter: &'value mut Array) {
            if self.error.is_some() {
                return;
            }
            let name = metadata.id.to_string();
            let value = match self.lease.device_value(&name) {
                Ok(value) => value,
                Err(error) => {
                    self.error = Some(error.to_string());
                    return;
                }
            };
            if parameter.shape() != value.shape() {
                self.error = Some(format!(
                    "resident parameter {name:?} has shape {:?}, expected {:?}",
                    value.shape(),
                    parameter.shape()
                ));
                return;
            }
            *parameter = value.clone();
            self.visited.insert(name);
        }
    }
    let mut binder = Binder {
        lease,
        visited: BTreeSet::new(),
        error: None,
    };
    module.visit_parameters_mut(&mut binder);
    if let Some(error) = binder.error {
        return Err(Error::Parallel(error));
    }
    let resident = lease
        .binding_names()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if binder.visited != resident {
        return Err(Error::Parallel(format!(
            "static resident bindings {:?} do not match parameter topology {:?}",
            resident, binder.visited
        )));
    }
    Ok(())
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

/// Builds a generic MLX layerwise policy from neutral parameter topologies.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_layerwise_policy<SM, U, F, I>(
    store: SharedCheckpointSource,
    static_modules: &mut SM,
    build: F,
    layout: ExecutionUnitLayout,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
) -> Result<(MlxLayerwisePolicy<U, F>, LayerwiseModelMetadata), Error>
where
    SM: Clone + Parameterized<Array>,
    U: Parameterized<Array>,
    F: MlxUnitFactory<U>,
    I: Fn(&str) -> bool,
{
    prepare_layerwise_policy_with_bindings(
        store,
        static_modules,
        build,
        layout,
        options,
        stream,
        weights_stream,
        ignored,
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        |_index, unit, store, _stream| {
            build_module_bindings(&MlxModule::new(unit), "", store).map_err(Into::into)
        },
    )
}

/// Builds the generic policy from caller-realized local binding selections.
///
/// Parallel composition uses this entry point to provide rank-local checkpoint
/// selections while retaining the same residency, overlap, and completion
/// algorithm used by replicated execution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_layerwise_policy_with_bindings<SM, U, F, I, SB, UB>(
    store: SharedCheckpointSource,
    static_modules: &mut SM,
    mut build: F,
    layout: ExecutionUnitLayout,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    ignored: I,
    static_bindings: SB,
    mut unit_bindings: UB,
) -> Result<(MlxLayerwisePolicy<U, F>, LayerwiseModelMetadata), Error>
where
    SM: Parameterized<Array>,
    U: Parameterized<Array>,
    F: MlxUnitFactory<U>,
    I: Fn(&str) -> bool,
    SB: FnOnce(
        &SM,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
    UB: FnMut(
        usize,
        U,
        &dyn eredu_checkpoint::store::CheckpointSource,
        &Stream,
    ) -> Result<Vec<WeightBinding>, Error>,
{
    let unit_count = layout.len();
    if unit_count == 0 {
        return Err(Error::Parallel(
            "generic MLX architecture declared no execution units".into(),
        ));
    }
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let depth = options.device_depth(unit_count);
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();

    let static_id = OffloadUnitId::new("model.static")?;
    let static_bindings = static_bindings(static_modules, store.as_ref())?;
    let static_bytes = binding_bytes(&static_bindings)?;
    consumed.extend(
        static_bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
    );
    definitions.push(OffloadUnit::new(static_id.clone(), static_bindings)?);
    specs.push(OffloadUnitSpec::new(
        static_id.clone(),
        static_bytes,
        ResidencyPolicy::Pinned,
        MemoryTier::Device,
    )?);

    let mut unit_ids = Vec::with_capacity(unit_count);
    let mut unit_bytes = Vec::with_capacity(unit_count);
    let mut layer_parameter_bytes = 0u64;
    let mut total_host_bytes = 0u64;
    let mut maximum_host_bytes = 0u64;
    for index in 0..unit_count {
        let unit = build.build(index, stream)?;
        let bindings = unit_bindings(index, unit, store.as_ref(), stream)?;
        let bytes = binding_bytes(&bindings)?;
        layer_parameter_bytes = layer_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("generic layer bytes overflowed".into()))?;
        let host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
        total_host_bytes = total_host_bytes
            .checked_add(host_bytes)
            .ok_or_else(|| Error::Parallel("generic host unit bytes overflowed".into()))?;
        maximum_host_bytes = maximum_host_bytes.max(host_bytes);
        consumed.extend(
            bindings
                .iter()
                .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned)),
        );
        let address = layout
            .address(index)
            .expect("validated layout covers every flat unit");
        let group_id = layout
            .group_id(address.group())
            .expect("validated layout names every execution group");
        let id = OffloadUnitId::new(format!(
            "model.{}.{:05}",
            group_id.as_str(),
            address.index()
        ))?;
        definitions.push(OffloadUnit::new(id.clone(), bindings)?);
        specs.push(OffloadUnitSpec::new(
            id.clone(),
            bytes,
            if fully_resident {
                ResidencyPolicy::Pinned
            } else if dense.is_some() {
                ResidencyPolicy::Cacheable
            } else {
                ResidencyPolicy::Windowed
            },
            if fully_resident {
                MemoryTier::Device
            } else if dense.is_some() {
                MemoryTier::Disk
            } else {
                MemoryTier::Host
            },
        )?);
        unit_ids.push(id);
        unit_bytes.push(bytes);
    }
    consumed.extend(store.materialized_source_keys());
    validate_unused(store.as_ref(), &consumed, options.strict_loading(), ignored)?;
    let device_window_bytes = (0..layout.group_count())
        .map(|group| {
            let range = layout
                .group_range(group)
                .expect("validated layout covers every execution group");
            largest_window_bytes(&unit_bytes[range], depth)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let host_required = match dense {
        Some(dense) if dense.host_budget_bytes > 0 => maximum_host_bytes
            .checked_mul(dense.host_lookahead as u64)
            .ok_or_else(|| Error::Parallel("generic host window bytes overflowed".into()))?,
        Some(_) => 0,
        None if fully_resident => 0,
        None => total_host_bytes,
    };
    validate_host_budget(offload, host_required)?;
    validate_device_budget(offload, static_bytes, device_window_bytes, depth)?;

    let plan = OffloadPlan::new(offload, specs)?;
    let residency_stream = if dense.is_some() {
        Stream::new_with_device(&stream.get_device()?)
    } else {
        stream.clone()
    };
    let residency = ResidencyManager::new_shared(
        Arc::clone(&store),
        plan,
        definitions,
        weights_stream.clone(),
        residency_stream,
    )?;
    residency.initialize()?;
    let static_lease = residency.acquire(&static_id, MemoryTier::Device)?;
    populate_parameterized(static_modules, &static_lease)?;
    let metadata = LayerwiseModelMetadata::new(
        "generic",
        None,
        unit_count,
        static_bytes,
        options.execution_residency(),
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_bytes,
        depth,
    );
    let dense_controller = dense
        .map(|options| {
            DenseStreamController::new(
                &residency,
                options,
                unit_count,
                layer_parameter_bytes,
                maximum_host_bytes,
                static_bytes,
                (0..layout.group_count()).map(|group| {
                    let range = layout
                        .group_range(group)
                        .expect("validated layout covers every execution group");
                    let id = layout
                        .group_id(group)
                        .expect("validated layout names every execution group")
                        .as_str()
                        .to_owned();
                    (id, unit_ids[range].to_vec())
                }),
            )
            .map(Arc::new)
        })
        .transpose()?;
    let policy = MlxLayerwisePolicy::new(
        residency,
        Arc::clone(&store),
        unit_ids,
        layout,
        depth,
        build,
        vec![static_lease],
        dense_controller,
        options.sample_backend_memory(),
        options.sample_process_memory(),
    )?;
    Ok((policy, metadata))
}

impl<U, F> LayerwisePolicy<MlxBackend, U> for MlxLayerwisePolicy<U, F>
where
    U: Parameterized<Array>,
    F: MlxUnitFactory<U>,
{
    type Lease = MlxUnitLease<U>;
    type Error = Error;

    fn begin(&mut self, initial: &Array, _stream: &Stream) -> Result<(), Self::Error> {
        let Some(dense) = &mut self.dense else {
            return Ok(());
        };
        if dense.windows.iter().any(Option::is_some)
            || dense.forward.is_some()
            || dense.groups.iter().any(Option::is_some)
        {
            return Err(Error::Parallel(
                "dense MLX policy began a forward while another remained active".into(),
            ));
        }
        let prefill = initial.dim(1) > 1;
        let forward = dense.controller.forward_guard(prefill, &self.residency)?;
        dense.prefill = prefill;
        dense.forward = Some(forward);
        Ok(())
    }

    fn acquire(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        stream: &Stream,
    ) -> Result<Self::Lease, Self::Error> {
        if self.layout.address(index) != Some(address) {
            return Err(Error::Parallel(format!(
                "execution unit {index} does not match group {} unit {}",
                address.group(),
                address.index()
            )));
        }
        self.reap_completed()?;
        if self.dense.is_some() {
            let group = address.group();
            let group_id = self
                .layout
                .group_id(group)
                .expect("validated unit address names its execution group")
                .as_str();
            let range = self
                .layout
                .group_range(group)
                .expect("validated unit address has a group range");
            let dense = self.dense.as_mut().expect("dense policy is active");
            if dense.groups[group].is_none() {
                dense.groups[group] = Some(dense.controller.group_guard(&self.residency, group_id));
            }
            if dense.windows[group].is_none() {
                dense.windows[group] = Some(dense.controller.transfer_window(
                    &self.residency,
                    group_id,
                    &self.unit_ids,
                    range,
                    dense.prefill,
                )?);
            }
            loop {
                let refill = self
                    .dense
                    .as_mut()
                    .and_then(|dense| dense.windows[group].as_mut())
                    .expect("dense forward begins before acquisition")
                    .refill();
                match refill {
                    Ok(()) => break,
                    Err(_) if !self.pending.is_empty() => self.drain_one()?,
                    Err(error) => return Err(error),
                }
            }
            let transfer = self
                .dense
                .as_mut()
                .and_then(|dense| dense.windows[group].as_mut())
                .expect("dense forward begins before acquisition")
                .next(stream)?;
            if transfer.index() != index {
                return Err(Error::Parallel(format!(
                    "dense transfer returned unit {}, expected {index}",
                    transfer.index()
                )));
            }
            let mut unit = MlxModule::new(self.build.build(index, stream)?);
            populate_module_from_lease(&mut unit, transfer.lease())?;
            return Ok(MlxUnitLease {
                unit,
                _transfer: MlxUnitTransfer::Dense {
                    _transfer: transfer,
                },
            });
        }
        // An ordinary lookahead transfer owns every lease in its window until
        // the preceding unit's exact completion.  Release that completed
        // window before requesting the overlapping next window; otherwise an
        // overlapping in-flight unit would wait for the very transfer guard
        // retained by `pending` on this thread.
        self.drain_one()?;
        self.trim_device_window(index, address)?;
        let group_range = self
            .layout
            .group_range(address.group())
            .expect("validated unit address has a group range");
        let end = index.saturating_add(self.window_depth).min(group_range.end);
        let requests = self.unit_ids[index..end]
            .iter()
            .cloned()
            .map(|id| (id, 1))
            .collect::<Vec<_>>();
        let transfer = loop {
            match self
                .residency
                .acquire_many_with_transfer(&requests, MemoryTier::Device)
            {
                Ok(transfer) => break transfer,
                Err(_) if !self.pending.is_empty() => self.drain_one()?,
                Err(error) => return Err(error.into()),
            }
        };
        transfer.order_after(stream)?;
        let mut unit = MlxModule::new(self.build.build(index, stream)?);
        populate_module_from_lease(&mut unit, &transfer.leases()[0])?;
        Ok(MlxUnitLease {
            unit,
            _transfer: MlxUnitTransfer::Ordinary {
                _transfer: transfer,
            },
        })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        _index: usize,
        _address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a Array,
        state_values: StateValues,
        context_values: ContextValues,
        _stream: &Stream,
    ) -> Result<(), Self::Error>
    where
        Array: 'a,
        StateValues: Iterator<Item = &'a Array>,
        ContextValues: Iterator<Item = &'a Array>,
    {
        let event = async_eval_with_event(
            std::iter::once(output)
                .chain(state_values)
                .chain(context_values),
        )?;
        self.pending.push_back((event, lease));
        Ok(())
    }

    fn finish(&mut self, output: &Array, _stream: &Stream) -> Result<(), Self::Error> {
        async_eval_with_event([output])?.synchronize()?;
        self.drain()?;
        if self.dense.is_none() && (self.sample_mlx_memory || self.sample_process_memory) {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        if let Some(dense) = &mut self.dense {
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                if let Some(group) = group.take() {
                    group.complete()?;
                }
            }
            if let Some(forward) = dense.forward.take() {
                forward.complete()?;
            }
        }
        Ok(())
    }
}

impl<U, F> Drop for MlxLayerwisePolicy<U, F> {
    fn drop(&mut self) {
        for (event, _) in &self.pending {
            let _ = event.synchronize();
        }
    }
}
