//! Architecture-independent execution of decoder models from resident layers.
//!
//! [`crate::backend::mlx::runtime::execution::layerwise::LayerwiseModel`] owns checkpoint
//! storage, residency, bounded device
//! windows, and synchronization. Model-family behavior is supplied by an
//! [`crate::backend::mlx::runtime::execution::layerwise::ArchitectureAdapter`].

use eredu_checkpoint::{
    store::{ResolvedCheckpointSource, SharedCheckpointSource, WeightStoreBackend},
    WeightQuantization,
};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, DenseDiskStreamReport, DenseStreamTelemetry, DenseTransferSchedule,
    ExecutionResidency, LayerWeightResidency, LayerwiseModelMetadata, OffloadUnit,
    ParallelModelInfo, StaticUnitBindings, WeightBinding, WeightResidency, DENSE_TRANSFER_WINDOW,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::Path,
    sync::Arc,
};

use safemlx::{module::ModuleParameters, transforms::async_eval_with_event, Array, Event, Stream};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_bindings, is_materialized_module_parameter,
        populate_module_from_lease, ModuleBindingError,
    },
    backend::mlx::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::mlx::runtime::checkpoint::recipe::RecipeDtype,
    backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore,
    backend::mlx::runtime::execution::inspection::{ActivationObserver, ActivationObserverProxy},
    backend::mlx::runtime::residency::dense_stream::BackgroundLayerPrefetch,
    backend::mlx::runtime::residency::manager::{
        host_capacity_upper_bound_for_bindings, ResidencyError, ResidencyManager, ResidentTransfer,
        ResidentUnitLease,
    },
    core::residency::{
        MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec,
        ResidencyLedgerError, ResidencyPolicy,
    },
};
use eredu_runtime::PagedCacheOptions;

use eredu_runtime::{ResidencyReport, ResidentLayerGroup, WeightMaterializationReport};

pub(crate) fn resolve_checkpoint_store<A: ArchitectureAdapter>(
    store: SharedCheckpointSource,
    adapter: &A,
) -> Result<SharedCheckpointSource, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend != WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = adapter.safetensors_checkpoint_plan()?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{} checkpoint contract did not resolve: {validation:?}",
                adapter.model_type()
            ))
        })?;
    Ok(Arc::new(ResolvedCheckpointSource::new(store, resolved)))
}

pub(crate) fn open_safetensors_weight_store(
    model_dir: &Path,
    max_mapped_shards: usize,
) -> Result<SharedCheckpointSource, Error> {
    Ok(Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    ))
}

pub(crate) fn validate_gguf_layerwise_source(
    checkpoint: &safemlx::ops::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    options: LayerWeightResidency,
) -> Result<crate::core::GgufArchitecture, Error> {
    let architecture_name = match metadata.get("general.architecture") {
        Some(safemlx::ops::GgufMetadataValue::String(name)) => name,
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key general.architecture has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing general.architecture".into(),
            ));
        }
    };
    let architecture = crate::core::GgufArchitecture::resolve(architecture_name)?;
    let residency = WeightResidency::with_layers(options);
    crate::composition::mlx::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    Ok(architecture)
}

pub(crate) struct DenseStreamController {
    options: DenseDiskStreamLoadOptions,
    background: Option<BackgroundLayerPrefetch>,
    telemetry: DenseStreamTelemetry,
}

impl DenseStreamController {
    pub(crate) fn new(
        manager: &ResidencyManager,
        options: DenseDiskStreamLoadOptions,
        planned_layer_count: usize,
        planned_layer_bytes: u64,
        maximum_host_layer_bytes: u64,
        pinned_static_device_bytes: u64,
        groups: impl IntoIterator<Item = (String, Vec<OffloadUnitId>)>,
    ) -> Result<Self, Error> {
        let background = (options.host_budget_bytes > 0)
            .then(|| {
                BackgroundLayerPrefetch::new(manager.clone(), options.background_queue_capacity)
            })
            .transpose()?;
        let transfer_stream_index = manager.device_stream_index()?;
        Ok(Self {
            options,
            background,
            telemetry: DenseStreamTelemetry::new(
                planned_layer_count,
                planned_layer_bytes,
                maximum_host_layer_bytes,
                pinned_static_device_bytes,
                transfer_stream_index,
                groups,
            ),
        })
    }

    pub(crate) fn transfer_window(
        self: &Arc<Self>,
        manager: &ResidencyManager,
        group: impl Into<String>,
        units: &[OffloadUnitId],
        indices: impl IntoIterator<Item = usize>,
        prefill: bool,
    ) -> Result<DenseTransferWindow, Error> {
        let indices = indices.into_iter().collect::<Vec<_>>();
        if let Some(&index) = indices.iter().find(|&&index| index >= units.len()) {
            return Err(LayerwiseModelError::InvalidDenseTransferWindow {
                index,
                unit_count: units.len(),
            }
            .into());
        }
        let mut window = DenseTransferWindow {
            controller: Arc::clone(self),
            manager: manager.clone(),
            group: group.into(),
            units: units.to_vec(),
            schedule: DenseTransferSchedule::new(indices, DENSE_TRANSFER_WINDOW)?,
            prefill,
        };
        if let Err(error) = window.refill() {
            if !window.schedule.has_ready() || !is_temporary_residency_contention(&error) {
                return Err(error);
            }
        }
        Ok(window)
    }

    fn observe_group(
        &self,
        manager: &ResidencyManager,
        group: &str,
        prefill: bool,
    ) -> Result<(), Error> {
        let (_, _, units, _) = manager.telemetry_snapshot()?;
        self.telemetry.observe_group(group, prefill, &units)?;
        Ok(())
    }

    fn record_group_execution(&self, group: &str) -> Result<(), Error> {
        self.telemetry.record_group_execution(group)?;
        Ok(())
    }

    pub(crate) fn clear_group(&self, manager: &ResidencyManager, group: &str) -> Result<(), Error> {
        manager.protect_group_window(&format!("dense:{group}:host"), &[], MemoryTier::Host)?;
        manager.protect_group_window(&format!("dense:{group}:device"), &[], MemoryTier::Device)?;
        if let Some(background) = &self.background {
            background.cancel()?;
        }
        Ok(())
    }

    pub(crate) fn forward_guard(
        self: &Arc<Self>,
        prefill: bool,
        manager: &ResidencyManager,
    ) -> Result<DenseStreamForwardGuard, Error> {
        let (_, offload, _, _) = manager.telemetry_snapshot()?;
        self.telemetry.begin_forward(prefill, &offload)?;
        Ok(DenseStreamForwardGuard {
            controller: Arc::clone(self),
            manager: manager.clone(),
            armed: true,
        })
    }

    fn commit_forward(&self, manager: &ResidencyManager) -> Result<(), Error> {
        if self.options.sample_backend_memory || self.options.sample_process_memory {
            manager.sample_memory(
                self.options.sample_backend_memory,
                self.options.sample_process_memory,
            )?;
        }
        let (_, offload, _, _) = manager.telemetry_snapshot()?;
        self.telemetry.commit_forward(&offload)?;
        Ok(())
    }

    fn abort_forward(&self) {
        self.telemetry.abort_forward();
    }

    pub(crate) fn group_guard(
        self: &Arc<Self>,
        manager: &ResidencyManager,
        group: &str,
    ) -> DenseStreamGroupGuard {
        DenseStreamGroupGuard {
            controller: Arc::clone(self),
            manager: manager.clone(),
            group: group.to_string(),
            armed: true,
        }
    }

    pub(crate) fn report(
        &self,
        manager: &ResidencyManager,
    ) -> Result<DenseDiskStreamReport, Error> {
        let residency = manager.report()?;
        let background = self
            .background
            .as_ref()
            .map(BackgroundLayerPrefetch::report)
            .transpose()?
            .unwrap_or_default();
        Ok(self.telemetry.report(residency, background)?)
    }
}
/// thread supported by MLX events. A window submits at most two device copies.
/// Callers consume one entry, evaluate and synchronize its compute work, drop
/// that entry, and then call [`Self::refill`] to submit the following layer.
pub(crate) struct DenseTransferWindow {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    group: String,
    units: Vec<OffloadUnitId>,
    schedule: DenseTransferSchedule<DensePreparedTransfer>,
    prefill: bool,
}

impl DenseTransferWindow {
    fn has_ready(&self) -> bool {
        self.schedule.has_ready()
    }

    fn is_exhausted(&self) -> bool {
        self.schedule.is_exhausted()
    }

    /// Takes the next transfer after ordering `consumer` behind its event.
    pub(crate) fn next(&mut self, consumer: &Stream) -> Result<DensePreparedTransfer, Error> {
        let (index, transfer) = self.schedule.pop_ready().ok_or({
            LayerwiseModelError::InvalidDenseTransferWindow {
                index: self.units.len(),
                unit_count: self.units.len(),
            }
        })?;
        debug_assert_eq!(transfer.index(), index);
        transfer.transfer.order_after(consumer)?;
        Ok(transfer)
    }

    /// Reprotects the current/next units and submits one replacement transfer.
    ///
    /// The completed [`DensePreparedTransfer`] must be dropped before this is
    /// called so the fixed two-layer device budget can admit the replacement.
    pub(crate) fn refill(&mut self) -> Result<(), Error> {
        let device_indices = self.schedule.desired_indices(DENSE_TRANSFER_WINDOW);
        let host_indices = if self.controller.background.is_some() {
            self.schedule
                .desired_indices(self.controller.options.host_lookahead)
        } else {
            Vec::new()
        };
        let device_units = device_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        let host_units = host_indices
            .iter()
            .map(|&index| self.units[index].clone())
            .collect::<Vec<_>>();
        self.manager.protect_group_window(
            &format!("dense:{}:host", self.group),
            &host_units,
            MemoryTier::Host,
        )?;
        self.manager.protect_group_window(
            &format!("dense:{}:device", self.group),
            &device_units,
            MemoryTier::Device,
        )?;
        if let Some(background) = &self.controller.background {
            for id in &host_units {
                background.submit(id)?;
            }
        }
        while self.schedule.can_admit() {
            let Some(index) = self.schedule.next_pending() else {
                break;
            };
            let id = &self.units[index];
            let _host = if host_indices.contains(&index) {
                self.controller
                    .background
                    .as_ref()
                    .map(|background| background.acquire(id))
                    .transpose()?
            } else {
                None
            };
            let transfer = self
                .manager
                .acquire_many_with_transfer(&[(id.clone(), 1)], MemoryTier::Device)?;
            self.schedule
                .admit(index, DensePreparedTransfer { index, transfer })?;
        }
        self.controller
            .observe_group(&self.manager, &self.group, self.prefill)?;
        Ok(())
    }
}

impl Drop for DenseTransferWindow {
    fn drop(&mut self) {
        let _ = self.controller.clear_group(&self.manager, &self.group);
    }
}

/// One populated-unit dependency taken from a [`DenseTransferWindow`].
pub(crate) struct DensePreparedTransfer {
    index: usize,
    transfer: ResidentTransfer,
}

impl DensePreparedTransfer {
    /// Returns the index in the group's authoritative unit list.
    pub(crate) const fn index(&self) -> usize {
        self.index
    }

    /// Returns the single resident lease protected by this transfer.
    pub(crate) fn lease(&self) -> &ResidentUnitLease {
        self.transfer
            .leases()
            .first()
            .expect("dense transfer always acquires one unit")
    }
}

pub(crate) struct DenseStreamForwardGuard {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    armed: bool,
}

impl DenseStreamForwardGuard {
    pub(crate) fn complete(mut self) -> Result<(), Error> {
        let result = self.controller.commit_forward(&self.manager);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for DenseStreamForwardGuard {
    fn drop(&mut self) {
        if self.armed {
            self.controller.abort_forward();
        }
    }
}

pub(crate) struct DenseStreamGroupGuard {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    group: String,
    armed: bool,
}

impl DenseStreamGroupGuard {
    pub(crate) fn complete(mut self) -> Result<(), Error> {
        let result = self
            .controller
            .clear_group(&self.manager, &self.group)
            .and_then(|()| self.controller.record_group_execution(&self.group));
        self.armed = false;
        result
    }
}

impl Drop for DenseStreamGroupGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.controller.clear_group(&self.manager, &self.group);
        }
    }
}

use eredu_runtime::{ExecutionGraph, ExecutionGroupSchedule};

/// Architecture contract for resident, bounded-residency, and distributed
/// execution.
///
/// Heterogeneous caches, architecture-specific inputs, multiple execution
/// groups, and retained recurrent state are represented directly rather than
/// being forced into a decoder-only KV-cache interface.
pub trait ArchitectureAdapter: Sized {
    /// Borrowed family-specific forward input.
    type Input<'a>;
    /// Complete architecture-owned cache and recurrent state.
    type Cache;
    /// Runtime execution unit. Families with heterogeneous blocks may use an enum.
    type Layer: ModuleParameters;
    /// Masks, positions, prepared media, or other per-forward state.
    type ForwardContext;

    /// Stable architecture identity used by residency metadata.
    fn model_type(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Returns the architecture-owned physical contract used to resolve the
    /// SafeTensors store before any binding recipe is inferred or materialized.
    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error>;

    /// Model-wide checkpoint quantization, when one uniform encoding exists.
    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        None
    }

    /// Returns whether a floating static matrix is represented by a packed
    /// parameter group in this adapter. Multimodal adapters override this for
    /// checkpoint components whose target modules intentionally stay dense.
    fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    /// Returns the exact cache compatibility identity for replicated or
    /// rank-local parallel execution.
    fn prompt_cache_model_identity(
        &self,
        _topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not declared a prompt-cache identity",
            std::any::type_name::<Self>()
        )))
    }

    /// Persists a validated architecture-owned cache.
    #[allow(clippy::too_many_arguments)]
    fn save_prompt_cache(
        &self,
        _cache: &mut Self::Cache,
        _destination: &Path,
        _descriptor: PromptCacheDescriptor,
        _prefix_token_ids: &[u32],
        _options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented prompt-cache persistence",
            std::any::type_name::<Self>()
        )))
    }

    /// Restores a validated architecture-owned cache.
    #[allow(clippy::too_many_arguments)]
    fn load_prompt_cache(
        &self,
        _directory: &Path,
        _expected: &PromptCacheDescriptor,
        _identity: &PromptCacheModelIdentity,
        _prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented prompt-cache restoration",
            std::any::type_name::<Self>()
        )))
    }

    /// Builds bindings for modules that remain pinned on the execution device.
    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error>;

    /// Builds only static units selected by their stable architecture id.
    ///
    /// Adapters should override this when binding construction consults a
    /// sharded store, so distributed stages do not open unowned static shards.
    fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        Ok(self
            .static_units(store)?
            .into_iter()
            .filter(|unit| select(unit.id().as_str()))
            .collect())
    }

    /// Assigns pinned leases to the adapter's static modules.
    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error>;

    /// Validates or initializes the complete cache before any weight lease is acquired.
    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error>;

    /// Embeds or prepares the input and creates family-owned forward context.
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Self::ForwardContext>, Error>;

    /// Embeds or prepares input under an explicit execution context.
    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.begin_forward(input, cache, execution.stream())
    }

    /// Declares the complete named execution-group dependency graph.
    fn execution_graph(&self) -> Result<ExecutionGraph, Error>;

    /// Returns whether a group is needed for this particular forward pass.
    ///
    /// This lets multimodal adapters skip vision groups during text-only decode.
    fn should_execute_group(&self, _group: usize, _context: &Self::ForwardContext) -> bool {
        true
    }

    /// Returns the number of ordered units in one group.
    fn layer_count(&self, group: usize) -> Result<usize, Error>;

    /// Creates a metadata-only runtime unit for one group position.
    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error>;

    /// Describes this architecture's physical checkpoint tensors using typed
    /// logical roles for tensor-parallel planning.
    ///
    /// Adapters without an exact tensor-parallel parameter plan fail closed.
    fn parallel_parameter_groups(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not declared tensor-parallel parameter roles",
            std::any::type_name::<Self>()
        )))
    }

    /// Registers placement for streamed and pinned parameters. Composite
    /// adapters can override this to reuse the planners of their nested model
    /// families instead of duplicating parameter-name logic.
    fn register_parallel_parameters(
        &self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        _stream: &Stream,
    ) -> Result<(), Error> {
        for group in self.parallel_parameter_groups(context)? {
            planner.register(group)?;
        }
        Ok(())
    }

    /// Rebuilds pinned modules whose parameter geometry is rank-local.
    ///
    /// The loader captures the global static bindings before invoking this
    /// hook, then applies the typed layout to those bindings before residency
    /// initialization.
    fn configure_parallel_static(
        &mut self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &eredu_runtime::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Creates a rank-local runtime unit from planned model geometry.
    fn new_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &eredu_runtime::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented rank-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Creates a rank-local runtime unit whose routed experts follow an
    /// authoritative expert assignment.
    ///
    /// Architectures supporting PP+EP implement this hook instead of exposing
    /// expert-bank representation details to the pipeline runtime.
    fn new_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented expert-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Creates a layer whose ordinary projections follow the TP layout while
    /// routed experts are restricted to the authoritative EP assignment.
    ///
    /// Architectures opt into triple-axis execution by implementing this one
    /// semantic composition point; pipeline placement remains external.
    fn new_tensor_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &eredu_runtime::LocalModelLayout,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented combined tensor/expert-local layer construction",
            std::any::type_name::<Self>()
        )))
    }

    /// Derives this rank's expert ownership from architecture metadata and the
    /// authoritative Cartesian topology.
    ///
    /// The default accepts an inactive EP axis and fails closed otherwise.
    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size > 1 {
            Err(Error::Parallel(format!(
                "architecture adapter {} has not declared expert ownership for EP size {}",
                std::any::type_name::<Self>(),
                topology.expert_parallel_size
            )))
        } else {
            Ok(None)
        }
    }

    /// Creates one runtime unit for replicated, TP-local, or EP-local
    /// execution from shared semantic inputs.
    ///
    /// Architecture adapters own the semantic composition of simultaneous TP
    /// and EP; PP remains the outer selection of execution units.
    fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        match (layout, assignment) {
            (None, None) => self.new_layer(group, index, stream),
            (Some(layout), None) => self.new_parallel_layer(group, index, layout, stream),
            (None, Some(assignment)) => {
                self.new_expert_parallel_layer(group, index, assignment, stream)
            }
            (Some(layout), Some(assignment)) => {
                self.new_tensor_expert_parallel_layer(group, index, layout, assignment, stream)
            }
        }
    }

    /// Returns the checkpoint prefix for one runtime unit.
    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String;

    /// Returns the stable residency unit name for one runtime unit.
    fn layer_unit_name(&self, group: usize, index: usize) -> String;
    /// Populates one temporary execution unit from its protected lease.
    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        Ok(populate_module_from_lease(layer, lease)?)
    }

    /// Builds direct or derived bindings for one runtime unit.
    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        Ok(build_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(group, index),
            store,
        )?)
    }

    /// Builds rank-local bindings for a tensor-parallel execution unit.
    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        build_parallel_module_bindings(
            layer,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    /// Builds rank-local bindings for an expert-parallel execution unit.
    ///
    /// The adapter owns checkpoint expert layout, packed companions, and the
    /// mapping from global expert ids to its layer representation.
    fn expert_parallel_layer_bindings(
        &self,
        _group: usize,
        _index: usize,
        _layer: &Self::Layer,
        _store: &dyn eredu_checkpoint::store::CheckpointSource,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        Err(Error::Parallel(format!(
            "architecture adapter {} has not implemented expert-local checkpoint bindings",
            std::any::type_name::<Self>()
        )))
    }

    /// Builds bindings for a layer that is simultaneously TP-sharded and
    /// restricted to EP-owned routed experts.
    ///
    /// The default composes the architecture's EP selection recipe with the
    /// shared semantic TP shard plan. Architectures only need to override this
    /// when their checkpoint representation requires a different ordering.
    #[allow(clippy::too_many_arguments)]
    fn tensor_expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings =
            self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)?;
        shard_layer_bindings(
            bindings,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    /// Builds bindings for the same replicated, TP-local, or EP-local unit
    /// geometry selected by [`Self::new_cartesian_layer`].
    #[allow(clippy::too_many_arguments)]
    fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        match (layout, assignment) {
            (None, None) => {
                // The execution layer can have transformed target geometry
                // (for example load-time affine quantization). Bindings must
                // continue to describe the adapter's source checkpoint
                // geometry and are transformed only during population.
                let source = self.new_layer(group, index, stream)?;
                self.layer_bindings(group, index, &source, store)
            }
            (Some(layout), None) => {
                self.parallel_layer_bindings(group, index, layer, store, layout, stream)
            }
            (None, Some(assignment)) => {
                self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)
            }
            (Some(layout), Some(assignment)) => self.tensor_expert_parallel_layer_bindings(
                group, index, layer, store, layout, assignment, stream,
            ),
        }
    }

    /// Returns checkpoint keys consumed by dependent units outside execution groups.
    fn additional_consumed_checkpoint_keys(
        &self,
        _store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Vec<String> {
        Vec::new()
    }

    /// Executes one populated unit while inspecting and mutating the complete cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Executes one populated unit with architecture-specific observation.
    ///
    /// The default provides stable unit boundary names. Adapters whose block
    /// math exposes richer observations override this hook without replacing
    /// residency, cache, or graph execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_with_observer<O: ActivationObserver>(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
        observer: &mut O,
    ) -> Result<Array, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        observer.observe(&format!("{prefix}.input"), hidden)?;
        let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output))
    }

    /// Executes one unit under an explicit replicated or TP context.
    ///
    /// The default preserves ordinary execution and rejects TP, ensuring a
    /// family cannot become distributed merely because it implements the
    /// resident adapter contract.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        if execution.is_tensor_parallel() {
            return Err(Error::Parallel(format!(
                "architecture adapter {} has not implemented tensor-parallel execution",
                std::any::type_name::<Self>()
            )));
        }
        self.forward_layer(
            group,
            index,
            layer,
            hidden,
            cache,
            context,
            execution.stream(),
        )
    }

    /// Returns every cache/state array that must be evaluated before lease release.
    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array>;

    /// Returns transient forward-context arrays that must be evaluated before lease release.
    fn retained_context_arrays<'a>(
        &self,
        _context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        Vec::new()
    }

    /// Selects or assembles the activation consumed by one ready group.
    ///
    /// Root groups receive `initial_hidden`. A group with one dependency uses
    /// that output by default. Multi-input groups must define an exact merge.
    #[allow(clippy::too_many_arguments)]
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        _cache: &mut Self::Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match dependency_outputs {
            [] => Ok(initial_hidden.clone()),
            [dependency] => Ok(dependency.clone()),
            _ => Err(LayerwiseModelError::UnmergedExecutionGroupInputs {
                group,
                inputs: dependency_outputs.len(),
            }
            .into()),
        }
    }

    /// Selects or assembles a ready group under an explicit execution context.
    #[allow(clippy::too_many_arguments)]
    fn begin_execution_group_with_execution(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        self.begin_execution_group(
            group,
            initial_hidden,
            dependency_outputs,
            cache,
            context,
            execution.stream(),
        )
    }

    /// Converts one group's output into the activation consumed by the next group.
    ///
    /// Multimodal adapters use this hook to merge encoded media before entering
    /// a text decoder. Homogeneous adapters keep the activation unchanged.
    fn complete_execution_group(
        &mut self,
        _group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(hidden.clone())
    }

    /// Converts group output under an explicit execution context.
    fn complete_execution_group_with_execution(
        &mut self,
        group: usize,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        self.complete_execution_group(group, hidden, cache, context, execution.stream())
    }

    /// Applies final normalization, projections, or family-specific output assembly.
    fn finish(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Produces final output under an explicit execution context.
    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        self.finish(hidden, cache, context, execution.stream())
    }

    /// Returns whether a checkpoint key is intentionally ignored by strict loading.
    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

/// Semantic adapter capability required by bounded load-time quantization.
///
/// Implementations rebuild the same architecture with packed matrix modules
/// while preserving architecture-owned execution choices such as multimodal
/// towers and externally resident experts. Checkpoint format is deliberately
/// absent from this contract: SafeTensors and dense GGUF use the same packed
/// overlay once their stores expose the adapter's semantic recipes.
pub(crate) trait LoadTimeQuantizableAdapter: ArchitectureAdapter {
    /// Rebuilds this adapter with `quantization` as its uniform packed matrix
    /// representation.
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error>;
}

/// Residency-owned execution engine for generalized adapters.
///
/// Group windows, lease lifetime, retained-state evaluation, stream
/// synchronization, and telemetry stay centralized here. Adapter code owns only
/// architecture math, cache validation, and runtime-unit construction.
pub struct LayerwiseModel<A: ArchitectureAdapter> {
    adapter: A,
    graph: ExecutionGraph,
    store: SharedCheckpointSource,
    residency: ResidencyManager,
    groups: Vec<ResidentLayerGroup>,
    static_leases: Vec<ResidentUnitLease>,
    resident_layers: Option<Vec<Vec<A::Layer>>>,
    // Keep every populated layer's source arrays protected for the model lifetime.
    _resident_layer_leases: Vec<Vec<ResidentUnitLease>>,
    dense_stream: Option<Arc<DenseStreamController>>,
    sample_mlx_memory: bool,
    sample_process_memory: bool,
    metadata: LayerwiseModelMetadata,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    execution_streams: Vec<Stream>,
    #[cfg(test)]
    force_serial_execution: bool,
    #[cfg(test)]
    last_ready_set_trace: ReadySetExecutionTrace,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ReadySetExecutionTrace {
    submissions: Vec<(usize, usize, i32)>,
    completion_boundaries: Vec<(usize, usize)>,
    independent_group_events: Vec<(usize, usize)>,
}

#[cfg(test)]
impl ReadySetExecutionTrace {
    pub(crate) fn submissions(&self) -> &[(usize, usize, i32)] {
        &self.submissions
    }

    pub(crate) fn completion_boundaries(&self) -> &[(usize, usize)] {
        &self.completion_boundaries
    }

    pub(crate) fn independent_group_events(&self) -> &[(usize, usize)] {
        &self.independent_group_events
    }
}

enum RetainedExecutionLayer<L> {
    Leased {
        _layer: L,
        _transfer: ResidentTransfer,
    },
    Transferred {
        _layer: L,
        _transfer: DensePreparedTransfer,
    },
}

struct GroupExecutionState<L> {
    stream: Stream,
    hidden: Option<Array>,
    next_layer: usize,
    completion: Option<Event>,
    retained_layer: Option<RetainedExecutionLayer<L>>,
    dense_window: Option<DenseTransferWindow>,
    dense_guard: Option<DenseStreamGroupGuard>,
    started: bool,
    ordered: bool,
    completed: bool,
    execute: bool,
}

impl<L> GroupExecutionState<L> {
    fn new(stream: Stream) -> Self {
        Self {
            stream,
            hidden: None,
            next_layer: 0,
            completion: None,
            retained_layer: None,
            dense_window: None,
            dense_guard: None,
            started: false,
            ordered: false,
            completed: false,
            execute: false,
        }
    }
}

impl<L> Drop for GroupExecutionState<L> {
    fn drop(&mut self) {
        // Normal completion clears this event before releasing retained state.
        // On failure or cancellation, drain independently submitted work on the
        // same host thread so its layers and transfer sources cannot unwind early.
        if let Some(completion) = &self.completion {
            let _ = completion.synchronize();
        }
    }
}

fn is_temporary_residency_contention(error: &Error) -> bool {
    matches!(
        error,
        Error::Residency(ResidencyError::Ledger(
            ResidencyLedgerError::BudgetExhausted { .. },
        )) | Error::LayerwiseModel(LayerwiseModelError::Residency(ResidencyError::Ledger(
            ResidencyLedgerError::BudgetExhausted { .. }
        )))
    )
}

impl<A: ArchitectureAdapter> LayerwiseModel<A> {
    /// Creates an engine from a validated residency manager and execution groups.
    pub fn new(
        adapter: A,
        graph: ExecutionGraph,
        store: SharedCheckpointSource,
        residency: ResidencyManager,
        groups: Vec<ResidentLayerGroup>,
        static_leases: Vec<ResidentUnitLease>,
    ) -> Result<Self, Error> {
        if groups.len() != graph.groups().len() {
            return Err(LayerwiseModelError::ExecutionGroupCount {
                adapter: graph.groups().len(),
                configured: groups.len(),
            }
            .into());
        }
        for (group_index, group) in groups.iter().enumerate() {
            let expected_id = graph.groups()[group_index].id();
            if group.id() != expected_id {
                return Err(LayerwiseModelError::ExecutionGroupIdentity {
                    slot: group_index,
                    adapter: expected_id.to_string(),
                    configured: group.id().to_string(),
                }
                .into());
            }
            let expected = adapter.layer_count(group_index)?;
            if expected != group.units().len() {
                return Err(LayerwiseModelError::ExecutionGroupLength {
                    group: group.id().to_string(),
                    adapter: expected,
                    configured: group.units().len(),
                }
                .into());
            }
        }
        let layer_count = groups.iter().map(|group| group.units().len()).sum();
        let metadata = LayerwiseModelMetadata::new(
            adapter.model_type(),
            adapter.quantization(),
            layer_count,
            0,
            ExecutionResidency::LayerwiseHost,
            0,
            0,
            0,
            0,
        );
        Ok(Self {
            adapter,
            graph,
            store,
            residency,
            groups,
            static_leases,
            resident_layers: None,
            _resident_layer_leases: Vec::new(),
            dense_stream: None,
            sample_mlx_memory: false,
            sample_process_memory: false,
            metadata,
            parallel_layout: None,
            parallel_topology: None,
            parallel_info: None,
            execution_streams: Vec::new(),
            #[cfg(test)]
            force_serial_execution: false,
            #[cfg(test)]
            last_ready_set_trace: ReadySetExecutionTrace::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn force_serial_reference(&mut self, force: bool) {
        self.force_serial_execution = force;
    }

    #[cfg(test)]
    pub(crate) fn ready_set_trace(&self) -> &ReadySetExecutionTrace {
        &self.last_ready_set_trace
    }

    fn materialize_resident_layers(&mut self, stream: &Stream) -> Result<(), Error> {
        let mut resident_layers = Vec::with_capacity(self.groups.len());
        let mut resident_layer_leases = Vec::with_capacity(self.groups.len());
        for (group_index, group) in self.groups.iter().enumerate() {
            let mut layers = Vec::with_capacity(group.units().len());
            let mut leases = Vec::with_capacity(group.units().len());
            for (index, id) in group.units().iter().enumerate() {
                let mut layer = if let Some(layout) = &self.parallel_layout {
                    self.adapter
                        .new_parallel_layer(group_index, index, layout, stream)?
                } else {
                    self.adapter.new_layer(group_index, index, stream)?
                };
                let lease = self.residency.acquire(id, MemoryTier::Device)?;
                self.adapter
                    .populate_layer(group_index, index, &mut layer, &lease)?;
                layers.push(layer);
                leases.push(lease);
            }
            resident_layers.push(layers);
            resident_layer_leases.push(leases);
        }
        self.resident_layers = Some(resident_layers);
        self._resident_layer_leases = resident_layer_leases;
        Ok(())
    }

    fn group_execution_streams(&mut self, stream: &Stream) -> Result<Vec<Stream>, Error> {
        #[cfg(test)]
        if self.force_serial_execution {
            return Ok((0..self.graph.groups().len())
                .map(|_| stream.clone())
                .collect());
        }
        if self.graph.groups().len() == 1 {
            return Ok(vec![stream.clone()]);
        }
        if self.execution_streams.is_empty() {
            let device = stream.get_device()?;
            self.execution_streams = (0..self.graph.groups().len())
                .map(|_| Stream::new_with_device(&device))
                .collect();
        }
        Ok(self.execution_streams.clone())
    }

    /// Enables optional allocator and process-memory samples after forward.
    pub fn with_memory_sampling(mut self, mlx: bool, process: bool) -> Self {
        self.sample_mlx_memory = mlx;
        self.sample_process_memory = process;
        self
    }

    /// Returns the architecture adapter.
    pub const fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Returns aggregate residency metadata for all execution groups.
    pub const fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns rank-local parallel placement information when loaded with a
    /// generalized distributed execution group.
    pub const fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Returns the exact typed rank-local parameter layout, when parallel.
    pub const fn parallel_layout(&self) -> Option<&eredu_runtime::LocalModelLayout> {
        self.parallel_layout.as_ref()
    }

    /// Returns the cache-relevant architecture fingerprint for this execution rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        Ok(self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?
            .architecture_fingerprint
            .clone())
    }

    /// Returns the complete cache identity, including every active parallel
    /// coordinate and the rank-local cache layout.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.adapter
            .prompt_cache_model_identity(self.parallel_topology)
    }

    /// Returns this execution rank's exact ordered cache-state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::core::attention::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?
            .layer_layout
            .clone())
    }

    /// Persists a compatible architecture-owned prefix cache. Parallel ranks
    /// publish into deterministic subdirectories below the supplied root.
    pub fn save_prompt_cache(
        &self,
        cache: &mut A::Cache,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let directory = self.prompt_cache_directory(root.as_ref());
        self.adapter.save_prompt_cache(
            cache,
            &directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    pub(crate) fn save_prompt_cache_with_validated_identity(
        &self,
        cache: &mut A::Cache,
        directory: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.adapter.save_prompt_cache(
            cache,
            directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible architecture-owned prefix cache from this rank's
    /// deterministic cache directory.
    pub fn load_prompt_cache(
        &self,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(A::Cache, PromptCacheManifest), Error> {
        let identity = self
            .adapter
            .prompt_cache_model_identity(self.parallel_topology)?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.adapter.load_prompt_cache(
            &self.prompt_cache_directory(root.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_validated_identity(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(A::Cache, PromptCacheManifest), Error> {
        self.adapter.load_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options,
            stream,
        )
    }

    fn prompt_cache_directory(&self, root: &Path) -> std::path::PathBuf {
        match self.parallel_topology {
            Some(topology) => root.join(format!("rank-{:05}", topology.global_rank)),
            None => root.to_path_buf(),
        }
    }

    pub(crate) fn prompt_cache_rank_identity(
        &self,
    ) -> Option<crate::core::cache::CacheRankIdentity> {
        self.parallel_topology
            .map(crate::backend::mlx::cache::prompt_cache_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    /// Binds state and persistence identity to an enclosing Cartesian runtime.
    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.parallel_topology = Some(topology);
    }

    /// Returns the mutable adapter for loader-time dependent-unit setup.
    pub(crate) fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// Returns a shared handle to the persistent checkpoint store.
    pub(crate) fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        Arc::clone(&self.store)
    }

    /// Returns the persistent backend-neutral checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn eredu_checkpoint::store::CheckpointSource) {
        self.store.as_ref()
    }

    /// Returns named execution groups in deterministic order.
    pub fn execution_groups(&self) -> &[ResidentLayerGroup] {
        &self.groups
    }

    /// Returns the validated dependency graph governing group execution.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    /// Returns the reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.residency
    }

    /// Returns a current residency and transfer report.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        Ok(self
            .residency
            .report()?
            .with_materialization(self.metadata.materialization().cloned()))
    }

    /// Returns dense-stream observations when that experimental policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense_stream
            .as_ref()
            .map(|streamer| streamer.report(&self.residency))
            .transpose()
    }

    /// Runs every graph-ready group while centrally enforcing lease safety.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_with_hooks(
            input,
            cache,
            stream,
            true,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                let execution =
                    crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(
                        stream,
                    );
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, &execution,
                )
            },
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs a rank-local layerwise model using the tensor-parallel subgroup.
    pub(crate) fn forward_tensor_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            true,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, execution,
                )
            },
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs TP execution groups and invokes a context hook after each unit.
    pub(crate) fn forward_tensor_parallel_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            false,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, execution,
                )
            },
            hook,
        )
    }

    /// Runs TP execution while allowing routed-expert evaluation to be replaced.
    ///
    /// The embedding, attention, dense/shared projections, cache geometry, and
    /// output head retain their tensor-parallel execution context. The caller
    /// replaces only the selected populated-layer operation and receives the
    /// same TP context, enabling an EP exchange inside a TP-sharded layer.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_tensor_parallel_with_layer_executor<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        executor: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
    {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            false,
            executor,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs TP execution with the architecture's ordinary layer semantics and
    /// returns the forward context retained by the pass.
    ///
    /// Embedded prediction heads consume the final decoder hidden state, so
    /// distributed callers need the same context that replicated execution
    /// exposes without replacing any layer operation.
    pub(crate) fn forward_tensor_parallel_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<(Array, A::ForwardContext), Error> {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            true,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, execution,
                )
            },
            |_, _, _| Ok(()),
        )
    }

    /// Runs TP execution with a caller-provided layer operation and returns
    /// the architecture context retained by the pass. This is the TP+EP
    /// counterpart of [`Self::forward_with_layer_executor_and_context`].
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_tensor_parallel_with_layer_executor_and_context<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        executor: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
    {
        self.forward_tensor_parallel_with_hooks(
            input,
            cache,
            group,
            stream,
            false,
            executor,
            |_, _, _| Ok(()),
        )
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::needless_range_loop
    )]
    fn forward_ready_set_with_hooks<'a, F, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        tensor_parallel_group: Option<&safemlx::distributed::Group>,
        stream: &Stream,
        batch_resident_groups: bool,
        mut executor: F,
        mut hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        #[cfg(test)]
        {
            self.last_ready_set_trace = ReadySetExecutionTrace::default();
        }
        let topology = match tensor_parallel_group {
            Some(_) => Some(self.parallel_topology.ok_or_else(|| {
                Error::Parallel(
                    "layerwise model was not loaded for tensor-parallel execution".into(),
                )
            })?),
            None => None,
        };
        let initial_execution = match (topology, tensor_parallel_group) {
            (Some(topology), Some(group)) => {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )?
            }
            (None, None) => {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(stream)
            }
            _ => unreachable!("topology and TP group are created together"),
        };
        self.adapter.validate_cache(cache)?;
        let eredu_runtime::LayeredForwardState {
            hidden: initial_hidden,
            mut context,
        } = self
            .adapter
            .begin_forward_with_execution(input, cache, &initial_execution)?;
        let prefill = initial_hidden.dim(1) > 1;
        let dense_forward = self
            .dense_stream
            .as_ref()
            .map(|streamer| streamer.forward_guard(prefill, &self.residency))
            .transpose()?;

        let mut initial_roots = vec![&initial_hidden];
        for group_index in 0..self.graph.groups().len() {
            initial_roots.extend(
                self.adapter
                    .retained_context_arrays(&context, group_index, 0),
            );
        }
        let initial_completion = async_eval_with_event(initial_roots)?;
        let mut states = self
            .group_execution_streams(stream)?
            .into_iter()
            .map(GroupExecutionState::new)
            .collect::<Vec<GroupExecutionState<A::Layer>>>();
        let batch_resident_groups = batch_resident_groups && self.resident_layers.is_some();
        let mut schedule = ExecutionGroupSchedule::new(&self.graph);
        let mut group_outputs: Vec<Option<Array>> = vec![None; self.graph.groups().len()];

        while states.iter().any(|state| !state.completed) {
            let mut progressed = false;

            // Completion is polled in stable group order. Resources are released
            // immediately after the exact unit/group event completes.
            for group_index in 0..states.len() {
                let complete = match states[group_index].completion.as_ref() {
                    Some(event) => event.is_complete()?,
                    None => false,
                };
                if !complete {
                    continue;
                }
                states[group_index].completion = None;
                states[group_index].retained_layer = None;
                if states[group_index].ordered {
                    if states[group_index].execute
                        && self.resident_layers.is_none()
                        && states[group_index].dense_window.is_none()
                    {
                        let resident_group = &self.groups[group_index];
                        let completed = states[group_index].next_layer - 1;
                        let end = completed
                            .saturating_add(resident_group.depth())
                            .min(resident_group.units().len());
                        resident_group
                            .trim_to(&self.residency, &resident_group.units()[completed..end])?;
                    }
                    if let Some(guard) = states[group_index].dense_guard.take() {
                        guard.complete()?;
                    }
                    states[group_index].dense_window = None;
                    states[group_index].completed = true;
                } else if let Some(window) = &mut states[group_index].dense_window {
                    match window.refill() {
                        Ok(()) => {}
                        Err(error) if is_temporary_residency_contention(&error) => {}
                        Err(error) => return Err(error),
                    }
                } else if self.resident_layers.is_none() {
                    let resident_group = &self.groups[group_index];
                    let completed = states[group_index].next_layer - 1;
                    let end = completed
                        .saturating_add(resident_group.depth())
                        .min(resident_group.units().len());
                    resident_group
                        .trim_to(&self.residency, &resident_group.units()[completed..end])?;
                }
                progressed = true;
            }

            // Record waits for every newly ready group before constructing any
            // consumer graph. Dependency output order is the adapter declaration
            // order, independent of producer completion order.
            let newly_ready = schedule.startable_groups().collect::<Vec<_>>();
            for group_index in newly_ready {
                if states[group_index].started {
                    continue;
                }
                let dependency_slots = schedule
                    .dependencies(group_index)
                    .expect("execution schedule uses the validated graph")
                    .to_vec();
                if dependency_slots.is_empty() {
                    initial_completion.wait_on(&states[group_index].stream)?;
                } else {
                    for &dependency in &dependency_slots {
                        if let Some(completion) = &states[dependency].completion {
                            completion.wait_on(&states[group_index].stream)?;
                        }
                    }
                }
                let dependency_outputs = dependency_slots
                    .iter()
                    .map(|&dependency| {
                        group_outputs[dependency]
                            .as_ref()
                            .expect("ready dependency has an ordered output")
                            .clone()
                    })
                    .collect::<Vec<_>>();
                let execution = match (topology, tensor_parallel_group) {
                    (Some(topology), Some(group)) => crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                        topology,
                        group,
                        &states[group_index].stream,
                    )?,
                    (None, None) => crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(&states[group_index].stream),
                    _ => unreachable!("topology and TP group are created together"),
                };
                let hidden = match self.adapter.begin_execution_group_with_execution(
                    group_index,
                    &initial_hidden,
                    &dependency_outputs,
                    cache,
                    &mut context,
                    &execution,
                ) {
                    Ok(hidden) => hidden,
                    Err(error) => {
                        schedule
                            .fail(group_index)
                            .expect("ready group belongs to the validated schedule");
                        return Err(error);
                    }
                };
                states[group_index].hidden = Some(hidden);
                states[group_index].started = true;
                states[group_index].execute =
                    self.adapter.should_execute_group(group_index, &context);
                for dependency in schedule
                    .started(group_index)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                {
                    group_outputs[dependency] = None;
                }
                progressed = true;
            }

            for group_index in 0..states.len() {
                if !states[group_index].started
                    || states[group_index].ordered
                    || states[group_index].completion.is_some()
                {
                    continue;
                }
                let resident_group = &self.groups[group_index];
                let layer_count = resident_group.units().len();
                let group_stream = states[group_index].stream.clone();
                let execution = match (topology, tensor_parallel_group) {
                    (Some(topology), Some(group)) => crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                        topology,
                        group,
                        &group_stream,
                    )?,
                    (None, None) => crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(&group_stream),
                    _ => unreachable!("topology and TP group are created together"),
                };

                if !states[group_index].execute {
                    let hidden = self.adapter.complete_execution_group_with_execution(
                        group_index,
                        states[group_index]
                            .hidden
                            .as_ref()
                            .expect("started group has hidden state"),
                        cache,
                        &mut context,
                        &execution,
                    )?;
                    let retained_context =
                        self.adapter
                            .retained_context_arrays(&context, group_index, layer_count);
                    let completion =
                        async_eval_with_event(std::iter::once(&hidden).chain(retained_context))?;
                    #[cfg(test)]
                    self.last_ready_set_trace
                        .completion_boundaries
                        .push((group_index, layer_count));
                    states[group_index].hidden = Some(hidden.clone());
                    states[group_index].completion = Some(completion);
                    states[group_index].ordered = true;
                    group_outputs[group_index] = Some(hidden);
                    schedule
                        .ordered(group_index)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    progressed = true;
                    continue;
                }

                if states[group_index].dense_guard.is_none() {
                    states[group_index].dense_guard = self
                        .dense_stream
                        .as_ref()
                        .map(|streamer| streamer.group_guard(&self.residency, resident_group.id()));
                }
                if states[group_index].dense_window.is_none() {
                    let result = self.dense_stream.as_ref().map(|streamer| {
                        streamer.transfer_window(
                            &self.residency,
                            resident_group.id(),
                            resident_group.units(),
                            states[group_index].next_layer..layer_count,
                            prefill,
                        )
                    });
                    match result {
                        Some(Ok(window)) => states[group_index].dense_window = Some(window),
                        Some(Err(error)) if is_temporary_residency_contention(&error) => continue,
                        Some(Err(error)) => return Err(error),
                        None => {}
                    }
                }

                if batch_resident_groups {
                    let mut hidden = states[group_index]
                        .hidden
                        .take()
                        .expect("started group has hidden state");
                    let start = states[group_index].next_layer;
                    for index in start..layer_count {
                        hidden = executor(
                            &mut self.adapter,
                            group_index,
                            index,
                            &mut self
                                .resident_layers
                                .as_mut()
                                .expect("resident batching requires materialized layers")
                                [group_index][index],
                            &hidden,
                            cache,
                            &mut context,
                            &execution,
                        )?;
                        hook(group_index, index, &mut context)?;
                        #[cfg(test)]
                        self.last_ready_set_trace.submissions.push((
                            group_index,
                            index,
                            group_stream.get_index()?,
                        ));
                    }
                    states[group_index].next_layer = layer_count;
                    hidden = self.adapter.complete_execution_group_with_execution(
                        group_index,
                        &hidden,
                        cache,
                        &mut context,
                        &execution,
                    )?;
                    let retained = (0..layer_count)
                        .flat_map(|index| self.adapter.retained_arrays(cache, group_index, index))
                        .collect::<Vec<_>>();
                    let retained_context =
                        self.adapter
                            .retained_context_arrays(&context, group_index, layer_count);
                    let completion = async_eval_with_event(
                        std::iter::once(&hidden)
                            .chain(retained)
                            .chain(retained_context),
                    )?;
                    #[cfg(test)]
                    {
                        self.last_ready_set_trace
                            .completion_boundaries
                            .push((group_index, layer_count));
                        let stream_index = group_stream.get_index()?;
                        for (other_group, other) in states.iter().enumerate() {
                            let distinct_event = other_group != group_index
                                && other.stream.get_index()? != stream_index
                                && other.completion.is_some();
                            if distinct_event {
                                self.last_ready_set_trace
                                    .independent_group_events
                                    .push((other_group, group_index));
                            }
                        }
                    }
                    states[group_index].hidden = Some(hidden.clone());
                    states[group_index].completion = Some(completion);
                    states[group_index].ordered = true;
                    // Resident parameters need no completion-gated lease release.
                    // Keep the event only as an on-device dependency for consumers;
                    // the final output synchronization covers the whole graph.
                    states[group_index].completed = true;
                    group_outputs[group_index] = Some(hidden);
                    schedule
                        .ordered(group_index)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                    progressed = true;
                    continue;
                }

                let index = states[group_index].next_layer;
                let mut retained_layer = None;
                let hidden = states[group_index]
                    .hidden
                    .as_ref()
                    .expect("started group has hidden state")
                    .clone();
                let layer_output = if let Some(layers) = &mut self.resident_layers {
                    executor(
                        &mut self.adapter,
                        group_index,
                        index,
                        &mut layers[group_index][index],
                        &hidden,
                        cache,
                        &mut context,
                        &execution,
                    )?
                } else if let Some(window) = &mut states[group_index].dense_window {
                    if !window.has_ready() {
                        match window.refill() {
                            Ok(()) => {}
                            Err(error) if is_temporary_residency_contention(&error) => continue,
                            Err(error) => return Err(error),
                        }
                    }
                    if window.is_exhausted() {
                        return Err(LayerwiseModelError::InvalidDenseTransferWindow {
                            index,
                            unit_count: layer_count,
                        }
                        .into());
                    }
                    let transfer = window.next(&group_stream)?;
                    debug_assert_eq!(transfer.index(), index);
                    let mut layer = match self.parallel_layout.as_ref() {
                        Some(layout) if tensor_parallel_group.is_some() => self
                            .adapter
                            .new_parallel_layer(group_index, index, layout, &group_stream)?,
                        _ => self.adapter.new_layer(group_index, index, &group_stream)?,
                    };
                    self.adapter.populate_layer(
                        group_index,
                        index,
                        &mut layer,
                        transfer.lease(),
                    )?;
                    let output = executor(
                        &mut self.adapter,
                        group_index,
                        index,
                        &mut layer,
                        &hidden,
                        cache,
                        &mut context,
                        &execution,
                    )?;
                    retained_layer = Some(RetainedExecutionLayer::Transferred {
                        _layer: layer,
                        _transfer: transfer,
                    });
                    output
                } else {
                    let end = index
                        .saturating_add(resident_group.depth())
                        .min(layer_count);
                    let requests = resident_group.units()[index..end]
                        .iter()
                        .cloned()
                        .map(|id| (id, 1))
                        .collect::<Vec<_>>();
                    let transfer = match self
                        .residency
                        .acquire_many_with_transfer(&requests, MemoryTier::Device)
                    {
                        Ok(transfer) => transfer,
                        Err(
                            error @ ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                                ..
                            }),
                        ) => {
                            let error = Error::Residency(error);
                            if states.iter().any(|state| state.completion.is_some()) {
                                continue;
                            }
                            return Err(error);
                        }
                        Err(error) => return Err(error.into()),
                    };
                    transfer.order_after(&group_stream)?;
                    let mut layer = match self.parallel_layout.as_ref() {
                        Some(layout) if tensor_parallel_group.is_some() => self
                            .adapter
                            .new_parallel_layer(group_index, index, layout, &group_stream)?,
                        _ => self.adapter.new_layer(group_index, index, &group_stream)?,
                    };
                    self.adapter.populate_layer(
                        group_index,
                        index,
                        &mut layer,
                        &transfer.leases()[0],
                    )?;
                    let output = executor(
                        &mut self.adapter,
                        group_index,
                        index,
                        &mut layer,
                        &hidden,
                        cache,
                        &mut context,
                        &execution,
                    )?;
                    retained_layer = Some(RetainedExecutionLayer::Leased {
                        _layer: layer,
                        _transfer: transfer,
                    });
                    output
                };
                hook(group_index, index, &mut context)?;
                states[group_index].next_layer += 1;
                let final_layer = states[group_index].next_layer == layer_count;
                let hidden = if final_layer {
                    self.adapter.complete_execution_group_with_execution(
                        group_index,
                        &layer_output,
                        cache,
                        &mut context,
                        &execution,
                    )?
                } else {
                    layer_output
                };
                let retained = self.adapter.retained_arrays(cache, group_index, index);
                let retained_context = self.adapter.retained_context_arrays(
                    &context,
                    group_index,
                    states[group_index].next_layer,
                );
                let completion = async_eval_with_event(
                    std::iter::once(&hidden)
                        .chain(retained)
                        .chain(retained_context),
                )?;
                #[cfg(test)]
                {
                    self.last_ready_set_trace
                        .completion_boundaries
                        .push((group_index, states[group_index].next_layer));
                    let stream_index = group_stream.get_index()?;
                    self.last_ready_set_trace
                        .submissions
                        .push((group_index, index, stream_index));
                    for (other_group, other) in states.iter().enumerate() {
                        let distinct_event = other_group != group_index
                            && other.stream.get_index()? != stream_index
                            && other.completion.is_some();
                        if distinct_event {
                            self.last_ready_set_trace
                                .independent_group_events
                                .push((other_group, group_index));
                        }
                    }
                }
                states[group_index].hidden = Some(hidden.clone());
                states[group_index].completion = Some(completion);
                states[group_index].retained_layer = retained_layer;
                if final_layer {
                    states[group_index].ordered = true;
                    group_outputs[group_index] = Some(hidden);
                    schedule
                        .ordered(group_index)
                        .map_err(|error| Error::Parallel(error.to_string()))?;
                }
                progressed = true;
            }

            if !progressed {
                let Some(event) = states.iter().find_map(|state| state.completion.as_ref()) else {
                    return Err(Error::Parallel(
                        "execution-group ready set made no progress without in-flight work".into(),
                    ));
                };
                event.synchronize()?;
            }
        }

        let hidden = group_outputs[self.graph.output()]
            .take()
            .expect("validated execution graph output was ordered");
        if let Some(completion) = states[self.graph.output()].completion.as_ref() {
            completion.wait_on(stream)?;
        }
        let final_execution = match (topology, tensor_parallel_group) {
            (Some(topology), Some(group)) => {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )?
            }
            (None, None) => {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(stream)
            }
            _ => unreachable!("topology and TP group are created together"),
        };
        let output =
            self.adapter
                .finish_with_execution(&hidden, cache, &context, &final_execution)?;
        async_eval_with_event([&output])?.synchronize()?;
        if self.dense_stream.is_none() && (self.sample_mlx_memory || self.sample_process_memory) {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        if let Some(guard) = dense_forward {
            guard.complete()?;
        }
        Ok((output, context))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn forward_tensor_parallel_with_hooks<'a, F, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        batch_resident_groups: bool,
        executor: F,
        hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        ) -> Result<Array, Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_ready_set_with_hooks(
            input,
            cache,
            Some(group),
            stream,
            batch_resident_groups,
            executor,
            hook,
        )
    }

    /// Runs a generalized forward pass while allowing the caller to replace
    /// execution of each populated layer.
    ///
    /// Residency, prefetch, lease lifetime, retained-array evaluation, and
    /// telemetry remain owned by this engine. Distributed execution uses this
    /// hook to replace only routed-expert evaluation while reusing the same
    /// architecture adapter and checkpoint bindings.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_with_layer_executor<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        executor: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
    {
        self.forward_with_hooks(input, cache, stream, false, executor, |_, _, _| Ok(()))
            .map(|(output, _)| output)
    }

    /// Runs the canonical execution path while exposing stable per-unit inputs
    /// and outputs to an activation observer.
    ///
    /// Observation is deliberately owned by the shared engine so fully
    /// resident, host-layerwise, and disk-streamed policies report identical
    /// names and intervention points.
    pub fn forward_with_observer<'a>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        observer: &mut dyn ActivationObserver,
    ) -> Result<Array, Error> {
        let mut observer = ActivationObserverProxy(observer);
        let output = self.forward_with_layer_executor(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                adapter.forward_layer_with_observer(
                    group,
                    index,
                    layer,
                    hidden,
                    cache,
                    context,
                    stream,
                    &mut observer,
                )
            },
        )?;
        observer.observe("model.logits", &output)?;
        Ok(output)
    }

    /// Runs a generalized pass with caller-provided populated-layer execution
    /// and returns the architecture context retained by that pass.
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_with_layer_executor_and_context<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        executor: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
    {
        self.forward_with_hooks(input, cache, stream, false, executor, |_, _, _| Ok(()))
    }

    /// Runs a generalized forward pass and invokes `hook` after each execution unit.
    ///
    /// Realtime autoregressive subgroups use this to turn one unit's logits into
    /// the token consumed by the next unit without moving lease ownership out of
    /// the shared residency engine.
    pub(crate) fn forward_with_context_hook<'a, F>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        hook: F,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_with_hooks(
            input,
            cache,
            stream,
            false,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                let execution =
                    crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::replicated(
                        stream,
                    );
                adapter.forward_layer_with_execution(
                    group, index, layer, hidden, cache, context, &execution,
                )
            },
            hook,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn forward_with_hooks<'a, F, H>(
        &mut self,
        input: A::Input<'a>,
        cache: &mut A::Cache,
        stream: &Stream,
        batch_resident_groups: bool,
        mut executor: F,
        hook: H,
    ) -> Result<(Array, A::ForwardContext), Error>
    where
        F: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Layer,
            &Array,
            &mut A::Cache,
            &mut A::ForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), Error>,
    {
        self.forward_ready_set_with_hooks(
            input,
            cache,
            None,
            stream,
            batch_resident_groups,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                executor(
                    adapter,
                    group,
                    index,
                    layer,
                    hidden,
                    cache,
                    context,
                    execution.stream(),
                )
            },
            hook,
        )
    }

    /// Clears one named execution group without affecting other groups.
    pub fn clear_device_group(&self, id: &str) -> Result<(), Error> {
        let group = self
            .groups
            .iter()
            .find(|group| group.id() == id)
            .ok_or_else(|| LayerwiseModelError::UnknownExecutionGroup(id.to_string()))?;
        if self.resident_layers.is_some() {
            return Ok(());
        }
        Ok(group.clear(&self.residency)?)
    }

    /// Clears every temporary device execution group.
    pub fn clear_all_device_groups(&self) -> Result<(), Error> {
        if self.resident_layers.is_some() {
            return Ok(());
        }
        for group in &self.groups {
            group.clear(&self.residency)?;
        }
        Ok(())
    }

    /// Returns the number of pinned static leases held by the engine.
    pub fn static_lease_count(&self) -> usize {
        self.static_leases.len()
    }
}

/// Builds a generalized layerwise model with independently bounded groups.
#[cfg(test)]
pub(crate) fn load_safetensors_layerwise_model<A, O>(
    model_dir: impl AsRef<Path>,
    adapter: A,
    options: O,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let store = open_safetensors_weight_store(model_dir.as_ref(), options.max_mapped_shards())?;
    load_layerwise_model(store, adapter, options, stream, weights_stream)
}

/// Builds a packed, disk-backed overlay for every quantizable static and
/// execution-unit binding declared by `source_adapter`, then loads the
/// quantized target adapter through the ordinary residency engine.
///
/// The source adapter is used only for semantic checkpoint recipes. No dense
/// module is populated. The target adapter is therefore free to expose packed
/// parameter trees, while every residency budget is computed from the packed
/// store metadata seen by [`load_layerwise_model`].
fn packed_weight_companion_dtypes(module: &impl ModuleParameters) -> BTreeMap<String, RecipeDtype> {
    let parameters = module.parameters().flatten();
    parameters
        .iter()
        .filter(|(_, parameter)| parameter.dtype() == safemlx::Dtype::Uint32)
        .map(|(name, _)| {
            let canonical =
                crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(name);
            let scales = canonical
                .strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.scales"))
                .unwrap_or_else(|| format!("{canonical}_scales"));
            let dtype = parameters
                .get(scales.as_str())
                .map(|parameter| {
                    crate::backend::mlx::runtime::checkpoint::recipe::recipe_dtype_from_mlx(
                        parameter.dtype(),
                    )
                })
                .unwrap_or(RecipeDtype::F32);
            (canonical, dtype)
        })
        .collect()
}

fn quantize_layerwise_store<A>(
    store: SharedCheckpointSource,
    source_adapter: &A,
    target_adapter: &A,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    A: ArchitectureAdapter,
{
    let mut recipes = BTreeMap::new();
    let mut collect =
        |bindings: &[WeightBinding],
         selected_local_weights: Option<&BTreeMap<String, RecipeDtype>>| {
            for binding in bindings {
                let recipe = binding.source_recipe();
                let metadata = recipe.infer(store.as_ref())?;
                if !matches!(
                    metadata.dtype(),
                    RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
                ) || metadata.shape().len() < 2
                {
                    continue;
                }
                let canonical_local =
                    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                        binding.name(),
                    );
                if selected_local_weights
                    .is_some_and(|selected| !selected.contains_key(&canonical_local))
                {
                    continue;
                }
                let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
                let companion_dtype = selected_local_weights
                    .and_then(|selected| selected.get(&canonical_local))
                    .cloned()
                    .unwrap_or(RecipeDtype::F32);
                match recipes.entry(target.to_string()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((recipe, companion_dtype));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() != &(recipe, companion_dtype) =>
                    {
                        return Err(Error::Quantization(format!(
                        "load-time quantization target {target:?} has conflicting semantic recipes"
                    )));
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
            Ok::<(), Error>(())
        };
    for unit in source_adapter.static_units(store.as_ref())? {
        let selected = unit
            .bindings()
            .iter()
            .filter(|binding| target_adapter.quantizes_static_binding(binding))
            .map(|binding| {
                (
                    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                        binding.name(),
                    ),
                    RecipeDtype::F32,
                )
            })
            .collect::<BTreeMap<_, _>>();
        collect(unit.bindings(), Some(&selected))?;
    }
    let graph = source_adapter.execution_graph()?;
    for group in 0..graph.groups().len() {
        for index in 0..source_adapter.layer_count(group)? {
            let layer = source_adapter.new_layer(group, index, stream)?;
            let target_layer = target_adapter.new_layer(group, index, stream)?;
            let selected = packed_weight_companion_dtypes(&target_layer);
            collect(
                &source_adapter.layer_bindings(group, index, &layer, store.as_ref())?,
                Some(&selected),
            )?;
        }
    }
    if recipes.is_empty() {
        return Err(Error::Quantization(format!(
            "architecture adapter {} declared no floating matrix bindings for load-time quantization",
            source_adapter.model_type()
        )));
    }
    let targets = recipes
        .into_iter()
        .map(|(target, (recipe, companion_dtype))| {
            let target = BoundedQuantizationTarget::from_recipe(target, recipe)?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companion_dtype)
                }
                WeightQuantization::MxFp4 => Ok(target),
                WeightQuantization::GgufIQuant { .. } => unreachable!(
                    "load-time materialization rejects checkpoint-native GGUF encodings"
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        Arc::clone(&store),
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        stream,
    )?);
    let report = transformed.report().clone();
    let transformed: SharedCheckpointSource = transformed;
    Ok((transformed, report))
}

/// Builds the shared bounded packed overlay from neutral parameter topologies.
///
/// Architecture composition supplies unloaded source and target modules; this
/// backend capability only inspects their stable parameter slots and recipes.
pub(crate) fn quantize_parameterized_store<SM, U, SF, TF>(
    store: SharedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    mut source_unit: SF,
    mut target_unit: TF,
    unit_count: usize,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: Clone + eredu_nn::Parameterized<Array>,
    U: eredu_nn::Parameterized<Array>,
    SF: FnMut(usize, &Stream) -> Result<U, Error>,
    TF: FnMut(usize, &Stream) -> Result<U, Error>,
{
    let mut recipes = BTreeMap::new();
    let mut collect = |bindings: &[WeightBinding], selected: &BTreeMap<String, RecipeDtype>| {
        for binding in bindings {
            let recipe = binding.source_recipe();
            let metadata = recipe.infer(store.as_ref())?;
            if !matches!(
                metadata.dtype(),
                RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
            ) || metadata.shape().len() < 2
            {
                continue;
            }
            let canonical =
                crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                    binding.name(),
                );
            let Some(companion_dtype) = selected.get(&canonical).cloned() else {
                continue;
            };
            let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
            match recipes.entry(target.to_string()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((recipe, companion_dtype));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() != &(recipe, companion_dtype) =>
                {
                    return Err(Error::Quantization(format!(
                        "load-time quantization target {target:?} has conflicting semantic recipes"
                    )));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
        Ok::<(), Error>(())
    };

    let source_static = crate::backend::mlx::nn::shared::MlxModule::new(source_static.clone());
    let target_static = crate::backend::mlx::nn::shared::MlxModule::new(target_static.clone());
    collect(
        &build_module_bindings(&source_static, "", store.as_ref())?,
        &packed_weight_companion_dtypes(&target_static),
    )?;
    for index in 0..unit_count {
        let source = crate::backend::mlx::nn::shared::MlxModule::new(source_unit(index, stream)?);
        let target = crate::backend::mlx::nn::shared::MlxModule::new(target_unit(index, stream)?);
        collect(
            &build_module_bindings(&source, "", store.as_ref())?,
            &packed_weight_companion_dtypes(&target),
        )?;
    }
    if recipes.is_empty() {
        return Err(Error::Quantization(
            "neutral parameter topology declared no floating matrix bindings for load-time quantization"
                .into(),
        ));
    }
    let targets = recipes
        .into_iter()
        .map(|(target, (recipe, companion_dtype))| {
            let target = BoundedQuantizationTarget::from_recipe(target, recipe)?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companion_dtype)
                }
                WeightQuantization::MxFp4 => Ok(target),
                WeightQuantization::GgufIQuant { .. } => unreachable!(
                    "load-time materialization rejects checkpoint-native GGUF encodings"
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        store,
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        stream,
    )?);
    let report = transformed.report().clone();
    let transformed: SharedCheckpointSource = transformed;
    Ok((transformed, report))
}

/// Loads an adapter directly or through the shared bounded packed overlay.
///
/// This is the authoritative standalone materialization route for both
/// SafeTensors and dense GGUF stores. Architecture code supplies semantic
/// bindings through the adapter; residency sees only the resulting packed
/// store and therefore budgets packed bytes rather than dense source bytes.
pub(crate) fn load_layerwise_model_with_quantization<A, O>(
    store: SharedCheckpointSource,
    source_adapter: A,
    options: O,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: LoadTimeQuantizableAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let store = resolve_checkpoint_store(store, &source_adapter)?;
    match quantization {
        Some(quantization) => {
            let target_adapter = source_adapter.load_time_quantized(quantization, stream)?;
            let (store, report) = quantize_layerwise_store(
                store,
                &source_adapter,
                &target_adapter,
                quantization,
                stream,
            )?;
            let mut model =
                load_layerwise_model(store, target_adapter, options, stream, weights_stream)?;
            model.metadata.set_materialization(Some(report));
            Ok(model)
        }
        None => load_layerwise_model(store, source_adapter, options, stream, weights_stream),
    }
}

/// Loads a tensor-parallel adapter directly or through the same bounded packed
/// overlay used by non-distributed residency.
pub(crate) fn load_tensor_parallel_layerwise_model_with_quantization<A, O>(
    store: SharedCheckpointSource,
    source_adapter: A,
    options: O,
    quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: LoadTimeQuantizableAdapter,
    O: Into<LayerWeightResidency>,
{
    let options = options.into();
    let store = resolve_checkpoint_store(store, &source_adapter)?;
    match quantization {
        Some(quantization) => {
            let target_adapter = source_adapter.load_time_quantized(quantization, stream)?;
            let (store, report) = quantize_layerwise_store(
                store,
                &source_adapter,
                &target_adapter,
                quantization,
                stream,
            )?;
            let mut model = load_tensor_parallel_layerwise_model(
                store,
                target_adapter,
                options,
                build,
                stream,
                weights_stream,
            )?;
            model.metadata.set_materialization(Some(report));
            Ok(model)
        }
        None => load_tensor_parallel_layerwise_model(
            store,
            source_adapter,
            options,
            build,
            stream,
            weights_stream,
        ),
    }
}

/// Builds a PP-stage-local packed overlay before pipeline residency planning.
///
/// The source adapter contributes semantic recipes for only the stage-owned
/// static roles and decoder range. The target adapter identifies projections
/// whose runtime parameter tree is packed. Complete or rank-selected expert
/// banks use the same matrix-row tiler as ordinary projections; an independent
/// expert store is only involved when that residency policy was requested.
pub(crate) struct PipelineStageQuantizationSelection<'a> {
    static_roles: &'a [&'a str],
    layer_groups: Vec<(usize, Range<usize>)>,
}

impl<'a> PipelineStageQuantizationSelection<'a> {
    pub(crate) fn new(
        static_roles: &'a [&'a str],
        layer_group: usize,
        layer_range: Range<usize>,
    ) -> Self {
        Self {
            static_roles,
            layer_groups: vec![(layer_group, layer_range)],
        }
    }

    pub(crate) fn with_layer_group(
        mut self,
        layer_group: usize,
        layer_range: Range<usize>,
    ) -> Self {
        if !layer_range.is_empty() {
            self.layer_groups.push((layer_group, layer_range));
        }
        self
    }
}

pub(crate) fn quantize_pipeline_stage_store<A>(
    store: SharedCheckpointSource,
    source_adapter: &A,
    target_adapter: &A,
    selection: PipelineStageQuantizationSelection<'_>,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    A: ArchitectureAdapter,
{
    let mut recipes = BTreeMap::new();
    let mut collect =
        |bindings: &[WeightBinding],
         selected_local_weights: Option<&BTreeMap<String, RecipeDtype>>| {
            for binding in bindings {
                let recipe = binding.source_recipe();
                let metadata = recipe.infer(store.as_ref())?;
                if !matches!(
                    metadata.dtype(),
                    RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
                ) || metadata.shape().len() < 2
                {
                    continue;
                }
                let canonical_local =
                    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                        binding.name(),
                    );
                if selected_local_weights
                    .is_some_and(|selected| !selected.contains_key(&canonical_local))
                {
                    continue;
                }
                let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
                let companion_dtype = selected_local_weights
                    .and_then(|selected| selected.get(&canonical_local))
                    .cloned()
                    .unwrap_or(RecipeDtype::F32);
                match recipes.entry(target.to_string()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((recipe, companion_dtype));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() != &(recipe, companion_dtype) =>
                    {
                        return Err(Error::Quantization(format!(
                        "pipeline load-time quantization target {target:?} has conflicting semantic recipes"
                    )));
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
            Ok::<(), Error>(())
        };

    for unit in source_adapter.static_units(store.as_ref())? {
        if !selection
            .static_roles
            .iter()
            .any(|role| unit.id().as_str().ends_with(&format!(".static.{role}")))
        {
            continue;
        }
        let selected = unit
            .bindings()
            .iter()
            .filter(|binding| target_adapter.quantizes_static_binding(binding))
            .map(|binding| {
                (
                    crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                        binding.name(),
                    ),
                    RecipeDtype::F32,
                )
            })
            .collect::<BTreeMap<_, _>>();
        collect(unit.bindings(), Some(&selected))?;
    }

    for (group, range) in selection.layer_groups {
        for index in range {
            let source_layer = source_adapter.new_layer(group, index, stream)?;
            let target_layer = target_adapter.new_layer(group, index, stream)?;
            let selected = packed_weight_companion_dtypes(&target_layer);
            collect(
                &source_adapter.layer_bindings(group, index, &source_layer, store.as_ref())?,
                Some(&selected),
            )?;
        }
    }

    if recipes.is_empty() {
        return Err(Error::Quantization(format!(
            "pipeline architecture adapter {} declared no floating matrix bindings for stage-local load-time quantization",
            source_adapter.model_type()
        )));
    }
    let targets = recipes
        .into_iter()
        .map(|(target, (recipe, companion_dtype))| {
            let target = BoundedQuantizationTarget::from_recipe(target, recipe)?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companion_dtype)
                }
                WeightQuantization::MxFp4 => Ok(target),
                WeightQuantization::GgufIQuant { .. } => unreachable!(
                    "load-time materialization rejects checkpoint-native GGUF encodings"
                ),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        store,
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        stream,
    )?);
    let report = transformed.report().clone();
    let transformed: SharedCheckpointSource = transformed;
    Ok((transformed, report))
}

fn bounded_quantization_working_set(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    targets: &[BoundedQuantizationTarget],
    quantization: WeightQuantization,
) -> Result<u64, Error> {
    let mut output_bytes = 0u64;
    let mut minimum_tile_bytes = 0u64;
    for target in targets {
        let metadata = target.source().infer(store)?;
        let shape = metadata.shape();
        if shape.len() < 2 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} must be a matrix or matrix bank, got shape {shape:?}",
                target.weight_name()
            )));
        }
        let row_axis = shape.len() - 2;
        let leading = shape[..row_axis]
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Error::Quantization("leading matrix count overflowed".into()))?;
        if leading == 0 || shape[row_axis] == 0 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} must contain at least one matrix row",
                target.weight_name()
            )));
        }
        let rows = leading
            .checked_mul(shape[row_axis])
            .ok_or_else(|| Error::Quantization("matrix-bank row count overflowed".into()))?
            as u64;
        let columns = shape[row_axis + 1];
        let group_size = usize::try_from(quantization.group_size())
            .map_err(|_| Error::Quantization("quantization group size is invalid".into()))?;
        if columns % group_size != 0 || columns % 32 != 0 {
            return Err(Error::Quantization(format!(
                "load-time quantization target {:?} input dimension {columns} must be divisible by group_size {group_size} and 32",
                target.weight_name()
            )));
        }
        let groups = (columns / group_size) as u64;
        let packed_row = (columns as u64)
            .checked_mul(quantization.bits() as u64)
            .and_then(|bits| bits.checked_div(8))
            .ok_or_else(|| Error::Quantization("packed row size overflowed".into()))?;
        let companion_row = if matches!(quantization, WeightQuantization::MxFp4) {
            groups
        } else {
            groups
                .checked_mul(target.affine_companion_bytes())
                .ok_or_else(|| Error::Quantization("packed scale row size overflowed".into()))?
        };
        let bias_row = if quantization.has_biases() {
            groups
                .checked_mul(target.affine_companion_bytes())
                .ok_or_else(|| Error::Quantization("packed bias row size overflowed".into()))?
        } else {
            0
        };
        let output_row = packed_row
            .checked_add(companion_row)
            .and_then(|bytes| bytes.checked_add(bias_row))
            .ok_or_else(|| Error::Quantization("packed output row size overflowed".into()))?;
        output_bytes = output_bytes
            .checked_add(
                rows.checked_mul(output_row)
                    .ok_or_else(|| Error::Quantization("packed target size overflowed".into()))?,
            )
            .ok_or_else(|| Error::Quantization("packed model size overflowed".into()))?;
        for matrix in 0..leading {
            let one_row = target
                .source()
                .select_bounded_matrix_rows(store, matrix, 0, 1)?;
            one_row.preflight_bounded(store)?;
            minimum_tile_bytes = minimum_tile_bytes.max(
                one_row
                    .peak_materialization_bytes(store)?
                    .checked_add(output_row)
                    .ok_or_else(|| Error::Quantization("conversion tile size overflowed".into()))?,
            );
        }
    }
    Ok(output_bytes.max(minimum_tile_bytes))
}

/// Builds a generalized layerwise model from an already cataloged checkpoint.
pub fn load_layerwise_model<A, O>(
    store: SharedCheckpointSource,
    mut adapter: A,
    options: O,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let store = resolve_checkpoint_store(store, &adapter)?;
    let options = options.into();
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut static_ids = Vec::new();
    for unit in adapter.static_units(store.as_ref())? {
        let (id, bindings) = unit.into_parts();
        static_ids.push(id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            id,
            bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let execution_graph = adapter.execution_graph()?;
    let mut groups = Vec::with_capacity(execution_graph.groups().len());
    let mut layer_parameter_bytes = 0u64;
    let mut host_layer_parameter_bytes = 0u64;
    let mut device_window_bytes = 0u64;
    let mut host_window_bytes = 0u64;
    let mut planned_layer_count = 0usize;
    let mut maximum_group_depth = 0usize;
    let mut maximum_host_layer_bytes = 0u64;
    for (group_index, group_spec) in execution_graph.groups().iter().enumerate() {
        let layer_count = adapter.layer_count(group_index)?;
        let depth = options.device_depth(layer_count);
        maximum_group_depth = maximum_group_depth.max(depth);
        if depth > layer_count {
            return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
        }
        if let Some(dense) = dense {
            if dense.host_budget_bytes > 0 && dense.host_lookahead > layer_count {
                return Err(LayerwiseModelError::InvalidHostLayerWindow {
                    depth: dense.host_lookahead,
                    layer_count,
                }
                .into());
            }
        }
        let mut layer_ids = Vec::with_capacity(layer_count);
        let mut layer_bytes = Vec::with_capacity(layer_count);
        let mut layer_host_bytes = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let layer = adapter.new_layer(group_index, index, stream)?;
            let bindings = adapter.layer_bindings(group_index, index, &layer, store.as_ref())?;
            let bytes = binding_bytes(&bindings)?;
            let host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
            maximum_host_layer_bytes = maximum_host_layer_bytes.max(host_bytes);
            host_layer_parameter_bytes = host_layer_parameter_bytes.checked_add(host_bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host execution-unit allocation-capacity total",
                },
            )?;
            layer_parameter_bytes = layer_parameter_bytes.checked_add(bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host execution-unit byte total",
                },
            )?;
            let id = OffloadUnitId::new(adapter.layer_unit_name(group_index, index))?;
            consumed.extend(
                bindings
                    .iter()
                    .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
            );
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
            planned_layer_count = planned_layer_count.checked_add(1).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "streamed execution-unit count",
                },
            )?;
            layer_ids.push(id);
            layer_bytes.push(bytes);
            layer_host_bytes.push(host_bytes);
        }
        let group_device_window = largest_window_bytes(&layer_bytes, depth)?;
        if dense.is_some() {
            device_window_bytes = device_window_bytes.max(group_device_window);
            if let Some(dense) = dense {
                if dense.host_budget_bytes > 0 {
                    host_window_bytes = host_window_bytes.max(largest_window_bytes(
                        &layer_host_bytes,
                        dense.host_lookahead,
                    )?);
                }
            }
        } else {
            device_window_bytes = device_window_bytes.checked_add(group_device_window).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "combined device execution-window byte total",
                },
            )?;
        }
        groups.push(ResidentLayerGroup::new(
            group_spec.id().to_string(),
            layer_ids,
            depth,
        )?);
    }

    consumed.extend(store.materialized_source_keys());
    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));

    validate_unused(store.as_ref(), &consumed, options.strict_loading(), |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    if fully_resident {
        validate_host_budget(offload, 0)?;
    } else if dense.is_some() {
        validate_host_budget(offload, host_window_bytes)?;
    } else {
        validate_host_budget(offload, host_layer_parameter_bytes)?;
    }
    validate_device_budget(
        offload,
        static_device_bytes,
        device_window_bytes,
        maximum_group_depth,
    )?;

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
    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;

    let mut model = LayerwiseModel::new(
        adapter,
        execution_graph,
        store,
        residency,
        groups,
        static_leases,
    )?
    .with_memory_sampling(
        options.sample_backend_memory(),
        options.sample_process_memory(),
    );
    if fully_resident {
        model.materialize_resident_layers(stream)?;
    }
    model.metadata = LayerwiseModelMetadata::new(
        model.adapter.model_type(),
        model.adapter.quantization(),
        planned_layer_count,
        static_device_bytes,
        options.execution_residency(),
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_layer_bytes,
        if fully_resident {
            planned_layer_count
        } else {
            maximum_group_depth
        },
    );
    if let Some(dense) = dense {
        let execution_groups = model
            .groups
            .iter()
            .map(|group| (group.id().to_string(), group.units().to_vec()))
            .collect::<Vec<_>>();
        model.dense_stream = Some(Arc::new(DenseStreamController::new(
            &model.residency,
            dense,
            planned_layer_count,
            layer_parameter_bytes,
            maximum_host_layer_bytes,
            static_device_bytes,
            execution_groups,
        )?));
    }
    Ok(model)
}

pub(crate) fn load_tensor_parallel_layerwise_model<A, O>(
    store: SharedCheckpointSource,
    mut adapter: A,
    options: O,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<A>, Error>
where
    A: ArchitectureAdapter,
    O: Into<LayerWeightResidency>,
{
    let store = resolve_checkpoint_store(store, &adapter)?;
    let options = options.into();
    let fully_resident = options.is_fully_resident();
    let dense = options.dense();
    let offload = options.offload()?;
    let mut planner = build.planner();
    adapter.register_parallel_parameters(build, &mut planner, stream)?;
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(format!(
            "architecture adapter {} declared no tensor-parallel execution-group parameters",
            adapter.model_type()
        )));
    }

    let static_units = adapter.static_units(store.as_ref())?;
    adapter.configure_parallel_static(build, &layout, stream)?;

    let mut definitions = Vec::new();
    let mut specs = Vec::new();
    let mut consumed = BTreeSet::new();
    let mut static_device_bytes = 0u64;
    let mut global_parameter_bytes = 0u64;
    let mut static_ids = Vec::new();
    for unit in static_units {
        let (id, bindings) = unit.into_parts();
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&bindings)?)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "global static parameter byte total",
            })?;
        let bindings = shard_layer_bindings(bindings, "", store.as_ref(), &layout)?;
        static_ids.push(id.clone());
        add_unit(
            &mut definitions,
            &mut specs,
            &mut consumed,
            id,
            bindings,
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            &mut static_device_bytes,
        )?;
    }

    let execution_graph = adapter.execution_graph()?;
    let mut groups = Vec::with_capacity(execution_graph.groups().len());
    let mut layer_parameter_bytes = 0u64;
    let mut host_layer_parameter_bytes = 0u64;
    let mut device_window_bytes = 0u64;
    let mut host_window_bytes = 0u64;
    let mut planned_layer_count = 0usize;
    let mut maximum_group_depth = 0usize;
    let mut maximum_host_layer_bytes = 0u64;
    for (group_index, group_spec) in execution_graph.groups().iter().enumerate() {
        let layer_count = adapter.layer_count(group_index)?;
        let depth = options.device_depth(layer_count);
        maximum_group_depth = maximum_group_depth.max(depth);
        if depth > layer_count {
            return Err(LayerwiseModelError::InvalidLayerWindow { depth, layer_count }.into());
        }
        if let Some(dense) = dense {
            if dense.host_budget_bytes > 0 && dense.host_lookahead > layer_count {
                return Err(LayerwiseModelError::InvalidHostLayerWindow {
                    depth: dense.host_lookahead,
                    layer_count,
                }
                .into());
            }
        }
        let mut layer_ids = Vec::with_capacity(layer_count);
        let mut layer_bytes = Vec::with_capacity(layer_count);
        let mut layer_host_bytes = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let global_layer = adapter.new_layer(group_index, index, stream)?;
            let global_bindings =
                adapter.layer_bindings(group_index, index, &global_layer, store.as_ref())?;
            global_parameter_bytes = global_parameter_bytes
                .checked_add(binding_bytes(&global_bindings)?)
                .ok_or(LayerwiseModelError::ArithmeticOverflow {
                    context: "global TP execution-unit byte total",
                })?;
            let layer = adapter.new_parallel_layer(group_index, index, &layout, stream)?;
            let bindings = adapter.parallel_layer_bindings(
                group_index,
                index,
                &layer,
                store.as_ref(),
                &layout,
                stream,
            )?;
            let bytes = binding_bytes(&bindings)?;
            let host_bytes = host_capacity_upper_bound_for_bindings(&bindings)?;
            maximum_host_layer_bytes = maximum_host_layer_bytes.max(host_bytes);
            host_layer_parameter_bytes = host_layer_parameter_bytes.checked_add(host_bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host TP execution-unit allocation-capacity total",
                },
            )?;
            layer_parameter_bytes = layer_parameter_bytes.checked_add(bytes).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "host TP execution-unit byte total",
                },
            )?;
            let id = OffloadUnitId::new(adapter.layer_unit_name(group_index, index))?;
            consumed.extend(
                bindings
                    .iter()
                    .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
            );
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
            planned_layer_count = planned_layer_count.checked_add(1).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "TP execution-unit count",
                },
            )?;
            layer_ids.push(id);
            layer_bytes.push(bytes);
            layer_host_bytes.push(host_bytes);
        }
        let group_device_window = largest_window_bytes(&layer_bytes, depth)?;
        if dense.is_some() {
            device_window_bytes = device_window_bytes.max(group_device_window);
            if let Some(dense) = dense {
                if dense.host_budget_bytes > 0 {
                    host_window_bytes = host_window_bytes.max(largest_window_bytes(
                        &layer_host_bytes,
                        dense.host_lookahead,
                    )?);
                }
            }
        } else {
            device_window_bytes = device_window_bytes.checked_add(group_device_window).ok_or(
                LayerwiseModelError::ArithmeticOverflow {
                    context: "combined TP device execution-window byte total",
                },
            )?;
        }
        groups.push(ResidentLayerGroup::new(
            group_spec.id().to_string(),
            layer_ids,
            depth,
        )?);
    }
    consumed.extend(store.materialized_source_keys());
    consumed.extend(adapter.additional_consumed_checkpoint_keys(store.as_ref()));
    validate_unused(store.as_ref(), &consumed, options.strict_loading(), |key| {
        adapter.ignores_checkpoint_key(key)
    })?;
    if fully_resident {
        validate_host_budget(offload, 0)?;
    } else if dense.is_some() {
        validate_host_budget(offload, host_window_bytes)?;
    } else {
        validate_host_budget(offload, host_layer_parameter_bytes)?;
    }
    validate_device_budget(
        offload,
        static_device_bytes,
        device_window_bytes,
        maximum_group_depth,
    )?;
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
    let static_leases = static_ids
        .iter()
        .map(|id| residency.acquire(id, MemoryTier::Device))
        .collect::<Result<Vec<_>, _>>()?;
    adapter.populate_static(&static_leases)?;
    let mut model = LayerwiseModel::new(
        adapter,
        execution_graph,
        store,
        residency,
        groups,
        static_leases,
    )?
    .with_memory_sampling(
        options.sample_backend_memory(),
        options.sample_process_memory(),
    );
    model.parallel_layout = Some(layout);
    model.parallel_topology = Some(build.topology());
    if fully_resident {
        model.materialize_resident_layers(stream)?;
    }
    model.metadata = LayerwiseModelMetadata::new(
        model.adapter.model_type(),
        model.adapter.quantization(),
        planned_layer_count,
        static_device_bytes,
        options.execution_residency(),
        layer_parameter_bytes,
        device_window_bytes,
        maximum_host_layer_bytes,
        if fully_resident {
            planned_layer_count
        } else {
            maximum_group_depth
        },
    );
    let local_parameter_bytes = static_device_bytes
        .checked_add(layer_parameter_bytes)
        .ok_or(LayerwiseModelError::ArithmeticOverflow {
            context: "rank-local parallel parameter byte total",
        })?;
    model.parallel_info = Some(ParallelModelInfo::new(
        build.topology(),
        model.adapter.model_type(),
        model
            .parallel_layout
            .as_ref()
            .expect("parallel layout was assigned")
            .tensors()
            .map(|(target, _)| target.to_string())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if fully_resident {
            local_parameter_bytes
        } else {
            static_device_bytes
        },
        static_device_bytes.checked_add(device_window_bytes).ok_or(
            LayerwiseModelError::ArithmeticOverflow {
                context: "maximum rank-local device parameter byte total",
            },
        )?,
    ));
    if let Some(dense) = dense {
        let execution_groups = model
            .groups
            .iter()
            .map(|group| (group.id().to_string(), group.units().to_vec()))
            .collect::<Vec<_>>();
        model.dense_stream = Some(Arc::new(DenseStreamController::new(
            &model.residency,
            dense,
            planned_layer_count,
            layer_parameter_bytes,
            maximum_host_layer_bytes,
            static_device_bytes,
            execution_groups,
        )?));
    }
    Ok(model)
}

fn packed_semantic_weight_name(name: &str) -> Option<String> {
    name.strip_suffix(".scales")
        .or_else(|| name.strip_suffix(".biases"))
        .map(|prefix| format!("{prefix}.weight"))
}

fn stored_tensor_selection(
    tensor: &eredu_runtime::LocalTensorLayout,
    stored_shape: &[usize],
) -> Result<crate::backend::mlx::runtime::checkpoint::store::TensorSelection, Error> {
    use crate::backend::mlx::runtime::checkpoint::store::TensorSelection;
    use eredu_runtime::TensorPlacement;

    let scale_boundary = |axis: usize, boundary: usize| -> Result<usize, Error> {
        let semantic = tensor.global_shape()[axis];
        let stored = stored_shape[axis];
        boundary
            .checked_mul(stored)
            .and_then(|value| value.checked_div(semantic))
            .filter(|scaled| scaled * semantic == boundary * stored)
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "semantic shard boundary {boundary} on axis {axis} is not aligned to packed storage shape {stored_shape:?} derived from {:?}",
                    tensor.global_shape()
                ))
            })
    };

    Ok(match tensor.placement() {
        TensorPlacement::Replicated | TensorPlacement::Local => TensorSelection::Full,
        TensorPlacement::Shard { axis, index, parts } => {
            let stored = stored_shape[*axis];
            if !stored.is_multiple_of(*parts) {
                return Err(Error::Parallel(format!(
                    "packed storage axis {axis} width {stored} cannot be divided among {parts} TP ranks"
                )));
            }
            let width = stored / *parts;
            TensorSelection::Range {
                axis: *axis,
                start: index * width,
                end: (index + 1) * width,
            }
        }
        TensorPlacement::Range { axis, start, end } => TensorSelection::Range {
            axis: *axis,
            start: scale_boundary(*axis, *start)?,
            end: scale_boundary(*axis, *end)?,
        },
        TensorPlacement::Indices { axis, indices } => {
            if stored_shape[*axis] != tensor.global_shape()[*axis] {
                return Err(Error::Parallel(format!(
                    "indexed TP placement on semantic axis {axis} cannot address packed storage shape {stored_shape:?} derived from {:?}",
                    tensor.global_shape()
                )));
            }
            TensorSelection::Indices {
                axis: *axis,
                indices: indices.clone(),
            }
        }
        TensorPlacement::Omit
        | TensorPlacement::Rank { .. }
        | TensorPlacement::PipelineStage { .. } => {
            return Err(Error::Parallel(format!(
                "execution-group binding has non-TP placement {:?}",
                tensor.placement()
            )))
        }
    })
}

pub(crate) fn shard_layer_bindings(
    bindings: Vec<WeightBinding>,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<WeightBinding>, Error> {
    use crate::backend::mlx::runtime::checkpoint::store::TensorSelection;

    let store_keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let canonical_name =
            crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(&format!(
                "{prefix}.{}",
                binding.name()
            ));
        let logical_target = binding.logical_target();
        let tensor = logical_target
            .and_then(|target| layout.tensor(target))
            .or_else(|| {
                logical_target.and_then(|logical| {
                    let canonical_logical =
                        crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(logical);
                    layout.tensors().find_map(|(target, tensor)| {
                        (crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(target)
                            == canonical_logical)
                            .then_some(tensor)
                    })
                })
            })
            .or_else(|| layout.tensor(binding.checkpoint_key()))
            .or_else(|| layout.tensor(&canonical_name))
            .or_else(|| {
                layout.tensors().find_map(|(target, tensor)| {
                    let canonical_target =
                        crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(target);
                    (canonical_target == binding.checkpoint_key()
                        || canonical_target == canonical_name)
                        .then_some(tensor)
                })
            })
            .or_else(|| {
                [
                    logical_target.map(str::to_string),
                    Some(binding.checkpoint_key().to_string()),
                    Some(canonical_name.clone()),
                ]
                .into_iter()
                .flatten()
                .filter_map(|name| packed_semantic_weight_name(&name))
                .find_map(|weight| {
                    layout.tensor(&weight).or_else(|| {
                        let canonical =
                            crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(&weight);
                        layout.tensor(&canonical)
                    })
                })
            });
        let Some(tensor) = tensor else {
            output.push(binding);
            continue;
        };
        // Direct bindings created by callers can describe a logical checkpoint
        // target that is deliberately absent from this physical store. Preserve
        // that contract by deriving its selection and byte count from semantic
        // layout alone. Physical and derived bindings use store metadata below,
        // which is required when packed storage geometry differs from the
        // semantic weight geometry.
        if binding.recipe().is_none() && !store_keys.contains(binding.checkpoint_key()) {
            let selection = stored_tensor_selection(tensor, tensor.global_shape())?;
            if selection == TensorSelection::Full {
                output.push(binding);
                continue;
            }
            let global_elements = tensor.global_shape().iter().product::<usize>();
            let local_elements = tensor.local_shape().iter().product::<usize>();
            let expected_bytes = binding
                .expected_bytes()
                .checked_mul(local_elements as u64)
                .and_then(|bytes| bytes.checked_div(global_elements as u64))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "cannot size rank-local binding {:?}",
                        binding.name()
                    ))
                })?;
            output.push(WeightBinding::new(
                binding.name(),
                binding.checkpoint_key(),
                selection,
                expected_bytes,
            )?);
            continue;
        }
        let recipe = binding.source_recipe();
        let metadata = recipe.infer(store)?;
        let selection = stored_tensor_selection(tensor, metadata.shape())?;
        if selection == crate::backend::mlx::runtime::checkpoint::store::TensorSelection::Full {
            output.push(binding);
            continue;
        }
        let recipe = recipe.select_bounded(store, selection)?;
        let expected_bytes = recipe.infer(store)?.byte_len();
        let sharded = WeightBinding::from_recipe(binding.name(), recipe, expected_bytes)?;
        output.push(sharded);
    }
    Ok(output)
}

fn build_parallel_module_bindings(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<WeightBinding>, Error> {
    use crate::backend::mlx::runtime::checkpoint::store::TensorSelection;
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let params = module.parameters().flatten();
    let mut names = params
        .iter()
        .filter(|(name, parameter)| is_materialized_module_parameter(name, parameter, &params))
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    names.sort();
    let mut bindings = Vec::with_capacity(names.len());
    for local_name in names {
        let parameter = params.get(local_name.as_str()).expect("known parameter");
        let destination = if prefix.is_empty() {
            local_name.clone()
        } else {
            format!("{prefix}.{local_name}")
        };
        let canonical =
            crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                &destination,
            );
        let checkpoint_key = if keys.contains(&destination) {
            destination.clone()
        } else if keys.contains(&canonical) {
            canonical.clone()
        } else {
            return Err(
                crate::backend::mlx::runtime::checkpoint::binding::ModuleBindingError::MissingParameter {
                    destination,
                }
                .into(),
            );
        };
        let metadata = store.source_metadata(&checkpoint_key)?;
        let tensor = layout
            .tensor(&checkpoint_key)
            .or_else(|| layout.tensor(&canonical))
            .or_else(|| {
                packed_semantic_weight_name(&checkpoint_key)
                    .or_else(|| packed_semantic_weight_name(&canonical))
                    .and_then(|weight| {
                        layout.tensor(&weight).or_else(|| {
                            let canonical =
                                crate::backend::mlx::runtime::checkpoint::binding::canonical_checkpoint_name(
                                    &weight,
                                );
                            layout.tensor(&canonical)
                        })
                    })
            });
        let (selection, expected_bytes) = if let Some(tensor) = tensor {
            let local_shape = parameter
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Error::Parallel(format!("invalid local shape for {destination}")))?;
            let selection = stored_tensor_selection(tensor, &metadata.logical_shape)?;
            let mut selected_shape = metadata.logical_shape.clone();
            match &selection {
                TensorSelection::Full => {}
                TensorSelection::Range { axis, start, end } => {
                    selected_shape[*axis] = end - start;
                }
                TensorSelection::Indices { axis, indices } => {
                    selected_shape[*axis] = indices.len();
                }
                TensorSelection::Contiguous { .. } => {
                    unreachable!("TP packed placement never emits a reshaped contiguous span")
                }
            }
            if selected_shape != local_shape {
                return Err(Error::Parallel(format!(
                    "planned packed local shape {:?} for {destination} does not match runtime {:?}",
                    selected_shape, local_shape
                )));
            }
            let recipe =
                crate::backend::mlx::runtime::checkpoint::recipe::DerivedWeightRecipe::source(
                    checkpoint_key.clone(),
                    selection.clone(),
                );
            let bytes = recipe.infer(store)?.byte_len();
            (selection, bytes)
        } else {
            let expected = parameter
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Error::Parallel(format!("invalid shape for {destination}")))?;
            if metadata.logical_shape != expected {
                return Err(Error::Parallel(format!(
                    "unplanned parameter {destination} has checkpoint shape {:?}, runtime expects {:?}",
                    metadata.logical_shape, expected
                )));
            }
            (TensorSelection::Full, metadata.encoded_byte_len as u64)
        };
        bindings.push(WeightBinding::new(
            local_name,
            checkpoint_key,
            selection,
            expected_bytes,
        )?);
    }
    Ok(bindings)
}

#[allow(clippy::too_many_arguments)]
fn add_unit(
    definitions: &mut Vec<OffloadUnit>,
    specs: &mut Vec<OffloadUnitSpec>,
    consumed: &mut BTreeSet<String>,
    id: OffloadUnitId,
    bindings: Vec<WeightBinding>,
    policy: ResidencyPolicy,
    tier: MemoryTier,
    byte_total: &mut u64,
) -> Result<(), Error> {
    let bytes = binding_bytes(&bindings)?;
    *byte_total = byte_total
        .checked_add(bytes)
        .ok_or(LayerwiseModelError::ArithmeticOverflow {
            context: "static device byte total",
        })?;
    consumed.extend(
        bindings
            .iter()
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_string)),
    );
    definitions.push(OffloadUnit::new(id.clone(), bindings)?);
    specs.push(OffloadUnitSpec::new(id, bytes, policy, tier)?);
    Ok(())
}

pub(crate) fn validate_unused<F>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    consumed: &BTreeSet<String>,
    strict: bool,
    ignored: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    if !strict {
        return Ok(());
    }
    let unused = store
        .source_keys()
        .into_iter()
        .chain(store.unclaimed_checkpoint_keys())
        .filter(|key| !consumed.contains(key))
        .filter(|key| !ignored(key))
        .collect::<Vec<_>>();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(LayerwiseModelError::UnexpectedCheckpointParameters { unused }.into())
    }
}

fn largest_window_bytes(layer_bytes: &[u64], depth: usize) -> Result<u64, Error> {
    let mut largest = 0u64;
    for start in 0..layer_bytes.len() {
        let mut current = 0u64;
        for bytes in layer_bytes.iter().skip(start).take(depth) {
            current =
                current
                    .checked_add(*bytes)
                    .ok_or(LayerwiseModelError::ArithmeticOverflow {
                        context: "device layer window byte total",
                    })?;
        }
        largest = largest.max(current);
    }
    Ok(largest)
}

pub(crate) fn validate_host_budget(config: OffloadConfig, required: u64) -> Result<(), Error> {
    if let Some(budget) = config.host_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::HostBudgetTooSmall { required, budget }.into());
        }
    }
    Ok(())
}

pub(crate) fn validate_device_budget(
    config: OffloadConfig,
    static_bytes: u64,
    window_bytes: u64,
    depth: usize,
) -> Result<(), Error> {
    let required =
        static_bytes
            .checked_add(window_bytes)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "static plus device-window byte total",
            })?;
    if let Some(budget) = config.device_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::DeviceBudgetTooSmall {
                static_bytes,
                window_bytes,
                depth,
                required,
                budget,
            }
            .into());
        }
    }
    Ok(())
}

/// Structured failures produced by the generic layerwise execution engine.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseModelError {
    /// A multi-input group did not define how to combine its dependencies.
    #[error(
        "execution group slot {group} has {inputs} dependency outputs but no merge implementation"
    )]
    UnmergedExecutionGroupInputs {
        /// Architecture group slot.
        group: usize,
        /// Number of ready dependency outputs.
        inputs: usize,
    },
    /// Adapter and configured execution-group counts differ.
    #[error("adapter declares {adapter} execution groups but {configured} were configured")]
    ExecutionGroupCount {
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// Adapter and configured group identities differ at one stable slot.
    #[error("execution group slot {slot} is {configured:?}, expected adapter group {adapter:?}")]
    ExecutionGroupIdentity {
        /// Architecture group slot.
        slot: usize,
        /// Adapter-declared identity.
        adapter: String,
        /// Configured residency identity.
        configured: String,
    },
    /// Adapter and configured unit counts differ for one execution group.
    #[error("execution group {group:?} has {configured} configured units but adapter declares {adapter}")]
    ExecutionGroupLength {
        /// Group id.
        group: String,
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// A requested execution group does not exist.
    #[error("unknown resident execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// The configured ordered layer window was invalid.
    #[error("device layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Decoder layer count.
        layer_count: usize,
    },
    /// The protected host lookahead exceeds an execution group.
    #[error("host layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidHostLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A dense transfer window contained an invalid or unordered unit index.
    #[error(
        "dense transfer window index {index} is out of order or outside {unit_count} planned units"
    )]
    InvalidDenseTransferWindow {
        /// Invalid unit index.
        index: usize,
        /// Available units.
        unit_count: usize,
    },
    /// Strict loading found unrelated checkpoint tensors.
    #[error("strict layerwise loading found unexpected checkpoint parameters: {unused:?}")]
    UnexpectedCheckpointParameters {
        /// Unexpected keys in stable order.
        unused: Vec<String>,
    },
    /// The host cannot retain the required decoder-layer allocation capacity.
    #[error("host budget {budget} bytes cannot contain {required} bytes of decoder host-transfer allocations")]
    HostBudgetTooSmall {
        /// Required charged host-allocation bytes.
        required: u64,
        /// Configured host budget.
        budget: u64,
    },
    /// The device cannot contain static weights plus the configured window.
    #[error("device budget {budget} bytes cannot contain {static_bytes} static bytes plus the depth-{depth} layer window ({window_bytes} bytes, {required} total)")]
    DeviceBudgetTooSmall {
        /// Pinned static device bytes.
        static_bytes: u64,
        /// Largest consecutive window bytes.
        window_bytes: u64,
        /// Configured layer count.
        depth: usize,
        /// Total required parameter bytes.
        required: u64,
        /// Configured device budget.
        budget: u64,
    },
    /// A cache vector had the wrong number of layers.
    #[error("layerwise cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// A cache entry was absent.
    #[error("layerwise cache entry {index} is missing")]
    MissingLayerCache {
        /// Missing decoder index.
        index: usize,
    },
    /// Checked byte or index arithmetic overflowed.
    #[error("layerwise model arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Module checkpoint binding failed.
    #[error(transparent)]
    ModuleBinding(#[from] ModuleBindingError),
    /// Residency execution failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}
