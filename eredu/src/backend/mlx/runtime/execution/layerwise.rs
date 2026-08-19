//! Architecture-independent execution of decoder models from resident layers.
//!
//! Reusable MLX materialization, residency-transfer, sharding, and pipeline
//! quantization capabilities used by the backend-neutral layered runtime.

use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
#[cfg(test)]
use eredu_runtime::OffloadUnit;
use eredu_runtime::{
    DenseDiskStreamLoadOptions, DenseDiskStreamReport, DenseStreamTelemetry, DenseTransferSchedule,
    LayerWeightResidency, StaticUnitBindings, WeightBinding, WeightResidency,
    DENSE_TRANSFER_WINDOW,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::Path,
    sync::Arc,
};

use safemlx::{module::ModuleParameters, Array, Stream};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::binding::{build_module_bindings, ModuleBindingError},
    backend::mlx::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::mlx::runtime::checkpoint::recipe::RecipeDtype,
    backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore,
    backend::mlx::runtime::residency::dense_stream::BackgroundLayerPrefetch,
    backend::mlx::runtime::residency::manager::{
        ResidencyError, ResidencyManager, ResidentTransfer, ResidentUnitLease,
    },
    core::residency::{MemoryTier, OffloadConfig, OffloadUnitId, ResidencyLedgerError},
};
#[cfg(test)]
use crate::{
    backend::mlx::runtime::checkpoint::binding::{binding_bytes, is_materialized_module_parameter},
    core::residency::{OffloadUnitSpec, ResidencyPolicy},
};

use eredu_runtime::WeightMaterializationReport;

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
    #[cfg(test)]
    fn has_ready(&self) -> bool {
        self.schedule.has_ready()
    }

    #[cfg(test)]
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

/// Residency-owned execution engine for generalized adapters.
///
/// Group windows, lease lifetime, retained-state evaluation, stream
/// synchronization, and telemetry stay centralized here. Adapter code owns only
/// architecture math, cache validation, and runtime-unit construction.
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

/// Builds a bounded packed overlay from native module trees and caller-owned
/// semantic checkpoint bindings.
///
/// This is the checkpoint-layout-aware counterpart to
/// [`quantize_parameterized_store`]. Architecture composition remains
/// responsible for recipes such as reshapes, slices, and renamed tensors;
/// the backend only selects packed matrix destinations and materializes the
/// shared overlay.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_module_store_with_bindings<SM, U, SF, TF, SB, UB>(
    store: SharedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    mut source_unit: SF,
    mut target_unit: TF,
    unit_count: usize,
    quantization: WeightQuantization,
    stream: &Stream,
    static_bindings: SB,
    mut unit_bindings: UB,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: ModuleParameters,
    U: ModuleParameters,
    SF: FnMut(usize, &Stream) -> Result<U, Error>,
    TF: FnMut(usize, &Stream) -> Result<U, Error>,
    SB: FnOnce(
        &SM,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
    UB: FnMut(
        usize,
        &U,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
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

    collect(
        &static_bindings(source_static, store.as_ref())?,
        &packed_weight_companion_dtypes(target_static),
    )?;
    for index in 0..unit_count {
        let source = source_unit(index, stream)?;
        let target = target_unit(index, stream)?;
        collect(
            &unit_bindings(index, &source, store.as_ref())?,
            &packed_weight_companion_dtypes(&target),
        )?;
    }
    if recipes.is_empty() {
        return Err(Error::Quantization(
            "native parameter topology declared no floating matrix bindings for load-time quantization"
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_pipeline_stage_store_with<L, SU, Q, SL, TL, LB>(
    store: SharedCheckpointSource,
    selection: PipelineStageQuantizationSelection<'_>,
    quantization: WeightQuantization,
    stream: &Stream,
    model_type: &str,
    source_static_units: SU,
    quantizes_static_binding: Q,
    mut source_layer: SL,
    mut target_layer: TL,
    mut layer_bindings: LB,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    L: ModuleParameters,
    SU: FnOnce(
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error>,
    Q: Fn(&WeightBinding) -> bool,
    SL: FnMut(usize, usize, &Stream) -> Result<L, Error>,
    TL: FnMut(usize, usize, &Stream) -> Result<L, Error>,
    LB: FnMut(
        usize,
        usize,
        &L,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error>,
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

    for unit in source_static_units(store.as_ref())? {
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
            .filter(|binding| quantizes_static_binding(binding))
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
            let source_layer = source_layer(group, index, stream)?;
            let target_layer = target_layer(group, index, stream)?;
            let selected = packed_weight_companion_dtypes(&target_layer);
            collect(
                &layer_bindings(group, index, &source_layer, store.as_ref())?,
                Some(&selected),
            )?;
        }
    }

    if recipes.is_empty() {
        return Err(Error::Quantization(format!(
            "pipeline architecture adapter {} declared no floating matrix bindings for stage-local load-time quantization",
            model_type
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

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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
