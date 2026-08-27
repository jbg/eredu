//! Architecture-independent execution of decoder models from resident layers.
//!
//! Reusable MLX materialization, residency-transfer, sharding, and pipeline
//! quantization capabilities used by the backend-neutral layered runtime.

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
    WeightQuantization,
};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, DenseDiskStreamReport, DenseStreamTelemetry, DenseTransferSchedule,
    StaticUnitBindings, WeightBinding, DENSE_TRANSFER_WINDOW,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    path::Path,
    sync::Arc,
};

use safemlx::{module::ModuleParameters, Dtype, Stream};

use crate::{
    backend::error::Error,
    backend::runtime::checkpoint::binding::{build_module_bindings, ModuleBindingError},
    backend::runtime::checkpoint::bounded_quantization::{
        BoundedQuantizationPlan, BoundedQuantizationTarget, BoundedQuantizedWeightStore,
    },
    backend::runtime::residency::dense_stream::BackgroundLayerPrefetch,
    backend::runtime::residency::manager::{
        ResidencyError, ResidencyManager, ResidentTransfer, ResidentUnitLease,
    },
};
use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitId, ResidencyLedgerError};

use eredu_nn::{LinearCompanionRole, ParameterMetadata, ParameterVisitor, Parameterized};
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

/// Opens a shared safetensors source with a bounded mapped-shard cache.
pub fn open_safetensors_weight_store(
    model_dir: &Path,
    max_mapped_shards: usize,
) -> Result<SharedCheckpointSource, Error> {
    Ok(Arc::new(
        SafetensorsWeightStore::open_with_max_mapped_shards(model_dir, max_mapped_shards)?,
    ))
}

/// Coordinates bounded host prefetch, device transfers, and dense-stream telemetry.
pub struct DenseStreamController {
    options: DenseDiskStreamLoadOptions,
    background: Option<BackgroundLayerPrefetch>,
    telemetry: DenseStreamTelemetry,
}

impl DenseStreamController {
    /// Creates a controller for a validated dense disk-stream plan.
    pub fn new(
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

    /// Opens a bounded device-transfer window for selected group-local units.
    pub fn transfer_window(
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

    /// Cancels prefetch and removes host/device protection for one group.
    pub fn clear_group(&self, manager: &ResidencyManager, group: &str) -> Result<(), Error> {
        manager.protect_group_window(&format!("dense:{group}:host"), &[], MemoryTier::Host)?;
        manager.protect_group_window(&format!("dense:{group}:device"), &[], MemoryTier::Device)?;
        if let Some(background) = &self.background {
            background.cancel()?;
        }
        Ok(())
    }

    /// Starts transactional telemetry for one model forward pass.
    pub fn forward_guard(
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

    /// Starts cleanup and execution accounting for one group.
    pub fn group_guard(
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

    /// Returns current dense-stream and residency telemetry.
    pub fn report(&self, manager: &ResidencyManager) -> Result<DenseDiskStreamReport, Error> {
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
pub struct DenseTransferWindow {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    group: String,
    units: Vec<OffloadUnitId>,
    schedule: DenseTransferSchedule<DensePreparedTransfer>,
    prefill: bool,
}

impl DenseTransferWindow {
    /// Takes the next transfer after ordering `consumer` behind its event.
    pub fn next(&mut self, consumer: &Stream) -> Result<DensePreparedTransfer, Error> {
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
    pub fn refill(&mut self) -> Result<(), Error> {
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
pub struct DensePreparedTransfer {
    index: usize,
    transfer: ResidentTransfer,
}

impl DensePreparedTransfer {
    /// Returns the index in the group's authoritative unit list.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the single resident lease protected by this transfer.
    pub fn lease(&self) -> &ResidentUnitLease {
        self.transfer
            .leases()
            .first()
            .expect("dense transfer always acquires one unit")
    }
}

/// Transactional guard for dense-stream forward telemetry.
pub struct DenseStreamForwardGuard {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    armed: bool,
}

impl DenseStreamForwardGuard {
    /// Commits the forward telemetry and disarms rollback-on-drop.
    pub fn complete(mut self) -> Result<(), Error> {
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

/// Cleanup guard for one dense-stream execution group.
pub struct DenseStreamGroupGuard {
    controller: Arc<DenseStreamController>,
    manager: ResidencyManager,
    group: String,
    armed: bool,
}

impl DenseStreamGroupGuard {
    /// Clears group protections and records successful execution.
    pub fn complete(mut self) -> Result<(), Error> {
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
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct PackedWeightCompanions {
    weight_name: String,
    scales_name: String,
    biases_name: Option<String>,
    affine_companion_dtype: RecipeDtype,
}

pub(crate) fn packed_weight_companions<M>(
    module: &M,
    quantization: WeightQuantization,
) -> Result<BTreeMap<String, PackedWeightCompanions>, Error>
where
    M: Parameterized<crate::MlxTensor>,
{
    struct Collector {
        parameters: BTreeMap<String, Dtype>,
        companions: BTreeMap<(String, LinearCompanionRole), (String, Dtype)>,
        error: Option<Error>,
    }

    impl<'a> ParameterVisitor<'a, crate::MlxTensor> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a crate::MlxTensor) {
            if self.error.is_some() {
                return;
            }
            let name = metadata.id.as_str().to_owned();
            self.parameters
                .insert(name.clone(), value.as_array().dtype());
            if let Some(role) = metadata.linear_companion {
                let Some(weight) = metadata.linear_companion_of else {
                    self.error = Some(Error::Quantization(format!(
                        "linear quantization companion {name:?} has no primary weight identity"
                    )));
                    return;
                };
                if self
                    .companions
                    .insert(
                        (weight.as_str().to_owned(), role),
                        (name.clone(), value.as_array().dtype()),
                    )
                    .is_some()
                {
                    self.error = Some(Error::Quantization(format!(
                        "linear weight {:?} declares more than one {role:?} companion",
                        weight.as_str()
                    )));
                }
            }
        }
    }

    let mut collector = Collector {
        parameters: BTreeMap::new(),
        companions: BTreeMap::new(),
        error: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(error) = collector.error {
        return Err(error);
    }
    let weights = collector
        .companions
        .keys()
        .map(|(weight, _)| weight.clone())
        .collect::<BTreeSet<_>>();
    weights
        .into_iter()
        .map(|weight_name| {
            let dtype = collector.parameters.get(&weight_name).ok_or_else(|| {
                Error::Quantization(format!(
                    "linear quantization companions reference missing weight {weight_name:?}"
                ))
            })?;
            if *dtype != Dtype::Uint32 {
                return Ok(None);
            }
            let (scales_name, scales_dtype) = collector
                .companions
                .remove(&(weight_name.clone(), LinearCompanionRole::Scale))
                .ok_or_else(|| {
                    Error::Quantization(format!(
                        "packed linear weight {weight_name:?} has no declared scale companion"
                    ))
                })?;
            let biases = collector
                .companions
                .remove(&(weight_name.clone(), LinearCompanionRole::AffineBias));
            if quantization.has_biases() && biases.is_none() {
                return Err(Error::Quantization(format!(
                    "affine packed linear weight {weight_name:?} has no declared bias companion"
                )));
            }
            let affine_companion_dtype = match scales_dtype {
                Dtype::Float16 => RecipeDtype::F16,
                Dtype::Bfloat16 => RecipeDtype::BF16,
                Dtype::Float32 => RecipeDtype::F32,
                Dtype::Uint8 if !quantization.has_biases() => RecipeDtype::F32,
                dtype => {
                    return Err(Error::Quantization(format!(
                        "packed linear weight {weight_name:?} has unsupported scale dtype {dtype:?}"
                    )))
                }
            };
            if let Some((_, biases_dtype)) = &biases {
                let expected = match affine_companion_dtype {
                    RecipeDtype::F16 => Dtype::Float16,
                    RecipeDtype::BF16 => Dtype::Bfloat16,
                    RecipeDtype::F32 => Dtype::Float32,
                    _ => unreachable!("selected affine companion dtype"),
                };
                if *biases_dtype != expected {
                    return Err(Error::Quantization(format!(
                        "packed linear weight {weight_name:?} has mismatched scale and bias dtypes"
                    )));
                }
            }
            Ok(Some((
                weight_name.clone(),
                PackedWeightCompanions {
                    weight_name,
                    scales_name,
                    biases_name: biases.map(|(name, _)| name),
                    affine_companion_dtype,
                },
            )))
        })
        .collect::<Result<Vec<_>, Error>>()
        .map(|targets| targets.into_iter().flatten().collect())
}

type QuantizationRecipes = BTreeMap<String, (DerivedWeightRecipe, PackedWeightCompanions)>;

fn collect_quantization_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    selected: &BTreeMap<String, PackedWeightCompanions>,
    recipes: &mut QuantizationRecipes,
    context: &str,
) -> Result<(), Error> {
    for binding in bindings {
        if binding.is_alias() {
            continue;
        }
        let recipe = binding.source_recipe();
        let metadata = recipe.infer(store)?;
        if !matches!(
            metadata.dtype(),
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
        ) || metadata.shape().len() < 2
        {
            continue;
        }
        let Some(companions) = selected.get(binding.name()).cloned() else {
            continue;
        };
        let target = companions.weight_name.clone();
        match recipes.entry(target.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((recipe, companions));
            }
            std::collections::btree_map::Entry::Occupied(entry)
                if entry.get() != &(recipe, companions) =>
            {
                return Err(Error::Quantization(format!(
                    "{context} target {target:?} has conflicting semantic recipes"
                )));
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod packed_weight_companion_tests {
    use super::*;
    use crate::backend::nn::shared::MlxNeuralBackend;
    use eredu_checkpoint::store::MemoryWeightStore;
    use eredu_checkpoint::AffineQuantization;
    use eredu_nn::{LinearFormat, LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec};
    use safemlx::{Device, DeviceType, ExecutionContext};

    fn parameter(name: &str) -> ParameterSpec {
        ParameterSpec::trainable(name).unwrap()
    }

    #[test]
    fn quantized_store_preserves_nonconventional_architecture_companion_names() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let weight_name = "encoder.blocks.3.projection.kernel";
        let source = MlxNeuralBackend::linear(
            LinearSpec {
                input: 64,
                output: 8,
                weight: parameter(weight_name),
                bias: None,
                format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            },
            context.stream(),
        )
        .unwrap();
        let format = LinearFormatSpec::affine(
            WeightQuantization::Affine(AffineQuantization::default()).into(),
            parameter("encoder.blocks.3.quantization.scale-table"),
            parameter("encoder.blocks.3.quantization.zero-points"),
        )
        .unwrap();
        let linear = MlxNeuralBackend::linear(
            LinearSpec {
                input: 64,
                output: 8,
                weight: parameter(weight_name),
                bias: None,
                format,
            },
            context.stream(),
        )
        .unwrap();

        let companions = packed_weight_companions(
            &linear,
            WeightQuantization::Affine(AffineQuantization::default()),
        )
        .unwrap();
        let target = companions
            .get("encoder.blocks.3.projection.kernel")
            .unwrap();
        assert_eq!(target.weight_name, "encoder.blocks.3.projection.kernel");
        assert_eq!(
            target.scales_name,
            "encoder.blocks.3.quantization.scale-table"
        );
        assert_eq!(
            target.biases_name.as_deref(),
            Some("encoder.blocks.3.quantization.zero-points")
        );

        let store: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                weight_name.to_owned(),
                safetensors::Dtype::F32,
                vec![8, 64],
                vec![0; 8 * 64 * size_of::<f32>()],
            )])
            .unwrap(),
        );
        let (quantized, _) = quantize_parameterized_module_store(
            store,
            &source,
            &linear,
            WeightQuantization::Affine(AffineQuantization::default()),
            context.stream(),
        )
        .unwrap();
        assert!(quantized
            .source_metadata("encoder.blocks.3.quantization.scale-table")
            .is_ok());
        assert!(quantized
            .source_metadata("encoder.blocks.3.quantization.zero-points")
            .is_ok());
        assert!(quantized
            .source_metadata("encoder.blocks.3.projection.kernel_scales")
            .is_err());
    }
}

/// Builds a quantized checkpoint overlay from neutral parameter topologies.
pub fn quantize_parameterized_store<SM, U, SF, TF>(
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
    SM: Clone + eredu_nn::Parameterized<crate::MlxTensor>,
    U: eredu_nn::Parameterized<crate::MlxTensor>,
    SF: FnMut(usize, &Stream) -> Result<U, Error>,
    TF: FnMut(usize, &Stream) -> Result<U, Error>,
{
    let mut recipes = BTreeMap::new();
    let source_static = crate::backend::nn::shared::MlxModule::new(source_static.clone());
    let target_static = crate::backend::nn::shared::MlxModule::new(target_static.clone());
    collect_quantization_recipes(
        store.as_ref(),
        &build_module_bindings(&source_static, "", store.as_ref())?,
        &packed_weight_companions(&target_static, quantization)?,
        &mut recipes,
        "load-time quantization",
    )?;
    for index in 0..unit_count {
        let source = crate::backend::nn::shared::MlxModule::new(source_unit(index, stream)?);
        let target = crate::backend::nn::shared::MlxModule::new(target_unit(index, stream)?);
        collect_quantization_recipes(
            store.as_ref(),
            &build_module_bindings(&source, "", store.as_ref())?,
            &packed_weight_companions(&target, quantization)?,
            &mut recipes,
            "load-time quantization",
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
        .map(|(target, (recipe, companions))| {
            let target = BoundedQuantizationTarget::from_recipe(
                target,
                companions.scales_name,
                companions.biases_name,
                recipe,
            )?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companions.affine_companion_dtype)
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

/// Builds a bounded packed overlay for one fully resident neutral module tree.
pub fn quantize_parameterized_module_store<M>(
    store: SharedCheckpointSource,
    source: &M,
    target: &M,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    M: Clone + eredu_nn::Parameterized<crate::MlxTensor>,
{
    quantize_parameterized_store(
        store,
        source,
        target,
        |_index, _stream| Ok(source.clone()),
        |_index, _stream| Ok(target.clone()),
        0,
        quantization,
        stream,
    )
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
pub fn quantize_module_store_with_bindings<SM, U, SF, TF, SB, UB>(
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
    SM: ModuleParameters + Parameterized<crate::MlxTensor>,
    U: ModuleParameters + Parameterized<crate::MlxTensor>,
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
    collect_quantization_recipes(
        store.as_ref(),
        &static_bindings(source_static, store.as_ref())?,
        &packed_weight_companions(target_static, quantization)?,
        &mut recipes,
        "load-time quantization",
    )?;
    for index in 0..unit_count {
        let source = source_unit(index, stream)?;
        let target = target_unit(index, stream)?;
        collect_quantization_recipes(
            store.as_ref(),
            &unit_bindings(index, &source, store.as_ref())?,
            &packed_weight_companions(&target, quantization)?,
            &mut recipes,
            "load-time quantization",
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
        .map(|(target, (recipe, companions))| {
            let target = BoundedQuantizationTarget::from_recipe(
                target,
                companions.scales_name,
                companions.biases_name,
                recipe,
            )?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companions.affine_companion_dtype)
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
pub struct PipelineStageQuantizationSelection<'a> {
    static_roles: &'a [&'a str],
    layer_groups: Vec<(usize, Range<usize>)>,
}

impl<'a> PipelineStageQuantizationSelection<'a> {
    /// Returns the static parameter roles owned by this pipeline stage.
    pub fn static_roles(&self) -> &'a [&'a str] {
        self.static_roles
    }

    /// Creates a selection with one stage-local layer group.
    pub fn new(static_roles: &'a [&'a str], layer_group: usize, layer_range: Range<usize>) -> Self {
        Self {
            static_roles,
            layer_groups: vec![(layer_group, layer_range)],
        }
    }

    /// Adds a non-empty stage-local layer group to the selection.
    pub fn with_layer_group(mut self, layer_group: usize, layer_range: Range<usize>) -> Self {
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
    static_companions: BTreeMap<String, PackedWeightCompanions>,
    source_static_units: SU,
    quantizes_static_binding: Q,
    mut source_layer: SL,
    mut target_layer: TL,
    mut layer_bindings: LB,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), Error>
where
    L: ModuleParameters + Parameterized<crate::MlxTensor>,
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
    for unit in source_static_units(store.as_ref())? {
        let selected_names = unit
            .bindings()
            .iter()
            .filter(|binding| quantizes_static_binding(binding))
            .map(|binding| binding.name().to_owned())
            .collect::<BTreeSet<_>>();
        let selected = static_companions
            .iter()
            .filter(|(name, _)| selected_names.contains(*name))
            .map(|(name, companions)| (name.clone(), companions.clone()))
            .collect::<BTreeMap<_, _>>();
        collect_quantization_recipes(
            store.as_ref(),
            unit.bindings(),
            &selected,
            &mut recipes,
            "pipeline load-time quantization",
        )?;
    }

    for (group, range) in selection.layer_groups {
        for index in range {
            let source_layer = source_layer(group, index, stream)?;
            let target_layer = target_layer(group, index, stream)?;
            let selected = packed_weight_companions(&target_layer, quantization)?;
            collect_quantization_recipes(
                store.as_ref(),
                &layer_bindings(group, index, &source_layer, store.as_ref())?,
                &selected,
                &mut recipes,
                "pipeline load-time quantization",
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
        .map(|(target, (recipe, companions))| {
            let target = BoundedQuantizationTarget::from_recipe(
                target,
                companions.scales_name,
                companions.biases_name,
                recipe,
            )?;
            match quantization {
                WeightQuantization::Affine(_) => {
                    target.with_affine_companion_dtype(companions.affine_companion_dtype)
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

fn stored_tensor_selection(
    tensor: &eredu_runtime::LocalTensorLayout,
    stored_shape: &[usize],
) -> Result<TensorSelection, Error> {
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

/// Applies the local parallel layout to architecture-declared layer bindings.
pub fn shard_layer_bindings(
    bindings: Vec<WeightBinding>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<WeightBinding>, Error> {
    let store_keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(bindings.len());
    for binding in bindings {
        if binding.is_alias() {
            output.push(binding);
            continue;
        }
        let logical_target = binding
            .logical_target()
            .ok_or_else(|| LayerwiseModelError::MissingParallelBindingTarget {
                binding: binding.name().to_owned(),
            })?
            .to_owned();
        let quantization_companions = binding
            .quantization_companions()
            .map(|(scales, biases)| (scales.to_owned(), biases.to_owned()));
        let tensor = layout.tensor(&logical_target).ok_or_else(|| {
            LayerwiseModelError::UnknownParallelBindingTarget {
                binding: binding.name().to_owned(),
                target: logical_target.clone(),
            }
        })?;
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
            let mut sharded = WeightBinding::new(
                binding.name(),
                binding.checkpoint_key(),
                selection,
                expected_bytes,
            )?;
            sharded = sharded.with_logical_target(logical_target)?;
            if let Some((scales, biases)) = quantization_companions {
                sharded = sharded.with_quantization_companions(scales, biases)?;
            }
            output.push(sharded);
            continue;
        }
        let recipe = binding.source_recipe();
        let metadata = recipe.infer(store)?;
        let selection = stored_tensor_selection(tensor, metadata.shape())?;
        if selection == TensorSelection::Full {
            output.push(binding);
            continue;
        }
        let recipe = recipe.select_bounded(store, selection)?;
        let expected_bytes = recipe.infer(store)?.byte_len();
        let mut sharded = WeightBinding::from_recipe(binding.name(), recipe, expected_bytes)?;
        sharded = sharded.with_logical_target(logical_target)?;
        if let Some((scales, biases)) = quantization_companions {
            sharded = sharded.with_quantization_companions(scales, biases)?;
        }
        output.push(sharded);
    }
    Ok(output)
}

#[cfg(test)]
mod shard_layer_bindings_tests {
    use super::*;
    use eredu_checkpoint::store::MemoryWeightStore;
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, ParameterRole, TensorPlacement};

    fn store() -> MemoryWeightStore {
        MemoryWeightStore::from_safetensors([(
            "model.weight".to_owned(),
            safetensors::Dtype::F32,
            vec![4, 2],
            vec![0; 4 * 2 * size_of::<f32>()],
        )])
        .unwrap()
    }

    fn layout() -> LocalModelLayout {
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "model.weight".into(),
            LocalTensorLayout::new(
                "projection",
                ParameterRole::ColumnProjection,
                vec![4, 2],
                vec![2, 2],
                TensorPlacement::Shard {
                    axis: 0,
                    index: 0,
                    parts: 2,
                },
                None,
                None,
                false,
            ),
        );
        layout
    }

    fn binding() -> WeightBinding {
        WeightBinding::new(
            "weight",
            "model.weight",
            TensorSelection::Full,
            (4 * 2 * size_of::<f32>()) as u64,
        )
        .unwrap()
    }

    #[test]
    fn sharding_uses_the_exact_architecture_logical_target() {
        let binding = binding().with_logical_target("model.weight").unwrap();

        let store = store();
        let sharded = shard_layer_bindings(vec![binding], &store, &layout()).unwrap();

        assert_eq!(
            sharded[0].source_recipe().infer(&store).unwrap().shape(),
            [2, 2]
        );
        assert_eq!(sharded[0].expected_bytes(), 16);
    }

    #[test]
    fn sharding_rejects_a_missing_architecture_logical_target() {
        let error = shard_layer_bindings(vec![binding()], &store(), &layout()).unwrap_err();

        assert!(matches!(
            error,
            Error::LayerwiseModel(LayerwiseModelError::MissingParallelBindingTarget {
                binding,
            }) if binding == "weight"
        ));
    }

    #[test]
    fn sharding_rejects_an_unmatched_architecture_logical_target() {
        let binding = binding().with_logical_target("model.weigth").unwrap();

        let error = shard_layer_bindings(vec![binding], &store(), &layout()).unwrap_err();

        assert!(matches!(
            error,
            Error::LayerwiseModel(LayerwiseModelError::UnknownParallelBindingTarget {
                binding,
                target,
            }) if binding == "weight" && target == "model.weigth"
        ));
    }
}

/// Rejects checkpoint tensors that were neither consumed nor explicitly ignored.
pub fn validate_unused<F>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    consumed: &BTreeSet<String>,
    ignored: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    let unused = store
        .source_keys()
        .into_iter()
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
mod validate_unused_tests {
    use super::*;
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    };

    struct ResolvedTestSource;

    impl CheckpointSource for ResolvedTestSource {
        fn source_keys(&self) -> Vec<String> {
            vec!["claimed.weight".into()]
        }

        fn source_metadata(&self, _key: &str) -> Result<TensorMetadata, StoreError> {
            unreachable!("unused-key validation does not read tensor metadata")
        }

        fn acquire_lease(
            &self,
            _request: TensorReadRequest,
        ) -> Result<CheckpointLease, StoreError> {
            unreachable!("unused-key validation does not acquire tensor payloads")
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            unreachable!("unused-key validation does not inspect storage diagnostics")
        }

        fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
            vec!["schema.allowed.extra".into()]
        }

        fn is_checkpoint_contract_resolved(&self) -> bool {
            true
        }
    }

    #[test]
    fn schema_unclaimed_keys_are_not_backend_unused_parameters() {
        let consumed = BTreeSet::from(["claimed.weight".into()]);
        validate_unused(&ResolvedTestSource, &consumed, |_| false).unwrap();
    }
}

/// Validates that required host storage fits the configured offload budget.
pub fn validate_host_budget(config: OffloadConfig, required: u64) -> Result<(), Error> {
    if let Some(budget) = config.host_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::HostBudgetTooSmall { required, budget }.into());
        }
    }
    Ok(())
}

/// Validates that static and window storage fit the configured device budget.
pub fn validate_device_budget(
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
    /// A non-alias binding omitted its architecture-owned parameter identity.
    #[error("parallel binding {binding:?} has no architecture-logical target")]
    MissingParallelBindingTarget {
        /// Resident-unit binding name.
        binding: String,
    },
    /// A binding's exact architecture-owned identity was absent from the local layout.
    #[error("parallel binding {binding:?} targets unknown architecture parameter {target:?}")]
    UnknownParallelBindingTarget {
        /// Resident-unit binding name.
        binding: String,
        /// Exact architecture-logical parameter identity.
        target: String,
    },
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
