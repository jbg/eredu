//! Exact checkpoint binding with backend-supplied destination geometry and encoding.

pub(crate) mod reads;
pub use reads::materialize_parameter_owners;

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::{
    quantization::{BoundedQuantizationPlan, BoundedQuantizationTarget, PreparedQuantizedSource},
    recipe::RecipeDtype,
    store::{CheckpointSource, SharedCheckpointSource, TensorSelection},
    WeightQuantization,
};
use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use crate::{LocalModelLayout, ReplicatedTextMaterializationTask, ReplicatedTextTransformGroup};
use eredu_checkpoint::quantization::WeightMaterializationReport;
use eredu_core::checkpoint_mechanisms::ExactParameterMaterializer;
use eredu_core::residency::parameter_binding::WeightBinding;

/// Validates one exact binding unit, realizes its physical owners, and publishes aliases.
pub fn materialize_checkpoint_bindings<B, V, E>(
    source: &dyn CheckpointSource,
    bindings: &[WeightBinding],
    materialize: impl FnOnce(&[&WeightBinding]) -> Result<BTreeMap<String, V>, E>,
) -> Result<BTreeMap<String, V>, BackendFailure>
where
    B: eredu_core::checkpoint_mechanisms::CheckpointParameterBackend,
    V: Clone,
    E: std::error::Error + Send + Sync + 'static,
{
    let plan = eredu_core::residency::parameter_binding::preflight_bindings::<B>(source, bindings)
        .map_err(BackendFailure::from_error)?;
    let owners = plan.owners().collect::<Vec<_>>();
    let mut values = materialize(&owners).map_err(BackendFailure::from_error)?;
    if values.len() != owners.len()
        || owners
            .iter()
            .any(|owner| !values.contains_key(owner.name()))
    {
        return Err(invalid(
            "materialized parameters differ from preflighted physical owners",
        ));
    }
    for (alias, owner) in plan.aliases() {
        values.insert(alias.name().to_owned(), values[owner.name()].clone());
    }
    Ok(values)
}

/// Runs exact transform groups in portable order on a retained source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_exact_transforms<B, P, SM, U>(
    store: &mut SharedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    source_units: &[U],
    target_units: &[U],
    layout: Option<&LocalModelLayout>,
    tasks: &[ReplicatedTextMaterializationTask],
    groups: &[ReplicatedTextTransformGroup],
    context: &<B::Tensor as Tensor>::Context,
) -> Result<WeightMaterializationReport, BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    SM: Parameterized<B::Tensor>,
    U: Parameterized<B::Tensor>,
{
    let mut combined = WeightMaterializationReport::default();
    for group in groups {
        let exact = group.tasks(tasks).map_err(BackendFailure::from_error)?;
        let targets = exact_text_transform_targets::<B, P, _, _>(
            store.as_ref(),
            source_static,
            target_static,
            source_units,
            target_units,
            layout,
            group.quantization(),
            &exact,
        )?;
        let (transformed, report) = materialize_exact_quantization::<B, P>(
            store.clone(),
            group.quantization(),
            targets,
            context,
        )?;
        *store = transformed;
        combined.merge(report);
    }
    Ok(combined)
}

/// Admits, realizes and publishes a portable exact-parameter conversion.
pub(crate) fn materialize_exact_quantization<B: NeuralBackend, P: ExactParameterMaterializer<B>>(
    source: SharedCheckpointSource,
    quantization: WeightQuantization,
    targets: Vec<BoundedQuantizationTarget>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(SharedCheckpointSource, WeightMaterializationReport), BackendFailure> {
    let plan = BoundedQuantizationPlan::for_source(source.as_ref(), quantization, targets)
        .map_err(BackendFailure::from_error)?;
    let prepared: SharedCheckpointSource = std::sync::Arc::new(
        PreparedQuantizedSource::new(source, plan).map_err(BackendFailure::from_error)?,
    );
    let report = crate::checkpoint_quantization::realize_prepared_quantization(&prepared, || {
        P::quantizer(context)
    })?
    .expect("new admitted conversion has a pending plan");
    Ok((prepared, report))
}

/// Projects selected transform tasks onto typed source and destination slots.
/// All source recipes, local placement, and companion identities are resolved
/// before any native encoding runs.
#[allow(clippy::too_many_arguments)]
pub fn exact_text_transform_targets<B, P, SM, U>(
    store: &dyn CheckpointSource,
    source_static: &SM,
    target_static: &SM,
    source_units: &[U],
    target_units: &[U],
    layout: Option<&LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<Vec<BoundedQuantizationTarget>, BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    SM: Parameterized<B::Tensor>,
    U: Parameterized<B::Tensor>,
{
    if source_units.len() != target_units.len() || tasks.is_empty() {
        return Err(invalid(
            "exact transform needs paired modules and a nonempty task group",
        ));
    }
    let mut requested = BTreeSet::new();
    for task in tasks {
        if !task.lowering().transforms_payload()
            || task.executable().weight_quantization() != Some(quantization)
            || !requested.insert(task.name())
        {
            return Err(invalid(format!(
                "invalid or duplicate transform task {:?}",
                task.name()
            )));
        }
    }
    let mut targets = BTreeMap::new();
    collect_transform_targets::<B, P, _>(
        source_static,
        target_static,
        store,
        layout,
        quantization,
        tasks,
        &mut targets,
    )?;
    for (source, target) in source_units.iter().zip(target_units) {
        collect_transform_targets::<B, P, _>(
            source,
            target,
            store,
            layout,
            quantization,
            tasks,
            &mut targets,
        )?;
    }
    let consumed: BTreeSet<_> = targets.keys().copied().collect();
    if consumed != requested {
        return Err(invalid(format!(
            "unresolved transform tasks: {:?}",
            requested.difference(&consumed)
        )));
    }
    Ok(targets.into_values().collect())
}

#[allow(clippy::too_many_arguments)]
fn collect_transform_targets<'a, B, P, M>(
    source: &M,
    target: &M,
    store: &dyn CheckpointSource,
    layout: Option<&LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&'a ReplicatedTextMaterializationTask],
    targets: &mut BTreeMap<&'a str, BoundedQuantizationTarget>,
) -> Result<(), BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    M: Parameterized<B::Tensor>,
{
    use eredu_nn::LinearCompanionRole::*;
    let source =
        crate::parameter::collect_module_parameters(source).map_err(BackendFailure::from_error)?;
    let target =
        crate::parameter::collect_module_parameters(target).map_err(BackendFailure::from_error)?;
    for task in tasks {
        let name = task.name();
        let Some((parameter, _)) = target.get(name) else {
            continue;
        };
        if P::parameter_target(parameter).is_none_or(|slot| slot.dtype != RecipeDtype::U32) {
            return Err(invalid(format!(
                "selected packed output {name:?} requires a U32 slot"
            )));
        }
        let mut scale = None;
        let mut bias = None;
        let mut dtype = None;
        for companion in task.output_companions() {
            let companion_name = companion.name();
            let slot = target
                .get(companion_name)
                .and_then(|(value, _)| P::parameter_target(value))
                .ok_or_else(|| {
                    invalid(format!(
                        "missing packed companion {companion_name:?} for {name:?}"
                    ))
                })?;
            let current = match slot.dtype {
                RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 => slot.dtype,
                RecipeDtype::U8 if !quantization.has_biases() => RecipeDtype::F32,
                other => {
                    return Err(invalid(format!(
                        "unsupported packed companion dtype {other:?} for {companion_name:?}"
                    )))
                }
            };
            if dtype
                .replace(current.clone())
                .is_some_and(|prior| prior != current)
            {
                return Err(invalid(format!(
                    "mismatched packed companion dtypes for {name:?}"
                )));
            }
            match companion.role() {
                Scale => scale = Some(companion_name),
                AffineBias => bias = Some(companion_name),
                GlobalScale => return Err(invalid("load-time NVFP4 encoding is unavailable")),
            }
        }
        let scale =
            scale.ok_or_else(|| invalid(format!("missing scale companion for {name:?}")))?;
        let slot = source
            .get(name)
            .and_then(|(value, _)| P::parameter_target(value))
            .ok_or_else(|| invalid(format!("missing source slot for {name:?}")))?;
        let mut recipe = task.source_recipe().map_err(BackendFailure::from_error)?;
        let mut metadata = recipe.infer(store).map_err(BackendFailure::from_error)?;
        if task
            .derived_output()
            .is_some_and(|expected| expected != &metadata)
        {
            return Err(invalid(format!(
                "transform for {name:?} differs from its admitted derived output"
            )));
        }
        if let Some(layout) = layout {
            let tensor = layout
                .tensor(name)
                .ok_or_else(|| invalid(format!("missing source placement for {name:?}")))?;
            for placement in tensor
                .additional_placements()
                .iter()
                .chain(std::iter::once(tensor.placement()))
            {
                let selection = crate::placement_selection(tensor, placement, metadata.shape())
                    .map_err(BackendFailure::from_error)?;
                if selection != TensorSelection::Full {
                    recipe = recipe
                        .select_bounded(store, selection)
                        .map_err(BackendFailure::from_error)?;
                    metadata = recipe.infer(store).map_err(BackendFailure::from_error)?;
                }
            }
        }
        let floating = |dtype: &RecipeDtype| {
            matches!(
                dtype,
                RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
            )
        };
        if !floating(&slot.dtype)
            || !floating(metadata.dtype())
            || metadata.shape().len() < 2
            || metadata.shape() != slot.shape
        {
            return Err(invalid(format!(
                "transform for {name:?} requires matching floating source geometry"
            )));
        }
        let bounded = BoundedQuantizationTarget::from_recipe(
            name.to_owned(),
            scale.to_owned(),
            bias.map(str::to_owned),
            recipe,
        )
        .map_err(BackendFailure::from_error)?;
        let bounded = match quantization {
            WeightQuantization::Affine(_) => bounded
                .with_affine_companion_dtype(metadata.dtype().clone())
                .map_err(BackendFailure::from_error)?,
            WeightQuantization::MxFp4 => bounded,
            _ => {
                return Err(invalid(format!(
                    "unsupported load-time encoding {quantization:?}"
                )))
            }
        };
        if targets.insert(name, bounded).is_some() {
            return Err(invalid(format!(
                "transform output {name:?} is defined in more than one module"
            )));
        }
    }
    Ok(())
}

/// Binds portable recipes to the exact permitted native destinations.
pub(crate) fn exact_bindings<B, P, M>(
    module: &M,
    store: &dyn CheckpointSource,
    tasks: &[&ReplicatedTextMaterializationTask],
    excluded: &BTreeSet<String>,
    layout: Option<&LocalModelLayout>,
) -> Result<Vec<WeightBinding>, BackendFailure>
where
    B: NeuralBackend,
    P: ExactParameterMaterializer<B>,
    M: Parameterized<B::Tensor>,
{
    crate::build_exact_replicated_text_bindings(
        module,
        store,
        tasks,
        excluded,
        layout,
        P::parameter_target,
        |_, recipe, source| P::lower_mxfp4(recipe, source),
    )
    .map_err(BackendFailure::from_error)
}

/// Exact prepared parameter bindings and slot ownership for a replicated model.
pub struct PreparedReplicatedParameterBindings {
    /// Source after the admitted transforms.
    pub source: SharedCheckpointSource,
    /// Canonical invocation and storage geometry.
    pub invocations: crate::ExecutionUnitAddressMap,
    /// Pinned parameters owned by this rank.
    pub static_bindings: Vec<WeightBinding>,
    /// Binding groups in physical storage order.
    pub unit_bindings: Vec<Vec<WeightBinding>>,
    /// Module destinations owned outside this parameter store.
    pub excluded_parameters: BTreeSet<String>,
    /// Complete set of local parameter tasks on a partitioned rank.
    pub local_parameters: Option<BTreeSet<String>>,
    /// Exact destination slots and their materialized source facts.
    pub slots: Vec<crate::parameter_operations::PreparedParameterSlot>,
    /// Neutral module parameter declarations.
    pub declarations: Vec<eredu_nn::ParameterMetadata>,
    /// Materialization telemetry from any admitted transformations.
    pub materialization: Option<WeightMaterializationReport>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_replicated_parameter_bindings<A, B, S, P>(
    mut store: SharedCheckpointSource,
    architecture: &A,
    units: &[A::Unit],
    invocations: &crate::ExecutionUnitAddressMap,
    source: Option<(&A, &[A::Unit])>,
    source_layout: Option<&LocalModelLayout>,
    target_layout: Option<&LocalModelLayout>,
    partition: &crate::ReplicatedTextMaterializationPartitionPlan,
    tasks: &[ReplicatedTextMaterializationTask],
    addressable: &[String],
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedParameterBindings, BackendFailure>
where
    B: NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
    P: ExactParameterMaterializer<B>,
{
    let layout = invocations.storage_layout();
    if units.len() != layout.len() {
        return Err(invalid(
            "constructed units differ from selected execution storage",
        ));
    }
    let static_tasks = partition
        .static_tasks(tasks)
        .map_err(BackendFailure::from_error)?;
    let unit_tasks = partition
        .unit_tasks(tasks)
        .map_err(BackendFailure::from_error)?;
    let groups =
        crate::group_replicated_text_transform_tasks(tasks).map_err(BackendFailure::from_error)?;
    let materialization = match (source, groups.is_empty()) {
        (None, true) => None,
        (Some((source, source_units)), false) if source_units.len() == units.len() => {
            Some(materialize_exact_transforms::<B, P, _, _>(
                &mut store,
                source.static_modules(),
                architecture.static_modules(),
                source_units,
                units,
                source_layout,
                tasks,
                &groups,
                context,
            )?)
        }
        (Some(_), true) => return Err(invalid("source architecture has no materialization tasks")),
        (None, false) => return Err(invalid("selected transform tasks have no source modules")),
        _ => {
            return Err(invalid(
                "source and target execution storage have different cardinality",
            ))
        }
    };

    let addressable = addressable.iter().cloned().collect::<BTreeSet<_>>();
    let mut excluded_parameters = addressable.clone();
    if target_layout.is_some() {
        let selected = targets(static_tasks.iter().copied());
        let declared = eredu_nn::validate_parameter_topology(architecture.static_modules())
            .map_err(BackendFailure::from_error)?;
        excluded_parameters.extend(
            declared
                .into_iter()
                .map(|parameter| parameter.id.to_string())
                .filter(|name| !selected.contains(name)),
        );
    }
    let static_bindings = exact_bindings::<B, P, _>(
        architecture.static_modules(),
        store.as_ref(),
        &static_tasks,
        &excluded_parameters,
        target_layout,
    )?;
    let unit_bindings = units
        .iter()
        .zip(&unit_tasks)
        .map(|(unit, tasks)| {
            exact_bindings::<B, P, _>(unit, store.as_ref(), tasks, &addressable, target_layout)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let addresses = (0..layout.len())
        .map(|ordinal| layout.address(ordinal).expect("declared physical owner"))
        .collect::<Vec<_>>();
    let (slots, declarations) =
        crate::parameter_operations::collect_prepared_layered_parameters::<A, B, S>(
            architecture,
            units,
            &addresses,
            &static_bindings,
            &unit_bindings,
            store.as_ref(),
        )?;
    Ok(PreparedReplicatedParameterBindings {
        source: store,
        invocations: invocations.clone(),
        static_bindings,
        unit_bindings,
        excluded_parameters,
        local_parameters: target_layout.map(|_| targets(tasks)),
        slots,
        declarations,
        materialization,
    })
}

fn targets<'a>(
    tasks: impl IntoIterator<Item = &'a ReplicatedTextMaterializationTask>,
) -> BTreeSet<String> {
    tasks
        .into_iter()
        .flat_map(|task| {
            std::iter::once(task.name().to_owned()).chain(
                task.output_companions()
                    .iter()
                    .map(|companion| companion.name().to_owned()),
            )
        })
        .collect()
}

fn invalid(detail: impl std::fmt::Display) -> BackendFailure {
    BackendFailure::from_error(eredu_nn::Error::backend(detail))
}

pub(crate) use companion_catalog::bind_packed_transform_targets;
pub mod companion_catalog;
