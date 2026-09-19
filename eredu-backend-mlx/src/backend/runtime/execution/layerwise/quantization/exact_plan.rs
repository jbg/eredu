//! Shared exact-task planning from cold or native parameter-slot geometry.
use super::*;
use crate::backend::runtime::checkpoint::{
    binding::mlx_workspace_binding_targets, bounded_quantization::ColdQuantization,
};
use eredu_runtime::ParameterBindingTarget;

/// The source recipe and destination projection determine the same packed plan
/// used by ordinary loading. No stream, native tensor or payload lease is created.
pub(crate) fn prepare_exact_quantization_from_destinations(
    store: RetainedCheckpointSource,
    modules: &[&BTreeMap<String, eredu_nn::workspace::WorkspaceLayout>],
    source_layout: Option<&eredu_runtime::LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<ColdQuantization, Error> {
    let targets = modules
        .iter()
        .map(|module| {
            mlx_workspace_binding_targets(module).ok_or_else(|| {
                Error::Quantization("invalid cold quantization destination representation".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (plan, destinations) = build(store.as_ref(), &targets, source_layout, quantization, tasks)?;
    let cold = ColdQuantization::prepare(store, plan)?;
    cold.validate_destinations(&destinations)?;
    Ok(cold)
}

pub(super) fn build(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    modules: &[BTreeMap<String, ParameterBindingTarget>],
    source_layout: Option<&eredu_runtime::LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<
    (
        BoundedQuantizationPlan,
        BTreeMap<String, ParameterBindingTarget>,
    ),
    Error,
> {
    quantization.validate()?;
    if quantization.gguf_iquant().is_some() {
        return Err(Error::Quantization(
            "checkpoint-native GGUF encodings cannot be produced by load-time quantization".into(),
        ));
    }
    let mut requested = BTreeMap::new();
    for task in tasks {
        if !matches!(
            task.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        ) {
            return Err(Error::Quantization(format!(
                "exact materialization task {:?} did not select a transform lowering",
                task.name()
            )));
        }
        if task.executable().weight_quantization() != Some(quantization) {
            return Err(Error::Quantization(format!(
                "exact materialization task {:?} does not match its format group",
                task.name()
            )));
        }
        if requested.insert(task.name(), *task).is_some() {
            return Err(Error::Quantization(format!(
                "exact materialization task {:?} was requested more than once",
                task.name()
            )));
        }
    }
    if requested.is_empty() {
        return Err(Error::Quantization(
            "exact replicated-text materialization received no tasks".into(),
        ));
    }
    let requested_names = requested.keys().copied().collect::<BTreeSet<_>>();

    let mut recipes = BTreeMap::new();
    let mut destinations = BTreeMap::new();
    for parameters in modules {
        for (name, mut companions) in exact_task_weight_companions(parameters, quantization, tasks)?
        {
            let task = requested[name.as_str()];
            let (recipe, metadata) =
                eredu_runtime::resolve_replicated_text_transform_source(store, task, source_layout)
                    .map_err(|error| match error {
                        eredu_runtime::TransformSourceError::Task { details } => {
                            Error::Quantization(details)
                        }
                        eredu_runtime::TransformSourceError::Placement(cause) => {
                            Error::Parallel(cause.to_string())
                        }
                        eredu_runtime::TransformSourceError::Recipe(cause) => cause.into(),
                    })?;
            // Unloaded floating destinations accept the admitted source precision.
            companions.affine_companion_dtype = metadata.dtype().clone();
            if recipes.insert(name.clone(), (recipe, companions)).is_some() {
                return Err(Error::Quantization(format!(
                    "selected materialization task {name:?} was bound more than once"
                )));
            }
            for output in std::iter::once(task.name()).chain(
                task.output_companions()
                    .iter()
                    .map(|companion| companion.name()),
            ) {
                let slot = parameters
                    .get(output)
                    .expect("validated companion identity");
                if destinations
                    .insert(output.to_owned(), slot.clone())
                    .is_some()
                {
                    return Err(Error::Quantization(format!(
                        "selected quantization output {output:?} is produced more than once"
                    )));
                }
            }
        }
    }
    validate_exact_consumption(&requested_names, &recipes)?;
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
    let working_set_bytes = bounded_quantization_working_set(store, &targets, quantization)?;
    Ok((
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        destinations,
    ))
}

fn exact_task_weight_companions(
    parameters: &BTreeMap<String, ParameterBindingTarget>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<BTreeMap<String, PackedWeightCompanions>, Error> {
    let mut selected = BTreeMap::new();
    for task in tasks {
        let Some(weight) = parameters.get(task.name()) else {
            continue;
        };
        let weight_dtype = &weight.dtype;
        if *weight_dtype != RecipeDtype::U32 {
            return Err(Error::Quantization(format!(
                "selected packed output {:?} has native dtype {weight_dtype:?}, expected U32",
                task.name()
            )));
        }
        let mut scales = None;
        let mut biases = None;
        let mut companion_dtype = None;
        for companion in task.output_companions() {
            let parameter = parameters.get(companion.name()).ok_or_else(|| {
                Error::Quantization(format!(
                    "selected materialization task {:?} names absent companion {:?}",
                    task.name(),
                    companion.name()
                ))
            })?;
            let dtype = match &parameter.dtype {
                RecipeDtype::F16 => RecipeDtype::F16,
                RecipeDtype::BF16 => RecipeDtype::BF16,
                RecipeDtype::F32 => RecipeDtype::F32,
                RecipeDtype::U8 if !quantization.has_biases() => RecipeDtype::F32,
                dtype => {
                    return Err(Error::Quantization(format!(
                        "selected companion {:?} has unsupported dtype {dtype:?}",
                        companion.name()
                    )))
                }
            };
            if companion_dtype
                .replace(dtype.clone())
                .is_some_and(|prior| prior != dtype)
            {
                return Err(Error::Quantization(format!(
                    "selected task {:?} has mismatched companion dtypes",
                    task.name()
                )));
            }
            match companion.role() {
                LinearCompanionRole::Scale => scales = Some(companion.name().to_owned()),
                LinearCompanionRole::AffineBias => biases = Some(companion.name().to_owned()),
            }
        }
        selected.insert(
            task.name().to_owned(),
            PackedWeightCompanions {
                weight_name: task.name().to_owned(),
                scales_name: scales.ok_or_else(|| {
                    Error::Quantization(format!(
                        "selected task {:?} has no declared scale companion",
                        task.name()
                    ))
                })?,
                biases_name: biases,
                affine_companion_dtype: companion_dtype.expect("scale companion supplies dtype"),
            },
        );
    }
    Ok(selected)
}

#[cfg(test)]
mod tests;
