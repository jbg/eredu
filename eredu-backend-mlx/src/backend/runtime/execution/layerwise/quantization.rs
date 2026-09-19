//! On-load quantization recipes, bounds, and task materialization.

use super::*;
mod exact_plan;
pub(crate) use exact_plan::prepare_exact_quantization_from_destinations;

/// Architecture-declared identities and scalar format of packed companions.
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
        fn visit(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a crate::MlxTensor,
        ) {
            if self.error.is_some() {
                return;
            }
            let name = metadata.id().as_str().to_owned();
            self.parameters
                .insert(name.clone(), value.as_array().dtype());
            if let Some(role) = metadata.linear_companion() {
                let Some(weight) = metadata.linear_companion_of() else {
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
    module.visit_parameters(&mut collector)?;
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

fn collect_exact_target_parameters<M: Parameterized<crate::MlxTensor>>(
    module: &M,
) -> Result<BTreeMap<String, eredu_runtime::ParameterBindingTarget>, Error> {
    #[derive(Default)]
    struct Collector {
        parameters: BTreeMap<String, eredu_runtime::ParameterBindingTarget>,
        invalid: Option<String>,
    }
    impl<'a> ParameterVisitor<'a, crate::MlxTensor> for Collector {
        fn visit(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a crate::MlxTensor,
        ) {
            let name = metadata.id().as_str().to_owned();
            match crate::backend::runtime::checkpoint::binding::mlx_parameter_binding_target(value)
            {
                Some(target) => {
                    self.parameters.insert(name, target);
                }
                None => self.invalid = Some(name),
            }
        }
    }
    let mut collector = Collector::default();
    module.visit_parameters(&mut collector)?;
    if let Some(name) = collector.invalid {
        return Err(Error::Quantization(format!(
            "invalid native quantization destination {name:?}"
        )));
    }
    Ok(collector.parameters)
}

fn validate_exact_quantization_sources<M>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    source: &M,
    destinations: &BTreeMap<String, eredu_runtime::ParameterBindingTarget>,
    plan: &BoundedQuantizationPlan,
) -> Result<(), Error>
where
    M: Parameterized<crate::MlxTensor>,
{
    struct SourceCollector {
        parameters: BTreeMap<String, (Vec<usize>, Dtype)>,
    }
    impl<'a> ParameterVisitor<'a, crate::MlxTensor> for SourceCollector {
        fn visit(
            &mut self,
            metadata: eredu_nn::ParameterMetadataView<'_>,
            value: &'a crate::MlxTensor,
        ) {
            let shape = value
                .as_array()
                .shape()
                .iter()
                .map(|&dimension| usize::try_from(dimension))
                .collect::<Result<Vec<_>, _>>();
            if let Ok(shape) = shape {
                self.parameters.insert(
                    metadata.id().as_str().to_owned(),
                    (shape, value.as_array().dtype()),
                );
            }
        }
    }
    let mut source_parameters = SourceCollector {
        parameters: BTreeMap::new(),
    };
    source.visit_parameters(&mut source_parameters)?;
    for target in plan
        .targets()
        .iter()
        .filter(|target| destinations.contains_key(target.weight_name()))
    {
        let name = target.weight_name();
        let (source_shape, source_dtype) =
            source_parameters.parameters.get(name).ok_or_else(|| {
                Error::Quantization(format!(
                    "selected materialization task {name:?} is absent from its source module"
                ))
            })?;
        let metadata = target.source().infer(store)?;
        if metadata.shape() != source_shape {
            return Err(Error::Quantization(format!(
                "selected materialization task {:?} source recipe has shape {:?}, native source module requires {:?}",
                name,
                metadata.shape(),
                source_shape
            )));
        }
        let native_source_dtype = match source_dtype {
            Dtype::Float16 => RecipeDtype::F16,
            Dtype::Bfloat16 => RecipeDtype::BF16,
            Dtype::Float32 => RecipeDtype::F32,
            dtype => {
                return Err(Error::Quantization(format!(
                "selected materialization task {:?} source module has unsupported dtype {dtype:?}",
                name
            )))
            }
        };
        if metadata.dtype() != &native_source_dtype {
            return Err(Error::Quantization(format!(
                "selected materialization task {:?} source recipe dtype {:?} differs from native source dtype {:?}",
                name,
                metadata.dtype(),
                native_source_dtype
            )));
        }
    }
    Ok(())
}

fn validate_exact_consumption(
    requested: &BTreeSet<&str>,
    recipes: &QuantizationRecipes,
) -> Result<(), Error> {
    let consumed = recipes.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let missing = requested.difference(&consumed).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::Quantization(format!(
            "selected materialization tasks were not consumed exactly once: {missing:?}"
        )));
    }
    let unselected = consumed.difference(requested).copied().collect::<Vec<_>>();
    if !unselected.is_empty() {
        return Err(Error::Quantization(format!(
            "exact materialization produced unselected targets: {unselected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod packed_weight_companion_tests;

/// Builds a quantized checkpoint overlay from neutral parameter topologies.
#[cfg(test)]
fn quantize_parameterized_store<SM, U, SF, TF>(
    store: RetainedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    mut source_unit: SF,
    mut target_unit: TF,
    unit_count: usize,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
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
    let transformed: RetainedCheckpointSource = transformed.into();
    Ok((transformed, report))
}

/// Builds a bounded packed overlay for one fully resident neutral module tree.
#[cfg(test)]
fn quantize_parameterized_module_store<M>(
    store: RetainedCheckpointSource,
    source: &M,
    target: &M,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
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
    store: RetainedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    mut source_unit: SF,
    mut target_unit: TF,
    unit_count: usize,
    quantization: WeightQuantization,
    stream: &Stream,
    static_bindings: SB,
    mut unit_bindings: UB,
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: Parameterized<crate::MlxTensor>,
    U: Parameterized<crate::MlxTensor>,
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
    let transformed: RetainedCheckpointSource = transformed.into();
    Ok((transformed, report))
}

/// Builds a bounded packed overlay for an exact set of selected replicated-text tasks.
///
/// Target identity, source provenance, recipe, format, and lowering come only
/// from `tasks`. Module traversal verifies the exact native source and output
/// handles named by those tasks; packed tensors outside the selected task set
/// are never added to the materialization plan.
#[allow(clippy::too_many_arguments)]
pub fn quantize_exact_replicated_text_tasks<SM, U>(
    store: RetainedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    source_units: &[U],
    target_units: &[U],
    source_layout: Option<&eredu_runtime::LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
    stream: &Stream,
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: Parameterized<crate::MlxTensor>,
    U: Parameterized<crate::MlxTensor>,
{
    let cold = prepare_exact_native_quantization(
        store,
        source_static,
        target_static,
        source_units,
        target_units,
        source_layout,
        quantization,
        tasks,
    )?;
    Ok(cold
        .allocate_ordinary()?
        .materialize_handoff(stream)?
        .into_parts())
}

/// Adopts the actual converted overlay after checking the selected native slots.
/// Metadata validation and identity comparison perform no payload reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adopt_exact_replicated_text_quantization<SM, U>(
    converted: crate::backend::runtime::checkpoint::bounded_quantization::ConvertedQuantization,
    store: RetainedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    source_units: &[U],
    target_units: &[U],
    source_layout: Option<&eredu_runtime::LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: Parameterized<crate::MlxTensor>,
    U: Parameterized<crate::MlxTensor>,
{
    let cold = prepare_exact_native_quantization(
        store,
        source_static,
        target_static,
        source_units,
        target_units,
        source_layout,
        quantization,
        tasks,
    )?;
    cold.validate_conversion(&converted)?;
    Ok(converted.into_parts())
}

#[allow(clippy::too_many_arguments)]
fn prepare_exact_native_quantization<SM, U>(
    store: RetainedCheckpointSource,
    source_static: &SM,
    target_static: &SM,
    source_units: &[U],
    target_units: &[U],
    source_layout: Option<&eredu_runtime::LocalModelLayout>,
    quantization: WeightQuantization,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<crate::backend::runtime::checkpoint::bounded_quantization::ColdQuantization, Error>
where
    SM: Parameterized<crate::MlxTensor>,
    U: Parameterized<crate::MlxTensor>,
{
    if source_units.len() != target_units.len() {
        return Err(Error::Quantization(
            "exact source and target materialization units differ in cardinality".into(),
        ));
    }
    let target_parameters = std::iter::once(collect_exact_target_parameters(target_static))
        .chain(target_units.iter().map(collect_exact_target_parameters))
        .collect::<Result<Vec<_>, _>>()?;
    let (plan, destinations) = exact_plan::build(
        store.as_ref(),
        &target_parameters,
        source_layout,
        quantization,
        tasks,
    )?;
    validate_exact_quantization_sources(
        store.as_ref(),
        source_static,
        &target_parameters[0],
        &plan,
    )?;
    for (index, source) in source_units.iter().enumerate() {
        validate_exact_quantization_sources(
            store.as_ref(),
            source,
            &target_parameters[index + 1],
            &plan,
        )?;
    }
    let cold =
        crate::backend::runtime::checkpoint::bounded_quantization::ColdQuantization::prepare(
            store, plan,
        )?;
    cold.validate_destinations(&destinations)?;
    Ok(cold)
}

/// Builds a bounded packed overlay from architecture-selected realtime tasks.
///
/// Source recipes and physical output names come exclusively from the selected
/// contract. Target modules are inspected only to verify that those exact
/// primary and companion handles exist with the native dtypes required by MLX.
pub fn quantize_exact_realtime_tasks<SM, U>(
    store: RetainedCheckpointSource,
    target_static: &SM,
    target_units: &[U],
    quantization: WeightQuantization,
    tasks: &[RealtimeMaterializationTask],
    stream: &Stream,
) -> Result<(RetainedCheckpointSource, WeightMaterializationReport), Error>
where
    SM: Parameterized<crate::MlxTensor>,
    U: Parameterized<crate::MlxTensor>,
{
    let mut available = packed_weight_companions(target_static, quantization)?;
    for unit in target_units {
        for (target, companions) in packed_weight_companions(unit, quantization)? {
            if available.insert(target.clone(), companions).is_some() {
                return Err(Error::Quantization(format!(
                    "selected realtime packed target {target:?} appears in more than one module"
                )));
            }
        }
    }
    let mut recipes = BTreeMap::new();
    for task in tasks.iter().filter(|task| {
        matches!(
            task.lowering().kind(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        )
    }) {
        let target = task.lowering().target().as_str();
        if task
            .lowering()
            .descriptor()
            .executable()
            .weight_quantization()
            != Some(quantization)
        {
            return Err(Error::Quantization(format!(
                "selected realtime transform {target:?} does not match its executable format"
            )));
        }
        let companions = available.remove(target).ok_or_else(|| {
            Error::Quantization(format!(
                "selected realtime transform {target:?} has no native packed target"
            ))
        })?;
        let scale = task.lowering().scale().ok_or_else(|| {
            Error::Quantization(format!(
                "selected realtime transform {target:?} has no scale component"
            ))
        })?;
        if scale.target().as_str() != companions.scales_name {
            return Err(Error::Quantization(format!(
                "selected realtime transform {target:?} scale identity differs from native target"
            )));
        }
        let selected_bias = task
            .lowering()
            .affine_bias()
            .map(|component| component.target().as_str());
        if selected_bias != companions.biases_name.as_deref() {
            return Err(Error::Quantization(format!(
                "selected realtime transform {target:?} bias identity differs from native target"
            )));
        }
        let primary = task.lowering().primary();
        let recipe = primary.recipe().ok_or_else(|| {
            Error::Quantization(format!(
                "selected realtime transform {target:?} has no source recipe"
            ))
        })?;
        if recipes
            .insert(target.to_owned(), (recipe.clone(), companions))
            .is_some()
        {
            return Err(Error::Quantization(format!(
                "selected realtime transform {target:?} appears more than once"
            )));
        }
    }
    if recipes.is_empty() {
        return Err(Error::Quantization(
            "selected realtime transform contains no packed matrix tasks".into(),
        ));
    }
    if !available.is_empty() {
        return Err(Error::Quantization(format!(
            "native packed modules contain unselected realtime targets: {:?}",
            available.keys().collect::<Vec<_>>()
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
        .collect::<Result<Vec<_>, Error>>()?;
    let working_set_bytes =
        bounded_quantization_working_set(store.as_ref(), &targets, quantization)?;
    let transformed = Arc::new(BoundedQuantizedWeightStore::create(
        store,
        BoundedQuantizationPlan::new(quantization, working_set_bytes, targets)?,
        stream,
    )?);
    let report = transformed.report().clone();
    let transformed: RetainedCheckpointSource = transformed.into();
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
        let live_output_row = output_row
            .checked_add(target.companion_cast_source_row_bytes(
                quantization,
                metadata.dtype(),
                columns,
            )?)
            .ok_or_else(|| Error::Quantization("live packed output row size overflowed".into()))?;
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
            minimum_tile_bytes = minimum_tile_bytes.max(
                one_row
                    .peak_materialization_bytes(store)?
                    .checked_add(live_output_row)
                    .ok_or_else(|| Error::Quantization("conversion tile size overflowed".into()))?,
            );
        }
    }
    Ok(output_bytes.max(minimum_tile_bytes))
}
