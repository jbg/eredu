//! Pure Nemotron-H SafeTensors and GGUF checkpoint plans.

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{StoredDtype, WeightQuantization};

use super::{LayerPolicy, ModelArgs};
use eredu_nn::DeclareLogicalParameters;

/// Derives a Nemotron-H configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Nemotron-H
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    formats: std::collections::HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Resolves all released Nemotron-H names to canonical runtime names.
pub fn normalized_checkpoint_keys(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    args: &ModelArgs,
) -> Result<BTreeMap<String, String>, String> {
    let mut normalized = BTreeMap::new();
    for raw in store.source_keys() {
        normalized.insert(normalized_parameter_key(&raw, args)?, raw);
    }
    Ok(normalized)
}

/// Resolves one released checkpoint identity through the selected Nemotron-H schedule.
///
/// This only names parameters; numerical storage transforms remain recipe-owned.
pub fn normalized_parameter_key(raw: &str, args: &ModelArgs) -> Result<String, String> {
    let mtp = args.mtp_policies().map_err(|error| error.to_string())?;
    let runtime = if let Some(rest) = raw.strip_prefix("backbone.embeddings.") {
        format!("model.embeddings.{rest}")
    } else if let Some(rest) = raw.strip_prefix("backbone.norm_f.") {
        format!("model.norm_f.{rest}")
    } else if let Some(rest) = raw.strip_prefix("backbone.layers.") {
        let (layer_text, suffix) = rest
            .split_once('.')
            .ok_or_else(|| format!("invalid Nemotron-H checkpoint key {raw:?}"))?;
        let layer = layer_text
            .parse::<usize>()
            .map_err(|error| format!("invalid Nemotron-H layer in {raw:?}: {error}"))?;
        if let Some(suffix) = suffix.strip_prefix("mixer.") {
            let field = match args.layer_schedule.get(layer) {
                Some(LayerPolicy::Mamba) => "mamba",
                Some(LayerPolicy::SelfAttention(_)) => "attention",
                Some(LayerPolicy::DenseMlp) => "mlp",
                Some(LayerPolicy::SparseMoe) => "moe",
                None => {
                    return Err(format!(
                        "checkpoint layer {layer} is outside Nemotron-H schedule"
                    ))
                }
            };
            format!("model.layers.{layer}.{field}.{suffix}")
        } else {
            format!("model.layers.{layer}.{suffix}")
        }
    } else if let Some(rest) = raw
        .strip_prefix("mtp.layers.")
        .or_else(|| raw.strip_prefix("model.mtp.layers."))
    {
        let (layer_text, suffix) = rest
            .split_once('.')
            .ok_or_else(|| format!("invalid Nemotron-H MTP checkpoint key {raw:?}"))?;
        let physical = layer_text
            .parse::<usize>()
            .map_err(|error| format!("invalid Nemotron-H MTP layer in {raw:?}: {error}"))?;
        if physical >= mtp.len() {
            return Err(format!(
                "checkpoint MTP layer {physical} is outside Nemotron-H schedule"
            ));
        }
        format!("model.mtp.layers.{physical}.{suffix}")
    } else {
        raw.to_owned()
    };
    Ok(runtime)
}

/// Returns canonical recipes for selected Nemotron-H static modules.
pub fn static_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    args: &ModelArgs,
    root: Option<&str>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    Ok(normalized_checkpoint_keys(store, args)?
        .into_iter()
        .filter(|(runtime, raw)| {
            runtime != raw
                && root.is_none_or(|root| runtime.starts_with(root))
                && (runtime.starts_with("model.embeddings.")
                    || runtime.starts_with("model.norm_f.")
                    || runtime.starts_with("lm_head."))
        })
        .map(|(runtime, raw)| {
            (
                runtime,
                DerivedWeightRecipe::source(raw, TensorSelection::Full),
            )
        })
        .collect())
}

fn expert_source_root(bank: &eredu_nn::GroupedRelu2Spec) -> Result<&str, String> {
    bank.up()
        .weight()
        .id
        .as_str()
        .strip_suffix(".up_proj")
        .ok_or_else(|| "Nemotron-H expert bank has no declared source root".into())
}

fn expert_binding_target(
    bank: &eredu_nn::GroupedRelu2Spec,
    binding: &str,
) -> Result<String, String> {
    let (projection, suffix) = binding
        .strip_prefix("up_proj")
        .map(|suffix| (bank.up(), suffix))
        .or_else(|| {
            binding
                .strip_prefix("down_proj")
                .map(|suffix| (bank.down(), suffix))
        })
        .filter(|(_, suffix)| matches!(*suffix, "" | "_scales" | "_biases"))
        .ok_or_else(|| format!("undeclared Nemotron-H expert binding {binding}"))?;
    Ok(format!("{}{suffix}", projection.weight().id))
}

/// Returns the complete recipe catalog for one Nemotron-H execution unit.
pub fn unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &super::Declaration,
    group: usize,
    index: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let definition = declaration.definition();
    let identity = definition
        .parameters()
        .unit_layout()
        .ordinal(group, index)
        .ok_or_else(|| format!("Nemotron-H has no unit {index} in group {group}"))?;
    let root = definition
        .unit(group, index)
        .map_err(|e| e.to_string())?
        .path
        .clone();
    let scope = declaration
        .block(identity)
        .ok_or("unresolved Nemotron invocation scope")?;
    let normalized = normalized_checkpoint_keys(store, args)?;
    let unit_prefix = format!("{root}.");
    let mut recipes = normalized
        .iter()
        .filter(|(runtime, raw)| runtime.starts_with(&unit_prefix) && *runtime != *raw)
        .map(|(runtime, raw)| {
            (
                runtime.clone(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if let super::declarations::TargetOperatorParameters::Mamba(spec) = &scope.operator {
        let a_log = format!("{root}.mamba.A_log");
        if let Some(raw) = normalized.get(&a_log) {
            recipes.insert(
                a_log,
                DerivedWeightRecipe::NegLog {
                    input: Box::new(DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Full,
                    )),
                },
            );
        }
        let conv = format!("{root}.mamba.conv1d.weight");
        if let Some(raw) = normalized.get(&conv) {
            let recipe = DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full);
            let expected = spec
                .convolution
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|d| d.parameter.id == spec.convolution.weight.id)
                .expect("declared convolution weight")
                .shape;
            if recipe
                .infer(store)
                .map_err(|error| error.to_string())?
                .shape()
                != expected
            {
                recipes.insert(
                    conv,
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: expected,
                    },
                );
            }
        }
    }
    let super::declarations::TargetOperatorParameters::Sparse(spec) = &scope.operator else {
        return Ok(recipes);
    };
    let bank = &spec.experts;
    let expert_root = expert_source_root(&bank)?;
    if !include_experts {
        let expert_prefix = format!("{expert_root}.");
        recipes.retain(|name, _| !name.starts_with(&expert_prefix));
        return Ok(recipes);
    }
    // Raw packed and split names are recipe inputs. Only the typed bank's
    // canonical outputs are published, including its encoding companions.
    recipes.retain(|name, _| !name.starts_with(&format!("{expert_root}.")));
    for (binding, recipe) in bank_recipes(&normalized, &bank, None)? {
        let target = expert_binding_target(&bank, &binding)?;
        if matches!(
            &recipe,
            DerivedWeightRecipe::Source { key, selection: TensorSelection::Full } if key == &target
        ) {
            continue;
        }
        recipes.insert(target, recipe);
    }
    Ok(recipes)
}

/// Returns the unit recipe catalog for one flat target/MTP residency ordinal.
pub fn unit_recipes_flat(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &super::Declaration,
    flat: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let layout = declaration.definition().parameters().unit_layout();
    let address = layout
        .address(flat)
        .ok_or_else(|| format!("Nemotron-H flat unit {flat} is outside its declared schedule"))?;
    unit_recipes(
        store,
        declaration,
        address.group(),
        address.index(),
        include_experts,
    )
}

/// Returns canonical lazy-loading recipes for one Nemotron-H routed expert.
pub fn expert_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &super::Declaration,
    identity_layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    let bank = declaration
        .expert_bank(identity_layer)
        .map_err(|error| error.to_string())?;
    expert_recipes_for_bank(store, args, identity_layer, expert, &bank)
}

fn expert_recipes_for_bank(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    args: &ModelArgs,
    identity_layer: usize,
    expert: usize,
    bank: &eredu_nn::GroupedRelu2Spec,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if expert >= bank.group_count() as usize {
        return Err(format!(
            "Nemotron-H has no expert {expert} in layer {identity_layer}"
        ));
    }
    let normalized = normalized_checkpoint_keys(store, args)?;
    bank_recipes(&normalized, bank, Some(expert))
}

fn bank_recipes(
    normalized: &BTreeMap<String, String>,
    bank: &eredu_nn::GroupedRelu2Spec,
    expert: Option<usize>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let prefix = expert_source_root(bank)?;
    let packed = normalized.contains_key(bank.up().weight().id.as_str());
    let range = expert.map_or(0..bank.group_count() as usize, |expert| expert..expert + 1);
    let mut recipes = BTreeMap::new();
    for (projection, spec) in [("up_proj", bank.up()), ("down_proj", bank.down())] {
        let primary = spec.weight().id.as_str();
        for (binding_suffix, source_suffix, present) in [
            ("", "weight", true),
            ("_scales", "scales", spec.format().scale().is_some()),
            ("_biases", "biases", spec.format().affine_bias().is_some()),
        ] {
            if !present {
                continue;
            }
            let recipe = if packed {
                let source = if binding_suffix.is_empty() {
                    primary.to_owned()
                } else {
                    format!("{primary}.{source_suffix}")
                };
                let raw = normalized
                    .get(&source)
                    .ok_or_else(|| format!("missing packed Nemotron-H tensor {source}"))?;
                DerivedWeightRecipe::source(
                    raw.clone(),
                    expert.map_or(TensorSelection::Full, |_| TensorSelection::Range {
                        axis: 0,
                        start: range.start,
                        end: range.end,
                    }),
                )
            } else {
                let inputs = range
                    .clone()
                    .map(|expert| {
                        let source = format!("{prefix}.{expert}.{projection}.{source_suffix}");
                        normalized
                            .get(&source)
                            .cloned()
                            .map(|raw| DerivedWeightRecipe::source(raw, TensorSelection::Full))
                            .ok_or_else(|| format!("missing split Nemotron-H tensor {source}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                DerivedWeightRecipe::Stack { axis: 0, inputs }
            };
            recipes.insert(format!("{projection}{binding_suffix}"), recipe);
        }
    }
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &super::Declaration,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let args = declaration.args();
    args.validate().map_err(|error| error.to_string())?;
    if !args.has_sparse_moe_layers() {
        return Err("Nemotron-H expert residency requires sparse MoE units".into());
    }
    let layout = declaration.definition().parameters().unit_layout();
    let banks = declaration.expert_banks();
    let capacity = banks.values().try_fold(0usize, |total, bank| {
        total
            .checked_add(bank.group_count() as usize)
            .ok_or_else(|| "Nemotron-H expert residency catalog size overflowed".to_string())
    })?;
    let mut units = Vec::with_capacity(capacity);

    for (identity_layer, bank) in banks {
        let address = layout
            .address(identity_layer)
            .ok_or_else(|| "unresolved Nemotron-H expert invocation".to_string())?;
        let owner_group = layout
            .group_id(address.group())
            .ok_or_else(|| "unresolved Nemotron-H expert owner".to_string())?
            .clone();
        let unit_path = declaration
            .definition()
            .unit(address.group(), address.index())
            .map_err(|error| error.to_string())?
            .path
            .clone();
        add_expert_residency_units(
            &mut units,
            store,
            args,
            SparseUnitResidency {
                identity_layer,
                owner_group,
                owner_unit: address.index(),
                unit_path,
            },
            &bank,
        )?;
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(store))
        .map_err(|error| error.to_string())
}

struct SparseUnitResidency {
    identity_layer: usize,
    owner_group: eredu_runtime::ExecutionGroupId,
    owner_unit: usize,
    unit_path: String,
}

fn add_expert_residency_units(
    units: &mut Vec<crate::ExpertResidencyUnit>,
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    args: &ModelArgs,
    topology: SparseUnitResidency,
    bank: &eredu_nn::GroupedRelu2Spec,
) -> Result<(), String> {
    let owner_group = topology.owner_group;
    for expert in 0..bank.group_count() as usize {
        let recipes = expert_recipes_for_bank(store, args, topology.identity_layer, expert, bank)?;
        let up_quantizable =
            !recipes.contains_key("up_proj_scales") && !recipes.contains_key("up_proj_biases");
        let down_quantizable =
            !recipes.contains_key("down_proj_scales") && !recipes.contains_key("down_proj_biases");
        let parameters = recipes
            .into_iter()
            .map(|(binding, recipe)| {
                let target = expert_binding_target(bank, &binding)?;
                let role = match binding.as_str() {
                    "up_proj" if up_quantizable => {
                        crate::ExpertParameterRole::quantizable_projection(
                            "up_proj_scales",
                            "up_proj_biases",
                        )
                    }
                    "down_proj" if down_quantizable => {
                        crate::ExpertParameterRole::quantizable_projection(
                            "down_proj_scales",
                            "down_proj_biases",
                        )
                    }
                    _ => crate::ExpertParameterRole::Preserved,
                };
                crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        units.push(
            crate::ExpertResidencyUnit::new(
                eredu_runtime::ParameterBankKey::new(0, topology.identity_layer, expert),
                owner_group.clone(),
                topology.owner_unit,
                &topology.unit_path,
                crate::ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .map_err(|error| error.to_string())?,
        );
    }
    Ok(())
}

fn logical_shape(
    spec: &impl DeclareLogicalParameters,
    slot: &eredu_nn::ParameterSpec,
) -> Result<Vec<usize>, String> {
    spec.logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|d| d.parameter.id == slot.id)
        .map(|d| d.shape)
        .ok_or_else(|| format!("unresolved logical parameter {}", slot.id))
}

pub(super) fn safetensors_plan_from_declaration(
    declaration: &super::Declaration,
) -> Result<SafetensorsCheckpointPlan, String> {
    let args = declaration.args();
    let statics = &declaration.statics;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    for declaration in statics
        .embedding_spec()
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        add_safe_declaration(&mut common, "backbone.embeddings.weight", &declaration)?;
    }
    add_safe_normalization(
        &mut common,
        "backbone.norm_f.weight",
        statics.normalization_spec(),
    )?;
    if let Some(head) = statics.head_spec() {
        for declaration in head.logical_parameters().map_err(|e| e.to_string())? {
            add_safe_declaration(&mut common, declaration.parameter.id.as_str(), &declaration)?;
        }
    }

    for (layer, block) in declaration.blocks.iter().enumerate() {
        let official = format!("backbone.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        add_safe_normalization(&mut common, &format!("{official}.norm.weight"), &block.norm)?;
        add_safe_layer(
            args,
            block,
            false,
            &official,
            &canonical,
            &format!("layer {layer}"),
            &mut common,
            &mut groups,
        )?;
    }

    if args.num_nextn_predict_layers > 0 {
        let (_, _, layout) =
            super::parameters::execution_schedule(args).map_err(|error| error.to_string())?;
        for (layer, parameters) in declaration.predictions.iter().enumerate() {
            let official = format!("mtp.layers.{layer}");
            let canonical = format!("model.mtp.layers.{layer}");
            add_safe_normalization(
                &mut common,
                &format!("{official}.norm.weight"),
                &parameters.block.norm,
            )?;
            add_safe_layer(
                args,
                &parameters.block,
                true,
                &official,
                &canonical,
                &format!("MTP physical layer {layer}"),
                &mut common,
                &mut groups,
            )?;
        }
        for group in 1..layout.group_count() {
            let range = layout
                .group_range(group)
                .ok_or("unresolved Nemotron-H prediction range")?;
            for relative in 0..range.len() {
                let physical = layout
                    .ordinal(group, relative)
                    .and_then(|ordinal| ordinal.checked_sub(args.num_hidden_layers as usize))
                    .ok_or("unresolved prediction invocation")?;
                let parameters = &declaration.predictions[physical];
                for declaration in parameters
                    .boundary_parameters()
                    .map_err(|error| error.to_string())?
                {
                    let canonical = declaration.parameter.id.as_str();
                    let official = canonical.strip_prefix("model.").ok_or_else(|| {
                        format!("invalid Nemotron-H prediction identity {canonical}")
                    })?;
                    add_safe_declaration(&mut common, official, &declaration)?;
                }
            }
        }
    }

    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("Nemotron-H SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

fn add_safe_layer(
    args: &ModelArgs,
    block: &super::declarations::TargetBlockParameters,
    prediction: bool,
    official: &str,
    canonical: &str,
    label: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    use super::declarations::TargetOperatorParameters as P;
    match &block.operator {
        P::Mamba(spec) => add_safe_mamba(spec, official, canonical, common),
        P::Attention(spec) => add_safe_attention(
            spec,
            official,
            canonical,
            if prediction { "mixer" } else { "attention" },
            common,
        ),
        P::Dense(spec) => add_safe_dense(spec, official, canonical, common),
        P::Sparse(spec) => add_safe_moe(
            args,
            spec,
            official,
            canonical,
            if prediction { "mixer" } else { "moe" },
            label,
            common,
            groups,
        ),
    }
}

fn add_safe_mamba(
    spec: &super::mamba::Mamba2Spec,
    official: &str,
    canonical: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let prefix = format!("{canonical}.mamba");
    let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
    declarations.sort_by_key(|d| {
        mamba_storage_order(
            d.parameter
                .id
                .as_str()
                .strip_prefix(&format!("{prefix}."))
                .expect("declared mixer prefix"),
        )
    });
    for declaration in declarations {
        let target = declaration.parameter.id.as_str();
        let field = target
            .strip_prefix(&format!("{prefix}."))
            .expect("declared mixer prefix");
        add_safe_declaration(output, &format!("{official}.mixer.{field}"), &declaration)?;
    }
    Ok(())
}

// Stable storage order is independent of the construction visitor's traversal order.
fn mamba_storage_order(field: &str) -> usize {
    [
        "in_proj.weight",
        "conv1d.weight",
        "dt_bias",
        "A_log",
        "D",
        "norm.weight",
        "out_proj.weight",
        "conv1d.bias",
        "in_proj.bias",
        "out_proj.bias",
    ]
    .iter()
    .position(|&candidate| candidate == field)
    .expect("declared Mamba slot")
}

fn add_safe_attention(
    spec: &super::attention::AttentionParameters,
    official: &str,
    canonical: &str,
    canonical_field: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let prefix = format!("{canonical}.{canonical_field}");
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let target = declaration.parameter.id.as_str();
        let field = target
            .strip_prefix(&format!("{prefix}."))
            .expect("declared attention prefix");
        add_safe_declaration(output, &format!("{official}.mixer.{field}"), &declaration)?;
    }
    Ok(())
}

fn add_safe_dense(
    spec: &super::mlp::DenseParameters,
    official: &str,
    canonical: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let prefix = format!("{canonical}.mlp");
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let target = declaration.parameter.id.as_str();
        let field = target
            .strip_prefix(&format!("{prefix}."))
            .expect("declared MLP prefix");
        add_safe_declaration(output, &format!("{official}.mixer.{field}"), &declaration)?;
    }
    Ok(())
}

fn add_safe_moe(
    args: &ModelArgs,
    spec: &super::mlp::SparseMoeSpec,
    official: &str,
    canonical: &str,
    canonical_field: &str,
    label: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let prefix = format!("{canonical}.{canonical_field}");
    let canonical_experts = format!("{prefix}.experts");
    let selector = &spec.selector;
    let shared = &spec.shared;
    let bank = &spec.experts;
    let mut ordinary = selector.logical_parameters().map_err(|e| e.to_string())?;
    for role in [
        eredu_nn::LogicalParameterRole::LinearWeight,
        eredu_nn::LogicalParameterRole::LinearBias,
    ] {
        for spec in shared.network.projections() {
            ordinary.extend(
                spec.logical_parameters()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .filter(|d| d.role == role),
            );
        }
    }
    for declaration in ordinary {
        let target = declaration.parameter.id.as_str();
        let field = target
            .strip_prefix(&format!("{prefix}."))
            .expect("declared MoE prefix");
        add_safe_declaration(common, &format!("{official}.mixer.{field}"), &declaration)?;
    }
    let official_experts = format!("{official}.mixer.experts");
    let declarations = bank.logical_parameters().map_err(|e| e.to_string())?;
    let mut packed = Vec::new();
    for declaration in &declarations {
        let target = declaration.parameter.id.as_str();
        let field = target
            .strip_prefix(&format!("{canonical_experts}."))
            .expect("declared expert prefix");
        add_safe_declaration(
            &mut packed,
            &format!("{official_experts}.{field}"),
            declaration,
        )?;
    }
    let mut split = Vec::new();
    visit_split_expert_sources(bank, |declaration, target, shape| {
        let suffix = target
            .strip_prefix(&canonical_experts)
            .expect("declared independent expert root");
        let format = declaration
            .linear_format
            .map(|_| args.weight_quantization_for(&target).into());
        split.extend(safe_alias_constraints(
            &format!("{official_experts}{suffix}"),
            &target,
            shape,
            format,
        )?);
        Ok(())
    })?;
    groups.push(AlternativeLayoutGroup {
        id: format!("Nemotron-H {label} routed experts"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                discriminator_keys: packed
                    .iter()
                    .filter(|tensor| tensor.role == eredu_checkpoint::schema::TensorRole::Tensor)
                    .map(|tensor| tensor.key.clone())
                    .collect(),
                tensors: packed,
            },
            LayoutVariant {
                id: "split".into(),
                discriminator_keys: split
                    .iter()
                    .filter(|tensor| tensor.role == eredu_checkpoint::schema::TensorRole::Tensor)
                    .map(|tensor| tensor.key.clone())
                    .collect(),
                tensors: split,
            },
        ],
    });
    Ok(())
}

/// Projects source matrix geometry by joining released identities to retained
/// logical slots. Format constraints select source names, never logical widths.
pub(crate) fn source_linear_shapes(
    declaration: &super::Declaration,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let args = declaration.args();
    let mut logical = declaration
        .definition()
        .parameters()
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| {
            matches!(
                member.logical_role(),
                Some(
                    eredu_nn::LogicalParameterRole::LinearWeight
                        | eredu_nn::LogicalParameterRole::Embedding
                )
            )
        })
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target().to_owned(), shape.to_vec()))
        })
        .collect::<BTreeMap<_, _>>();
    for bank in declaration.expert_banks().values() {
        visit_split_expert_sources(bank, |_, target, shape| {
            logical.insert(target, shape);
            Ok(())
        })?;
    }
    let plan = declaration.safetensors_plan()?;
    let mut sources = BTreeMap::new();
    for constraint in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .flat_map(|group| &group.variants)
            .flat_map(|variant| &variant.tensors),
    ) {
        let canonical = normalized_parameter_key(&constraint.key, args)?;
        if let Some(shape) = logical.get(&canonical) {
            sources.insert(constraint.key.clone(), shape.clone());
        }
    }
    Ok(sources)
}

fn visit_split_expert_sources(
    bank: &eredu_nn::GroupedRelu2Spec,
    mut visit: impl FnMut(
        &eredu_nn::LogicalParameterDeclaration<'_>,
        String,
        Vec<usize>,
    ) -> Result<(), String>,
) -> Result<(), String> {
    let root = expert_source_root(bank)?;
    let declarations = bank.logical_parameters().map_err(|e| e.to_string())?;
    for expert in 0..bank.group_count() as usize {
        for declaration in &declarations {
            let field = declaration
                .parameter
                .id
                .as_str()
                .strip_prefix(root)
                .ok_or("Nemotron-H expert projection differs from its declared root")?;
            visit(
                declaration,
                format!("{root}.{expert}{field}.weight"),
                declaration.shape[1..].to_vec(),
            )?;
        }
    }
    Ok(())
}

/// Source aliases borrow exact geometry and encoding eligibility from the neural slot.
pub(super) fn add_safe_declaration(
    output: &mut Vec<SafetensorsTensorConstraint>,
    official: &str,
    declaration: &eredu_nn::LogicalParameterDeclaration<'_>,
) -> Result<(), String> {
    output.extend(safe_alias_constraints(
        official,
        declaration.parameter.id.as_str(),
        declaration.shape.clone(),
        declaration.linear_format.map(|format| format.encoding()),
    )?);
    Ok(())
}

fn add_safe_normalization(
    output: &mut Vec<SafetensorsTensorConstraint>,
    official: &str,
    spec: &eredu_nn::NormalizationConstructionSpec,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        add_safe_declaration(output, official, &declaration)?;
    }
    Ok(())
}

fn safe_alias_constraints(
    official: &str,
    canonical: &str,
    shape: Vec<usize>,
    format: Option<eredu_checkpoint::LinearFormat>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let mut names = vec![official.to_string(), canonical.to_string()];
    names.sort();
    names.dedup();
    let primary = official.to_string();
    let aliases = names
        .iter()
        .filter(|name| **name != primary)
        .cloned()
        .collect::<Vec<_>>();
    let Some(quantization) = format.and_then(eredu_checkpoint::LinearFormat::weight_quantization)
    else {
        let mut tensor =
            SafetensorsTensorConstraint::required(primary, shape, StoredDtypeConstraint::Floating)
                .with_aliases(aliases);
        tensor.linear_format = format;
        return Ok(vec![tensor]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("Nemotron-H matrix {canonical:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized tensor {canonical:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed_shape = shape.clone();
    *packed_shape.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        primary,
        packed_shape,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases(aliases)
    .with_linear_format(quantization.into())];
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let prefixes = names
        .iter()
        .map(|name| outer_prefix(name).to_string())
        .collect::<BTreeSet<_>>();
    for (suffix, required) in [("scales", true), ("biases", quantization.has_biases())] {
        if !required {
            continue;
        }
        let primary_companion = format!("{}.{}", outer_prefix(official), suffix);
        let aliases = prefixes
            .iter()
            .map(|prefix| format!("{prefix}.{suffix}"))
            .filter(|name| name != &primary_companion)
            .collect::<Vec<_>>();
        constraints.push(
            SafetensorsTensorConstraint::required(
                primary_companion,
                companion_shape.clone(),
                companion_dtype(),
            )
            .with_aliases(aliases)
            .linear_companion(
                official,
                if suffix == "scales" {
                    eredu_checkpoint::schema::LinearCompanionKind::Scale
                } else {
                    eredu_checkpoint::schema::LinearCompanionKind::AffineBias
                },
            ),
        );
    }
    Ok(constraints)
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

pub(super) fn gguf_plan_from_declaration(
    declaration: &super::Declaration,
) -> Result<GgufCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            logical_shape(statics.embedding_spec(), &statics.embedding_spec().weight)?,
            TensorOperation::Matrix,
        ),
        gguf(
            "output_norm.weight",
            statics
                .normalization_spec()
                .logical_parameters()
                .map_err(|e| e.to_string())?[0]
                .shape
                .clone(),
            TensorOperation::Vector,
        ),
    ];
    if let Some(head) = statics.head_spec() {
        tensors.push(gguf(
            "output.weight",
            logical_shape(head, &head.weight)?,
            TensorOperation::Matrix,
        ));
    }
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        let block = format!("blk.{layer}");
        tensors.push(gguf(
            format!("{block}.attn_norm.weight"),
            scope.norm.logical_parameters().map_err(|e| e.to_string())?[0]
                .shape
                .clone(),
            TensorOperation::Vector,
        ));
        use super::declarations::TargetOperatorParameters as P;
        match &scope.operator {
            P::Mamba(spec) => add_gguf_mamba(spec, &block, &mut tensors)?,
            P::Attention(spec) => add_gguf_attention(spec, &block, &mut tensors)?,
            P::Dense(spec) => add_gguf_dense(spec, &block, &mut tensors)?,
            P::Sparse(spec) => add_gguf_moe(spec, &block, &mut tensors)?,
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new("Nemotron-H", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

fn add_gguf_mamba(
    spec: &super::mamba::Mamba2Spec,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let prefix = format!("{}.mamba", gguf_unit_root(block)?);
    let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
    declarations.sort_by_key(|d| {
        mamba_storage_order(
            d.parameter
                .id
                .as_str()
                .strip_prefix(&format!("{prefix}."))
                .expect("declared mixer prefix"),
        )
    });
    for declaration in declarations {
        let field = declaration
            .parameter
            .id
            .as_str()
            .strip_prefix(&format!("{prefix}."))
            .expect("declared mixer prefix");
        let name = gguf_unit_parameter_name(declaration.parameter.id.as_str())?;
        let mut shape = declaration.shape;
        let operation = match field {
            "in_proj.weight" | "out_proj.weight" => TensorOperation::Matrix,
            "conv1d.weight" => {
                // GGUF stores the depthwise kernel without the singleton input-channel axis.
                shape.remove(1);
                TensorOperation::Dense
            }
            "A_log" => TensorOperation::Dense,
            _ => TensorOperation::Vector,
        };
        output.push(gguf(format!("{block}.{name}"), shape, operation));
    }
    Ok(())
}

fn add_gguf_attention(
    spec: &super::attention::AttentionParameters,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let matrix = declaration.role == eredu_nn::LogicalParameterRole::LinearWeight;
        output.push(gguf(
            format!(
                "{block}.{}",
                gguf_unit_parameter_name(declaration.parameter.id.as_str())?
            ),
            declaration.shape,
            if matrix {
                TensorOperation::Matrix
            } else {
                TensorOperation::Vector
            },
        ));
    }
    Ok(())
}

fn add_gguf_dense(
    spec: &super::mlp::DenseParameters,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    // GGUF orders weights before optional biases.
    for role in [
        eredu_nn::LogicalParameterRole::LinearWeight,
        eredu_nn::LogicalParameterRole::LinearBias,
    ] {
        for spec in spec.network.projections() {
            for declaration in spec
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|d| d.role == role)
            {
                let matrix = role == eredu_nn::LogicalParameterRole::LinearWeight;
                output.push(gguf(
                    format!(
                        "{block}.{}",
                        gguf_unit_parameter_name(declaration.parameter.id.as_str())?
                    ),
                    declaration.shape,
                    if matrix {
                        TensorOperation::Matrix
                    } else {
                        TensorOperation::Vector
                    },
                ));
            }
        }
    }
    Ok(())
}

fn add_gguf_moe(
    spec: &super::mlp::SparseMoeSpec,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let selector = &spec.selector;
    let shared = &spec.shared;
    let bank = &spec.experts;
    let mut declarations = selector.logical_parameters().map_err(|e| e.to_string())?;
    for spec in shared.network.projections() {
        declarations.extend(
            spec.logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|d| d.role == eredu_nn::LogicalParameterRole::LinearWeight),
        );
    }
    declarations.extend(bank.logical_parameters().map_err(|e| e.to_string())?);
    for spec in shared.network.projections() {
        declarations.extend(
            spec.logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|d| d.role == eredu_nn::LogicalParameterRole::LinearBias),
        );
    }
    for declaration in declarations {
        let id = declaration.parameter.id.as_str();
        let name = gguf_unit_parameter_name(id)?;
        let mut tensor = gguf(
            format!("{block}.{name}"),
            declaration.shape,
            if declaration.role == eredu_nn::LogicalParameterRole::LinearWeight {
                TensorOperation::Matrix
            } else {
                TensorOperation::Vector
            },
        );
        if name == "exp_probs_b.bias" {
            tensor = tensor.with_aliases([format!("{block}.ffn_exp_probs_b.bias")]);
        }
        output.push(tensor);
    }
    Ok(())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Nemotron-H {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Nemotron-H {name} geometry overflows"))
}

const GGUF_MOE_PARAMETERS: [(&str, &str); 7] = [
    ("ffn_gate_inp", "gate"),
    ("exp_probs_b", "gate.e_score_correction_bias"),
    ("ffn_up_exps", "experts.up_proj"),
    ("ffn_down_exps", "experts.down_proj"),
    ("ffn_up_shexp", "shared_experts.up_proj"),
    ("ffn_down_shexp", "shared_experts.down_proj"),
    ("ffn_exp_probs_b", "gate.e_score_correction_bias"),
];

const GGUF_PARAMETERS: [(&str, &str); 16] = [
    ("attn_norm", "norm"),
    ("attn_q", "attention.q_proj"),
    ("attn_k", "attention.k_proj"),
    ("attn_v", "attention.v_proj"),
    ("attn_output", "attention.o_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ssm_in", "mamba.in_proj"),
    ("ssm_conv1d", "mamba.conv1d"),
    ("ssm_dt.bias", "mamba.dt_bias"),
    ("ssm_a", "mamba.A_log"),
    ("ssm_d", "mamba.D"),
    ("ssm_norm", "mamba.norm"),
    ("ssm_out", "mamba.out_proj"),
    ("rope_freqs", "rope_freqs"),
    ("ffn_norm", "ffn_norm"),
];

fn gguf_unit_root(block: &str) -> Result<String, String> {
    let layer = block
        .strip_prefix("blk.")
        .ok_or_else(|| format!("unresolved GGUF block {block}"))?;
    Ok(format!("model.layers.{layer}"))
}

fn gguf_unit_parameter_name(canonical: &str) -> Result<String, String> {
    let canonical = canonical
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .map(|(_, parameter)| parameter)
        .ok_or_else(|| format!("unresolved GGUF parameter {canonical}"))?;
    if let Some(parameter) = canonical.strip_prefix("moe.") {
        if let Some((source, target)) = GGUF_MOE_PARAMETERS
            .iter()
            .copied()
            .filter(|&(_, target)| {
                parameter == target || parameter.starts_with(&format!("{target}."))
            })
            .max_by_key(|&(_, target)| target.len())
        {
            let suffix = parameter.strip_prefix(target).expect("matched target");
            let suffix = if target == "gate.e_score_correction_bias" && suffix.is_empty() {
                ".bias"
            } else if target.starts_with("experts.") && suffix.is_empty() {
                ".weight"
            } else {
                suffix
            };
            // Both physical correction-bias spellings map to the same slot.
            let source = if source == "ffn_exp_probs_b" {
                "exp_probs_b"
            } else {
                source
            };
            return Ok(format!("{source}{suffix}"));
        }
    }
    for (source, target) in GGUF_PARAMETERS {
        if canonical == target || canonical.starts_with(&format!("{target}.")) {
            return Ok(canonical.replacen(target, source, 1));
        }
    }
    Err(format!("unresolved GGUF parameter {canonical}"))
}

/// Translates one physical GGUF tensor name to its canonical parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embeddings"),
        ("output_norm", "model.norm_f"),
        ("output", "lm_head"),
    ];
    for (source, target) in ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }

    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };

    for (source, target) in GGUF_MOE_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = parameter.strip_prefix(source).unwrap_or_default();
            let suffix = if target == "gate.e_score_correction_bias" && suffix == ".bias" {
                ""
            } else if target.starts_with("experts.") {
                match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                }
            } else {
                suffix
            };
            return format!("model.layers.{layer}.moe.{target}{suffix}");
        }
    }

    for (source, target) in GGUF_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}
