//! Pure Kimi Linear checkpoint plans, naming, and derived-weight recipes.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{TensorSelection, WeightStoreBackend},
    WeightQuantization,
};

use super::declarations::{FeedForwardParameters, MixerParameters};
use super::{Declaration, ModelArgs};
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

/// Derives a Kimi Linear configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Kimi Linear
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let source = Declaration::new(args.clone()).map_err(|error| error.to_string())?;
    for scope in &source.blocks {
        let FeedForwardParameters::Sparse(spec) = &scope.feed_forward else {
            continue;
        };
        let packed = spec
            .routed
            .experts
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let prefix = packed
            .gate_up()
            .parameter
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or("expert bank has no declared source root")?;
        crate::format_references::rehome_split_matrix_format(
            &mut formats,
            &packed.gate_up().parameter.id,
            [format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
        )?;
    }
    let mut target = args.clone();
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

fn canonical_recipe_name(name: &str) -> String {
    name.replace(".block_sparse_moe.", ".mlp.")
}

fn normalized_checkpoint_keys(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| (canonical_recipe_name(&raw), raw))
        .collect()
}

fn expert_source(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, String> {
    let runtime = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| normalized.contains_key(candidate))
        .ok_or_else(|| {
            format!("Kimi Linear checkpoint is missing expert {expert} projection under {prefix}")
        })?;
    Ok(DerivedWeightRecipe::source(
        normalized
            .get(&runtime)
            .expect("normalized Kimi expert key exists"),
        TensorSelection::Full,
    ))
}

/// Returns the complete neutral recipe catalog for one Kimi execution unit.
pub fn unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
    layer: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| format!("Kimi Linear has no declared layer {layer}"))?;
    let mut recipes = BTreeMap::new();
    let root = format!("model.layers.{layer}");
    let expert_prefix = format!("{root}.mlp.experts");
    let normalized = normalized_checkpoint_keys(store);
    for (runtime, raw) in &normalized {
        if runtime.starts_with(&format!("{root}.mlp."))
            && !runtime.starts_with(&expert_prefix)
            && runtime != raw
        {
            recipes.insert(
                runtime.clone(),
                DerivedWeightRecipe::source(raw, TensorSelection::Full),
            );
        }
    }

    if let MixerParameters::Kda(spec) = &scope.mixer {
        for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
            let name = declaration.parameter.id.as_str();
            if store.source_metadata(name).is_err() {
                continue;
            }
            if declaration.parameter.id == spec.a_log.id {
                let mut recipe = DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                    shape: declaration.shape,
                };
                if store.source_backend().map_err(|e| e.to_string())? == WeightStoreBackend::Gguf {
                    recipe = DerivedWeightRecipe::NegLog {
                        input: Box::new(recipe),
                    };
                }
                recipes.insert(name.to_owned(), recipe);
            } else if spec
                .convolutions
                .iter()
                .any(|c| c.weight.id == declaration.parameter.id)
            {
                recipes.insert(
                    name.to_owned(),
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                        shape: declaration.shape,
                    },
                );
            }
        }
    }

    if !include_experts {
        return Ok(recipes);
    }
    let FeedForwardParameters::Sparse(parameters) = &scope.feed_forward else {
        return Ok(recipes);
    };
    let bank = &parameters.routed.experts;
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    let gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let down = source.down().parameter.id.as_str().to_owned();
    if let (Some(gate_up_source), Some(down_source)) =
        (normalized.get(&gate_up), normalized.get(&down))
    {
        for (target, source) in [(&gate_up, gate_up_source), (&down, down_source)] {
            if source != target {
                recipes.insert(
                    target.clone(),
                    DerivedWeightRecipe::source(source, TensorSelection::Full),
                );
            }
        }
        return Ok(recipes);
    }
    let gate = format!("{expert_prefix}.gate_proj");
    let up = format!("{expert_prefix}.up_proj");
    if normalized.contains_key(&gate) && normalized.contains_key(&up) {
        recipes.insert(
            gate_up,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: [gate, up]
                    .into_iter()
                    .map(|runtime| {
                        DerivedWeightRecipe::source(
                            normalized.get(&runtime).expect("normalized projection key"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
        return Ok(recipes);
    }
    let experts = source.group_count();
    let mut gate_up_inputs = Vec::with_capacity(experts);
    let mut down_inputs = Vec::with_capacity(experts);
    for expert in 0..experts {
        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                expert_source(&normalized, &expert_prefix, expert, &["w1", "gate_proj"])?,
                expert_source(&normalized, &expert_prefix, expert, &["w3", "up_proj"])?,
            ],
        });
        down_inputs.push(expert_source(
            &normalized,
            &expert_prefix,
            expert,
            &["w2", "down_proj"],
        )?);
    }
    recipes.insert(
        gate_up,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: gate_up_inputs,
        },
    );
    recipes.insert(
        down,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: down_inputs,
        },
    );
    Ok(recipes)
}

/// Returns neutral lazy-loading recipes for one Kimi routed expert.
pub fn expert_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| format!("Kimi Linear has no declared layer {layer}"))?;
    let FeedForwardParameters::Sparse(p) = &scope.feed_forward else {
        return Err(format!("Kimi Linear layer {layer} has no declared bank"));
    };
    expert_recipes_for_bank(store, layer, expert, &p.routed.experts)
}

fn expert_recipes_for_bank(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    layer: usize,
    expert: usize,
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    if expert >= source.group_count() {
        return Err(format!(
            "Kimi Linear has no expert {expert} in layer {layer}"
        ));
    }
    let normalized = normalized_checkpoint_keys(store);
    let prefix = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved Kimi Linear expert source root")?;
    let packed_gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let packed_down = source.down().parameter.id.as_str().to_owned();
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let mut recipes = BTreeMap::new();
    if let (Some(gate_up), Some(down)) = (
        normalized.get(&packed_gate_up),
        normalized.get(&packed_down),
    ) {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::source(gate_up, selection.clone()),
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(down, selection.clone()),
        );
        for (target, source) in [
            ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
            ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
            ("down_proj_scales", format!("{packed_down}_scales")),
            ("down_proj_biases", format!("{packed_down}_biases")),
        ] {
            if let Some(source) = normalized.get(&source) {
                recipes.insert(
                    target.into(),
                    DerivedWeightRecipe::source(source, selection.clone()),
                );
            }
        }
        return Ok(recipes);
    }
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    if let (Some(gate), Some(up), Some(down)) = (
        normalized.get(&gate),
        normalized.get(&up),
        normalized.get(&packed_down),
    ) {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, selection.clone()),
                    DerivedWeightRecipe::source(up, selection.clone()),
                ],
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(down, selection.clone()),
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{prefix}.gate_proj{suffix}");
            let up = format!("{prefix}.up_proj{suffix}");
            if let (Some(gate), Some(up)) = (normalized.get(&gate), normalized.get(&up)) {
                recipes.insert(
                    format!("gate_up_proj{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, selection.clone()),
                            DerivedWeightRecipe::source(up, selection.clone()),
                        ],
                    },
                );
            }
            let down = format!("{packed_down}{suffix}");
            if let Some(down) = normalized.get(&down) {
                recipes.insert(
                    format!("down_proj{suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
        return Ok(recipes);
    }
    if source
        .encodings()
        .map_err(|e| e.to_string())?
        .iter()
        .any(|format| format.weight_quantization().is_some())
    {
        return Err("split Kimi Linear experts cannot be lazily load-time quantized".into());
    }
    let gate = expert_source(&normalized, &prefix, expert, &["w1", "gate_proj"])?;
    let up = expert_source(&normalized, &prefix, expert, &["w3", "up_proj"])?;
    let down = expert_source(&normalized, &prefix, expert, &["w2", "down_proj"])?;
    recipes.insert(
        "gate_up_proj".into(),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate, up],
            }],
        },
    );
    recipes.insert(
        "down_proj".into(),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![down],
        },
    );
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if declaration.expert_banks().is_empty() {
        return Err("independent expert residency requires Kimi Linear-MoE".into());
    }
    let owner_group = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for (layer, bank) in declaration.expert_banks() {
        let unit_path = format!("model.layers.{layer}");
        let source = bank.packed_parameters().map_err(|e| e.to_string())?;
        for expert in 0..source.group_count() {
            let recipes = expert_recipes_for_bank(store, layer, expert, &bank)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases");
            let parameters = recipes
                .into_iter()
                .map(|(binding, recipe)| {
                    let role = match binding.as_str() {
                        "gate_up_proj" if gate_up_quantizable => {
                            crate::ExpertParameterRole::quantizable_projection(
                                "gate_up_proj_scales",
                                "gate_up_proj_biases",
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
                    let (primary, suffix) = binding
                        .strip_prefix("gate_up_proj")
                        .map(|suffix| (source.gate_up(), suffix))
                        .or_else(|| {
                            binding
                                .strip_prefix("down_proj")
                                .map(|suffix| (source.down(), suffix))
                        })
                        .filter(|(_, suffix)| matches!(*suffix, "" | "_scales" | "_biases"))
                        .ok_or_else(|| format!("undeclared expert binding {binding}"))?;
                    let target = format!("{}{suffix}", primary.parameter.id);
                    crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    eredu_runtime::ParameterBankKey::new(0, layer, expert),
                    owner_group.clone(),
                    layer,
                    &unit_path,
                    crate::ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(store))
        .map_err(|error| error.to_string())
}

/// Builds the strict SafeTensors catalog plan.
pub fn safetensors_plan(declaration: &Declaration) -> Result<SafetensorsCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_spec(&mut common, statics.embedding_spec())?;
    add_safe_spec(&mut common, statics.normalization_spec())?;
    if let Some(head) = statics.head_spec() {
        add_safe_spec(&mut common, head)?;
    }
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        for spec in [&scope.input_norm, &scope.post_attention_norm] {
            add_safe_spec(&mut common, spec)?;
        }
        match &scope.mixer {
            MixerParameters::Kda(p) => add_safe_kda(p, &mut common)?,
            MixerParameters::Mla(p) => add_safe_spec(&mut common, p)?,
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                add_safe_spec(&mut common, spec)?;
            }
            FeedForwardParameters::Sparse(parameters) => {
                add_safe_moe(parameters, layer, &mut common, &mut groups)?
            }
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("model.mtp.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("Kimi Linear SafeTensors", common, groups, policy)
        .map_err(|e| e.to_string())
}

fn add_safe_spec(
    output: &mut Vec<SafetensorsTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        if matches!(
            declaration.role,
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
        ) {
            output.extend(source_matrix_constraints(
                declaration.parameter.id.as_str(),
                declaration.shape,
                declaration
                    .linear_format
                    .and_then(|format| format.encoding().weight_quantization()),
            )?);
        } else {
            output.push(safe_aliases(
                declaration.parameter.id.as_str(),
                declaration.shape,
            ));
        }
    }
    Ok(())
}

fn add_safe_kda(
    spec: &super::specs::KdaSpec,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let slot = declaration.parameter;
        if slot.id == spec.a_log.id || spec.convolutions.iter().any(|c| c.weight.id == slot.id) {
            let elements = declaration.shape.iter().product();
            output.push(
                safe_aliases(slot.id.as_str(), declaration.shape).with_element_count(elements),
            );
        } else if declaration.role == LogicalParameterRole::LinearWeight {
            output.extend(source_matrix_constraints(
                slot.id.as_str(),
                declaration.shape,
                declaration
                    .linear_format
                    .and_then(|format| format.encoding().weight_quantization()),
            )?);
        } else {
            output.push(safe_aliases(slot.id.as_str(), declaration.shape));
        }
    }
    Ok(())
}

fn add_safe_moe(
    parameters: &super::moe::SparseParameters,
    layer: usize,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    add_safe_spec(common, &parameters.routed.selector)?;
    add_safe_spec(common, &parameters.shared)?;
    let bank = &parameters.routed.experts;
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    let gate_shape = source.gate_up().shape.clone();
    let down_shape = source.down().shape.clone();
    let gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let down = source.down().parameter.id.as_str().to_owned();
    let [gate_up_format, down_format] = source.encodings().map_err(|e| e.to_string())?;
    let mut packed_tensors =
        source_matrix_constraints(&gate_up, gate_shape, gate_up_format.weight_quantization())?;
    packed_tensors.extend(source_matrix_constraints(
        &down,
        down_shape.clone(),
        down_format.weight_quantization(),
    )?);
    let mut split_tensors = Vec::new();
    let mut split_discriminators = Vec::new();
    visit_split_expert_sources(&source, |_, name, shape| {
        split_discriminators.push(name.clone());
        split_tensors.push(
            safe_aliases(&name, shape).with_linear_format(eredu_checkpoint::LinearFormat::Dense),
        );
        Ok(())
    })?;
    groups.push(AlternativeLayoutGroup {
        id: format!("Kimi Linear layer {layer} expert storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                tensors: packed_tensors,
                discriminator_keys: vec![gate_up, down],
            },
            LayoutVariant {
                id: "split".into(),
                tensors: split_tensors,
                discriminator_keys: split_discriminators,
            },
        ],
    });
    Ok(())
}

/// Projects ordinary source matrix geometry from the neural declaration and
/// the admitted independent-expert extension, without undoing physical packing.
pub(crate) fn source_linear_shapes(
    declaration: &Declaration,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let mut shapes = declaration
        .definition()
        .parameters()
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| member.logical_role() == Some(LogicalParameterRole::LinearWeight))
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target().to_owned(), shape.to_vec()))
        })
        .collect::<BTreeMap<_, _>>();
    for bank in declaration.expert_banks().values() {
        let source = bank.packed_parameters().map_err(|e| e.to_string())?;
        visit_split_expert_sources(&source, |_, name, shape| {
            shapes.insert(name, shape);
            Ok(())
        })?;
    }
    Ok(shapes)
}

fn visit_split_expert_sources(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    mut visit: impl FnMut(
        &eredu_nn::LogicalParameterDeclaration<'_>,
        String,
        Vec<usize>,
    ) -> Result<(), String>,
) -> Result<(), String> {
    let root = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("Kimi expert source has no declared gate/up root")?;
    let read_shape = source.unfused_shape();
    for expert in 0..source.group_count() {
        for (logical, physical, shape) in [
            (source.gate_up(), "w1", read_shape[1..].to_vec()),
            (source.down(), "w2", source.down().shape[1..].to_vec()),
            (source.gate_up(), "w3", read_shape[1..].to_vec()),
        ] {
            visit(logical, format!("{root}.{expert}.{physical}.weight"), shape)?;
        }
    }
    Ok(())
}

fn source_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![
            safe_aliases(name, shape).with_linear_format(eredu_checkpoint::LinearFormat::Dense)
        ]);
    };
    let mut constraints = eredu_checkpoint::schema::matrix_for_packed_words(
        name,
        physical_names(name)
            .into_iter()
            .filter(|candidate| candidate != name),
        shape,
        quantization,
        None,
    )
    .map_err(|error| error.to_string())?;
    constraints[0].linear_format = Some(quantization.into());
    Ok(constraints)
}

fn safe_aliases(key: &str, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    safe_aliases_with_dtype(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_aliases_with_dtype(
    key: &str,
    shape: Vec<usize>,
    dtype: StoredDtypeConstraint,
) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, dtype).with_aliases(
        physical_names(key)
            .into_iter()
            .filter(|candidate| candidate != key),
    )
}

fn physical_names(canonical: &str) -> Vec<String> {
    let mut names = vec![canonical.to_string()];
    if canonical.contains(".mlp.") {
        names.push(canonical.replace(".mlp.", ".block_sparse_moe."));
    }
    names.sort();
    names.dedup();
    names
}

/// Builds the GGUF physical catalog plan.
pub fn gguf_plan(declaration: &Declaration) -> Result<GgufCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut tensors = Vec::new();
    add_gguf_spec(&mut tensors, statics.embedding_spec())?;
    add_gguf_spec(&mut tensors, statics.normalization_spec())?;
    if let Some(head) = statics.head_spec() {
        add_gguf_spec(&mut tensors, head)?;
    }
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        let block = format!("blk.{layer}");
        for spec in [&scope.input_norm, &scope.post_attention_norm] {
            add_gguf_spec(&mut tensors, spec)?;
        }
        match &scope.mixer {
            MixerParameters::Kda(p) => add_gguf_kda(p, &mut tensors)?,
            MixerParameters::Mla(p) => add_gguf_spec(&mut tensors, p)?,
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                add_gguf_spec(&mut tensors, spec)?;
            }
            FeedForwardParameters::Sparse(parameters) => {
                add_gguf_spec(&mut tensors, &parameters.routed.selector)?;
                add_gguf_spec(&mut tensors, &parameters.shared)?;
                let bank = &parameters.routed.experts;
                let source = bank.packed_parameters().map_err(|e| e.to_string())?;
                let read_shape = source.unfused_shape();
                for (field, shape) in [
                    ("ffn_gate_exps.weight", read_shape.clone()),
                    ("ffn_up_exps.weight", read_shape.clone()),
                    ("ffn_down_exps.weight", source.down().shape.clone()),
                ] {
                    tensors.push(gguf(
                        format!("{block}.{field}"),
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "Kimi Linear GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|e| e.to_string())
}

fn add_gguf_spec(
    output: &mut Vec<GgufTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let operation = if matches!(
            declaration.role,
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
        ) {
            TensorOperation::Matrix
        } else {
            TensorOperation::Vector
        };
        output.push(gguf(
            gguf_parameter_name(declaration.parameter)?,
            declaration.shape,
            operation,
        ));
    }
    Ok(())
}

fn add_gguf_kda(
    spec: &super::specs::KdaSpec,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
    declarations.sort_by_key(|d| kda_gguf_storage_order(d.parameter.id.as_str()));
    for declaration in declarations {
        let name = gguf_parameter_name(declaration.parameter)?;
        if declaration.parameter.id == spec.a_log.id {
            let count = declaration.shape.iter().product::<usize>();
            output.push(
                gguf(&name, vec![count], TensorOperation::Vector)
                    .with_aliases([format!("{name}.weight")])
                    .with_element_count(count),
            );
        } else if spec
            .convolutions
            .iter()
            .any(|c| c.weight.id == declaration.parameter.id)
        {
            let count = declaration.shape.iter().product();
            output.push(
                gguf(name, declaration.shape, TensorOperation::Dense).with_element_count(count),
            );
        } else {
            let operation = if declaration.role == LogicalParameterRole::LinearWeight {
                TensorOperation::Matrix
            } else {
                TensorOperation::Vector
            };
            output.push(gguf(name, declaration.shape, operation));
        }
    }
    Ok(())
}

fn kda_gguf_storage_order(name: &str) -> usize {
    [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "dt_bias",
        "o_norm.weight",
        "o_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
    ]
    .iter()
    .position(|field| name.ends_with(&format!(".{field}")))
    .expect("declared KDA parameter")
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

const GGUF_ROOTS: [(&str, &str); 3] = [
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];

const GGUF_EXPERT_PARAMETERS: [(&str, &str); 3] = [
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
];

const GGUF_PARAMETERS: [(&str, &str); 30] = [
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_q_a", "self_attn.q_a_proj"),
    ("attn_q_b", "self_attn.q_b_proj"),
    ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
    ("attn_kv_b", "self_attn.kv_b_proj"),
    ("attn_k_b", "self_attn.k_b_proj"),
    ("attn_v_b", "self_attn.v_b_proj"),
    ("attn_q_a_norm", "self_attn.q_a_layernorm"),
    ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
    ("attn_output", "self_attn.o_proj"),
    ("ssm_conv1d_q", "self_attn.q_conv1d"),
    ("ssm_conv1d_k", "self_attn.k_conv1d"),
    ("ssm_conv1d_v", "self_attn.v_conv1d"),
    ("ssm_f_a", "self_attn.f_a_proj"),
    ("ssm_f_b", "self_attn.f_b_proj"),
    ("ssm_beta", "self_attn.b_proj"),
    ("ssm_g_a", "self_attn.g_a_proj"),
    ("ssm_g_b", "self_attn.g_b_proj"),
    ("ssm_norm", "self_attn.o_norm"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
    ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
    ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
    ("ffn_gate_inp", "mlp.gate"),
];

fn gguf_parameter_name(parameter: &eredu_nn::ParameterSpec) -> Result<String, String> {
    let canonical = parameter.id.as_str();
    for (source, target) in GGUF_ROOTS {
        if canonical == target || canonical.starts_with(&format!("{target}.")) {
            return Ok(canonical.replacen(target, source, 1));
        }
    }
    if let Some((layer, parameter)) = canonical
        .strip_prefix("model.layers.")
        .and_then(|s| s.split_once('.'))
    {
        if parameter == "self_attn.A_log" {
            return Ok(format!("blk.{layer}.ssm_a"));
        }
        if parameter == "self_attn.dt_bias" {
            return Ok(format!("blk.{layer}.ssm_dt.bias"));
        }
        if parameter == "mlp.gate.e_score_correction_bias" {
            return Ok(format!("blk.{layer}.exp_probs_b.bias"));
        }
        for (source, target) in GGUF_PARAMETERS {
            if parameter == target || parameter.starts_with(&format!("{target}.")) {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(target, source, 1)
                ));
            }
        }
    }
    Err(format!("unresolved Kimi GGUF declaration {canonical}"))
}

/// Translates a physical GGUF tensor name to canonical Kimi parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in GGUF_ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
    };
    for (source, target) in GGUF_EXPERT_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = match parameter.strip_prefix(source).unwrap_or_default() {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    if matches!(parameter, "exp_probs_b.bias" | "ffn_exp_probs_b.bias") {
        return format!("model.layers.{layer}.mlp.gate.e_score_correction_bias");
    }
    for (source, target) in GGUF_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    if parameter == "ssm_a" || parameter.starts_with("ssm_a.") {
        let suffix = parameter.strip_prefix("ssm_a").unwrap_or_default();
        let suffix = if suffix == ".weight" { "" } else { suffix };
        return format!("model.layers.{layer}.self_attn.A_log{suffix}");
    }
    if parameter == "ssm_dt.bias" || parameter == "ssm_dt" {
        return format!("model.layers.{layer}.self_attn.dt_bias");
    }
    name.to_owned()
}
