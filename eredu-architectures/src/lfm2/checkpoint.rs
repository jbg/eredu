//! Pure SafeTensors and GGUF checkpoint plans for LFM2.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::WeightQuantization;
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_core::AttentionPolicy;
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

use super::config::ModelArgs;
use super::{
    declarations::{FeedForwardParameters, MixerParameters},
    Declaration,
};

/// Derives an LFM2 configuration whose physical matrix formats reflect
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

/// Applies canonical checkpoint format metadata to a complete LFM2
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let source = Declaration::new(args.clone()).map_err(|error| error.to_string())?;
    for scope in &source.blocks {
        let FeedForwardParameters::Routed(spec) = &scope.feed_forward else {
            continue;
        };
        let packed = spec
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
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

fn expert_source(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, String> {
    let keys = store.source_keys();
    let key = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| keys.contains(candidate))
        .ok_or_else(|| {
            format!("LFM2 checkpoint is missing expert {expert} projection under {prefix}")
        })?;
    Ok(DerivedWeightRecipe::source(key, TensorSelection::Full))
}

/// Returns the complete derived-weight catalog for one LFM2 execution unit.
pub fn unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
    layer: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| format!("LFM2 has no declared layer {layer}"))?;
    let mut recipes = BTreeMap::new();
    if let MixerParameters::Convolution(spec) = &scope.mixer {
        let conv = spec.convolution.weight.id.as_str();
        if let Ok(metadata) = store.source_metadata(conv) {
            let expected = spec
                .convolution
                .logical_parameters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|d| d.parameter.id == spec.convolution.weight.id)
                .expect("declared convolution weight")
                .shape;
            let recipe = if metadata.logical_shape.len() == 2 {
                Some(DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(conv, TensorSelection::Full)),
                    shape: expected.clone(),
                })
            } else if metadata.logical_shape == [expected[0], expected[2], expected[1]]
                && metadata.logical_shape != expected
            {
                Some(DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::source(conv, TensorSelection::Full)),
                    axes: vec![0, 2, 1],
                })
            } else {
                None
            };
            if let Some(recipe) = recipe {
                recipes.insert(conv.to_owned(), recipe);
            }
        }
    }
    let FeedForwardParameters::Routed(parameters) = &scope.feed_forward else {
        return Ok(recipes);
    };
    let keys = store.source_keys();
    let bank = &parameters.experts;
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    let gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let down = source.down().parameter.id.as_str().to_owned();
    let prefix = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved LFM2 expert source root")?;
    if keys.contains(&gate_up) {
        return Ok(recipes);
    }
    if keys.contains(&format!("{prefix}.gate_proj")) && keys.contains(&format!("{prefix}.up_proj"))
    {
        recipes.insert(
            gate_up,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: ["gate_proj", "up_proj"]
                    .into_iter()
                    .map(|name| {
                        DerivedWeightRecipe::source(
                            format!("{prefix}.{name}"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
        return Ok(recipes);
    }
    let mut gate_up_inputs = Vec::new();
    let mut down_inputs = Vec::new();
    for expert in 0..source.group_count() {
        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?,
                expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?,
            ],
        });
        down_inputs.push(expert_source(
            store,
            &prefix,
            expert as i32,
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

/// Returns canonical lazy-loading recipes for one LFM2 routed expert.
pub fn expert_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let scope = declaration
        .blocks
        .get(layer)
        .ok_or_else(|| format!("LFM2 has no declared layer {layer}"))?;
    let FeedForwardParameters::Routed(parameters) = &scope.feed_forward else {
        return Err(format!("LFM2 layer {layer} has no declared expert bank"));
    };
    expert_recipes_for_bank(store, layer, expert, &parameters.experts)
}

fn expert_recipes_for_bank(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    layer: usize,
    expert: usize,
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    if expert >= source.group_count() {
        return Err(format!("LFM2 has no expert {expert} in layer {layer}"));
    }
    let prefix = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved LFM2 expert source root")?;
    let packed_gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let packed_down = source.down().parameter.id.as_str().to_owned();
    let keys = store.source_keys();
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let mut recipes = BTreeMap::new();
    if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
        for (name, key) in [
            ("gate_up_proj", packed_gate_up.clone()),
            ("down_proj", packed_down.clone()),
        ] {
            recipes.insert(
                name.into(),
                DerivedWeightRecipe::source(key, selection.clone()),
            );
        }
        for (name, key) in [
            ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
            ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
            ("down_proj_scales", format!("{packed_down}_scales")),
            ("down_proj_biases", format!("{packed_down}_biases")),
        ] {
            if keys.contains(&key) {
                recipes.insert(
                    name.into(),
                    DerivedWeightRecipe::source(key, selection.clone()),
                );
            }
        }
    } else if keys.contains(&format!("{prefix}.gate_proj"))
        && keys.contains(&format!("{prefix}.up_proj"))
        && keys.contains(&packed_down)
    {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(format!("{prefix}.gate_proj"), selection.clone()),
                    DerivedWeightRecipe::source(format!("{prefix}.up_proj"), selection.clone()),
                ],
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{prefix}.gate_proj{suffix}");
            let up = format!("{prefix}.up_proj{suffix}");
            if keys.contains(&gate) && keys.contains(&up) {
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
            if keys.contains(&down) {
                recipes.insert(
                    format!("down_proj{suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
    } else {
        if source
            .encodings()
            .map_err(|e| e.to_string())?
            .iter()
            .any(|format| format.weight_quantization().is_some())
        {
            return Err("split LFM2 experts cannot be lazily load-time quantized".into());
        }
        let gate = expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?;
        let up = expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?;
        let down = expert_source(store, &prefix, expert as i32, &["w2", "down_proj"])?;
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
    }
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    declaration: &Declaration,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if declaration.expert_banks().is_empty() {
        return Err("independent expert residency requires LFM2-MoE".into());
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

/// Builds the complete alternative-layout SafeTensors contract.
pub fn safetensors_plan(
    declaration: &Declaration,
    allow_derived_packed: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let statics = &declaration.statics;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_spec(&mut common, statics.embedding_spec())?;
    add_safe_spec(&mut common, statics.normalization_spec())?;
    if let Some(head) = statics.head_spec() {
        add_safe_spec(&mut common, head)?;
    }
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        for spec in [&scope.operator_norm, &scope.feed_forward_norm] {
            add_safe_spec(&mut common, spec)?;
        }
        match &scope.mixer {
            MixerParameters::Convolution(spec) => {
                let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
                declarations.sort_by_key(|d| convolution_storage_order(d.parameter.id.as_str()));
                for declaration in declarations {
                    let name = declaration.parameter.id.as_str();
                    if declaration.parameter.id == spec.convolution.weight.id {
                        let mut alternative = declaration.shape.clone();
                        alternative.swap(1, 2);
                        common.push(
                            safe(name, declaration.shape).with_alternate_shapes([alternative]),
                        );
                    } else if declaration.role == LogicalParameterRole::LinearWeight {
                        common.extend(source_matrix_constraints(
                            name,
                            declaration.shape,
                            declaration
                                .linear_format
                                .and_then(|format| format.encoding().weight_quantization()),
                        )?);
                    } else {
                        common.push(safe(name, declaration.shape));
                    }
                }
            }
            MixerParameters::Attention(spec, AttentionPolicy::Full) => {
                add_safe_spec(&mut common, spec)?;
            }
            MixerParameters::Attention(_, AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                for spec in [&spec.gate, &spec.down, &spec.up] {
                    add_safe_spec(&mut common, spec)?;
                }
            }
            FeedForwardParameters::Routed(parameters) => add_safe_moe(
                parameters,
                layer,
                allow_derived_packed,
                &mut common,
                &mut groups,
            )?,
        }
    }
    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("LFM2 SafeTensors", common, groups, policy)
        .map_err(|e| e.to_string())
}

fn add_safe_spec(
    output: &mut Vec<SafetensorsTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for declaration in spec.logical_parameters().map_err(|e| e.to_string())? {
        let name = declaration.parameter.id.as_str();
        if matches!(
            declaration.role,
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
        ) {
            output.extend(source_matrix_constraints(
                name,
                declaration.shape,
                declaration
                    .linear_format
                    .and_then(|format| format.encoding().weight_quantization()),
            )?);
        } else {
            output.push(safe(name, declaration.shape));
        }
    }
    Ok(())
}

fn convolution_storage_order(name: &str) -> usize {
    [
        "conv.weight",
        "in_proj.weight",
        "out_proj.weight",
        "conv.bias",
        "in_proj.bias",
        "out_proj.bias",
    ]
    .iter()
    .position(|field| name.ends_with(&format!(".{field}")))
    .expect("declared short-convolution slot")
}

fn add_safe_moe(
    parameters: &eredu_nn::RoutedGatedProductSpec,
    layer: usize,
    allow_derived_packed: bool,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    // The released router stays floating while expert projections may be encoded.
    for declaration in parameters
        .selector
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        common.push(safe(declaration.parameter.id.as_str(), declaration.shape));
    }
    let bank = &parameters.experts;
    let source = bank.packed_parameters().map_err(|e| e.to_string())?;
    let gate_up_shape = source.gate_up().shape.clone();
    let down_shape = source.down().shape.clone();
    let read_shape = source.unfused_shape();
    let prefix = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved LFM2 expert source root")?;
    let gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let down = source.down().parameter.id.as_str().to_owned();
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let [gate_up_format, down_format] = source.encodings().map_err(|e| e.to_string())?;
    let gate_up_quantization = gate_up_format.weight_quantization();
    let down_quantization = down_format.weight_quantization();
    let mut variants = vec![LayoutVariant {
        id: "fused packed".into(),
        tensors: [
            source_matrix_constraints(&gate_up, gate_up_shape, gate_up_quantization)?,
            source_matrix_constraints(&down, down_shape.clone(), down_quantization)?,
        ]
        .concat(),
        discriminator_keys: vec![gate_up.clone()],
    }];
    if allow_derived_packed {
        variants.push(LayoutVariant {
            id: "separate packed".into(),
            tensors: [
                source_matrix_constraints(&gate, read_shape.clone(), gate_up_quantization)?,
                source_matrix_constraints(&up, read_shape.clone(), gate_up_quantization)?,
                source_matrix_constraints(&down, down_shape.clone(), down_quantization)?,
            ]
            .concat(),
            discriminator_keys: vec![gate, up],
        });
    }
    if gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::new();
        let mut discriminators = Vec::new();
        visit_independent_expert_sources(&source, |_, canonical, alias, shape| {
            discriminators.push(canonical.clone());
            tensors.push(safe(canonical, shape).with_aliases([alias]));
            Ok(())
        })?;
        variants.push(LayoutVariant {
            id: "per-expert split".into(),
            tensors,
            discriminator_keys: discriminators,
        });
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("LFM2 layer {layer} routed expert storage"),
        required: true,
        variants,
    });
    Ok(())
}

/// Projects admitted ordinary matrix geometry from neural declarations and exact
/// source-layout extensions, before physical packing. Packed banks retain their
/// grouped geometry in the expert realization plan.
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
        let [gate_up, down] = source.encodings().map_err(|e| e.to_string())?;
        if gate_up.weight_quantization().is_none() && down.weight_quantization().is_none() {
            visit_independent_expert_sources(&source, |_, canonical, _, shape| {
                shapes.insert(canonical, shape);
                Ok(())
            })?;
        }
    }
    Ok(shapes)
}

/// Physical per-expert names reference the declared fused or down projection;
/// each source selects one group and, for gate/up, one row half.
fn visit_independent_expert_sources(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    mut visit: impl FnMut(
        &eredu_nn::LogicalParameterDeclaration<'_>,
        String,
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
        .ok_or("LFM2 expert source has no declared gate/up root")?;
    let read_shape = source.unfused_shape();
    for expert in 0..source.group_count() {
        for (logical, physical, alias, shape) in [
            (
                source.gate_up(),
                "w1",
                "gate_proj",
                read_shape[1..].to_vec(),
            ),
            (
                source.down(),
                "w2",
                "down_proj",
                source.down().shape[1..].to_vec(),
            ),
            (source.gate_up(), "w3", "up_proj", read_shape[1..].to_vec()),
        ] {
            visit(
                logical,
                format!("{root}.{expert}.{physical}.weight"),
                format!("{root}.{expert}.{alias}.weight"),
                shape,
            )?;
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
        return Ok(vec![safe(name, shape)]);
    };
    eredu_checkpoint::schema::matrix_for_packed_words(
        name,
        Vec::<String>::new(),
        shape,
        quantization,
        None,
    )
    .map_err(|error| error.to_string())
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

/// Builds the complete physical GGUF tensor contract.
pub fn gguf_plan(declaration: &Declaration) -> Result<GgufCheckpointPlan, String> {
    let is_moe = !declaration.expert_banks().is_empty();
    let statics = &declaration.statics;
    let mut tensors = Vec::new();
    add_gguf_spec(&mut tensors, statics.embedding_spec(), is_moe)?;
    add_gguf_spec(&mut tensors, statics.normalization_spec(), is_moe)?;
    if let Some(head) = statics.head_spec() {
        add_gguf_spec(&mut tensors, head, is_moe)?;
    }
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        let block = format!("blk.{layer}");
        for spec in [&scope.operator_norm, &scope.feed_forward_norm] {
            add_gguf_spec(&mut tensors, spec, is_moe)?;
        }
        match &scope.mixer {
            MixerParameters::Convolution(spec) => {
                let mut declarations = spec.logical_parameters().map_err(|e| e.to_string())?;
                declarations.sort_by_key(|d| convolution_storage_order(d.parameter.id.as_str()));
                for declaration in declarations {
                    let mut shape = declaration.shape;
                    let operation = if declaration.parameter.id == spec.convolution.weight.id {
                        shape.remove(1); // GGUF omits the singleton input-channel axis.
                        TensorOperation::Dense
                    } else if declaration.role == LogicalParameterRole::LinearWeight {
                        TensorOperation::Matrix
                    } else {
                        TensorOperation::Vector
                    };
                    tensors.push(gguf(
                        gguf_parameter_name(declaration.parameter.id.as_str(), is_moe)?,
                        shape,
                        operation,
                    ));
                }
            }
            MixerParameters::Attention(spec, AttentionPolicy::Full) => {
                add_gguf_spec(&mut tensors, spec, is_moe)?;
            }
            MixerParameters::Attention(_, AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }
        match &scope.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                for spec in [&spec.gate, &spec.down, &spec.up] {
                    add_gguf_spec(&mut tensors, spec, is_moe)?;
                }
            }
            FeedForwardParameters::Routed(parameters) => {
                let declarations = parameters
                    .selector
                    .logical_parameters()
                    .map_err(|e| e.to_string())?;
                // The correction bias follows the independent expert projections in GGUF.
                for declaration in declarations
                    .iter()
                    .filter(|d| d.role == LogicalParameterRole::LinearWeight)
                {
                    tensors.push(gguf(
                        gguf_parameter_name(declaration.parameter.id.as_str(), true)?,
                        declaration.shape.clone(),
                        TensorOperation::Matrix,
                    ));
                }
                let bank = &parameters.experts;
                let source = bank.packed_parameters().map_err(|e| e.to_string())?;
                let read_shape = source.unfused_shape();
                for (name, shape) in [
                    ("ffn_gate_exps.weight", read_shape.clone()),
                    ("ffn_up_exps.weight", read_shape.clone()),
                    ("ffn_down_exps.weight", source.down().shape.clone()),
                ] {
                    tensors.push(gguf(
                        format!("{block}.{name}"),
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
                for declaration in declarations
                    .into_iter()
                    .filter(|d| d.role != LogicalParameterRole::LinearWeight)
                {
                    tensors.push(
                        gguf(
                            gguf_parameter_name(declaration.parameter.id.as_str(), true)?,
                            declaration.shape,
                            TensorOperation::Vector,
                        )
                        .with_aliases([format!("{block}.exp_probs_b.bias")]),
                    );
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "LFM2 GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|e| e.to_string())
}

fn add_gguf_spec(
    output: &mut Vec<GgufTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
    is_moe: bool,
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
            gguf_parameter_name(declaration.parameter.id.as_str(), is_moe)?,
            declaration.shape,
            operation,
        ));
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

const GGUF_ROOTS: [(&str, &str); 3] = [
    ("token_embd", "model.embed_tokens"),
    ("token_embd_norm", "model.embedding_norm"),
    ("output", "lm_head"),
];

const GGUF_MOE_PARAMETERS: [(&str, &str); 6] = [
    ("ffn_gate_inp", "feed_forward.gate"),
    ("ffn_gate_exps", "feed_forward.experts.gate_proj"),
    ("ffn_up_exps", "feed_forward.experts.up_proj"),
    ("ffn_down_exps", "feed_forward.experts.down_proj"),
    ("ffn_exp_probs_b", "feed_forward.expert_bias"),
    ("exp_probs_b", "feed_forward.expert_bias"),
];

const GGUF_PARAMETERS: [(&str, &str); 14] = [
    ("shortconv.conv", "conv.conv"),
    ("shortconv.in_proj", "conv.in_proj"),
    ("shortconv.out_proj", "conv.out_proj"),
    ("attn_q_norm", "self_attn.q_layernorm"),
    ("attn_k_norm", "self_attn.k_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.out_proj"),
    ("attn_norm", "operator_norm"),
    ("ffn_norm", "ffn_norm"),
    ("ffn_gate", "feed_forward.w1"),
    ("ffn_down", "feed_forward.w2"),
    ("ffn_up", "feed_forward.w3"),
];

fn gguf_parameter_name(canonical: &str, is_moe: bool) -> Result<String, String> {
    for (source, target) in GGUF_ROOTS {
        if canonical == target || canonical.starts_with(&format!("{target}.")) {
            return Ok(canonical.replacen(target, source, 1));
        }
    }
    if let Some((layer, parameter)) = canonical
        .strip_prefix("model.layers.")
        .and_then(|s| s.split_once('.'))
    {
        if is_moe {
            // Prefer the most specific target, retaining the primary spelling for aliases.
            let mapping = GGUF_MOE_PARAMETERS
                .iter()
                .copied()
                .filter(|&(_, target)| {
                    parameter == target || parameter.starts_with(&format!("{target}."))
                })
                .fold(None, |best: Option<(&str, &str)>, candidate| {
                    if best.is_none_or(|(_, target)| candidate.1.len() > target.len()) {
                        Some(candidate)
                    } else {
                        best
                    }
                });
            if let Some((source, target)) = mapping {
                let suffix = parameter.strip_prefix(target).expect("matched parameter");
                let suffix = if target.ends_with("expert_bias") && suffix.is_empty() {
                    ".bias"
                } else if target.contains("experts.") && suffix.is_empty() {
                    ".weight"
                } else {
                    suffix
                };
                return Ok(format!("blk.{layer}.{source}{suffix}"));
            }
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
    Err(format!("unresolved LFM2 GGUF parameter {canonical}"))
}

/// Translates one physical GGUF tensor name to its canonical parameter identity.
pub fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    for (source, target) in GGUF_ROOTS {
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
    if is_moe {
        for (source, target) in GGUF_MOE_PARAMETERS {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let suffix = parameter.strip_prefix(source).unwrap_or_default();
                let suffix = if target.ends_with("expert_bias") && suffix == ".bias" {
                    ""
                } else if target.contains("experts.") {
                    match suffix {
                        ".weight" => "",
                        ".scales" => "_scales",
                        ".biases" => "_biases",
                        other => other,
                    }
                } else {
                    suffix
                };
                return format!("model.layers.{layer}.{target}{suffix}");
            }
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
