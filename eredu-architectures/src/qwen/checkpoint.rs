//! Pure checkpoint schemas, name translation, and recipes for Qwen text models.

use eredu_nn::DeclareLogicalParameters;
use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
    SafetensorsTensorConstraint, TensorOperation,
};
use eredu_checkpoint::{
    expert::{
        resolve_gated_product_expert_recipes, GatedProductExpertLayoutNames,
        GatedProductExpertRecipes, IndependentGatedProductExpertNames,
    },
    recipe::RecipeCatalog,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe, store::TensorSelection, LinearFormat, WeightQuantization,
};

use super::ModelArgs;

/// Derives a Qwen configuration whose physical matrix formats reflect
/// load-time quantization.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Qwen
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let source = super::Declaration::new(args.clone()).map_err(|error| error.to_string())?;
    if let Some(projections) = source.projections() {
        for layer in projections.owner_layers() {
            let bank = source
                .expert_bank(layer)
                .map_err(|error| error.to_string())?;
            let packed = bank
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            let prefix = expert_source_root(&packed)?;
            crate::format_references::rehome_split_matrix_format(
                &mut formats,
                &packed.gate_up().parameter.id,
                [
                    format!("{prefix}.gate_proj.weight"),
                    format!("{prefix}.up_proj.weight"),
                ],
            )?;
            crate::format_references::rehome_matrix_format(
                &mut formats,
                &packed.down().parameter.id,
                &format!("{prefix}.down_proj.weight"),
            )?;
        }
    }
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Builds the canonical Qwen SafeTensors catalog with its published layout alternatives.
pub fn safetensors_plan(
    declaration: &super::Declaration,
    allow_derived_expert_layouts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let args = declaration.args();
    let declarations = super::parallel::parameter_description(
        declaration,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|e| e.to_string())?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    let mut routers = std::collections::BTreeSet::new();
    if args.is_moe() {
        for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
            let spec = declaration
                .routed(layer)
                .map_err(|error| error.to_string())?;
            routers.insert(spec.selector.weight().id.as_str().to_owned());
            groups.push(expert_layout_group(
                &spec.experts,
                allow_derived_expert_layouts,
            )?);
        }
    }
    for group in declarations.groups() {
        if group.role() == eredu_runtime::ParameterRole::ExpertIntermediate {
            continue;
        }
        for parameter in group.members() {
            // Released routers have floating source storage, independently of
            // the execution encoding requested for the constructed selector.
            let format = if routers.contains(parameter.target()) {
                LinearFormat::Dense
            } else {
                parameter
                    .logical_linear_format()
                    .map_or(LinearFormat::Dense, |format| format.encoding())
            };
            common.extend(
                matrix_for_linear_format(
                    parameter.target(),
                    Vec::<String>::new(),
                    parameter.global_shape().to_vec(),
                    format,
                    None,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    }
    if args.tie_word_embeddings {
        // The optional redundant head retains its own source encoding; tied
        // execution still consumes the declared embedding storage.
        let (name, shape) = tied_output_source(declaration)?;
        let tensors = matrix_constraints(
            &name,
            shape,
            args.weight_quantization_for(&name),
            Vec::new(),
        )?;
        groups.push(AlternativeLayoutGroup {
            id: "redundant tied output head".into(),
            required: false,
            variants: vec![LayoutVariant {
                id: "present".into(),
                discriminator_keys: tensors.iter().map(|tensor| tensor.key.clone()).collect(),
                tensors,
            }],
        });
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", args.model_type),
        common,
        groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

pub(super) fn expert_source_root<'a>(
    source: &'a eredu_nn::PackedGatedProductParameters<'_>,
) -> Result<&'a str, String> {
    source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or_else(|| "Qwen expert bank has no declared source root".into())
}

pub(super) fn expert_binding_target(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    binding: &str,
) -> Result<String, String> {
    let (primary, suffix) = binding
        .strip_prefix("gate_up_proj")
        .map(|suffix| (source.gate_up(), suffix))
        .or_else(|| {
            binding
                .strip_prefix("down_proj")
                .map(|suffix| (source.down(), suffix))
        })
        .ok_or_else(|| format!("undeclared Qwen expert binding {binding}"))?;
    if suffix.is_empty() {
        return Ok(primary.parameter.id.to_string());
    }
    let format = primary
        .linear_format
        .ok_or("expert projection has no declared format")?;
    let parameter = match suffix {
        "_scales" => format.scale(),
        "_biases" => format.affine_bias(),
        "_global_scale" => format.global_scale(),
        _ => None,
    }
    .ok_or_else(|| format!("undeclared Qwen expert binding {binding}"))?;
    Ok(parameter.id.to_string())
}

/// Resolves packed, separate-packed, or per-expert sources into one canonical
/// bank under the normalized configuration's checkpoint root.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let args = declaration.args();
    if !args.is_moe() {
        return Err("Qwen expert recipes require Qwen3-MoE arguments".into());
    }
    let layer_count = dimension(args.num_hidden_layers, "num_hidden_layers")?;
    if layer >= layer_count {
        return Err(format!(
            "Qwen expert recipe layer {layer} is outside {layer_count} layers"
        ));
    }
    let bank = declaration
        .expert_bank(layer)
        .map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let prefix = expert_source_root(&source)?;
    let separate_suffix = if catalog
        .tensor_metadata(&format!("{prefix}.gate_proj"))
        .is_ok()
    {
        ""
    } else {
        ".weight"
    };
    let mut independent = Vec::new();
    visit_independent_expert_sources(&source, |_, canonical, aliases, _| {
        let selected = std::iter::once(canonical.clone())
            .chain(aliases)
            .find(|name| catalog.tensor_metadata(name).is_ok())
            .unwrap_or(canonical);
        independent.push(selected);
        Ok(())
    })?;
    let names = GatedProductExpertLayoutNames {
        target_gate_up: source.gate_up().parameter.id.to_string(),
        target_down: source.down().parameter.id.to_string(),
        packed_gate_up: source.gate_up().parameter.id.to_string(),
        packed_down: source.down().parameter.id.to_string(),
        separate_gate: format!("{prefix}.gate_proj{separate_suffix}"),
        separate_up: format!("{prefix}.up_proj{separate_suffix}"),
        separate_down: format!("{prefix}.down_proj{separate_suffix}"),
        independent: independent
            .chunks_exact(3)
            .map(|names| IndependentGatedProductExpertNames {
                gate: names[0].clone(),
                up: names[1].clone(),
                down: names[2].clone(),
            })
            .collect(),
    };
    resolve_gated_product_expert_recipes(catalog, &names).map_err(|error| error.to_string())
}

/// Returns architecture-owned recipes for one rank-local Qwen expert bank.
///
/// Expert identity is resolved through canonical per-expert recipes and then
/// assembled in rank order, preserving packed companions and adapting to
/// packed, split, or independently stored sources.
pub fn rank_local_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
    group_indices: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if group_indices.is_empty() {
        return Err("rank-local Qwen expert recipes require at least one expert".into());
    }
    let mut unique = std::collections::BTreeSet::new();
    let mut grouped = BTreeMap::<String, Vec<DerivedWeightRecipe>>::new();
    for &expert in group_indices {
        if !unique.insert(expert) {
            return Err(format!(
                "rank-local Qwen expert recipe contains duplicate expert {expert}"
            ));
        }
        for (binding, recipe) in expert_unit_recipes(catalog, declaration, layer, expert)? {
            grouped.entry(binding).or_default().push(recipe);
        }
    }
    let bank = declaration
        .expert_bank(layer)
        .map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    grouped
        .into_iter()
        .map(|(binding, mut inputs)| {
            let recipe = if inputs.len() == 1 {
                inputs.pop().expect("one rank-local expert recipe")
            } else {
                DerivedWeightRecipe::Concatenate { axis: 0, inputs }
            };
            recipe.infer(catalog).map_err(|error| error.to_string())?;
            Ok((expert_binding_target(&source, &binding)?, recipe))
        })
        .collect()
}

fn recipe_source_key<C: RecipeCatalog + ?Sized>(catalog: &C, base: &str) -> Option<String> {
    if catalog.tensor_metadata(base).is_ok() {
        return Some(base.to_owned());
    }
    let weight = format!("{base}.weight");
    catalog.tensor_metadata(&weight).is_ok().then_some(weight)
}

fn split_expert_source<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<String, String> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|key| catalog.tensor_metadata(key).is_ok())
        .ok_or_else(|| {
            format!("Qwen checkpoint is missing split expert {expert} projection {projections:?}")
        })
}

/// Returns neutral lazy-loading recipes for one Qwen routed expert.
pub fn expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let args = declaration.args();
    if !args.is_moe() {
        return Err("Qwen expert recipes require Qwen3-MoE arguments".into());
    }
    let layers = dimension(args.num_hidden_layers, "num_hidden_layers")?;
    if layer >= layers {
        return Err(format!(
            "Qwen expert recipe layer {layer} is outside {layers} layers"
        ));
    }
    let bank = declaration
        .expert_bank(layer)
        .map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let experts = source.group_count();
    if expert >= experts {
        return Err(format!(
            "Qwen expert {expert} is outside {experts} experts in layer {layer}"
        ));
    }
    let prefix = expert_source_root(&source)?;
    let packed_gate_up = source.gate_up().parameter.id.to_string();
    let packed_down = source.down().parameter.id.to_string();
    let packed_gate_up_source = recipe_source_key(catalog, &packed_gate_up);
    let packed_down_source = recipe_source_key(catalog, &packed_down);
    let split_gate_source = recipe_source_key(catalog, &format!("{prefix}.gate_proj"));
    let split_up_source = recipe_source_key(catalog, &format!("{prefix}.up_proj"));
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let materialize = |layout, gate_up, down| {
        eredu_checkpoint::expert::gated_product_materialization_recipes(
            catalog,
            &GatedProductExpertRecipes {
                target_gate_up: "gate_up_proj".into(),
                target_down: "down_proj".into(),
                layout,
                gate_up,
                down,
            },
        )
    };
    if let (Some(gate_up), Some(down)) = (packed_gate_up_source, packed_down_source.as_ref()) {
        return materialize(
            eredu_checkpoint::expert::GatedProductExpertStorageLayout::Packed,
            DerivedWeightRecipe::source(gate_up, selection.clone()),
            DerivedWeightRecipe::source(down, selection),
        );
    }
    if let (Some(gate), Some(up), Some(down)) =
        (split_gate_source, split_up_source, packed_down_source)
    {
        return materialize(
            eredu_checkpoint::expert::GatedProductExpertStorageLayout::SeparatePacked,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, selection.clone()),
                    DerivedWeightRecipe::source(up, selection.clone()),
                ],
            },
            DerivedWeightRecipe::source(down, selection),
        );
    }
    if source
        .encodings()
        .map_err(|error| error.to_string())?
        .iter()
        .any(|format| format.weight_quantization().is_some())
    {
        return Err(
            "split Qwen experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                .into(),
        );
    }
    let gate = split_expert_source(catalog, &prefix, expert, &["gate_proj", "w1"])?;
    let up = split_expert_source(catalog, &prefix, expert, &["up_proj", "w3"])?;
    let down = split_expert_source(catalog, &prefix, expert, &["down_proj", "w2"])?;
    materialize(
        eredu_checkpoint::expert::GatedProductExpertStorageLayout::Independent,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, TensorSelection::Full),
                    DerivedWeightRecipe::source(up, TensorSelection::Full),
                ],
            }],
        },
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::source(down, TensorSelection::Full)],
        },
    )
}

/// Builds the complete architecture-owned schedule for independent expert residency.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let args = declaration.args();
    if !args.is_moe() {
        return Err("Qwen expert residency requires Qwen3-MoE arguments".into());
    }
    let banks = declaration
        .expert_banks()
        .map_err(|error| error.to_string())?;
    let capacity = banks.values().try_fold(0usize, |capacity, bank| {
        capacity
            .checked_add(bank.group_count() as usize)
            .ok_or_else(|| "Qwen expert residency catalog size overflowed".to_owned())
    })?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(|error| error.to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for (layer, bank) in banks {
        let unit_path = format!("{}.layers.{layer}", args.parameter_root);
        let source = bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        for expert in 0..source.group_count() {
            let recipes = expert_unit_recipes(catalog, declaration, layer, expert)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases");
            let parameters = recipes
                .into_iter()
                .map(|(binding, recipe)| {
                    let target = expert_binding_target(&source, &binding)?;
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
        .and_then(|residency| residency.with_inferred_byte_geometry(catalog))
        .map_err(|error| error.to_string())
}

fn expert_layout_group(
    bank: &eredu_nn::GroupedGatedProductSpec,
    allow_derived_layouts: bool,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let prefix = expert_source_root(&source)?;
    let gate_up_shape = source.gate_up().shape.clone();
    let down_shape = source.down().shape.clone();
    let read_shape = source.unfused_shape();
    let experts = source.group_count();
    let [gate_up_format, down_format] = source.encodings().map_err(|error| error.to_string())?;
    let gate_up_quantization = gate_up_format.weight_quantization();
    let down_quantization = down_format.weight_quantization();
    let packed_gate_up = source.gate_up().parameter.id.to_string();
    let packed_down = source.down().parameter.id.to_string();
    let packed = expert_variant(
        "packed",
        vec![packed_gate_up.clone()],
        [
            (
                packed_gate_up,
                gate_up_shape,
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down.clone(),
                down_shape.clone(),
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let separate = expert_variant(
        "separate-packed",
        vec![gate.clone(), up.clone()],
        [
            (
                gate.clone(),
                read_shape.clone(),
                gate_up_quantization,
                Vec::new(),
            ),
            (
                up.clone(),
                read_shape.clone(),
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down,
                down_shape.clone(),
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let mut variants = vec![packed];
    if allow_derived_layouts {
        variants.push(separate);
    }
    if allow_derived_layouts && gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::with_capacity(checked_mul(experts, 3, "expert tensor count")?);
        let mut discriminators = Vec::with_capacity(tensors.capacity());
        visit_independent_expert_sources(&source, |_, canonical, aliases, shape| {
            discriminators.push(canonical.clone());
            tensors.push((canonical, shape, None, aliases));
            Ok(())
        })?;
        variants.push(expert_variant("split", discriminators, tensors)?);
    }
    Ok(AlternativeLayoutGroup {
        id: format!("{prefix} storage"),
        required: true,
        variants,
    })
}

fn visit_independent_expert_sources(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    mut visit: impl FnMut(
        &eredu_nn::LogicalParameterDeclaration<'_>,
        String,
        Vec<String>,
        Vec<usize>,
    ) -> Result<(), String>,
) -> Result<(), String> {
    let root = expert_source_root(source)?;
    let read = source.unfused_shape();
    for expert in 0..source.group_count() {
        for (logical, field, alias, shape) in [
            (source.gate_up(), "gate_proj", "w1", read[1..].to_vec()),
            (source.gate_up(), "up_proj", "w3", read[1..].to_vec()),
            (
                source.down(),
                "down_proj",
                "w2",
                source.down().shape[1..].to_vec(),
            ),
        ] {
            visit(
                logical,
                format!("{root}.{expert}.{field}.weight"),
                vec![format!("{root}.{expert}.{alias}.weight")],
                shape,
            )?;
        }
    }
    Ok(())
}

pub(super) fn tied_output_source(
    declaration: &super::Declaration,
) -> Result<(String, Vec<usize>), String> {
    let embedding = declaration.common().static_construction().embedding_spec();
    let shape = embedding
        .logical_parameters()
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("Qwen embedding declaration is missing")?
        .shape;
    Ok(("lm_head.weight".into(), shape))
}

/// Logical source matrices and admitted source-layout projections. Physical
/// packing and optional redundant tensors do not introduce parameter owners.
pub(crate) fn source_linear_shapes(
    declaration: &super::Declaration,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let args = declaration.args();
    let parameters = super::parallel::parameter_description(
        declaration,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let mut shapes = parameters
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| {
            member.logical_role() == Some(eredu_nn::LogicalParameterRole::LinearWeight)
        })
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target().to_owned(), shape.to_vec()))
        })
        .collect::<BTreeMap<_, _>>();
    if args.tie_word_embeddings {
        let (name, shape) = tied_output_source(declaration)?;
        shapes.insert(name, shape);
    }
    for bank in declaration
        .expert_banks()
        .map_err(|error| error.to_string())?
        .values()
    {
        let source = bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let [gate_up, down] = source.encodings().map_err(|error| error.to_string())?;
        if gate_up.weight_quantization().is_none() && down.weight_quantization().is_none() {
            visit_independent_expert_sources(&source, |_, name, _, shape| {
                shapes.insert(name, shape);
                Ok(())
            })?;
        }
    }
    Ok(shapes)
}

fn expert_variant(
    id: &str,
    discriminator_keys: Vec<String>,
    tensors: impl IntoIterator<Item = (String, Vec<usize>, Option<WeightQuantization>, Vec<String>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let mut constraints = Vec::new();
    for (name, shape, quantization, aliases) in tensors {
        constraints.extend(matrix_constraints(&name, shape, quantization, aliases)?);
    }
    Ok(LayoutVariant {
        id: id.into(),
        tensors: constraints,
        discriminator_keys,
    })
}

fn matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
    aliases: Vec<String>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    matrix_for_linear_format(name, aliases, shape, LinearFormat::from(quantization), None)
        .map_err(|error| error.to_string())
}

/// Builds the canonical Qwen GGUF catalog plan.
pub fn gguf_plan(declaration: &super::Declaration) -> Result<GgufCheckpointPlan, String> {
    let args = declaration.args();
    let declarations = super::parallel::parameter_description(
        declaration,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    for group in declarations.groups() {
        if group.role() == eredu_runtime::ParameterRole::ExpertIntermediate {
            continue;
        }
        for parameter in group.members() {
            tensors.push(gguf_tensor(
                gguf_parameter_name(parameter.target(), &args.parameter_root)?,
                parameter.global_shape().to_vec(),
                if parameter.logical_linear_format().is_some() {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            ));
        }
    }
    if args.is_moe() {
        for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
            let bank = declaration
                .expert_bank(layer)
                .map_err(|error| error.to_string())?;
            let source = bank
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            let block = format!("blk.{layer}");
            for (field, shape) in [
                ("ffn_gate_exps.weight", source.unfused_shape()),
                ("ffn_up_exps.weight", source.unfused_shape()),
                ("ffn_down_exps.weight", source.down().shape.clone()),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{field}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        }
    }
    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new(
        format!("{:?} GGUF", args.variant),
        tensors,
        Vec::new(),
        policy,
    )
    .map_err(|error| error.to_string())
}

fn gguf_tensor(
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
        .ok_or_else(|| format!("Qwen {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Qwen {name} geometry overflows"))
}

const GGUF_ROOT_NAMES: &[(&str, &str)] = &[
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];
const GGUF_EXPERT_NAMES: &[(&str, &str)] = &[
    ("ffn_gate_inp", "mlp.gate"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
];
const GGUF_LAYER_NAMES: &[(&str, &str)] = &[
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("rope_freqs", "rope_freqs"),
];
fn has_prefix(name: &str, prefix: &str) -> bool {
    name == prefix
        || name
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

/// Translates one GGUF tensor name into the canonical Qwen parameter identity.
pub fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    for &(source, target) in GGUF_ROOT_NAMES {
        if has_prefix(name, source) {
            return name.replacen(source, target, 1);
        }
    }
    let Some((layer, parameter)) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.to_owned();
    };
    for &(source, target) in GGUF_EXPERT_NAMES
        .iter()
        .filter(|_| is_moe)
        .chain(GGUF_LAYER_NAMES)
    {
        if has_prefix(parameter, source) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_owned()
}

fn gguf_parameter_name(name: &str, root: &str) -> Result<String, String> {
    for &(physical, canonical) in GGUF_ROOT_NAMES {
        let canonical = canonical.replacen("model", root, 1);
        if has_prefix(name, &canonical) {
            return Ok(name.replacen(&canonical, physical, 1));
        }
    }
    if let Some((layer, parameter)) = name
        .strip_prefix(&format!("{root}.layers."))
        .and_then(|rest| rest.split_once('.'))
    {
        for &(physical, canonical) in GGUF_EXPERT_NAMES.iter().chain(GGUF_LAYER_NAMES) {
            if has_prefix(parameter, canonical) {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(canonical, physical, 1)
                ));
            }
        }
    }
    Err(format!("Qwen GGUF has no storage declaration for {name}"))
}
