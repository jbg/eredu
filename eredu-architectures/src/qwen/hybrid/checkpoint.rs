//! SafeTensors contracts and fused-projection recipes for the hybrid decoder.

use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::text_checkpoint_aliases as aliases;

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
        GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, MatrixScaleNames,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
        TensorOperation,
    },
    store::{TensorSelection, WeightStoreBackend},
    WeightQuantization,
};

/// Derives a hybrid text configuration whose physical matrix formats reflect
/// load-time quantization.
pub fn load_time_quantization(
    config: &HybridConfig,
    quantization: WeightQuantization,
) -> Result<HybridConfig, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = config.policy().clone();
    target.fp8 = None;
    target.quantization = Some(quantization);
    target.linear_formats.clear();
    target.validate().map_err(|error| error.to_string())?;
    config.with_formats(target).map_err(|e| e.to_string())
}

/// Derives a conditional hybrid configuration whose text and aligned vision
/// matrix formats reflect load-time quantization.
pub fn conditional_load_time_quantization(
    config: &ParsedHybridConfig,
    quantization: WeightQuantization,
) -> Result<ParsedHybridConfig, String> {
    let mut target = config.clone();
    target.text = load_time_quantization(&config.text, quantization)?;
    if let Some(vision) = target.vision.as_mut() {
        *vision = std::sync::Arc::new(
            vision
                .with_quantization(quantization)
                .map_err(|e| e.to_string())?,
        );
        vision
            .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
            .map_err(|error| error.to_string())?;
    }
    Ok(target)
}

/// Applies canonical text and optional projector checkpoint formats to a
/// complete conditional hybrid configuration.
pub fn conditional_with_checkpoint_formats(
    config: &ParsedHybridConfig,
    text_formats: HashMap<String, WeightQuantization>,
    vision_formats: Option<HashMap<String, WeightQuantization>>,
) -> Result<ParsedHybridConfig, String> {
    let mut target = config.clone();
    let mut text = target.text.policy().clone();
    text.linear_formats = text_formats
        .into_iter()
        .map(|(name, format)| (name, format.into()))
        .collect();
    target.text = target.text.with_formats(text).map_err(|e| e.to_string())?;
    match (target.vision.as_mut(), vision_formats) {
        (Some(vision), Some(formats)) => {
            let mut config = vision.config().clone();
            config.linear_formats = formats
                .into_iter()
                .map(|(name, format)| (name, format.into()))
                .collect();
            *vision = std::sync::Arc::new(vision.with_formats(config).map_err(|e| e.to_string())?);
            vision
                .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
                .map_err(|error| error.to_string())?;
        }
        (None, None) => {}
        (Some(_), None) => {
            return Err("conditional Qwen projector formats are missing".into());
        }
        (None, Some(_)) => {
            return Err("conditional Qwen projector formats have no vision configuration".into());
        }
    }
    Ok(target)
}

/// Builds the projector checkpoint contract from an admitted conditional
/// Qwen3.5 composite.
///
/// Qwen3-Next, vision-mode, and decoder/projector width compatibility remain
/// architecture policy even when callers possess the normalized component
/// configurations independently.
pub fn conditional_projector_gguf_plan(
    config: &ParsedHybridConfig,
) -> Result<GgufCheckpointPlan, String> {
    if config.text.variant == HybridVariant::Qwen3Next {
        return Err("Qwen3-Next GGUF cannot attach a vision projector".into());
    }
    let vision = config
        .vision
        .as_ref()
        .ok_or_else(|| "conditional Qwen3.5 projector plan requires vision config".to_string())?;
    vision
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(|error| error.to_string())?;
    if vision.out_hidden_size != config.text.hidden_size {
        return Err(format!(
            "Qwen3.5 projector output {} does not match text hidden size {}",
            vision.out_hidden_size, config.text.hidden_size
        ));
    }
    crate::qwen::vision::gguf_plan(vision)
}

/// Returns all derived recipes owned by Qwen hybrid static modules.
pub fn static_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = super::safetensors::recipes(store, |name| {
        !name.starts_with("model.layers.") && !name.starts_with("model.visual.blocks.")
    })?;
    let patch0 = "model.visual.patch_embed.proj.weight.0";
    let patch1 = "model.visual.patch_embed.proj.weight.1";
    if store.source_metadata(patch0).is_ok() && store.source_metadata(patch1).is_ok() {
        recipes.insert(
            "model.visual.patch_embed.proj.weight".into(),
            crate::qwen::vision::temporal_patch_recipe([
                DerivedWeightRecipe::source(patch0, TensorSelection::Full),
                DerivedWeightRecipe::source(patch1, TensorSelection::Full),
            ]),
        );
    }
    if store.source_backend().map_err(|error| error.to_string())? == WeightStoreBackend::Gguf {
        let name = "model.norm.weight";
        if store.source_metadata(name).is_ok() {
            recipes.insert(
                name.into(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    Ok(recipes)
}

// Physical matrix companions use dotted names; grouped construction slots use
// underscores. Namespace aliases and companion spelling stay format-owned.
fn expert_checkpoint_aliases(name: &str) -> Vec<String> {
    let mut names = aliases(name);
    for suffix in ["scales", "biases"] {
        if let Some(primary) = name.strip_suffix(&format!("_{suffix}")) {
            let dotted = format!("{primary}.{suffix}");
            names.push(dotted.clone());
            names.extend(aliases(&dotted));
        }
    }
    names
}

/// Returns the complete derived-weight catalog for one target or MTP unit.
pub fn unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    config: &HybridConfig,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    config.validate().map_err(|error| error.to_string())?;
    let (_, _, layout) =
        super::parameters::execution_schedule(config).map_err(|error| error.to_string())?;
    let address = layout
        .address(flat)
        .ok_or_else(|| format!("Qwen hybrid has no declared unit {flat}"))?;
    let prefix = format!("model.layers.{flat}.");
    let mut recipes = super::safetensors::recipes(store, |name| name.starts_with(&prefix))?;
    if address.group() == 0
        && config.variant == HybridVariant::Qwen3Next
        && matches!(
            config.layer_schedule.get(flat),
            Some(HybridLayerPolicy::LinearAttention)
        )
    {
        let fused = format!("model.layers.{flat}.linear_attn.in_proj_qkvz.weight");
        if store.tensor_metadata(&fused).is_ok() {
            recipes.extend(
                qwen3_next_fused_recipes(store, config, flat)?
                    .iter()
                    .map(|(name, recipe)| (name.to_owned(), recipe.clone())),
            );
        }
    }
    if store.source_backend().map_err(|error| error.to_string())? == WeightStoreBackend::Gguf {
        add_gguf_unit_transforms(&mut recipes, store, config, flat)?;
    }
    if !config.is_moe() {
        return Ok(recipes);
    }
    let bank = super::block::expert_bank_spec(config, flat).map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let root = crate::qwen::checkpoint::expert_source_root(&source)?;
    for (binding, recipe) in crate::shared_routed::checkpoint::gated_expert_recipes(
        store,
        root,
        0..source.group_count(),
        TensorSelection::Full,
        expert_checkpoint_aliases,
    )? {
        let target = crate::qwen::checkpoint::expert_binding_target(&source, &binding)?;
        if recipe != DerivedWeightRecipe::source(&target, TensorSelection::Full) {
            recipes.insert(target, recipe);
        }
    }
    Ok(recipes)
}

/// Returns the derived-weight catalog for one canonical conditional-graph ordinal.
///
/// Conditional Qwen prepends its vision units to the target/MTP schedule. Keeping
/// that translation here makes the architecture's flat execution layout, rather
/// than a backend's reconstruction of group numbers, authoritative for recipes.
pub fn conditional_unit_recipes(
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    config: &ParsedHybridConfig,
    ordinal: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    config.text.validate().map_err(|error| error.to_string())?;
    let vision = config
        .vision
        .as_ref()
        .ok_or("conditional Qwen recipe selection requires vision config")?;
    vision
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(|error| error.to_string())?;
    let vision_layers = vision.layer_count();
    let (_, _, layout) =
        super::parameters::execution_schedule(&config.text).map_err(|error| error.to_string())?;
    if ordinal < vision_layers {
        let prefix = format!("model.visual.blocks.{ordinal}.");
        return super::safetensors::recipes(store, |name| name.starts_with(&prefix));
    }
    let decoder_ordinal = ordinal - vision_layers;
    if layout.address(decoder_ordinal).is_none() {
        return Err(format!(
            "conditional Qwen execution ordinal {ordinal} is outside its architecture layout"
        ));
    }
    unit_recipes(store, &config.text, decoder_ordinal)
}

/// Returns canonical lazy-loading recipes for one routed target or MTP expert.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if !config.is_moe() {
        return Err("independent expert recipes require a routed Qwen hybrid model".into());
    }
    let bank = super::block::expert_bank_spec(config, layer).map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    selected_expert_recipes(catalog, &source, expert)
}

/// Selects physical storage for one expert of an already resolved declaration.
fn selected_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if expert >= source.group_count() {
        return Err(format!("Qwen hybrid has no declared expert {expert}"));
    }
    crate::shared_routed::checkpoint::gated_expert_recipes(
        catalog,
        crate::qwen::checkpoint::expert_source_root(source)?,
        expert..expert + 1,
        TensorSelection::Range {
            axis: 0,
            start: expert,
            end: expert + 1,
        },
        expert_checkpoint_aliases,
    )
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
) -> Result<crate::ExpertResidencyCatalog, String> {
    config.validate().map_err(|error| error.to_string())?;
    if !config.is_moe() {
        return Err("Qwen hybrid expert residency requires a routed model".into());
    }
    let (schedule, _, layout) =
        super::parameters::execution_schedule(config).map_err(|error| error.to_string())?;
    let banks = super::block::expert_bank_specs(config).map_err(|error| error.to_string())?;
    let capacity = banks.values().try_fold(0usize, |capacity, bank| {
        capacity
            .checked_add(bank.group_count() as usize)
            .ok_or_else(|| "Qwen hybrid expert residency catalog size overflowed".to_string())
    })?;
    let mut units = Vec::with_capacity(capacity);
    for (layer, bank) in banks {
        let source = bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let address = layout
            .address(layer)
            .ok_or("unresolved Qwen hybrid bank invocation")?;
        let owner_group = layout
            .group_id(address.group())
            .ok_or("unresolved Qwen hybrid bank owner")?
            .clone();
        let owner_unit = address.index();
        let unit_path = schedule
            .unit_path(address.group(), address.index())
            .map_err(|error| error.to_string())?;
        for expert in 0..source.group_count() {
            let recipes = selected_expert_recipes(catalog, &source, expert)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases");
            let parameters = recipes
                .into_iter()
                .map(|(binding, recipe)| {
                    let target = crate::qwen::checkpoint::expert_binding_target(&source, &binding)?;
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
                    owner_unit,
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

fn add_gguf_unit_transforms(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    store: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
    config: &HybridConfig,
    flat: usize,
) -> Result<(), String> {
    let target_layers = usize::try_from(config.num_hidden_layers)
        .map_err(|_| "invalid Qwen hybrid layer count".to_string())?;
    let root = if flat < target_layers {
        format!("model.layers.{flat}")
    } else {
        format!("mtp.layers.{}", flat - target_layers)
    };
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ] {
        let name = format!("{root}.{suffix}");
        if store.source_metadata(&name).is_ok() {
            recipes.insert(
                name.clone(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    let a_log = format!("{root}.linear_attn.A_log");
    if store.source_metadata(&a_log).is_ok() {
        recipes.insert(
            a_log.clone(),
            DerivedWeightRecipe::NegLog {
                input: Box::new(DerivedWeightRecipe::source(a_log, TensorSelection::Full)),
            },
        );
    }
    let shared_gate = format!("{root}.mlp.shared_expert_gate.weight");
    if let Ok(metadata) = store.source_metadata(&shared_gate) {
        if metadata.logical_shape.len() == 1 {
            recipes.insert(
                shared_gate.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(
                        shared_gate,
                        TensorSelection::Full,
                    )),
                    shape: vec![1, metadata.logical_shape[0]],
                },
            );
        }
    }
    let conv = format!("{root}.linear_attn.conv1d.weight");
    if let Ok(metadata) = store.source_metadata(&conv) {
        if metadata.logical_shape.len() == 2 {
            let source = DerivedWeightRecipe::source(conv.clone(), TensorSelection::Full);
            let source = if config.variant == HybridVariant::Qwen3Next {
                source
            } else {
                crate::gated_delta::checkpoint::grouped_value_heads(
                    source,
                    &metadata.logical_shape,
                    0,
                    2 * config.linear_num_key_heads as usize * config.linear_key_head_dim as usize,
                    config.linear_num_key_heads as usize,
                    config.linear_num_value_heads as usize,
                )?
            };
            recipes.insert(
                conv.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(source),
                    shape: vec![metadata.logical_shape[0], 1, metadata.logical_shape[1]],
                },
            );
        }
    }
    if config.variant == HybridVariant::Qwen3Next {
        return Ok(());
    }
    for suffix in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.dt_bias",
        "linear_attn.A_log",
        "linear_attn.out_proj.weight",
    ] {
        let name = format!("{root}.{suffix}");
        let Ok(metadata) = store.source_metadata(&name) else {
            continue;
        };
        let base = recipes
            .remove(&name)
            .unwrap_or_else(|| DerivedWeightRecipe::source(name.clone(), TensorSelection::Full));
        let axis = usize::from(suffix == "linear_attn.out_proj.weight");
        let prefix = if suffix == "linear_attn.in_proj_qkv.weight" {
            2 * config.linear_num_key_heads as usize * config.linear_key_head_dim as usize
        } else {
            0
        };
        recipes.insert(
            name,
            crate::gated_delta::checkpoint::grouped_value_heads(
                base,
                &metadata.logical_shape,
                axis,
                prefix,
                config.linear_num_key_heads as usize,
                config.linear_num_value_heads as usize,
            )?,
        );
    }
    Ok(())
}

use super::{
    fp8_block_row_widths, fused_projection_widths, HybridConfig, HybridLayerPolicy, HybridVariant,
    ParsedHybridConfig,
};

/// Translates one hybrid projector GGUF name, including split patch weights.
pub fn translate_vision_gguf_weight_name(name: &str, deepstack: &[i32]) -> String {
    match name {
        "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
        "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
        _ => crate::qwen::vision::translate_gguf_weight_name(name, deepstack),
    }
}

/// Builds the strict hybrid text SafeTensors catalog.
pub fn safetensors_plan(config: &HybridConfig) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    let statics = config.statics();
    for declaration in statics
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        match declaration.role {
            LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding => matrix(
                config,
                &mut common,
                declaration.parameter.id.as_str(),
                declaration.shape,
            )?,
            _ => floating(
                &mut common,
                declaration.parameter.id.as_str(),
                declaration.shape,
            ),
        }
    }
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        add_block_at(
            config,
            &format!("model.layers.{layer}"),
            policy,
            &mut common,
            &mut groups,
        )?;
    }
    let mtp_layers = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| "mtp_num_hidden_layers must be non-negative")?;
    if mtp_layers > 0 {
        let shared = config
            .prediction_shared()
            .ok_or("undeclared hybrid prediction shared scope")?;
        for parameter in shared
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            match parameter.role {
                LogicalParameterRole::Normalization => floating(
                    &mut common,
                    parameter.parameter.id.as_str(),
                    parameter.shape,
                ),
                LogicalParameterRole::LinearWeight => common.extend(
                    matrix_for_linear_format(
                        parameter.parameter.id.as_str(),
                        Vec::<String>::new(),
                        parameter.shape,
                        parameter
                            .linear_format
                            .expect("declared prediction fusion format")
                            .encoding(),
                        None,
                    )
                    .map_err(|error| error.to_string())?,
                ),
                _ => return Err("unsupported shared prediction parameter role".into()),
            }
        }
        for depth in 0..mtp_layers {
            add_block_at(
                config,
                &format!("mtp.layers.{depth}"),
                HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
                &mut common,
                &mut groups,
            )?;
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", config.model_type),
        common,
        groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

/// Builds the complete SafeTensors catalog for a text-only or conditional hybrid model.
pub fn composite_safetensors_plan(
    config: &ParsedHybridConfig,
) -> Result<SafetensorsCheckpointPlan, String> {
    let mut text = safetensors_plan(&config.text)?;
    let Some(vision) = config.vision.as_ref() else {
        return Ok(text);
    };
    vision
        .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
        .map_err(|error| error.to_string())?;
    if vision.out_hidden_size != config.text.hidden_size {
        return Err(format!(
            "conditional Qwen projector output {} does not match text hidden size {}",
            vision.out_hidden_size, config.text.hidden_size
        ));
    }
    let vision = crate::qwen::vision::safetensors_plan(vision)?;
    text.common_tensors.extend(vision.common_tensors);
    text.layout_groups.extend(vision.layout_groups);
    let policy = text.catalog_policy;
    SafetensorsCheckpointPlan::new(
        format!("{} composite SafeTensors", config.text.model_type),
        text.common_tensors,
        text.layout_groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

/// Builds the canonical llama.cpp Qwen3-Next/Qwen3.5 tensor catalog.
pub fn gguf_plan(config: &HybridConfig) -> Result<GgufCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let declarations = super::parameters::parameter_description(
        config,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let prediction_groups = super::parameters::prediction_groups(&declarations)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut omitted = BTreeSet::new();
    let mut convolutions = BTreeSet::new();
    let mut dense_vectors = BTreeSet::new();
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        let root = format!("model.layers.{layer}");
        let physical = format!("blk.{layer}");
        let scope = super::BlockParameters::at(config, &root, policy).map_err(|e| e.to_string())?;
        if let super::specs::MixerParameters::Recurrent(spec) = &scope.mixer {
            convolutions.insert(spec.convolution.weight.id.as_str().to_owned());
            dense_vectors.insert(spec.transition_log.id.as_str().to_owned());
            if config.variant == HybridVariant::Qwen3Next {
                for projection in [
                    &spec.input_qkv,
                    &spec.input_gate,
                    &spec.input_beta,
                    &spec.input_decay,
                ] {
                    omitted.insert(projection.weight.id.as_str().to_owned());
                }
                for (name, shape) in fused_input_parameters(&spec, &root)? {
                    tensors.push(gguf_tensor(
                        gguf_parameter_name(&name)?,
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
            }
        }
        if let super::specs::FeedForwardParameters::Routed(spec) = &scope.feed_forward {
            let source = spec
                .routed
                .experts
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            omitted.extend([
                source.gate_up().parameter.id.as_str().to_owned(),
                source.down().parameter.id.as_str().to_owned(),
                spec.shared_gate.weight.id.as_str().to_owned(),
            ]);
            for (field, shape) in [
                ("ffn_gate_exps.weight", source.unfused_shape()),
                ("ffn_up_exps.weight", source.unfused_shape()),
                ("ffn_down_exps.weight", source.down().shape.clone()),
            ] {
                tensors.push(gguf_tensor(
                    format!("{physical}.{field}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
            let declaration = spec
                .shared_gate
                .logical_parameters()
                .map_err(|error| error.to_string())?
                .remove(0);
            let gate_name = gguf_parameter_name(declaration.parameter.id.as_str())?;
            let mut vector = declaration.shape.clone();
            vector.remove(0);
            groups.push(AlternativeLayoutGroup {
                id: format!("{physical} shared-expert gate rank"),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "matrix".into(),
                        tensors: vec![gguf_tensor(
                            &gate_name,
                            declaration.shape,
                            TensorOperation::Matrix,
                        )],
                        discriminator_keys: vec![gate_name.clone()],
                    },
                    LayoutVariant {
                        id: "vector".into(),
                        tensors: vec![gguf_tensor(&gate_name, vector, TensorOperation::Vector)],
                        discriminator_keys: vec![gate_name],
                    },
                ],
            });
        }
    }
    for group in declarations.groups() {
        // Prediction-only storage has separate source admission. Shared pinned
        // target resources retain their target authority and GGUF requirements.
        let prediction_only = match group.owner() {
            eredu_runtime::ParameterGroupOwner::ExecutionUnit { group, .. } => {
                prediction_groups.contains(group.as_str())
            }
            eredu_runtime::ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => consumers
                .iter()
                .all(|(group, _)| prediction_groups.contains(group.as_str())),
            _ => false,
        };
        if prediction_only {
            continue;
        }
        for parameter in group.members() {
            let name = parameter.target();
            if omitted.contains(name) {
                continue;
            }
            let mut shape = parameter.global_shape().to_vec();
            let operation = if convolutions.contains(name) {
                if shape.len() != 3 || shape[1] != 1 {
                    return Err(format!(
                        "hybrid convolution {name} cannot project to GGUF Conv1D storage"
                    ));
                }
                shape.remove(1);
                TensorOperation::Dense
            } else if dense_vectors.contains(name) {
                TensorOperation::Dense
            } else if parameter.logical_linear_format().is_some() {
                TensorOperation::Matrix
            } else {
                TensorOperation::Vector
            };
            tensors.push(gguf_tensor(gguf_parameter_name(name)?, shape, operation));
        }
    }
    let mut catalog = CatalogPolicy::non_strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new(
        format!("{} GGUF", config.model_type),
        tensors,
        groups,
        catalog,
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

const GGUF_ROOT_NAMES: &[(&str, &str)] = &[
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];
const GGUF_LAYER_NAMES: &[(&str, &str)] = &[
    ("attn_norm", "input_layernorm"),
    ("post_attention_norm", "post_attention_layernorm"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_qkvz", "linear_attn.in_proj_qkvz"),
    ("attn_qkv", "linear_attn.in_proj_qkv"),
    ("attn_gate", "linear_attn.in_proj_z"),
    ("ssm_beta", "linear_attn.in_proj_b"),
    ("ssm_alpha", "linear_attn.in_proj_a"),
    ("ssm_ba", "linear_attn.in_proj_ba"),
    ("ssm_conv1d", "linear_attn.conv1d"),
    ("ssm_dt.bias", "linear_attn.dt_bias"),
    ("ssm_a", "linear_attn.A_log"),
    ("ssm_norm", "linear_attn.norm"),
    ("ssm_out", "linear_attn.out_proj"),
    ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
    ("ffn_gate_inp", "mlp.gate"),
    ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
    ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
    ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("rope_freqs", "rope_freqs"),
];

/// Translates llama.cpp Qwen hybrid tensor identities to canonical parameters.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for &(source, target) in GGUF_ROOT_NAMES {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some((layer, parameter)) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.to_owned();
    };
    for &(source, target) in GGUF_LAYER_NAMES {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
            if target.starts_with("mlp.experts.") {
                suffix = match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                };
            }
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    name.to_owned()
}

fn gguf_parameter_name(name: &str) -> Result<String, String> {
    for &(physical, canonical) in GGUF_ROOT_NAMES {
        if name == canonical
            || name
                .strip_prefix(canonical)
                .is_some_and(|suffix| suffix.starts_with('.'))
        {
            return Ok(name.replacen(canonical, physical, 1));
        }
    }
    if let Some((layer, parameter)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
    {
        for &(physical, canonical) in GGUF_LAYER_NAMES {
            if parameter == canonical
                || parameter
                    .strip_prefix(canonical)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(canonical, physical, 1)
                ));
            }
        }
    }
    Err(format!("hybrid GGUF has no storage declaration for {name}"))
}

fn add_block_at(
    config: &HybridConfig,
    root: &str,
    policy: HybridLayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let scope = super::BlockParameters::at(config, root, policy).map_err(|e| e.to_string())?;
    for norm in &scope.norms {
        add_floating(norm, common)?;
    }
    match &scope.mixer {
        super::specs::MixerParameters::Attention(spec) => {
            for projection in spec.projection.projections() {
                add_projection(config, projection, common)?;
            }
            for norm in [&spec.query_norm, &spec.key_norm] {
                add_floating(norm, common)?;
            }
        }
        super::specs::MixerParameters::Recurrent(spec) => {
            add_linear_attention(config, root, spec, common, groups)?
        }
    }
    add_feed_forward(config, root, &scope.feed_forward, common, groups)
}

fn add_projection(
    config: &HybridConfig,
    spec: &eredu_nn::LinearSpec,
    common: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    for declaration in spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        if declaration.role == LogicalParameterRole::LinearWeight {
            matrix(
                config,
                common,
                declaration.parameter.id.as_str(),
                declaration.shape,
            )?;
        } else {
            floating(common, declaration.parameter.id.as_str(), declaration.shape);
        }
    }
    Ok(())
}

fn add_floating(
    spec: &impl DeclareLogicalParameters,
    common: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    for declaration in spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        floating(common, declaration.parameter.id.as_str(), declaration.shape);
    }
    Ok(())
}

fn fused_input_parameters(
    spec: &crate::gated_delta::GatedDeltaMixerSpec,
    root: &str,
) -> Result<[(String, Vec<usize>); 2], String> {
    let width = |first: &eredu_nn::LinearSpec, second: &eredu_nn::LinearSpec| {
        first
            .output
            .checked_add(second.output)
            .and_then(|width| usize::try_from(width).ok())
            .ok_or_else(|| "fused hybrid input geometry overflowed".to_owned())
    };
    Ok([
        (
            format!("{root}.linear_attn.in_proj_qkvz.weight"),
            vec![
                width(&spec.input_qkv, &spec.input_gate)?,
                spec.input_qkv.input as usize,
            ],
        ),
        (
            format!("{root}.linear_attn.in_proj_ba.weight"),
            vec![
                width(&spec.input_beta, &spec.input_decay)?,
                spec.input_beta.input as usize,
            ],
        ),
    ])
}

fn add_linear_attention(
    config: &HybridConfig,
    root: &str,
    spec: &crate::gated_delta::GatedDeltaMixerSpec,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let mut inputs = Vec::new();
    for projection in [
        &spec.input_qkv,
        &spec.input_gate,
        &spec.input_beta,
        &spec.input_decay,
    ] {
        for declaration in projection
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            inputs.push((
                declaration.parameter.id.as_str().to_owned(),
                declaration.shape,
            ));
        }
    }
    let mut variants = vec![layout(config, "split", inputs)?];
    if config.variant == HybridVariant::Qwen3Next {
        variants.push(layout(
            config,
            "fused",
            fused_input_parameters(&spec, root)?,
        )?);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root} linear-attention inputs"),
        required: true,
        variants,
    });
    add_floating(&spec.convolution, common)?;
    for declaration in spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        if [
            &spec.decay_bias.id,
            &spec.transition_log.id,
            &spec.normalization_weight.id,
        ]
        .contains(&&declaration.parameter.id)
        {
            floating(common, declaration.parameter.id.as_str(), declaration.shape);
        }
    }
    add_projection(config, &spec.output, common)
}

fn expert_parameters(
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<[(String, Vec<usize>); 2], String> {
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    Ok([
        (
            source.gate_up().parameter.id.as_str().to_owned(),
            source.gate_up().shape.clone(),
        ),
        (
            source.down().parameter.id.as_str().to_owned(),
            source.down().shape.clone(),
        ),
    ])
}

fn add_feed_forward(
    config: &HybridConfig,
    root: &str,
    parameters: &super::specs::FeedForwardParameters,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let prefix = format!("{root}.mlp");
    if let super::specs::FeedForwardParameters::Dense(spec) = parameters {
        for projection in spec.projections() {
            add_projection(config, projection, common)?;
        }
        return Ok(());
    }
    let super::specs::FeedForwardParameters::Routed(spec) = parameters else {
        unreachable!()
    };
    for declaration in spec
        .routed
        .selector
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        matrix(
            config,
            common,
            declaration.parameter.id.as_str(),
            declaration.shape,
        )?;
    }
    for projection in spec
        .shared
        .projections()
        .into_iter()
        .chain(std::iter::once(&spec.shared_gate))
    {
        add_projection(config, projection, common)?;
    }
    let packed_parameters = expert_parameters(&spec.routed.experts)?;
    let independent = independent_expert_parameters(&spec.routed.experts)?;
    groups.push(AlternativeLayoutGroup {
        id: format!("{prefix}.experts storage"),
        required: true,
        variants: vec![
            layout(config, "packed", packed_parameters)?,
            layout(config, "independent", independent)?,
        ],
    });
    Ok(())
}

fn independent_expert_parameters(
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<Vec<(String, Vec<usize>)>, String> {
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let root = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("hybrid expert source has no declared gate/up root")?;
    let read = source.unfused_shape();
    let mut independent = Vec::with_capacity(mul(source.group_count(), 3)?);
    for expert in 0..source.group_count() {
        for (field, shape) in [
            ("gate_proj", read[1..].to_vec()),
            ("up_proj", read[1..].to_vec()),
            ("down_proj", source.down().shape[1..].to_vec()),
        ] {
            independent.push((format!("{root}.{expert}.{field}.weight"), shape));
        }
    }
    Ok(independent)
}

/// Projects matrix source geometry from logical slots and explicit fused and
/// independent source layouts, without reversing physical packing.
pub(crate) fn source_linear_shapes(
    config: &HybridConfig,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let declaration = super::parameters::parameter_description(
        config,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let mut shapes = declaration
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
    if config.variant == HybridVariant::Qwen3Next {
        for (layer, policy) in config.layer_schedule.iter().enumerate() {
            if *policy == HybridLayerPolicy::LinearAttention {
                let root = format!("model.layers.{layer}");
                let scope =
                    super::BlockParameters::target(config, layer).map_err(|e| e.to_string())?;
                let super::specs::MixerParameters::Recurrent(spec) = &scope.mixer else {
                    return Err("fused source references a nonrecurrent unit".into());
                };
                shapes.extend(fused_input_parameters(spec, &root)?);
            }
        }
    }
    for bank in super::block::expert_bank_specs(config)
        .map_err(|error| error.to_string())?
        .values()
    {
        shapes.extend(independent_expert_parameters(bank)?);
    }
    Ok(shapes)
}

/// Projects the text and optional vision matrix geometry from their construction
/// declarations. Position tables keep their separate semantic role.
pub(crate) fn conditional_source_linear_shapes(
    config: &ParsedHybridConfig,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let mut shapes = source_linear_shapes(&config.text)?;
    if let Some(vision) = &config.vision {
        vision
            .validate_for(crate::qwen::vision::VisionMode::WindowScheduled)
            .map_err(|error| error.to_string())?;
        if vision.out_hidden_size != config.text.hidden_size {
            return Err(format!(
                "conditional Qwen projector output {} does not match text hidden size {}",
                vision.out_hidden_size, config.text.hidden_size
            ));
        }
        shapes.extend(crate::qwen::vision::source_linear_shapes(vision)?);
    }
    Ok(shapes)
}

/// Scalar storage extensions share one calibration value across split projections.
pub(crate) fn qwen3_next_global_sources(
    config: &HybridConfig,
    root: &str,
) -> Result<Vec<(String, [eredu_nn::ParameterId; 2])>, String> {
    if config.variant != HybridVariant::Qwen3Next {
        return Ok(Vec::new());
    }
    let mut sources = Vec::new();
    let scope = super::BlockParameters::at(config, root, HybridLayerPolicy::LinearAttention)
        .map_err(|e| e.to_string())?;
    let super::specs::MixerParameters::Recurrent(spec) = &scope.mixer else {
        unreachable!()
    };
    for (source, first, second) in [
        ("in_proj_qkvz", &spec.input_qkv, &spec.input_gate),
        ("in_proj_ba", &spec.input_beta, &spec.input_decay),
    ] {
        if let (Some(first_global), Some(second_global)) =
            (first.format.global_scale(), second.format.global_scale())
        {
            if first.format.row_layout().partitions() != 1
                || second.format.row_layout().partitions() != 1
            {
                return Err("Qwen3-Next fused global scales require scalar row layouts".into());
            }
            sources.push((
                format!("{root}.linear_attn.{source}.global_scale"),
                [first_global.id.clone(), second_global.id.clone()],
            ));
        }
    }
    Ok(sources)
}

/// Resolves group-major QKVZ/BA storage into canonical split parameters.
/// Weights, affine/FP8 companions and shared scalar global scales form one atomic recipe set.
pub fn qwen3_next_fused_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
    layer: usize,
) -> Result<AtomicRecipeSet, String> {
    if config.variant != HybridVariant::Qwen3Next {
        return Err("fused hybrid recipes require Qwen3-Next policy".into());
    }
    let prefix = format!("model.layers.{layer}.linear_attn");
    let (widths, ba_width) = fused_projection_widths(config).map_err(|error| error.to_string())?;
    let groups = dim(config.linear_num_key_heads, "linear_num_key_heads")?;
    let mut outputs = Vec::new();
    for suffix in ["weight", "scales", "biases"] {
        let qkvz = format!("{prefix}.in_proj_qkvz.{suffix}");
        if catalog.tensor_metadata(&qkvz).is_ok() {
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &qkvz,
                groups,
                &widths.map(|width| usize::try_from(width).unwrap()),
                [
                    (format!("{prefix}.in_proj_qkv.{suffix}"), vec![0, 1, 2]),
                    (format!("{prefix}.in_proj_z.{suffix}"), vec![3]),
                ],
            )?;
        }
        let ba = format!("{prefix}.in_proj_ba.{suffix}");
        if catalog.tensor_metadata(&ba).is_ok() {
            let width = usize::try_from(ba_width).map_err(|_| "invalid BA width")?;
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &ba,
                groups,
                &[width, width],
                [
                    (format!("{prefix}.in_proj_b.{suffix}"), vec![0]),
                    (format!("{prefix}.in_proj_a.{suffix}"), vec![1]),
                ],
            )?;
        }
    }
    for (source, targets) in qwen3_next_global_sources(config, &format!("model.layers.{layer}"))? {
        if let Ok(metadata) = catalog.tensor_metadata(&source) {
            if !metadata.logical_shape.is_empty()
                || metadata.stored_dtype != eredu_checkpoint::StoredDtype::F32
            {
                return Err(format!(
                    "Qwen3-Next fused global scale {source:?} must be an F32 scalar"
                ));
            }
            for target in targets {
                outputs.push((
                    target.as_str().to_owned(),
                    DerivedWeightRecipe::source(&source, TensorSelection::Full),
                ));
            }
        }
    }
    let scale = format!("{prefix}.in_proj_qkvz.weight_scale_inv");
    if catalog.tensor_metadata(&scale).is_ok() {
        let block_widths = fp8_block_row_widths(&widths)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|width| usize::try_from(width).map_err(|_| "invalid FP8 block width"))
            .collect::<Result<Vec<_>, _>>()?;
        add_grouped_outputs(
            catalog,
            &mut outputs,
            &scale,
            groups,
            &block_widths,
            [
                (
                    format!("{prefix}.in_proj_qkv.weight_scale_inv"),
                    vec![0, 1, 2],
                ),
                (format!("{prefix}.in_proj_z.weight_scale_inv"), vec![3]),
            ],
        )?;
    }
    if catalog
        .tensor_metadata(&format!("{prefix}.in_proj_ba.weight_scale_inv"))
        .is_ok()
    {
        return Err("Qwen3-Next fused BA must remain dense and cannot carry inverse scales".into());
    }
    if outputs.is_empty() {
        return Err(format!("no fused projections found for layer {layer}"));
    }
    AtomicRecipeSet::new(catalog, outputs).map_err(|error| error.to_string())
}

fn add_grouped_outputs<C, const N: usize>(
    catalog: &C,
    outputs: &mut Vec<(String, DerivedWeightRecipe)>,
    source: &str,
    groups: usize,
    widths: &[usize],
    targets: [(String, Vec<usize>); N],
) -> Result<(), String>
where
    C: RecipeCatalog + ?Sized,
{
    let group_width = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width).ok_or("grouped row width overflow")
    })?;
    for (target, components) in targets {
        if catalog.tensor_metadata(&target).is_ok() {
            return Err(format!(
                "fused output {target:?} collides with a physical tensor"
            ));
        }
        let mut indices = Vec::new();
        for component in components {
            let start = widths
                .get(..component)
                .ok_or("invalid grouped component")?
                .iter()
                .sum::<usize>();
            let width = *widths.get(component).ok_or("invalid grouped component")?;
            for group in 0..groups {
                let base = group
                    .checked_mul(group_width)
                    .and_then(|base| base.checked_add(start))
                    .ok_or("grouped row index overflow")?;
                indices.extend(base..base + width);
            }
        }
        outputs.push((
            target,
            DerivedWeightRecipe::source(source, TensorSelection::Indices { axis: 0, indices }),
        ));
    }
    Ok(())
}

fn layout(
    config: &HybridConfig,
    id: &str,
    tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    let discriminator_keys = tensors.iter().map(|(name, _)| name.clone()).collect();
    let mut constraints = Vec::new();
    for (name, shape) in tensors {
        matrix(config, &mut constraints, name, shape)?;
    }
    Ok(LayoutVariant {
        id: id.into(),
        discriminator_keys,
        tensors: constraints,
    })
}

fn matrix(
    config: &HybridConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl Into<String>,
    shape: Vec<usize>,
) -> Result<(), String> {
    let name = name.into();
    let aliases = aliases(&name);
    let format = config.linear_format(&name);
    let scale = matches!(format, eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)).then(|| {
        MatrixScaleNames {
            key: format!("{name}_scale_inv"),
            aliases: aliases
                .iter()
                .map(|name| format!("{name}_scale_inv"))
                .collect(),
        }
    });
    output.extend(
        matrix_for_linear_format(&name, aliases, shape, format, scale)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn floating(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl Into<String>,
    shape: Vec<usize>,
) {
    let name = name.into();
    output.push(
        SafetensorsTensorConstraint::required(&name, shape, StoredDtypeConstraint::Floating)
            .with_aliases(aliases(&name)),
    );
}

fn dim(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be positive"))
}

fn mul(left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| "checkpoint dimension overflow".into())
}
