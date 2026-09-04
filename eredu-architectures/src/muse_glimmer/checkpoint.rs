//! Composite artifact and checkpoint-name policy for Muse-Glimmer.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eredu_checkpoint::{
    composite::{
        ArtifactComponentSchema, ArtifactRole, ComponentId, ComponentParameterCatalog,
        CompositeArtifactError, CompositeArtifactSchema, ProjectorCompatibility,
    },
    expert::{
        resolve_gated_product_expert_recipes, GatedProductExpertLayoutNames,
        GatedProductExpertRecipes, IndependentGatedProductExpertNames,
    },
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
        GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
        SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
    },
    store::TensorSelection,
    WeightQuantization,
};

use super::DecoderConfig;

/// Derives a Muse-Glimmer configuration whose text and vision matrix formats
/// reflect load-time quantization instead of checkpoint-specific selections.
pub fn load_time_quantization(
    args: &DecoderConfig,
    quantization: WeightQuantization,
) -> Result<DecoderConfig, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    if let Some(vision) = &mut target.vision_config {
        vision.weight_quantization = Some(quantization);
        vision.quantized_weight_configs.clear();
    }
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies exact selected text and vision formats to a complete composite.
pub fn with_checkpoint_formats(
    args: &DecoderConfig,
    formats: HashMap<String, WeightQuantization>,
) -> Result<DecoderConfig, String> {
    let mut target = args.clone();
    let mut text = HashMap::new();
    let mut vision = HashMap::new();
    for (name, format) in formats {
        if name.starts_with("model.vision_") {
            vision.insert(name, format);
        } else {
            text.insert(name, format);
        }
    }
    target.quantization = None;
    target.quantized_weights = None;
    target.quantized_weight_configs = (!text.is_empty()).then_some(text);
    if let Some(config) = target.vision_config.as_mut() {
        config.weight_quantization = None;
        config.quantized_weight_configs = vision;
    }
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Decoder plus optional split projector/assistant artifact policy.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArtifactConfig {
    /// Decoder hidden width.
    pub hidden_size: usize,
    /// Image placeholder token.
    pub image_token_id: u32,
    /// Video placeholder token.
    pub video_token_id: u32,
    /// Whether a sibling vision projector is present.
    pub projector: bool,
    /// Whether the projector is the official image-only GGUF sidecar.
    pub image_only_projector: bool,
    /// Whether a required external DFlash assistant is present.
    pub assistant: bool,
}

impl ArtifactConfig {
    /// Builds the decoder-plus-siblings schema checked before loading.
    pub fn artifact_schema(&self) -> Result<CompositeArtifactSchema, CompositeArtifactError> {
        let mut siblings = Vec::new();
        if self.projector {
            siblings.push(ArtifactComponentSchema {
                component: ComponentId::new("vision_projector")?,
                role: ArtifactRole::Projector,
                required: true,
                architecture: "muse_glimmer".into(),
            });
        }
        if self.assistant {
            siblings.push(ArtifactComponentSchema {
                component: ComponentId::new("dflash_assistant")?,
                role: ArtifactRole::Assistant,
                required: true,
                architecture: "muse_glimmer".into(),
            });
        }
        CompositeArtifactSchema::new(
            ArtifactComponentSchema {
                component: ComponentId::new("decoder")?,
                role: ArtifactRole::Decoder,
                required: true,
                architecture: "muse_glimmer".into(),
            },
            siblings,
        )
    }

    /// Assigns stable logical roots to their artifact owner.
    pub fn parameter_catalog(&self) -> Result<ComponentParameterCatalog, CompositeArtifactError> {
        let mut components = vec![(
            ComponentId::new("decoder")?,
            vec![
                "model.embed_tokens.weight".into(),
                "model.norm.weight".into(),
                "lm_head.weight".into(),
            ],
        )];
        if self.projector {
            components.push((
                ComponentId::new("vision_projector")?,
                vec![
                    "model.vision_tower.patch_embedder.patch_embedding.weight".into(),
                    "model.vision_adapter.fc1.weight".into(),
                    "model.vision_projection.weight".into(),
                ],
            ));
        }
        if self.assistant {
            components.push((
                ComponentId::new("dflash_assistant")?,
                vec![
                    "model.encoder.weight".into(),
                    "model.layers.0.self_attn.q_proj.weight".into(),
                ],
            ));
        }
        ComponentParameterCatalog::new(components)
    }

    /// Builds the exact projector identity/width/token compatibility proof.
    pub fn projector_compatibility(
        &self,
        projector_architecture: impl Into<String>,
        projector_output_width: usize,
        projector_tokens: BTreeSet<u32>,
    ) -> ProjectorCompatibility {
        let decoder_tokens = if self.image_only_projector {
            BTreeSet::from([self.image_token_id])
        } else {
            BTreeSet::from([self.image_token_id, self.video_token_id])
        };
        ProjectorCompatibility {
            decoder_architecture: "muse_glimmer".into(),
            projector_architecture: projector_architecture.into(),
            decoder_hidden_width: self.hidden_size,
            projector_output_width,
            decoder_modality_tokens: decoder_tokens,
            projector_modality_tokens: projector_tokens,
        }
    }

    /// Rejects video requests for the official folded image-only GGUF projector.
    pub fn validate_video_admission(&self, contains_video: bool) -> Result<(), String> {
        if contains_video && self.image_only_projector {
            return Err("official Muse-Glimmer GGUF projector is image-only".into());
        }
        Ok(())
    }
}

/// Returns stable parameter identities for one SafeTensors artifact.
pub fn safetensors_parameter_names(args: &DecoderConfig) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".into(),
        "model.norm.weight".into(),
    ];
    if !args.tie_word_embeddings {
        names.push("lm_head.weight".into());
    }
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("model.layers.{layer}");
        for local in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.gate_proj.weight",
        ] {
            names.push(format!("{root}.{local}"));
        }
        if args.is_moe() {
            names.extend([
                format!("{root}.mlp.gate.weight"),
                format!("{root}.mlp.experts.gate_up_proj"),
                format!("{root}.mlp.experts.down_proj"),
            ]);
        } else {
            names.extend([
                format!("{root}.mlp.gate_proj.weight"),
                format!("{root}.mlp.up_proj.weight"),
                format!("{root}.mlp.down_proj.weight"),
            ]);
        }
    }
    names.extend(vision_parameter_names(args));
    names
}

fn vision_parameter_names(args: &DecoderConfig) -> Vec<String> {
    let Some(vision) = &args.vision_config else {
        return Vec::new();
    };
    let mut names = vec![
        "model.vision_tower.patch_embedder.patch_embedding.weight".into(),
        "model.vision_tower.patch_embedder.position_embedding_table.weight".into(),
        "model.vision_tower.ln_pre.weight".into(),
        "model.vision_tower.ln_pre.bias".into(),
        "model.vision_tower.ln_post.weight".into(),
        "model.vision_tower.ln_post.bias".into(),
        "model.vision_adapter.fc1.weight".into(),
        "model.vision_adapter.fc2.weight".into(),
        "model.vision_projection.weight".into(),
    ];
    for layer in 0..vision.layer_count() {
        let root = format!("model.vision_tower.layers.{layer}");
        for local in [
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
            "attn.q_proj.weight",
            "attn.q_proj.bias",
            "attn.k_proj.weight",
            "attn.k_proj.bias",
            "attn.v_proj.weight",
            "attn.v_proj.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ] {
            names.push(format!("{root}.{local}"));
        }
    }
    names
}

/// Builds the complete released SafeTensors catalog, including native vision
/// weights and every admitted dense or routed-expert representation.
pub fn safetensors_plan(args: &DecoderConfig) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "text hidden size")?;
    let vocabulary = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "text layer count")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head width")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head width")?,
        "key/value projection width",
    )?;
    let mut common = Vec::new();
    let mut alternatives = Vec::new();
    add_text_matrix(
        args,
        &mut common,
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        vec![vocabulary, hidden],
    )?;
    common.push(safe_alias(
        "model.language_model.norm.weight",
        "model.norm.weight",
        vec![hidden],
    ));
    if !args.tie_word_embeddings {
        add_text_matrix(
            args,
            &mut common,
            "lm_head.weight",
            "lm_head.weight",
            vec![vocabulary, hidden],
        )?;
    }
    for layer in 0..layers {
        let released = format!("model.language_model.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        for local in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            common.push(safe_alias(
                format!("{released}.{local}"),
                format!("{canonical}.{local}"),
                vec![hidden],
            ));
        }
        for (local, shape) in [
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("self_attn.gate_proj.weight", vec![query, hidden]),
        ] {
            add_text_matrix(
                args,
                &mut common,
                &format!("{released}.{local}"),
                &format!("{canonical}.{local}"),
                shape,
            )?;
        }
        if args.is_moe() {
            add_moe_layout(
                args,
                layer,
                &released,
                &canonical,
                hidden,
                &mut common,
                &mut alternatives,
            )?;
        } else {
            let intermediate = dimension(args.intermediate_size, "text intermediate size")?;
            for (local, shape) in [
                ("mlp.gate_proj.weight", vec![intermediate, hidden]),
                ("mlp.up_proj.weight", vec![intermediate, hidden]),
                ("mlp.down_proj.weight", vec![hidden, intermediate]),
            ] {
                add_text_matrix(
                    args,
                    &mut common,
                    &format!("{released}.{local}"),
                    &format!("{canonical}.{local}"),
                    shape,
                )?;
            }
        }
    }
    add_safetensors_vision(args, hidden, &mut common)?;
    SafetensorsCheckpointPlan::new(
        "Muse-Glimmer SafeTensors",
        common,
        alternatives,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Resolves every released SafeTensors alias and derived expert layout into
/// canonical architecture parameter identities.
///
/// The returned catalog is model-wide but contains only runtime parameter
/// outputs; independent or split expert tensors remain recipe inputs rather
/// than becoming spurious backend-visible destinations.
pub fn safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let plan = safetensors_plan(args)?;
    let constraints = plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .flat_map(|group| group.variants.iter())
            .flat_map(|variant| variant.tensors.iter()),
    );
    let mut recipes = BTreeMap::new();
    for tensor in constraints {
        let source = std::iter::once(tensor.key.clone())
            .chain(tensor.aliases.iter().cloned())
            .find(|name| catalog.tensor_metadata(name).is_ok());
        if let Some(source) = source {
            let target = canonical_safetensors_target(&tensor.key);
            if !is_expert_recipe_intermediate(&target) {
                recipes.insert(
                    target,
                    DerivedWeightRecipe::source(source, TensorSelection::Full),
                );
            }
        }
    }
    if args.is_moe() {
        for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
            let expert = safetensors_expert_recipes(catalog, args, layer)?;
            recipes.insert(expert.target_gate_up, expert.gate_up);
            recipes.insert(expert.target_down, expert.down);
        }
    }
    Ok(recipes)
}

fn is_expert_recipe_intermediate(target: &str) -> bool {
    let Some(local) = target.split(".mlp.experts.").nth(1) else {
        return false;
    };
    local
        .split('.')
        .next()
        .is_some_and(|segment| segment.parse::<usize>().is_ok())
        || local.starts_with("gate_proj")
        || local.starts_with("up_proj")
}

/// Selects the complete architecture-owned SafeTensors recipe group for pinned modules.
///
/// Backends must bind the returned group as a whole and reject unexpected leftovers.
pub fn static_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    select_safetensors_recipe_group(args, catalog, None)
}

/// Selects the complete architecture-owned SafeTensors recipe group for one execution unit.
///
/// `group` and `index` use the canonical Muse-Glimmer layout: vision is group zero and the text
/// decoder is group one. No backend-native parameter topology participates in selection.
pub fn unit_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    catalog: &C,
    group: usize,
    index: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let available = match group {
        0 => args
            .vision_config
            .as_ref()
            .map_or(0, |vision| vision.layer_count()),
        1 => dimension(args.num_hidden_layers, "text layer count")?,
        _ => {
            return Err(format!(
                "Muse-Glimmer recipe group {group} is outside two groups"
            ))
        }
    };
    if index >= available {
        return Err(format!(
            "Muse-Glimmer recipe unit {index} is outside group {group} with {available} units"
        ));
    }
    select_safetensors_recipe_group(args, catalog, Some((group, index)))
}

fn select_safetensors_recipe_group<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    catalog: &C,
    selected: Option<(usize, usize)>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = args
        .vision_config
        .as_ref()
        .map_or(0, |vision| vision.layer_count());
    let text_layers = dimension(args.num_hidden_layers, "text layer count")?;
    let mut selected_recipes = BTreeMap::new();
    for (target, recipe) in safetensors_recipes(args, catalog)? {
        let owner = if let Some(index) =
            indexed_recipe_target(&target, "model.vision_tower.layers.")?
        {
            if index >= vision_layers {
                return Err(format!(
                    "Muse-Glimmer recipe {target:?} names vision unit {index}, but only {vision_layers} exist"
                ));
            }
            Some((0, index))
        } else if let Some(index) = indexed_recipe_target(&target, "model.layers.")? {
            if index >= text_layers {
                return Err(format!(
                    "Muse-Glimmer recipe {target:?} names text unit {index}, but only {text_layers} exist"
                ));
            }
            Some((1, index))
        } else {
            None
        };
        if owner == selected {
            selected_recipes.insert(target, recipe);
        }
    }
    Ok(selected_recipes)
}

fn indexed_recipe_target(target: &str, root: &str) -> Result<Option<usize>, String> {
    let Some(rest) = target.strip_prefix(root) else {
        return Ok(None);
    };
    let Some((index, parameter)) = rest.split_once('.') else {
        return Err(format!(
            "architecture recipe target {target:?} has no parameter below its execution unit"
        ));
    };
    if parameter.is_empty() {
        return Err(format!(
            "architecture recipe target {target:?} has an empty unit parameter"
        ));
    }
    index
        .parse::<usize>()
        .map(Some)
        .map_err(|_| format!("architecture recipe target {target:?} has a non-numeric unit index"))
}

fn canonical_safetensors_target(source: &str) -> String {
    let canonical = source
        .replace("model.language_model.layers.", "model.layers.")
        .replace("model.language_model.embed_tokens.", "model.embed_tokens.")
        .replace("model.language_model.norm.", "model.norm.");
    if canonical.contains(".mlp.experts.") {
        if let Some(prefix) = canonical.strip_suffix(".scales") {
            return format!("{prefix}_scales");
        }
        if let Some(prefix) = canonical.strip_suffix(".biases") {
            return format!("{prefix}_biases");
        }
    }
    canonical
}

fn safetensors_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &DecoderConfig,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let target = format!("model.layers.{layer}.mlp.experts");
    let released = format!("model.language_model.layers.{layer}.mlp.experts");
    let packed_gate_up = first_existing(
        catalog,
        [
            format!("{target}.gate_up_proj"),
            format!("{released}.gate_up_proj"),
            format!("{released}.gate_up_proj.weight"),
        ],
    );
    let packed_down = first_existing(
        catalog,
        [
            format!("{target}.down_proj"),
            format!("{target}.down_proj.weight"),
            format!("{released}.down_proj"),
            format!("{released}.down_proj.weight"),
        ],
    );
    let separate_gate = first_existing(
        catalog,
        [
            format!("{target}.gate_proj"),
            format!("{target}.gate_proj.weight"),
            format!("{released}.gate_proj"),
            format!("{released}.gate_proj.weight"),
        ],
    );
    let separate_up = first_existing(
        catalog,
        [
            format!("{target}.up_proj"),
            format!("{target}.up_proj.weight"),
            format!("{released}.up_proj"),
            format!("{released}.up_proj.weight"),
        ],
    );
    let independent = (0..dimension(args.num_experts, "expert count")?)
        .map(|expert| {
            let root = format!("{released}.{expert}");
            IndependentGatedProductExpertNames {
                gate: format!("{root}.gate_proj.weight"),
                up: format!("{root}.up_proj.weight"),
                down: format!("{root}.down_proj.weight"),
            }
        })
        .collect();
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: format!("{target}.gate_up_proj"),
            target_down: format!("{target}.down_proj"),
            packed_gate_up: packed_gate_up.unwrap_or_default(),
            packed_down: packed_down.unwrap_or_default(),
            separate_gate: separate_gate.unwrap_or_default(),
            separate_up: separate_up.unwrap_or_default(),
            separate_down: first_existing(
                catalog,
                [
                    format!("{target}.down_proj"),
                    format!("{target}.down_proj.weight"),
                    format!("{released}.down_proj"),
                    format!("{released}.down_proj.weight"),
                ],
            )
            .unwrap_or_default(),
            independent,
        },
    )
    .map_err(|error| error.to_string())
}

fn first_existing<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    names: impl IntoIterator<Item = String>,
) -> Option<String> {
    names
        .into_iter()
        .find(|name| catalog.tensor_metadata(name).is_ok())
}

#[allow(clippy::too_many_arguments)]
fn add_moe_layout(
    args: &DecoderConfig,
    layer: usize,
    released: &str,
    canonical: &str,
    hidden: usize,
    common: &mut Vec<SafetensorsTensorConstraint>,
    alternatives: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let experts = dimension(args.num_experts, "expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "expert intermediate size")?;
    add_text_matrix(
        args,
        common,
        &format!("{released}.mlp.gate.weight"),
        &format!("{canonical}.mlp.gate.weight"),
        vec![experts, hidden],
    )?;
    let released_root = format!("{released}.mlp.experts");
    let canonical_root = format!("{canonical}.mlp.experts");
    let packed_gate_up = format!("{released_root}.gate_up_proj");
    let packed_down = format!("{released_root}.down_proj");
    let mut packed = Vec::new();
    add_text_matrix(
        args,
        &mut packed,
        &packed_gate_up,
        &format!("{canonical_root}.gate_up_proj"),
        vec![
            experts,
            checked_mul(2, intermediate, "fused expert width")?,
            hidden,
        ],
    )?;
    add_text_matrix(
        args,
        &mut packed,
        &packed_down,
        &format!("{canonical_root}.down_proj"),
        vec![experts, hidden, intermediate],
    )?;
    let split_gate = format!("{released_root}.gate_proj");
    let split_up = format!("{released_root}.up_proj");
    let split_down = format!("{released_root}.down_proj");
    let mut split = Vec::new();
    for (source, target, shape) in [
        (
            &split_gate,
            format!("{canonical_root}.gate_proj"),
            vec![experts, intermediate, hidden],
        ),
        (
            &split_up,
            format!("{canonical_root}.up_proj"),
            vec![experts, intermediate, hidden],
        ),
        (
            &split_down,
            format!("{canonical_root}.down_proj"),
            vec![experts, hidden, intermediate],
        ),
    ] {
        add_text_matrix(args, &mut split, source, &target, shape)?;
    }
    let mut independent = Vec::new();
    let mut independent_keys = Vec::new();
    for expert in 0..experts {
        for (local, shape) in [
            ("gate_proj.weight", vec![intermediate, hidden]),
            ("up_proj.weight", vec![intermediate, hidden]),
            ("down_proj.weight", vec![hidden, intermediate]),
        ] {
            let source = format!("{released_root}.{expert}.{local}");
            add_text_matrix(args, &mut independent, &source, &source, shape)?;
            independent_keys.push(source);
        }
    }
    alternatives.push(AlternativeLayoutGroup {
        id: format!("Muse-Glimmer layer {layer} expert storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed gate/up bank".into(),
                tensors: packed,
                discriminator_keys: vec![packed_gate_up, packed_down],
            },
            LayoutVariant {
                id: "split expert banks".into(),
                tensors: split,
                discriminator_keys: vec![split_gate, split_up, split_down],
            },
            LayoutVariant {
                id: "independent experts".into(),
                tensors: independent,
                discriminator_keys: independent_keys,
            },
        ],
    });
    Ok(())
}

fn add_safetensors_vision(
    args: &DecoderConfig,
    text_hidden: usize,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let vision = args.vision_config.as_ref().ok_or_else(|| {
        "Muse-Glimmer SafeTensors plan requires a vision configuration".to_string()
    })?;
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let intermediate = dimension(vision.intermediate_size, "vision intermediate size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let temporal = dimension(vision.temporal_patch_size, "vision temporal patch size")?;
    let patch_input = checked_mul(
        checked_mul(3, temporal, "vision temporal channel width")?,
        checked_mul(patch, patch, "vision patch area")?,
        "vision patch input width",
    )?;
    let positions = checked_mul(
        dimension(vision.position_height, "vision position height")?,
        dimension(vision.position_width, "vision position width")?,
        "vision position count",
    )?;
    for (name, shape) in [
        (
            "model.vision_tower.patch_embedder.patch_embedding.weight",
            vec![hidden, patch_input],
        ),
        (
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
            vec![positions, hidden],
        ),
    ] {
        add_vision_matrix(vision, output, name, shape)?;
    }
    for local in [
        "ln_pre.weight",
        "ln_pre.bias",
        "ln_post.weight",
        "ln_post.bias",
    ] {
        output.push(safe(
            local_with_root("model.vision_tower", local),
            vec![hidden],
        ));
    }
    for layer in 0..vision.layer_count() {
        let root = format!("model.vision_tower.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            output.push(safe(format!("{root}.{norm}.weight"), vec![hidden]));
            output.push(safe(format!("{root}.{norm}.bias"), vec![hidden]));
        }
        for projection in ["q_proj", "k_proj", "v_proj", "proj"] {
            let weight = format!("{root}.attn.{projection}.weight");
            add_vision_matrix(vision, output, &weight, vec![hidden, hidden])?;
            output.push(safe(format!("{root}.attn.{projection}.bias"), vec![hidden]));
        }
        let fc1 = format!("{root}.mlp.fc1.weight");
        add_vision_matrix(vision, output, &fc1, vec![intermediate, hidden])?;
        output.push(safe(format!("{root}.mlp.fc1.bias"), vec![intermediate]));
        let fc2 = format!("{root}.mlp.fc2.weight");
        add_vision_matrix(vision, output, &fc2, vec![hidden, intermediate])?;
        output.push(safe(format!("{root}.mlp.fc2.bias"), vec![hidden]));
    }
    let projector = dimension(args.projector_hidden_size, "projector hidden size")?;
    let vision_out = dimension(args.vision_out_hidden_size, "vision output width")?;
    for (name, shape) in [
        (
            "model.vision_adapter.fc1.weight",
            vec![projector, vision_out],
        ),
        (
            "model.vision_adapter.fc2.weight",
            vec![projector, projector],
        ),
        (
            "model.vision_projection.weight",
            vec![text_hidden, projector],
        ),
    ] {
        add_vision_matrix(vision, output, name, shape)?;
    }
    Ok(())
}

fn add_text_matrix(
    args: &DecoderConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    source: &str,
    canonical: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    let aliases = (source != canonical).then(|| canonical.to_string());
    output.extend(
        matrix_for_linear_format(
            source,
            aliases,
            shape,
            args.linear_format_for(canonical),
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn add_vision_matrix(
    vision: &super::VisionConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(
            name,
            std::iter::empty::<String>(),
            shape,
            vision.linear_format_for(name),
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

/// Builds the complete text GGUF catalog admitted for Muse-Glimmer.
pub fn gguf_plan(args: &DecoderConfig) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "text hidden size")?;
    let vocabulary = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "text layer count")?;
    let head = dimension(args.head_dim, "attention head width")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        head,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        head,
        "key/value projection width",
    )?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocabulary, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf(
            "output.weight",
            vec![vocabulary, hidden],
            TensorOperation::Matrix,
        ));
    }
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        for (local, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "post_attention_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "post_ffw_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            ("attn_q_norm.weight", vec![head], TensorOperation::Vector),
            ("attn_k_norm.weight", vec![head], TensorOperation::Vector),
            (
                "attn_q.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_k.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_v.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_output.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "attn_gate.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(gguf(format!("{root}.{local}"), shape, operation));
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "expert count")?;
            let intermediate = dimension(args.moe_intermediate_size, "expert intermediate size")?;
            tensors.extend([
                gguf(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_gate_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_up_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
        } else {
            let intermediate = dimension(args.intermediate_size, "text intermediate size")?;
            tensors.extend([
                gguf(
                    format!("{root}.ffn_gate.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_up.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_down.weight"),
                    vec![hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
        }
    }
    GgufCheckpointPlan::new(
        "Muse-Glimmer GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Builds the official image-only sibling projector GGUF catalog.
pub fn projector_gguf_plan(args: &DecoderConfig) -> Result<GgufCheckpointPlan, String> {
    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| "Muse-Glimmer projector plan requires vision geometry".to_string())?;
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let intermediate = dimension(vision.intermediate_size, "vision intermediate size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let merge = dimension(vision.merge_size, "vision merge size")?;
    let merged = checked_mul(
        hidden,
        checked_mul(merge, merge, "vision merge area")?,
        "merged vision width",
    )?;
    let mut tensors = vec![
        gguf(
            "v.patch_embd.weight",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf(
            "v.position_embd.weight",
            vec![1024, hidden],
            TensorOperation::Dense,
        ),
    ];
    for name in [
        "v.pre_ln.weight",
        "v.pre_ln.bias",
        "v.post_ln.weight",
        "v.post_ln.bias",
    ] {
        tensors.push(gguf(name, vec![hidden], TensorOperation::Dense));
    }
    for layer in 0..vision.layer_count() {
        let root = format!("v.blk.{layer}");
        for local in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
            tensors.push(gguf(
                format!("{root}.{local}"),
                vec![hidden],
                TensorOperation::Dense,
            ));
        }
        for projection in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            tensors.push(gguf(
                format!("{root}.{projection}.weight"),
                vec![hidden, hidden],
                TensorOperation::Matrix,
            ));
            tensors.push(gguf(
                format!("{root}.{projection}.bias"),
                vec![hidden],
                TensorOperation::Dense,
            ));
        }
        tensors.extend([
            gguf(
                format!("{root}.ffn_up.weight"),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.ffn_up.bias"),
                vec![intermediate],
                TensorOperation::Dense,
            ),
            gguf(
                format!("{root}.ffn_down.weight"),
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.ffn_down.bias"),
                vec![hidden],
                TensorOperation::Dense,
            ),
        ]);
    }
    tensors.extend([
        gguf("mm.0.weight", vec![4096, merged], TensorOperation::Matrix),
        gguf("mm.1.weight", vec![4096, 4096], TensorOperation::Matrix),
        gguf(
            "mm.2.weight",
            vec![dimension(args.hidden_size, "text hidden size")?, 4096],
            TensorOperation::Matrix,
        ),
    ]);
    GgufCheckpointPlan::new(
        "Muse-Glimmer projector GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn safe_alias(
    source: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    let source = source.into();
    let canonical = canonical.into();
    let aliases = (source != canonical).then_some(canonical);
    safe(source, shape).with_aliases(aliases)
}

fn safe(name: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
}

fn gguf(
    name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(name, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Muse-Glimmer {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Muse-Glimmer {name} geometry overflows"))
}

fn local_with_root(root: &str, local: &str) -> String {
    format!("{root}.{local}")
}

/// Resolves packed, separate, or independent expert tensors into neutral packed slots.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let root = format!("model.layers.{layer}.mlp.experts");
    let raw = format!("blk.{layer}");
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: format!("{root}.gate_up_proj"),
            target_down: format!("{root}.down_proj"),
            packed_gate_up: format!("{root}.gate_up_proj"),
            packed_down: format!("{root}.down_proj"),
            separate_gate: format!("{raw}.ffn_gate_exps.weight"),
            separate_up: format!("{raw}.ffn_up_exps.weight"),
            separate_down: format!("{raw}.ffn_down_exps.weight"),
            independent: Vec::new(),
        },
    )
    .map_err(|error| error.to_string())
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &DecoderConfig,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if !args.is_moe() {
        return Err("Muse-Glimmer expert residency requires a routed model".into());
    }
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let experts = dimension(args.num_experts, "expert count")?;
    let capacity = layers
        .checked_mul(experts)
        .ok_or_else(|| "Muse-Glimmer expert residency catalog size overflows".to_string())?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(super::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for layer in 0..layers {
        let unit_path = format!("model.layers.{layer}");
        let bank = match args.weight_convention {
            super::WeightConvention::HuggingFace => {
                safetensors_expert_recipes(catalog, args, layer)?
            }
            super::WeightConvention::Gguf => expert_recipes(catalog, layer)?,
        };
        for expert in 0..experts {
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert
                    .checked_add(1)
                    .ok_or_else(|| "Muse-Glimmer expert index overflowed".to_string())?,
            };
            let parameters = [
                (
                    "gate_up_proj",
                    bank.target_gate_up.clone(),
                    bank.gate_up.clone(),
                ),
                ("down_proj", bank.target_down.clone(), bank.down.clone()),
            ]
            .into_iter()
            .map(|(binding, target, recipe)| {
                let recipe = recipe
                    .select_bounded(catalog, selection.clone())
                    .map_err(|error| error.to_string())?;
                let role = match binding {
                    "gate_up_proj" => crate::ExpertParameterRole::quantizable_projection(
                        "gate_up_proj_scales",
                        "gate_up_proj_biases",
                    ),
                    "down_proj" => crate::ExpertParameterRole::quantizable_projection(
                        "down_proj_scales",
                        "down_proj_biases",
                    ),
                    _ => crate::ExpertParameterRole::Preserved,
                };
                crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    eredu_runtime::ParameterBankKey::new(layer, expert),
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

/// Translates one released dense text GGUF name to its neutral identity.
pub fn translate_text_gguf_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.into();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.into();
    };
    for (source, target) in [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_gate", "self_attn.gate_proj"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffw_norm", "post_feedforward_layernorm"),
        ("ffn_gate_inp", "mlp.gate"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_up", "mlp.up_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.into()
}

/// Translates one official projector GGUF name to its neutral identity.
pub fn translate_projector_gguf_name(name: &str) -> String {
    let exact = match name {
        "v.patch_embd.weight" => Some("vision_tower.patch_embedder.patch_embedding.weight"),
        "v.position_embd.weight" => {
            Some("vision_tower.patch_embedder.position_embedding_table.weight")
        }
        "v.pre_ln.weight" => Some("vision_tower.ln_pre.weight"),
        "v.pre_ln.bias" => Some("vision_tower.ln_pre.bias"),
        "v.post_ln.weight" => Some("vision_tower.ln_post.weight"),
        "v.post_ln.bias" => Some("vision_tower.ln_post.bias"),
        "mm.0.weight" => Some("vision_adapter.fc1.weight"),
        "mm.1.weight" => Some("vision_adapter.fc2.weight"),
        "mm.2.weight" => Some("vision_projection.weight"),
        _ => None,
    };
    if let Some(target) = exact {
        return format!("model.{target}");
    }
    let Some(rest) = name.strip_prefix("v.blk.") else {
        return name.into();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.into();
    };
    for (source, target) in [
        ("attn_q", "attn.q_proj"),
        ("attn_k", "attn.k_proj"),
        ("attn_v", "attn.v_proj"),
        ("attn_out", "attn.proj"),
        ("ffn_up", "mlp.fc1"),
        ("ffn_down", "mlp.fc2"),
        ("ln1", "norm1"),
        ("ln2", "norm2"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.vision_tower.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.into()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeCatalog},
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };

    use super::*;

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            logical_shape: shape.clone(),
            physical_shape: shape.clone(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    fn decoder(moe: bool) -> DecoderConfig {
        let mut value = serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 30,
            "video_token_id": 29,
            "out_hidden_size": 16,
            "projector_hidden_size": 8,
            "vision_config": {
                "model_type": "muse_glimmer_vision",
                "hidden_act": "gelu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "patch_size": 2,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "max_position_embeddings": 4,
                "layer_norm_eps": 1e-5,
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8,
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "layer_types": ["full_attention"],
                "layer_rope_theta": [0.0],
                "sliding_window": 16,
                "tie_word_embeddings": false,
                "hidden_activation": "silu",
                "attention_dropout": 0.0,
                "attention_bias": false,
                "mlp_bias": false,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611614,
                "final_logit_softcapping": 20.0
            }
        });
        if moe {
            value["text_config"]["intermediate_size"] = 0.into();
            value["text_config"]["moe_intermediate_size"] = 32.into();
            value["text_config"]["num_experts"] = 4.into();
            value["text_config"]["num_experts_per_tok"] = 2.into();
            value["text_config"]["norm_topk_prob"] = true.into();
        }
        DecoderConfig::from_hf_value(&value).unwrap()
    }

    #[test]
    fn selected_formats_are_partitioned_between_text_and_vision() {
        let source = decoder(false);
        let format = WeightQuantization::MxFp4;
        let text = "model.layers.0.self_attn.q_proj.weight";
        let vision = "model.vision_tower.layers.0.attn.q_proj.weight";
        let target = with_checkpoint_formats(
            &source,
            HashMap::from([(text.into(), format), (vision.into(), format)]),
        )
        .unwrap();
        assert_eq!(target.linear_format_for(text), format.into());
        assert_eq!(
            target
                .vision_config
                .as_ref()
                .unwrap()
                .linear_format_for(vision),
            format.into()
        );
    }

    #[test]
    fn split_projector_identity_and_image_only_policy_are_explicit() {
        let config = ArtifactConfig {
            hidden_size: 32,
            image_token_id: 10,
            video_token_id: 11,
            projector: true,
            image_only_projector: true,
            assistant: true,
        };
        assert_eq!(config.artifact_schema().unwrap().siblings().len(), 2);
        config
            .projector_compatibility("muse_glimmer", 32, BTreeSet::from([10]))
            .validate()
            .unwrap();
        assert!(config.validate_video_admission(true).is_err());
        assert_eq!(
            config
                .parameter_catalog()
                .unwrap()
                .owner("model.vision_projection.weight")
                .unwrap()
                .as_str(),
            "vision_projector"
        );
    }

    #[test]
    fn gguf_names_preserve_dense_text_and_projector_roots() {
        assert_eq!(
            translate_text_gguf_name("blk.2.attn_q.weight"),
            "model.layers.2.self_attn.q_proj.weight"
        );
        assert_eq!(
            translate_text_gguf_name("blk.1.ffn_gate_inp.weight"),
            "model.layers.1.mlp.gate.weight"
        );
        assert_eq!(
            translate_projector_gguf_name("v.blk.3.attn_out.weight"),
            "model.vision_tower.layers.3.attn.proj.weight"
        );
        assert_eq!(
            translate_text_gguf_name("blk.4.ffn_gate_exps.weight"),
            "model.layers.4.mlp.experts.gate_proj.weight"
        );
    }

    #[test]
    fn strict_plans_cover_native_vision_dense_and_all_moe_layouts() {
        let dense = decoder(false);
        let safe = safetensors_plan(&dense).unwrap();
        assert!(safe.catalog_policy.strict);
        assert!(safe.layout_groups.is_empty());
        assert!(safe.common_tensors.iter().any(|tensor| {
            tensor.key == "model.language_model.layers.0.self_attn.q_proj.weight"
                && tensor.aliases == ["model.layers.0.self_attn.q_proj.weight"]
        }));
        assert!(safe.common_tensors.iter().any(|tensor| {
            tensor.key == "model.vision_tower.patch_embedder.patch_embedding.weight"
        }));
        assert!(gguf_plan(&dense).unwrap().catalog_policy.strict);
        assert!(projector_gguf_plan(&dense).unwrap().catalog_policy.strict);

        let sparse = safetensors_plan(&decoder(true)).unwrap();
        assert_eq!(sparse.layout_groups.len(), 1);
        assert_eq!(sparse.layout_groups[0].variants.len(), 3);
        let sparse_gguf = gguf_plan(&decoder(true)).unwrap();
        assert!(sparse_gguf
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "blk.0.ffn_gate_exps.weight"));
    }

    #[test]
    fn safetensors_recipes_own_released_aliases_and_independent_expert_stacking() {
        let args = decoder(true);
        let mut tensors = BTreeMap::from([(
            "model.language_model.embed_tokens.weight".into(),
            metadata("model.language_model.embed_tokens.weight", vec![32, 32]),
        )]);
        for expert in 0..4 {
            let root = format!("model.language_model.layers.0.mlp.experts.{expert}");
            for (suffix, shape) in [
                ("gate_proj.weight", vec![32, 32]),
                ("up_proj.weight", vec![32, 32]),
                ("down_proj.weight", vec![32, 32]),
            ] {
                let name = format!("{root}.{suffix}");
                tensors.insert(name.clone(), metadata(&name, shape));
            }
        }
        let recipes = safetensors_recipes(&args, &Catalog(tensors.clone())).unwrap();
        assert!(matches!(
            recipes.get("model.embed_tokens.weight"),
            Some(DerivedWeightRecipe::Source { key, .. })
                if key == "model.language_model.embed_tokens.weight"
        ));
        assert!(matches!(
            recipes.get("model.layers.0.mlp.experts.gate_up_proj"),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 4
        ));
        assert!(matches!(
            recipes.get("model.layers.0.mlp.experts.down_proj"),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 4
        ));

        let pinned = static_safetensors_recipes(&args, &Catalog(tensors.clone())).unwrap();
        assert_eq!(
            pinned.keys().map(String::as_str).collect::<Vec<_>>(),
            ["model.embed_tokens.weight"]
        );
        let unit = unit_safetensors_recipes(&args, &Catalog(tensors), 1, 0).unwrap();
        assert_eq!(unit.len(), 2);
        assert!(unit.keys().all(|name| name.starts_with("model.layers.0.")));

        let split_root = "model.language_model.layers.0.mlp.experts";
        let split = Catalog(BTreeMap::from([
            (
                format!("{split_root}.gate_proj"),
                metadata(&format!("{split_root}.gate_proj"), vec![4, 32, 32]),
            ),
            (
                format!("{split_root}.up_proj"),
                metadata(&format!("{split_root}.up_proj"), vec![4, 32, 32]),
            ),
            (
                format!("{split_root}.down_proj"),
                metadata(&format!("{split_root}.down_proj"), vec![4, 32, 32]),
            ),
        ]));
        let split_unit = unit_safetensors_recipes(&args, &split, 1, 0).unwrap();
        assert_eq!(
            split_unit.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "model.layers.0.mlp.experts.down_proj",
                "model.layers.0.mlp.experts.gate_up_proj",
            ]
        );
        assert!(unit_safetensors_recipes(&args, &Catalog(BTreeMap::new()), 2, 0).is_err());
        assert!(unit_safetensors_recipes(&args, &Catalog(BTreeMap::new()), 1, 1).is_err());
    }

    #[test]
    fn residency_catalog_owns_expert_count_identity_and_compact_names() {
        let args = decoder(true);
        let root = "model.layers.0.mlp.experts";
        let catalog = Catalog(BTreeMap::from([
            (
                format!("{root}.gate_up_proj"),
                metadata(&format!("{root}.gate_up_proj"), vec![4, 64, 32]),
            ),
            (
                format!("{root}.down_proj"),
                metadata(&format!("{root}.down_proj"), vec![4, 32, 32]),
            ),
        ]));
        let residency = expert_residency_catalog(&catalog, &args).unwrap();
        assert_eq!(residency.units().len(), 4);
        let first = &residency.units()[0];
        assert_eq!(first.identity(), eredu_runtime::ParameterBankKey::new(0, 0));
        assert_eq!(first.owner_group().as_str(), "text_decoder");
        assert_eq!(first.owner_unit(), 0);
        assert_eq!(first.unit_path(), "model.layers.0");
        assert_eq!(
            first
                .parameters()
                .iter()
                .map(|parameter| (parameter.binding_name(), parameter.logical_target()))
                .collect::<Vec<_>>(),
            [
                ("gate_up_proj", "model.layers.0.mlp.experts.gate_up_proj"),
                ("down_proj", "model.layers.0.mlp.experts.down_proj"),
            ]
        );
        assert_eq!(
            residency.units()[3].identity(),
            eredu_runtime::ParameterBankKey::new(0, 3)
        );
    }
}
