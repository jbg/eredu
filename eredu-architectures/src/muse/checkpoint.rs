//! Composite artifact and checkpoint-name policy for Muse-Glimmer.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eredu_checkpoint::{
    composite::{
        ArtifactComponentSchema, ArtifactRole, ComponentId, ComponentParameterCatalog,
        CompositeArtifactError, CompositeArtifactSchema, ProjectorCompatibility,
    },
    expert::{
        gated_product_materialization_recipes, resolve_gated_product_expert_recipes,
        GatedProductExpertLayoutNames, GatedProductExpertRecipes,
        IndependentGatedProductExpertNames,
    },
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, matrix_for_nvfp4, AlternativeLayoutGroup, CatalogPolicy,
        GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, NvFp4Storage,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
        TensorOperation,
    },
    store::TensorSelection,
    WeightQuantization,
};

use super::{BlockNormalization, DecoderConfig, ResidualEquation, RoutedFeedForwardEquation};

/// Exact source views for independently consumed media towers. Keys retain their
/// admitted source spelling; namespace normalization only determines ownership.
pub(crate) fn media_source_roles(
    keys: &BTreeSet<String>,
) -> BTreeMap<crate::artifact_preparation::ArtifactSourceRole, BTreeSet<String>> {
    let mut roles = BTreeMap::<_, BTreeSet<_>>::new();
    for key in keys {
        if let Some(role) = media_source_role(key) {
            roles.entry(role).or_default().insert(key.clone());
        }
    }
    roles
}

pub(crate) fn media_source_role(
    key: &str,
) -> Option<crate::artifact_preparation::ArtifactSourceRole> {
    use crate::artifact_preparation::ArtifactSourceRole;
    let canonical = canonical_safetensors_target(key);
    if canonical.starts_with("model.audio_tower.") {
        Some(ArtifactSourceRole::Audio)
    } else if canonical.starts_with("model.vision_") {
        Some(ArtifactSourceRole::Vision)
    } else {
        None
    }
}

/// Resolves actual SafeTensors component coverage and per-weight MLX encodings.
/// Declared media geometry is retained only for towers present in this source view.
/// Companion preparation composes independently admitted views after primary admission.
pub fn resolve_safetensors_config<C>(
    args: &DecoderConfig,
    source_config: &serde_json::Value,
    catalog: &C,
) -> Result<DecoderConfig, String>
where
    C: RecipeCatalog + eredu_checkpoint::validation::SafetensorsCatalog + ?Sized,
{
    let keys = eredu_checkpoint::validation::SafetensorsCatalog::keys(catalog);
    let mut target = args.clone();
    let canonical = keys
        .iter()
        .map(|key| canonical_safetensors_target(key))
        .collect::<Vec<_>>();
    if !canonical
        .iter()
        .any(|key| key.starts_with("model.vision_tower."))
    {
        target.vision_config = None;
    }
    if !canonical
        .iter()
        .any(|key| key.starts_with("model.audio_tower."))
    {
        target.audio_config = None;
    }
    let quantization = match (
        source_config.get("quantization"),
        source_config.get("quantization_config"),
    ) {
        (Some(first), Some(second)) if first != second => {
            return Err("Muse root quantization declarations disagree".into())
        }
        (Some(value), _) | (_, Some(value)) => Some(value),
        _ => None,
    };
    let Some(quantization) = quantization else {
        target.validate().map_err(|error| error.to_string())?;
        return Ok(target);
    };
    if quantization.get("quant_method").is_some() || quantization.get("quant_algo").is_some() {
        if quantization
            .get("quant_method")
            .and_then(serde_json::Value::as_str)
            != Some("modelopt")
            || quantization
                .get("quant_algo")
                .and_then(serde_json::Value::as_str)
                != Some("NVFP4")
            || quantization
                .get("group_size")
                .and_then(serde_json::Value::as_u64)
                != Some(16)
        {
            return Err("unsupported Muse quantization declaration".into());
        }
        let mut formats = HashMap::new();
        for key in &keys {
            if let Some(weight) = key.strip_suffix("_weight_scale") {
                catalog
                    .tensor_metadata(weight)
                    .map_err(|error| error.to_string())?;
                for suffix in ["_weight_scale_2", "_input_scale"] {
                    catalog
                        .tensor_metadata(&format!("{weight}{suffix}"))
                        .map_err(|error| error.to_string())?;
                }
                formats.insert(
                    canonical_safetensors_target(weight),
                    WeightQuantization::NvFp4,
                );
            }
        }
        if formats.is_empty() {
            return Err("ModelOpt NVFP4 declaration has no encoded matrices".into());
        }
        return with_checkpoint_formats(&target, formats);
    }
    let default: WeightQuantization =
        serde_json::from_value(quantization.clone()).map_err(|error| error.to_string())?;
    let mut formats = HashMap::new();
    for key in &keys {
        let Some(module) = key.strip_suffix(".scales") else {
            continue;
        };
        let weight = format!("{module}.weight");
        catalog
            .tensor_metadata(&weight)
            .map_err(|error| error.to_string())?;
        let format = match quantization.get(module) {
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|error| format!("Muse weight {weight}: {error}"))?,
            None => default,
        };
        formats.insert(canonical_safetensors_target(&weight), format);
    }
    for (module, value) in quantization
        .as_object()
        .ok_or("Muse quantization must be an object")?
    {
        if matches!(module.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        let _: WeightQuantization = serde_json::from_value(value.clone())
            .map_err(|error| format!("Muse quantization override {module}: {error}"))?;
        let weight = format!("{module}.weight");
        catalog
            .tensor_metadata(&weight)
            .map_err(|error| error.to_string())?;
        if !keys.contains(&format!("{module}.scales")) {
            return Err(format!("Muse override {module} has no scale companion"));
        }
    }
    for layer in 0..dimension(target.num_hidden_layers, "text layer count")? {
        let root = format!("model.layers.{layer}.mlp.experts");
        match (
            formats.get(&format!("{root}.gate_proj")),
            formats.get(&format!("{root}.up_proj")),
        ) {
            (Some(gate), Some(up)) if gate == up => {
                formats.insert(format!("{root}.gate_up_proj"), *gate);
            }
            (None, None) => {}
            _ => {
                return Err(format!(
                    "Muse layer {layer} split gate/up physical encodings disagree"
                ))
            }
        }
    }
    target.quantization = None;
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    if let Some(vision) = target.vision_config.as_mut() {
        vision.weight_quantization = None;
        vision.quantized_weight_configs.clear();
    }
    with_checkpoint_formats(&target, formats)
}

/// Derives a Muse-Glimmer configuration whose text and aligned vision matrix formats
/// reflect load-time quantization instead of checkpoint-specific selections.
pub fn load_time_quantization(
    args: &DecoderConfig,
    quantization: WeightQuantization,
) -> Result<DecoderConfig, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    if quantization == WeightQuantization::NvFp4 {
        return Err("NVFP4 load-time calibration/encoding is unavailable".into());
    }
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    if let Some(vision) = &mut target.vision_config {
        vision.weight_quantization = None;
        vision.quantized_weight_configs.clear();
    }
    if target.vision_config.is_some() {
        // Derive candidates from the same checked geometry that owns source
        // admission. Small patch inputs and other unaligned media matrices
        // remain dense; explicit checkpoint formats are handled separately.
        let mut tensors = Vec::new();
        add_safetensors_vision(
            &target,
            dimension(target.hidden_size, "text hidden size")?,
            &mut tensors,
        )?;
        let group = usize::try_from(quantization.group_size())
            .map_err(|_| "negative quantization group size")?;
        if let Some(vision) = &mut target.vision_config {
            vision.quantized_weight_configs = tensors
                .into_iter()
                .filter(|tensor| {
                    tensor.shape.len() == 2
                        && tensor.shape[1] > 0
                        && tensor.shape[1].is_multiple_of(group)
                        && tensor.shape[1].is_multiple_of(32)
                })
                .map(|tensor| (tensor.key, quantization))
                .collect();
        }
    }
    if let Some(audio) = &mut target.audio_config {
        let group = usize::try_from(quantization.group_size())
            .map_err(|_| "negative quantization group size")?;
        audio.linear_formats = audio_tensor_layouts(audio)?
            .into_iter()
            .filter(|tensor| {
                tensor.operation == TensorOperation::Matrix
                    && tensor.shape[1].is_multiple_of(group)
                    && tensor.shape[1].is_multiple_of(32)
            })
            .map(|tensor| (tensor.canonical, quantization.into()))
            .collect();
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
    let mut audio = HashMap::new();
    for (name, format) in formats {
        if name.starts_with("model.vision_") {
            vision.insert(name, format);
        } else if name.starts_with("model.audio_tower.") {
            audio.insert(name, format.into());
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
    if let Some(config) = target.audio_config.as_mut() {
        config.linear_formats = audio;
    } else if !audio.is_empty() {
        return Err("audio checkpoint formats have no selected waveform tower".into());
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

fn decoder_norm_names(args: &DecoderConfig) -> Vec<&'static str> {
    let mut names = vec![
        "input_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ];
    if args.block_normalization == BlockNormalization::OutputScaled {
        names.push("post_attention_layernorm.weight");
    }
    names
}

pub(super) fn residual_parameter_names(args: &DecoderConfig) -> Vec<&'static str> {
    match args.residual_equation {
        ResidualEquation::Add => Vec::new(),
        ResidualEquation::Learned { .. } => vec![
            "post_attention_residual_gate.gate",
            "post_feedforward_residual_gate.gate",
        ],
        ResidualEquation::StoredCoefficients => vec![
            "post_attention_residual_gate.alpha",
            "post_attention_residual_gate.beta",
            "post_feedforward_residual_gate.alpha",
            "post_feedforward_residual_gate.beta",
        ],
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
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.gate_proj.weight",
        ] {
            names.push(format!("{root}.{local}"));
        }
        names.extend(
            decoder_norm_names(args)
                .into_iter()
                .chain(residual_parameter_names(args))
                .map(|name| format!("{root}.{name}")),
        );
        if args.is_moe() {
            names.extend([
                format!("{root}.mlp.gate.weight"),
                format!("{root}.mlp.experts.gate_up_proj"),
                format!("{root}.mlp.experts.down_proj"),
            ]);
            if let RoutedFeedForwardEquation::CompressedSigmoid { .. } = args.routed_feed_forward {
                names.extend(
                    [
                        "gate.e_score_correction_bias",
                        "pre_expert_proj.weight",
                        "pre_expert_norm.weight",
                        "experts.post_expert_norm.weight",
                        "post_expert_proj.weight",
                    ]
                    .map(|name| format!("{root}.mlp.{name}")),
                );
            }
        } else {
            names.extend([
                format!("{root}.mlp.gate_proj.weight"),
                format!("{root}.mlp.up_proj.weight"),
                format!("{root}.mlp.down_proj.weight"),
            ]);
        }
    }
    names.extend(vision_parameter_names(args));
    names.extend(audio_parameter_names(args));
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
    args.validate().map_err(|error| error.to_string())?;
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
        for local in decoder_norm_names(args)
            .into_iter()
            .chain(residual_parameter_names(args))
        {
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
    if args.vision_config.is_some() {
        add_safetensors_vision(args, hidden, &mut common)?;
    }
    add_safetensors_audio(args, &mut common)?;
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", args.identity.model_type()),
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
    let mut recipes = direct_safetensors_recipes(&plan, catalog);
    if args.is_moe() {
        for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
            let expert = safetensors_expert_recipes(catalog, args, layer)?;
            recipes.extend(gated_product_materialization_recipes(catalog, &expert)?);
        }
    }
    Ok(recipes)
}

fn direct_safetensors_recipes<C: RecipeCatalog + ?Sized>(
    plan: &SafetensorsCheckpointPlan,
    catalog: &C,
) -> BTreeMap<String, DerivedWeightRecipe> {
    let constraints = plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .flat_map(|group| group.variants.iter())
            .flat_map(|variant| variant.tensors.iter()),
    );
    let mut recipes = BTreeMap::new();
    for tensor in constraints {
        if tensor.role == eredu_checkpoint::schema::TensorRole::RecipeInput {
            continue;
        }
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
    recipes
}

pub(crate) fn bf16_media_plan(
    args: &DecoderConfig,
    roles: &BTreeSet<crate::artifact_preparation::ArtifactSourceRole>,
) -> Result<SafetensorsCheckpointPlan, String> {
    use crate::artifact_preparation::ArtifactSourceRole;
    args.validate().map_err(|e| e.to_string())?;
    if roles.is_empty()
        || roles
            .iter()
            .any(|role| !matches!(role, ArtifactSourceRole::Vision | ArtifactSourceRole::Audio))
    {
        return Err("BF16 media admission requires vision or audio source roles".into());
    }
    let mut tensors = Vec::new();
    if roles.contains(&ArtifactSourceRole::Vision) {
        add_safetensors_vision(
            args,
            dimension(args.hidden_size, "decoder width")?,
            &mut tensors,
        )?;
    }
    if roles.contains(&ArtifactSourceRole::Audio) {
        if args.audio_config.is_none() {
            return Err("media source omits its audio configuration".into());
        }
        add_safetensors_audio(args, &mut tensors)?;
    }
    for tensor in &mut tensors {
        tensor.dtype = StoredDtypeConstraint::Exact(eredu_checkpoint::StoredDtype::BF16);
    }
    SafetensorsCheckpointPlan::new(
        "Muse BF16 media companions",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

pub(crate) fn bf16_media_recipes<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    roles: &BTreeSet<crate::artifact_preparation::ArtifactSourceRole>,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    Ok(direct_safetensors_recipes(
        &bf16_media_plan(args, roles)?,
        catalog,
    ))
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
        2 => args
            .audio_config
            .as_ref()
            .map(|audio| audio.execution_units().map(|units| units.len()))
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or(0),
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
    let mut recipes = direct_safetensors_recipes(&safetensors_plan(args)?, catalog);
    if let Some((1, layer)) = selected.filter(|_| args.is_moe()) {
        let expert = safetensors_expert_recipes(catalog, args, layer)?;
        recipes.extend(gated_product_materialization_recipes(catalog, &expert)?);
    }
    for (target, recipe) in recipes {
        let owner = if target.starts_with("model.audio_tower.") {
            let audio = args
                .audio_config
                .as_ref()
                .ok_or("audio recipe has no declared tower")?;
            let index = audio
                .execution_units()
                .map_err(|error| error.to_string())?
                .iter()
                .position(|kind| kind.owns_parameter(&target))
                .ok_or_else(|| format!("unowned Muse audio recipe {target}"))?;
            Some((2, index))
        } else if let Some(index) = indexed_recipe_target(&target, "model.vision_tower.layers.")? {
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

pub(crate) fn canonical_safetensors_target(source: &str) -> String {
    let source = source.strip_prefix("language_model.").unwrap_or(source);
    let mut canonical = source
        .replace("model.language_model.layers.", "model.layers.")
        .replace("model.language_model.embed_tokens.", "model.embed_tokens.")
        .replace("model.language_model.norm.", "model.norm.")
        .replace(".mlp.switch_mlp.", ".mlp.experts.")
        .replace(".mlp.post_expert_norm.", ".mlp.experts.post_expert_norm.");
    if canonical.starts_with("vision_") || canonical.starts_with("audio_tower.") {
        canonical = format!("model.{canonical}");
    }
    for projection in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"] {
        if canonical.ends_with(&format!(".mlp.experts.{projection}.weight")) {
            canonical.truncate(canonical.len() - ".weight".len());
            break;
        }
    }
    if canonical.contains(".mlp.experts.") {
        for (physical, neutral) in [
            ("_weight_scale_2", "_global_scale"),
            ("_weight_scale", "_scales"),
            (".global_scale", "_global_scale"),
        ] {
            if let Some(prefix) = canonical.strip_suffix(physical) {
                return format!("{prefix}{neutral}");
            }
        }
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
        .flat_map(|name| {
            let mut choices = vec![name.clone()];
            choices.extend(safetensors_aliases(
                &name,
                &canonical_safetensors_target(&name),
            ));
            choices
        })
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
    let hidden = match args.routed_feed_forward {
        RoutedFeedForwardEquation::DirectSoftmax => hidden,
        RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } => {
            let latent = dimension(expert_width, "compressed expert width")?;
            for (local, shape) in [
                ("pre_expert_proj.weight", vec![latent, hidden]),
                ("post_expert_proj.weight", vec![hidden, latent]),
            ] {
                add_text_matrix(
                    args,
                    common,
                    &format!("{released}.mlp.{local}"),
                    &format!("{canonical}.mlp.{local}"),
                    shape,
                )?;
            }
            for (local, width) in [
                ("gate.e_score_correction_bias", experts),
                ("pre_expert_norm.weight", latent),
                ("experts.post_expert_norm.weight", latent),
            ] {
                common.push(safe_alias(
                    format!("{released}.mlp.{local}"),
                    format!("{canonical}.mlp.{local}"),
                    vec![width],
                ));
            }
            latent
        }
    };
    let released_root = format!("{released}.mlp.experts");
    let canonical_root = format!("{canonical}.mlp.experts");
    let packed_gate_up = format!("{released_root}.gate_up_proj");
    let packed_down = format!("{released_root}.down_proj");
    let mut packed = Vec::new();
    add_packed_expert_matrix(
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
    add_packed_expert_matrix(
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
                // Both bank layouts require the same down projection. Its
                // presence does not select either gate/up representation.
                discriminator_keys: vec![packed_gate_up],
            },
            LayoutVariant {
                id: "split expert banks".into(),
                tensors: split,
                discriminator_keys: vec![split_gate, split_up],
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

fn audio_parameter_names(args: &DecoderConfig) -> Vec<String> {
    let Some(audio) = &args.audio_config else {
        return Vec::new();
    };
    let mut names = [
        "norm.weight",
        "proj.weight",
        "proj.bias",
        "post_proj_norm.weight",
    ]
    .map(|name| format!("model.audio_tower.{name}"))
    .to_vec();
    for (stage, &depth) in audio.depths.iter().enumerate() {
        let root = format!("model.audio_tower.layers.{stage}");
        names.extend(
            [
                "patch_embedder.proj.weight",
                "patch_embedder.proj.bias",
                "patch_embedder.norm.weight",
            ]
            .map(|name| format!("{root}.{name}")),
        );
        for block in 0..depth {
            names.extend(
                [
                    "conv.weight",
                    "conv.bias",
                    "norm.weight",
                    "mlp.fc1.weight",
                    "mlp.fc1.bias",
                    "mlp.fc2.weight",
                    "mlp.fc2.bias",
                ]
                .map(|name| format!("{root}.conv_blocks.{block}.{name}")),
            );
        }
    }
    names
}

fn add_audio_matrix(
    audio: &super::AudioConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(
            name,
            std::iter::empty::<String>(),
            shape,
            audio.linear_format_for(name),
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

pub(crate) struct AudioTensorLayout {
    pub(crate) canonical: String,
    gguf: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) operation: TensorOperation,
}

pub(crate) fn audio_tensor_layouts(
    audio: &super::AudioConfig,
) -> Result<Vec<AudioTensorLayout>, String> {
    audio.validate().map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    let mut add = |canonical: String, gguf: String, shape, operation| {
        tensors.push(AudioTensorLayout {
            canonical,
            gguf,
            shape,
            operation,
        });
    };
    let mut flat_block = 0usize;
    for stage in 0..audio.hidden_sizes.len() {
        let width = dimension(audio.hidden_sizes[stage], "audio stage width")?;
        let input = if stage == 0 {
            1
        } else {
            dimension(audio.hidden_sizes[stage - 1], "audio input channels")?
        };
        let patch = dimension(audio.patch_sizes[stage], "audio patch size")?;
        let intermediate = dimension(
            audio
                .intermediate_size(stage)
                .map_err(|error| error.to_string())?,
            "audio MLP width",
        )?;
        let root = format!("model.audio_tower.layers.{stage}");
        add(
            format!("{root}.patch_embedder.proj.weight"),
            format!("a.patch.{stage}.weight"),
            vec![width, checked_mul(input, patch, "audio patch input width")?],
            TensorOperation::Matrix,
        );
        add(
            format!("{root}.patch_embedder.proj.bias"),
            format!("a.patch.{stage}.bias"),
            vec![width],
            TensorOperation::Vector,
        );
        add(
            format!("{root}.patch_embedder.norm.weight"),
            format!("a.patch.{stage}.norm.weight"),
            vec![width],
            TensorOperation::Vector,
        );
        for block in 0..audio.depths[stage] {
            let root = format!("{root}.conv_blocks.{block}");
            for (canonical, source, shape, operation) in [
                (
                    "conv.weight",
                    format!("a.conv1d.{flat_block}.weight"),
                    vec![
                        width,
                        1,
                        dimension(audio.conv_kernel_size, "audio kernel width")?,
                    ],
                    TensorOperation::Dense,
                ),
                (
                    "conv.bias",
                    format!("a.conv1d.{flat_block}.bias"),
                    vec![width],
                    TensorOperation::Vector,
                ),
                (
                    "norm.weight",
                    format!("a.conv1d.{flat_block}.norm.weight"),
                    vec![width],
                    TensorOperation::Vector,
                ),
                (
                    "mlp.fc1.weight",
                    format!("a.blk.{flat_block}.ffn_up.weight"),
                    vec![intermediate, width],
                    TensorOperation::Matrix,
                ),
                (
                    "mlp.fc1.bias",
                    format!("a.blk.{flat_block}.ffn_up.bias"),
                    vec![intermediate],
                    TensorOperation::Vector,
                ),
                (
                    "mlp.fc2.weight",
                    format!("a.blk.{flat_block}.ffn_down.weight"),
                    vec![width, intermediate],
                    TensorOperation::Matrix,
                ),
                (
                    "mlp.fc2.bias",
                    format!("a.blk.{flat_block}.ffn_down.bias"),
                    vec![width],
                    TensorOperation::Vector,
                ),
            ] {
                add(format!("{root}.{canonical}"), source, shape, operation);
            }
            flat_block = flat_block
                .checked_add(1)
                .ok_or("audio flat block index overflowed")?;
        }
    }
    let width = dimension(
        *audio.hidden_sizes.last().ok_or("audio stages are empty")?,
        "audio final width",
    )?;
    let language = dimension(audio.out_hidden_size, "audio language width")?;
    for (canonical, source, shape, operation) in [
        (
            "norm.weight",
            "a.pre_ln.weight",
            vec![width],
            TensorOperation::Vector,
        ),
        (
            "proj.weight",
            "mm.a.fc.weight",
            vec![language, width],
            TensorOperation::Matrix,
        ),
        (
            "proj.bias",
            "mm.a.fc.bias",
            vec![language],
            TensorOperation::Vector,
        ),
        (
            "post_proj_norm.weight",
            "a.post_ln.weight",
            vec![language],
            TensorOperation::Vector,
        ),
    ] {
        add(
            format!("model.audio_tower.{canonical}"),
            source.into(),
            shape,
            operation,
        );
    }
    Ok(tensors)
}

fn add_safetensors_audio(
    args: &DecoderConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let Some(audio) = &args.audio_config else {
        return Ok(());
    };
    for tensor in audio_tensor_layouts(audio)? {
        if tensor.operation == TensorOperation::Matrix {
            add_audio_matrix(audio, output, &tensor.canonical, tensor.shape)?;
        } else {
            output.push(safe(tensor.canonical, tensor.shape));
        }
    }
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
    let aliases = safetensors_aliases(source, canonical);
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

fn add_packed_expert_matrix(
    args: &DecoderConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    source: &str,
    canonical: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    if args.linear_format_for(canonical) != eredu_checkpoint::LinearFormat::NvFp4 {
        return add_text_matrix(args, output, source, canonical, shape);
    }
    let parts = if canonical.ends_with(".gate_up_proj") {
        2
    } else {
        1
    };
    output.extend(
        matrix_for_nvfp4(
            source,
            safetensors_aliases(source, canonical),
            shape.clone(),
            parts,
            NvFp4Storage::ModelOptBytes,
        )
        .map_err(|error| error.to_string())?,
    );
    let mut calibration_shape = shape[..shape.len() - 2].to_vec();
    if parts > 1 {
        calibration_shape.push(parts);
    }
    let mut calibration = SafetensorsTensorConstraint::required(
        format!("{source}_input_scale"),
        calibration_shape,
        StoredDtypeConstraint::Exact(eredu_checkpoint::StoredDtype::F32),
    );
    calibration.role = eredu_checkpoint::schema::TensorRole::RecipeInput;
    output.push(calibration);
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
            safetensors_aliases(name, name),
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
            if local == "post_attention_norm.weight"
                && args.block_normalization == BlockNormalization::InputScaledFeedForward
            {
                continue;
            }
            tensors.push(gguf(format!("{root}.{local}"), shape, operation));
        }
        match args.residual_equation {
            ResidualEquation::Add => {}
            ResidualEquation::Learned { .. } => {
                return Err(
                    "Muse GGUF residual gates require stored coefficient normalization".into(),
                )
            }
            ResidualEquation::StoredCoefficients => {
                for local in [
                    "attn_resid_alpha",
                    "attn_resid_beta",
                    "ffn_resid_alpha",
                    "ffn_resid_beta",
                ] {
                    tensors.push(gguf(
                        format!("{root}.{local}.weight"),
                        vec![hidden],
                        TensorOperation::Vector,
                    ));
                }
            }
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "expert count")?;
            let intermediate = dimension(args.moe_intermediate_size, "expert intermediate size")?;
            let expert_hidden = match args.routed_feed_forward {
                RoutedFeedForwardEquation::DirectSoftmax => hidden,
                RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } => {
                    let latent = dimension(expert_width, "compressed expert width")?;
                    for (local, shape, operation) in [
                        (
                            "ffn_latent_down.weight",
                            vec![latent, hidden],
                            TensorOperation::Matrix,
                        ),
                        (
                            "ffn_latent_up.weight",
                            vec![hidden, latent],
                            TensorOperation::Matrix,
                        ),
                        (
                            "ffn_pre_exp_norm.weight",
                            vec![latent],
                            TensorOperation::Vector,
                        ),
                        (
                            "ffn_post_exp_norm.weight",
                            vec![latent],
                            TensorOperation::Vector,
                        ),
                        ("exp_probs_b.bias", vec![experts], TensorOperation::Vector),
                    ] {
                        tensors.push(gguf(format!("{root}.{local}"), shape, operation));
                    }
                    latent
                }
            };
            tensors.extend([
                gguf(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_gate_exps.weight"),
                    vec![experts, intermediate, expert_hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_up_exps.weight"),
                    vec![experts, intermediate, expert_hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, expert_hidden, intermediate],
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
        format!("{} GGUF", args.identity.model_type()),
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
    if let Some(audio) = &args.audio_config {
        for tensor in audio_tensor_layouts(audio)? {
            tensors.push(gguf(tensor.gguf, tensor.shape, tensor.operation));
        }
    }
    GgufCheckpointPlan::new(
        format!("{} media GGUF", args.identity.model_type()),
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Lowers the published spatial patch kernel to the linear patch operator.
pub fn projector_gguf_recipes<C: RecipeCatalog + ?Sized>(
    args: &DecoderConfig,
    catalog: &C,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let Some(vision) = args.vision_config.as_ref() else {
        return Ok(BTreeMap::new());
    };
    let target = "model.vision_tower.patch_embedder.patch_embedding.weight";
    let source = first_existing(catalog, [target.to_owned(), "v.patch_embd.weight".into()])
        .ok_or_else(|| "Muse-Glimmer projector patch kernel is missing".to_owned())?;
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let width = checked_mul(3, checked_mul(patch, patch, "patch area")?, "patch width")?;
    let metadata = catalog
        .tensor_metadata(&source)
        .map_err(|error| error.to_string())?;
    if metadata.logical_shape != [hidden, 3, patch, patch]
        && metadata.logical_shape != [hidden, width]
    {
        return Err(format!(
            "Muse-Glimmer projector patch kernel has invalid shape {:?}",
            metadata.logical_shape
        ));
    }
    Ok(BTreeMap::from([(
        target.into(),
        DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source(source, TensorSelection::Full)),
            shape: vec![hidden, width],
        },
    )]))
}

fn safetensors_aliases(source: &str, canonical: &str) -> Vec<String> {
    let mut aliases = BTreeSet::new();
    if source != canonical {
        aliases.insert(canonical.to_owned());
    }
    let mut mlx = canonical.to_owned();
    for projection in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"] {
        let suffix = format!(".mlp.experts.{projection}");
        if mlx.ends_with(&suffix) {
            mlx = mlx.replace(&suffix, &format!(".mlp.switch_mlp.{projection}.weight"));
            break;
        }
    }
    mlx = mlx.replace(".mlp.experts.post_expert_norm.", ".mlp.post_expert_norm.");
    if mlx.starts_with("model.vision_") || mlx.starts_with("model.audio_tower.") {
        mlx = mlx.trim_start_matches("model.").to_owned();
    }
    if mlx != source {
        aliases.insert(mlx.clone());
    }
    if mlx.starts_with("model.") || mlx.starts_with("lm_head.") {
        aliases.insert(format!("language_model.{mlx}"));
    }
    aliases.remove(source);
    aliases.into_iter().collect()
}

fn safe_alias(
    source: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    let source = source.into();
    let canonical = canonical.into();
    let aliases = safetensors_aliases(&source, &canonical);
    safe(source, shape).with_aliases(aliases)
}

fn safe(name: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    let name = name.into();
    let aliases = safetensors_aliases(&name, &canonical_safetensors_target(&name));
    SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
        .with_aliases(aliases)
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
    let source = |canonical: String, raw: String| {
        first_existing(catalog, [canonical, raw]).unwrap_or_default()
    };
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: format!("{root}.gate_up_proj"),
            target_down: format!("{root}.down_proj"),
            packed_gate_up: format!("{root}.gate_up_proj"),
            packed_down: format!("{root}.down_proj"),
            separate_gate: source(
                format!("{root}.gate_proj.weight"),
                format!("{raw}.ffn_gate_exps.weight"),
            ),
            separate_up: source(
                format!("{root}.up_proj.weight"),
                format!("{raw}.ffn_up_exps.weight"),
            ),
            separate_down: source(
                format!("{root}.down_proj.weight"),
                format!("{raw}.ffn_down_exps.weight"),
            ),
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
        let outputs = gated_product_materialization_recipes(catalog, &bank)?;
        let mut materialization = Vec::new();
        for target in [&bank.target_gate_up, &bank.target_down] {
            for suffix in ["", "_scales", "_biases", "_global_scale"] {
                if let Some(output) = outputs.get_key_value(&format!("{target}{suffix}")) {
                    materialization.push(output);
                }
            }
        }
        let members = eredu_runtime::ParameterBankRecipes::new(
            catalog,
            materialization
                .iter()
                .map(|(target, recipe)| ((*target).clone(), (*recipe).clone()))
                .collect(),
            experts,
        )
        .map_err(|error| error.to_string())?;
        for expert in 0..experts {
            let selected = members
                .member(expert)
                .expect("validated expert bank member");
            let parameters = materialization
                .iter()
                .map(|(target, _)| {
                    let binding = target
                        .rsplit('.')
                        .next()
                        .ok_or("expert parameter has no local identity")?;
                    let recipe = selected
                        .get(target.as_str())
                        .expect("declared bank parameter")
                        .clone();
                    let role = if args.linear_format_for(target)
                        == eredu_checkpoint::LinearFormat::Dense
                    {
                        match binding {
                            "gate_up_proj" | "down_proj" => {
                                crate::ExpertParameterRole::quantizable_projection(
                                    format!("{binding}_scales"),
                                    format!("{binding}_biases"),
                                )
                            }
                            _ => crate::ExpertParameterRole::Preserved,
                        }
                    } else {
                        crate::ExpertParameterRole::Preserved
                    };
                    crate::ExpertParameterRecipe::new(binding, target.as_str(), recipe, role)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
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
        (
            "attn_resid_alpha.weight",
            "post_attention_residual_gate.alpha",
        ),
        (
            "attn_resid_beta.weight",
            "post_attention_residual_gate.beta",
        ),
        (
            "ffn_resid_alpha.weight",
            "post_feedforward_residual_gate.alpha",
        ),
        (
            "ffn_resid_beta.weight",
            "post_feedforward_residual_gate.beta",
        ),
        ("ffn_latent_down", "mlp.pre_expert_proj"),
        ("ffn_latent_up", "mlp.post_expert_proj"),
        ("ffn_pre_exp_norm", "mlp.pre_expert_norm"),
        ("ffn_post_exp_norm", "mlp.experts.post_expert_norm"),
        ("exp_probs_b.bias", "mlp.gate.e_score_correction_bias"),
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
pub fn translate_projector_gguf_name(name: &str, audio: Option<&super::AudioConfig>) -> String {
    if name.starts_with("a.") || name.starts_with("mm.a.") {
        if let Some(target) = audio
            .and_then(|audio| audio_tensor_layouts(audio).ok())
            .and_then(|tensors| tensors.into_iter().find(|tensor| tensor.gguf == name))
        {
            return target.canonical;
        }
        return name.into();
    }
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
