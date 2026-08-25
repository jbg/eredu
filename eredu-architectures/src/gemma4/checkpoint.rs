//! Composite artifact policy for Gemma 4 checkpoints.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eredu_checkpoint::composite::{
    ArtifactComponentSchema, ArtifactRole, ComponentId, ComponentParameterCatalog,
    CompositeArtifactError, CompositeArtifactSchema, ProjectorCompatibility,
};
use eredu_checkpoint::store::TensorSelection;
use eredu_checkpoint::{
    expert::{
        resolve_gated_product_expert_recipes, GatedProductExpertLayoutNames,
        GatedProductExpertRecipes,
    },
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
        GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
        SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation, TensorRequirement,
    },
    WeightQuantization,
};
use eredu_nn::AttentionValueSource;

use super::{FamilyConfig, FeedForwardPolicy, ModelArgs};

const RELEASED_LAYER_ROOT: &str = "model.language_model.layers";

/// Derives a Gemma 4 family configuration whose text and aligned media
/// matrix formats reflect load-time quantization.
pub fn load_time_quantization(
    config: &FamilyConfig,
    quantization: WeightQuantization,
) -> Result<FamilyConfig, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = config.clone();
    target.text.weight_quantization = Some(quantization);
    target.text.quantized_weights = None;
    target.text.quantized_weight_configs = None;
    if let Some(vision) = target.vision.as_mut() {
        vision.weight_quantization = Some(quantization);
        vision.quantized_weights = None;
        vision.quantized_weight_configs = None;
    }
    if let Some(audio) = target.audio.as_mut() {
        audio.weight_quantization = Some(quantization);
        audio.quantized_weights = None;
        audio.quantized_weight_configs = None;
    }
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical text checkpoint formats to a complete Gemma 4 family
/// configuration.
pub fn with_checkpoint_formats(
    config: &FamilyConfig,
    formats: HashMap<String, WeightQuantization>,
) -> Result<FamilyConfig, String> {
    let mut target = config.clone();
    target.text.quantized_weight_configs = Some(formats);
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

fn released_layer_path(layer: usize) -> String {
    format!("{RELEASED_LAYER_ROOT}.{layer}")
}

/// Backend-independent artifact geometry needed before weight loading.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Gemma4ArtifactConfig {
    /// Whether the primary model uses the unified architecture identity.
    pub unified: bool,
    /// Decoder hidden width.
    pub hidden_size: usize,
    /// Optional image placeholder identity.
    pub image_token_id: Option<u32>,
    /// Optional video placeholder identity.
    pub video_token_id: Option<u32>,
    /// Optional audio placeholder identity.
    pub audio_token_id: Option<u32>,
    /// Whether a sibling media projector is admitted.
    pub projector: bool,
    /// Whether an external assistant artifact is required.
    pub assistant: bool,
}

impl Gemma4ArtifactConfig {
    /// Returns the exact model identity admitted by the primary artifact.
    pub fn architecture(&self) -> &'static str {
        if self.unified {
            "gemma4_unified"
        } else {
            "gemma4"
        }
    }

    /// Builds the primary-plus-sibling artifact schema.
    pub fn artifact_schema(&self) -> Result<CompositeArtifactSchema, CompositeArtifactError> {
        if self.hidden_size == 0 {
            return Err(CompositeArtifactError::ProjectorWidthMismatch {
                decoder: 0,
                projector: 0,
            });
        }
        let architecture = self.architecture().to_owned();
        let mut siblings = Vec::new();
        if self.projector {
            siblings.push(ArtifactComponentSchema {
                component: ComponentId::new("media_projector")?,
                role: ArtifactRole::Projector,
                required: false,
                architecture: architecture.clone(),
            });
        }
        if self.assistant {
            siblings.push(ArtifactComponentSchema {
                component: ComponentId::new("assistant")?,
                role: ArtifactRole::Assistant,
                required: true,
                architecture: architecture.clone(),
            });
        }
        CompositeArtifactSchema::new(
            ArtifactComponentSchema {
                component: ComponentId::new("decoder")?,
                role: ArtifactRole::Decoder,
                required: true,
                architecture,
            },
            siblings,
        )
    }

    /// Builds component ownership for stable logical parameter identities.
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
                ComponentId::new("media_projector")?,
                vec![
                    "model.embed_vision.projection.weight".into(),
                    "model.embed_audio.embedding_projection.weight".into(),
                ],
            ));
        }
        if self.assistant {
            components.push((
                ComponentId::new("assistant")?,
                vec![
                    "masked_embedding.centroids.weight".into(),
                    "masked_embedding.token_ordering".into(),
                ],
            ));
        }
        ComponentParameterCatalog::new(components)
    }

    /// Builds the sibling-projector compatibility proof checked before loading.
    pub fn projector_compatibility(
        &self,
        projector_architecture: impl Into<String>,
        projector_output_width: usize,
        projector_modality_tokens: BTreeSet<u32>,
    ) -> ProjectorCompatibility {
        ProjectorCompatibility {
            decoder_architecture: self.architecture().into(),
            projector_architecture: projector_architecture.into(),
            decoder_hidden_width: self.hidden_size,
            projector_output_width,
            decoder_modality_tokens: [
                self.image_token_id,
                self.video_token_id,
                self.audio_token_id,
            ]
            .into_iter()
            .flatten()
            .collect(),
            projector_modality_tokens,
        }
    }
}

/// Builds the complete decoder portion of a Gemma 4 SafeTensors artifact.
/// Native media components are appended from the same normalized family
/// configuration, while residency remains a runtime concern.
pub fn safetensors_plan(family: &FamilyConfig) -> Result<SafetensorsCheckpointPlan, String> {
    let args = &family.text;
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let layers = args.num_hidden_layers();
    let vocabulary = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
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
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dimension(args.hidden_size_per_layer_input, "per-layer input width")?;
        let combined = checked_mul(layers, per_layer, "combined per-layer input width")?;
        let per_layer_vocab = dimension(
            args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
            "per-layer input vocabulary size",
        )?;
        for (source, target, shape) in [
            (
                "model.language_model.embed_tokens_per_layer.weight",
                "model.embed_tokens_per_layer.weight",
                vec![per_layer_vocab, combined],
            ),
            (
                "model.language_model.per_layer_model_projection.weight",
                "model.per_layer_model_projection.weight",
                vec![combined, hidden],
            ),
        ] {
            add_text_matrix(args, &mut common, source, target, shape)?;
        }
        common.push(safe_alias(
            "model.language_model.per_layer_projection_norm.weight",
            "model.per_layer_projection_norm.weight",
            vec![per_layer],
        ));
    }
    for layer in 0..layers {
        add_safetensors_layer(args, layer, hidden, &mut common, &mut groups)?;
    }
    if let Some(vision) = &family.vision {
        add_safetensors_vision(vision, hidden, &mut common)?;
    }
    if let Some(audio) = &family.audio {
        add_safetensors_audio(audio, hidden, &mut common)?;
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.extend([
        "multi_modal_projector.".into(),
        "model.multi_modal_projector.".into(),
        "model.vision_embedder.".into(),
    ]);
    SafetensorsCheckpointPlan::new("Gemma 4 SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

/// Builds the strict released sibling-projector GGUF catalog.
///
/// The media geometry is derived from the same neutral parameter plan used by
/// SafeTensors; only the released physical prefixes and GGUF encoding policy
/// differ.
pub fn mmproj_gguf_plan(family: &FamilyConfig) -> Result<GgufCheckpointPlan, String> {
    let safe = safetensors_plan(family)?;
    let mut tensors = Vec::new();
    for tensor in safe.common_tensors {
        let Some(key) = projector_physical_name(&tensor.key) else {
            continue;
        };
        if tensor.role != eredu_checkpoint::schema::TensorRole::Tensor {
            return Err(format!(
                "Gemma 4 projector GGUF does not admit companion tensor {:?}",
                tensor.key
            ));
        }
        let mut constraint = GgufTensorConstraint::required(
            key,
            tensor.shape,
            GgufTypeConstraint::OperationClass(TensorOperation::Dense),
        )
        .with_alternate_shapes(tensor.alternate_shapes);
        constraint.element_count = tensor.element_count;
        if tensor.requirement == TensorRequirement::Optional {
            constraint.requirement = TensorRequirement::Optional;
        }
        tensors.push(constraint);
    }
    if tensors.is_empty() {
        return Err("Gemma 4 projector plan has no enabled media component".into());
    }
    GgufCheckpointPlan::new(
        "Gemma 4 sibling projector GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn projector_physical_name(canonical: &str) -> Option<String> {
    for (prefix, physical) in [
        ("model.vision_tower", "vision_tower"),
        ("model.embed_vision", "embed_vision"),
        ("model.audio_tower", "audio_tower"),
        ("model.embed_audio", "embed_audio"),
    ] {
        if canonical == prefix || canonical.starts_with(&format!("{prefix}.")) {
            return Some(canonical.replacen(prefix, physical, 1));
        }
    }
    None
}

fn add_safetensors_layer(
    args: &ModelArgs,
    layer: usize,
    hidden: usize,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("Gemma 4 layer {layer} has no normalized policy"))?;
    let released = released_layer_path(layer);
    let canonical = format!("model.layers.{layer}");
    let head = policy.head_dim.get() as usize;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        head,
        "query projection width",
    )?;
    let key_value = checked_mul(
        policy.num_key_value_heads.get() as usize,
        head,
        "key/value projection width",
    )?;
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
    common.push(safe_alias(
        format!("{released}.layer_scalar"),
        format!("{canonical}.layer_scalar"),
        vec![1],
    ));
    let intermediate = policy.intermediate_size.get() as usize;
    for (local, shape) in [
        ("self_attn.q_proj.weight", vec![query, hidden]),
        ("self_attn.o_proj.weight", vec![hidden, query]),
        ("mlp.gate_proj.weight", vec![intermediate, hidden]),
        ("mlp.up_proj.weight", vec![intermediate, hidden]),
        ("mlp.down_proj.weight", vec![hidden, intermediate]),
    ] {
        add_text_matrix(
            args,
            common,
            &format!("{released}.{local}"),
            &format!("{canonical}.{local}"),
            shape,
        )?;
    }
    common.push(safe_alias(
        format!("{released}.self_attn.q_norm.weight"),
        format!("{canonical}.self_attn.q_norm.weight"),
        vec![head],
    ));
    if policy.key_value.owns_state() {
        add_text_matrix(
            args,
            common,
            &format!("{released}.self_attn.k_proj.weight"),
            &format!("{canonical}.self_attn.k_proj.weight"),
            vec![key_value, hidden],
        )?;
        common.push(safe_alias(
            format!("{released}.self_attn.k_norm.weight"),
            format!("{canonical}.self_attn.k_norm.weight"),
            vec![head],
        ));
        if policy.key_value.value() != Some(AttentionValueSource::ReuseKey) {
            add_text_matrix(
                args,
                common,
                &format!("{released}.self_attn.v_proj.weight"),
                &format!("{canonical}.self_attn.v_proj.weight"),
                vec![key_value, hidden],
            )?;
        }
    }
    if args.hidden_size_per_layer_input > 0 {
        let media = dimension(args.hidden_size_per_layer_input, "per-layer input width")?;
        for (local, shape) in [
            ("per_layer_input_gate.weight", vec![media, hidden]),
            ("per_layer_projection.weight", vec![hidden, media]),
        ] {
            add_text_matrix(
                args,
                common,
                &format!("{released}.{local}"),
                &format!("{canonical}.{local}"),
                shape,
            )?;
        }
        common.push(safe_alias(
            format!("{released}.post_per_layer_input_norm.weight"),
            format!("{canonical}.post_per_layer_input_norm.weight"),
            vec![hidden],
        ));
    }
    if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
        add_safetensors_experts(args, &released, &canonical, hidden, common, groups)?;
    }
    Ok(())
}

fn add_safetensors_experts(
    args: &ModelArgs,
    released: &str,
    canonical: &str,
    hidden: usize,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let experts = dimension(
        args.num_experts
            .ok_or_else(|| "Gemma 4 sparse layer has no expert count".to_string())?,
        "expert count",
    )?;
    let intermediate = dimension(
        args.moe_intermediate_size
            .ok_or_else(|| "Gemma 4 sparse layer has no expert width".to_string())?,
        "expert intermediate size",
    )?;
    for (local, shape) in [
        ("router.proj.weight", vec![experts, hidden]),
        ("router.scale", vec![hidden]),
        ("router.per_expert_scale", vec![experts]),
        ("post_feedforward_layernorm_1.weight", vec![hidden]),
        ("pre_feedforward_layernorm_2.weight", vec![hidden]),
        ("post_feedforward_layernorm_2.weight", vec![hidden]),
    ] {
        let source = format!("{released}.{local}");
        let target = format!("{canonical}.{local}");
        if local.ends_with(".weight") && local == "router.proj.weight" {
            add_text_matrix(args, common, &source, &target, shape)?;
        } else {
            common.push(safe_alias(source, target, shape));
        }
    }
    let released_root = format!("{released}.experts.switch_glu");
    let canonical_root = format!("{canonical}.experts.switch_glu");
    let mut split = Vec::new();
    let split_gate = format!("{released_root}.gate_proj.weight");
    let split_up = format!("{released_root}.up_proj.weight");
    for (source, target) in [
        (&split_gate, format!("{canonical_root}.gate_proj.weight")),
        (&split_up, format!("{canonical_root}.up_proj.weight")),
    ] {
        add_text_matrix(
            args,
            &mut split,
            source,
            &target,
            vec![experts, intermediate, hidden],
        )?;
    }
    let fused = format!("{released_root}.gate_up_proj.weight");
    let mut packed = Vec::new();
    add_text_matrix(
        args,
        &mut packed,
        &fused,
        &format!("{canonical_root}.gate_up_proj"),
        vec![
            experts,
            checked_mul(2, intermediate, "fused expert width")?,
            hidden,
        ],
    )?;
    groups.push(AlternativeLayoutGroup {
        id: format!("{canonical_root} gate/up storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "separate gate and up".into(),
                tensors: split,
                discriminator_keys: vec![split_gate, split_up],
            },
            LayoutVariant {
                id: "single fused gate and up".into(),
                tensors: packed,
                discriminator_keys: vec![fused],
            },
        ],
    });
    add_text_matrix(
        args,
        common,
        &format!("{released_root}.down_proj.weight"),
        &format!("{canonical_root}.down_proj"),
        vec![experts, hidden, intermediate],
    )?;
    Ok(())
}

fn add_text_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    source: &str,
    canonical: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(
            source,
            (source != canonical).then(|| canonical.to_owned()),
            shape,
            args.linear_format_for(canonical),
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn add_safetensors_vision(
    config: &super::VisionConfig,
    text_hidden: usize,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(config.hidden_size, "vision hidden size")?;
    let intermediate = dimension(config.intermediate_size, "vision intermediate size")?;
    let head = dimension(config.head_dim, "vision head width")?;
    let query = checked_mul(
        dimension(config.num_attention_heads, "vision attention head count")?,
        head,
        "vision query width",
    )?;
    let key_value = checked_mul(
        dimension(config.num_key_value_heads, "vision KV head count")?,
        head,
        "vision key/value width",
    )?;
    let patch = dimension(config.patch_size, "vision patch size")?;
    let patch_input = checked_mul(
        3,
        checked_mul(patch, patch, "vision patch area")?,
        "vision patch input width",
    )?;
    let patch_name = "model.vision_tower.patch_embedder.input_proj.weight";
    add_matrix(
        output,
        patch_name,
        vec![hidden, patch_input],
        config.linear_format_for(patch_name, patch_input as i32),
    )?;
    output.push(safe(
        "model.vision_tower.patch_embedder.position_embedding_table",
        vec![
            2,
            dimension(config.position_embedding_size, "vision position count")?,
            hidden,
        ],
    ));
    if config.standardize {
        for local in ["std_bias", "std_scale"] {
            output.push(safe(format!("model.vision_tower.{local}"), vec![hidden]));
        }
    }
    for layer in 0..dimension(config.num_hidden_layers, "vision layer count")? {
        let root = format!("model.vision_tower.encoder.layers.{layer}");
        for local in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            output.push(safe(format!("{root}.{local}"), vec![hidden]));
        }
        for (local, shape, input) in [
            ("self_attn.q_proj", vec![query, hidden], hidden),
            ("self_attn.k_proj", vec![key_value, hidden], hidden),
            ("self_attn.v_proj", vec![key_value, hidden], hidden),
            ("self_attn.o_proj", vec![hidden, query], query),
            ("mlp.gate_proj", vec![intermediate, hidden], hidden),
            ("mlp.up_proj", vec![intermediate, hidden], hidden),
            ("mlp.down_proj", vec![hidden, intermediate], intermediate),
        ] {
            add_clipped_matrix(output, config, &format!("{root}.{local}"), shape, input)?;
        }
        for local in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            output.push(safe(format!("{root}.{local}"), vec![head]));
        }
    }
    let projection = "model.embed_vision.embedding_projection.weight";
    add_matrix(
        output,
        projection,
        vec![text_hidden, hidden],
        config.linear_format_for(projection, hidden as i32),
    )
}

fn add_clipped_matrix(
    output: &mut Vec<SafetensorsTensorConstraint>,
    config: &super::VisionConfig,
    prefix: &str,
    shape: Vec<usize>,
    input: usize,
) -> Result<(), String> {
    let weight = format!("{prefix}.linear.weight");
    add_matrix(
        output,
        &weight,
        shape,
        config.linear_format_for(&weight, input as i32),
    )?;
    for local in ["input_min", "input_max", "output_min", "output_max"] {
        output.push(safe(format!("{prefix}.{local}"), Vec::new()));
    }
    Ok(())
}

fn add_safetensors_audio(
    config: &super::AudioConfig,
    text_hidden: usize,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(config.hidden_size, "audio hidden size")?;
    let [first, second] = config.subsampling_conv_channels.as_slice() else {
        return Err("Gemma 4 audio requires two subsampling channels".into());
    };
    let first = dimension(*first, "first audio channel count")?;
    let second = dimension(*second, "second audio channel count")?;
    let projection = dimension(config.output_proj_dims, "audio output width")?;
    for (name, shape) in [
        (
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
            vec![first, 3, 3, 1],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
            vec![first],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
            vec![second, 3, 3, first],
        ),
        (
            "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
            vec![second],
        ),
    ] {
        output.push(safe(name, shape));
    }
    let input_projection = "model.audio_tower.subsample_conv_projection.input_proj_linear.weight";
    let input_width = checked_mul(32, second, "audio subsampling width")?;
    add_matrix(
        output,
        input_projection,
        vec![hidden, input_width],
        config.linear_format_for(input_projection, input_width as i32),
    )?;
    let output_projection = "model.audio_tower.output_proj.weight";
    add_matrix(
        output,
        output_projection,
        vec![projection, hidden],
        config.linear_format_for(output_projection, hidden as i32),
    )?;
    output.push(safe("model.audio_tower.output_proj.bias", vec![projection]).optional());
    for layer in 0..dimension(config.num_hidden_layers, "audio layer count")? {
        let root = format!("model.audio_tower.layers.{layer}");
        for local in [
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
            "norm_out.weight",
        ] {
            output.push(safe(format!("{root}.{local}"), vec![hidden]));
        }
        let four_hidden = checked_mul(4, hidden, "audio feed-forward width")?;
        let two_hidden = checked_mul(2, hidden, "audio convolution gate width")?;
        for (local, shape, input) in [
            (
                "feed_forward1.ffw_layer_1",
                vec![four_hidden, hidden],
                hidden,
            ),
            (
                "feed_forward1.ffw_layer_2",
                vec![hidden, four_hidden],
                four_hidden,
            ),
            ("self_attn.q_proj", vec![hidden, hidden], hidden),
            ("self_attn.k_proj", vec![hidden, hidden], hidden),
            ("self_attn.v_proj", vec![hidden, hidden], hidden),
            ("self_attn.post", vec![hidden, hidden], hidden),
            ("lconv1d.linear_start", vec![two_hidden, hidden], hidden),
            ("lconv1d.linear_end", vec![hidden, hidden], hidden),
            (
                "feed_forward2.ffw_layer_1",
                vec![four_hidden, hidden],
                hidden,
            ),
            (
                "feed_forward2.ffw_layer_2",
                vec![hidden, four_hidden],
                four_hidden,
            ),
        ] {
            add_audio_clipped_matrix(output, config, &format!("{root}.{local}"), shape, input)?;
        }
        output.extend([
            safe(
                format!("{root}.self_attn.relative_k_proj.weight"),
                vec![hidden, hidden],
            ),
            safe(
                format!("{root}.self_attn.per_dim_scale"),
                vec![config.head_dim() as usize],
            ),
            safe(
                format!("{root}.lconv1d.depthwise_conv1d.weight"),
                vec![
                    hidden,
                    dimension(config.conv_kernel_size, "audio convolution kernel")?,
                    1,
                ],
            ),
        ]);
    }
    let media_projection = "model.embed_audio.embedding_projection.weight";
    add_matrix(
        output,
        media_projection,
        vec![text_hidden, projection],
        config.linear_format_for(media_projection, projection as i32),
    )
}

fn add_audio_clipped_matrix(
    output: &mut Vec<SafetensorsTensorConstraint>,
    config: &super::AudioConfig,
    prefix: &str,
    shape: Vec<usize>,
    input: usize,
) -> Result<(), String> {
    let weight = format!("{prefix}.linear.weight");
    add_matrix(
        output,
        &weight,
        shape,
        config.linear_format_for(&weight, input as i32),
    )?;
    for local in ["input_min", "input_max", "output_min", "output_max"] {
        output.push(safe(format!("{prefix}.{local}"), Vec::new()));
    }
    Ok(())
}

fn add_matrix(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    format: eredu_checkpoint::LinearFormat,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(name, std::iter::empty::<String>(), shape, format, None)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn safe_alias(
    source: impl Into<String>,
    canonical: impl Into<String>,
    shape: Vec<usize>,
) -> SafetensorsTensorConstraint {
    let source = source.into();
    let canonical = canonical.into();
    safe(source.clone(), shape).with_aliases((source != canonical).then_some(canonical))
}

fn safe(name: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Gemma 4 {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Gemma 4 {name} geometry overflows"))
}

/// Builds the strict text GGUF catalog, including shared-KV omissions and
/// split/fused routed-expert alternatives.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let layers = args.num_hidden_layers();
    let vocabulary = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = vec![
        gguf(
            "token_embd.weight",
            vec![vocabulary, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        common.push(gguf(
            "output.weight",
            vec![vocabulary, hidden],
            TensorOperation::Matrix,
        ));
    }
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dimension(args.hidden_size_per_layer_input, "per-layer input width")?;
        let combined = checked_mul(layers, per_layer, "combined per-layer input width")?;
        let per_layer_vocab = dimension(
            args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
            "per-layer input vocabulary size",
        )?;
        common.extend([
            gguf(
                "per_layer_token_embd.weight",
                vec![per_layer_vocab, combined],
                TensorOperation::Matrix,
            ),
            gguf(
                "per_layer_model_proj.weight",
                vec![combined, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                "per_layer_proj_norm.weight",
                vec![per_layer],
                TensorOperation::Vector,
            ),
        ]);
    }
    let attention_heads = dimension(args.num_attention_heads, "attention head count")?;
    let mut groups = Vec::new();
    let mut catalog = CatalogPolicy::strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| format!("Gemma 4 layer {layer} has no normalized policy"))?;
        let head = policy.head_dim.get() as usize;
        let query = checked_mul(attention_heads, head, "query projection width")?;
        let key_value = checked_mul(
            policy.num_key_value_heads.get() as usize,
            head,
            "key/value projection width",
        )?;
        for local in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
        ] {
            common.push(gguf(
                format!("{root}.{local}"),
                vec![hidden],
                TensorOperation::Vector,
            ));
        }
        let intermediate = policy.intermediate_size.get() as usize;
        common.extend([
            gguf(
                format!("{root}.layer_output_scale.weight"),
                vec![1],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_q.weight"),
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_output.weight"),
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_q_norm.weight"),
                vec![head],
                TensorOperation::Vector,
            ),
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
        if policy.key_value.owns_state() {
            common.extend([
                gguf(
                    format!("{root}.attn_k.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.attn_k_norm.weight"),
                    vec![head],
                    TensorOperation::Vector,
                ),
            ]);
            if policy.key_value.value() != Some(AttentionValueSource::ReuseKey) {
                common.push(gguf(
                    format!("{root}.attn_v.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ));
            }
        } else {
            catalog.allowed_prefixes.extend([
                format!("{root}.attn_k."),
                format!("{root}.attn_v."),
                format!("{root}.attn_k_norm."),
            ]);
        }
        if args.hidden_size_per_layer_input > 0 {
            let media = dimension(args.hidden_size_per_layer_input, "per-layer input width")?;
            common.extend([
                gguf(
                    format!("{root}.inp_gate.weight"),
                    vec![media, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.proj.weight"),
                    vec![hidden, media],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.post_norm.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ]);
        }
        if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
            let experts = dimension(
                args.num_experts
                    .ok_or_else(|| "Gemma 4 MoE has no expert count".to_string())?,
                "expert count",
            )?;
            let moe = dimension(
                args.moe_intermediate_size
                    .ok_or_else(|| "Gemma 4 MoE has no expert width".to_string())?,
                "expert width",
            )?;
            common.extend([
                gguf(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_gate_inp.scale"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.scale"),
                    vec![experts],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.post_ffw_norm_1.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.pre_ffw_norm_2.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.post_ffw_norm_2.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, hidden, moe],
                    TensorOperation::Matrix,
                ),
            ]);
            let gate = format!("{root}.ffn_gate_exps.weight");
            let up = format!("{root}.ffn_up_exps.weight");
            let fused = format!("{root}.ffn_gate_up_exps.weight");
            groups.push(AlternativeLayoutGroup {
                id: format!("{root} expert gate/up storage"),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "separate gate and up".into(),
                        tensors: vec![
                            gguf(&gate, vec![experts, moe, hidden], TensorOperation::Matrix),
                            gguf(&up, vec![experts, moe, hidden], TensorOperation::Matrix),
                        ],
                        discriminator_keys: vec![gate, up],
                    },
                    LayoutVariant {
                        id: "single fused gate and up".into(),
                        tensors: vec![gguf(
                            &fused,
                            vec![experts, checked_mul(2, moe, "fused expert width")?, hidden],
                            TensorOperation::Matrix,
                        )],
                        discriminator_keys: vec![fused],
                    },
                ],
            });
        }
    }
    GgufCheckpointPlan::new("Gemma 4 GGUF", common, groups, catalog)
        .map_err(|error| error.to_string())
}

fn gguf(
    name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(name, shape, GgufTypeConstraint::OperationClass(operation))
}

/// Translates one released Gemma 4 text GGUF name to neutral parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
        ("per_layer_token_embd", "model.embed_tokens_per_layer"),
        ("per_layer_model_proj", "model.per_layer_model_projection"),
        ("per_layer_proj_norm", "model.per_layer_projection_norm"),
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
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffw_norm_1", "post_feedforward_layernorm_1"),
        ("pre_ffw_norm_2", "pre_feedforward_layernorm_2"),
        ("post_ffw_norm_2", "post_feedforward_layernorm_2"),
        ("post_ffw_norm", "post_feedforward_layernorm"),
        ("layer_output_scale", "layer_scalar"),
        ("ffn_gate_inp", "router.proj"),
        ("ffn_gate_up_exps", "experts.switch_glu.gate_up_proj"),
        ("ffn_gate_exps", "experts.switch_glu.gate_proj"),
        ("ffn_up_exps", "experts.switch_glu.up_proj"),
        ("ffn_down_exps", "experts.switch_glu.down_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("inp_gate", "per_layer_input_gate"),
        ("proj", "per_layer_projection"),
        ("post_norm", "post_per_layer_input_norm"),
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

/// Resolves the released separate gate/up or already-fused sparse bank into
/// the canonical neutral expert slots used by the ordinary Gemma block.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("Gemma 4 expert recipe layer {layer} is out of range"))?;
    if policy.feed_forward != FeedForwardPolicy::DenseWithSparseMoe {
        return Err(format!("Gemma 4 layer {layer} has no routed experts"));
    }
    let prefix = format!("{}.experts.switch_glu", released_layer_path(layer));
    let source = |base: &str| {
        if catalog.tensor_metadata(base).is_ok() {
            base.to_owned()
        } else {
            format!("{base}.weight")
        }
    };
    let names = GatedProductExpertLayoutNames {
        target_gate_up: format!("{prefix}.gate_up_proj"),
        target_down: format!("{prefix}.down_proj"),
        packed_gate_up: source(&format!("{prefix}.gate_up_proj")),
        packed_down: source(&format!("{prefix}.down_proj")),
        separate_gate: source(&format!("{prefix}.gate_proj")),
        separate_up: source(&format!("{prefix}.up_proj")),
        separate_down: source(&format!("{prefix}.down_proj")),
        independent: Vec::new(),
    };
    resolve_gated_product_expert_recipes(catalog, &names).map_err(|error| error.to_string())
}

/// Returns the derived-weight catalog for one canonical Gemma execution ordinal.
///
/// The family graph places optional vision and audio units before the decoder.
/// Recipe callers pass the architecture-flat ordinal unchanged so this neutral
/// catalog remains the sole owner of that topology-to-decoder translation.
pub fn unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    family: &FamilyConfig,
    ordinal: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let vision_layers = family.vision.as_ref().map_or(Ok(0), |config| {
        usize::try_from(config.num_hidden_layers)
            .map_err(|_| "invalid Gemma 4 vision layer count".to_string())
    })?;
    let audio_layers = family.audio.as_ref().map_or(Ok(0), |config| {
        usize::try_from(config.num_hidden_layers)
            .map_err(|_| "invalid Gemma 4 audio layer count".to_string())
    })?;
    let decoder_start = vision_layers
        .checked_add(audio_layers)
        .ok_or_else(|| "Gemma 4 media unit count overflowed".to_string())?;
    if ordinal < decoder_start {
        return Ok(BTreeMap::new());
    }
    let layer = ordinal - decoder_start;
    if layer >= family.text.num_hidden_layers() {
        return Err(format!(
            "Gemma 4 execution ordinal {ordinal} is outside its architecture layout"
        ));
    }
    if family
        .text
        .layer_policy(layer)
        .is_none_or(|policy| policy.feed_forward != FeedForwardPolicy::DenseWithSparseMoe)
    {
        return Ok(BTreeMap::new());
    }
    let resolved = expert_recipes(catalog, &family.text, layer)?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let experts = usize::try_from(
        args.num_experts
            .ok_or_else(|| "Gemma 4 expert residency requires an expert count".to_string())?,
    )
    .ok()
    .filter(|count| *count > 0)
    .ok_or_else(|| "Gemma 4 expert count must be positive".to_string())?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for layer in 0..args.num_hidden_layers() {
        if args
            .layer_policy(layer)
            .is_none_or(|policy| policy.feed_forward != FeedForwardPolicy::DenseWithSparseMoe)
        {
            continue;
        }
        let unit_path = released_layer_path(layer);
        let bank = expert_recipes(catalog, args, layer)?;
        for expert in 0..experts {
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert
                    .checked_add(1)
                    .ok_or_else(|| "Gemma 4 expert index overflowed".to_string())?,
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
                    eredu_runtime::ExpertIdentity::new(layer, expert),
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
    crate::ExpertResidencyCatalog::new(units).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap, HashSet};

    use eredu_checkpoint::{
        expert::GatedProductExpertStorageLayout,
        recipe::{DerivedWeightRecipe, RecipeCatalog},
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };

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
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    fn sparse_args() -> ModelArgs {
        ModelArgs::from_hf_json(
            br#"{
                "model_type":"gemma4","hidden_size":16,"num_hidden_layers":1,
                "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":64,
                "max_position_embeddings":128,"layer_types":["full_attention"],
                "enable_moe_block":true,"num_experts":4,"top_k_experts":2,
                "moe_intermediate_size":8
            }"#,
        )
        .unwrap()
    }

    fn sparse_family() -> FamilyConfig {
        FamilyConfig::from_hf_json(
            br#"{
                "model_type":"gemma4","tie_word_embeddings":false,
                "image_token_id":60,"audio_token_id":61,
                "text_config":{
                    "hidden_size":32,"num_hidden_layers":1,"intermediate_size":64,
                    "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                    "rms_norm_eps":0.000001,"vocab_size":64,"max_position_embeddings":128,
                    "layer_types":["full_attention"],"enable_moe_block":true,
                    "num_experts":4,"top_k_experts":2,"moe_intermediate_size":32
                },
                "vision_config":{
                    "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                    "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
                    "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                    "rms_norm_eps":0.000001
                },
                "audio_config":{
                    "hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                    "output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,
                    "attention_context_left":5,"attention_context_right":0,
                    "attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,
                    "residual_weight":0.5,"rms_norm_eps":0.000001,
                    "subsampling_conv_channels":[4,8]
                }
            }"#,
        )
        .unwrap()
    }

    fn config() -> Gemma4ArtifactConfig {
        Gemma4ArtifactConfig {
            unified: true,
            hidden_size: 32,
            image_token_id: Some(100),
            video_token_id: None,
            audio_token_id: Some(101),
            projector: true,
            assistant: true,
        }
    }

    #[test]
    fn load_time_quantization_replaces_all_family_checkpoint_formats() {
        let mut source = sparse_family();
        let checkpoint_quantization =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 8).unwrap());
        source.text.quantized_weights = Some(HashSet::from(["text.weight".into()]));
        source.text.quantized_weight_configs = Some(HashMap::from([(
            "text.weight".into(),
            checkpoint_quantization,
        )]));
        let vision = source.vision.as_mut().unwrap();
        vision.quantized_weights = Some(HashSet::from(["vision.weight".into()]));
        vision.quantized_weight_configs = Some(HashMap::from([(
            "vision.weight".into(),
            checkpoint_quantization,
        )]));
        let audio = source.audio.as_mut().unwrap();
        audio.quantized_weights = Some(HashSet::from(["audio.weight".into()]));
        audio.quantized_weight_configs = Some(HashMap::from([(
            "audio.weight".into(),
            checkpoint_quantization,
        )]));

        let requested =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let target = load_time_quantization(&source, requested).unwrap();

        assert_eq!(target.text.weight_quantization, Some(requested));
        assert_eq!(target.text.quantized_weights, None);
        assert_eq!(target.text.quantized_weight_configs, None);
        let vision = target.vision.unwrap();
        assert_eq!(vision.weight_quantization, Some(requested));
        assert_eq!(vision.quantized_weights, None);
        assert_eq!(vision.quantized_weight_configs, None);
        let audio = target.audio.unwrap();
        assert_eq!(audio.weight_quantization, Some(requested));
        assert_eq!(audio.quantized_weights, None);
        assert_eq!(audio.quantized_weight_configs, None);
        assert!(source.text.quantized_weight_configs.is_some());
        assert!(source
            .vision
            .as_ref()
            .unwrap()
            .quantized_weight_configs
            .is_some());
        assert!(source
            .audio
            .as_ref()
            .unwrap()
            .quantized_weight_configs
            .is_some());
    }

    #[test]
    fn artifact_schema_freezes_optional_projector_and_required_assistant() {
        let config = config();
        let schema = config.artifact_schema().unwrap();
        assert_eq!(schema.primary().architecture, "gemma4_unified");
        assert_eq!(schema.siblings().len(), 2);
        assert!(!schema.siblings()[0].required);
        assert!(schema.siblings()[1].required);
        let catalog = config.parameter_catalog().unwrap();
        assert_eq!(
            catalog
                .owner("masked_embedding.token_ordering")
                .unwrap()
                .as_str(),
            "assistant"
        );
    }

    #[test]
    fn sibling_projector_must_match_identity_width_and_modality_tokens() {
        let config = config();
        config
            .projector_compatibility("gemma4_unified", 32, BTreeSet::from([100, 101]))
            .validate()
            .unwrap();
        assert!(config
            .projector_compatibility("gemma4", 32, BTreeSet::from([100, 101]))
            .validate()
            .is_err());
        assert!(config
            .projector_compatibility("gemma4_unified", 16, BTreeSet::from([100, 101]))
            .validate()
            .is_err());
    }

    #[test]
    fn separate_expert_bank_derives_exact_fused_neutral_targets() {
        let prefix = "model.language_model.layers.0.experts.switch_glu";
        let tensors = BTreeMap::from([
            (
                format!("{prefix}.gate_proj.weight"),
                metadata(&format!("{prefix}.gate_proj.weight"), vec![4, 8, 16]),
            ),
            (
                format!("{prefix}.up_proj.weight"),
                metadata(&format!("{prefix}.up_proj.weight"), vec![4, 8, 16]),
            ),
            (
                format!("{prefix}.down_proj.weight"),
                metadata(&format!("{prefix}.down_proj.weight"), vec![4, 16, 8]),
            ),
        ]);
        let catalog = Catalog(tensors);
        let recipes = expert_recipes(&catalog, &sparse_args(), 0).unwrap();
        assert_eq!(
            recipes.layout,
            GatedProductExpertStorageLayout::SeparatePacked
        );
        assert_eq!(recipes.target_gate_up, format!("{prefix}.gate_up_proj"));
        assert_eq!(
            recipes.gate_up.infer(&catalog).unwrap().shape(),
            &[4, 16, 16]
        );
        assert!(matches!(
            recipes.gate_up,
            DerivedWeightRecipe::Concatenate { axis: 1, .. }
        ));
    }

    #[test]
    fn residency_catalog_owns_sparse_schedule_identity_and_bindings() {
        let args = sparse_args();
        let root = "model.language_model.layers.0.experts.switch_glu";
        let catalog = Catalog(BTreeMap::from([
            (
                format!("{root}.gate_up_proj"),
                metadata(&format!("{root}.gate_up_proj"), vec![4, 16, 16]),
            ),
            (
                format!("{root}.down_proj"),
                metadata(&format!("{root}.down_proj"), vec![4, 16, 8]),
            ),
        ]));
        let residency = expert_residency_catalog(&catalog, &args).unwrap();
        assert_eq!(residency.units().len(), 4);
        let first = &residency.units()[0];
        assert_eq!(first.identity(), eredu_runtime::ExpertIdentity::new(0, 0));
        assert_eq!(first.owner_group().as_str(), "text_decoder");
        assert_eq!(first.owner_unit(), 0);
        assert_eq!(first.unit_path(), "model.language_model.layers.0");
        assert_eq!(
            first
                .parameters()
                .iter()
                .map(|parameter| (parameter.binding_name(), parameter.logical_target()))
                .collect::<Vec<_>>(),
            [
                (
                    "gate_up_proj",
                    "model.language_model.layers.0.experts.switch_glu.gate_up_proj"
                ),
                (
                    "down_proj",
                    "model.language_model.layers.0.experts.switch_glu.down_proj"
                ),
            ]
        );
        assert_eq!(
            residency.units()[3].identity(),
            eredu_runtime::ExpertIdentity::new(0, 3)
        );
    }

    #[test]
    fn strict_plans_cover_media_shared_policy_and_expert_alternatives() {
        let family = sparse_family();
        let safe = safetensors_plan(&family).unwrap();
        assert!(safe.catalog_policy.strict);
        assert_eq!(safe.layout_groups.len(), 1);
        assert_eq!(safe.layout_groups[0].variants.len(), 2);
        assert!(safe
            .common_tensors
            .iter()
            .any(|tensor| { tensor.key == "model.vision_tower.patch_embedder.input_proj.weight" }));
        assert!(safe.common_tensors.iter().any(|tensor| {
            tensor.key == "model.audio_tower.subsample_conv_projection.layer0.conv.weight"
        }));
        let gguf = gguf_plan(&family.text).unwrap();
        assert!(gguf.catalog_policy.strict);
        assert_eq!(gguf.layout_groups.len(), 1);
        assert_eq!(
            translate_gguf_weight_name("blk.0.ffn_gate_up_exps.weight"),
            "model.layers.0.experts.switch_glu.gate_up_proj.weight"
        );
    }

    #[test]
    fn unit_recipes_use_the_composite_architecture_ordinal() {
        let family = sparse_family();
        let root = "model.language_model.layers.0.experts.switch_glu";
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

        assert!(unit_recipes(&catalog, &family, 0).unwrap().is_empty());
        assert!(unit_recipes(&catalog, &family, 1).unwrap().is_empty());
        let decoder = unit_recipes(&catalog, &family, 2).unwrap();
        assert!(decoder.contains_key(&format!("{root}.gate_up_proj")));
        assert!(decoder.contains_key(&format!("{root}.down_proj")));
        assert!(unit_recipes(&catalog, &family, 3).is_err());
    }
}
