//! Composite artifact policy for Gemma 4 checkpoints.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eredu_checkpoint::composite::{
    ArtifactComponentSchema, ArtifactRole, ComponentId, ComponentParameterCatalog,
    CompositeArtifactError, CompositeArtifactSchema, ProjectorCompatibility,
};
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

use super::{FamilyConfig, FeedForwardPolicy, ModelArgs};

const RELEASED_LAYER_ROOT: &str = "model.language_model.layers";

/// Derives a Gemma 4 family configuration whose text and aligned media
/// matrix formats reflect load-time quantization.
pub fn load_time_quantization(
    config: &FamilyConfig,
    quantization: WeightQuantization,
) -> Result<FamilyConfig, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = config.policy().clone();
    target.text.weight_quantization = Some(quantization);
    target.text.quantized_weights = None;
    target.text.quantized_weight_configs = None;
    if let Some(vision) = target.vision.as_mut() {
        *vision = vision
            .with_formats(Some(quantization), None, None)
            .map_err(|e| e.to_string())?;
    }
    if let Some(audio) = target.audio.as_mut() {
        *audio = audio
            .with_formats(Some(quantization), None, None)
            .map_err(|e| e.to_string())?;
    }
    config
        .project_storage(target)
        .map_err(|error| error.to_string())
}

/// Applies canonical text checkpoint formats to a complete Gemma 4 family
/// configuration.
pub fn with_checkpoint_formats(
    config: &FamilyConfig,
    formats: HashMap<String, WeightQuantization>,
) -> Result<FamilyConfig, String> {
    let mut target = config.policy().clone();
    let mut text = HashMap::new();
    let mut vision = HashMap::new();
    let mut audio = HashMap::new();
    for (name, format) in formats {
        if name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.") {
            vision.insert(name.clone(), format);
            if name.starts_with("model.embed_vision.") {
                text.insert(name, format);
            }
        } else if name.starts_with("model.audio_tower.") || name.starts_with("model.embed_audio.") {
            audio.insert(name.clone(), format);
            if name.starts_with("model.embed_audio.") {
                text.insert(name, format);
            }
        } else {
            text.insert(name, format);
        }
    }
    target.text.weight_quantization = None;
    target.text.quantized_weights = None;
    target.text.quantized_weight_configs = (!text.is_empty()).then_some(text);
    if let Some(config) = target.vision.as_mut() {
        *config = config
            .with_formats(None, None, (!vision.is_empty()).then_some(vision))
            .map_err(|e| e.to_string())?;
    } else if !vision.is_empty() {
        return Err("vision formats have no declared media owner".into());
    }
    if let Some(config) = target.audio.as_mut() {
        *config = config
            .with_formats(None, None, (!audio.is_empty()).then_some(audio))
            .map_err(|e| e.to_string())?;
    } else if !audio.is_empty() {
        return Err("audio formats have no declared media owner".into());
    }
    config
        .project_storage(target)
        .map_err(|error| error.to_string())
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
                    super::media::ProjectionKind::Vision.gemma_weight().into(),
                    super::media::ProjectionKind::Audio.gemma_weight().into(),
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
    shared_safetensors_plan(
        &family.text,
        family.vision.as_ref(),
        "model.vision_tower",
        family.audio.as_ref(),
        family.pinned(),
        family.projectors(),
        family.blocks(),
        family.dense_scopes(),
        family.attention_scopes(),
        family.routed_scopes(),
    )
}

/// Projects matrix-shaped source geometry by resolving format identities to
/// construction declarations. Lookup tables retain their independent semantics;
/// floating source exceptions never require inverse packing arithmetic.
pub(crate) fn source_linear_shapes(
    family: &FamilyConfig,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let declaration =
        super::model::parameter_description(family).map_err(|error| error.to_string())?;
    let pinned = family.pinned();
    let embedding = pinned.embedding().weight.id.as_str();
    let logical = declaration
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| member.target() != embedding)
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target(), shape))
        })
        .collect::<BTreeMap<_, _>>();
    let plan = safetensors_plan(family)?;
    let mut shapes = BTreeMap::new();
    for source in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .flat_map(|group| &group.variants)
            .flat_map(|variant| &variant.tensors),
    ) {
        if let Some(shape) = source_matrix_shape(source, &logical, embedding)? {
            shapes.insert(source.key.clone(), shape);
        }
    }
    Ok(shapes)
}

fn source_matrix_shape(
    source: &SafetensorsTensorConstraint,
    logical: &BTreeMap<&str, &[usize]>,
    embedding: &str,
) -> Result<Option<Vec<usize>>, String> {
    if source.role != eredu_checkpoint::schema::TensorRole::Tensor || source.shape.len() != 2 {
        return Ok(None);
    }
    let names =
        || std::iter::once(source.key.as_str()).chain(source.aliases.iter().map(String::as_str));
    if names().any(|name| name == embedding) {
        return Ok(None);
    }
    let mut resolved = names().filter_map(|name| logical.get(name).copied());
    let shape = resolved.next().ok_or_else(|| {
        format!(
            "Gemma source matrix {:?} has no declared logical geometry",
            source.key
        )
    })?;
    if resolved.any(|other| other != shape) {
        return Err(format!(
            "Gemma source matrix {:?} aliases have conflicting logical geometry",
            source.key
        ));
    }
    Ok(Some(shape.to_vec()))
}

pub(crate) fn shared_safetensors_plan(
    args: &ModelArgs,
    vision: Option<&super::VisionConfig>,
    vision_root: &str,
    audio: Option<&super::AudioConfig>,
    declaration: &super::text::pinned::Declaration,
    projectors: &[super::media::ProjectionSpec],
    blocks: &[super::text::specs::BlockSpec],
    dense_scopes: &[super::text::specs::DenseSpec],
    attention_scopes: &[super::text::specs::AttentionSpec],
    routed_scopes: &[Option<super::text::routed::Declaration>],
) -> Result<SafetensorsCheckpointPlan, String> {
    declaration.validate_for(args).map_err(|e| e.to_string())?;
    let layers = args.num_hidden_layers();
    let mut common = Vec::new();
    let mut groups = Vec::new();
    use eredu_nn::DeclareLogicalParameters;
    for parameter in declaration
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        let id = parameter.parameter.id.as_str();
        let (source, target) =
            if let Some(local) = id.strip_prefix(&format!("{}.", declaration.root())) {
                (
                    format!("model.language_model.{local}"),
                    format!("model.{local}"),
                )
            } else {
                (id.to_owned(), id.to_owned())
            };
        if let Some(format) = parameter.linear_format {
            common.extend(
                matrix_for_linear_format(
                    &source,
                    (&source != &target).then_some(target),
                    parameter.shape,
                    format.encoding(),
                    None,
                )
                .map_err(|e| e.to_string())?,
            );
        } else {
            common.push(safe_alias(source, target, parameter.shape));
        }
    }
    for layer in 0..layers {
        add_safetensors_layer(
            args,
            layer,
            blocks.get(layer).ok_or("missing Gemma block owner")?,
            dense_scopes.get(layer).ok_or("missing Gemma dense owner")?,
            attention_scopes
                .get(layer)
                .ok_or("missing Gemma attention owner")?,
            routed_scopes
                .get(layer)
                .ok_or("missing Gemma routed scope")?
                .as_ref(),
            &mut common,
            &mut groups,
        )?;
    }
    if let Some(vision) = vision {
        add_safetensors_vision(vision, vision_root, &mut common)?;
    }
    if let Some(audio) = audio {
        add_safetensors_audio(audio, &mut common)?;
    }
    for spec in projectors {
        let source = spec.kind.gemma_weight();
        for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
            if parameter.parameter.id != spec.projection.weight.id {
                return Err("Gemma projector storage extension does not cover bias slots".into());
            }
            add_matrix(
                &mut common,
                source,
                parameter.shape,
                spec.projection.format.encoding(),
            )?;
        }
    }
    for constraint in &mut common {
        add_mlx_vlm_aliases(constraint);
    }
    for group in &mut groups {
        for variant in &mut group.variants {
            for constraint in &mut variant.tensors {
                add_mlx_vlm_aliases(constraint);
            }
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.extend([
        "multi_modal_projector.".into(),
        "model.multi_modal_projector.".into(),
        "model.vision_embedder.".into(),
        "vision_embedder.".into(),
    ]);
    // A text-only conversion may keep the media-to-text projections of the
    // towers it dropped; with no encoder declared they are dead weight, not an
    // unexpected tensor.
    if vision.is_none() {
        policy
            .allowed_prefixes
            .extend(["model.embed_vision.".into(), "embed_vision.".into()]);
    }
    if audio.is_none() {
        policy
            .allowed_prefixes
            .extend(["model.embed_audio.".into(), "embed_audio.".into()]);
    }
    SafetensorsCheckpointPlan::new("Gemma 4 SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

/// Physical name of a released-layout tensor in the `mlx-vlm` /
/// `mlx-community` SafeTensors conversions, which nest the decoder under
/// `language_model.model` and keep the media towers and projections at the
/// root without the `model.` prefix. Returns `None` for names that already
/// use a layout those conversions share with the released one.
fn mlx_vlm_physical_name(name: &str) -> Option<String> {
    for (canonical, physical) in [
        ("model.language_model.", "language_model.model."),
        ("lm_head.", "language_model.lm_head."),
        ("model.vision_tower.", "vision_tower."),
        ("model.embed_vision.", "embed_vision."),
        ("model.audio_tower.", "audio_tower."),
        ("model.embed_audio.", "embed_audio."),
    ] {
        if let Some(rest) = name.strip_prefix(canonical) {
            return Some(format!("{physical}{rest}"));
        }
    }
    None
}

/// Exact optional audio bias presence across admitted primary and companion sources.
pub(crate) fn audio_output_projection_bias<P>(
    inspection: &eredu_core::artifact::ArtifactInspection<P>,
) -> bool {
    let projector = inspection.validated_gguf().and_then(|gguf| {
        gguf.companion(&eredu_core::artifact::ArtifactCompanionRole::MediaProjector)
            .map(|companion| companion.checkpoint())
    });
    audio_output_projection_bias_in(inspection.tensors(), projector)
}

fn audio_output_projection_bias_in(
    tensors: &eredu_core::checkpoint::TensorCatalog,
    projector: Option<&eredu_gguf::Checkpoint>,
) -> bool {
    let canonical = "model.audio_tower.output_proj.bias";
    tensors.get(canonical).is_some()
        || mlx_vlm_physical_name(canonical).is_some_and(|physical| tensors.get(&physical).is_some())
        || projector_physical_name(canonical).is_some_and(|physical| {
            projector.is_some_and(|projector| {
                projector
                    .tensors()
                    .any(|tensor| tensor.descriptor().name == physical)
            })
        })
}

/// Adds the `mlx-vlm` physical alias of a constraint's key and existing
/// aliases so one plan admits the released, canonical, and MLX conversions.
pub(crate) fn add_mlx_vlm_aliases(constraint: &mut SafetensorsTensorConstraint) {
    let extra = std::iter::once(&constraint.key)
        .chain(constraint.aliases.iter())
        .filter_map(|name| mlx_vlm_physical_name(name))
        .filter(|physical| physical != &constraint.key && !constraint.aliases.contains(physical))
        .collect::<Vec<_>>();
    for physical in extra {
        if !constraint.aliases.contains(&physical) {
            constraint.aliases.push(physical);
        }
    }
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
    block: &super::text::specs::BlockSpec,
    dense: &super::text::specs::DenseSpec,
    attention: &super::text::specs::AttentionSpec,
    routed: Option<&super::text::routed::Declaration>,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let spec = super::text::specs::DecoderBlockSpec::from_scopes(
        args,
        layer,
        block
            .in_namespace(layer, "model.layers")
            .map_err(|e| e.to_string())?,
        dense
            .in_namespace(layer, "model.layers")
            .map_err(|e| e.to_string())?,
        attention
            .in_namespace(layer, "model.layers")
            .map_err(|e| e.to_string())?,
        routed
            .map(|s| s.in_namespace(args, layer, "model.layers"))
            .transpose()
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    add_safetensors_block(layer, &spec, &released_layer_path(layer), common, groups)
}

pub(crate) fn add_safetensors_block(
    layer: usize,
    spec: &super::text::specs::DecoderBlockSpec,
    released: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let canonical = format!("model.layers.{layer}");
    let attention = &spec.attention;
    let dense = &spec.dense;
    let block = &spec.block;
    for parameter in attention
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .chain(dense.logical_parameters().map_err(|e| e.to_string())?)
        .chain(block.logical_parameters().map_err(|e| e.to_string())?)
    {
        let target = parameter.parameter.id.as_str();
        let local = target
            .strip_prefix(&format!("{canonical}."))
            .ok_or("Gemma ordinary extension references another invocation")?;
        let source = format!("{released}.{local}");
        if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
            add_text_matrix(
                parameter
                    .linear_format
                    .ok_or("matrix slot has no declared encoding")?
                    .encoding(),
                common,
                &source,
                target,
                parameter.shape,
            )?;
        } else {
            common.push(safe_alias(source, target, parameter.shape));
        }
    }
    if let Some(routed) = &spec.routed {
        add_safetensors_experts(routed, released, &canonical, common, groups)?;
    }
    Ok(())
}

fn add_safetensors_experts(
    declaration: &super::text::routed::Declaration,
    released: &str,
    canonical: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let source = declaration
        .routed
        .experts
        .packed_parameters()
        .map_err(|e| e.to_string())?;
    let gate_up = source.gate_up();
    let down = source.down();
    let fused_shape = gate_up.shape.clone();
    let down_shape = down.shape.clone();
    let split_shape = source.unfused_shape();
    for parameter in declaration
        .router_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .chain(
            declaration.norms[..]
                .logical_parameters()
                .map_err(|e| e.to_string())?,
        )
    {
        let target = parameter.parameter.id.as_str();
        let local = target
            .strip_prefix(&format!("{canonical}."))
            .ok_or("Gemma routed extension references another unit")?;
        let source = format!("{released}.{local}");
        if parameter.parameter.id == declaration.routed.selector.weight().id {
            add_text_matrix(
                parameter
                    .linear_format
                    .ok_or("matrix slot has no declared encoding")?
                    .encoding(),
                common,
                &source,
                target,
                parameter.shape,
            )?;
        } else {
            common.push(safe_alias(source, target, parameter.shape));
        }
    }
    let released_root = format!("{released}.experts.switch_glu");
    let canonical_root = gate_up
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved Gemma expert source root")?;
    let mut split = Vec::new();
    let split_gate = format!("{released_root}.gate_proj.weight");
    let split_up = format!("{released_root}.up_proj.weight");
    for ((source, target), format) in [
        (&split_gate, format!("{canonical_root}.gate_proj.weight")),
        (&split_up, format!("{canonical_root}.up_proj.weight")),
    ]
    .into_iter()
    .zip(declaration.split_formats)
    {
        add_text_matrix(format, &mut split, source, &target, split_shape.clone())?;
    }
    let fused = format!("{released_root}.gate_up_proj.weight");
    let mut packed = Vec::new();
    add_text_matrix(
        gate_up
            .linear_format
            .ok_or("expert bank has no declared encoding")?
            .encoding(),
        &mut packed,
        &fused,
        gate_up.parameter.id.as_str(),
        fused_shape,
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
        down.linear_format
            .ok_or("expert bank has no declared encoding")?
            .encoding(),
        common,
        &format!("{released_root}.down_proj.weight"),
        down.parameter.id.as_str(),
        down_shape,
    )?;
    Ok(())
}

fn add_text_matrix(
    format: eredu_checkpoint::LinearFormat,
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
            format,
            None,
        )
        .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn add_safetensors_vision(
    config: &super::VisionConfig,
    vision_root: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let spec = config.pinned(vision_root).map_err(|e| e.to_string())?;
    add_vision_parameters(
        output,
        vision_root,
        spec.logical_parameters().map_err(|e| e.to_string())?,
    )?;
    for layer in 0..dimension(config.num_hidden_layers, "vision layer count")? {
        let spec = config
            .layer(layer, vision_root)
            .map_err(|e| e.to_string())?;
        add_vision_parameters(
            output,
            vision_root,
            spec.logical_parameters().map_err(|e| e.to_string())?,
        )?;
    }
    Ok(())
}

fn add_vision_parameters(
    output: &mut Vec<SafetensorsTensorConstraint>,
    vision_root: &str,
    parameters: Vec<eredu_nn::LogicalParameterDeclaration<'_>>,
) -> Result<(), String> {
    for parameter in parameters {
        let id = parameter.parameter.id.as_str();
        // Published source names share the Gemma tower tree; construction keeps
        // the actual family identity when selecting per-parameter encodings.
        let source = id.replacen(vision_root, "model.vision_tower", 1);
        if let Some(format) = parameter.linear_format {
            add_matrix(output, &source, parameter.shape, format.encoding())?;
        } else {
            output.push(safe(source, parameter.shape));
        }
    }
    Ok(())
}

fn add_safetensors_audio(
    config: &super::AudioConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let pinned = config.pinned();
    add_audio_parameters(
        output,
        pinned.logical_parameters().map_err(|e| e.to_string())?,
        pinned.output.bias.as_ref().map(|p| &p.id),
        None,
    )?;
    for spec in config.layers() {
        // Released relative-position matrices remain floating source tensors;
        // their selected materialization encoding is a separate storage concern.
        let relative = &spec.attention.relative_key.weight.id;
        add_audio_parameters(
            output,
            spec.logical_parameters().map_err(|e| e.to_string())?,
            None,
            Some(relative),
        )?;
    }
    Ok(())
}

fn add_audio_parameters(
    output: &mut Vec<SafetensorsTensorConstraint>,
    parameters: Vec<eredu_nn::LogicalParameterDeclaration<'_>>,
    optional: Option<&eredu_nn::ParameterId>,
    floating_source: Option<&eredu_nn::ParameterId>,
) -> Result<(), String> {
    for parameter in parameters {
        let name = parameter.parameter.id.as_str();
        if let Some(format) = parameter.linear_format {
            let encoding = if floating_source == Some(&parameter.parameter.id) {
                eredu_checkpoint::LinearFormat::Dense
            } else {
                format.encoding()
            };
            add_matrix(output, name, parameter.shape, encoding)?;
        } else {
            let mut tensor = safe(name, parameter.shape);
            if optional == Some(&parameter.parameter.id) {
                tensor = tensor.optional();
            }
            output.push(tensor);
        }
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

// Bias ownership follows the same projection schedule as the executable.
// Shared-KV consumers project only Q/O; reused values have no separate V bias.
/// Builds the strict text GGUF catalog, including shared-KV omissions and
/// split/fused routed-expert alternatives.
pub fn gguf_plan(family: &FamilyConfig) -> Result<GgufCheckpointPlan, String> {
    shared_gguf_plan(
        &family.text,
        family.pinned(),
        family.blocks(),
        family.dense_scopes(),
        family.attention_scopes(),
        family.routed_scopes(),
    )
}

pub(crate) fn shared_gguf_plan(
    args: &ModelArgs,
    declaration: &super::text::pinned::Declaration,
    block_scopes: &[super::text::specs::BlockSpec],
    dense_scopes: &[super::text::specs::DenseSpec],
    attention_scopes: &[super::text::specs::AttentionSpec],
    routed_scopes: &[Option<super::text::routed::Declaration>],
) -> Result<GgufCheckpointPlan, String> {
    declaration.validate_for(args).map_err(|e| e.to_string())?;
    let layers = args.num_hidden_layers();
    let mut common = static_gguf_slots(declaration)?
        .into_iter()
        .map(|(parameter, source)| {
            gguf(
                source,
                parameter.shape,
                if matches!(
                    parameter.role,
                    eredu_nn::LogicalParameterRole::Embedding
                        | eredu_nn::LogicalParameterRole::LinearWeight
                ) {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            )
        })
        .collect::<Vec<_>>();
    let mut groups = Vec::new();
    let mut catalog = CatalogPolicy::strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    let blocks = (0..layers)
        .map(|layer| {
            super::text::specs::DecoderBlockSpec::from_scopes(
                args,
                layer,
                block_scopes
                    .get(layer)
                    .ok_or_else(|| eredu_nn::Error::backend("missing Gemma block owner"))?
                    .in_namespace(layer, "model.layers")?,
                dense_scopes
                    .get(layer)
                    .ok_or_else(|| eredu_nn::Error::backend("missing Gemma dense owner"))?
                    .in_namespace(layer, "model.layers")?,
                attention_scopes
                    .get(layer)
                    .ok_or_else(|| eredu_nn::Error::backend("missing Gemma attention owner"))?
                    .in_namespace(layer, "model.layers")?,
                routed_scopes
                    .get(layer)
                    .ok_or_else(|| eredu_nn::Error::backend("missing Gemma routed scope"))?
                    .as_ref()
                    .map(|s| s.in_namespace(args, layer, "model.layers"))
                    .transpose()?,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    add_gguf_blocks(&blocks, &mut common, &mut groups, &mut catalog)?;
    GgufCheckpointPlan::new("Gemma 4 GGUF", common, groups, catalog)
        .map_err(|error| error.to_string())
}

pub(crate) fn add_gguf_blocks(
    blocks: &[super::text::specs::DecoderBlockSpec],
    common: &mut Vec<GgufTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<GgufTensorConstraint>>,
    catalog: &mut CatalogPolicy,
) -> Result<(), String> {
    for (layer, spec) in blocks.iter().enumerate() {
        let root = format!("blk.{layer}");
        let attention = &spec.attention;
        let dense = &spec.dense;
        for (parameter, source) in ordinary_gguf_slots(attention, dense)? {
            common.push(gguf(
                format!("{root}.{source}"),
                parameter.shape,
                if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            ));
        }
        let block = &spec.block;
        for (parameter, source) in block_gguf_slots(block)? {
            common.push(gguf(
                format!("{root}.{source}"),
                parameter.shape,
                if parameter.role == eredu_nn::LogicalParameterRole::LinearWeight {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            ));
        }
        if attention.key.is_none() {
            catalog.allowed_prefixes.extend(
                shared_kv_gguf_exclusions(layer)
                    .into_iter()
                    .map(|(source, _)| format!("{source}.")),
            );
        }
        if let Some(declaration) = &spec.routed {
            let source = declaration
                .routed
                .experts
                .packed_parameters()
                .map_err(|e| e.to_string())?;
            let fused_shape = source.gate_up().shape.clone();
            let down_shape = source.down().shape.clone();
            let split_shape = source.unfused_shape();
            for (parameter, source) in routed_gguf_common(declaration)? {
                let operation = if parameter.parameter.id == declaration.routed.selector.weight().id
                {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                };
                common.push(gguf(format!("{root}.{source}"), parameter.shape, operation));
            }
            common.push(gguf(
                format!("{root}.ffn_down_exps.weight"),
                down_shape,
                TensorOperation::Matrix,
            ));
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
                            gguf(&gate, split_shape.clone(), TensorOperation::Matrix),
                            gguf(&up, split_shape, TensorOperation::Matrix),
                        ],
                        discriminator_keys: vec![gate, up],
                    },
                    LayoutVariant {
                        id: "single fused gate and up".into(),
                        tensors: vec![gguf(&fused, fused_shape, TensorOperation::Matrix)],
                        discriminator_keys: vec![fused],
                    },
                ],
            });
        }
    }
    Ok(())
}

fn gguf(
    name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(name, shape, GgufTypeConstraint::OperationClass(operation))
}

fn static_gguf_slots(
    declaration: &super::text::pinned::Declaration,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, String)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut slots = Vec::new();
    for parameter in declaration
        .embedding()
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        slots.push((parameter, "token_embd.weight".into()));
    }
    for parameter in declaration
        .normalization()
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        slots.push((parameter, "output_norm.weight".into()));
    }
    if let Some(head) = declaration.head() {
        for parameter in head.logical_parameters().map_err(|e| e.to_string())? {
            slots.push((parameter, "output.weight".into()));
        }
    }
    if let Some(spec) = &declaration.per_layer {
        slots.extend(per_layer_gguf_slots(spec)?);
    }
    Ok(slots)
}

pub(crate) fn per_layer_gguf_slots(
    spec: &super::text::pinned::PerLayerSpec,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, String)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut slots = Vec::new();
    for parameter in spec
        .embedding
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        slots.push((parameter, "per_layer_token_embd.weight".into()));
    }
    for parameter in spec
        .projection
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        slots.push((parameter, "per_layer_model_proj.weight".into()));
    }
    for parameter in spec.norm.logical_parameters().map_err(|e| e.to_string())? {
        slots.push((parameter, "per_layer_proj_norm.weight".into()));
    }
    Ok(slots)
}

fn block_gguf_slots(
    block: &super::text::specs::BlockSpec,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, String)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut slots = Vec::new();
    for (spec, source) in [
        (&block.input, "attn_norm.weight"),
        (&block.post_attention, "post_attention_norm.weight"),
        (&block.pre_feed_forward, "ffn_norm.weight"),
        (&block.post_feed_forward, "post_ffw_norm.weight"),
    ] {
        for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
            slots.push((parameter, source.into()));
        }
    }
    for parameter in block
        .residual_scale
        .logical_parameters()
        .map_err(|e| e.to_string())?
    {
        slots.push((parameter, "layer_output_scale.weight".into()));
    }
    if let Some(spec) = &block.per_layer {
        for (slot, source) in [(&spec.gate, "inp_gate"), (&spec.projection, "proj")] {
            for parameter in slot.logical_parameters().map_err(|e| e.to_string())? {
                let suffix = match parameter.role {
                    eredu_nn::LogicalParameterRole::LinearWeight => "weight",
                    eredu_nn::LogicalParameterRole::LinearBias => "bias",
                    _ => return Err("unresolved per-layer GGUF slot".into()),
                };
                slots.push((parameter, format!("{source}.{suffix}")));
            }
        }
        for parameter in spec.norm.logical_parameters().map_err(|e| e.to_string())? {
            slots.push((parameter, "post_norm.weight".into()));
        }
    }
    Ok(slots)
}

// Published GGUFs may retain unused K/V tensors on shared consumers. These are
// source exclusions, not parameter slots, aliases of the publisher, or replicas.
fn shared_kv_gguf_exclusions(layer: usize) -> [(String, String); 3] {
    [
        ("attn_k", "k_proj"),
        ("attn_v", "v_proj"),
        ("attn_k_norm", "k_norm"),
    ]
    .map(|(source, target)| {
        (
            format!("blk.{layer}.{source}"),
            format!("model.layers.{layer}.self_attn.{target}"),
        )
    })
}

fn ordinary_gguf_slots<'a>(
    attention: &'a super::text::specs::AttentionSpec,
    dense: &'a super::text::specs::DenseSpec,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'a>, String)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut slots = Vec::new();
    for (spec, source) in [
        (Some(&attention.query), "attn_q"),
        (attention.key.as_ref(), "attn_k"),
        (attention.value.as_ref(), "attn_v"),
        (Some(&attention.output), "attn_output"),
        (Some(&dense.product.gate), "ffn_gate"),
        (Some(&dense.product.up), "ffn_up"),
        (Some(&dense.product.down), "ffn_down"),
    ] {
        if let Some(spec) = spec {
            for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
                let suffix = match parameter.role {
                    eredu_nn::LogicalParameterRole::LinearWeight => "weight",
                    eredu_nn::LogicalParameterRole::LinearBias => "bias",
                    _ => return Err("unresolved ordinary GGUF projection slot".into()),
                };
                slots.push((parameter, format!("{source}.{suffix}")));
            }
        }
    }
    for (spec, source) in [
        (Some(&attention.query_norm), "attn_q_norm"),
        (attention.key_norm.as_ref(), "attn_k_norm"),
    ] {
        if let Some(spec) = spec {
            for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
                slots.push((parameter, format!("{source}.weight")));
            }
        }
    }
    Ok(slots)
}

fn routed_gguf_common(
    declaration: &super::text::routed::Declaration,
) -> Result<Vec<(eredu_nn::LogicalParameterDeclaration<'_>, &'static str)>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let mut parameters = declaration.router_parameters().map_err(|e| e.to_string())?;
    parameters.extend(
        declaration.norms[..]
            .logical_parameters()
            .map_err(|e| e.to_string())?,
    );
    let sources = [
        "ffn_gate_inp.weight",
        "ffn_gate_inp.scale",
        "ffn_down_exps.scale",
        "post_ffw_norm_1.weight",
        "pre_ffw_norm_2.weight",
        "post_ffw_norm_2.weight",
    ];
    if parameters.len() != sources.len() {
        return Err("Gemma routed GGUF extension does not cover declared slots".into());
    }
    Ok(parameters.into_iter().zip(sources).collect())
}

fn insert_gguf_bank_bindings(
    parameters: &mut BTreeMap<String, String>,
    source: &str,
    target: &str,
) -> Result<(), String> {
    let mut entries = vec![(format!("{source}.weight"), target.to_owned())];
    for (suffix, role) in [
        ("scales", eredu_nn::LinearCompanionRole::Scale),
        ("biases", eredu_nn::LinearCompanionRole::AffineBias),
    ] {
        entries.push((
            format!("{source}.{suffix}"),
            crate::linear_format::standard_expert_companion_name(target, role),
        ));
    }
    for (source, target) in entries {
        if parameters.insert(source.clone(), target).is_some() {
            return Err(format!("conflicting Gemma GGUF bank alias {source}"));
        }
    }
    Ok(())
}

fn insert_gguf_bindings(
    parameters: &mut BTreeMap<String, String>,
    bindings: impl IntoIterator<Item = (String, String, bool)>,
) -> Result<(), String> {
    for (source, target, matrix) in bindings {
        if parameters.insert(source.clone(), target.clone()).is_some() {
            return Err(format!("conflicting Gemma GGUF alias {source}"));
        }
        if matrix {
            let source = source
                .strip_suffix(".weight")
                .ok_or("matrix source has no weight suffix")?;
            let target = target
                .strip_suffix(".weight")
                .ok_or("matrix source target has no weight suffix")?;
            for suffix in ["scales", "biases"] {
                let source = format!("{source}.{suffix}");
                if parameters
                    .insert(source.clone(), format!("{target}.{suffix}"))
                    .is_some()
                {
                    return Err(format!("conflicting Gemma GGUF alias {source}"));
                }
            }
        }
    }
    Ok(())
}

/// Prepared source aliases referencing typed ordinary and sparse construction slots.
#[derive(Debug, Clone)]
pub struct TextGgufNames {
    parameters: BTreeMap<String, String>,
    source_exclusions: BTreeMap<String, String>,
}
impl TextGgufNames {
    /// Resolves declared projection identities before source conversion or format selection.
    pub fn new(
        args: &ModelArgs,
        declaration: &super::text::pinned::Declaration,
        block_scopes: &[super::text::specs::BlockSpec],
        dense_scopes: &[super::text::specs::DenseSpec],
        attention_scopes: &[super::text::specs::AttentionSpec],
        routed_scopes: &[Option<super::text::routed::Declaration>],
    ) -> Result<Self, String> {
        declaration.validate_for(args).map_err(|e| e.to_string())?;
        let bindings = static_gguf_slots(declaration)?
            .into_iter()
            .map(|(parameter, source)| {
                let id = parameter.parameter.id.as_str();
                let target = id
                    .strip_prefix(&format!("{}.", declaration.root()))
                    .map_or_else(|| id.to_owned(), |local| format!("model.{local}"));
                (
                    source,
                    target,
                    matches!(
                        parameter.role,
                        eredu_nn::LogicalParameterRole::Embedding
                            | eredu_nn::LogicalParameterRole::LinearWeight
                    ),
                )
            });

        let bindings = bindings.collect::<Vec<_>>();
        let blocks = (0..args.num_hidden_layers())
            .map(|layer| {
                super::text::specs::DecoderBlockSpec::from_scopes(
                    args,
                    layer,
                    block_scopes
                        .get(layer)
                        .ok_or_else(|| eredu_nn::Error::backend("missing Gemma block owner"))?
                        .in_namespace(layer, "model.layers")?,
                    dense_scopes
                        .get(layer)
                        .ok_or_else(|| eredu_nn::Error::backend("missing Gemma dense owner"))?
                        .in_namespace(layer, "model.layers")?,
                    attention_scopes
                        .get(layer)
                        .ok_or_else(|| eredu_nn::Error::backend("missing Gemma attention owner"))?
                        .in_namespace(layer, "model.layers")?,
                    routed_scopes
                        .get(layer)
                        .ok_or_else(|| eredu_nn::Error::backend("missing Gemma routed scope"))?
                        .as_ref()
                        .map(|s| s.in_namespace(args, layer, "model.layers"))
                        .transpose()?,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Self::from_scopes(bindings, &blocks)
    }
    pub(crate) fn from_scopes(
        bindings: impl IntoIterator<Item = (String, String, bool)>,
        blocks: &[super::text::specs::DecoderBlockSpec],
    ) -> Result<Self, String> {
        let mut parameters = BTreeMap::new();
        let mut source_exclusions = BTreeMap::new();
        insert_gguf_bindings(&mut parameters, bindings)?;
        for (layer, spec) in blocks.iter().enumerate() {
            let attention = &spec.attention;
            if attention.key.is_none() {
                source_exclusions.extend(shared_kv_gguf_exclusions(layer));
            }
            let dense = &spec.dense;
            let block = &spec.block;
            let mut bindings = ordinary_gguf_slots(attention, dense)?
                .into_iter()
                .chain(block_gguf_slots(block)?)
                .map(|(p, source)| {
                    (
                        format!("blk.{layer}.{source}"),
                        p.parameter.id.as_str().to_owned(),
                        p.role == eredu_nn::LogicalParameterRole::LinearWeight,
                    )
                })
                .collect::<Vec<_>>();
            if let Some(declaration) = &spec.routed {
                for (parameter, source) in routed_gguf_common(declaration)? {
                    bindings.push((
                        format!("blk.{layer}.{source}"),
                        parameter.parameter.id.as_str().to_owned(),
                        parameter.parameter.id == declaration.routed.selector.weight().id,
                    ));
                }
                let source = declaration
                    .routed
                    .experts
                    .packed_parameters()
                    .map_err(|e| e.to_string())?;
                let gate_up = source.gate_up();
                let down = source.down();
                let root = gate_up
                    .parameter
                    .id
                    .as_str()
                    .strip_suffix(".gate_up_proj")
                    .ok_or("unresolved Gemma expert source root")?;
                for (source, target) in [
                    ("ffn_gate_up_exps", gate_up.parameter.id.as_str()),
                    ("ffn_down_exps", down.parameter.id.as_str()),
                ] {
                    insert_gguf_bank_bindings(
                        &mut parameters,
                        &format!("blk.{layer}.{source}"),
                        target,
                    )?;
                }
                for (source, target) in [
                    ("ffn_gate_exps", format!("{root}.gate_proj.weight")),
                    ("ffn_up_exps", format!("{root}.up_proj.weight")),
                ] {
                    bindings.push((format!("blk.{layer}.{source}.weight"), target, true));
                }
            }
            insert_gguf_bindings(&mut parameters, bindings)?;
        }
        Ok(Self {
            parameters,
            source_exclusions,
        })
    }
    /// Translates the standalone text source view used by assistant and decoder recipes.
    pub fn translate(&self, name: &str) -> String {
        if let Some(target) = self.parameters.get(name) {
            return target.clone();
        }
        for (source, target) in &self.source_exclusions {
            if name == source || name.starts_with(&format!("{source}.")) {
                return name.replacen(source, target, 1);
            }
        }
        name.into()
    }
    /// Projects source identities into the complete multimodal family's parameter tree.
    pub fn translate_family(&self, name: &str) -> String {
        let name = self.translate(name);
        match name.strip_prefix("model.") {
            Some(local) => format!("model.language_model.{local}"),
            None => name,
        }
    }
}

/// Resolves the released separate gate/up or already-fused sparse bank into
/// the canonical neutral expert slots used by the ordinary Gemma block.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let declaration = config
        .routed(layer)
        .map_err(|e| e.to_string())?
        .ok_or("layer has no routed source")?;
    expert_recipes_for_declaration(catalog, declaration)
}

fn expert_recipes_for_declaration<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::text::routed::Declaration,
) -> Result<GatedProductExpertRecipes, String> {
    let source = |base: &str| {
        if catalog.tensor_metadata(base).is_ok() {
            base.to_owned()
        } else {
            format!("{base}.weight")
        }
    };
    let names = expert_layout_names(declaration, source)?;
    resolve_gated_product_expert_recipes(catalog, &names).map_err(|error| error.to_string())
}

pub(super) fn expert_layout_names(
    declaration: &super::text::routed::Declaration,
    resolve_source: impl Fn(&str) -> String,
) -> Result<GatedProductExpertLayoutNames, String> {
    let source = declaration
        .routed
        .experts
        .packed_parameters()
        .map_err(|e| e.to_string())?;
    let gate_up = source.gate_up();
    let down = source.down();
    let prefix = gate_up
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved Gemma expert root")?;
    Ok(GatedProductExpertLayoutNames {
        target_gate_up: gate_up.parameter.id.as_str().to_owned(),
        target_down: down.parameter.id.as_str().to_owned(),
        packed_gate_up: resolve_source(&format!("{prefix}.gate_up_proj")),
        packed_down: resolve_source(&format!("{prefix}.down_proj")),
        separate_gate: resolve_source(&format!("{prefix}.gate_proj")),
        separate_up: resolve_source(&format!("{prefix}.up_proj")),
        separate_down: resolve_source(&format!("{prefix}.down_proj")),
        independent: Vec::new(),
    })
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
    let resolved = expert_recipes(catalog, family, layer)?;
    eredu_checkpoint::expert::gated_product_materialization_recipes(catalog, &resolved)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &FamilyConfig,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let experts = config
        .routed_layers()
        .next()
        .map(|(_, scope)| scope)
        .ok_or("Gemma 4 expert residency requires a sparse invocation")?
        .routed
        .experts
        .group_count() as usize;
    let owner_group = eredu_runtime::ExecutionGroupId::new(super::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let banks = config.routed_layers().map(|(layer, declaration)| {
        expert_recipes_for_declaration(catalog, declaration)
            .map(|recipes| (layer, released_layer_path(layer), recipes))
    });
    crate::expert_residency::gated_product_residency_catalog(
        catalog,
        owner_group,
        0,
        experts,
        banks,
    )
}
