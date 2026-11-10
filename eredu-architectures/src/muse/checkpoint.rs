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

use super::DecoderConfig;

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
    let canonical = keys
        .iter()
        .map(|key| canonical_safetensors_target(key))
        .collect::<Vec<_>>();
    let target = args
        .with_media(
            args.vision_config.clone().filter(|_| {
                canonical
                    .iter()
                    .any(|key| key.starts_with("model.vision_tower."))
            }),
            args.audio_config.clone().filter(|_| {
                canonical
                    .iter()
                    .any(|key| key.starts_with("model.audio_tower."))
            }),
        )
        .map_err(|e| e.to_string())?;
    let quantization = match (
        source_config.get("quantization"),
        source_config.get("quantization_config"),
    ) {
        (Some(first), Some(second)) if first != second => {
            return Err("Muse root quantization declarations disagree".into());
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
    let target = args
        .with_formats(Some(quantization), None, None)
        .map_err(|e| e.to_string())?;
    let mut vision = target.vision_config.clone();
    let mut audio = target.audio_config.clone();
    if let Some(vision) = &mut vision {
        use eredu_nn::DeclareLogicalParameters;
        let declarations = vision.declaration();
        let group = usize::try_from(quantization.group_size())
            .map_err(|_| "negative quantization group size")?;
        let formats = declarations
            .logical_parameters()
            .map_err(|error| error.to_string())?
            .into_iter()
            .filter(|parameter| {
                parameter.linear_format.is_some()
                    && parameter.shape[1].is_multiple_of(group)
                    && parameter.shape[1].is_multiple_of(32)
            })
            .map(|parameter| (parameter.parameter.id.as_str().to_owned(), quantization))
            .collect();
        *vision = vision
            .with_formats(formats, None)
            .map_err(|e| e.to_string())?;
    }
    if let Some(audio) = &mut audio {
        let group = usize::try_from(quantization.group_size())
            .map_err(|_| "negative quantization group size")?;
        let declarations = audio.declaration();
        let formats = declarations
            .units
            .iter()
            .flat_map(|(_, spec)| spec.projections())
            .filter(|spec| {
                (spec.input as usize).is_multiple_of(group)
                    && (spec.input as usize).is_multiple_of(32)
            })
            .map(|spec| (spec.weight.id.as_str().to_owned(), quantization.into()))
            .collect();
        *audio = audio.with_formats(formats).map_err(|e| e.to_string())?;
    }
    target.with_media(vision, audio).map_err(|e| e.to_string())
}

/// Projects checkpoint encodings onto retained text/media owners and source layouts.
pub fn with_checkpoint_formats(
    args: &DecoderConfig,
    formats: HashMap<String, WeightQuantization>,
) -> Result<DecoderConfig, String> {
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
    let target = args
        .with_formats(None, None, (!text.is_empty()).then_some(text))
        .map_err(|e| e.to_string())?;
    let mut vision_config = target.vision_config.clone();
    let mut audio_config = target.audio_config.clone();
    if let Some(config) = vision_config.as_mut() {
        *config = config
            .with_formats(vision, None)
            .map_err(|e| e.to_string())?;
    } else if !vision.is_empty() {
        return Err("vision checkpoint formats have no selected vision tower".into());
    }
    if let Some(config) = audio_config.as_mut() {
        *config = config.with_formats(audio).map_err(|e| e.to_string())?;
    } else if !audio.is_empty() {
        return Err("audio checkpoint formats have no selected waveform tower".into());
    }
    target
        .with_media(vision_config, audio_config)
        .map_err(|e| e.to_string())
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

/// Projects source matrices from construction slots and explicit storage extensions.
pub(crate) fn source_linear_shapes(
    args: &DecoderConfig,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    use eredu_nn::DeclareLogicalParameters;
    args.validate().map_err(|error| error.to_string())?;
    let mut shapes = BTreeMap::new();
    let mut insert = |name: String, shape: Vec<usize>| -> Result<(), String> {
        if let Some(previous) = shapes.insert(name.clone(), shape.clone()) {
            if previous != shape {
                return Err(format!("conflicting Muse source geometry for {name}"));
            }
        }
        Ok(())
    };
    if args.weight_convention == super::WeightConvention::Gguf {
        let mut tensors = static_gguf_layouts(args)?;
        for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
            tensors.extend(attention_gguf_layouts(args, layer)?);
            tensors.extend(if args.is_moe() {
                routed_gguf_layouts(args, layer)?
            } else {
                dense_gguf_layouts(args, layer)?
            });
        }
        for tensor in tensors {
            if tensor.shape.len() == 2 && tensor.gguf != "token_embd.weight" {
                insert(tensor.canonical, tensor.shape)?;
            }
        }
        if let Some(vision) = &args.vision_config {
            // Validate the released image-only storage contract, including the
            // position table and adapter widths. Flatten the declared spatial
            // kernel without the temporal folding used by SafeTensors.
            for tensor in vision_gguf_layouts(vision)? {
                if tensor.shape.len() == 2 {
                    insert(tensor.canonical, tensor.shape)?;
                }
            }
            let statics = &vision.declaration().statics;
            let spatial = statics
                .spatial_source()
                .map_err(|error| error.to_string())?;
            insert(
                spatial.weight.id.as_str().into(),
                spatial.matrix_shape.to_vec(),
            )?;
            if let Some(audio) = &args.audio_config {
                for tensor in audio_gguf_layouts(audio)? {
                    if tensor.shape.len() == 2 {
                        insert(tensor.canonical, tensor.shape)?;
                    }
                }
            }
        }
        return Ok(shapes);
    }
    let construction = args.pinned();
    let embedding = construction.embedding_spec().weight.id.as_str();
    let mut visit = |spec: &dyn DeclareLogicalParameters| -> Result<(), String> {
        for parameter in spec
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            if parameter.shape.len() == 2
                && parameter.linear_format.is_some()
                && parameter.parameter.id.as_str() != embedding
            {
                insert(parameter.parameter.id.as_str().into(), parameter.shape)?;
            }
        }
        Ok(())
    };
    visit(construction)?;
    for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
        let attention = args
            .attention(layer)
            .map(|spec| &spec.parameters)
            .map_err(|error| error.to_string())?;
        for projection in attention.projections() {
            visit(projection)?;
        }
        if args.is_moe() {
            let declaration = args.routed(layer).map_err(|error| error.to_string())?;
            visit(declaration)?;
        } else {
            let dense = args.dense(layer).map_err(|error| error.to_string())?;
            visit(dense)?;
        }
    }
    if let Some(vision) = &args.vision_config {
        visit(vision.declaration())?;
    }
    if let Some(audio) = &args.audio_config {
        let declarations = audio.declaration();
        for (_, unit) in &declarations.units {
            visit(unit)?;
        }
    }
    if args.is_moe() {
        for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
            visit_expert_source_matrices(args, layer, |source| {
                if source.layout == ExpertSourceLayout::Independent {
                    insert(source.canonical, source.shape)?;
                }
                Ok(())
            })?;
        }
    }
    Ok(shapes)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum ExpertSourceLayout {
    Split,
    Independent,
}

/// A physical layout projection; ownership and geometry originate in the packed bank.
struct ExpertMatrixSource {
    layout: ExpertSourceLayout,
    source: String,
    canonical: String,
    owner: eredu_nn::ParameterId,
    shape: Vec<usize>,
}
fn visit_expert_source_matrices(
    args: &DecoderConfig,
    layer: usize,
    mut visit: impl FnMut(ExpertMatrixSource) -> Result<(), String>,
) -> Result<(), String> {
    let declaration = args.routed(layer).map_err(|e| e.to_string())?;
    let bank = declaration
        .bank()
        .packed_parameters()
        .map_err(|e| e.to_string())?;
    let (root, _, _) = declared_expert_targets(&bank)?;
    let released = format!("model.language_model.layers.{layer}.mlp.experts");
    for (field, owner, shape) in [
        ("gate_proj", bank.gate_up(), bank.unfused_shape()),
        ("up_proj", bank.gate_up(), bank.unfused_shape()),
        ("down_proj", bank.down(), bank.down().shape.clone()),
    ] {
        visit(ExpertMatrixSource {
            layout: ExpertSourceLayout::Split,
            source: format!("{released}.{field}"),
            canonical: format!("{root}.{field}"),
            owner: owner.parameter.id.clone(),
            shape: shape.clone(),
        })?;
        if args.weight_convention == super::WeightConvention::HuggingFace {
            for expert in 0..bank.group_count() {
                visit(ExpertMatrixSource {
                    layout: ExpertSourceLayout::Independent,
                    source: format!("{released}.{expert}.{field}.weight"),
                    canonical: format!("{root}.{expert}.{field}.weight"),
                    owner: owner.parameter.id.clone(),
                    shape: shape[1..].to_vec(),
                })?;
            }
        }
    }
    Ok(())
}

/// Derives executable bank formats while retaining the physical source encodings.
/// Encoded members of a fused bank must have one homogeneous representation.
pub(crate) fn normalize_expert_formats(
    args: &DecoderConfig,
    default: Option<WeightQuantization>,
    selected: Option<&std::collections::HashSet<String>>,
    formats: &mut Option<HashMap<String, WeightQuantization>>,
) -> Result<(), String> {
    let names = || {
        formats
            .iter()
            .flat_map(|formats| formats.keys())
            .chain(selected.into_iter().flatten())
    };
    if names().next().is_none() {
        return Ok(());
    }
    let owned = eredu_nn::DeclareLogicalParameters::logical_parameters(args)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|p| p.parameter.id.as_str())
        .collect::<BTreeSet<_>>();
    if names().all(|name| owned.contains(name.as_str())) {
        return Ok(());
    }
    for layer in args.routed_layers().keys().copied() {
        let mut groups = BTreeMap::<_, Vec<String>>::new();
        visit_expert_source_matrices(args, layer, |source| {
            if source.canonical != source.owner.as_str() {
                groups
                    .entry((source.layout, source.owner))
                    .or_default()
                    .push(source.canonical);
            }
            Ok(())
        })?;
        for ((_, owner), names) in groups {
            if !names.iter().any(|name| {
                formats.as_ref().is_some_and(|f| f.contains_key(name))
                    || selected.is_some_and(|names| names.contains(name))
            }) {
                continue;
            }
            let requested = names
                .iter()
                .map(|name| {
                    formats
                        .as_ref()
                        .and_then(|f| f.get(name))
                        .copied()
                        .or_else(|| {
                            default.filter(|_| selected.is_none_or(|names| names.contains(name)))
                        })
                })
                .collect::<Vec<_>>();
            let Some(format) = requested.iter().copied().flatten().next() else {
                continue;
            };
            if requested.iter().any(|candidate| *candidate != Some(format)) {
                return Err(format!(
                    "Muse expert source encodings are incomplete or heterogeneous for {owner}"
                ));
            }
            format.validate().map_err(|e| e.to_string())?;
            let formats = formats.get_or_insert_default();
            if formats
                .get(owner.as_str())
                .is_some_and(|old| *old != format)
            {
                return Err(format!(
                    "Muse fused and source expert encodings disagree for {owner}"
                ));
            }
            formats.insert(owner.to_string(), format);
            for name in names {
                formats.insert(name, format);
            }
        }
    }
    Ok(())
}

pub(crate) fn expert_source_formats(
    args: &DecoderConfig,
) -> Result<Vec<(String, eredu_checkpoint::LinearFormat)>, String> {
    let mut references = Vec::new();
    for layer in args.routed_layers().keys().copied() {
        visit_expert_source_matrices(args, layer, |source| {
            let format = args.linear_format_for(&source.canonical);
            if args.weight_convention == super::WeightConvention::HuggingFace {
                matrix_for_linear_format(
                    &source.canonical,
                    Vec::<String>::new(),
                    source.shape,
                    format,
                    None,
                )
                .map_err(|e| e.to_string())?;
            }
            references.push((source.canonical, format));
            Ok(())
        })?;
    }
    Ok(references)
}

/// Builds exact SafeTensors requirements and explicit released layout alternatives.
pub fn safetensors_plan(args: &DecoderConfig) -> Result<SafetensorsCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    let hidden = dimension(args.hidden_size, "text hidden size")?;
    let layers = dimension(args.num_hidden_layers, "text layer count")?;
    let mut common = Vec::new();
    let mut alternatives = Vec::new();
    use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
    let construction = args.pinned();
    for parameter in construction
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        let canonical = parameter.parameter.id.as_str();
        let source = canonical.strip_prefix("model.").map_or_else(
            || canonical.to_owned(),
            |local| format!("model.language_model.{local}"),
        );
        if matches!(
            parameter.role,
            LogicalParameterRole::Embedding | LogicalParameterRole::LinearWeight
        ) {
            add_text_matrix(args, &mut common, &source, canonical, parameter.shape)?;
        } else {
            common.push(safe_alias(source, canonical, parameter.shape));
        }
    }
    for layer in 0..layers {
        let released = format!("model.language_model.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        let scalar = args.scalar(layer).map_err(|error| error.to_string())?;
        for parameter in scalar
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let name = parameter.parameter.id.as_str();
            let local = name
                .strip_prefix(&format!("{canonical}."))
                .ok_or("scalar source extension references another unit")?;
            common.push(safe_alias(
                format!("{released}.{local}"),
                name,
                parameter.shape,
            ));
        }
        use eredu_nn::DeclareLogicalParameters;
        let attention = args
            .attention(layer)
            .map(|spec| &spec.parameters)
            .map_err(|error| error.to_string())?;
        for projection in attention.projections() {
            for parameter in projection
                .logical_parameters()
                .map_err(|error| error.to_string())?
            {
                let name = parameter.parameter.id.as_str();
                let local = name
                    .strip_prefix(&format!("{canonical}."))
                    .ok_or("attention source extension references another unit")?;
                add_text_matrix(
                    args,
                    &mut common,
                    &format!("{released}.{local}"),
                    name,
                    parameter.shape,
                )?;
            }
        }
        if args.is_moe() {
            add_moe_layout(
                args,
                layer,
                &released,
                &canonical,
                &mut common,
                &mut alternatives,
            )?;
        } else {
            let projections = args.dense(layer).map_err(|error| error.to_string())?;
            for projection in projections.projections() {
                for parameter in projection
                    .logical_parameters()
                    .map_err(|error| error.to_string())?
                {
                    let name = parameter.parameter.id.as_str();
                    let local = name
                        .strip_prefix(&format!("{canonical}."))
                        .ok_or("dense source extension references another unit")?;
                    add_text_matrix(
                        args,
                        &mut common,
                        &format!("{released}.{local}"),
                        name,
                        parameter.shape,
                    )?;
                }
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
/// decoder is group one, and waveform units are group two. Selection uses no native topology.
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
                "Muse-Glimmer recipe group {group} is outside three groups"
            ));
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
    let audio_owners = args
        .audio_config
        .as_ref()
        .map(|audio| audio.declaration().owners())
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    for (target, recipe) in recipes {
        let owner = if target.starts_with("model.audio_tower.") {
            let index = audio_owners
                .get(&target)
                .copied()
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
    let declaration = args.routed(layer).map_err(|error| error.to_string())?;
    let source = declaration
        .bank()
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let (target, target_gate_up, target_down) = declared_expert_targets(&source)?;
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
    let independent = (0..source.group_count())
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
            target_gate_up,
            target_down,
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
    common: &mut Vec<SafetensorsTensorConstraint>,
    alternatives: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    use eredu_nn::DeclareLogicalParameters;
    let declaration = args.routed(layer).map_err(|error| error.to_string())?;
    let add_parameter = |parameter: eredu_nn::LogicalParameterDeclaration<'_>,
                         matrix: bool,
                         common: &mut Vec<SafetensorsTensorConstraint>|
     -> Result<(), String> {
        let name = parameter.parameter.id.as_str();
        let local = name
            .strip_prefix(&format!("{canonical}."))
            .ok_or("routed source extension references another unit")?;
        let source = format!("{released}.{local}");
        if matrix {
            add_text_matrix(args, common, &source, name, parameter.shape)
        } else {
            common.push(safe_alias(source, name, parameter.shape));
            Ok(())
        }
    };
    for parameter in declaration
        .selector()
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        let matrix = parameter.parameter.id == declaration.selector().weight().id;
        add_parameter(parameter, matrix, common)?;
    }
    if let Some(spec) = declaration.compression() {
        for parameter in spec
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let matrix = parameter.parameter.id == spec.input.weight.id
                || parameter.parameter.id == spec.output.weight.id;
            add_parameter(parameter, matrix, common)?;
        }
    }
    let source = declaration
        .bank()
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    if source.gate_up_bias().is_some() || source.down_bias().is_some() {
        return Err("Muse expert storage extension does not cover bank bias slots".into());
    }
    let released_root = format!("{released}.mlp.experts");
    let packed_gate_up = format!("{released_root}.gate_up_proj");
    let packed_down = format!("{released_root}.down_proj");
    let mut packed = Vec::new();
    add_packed_expert_matrix(&mut packed, &packed_gate_up, source.gate_up())?;
    add_packed_expert_matrix(&mut packed, &packed_down, source.down())?;
    let split_gate = format!("{released_root}.gate_proj");
    let split_up = format!("{released_root}.up_proj");
    let mut split = Vec::new();
    let mut independent = Vec::new();
    let mut independent_keys = Vec::new();
    visit_expert_source_matrices(args, layer, |source| {
        let output = match source.layout {
            ExpertSourceLayout::Split => &mut split,
            ExpertSourceLayout::Independent => {
                independent_keys.push(source.source.clone());
                &mut independent
            }
        };
        // Independent slices keep their published source aliases; their format
        // reference is canonical, without introducing a logical parameter owner.
        let aliases = safetensors_aliases(
            &source.source,
            match source.layout {
                ExpertSourceLayout::Split => &source.canonical,
                ExpertSourceLayout::Independent => &source.source,
            },
        );
        output.extend(
            matrix_for_linear_format(
                &source.source,
                aliases,
                source.shape,
                args.linear_format_for(&source.canonical),
                None,
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(())
    })?;
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

fn add_audio_matrix(
    format: eredu_checkpoint::LinearFormat,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(name, std::iter::empty::<String>(), shape, format, None)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

struct DeclaredGgufTensor {
    canonical: String,
    gguf: String,
    shape: Vec<usize>,
    operation: TensorOperation,
}

fn audio_gguf_layouts(audio: &super::AudioConfig) -> Result<Vec<DeclaredGgufTensor>, String> {
    use super::audio::{specs::UnitSpec, AudioUnitKind};
    use eredu_nn::DeclareLogicalParameters;
    let declarations = audio.declaration();
    let mut tensors = Vec::new();
    // Storage aliases reference typed construction slots; they do not supply geometry.
    let mut bind = |spec: &dyn DeclareLogicalParameters,
                    sources: Vec<(String, TensorOperation)>|
     -> Result<(), String> {
        let parameters = spec
            .logical_parameters()
            .map_err(|error| error.to_string())?;
        if parameters.len() != sources.len() {
            return Err("Muse audio storage extension does not cover its declared slots".into());
        }
        for (parameter, (gguf, operation)) in parameters.into_iter().zip(sources) {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf,
                shape: parameter.shape,
                operation,
            });
        }
        Ok(())
    };
    let mut flat_block = 0usize;
    for (kind, unit) in &declarations.units {
        match (kind, unit) {
            (AudioUnitKind::Patch { stage }, UnitSpec::Patch(spec)) => {
                bind(
                    &spec.projection,
                    vec![
                        (format!("a.patch.{stage}.weight"), TensorOperation::Matrix),
                        (format!("a.patch.{stage}.bias"), TensorOperation::Vector),
                    ],
                )?;
                bind(
                    &spec.norm,
                    vec![(
                        format!("a.patch.{stage}.norm.weight"),
                        TensorOperation::Vector,
                    )],
                )?;
            }
            (AudioUnitKind::Convolution { .. }, UnitSpec::Convolution(spec)) => {
                bind(
                    &spec.convolution,
                    vec![
                        (
                            format!("a.conv1d.{flat_block}.weight"),
                            TensorOperation::Dense,
                        ),
                        (
                            format!("a.conv1d.{flat_block}.bias"),
                            TensorOperation::Vector,
                        ),
                    ],
                )?;
                bind(
                    &spec.norm,
                    vec![(
                        format!("a.conv1d.{flat_block}.norm.weight"),
                        TensorOperation::Vector,
                    )],
                )?;
                for (projection, name) in [
                    (&spec.network.up, "ffn_up"),
                    (&spec.network.down, "ffn_down"),
                ] {
                    bind(
                        projection,
                        vec![
                            (
                                format!("a.blk.{flat_block}.{name}.weight"),
                                TensorOperation::Matrix,
                            ),
                            (
                                format!("a.blk.{flat_block}.{name}.bias"),
                                TensorOperation::Vector,
                            ),
                        ],
                    )?;
                }
                flat_block = flat_block
                    .checked_add(1)
                    .ok_or("audio flat block index overflowed")?;
            }
            (AudioUnitKind::Projection, UnitSpec::Projection(spec)) => {
                bind(
                    &spec.norm,
                    vec![("a.pre_ln.weight".into(), TensorOperation::Vector)],
                )?;
                bind(
                    &spec.projection,
                    vec![
                        ("mm.a.fc.weight".into(), TensorOperation::Matrix),
                        ("mm.a.fc.bias".into(), TensorOperation::Vector),
                    ],
                )?;
                bind(
                    &spec.post_norm,
                    vec![("a.post_ln.weight".into(), TensorOperation::Vector)],
                )?;
            }
            _ => return Err("Muse audio storage extension references an incompatible unit".into()),
        }
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
    use eredu_nn::DeclareLogicalParameters;
    let declarations = audio.declaration();
    for (_, unit) in &declarations.units {
        for parameter in unit
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let name = parameter.parameter.id.as_str();
            if let Some(format) = parameter.linear_format {
                add_audio_matrix(format.encoding(), output, name, parameter.shape)?;
            } else {
                output.push(safe(name, parameter.shape));
            }
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
    use eredu_nn::DeclareLogicalParameters;
    let declarations = vision.declaration();
    if vision.language_hidden_size as usize != text_hidden {
        return Err("Muse vision projection differs from the decoder width".into());
    }
    for parameter in declarations
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        let name = parameter.parameter.id.as_str();
        if let Some(format) = parameter.linear_format {
            add_vision_matrix(format.encoding(), output, name, parameter.shape)?;
        } else {
            output.push(safe(name, parameter.shape));
        }
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
    output: &mut Vec<SafetensorsTensorConstraint>,
    source: &str,
    parameter: &eredu_nn::LogicalParameterDeclaration<'_>,
) -> Result<(), String> {
    let canonical = parameter.parameter.id.as_str();
    let shape = parameter.shape.clone();
    let format = parameter
        .linear_format
        .ok_or("Muse packed matrix has no declared encoding")?;
    if format.encoding() != eredu_checkpoint::LinearFormat::NvFp4 {
        output.extend(
            matrix_for_linear_format(
                source,
                safetensors_aliases(source, canonical),
                shape,
                format.encoding(),
                None,
            )
            .map_err(|error| error.to_string())?,
        );
        return Ok(());
    }
    let parts = format.row_layout().partitions();
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
    format: eredu_checkpoint::LinearFormat,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(
        matrix_for_linear_format(name, safetensors_aliases(name, name), shape, format, None)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn attention_gguf_layouts(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<DeclaredGgufTensor>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let attention = &args
        .attention(layer)
        .map_err(|error| error.to_string())?
        .parameters;
    let mut tensors = Vec::new();
    for (projection, source) in attention.projections().into_iter().zip([
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_gate",
    ]) {
        for parameter in projection
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf: format!("blk.{layer}.{source}.weight"),
                shape: parameter.shape,
                operation: TensorOperation::Matrix,
            });
        }
    }
    for (norm, source) in attention.norms.iter().zip(["attn_q_norm", "attn_k_norm"]) {
        for parameter in norm
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf: format!("blk.{layer}.{source}.weight"),
                shape: parameter.shape,
                operation: TensorOperation::Vector,
            });
        }
    }
    Ok(tensors)
}

fn dense_gguf_layouts(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<DeclaredGgufTensor>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let projections = args.dense(layer).map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    for (projection, source) in projections
        .projections()
        .into_iter()
        .zip(["ffn_gate", "ffn_up", "ffn_down"])
    {
        for parameter in projection
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf: format!("blk.{layer}.{source}.weight"),
                shape: parameter.shape,
                operation: TensorOperation::Matrix,
            });
        }
    }
    Ok(tensors)
}

fn static_gguf_layouts(args: &DecoderConfig) -> Result<Vec<DeclaredGgufTensor>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let construction = args.pinned();
    let embedding = construction.embedding_spec();
    let head = construction.head_spec();
    construction
        .logical_parameters()
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|parameter| {
            let id = parameter.parameter.id.as_str();
            let (source, operation) = if id == embedding.weight.id.as_str() {
                ("token_embd.weight", TensorOperation::Matrix)
            } else if head.filter(|head| head.weight.id.as_str() == id).is_some() {
                ("output.weight", TensorOperation::Matrix)
            } else if parameter.role == eredu_nn::LogicalParameterRole::Normalization {
                ("output_norm.weight", TensorOperation::Vector)
            } else {
                return Err(format!("unresolved Muse static GGUF slot {id}"));
            };
            Ok(DeclaredGgufTensor {
                canonical: id.into(),
                gguf: source.into(),
                shape: parameter.shape,
                operation,
            })
        })
        .collect()
}
fn scalar_gguf_layouts(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<DeclaredGgufTensor>, String> {
    use super::equations::specs::ResidualSpec;
    use eredu_nn::DeclareLogicalParameters;
    let scalar = args.scalar(layer).map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    let mut bind = |spec: &dyn DeclareLogicalParameters, source: &str| -> Result<(), String> {
        for parameter in spec
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf: format!("blk.{layer}.{source}.weight"),
                shape: parameter.shape,
                operation: TensorOperation::Vector,
            });
        }
        Ok(())
    };
    for (spec, source) in [
        (&scalar.input as &dyn DeclareLogicalParameters, "attn_norm"),
        (&scalar.post_attention, "post_attention_norm"),
        (&scalar.pre_feed_forward, "ffn_norm"),
        (&scalar.post_feed_forward, "post_ffw_norm"),
    ] {
        bind(spec, source)?;
    }
    for (spec, source) in scalar.residuals.iter().zip(["attn_resid", "ffn_resid"]) {
        match spec {
            ResidualSpec::Add => {}
            ResidualSpec::Learned(_) => {
                return Err(
                    "Muse GGUF residual gates require stored coefficient normalization".into(),
                );
            }
            ResidualSpec::Stored(spec) => {
                let parameters = spec
                    .logical_parameters()
                    .map_err(|error| error.to_string())?;
                for (parameter, suffix) in parameters.into_iter().zip(["alpha", "beta"]) {
                    tensors.push(DeclaredGgufTensor {
                        canonical: parameter.parameter.id.as_str().to_owned(),
                        gguf: format!("blk.{layer}.{source}_{suffix}.weight"),
                        shape: parameter.shape,
                        operation: TensorOperation::Vector,
                    });
                }
            }
        }
    }
    Ok(tensors)
}

fn routed_gguf_layouts(
    args: &DecoderConfig,
    layer: usize,
) -> Result<Vec<DeclaredGgufTensor>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let declaration = args.routed(layer).map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    let mut bind = |spec: &dyn DeclareLogicalParameters,
                    sources: &[&str],
                    matrix: bool|
     -> Result<(), String> {
        let parameters = spec
            .logical_parameters()
            .map_err(|error| error.to_string())?;
        if parameters.len() != sources.len() {
            return Err("Muse routed GGUF extension does not cover its slots".into());
        }
        for (parameter, source) in parameters.into_iter().zip(sources) {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf: format!("blk.{layer}.{source}"),
                shape: parameter.shape,
                operation: if matrix {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            });
        }
        Ok(())
    };
    if let Some(compression) = declaration.compression() {
        bind(&compression.input, &["ffn_latent_down.weight"], true)?;
        bind(&compression.output, &["ffn_latent_up.weight"], true)?;
        bind(&compression.norms[0], &["ffn_pre_exp_norm.weight"], false)?;
        bind(&compression.norms[1], &["ffn_post_exp_norm.weight"], false)?;
    }
    for parameter in declaration
        .selector()
        .logical_parameters()
        .map_err(|error| error.to_string())?
    {
        let (source, operation) = if parameter.parameter.id == declaration.selector().weight().id {
            ("ffn_gate_inp.weight", TensorOperation::Matrix)
        } else if Some(&parameter.parameter.id)
            == declaration.selector().correction_bias().map(|p| &p.id)
        {
            ("exp_probs_b.bias", TensorOperation::Vector)
        } else {
            return Err("unresolved Muse selector GGUF slot".into());
        };
        tensors.push(DeclaredGgufTensor {
            canonical: parameter.parameter.id.as_str().to_owned(),
            gguf: format!("blk.{layer}.{source}"),
            shape: parameter.shape,
            operation,
        });
    }
    let source = declaration
        .bank()
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let root = source
        .gate_up()
        .parameter
        .id
        .as_str()
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved Muse packed bank")?;
    let split = source.unfused_shape();
    let down_shape = source.down().shape.clone();
    // Split sources are checkpoint inputs, not additional parameter owners.
    for (source, local, shape) in [
        ("ffn_gate_exps", "gate_proj", split.clone()),
        ("ffn_up_exps", "up_proj", split),
        ("ffn_down_exps", "down_proj", down_shape),
    ] {
        tensors.push(DeclaredGgufTensor {
            canonical: format!("{root}.{local}.weight"),
            gguf: format!("blk.{layer}.{source}.weight"),
            shape,
            operation: TensorOperation::Matrix,
        });
    }
    Ok(tensors)
}

// Source conversion precedes retained execution-format selection. Matrix outputs
// can therefore include affine or MXFP4 companions even with a dense execution spec.
fn gguf_source_bindings(tensor: DeclaredGgufTensor) -> Result<Vec<(String, String)>, String> {
    let mut bindings = vec![(tensor.gguf.clone(), tensor.canonical.clone())];
    if tensor.operation == TensorOperation::Matrix {
        let source_prefix = tensor
            .gguf
            .strip_suffix(".weight")
            .ok_or("GGUF matrix has no weight suffix")?;
        let source_format = crate::linear_format::standard_linear_format(
            &tensor.canonical,
            eredu_checkpoint::LinearFormat::Affine(
                eredu_checkpoint::AffineQuantization::new(32, 4)
                    .map_err(|error| error.to_string())?,
            ),
        )
        .map_err(|error| error.to_string())?;
        for (suffix, parameter) in [
            ("scales", source_format.scale()),
            ("biases", source_format.affine_bias()),
        ] {
            if let Some(parameter) = parameter {
                bindings.push((
                    format!("{source_prefix}.{suffix}"),
                    parameter.id.as_str().to_owned(),
                ));
            }
        }
    }
    Ok(bindings)
}

/// Builds the complete text GGUF catalog admitted for Muse-Glimmer.
pub fn gguf_plan(args: &DecoderConfig) -> Result<GgufCheckpointPlan, String> {
    if args.weight_convention != super::WeightConvention::Gguf {
        return Err("Muse GGUF admission requires GGUF-normalized source declarations".into());
    }
    let layers = dimension(args.num_hidden_layers, "text layer count")?;
    let mut tensors = static_gguf_layouts(args)?
        .into_iter()
        .map(|tensor| gguf(tensor.gguf, tensor.shape, tensor.operation))
        .collect::<Vec<_>>();
    for layer in 0..layers {
        for tensor in scalar_gguf_layouts(args, layer)?
            .into_iter()
            .chain(attention_gguf_layouts(args, layer)?)
        {
            tensors.push(gguf(tensor.gguf, tensor.shape, tensor.operation));
        }
        if args.is_moe() {
            for tensor in routed_gguf_layouts(args, layer)? {
                tensors.push(gguf(tensor.gguf, tensor.shape, tensor.operation));
            }
        } else {
            for tensor in dense_gguf_layouts(args, layer)? {
                tensors.push(gguf(tensor.gguf, tensor.shape, tensor.operation));
            }
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

// Image-only storage extensions reference the ordinary construction slots.
fn vision_gguf_layouts(vision: &super::VisionConfig) -> Result<Vec<DeclaredGgufTensor>, String> {
    use eredu_nn::DeclareLogicalParameters;
    let declarations = vision.declaration();
    let statics = &declarations.statics;
    if statics.position.vocabulary != 1024 || statics.adapter.up.output != 4096 {
        return Err(
            "published Muse image-only GGUF requires 1024 position rows and a 4096-wide adapter"
                .into(),
        );
    }
    let mut tensors = Vec::new();
    let mut bind = |spec: &dyn DeclareLogicalParameters,
                    names: Vec<String>,
                    matrix: bool,
                    physical: Option<Vec<usize>>|
     -> Result<(), String> {
        let parameters = spec
            .logical_parameters()
            .map_err(|error| error.to_string())?;
        if parameters.len() != names.len() {
            return Err("Muse vision storage extension does not cover its declared slots".into());
        }
        for (index, (parameter, gguf)) in parameters.into_iter().zip(names).enumerate() {
            tensors.push(DeclaredGgufTensor {
                canonical: parameter.parameter.id.as_str().to_owned(),
                gguf,
                shape: physical.clone().unwrap_or(parameter.shape),
                operation: if matrix && index == 0 {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Dense
                },
            });
        }
        Ok(())
    };
    bind(
        &statics.patch,
        vec!["v.patch_embd.weight".into()],
        false,
        Some(
            statics
                .spatial_source()
                .map_err(|error| error.to_string())?
                .kernel_shape
                .to_vec(),
        ),
    )?;
    bind(
        &statics.position,
        vec!["v.position_embd.weight".into()],
        false,
        None,
    )?;
    for (norm, prefix) in statics.norms.iter().zip(["v.pre_ln", "v.post_ln"]) {
        bind(
            norm,
            vec![format!("{prefix}.weight"), format!("{prefix}.bias")],
            false,
            None,
        )?;
    }
    for (layer, spec) in declarations.blocks.iter().enumerate() {
        for (norm, name) in spec.norms.iter().zip(["ln1", "ln2"]) {
            bind(
                norm,
                vec![
                    format!("v.blk.{layer}.{name}.weight"),
                    format!("v.blk.{layer}.{name}.bias"),
                ],
                false,
                None,
            )?;
        }
        for (projection, name) in spec
            .attention
            .projections()
            .into_iter()
            .chain(spec.mlp.projections())
            .zip([
                "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down",
            ])
        {
            bind(
                projection,
                vec![
                    format!("v.blk.{layer}.{name}.weight"),
                    format!("v.blk.{layer}.{name}.bias"),
                ],
                true,
                None,
            )?;
        }
    }
    for (index, projection) in statics
        .adapter
        .projections()
        .into_iter()
        .chain([&statics.projection])
        .enumerate()
    {
        bind(projection, vec![format!("mm.{index}.weight")], true, None)?;
    }
    Ok(tensors)
}

/// Resolves released media GGUF names to exact declared parameter identities.
pub fn projector_gguf_name_mapping(
    args: &DecoderConfig,
) -> Result<BTreeMap<String, String>, String> {
    let mut tensors = args
        .vision_config
        .as_ref()
        .map(vision_gguf_layouts)
        .transpose()?
        .unwrap_or_default();
    if let Some(audio) = &args.audio_config {
        tensors.extend(audio_gguf_layouts(audio)?);
    }
    let mut names = BTreeMap::new();
    for tensor in tensors {
        for (source, target) in gguf_source_bindings(tensor)? {
            if names.insert(source.clone(), target).is_some() {
                return Err(format!("conflicting Muse media GGUF name {source}"));
            }
        }
    }
    Ok(names)
}

/// Builds the official image-only sibling projector GGUF catalog.
pub fn projector_gguf_plan(args: &DecoderConfig) -> Result<GgufCheckpointPlan, String> {
    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| "Muse-Glimmer projector plan requires vision geometry".to_string())?;
    let mut tensors = vision_gguf_layouts(vision)?
        .into_iter()
        .map(|tensor| gguf(tensor.gguf, tensor.shape, tensor.operation))
        .collect::<Vec<_>>();
    if let Some(audio) = &args.audio_config {
        for tensor in audio_gguf_layouts(audio)? {
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
    let spec = &vision.declaration().statics;
    let spatial = spec.spatial_source().map_err(|error| error.to_string())?;
    let target = spatial.weight.id.as_str();
    let source = first_existing(catalog, [target.to_owned(), "v.patch_embd.weight".into()])
        .ok_or_else(|| "Muse-Glimmer projector patch kernel is missing".to_owned())?;
    let metadata = catalog
        .tensor_metadata(&source)
        .map_err(|error| error.to_string())?;
    if metadata.logical_shape != spatial.kernel_shape
        && metadata.logical_shape != spatial.matrix_shape
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
            shape: spatial.matrix_shape.to_vec(),
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

fn declared_expert_targets(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
) -> Result<(String, String, String), String> {
    let gate_up = source.gate_up().parameter.id.as_str().to_owned();
    let root = gate_up
        .strip_suffix(".gate_up_proj")
        .ok_or("unresolved Muse expert source root")?
        .to_owned();
    Ok((
        root,
        gate_up,
        source.down().parameter.id.as_str().to_owned(),
    ))
}

/// Resolves packed, separate, or independent expert tensors into neutral packed slots.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &DecoderConfig,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let declaration = args.routed(layer).map_err(|error| error.to_string())?;
    let source = declaration
        .bank()
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let (root, target_gate_up, target_down) = declared_expert_targets(&source)?;
    let raw = format!("blk.{layer}");
    let source = |canonical: String, raw: String| {
        first_existing(catalog, [canonical, raw]).unwrap_or_default()
    };
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: target_gate_up.clone(),
            target_down: target_down.clone(),
            packed_gate_up: target_gate_up,
            packed_down: target_down,
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
    let declarations = args.routed_layers();
    let capacity = declarations
        .values()
        .try_fold(0usize, |capacity, declaration| {
            capacity
                .checked_add(declaration.bank().group_count() as usize)
                .ok_or_else(|| "Muse-Glimmer expert residency catalog size overflows".to_string())
        })?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(super::TEXT_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for (layer, declaration) in declarations {
        let source = declaration
            .bank()
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let experts = source.group_count();
        let formats = source.encodings().map_err(|error| error.to_string())?;
        let unit_path = format!("model.layers.{layer}");
        let bank = match args.weight_convention {
            super::WeightConvention::HuggingFace => {
                safetensors_expert_recipes(catalog, args, *layer)?
            }
            super::WeightConvention::Gguf => expert_recipes(catalog, args, *layer)?,
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
                    let quantizable = [source.gate_up(), source.down()]
                        .into_iter()
                        .zip(formats)
                        .any(|(parameter, format)| {
                            parameter.parameter.id.as_str() == target.as_str()
                                && format == eredu_checkpoint::LinearFormat::Dense
                        });
                    let role = if quantizable {
                        crate::ExpertParameterRole::quantizable_projection(
                            format!("{binding}_scales"),
                            format!("{binding}_biases"),
                        )
                    } else {
                        crate::ExpertParameterRole::Preserved
                    };
                    crate::ExpertParameterRecipe::new(binding, target.as_str(), recipe, role)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    eredu_runtime::ParameterBankKey::new(0, *layer, expert),
                    owner_group.clone(),
                    *layer,
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

/// Prepared released text GGUF aliases with ordinary projection slots and companions resolved.
#[derive(Debug, Clone)]
pub struct TextGgufNames {
    projections: BTreeMap<String, String>,
}
impl TextGgufNames {
    /// Resolves ordinary projection aliases against typed construction geometry before payload reads.
    pub fn new(args: &DecoderConfig) -> Result<Self, String> {
        let layers = dimension(args.num_hidden_layers, "text layer count")?;
        let mut projections = BTreeMap::new();
        let mut tensors = static_gguf_layouts(args)?;
        for layer in 0..layers {
            tensors.extend(scalar_gguf_layouts(args, layer)?);
            tensors.extend(attention_gguf_layouts(args, layer)?);
            if args.is_moe() {
                tensors.extend(routed_gguf_layouts(args, layer)?);
            } else {
                tensors.extend(dense_gguf_layouts(args, layer)?);
            }
        }
        for tensor in tensors {
            let bindings = gguf_source_bindings(tensor)?;
            for (source, target) in bindings {
                if projections.insert(source.clone(), target).is_some() {
                    return Err(format!("conflicting Muse projection alias {source}"));
                }
            }
        }
        Ok(Self { projections })
    }
    /// Translates declared projection outputs and explicit remaining format extensions.
    pub fn translate(&self, name: &str) -> String {
        self.projections
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.into())
    }
}
