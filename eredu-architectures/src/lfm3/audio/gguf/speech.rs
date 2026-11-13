//! Portable canonical projections for GGUF speech input and depth generation.

use super::{AdmittedAudioGguf, AudioGgufError};
use crate::lfm3::audio::{BindingSource, Declaration, DepthDeclaration};
use eredu_checkpoint::{
    gguf_store::{GgufCatalog, GgufWeightStore},
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::GgufCheckpointPlan,
    store::{SharedCheckpointSource, TensorSelection},
};
use eredu_codec::liquid::{Declaration as CodecDeclaration, GgufDecoderBindings};
use eredu_gguf::{Checkpoint, TranslatedTensorLayout};
use eredu_nn::{DeclareLogicalParameters, LogicalParameterDeclaration};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

type Recipes = BTreeMap<String, DerivedWeightRecipe>;
const PACKED_EMBEDDINGS: &str = "a.gen.code.embd.weight";
const PACKED_HEAD_NORMS: &str = "a.gen.code.head_norm.weight";

/// Canonical speech input and depth binding recipes for an admitted projector.
pub struct AudioGgufSpeechBindings {
    projector: Checkpoint,
    plan: GgufCheckpointPlan,
    mapping: Vec<TranslatedTensorLayout>,
    catalog: GgufCatalog,
    input: Recipes,
    depth: Recipes,
    pub(in crate::lfm3) input_declaration: Declaration,
    pub(in crate::lfm3) depth_declaration: DepthDeclaration,
}

/// Restricted retained speech component sources, backed by the same GGUF projector.
pub struct AudioGgufSpeechSources {
    /// Speech input encoder and projector weights.
    pub input: BindingSource,
    /// Depth decoder and codebook feedback weights.
    pub depth: BindingSource,
}

impl AdmittedAudioGguf {
    /// Compiles the admitted projector against the portable speech declarations.
    /// All shapes are inferred without reading tensor payloads.
    pub fn speech_bindings(
        &self,
        input: &Declaration,
        depth: &DepthDeclaration,
    ) -> Result<AudioGgufSpeechBindings, AudioGgufError> {
        if !depth.matches_gguf_execution() {
            return Err(invalid(
                "speech execution geometry differs from the released GGUF projector",
            ));
        }
        let (input_declaration, input_formats) = select_input_formats(self.projector(), input)?;
        let mut depth_declaration = depth.clone();
        let depth_formats = select_formats(self.projector(), depth, depth_recipe)?;
        depth_declaration
            .select_formats(&depth_formats)
            .map_err(invalid)?;
        let projector = speech_projector_catalog(
            self.projector().clone(),
            input_declaration
                .logical_parameters()
                .map_err(invalid)?
                .iter()
                .filter(|parameter| parameter.linear_format.is_some())
                .filter_map(input_recipe)
                .chain(
                    depth_declaration
                        .logical_parameters()
                        .map_err(invalid)?
                        .iter()
                        .filter(|parameter| parameter.linear_format.is_some())
                        .filter_map(depth_recipe),
                ),
        )?;
        let mapping = projector.translated_outputs(str::to_owned)?;
        let catalog =
            GgufCatalog::from_resolved_checkpoint(&projector, self.layouts().1, &mapping)?;
        let input = compile(&input_declaration, &catalog, input_recipe, &input_formats)?;
        let depth = compile(&depth_declaration, &catalog, depth_recipe, &depth_formats)?;
        let physical = catalog.keys();
        require_coverage(
            &input,
            physical.iter().filter(|key| {
                key.starts_with("a.") && !key.starts_with("a.gen.") || key.starts_with("mm.a.")
            }),
        )?;
        require_coverage(
            &depth,
            physical.iter().filter(|key| key.starts_with("a.gen.code.")),
        )?;
        Ok(AudioGgufSpeechBindings {
            projector,
            plan: self.projector_plan.clone(),
            mapping,
            catalog,
            input,
            depth,
            input_declaration,
            depth_declaration,
        })
    }
}

fn require_coverage<'a>(
    recipes: &Recipes,
    physical: impl Iterator<Item = &'a String>,
) -> Result<(), AudioGgufError> {
    let covered = recipes
        .values()
        .flat_map(DerivedWeightRecipe::source_keys)
        .collect::<BTreeSet<_>>();
    if covered != physical.map(String::as_str).collect() {
        return Err(invalid(
            "canonical speech recipes do not cover their GGUF scope",
        ));
    }
    Ok(())
}

impl AudioGgufSpeechBindings {
    pub(in crate::lfm3) fn projector_catalog(&self) -> &GgufCatalog {
        &self.catalog
    }

    /// Exact input encoder and projector recipes keyed by canonical parameter name.
    pub fn input_recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.input
    }

    /// Exact depth and feedback recipes keyed by canonical parameter name.
    pub fn depth_recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.depth
    }

    /// Checks the LiquidCodec decoder projection against this projector's metadata.
    pub fn codec_decoder_bindings(
        &self,
        declaration: &CodecDeclaration,
    ) -> Result<GgufDecoderBindings, AudioGgufError> {
        GgufDecoderBindings::compile(declaration, &self.catalog).map_err(invalid)
    }

    /// Retains bounded weight leases from the admitted file-backed projector.
    /// Header-only provenance cannot grant payload access.
    pub fn retain(
        self,
        max_cached_readers: usize,
        prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
    ) -> Result<AudioGgufSpeechSources, AudioGgufError> {
        let source = self.open_source(max_cached_readers, prepared_cache)?;
        self.restrict_speech(source)
    }

    /// Retains the speech components and a decoder-only LiquidCodec binding from one projector.
    pub fn retain_with_codec_decoder(
        self,
        declaration: &CodecDeclaration,
        max_cached_readers: usize,
        prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
    ) -> Result<(AudioGgufSpeechSources, CodecDeclaration, BindingSource), AudioGgufError> {
        let (speech, declaration, codec, _) =
            self.retain_with_projector(declaration, max_cached_readers, prepared_cache)?;
        Ok((speech, declaration, codec))
    }

    pub(in crate::lfm3) fn retain_with_projector(
        self,
        declaration: &CodecDeclaration,
        max_cached_readers: usize,
        prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
    ) -> Result<
        (
            AudioGgufSpeechSources,
            CodecDeclaration,
            BindingSource,
            SharedCheckpointSource,
        ),
        AudioGgufError,
    > {
        let codec = self.codec_decoder_bindings(declaration)?;
        let declaration = codec.declaration().clone();
        let source = self.open_source(max_cached_readers, prepared_cache)?;
        let (codec_source, recipes) = codec.retain(source.clone()).map_err(invalid)?;
        let codec = BindingSource::from_recipes(codec_source, "LiquidCodec GGUF decoder", recipes)?;
        Ok((
            self.restrict_speech(source.clone())?,
            declaration,
            codec,
            source,
        ))
    }

    fn open_source(
        &self,
        max_cached_readers: usize,
        prepared_cache: Option<&eredu_checkpoint::prepared_cache::PreparedCheckpointCachePolicy>,
    ) -> Result<SharedCheckpointSource, AudioGgufError> {
        Ok(Arc::new(
            GgufWeightStore::builder()
                .require_prepared_decoding(prepared_cache)
                .max_cached_readers(max_cached_readers)?
                .add_checkpoint(self.projector.clone(), &self.plan, &self.mapping)?
                .build()?,
        ))
    }

    fn restrict_speech(
        self,
        source: SharedCheckpointSource,
    ) -> Result<AudioGgufSpeechSources, AudioGgufError> {
        Ok(AudioGgufSpeechSources {
            input: BindingSource::from_recipes(
                source.clone(),
                "LFM3 Audio GGUF input",
                self.input,
            )?,
            depth: BindingSource::from_recipes(source, "LFM3 Audio GGUF depth", self.depth)?
                .with_shared_sources([PACKED_EMBEDDINGS, PACKED_HEAD_NORMS]),
        })
    }
}

pub(in crate::lfm3) fn select_input_formats(
    checkpoint: &Checkpoint,
    declaration: &Declaration,
) -> Result<
    (
        Declaration,
        BTreeMap<String, eredu_checkpoint::LinearFormat>,
    ),
    AudioGgufError,
> {
    if !declaration.matches_gguf_execution() {
        return Err(invalid(
            "speech input geometry differs from the released GGUF projector",
        ));
    }
    let formats = select_formats(checkpoint, declaration, input_recipe)?;
    let mut declaration = declaration.clone();
    declaration.select_formats(&formats).map_err(invalid)?;
    Ok((declaration, formats))
}

pub(in crate::lfm3) fn input_projector_catalog(
    checkpoint: Checkpoint,
    declaration: &Declaration,
) -> Result<Checkpoint, AudioGgufError> {
    let parameters = declaration.logical_parameters().map_err(invalid)?;
    speech_projector_catalog(
        checkpoint,
        parameters
            .iter()
            .filter(|parameter| parameter.linear_format.is_some())
            .filter_map(input_recipe),
    )
}

pub(in crate::lfm3) fn input_encoder_recipes(
    declaration: &Declaration,
    catalog: &(impl RecipeCatalog + ?Sized),
) -> Result<Recipes, AudioGgufError> {
    let formats = declaration
        .logical_parameters()
        .map_err(invalid)?
        .into_iter()
        .filter_map(|parameter| {
            parameter
                .linear_format
                .map(|format| (parameter.parameter.id.to_string(), format.encoding()))
        })
        .collect();
    compile(declaration, catalog, input_recipe, &formats)
}

fn select_formats(
    checkpoint: &Checkpoint,
    declaration: &impl DeclareLogicalParameters,
    recipe: fn(&LogicalParameterDeclaration<'_>) -> Option<DerivedWeightRecipe>,
) -> Result<BTreeMap<String, eredu_checkpoint::LinearFormat>, AudioGgufError> {
    let physical = checkpoint
        .shards()
        .iter()
        .flat_map(|shard| {
            shard
                .tensors()
                .iter()
                .map(move |tensor| (tensor.descriptor().name.as_str(), (tensor, shard.endian())))
        })
        .collect::<BTreeMap<_, _>>();
    declaration
        .logical_parameters()
        .map_err(invalid)?
        .into_iter()
        .filter(|parameter| parameter.linear_format.is_some())
        .map(|parameter| {
            let name = parameter.parameter.id.to_string();
            let Some(DerivedWeightRecipe::Source { key, .. }) = recipe(&parameter) else {
                return Err(invalid(format!(
                    "missing direct speech matrix recipe for {name}"
                )));
            };
            let (tensor, endian) = physical
                .get(key.as_str())
                .copied()
                .ok_or_else(|| invalid(format!("missing speech tensor {key}")))?;
            Ok((
                name,
                crate::linear_format::gguf_tensor_format(tensor, endian).map_err(invalid)?,
            ))
        })
        .collect()
}

fn speech_projector_catalog(
    checkpoint: Checkpoint,
    packed: impl IntoIterator<Item = DerivedWeightRecipe>,
) -> Result<Checkpoint, AudioGgufError> {
    let packed = packed
        .into_iter()
        .filter_map(|recipe| match recipe {
            DerivedWeightRecipe::Source { key, .. } => Some(key),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let dense = checkpoint
        .tensors()
        .filter_map(|tensor| {
            let descriptor = tensor.descriptor();
            ((descriptor.name.starts_with("a.") || descriptor.name.starts_with("mm.a."))
                && !descriptor.name.starts_with("a.gen.wav.")
                && !packed.contains(&descriptor.name)
                && descriptor
                    .ggml_type
                    .block_and_bytes()
                    .is_ok_and(|(block, _)| block > 1))
            .then(|| descriptor.name.clone())
        })
        .collect::<Vec<_>>();
    Ok(checkpoint
        .into_tensor_representation(dense, eredu_gguf::QuantizedTensorRepresentation::DecodedF32)?)
}

fn compile(
    declaration: &impl DeclareLogicalParameters,
    catalog: &(impl RecipeCatalog + ?Sized),
    recipe: fn(&LogicalParameterDeclaration<'_>) -> Option<DerivedWeightRecipe>,
    formats: &BTreeMap<String, eredu_checkpoint::LinearFormat>,
) -> Result<Recipes, AudioGgufError> {
    let mut recipes = BTreeMap::new();
    for parameter in declaration.logical_parameters().map_err(invalid)? {
        let key = parameter.parameter.id.as_str();
        let recipe =
            recipe(&parameter).ok_or_else(|| invalid(format!("no GGUF projection for {key}")))?;
        let member = eredu_runtime::ParameterMemberSpec::from_logical_declaration(
            &parameter,
            eredu_runtime::MemberSharding::Replicated,
        );
        let members = if let Some(format) = parameter.linear_format {
            eredu_runtime::expand_linear_format_parameter(&member, format).map_err(invalid)?
        } else {
            vec![member]
        };
        let packed = formats
            .get(key)
            .is_some_and(|&format| format != eredu_checkpoint::LinearFormat::Dense);
        for member in members {
            let mut selected = if let Some(companion) = member.linear_companion() {
                let DerivedWeightRecipe::Source { key, selection } = &recipe else {
                    return Err(invalid(
                        "speech format companion requires a direct GGUF matrix",
                    ));
                };
                let prefix = key
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("invalid GGUF weight key"))?;
                let suffix = match companion {
                    eredu_nn::LinearCompanionRole::Scale => "scales",
                    eredu_nn::LinearCompanionRole::AffineBias => "biases",
                    _ => return Err(invalid("unsupported GGUF speech companion")),
                };
                DerivedWeightRecipe::source(format!("{prefix}.{suffix}"), selection.clone())
            } else {
                recipe.clone()
            };
            if let DerivedWeightRecipe::Source { selection, .. } = &mut selected {
                if let TensorSelection::Contiguous {
                    offset_elements,
                    shape,
                } = selection
                {
                    let index = *offset_elements / parameter.shape.iter().product::<usize>();
                    *shape = member.global_shape().to_vec();
                    *offset_elements = index * shape.iter().product::<usize>();
                }
            }
            let inferred = selected
                .infer(catalog)
                .map_err(|error| invalid(format!("{key}: {error}")))?;
            if inferred.shape != member.global_shape()
                || (packed
                    && member.linear_companion().is_none()
                    && inferred.dtype == eredu_checkpoint::recipe::RecipeDtype::F32)
            {
                return Err(invalid(format!(
                    "{key}: projected shape {:?}, expected {:?}",
                    inferred.shape,
                    member.global_shape()
                )));
            }
            recipes.insert(member.target().to_owned(), selected);
        }
    }
    Ok(recipes)
}

fn invalid(error: impl std::fmt::Display) -> AudioGgufError {
    AudioGgufError::Declaration(error.to_string())
}

fn source(name: impl Into<String>) -> DerivedWeightRecipe {
    DerivedWeightRecipe::source(name, TensorSelection::Full)
}

fn reshape(name: impl Into<String>, shape: &[usize]) -> DerivedWeightRecipe {
    DerivedWeightRecipe::Reshape {
        input: Box::new(source(name)),
        shape: shape.to_vec(),
    }
}

fn input_recipe(parameter: &LogicalParameterDeclaration<'_>) -> Option<DerivedWeightRecipe> {
    let key = parameter.parameter.id.as_str();
    if let Some(suffix) = key.strip_prefix("model.audio_encoder.downsample.") {
        let (layer, field) = suffix.split_once('.')?;
        return match layer {
            "lin_out" => Some(source(format!("a.pre_encode.out.{field}"))),
            _ => {
                let index = match layer {
                    "conv_in" => 0,
                    "conv_mid" => 2,
                    "lin_mid" => 3,
                    "conv_end" => 5,
                    "lin_end" => 6,
                    _ => return None,
                };
                let name = format!("a.conv1d.{index}.{field}");
                Some(if field == "bias" {
                    reshape(name, &parameter.shape)
                } else {
                    source(name)
                })
            }
        };
    }
    if let Some(suffix) = key.strip_prefix("model.audio_projector.") {
        let (layer, field) = suffix.split_once('.')?;
        let index = match layer {
            "layer_norm" => 0,
            "linear_1" => 1,
            "linear_2" => 3,
            _ => return None,
        };
        return Some(source(format!("mm.a.mlp.{index}.{field}")));
    }
    if key == "model.audio_encoder.lfm.embedding_norm.weight" {
        return Some(source("a.post_ln.weight"));
    }
    block_recipe(
        key.strip_prefix("model.audio_encoder.lfm.layers.")?,
        "a.blk",
        &parameter.shape,
    )
}

fn depth_recipe(parameter: &LogicalParameterDeclaration<'_>) -> Option<DerivedWeightRecipe> {
    let key = parameter.parameter.id.as_str();
    if key == "model.codebook_embedding.weight" {
        return Some(source("a.gen.code.out_embd.weight"));
    }
    let suffix = key.strip_prefix("model.depth_decoder.")?;
    match suffix {
        "model.embedding_norm.weight" => Some(source("a.gen.code.output_norm.weight")),
        "mask_embedding" => Some(source("a.gen.code.mask_embd")),
        "depth_linear.weight" => Some(source("a.gen.code.proj_in.weight")),
        "depth_linear.bias" => Some(source("a.gen.code.proj_in.bias")),
        _ if suffix.starts_with("depth_embeddings.") => {
            let (index, field) = suffix.strip_prefix("depth_embeddings.")?.split_once('.')?;
            let index = index.parse::<usize>().ok().filter(|&index| index < 12)?;
            let physical = match field {
                "embedding.weight" => PACKED_EMBEDDINGS,
                "embedding_norm.weight" => PACKED_HEAD_NORMS,
                _ => return None,
            };
            let elements = parameter
                .shape
                .iter()
                .try_fold(1usize, |n, width| n.checked_mul(*width))?;
            Some(DerivedWeightRecipe::source(
                physical,
                TensorSelection::Contiguous {
                    offset_elements: index.checked_mul(elements)?,
                    shape: parameter.shape.clone(),
                },
            ))
        }
        _ => block_recipe(
            suffix.strip_prefix("model.layers.")?,
            "a.gen.code.blk",
            &parameter.shape,
        ),
    }
}

fn block_recipe(suffix: &str, prefix: &str, shape: &[usize]) -> Option<DerivedWeightRecipe> {
    let (index, key) = suffix.split_once('.')?;
    let key = key.strip_suffix(".weight")?;
    let physical = match key {
        "operator_norm" => "ln1",
        "ffn_norm" => "ln2",
        "feed_forward.w1" => "ffn_gate",
        "feed_forward.w2" => "ffn_down",
        "feed_forward.w3" => "ffn_up",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.q_norm" | "self_attn.q_layernorm" => "attn_q_norm",
        "self_attn.k_norm" | "self_attn.k_layernorm" => "attn_k_norm",
        "self_attn.o_proj" | "self_attn.out_proj" => "attn_out",
        "self_attn.gate_proj" => "attn_gate",
        "conv.conv" => "conv_dw",
        "conv.in_proj" => "conv_pw1",
        "conv.out_proj" => "conv_pw2",
        _ => return None,
    };
    let name = format!("{prefix}.{index}.{physical}.weight");
    Some(if physical == "conv_dw" {
        reshape(name, shape)
    } else {
        source(name)
    })
}
