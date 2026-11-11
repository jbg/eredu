//! Portable canonical projections for GGUF speech input and depth generation.

use super::{projector_plan, AdmittedAudioGguf, AudioGgufError};
use crate::lfm3::audio::{BindingSource, Declaration, DepthDeclaration};
use eredu_checkpoint::{
    gguf_store::{open_prepared_gguf_source, GgufCatalog},
    recipe::DerivedWeightRecipe,
    schema::GgufCheckpointPlan,
    store::{SharedCheckpointSource, TensorSelection},
};
use eredu_codec::liquid::{Declaration as CodecDeclaration, GgufDecoderBindings};
use eredu_gguf::{Checkpoint, QuantizedTensorRepresentation, TranslatedTensorLayout};
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
        if !input.matches_gguf_execution() || !depth.matches_gguf_execution() {
            return Err(invalid(
                "speech execution geometry differs from the released GGUF projector",
            ));
        }
        let quantized = self
            .projector()
            .tensors()
            .filter_map(|tensor| {
                let descriptor = tensor.descriptor();
                descriptor
                    .ggml_type
                    .block_and_bytes()
                    .ok()
                    .filter(|(block, _)| *block > 1)
                    .map(|_| descriptor.name.clone())
            })
            .collect::<Vec<_>>();
        let projector = self
            .projector()
            .clone()
            .into_tensor_representation(quantized, QuantizedTensorRepresentation::DecodedF32)?;
        let mapping = projector.translated_outputs(str::to_owned)?;
        let catalog =
            GgufCatalog::from_resolved_checkpoint(&projector, self.layouts().1, &mapping)?;
        let input = compile(input, &catalog, input_recipe)?;
        let depth = compile(depth, &catalog, depth_recipe)?;
        let physical = catalog.keys();
        require_coverage(
            &input,
            physical.iter().filter(|key| !key.starts_with("a.gen.")),
        )?;
        require_coverage(
            &depth,
            physical.iter().filter(|key| key.starts_with("a.gen.code.")),
        )?;
        Ok(AudioGgufSpeechBindings {
            projector,
            plan: projector_plan(self.encodings().1)?,
            mapping,
            catalog,
            input,
            depth,
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
    ) -> Result<AudioGgufSpeechSources, AudioGgufError> {
        let source = self.open_source(max_cached_readers)?;
        self.restrict_speech(source)
    }

    /// Retains the speech components and a decoder-only LiquidCodec binding from one projector.
    pub fn retain_with_codec_decoder(
        self,
        declaration: &CodecDeclaration,
        max_cached_readers: usize,
    ) -> Result<(AudioGgufSpeechSources, BindingSource), AudioGgufError> {
        let codec = self.codec_decoder_bindings(declaration)?;
        let source = self.open_source(max_cached_readers)?;
        let (codec_source, recipes) = codec.retain(source.clone()).map_err(invalid)?;
        let codec = BindingSource::from_recipes(codec_source, "LiquidCodec GGUF decoder", recipes)?;
        Ok((self.restrict_speech(source)?, codec))
    }

    fn open_source(
        &self,
        max_cached_readers: usize,
    ) -> Result<SharedCheckpointSource, AudioGgufError> {
        Ok(Arc::new(open_prepared_gguf_source(
            self.projector.clone(),
            &self.plan,
            &self.mapping,
            max_cached_readers,
        )?))
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

fn compile(
    declaration: &impl DeclareLogicalParameters,
    catalog: &GgufCatalog,
    recipe: fn(&LogicalParameterDeclaration<'_>) -> Option<DerivedWeightRecipe>,
) -> Result<Recipes, AudioGgufError> {
    let mut recipes = BTreeMap::new();
    for parameter in declaration.logical_parameters().map_err(invalid)? {
        let key = parameter.parameter.id.as_str();
        let recipe =
            recipe(&parameter).ok_or_else(|| invalid(format!("no GGUF projection for {key}")))?;
        let inferred = recipe
            .infer(catalog)
            .map_err(|error| invalid(format!("{key}: {error}")))?;
        if inferred.shape != parameter.shape {
            return Err(invalid(format!(
                "{key}: projected shape {:?}, expected {:?}",
                inferred.shape, parameter.shape,
            )));
        }
        recipes.insert(key.into(), recipe);
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
            "lin_out" if field == "weight" => {
                Some(reshape("a.pre_encode.out.weight", &parameter.shape))
            }
            "lin_out" => Some(source("a.pre_encode.out.bias")),
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
