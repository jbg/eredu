//! Retained admission for the Audio checkpoint and its LiquidCodec companion.

use super::{DepthDeclaration, ProcessorConfig};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    safetensors::SafetensorsMetadataCatalog,
    schema::{CatalogPolicy, SafetensorsCheckpointPlan},
    store::{
        open_validated_safetensors_source, RestrictedCheckpointSource, SharedCheckpointSource,
        TensorSelection,
    },
    validation::{resolve_safetensors_plan, ResolvedCheckpointPlan},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Failure to admit a complete local LFM3 Audio SafeTensors artifact.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// A required companion file cannot be read.
    #[error("could not read {path:?}: {source}")]
    Read {
        /// Required companion file.
        path: PathBuf,
        /// File access failure.
        source: std::io::Error,
    },
    /// A required configuration file does not contain valid JSON.
    #[error("invalid LFM3 Audio JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Neural configuration or parameter topology is invalid.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    /// The LiquidCodec companion is invalid.
    #[error(transparent)]
    Codec(#[from] eredu_codec::Error),
    /// A header or retained source cannot be admitted.
    #[error("LFM3 Audio checkpoint admission failed: {0}")]
    Checkpoint(String),
}

/// Token identities used by the Audio checkpoint's language and speech protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTokens {
    /// Placeholder for one projected input speech position.
    pub input: i32,
    /// Placeholder for one speech feedback frame.
    pub output: i32,
    /// Switches to speech generation.
    pub audio_start: i32,
    /// Ends the text part of an interleaved response.
    pub text_end: i32,
    /// Starts an interleaved response.
    pub mixed_start: i32,
}

impl AudioTokens {
    fn from_config(config: &Value, vocabulary: i32) -> Result<Self, ArtifactError> {
        let id = |key: &str, default: i32| {
            let value = match config.get(key) {
                None => Some(default),
                Some(value) => value.as_i64().and_then(|n| i32::try_from(n).ok()),
            };
            value
                .filter(|n| (0..vocabulary).contains(n))
                .ok_or_else(|| {
                    ArtifactError::Checkpoint(format!("invalid speech token identity {key}"))
                })
        };
        let tokens = Self {
            input: id("audio_token_id", 125031)?,
            output: id("audio_out_token_id", 125032)?,
            audio_start: id("audio_start_token_id", 125012)?,
            text_end: id("text_end_token_id", 125014)?,
            mixed_start: id("mixed_start_token_id", 125015)?,
        };
        let ids = [
            tokens.input,
            tokens.output,
            tokens.audio_start,
            tokens.text_end,
            tokens.mixed_start,
        ];
        if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
            return Err(ArtifactError::Checkpoint(
                "speech token identities overlap".into(),
            ));
        }
        Ok(tokens)
    }
}

/// One exact source projection and its canonical parameter recipes.
#[derive(Clone)]
pub struct BindingSource {
    source: SharedCheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
}

impl BindingSource {
    /// Projects the resident F32 execution profile before native materialization.
    pub(super) fn widened_f32(&self) -> Result<(Self, u64), ArtifactError> {
        let mut recipes = self.recipes.clone();
        let mut bytes = 0u64;
        for key in self.source.source_keys() {
            let metadata = self.source.source_metadata(&key).map_err(checkpoint)?;
            let size = metadata
                .logical_shape
                .iter()
                .try_fold(4u64, |size, &dimension| {
                    size.checked_mul(u64::try_from(dimension).ok()?)
                })
                .ok_or_else(|| checkpoint("resident Audio parameter size overflowed"))?;
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| checkpoint("resident Audio parameter size overflowed"))?;
            recipes
                .entry(key.clone())
                .or_insert_with(|| DerivedWeightRecipe::source(key, TensorSelection::Full));
        }
        for recipe in recipes.values_mut() {
            *recipe = DerivedWeightRecipe::Cast {
                input: Box::new(recipe.clone()),
                dtype: RecipeDtype::F32,
            };
        }
        Ok((
            Self {
                source: self.source.clone(),
                recipes,
            },
            bytes,
        ))
    }

    /// Exact retained source for this scope, without checkpoint rediscovery.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }

    /// Canonical destinations whose physical identities or layout differ.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.recipes
    }
}

/// Released tokenizer and generation assets retained at preparation.
#[derive(Debug, Clone)]
pub struct TextAssets {
    /// Serialized tokenizer vocabulary and tokenization rules.
    pub tokenizer: Arc<[u8]>,
    /// Checkpoint-specific tokenizer configuration.
    pub tokenizer_config: Arc<[u8]>,
    /// Special token mapping.
    pub special_tokens: Arc<[u8]>,
    /// Checkpoint chat template.
    pub chat_template: Arc<str>,
    /// Checkpoint sampling and termination defaults.
    pub generation: Value,
}

/// Complete admitted Audio artifact with independently restricted parameter scopes.
#[derive(Clone)]
pub struct PreparedAudio {
    /// Causal language declaration shared with the text route.
    pub language: Arc<crate::lfm3::Declaration>,
    /// Speech input downsampler, encoder and projector declaration.
    pub encoder: super::Declaration,
    /// Speech output depth decoder and feedback declaration.
    pub depth: DepthDeclaration,
    /// LiquidCodec companion declaration.
    pub codec: eredu_codec::liquid::Declaration,
    /// Released host PCM processor policy.
    pub processor: ProcessorConfig,
    /// Special token identities in the language vocabulary.
    pub tokens: AudioTokens,
    interleaved_text_slots: usize,
    interleaved_speech_slots: usize,
    /// Retained tokenizer, template and generation configuration.
    pub text_assets: TextAssets,
    /// Restricted language checkpoint binding.
    pub language_weights: BindingSource,
    /// Restricted encoder/projector checkpoint binding.
    pub encoder_weights: BindingSource,
    /// Restricted depth/feedback checkpoint binding.
    pub depth_weights: BindingSource,
    /// Restricted codec companion binding.
    pub codec_weights: BindingSource,
}

fn read(root: &Path, name: &str) -> Result<Vec<u8>, ArtifactError> {
    let path = root.join(name);
    std::fs::read(&path).map_err(|source| ArtifactError::Read { path, source })
}

fn checkpoint(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Checkpoint(error.to_string())
}

fn discover(root: &Path) -> Result<SafetensorsMetadataCatalog, ArtifactError> {
    SafetensorsMetadataCatalog::discover(root).map_err(checkpoint)
}

fn resolve(
    catalog: &SafetensorsMetadataCatalog,
    plan: &SafetensorsCheckpointPlan,
) -> Result<ResolvedCheckpointPlan, ArtifactError> {
    resolve_safetensors_plan(catalog, plan).map_err(|error| checkpoint(format!("{error:?}")))
}

fn open(
    catalog: &SafetensorsMetadataCatalog,
    plan: &SafetensorsCheckpointPlan,
    resolution: ResolvedCheckpointPlan,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, ArtifactError> {
    open_validated_safetensors_source(
        catalog.admitted_shards(),
        catalog.tensors().clone(),
        plan,
        resolution,
        max_cached_shards,
    )
    .map_err(checkpoint)
}

fn restrict(
    source: &SharedCheckpointSource,
    resolution: &ResolvedCheckpointPlan,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<BindingSource, ArtifactError> {
    let source = RestrictedCheckpointSource::including(
        source.clone(),
        resolution.identity(),
        resolution.source_keys().clone(),
    )
    .map_err(checkpoint)?;
    Ok(BindingSource {
        source: Arc::new(source),
        recipes,
    })
}

impl PreparedAudio {
    /// Number of text slots per interleaved block.
    pub fn interleaved_text_slots(&self) -> usize {
        self.interleaved_text_slots
    }

    /// Number of speech slots per interleaved block.
    pub fn interleaved_speech_slots(&self) -> usize {
        self.interleaved_speech_slots
    }

    /// Admits the model and bundled audio_tokenizer without materializing tensor payloads.
    pub fn open(root: &Path, max_cached_shards: usize) -> Result<Self, ArtifactError> {
        let config: Value = serde_json::from_slice(&read(root, "config.json")?)?;
        Self::open_source(root, root, &config, max_cached_shards)
    }

    /// Admits an exact model payload with separately resolved Audio assets.
    /// The companion codec lives under `audio_tokenizer` in the asset root.
    pub(crate) fn open_source(
        model: &Path,
        root: &Path,
        config: &Value,
        max_cached_shards: usize,
    ) -> Result<Self, ArtifactError> {
        let cadence = |key: &str, default: usize| match config.get(key) {
            None => Ok(default),
            Some(value) => value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .ok_or_else(|| checkpoint(format!("invalid LFM3 Audio cadence {key}"))),
        };
        let interleaved_text_slots = cadence("interleaved_n_text", 6)?;
        let interleaved_speech_slots = cadence("interleaved_n_audio", 12)?;
        let encoder = super::Declaration::from_config(config)?;
        let depth = DepthDeclaration::from_config(config)?;
        let mut text = config.get("lfm").unwrap_or(&config["text_config"]).clone();
        if let Some(tied) = config.get("tie_word_embeddings") {
            text.as_object_mut()
                .ok_or_else(|| checkpoint("missing language config object"))?
                .insert("tie_word_embeddings".into(), tied.clone());
        }
        let language = crate::lfm3::model_args_from_config_value(&text).map_err(checkpoint)?;
        if language.weight_quantization.is_some() {
            return Err(checkpoint(
                "the dense LFM3 Audio contract requires floating source weights",
            ));
        }
        let tokens = AudioTokens::from_config(config, language.vocab_size)?;
        let language = Arc::new(crate::lfm3::Declaration::new(language)?);
        let processor = ProcessorConfig::from_feature_extractor(&serde_json::from_slice(&read(
            root,
            "preprocessor_config.json",
        )?)?)?;
        let text_assets = TextAssets {
            tokenizer: read(root, "tokenizer.json")?.into(),
            tokenizer_config: read(root, "tokenizer_config.json")?.into(),
            special_tokens: read(root, "special_tokens_map.json")?.into(),
            chat_template: String::from_utf8(read(root, "chat_template.jinja")?)
                .map_err(checkpoint)?
                .into(),
            generation: serde_json::from_slice(&read(root, "generation_config.json")?)?,
        };

        let mut language_plan =
            crate::lfm3::safetensors_plan(&language, false).map_err(checkpoint)?;
        for tensor in &mut language_plan.common_tensors {
            if let Some(suffix) = tensor.key.strip_prefix("model.") {
                tensor.aliases.extend([
                    format!("model.language_model.{suffix}"),
                    format!("lfm.model.{suffix}"),
                ]);
            } else if tensor.key == "lm_head.weight" {
                tensor.aliases.push("lfm.lm_head.weight".into());
            }
        }
        let input_plan = encoder.safetensors_plan()?;
        let depth_plan = depth.safetensors_plan()?;
        let mut policy = CatalogPolicy::strict();
        let mut tensors = Vec::new();
        let mut groups = Vec::new();
        for plan in [&language_plan, &input_plan, &depth_plan] {
            tensors.extend(plan.common_tensors.iter().cloned());
            groups.extend(plan.layout_groups.iter().cloned());
            policy
                .explicitly_allowed_keys
                .extend(plan.catalog_policy.explicitly_allowed_keys.iter().cloned());
            policy
                .allowed_suffixes
                .extend(plan.catalog_policy.allowed_suffixes.iter().cloned());
        }
        let model_plan =
            SafetensorsCheckpointPlan::new("LFM3 Audio SafeTensors", tensors, groups, policy)
                .map_err(checkpoint)?;
        let catalog = discover(model)?;
        let combined = resolve(&catalog, &model_plan)?;
        let language_resolution = resolve(&catalog, &language_plan)?;
        let encoder_resolution = resolve(&catalog, &input_plan)?;
        let depth_resolution = resolve(&catalog, &depth_plan)?;
        let source = open(&catalog, &model_plan, combined, max_cached_shards)?;
        let mut language_recipes = BTreeMap::new();
        for tensor in &language_plan.common_tensors {
            let physical = std::iter::once(&tensor.key)
                .chain(&tensor.aliases)
                .find(|key| language_resolution.source_keys().contains(*key))
                .ok_or_else(|| {
                    checkpoint(format!("no admitted language source for {}", tensor.key))
                })?;
            let metadata = source.source_metadata(physical).map_err(checkpoint)?;
            let shape: Vec<usize> = metadata
                .logical_shape
                .iter()
                .map(|&n| usize::try_from(n))
                .collect::<Result<_, _>>()
                .map_err(checkpoint)?;
            if shape != tensor.shape {
                if !tensor.key.ends_with(".conv.conv.weight")
                    || tensor.shape.len() != 3
                    || shape != [tensor.shape[0], tensor.shape[2], tensor.shape[1]]
                {
                    return Err(checkpoint(format!(
                        "unsupported admitted layout for {}",
                        tensor.key
                    )));
                }
                let recipe = DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::source(physical, TensorSelection::Full)),
                    axes: vec![0, 2, 1],
                };
                language_recipes.insert(tensor.key.clone(), recipe);
            } else if physical != &tensor.key {
                language_recipes.insert(
                    tensor.key.clone(),
                    DerivedWeightRecipe::source(physical, TensorSelection::Full),
                );
            }
        }
        let language_weights = restrict(&source, &language_resolution, language_recipes)?;
        let encoder_weights = restrict(
            &source,
            &encoder_resolution,
            encoder.safetensors_recipes(&encoder_resolution)?,
        )?;
        let depth_weights = restrict(
            &source,
            &depth_resolution,
            depth.safetensors_recipes(&depth_resolution)?,
        )?;

        let codec_root = root.join("audio_tokenizer");
        let codec_config =
            eredu_codec::liquid::Config::from_json(&read(&codec_root, "config.json")?)?;
        if codec_config.codebook_size()? != depth.eos_code()
            || usize::try_from(codec_config.n_q).ok() != Some(depth.schedule().iter().sum())
            || codec_config.sampling_rate != 24000
        {
            return Err(checkpoint(
                "LiquidCodec companion does not match the speech output vocabulary or sample rate",
            ));
        }
        let codec = eredu_codec::liquid::Declaration::new(codec_config)?;
        let codec_plan = codec.safetensors_plan()?;
        let catalog = discover(&codec_root)?;
        let resolution = resolve(&catalog, &codec_plan)?;
        let mut recipes = BTreeMap::new();
        for tensor in &codec_plan.common_tensors {
            if !resolution.source_keys().contains(&tensor.key) {
                let alias = tensor
                    .aliases
                    .iter()
                    .find(|alias| resolution.source_keys().contains(*alias))
                    .ok_or_else(|| {
                        checkpoint(format!("no admitted codec source for {}", tensor.key))
                    })?;
                recipes.insert(
                    tensor.key.clone(),
                    DerivedWeightRecipe::source(alias, TensorSelection::Full),
                );
            }
        }
        let codec_weights = BindingSource {
            source: open(&catalog, &codec_plan, resolution, max_cached_shards)?,
            recipes,
        };

        Ok(Self {
            language,
            encoder,
            depth,
            codec,
            processor,
            tokens,
            interleaved_text_slots,
            interleaved_speech_slots,
            text_assets,
            language_weights,
            encoder_weights,
            depth_weights,
            codec_weights,
        })
    }
}
