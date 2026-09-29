//! Source-free vision declarations and exact retained source binding.
use super::*;
use crate::qwen::vision::{self, VisionConfig, VisionMode};
use crate::qwen4_exp::config::MediaTokens;
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    store::{TensorMetadata, TensorSelection, TensorSourceProvenance},
};
use eredu_runtime::ReplicatedTextPhysicalSource;

/// Exact projector source factory inputs, validated before opening payload sources.
#[derive(Clone)]
pub struct GgufVisionSourcePlan {
    checkpoint: eredu_gguf::Checkpoint,
    checkpoint_plan: eredu_checkpoint::schema::GgufCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    catalog: eredu_checkpoint::gguf_store::GgufCatalog,
}
impl GgufVisionSourcePlan {
    /// Retained container headers and chosen representations.
    pub fn checkpoint(&self) -> &eredu_gguf::Checkpoint {
        &self.checkpoint
    }
    /// Exact architecture schema.
    pub fn checkpoint_plan(&self) -> &eredu_checkpoint::schema::GgufCheckpointPlan {
        &self.checkpoint_plan
    }
    /// Admitted physical-to-canonical mapping.
    pub fn mapping(&self) -> &[eredu_gguf::TranslatedTensorLayout] {
        &self.mapping
    }
    /// Resolved source contract used by the source factory.
    pub fn resolution(&self) -> &eredu_checkpoint::validation::ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Canonical metadata only; no readable methods are available.
    pub fn catalog(&self) -> &eredu_checkpoint::gguf_store::GgufCatalog {
        &self.catalog
    }
}

/// Token-independent projector geometry, recipes and exact physical source identities.
#[derive(Clone)]
struct VisionDeclaration {
    config: VisionConfig,
    expected: BTreeMap<String, PreparedTensorSource>,
    physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    static_recipes: BTreeMap<String, DerivedWeightRecipe>,
    blocks: Vec<BTreeMap<String, DerivedWeightRecipe>>,
}
impl RecipeCatalog for VisionDeclaration {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.expected
            .get(key)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }
}

/// Source-free GGUF projector admission before tokenizer special IDs are known.
/// A declaration cannot become an executable vision plan until all four IDs bind.
#[derive(Clone)]
pub struct GgufVisionPlan {
    source: Arc<GgufVisionSourcePlan>,
    target: Config,
    declaration: VisionDeclaration,
    processor: Option<crate::processor_plan::QwenProcessorPlan>,
}

/// Immutable image/video declaration with exact tokenizer IDs and no readable source.
#[derive(Clone)]
pub struct VisionPlan {
    gguf: Option<Arc<GgufVisionSourcePlan>>,
    declaration: VisionDeclaration,
    processor: Option<crate::processor_plan::QwenProcessorPlan>,
    media: MediaTokens,
}
impl RecipeCatalog for VisionPlan {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.declaration.tensor_metadata(key)
    }
}
impl VisionPlan {
    /// Admits an integrated SafeTensors tower from exact headers and physical identities.
    pub fn safetensors<C: RecipeCatalog + ?Sized>(
        target: &Config,
        catalog: &C,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<Self, PreparationError> {
        let config = target
            .vision
            .clone()
            .ok_or(PreparationError::MissingVision)?;
        let media = target
            .media
            .clone()
            .ok_or(PreparationError::MissingVision)?;
        let expected = physical.into_iter().filter(|(key, _)| key.starts_with("model.visual."))
            .map(|(key, physical)| {
                let mut metadata = catalog.tensor_metadata(&key)?;
                // The released conversion excludes the vision tower from FP8.
                // Preserve scalar geometry independently of a caller's byte declaration.
                let scalar_bytes = match metadata.stored_dtype {
                    eredu_checkpoint::StoredDtype::F32 => Some(4u64),
                    eredu_checkpoint::StoredDtype::F16 | eredu_checkpoint::StoredDtype::BF16 => Some(2u64),
                    _ => None,
                };
                let expected_bytes = scalar_bytes.and_then(|width| {
                    metadata.logical_shape.iter().try_fold(width, |bytes, &axis| {
                        bytes.checked_mul(u64::try_from(axis).ok()?)
                    })
                });
                if metadata.name != key
                    || metadata.physical_shape != metadata.logical_shape
                    || expected_bytes != Some(metadata.encoded_byte_len) || metadata.encoded_byte_len != physical.encoded_byte_len()
                    || physical.catalog_key() != key
                    || metadata.backing_shard.as_deref().is_some_and(|path| path != physical.shard())
                    || !matches!(physical.source_encoding(), eredu_checkpoint::SourceTensorEncoding::Safetensors(dtype) if dtype == &metadata.stored_dtype)
                {
                    return Err(StoreError::PreparedCatalogMismatch { key });
                }
                metadata.backing_shard = Some(physical.shard().to_owned());
                let provenance = TensorSourceProvenance {
                    catalog_key: physical.catalog_key().to_owned(), physical_tensor: physical.tensor().to_owned(),
                    output: physical.output().to_owned(), backing_shard: Some(physical.shard().to_owned()),
                    source_encoding: physical.source_encoding().clone(),
                };
                Ok((key, PreparedTensorSource { metadata, provenance }))
            }).collect::<Result<BTreeMap<_, _>, StoreError>>()?;
        struct Catalog<'a>(&'a BTreeMap<String, PreparedTensorSource>);
        impl eredu_checkpoint::validation::SafetensorsCatalog for Catalog<'_> {
            fn keys(&self) -> Vec<String> {
                self.0.keys().cloned().collect()
            }
            fn metadata(
                &self,
                key: &str,
            ) -> Result<eredu_checkpoint::validation::CatalogTensorMetadata, String> {
                let value = self
                    .0
                    .get(key)
                    .ok_or_else(|| format!("missing vision header {key}"))?;
                Ok(eredu_checkpoint::validation::CatalogTensorMetadata {
                    shape: value.metadata.logical_shape.clone(),
                    stored_dtype: value.metadata.stored_dtype.clone(),
                })
            }
        }
        let checkpoint = vision::safetensors_plan(&config, "model.visual")
            .map_err(PreparationError::Contract)?;
        resolve_safetensors_plan(&Catalog(&expected), &checkpoint).map_err(|error| {
            PreparationError::Contract(format!("vision contract did not resolve: {error:?}"))
        })?;
        let recipes = expected
            .keys()
            .map(|key| {
                (
                    key.clone(),
                    DerivedWeightRecipe::source(key, TensorSelection::Full),
                )
            })
            .collect();
        let plan = Self::new(config, media, expected, recipes)?;
        plan.validate_target(target, None)?;
        Ok(plan)
    }

    fn new(
        config: VisionConfig,
        media: MediaTokens,
        expected: BTreeMap<String, PreparedTensorSource>,
        recipes: BTreeMap<String, DerivedWeightRecipe>,
    ) -> Result<Self, PreparationError> {
        Ok(Self {
            declaration: VisionDeclaration::new(config, expected, recipes)?,
            processor: None,
            media,
            gguf: None,
        })
    }

    /// Checks decoder compatibility, including an optional GGUF n-gram image-reset ID.
    pub fn validate_target(
        &self,
        target: &Config,
        ngram_image_token: Option<u32>,
    ) -> Result<(), PreparationError> {
        validate_target_geometry(&self.declaration.config, target)?;
        validate_media_tokens(&self.media, target, ngram_image_token)
    }

    /// Retains optional host processor artifacts without reopening files.
    pub fn with_processor_json(
        mut self,
        image: Option<&[u8]>,
        video: Option<&[u8]>,
    ) -> Result<Self, PreparationError> {
        self.processor = crate::processor_plan::QwenProcessorPlan::from_visual_json(
            crate::processor_plan::MediaFraming {
                start_token_id: self.media.start,
                end_token_id: self.media.end,
            },
            image,
            video,
        )
        .map_err(|error| PreparationError::Contract(error.to_string()))?;
        self.validate_processor()?;
        Ok(self)
    }
    /// Retains an already admitted processor without changing its framing policy.
    pub fn with_processor(
        mut self,
        processor: crate::processor_plan::QwenProcessorPlan,
    ) -> Result<Self, PreparationError> {
        self.processor = Some(processor);
        self.validate_processor()?;
        Ok(self)
    }
    fn validate_processor(&self) -> Result<(), PreparationError> {
        let framing = crate::processor_plan::MediaFraming {
            start_token_id: self.media.start,
            end_token_id: self.media.end,
        };
        if self
            .processor
            .as_ref()
            .is_some_and(|processor| processor.framing() != Some(framing))
        {
            return Err(PreparationError::VisionMismatch {
                field: "processor media framing",
            });
        }
        validate_processor(&self.declaration.config, self.processor.as_ref())
    }
    /// Exact shared vision equations and per-matrix encodings.
    pub fn config(&self) -> &VisionConfig {
        &self.declaration.config
    }
    /// Original token IDs supplied by target policy.
    pub fn media_tokens(&self) -> &MediaTokens {
        &self.media
    }
    /// Optional retained raw-media processor policy.
    pub fn processor(&self) -> Option<&crate::processor_plan::QwenProcessorPlan> {
        self.processor.as_ref()
    }
    /// Complete canonical recipe set, without payload access.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.declaration.recipes
    }
    /// Static ingress and merger recipes.
    pub fn static_recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.declaration.static_recipes
    }
    /// One independently addressable encoder block.
    pub fn block_recipes(
        &self,
        layer: usize,
    ) -> Result<&BTreeMap<String, DerivedWeightRecipe>, PreparationError> {
        self.declaration.blocks.get(layer).ok_or_else(|| {
            PreparationError::Contract("vision block outside declared schedule".into())
        })
    }
    /// Exact physical source declarations used by cold parameter selection.
    pub fn physical_sources(&self) -> &BTreeMap<String, ReplicatedTextPhysicalSource> {
        &self.declaration.physical
    }
    /// Separate GGUF projector factory inputs, when this role uses that container.
    pub fn gguf_source(&self) -> Option<&GgufVisionSourcePlan> {
        self.gguf.as_deref()
    }

    /// Checks every retained metadata and provenance field before binding owner views.
    /// No weight or control payload is acquired.
    pub fn bind(self, source: SharedCheckpointSource) -> Result<PreparedVision, PreparationError> {
        let actual = source.source_keys().into_iter().collect::<BTreeSet<_>>();
        if let Some(key) = self
            .declaration
            .expected
            .keys()
            .find(|key| !actual.contains(*key))
        {
            return Err(StoreError::UnknownTensor { key: key.clone() }.into());
        }
        let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
            source,
            "qwen4_exp retained vision",
            self.declaration.expected.keys().cloned().collect(),
        )?);
        let source: SharedCheckpointSource = Arc::new(PreparedCheckpointSource::new(
            source,
            self.declaration.expected.clone(),
        )?);
        let static_parameters =
            PreparedParameters::new(source.clone(), self.declaration.static_recipes.clone())?;
        let blocks = self
            .declaration
            .blocks
            .iter()
            .map(|recipes| PreparedParameters::new(source.clone(), recipes.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedVision {
            plan: Arc::new(self),
            source,
            static_parameters,
            blocks,
        })
    }
}

fn validate_target_geometry(
    config: &VisionConfig,
    target: &Config,
) -> Result<(), PreparationError> {
    if config.out_hidden_size != target.hidden_size {
        return Err(PreparationError::VisionMismatch {
            field: "output width",
        });
    }
    if config.deepstack_layer_count() != 0 {
        return Err(PreparationError::VisionMismatch {
            field: "DeepStack injection",
        });
    }
    if let Some(expected) = &target.vision {
        let mut expected = expected.clone();
        expected.linear_formats.clear();
        let mut actual = config.clone();
        actual.linear_formats.clear();
        if expected != actual {
            return Err(PreparationError::VisionMismatch {
                field: "vision geometry",
            });
        }
    }
    Ok(())
}

fn validate_media_tokens(
    media: &MediaTokens,
    target: &Config,
    ngram_image_token: Option<u32>,
) -> Result<(), PreparationError> {
    let ids = [media.image, media.video, media.start, media.end];
    if ids
        .iter()
        .any(|id| i32::try_from(*id).map_or(true, |id| id >= target.vocabulary))
        || ids.into_iter().collect::<BTreeSet<_>>().len() != 4
        || ngram_image_token.is_some_and(|id| id != media.image)
        || target
            .media
            .as_ref()
            .is_some_and(|m| [m.image, m.video, m.start, m.end] != ids)
    {
        return Err(PreparationError::VisionMismatch {
            field: "media token IDs",
        });
    }
    Ok(())
}

fn validate_processor(
    config: &VisionConfig,
    processor: Option<&crate::processor_plan::QwenProcessorPlan>,
) -> Result<(), PreparationError> {
    if let Some(plan) = processor {
        plan.validate_patches(crate::processor_plan::QwenPatchPlan {
            patch_size: config.patch_size as usize,
            temporal_patch_size: config.temporal_patch_size as usize,
            merge_size: config.spatial_merge_size as usize,
        })
        .map_err(|error| PreparationError::Contract(error.to_string()))?;
    }
    Ok(())
}

impl VisionDeclaration {
    fn new(
        config: VisionConfig,
        expected: BTreeMap<String, PreparedTensorSource>,
        recipes: BTreeMap<String, DerivedWeightRecipe>,
    ) -> Result<Self, PreparationError> {
        config
            .validate_for(VisionMode::DeepStack)
            .map_err(|error| PreparationError::Contract(error.to_string()))?;
        let physical = expected
            .iter()
            .map(|(key, entry)| {
                let p = &entry.provenance;
                let shard = p.backing_shard.clone().ok_or_else(|| {
                    PreparationError::Contract(format!("missing backing shard for {key}"))
                })?;
                let physical = ReplicatedTextPhysicalSource::new(
                    p.catalog_key.clone(),
                    p.physical_tensor.clone(),
                    shard,
                    p.output.clone(),
                    p.source_encoding.clone(),
                    entry.metadata.encoded_byte_len,
                )
                .map_err(|error| PreparationError::Contract(error.to_string()))?;
                Ok((key.clone(), physical))
            })
            .collect::<Result<_, PreparationError>>()?;
        let static_recipes = recipes
            .iter()
            .filter(|(key, _)| !key.starts_with("model.visual.blocks."))
            .map(|(key, recipe)| (key.clone(), recipe.clone()))
            .collect();
        let blocks = (0..config.layer_count())
            .map(|layer| {
                let prefix = format!("model.visual.blocks.{layer}.");
                recipes
                    .iter()
                    .filter(|(key, _)| key.starts_with(&prefix))
                    .map(|(key, recipe)| (key.clone(), recipe.clone()))
                    .collect()
            })
            .collect();
        let plan = Self {
            config,
            expected,
            physical,
            recipes,
            static_recipes,
            blocks,
        };
        for recipe in plan.recipes.values() {
            recipe
                .infer(&plan)
                .map_err(|error| PreparationError::Contract(error.to_string()))?;
        }
        Ok(plan)
    }
}

impl GgufVisionPlan {
    /// Admits the published separate GGUF projector using retained headers only.
    pub fn prepare(
        target: &Config,
        checkpoint: &eredu_gguf::Checkpoint,
    ) -> Result<Self, PreparationError> {
        let invalid = |e: String| PreparationError::Contract(e);
        struct Catalog<'a>(&'a eredu_gguf::Checkpoint);
        impl vision::VisionGgufCatalog for Catalog<'_> {
            fn shape(&self, name: &str) -> Option<Vec<usize>> {
                self.0
                    .tensors()
                    .find(|t| t.descriptor().name == name)?
                    .descriptor()
                    .row_major_shape()
                    .into_iter()
                    .map(|d| usize::try_from(d).ok())
                    .collect()
            }
        }
        let metadata = checkpoint
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if checkpoint.metadata().get("clip.use_gelu")
            != Some(&eredu_gguf::MetadataValue::Bool(true))
            || checkpoint
                .metadata()
                .get("clip.vision.attention.layer_norm_epsilon")
                .and_then(eredu_gguf::MetadataValue::as_f32)
                != Some(1e-6)
        {
            return Err(PreparationError::VisionMismatch {
                field: "vision activation/normalization",
            });
        }
        let mut config = vision::config_from_gguf_catalog(
            &Catalog(checkpoint),
            &metadata,
            VisionMode::DeepStack,
        )
        .map_err(|e| invalid(e.to_string()))?;
        validate_target_geometry(&config, target)?;
        let plan = vision::gguf_plan(&config, target.hidden_size).map_err(invalid)?;
        for shard in checkpoint.shards() {
            for tensor in shard.tensors() {
                let name = vision::translate_gguf_weight_name(&tensor.descriptor().name, &[]);
                let format = crate::linear_format::gguf_tensor_format(tensor, shard.endian())
                    .map_err(invalid)?;
                if format != LinearFormat::Dense {
                    config.linear_formats.insert(name, format);
                }
            }
        }
        let mapping = checkpoint
            .translated_outputs(|n| match n {
                "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
                "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
                _ => vision::translate_gguf_weight_name(n, &[]),
            })
            .map_err(|e| invalid(e.to_string()))?;
        let resolution = eredu_checkpoint::validation::resolve_gguf_plan(checkpoint, &plan)
            .map_err(|error| invalid(format!("GGUF vision contract did not resolve: {error:?}")))?;
        let catalog = eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(
            checkpoint,
            &resolution,
            &mapping,
        )?;
        let mut recipes: BTreeMap<_, _> = catalog
            .keys()
            .into_iter()
            .map(|n| {
                (
                    n.clone(),
                    DerivedWeightRecipe::source(n, TensorSelection::Full),
                )
            })
            .collect();
        let names = [
            "model.visual.patch_embed.proj.weight.0",
            "model.visual.patch_embed.proj.weight.1",
        ];
        let mut inputs = [
            recipes
                .remove(names[0])
                .ok_or_else(|| invalid("missing first temporal patch".into()))?,
            recipes
                .remove(names[1])
                .ok_or_else(|| invalid("missing second temporal patch".into()))?,
        ];
        if inputs[0]
            .infer(&catalog)
            .map_err(|e| invalid(e.to_string()))?
            .dtype
            != inputs[1]
                .infer(&catalog)
                .map_err(|e| invalid(e.to_string()))?
                .dtype
        {
            inputs = inputs.map(|input| DerivedWeightRecipe::Cast {
                input: Box::new(input),
                dtype: eredu_checkpoint::recipe::RecipeDtype::F32,
            });
        }
        recipes.insert(
            "model.visual.patch_embed.proj.weight".into(),
            vision::temporal_patch_recipe(inputs),
        );
        let expected = catalog
            .keys()
            .into_iter()
            .map(|key| {
                Ok((
                    key.clone(),
                    PreparedTensorSource {
                        metadata: catalog.metadata(&key)?,
                        provenance: catalog.source_provenance(&key)?,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, StoreError>>()?;
        let source = GgufVisionSourcePlan {
            checkpoint: checkpoint.clone(),
            checkpoint_plan: plan,
            mapping,
            resolution,
            catalog,
        };
        let declaration = VisionDeclaration::new(config, expected, recipes)?;
        let mut vision = Self {
            target: target.clone(),
            declaration,
            processor: None,
            source: Arc::new(source),
        };
        if metadata.contains_key("clip.vision.image_mean")
            || metadata.contains_key("clip.vision.image_std")
        {
            let processor = crate::processor_plan::QwenProcessorPlan::from_gguf_metadata(
                &metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )
            .map_err(|e| PreparationError::Contract(e.to_string()))?;
            vision.processor = Some(processor);
            validate_processor(&vision.declaration.config, vision.processor.as_ref())?;
        }
        Ok(vision)
    }

    /// Admitted target geometry retained through later tokenizer resolution.
    pub fn target_config(&self) -> &Config {
        &self.target
    }
    /// Exact shared vision equations and per-matrix encodings.
    pub fn config(&self) -> &VisionConfig {
        &self.declaration.config
    }
    /// Retained host processor policy; framing IDs have not yet been bound.
    pub fn processor(&self) -> Option<&crate::processor_plan::QwenProcessorPlan> {
        self.processor.as_ref()
    }
    /// Complete canonical recipe set admitted without payload access.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.declaration.recipes
    }
    /// Exact physical identities available before special token resolution.
    pub fn physical_sources(&self) -> &BTreeMap<String, ReplicatedTextPhysicalSource> {
        &self.declaration.physical
    }
    /// Exact separate-projector source declaration.
    pub fn gguf_source(&self) -> &GgufVisionSourcePlan {
        &self.source
    }
    /// Binds the complete tokenizer contract, including the target n-gram reset ID.
    /// This does not open or read either artifact.
    pub fn bind_media_tokens(
        self,
        media: MediaTokens,
        ngram_image_token: Option<u32>,
    ) -> Result<VisionPlan, PreparationError> {
        validate_media_tokens(&media, &self.target, ngram_image_token)?;
        let mut processor = self.processor;
        if let Some(processor) = &mut processor {
            processor.bind_framing(crate::processor_plan::MediaFraming {
                start_token_id: media.start,
                end_token_id: media.end,
            });
        }
        Ok(VisionPlan {
            declaration: self.declaration,
            processor,
            media,
            gguf: Some(self.source),
        })
    }
}

impl std::fmt::Debug for GgufVisionPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgufVisionPlan")
            .field("config", self.config())
            .field("processor", &self.processor)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for VisionPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VisionPlan")
            .field("config", self.config())
            .field("media", &self.media)
            .field("processor", &self.processor)
            .finish_non_exhaustive()
    }
}
