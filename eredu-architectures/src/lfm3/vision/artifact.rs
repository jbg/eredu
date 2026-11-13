//! Released LFM3 VL SafeTensors admission.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    schema::{CatalogPolicy, SafetensorsCheckpointPlan},
    store::{
        CompositeCheckpointSource, RestrictedCheckpointSource, SharedCheckpointSource,
        TensorSelection,
    },
};
use eredu_nn::Error;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    processor_execution::PreparedProcessor,
    processor_plan::{MediaFraming, QwenPatchPlan, QwenProcessorPlan},
};

use super::{Declaration, ProjectorConfig, VisionUnit};

/// Materialization contract for a single vision invocation.
pub struct PreparedVisionWeights {
    /// Semantic owner of the retained parameters.
    pub unit: VisionUnit,
    /// View that authorizes only this invocation's physical tensors.
    pub source: SharedCheckpointSource,
    /// Canonical parameter destinations and requested conversion.
    pub recipes: BTreeMap<String, DerivedWeightRecipe>,
    /// Materialized parameter bytes, excluding activations and scratch.
    pub parameter_bytes: u64,
}

impl Declaration {
    /// Resolves independently materializable units using only retained metadata.
    pub fn prepare_weight_units(
        &self,
        source: &SharedCheckpointSource,
        dtype: RecipeDtype,
    ) -> Result<Vec<PreparedVisionWeights>, Error> {
        self.units()
            .map(|unit| {
                let mut recipes = BTreeMap::new();
                let mut parameter_bytes = 0u64;
                for parameter in self.unit_parameters(unit)? {
                    let name = parameter.parameter.id.to_string();
                    let recipe = DerivedWeightRecipe::Cast {
                        input: Box::new(DerivedWeightRecipe::source(&name, TensorSelection::Full)),
                        dtype: dtype.clone(),
                    };
                    let metadata = recipe
                        .infer(source.as_ref())
                        .map_err(Error::backend_source)?;
                    if metadata.shape != parameter.shape || recipes.insert(name, recipe).is_some() {
                        return Err(Error::backend(format!(
                            "Invalid parameter geometry in LFM3 vision {unit:?}"
                        )));
                    }
                    parameter_bytes = parameter_bytes
                        .checked_add(metadata.byte_len)
                        .ok_or_else(|| Error::backend("LFM3 vision unit size overflow"))?;
                }
                let source = RestrictedCheckpointSource::including(
                    source.clone(),
                    format!("LFM3 vision {unit:?}"),
                    recipes.keys().cloned().collect(),
                )
                .map_err(Error::backend_source)?;
                Ok(PreparedVisionWeights {
                    unit,
                    source: Arc::new(source),
                    recipes,
                    parameter_bytes,
                })
            })
            .collect()
    }
}

/// Media marker IDs declared by the VL checkpoint.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct MediaTokens {
    /// One token per projected image position.
    pub image_token_id: i32,
    /// One token per projected video position.
    pub video_token_id: i32,
    /// Opens a media span.
    pub vision_start_token_id: i32,
    /// Closes a media span.
    pub vision_end_token_id: i32,
}

/// Cold decoder construction for one admitted VL artifact.
pub struct PreparedVlDecoder {
    declaration: Arc<crate::lfm3::Declaration>,
    parameters: eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
}

/// Decoder resources restricted to one rank of an LFM3 VL tensor cohort.
pub struct PreparedVlTensorDecoder {
    prepared: PreparedVlDecoder,
    geometry: crate::lfm3::LocalGeometry,
    rank: eredu_core::ParallelRankTopology,
}

impl PreparedVlTensorDecoder {
    /// Exact rank-local resources used for residency sizing.
    pub fn parameters(
        &self,
    ) -> &eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>> {
        self.prepared.parameters()
    }

    /// Rank identity retained before native collectives are bound.
    pub fn rank(&self) -> eredu_core::ParallelRankTopology {
        self.rank
    }

    /// Constructs the rank-local language equations and retained state layout.
    pub fn into_layerwise<B>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            crate::lfm3::LayeredModel<B>,
            eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
            eredu_runtime::StateLayout,
        ),
        Error,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let state = self.geometry.state_layout().clone();
        let (model, parameters) = self.prepared.construct(Some(self.geometry), context)?;
        Ok((model, parameters, state))
    }
}

impl PreparedVlDecoder {
    fn prepare(
        identity: &str,
        declaration: Arc<crate::lfm3::Declaration>,
        source: SharedCheckpointSource,
        recipes: &BTreeMap<String, DerivedWeightRecipe>,
        dtype: RecipeDtype,
    ) -> Result<Self, Error> {
        let parameters = eredu_runtime::PreparedComponent::prepare(
            identity,
            declaration.definition().clone(),
            source,
            |member| -> Result<_, eredu_runtime::ComponentPreparationError> {
                let recipe = recipes.get(member.target()).cloned().ok_or_else(|| {
                    eredu_runtime::ComponentPreparationError::ArchitectureMismatch(format!(
                        "missing LFM3 conditional decoder parameter {}",
                        member.target()
                    ))
                })?;
                Ok(DerivedWeightRecipe::Cast {
                    input: Box::new(recipe),
                    dtype: dtype.clone(),
                })
            },
        )
        .map_err(Error::backend_source)?;
        Ok(Self {
            declaration,
            parameters,
        })
    }

    /// Exact retained bindings and addresses for residency sizing.
    pub fn parameters(
        &self,
    ) -> &eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>> {
        &self.parameters
    }

    /// Selects rank-local pipeline resources with vocabulary output.
    pub fn select_pipeline_rank(
        self,
        rank: eredu_core::ParallelRankTopology,
        prefix: std::num::NonZeroUsize,
    ) -> Result<crate::lfm3::PreparedLanguagePartition, eredu_core::BackendFailure> {
        crate::lfm3::PreparedLanguagePartition::select(
            self.declaration,
            self.parameters,
            rank,
            Some(prefix),
        )
    }

    pub(super) fn select_rank_placement(
        &self,
        rank: eredu_core::ParallelRankTopology,
        prefix: std::num::NonZeroUsize,
    ) -> Result<super::SelectedVlRankPlacement, Error> {
        super::SelectedVlRankPlacement::select(self.declaration.clone(), rank, prefix)
    }

    pub(super) fn bind_rank_placement(
        self,
        placement: crate::lfm3::SelectedLanguagePartition,
    ) -> Result<crate::lfm3::PreparedLanguagePartition, Error> {
        if placement.args() != self.declaration.args() {
            return Err(Error::backend(
                "selected VL rank differs from the retained decoder configuration",
            ));
        }
        placement
            .bind(self.parameters)
            .map_err(Error::backend_source)
    }

    /// Retains exact tensor-parallel parameter and cache geometry for one rank.
    pub fn select_tensor_rank(
        mut self,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<PreparedVlTensorDecoder, Error> {
        let layout = self
            .parameters
            .tensor_parallel_layout(rank)
            .map_err(Error::backend_source)?;
        let geometry = crate::lfm3::local_geometry(&self.declaration, &layout)
            .map_err(Error::backend_source)?;
        self.parameters = self
            .parameters
            .select(&layout)
            .map_err(Error::backend_source)?;
        Ok(PreparedVlTensorDecoder {
            prepared: self,
            geometry,
            rank,
        })
    }

    /// Constructs the architecture with its retained resource and state geometry.
    pub fn into_layerwise<B>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            crate::lfm3::LayeredModel<B>,
            eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
        ),
        Error,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.construct(None, context)
    }

    fn construct<B>(
        self,
        local: Option<crate::lfm3::LocalGeometry>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            crate::lfm3::LayeredModel<B>,
            eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
        ),
        Error,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let architecture = crate::lfm3::CompleteLfm3Kernel::<B>::from_selected_declaration(
            self.declaration,
            self.parameters.declaration().clone(),
            local,
            context,
        )?;
        Ok((architecture, self.parameters))
    }
}

pub(crate) const DECODER_ROLE: &str = "lfm3.vl.decoder";

fn decision_pixel_cap(config: &Value) -> Result<u64, Error> {
    config["d1_vision_max_pixels"]
        .as_u64()
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::backend("invalid d1 image pixel cap"))
}

/// Backend-neutral released VL declaration and retained checkpoint source.
#[derive(Clone)]
pub struct PreparedVisionLanguage {
    /// Shared LFM3 language decoder geometry.
    pub language: Arc<crate::lfm3::Declaration>,
    /// Independently constructible vision units.
    pub vision: Declaration,
    /// Checkpoint-specific media markers.
    pub tokens: MediaTokens,
    decoder_source: SharedCheckpointSource,
    vision_source: SharedCheckpointSource,
    vision_recipes: BTreeMap<String, DerivedWeightRecipe>,
    /// Canonical decoder names mapped to the checkpoint's nested language scope.
    pub language_recipes: BTreeMap<String, DerivedWeightRecipe>,
    processor: QwenProcessorPlan,
    component: Arc<eredu_runtime::ComponentDefinition>,
    architecture_fingerprint: String,
}

/// Normalized generative VL geometry and canonical SafeTensors bindings.
#[derive(Debug, Clone)]
pub struct VisionLanguageDeclaration {
    pub(crate) language: Arc<crate::lfm3::Declaration>,
    pub(crate) vision: Declaration,
    tokens: MediaTokens,
    language_recipes: BTreeMap<String, DerivedWeightRecipe>,
    checkpoint: SafetensorsCheckpointPlan,
    vision_keys: BTreeSet<String>,
    component: Arc<eredu_runtime::ComponentDefinition>,
    architecture_fingerprint: String,
}

impl VisionLanguageDeclaration {
    pub(crate) fn from_config(config: &Value) -> Result<Self, Error> {
        if config["model_type"] != "lfm3_vl"
            || config["architectures"] != serde_json::json!(["Lfm3VlForConditionalGeneration"])
        {
            return Err(Error::backend("expected an LFM3 VL conditional checkpoint"));
        }
        Self::from_checkpoint_config(config, "LFM3 VL conditional SafeTensors", None)
    }

    pub(in crate::lfm3) fn prepare_decision(
        root: &Path,
        config: &Value,
        dtype: RecipeDtype,
        max_cached_shards: usize,
        processor: Option<&[u8]>,
        omni_tokens: Option<MediaTokens>,
        audio_processor: Option<crate::lfm3::decision::DecisionAudioProcessor>,
    ) -> Result<
        (
            PreparedVlDecoder,
            Option<crate::lfm3::decision::PreparedDecisionVision>,
            Option<crate::lfm3::decision::PreparedDecisionAudio>,
        ),
        Error,
    > {
        use eredu_checkpoint::{
            safetensors::SafetensorsMetadataCatalog, store::open_validated_safetensors_source,
            validation::resolve_safetensors_plan,
        };

        if !crate::lfm3::decision::is_d1_config(config) {
            return Err(Error::backend("expected a d1 decision checkpoint"));
        }
        let declaration =
            Self::from_checkpoint_config(config, "d1 vision SafeTensors", omni_tokens)?;
        let catalog = SafetensorsMetadataCatalog::discover(root).map_err(Error::backend_source)?;
        let resolve = |plan: &SafetensorsCheckpointPlan| {
            resolve_safetensors_plan(&catalog, plan).map_err(|report| {
                Error::backend(format!("d1 checkpoint schema differs: {report:?}"))
            })
        };
        let mut plan = declaration.checkpoint().clone();
        let mut speech_keys = BTreeSet::new();
        let audio = if omni_tokens.is_some() {
            use crate::lfm3::audio::{Declaration as AudioEncoder, DepthDeclaration};
            let encoder = AudioEncoder::from_speech_config(config)?;
            let input = encoder.safetensors_plan()?;
            let input_resolution = resolve(&input)?;
            speech_keys.extend(input_resolution.source_keys().iter().cloned());
            let depth = DepthDeclaration::from_speech_config(config)?.safetensors_plan()?;
            speech_keys.extend(resolve(&depth)?.source_keys().iter().cloned());
            let mut tensors = plan.common_tensors;
            tensors.extend(input.common_tensors);
            tensors.extend(depth.common_tensors);
            let mut policy = plan.catalog_policy;
            policy
                .explicitly_allowed_keys
                .extend(depth.catalog_policy.explicitly_allowed_keys);
            plan = SafetensorsCheckpointPlan::new(
                "d1 Omni SafeTensors",
                tensors,
                plan.layout_groups,
                policy,
            )
            .map_err(Error::backend_source)?;
            audio_processor.map(|processor| (encoder, input_resolution, processor))
        } else {
            None
        };
        let resolution = resolve(&plan)?;
        let roles = declaration.source_roles(
            &resolution
                .source_keys()
                .difference(&speech_keys)
                .cloned()
                .collect(),
        )?;
        let source = open_validated_safetensors_source(
            catalog.admitted_shards(),
            catalog.tensors().clone(),
            &plan,
            resolution,
            max_cached_shards,
        )
        .map_err(Error::backend_source)?;
        let audio = audio
            .map(|(encoder, resolution, processor)| {
                crate::lfm3::decision::PreparedDecisionAudio::new(
                    encoder,
                    &resolution,
                    source.clone(),
                    processor,
                    dtype.clone(),
                )
            })
            .transpose()?;
        let [(_, decoder_keys), (_, vision_keys)] = roles;
        let vision = processor
            .map(|bytes| {
                let admission = declaration.prompt_admission()?;
                let processor = declaration
                    .processor(bytes)
                    .map_err(Error::backend_source)?;
                let vision_source = Arc::new(
                    RestrictedCheckpointSource::including(source.clone(), "d1 vision", vision_keys)
                        .map_err(Error::backend_source)?,
                );
                let vision_source: SharedCheckpointSource = vision_source;
                let vision = super::PreparedVisionExecution::new(
                    declaration.vision.clone(),
                    &vision_source,
                    dtype.clone(),
                )?;
                Ok::<_, Error>(crate::lfm3::decision::PreparedDecisionVision {
                    vision,
                    processor: PreparedProcessor::from_qwen(processor),
                    admission,
                    max_pixels: decision_pixel_cap(config)?,
                    tokens: declaration.tokens,
                    width: declaration.language.args().hidden_size,
                })
            })
            .transpose()?;
        let source = RestrictedCheckpointSource::including(source, "d1 decoder", decoder_keys)
            .map_err(Error::backend_source)?;
        let decoder = PreparedVlDecoder::prepare(
            "lfm3/decision/decoder",
            declaration.language,
            Arc::new(source),
            &declaration.language_recipes,
            dtype,
        )?;
        Ok((decoder, vision, audio))
    }

    pub(in crate::lfm3) fn prepare_decision_gguf(
        checkpoint: eredu_gguf::Checkpoint,
        projector: Option<(&Path, &[u8])>,
        config: &Value,
        omni_tokens: Option<MediaTokens>,
        audio_processor: Option<crate::lfm3::decision::DecisionAudioProcessor>,
        dtype: RecipeDtype,
        max_cached_readers: usize,
    ) -> Result<
        (
            PreparedVlDecoder,
            Option<crate::lfm3::decision::PreparedDecisionVision>,
            Option<crate::lfm3::decision::PreparedDecisionAudio>,
        ),
        Error,
    > {
        let declaration = Self::from_checkpoint_config(config, "d1 GGUF", omni_tokens)?;
        let (source, recipes) = super::gguf::retain_decision_backbone(
            checkpoint,
            &declaration.language,
            omni_tokens.is_some(),
            max_cached_readers,
        )?;
        let media = projector
            .map(|(path, processor)| {
                let admission = declaration.prompt_admission()?;
                let processor = declaration
                    .processor(processor)
                    .map_err(Error::backend_source)?;
                let (source, recipes) = super::gguf::retain_decision_projector(
                    path,
                    &declaration.vision,
                    omni_tokens.is_some(),
                    max_cached_readers,
                )?;
                let audio = audio_processor
                    .map(|processor| {
                        crate::lfm3::decision::PreparedDecisionAudio::from_gguf(
                            crate::lfm3::audio::Declaration::from_speech_config(config)?,
                            source.clone(),
                            processor,
                            dtype.clone(),
                        )
                    })
                    .transpose()?;
                let vision = super::PreparedVisionExecution::from_canonical_recipes(
                    declaration.vision.clone(),
                    source,
                    &recipes,
                    dtype.clone(),
                )?;
                Ok::<_, Error>((
                    crate::lfm3::decision::PreparedDecisionVision {
                        vision,
                        processor: PreparedProcessor::from_qwen(processor),
                        admission,
                        max_pixels: decision_pixel_cap(config)?,
                        tokens: declaration.tokens,
                        width: declaration.language.args().hidden_size,
                    },
                    audio,
                ))
            })
            .transpose()?;
        let decoder = PreparedVlDecoder::prepare(
            "lfm3/decision/decoder",
            declaration.language,
            source,
            &recipes,
            dtype,
        )?;
        let (vision, audio) = media.map_or((None, None), |(vision, audio)| (Some(vision), audio));
        Ok((decoder, vision, audio))
    }

    pub(in crate::lfm3) fn from_checkpoint_config(
        config: &Value,
        identity: &str,
        tokens: Option<MediaTokens>,
    ) -> Result<Self, Error> {
        let args = crate::lfm3::model_args_from_config_value(&config["text_config"])
            .map_err(Error::backend_source)?;
        let projector = ProjectorConfig::from_config_value(config)?;
        if projector.text_hidden_size != args.hidden_size {
            return Err(Error::backend("vision projector and decoder widths differ"));
        }
        let tokens = match tokens {
            Some(tokens) => tokens,
            None => serde_json::from_value(config.clone()).map_err(Error::backend_source)?,
        };
        let mut ids = [
            tokens.image_token_id,
            tokens.video_token_id,
            tokens.vision_start_token_id,
            tokens.vision_end_token_id,
        ];
        ids.sort_unstable();
        if ids[0] < 0 || ids[3] >= args.vocab_size || ids.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(Error::backend("invalid LFM3 VL media token IDs"));
        }

        let language = Arc::new(crate::lfm3::Declaration::with_decision_policy(
            args,
            crate::lfm3::decision::is_d1_config(config),
        )?);
        let vision = Declaration::new(
            serde_json::from_value(config["vision_config"].clone())
                .map_err(Error::backend_source)?,
            projector,
            "model.vision_tower",
            "model.multi_modal_projector",
        )?;
        let mut language_plan =
            crate::lfm3::safetensors_plan(&language, false).map_err(Error::backend)?;
        let mut language_recipes = BTreeMap::new();
        for tensor in &mut language_plan.common_tensors {
            let source = tensor.key.strip_prefix("model.").map_or_else(
                || tensor.key.clone(),
                |suffix| format!("model.language_model.{suffix}"),
            );
            language_recipes.insert(
                tensor.key.clone(),
                DerivedWeightRecipe::source(&source, TensorSelection::Full),
            );
            tensor.key = source;
        }
        let vision_plan = vision.safetensors_plan()?;
        let vision_keys = vision_plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.clone())
            .collect();
        language_plan
            .common_tensors
            .extend(vision_plan.common_tensors);
        let plan = SafetensorsCheckpointPlan::new(
            identity,
            language_plan.common_tensors,
            language_plan.layout_groups,
            CatalogPolicy::strict(),
        )
        .map_err(Error::backend_source)?;
        use sha2::Digest;
        let digest =
            sha2::Sha256::digest(serde_json::to_vec(config).map_err(Error::backend_source)?);
        let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let architecture_fingerprint = format!("lfm3-vl-v2:{digest}");
        let component = Arc::new(super::component::definition(&language, &vision)?);
        Ok(Self {
            language,
            vision,
            tokens,
            language_recipes,
            checkpoint: plan,
            vision_keys,
            component,
            architecture_fingerprint,
        })
    }

    /// Combined logical graph, storage ownership and decoder state layout.
    pub fn component(&self) -> &Arc<eredu_runtime::ComponentDefinition> {
        &self.component
    }

    pub(crate) fn architecture_fingerprint(&self) -> &str {
        &self.architecture_fingerprint
    }

    pub(crate) fn physical_parameter_key<'a>(&'a self, target: &'a str) -> Result<&'a str, Error> {
        match self.language_recipes.get(target) {
            Some(DerivedWeightRecipe::Source {
                key,
                selection: TensorSelection::Full,
            }) => Ok(key),
            None if self.vision_keys.contains(target) => Ok(target),
            _ => Err(Error::backend(format!(
                "VL parameter {target:?} has no direct checkpoint source"
            ))),
        }
    }

    /// Derives prepared-input admission from normalized geometry alone.
    pub fn prompt_admission(&self) -> Result<super::VisionPromptAdmission, Error> {
        super::VisionPromptAdmission::from_declarations(
            self.tokens,
            &self.vision,
            self.language.args(),
        )
    }

    pub(crate) fn checkpoint(&self) -> &SafetensorsCheckpointPlan {
        &self.checkpoint
    }

    pub(crate) fn source_roles(
        &self,
        admitted: &BTreeSet<String>,
    ) -> Result<
        [(
            crate::artifact_preparation::ArtifactSourceRole,
            BTreeSet<String>,
        ); 2],
        Error,
    > {
        use crate::artifact_preparation::ArtifactSourceRole;
        let decoder: BTreeSet<String> = self
            .language_recipes
            .values()
            .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
            .collect();
        if !decoder.is_disjoint(&self.vision_keys)
            || !decoder.union(&self.vision_keys).eq(admitted.iter())
        {
            return Err(Error::backend(
                "LFM3 VL source roles differ from the admitted tensor catalog",
            ));
        }
        Ok([
            (ArtifactSourceRole::Component(DECODER_ROLE), decoder),
            (ArtifactSourceRole::Vision, self.vision_keys.clone()),
        ])
    }

    pub(crate) fn processor(
        &self,
        bytes: &[u8],
    ) -> Result<QwenProcessorPlan, crate::processor_plan::ProcessorPlanError> {
        let processor = QwenProcessorPlan::from_lfm3_vl(
            MediaFraming {
                start_token_id: self.tokens.vision_start_token_id as u32,
                end_token_id: self.tokens.vision_end_token_id as u32,
            },
            bytes,
        )?;
        let geometry = self.vision.config();
        processor.validate_patches(QwenPatchPlan {
            patch_size: geometry.patch_size as usize,
            temporal_patch_size: geometry.temporal_patch_size as usize,
            merge_size: geometry.spatial_merge_size as usize,
        })?;
        Ok(processor)
    }
}

impl PreparedVisionLanguage {
    /// Inspects and retains the released generative VL checkpoint.
    pub fn open(directory: &Path, max_cached_shards: usize) -> Result<Self, Error> {
        let inspection =
            crate::configuration::inspect_artifact(directory).map_err(Error::backend_source)?;
        Self::from_inspection(&inspection, max_cached_shards)
    }

    /// Opens a released VL GGUF pair with the publisher's model and processor assets.
    pub fn open_gguf(
        backbone: &Path,
        projector: &Path,
        assets: &Path,
        max_cached_readers: usize,
    ) -> Result<Self, Error> {
        let config = std::fs::read(assets.join("config.json")).map_err(Error::backend_source)?;
        let declaration = VisionLanguageDeclaration::from_config(
            &serde_json::from_slice(&config).map_err(Error::backend_source)?,
        )?;
        let processor =
            std::fs::read(assets.join("processor_config.json")).map_err(Error::backend_source)?;
        let processor = declaration
            .processor(&processor)
            .map_err(Error::backend_source)?;
        let (decoder_source, language_recipes) =
            super::gguf::retain_backbone(backbone, &declaration.language, max_cached_readers)?;
        let (vision_source, vision_recipes) =
            super::gguf::retain_projector(projector, &declaration.vision, max_cached_readers)?;
        Ok(Self {
            language: declaration.language,
            vision: declaration.vision,
            tokens: declaration.tokens,
            decoder_source,
            vision_source,
            vision_recipes,
            language_recipes,
            processor,
            component: declaration.component,
            architecture_fingerprint: format!("{}:gguf", declaration.architecture_fingerprint),
        })
    }

    /// Uses the ordinary loader's admitted tensor catalog and retained processor.
    pub fn from_inspection(
        inspection: &eredu_core::ArtifactInspection<
            crate::processor_plan::ArtifactArchitecturePlan,
        >,
        max_cached_shards: usize,
    ) -> Result<Self, Error> {
        if inspection.architecture_plan().model_kind() != crate::configuration::ModelKind::Lfm3Vl {
            return Err(Error::backend("expected a generative LFM3 VL admission"));
        }
        let declarations = crate::artifact_preparation::source_declarations(inspection)
            .map_err(Error::backend_source)?;
        let bound = declarations
            .bind(max_cached_shards)
            .map_err(Error::backend_source)?;
        Self::from_roles(inspection.architecture_plan(), &bound.roles)
    }

    /// Builds VL components from sources retained by selected model preparation.
    pub fn from_source_graph(
        graph: &crate::prepared_sources::PreparedModelSourceGraph,
    ) -> Result<Self, Error> {
        Self::from_roles(graph.architecture(), graph.component_sources())
    }

    fn from_roles(
        plan: &crate::processor_plan::ArtifactArchitecturePlan,
        roles: &BTreeMap<crate::artifact_preparation::ArtifactSourceRole, SharedCheckpointSource>,
    ) -> Result<Self, Error> {
        use crate::artifact_preparation::ArtifactSourceRole;

        let declaration = plan
            .lfm3_vl_declaration()
            .ok_or_else(|| Error::backend("expected a generative LFM3 VL admission"))?;
        let processor = plan
            .qwen_patch_processor()
            .ok_or_else(|| Error::backend("LFM3 VL admission has no retained processor"))?;
        let source = |role| {
            roles
                .get(&role)
                .cloned()
                .ok_or_else(|| Error::backend(format!("LFM3 VL source graph omitted {role:?}")))
        };
        let decoder_source = source(ArtifactSourceRole::Component(DECODER_ROLE))?;
        let vision_source = source(ArtifactSourceRole::Vision)?;
        let (language_recipes, vision_recipes) = if plan.gguf_plan().is_some() {
            (
                super::gguf::backbone_recipes(decoder_source.as_ref(), &declaration.language)?,
                super::gguf::projector_recipes(vision_source.as_ref(), &declaration.vision)?,
            )
        } else {
            (declaration.language_recipes.clone(), BTreeMap::new())
        };
        Ok(Self {
            language: declaration.language.clone(),
            vision: declaration.vision.clone(),
            tokens: declaration.tokens,
            decoder_source,
            vision_source,
            vision_recipes,
            language_recipes,
            processor: processor.clone(),
            component: declaration.component().clone(),
            architecture_fingerprint: plan
                .lfm3_vl_execution_identity()
                .expect("admitted VL declaration"),
        })
    }

    /// Builds the complete component's weight bindings from retained source views.
    ///
    /// Recipes use canonical parameter names and preserve one resource per
    /// invocation, with decoder statics pinned independently.
    pub fn prepare_component(
        &self,
        dtype: RecipeDtype,
    ) -> Result<eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>, Error>
    {
        let source = self.component_source()?;
        eredu_runtime::PreparedComponent::prepare(
            self.component.id(),
            self.component.clone(),
            source,
            |member| -> Result<_, eredu_runtime::ComponentPreparationError> {
                let target = member.target();
                let recipe = self
                    .language_recipes
                    .get(target)
                    .or_else(|| self.vision_recipes.get(target))
                    .cloned()
                    .unwrap_or_else(|| DerivedWeightRecipe::source(target, TensorSelection::Full));
                Ok(DerivedWeightRecipe::Cast {
                    input: Box::new(recipe),
                    dtype: dtype.clone(),
                })
            },
        )
        .map_err(Error::backend_source)
    }

    pub(super) fn component_source(&self) -> Result<SharedCheckpointSource, Error> {
        let source = CompositeCheckpointSource::new([
            self.vision_source.clone(),
            self.decoder_source.clone(),
        ])
        .map_err(Error::backend_source)?;
        Ok(Arc::new(source))
    }

    /// Builds the typed architecture adapter over the exact prepared component.
    pub fn prepare_layerwise<B>(
        &self,
        dtype: RecipeDtype,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        (
            super::VlLayeredModel<B>,
            eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
        ),
        Error,
    >
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let prepared = self.prepare_component(dtype)?;
        let architecture = self.layerwise_for::<B>(&prepared, context)?;
        Ok((architecture, prepared))
    }

    pub(super) fn layerwise_for<B>(
        &self,
        prepared: &eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<super::VlLayeredModel<B>, Error>
    where
        B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        if prepared.declaration() != &self.component {
            return Err(Error::backend("VL bindings differ from the admitted graph"));
        }
        super::VlKernel::from_declaration(
            prepared.declaration().clone(),
            self.vision.clone(),
            self.language.clone(),
            self.architecture_fingerprint.clone(),
            context,
        )
    }

    pub(super) fn prepare_weight_components(
        &self,
        dtype: RecipeDtype,
    ) -> Result<(super::PreparedVisionExecution, PreparedVlDecoder), Error> {
        self.project_weight_components(self.prepare_component(dtype)?)
    }

    pub(super) fn project_weight_components(
        &self,
        component: eredu_runtime::PreparedComponent<Arc<eredu_runtime::ComponentDefinition>>,
    ) -> Result<(super::PreparedVisionExecution, PreparedVlDecoder), Error> {
        let vision = super::PreparedVisionExecution::from_component(
            self.vision.clone(),
            self.vision_source.clone(),
            &component,
        )?;
        let parameters = component
            .project(
                "lfm3/vision-language/decoder",
                self.language.definition().clone(),
            )
            .map_err(Error::backend_source)?;
        Ok((
            vision,
            PreparedVlDecoder {
                declaration: self.language.clone(),
                parameters,
            },
        ))
    }

    /// Prepares vision units from their restricted source role.
    pub fn prepare_vision_execution(
        &self,
        dtype: RecipeDtype,
    ) -> Result<super::PreparedVisionExecution, Error> {
        if self.vision_recipes.is_empty() {
            return super::PreparedVisionExecution::new(
                self.vision.clone(),
                &self.vision_source,
                dtype,
            );
        }
        super::PreparedVisionExecution::from_canonical_recipes(
            self.vision.clone(),
            self.vision_source.clone(),
            &self.vision_recipes,
            dtype,
        )
    }

    /// Prepares decoder bindings for the same retained VL checkpoint.
    pub fn prepare_language_execution(
        &self,
        dtype: RecipeDtype,
    ) -> Result<PreparedVlDecoder, Error> {
        PreparedVlDecoder::prepare(
            "lfm3/vision-language/decoder",
            self.language.clone(),
            self.decoder_source.clone(),
            &self.language_recipes,
            dtype,
        )
    }

    /// Retained image/video transforms, sampling and timestamp framing.
    pub fn processor(&self) -> PreparedProcessor {
        PreparedProcessor::from_qwen(self.processor.clone())
    }
}
