//! Neutral Gemma 4 binding to MLX storage, state, and residency policy.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::gemma4::{
    AudioInput, DecoderInputPart, FamilyConfig, LayeredModel as Architecture, ModelInput, Unit,
    VisionInput,
};
use eredu_checkpoint::{
    store::{CheckpointSource, SharedCheckpointSource},
    WeightQuantization,
};
use eredu_nn::{ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    CacheResidencyPolicy, CausalModel, ExecutionGraph, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, LayeredForwardState, LayerwiseRuntime, PagedCacheOptions,
    ParallelLayeredArchitecture, ParallelModelInfo, RuntimeState, StaticUnitBindings,
    WeightBinding, WeightResidency,
};
use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        maximum, pad, GgufCheckpoint, GgufMetadataValue, PadWidth,
    },
    Array, Dtype, Stream,
};

use crate::backend::mlx::{
    error::Error,
    nn::{
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        shared::{MlxBackend, MlxModule, MlxNamedModule},
    },
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::{
            binding::{
                binding_bytes, build_module_bindings, build_module_bindings_with_recipes_excluding,
                materialize_module_bindings, populate_module_from_arrays_excluding,
                populate_module_from_lease_excluding,
            },
            load::{gguf_metadata, gguf_quantization_configs, GgufTensorNames},
            quantization::should_quantize_on_load,
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitFactory,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_parameterized_module_store,
                quantize_parameterized_store, shard_layer_bindings,
            },
        },
        media::input,
        residency::expert_cache::{ExpertCache, ExpertCacheReport},
    },
};
use crate::composition::mlx::artifact::find_sibling_mmproj;

type NeutralArchitecture = Architecture<MlxBackend>;
type NeutralUnit = Unit<MlxBackend>;
type NeutralAssistant = eredu_architectures::gemma4::Assistant<MlxBackend>;
pub(crate) type Gemma4PipelineUnit = MlxModule<NeutralUnit>;

/// Opaque neutral state retained while placed Gemma media roots execute.
pub(crate) struct Gemma4PipelineIngressState {
    forward: Option<LayeredForwardState<Array, eredu_architectures::gemma4::ForwardContext<Array>>>,
    state: MlxHybridState,
    vision_hidden: Option<Array>,
    vision_state: Option<eredu_architectures::gemma4::VisionState<Array>>,
    audio_hidden: Option<Array>,
    audio_valid: Option<Vec<i32>>,
}

/// Decoder-ready neutral Gemma pipeline payload.
pub(crate) struct Gemma4PipelineIngressOutput {
    pub(crate) hidden: Array,
    pub(crate) per_layer_inputs: Option<Array>,
}

/// Pipeline/loading binder over the ordinary neutral Gemma 4 architecture.
pub(crate) struct Gemma4PipelineAdapter {
    args: FamilyConfig,
    architecture: NeutralArchitecture,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_layout: Option<eredu_runtime::LocalModelLayout>,
    local_args: Option<Arc<Vec<eredu_architectures::gemma4::ModelArgs>>>,
    local_text: Option<eredu_architectures::gemma4::ModelArgs>,
    parallel_media_range: Option<std::ops::Range<i32>>,
    external_experts: bool,
}

impl Gemma4PipelineAdapter {
    pub(crate) fn new(
        args: FamilyConfig,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture: NeutralArchitecture::new(args.clone(), stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            args,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_layout: None,
            local_args: None,
            local_text: None,
            parallel_media_range: None,
            external_experts,
        })
    }

    fn static_modules(&self) -> &eredu_architectures::gemma4::StaticModules<MlxBackend> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &self.architecture,
        )
    }

    fn static_modules_mut(
        &mut self,
    ) -> &mut eredu_architectures::gemma4::StaticModules<MlxBackend> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
            &mut self.architecture,
        )
    }

    pub(crate) fn architecture_mut(&mut self) -> &mut NeutralArchitecture {
        &mut self.architecture
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let modules = self.static_modules();
        let mut units = Vec::new();
        macro_rules! push_leaf {
            ($role:literal, $module:expr, $prefix:literal, $packed:expr) => {
                if select(concat!("gemma4.static.", $role)) {
                    let prefix = concat!($prefix, ".");
                    let bindings = build_module_bindings(
                        &MlxModule::new($module.clone()),
                        "",
                        store,
                    )?
                    .into_iter()
                    .map(|binding| {
                        let local = binding
                            .name()
                            .strip_prefix(prefix)
                            .ok_or_else(|| {
                                Error::Parallel(format!(
                                    "Gemma 4 static binding {:?} does not start with {prefix:?}",
                                    binding.name()
                                ))
                            })?
                            .to_string();
                        let local = if $packed && local == "weight" {
                            "inner.weight"
                        } else {
                            local.as_str()
                        };
                        binding.with_name(local).map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                    units.push(StaticUnitBindings::new(
                        concat!("gemma4.static.", $role),
                        bindings,
                    )?);
                }
            };
        }
        macro_rules! push {
            ($role:literal, $module:expr) => {
                if select(concat!("gemma4.static.", $role)) {
                    units.push(StaticUnitBindings::new(
                        concat!("gemma4.static.", $role),
                        build_module_bindings(&MlxModule::new($module.clone()), "", store)?,
                    )?);
                }
            };
        }
        push_leaf!(
            "embedding",
            modules.text.embeddings,
            "model.language_model.embed_tokens",
            self.args
                .text
                .linear_format_for("model.language_model.embed_tokens.weight")
                .weight_quantization()
                .is_some()
        );
        if let Some(module) = &modules.text.per_layer_embeddings {
            push_leaf!(
                "per_layer_embedding",
                module,
                "model.language_model.embed_tokens_per_layer",
                self.args
                    .text
                    .linear_format_for("model.language_model.embed_tokens_per_layer.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.text.per_layer_projection {
            push_leaf!(
                "per_layer_projection",
                module,
                "model.language_model.per_layer_model_projection",
                self.args
                    .text
                    .linear_format_for("model.language_model.per_layer_model_projection.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.text.per_layer_norm {
            push_leaf!(
                "per_layer_norm",
                module,
                "model.language_model.per_layer_projection_norm",
                false
            );
        }
        push_leaf!(
            "norm",
            modules.text.norm,
            "model.language_model.norm",
            false
        );
        if let Some(module) = &modules.text.head {
            push_leaf!(
                "output",
                module,
                "lm_head",
                self.args
                    .text
                    .linear_format_for("lm_head.weight")
                    .weight_quantization()
                    .is_some()
            );
        }
        if let Some(module) = &modules.vision {
            push!("vision", module);
        }
        if let Some(module) = &modules.vision_projection {
            push!("vision_projection", module);
        }
        if let Some(module) = &modules.audio {
            push!("audio", module);
        }
        if let Some(module) = &modules.audio_projection {
            push!("audio_projection", module);
        }
        Ok(units)
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self
                .args
                .vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize)),
            1 => Ok(self
                .args
                .audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize)),
            2 => Ok(self.args.text.num_hidden_layers()),
            _ => Err(Error::Parallel(format!(
                "Gemma 4 has no execution group {group}"
            ))),
        }
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineUnit, Error> {
        self.layer_count(group)?;
        let unit = match group {
            0 => NeutralUnit::Vision(
                eredu_architectures::gemma4::VisionLayer::new(
                    self.args.vision.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture("Gemma 4 vision config is missing".into())
                    })?,
                    index,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            ),
            1 => NeutralUnit::Audio(
                eredu_architectures::gemma4::AudioLayer::new(
                    self.args.audio.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture("Gemma 4 audio config is missing".into())
                    })?,
                    index,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            ),
            2 => {
                let args = match layout {
                    Some(_) => self
                        .local_args
                        .as_ref()
                        .and_then(|args| args.get(index))
                        .ok_or_else(|| {
                            Error::Parallel("missing local Gemma 4 layer args".into())
                        })?,
                    None => &self.args.text,
                };
                NeutralUnit::Text(
                    eredu_architectures::gemma4::DenseBlock::new(args, index, stream)
                        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                )
            }
            _ => unreachable!(),
        };
        Ok(MlxModule::new(unit))
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Gemma4PipelineUnit, Error> {
        self.new_cartesian_layer(group, index, None, stream)
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Gemma4PipelineUnit,
        store: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let recipes = if group == 2 && !self.external_experts {
            gemma4_unit_recipes(&self.args.text, index, store)?
        } else {
            BTreeMap::new()
        };
        build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
            self.external_experts && name.contains(".experts.switch_glu.")
        })
        .map_err(Into::into)
    }

    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Gemma4PipelineUnit,
        store: &dyn CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global;
        let layer = if group == 2 && layout.is_some() {
            global = self.new_cartesian_layer(group, index, None, stream)?;
            &global
        } else {
            layer
        };
        let recipes = if group == 2 && !self.external_experts {
            gemma4_unit_recipes(&self.args.text, index, store)?
        } else {
            BTreeMap::new()
        };
        let bindings =
            build_module_bindings_with_recipes_excluding(layer, "", store, recipes, |name| {
                self.external_experts && name.contains(".experts.switch_glu.")
            })?;
        match (group, layout) {
            (2, Some(layout)) => shard_layer_bindings(
                bindings,
                &format!("model.language_model.layers.{index}"),
                store,
                layout,
            ),
            _ => Ok(bindings),
        }
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    ) -> Result<(), Error> {
        for group in eredu_architectures::gemma4::static_parameter_groups(&self.args.text)? {
            planner.register(group)?;
        }
        for index in 0..self.args.text.num_hidden_layers() {
            for group in
                eredu_architectures::gemma4::layer_parameter_groups(&self.args.text, index)?
            {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_layout = Some(layout.clone());
        let local_args = Arc::new(
            (0..self.args.text.num_hidden_layers())
                .map(|index| {
                    eredu_architectures::gemma4::local_block_args(&self.args.text, index, layout)
                        .map_err(|error| Error::Parallel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut local_text = local_args
            .first()
            .cloned()
            .ok_or_else(|| Error::Parallel("Gemma 4 has no decoder layers".into()))?;
        local_text.layer_schedule = eredu_core::LayerSchedule::new(
            local_args.len(),
            local_args
                .iter()
                .enumerate()
                .map(|(index, args)| {
                    args.layer_policy(index).ok_or_else(|| {
                        Error::Parallel(format!("missing local Gemma 4 layer policy {index}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        self.local_args = Some(local_args);
        self.local_text = Some(local_text);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.text.vocab_size as usize,
            self.args.text.hidden_size,
            self.args
                .text
                .linear_format_for("model.language_model.embed_tokens.weight")
                .weight_quantization(),
            build,
            stream,
        )?);
        self.parallel_lm_head = (!self.args.text.tie_word_embeddings)
            .then(|| {
                VocabParallelLmHead::unloaded(
                    self.args.text.hidden_size,
                    self.args.text.vocab_size as usize,
                    self.args
                        .text
                        .linear_format_for("lm_head.weight")
                        .weight_quantization(),
                    build,
                    stream,
                )
            })
            .transpose()?;
        if self.args.text.hidden_size_per_layer_input > 0 {
            let range = crate::core::balanced_contiguous_range(
                self.args.text.hidden_size_per_layer_input as usize,
                build.topology().tensor_parallel_size,
                build.topology().tensor_parallel_rank,
                false,
            )?;
            self.parallel_media_range = Some(range.start as i32..range.end as i32);
        }
        Ok(())
    }

    pub(crate) fn embedding_mut(&mut self) -> &mut crate::backend::mlx::nn::shared::MlxEmbedding {
        &mut self.static_modules_mut().text.embeddings
    }

    pub(crate) fn per_layer_embedding_mut(
        &mut self,
    ) -> Option<&mut crate::backend::mlx::nn::shared::MlxEmbedding> {
        self.static_modules_mut().text.per_layer_embeddings.as_mut()
    }

    pub(crate) fn per_layer_projection_mut(
        &mut self,
    ) -> Option<&mut crate::backend::mlx::nn::shared::MlxLinear> {
        self.static_modules_mut().text.per_layer_projection.as_mut()
    }

    pub(crate) fn per_layer_norm_mut(
        &mut self,
    ) -> Option<&mut crate::backend::mlx::nn::shared::MlxRmsNorm> {
        self.static_modules_mut().text.per_layer_norm.as_mut()
    }

    pub(crate) fn norm_mut(&mut self) -> &mut crate::backend::mlx::nn::shared::MlxRmsNorm {
        &mut self.static_modules_mut().text.norm
    }

    pub(crate) fn output_mut(&mut self) -> Option<&mut crate::backend::mlx::nn::shared::MlxLinear> {
        self.static_modules_mut().text.head.as_mut()
    }

    pub(crate) fn vision_mut(
        &mut self,
    ) -> Option<
        crate::backend::mlx::nn::shared::MlxModuleRef<
            '_,
            eredu_architectures::gemma4::VisionStatic<MlxBackend>,
        >,
    > {
        self.static_modules_mut()
            .vision
            .as_mut()
            .map(crate::backend::mlx::nn::shared::MlxModuleRef::new)
    }

    pub(crate) fn vision_projection_mut(
        &mut self,
    ) -> Option<
        crate::backend::mlx::nn::shared::MlxModuleRef<
            '_,
            eredu_architectures::gemma4::ModalityProjector<MlxBackend>,
        >,
    > {
        self.static_modules_mut()
            .vision_projection
            .as_mut()
            .map(crate::backend::mlx::nn::shared::MlxModuleRef::new)
    }

    pub(crate) fn audio_mut(
        &mut self,
    ) -> Option<
        crate::backend::mlx::nn::shared::MlxModuleRef<
            '_,
            eredu_architectures::gemma4::AudioStatic<MlxBackend>,
        >,
    > {
        self.static_modules_mut()
            .audio
            .as_mut()
            .map(crate::backend::mlx::nn::shared::MlxModuleRef::new)
    }

    pub(crate) fn audio_projection_mut(
        &mut self,
    ) -> Option<
        crate::backend::mlx::nn::shared::MlxModuleRef<
            '_,
            eredu_architectures::gemma4::ModalityProjector<MlxBackend>,
        >,
    > {
        self.static_modules_mut()
            .audio_projection
            .as_mut()
            .map(crate::backend::mlx::nn::shared::MlxModuleRef::new)
    }

    pub(crate) fn parallel_embedding_mut(&mut self) -> Option<&mut VocabParallelEmbedding> {
        self.parallel_embedding.as_mut()
    }

    pub(crate) fn parallel_lm_head_mut(&mut self) -> Option<&mut VocabParallelLmHead> {
        self.parallel_lm_head.as_mut()
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let args = self.local_text.as_ref().unwrap_or(&self.args.text);
        let layout = eredu_architectures::gemma4::state_layout(args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut identity = eredu_runtime::ModelStateIdentity {
            model_family: "gemma4".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: self.args.text.architecture_fingerprint(),
            layer_count: layout.len(),
            global_layer_start: 0,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layout.len()],
            topology: Default::default(),
        }
        .prompt_cache_identity(&layout)
        .map_err(|error| Error::Parallel(error.to_string()))?;
        if let Some(topology) = topology {
            identity.topology = crate::backend::mlx::cache::prompt_cache_topology(topology);
        }
        Ok(identity)
    }

    pub(crate) fn pipeline_per_layer_width(&self) -> i32 {
        self.parallel_media_range
            .as_ref()
            .map_or(self.args.text.hidden_size_per_layer_input, |range| {
                range.end - range.start
            })
    }

    pub(crate) fn pipeline_state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        let args = self.local_text.as_ref().unwrap_or(&self.args.text);
        eredu_architectures::gemma4::state_layout(args)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressState, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        let mut state = MlxHybridState::device(
            eredu_architectures::gemma4::state_layout(&self.args.text)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        )?;
        let mut forward = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::begin_forward(
            &mut self.architecture,
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            &mut state,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let vision_hidden = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::should_execute_group(&self.architecture, 0, &forward.context)
        .then(|| {
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_execution_group(
                &mut self.architecture,
                0,
                &forward.hidden,
                &[],
                &mut state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
        .transpose()?;
        let audio_hidden = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::should_execute_group(&self.architecture, 1, &forward.context)
        .then(|| {
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::begin_execution_group(
                &mut self.architecture,
                1,
                &forward.hidden,
                &[],
                &mut state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        })
        .transpose()?;
        Ok(Gemma4PipelineIngressState {
            forward: Some(forward),
            state,
            vision_hidden,
            vision_state: None,
            audio_hidden,
            audio_valid: None,
        })
    }

    pub(crate) fn begin_pipeline_continuation(
        &mut self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressState, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let vision_hidden = prepared.vision_input().map(|input| input.patches.clone());
        let vision_state = prepared
            .vision_input()
            .map(|input| {
                self.static_modules()
                    .vision
                    .as_ref()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture("Gemma 4 has no vision tower".into())
                    })?
                    .prepare_state(input, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            })
            .transpose()?;
        let audio_hidden = prepared.audio_input().map(|input| input.features.clone());
        let audio_valid = prepared
            .audio_input()
            .map(|input| input.valid_subsampled_frames.to_vec());
        Ok(Gemma4PipelineIngressState {
            forward: None,
            state: MlxHybridState::device(
                eredu_architectures::gemma4::state_layout(&self.args.text)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
            )?,
            vision_hidden,
            vision_state,
            audio_hidden,
            audio_valid,
        })
    }

    pub(crate) fn pipeline_ingress_active(
        &self,
        group: &str,
        state: &Gemma4PipelineIngressState,
    ) -> Result<bool, Error> {
        match group {
            "vision_encoder" => Ok(state.vision_hidden.is_some()),
            "audio_encoder" => Ok(state.audio_hidden.is_some()),
            _ => Err(Error::Parallel(format!(
                "Gemma 4 has no placed media group {group:?}"
            ))),
        }
    }

    pub(crate) fn pipeline_ingress_arrays(
        &self,
        group: &str,
        state: &Gemma4PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        let hidden = match group {
            "vision_encoder" => state.vision_hidden.as_ref(),
            "audio_encoder" => state.audio_hidden.as_ref(),
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        Ok(hidden.cloned().into_iter().collect())
    }

    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        group: &str,
        state: &mut Gemma4PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let slot = match group {
            "vision_encoder" => &mut state.vision_hidden,
            "audio_encoder" => &mut state.audio_hidden,
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        match (slot.is_some(), arrays.as_slice()) {
            (true, [hidden]) => {
                *slot = Some(hidden.clone());
                Ok(())
            }
            (false, []) => Ok(()),
            (active, _) => Err(Error::Parallel(format!(
                "Gemma 4 {group} payload has {} arrays for active={active}",
                arrays.len()
            ))),
        }
    }

    pub(crate) fn merge_pipeline_ingress_arrays(
        &self,
        state: &mut Gemma4PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let expected =
            usize::from(state.vision_hidden.is_some()) + usize::from(state.audio_hidden.is_some());
        if arrays.len() != expected {
            return Err(Error::Parallel(format!(
                "Gemma 4 media merger produced {} arrays, expected {expected}",
                arrays.len()
            )));
        }
        let mut arrays = arrays.into_iter();
        if state.vision_hidden.is_some() {
            state.vision_hidden = arrays.next();
        }
        if state.audio_hidden.is_some() {
            state.audio_hidden = arrays.next();
        }
        Ok(())
    }

    pub(crate) fn forward_pipeline_media_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Gemma4PipelineUnit,
        state: &mut Gemma4PipelineIngressState,
        stream: &Stream,
    ) -> Result<(), Error> {
        let hidden = match group {
            0 => state.vision_hidden.as_ref(),
            1 => state.audio_hidden.as_ref(),
            _ => None,
        }
        .ok_or_else(|| Error::Parallel("Gemma 4 media group has no activation".into()))?
        .clone();
        let output = if let Some(forward) = state.forward.as_mut() {
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::forward_unit(
                &mut self.architecture,
                group,
                index,
                &mut **layer,
                &hidden,
                &mut state.state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
        } else {
            match (group, &mut **layer) {
                (0, NeutralUnit::Vision(layer)) => self
                    .static_modules()
                    .vision
                    .as_ref()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture("Gemma 4 vision static is missing".into())
                    })?
                    .forward_layer(
                        layer,
                        &hidden,
                        state.vision_state.as_ref().ok_or_else(|| {
                            Error::Parallel("Gemma 4 vision continuation state is missing".into())
                        })?,
                        stream,
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                (1, NeutralUnit::Audio(layer)) => layer
                    .forward(
                        &hidden,
                        state.audio_valid.as_deref().ok_or_else(|| {
                            Error::Parallel("Gemma 4 audio continuation state is missing".into())
                        })?,
                        stream,
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
                _ => return Err(Error::Parallel("Gemma 4 media unit/group mismatch".into())),
            }
        };
        match group {
            0 => state.vision_hidden = Some(output),
            1 => state.audio_hidden = Some(output),
            _ => unreachable!(),
        }
        Ok(())
    }

    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: Gemma4PipelineIngressState,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressOutput, Error> {
        let mut forward = state.forward.take().ok_or_else(|| {
            Error::Parallel("Gemma 4 media finalization requires the primary ingress state".into())
        })?;
        if let Some(hidden) = state.vision_hidden.take() {
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::complete_execution_group(
                &mut self.architecture,
                0,
                &hidden,
                &mut state.state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        if let Some(hidden) = state.audio_hidden.take() {
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::complete_execution_group(
                &mut self.architecture,
                1,
                &hidden,
                &mut state.state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        }
        let (hidden, tokens) = self
            .architecture
            .assemble_pipeline_text(&forward.context, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let mut per_layer_inputs = self
            .architecture
            .pipeline_per_layer_inputs(&tokens, &hidden, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        if let (Some(inputs), Some(range)) = (&per_layer_inputs, &self.parallel_media_range) {
            per_layer_inputs = Some(inputs.try_index_device((.., .., .., range.clone()), stream)?);
        }
        NeutralArchitecture::set_pipeline_per_layer_inputs(
            &mut forward.context,
            per_layer_inputs.clone(),
        );
        Ok(Gemma4PipelineIngressOutput {
            hidden,
            per_layer_inputs,
        })
    }

    pub(crate) fn prepare_pipeline_tokens<S>(
        &mut self,
        tokens: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        state: &mut S,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, eredu_architectures::gemma4::ForwardContext<Array>>, Error>
    where
        S: eredu_runtime::LayerRuntimeState<MlxBackend>,
        S::LayerState: eredu_nn::AttentionCache<Array>,
    {
        if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
            let embeddings = self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Error::Parallel("Gemma 4 TP pipeline has no embedding".into()))?
                .forward(tokens, execution)?;
            let mut forward = self
                .architecture
                .begin_parallel_text(tokens, embeddings, state, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let mut per_layer_inputs = self
                .architecture
                .pipeline_per_layer_inputs(tokens, &forward.hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            if let (Some(inputs), Some(range)) = (&per_layer_inputs, &self.parallel_media_range) {
                per_layer_inputs =
                    Some(inputs.try_index_device((.., .., .., range.clone()), stream)?);
            }
            NeutralArchitecture::set_pipeline_per_layer_inputs(
                &mut forward.context,
                per_layer_inputs,
            );
            return Ok(forward);
        }
        let parts = [DecoderInputPart::Text(tokens)];
        let mut forward =
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, S>>::begin_forward(
                &mut self.architecture,
                ModelInput {
                    parts: &parts,
                    vision: None,
                    audio: None,
                    per_layer_tokens: None,
                    mask: None,
                },
                state,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        forward.hidden =
            <NeutralArchitecture as LayeredArchitecture<MlxBackend, S>>::begin_execution_group(
                &mut self.architecture,
                2,
                &forward.hidden,
                &[],
                state,
                &mut forward.context,
                stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(forward)
    }

    pub(crate) fn resume_pipeline_text<S>(
        &self,
        hidden: Array,
        mask: Option<Array>,
        per_layer_inputs: Option<Array>,
        state: &mut S,
    ) -> Result<LayeredForwardState<Array, eredu_architectures::gemma4::ForwardContext<Array>>, Error>
    where
        S: eredu_runtime::LayerRuntimeState<MlxBackend>,
        S::LayerState: eredu_nn::AttentionCache<Array>,
    {
        self.architecture
            .resume_pipeline_text(hidden, mask, per_layer_inputs, state)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub(crate) fn finish_pipeline_text(
        &mut self,
        hidden: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(execution) = execution.filter(|execution| execution.is_tensor_parallel()) {
            let hidden = self
                .architecture
                .final_parallel_hidden(hidden, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let logits = match &mut self.parallel_lm_head {
                Some(head) => head.forward(&hidden, execution)?,
                None => self
                    .parallel_embedding
                    .as_mut()
                    .ok_or_else(|| {
                        Error::Parallel("Gemma 4 tied TP pipeline has no embedding shard".into())
                    })?
                    .project_logits(&hidden, execution)?,
            }
            .all_gather(execution)?;
            return self
                .architecture
                .finish_parallel_logits(logits, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
        }
        self.architecture
            .project_pipeline_logits(hidden, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}
type Resident = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralUnit>,
>;
type Bounded = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralUnit, UnitFactory>,
>;
type ParallelResident = LayerwiseRuntime<
    Gemma4ParallelComposition,
    MlxBackend,
    MlxHybridState,
    MlxResidentPolicy<eredu_architectures::gemma4::DenseBlock<MlxBackend>>,
>;
type ParallelBounded = LayerwiseRuntime<
    Gemma4ParallelComposition,
    MlxBackend,
    MlxHybridState,
    MlxLayerwisePolicy<eredu_architectures::gemma4::DenseBlock<MlxBackend>, ParallelUnitFactory>,
>;

#[derive(Clone)]
struct UnitFactory {
    args: FamilyConfig,
    vision_layers: usize,
    audio_layers: usize,
    external_experts: bool,
}

impl MlxUnitFactory<NeutralUnit> for UnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<NeutralUnit, Error> {
        if ordinal < self.vision_layers {
            eredu_architectures::gemma4::VisionLayer::new(
                self.args.vision.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture("Gemma 4 vision config is missing".into())
                })?,
                ordinal,
                stream,
            )
            .map(NeutralUnit::Vision)
        } else if ordinal < self.vision_layers + self.audio_layers {
            eredu_architectures::gemma4::AudioLayer::new(
                self.args.audio.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture("Gemma 4 audio config is missing".into())
                })?,
                ordinal - self.vision_layers,
                stream,
            )
            .map(NeutralUnit::Audio)
        } else {
            eredu_architectures::gemma4::DenseBlock::new(
                &self.args.text,
                ordinal - self.vision_layers - self.audio_layers,
                stream,
            )
            .map(NeutralUnit::Text)
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralUnit>,
        lease: &crate::backend::mlx::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && name.contains(".experts.switch_glu.")
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct ParallelUnitFactory {
    args: Arc<Vec<eredu_architectures::gemma4::ModelArgs>>,
}

impl MlxUnitFactory<eredu_architectures::gemma4::DenseBlock<MlxBackend>> for ParallelUnitFactory {
    fn build(
        &mut self,
        index: usize,
        stream: &Stream,
    ) -> Result<eredu_architectures::gemma4::DenseBlock<MlxBackend>, Error> {
        let args = self.args.get(index).ok_or_else(|| {
            Error::Parallel(format!(
                "parallel Gemma 4 layer {index} is outside {} local layouts",
                self.args.len()
            ))
        })?;
        eredu_architectures::gemma4::DenseBlock::new(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

/// MLX vocabulary shards around the backend-neutral Gemma 4 decoder.
pub struct Gemma4ParallelComposition {
    architecture: NeutralArchitecture,
    embedding: MlxNamedModule<VocabParallelEmbedding>,
    output: Option<MlxNamedModule<VocabParallelLmHead>>,
    vision_layers: Vec<eredu_architectures::gemma4::VisionLayer<MlxBackend>>,
    audio_layers: Vec<eredu_architectures::gemma4::AudioLayer<MlxBackend>>,
    local_args: Arc<Vec<eredu_architectures::gemma4::ModelArgs>>,
    local_text: eredu_architectures::gemma4::ModelArgs,
    topology: crate::backend::mlx::MlxParallelContext,
}

/// Rank-local input accepted by the neutral Gemma 4 TP composition.
pub enum Gemma4ParallelInput<'a> {
    /// Ordinary text-only decode or prefill input.
    Text(&'a Array),
    /// Prepared ordered text/media input for prefill.
    Prepared(ModelInput<'a, Array>),
}

impl Gemma4ParallelComposition {
    fn new(
        args: FamilyConfig,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let local_args = Arc::new(
            (0..args.text.num_hidden_layers())
                .map(|index| {
                    eredu_architectures::gemma4::local_block_args(&args.text, index, layout)
                        .map_err(|error| Error::Parallel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut local_text = local_args
            .first()
            .cloned()
            .ok_or_else(|| Error::Parallel("Gemma 4 has no decoder layers".into()))?;
        local_text.layer_schedule = eredu_core::LayerSchedule::new(
            local_args.len(),
            local_args
                .iter()
                .enumerate()
                .map(|(index, args)| {
                    args.layer_policy(index).ok_or_else(|| {
                        Error::Parallel(format!("missing local Gemma 4 layer policy {index}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let embedding_name = "model.language_model.embed_tokens.weight";
        let output_name = "lm_head.weight";
        let embedding = MlxNamedModule::new(
            VocabParallelEmbedding::unloaded(
                args.text.vocab_size as usize,
                args.text.hidden_size,
                args.text
                    .linear_format_for(embedding_name)
                    .weight_quantization(),
                build,
                stream,
            )?,
            ParameterSpec::trainable(embedding_name)
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None,
        )?;
        let output = (!args.text.tie_word_embeddings)
            .then(|| {
                Ok::<_, Error>(MlxNamedModule::new(
                    VocabParallelLmHead::unloaded(
                        args.text.hidden_size,
                        args.text.vocab_size as usize,
                        args.text
                            .linear_format_for(output_name)
                            .weight_quantization(),
                        build,
                        stream,
                    )?,
                    ParameterSpec::trainable(output_name)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                    None,
                )?)
            })
            .transpose()?;
        let vision_layers = args
            .vision
            .as_ref()
            .map(|vision| {
                (0..vision.num_hidden_layers as usize)
                    .map(|index| {
                        eredu_architectures::gemma4::VisionLayer::new(vision, index, stream)
                            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .transpose()?
            .unwrap_or_default();
        let audio_layers = args
            .audio
            .as_ref()
            .map(|audio| {
                (0..audio.num_hidden_layers as usize)
                    .map(|index| {
                        eredu_architectures::gemma4::AudioLayer::new(audio, index, stream)
                            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            architecture,
            embedding,
            output,
            vision_layers,
            audio_layers,
            local_args,
            local_text,
            topology: build.topology(),
        })
    }

    fn execution_context<'a>(
        &self,
        group: &'a safemlx::distributed::Group,
        stream: &'a Stream,
    ) -> Result<
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'a>,
        Error,
    > {
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            self.topology,
            group,
            stream,
        )
    }
}

impl Parameterized<Array> for Gemma4ParallelComposition {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        self.embedding.visit_parameters(visitor);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules(&self.architecture);
        modules.text.norm.visit_parameters(visitor);
        modules.vision.visit_parameters(visitor);
        modules.vision_projection.visit_parameters(visitor);
        modules.audio.visit_parameters(visitor);
        modules.audio_projection.visit_parameters(visitor);
        self.vision_layers.visit_parameters(visitor);
        self.audio_layers.visit_parameters(visitor);
        if let Some(output) = &self.output {
            output.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        self.embedding.visit_parameters_mut(visitor);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(&mut self.architecture);
        modules.text.norm.visit_parameters_mut(visitor);
        modules.vision.visit_parameters_mut(visitor);
        modules.vision_projection.visit_parameters_mut(visitor);
        modules.audio.visit_parameters_mut(visitor);
        modules.audio_projection.visit_parameters_mut(visitor);
        self.vision_layers.visit_parameters_mut(visitor);
        self.audio_layers.visit_parameters_mut(visitor);
        if let Some(output) = &mut self.output {
            output.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.embedding.set_trainable(trainable);
        let modules = <NeutralArchitecture as LayeredArchitecture<
            MlxBackend,
            MlxHybridState,
        >>::static_modules_mut(&mut self.architecture);
        modules.text.norm.set_trainable(trainable);
        modules.vision.set_trainable(trainable);
        modules.vision_projection.set_trainable(trainable);
        modules.audio.set_trainable(trainable);
        modules.audio_projection.set_trainable(trainable);
        self.vision_layers.set_trainable(trainable);
        self.audio_layers.set_trainable(trainable);
        if let Some(output) = &mut self.output {
            output.set_trainable(trainable);
        }
    }
}

impl LayeredArchitecture<MlxBackend, MlxHybridState> for Gemma4ParallelComposition {
    type Input<'a> = Gemma4ParallelInput<'a>;
    type StaticModules = Self;
    type Unit = eredu_architectures::gemma4::DenseBlock<MlxBackend>;
    type ForwardContext = eredu_architectures::gemma4::ForwardContext<Array>;
    type RetainedContextValues<'a> = std::vec::IntoIter<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.architecture.args().model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "parallel Gemma 4 decoder has no execution group {group}"
            )));
        }
        Ok(self.local_args.len())
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::Parallel(format!(
                "parallel Gemma 4 layer {index} is out of range"
            )));
        }
        Ok(format!("model.language_model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel("invalid Gemma 4 TP execution group".into()));
        }
        eredu_architectures::gemma4::DenseBlock::new(
            self.local_args
                .get(index)
                .ok_or_else(|| Error::Parallel("missing Gemma 4 local layer args".into()))?,
            index,
            stream,
        )
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        _input: Self::Input<'a>,
        _state: &mut MlxHybridState,
        _stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        Err(Error::Parallel(
            "parallel Gemma 4 composition requires a collective context".into(),
        ))
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _unit: &mut Self::Unit,
        _hidden: &Array,
        _state: &mut MlxHybridState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Gemma 4 composition requires a collective context".into(),
        ))
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _state: &mut MlxHybridState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::Parallel(format!(
                "parallel Gemma 4 group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn finish_forward(
        &mut self,
        _hidden: &Array,
        _state: &mut MlxHybridState,
        _forward: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Gemma 4 composition requires a collective context".into(),
        ))
    }

    fn retained_context_values<'a>(
        &'a self,
        _forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        Vec::new().into_iter()
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MlxHybridState> for Gemma4ParallelComposition {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        match input {
            Gemma4ParallelInput::Text(tokens) => {
                let embeddings = self.embedding.forward(tokens, &execution)?;
                self.architecture
                    .begin_parallel_text(tokens, embeddings, state, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }
            Gemma4ParallelInput::Prepared(input) => {
                let embeddings = input
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        DecoderInputPart::Text(tokens) => Some(
                            self.embedding
                                .forward(tokens, &execution)
                                .map_err(|error| Error::Parallel(error.to_string())),
                        ),
                        DecoderInputPart::Image(_)
                        | DecoderInputPart::Video(_)
                        | DecoderInputPart::Audio(_)
                        | DecoderInputPart::Projected { .. } => None,
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.architecture
                    .begin_parallel_input(
                        input,
                        &embeddings,
                        &mut self.vision_layers,
                        &mut self.audio_layers,
                        state,
                        stream,
                    )
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
            }
        }
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        state: &mut MlxHybridState,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        if group_index != 0 {
            return Err(Error::Parallel("invalid Gemma 4 TP execution group".into()));
        }
        self.architecture
            .forward_text_unit_parallel(index, unit, hidden, state, forward, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _state: &mut MlxHybridState,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .architecture
            .final_parallel_hidden(hidden, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let logits = match &mut self.output {
            Some(output) => output.forward(&hidden, &execution)?,
            None => self.embedding.project_logits(&hidden, &execution)?,
        }
        .all_gather(&execution)?;
        self.architecture
            .finish_parallel_logits(logits, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

enum Execution {
    Resident(Resident),
    Bounded(Bounded),
    ParallelResident(Box<ParallelResident>),
    ParallelBounded(Box<ParallelBounded>),
}

#[allow(clippy::too_many_arguments)]
fn forward_external_experts<P>(
    architecture: &mut NeutralArchitecture,
    group: usize,
    index: usize,
    unit: &mut NeutralUnit,
    hidden: &Array,
    state: &mut MlxHybridState,
    forward: &mut eredu_architectures::gemma4::ForwardContext<Array>,
    stream: &Stream,
    provider: &mut P,
) -> Result<Array, eredu_nn::Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    if group == 2 {
        let NeutralUnit::Text(block) = unit else {
            return Err(eredu_nn::Error::backend(
                "Gemma 4 text execution group received a media unit",
            ));
        };
        return architecture.forward_text_unit_with_provider(
            index,
            block,
            hidden,
            state,
            forward,
            if hidden.dim(1) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            provider,
            stream,
        );
    }
    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
}

/// One neutral Gemma 4 object shared by resident and bounded execution.
pub struct Gemma4Model {
    args: FamilyConfig,
    state_layout: eredu_runtime::StateLayout,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    execution: Execution,
    expert_cache: Option<ExpertCache>,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
}

/// Fully resident external assistant built from the neutral Gemma equations.
pub(crate) struct Gemma4AssistantModel {
    pub(crate) config: eredu_architectures::gemma4::AssistantConfig,
    module: MlxModule<NeutralAssistant>,
}

impl Gemma4AssistantModel {
    pub(crate) fn max_proposals(&self) -> usize {
        self.module.max_proposals()
    }

    pub(crate) fn begin_round(
        &self,
        shared_kv: eredu_architectures::gemma4::SharedAttentionStates<Array>,
        kv_offset: i32,
        hidden: Array,
    ) -> eredu_architectures::gemma4::AssistantState<Array> {
        self.module.begin_round(shared_kv, kv_offset, hidden)
    }

    pub(crate) fn draft_step(
        &mut self,
        embedding: &Array,
        state: &mut eredu_architectures::gemma4::AssistantState<Array>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.module
            .draft_step::<crate::backend::mlx::runtime::cache::kv::ConcatKeyValueCache>(
                embedding, state, stream,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

/// Loads the released SafeTensors assistant into the backend-neutral module.
pub(crate) fn load_assistant_safetensors(
    model_dir: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    let bytes = std::fs::read(model_dir.join("config.json"))?;
    let source_config = eredu_architectures::gemma4::AssistantConfig::from_json(&bytes)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let requested = options
        .quantization
        .map(|requested| {
            if source_config.use_ordered_embeddings {
                return Err(Error::Quantization(
                    "Gemma 4 ordered assistant heads cannot be quantized".into(),
                ));
            }
            should_quantize_on_load("Gemma 4 assistant", source_config.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut config = source_config.clone();
    if let Some(requested) = requested {
        if config.use_ordered_embeddings {
            return Err(Error::Quantization(
                "Gemma 4 ordered assistant heads cannot be quantized".into(),
            ));
        }
        config.quantization = Some(requested);
        config.text_config.weight_quantization = Some(requested);
    }
    let store =
        open_safetensors_weight_store(model_dir, options.weight_residency.max_mapped_shards())?;
    let store = if let Some(requested) = requested {
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0
    } else {
        store
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

pub(crate) fn load_assistant_gguf(
    gguf_file: &Path,
    options: crate::backend::mlx::ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4AssistantModel, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
            "Gemma 4 assistant loading supports fully resident weights only".into(),
        ));
    }
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "Gemma 4 assistant loading requires replicated placement".into(),
        ));
    }
    struct Catalog<'a>(&'a GgufCheckpoint);
    impl eredu_architectures::gemma4::GgufTensorCatalog for Catalog<'_> {
        fn contains(&self, name: &str) -> bool {
            self.0.contains_gguf_tensor(name)
        }
    }
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    let mut config = eredu_architectures::gemma4::AssistantConfig::from_gguf_metadata(
        &Catalog(&checkpoint),
        &metadata,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let formats = gguf_quantization_configs(
        &checkpoint,
        eredu_architectures::gemma4::translate_assistant_gguf_weight_name,
    )?;
    if !formats.is_empty() {
        config.text_config.quantized_weight_configs = Some(formats);
    }
    if config.use_ordered_embeddings && options.quantization.is_some() {
        return Err(Error::Quantization(
            "Gemma 4 ordered assistant heads cannot be quantized".into(),
        ));
    }
    crate::backend::mlx::validate_gguf_quantization_source(
        &checkpoint,
        &metadata,
        options.quantization,
    )?;
    let plan = eredu_architectures::gemma4::assistant_gguf_plan(&config)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: SharedCheckpointSource = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(options.weight_residency.max_mapped_shards())?
            .add_checkpoint(checkpoint.catalog().clone(), &plan, |name| {
                eredu_architectures::gemma4::translate_assistant_gguf_weight_name(name)
            })?
            .build()?,
    );
    let source_config = config.clone();
    let (store, config) = if let Some(requested) = options.quantization {
        config.quantization = Some(requested);
        config.text_config.weight_quantization = Some(requested);
        config.text_config.quantized_weight_configs = None;
        let source = NeutralAssistant::new(source_config, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let target = NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        (
            quantize_parameterized_module_store(store, &source, &target, requested, stream)?.0,
            config,
        )
    } else {
        (store, config)
    };
    let mut module = MlxModule::new(
        NeutralAssistant::new(config.clone(), stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    );
    let bindings = build_module_bindings(&module, "", store.as_ref())?;
    let arrays = materialize_module_bindings(store.as_ref(), &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut module, &arrays, |_| false)?;
    Ok(Gemma4AssistantModel { config, module })
}

/// Ordinary target outputs retained by the neutral speculative adapter.
pub(crate) struct Gemma4MtpOutput {
    pub(crate) logits: Array,
    pub(crate) hidden: Array,
    pub(crate) shared_kv: eredu_architectures::gemma4::SharedAttentionStates<Array>,
}

impl Gemma4Model {
    pub fn args(&self) -> &FamilyConfig {
        &self.args
    }

    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone())
            .expect("validated Gemma 4 state must be realizable")
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let rank = self.parallel_info.as_ref().and_then(|info| {
                    crate::backend::mlx::cache::prompt_cache_topology(info.topology())
                        .cache_rank_identity()
                });
                MlxHybridState::paged(
                    self.state_layout.clone(),
                    CacheResidencyManager::new(options)
                        .map_err(|error| Error::Parallel(error.to_string()))?,
                    rank,
                )
                .map_err(Into::into)
            }
        }
    }

    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.state_layout.layers().clone())
    }

    fn prompt_identity(&self) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(eredu_core::cache::PromptCacheTopology::default, |info| {
                crate::backend::mlx::cache::prompt_cache_topology(info.topology())
            });
        eredu_runtime::ModelStateIdentity {
            model_family: "gemma4".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: self.args.architecture_fingerprint(),
            layer_count: self.state_layout.len(),
            global_layer_start: 0,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; self.state_layout.len()],
            topology,
        }
        .prompt_cache_identity(&self.state_layout)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, eredu_core::cache::PromptCacheManifest), Error> {
        let identity = self.prompt_identity()?;
        let rank = identity.topology.cache_rank_identity();
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory.as_ref(), &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::paged(self.state_layout.clone(), manager, rank)?;
        let processed = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix length exceeds i32".into()))?;
        state.restore_prompt_cache_state(tensors, processed, &identity.layer_prefix_offsets)?;
        Ok((state, manifest))
    }

    pub fn save_prompt_cache(
        &self,
        state: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &eredu_core::cache::PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<eredu_core::cache::PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(
            &descriptor,
            &self.prompt_identity()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        state
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        let report = match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report()?,
            Execution::Bounded(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelResident(runtime) => runtime.policy().residency_report()?,
            Execution::ParallelBounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ParallelResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ParallelBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ParallelResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ParallelBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn forward_with_capture(
        &mut self,
        input: ModelInput<'_, Array>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<
        (
            Array,
            eredu_architectures::gemma4::ForwardContext<Array>,
            Array,
        ),
        Error,
    > {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel execution requires a collective session".into(),
            ));
        }
        if state.layout() != &self.state_layout {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 cache layout mismatch".into(),
            ));
        }
        let mut final_text_hidden = None;
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.text.clone();
            let mut provider =
                crate::composition::gemma4_expert::cached_provider(&expert_cache, &args);
            let result = match &mut self.execution {
                Execution::Resident(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        state,
                        stream,
                        |architecture, group, index, unit, hidden, state, forward, stream| {
                            forward_external_experts(
                                architecture,
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut provider,
                            )
                        },
                        |group, _index, hidden, _forward| {
                            if group == 2 {
                                final_text_hidden = Some(hidden.clone());
                            }
                            Ok(())
                        },
                    ),
                Execution::Bounded(runtime) => runtime
                    .forward_with_unit_executor_and_activation_hook(
                        input,
                        state,
                        stream,
                        |architecture, group, index, unit, hidden, state, forward, stream| {
                            forward_external_experts(
                                architecture,
                                group,
                                index,
                                unit,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut provider,
                            )
                        },
                        |group, _index, hidden, _forward| {
                            if group == 2 {
                                final_text_hidden = Some(hidden.clone());
                            }
                            Ok(())
                        },
                    ),
                Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
            };
            drop(provider);
            self.expert_cache = Some(expert_cache);
            let (logits, forward) =
                result.map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let hidden = final_text_hidden.ok_or_else(|| {
                Error::UnsupportedArchitecture("Gemma 4 text graph produced no activation".into())
            })?;
            return Ok((logits, forward, hidden));
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                state,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, _forward| {
                    if group == 2 {
                        final_text_hidden = Some(hidden.clone());
                    }
                    Ok(())
                },
            ),
            Execution::Bounded(runtime) => runtime.forward_with_unit_executor_and_activation_hook(
                input,
                state,
                stream,
                |architecture, group, index, unit, hidden, state, forward, stream| {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, stream)
                },
                |group, _index, hidden, _forward| {
                    if group == 2 {
                        final_text_hidden = Some(hidden.clone());
                    }
                    Ok(())
                },
            ),
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let hidden = final_text_hidden.ok_or_else(|| {
            Error::UnsupportedArchitecture("Gemma 4 text graph produced no activation".into())
        })?;
        Ok((result.0, result.1, hidden))
    }

    fn forward(
        &mut self,
        input: ModelInput<'_, Array>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward_with_capture(input, state, stream)
            .map(|(logits, _, _)| logits)
    }

    pub(crate) fn embed_mtp_token(&mut self, token: u32, stream: &Stream) -> Result<Array, Error> {
        if matches!(
            self.execution,
            Execution::ParallelResident(_) | Execution::ParallelBounded(_)
        ) {
            return Err(Error::Parallel(
                "Gemma 4 assistant embedding is unavailable in tensor-parallel execution".into(),
            ));
        }
        let tokens = Array::from_slice(&[token], &[1, 1]);
        match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.architecture_mut().token_embeddings(&tokens, stream)
            }
            Execution::Bounded(runtime) => {
                runtime.architecture_mut().token_embeddings(&tokens, stream)
            }
            Execution::ParallelResident(_) | Execution::ParallelBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn mtp_output(
        &mut self,
        input: ModelInput<'_, Array>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        let (logits, forward, hidden) = self.forward_with_capture(input, state, stream)?;
        Ok(Gemma4MtpOutput {
            logits,
            hidden,
            shared_kv: forward.shared_attention_states().clone(),
        })
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.mtp_output(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Gemma4MtpOutput, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.mtp_output(
            ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub fn forward_tokens(
        &mut self,
        tokens: &Array,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let parts = [DecoderInputPart::Text(tokens)];
        self.forward(
            ModelInput {
                parts: &parts,
                vision: None,
                audio: None,
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel cache layout mismatch".into(),
            ));
        }
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(Gemma4ParallelInput::Text(tokens), state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(Gemma4ParallelInput::Text(tokens), state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "Gemma 4 model was not loaded for tensor parallelism".into(),
            )),
        }
    }

    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if state.layout() != &self.state_layout {
            return Err(Error::Parallel(
                "Gemma 4 tensor-parallel cache layout mismatch".into(),
            ));
        }
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        let input = Gemma4ParallelInput::Prepared(ModelInput {
            parts: &parts,
            vision: prepared.vision_input(),
            audio: prepared.audio_input(),
            per_layer_tokens: None,
            mask: None,
        });
        match &mut self.execution {
            Execution::ParallelResident(runtime) => runtime
                .forward_parallel(input, state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::ParallelBounded(runtime) => runtime
                .forward_parallel(input, state, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            Execution::Resident(_) | Execution::Bounded(_) => Err(Error::Parallel(
                "Gemma 4 model was not loaded for tensor parallelism".into(),
            )),
        }
    }

    pub fn forward_input(
        &mut self,
        typed: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        input::validate(typed)?;
        let prepared = PreparedParts::new(&self.args, typed, stream)?;
        let parts = prepared.decoder_parts();
        self.forward(
            ModelInput {
                parts: &parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            stream,
        )
    }
}

struct PreparedVision {
    patches: Array,
    positions: Array,
    valid: Array,
    key_mask: Array,
    grid_extents: Vec<(i32, i32)>,
}

struct PreparedVisionPart {
    patches: Array,
    positions: Array,
    height: i32,
    width: i32,
}

struct PreparedAudio {
    features: Array,
    input_mask: Array,
    first_stage_mask: Array,
    valid: Vec<i32>,
}

struct PreparedAudioPart {
    features: Array,
    mask: Array,
    valid_frames: i32,
}

struct PreparedParts {
    tokens: Vec<Array>,
    modalities: Vec<input::Modality>,
    projected: Vec<Option<Array>>,
    vision_parts: Vec<PreparedVisionPart>,
    vision: Option<PreparedVision>,
    audio_parts: Vec<PreparedAudioPart>,
    audio: Option<PreparedAudio>,
}

impl PreparedParts {
    fn new(
        args: &FamilyConfig,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut prepared = Self {
            tokens: Vec::with_capacity(typed.parts.len()),
            modalities: Vec::with_capacity(typed.parts.len()),
            projected: Vec::with_capacity(typed.parts.len()),
            vision_parts: Vec::new(),
            vision: None,
            audio_parts: Vec::new(),
            audio: None,
        };
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    prepared.tokens.push(tokens.clone());
                    prepared.modalities.push(input::Modality::Text);
                    prepared.projected.push(None);
                }
                (
                    modality @ (input::Modality::Image | input::Modality::Video),
                    input::InputPayload::Tensor(patches),
                ) => prepared.push_vision(args, modality, patches, part.metadata)?,
                (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                    prepared.push_audio(args, features, part.metadata)?
                }
                (modality, input::InputPayload::Embeddings(embeddings)) => {
                    input::ensure_hidden_size(
                        embeddings,
                        args.text.hidden_size,
                        "Gemma 4 projected embeddings",
                    )?;
                    let token = modality_token(args, modality)?;
                    prepared.tokens.push(Array::from_slice(
                        &vec![token; embeddings.dim(1) as usize],
                        &[1, embeddings.dim(1)],
                    ));
                    prepared.modalities.push(modality);
                    prepared.projected.push(Some(embeddings.clone()));
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Gemma 4 does not accept this {} payload",
                        modality.as_str()
                    )))
                }
            }
        }
        prepared.finish_vision(stream)?;
        prepared.finish_audio(stream)?;
        Ok(prepared)
    }

    fn push_vision(
        &mut self,
        args: &FamilyConfig,
        modality: input::Modality,
        patches: &Array,
        metadata: input::InputMetadata<'_>,
    ) -> Result<(), Error> {
        let positions = metadata.patch_positions.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires patch positions",
                modality.as_str()
            ))
        })?;
        metadata.patch_grid.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires a prepared patch grid",
                modality.as_str()
            ))
        })?;
        let [time, height, width] = metadata.patch_extent.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} input requires a host-known patch extent",
                modality.as_str()
            ))
        })?;
        if time != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 {} parts must contain one prepared frame; got {time}",
                modality.as_str()
            )));
        }
        let pool = args
            .vision
            .as_ref()
            .ok_or_else(|| Error::UnsupportedArchitecture("Gemma 4 has no vision tower".into()))?
            .pooling_kernel_size;
        let count = (height / pool) * (width / pool);
        let token = modality_token(args, modality)?;
        self.tokens
            .push(Array::from_slice(&vec![token; count as usize], &[1, count]));
        self.modalities.push(modality);
        self.projected.push(None);

        self.vision_parts.push(PreparedVisionPart {
            patches: patches.clone(),
            positions: positions.clone(),
            height,
            width,
        });
        Ok(())
    }

    fn finish_vision(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.vision_parts.is_empty() {
            return Ok(());
        }
        let max_patches = self
            .vision_parts
            .iter()
            .map(|part| part.patches.dim(1))
            .max()
            .unwrap_or(0);
        let patches = self
            .vision_parts
            .iter()
            .map(|part| pad_sequence(&part.patches, max_patches, 0, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let positions = self
            .vision_parts
            .iter()
            .map(|part| pad_sequence(&part.positions, max_patches, -1, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let patch_refs = patches.iter().collect::<Vec<_>>();
        let position_refs = positions.iter().collect::<Vec<_>>();
        let patches = concatenate_axis(&patch_refs, 0, stream)?;
        let positions = concatenate_axis(&position_refs, 0, stream)?;
        let x = positions.try_index_device((.., .., 0), stream)?;
        let y = positions.try_index_device((.., .., 1), stream)?;
        let padding = x
            .eq(Array::from_int(-1), stream)?
            .logical_and(&y.eq(Array::from_int(-1), stream)?, stream)?;
        let sanitized = maximum(positions.clone(), Array::from_int(0), stream)?;
        let valid = padding
            .logical_not(stream)?
            .as_dtype(Dtype::Float32, stream)?
            .try_index_device((.., .., NewAxis), stream)?;
        let key_mask = padding
            .try_index_device((.., NewAxis, NewAxis, ..), stream)?
            .as_dtype(Dtype::Float32, stream)?
            .multiply(Array::from_f32(-1.0e9), stream)?;
        self.vision = Some(PreparedVision {
            patches,
            positions: sanitized,
            valid,
            key_mask,
            grid_extents: self
                .vision_parts
                .iter()
                .map(|part| (part.height, part.width))
                .collect(),
        });
        Ok(())
    }

    fn push_audio(
        &mut self,
        args: &FamilyConfig,
        features: &Array,
        metadata: input::InputMetadata<'_>,
    ) -> Result<(), Error> {
        let mask = metadata.audio_mask.ok_or_else(|| {
            Error::UnsupportedArchitecture("Gemma 4 audio input requires an audio mask".into())
        })?;
        let valid_frames = metadata.audio_valid_frames.ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Gemma 4 audio input requires a host-known valid-frame extent".into(),
            )
        })?;
        let valid = (valid_frames + 3) / 4;
        let token = modality_token(args, input::Modality::Audio)?;
        self.tokens
            .push(Array::from_slice(&vec![token; valid as usize], &[1, valid]));
        self.modalities.push(input::Modality::Audio);
        self.projected.push(None);

        self.audio_parts.push(PreparedAudioPart {
            features: features.clone(),
            mask: mask.clone(),
            valid_frames,
        });
        Ok(())
    }

    fn finish_audio(&mut self, stream: &Stream) -> Result<(), Error> {
        if self.audio_parts.is_empty() {
            return Ok(());
        }
        let max_frames = self
            .audio_parts
            .iter()
            .map(|part| part.features.dim(1))
            .max()
            .unwrap_or(0);
        let features = self
            .audio_parts
            .iter()
            .map(|part| pad_sequence(&part.features, max_frames, 0, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let masks = self
            .audio_parts
            .iter()
            .map(|part| {
                let extra = max_frames - part.mask.dim(1);
                if extra == 0 {
                    Ok(part.mask.clone())
                } else {
                    Ok(pad(
                        &part.mask,
                        PadWidth::from(&[(0, 0), (0, extra)][..]),
                        Array::from_bool(false),
                        None,
                        stream,
                    )?)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let feature_refs = features.iter().collect::<Vec<_>>();
        let mask_refs = masks.iter().collect::<Vec<_>>();
        let features = concatenate_axis(&feature_refs, 0, stream)?;
        let mask = concatenate_axis(&mask_refs, 0, stream)?;
        let input_mask = mask
            .as_dtype(features.dtype(), stream)?
            .try_index_device((.., .., NewAxis), stream)?;
        let first_frames = (max_frames + 1) / 2;
        let first_stage = self
            .audio_parts
            .iter()
            .map(|part| {
                let first_valid = (part.valid_frames + 1) / 2;
                let mut mask = vec![0.0f32; first_frames as usize];
                mask[..first_valid as usize].fill(1.0);
                Array::from_slice(&mask, &[1, first_frames, 1, 1])
            })
            .collect::<Vec<_>>();
        let first_refs = first_stage.iter().collect::<Vec<_>>();
        self.audio = Some(PreparedAudio {
            features,
            input_mask,
            first_stage_mask: concatenate_axis(&first_refs, 0, stream)?,
            valid: self
                .audio_parts
                .iter()
                .map(|part| (part.valid_frames + 3) / 4)
                .collect(),
        });
        Ok(())
    }

    fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, Array>> {
        self.tokens
            .iter()
            .zip(&self.modalities)
            .zip(&self.projected)
            .map(|((tokens, modality), embeddings)| {
                if let Some(embeddings) = embeddings {
                    DecoderInputPart::Projected { tokens, embeddings }
                } else {
                    match modality {
                        input::Modality::Text => DecoderInputPart::Text(tokens),
                        input::Modality::Image => DecoderInputPart::Image(tokens),
                        input::Modality::Video => DecoderInputPart::Video(tokens),
                        input::Modality::Audio => DecoderInputPart::Audio(tokens),
                    }
                }
            })
            .collect()
    }

    fn vision_input(&self) -> Option<VisionInput<'_, Array>> {
        self.vision.as_ref().map(|vision| VisionInput {
            patches: &vision.patches,
            position_ids: &vision.positions,
            position_valid: &vision.valid,
            key_mask: &vision.key_mask,
            grid_extents: &vision.grid_extents,
        })
    }

    fn audio_input(&self) -> Option<AudioInput<'_, Array>> {
        self.audio.as_ref().map(|audio| AudioInput {
            features: &audio.features,
            input_mask: &audio.input_mask,
            first_stage_mask: &audio.first_stage_mask,
            valid_subsampled_frames: &audio.valid,
        })
    }
}

fn pad_sequence(value: &Array, sequence: i32, fill: i32, stream: &Stream) -> Result<Array, Error> {
    let extra = sequence - value.dim(1);
    if extra < 0 {
        return Err(Error::UnsupportedArchitecture(
            "prepared media sequence exceeds its batch padding extent".into(),
        ));
    }
    if extra == 0 {
        return Ok(value.clone());
    }
    Ok(pad(
        value,
        PadWidth::from(&[(0, 0), (0, extra), (0, 0)][..]),
        Array::from_int(fill).as_dtype(value.dtype(), stream)?,
        None,
        stream,
    )?)
}

fn modality_token(args: &FamilyConfig, modality: input::Modality) -> Result<u32, Error> {
    match modality {
        input::Modality::Text => Some(args.text.pad_token_id),
        input::Modality::Image => args.image_token_id,
        input::Modality::Video => args.video_token_id,
        input::Modality::Audio => args.audio_token_id,
    }
    .and_then(|token| u32::try_from(token).ok())
    .ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Gemma 4 has no valid {} placeholder",
            modality.as_str()
        ))
    })
}

impl CausalModel<MlxHybridState> for Gemma4Model {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_input(input, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        tokens: &Array,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_tokens(tokens, state, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

fn execution_layout(args: &FamilyConfig) -> Result<ExecutionUnitLayout, Error> {
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![
            eredu_runtime::ExecutionGroupSpec::root("vision"),
            eredu_runtime::ExecutionGroupSpec::root("audio"),
            eredu_runtime::ExecutionGroupSpec::with_dependencies(
                "text_decoder",
                ["vision", "audio"],
            ),
        ],
        "text_decoder",
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(
        &graph,
        [
            args.vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            args.audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            args.text.num_hidden_layers(),
        ],
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn resolve_pipeline_store(
    store: SharedCheckpointSource,
    args: &FamilyConfig,
) -> Result<SharedCheckpointSource, Error> {
    let plan = eredu_architectures::gemma4::safetensors_plan(args)
        .map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|error| {
            Error::UnsupportedArchitecture(format!(
                "Gemma 4 checkpoint contract did not resolve: {error:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

pub(crate) fn load_pipeline_config(model_dir: &Path) -> Result<FamilyConfig, Error> {
    FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn quantize_store(
    store: SharedCheckpointSource,
    source: &FamilyConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        SharedCheckpointSource,
        FamilyConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target = source.clone();
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
    let source_architecture = NeutralArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target_architecture = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let factory = |args: &FamilyConfig| UnitFactory {
        args: args.clone(),
        vision_layers: args
            .vision
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        audio_layers: args
            .audio
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize),
        external_experts: false,
    };
    let mut source_factory = factory(source);
    let mut target_factory = factory(&target);
    let unit_count = source_factory
        .vision_layers
        .checked_add(source_factory.audio_layers)
        .and_then(|count| count.checked_add(source.text.num_hidden_layers()))
        .ok_or_else(|| Error::Quantization("Gemma 4 unit count overflowed".into()))?;
    let (store, report) = quantize_parameterized_store(
        store,
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        ),
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        ),
        move |index, stream| source_factory.build(index, stream),
        move |index, stream| target_factory.build(index, stream),
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

fn load_store(
    store: SharedCheckpointSource,
    args: FamilyConfig,
    residency: eredu_runtime::LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<Gemma4Model, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let vision_layers = args
        .vision
        .as_ref()
        .map_or(0, |config| config.num_hidden_layers as usize);
    let audio_layers = args
        .audio
        .as_ref()
        .map_or(0, |config| config.num_hidden_layers as usize);
    let binding_args = args.text.clone();
    let text_start = vision_layers + audio_layers;
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules_mut(
            &mut architecture,
        ),
        UnitFactory {
            args: args.clone(),
            vision_layers,
            audio_layers,
            external_experts,
        },
        execution_layout(&args)?,
        residency,
        stream,
        weights_stream,
        move |key| external_experts && key.contains(".experts.switch_glu."),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |ordinal, unit, store, _stream| {
            let recipes = if !external_experts && ordinal >= text_start {
                let layer = ordinal - text_start;
                if binding_args.layer_policy(layer).is_some_and(|policy| {
                    policy.feed_forward
                        == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
                }) {
                    let resolved = eredu_architectures::gemma4::expert_recipes(
                        store,
                        &binding_args,
                        "model.language_model.layers",
                        layer,
                    )
                    .map_err(Error::UnsupportedArchitecture)?;
                    BTreeMap::from([
                        (resolved.target_gate_up, resolved.gate_up),
                        (resolved.target_down, resolved.down),
                    ])
                } else {
                    BTreeMap::new()
                }
            } else {
                BTreeMap::new()
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".experts.switch_glu."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization);
    metadata.set_materialization(materialization);
    let execution = if residency.is_fully_resident() {
        Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(Gemma4Model {
        state_layout: eredu_architectures::gemma4::state_layout(&args.text)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        args,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: None,
    })
}

fn gemma4_unit_recipes(
    args: &eredu_architectures::gemma4::ModelArgs,
    layer: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    if args.layer_policy(layer).is_some_and(|policy| {
        policy.feed_forward == eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
    }) {
        let resolved = eredu_architectures::gemma4::expert_recipes(
            store,
            args,
            "model.language_model.layers",
            layer,
        )
        .map_err(Error::UnsupportedArchitecture)?;
        Ok(BTreeMap::from([
            (resolved.target_gate_up, resolved.gate_up),
            (resolved.target_down, resolved.down),
        ]))
    } else {
        Ok(BTreeMap::new())
    }
}

fn load_parallel_store(
    store: SharedCheckpointSource,
    args: FamilyConfig,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let layer_count = args.text.num_hidden_layers();
    let mut planner = build.planner();
    for group in eredu_architectures::gemma4::static_parameter_groups(&args.text)? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        for group in eredu_architectures::gemma4::layer_parameter_groups(&args.text, index)? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Gemma 4 declared no tensor-parallel parameters".into(),
        ));
    }
    let mut composition = Gemma4ParallelComposition::new(args.clone(), build, &layout, stream)?;
    let state_layout = eredu_architectures::gemma4::state_layout(&composition.local_text)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let local_args = Arc::clone(&composition.local_args);
    let unit_layout = ExecutionUnitLayout::new(
        &ExecutionGraph::chain(["text_decoder"])
            .map_err(|error| Error::Parallel(error.to_string()))?,
        [layer_count],
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;

    let global_architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let global_static = MlxModule::new(
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxHybridState>>::static_modules(
            &global_architecture,
        )
        .clone(),
    );
    let mut global_static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    if let Some(vision) = &args.vision {
        for index in 0..vision.num_hidden_layers as usize {
            let layer = MlxModule::new(
                eredu_architectures::gemma4::VisionLayer::<MlxBackend>::new(vision, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            global_static_bindings.extend(build_module_bindings(&layer, "", store.as_ref())?);
        }
    }
    if let Some(audio) = &args.audio {
        for index in 0..audio.num_hidden_layers as usize {
            let layer = MlxModule::new(
                eredu_architectures::gemma4::AudioLayer::<MlxBackend>::new(audio, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            global_static_bindings.extend(build_module_bindings(&layer, "", store.as_ref())?);
        }
    }
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for index in 0..layer_count {
        let unit = MlxModule::new(
            eredu_architectures::gemma4::DenseBlock::<MlxBackend>::new(&args.text, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        );
        let recipes = gemma4_unit_recipes(&args.text, index, store.as_ref())?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(
                &build_module_bindings_with_recipes_excluding(
                    &unit,
                    "",
                    store.as_ref(),
                    recipes,
                    |_| false,
                )?,
            )?)
            .ok_or_else(|| Error::Parallel("Gemma 4 global parameter bytes overflowed".into()))?;
    }

    let static_layout = Arc::new(layout);
    let unit_sharding = Arc::clone(&static_layout);
    let report_layout = Arc::clone(&static_layout);
    let binding_args = args.text.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut composition,
        ParallelUnitFactory { args: local_args },
        unit_layout,
        residency,
        stream,
        weights_stream,
        |_| false,
        move |_modules, store| {
            shard_layer_bindings(global_static_bindings, "", store, &static_layout)
        },
        move |index, _local, store, stream| {
            let global = MlxModule::new(
                eredu_architectures::gemma4::DenseBlock::<MlxBackend>::new(
                    &binding_args,
                    index,
                    stream,
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            let recipes = gemma4_unit_recipes(&binding_args, index, store)?;
            let bindings =
                build_module_bindings_with_recipes_excluding(&global, "", store, recipes, |_| {
                    false
                })?;
            shard_layer_bindings(
                bindings,
                &format!("model.language_model.layers.{index}"),
                store,
                &unit_sharding,
            )
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.text.weight_quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Gemma 4 local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Gemma 4 device parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        report_layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if residency.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let execution = if residency.is_fully_resident() {
        Execution::ParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        Execution::ParallelBounded(Box::new(LayerwiseRuntime::new(composition, policy)))
    };
    Ok(Gemma4Model {
        args,
        state_layout,
        metadata,
        execution,
        expert_cache: None,
        parallel_info: Some(parallel_info),
    })
}

pub(crate) fn load_safetensors_tensor_parallel(
    model_dir: impl AsRef<Path>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_pipeline_store(store, &args)?;
    load_parallel_store(store, args, residency, build, stream, weights_stream)
}

pub(crate) fn load_gguf_tensor_parallel(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Gemma4Model, Vec<u32>), Error> {
    let (store, args) = open_pipeline_gguf_store(
        gguf_file,
        checkpoint,
        metadata,
        residency.max_mapped_shards(),
    )?;
    let eos = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    Ok((
        load_parallel_store(store, args, residency, build, stream, weights_stream)?,
        eos,
    ))
}

fn attach_expert_cache(
    model: &mut Gemma4Model,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::gemma4_expert::expert_catalog(
        &model.args.text,
        store.as_ref(),
        stream,
    )?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors through one neutral family object and residency policy.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let model_dir = model_dir.as_ref();
    let args = FamilyConfig::from_hf_json(&std::fs::read(model_dir.join("config.json"))?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let store = open_safetensors_weight_store(model_dir, residency.max_mapped_shards())?;
    let store = resolve_pipeline_store(store, &args)?;
    let requested = quantization
        .map(|requested| {
            should_quantize_on_load("Gemma 4", args.text.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, args, materialization) = match requested {
        Some(quantization) => {
            let (store, args, report) = quantize_store(store, &args, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        materialization,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads a Gemma 4 decoder and optional sibling media projector through the
/// same neutral family object.
pub fn load_gguf(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4Model, Error> {
    let expert_options = residency.expert_cache();
    let (store, args) = open_pipeline_gguf_store(
        gguf_file,
        checkpoint,
        metadata,
        residency.max_mapped_shards(),
    )?;
    let mut model = load_store(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

pub(crate) fn open_pipeline_gguf_store(
    gguf_file: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_cached_readers: usize,
) -> Result<(SharedCheckpointSource, FamilyConfig), Error> {
    let projector = find_sibling_mmproj(gguf_file, "gemma4")?
        .map(GgufCheckpoint::open)
        .transpose()?;
    let projector_metadata = projector
        .as_ref()
        .map(crate::backend::mlx::runtime::checkpoint::load::gguf_metadata);
    if let Some(metadata) = projector_metadata.as_ref() {
        eredu_architectures::gemma4::validate_projector_identity(metadata)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    }
    let names = checkpoint
        .catalog()
        .tensors()
        .flat_map(|tensor| tensor.outputs())
        .map(|output| output.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut text = eredu_architectures::gemma4::ModelArgs::from_gguf_metadata(&names, metadata)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    text.quantized_weight_configs = Some(gguf_quantization_configs(
        checkpoint,
        eredu_architectures::gemma4::translate_gguf_weight_name,
    )?);
    let args = eredu_architectures::gemma4::family_from_gguf_metadata(
        text,
        metadata,
        projector_metadata.as_ref(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    if let Some(projector) = projector.as_ref() {
        let quantized = gguf_quantization_configs(
            projector,
            eredu_architectures::gemma4::translate_mmproj_weight_name,
        )?;
        if !quantized.is_empty() {
            return Err(Error::UnsupportedArchitecture(
                "Gemma 4 projector GGUF admits only dense F16, BF16, or F32 tensors".into(),
            ));
        }
        let tokens = [
            args.image_token_id,
            args.video_token_id,
            args.audio_token_id,
        ]
        .into_iter()
        .flatten()
        .filter_map(|token| u32::try_from(token).ok())
        .collect::<std::collections::BTreeSet<_>>();
        eredu_architectures::gemma4::Gemma4ArtifactConfig {
            unified: args.model_type == "gemma4_unified",
            hidden_size: args.text.hidden_size as usize,
            image_token_id: args.image_token_id.and_then(|token| token.try_into().ok()),
            video_token_id: args.video_token_id.and_then(|token| token.try_into().ok()),
            audio_token_id: args.audio_token_id.and_then(|token| token.try_into().ok()),
            projector: true,
            assistant: false,
        }
        .projector_compatibility(
            args.model_type.clone(),
            args.text.hidden_size as usize,
            tokens,
        )
        .validate()
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    }
    let plan = eredu_architectures::gemma4::gguf_plan(&args.text)
        .map_err(Error::UnsupportedArchitecture)?;
    let builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_cached_readers)?
        .add_checkpoint(checkpoint.catalog().clone(), &plan, |name| {
            eredu_architectures::gemma4::translate_gguf_weight_name(name)
        })?;
    let builder = if let Some(projector) = projector.as_ref() {
        let plan = eredu_architectures::gemma4::mmproj_gguf_plan(&args)
            .map_err(Error::UnsupportedArchitecture)?;
        builder.add_checkpoint(projector.catalog().clone(), &plan, |name| {
            eredu_architectures::gemma4::translate_mmproj_weight_name(name)
        })?
    } else {
        builder
    };
    let store: SharedCheckpointSource = Arc::new(builder.build()?);
    Ok((store, args))
}
