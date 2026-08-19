//! Bounded layer execution for Moshi and PersonaPlex realtime token models.

use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, LayerWeightResidency, LayeredArchitecture,
    LayeredForwardState, LayerwiseRuntime, ParallelLayeredArchitecture, ResidencyReport,
    ResidentLayerGroupReport, WeightBinding,
};

use eredu_checkpoint::WeightQuantization;
use std::{collections::BTreeMap, path::Path, sync::Arc};

use safemlx::{
    error::Exception,
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, stack_axis, zeros},
    random::RandomState,
    Array, Stream,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxParameterTree},
    backend::mlx::runtime::checkpoint::artifact::LoadedArtifactIdentity,
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, build_module_bindings,
    },
    backend::mlx::runtime::checkpoint::store::TensorSelection,
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, recipe::DerivedWeightRecipe,
    },
    backend::mlx::runtime::execution::{
        generic::{
            prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            MlxUnitFactory,
        },
        layerwise::{
            open_safetensors_weight_store, quantize_module_store_with_bindings,
            shard_layer_bindings,
        },
    },
    backend::mlx::runtime::generation::sampler::Sampler,
    composition::mlx::realtime::MlxRealtimeOutput,
    composition::mlx_architectures::moshi::model::{
        self as resident, DepFormerSlice, MoshiLayerwiseStatic, MoshiTransformerLayer,
    },
    core::realtime::RealtimeSpeechConfig,
};

pub use crate::composition::mlx_architectures::moshi::model::{
    GenerationState, GenerationStepWithLogits, ModelArgs, MoshiCache, SampleStepOutput,
    TokenStepOutput,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CheckpointLayout {
    Native,
    Pytorch,
}

/// Moshi-family model with independent temporal and depth-codebook residency windows.
pub struct MoshiLayerwiseModel {
    execution: MoshiExecution,
    artifact_identity: LoadedArtifactIdentity,
}

type MoshiUnit = MlxParameterTree<MoshiExecutionUnit>;
type MoshiStatic = MlxParameterTree<MoshiLayerwiseStatic>;
type MoshiResidentRuntime =
    LayerwiseRuntime<MoshiArchitecture, MlxBackend, MoshiCache, MlxResidentPolicy<MoshiUnit>>;
type MoshiLayerwiseRuntime = LayerwiseRuntime<
    MoshiArchitecture,
    MlxBackend,
    MoshiCache,
    MlxLayerwisePolicy<MoshiUnit, MoshiUnitFactory>,
>;

enum MoshiExecution {
    Resident(MoshiResidentRuntime),
    Layerwise(MoshiLayerwiseRuntime),
}

impl MoshiExecution {
    fn forward_with_context_hook<'a, H>(
        &mut self,
        input: MoshiLayerwiseInput<'a>,
        cache: &mut MoshiCache,
        stream: &Stream,
        hook: H,
    ) -> Result<(Array, MoshiForwardContext), eredu_runtime::LayerwiseRuntimeError<Error, Error>>
    where
        H: FnMut(usize, usize, &mut MoshiForwardContext) -> Result<(), Error>,
    {
        match self {
            Self::Resident(execution) => {
                execution.forward_with_context_hook(input, cache, stream, hook)
            }
            Self::Layerwise(execution) => {
                execution.forward_with_context_hook(input, cache, stream, hook)
            }
        }
    }

    fn forward_parallel_with_context_hook<'a, H>(
        &mut self,
        input: MoshiLayerwiseInput<'a>,
        cache: &mut MoshiCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        hook: H,
    ) -> Result<(Array, MoshiForwardContext), eredu_runtime::LayerwiseRuntimeError<Error, Error>>
    where
        H: FnMut(usize, usize, &mut MoshiForwardContext) -> Result<(), Error>,
    {
        match self {
            Self::Resident(execution) => {
                execution.forward_parallel_with_context_hook(input, cache, group, stream, hook)
            }
            Self::Layerwise(execution) => {
                execution.forward_parallel_with_context_hook(input, cache, group, stream, hook)
            }
        }
    }
}

#[derive(Clone)]
struct MoshiUnitFactory {
    args: ModelArgs,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
}

impl MlxUnitFactory<MoshiUnit> for MoshiUnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<MoshiUnit, Error> {
        let (group, index) = moshi_unit_address(&self.args, ordinal)?;
        let unit = build_moshi_unit(
            &self.args,
            group,
            index,
            self.parallel_layout.as_deref(),
            stream,
        )?;
        MlxParameterTree::new(unit, "").map_err(|error| Error::Parallel(error.to_string()))
    }
}

impl MoshiLayerwiseModel {
    pub(crate) fn with_artifact_identity(mut self, identity: LoadedArtifactIdentity) -> Self {
        self.artifact_identity = identity;
        self
    }

    pub(crate) fn artifact_identity(&self) -> &LoadedArtifactIdentity {
        &self.artifact_identity
    }

    /// Returns the parsed Moshi-family configuration.
    pub fn args(&self) -> &ModelArgs {
        match &self.execution {
            MoshiExecution::Resident(execution) => execution.architecture().args(),
            MoshiExecution::Layerwise(execution) => execution.architecture().args(),
        }
    }

    /// Allocates empty temporal and within-frame depth caches.
    pub fn new_cache(&self) -> MoshiCache {
        new_cache(self.args())
    }

    /// Returns the codec-token stream geometry consumed by realtime scheduling.
    pub fn realtime_config(&self) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            self.args().n_q as usize,
            self.args().input_audio_codebooks() as usize,
            self.args().generated_audio_codebooks() as usize,
            self.args().dep_q as usize,
            self.args().text_padding_token(),
            self.args().audio_padding_token(),
            self.args()
                .audio_delays()
                .iter()
                .map(|delay| *delay as usize)
                .collect(),
        )
        .expect("validated Moshi arguments have valid realtime geometry")
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            MoshiExecution::Resident(execution) => execution.policy().residency_report(),
            MoshiExecution::Layerwise(execution) => execution.policy().residency_report(),
        }
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            MoshiExecution::Resident(_) => Ok(None),
            MoshiExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
        }
    }

    /// Returns residency attributed to the temporal and depth execution groups.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        match &self.execution {
            MoshiExecution::Resident(execution) => execution.policy().execution_group_reports(),
            MoshiExecution::Layerwise(execution) => execution.policy().execution_group_reports(),
        }
    }

    /// Clears one temporary execution group without affecting the other group.
    pub fn clear_device_group(&self, group: &str) -> Result<(), Error> {
        match &self.execution {
            MoshiExecution::Resident(execution) => execution.policy().clear_device_group(group),
            MoshiExecution::Layerwise(execution) => execution.policy().clear_device_group(group),
        }
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            MoshiExecution::Resident(execution) => execution.policy().checkpoint_store(),
            MoshiExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
        }
    }

    /// Runs one frame with teacher-forced depth inputs.
    pub fn token_step(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        depth_tokens: &Array,
        cache: &mut MoshiCache,
        stream: &Stream,
    ) -> Result<TokenStepOutput, Exception> {
        let (_, context) = self
            .execution
            .forward_with_context_hook(
                MoshiLayerwiseInput::TeacherForced {
                    text_token,
                    audio_tokens,
                    depth_tokens,
                },
                cache,
                stream,
                |_, _, _| Ok(()),
            )
            .map_err(layerwise_exception)?;
        context.into_token_output()
    }

    /// Runs one teacher-forced frame through rank-local temporal and depth groups.
    pub fn token_step_tensor_parallel(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        depth_tokens: &Array,
        cache: &mut MoshiCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<TokenStepOutput, Exception> {
        let (_, context) = self
            .execution
            .forward_parallel_with_context_hook(
                MoshiLayerwiseInput::TeacherForced {
                    text_token,
                    audio_tokens,
                    depth_tokens,
                },
                cache,
                group,
                stream,
                |_, _, _| Ok(()),
            )
            .map_err(layerwise_exception)?;
        context.into_token_output()
    }

    /// Runs one frame with caller-provided text and audio samplers.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_step<TS: Sampler, AS: Sampler>(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        cache: &mut MoshiCache,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<SampleStepOutput, Exception> {
        self.sample_step_forced(
            text_token,
            audio_tokens,
            cache,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            None,
            None,
            None,
            prng_state,
            stream,
        )
    }

    /// Runs one autoregressive frame through rank-local temporal and depth groups.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_step_tensor_parallel<TS: Sampler, AS: Sampler>(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        cache: &mut MoshiCache,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        mut prng_state: Option<&mut RandomState>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<SampleStepOutput, Exception> {
        let depth_count = self.args().dep_q as usize;
        let temporal_layers = self.args().num_layers as usize;
        if audio_samplers.len() != depth_count {
            return Err(Exception::custom(format!(
                "Moshi requires one audio sampler per generated codebook (expected {depth_count}, got {})",
                audio_samplers.len()
            )));
        }
        let (_, context) = self
            .execution
            .forward_parallel_with_context_hook(
                MoshiLayerwiseInput::Autoregressive {
                    text_token,
                    audio_tokens,
                    forced_text_token: None,
                    forced_audio_tokens: None,
                    forced_audio_codebooks: None,
                },
                cache,
                group,
                stream,
                |execution_group, index, context| {
                    if execution_group == 0 && index + 1 == temporal_layers {
                        let local = if group.rank() == 0 {
                            text_sampler.sample(
                                context
                                    .text_logits
                                    .as_ref()
                                    .expect("last temporal layer logits"),
                                text_temperature,
                                prng_state.as_deref_mut(),
                                stream,
                            )?
                        } else {
                            zeros::<u32>(&[text_token.dim(0), 1], stream)?
                        };
                        let text = safemlx::distributed::all_sum(&local, group, stream)?;
                        context.previous = Some(text.clone());
                        context.sampled_text = Some(text);
                    } else if execution_group == 1 {
                        let local = if group.rank() == 0 {
                            audio_samplers[index].sample(
                                context.current_audio_logits.as_ref().expect("depth logits"),
                                audio_temperature,
                                prng_state.as_deref_mut(),
                                stream,
                            )?
                        } else {
                            zeros::<u32>(&[text_token.dim(0), 1], stream)?
                        };
                        let next = safemlx::distributed::all_sum(&local, group, stream)?;
                        context
                            .predicted_audio
                            .push(next.squeeze_axes(&[-1], stream)?);
                        context.previous = Some(next);
                    }
                    Ok(())
                },
            )
            .map_err(layerwise_exception)?;
        let text = context
            .sampled_text
            .as_ref()
            .expect("autoregressive text token")
            .clone();
        let audio = stack_axis(&context.predicted_audio, 1, stream)?;
        Ok(SampleStepOutput {
            text_token: text,
            audio_tokens: audio,
            logits: context.into_token_output()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_step_forced<TS: Sampler, AS: Sampler>(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        cache: &mut MoshiCache,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        forced_text_token: Option<&Array>,
        forced_audio_tokens: Option<&Array>,
        forced_audio_codebooks: Option<&[bool]>,
        mut prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<SampleStepOutput, Exception> {
        let depth_count = self.args().dep_q as usize;
        let temporal_layers = self.args().num_layers as usize;
        if audio_samplers.len() != depth_count {
            return Err(Exception::custom(format!(
                "Moshi requires one audio sampler per generated codebook (expected {depth_count}, got {})",
                audio_samplers.len()
            )));
        }
        validate_forced_depth(
            forced_audio_tokens,
            forced_audio_codebooks,
            text_token.dim(0),
            depth_count,
        )?;
        if let Some(token) = forced_text_token {
            if token.shape() != [text_token.dim(0), 1] {
                return Err(Exception::custom(format!(
                    "Moshi forced text token must have shape [batch, 1], got {:?}",
                    token.shape()
                )));
            }
        }

        let (_, context) = self
            .execution
            .forward_with_context_hook(
                MoshiLayerwiseInput::Autoregressive {
                    text_token,
                    audio_tokens,
                    forced_text_token,
                    forced_audio_tokens,
                    forced_audio_codebooks,
                },
                cache,
                stream,
                |group, index, context| {
                    if group == 0 && index + 1 == temporal_layers {
                        let sampled = text_sampler.sample(
                            context
                                .text_logits
                                .as_ref()
                                .expect("last temporal layer logits"),
                            text_temperature,
                            prng_state.as_deref_mut(),
                            stream,
                        )?;
                        let text = context
                            .forced_text_token
                            .as_ref()
                            .cloned()
                            .unwrap_or(sampled);
                        context.previous = Some(text.clone());
                        context.sampled_text = Some(text);
                    } else if group == 1 {
                        let forced = context
                            .forced_audio_codebooks
                            .as_ref()
                            .filter(|mask| mask[index])
                            .and(context.forced_audio_tokens.as_ref())
                            .map(|tokens| tokens.try_index_device((.., index as i32), stream))
                            .transpose()?;
                        let next = match forced {
                            Some(token) => token.expand_dims(1, stream)?,
                            None => audio_samplers[index].sample(
                                context.current_audio_logits.as_ref().expect("depth logits"),
                                audio_temperature,
                                prng_state.as_deref_mut(),
                                stream,
                            )?,
                        };
                        context
                            .predicted_audio
                            .push(next.squeeze_axes(&[-1], stream)?);
                        context.previous = Some(next);
                    }
                    Ok(())
                },
            )
            .map_err(layerwise_exception)?;

        let text = context
            .sampled_text
            .as_ref()
            .expect("autoregressive text token")
            .clone();
        let audio = stack_axis(&context.predicted_audio, 1, stream)?;
        Ok(SampleStepOutput {
            text_token: text,
            audio_tokens: audio,
            logits: context.into_token_output()?,
        })
    }

    /// Runs one frame with greedy sampling.
    pub fn greedy_step(
        &mut self,
        text_token: &Array,
        audio_tokens: &Array,
        cache: &mut MoshiCache,
        stream: &Stream,
    ) -> Result<resident::SampleStepOutput, Exception> {
        let mut text_sampler = crate::backend::mlx::runtime::generation::sampler::DefaultSampler;
        let mut audio_samplers = (0..self.args().dep_q)
            .map(|_| crate::backend::mlx::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        self.sample_step(
            text_token,
            audio_tokens,
            cache,
            &mut text_sampler,
            &mut audio_samplers,
            0.0,
            0.0,
            None,
            stream,
        )
    }

    /// Creates a fresh delayed-stream realtime session.
    pub fn new_generation_state(&self) -> resident::GenerationState {
        resident::GenerationState {
            cache: self.new_cache(),
            frames: Vec::new(),
            previous_text: None,
            step: 0,
        }
    }

    /// Consumes one encoded input-side frame and advances generation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_step<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut resident::GenerationState,
        input_audio_tokens: &Array,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxRealtimeOutput, Exception> {
        self.generate_step_forced(
            state,
            input_audio_tokens,
            None,
            None,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            prng_state,
            stream,
        )
    }

    /// Advances generation with optional forced generated audio and text.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_step_forced<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxRealtimeOutput, Exception> {
        Ok(self
            .generate_step_forced_with_logits(
                state,
                input_audio_tokens,
                forced_generated_audio_tokens,
                forced_text_token,
                text_sampler,
                audio_samplers,
                text_temperature,
                audio_temperature,
                prng_state,
                stream,
            )?
            .output)
    }

    /// Advances generation while retaining the text and audio decision logits.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_step_forced_with_logits<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepWithLogits, Exception> {
        if self.args().existing_text_padding_id.is_some() && self.args().dep_q == self.args().n_q {
            return self.generate_step_pytorch_style_with_logits(
                state,
                input_audio_tokens,
                forced_generated_audio_tokens,
                forced_text_token,
                text_sampler,
                audio_samplers,
                text_temperature,
                audio_temperature,
                prng_state,
                stream,
            );
        }
        self.generate_step_native_style_with_logits(
            state,
            input_audio_tokens,
            forced_generated_audio_tokens,
            forced_text_token,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            prng_state,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_step_native_style_with_logits<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepWithLogits, Exception> {
        let args = self.args().clone();
        let input_codebooks = args.input_audio_codebooks();
        if input_audio_tokens.shape().len() != 2 || input_audio_tokens.dim(1) != input_codebooks {
            return Err(Exception::custom(format!(
                "Moshi encoded input must have shape [batch, {input_codebooks}], got {:?}",
                input_audio_tokens.shape()
            )));
        }
        let batch = input_audio_tokens.dim(0);
        let generated_codebooks = args.generated_audio_codebooks();
        validate_generated_audio(forced_generated_audio_tokens, batch, generated_codebooks)?;

        let mut frame = vec![None; args.n_q as usize];
        for codebook in 0..input_codebooks {
            frame[(generated_codebooks + codebook) as usize] = Some(
                input_audio_tokens
                    .try_index_device((.., codebook), stream)?
                    .expand_dims(1, stream)?,
            );
        }
        if let Some(tokens) = forced_generated_audio_tokens {
            for codebook in 0..generated_codebooks {
                frame[codebook as usize] = Some(
                    tokens
                        .try_index_device((.., codebook), stream)?
                        .expand_dims(1, stream)?,
                );
            }
        }
        state.frames.push(frame);

        let padding = Array::full::<i32>(
            &[batch, 1],
            Array::from_int(args.audio_padding_token()),
            stream,
        )?;
        let mut delayed = Vec::with_capacity(args.n_q as usize);
        for (codebook, &delay) in args.audio_delays().iter().enumerate() {
            let source = state.step as isize - 1 - delay as isize;
            delayed.push(if source < 0 {
                padding.clone()
            } else {
                state.frames[source as usize][codebook]
                    .as_ref()
                    .ok_or_else(|| {
                        Exception::custom(format!(
                            "Moshi delayed stream is missing codebook {codebook} at frame {source}"
                        ))
                    })?
                    .clone()
            });
        }
        let delayed = safemlx::ops::concatenate_axis(&delayed, 1, stream)?;
        let text_input = state.previous_text.clone().unwrap_or(Array::full::<i32>(
            &[batch, 1],
            Array::from_int(args.text_padding_token()),
            stream,
        )?);

        let mut forced_depth = Vec::new();
        let mut forced_mask = vec![false; args.dep_q as usize];
        if forced_generated_audio_tokens.is_some() || args.dep_q > generated_codebooks {
            for codebook in 0..args.dep_q {
                if codebook < generated_codebooks {
                    if let Some(tokens) = forced_generated_audio_tokens {
                        forced_depth.push(tokens.try_index_device((.., codebook), stream)?);
                        forced_mask[codebook as usize] = true;
                    } else {
                        forced_depth.push(Array::zeros::<i32>(&[batch], stream)?);
                    }
                } else {
                    let input_index = codebook - generated_codebooks;
                    if input_index < input_codebooks {
                        forced_depth
                            .push(input_audio_tokens.try_index_device((.., input_index), stream)?);
                        forced_mask[codebook as usize] = true;
                    } else {
                        forced_depth.push(Array::zeros::<i32>(&[batch], stream)?);
                    }
                }
            }
        }
        let forced_depth = if forced_depth.is_empty() {
            None
        } else {
            Some(stack_axis(&forced_depth, 1, stream)?)
        };
        let sampled = self.sample_step_forced(
            &text_input,
            &delayed,
            &mut state.cache,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            forced_text_token,
            forced_depth.as_ref(),
            forced_depth.as_ref().map(|_| forced_mask.as_slice()),
            prng_state,
            stream,
        )?;

        for (codebook, &delay) in args
            .audio_delays()
            .iter()
            .take(generated_codebooks as usize)
            .enumerate()
        {
            let target = state.step as isize - delay as isize;
            if target >= 0 {
                state.frames[target as usize][codebook] = Some(
                    forced_generated_audio_tokens
                        .unwrap_or(&sampled.audio_tokens)
                        .try_index_device((.., codebook as i32), stream)?
                        .expand_dims(1, stream)?,
                );
            }
        }
        let max_delay = args.audio_delays().iter().copied().max().unwrap_or(0) as usize;
        let output_audio_tokens = state
            .step
            .checked_sub(max_delay)
            .map(|index| {
                let tokens = state.frames[index]
                    .iter()
                    .take(generated_codebooks as usize)
                    .map(|token| {
                        token.clone().ok_or_else(|| {
                            Exception::custom(format!(
                                "Moshi generated stream is incomplete at frame {index}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                safemlx::ops::concatenate_axis(&tokens, 1, stream)
            })
            .transpose()?;
        state.previous_text = Some(sampled.text_token.clone());
        state.step += 1;
        Ok(GenerationStepWithLogits {
            output: MlxRealtimeOutput {
                text_token: sampled.text_token,
                sampled_audio_tokens: sampled.audio_tokens,
                output_audio_tokens,
            },
            text_logits: Some(sampled.logits.text_logits),
            audio_logits: sampled.logits.audio_logits,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_step_pytorch_style_with_logits<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<GenerationStepWithLogits, Exception> {
        let args = self.args().clone();
        let input_codebooks = args.input_audio_codebooks();
        if input_audio_tokens.shape().len() != 2 || input_audio_tokens.dim(1) != input_codebooks {
            return Err(Exception::custom(format!(
                "Moshi encoded input must have shape [batch, {input_codebooks}], got {:?}",
                input_audio_tokens.shape()
            )));
        }
        let batch = input_audio_tokens.dim(0);
        let generated_codebooks = args.generated_audio_codebooks();
        validate_generated_audio(forced_generated_audio_tokens, batch, generated_codebooks)?;
        if let Some(token) = forced_text_token {
            if token.shape() != [batch, 1] {
                return Err(Exception::custom(format!(
                    "Moshi forced text token must have shape [batch, 1], got {:?}",
                    token.shape()
                )));
            }
        }

        let slots = args.n_q as usize + 1;
        let offset = state.step;
        for codebook in 0..input_codebooks {
            let slot = 1 + generated_codebooks + codebook;
            let position = offset + args.delays[slot as usize] as usize;
            ensure_token_position(&mut state.frames, position, slots);
            state.frames[position][slot as usize] = Some(
                input_audio_tokens
                    .try_index_device((.., codebook), stream)?
                    .expand_dims(1, stream)?,
            );
        }
        if let Some(tokens) = forced_generated_audio_tokens {
            for codebook in 0..generated_codebooks {
                let slot = 1 + codebook;
                let position = offset + args.delays[slot as usize] as usize;
                ensure_token_position(&mut state.frames, position, slots);
                state.frames[position][slot as usize] = Some(
                    tokens
                        .try_index_device((.., codebook), stream)?
                        .expand_dims(1, stream)?,
                );
            }
        }
        if let Some(token) = forced_text_token {
            let position = offset + args.delays[0] as usize;
            ensure_token_position(&mut state.frames, position, slots);
            state.frames[position][0] = Some(token.clone());
        }
        ensure_token_position(&mut state.frames, offset, slots);
        for (slot, &delay) in args.delays.iter().enumerate() {
            if offset <= delay as usize {
                state.frames[offset][slot] = Some(if slot == 0 {
                    Array::full::<i32>(&[batch, 1], Array::from_int(args.text_card), stream)?
                } else {
                    Array::full::<i32>(
                        &[batch, 1],
                        Array::from_int(args.audio_padding_token()),
                        stream,
                    )?
                });
            }
        }
        if offset == 0 {
            state.step += 1;
            return Ok(GenerationStepWithLogits {
                output: MlxRealtimeOutput {
                    text_token: Array::full::<i32>(
                        &[batch, 1],
                        Array::from_int(args.text_card),
                        stream,
                    )?,
                    sampled_audio_tokens: Array::full::<i32>(
                        &[batch, args.dep_q],
                        Array::from_int(args.audio_padding_token()),
                        stream,
                    )?,
                    output_audio_tokens: None,
                },
                text_logits: None,
                audio_logits: Vec::new(),
            });
        }

        let input_position = offset - 1;
        let target_position = offset;
        let text_input = token_position(&state.frames, input_position, 0)?;
        let mut audio_inputs = Vec::with_capacity(args.n_q as usize);
        for slot in 1..=args.n_q as usize {
            audio_inputs.push(token_position(&state.frames, input_position, slot)?);
        }
        let audio_input = safemlx::ops::concatenate_axis(&audio_inputs, 1, stream)?;
        ensure_token_position(&mut state.frames, target_position, slots);
        let forced_text = state.frames[target_position][0].clone();
        let mut forced_depth = Vec::with_capacity(args.dep_q as usize);
        let mut forced_mask = vec![false; args.dep_q as usize];
        for codebook in 0..args.dep_q {
            let slot = 1 + codebook as usize;
            if let Some(token) = &state.frames[target_position][slot] {
                forced_depth.push(token.squeeze_axes(&[-1], stream)?);
                forced_mask[codebook as usize] = true;
            } else {
                forced_depth.push(Array::zeros::<i32>(&[batch], stream)?);
            }
        }
        let forced_depth = stack_axis(&forced_depth, 1, stream)?;
        let sampled = self.sample_step_forced(
            &text_input,
            &audio_input,
            &mut state.cache,
            text_sampler,
            audio_samplers,
            text_temperature,
            audio_temperature,
            forced_text.as_ref(),
            Some(&forced_depth),
            Some(&forced_mask),
            prng_state,
            stream,
        )?;
        if state.frames[target_position][0].is_none() {
            state.frames[target_position][0] = Some(sampled.text_token.clone());
        }
        for codebook in 0..args.dep_q {
            let slot = 1 + codebook as usize;
            if state.frames[target_position][slot].is_none() {
                state.frames[target_position][slot] = Some(
                    sampled
                        .audio_tokens
                        .try_index_device((.., codebook), stream)?
                        .expand_dims(1, stream)?,
                );
            }
        }
        let max_delay = args.delays.iter().copied().max().unwrap_or(0) as usize;
        let output_audio_tokens = if offset <= max_delay {
            None
        } else {
            let base = offset - max_delay;
            let mut tokens = Vec::with_capacity(generated_codebooks as usize);
            for codebook in 0..generated_codebooks {
                let slot = 1 + codebook as usize;
                let position = base + args.delays[slot] as usize;
                tokens.push(token_position(&state.frames, position, slot)?);
            }
            Some(safemlx::ops::concatenate_axis(&tokens, 1, stream)?)
        };
        state.previous_text = Some(sampled.text_token.clone());
        state.step += 1;
        Ok(GenerationStepWithLogits {
            output: MlxRealtimeOutput {
                text_token: sampled.text_token,
                sampled_audio_tokens: sampled.audio_tokens,
                output_audio_tokens,
            },
            text_logits: Some(sampled.logits.text_logits),
            audio_logits: sampled.logits.audio_logits,
        })
    }
}

fn validate_generated_audio(
    tokens: Option<&Array>,
    batch: i32,
    generated_codebooks: i32,
) -> Result<(), Exception> {
    if let Some(tokens) = tokens {
        if tokens.shape() != [batch, generated_codebooks] {
            return Err(Exception::custom(format!(
                "Moshi forced generated audio must have shape [batch, {generated_codebooks}], got {:?}",
                tokens.shape()
            )));
        }
    }
    Ok(())
}

fn ensure_token_position(frames: &mut Vec<Vec<Option<Array>>>, position: usize, slots: usize) {
    while frames.len() <= position {
        frames.push(vec![None; slots]);
    }
}

fn token_position(
    frames: &[Vec<Option<Array>>],
    position: usize,
    slot: usize,
) -> Result<Array, Exception> {
    frames
        .get(position)
        .and_then(|frame| frame.get(slot))
        .and_then(Clone::clone)
        .ok_or_else(|| {
            Exception::custom(format!(
                "Moshi delayed stream is missing slot {slot} at position {position}"
            ))
        })
}

/// Loads a native MLX-layout Moshi checkpoint through bounded layer residency.
pub fn load_moshi_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let source = super::checkpoint::source_path(model_dir, &args);
    load_with_layout(
        source,
        args,
        CheckpointLayout::Native,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads native Moshi with rank-local temporal and depth transformers.
#[cfg(test)]
pub(crate) fn load_moshi_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let source = super::checkpoint::source_path(model_dir, &args);
    super::checkpoint::validate_safetensors_path(&source, &args)?;
    load_parallel_with_layout(
        open_safetensors_weight_store(&source, options.max_mapped_shards())?,
        args,
        CheckpointLayout::Native,
        options,
        build,
        stream,
        weights_stream,
    )
}

/// Loads the released PersonaPlex PyTorch checkpoint through bounded layer residency.
pub fn load_personaplex_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let metadata =
        crate::composition::mlx_architectures::moshi::personaplex::get_model_metadata(model_dir)?;
    let mut args = crate::composition::mlx_architectures::moshi::personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    load_with_layout(
        model_dir,
        args,
        CheckpointLayout::Pytorch,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads an explicit PyTorch-layout Moshi-family checkpoint through the
/// canonical generalized engine.
pub fn load_pytorch_layerwise_model(
    args: ModelArgs,
    checkpoint: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    load_with_layout(
        checkpoint,
        args,
        CheckpointLayout::Pytorch,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads PersonaPlex with rank-local temporal and depth transformers.
pub fn load_personaplex_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let metadata =
        crate::composition::mlx_architectures::moshi::personaplex::get_model_metadata(model_dir)?;
    let mut args = crate::composition::mlx_architectures::moshi::personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    load_parallel_with_layout(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        args,
        CheckpointLayout::Pytorch,
        options,
        build,
        stream,
        weights_stream,
    )
}

fn load_with_layout(
    source: impl AsRef<Path>,
    args: ModelArgs,
    layout: CheckpointLayout,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let source = source.as_ref();
    let options = options.into();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Moshi family", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(source, options.max_mapped_shards())?;
    let store = resolve_moshi_store(store, &args, layout)?;
    let (store, args) = match quantize_on_load {
        Some(quantization) => quantize_moshi_store(store, &args, layout, quantization, stream)?,
        None => (store, args),
    };
    let mut architecture = MoshiArchitecture::new(args.clone(), stream)?;
    let unit_layout = moshi_execution_layout(&args)?;
    let factory = MoshiUnitFactory {
        args,
        parallel_layout: None,
    };
    let binding_layout = layout;
    let binding_args = architecture.args.clone();
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        unit_layout,
        options,
        stream,
        weights_stream,
        |_| false,
        move |modules, store| moshi_static_bindings(binding_layout, modules, store),
        move |ordinal, unit, store, _| {
            let (group, index) = moshi_unit_address(&binding_args, ordinal)?;
            moshi_unit_bindings(binding_layout, &binding_args, group, index, &unit, store)
        },
    )?;
    let execution = if options.is_fully_resident() {
        MoshiExecution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        MoshiExecution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MoshiLayerwiseModel {
        execution,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
}

fn load_parallel_with_layout(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    checkpoint_layout: CheckpointLayout,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let store = resolve_moshi_store(store, &args, checkpoint_layout)?;
    let mut architecture = MoshiArchitecture::new(args.clone(), stream)?;
    let mut planner = build.planner();
    for group in architecture.parallel_parameter_groups(build)? {
        planner.register(group)?;
    }
    let (_, local_layout) = planner.finish()?;
    if local_layout.is_empty() {
        return Err(Error::Parallel(
            "Moshi declared no tensor-parallel parameters".into(),
        ));
    }
    architecture.configure_parallel_static(build, &local_layout, stream)?;

    let global_static = MoshiLayerwiseStatic::new(&args, stream)?;
    let global_static = MlxParameterTree::new(global_static, "")
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let static_bindings = moshi_static_bindings(checkpoint_layout, &global_static, store.as_ref())?;
    let unit_layout = moshi_execution_layout(&args)?;
    let shared_layout = Arc::new(local_layout);
    let factory = MoshiUnitFactory {
        args: args.clone(),
        parallel_layout: Some(Arc::clone(&shared_layout)),
    };
    let binding_args = args.clone();
    let binding_local_layout = Arc::clone(&shared_layout);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        unit_layout,
        options,
        stream,
        weights_stream,
        |_| false,
        move |_, store| shard_layer_bindings(static_bindings, "", store, &shared_layout),
        move |ordinal, _local, store, stream| {
            let (group, index) = moshi_unit_address(&binding_args, ordinal)?;
            let global = build_moshi_unit(&binding_args, group, index, None, stream)?;
            let global = MlxParameterTree::new(global, "")
                .map_err(|error| Error::Parallel(error.to_string()))?;
            let bindings = moshi_unit_bindings(
                checkpoint_layout,
                &binding_args,
                group,
                index,
                &global,
                store,
            )?;
            shard_layer_bindings(
                bindings,
                &moshi_unit_prefix(group, index),
                store,
                &binding_local_layout,
            )
        },
    )?;
    let execution = if options.is_fully_resident() {
        MoshiExecution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        MoshiExecution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MoshiLayerwiseModel {
        execution,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
}

fn resolve_moshi_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
    layout: CheckpointLayout,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = match layout {
        CheckpointLayout::Native => super::checkpoint::safetensors_plan(args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        CheckpointLayout::Pytorch => super::personaplex_checkpoint::safetensors_plan(args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
    };
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "Moshi checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_moshi_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &ModelArgs,
    layout: CheckpointLayout,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        ModelArgs,
    ),
    Error,
> {
    let mut target_args = source_args.clone();
    target_args.quantization = Some(quantization);
    let source_static = MoshiLayerwiseStatic::new(source_args, stream)?;
    let target_static = MoshiLayerwiseStatic::new(&target_args, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let count = usize::try_from(source_args.num_layers + source_args.dep_q)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Moshi unit count".into()))?;
    let (store, _) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |ordinal, stream| {
            let (group, index) = moshi_unit_address(&source_units, ordinal)?;
            build_moshi_unit(&source_units, group, index, None, stream)
        },
        move |ordinal, stream| {
            let (group, index) = moshi_unit_address(&target_units, ordinal)?;
            build_moshi_unit(&target_units, group, index, None, stream)
        },
        count,
        quantization,
        stream,
        move |module, store| match layout {
            CheckpointLayout::Native => Ok(build_module_bindings(module, "", store)?),
            CheckpointLayout::Pytorch => pytorch_static_bindings(module, store),
        },
        move |ordinal, module, store| {
            let (group, index) = moshi_unit_address(&binding_args, ordinal)?;
            match layout {
                CheckpointLayout::Native => Ok(build_module_bindings(
                    module,
                    &moshi_unit_prefix(group, index),
                    store,
                )?),
                CheckpointLayout::Pytorch => pytorch_layer_bindings(
                    module,
                    &moshi_unit_prefix(group, index),
                    group,
                    index,
                    binding_args.dep_q as usize,
                    store,
                ),
            }
        },
    )?;
    Ok((store, target_args))
}

/// Family-specific input for teacher-forced or autoregressive depth execution.
pub(crate) enum MoshiLayerwiseInput<'a> {
    /// Caller supplies the token embedded by every depth slice.
    TeacherForced {
        text_token: &'a Array,
        audio_tokens: &'a Array,
        depth_tokens: &'a Array,
    },
    /// Each depth slice consumes the token selected after the previous slice.
    Autoregressive {
        text_token: &'a Array,
        audio_tokens: &'a Array,
        forced_text_token: Option<&'a Array>,
        forced_audio_tokens: Option<&'a Array>,
        forced_audio_codebooks: Option<&'a [bool]>,
    },
}

/// Per-frame state shared between temporal and depth execution groups.
pub(crate) struct MoshiForwardContext {
    temporal_input: Array,
    temporal_output: Option<Array>,
    text_logits: Option<Array>,
    audio_logits: Vec<Array>,
    depth_tokens: Option<Array>,
    previous: Option<Array>,
    sampled_text: Option<Array>,
    predicted_audio: Vec<Array>,
    current_audio_logits: Option<Array>,
    forced_text_token: Option<Array>,
    forced_audio_tokens: Option<Array>,
    forced_audio_codebooks: Option<Vec<bool>>,
    autoregressive: bool,
}

impl MoshiForwardContext {
    fn into_token_output(self) -> Result<TokenStepOutput, Exception> {
        Ok(TokenStepOutput {
            temporal_input: self.temporal_input,
            temporal_layer_traces: Vec::new(),
            text_logits: self
                .text_logits
                .ok_or_else(|| Exception::custom("Moshi temporal logits were not produced"))?,
            audio_logits: self.audio_logits,
            temporal_output: self
                .temporal_output
                .ok_or_else(|| Exception::custom("Moshi temporal output was not produced"))?,
        })
    }
}

/// One temporary temporal layer or one complete depth-codebook slice.
pub(crate) enum MoshiExecutionUnit {
    Temporal(Box<MoshiTransformerLayer>),
    Depth(Box<DepFormerSlice>),
}

impl ModuleParameters for MoshiExecutionUnit {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Temporal(unit) => unit.num_parameters(),
            Self::Depth(unit) => unit.num_parameters(),
        }
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Temporal(unit) => unit.parameters(),
            Self::Depth(unit) => unit.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Temporal(unit) => unit.parameters_mut(),
            Self::Depth(unit) => unit.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Temporal(unit) => unit.trainable_parameters(),
            Self::Depth(unit) => unit.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Temporal(unit) => unit.freeze_parameters(recursive),
            Self::Depth(unit) => unit.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Temporal(unit) => unit.unfreeze_parameters(recursive),
            Self::Depth(unit) => unit.unfreeze_parameters(recursive),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Temporal(unit) => unit.all_frozen(),
            Self::Depth(unit) => unit.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Temporal(unit) => unit.any_frozen(),
            Self::Depth(unit) => unit.any_frozen(),
        }
    }
}

/// Neutral layered composition for native Moshi and released PersonaPlex layouts.
pub(crate) struct MoshiArchitecture {
    args: ModelArgs,
    static_modules: MoshiStatic,
}

impl MoshiArchitecture {
    fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let static_modules = MoshiLayerwiseStatic::new(&args, stream)?;
        Ok(Self {
            static_modules: MlxParameterTree::new(static_modules, "")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            args,
        })
    }

    /// Returns parsed model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }
}

fn moshi_execution_layout(args: &ModelArgs) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["temporal_transformer", "depth_codebook_slices"])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(&graph, [args.num_layers as usize, args.dep_q as usize])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn moshi_unit_address(args: &ModelArgs, ordinal: usize) -> Result<(usize, usize), Error> {
    let temporal = args.num_layers as usize;
    if ordinal < temporal {
        return Ok((0, ordinal));
    }
    let depth = ordinal - temporal;
    if depth < args.dep_q as usize {
        return Ok((1, depth));
    }
    Err(Error::UnsupportedArchitecture(format!(
        "Moshi execution unit {ordinal} is outside {} units",
        temporal + args.dep_q as usize
    )))
}

fn moshi_unit_prefix(group: usize, index: usize) -> String {
    if group == 0 {
        format!("transformer.layers.{index}")
    } else {
        format!("depformer.slices.{index}")
    }
}

fn build_moshi_unit(
    args: &ModelArgs,
    group: usize,
    index: usize,
    parallel_layout: Option<&eredu_runtime::LocalModelLayout>,
    stream: &Stream,
) -> Result<MoshiExecutionUnit, Error> {
    let Some(layout) = parallel_layout else {
        return match group {
            0 => Ok(MoshiExecutionUnit::Temporal(Box::new(
                MoshiTransformerLayer::new_temporal(args, stream)?,
            ))),
            1 => Ok(MoshiExecutionUnit::Depth(Box::new(
                DepFormerSlice::new_for_index(args, index, stream)?,
            ))),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            ))),
        };
    };
    let (prefix, dim, heads, feed_forward) = match group {
        0 => (
            format!("transformer.layers.{index}"),
            args.dim,
            args.num_heads,
            args.dim_feedforward.unwrap_or(4 * args.dim),
        ),
        1 => (
            format!("depformer.slices.{index}.transformer.layers.0"),
            args.depformer_dim,
            args.depformer_num_heads,
            args.depformer_dim_feedforward
                .unwrap_or(4 * args.depformer_dim),
        ),
        _ => {
            return Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            )))
        }
    };
    let local_qkv = layout
        .tensor(&format!("{prefix}.self_attn.in_proj.weight"))
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} attention")))?
        .local_shape()[0];
    let local_in = layout
        .tensor(&format!("{prefix}.gating.linear_in.weight"))
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLP")))?
        .local_shape()[0];
    let head_dim = dim / heads;
    let local_heads = i32::try_from(local_qkv / 3)
        .map_err(|_| Error::Parallel("Moshi local attention width exceeds i32".into()))?
        / head_dim;
    let global_hidden = if feed_forward == 4 * dim {
        11 * dim / 4
    } else {
        2 * feed_forward / 3
    };
    let local_hidden = i32::try_from(local_in / 2)
        .map_err(|_| Error::Parallel("Moshi local MLP width exceeds i32".into()))?;
    if local_hidden > global_hidden {
        return Err(Error::Parallel(format!(
            "Moshi local MLP width {local_hidden} exceeds global width {global_hidden}"
        )));
    }
    match group {
        0 => Ok(MoshiExecutionUnit::Temporal(Box::new(
            MoshiTransformerLayer::new_temporal_tensor_parallel(
                args,
                local_heads,
                local_hidden,
                stream,
            )?,
        ))),
        1 => Ok(MoshiExecutionUnit::Depth(Box::new(
            DepFormerSlice::new_for_index_tensor_parallel(
                args,
                index,
                local_heads,
                local_hidden,
                stream,
            )?,
        ))),
        _ => unreachable!("validated Moshi execution group"),
    }
}

fn moshi_static_bindings(
    layout: CheckpointLayout,
    module: &MoshiStatic,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    match layout {
        CheckpointLayout::Native => Ok(build_module_bindings(&**module, "", store)?),
        CheckpointLayout::Pytorch => pytorch_static_bindings(&**module, store),
    }
}

fn moshi_unit_bindings(
    layout: CheckpointLayout,
    args: &ModelArgs,
    group: usize,
    index: usize,
    unit: &MoshiUnit,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    match layout {
        CheckpointLayout::Native => Ok(build_module_bindings(
            &**unit,
            &moshi_unit_prefix(group, index),
            store,
        )?),
        CheckpointLayout::Pytorch => pytorch_layer_bindings(
            &**unit,
            &moshi_unit_prefix(group, index),
            group,
            index,
            args.dep_q as usize,
            store,
        ),
    }
}

fn begin_moshi_forward<'a>(
    architecture: &mut MoshiArchitecture,
    input: MoshiLayerwiseInput<'a>,
    cache: &mut MoshiCache,
    parallel: Option<&safemlx::distributed::Group>,
    stream: &Stream,
) -> Result<LayeredForwardState<Array, MoshiForwardContext>, Error> {
    if cache.temporal.len() != architecture.args.num_layers as usize
        || cache.depth.len() != architecture.args.depformer_num_layers as usize
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "Moshi cache has {} temporal and {} depth layers; expected {} and {}",
            cache.temporal.len(),
            cache.depth.len(),
            architecture.args.num_layers,
            architecture.args.depformer_num_layers
        )));
    }
    let (text, audio, depth, forced_text, forced_audio, forced_mask, autoregressive) = match input {
        MoshiLayerwiseInput::TeacherForced {
            text_token,
            audio_tokens,
            depth_tokens,
        } => {
            if depth_tokens.shape().len() != 2
                || depth_tokens.dim(0) != text_token.dim(0)
                || depth_tokens.dim(1) != architecture.args.dep_q
            {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Moshi depth input must have shape [batch, {}]",
                    architecture.args.dep_q
                )));
            }
            (
                text_token,
                audio_tokens,
                Some(depth_tokens.clone()),
                None,
                None,
                None,
                false,
            )
        }
        MoshiLayerwiseInput::Autoregressive {
            text_token,
            audio_tokens,
            forced_text_token,
            forced_audio_tokens,
            forced_audio_codebooks,
        } => (
            text_token,
            audio_tokens,
            None,
            forced_text_token.cloned(),
            forced_audio_tokens.cloned(),
            forced_audio_codebooks.map(ToOwned::to_owned),
            true,
        ),
    };
    cache.reset_depth();
    let hidden = match parallel {
        Some(group) => architecture.static_modules.temporal_input_tensor_parallel(
            &architecture.args,
            text,
            audio,
            group,
            stream,
        )?,
        None => {
            architecture
                .static_modules
                .temporal_input(&architecture.args, text, audio, stream)?
        }
    };
    Ok(LayeredForwardState {
        context: MoshiForwardContext {
            temporal_input: hidden.clone(),
            temporal_output: None,
            text_logits: None,
            audio_logits: Vec::with_capacity(architecture.args.dep_q as usize),
            depth_tokens: depth,
            previous: None,
            sampled_text: None,
            predicted_audio: Vec::with_capacity(architecture.args.dep_q as usize),
            current_audio_logits: None,
            forced_text_token: forced_text,
            forced_audio_tokens: forced_audio,
            forced_audio_codebooks: forced_mask,
            autoregressive,
        },
        hidden,
    })
}

impl MoshiArchitecture {
    fn parallel_parameter_groups(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        use eredu_runtime::{
            MemberSharding, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        };
        use safemlx::ops::quantized_packed_dimension;

        let mut groups = Vec::new();
        let mut register_vocab = |prefix: String, vocabulary: usize, dimensions: i32| {
            let packed = self.args.quantization.map_or(dimensions, |quantization| {
                quantized_packed_dimension(dimensions, quantization.bits())
            }) as usize;
            let mut members = vec![ParameterMemberSpec::new(
                format!("{prefix}.weight"),
                [vocabulary, packed],
                MemberSharding::Balanced { axis: 0 },
            )];
            if let Some(quantization) = self.args.quantization {
                let companions = [
                    vocabulary,
                    (dimensions / quantization.group_size()) as usize,
                ];
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.scales"),
                    companions,
                    MemberSharding::Balanced { axis: 0 },
                ));
                if quantization.has_biases() {
                    members.push(ParameterMemberSpec::new(
                        format!("{prefix}.biases"),
                        companions,
                        MemberSharding::Balanced { axis: 0 },
                    ));
                }
            }
            groups.push(ParameterGroupSpec::new(
                prefix,
                ParameterRole::Vocabulary,
                members,
            )?);
            Ok::<_, Error>(())
        };
        register_vocab(
            "text_emb".into(),
            (self.args.text_card + 1) as usize,
            self.args.dim,
        )?;
        for index in 0..self.args.n_q {
            register_vocab(
                format!("audio_embs.{index}"),
                (self.args.card + 1) as usize,
                self.args.dim,
            )?;
        }
        register_vocab(
            "text_linear".into(),
            self.args.text_card as usize,
            self.args.dim,
        )?;
        let mut register_transformer = |prefix: String,
                                        dim: i32,
                                        feed_forward: i32,
                                        layers: i32|
         -> Result<(), Error> {
            let hidden = if feed_forward == 4 * dim {
                11 * dim / 4
            } else {
                2 * feed_forward / 3
            };
            let quant = self.args.quantization;
            let packed = |width: i32| {
                quant.map_or(width, |q| quantized_packed_dimension(width, q.bits())) as usize
            };
            let linear_members =
                |target: String, output: i32, input: i32, sharding: MemberSharding| {
                    let mut members = vec![ParameterMemberSpec::new(
                        format!("{target}.weight"),
                        [output as usize, packed(input)],
                        sharding.clone(),
                    )];
                    if let Some(q) = quant {
                        let companion = match sharding {
                            MemberSharding::Equal { axis: 0 }
                            | MemberSharding::Segmented { axis: 0, .. } => sharding.clone(),
                            MemberSharding::Equal { axis: 1 } => MemberSharding::Equal { axis: 1 },
                            _ => sharding.clone(),
                        };
                        members.push(ParameterMemberSpec::new(
                            format!("{target}.scales"),
                            [output as usize, (input / q.group_size()) as usize],
                            companion.clone(),
                        ));
                        if q.has_biases() {
                            members.push(ParameterMemberSpec::new(
                                format!("{target}.biases"),
                                [output as usize, (input / q.group_size()) as usize],
                                companion,
                            ));
                        }
                    }
                    members
                };
            for layer in 0..layers {
                let layer = format!("{prefix}.layers.{layer}");
                let segments = vec![
                    0..dim as usize,
                    dim as usize..2 * dim as usize,
                    2 * dim as usize..3 * dim as usize,
                ];
                groups.push(ParameterGroupSpec::new(
                    format!("{layer}.attention.input"),
                    ParameterRole::Segmented,
                    linear_members(
                        format!("{layer}.self_attn.in_proj"),
                        3 * dim,
                        dim,
                        MemberSharding::Segmented { axis: 0, segments },
                    ),
                )?);
                groups.push(ParameterGroupSpec::new(
                    format!("{layer}.attention.output"),
                    ParameterRole::RowProjection,
                    linear_members(
                        format!("{layer}.self_attn.out_proj"),
                        dim,
                        dim,
                        MemberSharding::Equal { axis: 1 },
                    ),
                )?);
                let mlp_segments = vec![0..hidden as usize, hidden as usize..2 * hidden as usize];
                groups.push(ParameterGroupSpec::new(
                    format!("{layer}.mlp.input"),
                    ParameterRole::Segmented,
                    linear_members(
                        format!("{layer}.gating.linear_in"),
                        2 * hidden,
                        dim,
                        MemberSharding::Segmented {
                            axis: 0,
                            segments: mlp_segments,
                        },
                    ),
                )?);
                groups.push(ParameterGroupSpec::new(
                    format!("{layer}.mlp.output"),
                    ParameterRole::RowProjection,
                    linear_members(
                        format!("{layer}.gating.linear_out"),
                        dim,
                        hidden,
                        MemberSharding::Equal { axis: 1 },
                    ),
                )?);
            }
            Ok(())
        };
        register_transformer(
            "transformer".into(),
            self.args.dim,
            self.args.dim_feedforward.unwrap_or(4 * self.args.dim),
            self.args.num_layers,
        )?;
        for slice in 0..self.args.dep_q {
            register_transformer(
                format!("depformer.slices.{slice}.transformer"),
                self.args.depformer_dim,
                self.args
                    .depformer_dim_feedforward
                    .unwrap_or(4 * self.args.depformer_dim),
                self.args.depformer_num_layers,
            )?;
        }
        Ok(groups)
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let modules =
            MoshiLayerwiseStatic::new_tensor_parallel(&self.args, context.topology(), stream)?;
        self.static_modules = MlxParameterTree::new(modules, "")
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(())
    }
}

impl LayeredArchitecture<MlxBackend, MoshiCache> for MoshiArchitecture {
    type Input<'a> = MoshiLayerwiseInput<'a>;
    type StaticModules = MoshiStatic;
    type Unit = MoshiUnit;
    type ForwardContext = MoshiForwardContext;
    type RetainedContextValues<'a> = std::vec::IntoIter<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        "moshi"
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["temporal_transformer", "depth_codebook_slices"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.num_layers as usize),
            1 => Ok(self.args.dep_q as usize),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            ))),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "Moshi execution group {group} has no unit {index}"
            )));
        }
        Ok(moshi_unit_prefix(group, index))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        let unit = build_moshi_unit(&self.args, group, index, None, stream)?;
        MlxParameterTree::new(unit, "").map_err(|error| Error::Parallel(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut MoshiCache,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        begin_moshi_forward(self, input, cache, None, stream)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _cache: &mut MoshiCache,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, [temporal]) => Ok((*temporal).clone()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi execution group {group} received {} dependencies",
                dependencies.len()
            ))),
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut MoshiCache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, &mut **layer) {
            (0, MoshiExecutionUnit::Temporal(layer)) => {
                let output =
                    layer.forward_layerwise(hidden.clone(), &mut cache.temporal[index], stream)?;
                if index + 1 == self.args.num_layers as usize {
                    let (temporal, logits) =
                        self.static_modules.finish_temporal(&output, stream)?;
                    context.temporal_output = Some(temporal.clone());
                    context.text_logits = Some(logits);
                    Ok(temporal)
                } else {
                    Ok(output)
                }
            }
            (1, MoshiExecutionUnit::Depth(slice)) => {
                let previous = if context.autoregressive {
                    context
                        .previous
                        .as_ref()
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "Moshi depth execution started before text sampling".into(),
                            )
                        })?
                        .clone()
                } else {
                    context
                        .depth_tokens
                        .as_ref()
                        .expect("teacher-forced depth tokens")
                        .try_index_device((.., index as i32), stream)?
                        .expand_dims(1, stream)?
                };
                let logits = slice.forward_layerwise(
                    context.temporal_output.as_ref().expect("temporal output"),
                    &previous,
                    context.autoregressive,
                    &mut cache.depth,
                    stream,
                )?;
                context.current_audio_logits = Some(logits.clone());
                context.audio_logits.push(logits);
                Ok(hidden.clone())
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi execution unit does not match group {group}"
            ))),
        }
    }

    fn retained_context_values<'a>(
        &self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::once(&context.temporal_input)
            .chain(context.temporal_output.iter())
            .chain(context.text_logits.iter())
            .chain(context.audio_logits.iter())
            .chain(context.previous.iter())
            .chain(context.sampled_text.iter())
            .chain(context.predicted_audio.iter())
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn finish_forward(
        &mut self,
        _hidden: &Array,
        _cache: &mut MoshiCache,
        context: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        context
            .text_logits
            .clone()
            .ok_or_else(|| Error::UnsupportedArchitecture("Moshi produced no text logits".into()))
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MoshiCache> for MoshiArchitecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut MoshiCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        begin_moshi_forward(self, input, cache, Some(group), stream)
    }

    fn forward_unit_parallel(
        &mut self,
        execution_group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        cache: &mut MoshiCache,
        context: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut **unit {
            MoshiExecutionUnit::Temporal(layer) if execution_group == 0 => {
                let output = layer.forward_tensor_parallel(
                    hidden.clone(),
                    &mut cache.temporal[index],
                    group,
                    stream,
                )?;
                if index + 1 == self.args.num_layers as usize {
                    let (temporal, logits) = self
                        .static_modules
                        .finish_temporal_tensor_parallel(&output, group, stream)?;
                    context.temporal_output = Some(temporal.clone());
                    context.text_logits = Some(logits);
                    Ok(temporal)
                } else {
                    Ok(output)
                }
            }
            MoshiExecutionUnit::Depth(slice) if execution_group == 1 => {
                let previous = if context.autoregressive {
                    context
                        .previous
                        .as_ref()
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "Moshi depth execution started before text sampling".into(),
                            )
                        })?
                        .clone()
                } else {
                    context
                        .depth_tokens
                        .as_ref()
                        .expect("teacher-forced depth tokens")
                        .try_index_device((.., index as i32), stream)?
                        .expand_dims(1, stream)?
                };
                let logits = slice.forward_tensor_parallel(
                    context.temporal_output.as_ref().expect("temporal output"),
                    &previous,
                    context.autoregressive,
                    &mut cache.depth,
                    group,
                    stream,
                )?;
                context.current_audio_logits = Some(logits.clone());
                context.audio_logits.push(logits);
                Ok(hidden.clone())
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi TP execution unit does not match group {execution_group}"
            ))),
        }
    }

    fn finish_forward_parallel(
        &mut self,
        _hidden: &Array,
        _cache: &mut MoshiCache,
        context: &Self::ForwardContext,
        _group: &safemlx::distributed::Group,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        context
            .text_logits
            .clone()
            .ok_or_else(|| Error::UnsupportedArchitecture("Moshi produced no text logits".into()))
    }
}

fn new_cache(args: &ModelArgs) -> MoshiCache {
    MoshiCache::new(args)
}

fn pytorch_static_bindings(
    module: &MoshiLayerwiseStatic,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let mut recipes = BTreeMap::new();
    for name in module.parameters().flatten().keys() {
        let name = name.as_ref();
        let source = if let Some(rest) = name.strip_prefix("audio_embs.") {
            let (index, suffix) = rest.split_once('.').expect("audio embedding parameter");
            format!("emb.{index}.{suffix}")
        } else if name == "out_norm.weight" {
            recipes.insert(
                name.to_string(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(source_full("out_norm.alpha")),
                    shape: vec![module.parameters().flatten()[name].dim(0) as usize],
                },
            );
            continue;
        } else {
            name.to_string()
        };
        recipes.insert(name.to_string(), source_full(source));
    }
    Ok(
        build_module_binding_plan_with_recipes(module, "", store, recipes)?
            .build_bindings(store)?,
    )
}

fn pytorch_layer_bindings(
    module: &MoshiExecutionUnit,
    prefix: &str,
    group: usize,
    index: usize,
    depth_count: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let mut recipes = BTreeMap::new();
    for name in module.parameters().flatten().keys() {
        let name = name.as_ref();
        let recipe = if group == 0 {
            temporal_recipe(name, index, module, store)
        } else {
            depth_recipe(name, index, depth_count, store, module)?
        };
        recipes.insert(name.to_string(), recipe);
    }
    Ok(
        build_module_binding_plan_with_recipes(module, prefix, store, recipes)?
            .build_bindings(store)?,
    )
}

fn temporal_recipe(
    name: &str,
    layer: usize,
    module: &MoshiExecutionUnit,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> DerivedWeightRecipe {
    if name == "norm1.weight" || name == "norm2.weight" {
        let norm = name.strip_suffix(".weight").unwrap();
        return DerivedWeightRecipe::Reshape {
            input: Box::new(source_full(format!(
                "transformer.layers.{layer}.{norm}.alpha"
            ))),
            shape: vec![module.parameters().flatten()[name].dim(0) as usize],
        };
    }
    if name == "self_attn.in_proj.weight" {
        let packed = format!("transformer.layers.{layer}.self_attn.in_proj_weight");
        let native = format!("transformer.layers.{layer}.self_attn.in_proj.weight");
        return source_full(if store.source_keys().iter().any(|key| key == &packed) {
            packed
        } else {
            native
        });
    }
    source_full(format!("transformer.layers.{layer}.{name}"))
}

fn depth_recipe(
    name: &str,
    slice: usize,
    depth_count: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    module: &MoshiExecutionUnit,
) -> Result<DerivedWeightRecipe, Error> {
    if let Some(suffix) = name.strip_prefix("emb.") {
        return Ok(source_full(if slice == 0 {
            format!("depformer_text_emb.{suffix}")
        } else {
            format!("depformer_emb.{}.{suffix}", slice - 1)
        }));
    }
    if let Some(suffix) = name.strip_prefix("linear_in.") {
        return Ok(source_full(format!("depformer_in.{slice}.{suffix}")));
    }
    if let Some(suffix) = name.strip_prefix("linear_out.") {
        return Ok(source_full(format!("linears.{slice}.{suffix}")));
    }
    let rest = name
        .strip_prefix("transformer.layers.")
        .expect("depth transformer parameter");
    let (layer, suffix) = rest.split_once('.').expect("depth layer parameter");
    if suffix == "norm1.weight" || suffix == "norm2.weight" {
        let norm = suffix.strip_suffix(".weight").unwrap();
        return Ok(DerivedWeightRecipe::Reshape {
            input: Box::new(source_full(format!(
                "depformer.layers.{layer}.{norm}.alpha"
            ))),
            shape: vec![module.parameters().flatten()[name].dim(0) as usize],
        });
    }
    if let Some(attention) = suffix.strip_prefix("self_attn.") {
        let key = if attention == "in_proj.weight" {
            let packed = format!("depformer.layers.{layer}.self_attn.in_proj_weight");
            let native = format!("depformer.layers.{layer}.self_attn.in_proj.weight");
            if store.source_keys().iter().any(|key| key == &packed) {
                packed
            } else {
                native
            }
        } else {
            format!("depformer.layers.{layer}.self_attn.{attention}")
        };
        let rows = store.source_metadata(&key)?.logical_shape[0];
        if rows % depth_count != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "PersonaPlex tensor {key} cannot be split across {depth_count} codebooks"
            )));
        }
        let chunk = rows / depth_count;
        return Ok(DerivedWeightRecipe::source(
            key,
            TensorSelection::Range {
                axis: 0,
                start: slice * chunk,
                end: (slice + 1) * chunk,
            },
        ));
    }
    let gating = suffix
        .strip_prefix("gating.")
        .expect("depth gating parameter");
    Ok(source_full(format!(
        "depformer.layers.{layer}.gating.{slice}.{gating}"
    )))
}

fn source_full(key: impl Into<String>) -> DerivedWeightRecipe {
    DerivedWeightRecipe::source(key, TensorSelection::Full)
}

fn validate_forced_depth(
    tokens: Option<&Array>,
    mask: Option<&[bool]>,
    batch: i32,
    depth_count: usize,
) -> Result<(), Exception> {
    if let Some(tokens) = tokens {
        if tokens.shape() != [batch, depth_count as i32] {
            return Err(Exception::custom(format!(
                "Moshi forced depth tokens must have shape [batch, {depth_count}], got {:?}",
                tokens.shape()
            )));
        }
    }
    if let Some(mask) = mask {
        if mask.len() != depth_count {
            return Err(Exception::custom(format!(
                "Moshi forced depth mask must have {depth_count} entries, got {}",
                mask.len()
            )));
        }
    }
    Ok(())
}

fn layerwise_exception(error: impl std::fmt::Display) -> Exception {
    Exception::custom(error.to_string())
}

#[cfg(test)]
mod neutral_runtime_tests {
    use super::*;

    #[test]
    fn execution_layout_preserves_temporal_and_depth_groups() {
        let args = ModelArgs::v0_1();
        let layout = moshi_execution_layout(&args).unwrap();
        assert_eq!(layout.group_count(), 2);
        assert_eq!(layout.group_id(0).unwrap().as_str(), "temporal_transformer");
        assert_eq!(
            layout.group_id(1).unwrap().as_str(),
            "depth_codebook_slices"
        );
        assert_eq!(
            layout.group_range(0).unwrap().len(),
            args.num_layers as usize
        );
        assert_eq!(layout.group_range(1).unwrap().len(), args.dep_q as usize);
        assert_eq!(moshi_unit_address(&args, 0).unwrap(), (0, 0));
        assert_eq!(
            moshi_unit_address(&args, args.num_layers as usize).unwrap(),
            (1, 0)
        );
        assert!(moshi_unit_address(&args, args.num_layers as usize + args.dep_q as usize).is_err());
    }
}
