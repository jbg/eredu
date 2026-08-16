//! Bounded layer execution for Moshi and PersonaPlex realtime token models.

use std::{collections::BTreeMap, path::Path};

use safemlx::{
    error::Exception,
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, stack_axis, zeros},
    random::RandomState,
    Array, Stream,
};

use crate::{
    api::realtime::{RealtimeSpeechConfig, RealtimeStepOutput},
    architectures::moshi::model::{
        self as resident, DepFormerSlice, ModelArgs, MoshiCache, MoshiLayerwiseStatic,
        MoshiTransformerLayer, SampleStepOutput, TokenStepOutput,
    },
    error::Error,
    runtime::cache::KeyValueCache,
    runtime::checkpoint::artifact::LoadedArtifactIdentity,
    runtime::checkpoint::binding::{
        build_module_bindings, build_module_bindings_with_recipes, populate_module_from_lease,
    },
    runtime::checkpoint::recipe::DerivedWeightRecipe,
    runtime::checkpoint::store::{TensorSelection, WeightStore},
    runtime::execution::layerwise::{
        load_layerwise_model, load_safetensors_layerwise_model,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store,
        transformed_module_weight_store, ArchitectureAdapter, LayerWeightResidency,
        LayerwiseForwardState, LayerwiseModel, StaticUnitBindings,
    },
    runtime::generation::sampler::Sampler,
    runtime::residency::manager::{
        ResidencyReport, ResidentLayerGroupReport, ResidentUnitLease, WeightBinding,
    },
};

const STATIC_UNIT: &str = "moshi.static";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CheckpointLayout {
    Native,
    Pytorch,
}

/// Moshi-family model with independent temporal and depth-codebook residency windows.
pub struct MoshiLayerwiseModel {
    execution: LayerwiseModel<MoshiLayerwiseAdapter>,
    artifact_identity: LoadedArtifactIdentity,
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
        self.execution.adapter().args()
    }

    /// Allocates empty temporal and within-frame depth caches.
    pub fn new_cache(&self) -> MoshiCache {
        new_cache(self.args())
    }

    /// Returns the codec-token stream geometry consumed by realtime scheduling.
    pub fn realtime_config(&self) -> RealtimeSpeechConfig<'_> {
        RealtimeSpeechConfig {
            total_audio_codebooks: self.args().n_q,
            input_audio_codebooks: self.args().input_audio_codebooks(),
            generated_audio_codebooks: self.args().generated_audio_codebooks(),
            depth_audio_codebooks: self.args().dep_q,
            text_padding_token: self.args().text_padding_token(),
            audio_padding_token: self.args().audio_padding_token(),
            audio_delays: self.args().audio_delays(),
        }
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns residency attributed to the temporal and depth execution groups.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.execution
            .execution_groups()
            .iter()
            .map(|group| {
                group
                    .report(self.execution.residency_manager())
                    .map_err(Error::from)
            })
            .collect()
    }

    /// Clears one temporary execution group without affecting the other group.
    pub fn clear_device_group(&self, group: &str) -> Result<(), Error> {
        self.execution.clear_device_group(group)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
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
            .forward_tensor_parallel_with_context_hook(
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
            .forward_tensor_parallel_with_context_hook(
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
        let mut text_sampler = crate::runtime::generation::sampler::DefaultSampler;
        let mut audio_samplers = (0..self.args().dep_q)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
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
    ) -> Result<RealtimeStepOutput, Exception> {
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
        state: &mut resident::GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Exception> {
        if self.args().existing_text_padding_id.is_some() && self.args().dep_q == self.args().n_q {
            return self.generate_step_pytorch_style(
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
        self.generate_step_native_style(
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
    fn generate_step_native_style<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut resident::GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Exception> {
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
        Ok(RealtimeStepOutput {
            text_token: sampled.text_token,
            sampled_audio_tokens: sampled.audio_tokens,
            output_audio_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_step_pytorch_style<TS: Sampler, AS: Sampler>(
        &mut self,
        state: &mut resident::GenerationState,
        input_audio_tokens: &Array,
        forced_generated_audio_tokens: Option<&Array>,
        forced_text_token: Option<&Array>,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Exception> {
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
            return Ok(RealtimeStepOutput {
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
        Ok(RealtimeStepOutput {
            text_token: sampled.text_token,
            sampled_audio_tokens: sampled.audio_tokens,
            output_audio_tokens,
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
    options: impl Into<crate::runtime::execution::layerwise::LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let weights_name = args
        .moshi_name
        .clone()
        .unwrap_or_else(|| "model.safetensors".to_string());
    let source = if weights_name == "model.safetensors"
        && model_dir.join("model.safetensors.index.json").exists()
    {
        model_dir.to_path_buf()
    } else {
        model_dir.join(weights_name)
    };
    load_with_layout(
        source,
        args,
        CheckpointLayout::Native,
        options,
        stream,
        weights_stream,
    )
}

/// Loads native Moshi with rank-local temporal and depth transformers.
pub fn load_moshi_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<crate::runtime::execution::layerwise::LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let weights_name = args
        .moshi_name
        .clone()
        .unwrap_or_else(|| "model.safetensors".to_string());
    let source = if weights_name == "model.safetensors"
        && model_dir.join("model.safetensors.index.json").exists()
    {
        model_dir.to_path_buf()
    } else {
        model_dir.join(weights_name)
    };
    let adapter = MoshiLayerwiseAdapter::new(args, CheckpointLayout::Native, stream)?;
    Ok(MoshiLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(&source, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
}

/// Loads the released PersonaPlex PyTorch checkpoint through bounded layer residency.
pub fn load_personaplex_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<crate::runtime::execution::layerwise::LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::PersonaPlex,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let metadata = crate::architectures::moshi::personaplex::get_model_metadata(model_dir)?;
    let mut args = crate::architectures::moshi::personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    load_with_layout(
        model_dir,
        args,
        CheckpointLayout::Pytorch,
        options,
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
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    load_with_layout(
        checkpoint,
        args,
        CheckpointLayout::Pytorch,
        options,
        stream,
        weights_stream,
    )
}

/// Hands a completed Moshi/PersonaPlex load-time transformation to the
/// canonical generalized execution engine.
pub(crate) fn execute_transformed_model(
    model: resident::Model,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let adapter = MoshiLayerwiseAdapter::new(model.args.clone(), CheckpointLayout::Native, stream)?;
    let store = transformed_module_weight_store(&model)?;
    Ok(MoshiLayerwiseModel {
        execution: load_layerwise_model(
            store,
            adapter,
            LayerWeightResidency::FullyResident,
            stream,
            weights_stream,
        )?,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
}

/// Loads PersonaPlex with rank-local temporal and depth transformers.
pub fn load_personaplex_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<crate::runtime::execution::layerwise::LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::PersonaPlex,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let metadata = crate::architectures::moshi::personaplex::get_model_metadata(model_dir)?;
    let mut args = crate::architectures::moshi::personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    let adapter = MoshiLayerwiseAdapter::new(args, CheckpointLayout::Pytorch, stream)?;
    Ok(MoshiLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
}

fn load_with_layout(
    source: impl AsRef<Path>,
    args: ModelArgs,
    layout: CheckpointLayout,
    options: impl Into<crate::runtime::execution::layerwise::LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MoshiLayerwiseModel, Error> {
    let adapter = MoshiLayerwiseAdapter::new(args, layout, stream)?;
    Ok(MoshiLayerwiseModel {
        execution: load_safetensors_layerwise_model(
            source,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
        artifact_identity: LoadedArtifactIdentity::in_memory(),
    })
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

/// Shared adapter for native Moshi and released PersonaPlex layouts.
pub(crate) struct MoshiLayerwiseAdapter {
    args: ModelArgs,
    layout: CheckpointLayout,
    static_modules: MoshiLayerwiseStatic,
}

impl MoshiLayerwiseAdapter {
    fn new(args: ModelArgs, layout: CheckpointLayout, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            static_modules: MoshiLayerwiseStatic::new(&args, stream)?,
            args,
            layout,
        })
    }

    /// Returns parsed model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }
}

impl ArchitectureAdapter for MoshiLayerwiseAdapter {
    type Input<'a> = MoshiLayerwiseInput<'a>;
    type Cache = MoshiCache;
    type Layer = MoshiExecutionUnit;
    type ForwardContext = MoshiForwardContext;

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        let bindings = match self.layout {
            CheckpointLayout::Native => build_module_bindings(&self.static_modules, "", store)?,
            CheckpointLayout::Pytorch => pytorch_static_bindings(&self.static_modules, store)?,
        };
        Ok(vec![StaticUnitBindings::new(STATIC_UNIT, bindings)?])
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Moshi adapter received {} static leases, expected 1",
                leases.len()
            )));
        }
        Ok(populate_module_from_lease(
            &mut self.static_modules,
            &leases[0],
        )?)
    }

    fn validate_cache(&self, cache: &mut MoshiCache) -> Result<(), Error> {
        if cache.temporal.len() != self.args.num_layers as usize
            || cache.depth.len() != self.args.depformer_num_layers as usize
        {
            return Err(Error::UnsupportedArchitecture(format!(
                "Moshi cache has {} temporal and {} depth layers; expected {} and {}",
                cache.temporal.len(),
                cache.depth.len(),
                self.args.num_layers,
                self.args.depformer_num_layers
            )));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let (text, audio, depth, forced_text, forced_audio, forced_mask, autoregressive) =
            match input {
                MoshiLayerwiseInput::TeacherForced {
                    text_token,
                    audio_tokens,
                    depth_tokens,
                } => {
                    if depth_tokens.shape().len() != 2
                        || depth_tokens.dim(0) != text_token.dim(0)
                        || depth_tokens.dim(1) != self.args.dep_q
                    {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Moshi depth input must have shape [batch, {}]",
                            self.args.dep_q
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
        let hidden = self
            .static_modules
            .temporal_input(&self.args, text, audio, stream)?;
        Ok(LayerwiseForwardState {
            context: MoshiForwardContext {
                temporal_input: hidden.clone(),
                temporal_output: None,
                text_logits: None,
                audio_logits: Vec::with_capacity(self.args.dep_q as usize),
                depth_tokens: depth,
                previous: None,
                sampled_text: None,
                predicted_audio: Vec::with_capacity(self.args.dep_q as usize),
                current_audio_logits: None,
                forced_text_token: forced_text,
                forced_audio_tokens: forced_audio,
                forced_audio_codebooks: forced_mask,
                autoregressive,
            },
            hidden,
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(group) = execution.group() else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let (text, audio, depth, forced_text, forced_audio, forced_mask, autoregressive) =
            match input {
                MoshiLayerwiseInput::TeacherForced {
                    text_token,
                    audio_tokens,
                    depth_tokens,
                } => (
                    text_token,
                    audio_tokens,
                    Some(depth_tokens.clone()),
                    None,
                    None,
                    None,
                    false,
                ),
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
        let hidden = self.static_modules.temporal_input_tensor_parallel(
            &self.args,
            text,
            audio,
            group,
            execution.stream(),
        )?;
        Ok(LayerwiseForwardState {
            context: MoshiForwardContext {
                temporal_input: hidden.clone(),
                temporal_output: None,
                text_logits: None,
                audio_logits: Vec::with_capacity(self.args.dep_q as usize),
                depth_tokens: depth,
                previous: None,
                sampled_text: None,
                predicted_audio: Vec::with_capacity(self.args.dep_q as usize),
                current_audio_logits: None,
                forced_text_token: forced_text,
                forced_audio_tokens: forced_audio,
                forced_audio_codebooks: forced_mask,
                autoregressive,
            },
            hidden,
        })
    }

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::runtime::execution::layerwise::ExecutionGroupDag::chain([
            "temporal_transformer",
            "depth_codebook_slices",
        ])
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.num_layers as usize),
            1 => Ok(self.args.dep_q as usize),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            ))),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        match group {
            0 => Ok(MoshiExecutionUnit::Temporal(Box::new(
                MoshiTransformerLayer::new_temporal(&self.args, stream)?,
            ))),
            1 => Ok(MoshiExecutionUnit::Depth(Box::new(
                DepFormerSlice::new_for_index(&self.args, index, stream)?,
            ))),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            ))),
        }
    }

    fn parallel_parameter_groups(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<crate::runtime::distributed::parallel::ParameterGroupSpec>, Error> {
        use crate::runtime::distributed::parallel::{
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
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.static_modules =
            MoshiLayerwiseStatic::new_tensor_parallel(&self.args, context.topology(), stream)?;
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let (prefix, dim, heads, feed_forward) = if group == 0 {
            (
                format!("transformer.layers.{index}"),
                self.args.dim,
                self.args.num_heads,
                self.args.dim_feedforward.unwrap_or(4 * self.args.dim),
            )
        } else {
            (
                format!("depformer.slices.{index}.transformer.layers.0"),
                self.args.depformer_dim,
                self.args.depformer_num_heads,
                self.args
                    .depformer_dim_feedforward
                    .unwrap_or(4 * self.args.depformer_dim),
            )
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
        let _ = global_hidden;
        let local_hidden = i32::try_from(local_in / 2)
            .map_err(|_| Error::Parallel("Moshi local MLP width exceeds i32".into()))?;
        match group {
            0 => Ok(MoshiExecutionUnit::Temporal(Box::new(
                MoshiTransformerLayer::new_temporal_tensor_parallel(
                    &self.args,
                    local_heads,
                    local_hidden,
                    stream,
                )?,
            ))),
            1 => Ok(MoshiExecutionUnit::Depth(Box::new(
                DepFormerSlice::new_for_index_tensor_parallel(
                    &self.args,
                    index,
                    local_heads,
                    local_hidden,
                    stream,
                )?,
            ))),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi has no execution group {group}"
            ))),
        }
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("transformer.layers.{index}")
        } else {
            format!("depformer.slices.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("moshi.temporal.{index:05}")
        } else {
            format!("moshi.depth_slice.{index:05}")
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        match self.layout {
            CheckpointLayout::Native => Ok(build_module_bindings(
                layer,
                &self.layer_checkpoint_prefix(group, index),
                store,
            )?),
            CheckpointLayout::Pytorch => {
                pytorch_layer_bindings(layer, group, index, self.args.dep_q as usize, store)
            }
        }
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, layer) {
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

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.forward_layer(
                group,
                index,
                layer,
                hidden,
                cache,
                context,
                execution.stream(),
            );
        };
        match layer {
            MoshiExecutionUnit::Temporal(layer) if group == 0 => {
                let output = layer.forward_tensor_parallel(
                    hidden.clone(),
                    &mut cache.temporal[index],
                    tp_group,
                    execution.stream(),
                )?;
                if index + 1 == self.args.num_layers as usize {
                    let (temporal, logits) = self.static_modules.finish_temporal_tensor_parallel(
                        &output,
                        tp_group,
                        execution.stream(),
                    )?;
                    context.temporal_output = Some(temporal.clone());
                    context.text_logits = Some(logits);
                    Ok(temporal)
                } else {
                    Ok(output)
                }
            }
            MoshiExecutionUnit::Depth(slice) if group == 1 => {
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
                        .try_index_device((.., index as i32), execution.stream())?
                        .expand_dims(1, execution.stream())?
                };
                let logits = slice.forward_tensor_parallel(
                    context.temporal_output.as_ref().expect("temporal output"),
                    &previous,
                    context.autoregressive,
                    &mut cache.depth,
                    tp_group,
                    execution.stream(),
                )?;
                context.current_audio_logits = Some(logits.clone());
                context.audio_logits.push(logits);
                Ok(hidden.clone())
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Moshi TP execution unit does not match group {group}"
            ))),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if group == 0 {
            cache.temporal[index].retained_arrays()
        } else {
            cache
                .depth
                .iter()
                .flat_map(KeyValueCache::retained_arrays)
                .collect()
        }
    }

    fn retained_context_arrays<'a>(
        &self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        std::iter::once(&context.temporal_input)
            .chain(context.temporal_output.iter())
            .chain(context.text_logits.iter())
            .chain(context.audio_logits.iter())
            .chain(context.previous.iter())
            .chain(context.sampled_text.iter())
            .chain(context.predicted_audio.iter())
            .collect()
    }

    fn finish(
        &mut self,
        _hidden: &Array,
        _cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        context
            .text_logits
            .clone()
            .ok_or_else(|| Error::UnsupportedArchitecture("Moshi produced no text logits".into()))
    }
}

fn new_cache(args: &ModelArgs) -> MoshiCache {
    MoshiCache {
        temporal: vec![
            crate::runtime::cache::ConcatKeyValueCache::new_with_max_size_and_step(
                args.context + 1,
                256
            );
            args.num_layers as usize
        ],
        depth: vec![
            crate::runtime::cache::ConcatKeyValueCache::new();
            args.depformer_num_layers as usize
        ],
    }
}

fn pytorch_static_bindings(
    module: &MoshiLayerwiseStatic,
    store: &dyn WeightStore,
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
    Ok(build_module_bindings_with_recipes(
        module, "", store, recipes,
    )?)
}

fn pytorch_layer_bindings(
    module: &MoshiExecutionUnit,
    group: usize,
    index: usize,
    depth_count: usize,
    store: &dyn WeightStore,
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
    Ok(build_module_bindings_with_recipes(
        module, "", store, recipes,
    )?)
}

fn temporal_recipe(
    name: &str,
    layer: usize,
    module: &MoshiExecutionUnit,
    store: &dyn WeightStore,
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
        return source_full(if store.keys().iter().any(|key| key == &packed) {
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
    store: &dyn WeightStore,
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
            if store.keys().iter().any(|key| key == &packed) {
                packed
            } else {
                native
            }
        } else {
            format!("depformer.layers.{layer}.self_attn.{attention}")
        };
        let rows = store.metadata(&key)?.shape[0];
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

fn layerwise_exception(error: Error) -> Exception {
    Exception::custom(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs, path::Path};

    use safemlx::{
        distributed::{Backend, Group},
        module::ModuleParameters,
        ops::{concatenate_axis, ones_dtype, zeros_dtype},
        Array, Device, DeviceType, ExecutionContext, Stream,
    };

    use super::*;
    use crate::{
        api::realtime::{
            generate_encoded_greedy, LoadedRealtimeModel, RealtimeInferenceScheduler,
            RealtimeSampling,
        },
        api::{moshi as eager, ModelLoadOptions},
        runtime::distributed::{
            parallel::{ParallelBuildContext, ShardingPolicy},
            topology::{DeviceAssignment, ParallelTopology},
        },
        runtime::execution::layerwise::{
            LayerWeightResidency, LayerwiseLoadOptions, WeightResidency,
        },
        runtime::generation::sampler::DefaultSampler,
        runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
        runtime::residency::policy::{MemoryTier, OffloadConfig},
        RequestId, SchedulerLimits,
    };

    fn config() -> &'static str {
        r#"{
            "model_type": "moshi",
            "dim": 16, "text_card": 32, "n_q": 4, "dep_q": 2, "card": 8,
            "num_heads": 4, "num_layers": 2, "causal": true, "context": 16,
            "max_period": 10000, "positional_embedding": "rope",
            "depformer_dim": 8, "depformer_dim_feedforward": 32,
            "depformer_num_heads": 2, "depformer_num_layers": 2,
            "depformer_context": 2, "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 0, 1]
        }"#
    }

    fn personaplex_style_config() -> &'static str {
        r#"{
            "model_type": "personaplex",
            "dim": 16, "text_card": 32, "existing_text_padding_id": 3,
            "n_q": 4, "dep_q": 4, "generated_audio_codebooks": 2, "card": 8,
            "num_heads": 4, "num_layers": 2, "causal": true, "context": 16,
            "max_period": 10000, "positional_embedding": "rope",
            "depformer_dim": 8, "depformer_dim_feedforward": 32,
            "depformer_num_heads": 2, "depformer_num_layers": 2,
            "depformer_context": 4, "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 0, 1]
        }"#
    }

    fn initialize(model: &mut eager::Model, stream: &Stream) {
        let mut names = model
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = model.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("norm1.weight")
                || name.ends_with("norm2.weight")
                || name == "out_norm.weight"
            {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else if name.ends_with(".bias") {
                zeros_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0005 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &eager::Model) {
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| (name.to_string(), *value))
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(dir.join("config.json"), config()).unwrap();
    }

    fn write_pytorch_fixture(path: &Path, model: &eager::Model, stream: &Stream) {
        let params = model.parameters().flatten();
        let mut direct = BTreeMap::<String, Array>::new();
        let mut packed = BTreeMap::<String, Vec<(usize, Array)>>::new();
        for (name, value) in params.iter() {
            let name = name.as_ref();
            if let Some(rest) = name.strip_prefix("audio_embs.") {
                direct.insert(format!("emb.{rest}"), (*value).clone());
            } else if name == "out_norm.weight" {
                direct.insert(
                    "out_norm.alpha".into(),
                    value.reshape(&[1, value.dim(0)], stream).unwrap(),
                );
            } else if name.starts_with("text_emb.") || name.starts_with("text_linear.") {
                direct.insert(name.to_string(), (*value).clone());
            } else if let Some(rest) = name.strip_prefix("transformer.layers.") {
                let source = if rest.ends_with("norm1.weight") || rest.ends_with("norm2.weight") {
                    format!("transformer.layers.{}", rest.replace(".weight", ".alpha"))
                } else if rest.ends_with("self_attn.in_proj.weight") {
                    format!(
                        "transformer.layers.{}",
                        rest.replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
                    )
                } else {
                    format!("transformer.layers.{rest}")
                };
                let transformed = if source.ends_with(".alpha") {
                    value.reshape(&[1, value.dim(0)], stream).unwrap()
                } else {
                    (*value).clone()
                };
                direct.insert(source, transformed);
            } else if let Some(rest) = name.strip_prefix("depformer.slices.") {
                let (slice, rest) = rest.split_once('.').unwrap();
                let slice = slice.parse::<usize>().unwrap();
                if let Some(suffix) = rest.strip_prefix("emb.") {
                    direct.insert(
                        if slice == 0 {
                            format!("depformer_text_emb.{suffix}")
                        } else {
                            format!("depformer_emb.{}.{suffix}", slice - 1)
                        },
                        (*value).clone(),
                    );
                } else if let Some(suffix) = rest.strip_prefix("linear_in.") {
                    direct.insert(format!("depformer_in.{slice}.{suffix}"), (*value).clone());
                } else if let Some(suffix) = rest.strip_prefix("linear_out.") {
                    direct.insert(format!("linears.{slice}.{suffix}"), (*value).clone());
                } else {
                    let rest = rest.strip_prefix("transformer.layers.").unwrap();
                    let (layer, suffix) = rest.split_once('.').unwrap();
                    if suffix == "norm1.weight" || suffix == "norm2.weight" {
                        if slice == 0 {
                            direct.insert(
                                format!(
                                    "depformer.layers.{layer}.{}",
                                    suffix.replace(".weight", ".alpha")
                                ),
                                value.reshape(&[1, value.dim(0)], stream).unwrap(),
                            );
                        }
                    } else if let Some(attention) = suffix.strip_prefix("self_attn.") {
                        let attention = if attention == "in_proj.weight" {
                            "in_proj_weight"
                        } else {
                            attention
                        };
                        packed
                            .entry(format!("depformer.layers.{layer}.self_attn.{attention}"))
                            .or_default()
                            .push((slice, (*value).clone()));
                    } else {
                        let gating = suffix.strip_prefix("gating.").unwrap();
                        direct.insert(
                            format!("depformer.layers.{layer}.gating.{slice}.{gating}"),
                            (*value).clone(),
                        );
                    }
                }
            }
        }
        for (key, mut slices) in packed {
            slices.sort_by_key(|(slice, _)| *slice);
            let values = slices.iter().map(|(_, value)| value).collect::<Vec<_>>();
            direct.insert(key, concatenate_axis(&values, 0, stream).unwrap());
        }
        let arrays = direct.into_iter().collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            path,
        )
        .unwrap();
    }

    fn fixture(gpu: &ExecutionContext) -> tempfile::TempDir {
        let args: ModelArgs = serde_json::from_str(config()).unwrap();
        let mut model = eager::Model::new(args, gpu.stream()).unwrap();
        initialize(&mut model, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &model);
        dir
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn realtime_session_identity_tracks_checkpoint_content_across_residency() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let source = fixture(&gpu);
        let resident =
            crate::api::realtime::load_model(source.path(), gpu.stream(), cpu.stream()).unwrap();
        let layerwise = crate::api::realtime::load_model_with_options(
            source.path(),
            ModelLoadOptions::default().with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            )),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();

        let request = RequestId::new(19);
        let samplers = || vec![DefaultSampler, DefaultSampler];
        let mut resident_scheduler =
            RealtimeInferenceScheduler::new(&resident, SchedulerLimits::new(1, 1).unwrap())
                .unwrap();
        resident_scheduler
            .register_request(
                &resident,
                request,
                DefaultSampler,
                samplers(),
                RealtimeSampling::greedy(),
            )
            .unwrap();
        let session = resident_scheduler.release_request(request).unwrap();

        let mut layerwise_scheduler =
            RealtimeInferenceScheduler::new(&layerwise, SchedulerLimits::new(1, 1).unwrap())
                .unwrap();
        layerwise_scheduler
            .register_request_with_session(&layerwise, request, session)
            .unwrap();
        let session = layerwise_scheduler.release_request(request).unwrap();

        let changed = tempfile::tempdir().unwrap();
        fs::copy(
            source.path().join("config.json"),
            changed.path().join("config.json"),
        )
        .unwrap();
        let mut weights = fs::read(source.path().join("model.safetensors")).unwrap();
        let last = weights.last_mut().unwrap();
        *last ^= 1;
        fs::write(changed.path().join("model.safetensors"), weights).unwrap();
        let changed_model =
            crate::api::realtime::load_model(changed.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut changed_scheduler =
            RealtimeInferenceScheduler::new(&changed_model, SchedulerLimits::new(1, 1).unwrap())
                .unwrap();
        let error = changed_scheduler
            .register_request_with_session(&changed_model, request, session)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("checkpoint artifact fingerprint"));
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn tensor_parallel_dense_stream_loads_realtime_static_and_groups() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let dir = fixture(&gpu);
        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let topology =
            ParallelTopology::from_rank(1, 0, 1, 1, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
                .unwrap();
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let model = load_moshi_tensor_parallel_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(options),
            build,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(report
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().contains("moshi.temporal.")
                || unit.id().as_str().contains("moshi.depth."))
            .all(|unit| unit.planned_tier() == MemoryTier::Disk));
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 5e-5, "{left} != {right}");
        }
    }

    fn assert_tokens_equal(left: &Array, right: &Array, stream: &Stream) {
        let left_array = left.as_dtype(safemlx::Dtype::Int32, stream).unwrap();
        let right_array = right.as_dtype(safemlx::Dtype::Int32, stream).unwrap();
        let left = left_array.evaluated().unwrap();
        let right = right_array.evaluated().unwrap();
        assert_eq!(left.as_slice::<i32>(), right.as_slice::<i32>());
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn native_teacher_forced_and_realtime_cache_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let dir = fixture(&gpu);
        let mut resident = eager::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_moshi_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let text = Array::from_slice(&[1i32], &[1, 1]);
        let audio = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let depth = Array::from_slice(&[2i32, 3], &[1, 2]);
        let mut resident_cache = resident.new_cache();
        let mut layerwise_cache = layerwise.new_cache();
        let expected = resident
            .token_step(&text, &audio, &depth, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .token_step(&text, &audio, &depth, &mut layerwise_cache, gpu.stream())
            .unwrap();
        assert_close(&expected.text_logits, &actual.text_logits);
        assert_close(&expected.temporal_output, &actual.temporal_output);
        for (expected, actual) in expected.audio_logits.iter().zip(&actual.audio_logits) {
            assert_close(expected, actual);
        }

        let mut resident_state = resident::GenerationState::new(&resident);
        let mut layerwise_state = layerwise.new_generation_state();
        let input = Array::from_slice(&[4i32, 5], &[1, 2]);
        let mut resident_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut layerwise_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut resident_audio = (0..2)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        let mut layerwise_audio = (0..2)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        for _ in 0..3 {
            let expected = resident
                .generate_step(
                    &mut resident_state,
                    &input,
                    &mut resident_text,
                    &mut resident_audio,
                    0.0,
                    0.0,
                    None,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .generate_step(
                    &mut layerwise_state,
                    &input,
                    &mut layerwise_text,
                    &mut layerwise_audio,
                    0.0,
                    0.0,
                    None,
                    gpu.stream(),
                )
                .unwrap();
            assert_tokens_equal(&expected.text_token, &actual.text_token, gpu.stream());
            assert_tokens_equal(
                &expected.sampled_audio_tokens,
                &actual.sampled_audio_tokens,
                gpu.stream(),
            );
            assert_eq!(
                expected.output_audio_tokens.is_some(),
                actual.output_audio_tokens.is_some()
            );
        }
        assert_eq!(resident_state.step(), layerwise_state.step());
        let groups = layerwise.execution_group_reports().unwrap();
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|group| group.host_bytes() > 0));
        assert!(groups.iter().all(|group| group.device_units() <= 1));
        layerwise
            .clear_device_group("temporal_transformer")
            .unwrap();
        let groups = layerwise.execution_group_reports().unwrap();
        assert_eq!(groups[0].device_units(), 0);
        assert!(groups[1].device_units() <= 1);
        assert_eq!(
            layerwise.execution.execution_groups()[0].id(),
            "temporal_transformer"
        );
        assert_eq!(
            layerwise.execution.execution_groups()[1].id(),
            "depth_codebook_slices"
        );

        let loaded = crate::api::realtime::load_model_with_options(
            dir.path(),
            crate::api::ModelLoadOptions::default().with_weight_residency(
                crate::runtime::execution::layerwise::WeightResidency::layerwise_host(
                    LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
                ),
            ),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(matches!(
            &loaded,
            crate::api::realtime::LoadedRealtimeModel::Moshi(_)
        ));
        assert_eq!(loaded.execution_group_reports().unwrap().unwrap().len(), 2);
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn dense_stream_teacher_forced_and_realtime_multigroup_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let dir = fixture(&gpu);
        let mut resident = eager::load_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let mut streamed = load_moshi_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(dense),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let initial = streamed.dense_stream_report().unwrap().unwrap();
        assert!(initial
            .residency()
            .units()
            .iter()
            .filter(|unit| {
                ["moshi.temporal.", "moshi.depth_slice."]
                    .iter()
                    .any(|prefix| unit.id().as_str().starts_with(prefix))
            })
            .all(|unit| {
                unit.planned_tier() == MemoryTier::Disk
                    && !unit.host_resident()
                    && !unit.device_resident()
            }));

        let text = Array::from_slice(&[1i32], &[1, 1]);
        let audio = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let depth = Array::from_slice(&[2i32, 3], &[1, 2]);
        let mut resident_cache = resident.new_cache();
        let mut streamed_cache = streamed.new_cache();
        let expected = resident
            .token_step(&text, &audio, &depth, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = streamed
            .token_step(&text, &audio, &depth, &mut streamed_cache, gpu.stream())
            .unwrap();
        assert_close(&expected.text_logits, &actual.text_logits);
        assert_close(&expected.temporal_output, &actual.temporal_output);
        for (expected, actual) in expected.audio_logits.iter().zip(&actual.audio_logits) {
            assert_close(expected, actual);
        }

        let mut resident_state = resident::GenerationState::new(&resident);
        let mut streamed_state = streamed.new_generation_state();
        let input = Array::from_slice(&[4i32, 5], &[1, 2]);
        let mut resident_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut streamed_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut resident_audio = (0..2)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        let mut streamed_audio = (0..2)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        for _ in 0..2 {
            let expected = resident
                .generate_step(
                    &mut resident_state,
                    &input,
                    &mut resident_text,
                    &mut resident_audio,
                    0.0,
                    0.0,
                    None,
                    gpu.stream(),
                )
                .unwrap();
            let actual = streamed
                .generate_step(
                    &mut streamed_state,
                    &input,
                    &mut streamed_text,
                    &mut streamed_audio,
                    0.0,
                    0.0,
                    None,
                    gpu.stream(),
                )
                .unwrap();
            assert_tokens_equal(&expected.text_token, &actual.text_token, gpu.stream());
            assert_tokens_equal(
                &expected.sampled_audio_tokens,
                &actual.sampled_audio_tokens,
                gpu.stream(),
            );
        }
        assert_eq!(resident_state.step(), streamed_state.step());
        let report = streamed.dense_stream_report().unwrap().unwrap();
        assert!(report.decode_forwards() >= 3);

        let loaded = crate::api::realtime::load_model_with_options(
            dir.path(),
            crate::api::ModelLoadOptions::default().with_weight_residency(
                crate::runtime::execution::layerwise::WeightResidency::dense_disk_stream(dense),
            ),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(matches!(
            &loaded,
            crate::api::realtime::LoadedRealtimeModel::Moshi(_)
        ));
        assert!(loaded.dense_stream_report().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn native_offline_encoded_sequence_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let dir = fixture(&gpu);
        let resident = load_moshi_layerwise_model(
            dir.path(),
            LayerWeightResidency::FullyResident,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let layerwise = load_moshi_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let input = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[1, 2, 3]);
        let mut resident = LoadedRealtimeModel::Moshi(resident);
        let mut layerwise = LoadedRealtimeModel::Moshi(layerwise);
        let expected = generate_encoded_greedy(&mut resident, &input, gpu.stream()).unwrap();
        let actual = generate_encoded_greedy(&mut layerwise, &input, gpu.stream()).unwrap();
        assert_eq!(
            expected.text_tokens.evaluated().unwrap().as_slice::<u32>(),
            actual.text_tokens.evaluated().unwrap().as_slice::<u32>()
        );
        assert_eq!(
            expected.audio_tokens.evaluated().unwrap().as_slice::<u32>(),
            actual.audio_tokens.evaluated().unwrap().as_slice::<u32>()
        );
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn pytorch_layout_forced_prompt_cache_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let args: ModelArgs = serde_json::from_str(personaplex_style_config()).unwrap();
        let mut fixture = eager::Model::new(args.clone(), gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_pytorch_fixture(&path, &fixture, gpu.stream());

        let mut resident =
            eager::load_pytorch_safetensors_model(args.clone(), &path, gpu.stream(), cpu.stream())
                .unwrap();
        let mut layerwise = load_with_layout(
            &path,
            args,
            CheckpointLayout::Pytorch,
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut resident_state = resident::GenerationState::new(&resident);
        let mut layerwise_state = layerwise.new_generation_state();
        let user = Array::from_slice(&[3i32, 4], &[1, 2]);
        let agent = Array::from_slice(&[1i32, 2], &[1, 2]);
        let forced_text = Array::from_slice(&[5i32], &[1, 1]);
        let mut resident_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut layerwise_text = crate::runtime::generation::sampler::DefaultSampler;
        let mut resident_audio = (0..4)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();
        let mut layerwise_audio = (0..4)
            .map(|_| crate::runtime::generation::sampler::DefaultSampler)
            .collect::<Vec<_>>();

        for forced in [true, true, false] {
            let expected = if forced {
                resident
                    .generate_step_forced(
                        &mut resident_state,
                        &user,
                        Some(&agent),
                        Some(&forced_text),
                        &mut resident_text,
                        &mut resident_audio,
                        0.0,
                        0.0,
                        None,
                        gpu.stream(),
                    )
                    .unwrap()
            } else {
                resident
                    .generate_step(
                        &mut resident_state,
                        &user,
                        &mut resident_text,
                        &mut resident_audio,
                        0.0,
                        0.0,
                        None,
                        gpu.stream(),
                    )
                    .unwrap()
            };
            let actual = if forced {
                layerwise
                    .generate_step_forced(
                        &mut layerwise_state,
                        &user,
                        Some(&agent),
                        Some(&forced_text),
                        &mut layerwise_text,
                        &mut layerwise_audio,
                        0.0,
                        0.0,
                        None,
                        gpu.stream(),
                    )
                    .unwrap()
            } else {
                layerwise
                    .generate_step(
                        &mut layerwise_state,
                        &user,
                        &mut layerwise_text,
                        &mut layerwise_audio,
                        0.0,
                        0.0,
                        None,
                        gpu.stream(),
                    )
                    .unwrap()
            };
            assert_tokens_equal(&expected.text_token, &actual.text_token, gpu.stream());
            assert_tokens_equal(
                &expected.sampled_audio_tokens,
                &actual.sampled_audio_tokens,
                gpu.stream(),
            );
        }
        assert_eq!(resident_state.step(), 3);
        assert_eq!(resident_state.step(), layerwise_state.step());
    }
}
