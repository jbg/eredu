//! Generic ordinary-decoder operations for the portable independent-draft transaction.
use super::*;
use crate::composition::mlx::{
    MlxModelInput,
    replicated_text::{ErasedReplicatedTextExecutable, MlxPredictionTargetState},
};
use eredu_runtime::speculative::autoregressive::{AutoregressiveMechanisms, AutoregressivePass};
use std::rc::Rc;

pub(crate) struct MlxAutoregressiveState {
    native: MlxPredictionTargetState,
    stream: Stream,
}
pub(crate) struct MlxAutoregressiveMechanisms;
#[derive(Clone)]
pub(crate) struct MlxAutoregressiveOutput {
    logits: Array,
    stream: Stream,
}
impl MlxAutoregressiveState {
    #[cfg(test)]
    pub(crate) fn retained_arrays(&self) -> Vec<&Array> {
        self.native.retained_arrays()
    }

    pub(crate) fn publish(
        &mut self,
        model: &mut dyn ErasedReplicatedTextExecutable,
    ) -> Result<(), Error> {
        model.exchange_autoregressive_cache(&mut self.native, &self.stream)
    }
}
impl MlxAutoregressiveOutput {
    pub(crate) fn retained_bytes(&self) -> Option<u64> {
        (self.logits.nbytes() as u64)
            .checked_mul(2)?
            .checked_add(4096)
    }
}
fn exception(e: Error) -> Exception {
    Exception::from_source(e)
}
type SharedObserver<'a> =
    dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Exception> + 'a;

fn observed_forward<'a, R>(
    observer: Option<&'a mut SharedObserver<'a>>,
    phase: eredu_core::speculative::SpeculativeActivationPhase,
    geometry: impl Into<eredu_runtime::inspection::SpeculativeActivationGeometry>,
    execute: impl FnOnce(&mut dyn eredu_runtime::ActivationObserver<Array, Error>) -> Result<R, Error>,
) -> Result<R, Exception> {
    eredu_runtime::inspection::with_speculative_activation(observer, phase, geometry, |observer| {
        let Some(observer) = observer else {
            return execute(&mut eredu_runtime::NoopObserver).map_err(exception);
        };
        let mut bridge = eredu_runtime::inspection::ObserverErrorBridge::new(
            observer,
            exception,
            |error: &Exception| Error::Exception(Exception::custom(error.to_string())),
        );
        let mut arrays = crate::composition::mlx::session::ArrayObserverAdapter {
            inner: &mut bridge,
            routed_path: None,
            routed_invocation_active: false,
        };
        let result = execute(&mut arrays).map_err(exception);
        bridge.resolve(result)
    })
}

impl AutoregressiveMechanisms for MlxAutoregressiveMechanisms {
    fn logits_snapshot_estimate(
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::state_snapshot::estimate_array(logits)
    }
    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        context.coordinate_speculative_step(local)
    }
    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        context.agree_text_preparation(stage, status)
    }
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        context.coordinate_decoding_control(local)
    }
    type Model = dyn ErasedReplicatedTextExecutable;
    type Input = MlxModelInput;
    type State = MlxAutoregressiveState;
    type Checkpoint = Rc<MlxAutoregressiveState>;
    type Output = MlxAutoregressiveOutput;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = scheduler::SpeculativeComponentTimings;
    type Error = Exception;
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Exception> {
        let stream = match pass {
            AutoregressivePass::DraftPrefill => context.draft(),
            _ => context.target(),
        };
        Ok(MlxAutoregressiveState {
            native: model.prepare_autoregressive_cache().map_err(exception)?,
            stream: stream.clone(),
        })
    }
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        _: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<(Self::Output, usize), Exception> {
        let mut count = 0;
        let output = context
            .observe_target(|observer| {
                input.with_borrowed(|input| {
                    let (output, positions) = model.prepared_context_forward(
                        input,
                        &mut state.native,
                        &state.stream,
                        observer,
                    )?;
                    count = positions;
                    Ok(output)
                })
            })
            .map_err(exception)?;
        Ok((
            MlxAutoregressiveOutput {
                logits: output,
                stream: state.stream.clone(),
            },
            count,
        ))
    }
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        _: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, Exception> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("causal append exceeds native sequence geometry"))?;
        if width == 0 {
            return Err(Exception::custom(
                "causal append requires a nonempty sequence",
            ));
        }
        let tokens = Array::from_slice(tokens, &[1, width]);
        context
            .observe_target(|observer| {
                model.attention_forward_with_observer(
                    &tokens,
                    None,
                    eredu_core::decoding::AttentionInvocation::CAUSAL,
                    Some(&mut state.native),
                    false,
                    &state.stream,
                    observer,
                )
            })
            .map(|logits| MlxAutoregressiveOutput {
                logits,
                stream: state.stream.clone(),
            })
            .map_err(exception)
    }
    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Exception> {
        Ok(Rc::new(MlxAutoregressiveState {
            native: state.native.checkpoint_view()?,
            stream: state.stream.clone(),
        }))
    }
    fn restore(saved: &Self::Checkpoint, _: Self::Context<'_>) -> Result<Self::State, Exception> {
        Ok(MlxAutoregressiveState {
            native: saved.native.checkpoint_view()?,
            stream: saved.stream.clone(),
        })
    }
    fn retain_verified_prefix(
        state: &mut Self::State,
        saved: &Self::Checkpoint,
        accepted_inputs: usize,
        context: Self::Context<'_>,
    ) -> Result<bool, Exception> {
        let length = saved
            .native
            .generation()
            .map_err(exception)?
            .checked_add(accepted_inputs as u64)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| Exception::custom("accepted prefix exceeds native geometry"))?;
        state.native.retain_causal_prefix(length, context.target())
    }
    fn estimate(
        saved: &Self::Checkpoint,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::estimate_state(saved)
    }
    fn checkpoint_retained_bytes(saved: &Self::Checkpoint) -> Option<u64> {
        saved
            .native
            .control_estimate()?
            .retained_bytes
            .checked_add(saved.native.control_growth(0)?)
    }
    fn estimate_state(
        state: &Self::State,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.native.control_estimate()
    }
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        model: &mut Self::Model,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        model.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        model: &Self::Model,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        eredu_core::BackendFailure,
    > {
        model.parameter_conversion_retention()
    }

    fn memory_observation(
        model: &Self::Model,
        state: &Self::State,
        additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeModelMemoryObservation>,
        eredu_core::BackendFailure,
    > {
        let position = state
            .native
            .generation()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let retained = state.native.control_estimate().map(|e| e.retained_bytes);
        let current = retained.and_then(|n| n.checked_add(state.native.control_growth(0)?));
        let peak = retained
            .and_then(|n| n.checked_add(state.native.control_growth(additional_positions)?));
        let parameters = crate::composition::mlx::capability::static_memory_from_residency(
            model
                .residency_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
            model
                .parameter_bank_report()
                .map_err(eredu_core::BackendFailure::from_error)?,
        )
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Some(
            eredu_core::speculative::SpeculativeModelMemoryObservation {
                parameter_conversions: model
                    .residency_report()
                    .map_err(eredu_core::BackendFailure::from_error)?
                    .and_then(|r| r.device_parameter_conversions().map(<[_]>::to_vec)),
                current_positions: position,
                current_state_bytes: current,
                peak_state_bytes: peak,
                parameters,
                available: crate::composition::mlx::capability::available_memory()
                    .map_err(eredu_core::BackendFailure::from_error)?,
                allocator_cache_limit: match crate::allocator_cache_policy() {
                    Ok(policy) => eredu_core::Observed::exact(
                        policy.limit_bytes,
                        "current MLX allocator-cache policy",
                    ),
                    Err(error) => eredu_core::Observed::unavailable(error.to_string()),
                },
            },
        ))
    }
    fn logits(
        output: &Self::Output,
        position: usize,
        _: Self::Context<'_>,
    ) -> Result<Array, Exception> {
        output
            .logits
            .try_index_device((.., position as i32, ..), &output.stream)
    }
    fn completion(
        output: &Self::Output,
        _: Self::Context<'_>,
    ) -> Result<Self::Completion, Exception> {
        MlxSpeculativeCompletion::submit([&output.logits])
    }
    fn invalid(message: &'static str) -> Exception {
        Exception::custom(message)
    }
}

impl eredu_runtime::speculative::shared_backbone::SharedBackboneMechanisms
    for MlxAutoregressiveMechanisms
{
    type ActivationTensor = MlxTensor;
    fn activation_observer(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'_>,
    ) -> Result<
        Option<Box<SharedObserver<'static>>>,
        eredu_core::speculative::SpeculativeControlError,
    > {
        match context.capture_binding() {
            Some(binding) => binding.observer(plan, request, context.target()),
            None => crate::composition::mlx::session::bounded_capture::speculative_capture(
                plan,
                request,
                context.target(),
            ),
        }
    }
    fn prefill_observed(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        context: Self::Context<'_>,
        mut observer: Option<&mut SharedObserver<'_>>,
    ) -> Result<(Self::Output, usize), Exception> {
        if observer.is_none() {
            return Self::prefill(
                model,
                input,
                state,
                AutoregressivePass::TargetPrefill,
                context,
            );
        }
        let plans = input
            .with_borrowed(|input| model.prepared_input_plans(input))
            .map_err(Exception::from_source)?;
        let media = eredu_architectures::media_plan::capture_media_shape(&plans)
            .map_err(Exception::from_source)?;
        let positions = plans
            .iter()
            .try_fold(0usize, |sum, plan| {
                sum.checked_add(usize::try_from(plan.decoder_positions()).ok()?)
            })
            .ok_or_else(|| Exception::custom("prepared observation geometry overflows"))?;
        let (logits, count) = observed_forward(
            observer
                .as_mut()
                .map(|observer| &mut **observer as &mut SharedObserver<'_>),
            eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill,
            eredu_runtime::inspection::SpeculativeActivationGeometry {
                sequence: positions,
                media,
            },
            |observer| {
                input.with_borrowed(|input| {
                    model.prepared_context_forward(
                        input,
                        &mut state.native,
                        &state.stream,
                        observer,
                    )
                })
            },
        )?;
        Ok((
            MlxAutoregressiveOutput {
                logits,
                stream: state.stream.clone(),
            },
            count,
        ))
    }
    fn decode_observed(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
        mut observer: Option<&mut SharedObserver<'_>>,
    ) -> Result<Self::Output, Exception> {
        if observer.is_none() {
            return Self::decode(model, tokens, state, pass, context);
        }
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("causal observation geometry overflows"))?;
        if width == 0 {
            return Err(Exception::custom(
                "causal observation requires nonempty input",
            ));
        }
        let phase = match pass {
            AutoregressivePass::Verification => {
                eredu_core::speculative::SpeculativeActivationPhase::Verification
            }
            _ => eredu_core::speculative::SpeculativeActivationPhase::TargetReplay,
        };
        let input = Array::from_slice(tokens, &[1, width]);
        let logits = observed_forward(
            observer
                .as_mut()
                .map(|observer| &mut **observer as &mut SharedObserver<'_>),
            phase,
            tokens.len(),
            |observer| {
                model.attention_forward_with_observer(
                    &input,
                    None,
                    eredu_core::decoding::AttentionInvocation::CAUSAL,
                    Some(&mut state.native),
                    false,
                    &state.stream,
                    observer,
                )
            },
        )?;
        Ok(MlxAutoregressiveOutput {
            logits,
            stream: state.stream.clone(),
        })
    }
    type DraftOptions = eredu_runtime::speculative::shared_backbone::SharedBackboneDraftOptions;
    type ProposalRandomness = RandomState;
    type RefinementState =
        eredu_runtime::generation::refinement::RefinementState<MlxSamplingBackend>;
    fn refinement_state_estimate(
        state: &Self::RefinementState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        use eredu_runtime::generation::refinement::RefinementValue;
        let host = state.snapshot_host_bytes()?;
        let mut estimate = Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: host,
            copy_bytes: host,
        });
        state.visit_retained_values(|value| {
            let array = match value {
                RefinementValue::Tokens(value)
                | RefinementValue::Scores(value)
                | RefinementValue::Mask(value) => value.as_array(),
                RefinementValue::Random(value) => value.as_array(),
            };
            estimate = estimate.and_then(|sum| {
                let value = super::state_snapshot::estimate_array(array)?;
                Some(eredu_core::execution_control::SnapshotEstimate {
                    retained_bytes: sum.retained_bytes.checked_add(value.retained_bytes)?,
                    copy_bytes: sum.copy_bytes.checked_add(value.copy_bytes)?,
                })
            });
        });
        estimate
    }

    fn prepare_draft_options(
        model: &Self::Model,
        options: eredu_core::decoding::SpeculationOptions,
        temperature: f32,
    ) -> Result<Self::DraftOptions, eredu_core::decoding::DecodingModeError> {
        model
            .capability_estimate()
            .shared_backbone_draft()
            .ok_or(eredu_core::decoding::DecodingModeError::Unsupported(
                "no retained shared-backbone declaration",
            ))?
            .prepare(options, temperature)
    }
    fn draft_capabilities(model: &Self::Model) -> eredu_core::decoding::DecodingCapabilities {
        let estimate = model.capability_estimate();
        let shared = estimate.shared_backbone_draft();
        let ordinary = estimate.ordinary_refinement();
        eredu_core::decoding::DecodingCapabilities {
            autoregressive: ordinary.is_none_or(|policy| policy.causal_sampling),
            diffusion: ordinary.is_some(),
            speculation: shared.is_some(),
            mode_changes: true,
            entropy_feedback: ordinary.is_some_and(|policy| {
                matches!(
                    policy.policy,
                    eredu_core::refinement::RefinementPolicy::EntropyFeedback(_)
                )
            }),
            masked_unmasking: ordinary.is_some_and(|policy| {
                matches!(
                    policy.policy,
                    eredu_core::refinement::RefinementPolicy::MaskedUnmasking { .. }
                )
            }),
            mask_token_id: ordinary.and_then(|policy| match policy.policy {
                eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
                    mask_token_id, ..
                } => Some(mask_token_id),
                _ => None,
            }),
            draft_adapter: shared.is_some_and(|policy| policy.adapter_available()),
            ..Default::default()
        }
    }
    fn draft_options_bytes(_: &Self::DraftOptions) -> Option<u64> {
        Some(std::mem::size_of::<Self::DraftOptions>() as u64)
    }

    fn proposal_randomness_bytes(random: &RandomState) -> Option<u64> {
        let survey = safemlx::inspect_storage(&[random.as_array()]).ok()?;
        let backing = &survey.backings[*survey.values.first()?.as_ref()?];
        (backing.allocator_capacity_bytes? as u64)
            .checked_add(std::mem::size_of::<RandomState>() as u64)?
            .checked_add(4096)
    }

    fn refine(
        model: &mut Self::Model,
        prefix: &Self::Checkpoint,
        next_logits: &Array,
        committed: &[u32],
        requested: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        randomness: &mut RandomState,
        paused: &mut Option<Self::RefinementState>,
        context: Self::Context<'_>,
        mut observer: Option<&mut SharedObserver<'_>>,
    ) -> Result<Option<eredu_core::speculative::RefinementProgress<Array>>, Exception> {
        use eredu_core::refinement::{RefinementCanvas, RefinementPolicy};
        use eredu_runtime::generation::refinement::{RefinementBackend, RefinementState};
        let declaration = model
            .capability_estimate()
            .ordinary_refinement()
            .ok_or_else(|| Exception::custom("no retained refinement declaration"))?;
        let policy = declaration
            .prepare(requested, temperature)
            .map_err(Exception::from_source)?;
        let geometry = declaration.geometry;
        if committed.len() >= geometry.canvas_length.get() as usize {
            return Err(Exception::custom(
                "refinement prefix exceeds admitted geometry",
            ));
        }
        let stream = context.target();
        if paused.is_none() {
            let layout = if let RefinementPolicy::MaskedUnmasking {
                config,
                mask_token_id,
            } = policy
            {
                let mut layout = RefinementCanvas {
                    seed_tokens: vec![None; geometry.canvas_length.get() as usize],
                    pinned_positions: (0..committed.len())
                        .map(|position| position as u32)
                        .collect(),
                };
                for (seed, token) in layout.seed_tokens.iter_mut().zip(committed) {
                    *seed = Some(*token);
                }
                if committed.is_empty() && declaration.seed_first {
                    let mask =
                        MlxTensor::from_array(Array::from_slice(&[mask_token_id as i32], &[1]));
                    let sampled = MlxSamplingBackend::masked_refinement_step(
                        &mask,
                        None,
                        &MlxTensor::from_array(next_logits.clone()),
                        mask_token_id,
                        config,
                        1,
                        randomness,
                        stream,
                    )?
                    .ok_or_else(|| Exception::custom("masked seed sampling is unavailable"))?;
                    let seed = sampled
                        .canvas
                        .as_array()
                        .as_dtype(safemlx::Dtype::Uint32, stream)?
                        .into_evaluated()?
                        .try_to_vec::<u32>()
                        .map_err(Exception::from_source)?[0];
                    layout.seed_tokens[0] = Some(seed);
                    if seed != mask_token_id {
                        layout.pinned_positions.push(0);
                    }
                }
                layout
                    .validate(
                        geometry.canvas_length.get(),
                        geometry.vocabulary_size.get() as usize,
                    )
                    .map_err(Exception::from_source)?;
                Some(layout)
            } else {
                if !committed.is_empty() {
                    return Err(Exception::custom(
                        "entropy refinement has no pinned selected prefix",
                    ));
                }
                None
            };
            *paused = Some(
                RefinementState::<MlxSamplingBackend>::new(
                    policy,
                    geometry.canvas_length,
                    eredu_runtime::TokenDomain::new(geometry.vocabulary_size.get() as usize),
                    next_logits.dtype(),
                    randomness.clone(),
                    layout.as_ref(),
                    stream,
                )
                .map_err(Exception::from_source)?,
            );
        }
        let mut transient_prefix = Self::restore(prefix, context)?;
        let refinement = paused.as_mut().expect("initialized numerical continuation");
        if !refinement.is_finished() {
            let canvas = refinement
                .canvas()
                .as_array()
                .reshape(&[1, geometry.canvas_length.get() as i32], stream)?;
            let feedback = refinement
                .self_conditioning()
                .map(|value| value.as_array().reshape(&[1, geometry.canvas_length.get() as i32, geometry.vocabulary_size.get() as i32], stream))
                .transpose()?;
            let scores = observed_forward(
                observer
                    .as_mut()
                    .map(|observer| &mut **observer as &mut SharedObserver<'_>),
                eredu_core::speculative::SpeculativeActivationPhase::Refinement {
                    iteration: refinement.completed_steps(),
                },
                geometry.canvas_length.get() as usize,
                |observer| {
                    model.attention_forward_with_observer(
                        &canvas,
                        feedback.as_ref(),
                        eredu_core::decoding::AttentionInvocation::DIFFUSION,
                        Some(&mut transient_prefix.native),
                        false,
                        stream,
                        observer,
                    )
                },
            )?;
            let scores = scores.reshape(
                &[
                    geometry.canvas_length.get() as i32,
                    geometry.vocabulary_size.get() as i32,
                ],
                stream,
            )?;
            refinement
                .advance(&MlxTensor::from_array(scores), stream)
                .map_err(Exception::from_source)?;
        }
        let mut retained = Vec::new();
        refinement.visit_retained_values(|value| {
            use eredu_runtime::generation::refinement::RefinementValue;
            let array = match value {
                RefinementValue::Tokens(value) | RefinementValue::Scores(value) | RefinementValue::Mask(value) => value.as_array(),
                RefinementValue::Random(value) => value.as_array(),
            };
            retained.push(array.clone());
        });
        super::state_snapshot::settle(retained.iter().collect())?;
        *randomness = refinement.clone().into_random();
        if !refinement.is_finished() {
            return Ok(Some(eredu_core::speculative::RefinementProgress::Pending {
                width: geometry.canvas_length.get() as usize,
                evaluated_positions: geometry.canvas_length.get() as usize,
                iterations: 1,
            }));
        }
        let tokens = refinement
            .candidates()
            .ok_or_else(|| Exception::custom("refinement has no selected canvas"))?
            .as_array()
            .as_dtype(safemlx::Dtype::Uint32, stream)?
            .into_evaluated()?
            .try_to_vec::<u32>()
            .map_err(Exception::from_source)?;
        let scores = refinement
            .final_scores()
            .ok_or_else(|| Exception::custom("refinement has no position scores"))?;
        let rows = tokens
            .into_iter()
            .enumerate()
            .map(|(position, token)| {
                scores
                    .as_array()
                    .try_index_device((position as i32..position as i32 + 1, ..), stream)
                    .map(|scores| (token, scores))
            })
            .collect::<Result<Vec<_>, _>>()?;
        *randomness = paused
            .take()
            .expect("completed numerical continuation")
            .into_random();
        Ok(Some(eredu_core::speculative::RefinementProgress::Complete(
            eredu_core::speculative::RefinementBlock {
                positions: rows,
                iterations: 1,
                evaluated_positions: geometry.canvas_length.get() as usize,
            },
        )))
    }

    fn draft(
        model: &mut Self::Model,
        prefix: &Self::Checkpoint,
        last_token: u32,
        capacity: usize,
        options: &Self::DraftOptions,
        randomness: &mut RandomState,
        context: Self::Context<'_>,
        mut observer: Option<&mut SharedObserver<'_>>,
    ) -> Result<Vec<u32>, Exception> {
        use eredu_runtime::generation::refinement::RefinementState;
        let geometry = options.geometry();
        if model.block_refinement_geometry() != Some(geometry)
            || capacity == 0
            || capacity >= geometry.canvas_length.get() as usize
            || context.crosses_devices()
        {
            return Err(Exception::custom(
                "shared-backbone draft does not match retained canvas or device geometry",
            ));
        }
        let mut layout = eredu_core::refinement::RefinementCanvas {
            seed_tokens: vec![None; geometry.canvas_length.get() as usize],
            pinned_positions: vec![0],
        };
        layout.seed_tokens[0] = Some(last_token);
        layout
            .validate(
                geometry.canvas_length.get(),
                geometry.vocabulary_size.get() as usize,
            )
            .map_err(Exception::from_source)?;
        let stream = context.draft();
        let mut transient_prefix = Self::restore(prefix, context)?;
        let mut refinement = RefinementState::<MlxSamplingBackend>::new(
            options.policy(),
            geometry.canvas_length,
            eredu_runtime::TokenDomain::new(geometry.vocabulary_size.get() as usize),
            safemlx::Dtype::Float32,
            randomness.clone(),
            Some(&layout),
            stream,
        )
        .map_err(Exception::from_source)?;
        while !refinement.is_finished() {
            let canvas = refinement
                .canvas()
                .as_array()
                .reshape(&[1, geometry.canvas_length.get() as i32], stream)?;
            let scores = observed_forward(
                observer
                    .as_mut()
                    .map(|observer| &mut **observer as &mut SharedObserver<'_>),
                eredu_core::speculative::SpeculativeActivationPhase::MaskedDraft {
                    iteration: refinement.completed_steps(),
                },
                geometry.canvas_length.get() as usize,
                |observer| {
                    model.attention_forward_with_observer(
                        &canvas,
                        None,
                        options.invocation(),
                        Some(&mut transient_prefix.native),
                        false,
                        stream,
                        observer,
                    )
                },
            )?;
            let scores = scores.reshape(
                &[
                    geometry.canvas_length.get() as i32,
                    geometry.vocabulary_size.get() as i32,
                ],
                stream,
            )?;
            refinement
                .advance(&MlxTensor::from_array(scores), stream)
                .map_err(Exception::from_source)?;
        }
        // Selected frozen identities survive finalization; only this bounded
        // proposal slice leaves native storage. The seeded pending token is omitted.
        let tokens = refinement
            .candidates()
            .ok_or_else(|| Exception::custom("masked draft has no settled selections"))?
            .as_array()
            .try_index_device(1..(capacity + 1) as i32, stream)?;
        let tokens = tokens
            .as_dtype(safemlx::Dtype::Uint32, stream)?
            .into_evaluated()?
            .try_to_vec::<u32>()
            .map_err(Exception::from_source)?;
        *randomness = refinement.into_random();
        Ok(tokens)
    }
}
