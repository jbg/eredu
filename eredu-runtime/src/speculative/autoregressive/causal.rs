//! Single-model causal and refinement invocation through the shared transaction.
use super::*;

/// One ordinary model and one canonical cache, without assistant state.
/// Optional target refinement owns separate numerical state and randomness.
/// The common request table uses zero proposals; settled suffix append uses the
/// same checkpoint, exact completion and rollback contract.
pub struct CausalExecutor<'a, M: AutoregressiveMechanisms, R: TargetRefinement<M>> {
    model: &'a mut M::Model,
    refinement: R,
}

/// Immutable prefix and numerical continuation for an ordinary single-model lane.
#[derive(Clone)]
pub struct CausalSeed<C, R> {
    prefix: Option<C>,
    refinement: R,
}
impl<C, R> CausalSeed<C, R> {
    /// Binds an existing canonical boundary without executing another prompt.
    pub fn new(prefix: Option<C>, refinement: R) -> Self {
        Self { prefix, refinement }
    }
}
impl<'a, M: AutoregressiveMechanisms, R: TargetRefinement<M>> CausalExecutor<'a, M, R> {
    /// Borrows the retained parameter owner without constructing another model.
    pub fn new(model: &'a mut M::Model, refinement: R) -> Self {
        Self { model, refinement }
    }
    /// Realizes an isolated mutable causal lane using the model's ordinary geometry.
    pub fn new_cache(&mut self, context: M::Context<'_>) -> Result<M::State, M::Error> {
        M::empty(self.model, AutoregressivePass::TargetPrefill, context)
    }
}
impl<M: AutoregressiveMechanisms, R: TargetRefinement<M>> SpeculativeExecutor
    for CausalExecutor<'_, M, R>
{
    type Input = AutoregressiveInput<M::Input, R::State>;
    type Cache = M::State;
    type TargetState = CausalSeed<M::Checkpoint, R::State>;
    type DraftState = R::State;
    type CacheCheckpoint = M::Checkpoint;
    type Verification = CausalVerification<M::Output>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a> where Self: 'a;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        let mut capabilities = self.refinement.capabilities(self.model);
        capabilities.speculation = false;
        capabilities.max_draft_tokens = None;
        capabilities.mode_changes = capabilities.diffusion;
        capabilities.with_completion_support(
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::NativeCancel,
            ),
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            ),
        )
    }
    fn refine_block<'a>(
        &mut self,
        state: &mut Self::TargetState,
        logits: &M::Logits,
        committed: &[u32],
        policy: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        context: M::Context<'a>,
    ) -> Result<Option<eredu_core::speculative::RefinementProgress<M::Logits>>, M::Error>  where Self: 'a {
        let Some(prefix) = &state.prefix else {
            return Ok(None);
        };
        self.refinement.advance(
            self.model,
            prefix,
            &mut state.refinement,
            logits,
            committed,
            policy,
            temperature,
            context,
        )
    }
    fn max_proposals(&self) -> usize {
        0
    }
    fn logits_snapshot_estimate(&self, logits: &M::Logits) -> Option<SnapshotEstimate> {
        M::logits_snapshot_estimate(logits)
    }
    fn coordinate_speculative_step<'context>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: M::Context<'context>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>  where Self: 'context {
        M::coordinate_speculative_step(local, context)
    }
    fn coordinate_decoding_control<'context>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: M::Context<'context>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>  where Self: 'context {
        M::coordinate_decoding_control(local, context)
    }
    fn agree_text_preparation<'context>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: M::Context<'context>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
     where Self: 'context {
        M::agree_text_preparation(stage, status, context)
    }
    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut M::State,
        context: M::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, M::Logits>, M::Error>  where Self: 'context {
        let (output, count) = M::prefill(
            self.model,
            &input.input,
            cache,
            AutoregressivePass::TargetPrefill,
            context,
        )?;
        if count == 0 {
            return Err(M::invalid("causal generation requires a nonempty prompt"));
        }
        Ok(SpeculativePrefill::new(
            M::logits(&output, count - 1, context)?,
            CausalSeed {
                prefix: if self.refinement.capabilities(self.model).diffusion {
                    Some(M::checkpoint(cache)?)
                } else {
                    None
                },
                refinement: input.refinement,
            },
            count,
        ))
    }
    fn begin_proposal<'context>(
        &mut self,
        state: &Self::TargetState,
        _: u32,
        capacity: usize,
        _: M::Context<'context>,
    ) -> Result<Self::DraftState, M::Error>  where Self: 'context {
        if capacity != 0 {
            return Err(M::invalid("causal execution has no retained draft source"));
        }
        Ok(state.refinement.clone())
    }
    fn proposal_step<'context>(
        &mut self,
        _: &mut Self::DraftState,
        _: u32,
        _: M::Context<'context>,
    ) -> Result<eredu_core::SpeculativeProposalStep<M::Logits>, M::Error>  where Self: 'context {
        Err(M::invalid("causal execution has no retained draft source"))
    }
    fn checkpoint(&self, cache: &M::State) -> Result<M::Checkpoint, M::Error> {
        M::checkpoint(cache)
    }
    fn restore_checkpoint<'context>(
        &mut self,
        cache: &mut M::State,
        saved: &M::Checkpoint,
        context: M::Context<'context>,
    ) -> Result<(), M::Error>  where Self: 'context {
        *cache = M::restore(saved, context)?;
        Ok(())
    }
    fn submit_verification<'context>(
        &mut self,
        tokens: &[u32],
        cache: &mut M::State,
        context: M::Context<'context>,
    ) -> Result<Submission<Self::Verification, M::Completion>, M::Error>  where Self: 'context {
        causal_verification::submit::<M>(self.model, tokens, cache, context)
    }
    fn verification_logits<'context>(
        &self,
        output: &Self::Verification,
        position: usize,
        context: M::Context<'context>,
    ) -> Result<M::Logits, M::Error>  where Self: 'context {
        causal_verification::logits::<M>(output, position, context)
    }

    fn commit_verification<'context>(
        &mut self,
        output: Self::Verification,
        refinement: Self::DraftState,
        cache: &mut M::State,
        _: &M::Checkpoint,
        retained: usize,
        _: M::Context<'context>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, M::Error>  where Self: 'context {
        // Every input in a causal append is already committed. There is no
        // speculative proposal suffix that can be rejected or replayed.
        if retained == 0 || retained != output.inputs.len() {
            return Err(M::invalid(
                "causal append must retain its complete committed suffix",
            ));
        }
        Ok(SpeculativeCommit::new(
            CausalSeed {
                prefix: if self.refinement.capabilities(self.model).diffusion {
                    Some(M::checkpoint(cache)?)
                } else {
                    None
                },
                refinement,
            },
            0,
        ))
    }
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Ok(
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target: M::trim_parameter_conversions(self.model)?,
                external_drafter: None,
            },
        )
    }
    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(Some(
            eredu_core::residency::ExecutionConversionRetentionReport {
                target: M::parameter_conversion_retention(self.model)?,
                external_drafter: None,
            },
        ))
    }
    fn continuation_memory_observation(
        &self,
        cache: &M::State,
        state: &Self::TargetState,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(
            M::memory_observation(self.model, cache, additional)?.map(|target| {
                eredu_core::speculative::SpeculativeContinuationObservation {
                    target,
                    draft: None,
                    embedded: None,
                    parameter_conversions: None,
                    seed_bytes: state
                        .prefix
                        .as_ref()
                        .map_or(Some(0), M::checkpoint_retained_bytes)
                        .and_then(|bytes| {
                            bytes.checked_add(
                                self.refinement.estimate(&state.refinement)?.retained_bytes,
                            )
                        }),
                }
            }),
        )
    }
    fn control_snapshot_estimate(
        &self,
        cache: &M::State,
        state: &Self::TargetState,
    ) -> Option<SnapshotEstimate> {
        let cache = M::estimate_state(cache)?;
        let prefix = state.prefix.as_ref().map_or(
            Some(SnapshotEstimate {
                retained_bytes: 0,
                copy_bytes: 0,
            }),
            M::estimate,
        )?;
        let numerical = self.refinement.estimate(&state.refinement)?;
        Some(SnapshotEstimate {
            retained_bytes: cache
                .retained_bytes
                .checked_add(prefix.retained_bytes)?
                .checked_add(numerical.retained_bytes)?,
            copy_bytes: cache
                .copy_bytes
                .checked_add(prefix.copy_bytes)?
                .checked_add(numerical.copy_bytes)?,
        })
    }
    fn control_snapshot<'context>(
        &self,
        cache: &M::State,
        state: &Self::TargetState,
        _: M::Context<'context>,
    ) -> Result<
        Option<(M::Checkpoint, Self::TargetState)>,
        eredu_core::speculative::SpeculativeControlError,
    >  where Self: 'context {
        Ok(Some((
            M::checkpoint(cache)
                .map_err(eredu_core::speculative::SpeculativeControlError::backend)?,
            state.clone(),
        )))
    }
    fn restore_control_snapshot<'context>(
        &mut self,
        cache: &mut M::State,
        saved: &M::Checkpoint,
        state: &Self::TargetState,
        context: M::Context<'context>,
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError>  where Self: 'context {
        self.restore_checkpoint(cache, saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(state.clone()))
    }
}
