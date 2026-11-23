//! Speculative sampling over host score rows with explicit role-separated randomness.

#[cfg(feature = "tensor-capture")]
use eredu_core::refinement::PredictionScoreKind;

use eredu_core::sampling::{
    host::{HostSampling, HostScores},
    host_random::HostRandomState,
};
use std::marker::PhantomData;

use eredu_core::{
    capture::SpeculativeCaptureTypes,
    sampling::{SamplingBackend, TokenDomain},
    speculative::SpeculativeControlError,
    BackendFailure, BackendFailureKind, SamplingPlacement, SpeculativeDraftRandomPosition,
    SpeculativeSampling,
};

use crate::SpeculativeSampler;

/// Conversion from an admitted score row to the shared host sampler.
pub trait HostScoreSource<Q: SpeculativeCaptureTypes> {
    /// Exact input row representation used by the selected executor.
    type Logits: Clone;

    /// Resolves the selected row on its owning context.
    fn read(
        logits: &Self::Logits,
        placement: SamplingPlacement,
        context: crate::capture::SpeculativeContextFor<'_, Q>,
    ) -> Result<HostScores, BackendFailure>;
}

/// Already materialized host score rows.
pub struct OwnedHostScores;

impl<Q: SpeculativeCaptureTypes> HostScoreSource<Q> for OwnedHostScores {
    type Logits = HostScores;

    fn read(
        logits: &HostScores,
        _: SamplingPlacement,
        _: crate::capture::SpeculativeContextFor<'_, Q>,
    ) -> Result<HostScores, BackendFailure> {
        Ok(logits.clone())
    }
}

/// Requests one native predictor row through the backend-neutral readout contract.
pub struct NativeHostScores;

impl<Q: eredu_core::backend::local_text::HostSpeculativeScoreRead> HostScoreSource<Q>
    for NativeHostScores
{
    type Logits = Q::Logits;

    fn read(
        logits: &Q::Logits,
        placement: SamplingPlacement,
        context: crate::capture::SpeculativeContextFor<'_, Q>,
    ) -> Result<HostScores, BackendFailure> {
        let queue = match placement {
            SamplingPlacement::Draft => context.draft(),
            _ => context.target(),
        };
        Q::read_predictor_scores(logits, queue).and_then(HostScores::from_row)
    }
}

/// Adapts the shared sampler policy and host arithmetic to the speculative scheduler.
pub struct HostSpeculativeSampling<S, Q, R = OwnedHostScores> {
    inner: S,
    queue: PhantomData<fn() -> (Q, R)>,
    #[cfg(feature = "tensor-capture")]
    capture: Option<std::rc::Rc<std::cell::RefCell<super::sampling::LogitCapture>>>,
}

impl<S, Q, R> HostSpeculativeSampling<S, Q, R> {
    /// Installs the admitted sampler policy without deriving random streams.
    pub const fn new(inner: S) -> Self {
        Self {
            inner,
            queue: PhantomData,
            #[cfg(feature = "tensor-capture")]
            capture: None,
        }
    }
}

impl<S: Clone, Q, R> Clone for HostSpeculativeSampling<S, Q, R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            queue: PhantomData,
            #[cfg(feature = "tensor-capture")]
            capture: self.capture.clone(),
        }
    }
}

fn random_error() -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other("stochastic speculation requires an explicitly seeded host stream"),
    )
}

impl<S, Q, R> SpeculativeSampling for HostSpeculativeSampling<S, Q, R>
where
    S: SpeculativeSampler<HostSampling> + Clone,
    Q: SpeculativeCaptureTypes<
            Transport: 'static,
            InvocationTensor: 'static,
            ObserverError: 'static,
            Scores: 'static,
        > + 'static,
    R: HostScoreSource<Q> + 'static,
{
    type Logits = R::Logits;
    type Distribution = HostScores;
    type Seed = u64;
    type RandomState = HostRandomState;
    type DraftRandomness = u64;
    type RandomnessRoot = u64;
    type Context<'a>
        = crate::capture::SpeculativeContextFor<'a, Q>
    where
        Self: 'a;
    type Error = BackendFailure;

    fn control_requires_positive_temperature(&self) -> Option<bool> {
        self.inner.control_requires_positive_temperature()
    }

    fn control_seed<'a>(seed: u64, _: Self::Context<'a>) -> Result<u64, SpeculativeControlError>
    where
        Self: 'a,
    {
        Ok(seed)
    }

    fn control_force_next(
        &mut self,
        token: u32,
        vocabulary: usize,
        position: usize,
    ) -> Result<(), SpeculativeControlError> {
        self.inner
            .control_force_next(token, TokenDomain::new(vocabulary), position)
    }

    fn control_clear_forced(&mut self) -> bool {
        self.inner.control_clear_forced()
    }

    fn control_pending_forced(&self) -> Option<u32> {
        self.inner.control_pending_forced()
    }

    #[cfg(feature = "tensor-capture")]
    fn enable_control_capture(
        &mut self,
        plan: eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        super::autoregressive::validate_control_capture(&plan)?;
        if !plan.is_empty() {
            self.capture = Some(std::rc::Rc::new(std::cell::RefCell::new(
                super::sampling::LogitCapture::new(plan),
            )));
        }
        Ok(())
    }

    #[cfg(feature = "tensor-capture")]
    fn take_control_captures(
        &mut self,
    ) -> Vec<eredu_core::speculative::SpeculativePredictionCapture> {
        self.capture
            .as_ref()
            .map_or_else(Vec::new, |capture| capture.borrow_mut().drain())
    }

    fn control_snapshot_bytes(
        &self,
        target: Option<&HostRandomState>,
        draft: Option<&u64>,
    ) -> Option<u64> {
        self.inner
            .control_snapshot_bytes()?
            .checked_add(randomness_bytes(target, draft))
    }

    fn continuation_storage_bytes(
        &self,
        target: Option<&HostRandomState>,
        draft: Option<&u64>,
        additional: u64,
    ) -> Option<u64> {
        self.inner
            .continuation_storage_bytes(additional)?
            .checked_add(randomness_bytes(target, draft))?
            .checked_add((size_of::<HostRandomState>() + size_of::<u64>()) as u64)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        self.inner.supports_exact_optimistic_promotion()
    }

    fn grammar_is_complete(&mut self) -> Result<bool, BackendFailure> {
        self.inner.grammar_is_complete()
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, BackendFailure> {
        self.inner.prefix_is_complete(history)
    }

    fn randomness_root<'a>(seed: Option<u64>, _: Self::Context<'a>) -> Result<u64, BackendFailure>
    where
        Self: 'a,
    {
        seed.ok_or_else(random_error)
    }

    fn target_randomness_from_root<'a>(
        seed: &mut u64,
        _: Self::Context<'a>,
    ) -> Result<HostRandomState, BackendFailure>
    where
        Self: 'a,
    {
        Ok(HostRandomState::with_seed(*seed))
    }

    fn draft_randomness_from_root<'a>(
        seed: &mut u64,
        _: Self::Context<'a>,
    ) -> Result<u64, BackendFailure>
    where
        Self: 'a,
    {
        Ok(HostRandomState::with_stream(*seed, 1).next_u64())
    }

    fn draft_randomness_at<'a>(
        root: &u64,
        position: SpeculativeDraftRandomPosition,
        _: Self::Context<'a>,
    ) -> Result<HostRandomState, BackendFailure>
    where
        Self: 'a,
    {
        let stream = u64::try_from(position.get()).map_err(BackendFailure::from_error)?;
        Ok(HostRandomState::with_stream(*root, stream))
    }

    fn process_logits<'a>(
        &mut self,
        logits: &R::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<HostScores, BackendFailure>
    where
        Self: 'a,
    {
        let scores = R::read(logits, placement, context)?;
        #[cfg(feature = "tensor-capture")]
        if let Some(capture) = &self.capture {
            use eredu_core::speculative::SpeculativeCaptureRole;
            let role = match placement {
                SamplingPlacement::Target => SpeculativeCaptureRole::Target,
                SamplingPlacement::Draft => SpeculativeCaptureRole::Draft,
                _ => {
                    return Err(BackendFailure::new(
                        BackendFailureKind::Unsupported,
                        std::io::Error::other("no host observation for this sampling placement"),
                    ))
                }
            };
            return self.inner.process_logits_with_capture(
                &scores,
                temperature,
                history,
                &(),
                |logits, domain| {
                    capture.borrow_mut().observe_read_only(
                        &mut score_capture::HostScoreCapture(domain),
                        logits,
                        role,
                        history.len() as u64,
                        PredictionScoreKind::CausalConditional,
                    )?;
                    Ok(logits.clone())
                },
            );
        }
        let _ = placement;
        self.inner
            .process_logits(&scores, temperature, history, &())
    }

    fn finalize_refinement_position<'a>(
        &mut self,
        logits: &R::Logits,
        selected: u32,
        mask_token_id: Option<u32>,
        history: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Option<(u32, HostScores)>, BackendFailure>
    where
        Self: 'a,
    {
        use crate::generation::refinement::{
            finalization::{finalize_refinement_position, RefinementSelection},
            host::{scalar, HostRefinement, ScoreMatrix},
        };

        let scores = R::read(logits, SamplingPlacement::Target, context)?;
        let domain = TokenDomain::new(scores.values().len());
        let scores = self.inner.refinement_scores_with_capture(
            &scores,
            history,
            &(),
            |scores, _domain| {
                #[cfg(feature = "tensor-capture")]
                if let Some(capture) = &self.capture {
                    capture.borrow_mut().observe_read_only(
                        &mut score_capture::HostScoreCapture(_domain),
                        scores,
                        eredu_core::speculative::SpeculativeCaptureRole::Target,
                        history.len() as u64,
                        if mask_token_id.is_some() {
                            PredictionScoreKind::RefinementPosition
                        } else {
                            PredictionScoreKind::ConstrainedFinalization
                        },
                    )?;
                }
                Ok(scores.clone())
            },
        )?;
        let candidate = std::sync::Arc::from([selected]);
        let finalized = finalize_refinement_position::<HostRefinement>(
            &ScoreMatrix::from_row(&scores),
            0,
            mask_token_id.map(|mask_token_id| RefinementSelection {
                tokens: &candidate,
                mask_token_id,
            }),
            domain,
            &eredu_core::TokenFilter::All,
            &(),
        )
        .map_err(BackendFailure::from_error)?;
        Ok(Some((scalar(&finalized.token)?, finalized.scores)))
    }

    fn sample<'a>(
        &self,
        distribution: &HostScores,
        temperature: f32,
        random: Option<&mut HostRandomState>,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<u32, BackendFailure>
    where
        Self: 'a,
    {
        self.inner
            .sample_processed(distribution, temperature, random, &())
    }

    fn probability_at<'a>(
        &self,
        distribution: &HostScores,
        token: u32,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<f32, BackendFailure>
    where
        Self: 'a,
    {
        HostSampling::token_probability(distribution, token, &())
    }

    fn sample_unit_interval<'a>(
        &self,
        random: Option<&mut HostRandomState>,
        _: Self::Context<'a>,
    ) -> Result<f32, BackendFailure>
    where
        Self: 'a,
    {
        Ok(random.ok_or_else(random_error)?.unit_interval_f32())
    }

    fn positive_probability_difference<'a>(
        &self,
        target: &HostScores,
        draft: &HostScores,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<Option<HostScores>, BackendFailure>
    where
        Self: 'a,
    {
        target.positive_difference(draft)
    }

    fn update_sampler_state<'a>(
        &mut self,
        distribution: &HostScores,
        token: u32,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<(), BackendFailure>
    where
        Self: 'a,
    {
        self.inner.commit_token(distribution, token, &())
    }
}

fn randomness_bytes(target: Option<&HostRandomState>, draft: Option<&u64>) -> u64 {
    (target.map_or(0, size_of_val) + draft.map_or(0, size_of_val)) as u64
}

#[cfg(feature = "tensor-capture")]
mod score_capture;
