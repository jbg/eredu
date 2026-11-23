//! Portable speculative sampling, score intervention, and capture accounting.

use std::{cell::RefCell, rc::Rc};

use eredu_core::{
    capture::{AdmittedCapturePlan, CaptureError, CapturePhase, CaptureTokenDomain},
    intervention::AdmittedInterventionPlan,
    refinement::PredictionScoreKind,
    speculative::{
        SpeculativeCaptureRole, SpeculativeControlError, SpeculativeInterventionPlan,
        SpeculativePredictionCapture,
    },
    SamplingPlacement, SpeculativeDraftRandomPosition, SpeculativeSampling,
    MODEL_LOGITS_OBSERVATION_PATH,
};

use crate::{capture::CaptureSession, SpeculativeSampler};
use eredu_core::sampling::{SamplingBackend, TokenDomain};

use eredu_core::speculative::sampling::SpeculativeSamplingPrimitives;

/// Queue and observer assignment derived by the portable sampler.
pub type SamplingContext<'a, M> =
    crate::capture::SpeculativeContextFor<'a, <M as SpeculativeSamplingPrimitives>::CaptureBinding>;

type Logits<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::Logits;
type Random<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::RandomState;
type Queue<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::Context;

fn sampling_queue<'a, M: SpeculativeSamplingPrimitives>(
    placement: SamplingPlacement,
    context: SamplingContext<'a, M>,
) -> Result<&'a Queue<M>, M::Error> {
    match placement {
        SamplingPlacement::Target => Ok(context.target()),
        SamplingPlacement::Draft => Ok(context.draft()),
        _ => Err(M::from_error(std::io::Error::other(
            "unsupported speculative sampling placement requires an explicit stream",
        ))),
    }
}

/// Portable sampler and observer shared by speculative branches of one request.
/// Observer clones share one nonrefundable ledger. Interventions are branch-local.
#[derive(Clone)]
pub struct InstalledSpeculativeSampling<S, M> {
    inner: S,
    capture: Option<Rc<RefCell<LogitCapture>>>,
    interventions: Vec<SpeculativeInterventionPlan>,
    mechanisms: std::marker::PhantomData<M>,
}

impl<S, M> InstalledSpeculativeSampling<S, M> {
    /// Installs ordinary sampling policy for an admitted executor.
    pub const fn new(inner: S) -> Self {
        Self {
            inner,
            capture: None,
            interventions: Vec::new(),
            mechanisms: std::marker::PhantomData,
        }
    }

    /// Returns the retained ordinary sampling policy.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Borrows the retained ordinary sampling policy.
    pub const fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S, M> SpeculativeSampling for InstalledSpeculativeSampling<S, M>
where
    M: SpeculativeSamplingPrimitives,
    S: SpeculativeSampler<M::Backend> + Clone,
{
    type Logits = M::Value;
    type Distribution = M::Value;
    type Seed = M::Seed;
    type RandomState = Random<M>;
    type DraftRandomness = M::DraftRandomness;
    type RandomnessRoot = M::RandomnessRoot;
    type Context<'a>
        = SamplingContext<'a, M>
    where
        Self: 'a;
    type Error = M::Error;
    fn control_requires_positive_temperature(&self) -> Option<bool> {
        self.inner.control_requires_positive_temperature()
    }
    fn control_seed<'a>(
        seed: u64,
        context: Self::Context<'a>,
    ) -> Result<Self::Seed, eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'a,
    {
        let _ = context;
        M::seed(seed).map_err(SpeculativeControlError::backend)
    }
    fn control_force_next(
        &mut self,
        token: u32,
        vocabulary: usize,
        position: usize,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        self.inner.control_force_next(
            token,
            eredu_core::sampling::TokenDomain::new(vocabulary),
            position,
        )
    }
    fn control_clear_forced(&mut self) -> bool {
        self.inner.control_clear_forced()
    }
    fn control_pending_forced(&self) -> Option<u32> {
        self.inner.control_pending_forced()
    }
    fn control_snapshot_bytes(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
    ) -> Option<u64> {
        let mut bytes = self
            .inner
            .control_snapshot_bytes()?
            .checked_add(M::snapshot_randomness(target, draft)?)?;
        for plan in &self.interventions {
            bytes = bytes.checked_add(
                crate::execution_control::admitted_intervention_storage_bytes(&plan.plan)?,
            )?;
        }
        bytes.checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn continuation_storage_bytes(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
        additional_tokens: u64,
    ) -> Option<u64> {
        // Current allowance includes RNG and intervention ownership; policy's
        // total continuation bound supplies all future history/grammar growth.
        self.control_snapshot_bytes(target, draft)?
            .checked_add(self.inner.continuation_storage_bytes(additional_tokens)?)?
            .checked_add(M::continuation_randomness_bytes())
    }

    fn enable_control_capture(
        &mut self,
        plan: AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        crate::speculative::autoregressive::validate_control_capture(&plan)?;
        self.capture = Some(std::rc::Rc::new(std::cell::RefCell::new(
            LogitCapture::new(plan),
        )));
        Ok(())
    }

    fn take_control_captures(
        &mut self,
    ) -> Vec<eredu_core::speculative::SpeculativePredictionCapture> {
        self.capture
            .as_ref()
            .map_or_else(Vec::new, |capture| capture.borrow_mut().drain())
    }

    fn control_intervene(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        use eredu_core::speculative::SpeculativeControlError;
        if plans.len() > 2 || (plans.len() == 2 && plans[0].role == plans[1].role) {
            return Err(SpeculativeControlError::Invalid(
                "supply at most one intervention plan for each model role",
            ));
        }
        if !plans.is_empty() {
            let capture = self
                .capture
                .as_ref()
                .ok_or(SpeculativeControlError::Invalid(
                    "tensor interventions require an explicit capture/evidence budget",
                ))?
                .borrow();
            for plan in &plans {
                if plan
                    .plan
                    .points()
                    .iter()
                    .any(|point| point.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                {
                    return Err(SpeculativeControlError::Unsupported(
                        "speculative tensor edits currently support model.logits only",
                    ));
                }
                crate::intervention::preflight(
                    capture.session.plan(),
                    &plan.plan,
                    M::intervention_estimator().as_ref(),
                )?;
            }
        }
        self.interventions = plans;
        Ok(())
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeSampler::<M::Backend>::supports_exact_optimistic_promotion(&self.inner)
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        SpeculativeSampler::<M::Backend>::grammar_is_complete(&mut self.inner)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        SpeculativeSampler::<M::Backend>::prefix_is_complete(&self.inner, history)
    }

    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a,
    {
        {
            let _ = context;
            M::randomness_root(seed)
        }
    }

    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        M::target_randomness_from_root(root, context.target())
    }

    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a,
    {
        let key = M::draft_randomness_from_root(root, context.target())?;
        if context.crosses_devices() {
            return M::copy_draft_key(&key, context.draft());
        }
        if context.is_split() {
            let _handoff = context.wait_for_target_outputs([M::draft_key_root(&key)])?;
        }
        Ok(key)
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        M::draft_randomness_at(root, position, context.draft())
    }

    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_queue::<M>(placement, context)?;
        let logits = M::wrap(logits);
        let Some(capture) = &self.capture else {
            return self
                .inner
                .process_logits(&logits, temperature, history, stream)
                .map(M::unwrap);
        };
        let role = match placement {
            SamplingPlacement::Target => SpeculativeCaptureRole::Target,
            SamplingPlacement::Draft => SpeculativeCaptureRole::Draft,
            _ => {
                return Err(M::from_error(CaptureError::Unsupported(
                    "unsupported speculative capture role".into(),
                )))
            }
        };
        let observation = PredictionObservation {
            role,
            position: history.len() as u64,
            kind: PredictionScoreKind::CausalConditional,
            intervention: self
                .interventions
                .iter()
                .find(|plan| plan.role == role)
                .map(|plan| &plan.plan),
        };
        self.inner
            .process_logits_with_capture(&logits, temperature, history, stream, |logits, domain| {
                capture
                    .borrow_mut()
                    .observe::<M>(logits, stream, domain, observation)
            })
            .map(M::unwrap)
    }

    fn finalize_refinement_position<'a>(
        &mut self,
        logits: &Self::Logits,
        selected: u32,
        mask_token_id: Option<u32>,
        history: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Option<(u32, Self::Logits)>, Self::Error>
    where
        Self: 'a,
    {
        use crate::generation::refinement::finalization::{
            finalize_refinement_position, RefinementSelection,
        };
        let stream = sampling_queue::<M>(SamplingPlacement::Target, context)?;
        let scores = self.inner.refinement_scores_with_capture(
            &M::wrap(logits),
            history,
            stream,
            |scores, domain| {
                let Some(capture) = &self.capture else {
                    return Ok(scores.clone());
                };
                let role = SpeculativeCaptureRole::Target;
                let observation = PredictionObservation {
                    role,
                    position: history.len() as u64,
                    kind: if mask_token_id.is_some() {
                        PredictionScoreKind::RefinementPosition
                    } else {
                        PredictionScoreKind::ConstrainedFinalization
                    },
                    intervention: self
                        .interventions
                        .iter()
                        .find(|plan| plan.role == role)
                        .map(|plan| &plan.plan),
                };
                capture
                    .borrow_mut()
                    .observe::<M>(scores, stream, domain, observation)
            },
        )?;
        let candidate = M::candidate(selected);
        let finalized = finalize_refinement_position::<M::Backend>(
            &scores,
            0,
            mask_token_id.map(|mask_token_id| RefinementSelection {
                tokens: &candidate,
                mask_token_id,
            }),
            TokenDomain::new(M::vocabulary(logits)),
            &eredu_core::TokenFilter::All,
            stream,
        )
        .map_err(M::from_error)?;
        let token = M::Backend::token_id(&finalized.token, stream)?;
        Ok(Some((token, M::unwrap(finalized.scores))))
    }
    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_queue::<M>(placement, context)?;
        let token = SpeculativeSampler::<M::Backend>::sample_processed(
            &self.inner,
            &M::wrap(distribution),
            temperature,
            randomness,
            stream,
        )?;
        M::sample_token(token, stream)
    }

    fn probability_at<'a>(
        &self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        M::probability_at(
            distribution,
            token,
            sampling_queue::<M>(placement, context)?,
        )
    }

    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        M::sample_unit_interval(randomness, context.target())
    }

    fn positive_probability_difference<'a>(
        &self,
        left: &Self::Distribution,
        right: &Self::Distribution,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::Distribution>, Self::Error>
    where
        Self: 'a,
    {
        M::positive_probability_difference(left, right, sampling_queue::<M>(placement, context)?)
    }

    fn update_sampler_state<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        SpeculativeSampler::<M::Backend>::commit_token(
            &mut self.inner,
            &M::wrap(distribution),
            token,
            sampling_queue::<M>(placement, context)?,
        )
    }

    fn prepare_verification<'a>(
        &self,
        distributions: &mut [&mut Self::Distribution],
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 {
            return Ok(());
        }
        if context.crosses_devices() {
            M::copy_distributions(distributions, context.target())?;
        } else if context.is_split() {
            let _handoff = context.wait_for_draft_outputs(
                distributions
                    .iter()
                    .map(|value| M::distribution_root(value)),
            )?;
        }
        Ok(())
    }
}

pub(super) struct LogitCapture {
    session: CaptureSession,
    records: Vec<SpeculativePredictionCapture>,
}

struct PredictionObservation<'a> {
    role: SpeculativeCaptureRole,
    position: u64,
    kind: PredictionScoreKind,
    intervention: Option<&'a AdmittedInterventionPlan>,
}

impl LogitCapture {
    pub(super) fn new(plan: AdmittedCapturePlan) -> Self {
        Self {
            session: CaptureSession::new(plan),
            records: Vec::new(),
        }
    }

    pub(super) fn drain(&mut self) -> Vec<SpeculativePredictionCapture> {
        std::mem::take(&mut self.records)
    }

    fn begin(&mut self, position: u64) -> Result<(), CaptureError> {
        let phase = if position == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        self.session.begin_step(phase, position)
    }

    fn record(&mut self, role: SpeculativeCaptureRole, position: u64, kind: PredictionScoreKind) {
        if let Some(capture) = self.session.take_step() {
            self.records.push(SpeculativePredictionCapture {
                role,
                position,
                score_kind: kind,
                capture,
            });
        }
    }

    #[cfg(feature = "tensor-capture")]
    pub(super) fn observe_read_only<
        B: eredu_core::capture::CaptureBackend<Error = eredu_core::BackendFailure>,
    >(
        &mut self,
        backend: &mut B,
        scores: &B::Tensor,
        role: SpeculativeCaptureRole,
        position: u64,
        kind: PredictionScoreKind,
    ) -> Result<(), eredu_core::BackendFailure> {
        self.begin(position)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let result = self
            .session
            .observe(backend, MODEL_LOGITS_OBSERVATION_PATH, scores);
        self.record(role, position, kind);
        result.map_err(crate::capture::CaptureExecutionError::into_operation_error)
    }

    fn observe<M: SpeculativeSamplingPrimitives>(
        &mut self,
        logits: &Logits<M>,
        queue: &Queue<M>,
        domain: Option<CaptureTokenDomain<'_>>,
        observation: PredictionObservation<'_>,
    ) -> Result<Logits<M>, M::Error> {
        self.session
            .select_prediction_interventions(
                observation.intervention.cloned(),
                M::intervention_estimator(),
            )
            .map_err(M::from_error)?;
        self.begin(observation.position).map_err(M::from_error)?;
        let tensor = M::capture_scores(logits, queue)?;
        let mut backend = M::score_capture(queue, domain);
        self.session
            .observe(&mut backend, MODEL_LOGITS_OBSERVATION_PATH, &tensor)
            .map_err(M::from_error)?;
        let effective = self
            .session
            .intervene(&mut backend, MODEL_LOGITS_OBSERVATION_PATH, &tensor)
            .map_err(M::from_error)?;
        self.session.finish_interventions().map_err(M::from_error)?;
        self.record(observation.role, observation.position, observation.kind);
        effective.map_or_else(
            || Ok(logits.clone()),
            |value| M::restore_scores(value, logits, queue),
        )
    }
}
