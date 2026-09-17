use super::*;
use crate::backend::managed_memory::{NativeMemoryOwner, NativeMemoryRetention};

mod host_owner;
mod capture;
mod interventions;
use host_owner::Policy;
mod ownership;
use ownership::{KeyValue, derivation_memory, with_memory_recovery};
pub use ownership::{MlxSpeculativeRandomState, MlxSpeculativeSeed};

pub(crate) mod numerical;

pub(crate) mod logits;
use logits::LogitsSource;

mod distribution;
pub use distribution::MlxSpeculativeDistribution;

#[cfg(test)]
mod distribution_tests;
#[cfg(test)]
mod ownership_tests;

/// MLX implementation of opaque speculative sampling operations.
pub struct MlxSpeculativeSampling<S, E = Exception, L = Array> {
    inner: Policy<S>,
    capture: Option<std::rc::Rc<std::cell::RefCell<LogitCapture>>>,
    original_capture: Option<capture::OriginalCapture>,
    interventions: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    // Original source copies are immutable C owners. Fixed role slots carry no
    // caller Vec and survive policy snapshots as aliases of the actual source.
    original_interventions: Option<interventions::OriginalInterventions>,
    // Policy and capture clones may share interior native state. Share additions
    // to their authority as well, and drop payloads before the final owner.
    memory_retention: Option<std::rc::Rc<std::cell::RefCell<NativeMemoryRetention>>>,
    error: std::marker::PhantomData<fn() -> (E, L)>,
}

impl<S> MlxSpeculativeSampling<S> {
    /// Wraps one runtime sampling policy for MLX execution.
    pub fn new(inner: S) -> Self {
        Self {
            inner: Policy::Ordinary(inner),
            capture: None,
            original_capture: None,
            interventions: Vec::new(),
            original_interventions: None,
            memory_retention: Some(std::rc::Rc::default()),
            error: std::marker::PhantomData,
        }
    }

    /// Retargets only the shared driver's error type. Numerical workers,
    /// policy, source owners and completion behavior are moved unchanged.
    pub(crate) fn with_error<E>(self) -> MlxSpeculativeSampling<S, E> {
        MlxSpeculativeSampling {
            inner: self.inner,
            capture: self.capture,
            original_capture: self.original_capture,
            interventions: self.interventions,
            original_interventions: self.original_interventions,
            memory_retention: self.memory_retention,
            error: std::marker::PhantomData,
        }
    }
}
impl<S: Clone, E, L> Clone for MlxSpeculativeSampling<S, E, L> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            capture: self.capture.clone(),
            original_capture: self.original_capture.clone(),
            interventions: self.interventions.clone(),
            original_interventions: self.original_interventions.clone(),
            memory_retention: self.memory_retention.clone(),
            error: std::marker::PhantomData,
        }
    }
}
impl<S, E, L> MlxSpeculativeSampling<S, E, L> {
    pub(crate) fn with_logits<N>(self) -> MlxSpeculativeSampling<S, E, N> {
        MlxSpeculativeSampling {
            inner: self.inner,
            capture: self.capture,
            original_capture: self.original_capture,
            interventions: self.interventions,
            original_interventions: self.original_interventions,
            memory_retention: self.memory_retention,
            error: std::marker::PhantomData,
        }
    }

    /// Retains the preparation authority even when greedy decoding needs no RNG.
    pub(crate) fn with_memory_owner(self, owner: &NativeMemoryOwner) -> Self {
        if let Some(memory) = &self.memory_retention {
            memory.borrow_mut().retain(owner);
        }
        self
    }

    /// Admit all source domains before any callback or native operation.
    fn operation_memory<'a>(
        &self,
        sources: impl IntoIterator<Item = &'a NativeMemoryRetention>,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<NativeMemoryRetention, Exception> {
        let mut retained = self
            .memory_retention
            .as_ref()
            .expect("ordinary sampler memory")
            .borrow()
            .clone();
        for source in sources {
            retained.extend_from(source);
        }
        derivation_memory(&retained, context)
    }

    /// Even an immutable policy callback may retain native values internally.
    /// Share its authority with earlier sampler clones without holding a borrow
    /// across user callbacks.
    fn retain_callback_memory(&self, memory: &NativeMemoryRetention) {
        self.memory_retention
            .as_ref()
            .expect("ordinary sampler memory")
            .borrow_mut()
            .extend_from(memory);
    }

    /// Wraps caller-owned storage without allocating or evaluating native data.
    pub(crate) fn seed_from_array(value: Array) -> MlxSpeculativeSeed {
        MlxSpeculativeSeed::from_array(value)
    }

    /// Returns the sampling policy after a test generation completes.
    #[cfg(test)]
    pub fn into_inner(self) -> S {
        self.inner.into_ordinary()
    }

    #[cfg(test)]
    pub fn inner(&self) -> &S {
        self.inner.source()
    }
}

impl<S, E, L> SpeculativeSampling for MlxSpeculativeSampling<S, E, L>
where
    S: SpeculativeSampler<MlxSamplingBackend> + Clone,
    E: std::error::Error + Send + Sync + From<Exception> + From<L::OriginalError> + 'static,
    L: LogitsSource,
{
    type Logits = L;
    type Distribution = MlxSpeculativeDistribution;
    type Seed = MlxSpeculativeSeed;
    type RandomState = MlxSpeculativeRandomState;
    type DraftRandomness = MlxSpeculativeSeed;
    type RandomnessRoot = MlxSpeculativeRandomState;
    type Context<'a>
        = SpeculativeExecutionStreams<'a>
    where
        Self: 'a;
    type Error = E;

    fn control_requires_positive_temperature(&self) -> Option<bool> {
        self.inner.source().control_requires_positive_temperature()
    }
    fn control_seed<'a>(
        seed: u64,
        context: Self::Context<'a>,
    ) -> Result<Self::Seed, eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'a,
    {
        if let Some((sources, _)) = context.original_numerical() {
            let failure = |cause| {
                eredu_core::speculative::SpeculativeControlError::backend_with_retained(
                    sources.retain_error(cause),
                    Error::take_retained_backend_failure,
                )
            };
            reserve_original_key::<Self::Seed, eredu_core::speculative::SpeculativeControlError, L>(
                None, SamplingPlacement::Target, context,
            ).map_err(failure)?;
            return numerical::create_key(seed, context)
                .map(|value| MlxSpeculativeSeed {
                    value: KeyValue::Original(value),
                    memory_retention: NativeMemoryRetention::default(),
                })
                .map_err(failure);
        }
        let owner = NativeMemoryOwner::acquire(&context.memory_pool())
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        let memory = NativeMemoryRetention::from_owner(&owner);
        with_memory_recovery(memory.clone(), || {
            Ok(MlxSpeculativeSeed {
                value: KeyValue::Ordinary(safemlx::random::key(seed)?),
                memory_retention: memory,
            })
        })
        .map_err(eredu_core::speculative::SpeculativeControlError::backend)
    }
    fn control_force_next(
        &mut self,
        token: u32,
        vocabulary: usize,
        position: usize,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        if self.inner.is_original() {
            return self.force_original(
                token,
                eredu_runtime::TokenDomain::new(vocabulary),
                position,
            );
        }
        self.inner
            .ordinary_mut()
            .expect("ordinary sampler")
            .control_force_next(token, eredu_runtime::TokenDomain::new(vocabulary), position)
    }
    fn control_clear_forced(&mut self) -> bool {
        self.inner
            .ordinary_mut()
            .is_some_and(|source| source.control_clear_forced())
    }
    fn try_control_clear_forced(
        &mut self,
    ) -> Result<bool, eredu_core::speculative::SpeculativeControlError> {
        if self.inner.is_original() {
            self.clear_original_forced()
        } else {
            Ok(self.control_clear_forced())
        }
    }
    fn control_pending_forced(&self) -> Option<u32> {
        self.inner.source().control_pending_forced()
    }
    fn control_snapshot_bytes(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
    ) -> Option<u64> {
        if self.inner.is_original() {
            return self.original_snapshot_bytes(target, draft);
        }
        let mut bytes = self.inner.source().control_snapshot_bytes()?;
        let target_array = match target {
            Some(target) => Some(target.value.ordinary()?.as_array()),
            None => None,
        };
        let draft_array = match draft {
            Some(draft) => Some(draft.value.ordinary()?),
            None => None,
        };
        for array in target_array.into_iter().chain(draft_array) {
            bytes = bytes
                .checked_add(array.nbytes() as u64)?
                .checked_add(4096)?;
        }
        bytes = bytes
            .checked_add(std::mem::size_of::<std::cell::RefCell<NativeMemoryRetention>>() as u64)?
            .checked_add(
                self.memory_retention
                    .as_ref()?
                    .borrow()
                    .logical_metadata_bytes()?,
            )?;
        if let Some(target) = target {
            bytes = bytes
                .checked_add(std::mem::size_of::<MlxSpeculativeRandomState>() as u64)?
                .checked_add(target.memory_retention.logical_metadata_bytes()?)?;
        }
        if let Some(draft) = draft {
            bytes = bytes
                .checked_add(std::mem::size_of::<MlxSpeculativeSeed>() as u64)?
                .checked_add(draft.memory_retention.logical_metadata_bytes()?)?;
        }
        for plan in &self.interventions {
            bytes = bytes.checked_add(
                eredu_runtime::execution_control::admitted_intervention_storage_bytes(&plan.plan)?,
            )?;
        }
        bytes.checked_add(std::mem::size_of::<Self>() as u64)
    }

    fn control_snapshot_metadata_bytes(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
    ) -> Option<usize> {
        self.original_snapshot_metadata(target, draft)
    }
    fn copy_control_snapshot(
        &self,
        target: Option<&Self::RandomState>,
        draft: Option<&Self::DraftRandomness>,
        host: eredu_core::HostPreparationAuthority,
    ) -> Result<
        (
            Self,
            Option<Self::RandomState>,
            Option<Self::DraftRandomness>,
        ),
        eredu_core::speculative::SpeculativeControlError,
    > {
        self.copy_snapshot(target, draft, host)
    }

    fn enable_control_capture(
        &mut self,
        plan: eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        if let Some(refusal) = self.inner.capture_refusal() {
            return Err(refusal);
        }
        validate_control_capture(&plan)?;
        self.capture = Some(std::rc::Rc::new(std::cell::RefCell::new(LogitCapture {
            session: eredu_runtime::capture::CaptureSession::new(plan),
            records: Vec::new(),
        })));
        Ok(())
    }

    fn enable_control_capture_prepared<'a>(
        &mut self,
        plan: eredu_core::capture::AdmittedCapturePlan,
        context: Self::Context<'a>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'a,
    {
        use eredu_core::speculative::SpeculativeControlError;
        if context.original_numerical().is_none() {
            return self.enable_control_capture(plan).map_err(Into::into);
        }
        if !self.inner.is_original() || self.capture.is_some() || self.original_capture.is_some() {
            return Err(SpeculativeControlError::Invalid("capture observer is already installed or belongs to another source"));
        }
        self.original_capture = Some(capture::OriginalCapture::prepare(&plan, context)
            .map_err(|cause| SpeculativeControlError::backend_with_retained(cause, Error::take_retained_backend_failure))?);
        Ok(())
    }

    fn take_control_captures(
        &mut self,
    ) -> eredu_core::SpeculativeBuffer<eredu_core::speculative::SpeculativePredictionCapture> {
        if let Some(capture) = &self.original_capture { return capture.take(); }
        // This collector is ordinary-only. Moving its Vec into the ordinary
        // buffer variant neither attaches H nor qualifies original capture.
        self.capture
            .as_ref()
            .map_or_else(Default::default, |capture| {
                std::mem::take(&mut capture.borrow_mut().records).into()
            })
    }

    fn validate_control_interventions_prepared<'a>(
        &self,
        plans: &[eredu_core::speculative::SpeculativeInterventionPlan],
        context: Self::Context<'a>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError>
    where Self: 'a,
    {
        if !self.inner.is_original() || context.original_numerical().is_none() {
            return if plans.is_empty() {Ok(())} else {Err(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "loaded execution has no speculative intervention discovery"))};
        }
        interventions::validate(plans,context)
    }
    fn control_intervene_prepared<'a>(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
        context: Self::Context<'a>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError>
    where Self: 'a,
    {
        if context.original_numerical().is_none() {return self.control_intervene(plans);}
        if !self.inner.is_original() || self.capture.is_some() || !self.interventions.is_empty() {
            return Err(eredu_core::speculative::SpeculativeControlError::Invalid(
                "original interventions require the original policy and capture source"));
        }
        let next=interventions::prepare(&plans,self.original_capture.as_ref(),context)?;
        // All validation, shared preflight and fresh source copies completed.
        // Replacing fixed slots is the single commit; failures preserve old edits.
        self.original_interventions=next;
        Ok(())
    }
    fn control_intervene(
        &mut self,
        plans: Vec<eredu_core::speculative::SpeculativeInterventionPlan>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        use eredu_core::speculative::SpeculativeControlError;
        if self.inner.is_original() {
            // Clearing an already empty original intervention set is a no-op;
            // do not adopt even an empty caller Vec's spare allocation.
            return if plans.is_empty() && self.original_interventions.is_none() {
                Ok(())
            } else {
                Err(SpeculativeControlError::Unsupported(
                    "original sampler intervention requires its admitted capture producer",
                ))
            };
        }
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
                eredu_runtime::intervention::preflight(
                    capture.session.plan(),
                    &plan.plan,
                    &crate::composition::mlx::session::intervention::NativeInterventionEstimator,
                )?;
            }
        }
        self.interventions = plans;
        Ok(())
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(
            self.inner.source(),
        )
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        if self.inner.is_original() && self.inner.source().prepared_grammar_controller().is_some() {
            return self.current_grammar_original().map_err(|cause| E::from(L::original_error(cause)));
        }
        if let Some(complete) = self.inner.grammar_complete() {
            return Ok(complete);
        }
        SpeculativeSampler::<MlxSamplingBackend>::grammar_is_complete(
            self.inner.ordinary_mut().expect("ordinary sampler"),
        )
        .map_err(E::from)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        if self.inner.is_original() {
            return self
                .original_prefix_is_complete(history)
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        SpeculativeSampler::<MlxSamplingBackend>::prefix_is_complete(self.inner.source(), history)
            .map_err(E::from)
    }

    fn randomness_root<'a>(
        seed: Option<Self::Seed>,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomnessRoot, Self::Error>
    where
        Self: 'a,
    {
        let seed = match seed {
            Some(seed) => seed,
            None if context.original_numerical().is_some() => {
                return Err(E::from(L::original_error(Error::InvalidOperation(
                    "random operations require an explicit PRNG key",
                ))));
            }
            None => {
                return Err(E::from(Exception::custom(
                    "random operations require an explicit PRNG key",
                )));
            }
        };
        if let Some(key) = seed.value.original() {
            reserve_original_key::<Self::RandomnessRoot, E, L>(
                Some(key),
                SamplingPlacement::Target,
                context,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?;
            let KeyValue::Original(key) = seed.value else {
                unreachable!()
            };
            return Ok(MlxSpeculativeRandomState {
                value: KeyValue::Original(key),
                memory_retention: NativeMemoryRetention::default(),
            });
        }
        reject_original_context(context).map_err(|cause| E::from(L::original_error(cause)))?;
        // Raw caller keys carry no new-allocation permission. Bind context
        // ownership before the first native split, including standalone calls.
        let memory = derivation_memory(&seed.memory_retention, context)?;
        Ok(MlxSpeculativeRandomState {
            value: KeyValue::Ordinary(RandomState::from_key(seed.value.into_ordinary()?)),
            memory_retention: memory,
        })
    }

    fn target_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        if let Some(key) = root.value.original_mut() {
            reserve_original_key::<Self::RandomState, E, L>(
                Some(key),
                SamplingPlacement::Target,
                context,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?;
            return numerical::next_key(key, context)
                .map(|key| MlxSpeculativeRandomState {
                    value: KeyValue::Original(key),
                    memory_retention: NativeMemoryRetention::default(),
                })
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        reject_original_context(context).map_err(|cause| E::from(L::original_error(cause)))?;
        root.memory_retention = derivation_memory(&root.memory_retention, context)?;
        let memory = root.memory_retention.clone();
        with_memory_recovery(memory.clone(), || {
            Ok(MlxSpeculativeRandomState {
                value: KeyValue::Ordinary(RandomState::from_key(
                    root.value
                        .ordinary_mut()
                        .expect("ordinary key")
                        .next_key(context.target())?,
                )),
                memory_retention: memory,
            })
        })
        .map_err(E::from)
    }

    fn draft_randomness_from_root<'a>(
        root: &mut Self::RandomnessRoot,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftRandomness, Self::Error>
    where
        Self: 'a,
    {
        if let Some(key) = root.value.original_mut() {
            // Root advancement remains on Target. The returned key is already
            // complete; Draft key_at authenticates its actual source placement
            // before running the same split/select worker on Draft.
            reserve_original_key::<Self::DraftRandomness, E, L>(
                Some(key),
                SamplingPlacement::Draft,
                context,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?;
            reserve_original_key::<(), E, L>(Some(key), SamplingPlacement::Target, context)
                .map_err(|cause| E::from(L::original_error(cause)))?;
            return numerical::next_key(key, context)
                .and_then(|key|if context.crosses_devices(){
                    numerical::copy_key_to(&key,SamplingPlacement::Target,SamplingPlacement::Draft,context)
                }else{Ok(key)})
                .map(|key| MlxSpeculativeSeed {
                    value: KeyValue::Original(key),
                    memory_retention: NativeMemoryRetention::default(),
                })
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        reject_original_context(context).map_err(|cause| E::from(L::original_error(cause)))?;
        root.memory_retention = derivation_memory(&root.memory_retention, context)?;
        let memory = root.memory_retention.clone();
        with_memory_recovery(memory.clone(), || {
            let draft_key = root
                .value
                .ordinary_mut()
                .expect("ordinary key")
                .next_key(context.target())?;
            let draft_key = if context.is_split() {
                if context.crosses_devices() {
                    async_eval_with_event([&draft_key])?.synchronize()?;
                    let copied = draft_key.copy(context.draft())?;
                    async_eval_with_event([&copied])?.synchronize()?;
                    copied
                } else {
                    let _completion = context.wait_for_target_outputs([&draft_key])?;
                    draft_key
                }
            } else {
                draft_key
            };
            Ok(MlxSpeculativeSeed {
                value: KeyValue::Ordinary(draft_key),
                memory_retention: memory,
            })
        })
        .map_err(E::from)
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        if let Some(key) = root.value.original() {
            reserve_original_key::<Self::RandomState, E, L>(
                Some(key),
                SamplingPlacement::Draft,
                context,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?;
            return numerical::key_at(key, position, context)
                .map(|key| MlxSpeculativeRandomState {
                    value: KeyValue::Original(key),
                    memory_retention: NativeMemoryRetention::default(),
                })
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        reject_original_context(context).map_err(|cause| E::from(L::original_error(cause)))?;
        let memory = derivation_memory(&root.memory_retention, context)?;
        with_memory_recovery(memory.clone(), || {
            Ok(MlxSpeculativeRandomState {
                value: KeyValue::Ordinary(RandomState::from_key(
                    crate::backend::random::split_key_at(
                        root.value.ordinary().expect("ordinary key"),
                        position.get(),
                        context.draft(),
                    )?,
                )),
                memory_retention: memory,
            })
        })
        .map_err(E::from)
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
        if context.original_numerical().is_some() {
            let source = logits.original().ok_or_else(|| {
                E::from(L::original_error(Error::InvalidOperation(
                    "original sampling requires a completed source",
                )))
            })?;
            reserve_original_consumer::<Self::Distribution, E, L>(
                source, None, placement, context, 0,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?;
            if self.capture.is_some() || !self.interventions.is_empty() {
                return Err(E::from(L::original_error(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))));
            }
            let policy = self.inner.original_source()
                .map_err(|cause| E::from(L::original_error(cause)))?;
            let processed = match &self.original_capture {
                Some(capture) => capture.process_with_interventions(policy, source, temperature, history, placement, context,
                    interventions::for_placement(&self.original_interventions,placement)),
                None => numerical::process_policy_at(policy, source, temperature, history, placement, context),
            };
            return processed.and_then(|value| {
                MlxSpeculativeDistribution::original(
                    value,
                    context
                        .original_numerical()
                        .expect("validated original context")
                        .0,
                )
            })
            .map_err(|cause| E::from(L::original_error(cause)));
        }
        if self.original_interventions.is_some() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original intervention source requires its admitted request context"))));
        }
        let logits = logits.ordinary().ok_or_else(|| {
            E::from(L::original_error(Error::InvalidOperation(
                "original logits require their admitted source context",
            )))
        })?;
        let role = match placement {
            SamplingPlacement::Target => eredu_core::speculative::SpeculativeCaptureRole::Target,
            SamplingPlacement::Draft => eredu_core::speculative::SpeculativeCaptureRole::Draft,
            _ => return Err(Exception::custom("unsupported speculative sampling role").into()),
        };
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original policy requires its admitted numerical source",
            ))));
        }
        let stream = sampling_stream(placement, context)?;
        let memory = self.operation_memory([], context)?;
        self.retain_callback_memory(&memory);
        with_memory_recovery(memory.clone(), || {
            let logits = MlxTensor::from_array(logits.clone());
            let processed = if let Some(capture) = &self.capture {
                SpeculativeSampler::<MlxSamplingBackend>::process_logits_with_capture(
                    self.inner.ordinary_mut().expect("ordinary sampler"),
                    &logits,
                    temperature,
                    history,
                    stream,
                    |logits, domain| {
                        let effective = capture.borrow_mut().observe(
                            logits.as_array(),
                            history.len() as u64,
                            placement,
                            self.interventions
                                .iter()
                                .find(|p| p.role == role)
                                .map(|p| &p.plan),
                            stream,
                            domain,
                        )?;
                        Ok(effective
                            .map(MlxTensor::from_array)
                            .unwrap_or_else(|| logits.clone()))
                    },
                )?
            } else {
                SpeculativeSampler::<MlxSamplingBackend>::process_logits(
                    self.inner.ordinary_mut().expect("ordinary sampler"),
                    &logits,
                    temperature,
                    history,
                    stream,
                )?
            };
            Ok(MlxSpeculativeDistribution::new(
                processed.into_array(),
                memory,
            ))
        })
        .map_err(E::from)
    }

    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        mut randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        if let Some(value) = distribution.original_value() {
            reserve_original_consumer::<u32, E, L>(value, None, placement, context, 0)
                .map_err(|cause| E::from(L::original_error(cause)))?;
            if temperature > 0.0 {
                let key = randomness
                    .as_deref_mut()
                    .and_then(|state| state.value.original_mut())
                    .ok_or_else(|| {
                        E::from(L::original_error(Error::PrefillControl(
                            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                        )))
                    })?;
                reserve_original_key::<u32, E, L>(Some(key), placement, context)
                    .map_err(|cause| E::from(L::original_error(cause)))?;
                return numerical::sample_stochastic_at(
                    self.inner
                        .original_source()
                        .map_err(|cause| E::from(L::original_error(cause)))?,
                    value,
                    temperature,
                    key,
                    placement,
                    context,
                )
                .map_err(|cause| E::from(L::original_error(cause)));
            }
            return numerical::sample_greedy_at(
                self.inner
                    .original_source()
                    .map_err(|cause| E::from(L::original_error(cause)))?,
                value,
                temperature,
                placement,
                context,
            )
            .map_err(|cause| E::from(L::original_error(cause)));
        }
        if context.original_numerical().is_some() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original sampling requires a completed distribution",
            ))));
        }
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original policy requires its admitted numerical source",
            ))));
        }
        if randomness
            .as_ref()
            .is_some_and(|state| state.value.original().is_some())
        {
            return Err(E::from(L::original_error(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))));
        }
        let stream = sampling_stream(placement, context)?;
        let memory = self.operation_memory(
            std::iter::once(distribution.memory()).chain(
                randomness
                    .as_ref()
                    .map(|randomness| &randomness.memory_retention),
            ),
            context,
        )?;
        self.retain_callback_memory(&memory);
        if let Some(randomness) = &mut randomness {
            randomness.memory_retention.extend_from(&memory);
        }
        with_memory_recovery(memory, || {
            let token = SpeculativeSampler::<MlxSamplingBackend>::sample_processed(
                self.inner.source(),
                &MlxTensor::from_array(distribution.value().clone()),
                temperature,
                randomness.map(|randomness| randomness.value.ordinary_mut().expect("ordinary key")),
                stream,
            )?;
            eval([token.as_array()])?;
            Ok(token.into_array().item::<u32>(stream))
        })
        .map_err(E::from)
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
        if let Some(value) = distribution.original_value() {
            reserve_original_consumer::<f32, E, L>(value, None, placement, context, 2)
                .map_err(|cause| E::from(L::original_error(cause)))?;
            let probabilities = match numerical_phase(
                context, placement,
                eredu_runtime::speculative::numerical::SpeculativeNumericalKind::Normalize,
                value,
                None,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?
            {
                numerical::NumericalOutput::Distribution(value) => value,
                _ => unreachable!("normalization output"),
            };
            return match numerical_phase(
                context, placement,
                eredu_runtime::speculative::numerical::SpeculativeNumericalKind::ProbabilityAt {
                    token,
                },
                &probabilities,
                None,
            )
            .map_err(|cause| E::from(L::original_error(cause)))?
            {
                numerical::NumericalOutput::Probability(value) => Ok(value),
                _ => unreachable!("probability output"),
            };
        }
        if context.original_numerical().is_some() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original sampling requires a completed distribution",
            ))));
        }
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original policy requires its admitted numerical source",
            ))));
        }
        let stream = sampling_stream(placement, context)?;
        let memory = self.operation_memory([distribution.memory()], context)?;
        with_memory_recovery(memory, || {
            let probabilities = probabilities(distribution.value(), stream)?;
            array_probability_at(&probabilities, token, stream)
        })
        .map_err(E::from)
    }

    fn sample_unit_interval<'a>(
        &self,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<f32, Self::Error>
    where
        Self: 'a,
    {
        let randomness = match randomness {
            Some(randomness) => randomness,
            None if self.inner.is_original() || context.original_numerical().is_some() => {
                return Err(E::from(L::original_error(Error::InvalidOperation(
                    "stochastic speculative decoding requires a PRNG key",
                ))));
            }
            None => {
                return Err(E::from(Exception::custom(
                    "stochastic speculative decoding requires a PRNG key",
                )));
            }
        };
        if let Some(key) = randomness.value.original_mut() {
            self.inner
                .original_source()
                .map_err(|cause| E::from(L::original_error(cause)))?;
            reserve_original_key::<f32, E, L>(Some(key), SamplingPlacement::Target, context)
                .map_err(|cause| E::from(L::original_error(cause)))?;
            return numerical::sample_unit_interval(key, context)
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))));
        }
        reject_original_context(context).map_err(|cause| E::from(L::original_error(cause)))?;
        let memory = self.operation_memory([&randomness.memory_retention], context)?;
        randomness.memory_retention.extend_from(&memory);
        with_memory_recovery(memory, || {
            uniform(
                Some(randomness.value.ordinary_mut().expect("ordinary key")),
                context.target(),
            )
        })
        .map_err(E::from)
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
        match (left.original_value(), right.original_value()) {
            (Some(left), Some(right)) => {
                reserve_original_consumer::<Option<Self::Distribution>, E, L>(
                    left,
                    Some(right),
                    placement,
                    context,
                    1,
                )
                .map_err(|cause| E::from(L::original_error(cause)))?;
                return match numerical_phase(
                    context, placement,
                    eredu_runtime::speculative::numerical::SpeculativeNumericalKind::Correction,
                    left,
                    Some(right),
                )
                .map_err(|cause| E::from(L::original_error(cause)))?
                {
                    numerical::NumericalOutput::Correction(Some(value)) => {
                        MlxSpeculativeDistribution::original(
                            value,
                            context
                                .original_numerical()
                                .expect("validated original context")
                                .0,
                        )
                        .map(Some)
                        .map_err(|cause| E::from(L::original_error(cause)))
                    }
                    numerical::NumericalOutput::Correction(None) => Ok(None),
                    _ => unreachable!("correction output"),
                };
            }
            (None, None) => {}
            _ => {
                return Err(E::from(L::original_error(Error::InvalidOperation(
                    "speculative correction sources have different admission modes",
                ))));
            }
        }
        if context.original_numerical().is_some() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original sampling requires a completed distribution",
            ))));
        }
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original policy requires its admitted numerical source",
            ))));
        }
        let stream = sampling_stream(placement, context)?;
        let memory = self.operation_memory([left.memory(), right.memory()], context)?;
        with_memory_recovery(memory.clone(), || {
            let (difference, mass) = eredu_runtime::speculative::numerical::correction::<
                numerical::operators::Native,
            >(left.value(), right.value(), stream)?;
            let mass = mass.item::<f32>(stream);
            if !eredu_runtime::speculative::numerical::correction_has_mass(mass) {
                Ok(None)
            } else {
                Ok(Some(MlxSpeculativeDistribution::new(
                    eredu_runtime::speculative::numerical::correction_logits::<
                        numerical::operators::Native,
                    >(&difference, stream)?,
                    memory,
                )))
            }
        })
        .map_err(E::from)
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
        if let Some(value) = distribution.original_value() {
            reserve_original_consumer::<(), E, L>(value, None, placement, context, 0)
                .map_err(|cause| E::from(L::original_error(cause)))?;
            return self
                .commit_original(value, token, placement, context)
                .map_err(|cause| E::from(L::original_error(cause)));
        }
        if context.original_numerical().is_some() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original sampling requires a completed distribution",
            ))));
        }
        if self.inner.is_original() {
            return Err(E::from(L::original_error(Error::InvalidOperation(
                "original policy requires its admitted numerical source",
            ))));
        }
        let stream = sampling_stream(placement, context)?;
        let memory = self.operation_memory([distribution.memory()], context)?;
        self.retain_callback_memory(&memory);
        with_memory_recovery(memory, || {
            SpeculativeSampler::<MlxSamplingBackend>::commit_token(
                self.inner.ordinary_mut().expect("ordinary sampler"),
                &MlxTensor::from_array(distribution.value().clone()),
                token,
                stream,
            )
        })
        .map_err(E::from)
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
        if distributions.is_empty() || temperature == 0.0 || !context.is_split() {
            return Ok(());
        }
        if self.inner.is_original()
            || distributions.iter().any(|value|value.original_value().is_some()) {
            if context.original_external().is_none()
                || !matches!(context.topology(),eredu_core::SpeculativeExecutionTopology::SameDeviceSplit
                    |eredu_core::SpeculativeExecutionTopology::CrossDeviceSplit) {
                return Err(E::from(L::original_error(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound))));
            }
            let (sources,_)=context.original_numerical().ok_or_else(||E::from(L::original_error(
                Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))))?;
            let frames=[std::mem::size_of::<(&mut [&mut Self::Distribution],f32,Self::Context<'_>)>(),
                std::mem::size_of::<std::slice::IterMut<'_,&mut Self::Distribution>>(),
                std::mem::size_of::<Result<(),Self::Error>>()];
            sources.metadata_funding().reserve_metadata(frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
                .ok_or_else(||E::from(L::original_error(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))))?)
                .map_err(|cause|E::from(L::original_error(Error::WorkspacePlanning(cause))))?;
            for distribution in distributions {
                let value=distribution.original_value().ok_or_else(||E::from(L::original_error(
                    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))))?;
                reserve_original_consumer::<(),E,L>(value,None,SamplingPlacement::Target,context,0)
                    .map_err(|cause|E::from(L::original_error(cause)))?;
                if context.crosses_devices(){
                    let copied=numerical::copy_value_to(value,SamplingPlacement::Draft,SamplingPlacement::Target,context)
                        .map_err(|cause|E::from(L::original_error(cause)))?;
                    let completed=MlxSpeculativeDistribution::original(copied,sources)
                        .map_err(|cause|E::from(L::original_error(cause)))?;
                    **distribution=completed;
                }else{
                    numerical::NumericalProducer::validate_input_at(value,context,SamplingPlacement::Target)
                        .map_err(|cause|E::from(L::original_error(cause)))?;
                }
            }
            // Same-device values keep their completed source owners. A copied
            // distribution replaces its source only after destination completion;
            // later failure preserves all prior outputs and untouched inputs.
            return Ok(());
        }
        let memory = self.operation_memory(
            distributions
                .iter()
                .map(|distribution| distribution.memory()),
            context,
        )?;
        // Admission covers every source before the first submission or value
        // replacement. A later failed transfer cannot strip an earlier output
        // or untouched source of any authority used by this operation.
        for distribution in distributions.iter_mut() {
            distribution.memory_mut().extend_from(&memory);
        }
        with_memory_recovery(memory, || {
            if context.crosses_devices() {
                async_eval_with_event(
                    distributions
                        .iter()
                        .map(|distribution| distribution.value()),
                )?
                .synchronize()?;
                for distribution in distributions {
                    *distribution.value_mut() = distribution.value().copy(context.target())?;
                }
            } else {
                let _completion = context.wait_for_draft_outputs(
                    distributions
                        .iter()
                        .map(|distribution| distribution.value()),
                )?;
            }
            Ok(())
        })
        .map_err(E::from)
    }
}

fn reject_original_context(context: SpeculativeExecutionStreams<'_>) -> Result<(), Error> {
    if context.original_numerical().is_some() {
        return Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ));
    }
    Ok(())
}

/// Authenticate the closed source and selected stream before constructing an
/// outer RNG carrier. Its native value and completion remain producer-owned.
fn reserve_original_key<T, E, L: LogitsSource>(
    key: Option<&numerical::OriginalNumericalKey>,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error> {
    use eredu_nn::workspace::WorkspaceMetadataFundingError;
    use std::mem::{size_of, size_of_val};
    let (sources, environment) = context.original_numerical_for(placement).ok_or(Error::PrefillControl(
        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
    ))?;
    sources.validate_environment(environment)?;
    if let Some(key) = key {
        reserve_original_consumer::<T, E, L>(key.value(), None, placement, context, 0)?;
    } else {
        let stream = match placement {
            SamplingPlacement::Target => context.target(),
            SamplingPlacement::Draft => context.draft(),
            _ => {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ));
            }
        };
        if stream != environment.stream() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
    }
    let parts = [
        size_of::<Option<&numerical::OriginalNumericalKey>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<SamplingPlacement>(),
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<Result<T, E>>(),
        size_of::<Result<eredu_core::BackendFailure, Error>>(),
        size_of::<NativeMemoryRetention>(),
        size_of::<KeyValue<Array>>(),
        size_of::<KeyValue<RandomState>>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}

/// Pays only the newly introduced consumer frames. Numerical producers pay
/// their own plans, native values and completion controls; those are not repeated
/// here. Both sources must belong to this request before either account is used.
fn reserve_original_consumer<T, E, L: LogitsSource>(
    left: &numerical::OriginalNumericalValue,
    right: Option<&numerical::OriginalNumericalValue>,
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'_>,
    phase_calls: usize,
) -> Result<(), Error> {
    use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
    use std::mem::{size_of, size_of_val};
    let stream = match placement {
        SamplingPlacement::Target => context.target(),
        SamplingPlacement::Draft => context.draft(),
        _ => {
            return Err(Error::InvalidOperation(
                "unsupported speculative sampling placement",
            ));
        }
    };
    let (sources, environment) = context.original_numerical_for(placement).ok_or(Error::InvalidOperation(
        "numerical value requires its admitted source context",
    ))?;
    let funding = left.validate_consumer(sources)?;
    if let Some(right) = right {
        right.validate_consumer(sources)?;
    }
    if stream != environment.stream() {
        return Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ));
    }
    sources.validate_environment(environment)?;
    let parts = [
        size_of::<&numerical::OriginalNumericalValue>(),
        size_of::<Option<&numerical::OriginalNumericalValue>>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
        size_of::<SamplingPlacement>(),
        size_of::<Result<&WorkspaceMetadataFunding, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), E>>(),
        size_of::<T>(),
        size_of::<Result<T, Error>>(),
        size_of::<Result<T, E>>(),
        size_of::<Error>(),
        size_of::<L::OriginalError>(),
        size_of::<E>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|bytes| {
            if right.is_some() {
                bytes.checked_add(size_of::<Result<&WorkspaceMetadataFunding, Error>>())
            } else {
                Some(bytes)
            }
        })
        .and_then(|bytes| {
            if phase_calls == 0 {
                Some(bytes)
            } else {
                numerical_phase_control_bytes()
                    .and_then(|one| one.checked_mul(phase_calls))
                    .and_then(|phases| bytes.checked_add(phases))
            }
        })
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}

/// Extra helper return/argument frames outside NumericalProducer::execute.
fn numerical_phase_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<SpeculativeExecutionStreams<'_>>(),size_of::<SamplingPlacement>(),
        size_of::<eredu_runtime::speculative::numerical::SpeculativeNumericalKind>(),
        size_of::<&numerical::OriginalNumericalValue>(),
        size_of::<Option<&numerical::OriginalNumericalValue>>(),
        size_of::<numerical::NumericalOutput>(),
        size_of::<Result<numerical::NumericalOutput, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn numerical_phase(
    context: SpeculativeExecutionStreams<'_>,placement:SamplingPlacement,
    kind: eredu_runtime::speculative::numerical::SpeculativeNumericalKind,
    left: &numerical::OriginalNumericalValue,
    right: Option<&numerical::OriginalNumericalValue>,
) -> Result<numerical::NumericalOutput, Error> {
    numerical::NumericalProducer::execute_at(context,placement,kind,left,right)
}

fn sampling_stream<'a>(
    placement: SamplingPlacement,
    context: SpeculativeExecutionStreams<'a>,
) -> Result<&'a Stream, Exception> {
    match placement {
        SamplingPlacement::Target => Ok(context.target()),
        SamplingPlacement::Draft => Ok(context.draft()),
        _ => Err(Exception::custom(
            "unsupported speculative sampling placement requires an explicit stream",
        )),
    }
}

fn probabilities(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    eredu_runtime::speculative::numerical::normalize::<numerical::operators::Native>(logits, stream)
}

pub(super) fn array_probability_at(
    probabilities: &Array,
    token: u32,
    stream: &Stream,
) -> Result<f32, Exception> {
    use eredu_runtime::speculative::numerical::SpeculativeProbabilityBackend;
    let value = numerical::operators::Native::select(probabilities, token, stream)?;
    Ok(value.item::<f32>(stream))
}

fn uniform(state: Option<&mut RandomState>, stream: &Stream) -> Result<f32, Exception> {
    let state = state
        .ok_or_else(|| Exception::custom("stochastic speculative decoding requires a PRNG key"))?;
    Ok(state.uniform_unit_interval(stream)?.item::<f32>(stream))
}

struct LogitCapture {
    session: eredu_runtime::capture::CaptureSession,
    records: Vec<eredu_core::speculative::SpeculativePredictionCapture>,
}
impl LogitCapture {
    fn observe(
        &mut self,
        logits: &Array,
        position: u64,
        placement: SamplingPlacement,
        intervention: Option<&eredu_core::intervention::AdmittedInterventionPlan>,
        stream: &Stream,
        domain: Option<eredu_core::capture::CaptureTokenDomain<'_>>,
    ) -> Result<Option<Array>, Exception> {
        use eredu_core::{
            capture::CapturePhase,
            speculative::{SpeculativeCaptureRole, SpeculativePredictionCapture},
        };
        let role = match placement {
            SamplingPlacement::Target => SpeculativeCaptureRole::Target,
            SamplingPlacement::Draft => SpeculativeCaptureRole::Draft,
            _ => {
                return Err(Exception::custom(
                    "unsupported speculative capture placement",
                ));
            }
        };
        let phase = if position == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        self.session
            .select_prediction_interventions(
                intervention.cloned(),
                std::sync::Arc::new(
                    crate::composition::mlx::session::intervention::NativeInterventionEstimator,
                ),
            )
            .map_err(Exception::from_source)?;
        self.session
            .begin_step(phase, position)
            .map_err(Exception::from_source)?;
        // Sampling receives one vocabulary row. Normalize only its logical shape
        // to the ordinary model.logits catalog; the collector owns reservation,
        // evaluation, bounded transformation and host transfer.
        let values = logits.reshape(&[1, 1, -1], stream)?;
        let tensor = MlxTensor::from_array(values);
        let mut backend = crate::composition::mlx::session::bounded_capture::NativeCapture {
            partition: None,
            stream,
            domain,
        };
        self.session
            .observe(
                &mut backend,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                &tensor,
            )
            .map_err(crate::composition::mlx::session::bounded_capture::capture_error)
            .map_err(Exception::from_source)?;
        let effective = self
            .session
            .intervene(
                &mut backend,
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                &tensor,
            )
            .map_err(crate::composition::mlx::session::bounded_capture::capture_error)
            .map_err(Exception::from_source)?;
        self.session
            .finish_interventions()
            .map_err(Exception::from_source)?;
        if let Some(capture) = self.session.take_step() {
            self.records.push(SpeculativePredictionCapture {
                role,
                position,
                capture: eredu_core::capture::CapturedStepDelivery::Legacy(capture),
            });
        }
        effective
            .map(|value| value.into_array().reshape(logits.shape(), stream))
            .transpose()
    }
}

pub(crate) fn validate_control_capture(
    plan: &eredu_core::capture::AdmittedCapturePlan,
) -> Result<(), eredu_core::capture::CaptureError> {
    if plan.is_empty() {
        return Ok(());
    }
    if plan.request().batch != 1 || plan.request().prompt_tokens != 1 {
        return Err(eredu_core::capture::CaptureError::Invalid(
            "speculative logits require one-row capture admission".into(),
        ));
    }
    if plan
        .plan()
        .selections
        .iter()
        .any(|s| s.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    {
        return Err(eredu_core::capture::CaptureError::Unsupported(
                "controlled MLX speculation currently captures model.logits; layer captures require draft/verification phase attribution".into()));
    }
    Ok(())
}
