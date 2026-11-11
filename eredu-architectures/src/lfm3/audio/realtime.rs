//! Incremental speech generation over the shared realtime transaction.

use super::{
    AudioPhase, AudioSchedule, AudioScheduleError, DepthDeclaration, DepthFrame, DepthProgress,
    PreparedAudio,
};
use eredu_core::{
    execution_control::{SnapshotEstimate, SnapshotLimits, SnapshotResourceKind, SnapshotUsage},
    scheduler::SemanticStateTransaction,
    BackendFailure, Completion, TokenFilter,
};
use eredu_nn::{Index, Tensor, TensorElementType};
use eredu_runtime::{
    execution_control::{
        snapshot_value_estimate, ForkStateEstimate, IsolatedStateCopy, IsolatedStateError,
        IsolatedStateFork, IsolatedStateSnapshot, SnapshotBudget, SnapshotReservation,
        SnapshotRetainedGrowth, SnapshotValue,
    },
    DenseSamplingBackend, GenerationSampler, RealtimeFrameExecutionError, RealtimeFrameTransition,
    RealtimeGenerationBranch, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeGenerationTransactionError, Sampler, SamplingBackend,
};
use std::{cell::RefCell, rc::Rc};

/// Public output of one completed Audio advancement.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioAdvance {
    /// A depth pass completed. Its tentative code values remain private.
    Refining,
    /// A committed language token. PCM may contain a final codec tail.
    Text {
        /// Canonical language token ID.
        token: i32,
        /// Final 24-kHz mono samples when this token closes the response.
        pcm_tail: Vec<f32>,
    },
    /// A committed speech frame with newly stable codec samples.
    Speech {
        /// One code per codebook.
        codes: Vec<i32>,
        /// Newly emitted 24-kHz mono samples.
        pcm: Vec<f32>,
    },
    /// Speech EOS and the corresponding 24-kHz codec tail.
    SpeechEnd(Vec<f32>),
    /// No further advances are available.
    Finished,
}

/// Failure in realtime Audio preparation or advancement.
#[derive(Debug, thiserror::Error)]
pub enum AudioSessionError {
    /// The checkpoint's generation defaults cannot be used by the sampler.
    #[error("invalid LFM3 Audio sampling controls")]
    Sampling,
    /// The speech schedule rejected the request or transition.
    #[error(transparent)]
    Schedule(#[from] AudioScheduleError),
    /// The shared generation owner rejected state admission.
    #[error(transparent)]
    Admission(#[from] RealtimeGenerationTransactionError),
    /// A shared transition did not commit.
    #[error(transparent)]
    Transition(#[from] RealtimeFrameExecutionError),
    /// Another branch is already advancing on the shared graph.
    #[error("the shared LFM3 Audio graph is busy")]
    Busy,
    /// Native retention could not be established before advancement.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// Portable failure when capturing, restoring or forking a resident Audio session.
#[derive(Debug, thiserror::Error)]
pub enum AudioSnapshotError {
    /// The session has no snapshot budget.
    #[error("LFM3 Audio snapshots are not configured")]
    NotConfigured,
    /// Snapshot limits can only be configured once.
    #[error("LFM3 Audio snapshots are already configured")]
    AlreadyConfigured,
    /// Explicit cancellation is terminal.
    #[error("LFM3 Audio session is cancelled")]
    Cancelled,
    /// A saved model state originated in a different session.
    #[error("LFM3 Audio snapshot belongs to another session")]
    ForeignSession,
    /// The shared state owner rejected or failed the copy.
    #[error(transparent)]
    State(#[from] IsolatedStateError),
}

/// Retained checkpoint sampling controls and bounded per-lane history.
#[derive(Debug, Clone)]
pub struct AudioSampling {
    text_temperature: f32,
    speech_temperature: f32,
    text_history: usize,
    speech_history: usize,
    text_sampler: GenerationSampler,
    speech_sampler: GenerationSampler,
}

impl SnapshotValue for AudioSampling {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.text_sampler.estimate_snapshot_value(),
            self.speech_sampler.estimate_snapshot_value(),
        ])
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            text_sampler: self.text_sampler.copy_snapshot_value(retention)?,
            speech_sampler: self.speech_sampler.copy_snapshot_value(retention)?,
            ..*self
        })
    }
}

impl AudioSampling {
    /// Reads the publisher's text and speech sampling settings.
    pub fn from_artifact(artifact: &PreparedAudio) -> Result<Self, AudioSessionError> {
        let config = &artifact.text_assets.generation;
        let float = |key: &str, default: f32| {
            let value = match config.get(key).filter(|v| !v.is_null()) {
                Some(value) => value.as_f64().ok_or(AudioSessionError::Sampling)? as f32,
                None => default,
            };
            value
                .is_finite()
                .then_some(value)
                .ok_or(AudioSessionError::Sampling)
        };
        let integer = |key: &str, default: i32| {
            match config.get(key).filter(|v| !v.is_null()) {
                Some(value) => value.as_i64().and_then(|v| i32::try_from(v).ok()),
                None => Some(default),
            }
            .filter(|v| *v >= 0)
            .ok_or(AudioSessionError::Sampling)
        };
        let text_sampling = config
            .get("do_sample")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let text_temperature = if text_sampling {
            float("temperature", 1.0)?
        } else {
            0.0
        };
        let speech_temperature = float("audio_temperature", 0.0)?;
        let text_history = integer("text_repetition_window", 32)?;
        let speech_history = integer("audio_repetition_window", 64)?;
        let text_top_k = if text_sampling {
            integer("top_k", 50)?
        } else {
            0
        };
        let mut speech_top_k = integer("audio_top_k", 0)?;
        let top_p = if text_sampling {
            float("top_p", 1.0)?
        } else {
            1.0
        };
        let min_p = if text_sampling {
            float("min_p", 0.0)?
        } else {
            0.0
        };
        let text_penalty = float("repetition_penalty", 1.0)?;
        let speech_penalty = float("audio_repetition_penalty", 1.0)?;
        if text_temperature < 0.0
            || speech_temperature < 0.0
            || text_penalty <= 0.0
            || speech_penalty <= 0.0
            || !(0.0 < top_p && top_p <= 1.0)
            || !(0.0..=1.0).contains(&min_p)
        {
            return Err(AudioSessionError::Sampling);
        }
        let speech_temperature =
            if config.get("audio_temperature").is_none_or(|v| v.is_null()) && speech_top_k > 0 {
                1.0
            } else {
                speech_temperature
            };
        if speech_temperature == 0.0 {
            speech_top_k = 0;
        }
        let (text_top_k, top_p, min_p) = if text_temperature == 0.0 {
            (0, 1.0, 0.0)
        } else {
            (text_top_k, top_p, min_p)
        };
        Ok(Self {
            text_temperature,
            speech_temperature,
            text_history: text_history as usize,
            speech_history: speech_history as usize,
            text_sampler: GenerationSampler::new()
                .top_k(text_top_k)
                .top_p(top_p)
                .min_p(min_p)
                .penalties(text_penalty, text_history, 0.0, 0.0),
            speech_sampler: GenerationSampler::new()
                .top_k(speech_top_k)
                .top_p(1.0)
                .min_p(0.0)
                .penalties(speech_penalty, speech_history, 0.0, 0.0),
        })
    }
}

/// One selected model mechanism, including native state and completion.
/// The kernel exposes score rows and applies committed embeddings; the family
/// driver owns sampling, text/speech mode switches, EOS and slot accounting.
pub trait AudioExecutionKernel {
    /// Native tensor handle.
    type Tensor: Tensor;
    /// Native sampling failure retained as a source.
    type SamplingError: std::error::Error + Send + Sync + 'static;
    /// Existing sampler and explicit random mechanism.
    type SamplerBackend: DenseSamplingBackend<
        Logits = Self::Tensor,
        Token = Self::Tensor,
        Context = <Self::Tensor as Tensor>::Context,
        Error = Self::SamplingError,
    >;
    /// Canonical model state with a transaction-local branch.
    type State: SemanticStateTransaction<Error = BackendFailure>;
    /// Exact completion of a step, including its retained cache resources.
    type Completion: Completion;

    /// Selected native context.
    fn context(&self) -> &<Self::Tensor as Tensor>::Context;
    /// Retained depth declaration.
    fn depth(&self) -> &DepthDeclaration;
    /// Scores one text decision from the current normalized language position.
    fn text_logits(
        &mut self,
        state: &<Self::State as SemanticStateTransaction>::Branch,
    ) -> Result<Self::Tensor, BackendFailure>;
    /// Scores one full bidirectional depth pass.
    fn depth_logits(
        &mut self,
        state: &<Self::State as SemanticStateTransaction>::Branch,
        frame: &DepthFrame,
    ) -> Result<Self::Tensor, BackendFailure>;
    /// Appends a text token through the shared language traversal.
    fn append_text(
        &mut self,
        state: &mut <Self::State as SemanticStateTransaction>::Branch,
        token: i32,
    ) -> Result<(), BackendFailure>;
    /// Appends speech feedback. Only ordinary speech codes enter the codec.
    fn append_speech(
        &mut self,
        state: &mut <Self::State as SemanticStateTransaction>::Branch,
        codes: &[i32],
        eos: bool,
    ) -> Result<Vec<f32>, BackendFailure>;
    /// Flushes the active codec tail, if any.
    fn flush_speech(
        &mut self,
        state: &mut <Self::State as SemanticStateTransaction>::Branch,
    ) -> Result<Vec<f32>, BackendFailure>;
    /// Returns completion after settling every state/output root from the step.
    /// A failure path must preserve submitted resources until safe release.
    fn complete(
        &mut self,
        state: &<Self::State as SemanticStateTransaction>::Branch,
    ) -> Result<Self::Completion, BackendFailure>;

    /// Holds the branch's reservation through all native work in an advance.
    /// Unresolved work must retain it even when the operation fails or unwinds.
    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure>;
}

type Random<K> = <<K as AudioExecutionKernel>::SamplerBackend as SamplingBackend>::RandomState;
type ModelBranch<K> = <<K as AudioExecutionKernel>::State as SemanticStateTransaction>::Branch;
type Branch<K> = RealtimeGenerationBranch<
    ModelBranch<K>,
    AudioSchedule,
    AudioSampling,
    GenerationSampler,
    Random<K>,
    <K as AudioExecutionKernel>::Completion,
>;
type Generation<K> = RealtimeGenerationState<
    <K as AudioExecutionKernel>::State,
    AudioSchedule,
    AudioSampling,
    GenerationSampler,
    Random<K>,
    <K as AudioExecutionKernel>::Completion,
>;

/// One controlled synchronous session using the shared atomic generation owner.
pub struct AudioSession<K: AudioExecutionKernel> {
    transition: Rc<RefCell<Transition<K>>>,
    generation: Generation<K>,
    cancelled: bool,
    snapshot_budget: Option<SnapshotBudget>,
    fork: Option<ForkRetention<K>>,
}

struct ForkRetention<K: AudioExecutionKernel> {
    growth: SnapshotRetainedGrowth,
    estimate: fn(&Branch<K>) -> Result<u64, BackendFailure>,
}

/// Reusable idle Audio state, including unfinished refinement and codec carry.
pub struct AudioSnapshot<K: AudioExecutionKernel>(IsolatedStateSnapshot<Generation<K>>);

impl<K: AudioExecutionKernel> AudioSession<K>
where
    K::State: IsolatedStateCopy,
    Random<K>: SnapshotValue,
{
    /// Sets this session's snapshot limits once. No state is copied.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        if self.snapshot_budget.is_some() {
            return Err(AudioSnapshotError::AlreadyConfigured);
        }
        self.snapshot_budget = Some(SnapshotBudget::new(limits));
        Ok(())
    }

    /// Reports live retention and cumulative copy charges for this session.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }

    /// Saves the exact next advancement without running another model step.
    pub fn snapshot(&self) -> Result<AudioSnapshot<K>, AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        Ok(AudioSnapshot(IsolatedStateSnapshot::capture(
            &self.generation,
            budget,
        )?))
    }

    /// Rewinds generation to a saved boundary from this session.
    /// The caller owns any output already returned by previous advances.
    pub fn restore(&mut self, saved: &AudioSnapshot<K>) -> Result<(), AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        saved.0.restore(&mut self.generation)?;
        Ok(())
    }

    fn check_snapshot_boundary(&self) -> Result<(), AudioSnapshotError> {
        if self.cancelled {
            return Err(AudioSnapshotError::Cancelled);
        }
        Ok(())
    }
}

impl<K: AudioExecutionKernel> AudioSession<K>
where
    K::State: IsolatedStateFork + ForkStateEstimate,
    Random<K>: Clone + SnapshotValue,
{
    /// Copies the current settled continuation without replaying prefill.
    /// The child shares the resident graph and copy budget, and advances independently.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        let state_estimate = self
            .generation
            .estimate_isolated_copy()
            .map_err(|error| IsolatedStateError::Backend(failure(error)))?;
        // The child owns an origin and heap-backed reservation/high-water counters.
        let estimate = snapshot_value_estimate::<Self>([
            Ok(state_estimate),
            Ok(SnapshotEstimate {
                retained_bytes: 256,
                copy_bytes: 256,
            }),
        ])
        .map_err(IsolatedStateError::Backend)?;
        let retention = budget
            .reserve(SnapshotResourceKind::Branch, Some(estimate))
            .map_err(IsolatedStateError::Admission)?;
        let generation = self
            .generation
            .fork_isolated(&retention)
            .map_err(|error| IsolatedStateError::Backend(failure(error)))?;
        Ok(Self {
            transition: self.transition.clone(),
            generation,
            cancelled: false,
            snapshot_budget: Some(budget.clone()),
            fork: Some(ForkRetention {
                growth: SnapshotRetainedGrowth::new(&retention, state_estimate.retained_bytes),
                estimate: Generation::<K>::estimate_fork_branch,
            }),
        })
    }

    #[cfg(feature = "audio")]
    pub(super) fn inherit_fork_growth(
        &mut self,
        growth: SnapshotRetainedGrowth,
    ) -> Result<(), AudioSnapshotError> {
        let estimate = self
            .generation
            .estimate_isolated_copy()
            .map_err(|error| IsolatedStateError::Backend(failure(error)))?;
        growth
            .admit(estimate.retained_bytes)
            .map_err(IsolatedStateError::Admission)?;
        self.fork = Some(ForkRetention {
            growth,
            estimate: Generation::<K>::estimate_fork_branch,
        });
        Ok(())
    }
}

impl<K: AudioExecutionKernel> AudioSession<K>
where
    Random<K>: Clone,
{
    /// Adopts prefetched language state and the exact initialized speech schedule.
    pub fn new(
        kernel: K,
        state: K::State,
        schedule: AudioSchedule,
        sampling: AudioSampling,
        random: Random<K>,
    ) -> Result<Self, AudioSessionError> {
        Self::with_transition(
            Rc::new(RefCell::new(Transition(kernel))),
            state,
            schedule,
            sampling,
            random,
        )
    }

    pub(super) fn with_transition(
        transition: Rc<RefCell<Transition<K>>>,
        state: K::State,
        schedule: AudioSchedule,
        sampling: AudioSampling,
        random: Random<K>,
    ) -> Result<Self, AudioSessionError> {
        let mut samplers = vec![sampling.text_sampler.clone()];
        samplers
            .extend((0..schedule.policy().codebooks()).map(|_| sampling.speech_sampler.clone()));
        let contract = RealtimeGenerationContract::new(schedule.policy().clone(), samplers.len());
        let generation = RealtimeGenerationState::new(
            state,
            contract,
            schedule,
            sampling,
            samplers,
            Some(random),
        )?;
        Ok(Self {
            transition,
            generation,
            cancelled: false,
            snapshot_budget: None,
            fork: None,
        })
    }

    #[cfg(feature = "audio")]
    pub(super) fn inherit_snapshot_budget(&mut self, budget: Option<SnapshotBudget>) {
        self.snapshot_budget = budget;
    }

    /// Executes a text decision or one depth pass and publishes it on completion.
    pub fn advance(&mut self) -> Result<AudioAdvance, AudioSessionError> {
        if self.cancelled || self.phase() == AudioPhase::Finished {
            return Ok(AudioAdvance::Finished);
        }
        let mut transition = self
            .transition
            .try_borrow_mut()
            .map_err(|_| AudioSessionError::Busy)?;
        let result = if let Some(fork) = &self.fork {
            K::with_fork_retention(fork.growth.clone(), || {
                self.generation
                    .execute_frame_transition_checked(&(), &mut *transition, |branch| {
                        let bytes = (fork.estimate)(branch)?;
                        fork.growth.admit(bytes).map_err(failure)
                    })
            })?
        } else {
            self.generation
                .execute_frame_transition(&(), &mut *transition)
        };
        result.map_err(Into::into)
    }

    /// Current phase, including cancellation.
    pub fn phase(&self) -> AudioPhase {
        if self.cancelled {
            AudioPhase::Finished
        } else {
            self.generation.schedule_state().phase()
        }
    }

    /// Language slots committed by successful advances.
    pub fn committed_slots(&self) -> usize {
        self.generation.schedule_state().committed_slots()
    }

    /// Stops between settled advances. Unfinished codebooks and codec carry
    /// produce no further public events.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Drains the same controlled driver, omitting intermediate refinement events.
    pub fn events(&mut self) -> impl Iterator<Item = Result<AudioAdvance, AudioSessionError>> + '_ {
        let mut done = false;
        std::iter::from_fn(move || {
            if done {
                return None;
            }
            loop {
                match self.advance() {
                    Ok(AudioAdvance::Finished) => {
                        done = true;
                        return None;
                    }
                    Ok(AudioAdvance::Refining) => continue,
                    result => {
                        done = result.is_err();
                        return Some(result);
                    }
                }
            }
        })
    }
}

pub(super) struct Transition<K>(pub(super) K);

impl<K: AudioExecutionKernel>
    RealtimeFrameTransition<
        ModelBranch<K>,
        AudioSchedule,
        AudioSampling,
        GenerationSampler,
        Random<K>,
        K::Completion,
    > for Transition<K>
where
    Random<K>: Clone,
{
    type Input = ();
    type Output = AudioAdvance;
    type Error = BackendFailure;

    fn execute(
        &mut self,
        _: &(),
        branch: &mut Branch<K>,
    ) -> Result<(AudioAdvance, K::Completion), BackendFailure> {
        let event = match branch.schedule_state().phase() {
            AudioPhase::Text => self.text(branch)?,
            AudioPhase::Speech => self.speech(branch)?,
            AudioPhase::Finished => return Err(failure(AudioScheduleError::OutOfPhase)),
        };
        Ok((event, self.0.complete(branch.model_state())?))
    }
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

fn sample<K: AudioExecutionKernel>(
    sampler: &mut GenerationSampler,
    logits: &K::Tensor,
    temperature: f32,
    random: Option<&mut Random<K>>,
    context: &<K::Tensor as Tensor>::Context,
) -> Result<i32, BackendFailure> {
    let scaled;
    let (logits, temperature) = if temperature > 0.0 && temperature != 1.0 {
        scaled =
            K::SamplerBackend::scale_temperature(logits, temperature, context).map_err(failure)?;
        (&scaled, 1.0)
    } else {
        (logits, temperature)
    };
    let token = <GenerationSampler as Sampler<K::SamplerBackend>>::sample(
        sampler,
        logits,
        temperature,
        random,
        context,
    )
    .map_err(failure)?;
    i32::try_from(K::SamplerBackend::token_id(&token, context).map_err(failure)?).map_err(failure)
}

fn record(sampler: &mut GenerationSampler, id: i32, limit: usize) {
    if limit == 0 {
        return;
    }
    let history = sampler.generated_tokens();
    if history.len() < limit {
        sampler.accept_token(id as u32);
    } else {
        let values = history[history.len() + 1 - limit..]
            .iter()
            .copied()
            .chain(std::iter::once(id as u32))
            .collect::<Vec<_>>();
        sampler.set_generated_tokens(values);
    }
}

impl<K: AudioExecutionKernel> Transition<K>
where
    Random<K>: Clone,
{
    fn text(&mut self, branch: &mut Branch<K>) -> Result<AudioAdvance, BackendFailure> {
        let logits = self.0.text_logits(branch.model_state())?;
        let context = self.0.context();
        let mut allowed = vec![true; logits.shape().last().copied().unwrap_or(0).max(0) as usize];
        for id in branch.schedule_state().policy().suppressed_tokens() {
            *allowed
                .get_mut(id as usize)
                .ok_or_else(|| failure(AudioScheduleError::Incompatible))? = false;
        }
        let filter = TokenFilter::allowed(allowed).map_err(failure)?;
        let logits = logits
            .cast_float(TensorElementType::F32, context)
            .map_err(failure)?;
        let logits =
            K::SamplerBackend::apply_token_filter(&logits, &filter, context).map_err(failure)?;
        let mut sampler = branch.samplers()[0].clone();
        let temperature = branch.sampling().text_temperature;
        let token = branch
            .with_random_state_transaction(|random| {
                sample::<K>(&mut sampler, &logits, temperature, random.as_mut(), context)
            })
            .map_err(failure)?;
        branch
            .schedule_state_mut()
            .commit_text(token)
            .map_err(failure)?;
        let history = branch.sampling().text_history;
        record(&mut branch.samplers_mut()[0], token, history);
        self.0.append_text(branch.model_state_mut(), token)?;
        let pcm_tail = if branch.schedule_state().phase() == AudioPhase::Finished {
            self.0.flush_speech(branch.model_state_mut())?
        } else {
            Vec::new()
        };
        Ok(AudioAdvance::Text { token, pcm_tail })
    }

    fn speech(&mut self, branch: &mut Branch<K>) -> Result<AudioAdvance, BackendFailure> {
        if branch.schedule_state().depth_frame().is_none() {
            let count = branch.schedule_state().policy().codebooks();
            let order = branch
                .with_random_state_transaction(|random| {
                    permutation::<K>(
                        count,
                        random
                            .as_mut()
                            .ok_or_else(|| failure(AudioSessionError::Sampling))?,
                        self.0.context(),
                    )
                })
                .map_err(failure)?;
            branch
                .schedule_state_mut()
                .begin_depth(self.0.depth(), &order)
                .map_err(failure)?;
        }
        let frame = branch
            .schedule_state()
            .depth_frame()
            .ok_or_else(|| failure(AudioScheduleError::OutOfPhase))?;
        let (pass, slots) = frame
            .next_pass()
            .ok_or_else(|| failure(AudioScheduleError::OutOfPhase))?;
        let slots = slots.to_vec();
        let logits = self.0.depth_logits(branch.model_state(), frame)?;
        let context = self.0.context();
        let temperature = branch.sampling().speech_temperature;
        let mut samplers = slots
            .iter()
            .map(|&slot| branch.samplers()[slot + 1].clone())
            .collect::<Vec<_>>();
        let sampled = branch
            .with_random_state_transaction(|random| {
                let mut sampled = Vec::with_capacity(slots.len());
                for (slot, sampler) in slots.into_iter().zip(&mut samplers) {
                    let row = logits
                        .index(&[Index::At(0), Index::At(slot as i32)], context)
                        .and_then(|row| row.cast_float(TensorElementType::F32, context))
                        .map_err(failure)?;
                    sampled.push(sample::<K>(
                        sampler,
                        &row,
                        temperature,
                        random.as_mut(),
                        context,
                    )?);
                }
                Ok::<_, BackendFailure>(sampled)
            })
            .map_err(failure)?;
        match branch
            .schedule_state_mut()
            .reveal(pass, &sampled)
            .map_err(failure)?
        {
            DepthProgress::Pending => Ok(AudioAdvance::Refining),
            DepthProgress::End => {
                let codes =
                    vec![self.0.depth().eos_code(); branch.schedule_state().policy().codebooks()];
                self.0
                    .append_speech(branch.model_state_mut(), &codes, true)?;
                Ok(AudioAdvance::SpeechEnd(
                    self.0.flush_speech(branch.model_state_mut())?,
                ))
            }
            DepthProgress::Audio(codes) => {
                let limit = branch.sampling().speech_history;
                for (sampler, &code) in branch.samplers_mut()[1..].iter_mut().zip(&codes) {
                    record(sampler, code, limit);
                }
                let mut pcm = self
                    .0
                    .append_speech(branch.model_state_mut(), &codes, false)?;
                if branch.schedule_state().phase() == AudioPhase::Finished {
                    pcm.extend(self.0.flush_speech(branch.model_state_mut())?);
                }
                Ok(AudioAdvance::Speech { codes, pcm })
            }
        }
    }
}

// IID ranks yield a uniform permutation conditional on no ties. Retry a tied
// finite-precision draw instead of biasing the earliest slot.
fn permutation<K: AudioExecutionKernel>(
    count: usize,
    random: &mut Random<K>,
    context: &<K::Tensor as Tensor>::Context,
) -> Result<Vec<usize>, BackendFailure> {
    for _ in 0..8 {
        let draws = K::SamplerBackend::uniform_f32(&[count as i32], random, context)
            .map_err(failure)?
            .to_f32_vec(context)
            .map_err(failure)?;
        if draws.len() != count || draws.iter().any(|&x| !(0.0..1.0).contains(&x)) {
            return Err(failure(AudioSessionError::Sampling));
        }
        let mut order = (0..count).collect::<Vec<_>>();
        order.sort_unstable_by(|&a, &b| draws[a].total_cmp(&draws[b]));
        if order
            .windows(2)
            .all(|pair| draws[pair[0]] != draws[pair[1]])
        {
            return Ok(order);
        }
    }
    Err(failure(AudioSessionError::Sampling))
}
