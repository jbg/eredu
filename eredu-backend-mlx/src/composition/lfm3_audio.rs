//! MLX binding for the architecture-owned resident Audio driver.

use crate::backend::{
    nn::shared::MlxNeuralBackend,
    random::RandomState,
    runtime::{
        cache::state::MlxHybridState,
        checkpoint::binding::{
            build_neutral_module_bindings_with_recipes, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        generation::MlxSamplingBackend,
    },
    submission_recovery::{Recovery, Retention, Status},
    ExecutionContext,
};
use crate::{MlxTensor, Stream};
use eredu_architectures::lfm3::audio::{
    AudioAdvance, AudioPhase, AudioRequest, AudioSession, AudioSessionError, AudioSnapshot,
    AudioSnapshotError, AudioTranscription, AudioTranscriptionBlock, AudioTranscriptionError,
    AudioTranscriptionSnapshot, BindingSource, LogMelFeatures, PreparedAudio,
    PreparedResidentAudio, ResidentAudio, ResidentAudioBinding, ResidentAudioError,
};
#[cfg(feature = "audio")]
use eredu_architectures::lfm3::audio::{
    AudioIngress, AudioIngressError, AudioIngressProgress, AudioIngressRequest,
    AudioIngressSnapshot, AudioPcmTranscription, AudioPcmTranscriptionSnapshot,
};
use eredu_core::{
    execution_control::{SnapshotLimits, SnapshotUsage},
    BackendFailure,
};
use eredu_nn::Parameterized;
use eredu_runtime::{execution_control::SnapshotRetainedGrowth, RuntimeLayerState, StateLayout};

struct Binding {
    execution: Stream,
    weights: Stream,
}

struct RetainedBranch {
    _growth: SnapshotRetainedGrowth,
}

impl Retention for RetainedBranch {
    fn observe(&self, _: Status) {}
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

impl ResidentAudioBinding<MlxNeuralBackend> for Binding {
    type State = MlxHybridState;
    type SamplingError = safemlx::error::Exception;
    type Sampling = MlxSamplingBackend;

    fn context(&self) -> &Stream {
        &self.execution
    }

    fn bind<M: Parameterized<MlxTensor>>(
        &self,
        module: &mut M,
        weights: &BindingSource,
    ) -> Result<(), BackendFailure> {
        let source = weights.source();
        let mut recipes = weights.recipes().clone();
        let bindings =
            build_neutral_module_bindings_with_recipes(module, source.as_ref(), &mut recipes)
                .map_err(failure)?;
        let arrays =
            materialize_module_bindings(source.as_ref(), &bindings, &self.weights, self.context())
                .map_err(failure)?;
        populate_module_from_arrays_excluding(module, &arrays, |_| false).map_err(failure)
    }

    fn state(&self, layout: &StateLayout) -> Result<Self::State, BackendFailure> {
        MlxHybridState::device(layout.clone(), &[]).map_err(failure)
    }

    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure> {
        // This binding constructs only device-local state. Append roots are immutable.
        state.deep_clone_state().map_err(failure)
    }

    fn retained_values(state: &Self::State) -> Vec<&MlxTensor> {
        state
            .layers()
            .iter()
            .flat_map(RuntimeLayerState::retained_values)
            .collect()
    }

    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure> {
        let _scope = Recovery::begin(RetainedBranch { _growth: retention }).map_err(failure)?;
        Ok(operation())
    }
}

type Model = ResidentAudio<MlxNeuralBackend, Binding>;

/// Bound resident LFM3 Audio graph using exact F32 widening of floating source weights.
pub struct MlxAudioModel(Model);

impl MlxAudioModel {
    /// Binds an admitted Audio artifact on an explicitly selected device.
    /// The weights context is a separate host-side materialization stream.
    pub fn load(
        artifact: PreparedAudio,
        execution: ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        let prepared = PreparedResidentAudio::select(artifact)?;
        let weights = ExecutionContext::cpu().map_err(failure)?;
        Self::load_prepared(prepared, &weights, &execution)
    }

    /// Binds a retained resident selection on the supplied execution and weight streams.
    pub fn load_prepared(
        prepared: PreparedResidentAudio,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        Ok(Self(ResidentAudio::from_prepared(
            prepared,
            Binding {
                execution: execution.stream().clone(),
                weights: weights.stream().clone(),
            },
        )?))
    }

    /// Prefills the request and creates a controlled session with an explicit seed.
    pub fn start(
        self,
        request: AudioRequest<'_>,
        seed: u64,
    ) -> Result<MlxAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxAudioSession(self.0.start(request, random)?))
    }

    /// Prepares the checkpoint's live English transcription session.
    pub fn prepare_live_transcription(
        self,
    ) -> Result<MlxAudioTranscription, AudioTranscriptionError> {
        self.0
            .prepare_live_transcription()
            .map(MlxAudioTranscription)
    }

    /// Prepares LiveASR from a stream of mono 16-kHz PCM.
    #[cfg(feature = "audio")]
    pub fn prepare_pcm_transcription(
        self,
    ) -> Result<MlxAudioPcmTranscription, AudioTranscriptionError> {
        self.0
            .prepare_pcm_transcription()
            .map(MlxAudioPcmTranscription)
    }

    /// Prepares one streamed mono PCM utterance between the supplied token spans.
    #[cfg(feature = "audio")]
    pub fn prepare_ingress(
        self,
        request: AudioIngressRequest<'_>,
        seed: u64,
    ) -> Result<MlxAudioIngress, AudioIngressError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxAudioIngress(self.0.prepare_ingress(request, random)?))
    }
}

/// Controlled PCM ingress for a resident LFM3 Audio prompt.
#[cfg(feature = "audio")]
pub struct MlxAudioIngress(AudioIngress<MlxNeuralBackend, Binding>);

/// Saved continuation for one resident Audio prompt ingress.
#[cfg(feature = "audio")]
pub struct MlxAudioIngressSnapshot(AudioIngressSnapshot<MlxNeuralBackend, Binding>);

#[cfg(feature = "audio")]
impl MlxAudioIngress {
    /// Forks a settled streamed prompt on the same resident graph.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.0.fork().map(Self)
    }

    /// Configures the shared ingress and response snapshot limits.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.0.enable_snapshots(limits)
    }

    /// Reports current retained storage and cumulative copying.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.0.snapshot_usage()
    }

    /// Saves the settled prompt continuation.
    pub fn snapshot(&self) -> Result<MlxAudioIngressSnapshot, AudioSnapshotError> {
        self.0.snapshot().map(MlxAudioIngressSnapshot)
    }

    /// Restores PCM carry and both model caches from this ingress.
    pub fn restore(&mut self, saved: &MlxAudioIngressSnapshot) -> Result<(), AudioSnapshotError> {
        self.0.restore(&saved.0)
    }

    /// Current committed sample and audio-token positions.
    pub fn progress(&self) -> AudioIngressProgress {
        self.0.progress()
    }

    /// Accepts finite mono 16-kHz PCM and advances complete input positions.
    pub fn submit(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AudioIngressProgress, AudioIngressError> {
        let _scope = Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)?;
        self.0.submit(samples, sample_rate)
    }

    /// Flushes the utterance and starts the same response driver as finite prompts.
    pub fn finish(self) -> Result<MlxAudioSession, AudioIngressError> {
        let _scope = Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)?;
        self.0.finish().map(MlxAudioSession)
    }

    /// Drops the incomplete prompt without starting a response.
    pub fn cancel(self) {
        self.0.cancel();
    }
}

/// Controlled live transcription over one resident Audio graph.
pub struct MlxAudioTranscription(AudioTranscription<MlxNeuralBackend, Binding>);

/// Reusable saved continuation for this LiveASR input mode.
pub struct MlxAudioTranscriptionSnapshot(AudioTranscriptionSnapshot<MlxNeuralBackend, Binding>);

impl MlxAudioTranscription {
    /// Forks the settled continuation on the same resident graph.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.0.fork().map(Self)
    }

    /// Configures bounded snapshot and restore.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.0.enable_snapshots(limits)
    }

    /// Reports retained snapshots and cumulative copy work.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.0.snapshot_usage()
    }

    /// Saves the current settled continuation.
    pub fn snapshot(&self) -> Result<MlxAudioTranscriptionSnapshot, AudioSnapshotError> {
        self.0.snapshot().map(MlxAudioTranscriptionSnapshot)
    }

    /// Restores a saved continuation from this session.
    pub fn restore(
        &mut self,
        saved: &MlxAudioTranscriptionSnapshot,
    ) -> Result<(), AudioSnapshotError> {
        self.0.restore(&saved.0)
    }

    /// Maximum mel frames in one block, derived from the checkpoint cadence.
    pub fn frames_per_block(&self) -> usize {
        self.0.frames_per_block()
    }

    /// Processes one mel block and publishes its transcript after completion.
    pub fn advance(
        &mut self,
        features: &LogMelFeatures,
        final_chunk: bool,
        restart: bool,
    ) -> Result<AudioTranscriptionBlock, AudioTranscriptionError> {
        let _scope = Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)?;
        self.0.advance(features, final_chunk, restart)
    }

    /// Stops between settled blocks.
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
}

/// Controlled LiveASR with bounded PCM input.
#[cfg(feature = "audio")]
pub struct MlxAudioPcmTranscription(AudioPcmTranscription<MlxNeuralBackend, Binding>);

/// Reusable saved continuation for this LiveASR input mode.
#[cfg(feature = "audio")]
pub struct MlxAudioPcmTranscriptionSnapshot(
    AudioPcmTranscriptionSnapshot<MlxNeuralBackend, Binding>,
);

#[cfg(feature = "audio")]
impl MlxAudioPcmTranscription {
    /// Forks the settled continuation on the same resident graph.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.0.fork().map(Self)
    }

    /// Configures bounded snapshot and restore.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.0.enable_snapshots(limits)
    }

    /// Reports retained snapshots and cumulative copy work.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.0.snapshot_usage()
    }

    /// Saves the current settled continuation.
    pub fn snapshot(&self) -> Result<MlxAudioPcmTranscriptionSnapshot, AudioSnapshotError> {
        self.0.snapshot().map(MlxAudioPcmTranscriptionSnapshot)
    }

    /// Restores a saved continuation from this session.
    pub fn restore(
        &mut self,
        saved: &MlxAudioPcmTranscriptionSnapshot,
    ) -> Result<(), AudioSnapshotError> {
        self.0.restore(&saved.0)
    }

    /// Reports sample capacity before the next block must be advanced.
    pub fn available_samples(&self) -> usize {
        self.0.available_samples()
    }

    /// Accepts finite mono 16-kHz PCM.
    pub fn submit(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(), AudioTranscriptionError> {
        self.0.submit(samples, sample_rate)
    }

    /// Flushes PCM preprocessing and closes input.
    pub fn finish_input(&mut self) -> Result<(), AudioTranscriptionError> {
        self.0.finish_input()
    }

    /// Advances at most one LiveASR block.
    pub fn advance(
        &mut self,
        restart: bool,
    ) -> Result<Option<AudioTranscriptionBlock>, AudioTranscriptionError> {
        let _scope = Recovery::begin(Vec::<safemlx::Array>::new()).map_err(failure)?;
        self.0.advance(restart)
    }

    /// Whether the transcript is finished or cancelled.
    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    /// Cancels the stream at the current block boundary.
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
}

/// Batch-one resident speech session exposing portable committed events.
pub struct MlxAudioSession(AudioSession<Model>);

/// Reusable saved Audio continuation, charged to the originating session.
pub struct MlxAudioSnapshot(AudioSnapshot<Model>);

impl MlxAudioSession {
    /// Configures bounded snapshots and restores for the session.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.0.enable_snapshots(limits)
    }

    /// Reports retention and cumulative copy work.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.0.snapshot_usage()
    }

    /// Captures the settled generation boundary, including pending depth work.
    pub fn snapshot(&self) -> Result<MlxAudioSnapshot, AudioSnapshotError> {
        self.0.snapshot().map(MlxAudioSnapshot)
    }

    /// Restores a boundary saved by this session.
    pub fn restore(&mut self, saved: &MlxAudioSnapshot) -> Result<(), AudioSnapshotError> {
        self.0.restore(&saved.0)
    }

    /// Forks the current settled continuation on the same resident graph.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.0.fork().map(Self)
    }

    /// Executes a single text decision or depth pass.
    pub fn advance(&mut self) -> Result<AudioAdvance, AudioSessionError> {
        self.0.advance()
    }

    /// Drains the same controlled driver, omitting intermediate depth progress.
    pub fn events(&mut self) -> impl Iterator<Item = Result<AudioAdvance, AudioSessionError>> + '_ {
        self.0.events()
    }

    /// Current phase of the session.
    pub fn phase(&self) -> AudioPhase {
        self.0.phase()
    }

    /// Number of committed language slots.
    pub fn committed_slots(&self) -> usize {
        self.0.committed_slots()
    }

    /// Cancels at the next idle boundary.
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
}
