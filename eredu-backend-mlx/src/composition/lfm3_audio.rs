//! MLX binding for the architecture-owned Audio driver.

use crate::backend::{
    nn::shared::MlxNeuralBackend,
    random::RandomState,
    runtime::{
        cache::state::MlxHybridState,
        execution::generic::{
            prepare_layerwise_policy_from_bindings, prepare_parameter_unit_policy,
            MlxSelectedLayerwisePolicy,
        },
        generation::MlxSamplingBackend,
    },
    submission_recovery::{Recovery, Retention, Status},
    ExecutionContext,
};
use crate::{MlxTensor, Stream};
use eredu_architectures::lfm3::audio::TensorParallelAudioLanguage;
#[cfg(feature = "audio")]
use eredu_architectures::lfm3::audio::{
    AudioIngress, AudioIngressError, AudioIngressProgress, AudioIngressRequest,
    AudioIngressSnapshot, AudioPcmTranscription, AudioPcmTranscriptionSnapshot,
};
use eredu_architectures::lfm3::{
    self,
    audio::{
        AudioAdvance, AudioParameterResidency, AudioPhase, AudioRequest, AudioSession,
        AudioSessionError, AudioSnapshot, AudioSnapshotError, AudioTranscription,
        AudioTranscriptionBlock, AudioTranscriptionError, AudioTranscriptionSnapshot, DepthModule,
        EncoderModule, LogMelFeatures, OmniAudioRequest, PolicyDepth, PolicyEncoder, PreparedAudio,
        PreparedAudioCodec, PreparedAudioDepth, PreparedAudioEncoder, PreparedAudioLanguage,
        PreparedResidentAudio, ResidentAudio, ResidentAudioBinding, ResidentAudioError,
    },
};
use eredu_codec::liquid::{DecoderModule, PolicyDecoder};
use eredu_core::{
    execution_control::{SnapshotLimits, SnapshotUsage},
    BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome, ParallelRankTopology,
    ParallelTopology,
};
use eredu_nn::Parameterized;
use eredu_runtime::{
    execution_control::SnapshotRetainedGrowth, EmptyParameterModule, ExecutionUnitAddressMap,
    LayerWeightResidency, LayerwiseRuntime, RuntimeLayerState, StateLayout, WeightBinding,
};
use std::marker::PhantomData;

type Language = LayerwiseRuntime<
    lfm3::LayeredModel<MlxNeuralBackend>,
    MlxNeuralBackend,
    MlxHybridState,
    MlxSelectedLayerwisePolicy<lfm3::Block<MlxNeuralBackend>, ()>,
>;

type Encoder = PolicyEncoder<
    MlxNeuralBackend,
    MlxSelectedLayerwisePolicy<EncoderModule<MlxNeuralBackend>, ()>,
>;

type Depth =
    PolicyDepth<MlxNeuralBackend, MlxSelectedLayerwisePolicy<DepthModule<MlxNeuralBackend>, ()>>;

type Codec = PolicyDecoder<
    MlxNeuralBackend,
    MlxSelectedLayerwisePolicy<DecoderModule<MlxNeuralBackend>, ()>,
>;

type TensorLanguage = TensorParallelAudioLanguage<
    MlxNeuralBackend,
    MlxHybridState,
    MlxSelectedLayerwisePolicy<lfm3::Block<MlxNeuralBackend>, ()>,
>;

trait LanguagePlacement {
    type Execution: eredu_architectures::lfm3::audio::AudioLanguage<
        MlxNeuralBackend,
        MlxHybridState,
    >;

    fn bind(
        &self,
        binding: &Binding<Self>,
        prepared: PreparedAudioLanguage,
    ) -> Result<Self::Execution, BackendFailure>
    where
        Self: Sized;
}

impl LanguagePlacement for () {
    type Execution = Language;

    fn bind(
        &self,
        binding: &Binding<Self>,
        prepared: PreparedAudioLanguage,
    ) -> Result<Language, BackendFailure> {
        binding.bind_unpartitioned_language(prepared, None)
    }
}

struct TensorPlacement {
    rank: ParallelRankTopology,
    group: safemlx::distributed::Group,
}

impl LanguagePlacement for TensorPlacement {
    type Execution = TensorLanguage;

    fn bind(
        &self,
        binding: &Binding<Self>,
        prepared: PreparedAudioLanguage,
    ) -> Result<TensorLanguage, BackendFailure> {
        let runtime = binding.bind_unpartitioned_language(prepared, Some(self.rank))?;
        TensorParallelAudioLanguage::new(
            runtime,
            crate::backend::runtime::distributed::Group::uncontracted(&self.group),
        )
    }
}

struct Binding<L = ()> {
    execution: Stream,
    weights: Stream,
    residency: AudioParameterResidency,
    placement: L,
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

impl<L> Binding<L> {
    fn bind_units<U: Parameterized<MlxTensor> + 'static>(
        &self,
        source: eredu_checkpoint::store::SharedCheckpointSource,
        invocations: ExecutionUnitAddressMap,
        units: Vec<Vec<WeightBinding>>,
        residency: LayerWeightResidency,
        resident_units: impl FnOnce() -> Result<Vec<U>, BackendFailure>,
    ) -> Result<MlxSelectedLayerwisePolicy<U, ()>, BackendFailure> {
        let (policy, _) = prepare_parameter_unit_policy(
            source,
            &mut EmptyParameterModule,
            (),
            residency,
            &self.execution,
            &self.weights,
            |_| false,
            invocations,
            Vec::new(),
            units,
            Vec::new(),
        )
        .map_err(failure)?;
        if residency.is_fully_resident() {
            Ok(MlxSelectedLayerwisePolicy::resident(
                policy
                    .into_resident_units(resident_units()?, &self.execution)
                    .map_err(failure)?,
            ))
        } else {
            Ok(MlxSelectedLayerwisePolicy::bounded(policy))
        }
    }
}

impl<L: LanguagePlacement> Binding<L> {
    fn bind_unpartitioned_language(
        &self,
        prepared: PreparedAudioLanguage,
        rank: Option<ParallelRankTopology>,
    ) -> Result<Language, BackendFailure> {
        crate::backend::submission_recovery::detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (mut architecture, parameters) = match rank {
                    Some(rank) => prepared
                        .into_tensor_parallel::<MlxNeuralBackend, MlxHybridState>(
                            rank,
                            self.context(),
                        )?,
                    None => prepared
                        .into_layerwise::<MlxNeuralBackend, MlxHybridState>(self.context())?,
                };
                let (_, invocations, source, units, pinned) = parameters.into_parts();
                let (policy, _) = prepare_layerwise_policy_from_bindings(
                    source,
                    &mut architecture,
                    (),
                    PhantomData::<MlxHybridState>,
                    self.residency.language,
                    self.context(),
                    &self.weights,
                    |_| false,
                    invocations,
                    pinned.into_values().flatten().collect(),
                    units,
                )
                .map_err(failure)?;
                let policy = if self.residency.language.is_fully_resident() {
                    MlxSelectedLayerwisePolicy::resident(
                        policy
                            .into_resident(
                                &architecture,
                                self.context(),
                                PhantomData::<MlxHybridState>,
                            )
                            .map_err(failure)?,
                    )
                } else {
                    MlxSelectedLayerwisePolicy::bounded(policy)
                };
                Ok(LayerwiseRuntime::new(architecture, policy))
            },
            failure,
        )
    }
}

impl<L: LanguagePlacement> ResidentAudioBinding<MlxNeuralBackend> for Binding<L> {
    type State = MlxHybridState;
    type Language = L::Execution;
    type Encoder = Encoder;
    type Depth = Depth;
    type Codec = Codec;

    fn bind_codec(&self, prepared: PreparedAudioCodec) -> Result<Codec, BackendFailure> {
        crate::backend::submission_recovery::detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (declaration, source, invocations, units) = prepared.into_parts();
                let policy = self.bind_units(
                    source,
                    invocations.clone(),
                    units,
                    self.residency.codec,
                    || Codec::resident_units(&declaration, self.context()).map_err(failure),
                )?;
                PolicyDecoder::new(declaration, invocations, policy, self.context())
                    .map_err(failure)
            },
            failure,
        )
    }

    fn bind_depth(&self, prepared: PreparedAudioDepth) -> Result<Depth, BackendFailure> {
        crate::backend::submission_recovery::detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (declaration, source, invocations, units) = prepared.into_parts();
                let policy = self.bind_units(
                    source,
                    invocations.clone(),
                    units,
                    self.residency.depth,
                    || Depth::resident_units(&declaration, self.context()).map_err(failure),
                )?;
                PolicyDepth::new(declaration, invocations, policy).map_err(failure)
            },
            failure,
        )
    }

    fn bind_encoder(&self, prepared: PreparedAudioEncoder) -> Result<Encoder, BackendFailure> {
        crate::backend::submission_recovery::detached_preparation(
            Vec::<safemlx::Array>::new(),
            || {
                let (declaration, source, invocations, units) = prepared.into_parts();
                let policy = self.bind_units(
                    source,
                    invocations.clone(),
                    units,
                    self.residency.encoder,
                    || Encoder::resident_units(&declaration, self.context()).map_err(failure),
                )?;
                PolicyEncoder::new(declaration, invocations, policy).map_err(failure)
            },
            failure,
        )
    }

    fn bind_language(
        &self,
        prepared: PreparedAudioLanguage,
    ) -> Result<L::Execution, BackendFailure> {
        self.placement.bind(self, prepared)
    }

    type SamplingError = safemlx::error::Exception;
    type Sampling = MlxSamplingBackend;

    fn context(&self) -> &Stream {
        &self.execution
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

/// Bound LFM3 Audio graph consuming the selected F32 binding recipes.
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
        Self::load_prepared_with_language_residency(
            prepared,
            LayerWeightResidency::FullyResident,
            weights,
            execution,
        )
    }

    /// Binds resident speech components and selects residency for the language blocks.
    /// The supplied budget covers the language policy, including its pinned statics.
    /// Resident speech parameters and session state require separate capacity.
    pub fn load_prepared_with_language_residency(
        prepared: PreparedResidentAudio,
        language_residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        Self::load_prepared_with_residency(
            prepared,
            language_residency,
            LayerWeightResidency::FullyResident,
            weights,
            execution,
        )
    }

    /// Selects independent language and speech-input parameter policies.
    /// Speech generation weights and session state require separate capacity.
    pub fn load_prepared_with_residency(
        prepared: PreparedResidentAudio,
        language_residency: LayerWeightResidency,
        encoder_residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        Self::load_prepared_with_parameter_residency(
            prepared,
            AudioParameterResidency {
                language: language_residency,
                encoder: encoder_residency,
                ..Default::default()
            },
            weights,
            execution,
        )
    }

    /// Applies independent residency budgets to each parameter policy in the Audio graph.
    /// Mutable session state and runtime scratch require separate capacity.
    pub fn load_prepared_with_parameter_residency(
        prepared: PreparedResidentAudio,
        residency: AudioParameterResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        Ok(Self(ResidentAudio::from_prepared(
            prepared,
            Binding {
                execution: execution.stream().clone(),
                weights: weights.stream().clone(),
                residency,
                placement: (),
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

    /// Prefills projected Omni visual rows and speech features into a controlled session.
    pub fn start_omni(
        self,
        request: OmniAudioRequest<'_, MlxTensor>,
        seed: u64,
    ) -> Result<MlxAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxAudioSession(self.0.start_omni(request, random)?))
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
    /// Forks a settled streamed prompt on the same selected graph.
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
    /// Forks the settled continuation on the same selected graph.
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
    /// Forks the settled continuation on the same selected graph.
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

/// Batch-one speech session exposing portable committed events.
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

    /// Forks the current settled continuation on the same selected graph.
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

type TensorModel = ResidentAudio<MlxNeuralBackend, Binding<TensorPlacement>>;

/// Rank-local Audio graph with a tensor-sharded language backbone and replicated speech components.
/// The caller supplies an established cohort and coordinates request ordering and failures.
pub struct MlxTensorParallelAudioModel(TensorModel);

impl MlxTensorParallelAudioModel {
    /// Binds this rank from the retained Audio checkpoint.
    pub fn load(
        prepared: PreparedResidentAudio,
        residency: AudioParameterResidency,
        group: safemlx::distributed::Group,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        let topology = ParallelTopology::new(group.size(), 1, 1, 1).map_err(failure)?;
        let rank = ParallelRankTopology::new(topology, group.rank()).map_err(failure)?;
        Ok(Self(ResidentAudio::from_prepared(
            prepared,
            Binding {
                execution: execution.stream().clone(),
                weights: weights.stream().clone(),
                residency,
                placement: TensorPlacement { rank, group },
            },
        )?))
    }

    /// Prefills the same request and seed on every rank in the cohort.
    pub fn start(
        self,
        request: AudioRequest<'_>,
        seed: u64,
    ) -> Result<MlxTensorParallelAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxTensorParallelAudioSession(
            self.0.start(request, random)?,
        ))
    }
}

/// Controlled rank-local continuation. All ranks advance the same boundary together.
pub struct MlxTensorParallelAudioSession(AudioSession<TensorModel>);

impl MlxTensorParallelAudioSession {
    /// Advances one text decision or depth pass in cohort order.
    pub fn advance(&mut self) -> Result<AudioAdvance, AudioSessionError> {
        self.0.advance()
    }

    /// Drains committed output from the shared controlled driver.
    pub fn events(&mut self) -> impl Iterator<Item = Result<AudioAdvance, AudioSessionError>> + '_ {
        self.0.events()
    }
}

type AudioPipeline = crate::backend::runtime::execution::pipeline::MlxComponentPipeline<
    lfm3::PartitionedLayeredModel<MlxNeuralBackend>,
    MlxHybridState,
>;

struct PipelinePlacement {
    rank: ParallelRankTopology,
    world: safemlx::distributed::Group,
}

struct PipelineLanguage {
    runtime: AudioPipeline,
    layout: StateLayout,
    output: bool,
    vocabulary: i32,
    epoch: DistributedCommitEpoch,
    stream: Stream,
}

impl PipelineLanguage {
    fn next_epoch(&mut self) -> Result<DistributedCommitEpoch, BackendFailure> {
        let epoch = self.epoch;
        self.epoch = epoch.next().ok_or_else(|| {
            failure(eredu_nn::Error::backend(
                "Audio distributed commit epoch exhausted",
            ))
        })?;
        Ok(epoch)
    }
}

impl LanguagePlacement for PipelinePlacement {
    type Execution = PipelineLanguage;

    fn bind(
        &self,
        binding: &Binding<Self>,
        prepared: PreparedAudioLanguage,
    ) -> Result<PipelineLanguage, BackendFailure> {
        let prepared = prepared.select_pipeline_rank(self.rank)?;
        let communication = prepared.pipeline_communication()?;
        let layout = prepared
            .partition()
            .state()
            .ok_or_else(|| {
                failure(eredu_nn::Error::backend(
                    "Audio pipeline stage has no cache layout",
                ))
            })?
            .layout()
            .clone();
        let output = prepared.partition().ownership().owns_output();
        let (architecture, source, invocations, pinned, units) =
            prepared.into_layerwise::<MlxNeuralBackend, MlxHybridState>(&binding.execution)?;
        let vocabulary = architecture.kernel().args().vocab_size;
        let addressed = units
            .into_iter()
            .enumerate()
            .map(|(i, unit)| {
                (
                    invocations
                        .storage_layout()
                        .address(i)
                        .expect("retained Audio owner"),
                    unit,
                )
            })
            .collect();
        let (runtime, _) = AudioPipeline::with_pinned(
            architecture,
            source,
            invocations,
            pinned,
            addressed,
            communication.manifest,
            communication.plan,
            communication.tensor_group,
            eredu_runtime::PipelineActivationDtype::Float32,
            &self.world,
            binding.residency.language,
            &binding.weights,
            &binding.execution,
        )?;
        Ok(PipelineLanguage {
            runtime,
            layout,
            output,
            vocabulary,
            epoch: DistributedCommitEpoch::FIRST,
            stream: binding.execution.clone(),
        })
    }
}

impl eredu_architectures::lfm3::audio::AudioLanguage<MlxNeuralBackend, MlxHybridState>
    for PipelineLanguage
{
    fn state_layout(&self) -> &StateLayout {
        &self.layout
    }

    fn forward_prompt(
        &mut self,
        prompt: eredu_architectures::lfm3::audio::SpeechPrompt<'_, MlxTensor>,
        state: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<MlxTensor, BackendFailure> {
        let descriptor = prompt.collective_descriptor(stream).map_err(failure);
        let agreed = self.runtime.agree_descriptor(
            [0x415544494f505245, 0, 0, 0, 0, 0, 0, 0],
            descriptor.is_ok(),
            stream,
        );
        let descriptor = descriptor?;
        agreed?;
        let epoch = self.next_epoch()?;
        let (hidden, outcome) = self.runtime.forward(
            prompt.input(),
            state,
            descriptor,
            epoch,
            stream,
            prompt
                .retained_tensors()
                .map(|value| value.as_array().clone())
                .collect(),
            |state| {
                state
                    .layers()
                    .iter()
                    .flat_map(RuntimeLayerState::retained_values)
                    .map(|value| value.as_array().clone())
                    .collect()
            },
            |output| Ok((output.clone(), Vec::new())),
        )?;
        if outcome != DistributedCommitOutcome::Committed(epoch) {
            return Err(failure(eredu_nn::Error::backend(
                "Audio language pipeline did not settle",
            )));
        }
        Ok(hidden)
    }

    fn project_normalized(
        &mut self,
        hidden: &MlxTensor,
        stream: &Stream,
    ) -> Result<MlxTensor, BackendFailure> {
        use eredu_nn::Tensor;
        self.runtime.readout(
            stream,
            vec![hidden.as_array().clone()],
            |architecture, parallel| {
                if self.output {
                    architecture
                        .kernel_mut()
                        .project_audio_position(hidden, parallel, stream)
                        .map_err(failure)
                } else {
                    MlxTensor::full_f32(0.0, &[hidden.dim(0), self.vocabulary], stream)
                        .map_err(failure)
                }
            },
        )
    }

    fn uses_collective_commit(&self) -> bool {
        true
    }

    fn agree_audio(&mut self, descriptor: [u64; 8], success: bool) -> Result<(), BackendFailure> {
        self.runtime
            .agree_descriptor(descriptor, success, &self.stream)
    }

    fn commit_audio(&mut self) -> Result<(), BackendFailure> {
        let epoch = self.next_epoch()?;
        self.runtime.commit(epoch, &self.stream)
    }
}

/// One rank of an Audio session using the shared partition and commit protocol.
pub struct MlxPipelineAudioModel(ResidentAudio<MlxNeuralBackend, Binding<PipelinePlacement>>);

impl MlxPipelineAudioModel {
    /// Binds retained weights for a tensor, pipeline, or combined parallel rank.
    ///
    /// A single pipeline stage uses the same collective publication protocol.
    pub fn load(
        prepared: PreparedResidentAudio,
        residency: AudioParameterResidency,
        topology: ParallelTopology,
        world: safemlx::distributed::Group,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, ResidentAudioError> {
        if topology.world_size() < 2 || topology.world_size() != world.size() {
            return Err(failure(eredu_nn::Error::backend(
                "Audio partition topology requires a matching multi-rank native world",
            ))
            .into());
        }
        let rank = ParallelRankTopology::new(topology, world.rank()).map_err(failure)?;
        Ok(Self(ResidentAudio::from_prepared(
            prepared,
            Binding {
                execution: execution.stream().clone(),
                weights: weights.stream().clone(),
                residency,
                placement: PipelinePlacement { rank, world },
            },
        )?))
    }

    /// Starts the same request and seed on every rank.
    pub fn start(
        self,
        request: AudioRequest<'_>,
        seed: u64,
    ) -> Result<MlxPipelineAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxPipelineAudioSession(self.0.start(request, random)?))
    }

    /// Prefills the same projected Omni request on every rank in the pipeline cohort.
    ///
    /// Each rank supplies the same tokens, features, projected rows, and seed.
    /// Advance the returned sessions in the same order across ranks.
    pub fn start_omni(
        self,
        request: OmniAudioRequest<'_, MlxTensor>,
        seed: u64,
    ) -> Result<MlxPipelineAudioSession, ResidentAudioError> {
        let random = RandomState::with_seed(seed).map_err(failure)?;
        Ok(MlxPipelineAudioSession(self.0.start_omni(request, random)?))
    }
}

/// Controlled rank-local continuation with all-rank publication agreement.
pub struct MlxPipelineAudioSession(
    AudioSession<ResidentAudio<MlxNeuralBackend, Binding<PipelinePlacement>>>,
);

impl MlxPipelineAudioSession {
    /// Advances one shared text decision or depth pass.
    pub fn advance(&mut self) -> Result<AudioAdvance, AudioSessionError> {
        self.0.advance()
    }

    /// Number of language slots committed by the all-rank transaction.
    pub fn committed_slots(&self) -> usize {
        self.0.committed_slots()
    }

    /// Drains the same controlled driver.
    pub fn events(&mut self) -> impl Iterator<Item = Result<AudioAdvance, AudioSessionError>> + '_ {
        self.0.events()
    }
}
