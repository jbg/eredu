//! Coupled VoiceChat state over one caller-owned neural transaction.
//!
//! Backbone and TTS caches share a single transaction owner. Their mutable
//! storage is branched by that owner, never by cloning native cache handles.
//! Input, speech and retained audio use the component APIs' immutable tensor
//! predecessors; component execution replaces them with candidate successors.

use eredu_codec::ear_decoder::DecoderState;
use eredu_core::scheduler::SemanticStateTransaction;
use eredu_core::{BackendFailure, execution_control::SnapshotEstimate};
use eredu_nn::{RetainedTensorRoots, Tensor};
use eredu_runtime::StateLayout;
use eredu_runtime::execution_control::{
    ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork, SnapshotReservation, SnapshotValue,
    snapshot_value_estimate,
};

use super::{
    channel_decision::FeedbackTokens, input::InputState, perception::PerceptionState,
    speech::SpeechState,
};

/// Host-only feedback correction on an unpublished conversational candidate.
/// The enclosing semantic owner applies this only when an interruption commits;
/// neural caches, speech-code carry, RNG and component positions are retained.
pub trait InterruptionFeedback {
    /// Replaces the next backbone feedback with the released stopped-agent and
    /// quiet-function controls without appending a model position.
    fn reset_interruption_feedback(&mut self);
}

impl<T: Tensor, N, C, K> InterruptionFeedback for StateBranch<T, N, C, K> {
    fn reset_interruption_feedback(&mut self) {
        self.feedback = FeedbackTokens::new(
            super::channels::TEXT_EOS_ID as u32,
            super::channels::TEXT_PAD_ID as u32,
        )
        .expect("released interruption controls are valid");
    }
}

impl<M, Q, P, R, C> InterruptionFeedback
    for eredu_runtime::RealtimeGenerationBranch<M, Q, P, super::agent_sampling::AgentSampler, R, C>
where
    M: InterruptionFeedback,
    Q: eredu_runtime::RealtimeScheduleState,
{
    fn reset_interruption_feedback(&mut self) {
        self.model_state_mut().reset_interruption_feedback();
        if let Some(sampler) = self.samplers_mut().first_mut() {
            sampler.reset_last_effective_for_interruption();
        }
    }
}

/// Bounded projected features awaiting their first backbone invocation.
#[derive(Debug, thiserror::Error)]
pub enum PendingAudioError {
    /// Keeping another frame would exceed the admitted source-audio bound.
    #[error("VoiceChat pending audio requires {required} frames, capacity is {capacity}")]
    Capacity {
        /// Number of frames after the requested append.
        required: u64,
        /// Admitted maximum number of source frames.
        capacity: u64,
    },
    /// Host storage for an immutable projected-frame handle could not be reserved.
    #[error("cannot reserve VoiceChat pending audio: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    /// No projected frame remains to replay.
    #[error("VoiceChat has no pending projected audio")]
    Empty,
}

/// Invalid cold geometry for the two coupled neural cache families.
#[derive(Debug, thiserror::Error)]
pub enum NeuralStateGeometryError {
    /// The retained Nemotron configuration cannot describe its mutable state.
    #[error("invalid VoiceChat backbone state: {0}")]
    Backbone(#[source] crate::nemotron_h::ConfigError),
    /// The retained causal TTS configuration cannot describe its attention state.
    #[error("invalid VoiceChat TTS state: {0}")]
    Tts(#[source] eredu_nn::Error),
    /// The bidirectional character encoder has no persistent causal cache.
    #[error("VoiceChat persistent TTS state requires the causal generator configuration")]
    NonCausalTts,
    /// Selected Nemotron state differs from the retained architecture.
    #[error("selected VoiceChat backbone state layout differs from retained geometry")]
    BackboneLayout,
    /// Selected TTS state differs from the retained architecture.
    #[error("selected VoiceChat TTS state layout differs from retained geometry")]
    TtsLayout,
}

/// Architecture-owned neural state geometry, independent of storage policy.
///
/// Nemotron processes one conversation lane. TTS classifier-free guidance uses
/// two batch rows in each cache, preserving conditioned and unconditioned state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NeuralStateGeometry {
    backbone: StateLayout,
    tts: StateLayout,
}

impl NeuralStateGeometry {
    /// Derives state through the existing Nemotron and shared decoder contracts.
    pub fn new(
        backbone: &crate::nemotron_h::ModelArgs,
        tts: &super::config::GemmaConfig,
    ) -> Result<Self, NeuralStateGeometryError> {
        if !tts.causal {
            return Err(NeuralStateGeometryError::NonCausalTts);
        }
        let tts = super::tts_transformer::TransformerConfig::new(tts)
            .map_err(NeuralStateGeometryError::Tts)?;
        Ok(Self {
            backbone: crate::nemotron_h::state_layout(backbone)
                .map_err(NeuralStateGeometryError::Backbone)?,
            tts: crate::decoder::state_layout(&tts).map_err(NeuralStateGeometryError::Tts)?,
        })
    }

    /// Complete Nemotron attention, convolution and recurrent state geometry.
    pub fn backbone_layout(&self) -> &StateLayout {
        &self.backbone
    }

    /// Complete causal TTS attention geometry, including each sliding window.
    pub fn tts_layout(&self) -> &StateLayout {
        &self.tts
    }

    /// Number of conversation rows in each Nemotron cache tensor.
    pub const fn backbone_batch_size(&self) -> usize {
        1
    }

    /// Conditioned and unconditioned rows in each classifier-free guidance cache.
    pub const fn tts_batch_size(&self) -> usize {
        2
    }

    /// Rejects mismatched selected layouts before any native cache allocation.
    pub fn validate_layouts(
        &self,
        backbone: &StateLayout,
        tts: &StateLayout,
    ) -> Result<(), NeuralStateGeometryError> {
        if backbone != &self.backbone {
            return Err(NeuralStateGeometryError::BackboneLayout);
        }
        if tts != &self.tts {
            return Err(NeuralStateGeometryError::TtsLayout);
        }
        Ok(())
    }
}

/// Borrowed access to both neural cache families in one unpublished branch.
/// The implementation owns their joint branch, commit and rollback semantics.
pub trait NeuralStateParts {
    /// Existing Nemotron layered runtime state.
    type Backbone;
    /// Selected causal TTS state, owned by this same unpublished transaction.
    type TtsState: ?Sized;

    /// Borrows disjoint cache families without transferring their transaction owner.
    fn parts_mut(&mut self) -> (&mut Self::Backbone, &mut Self::TtsState);
}

/// Canonical component state retained by the shared realtime generation owner.
/// Native completion is established by that owner before commit or discard.
pub struct State<T: Tensor, N, C = PerceptionState<T>, K = DecoderState<T>> {
    neural: N,
    input: InputState<T, C>,
    speech: Option<SpeechState<T, K>>,
    retained_audio: Option<T>,
    pending_audio: Vec<T>,
    pending_audio_replaying: bool,
    feedback: FeedbackTokens,
}

/// Unpublished component candidates retained together until exact completion.
pub struct StateBranch<T: Tensor, N, C = PerceptionState<T>, K = DecoderState<T>> {
    started: bool,
    /// One joint owner of backbone and TTS cache candidates.
    pub neural: N,
    /// Host PCM carry, perception caches and RNNT hypotheses.
    pub input: InputState<T, C>,
    /// Speech feedback and the selected local codec carry; absent before prefill.
    pub speech: Option<SpeechState<T, K>>,
    /// Most recently consumed projected perception frame.
    pub retained_audio: Option<T>,
    /// Ordered perception frames not yet consumed by the language backbone.
    pending_audio: Vec<T>,
    pending_audio_replaying: bool,
    /// Effective previous agent and function tokens.
    pub feedback: FeedbackTokens,
}

impl<T: Tensor, N, C, K> StateBranch<T, N, C, K> {
    /// Borrows all component roots, including carry independent of visible output.
    /// The native submission owner must additionally retain and complete the
    /// joint `neural` cache roots; this traversal deliberately excludes them.
    pub fn visit_component_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T))
    where
        C: RetainedTensorRoots<T>,
        K: RetainedTensorRoots<T>,
    {
        self.input.visit_retained_tensors(visitor);
        if let Some(speech) = &self.speech {
            speech.visit_retained_tensors(visitor);
        }
        if let Some(audio) = &self.retained_audio {
            visitor(audio);
        }
        for audio in &self.pending_audio {
            visitor(audio);
        }
    }

    /// Number of perception frames awaiting their first backbone invocation.
    pub fn pending_audio_frames(&self) -> usize {
        self.pending_audio.len()
    }

    /// Whether this retained batch has already consumed its interruption feedback.
    pub fn pending_audio_replaying(&self) -> bool {
        self.pending_audio_replaying
    }

    /// Retains one immutable projected frame without admitting unbounded carry.
    /// The caller supplies the same maximum source-frame count as the schedule.
    pub fn retain_pending_audio(
        &mut self,
        audio: T,
        capacity: u64,
    ) -> Result<(), PendingAudioError> {
        let required = (self.pending_audio.len() as u64).saturating_add(1);
        if required > capacity {
            return Err(PendingAudioError::Capacity { required, capacity });
        }
        self.pending_audio
            .try_reserve_exact(1)
            .map_err(PendingAudioError::Allocation)?;
        self.pending_audio.push(audio);
        self.pending_audio_replaying = false;
        Ok(())
    }

    /// Marks this candidate as replaying the retained batch.
    /// Returns `true` only for its first replay, when interruption feedback must
    /// replace the previous tool-channel feedback before the backbone advances.
    pub fn begin_pending_audio_replay(&mut self) -> bool {
        if self.pending_audio_replaying || self.pending_audio.is_empty() {
            return false;
        }
        self.pending_audio_replaying = true;
        true
    }

    /// Takes the oldest unheard frame on this unpublished candidate only.
    /// Discarding the branch leaves the canonical queue unchanged.
    pub fn replay_pending_audio(&mut self) -> Result<T, PendingAudioError> {
        if self.pending_audio.is_empty() {
            return Err(PendingAudioError::Empty);
        }
        Ok(self.pending_audio.remove(0))
    }

    /// Releases perception carry after a normal return from tool work.
    pub fn clear_pending_audio(&mut self) {
        self.pending_audio = Vec::new();
        self.pending_audio_replaying = false;
    }

    /// Claims this candidate for exactly one coupled neural invocation.
    /// Call after preflight; a failed invocation retains partially advanced
    /// caches and must be discarded through its completion owner, never retried.
    /// Returns `false` without changing state when execution already started.
    pub fn begin_execution(&mut self) -> bool {
        if self.started {
            return false;
        }
        self.started = true;
        true
    }
}

impl<T: Tensor, N, C, K> State<T, N, C, K> {
    /// Borrows all retained component roots, excluding the separately owned
    /// joint neural caches. Traversal never evaluates or synchronizes tensors.
    pub fn visit_component_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T))
    where
        C: RetainedTensorRoots<T>,
        K: RetainedTensorRoots<T>,
    {
        self.input.visit_retained_tensors(visitor);
        if let Some(speech) = &self.speech {
            speech.visit_retained_tensors(visitor);
        }
        if let Some(audio) = &self.retained_audio {
            visitor(audio);
        }
        for audio in &self.pending_audio {
            visitor(audio);
        }
    }

    /// Creates cold component state; startup speech preparation is transactional.
    /// The caller supplies admitted initial feedback IDs and empty neural caches.
    pub fn new(input: InputState<T, C>, neural: N, feedback: FeedbackTokens) -> Self {
        Self {
            neural,
            input,
            speech: None,
            retained_audio: None,
            pending_audio: Vec::new(),
            pending_audio_replaying: false,
            feedback,
        }
    }

    /// Joint canonical neural owner, available for read-only accounting.
    pub fn neural(&self) -> &N {
        &self.neural
    }

    /// Canonical input carry and neural input caches.
    pub fn input(&self) -> &InputState<T, C> {
        &self.input
    }

    /// Admits or finishes host PCM when the enclosing session allows mutation.
    /// The session serializes this access with outstanding component branches.
    pub fn input_mut(&mut self) -> &mut InputState<T, C> {
        &mut self.input
    }

    /// Canonical speech feedback and codec carry.
    pub fn speech(&self) -> Option<&SpeechState<T, K>> {
        self.speech.as_ref()
    }

    /// Most recently committed projected perception frame.
    pub fn retained_audio(&self) -> Option<&T> {
        self.retained_audio.as_ref()
    }

    /// Number of perception frames awaiting their first backbone invocation.
    pub fn pending_audio_frames(&self) -> usize {
        self.pending_audio.len()
    }

    /// Effective previous channel tokens at the committed boundary.
    pub fn feedback(&self) -> FeedbackTokens {
        self.feedback
    }
}

impl<T: Tensor, N: SemanticStateTransaction, C: Clone, K: Clone> SemanticStateTransaction
    for State<T, N, C, K>
{
    type Branch = StateBranch<T, N::Branch, C, K>;
    type Error = N::Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(StateBranch {
            started: false,
            neural: self.neural.branch()?,
            input: self.input.clone(),
            speech: self.speech.clone(),
            retained_audio: self.retained_audio.clone(),
            pending_audio: self.pending_audio.clone(),
            pending_audio_replaying: self.pending_audio_replaying,
            feedback: self.feedback,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        // There is exactly one fallible commit. Its owner validates and adopts
        // both neural cache families atomically; only then can host/immutable
        // component candidates replace the canonical boundary.
        self.neural.commit_branch(branch.neural)?;
        self.input = branch.input;
        self.speech = branch.speech;
        self.retained_audio = branch.retained_audio;
        self.pending_audio = branch.pending_audio;
        self.pending_audio_replaying = branch.pending_audio_replaying;
        self.feedback = branch.feedback;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        N::discard_branch(branch.neural)
    }

    fn permits_parallel_branches(&self) -> bool {
        self.neural.permits_parallel_branches()
    }
}

impl<
    T: Tensor + SnapshotValue,
    N: IsolatedStateCopy,
    C: Clone + SnapshotValue,
    K: Clone + SnapshotValue,
> IsolatedStateCopy for State<T, N, C, K>
{
    type Error = BackendFailure;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
        snapshot_value_estimate::<Self>([
            self.neural
                .estimate_isolated_copy()
                .map_err(BackendFailure::from_error),
            self.input.estimate_snapshot_value(),
            self.speech.estimate_snapshot_value(),
            self.retained_audio.estimate_snapshot_value(),
            self.pending_audio.estimate_snapshot_value(),
        ])
    }
    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_components(retention, |neural| neural.copy_isolated(retention))
    }
    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
        self.neural
            .validate_isolated_restore(&saved.neural)
            .map_err(BackendFailure::from_error)
    }
    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
        self.validate_isolated_restore(&saved)?;
        self.neural
            .restore_isolated(saved.neural)
            .map_err(BackendFailure::from_error)?;
        self.input = saved.input;
        self.speech = saved.speech;
        self.retained_audio = saved.retained_audio;
        self.pending_audio = saved.pending_audio;
        self.pending_audio_replaying = saved.pending_audio_replaying;
        self.feedback = saved.feedback;
        Ok(())
    }
}

impl<
    T: Tensor + SnapshotValue,
    N: IsolatedStateCopy,
    C: Clone + SnapshotValue,
    K: Clone + SnapshotValue,
> State<T, N, C, K>
{
    fn copy_components(
        &self,
        retention: &SnapshotReservation,
        copy_neural: impl FnOnce(&N) -> Result<N, N::Error>,
    ) -> Result<Self, BackendFailure> {
        // All component copies complete before their result can be installed.
        // Neural copying retains the same lease on any unresolved failure.
        let input = self.input.copy_snapshot_value(retention)?;
        let speech = self.speech.copy_snapshot_value(retention)?;
        let retained_audio = self.retained_audio.copy_snapshot_value(retention)?;
        let pending_audio = self.pending_audio.copy_snapshot_value(retention)?;
        let neural = copy_neural(&self.neural).map_err(BackendFailure::from_error)?;
        Ok(Self {
            neural,
            input,
            speech,
            retained_audio,
            pending_audio,
            pending_audio_replaying: self.pending_audio_replaying,
            feedback: self.feedback,
        })
    }
}

impl<
    T: Tensor + SnapshotValue,
    N: IsolatedStateFork,
    C: Clone + SnapshotValue,
    K: Clone + SnapshotValue,
> IsolatedStateFork for State<T, N, C, K>
{
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
        self.copy_components(retention, |neural| neural.fork_isolated(retention))
    }
}

impl<
    T: Tensor + SnapshotValue,
    N: ForkStateEstimate,
    C: Clone + SnapshotValue,
    K: Clone + SnapshotValue,
> ForkStateEstimate for State<T, N, C, K>
{
    fn estimate_fork_branch(branch: &Self::Branch) -> Result<u64, BackendFailure> {
        let neural = N::estimate_fork_branch(&branch.neural)?;
        Ok(snapshot_value_estimate::<Self>([
            Ok(SnapshotEstimate {
                retained_bytes: neural,
                copy_bytes: 0,
            }),
            branch.input.estimate_snapshot_value(),
            branch.speech.estimate_snapshot_value(),
            branch.retained_audio.estimate_snapshot_value(),
            branch.pending_audio.estimate_snapshot_value(),
        ])?
        .retained_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{AttentionPolicy, cache::LayerCachePolicy};

    #[test]
    fn voicechat_neural_geometry_preserves_guidance_lanes_and_attention_windows() {
        // Compact hybrid backbone exercises recurrent, attention and stateless
        // units; TTS uses the pinned released configuration without resizing.
        let backbone = crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":3,
            "hybrid_override_pattern":"M*-", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3
        }))
        .unwrap();
        let config = super::super::config::VoiceChatConfig::from_json(include_str!(
            "fixtures/official-inference-config.json"
        ))
        .unwrap();
        let geometry = NeuralStateGeometry::new(&backbone, &config.tts.backbone).unwrap();
        assert_eq!(geometry.backbone_batch_size(), 1);
        assert_eq!(geometry.tts_batch_size(), 2);
        assert_eq!(geometry.backbone_layout().len(), 3);
        assert!(matches!(
            geometry.backbone_layout().layer(0),
            Some(LayerCachePolicy::FixedState { .. })
        ));
        assert!(matches!(
            geometry.backbone_layout().layer(1),
            Some(LayerCachePolicy::KeyValue { .. })
        ));
        assert_eq!(
            geometry.backbone_layout().layer(2),
            Some(&LayerCachePolicy::NoState)
        );
        assert_eq!(geometry.tts_layout().len(), 28);
        for (index, layer) in geometry.tts_layout().layers().iter().enumerate() {
            let expected_attention = if [5, 11, 17, 23].contains(&index) {
                AttentionPolicy::Full
            } else {
                AttentionPolicy::sliding(7500).unwrap()
            };
            assert_eq!(
                layer,
                &LayerCachePolicy::KeyValue {
                    attention: expected_attention,
                    num_key_value_heads: std::num::NonZeroU32::new(16).unwrap(),
                    head_dim: std::num::NonZeroU32::new(72).unwrap(),
                }
            );
        }
        geometry
            .validate_layouts(geometry.backbone_layout(), geometry.tts_layout())
            .unwrap();
        assert!(matches!(
            geometry.validate_layouts(geometry.tts_layout(), geometry.tts_layout()),
            Err(NeuralStateGeometryError::BackboneLayout)
        ));
        assert!(matches!(
            geometry.validate_layouts(geometry.backbone_layout(), geometry.backbone_layout()),
            Err(NeuralStateGeometryError::TtsLayout)
        ));
        assert!(matches!(
            NeuralStateGeometry::new(&backbone, &config.tts.character_encoder),
            Err(NeuralStateGeometryError::NonCausalTts)
        ));
    }
}
