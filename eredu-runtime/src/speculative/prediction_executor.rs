//! Typed prediction equations and state lifecycle contracts.

use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    Tensor,
};

use super::{PredictionExtensionMaterializer, PredictionModuleVisitor, PredictionOperationInvoker};

mod snapshot;
mod state_lifecycle;

pub use state_lifecycle::{
    ModelPredictionStateProfile, PoolingPredictionStateProfile, PredictionStateLifecycle,
    SequentialPredictionStateProfile,
};

/// Checks materialized capture shapes against the retained realization and frontier.
pub fn validate_selected_capture_shapes(
    selected: &crate::SelectedSpeculativeRealization,
    lane: &crate::SpeculativeLaneIdentity,
    shapes: Vec<Vec<usize>>,
) -> Result<(), crate::SpeculativeCaptureError> {
    let actual = selected.requirements().capture().instantiate(shapes)?;
    let values = vec![(); actual.entries().len()];
    let envelope = crate::SpeculativeCaptureEnvelope::new(
        crate::SpeculativeCaptureMetadata::new(actual, lane.capture_generation()),
        values,
    )?;
    selected.validate_capture(lane, &envelope)
}

/// Extension execution supported by the ordinary target.
pub trait PredictionTarget<B>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Exact prediction extension compatible with this target.
    type Extension<M>: MaterializedPredictionExecutor<Self, B, M>
    where
        M: PredictionExtensionMaterializer<B>,
        Self: Sized;
}

/// Declares the prediction equations paired with a family kernel's target.
pub trait PredictionKernel<B, A>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
{
    /// Extension implemented for this kernel's typed target.
    type Extension<M>: MaterializedPredictionExecutor<A, B, M>
    where
        M: PredictionExtensionMaterializer<B>;
}

impl<B, D, K, R> PredictionTarget<B> for crate::ComponentAdapter<D, K, R>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    K: PredictionKernel<B, Self>,
{
    type Extension<M>
        = K::Extension<M>
    where
        M: PredictionExtensionMaterializer<B>;
}

/// Architecture-owned executable strategy for one exactly paired materialized
/// prediction extension.
///
/// Implementations own proposal depth, lane-state geometry, prefill/advance
/// ordering, sequential versus fused execution, and every family-specific
/// operation. A backend supplies only typed operation invocation and storage.
pub trait MaterializedPredictionExecutor<A, B, M>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
{
    /// Exact mutable state retained for one speculative lane.
    type LaneState: Clone + 'static;
    /// Static state-membership and lifecycle declaration for this lane.
    type StateProfile: PredictionStateLifecycle<B, M, State = Self::LaneState>;

    /// Completes every retained lane value plus the actual phase outputs. The
    /// architecture supplies state membership; the backend settles native work.
    fn complete_state(
        &self,
        state: &mut Self::LaneState,
        outputs: &[&B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        Self::StateProfile::complete_state(state, outputs, context)
    }

    /// Visits every materialized prediction parameter owner exactly once in
    /// stable ordinal order. Shared target parameters remain with the target.
    fn visit_modules<V: PredictionModuleVisitor<B, M>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error>;

    /// Observes architecture-owned state membership through native mechanisms.
    fn memory_observation(
        &self,
        state: &Self::LaneState,
        additional: u64,
    ) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
        Self::StateProfile::memory_observation(state, additional)
    }

    /// Complete prediction-lane copy bound with architecture-owned membership.
    fn snapshot_estimate(
        &self,
        state: &Self::LaneState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::StateProfile::snapshot_estimate(state)
    }
    /// Copies every prediction state member into durable independent storage.
    fn snapshot(
        &self,
        state: &Self::LaneState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self::LaneState>, eredu_core::BackendFailure> {
        Self::StateProfile::snapshot(state, context)
    }

    /// Returns the proposal depth exposed to the scheduler.
    fn depth(&self) -> usize;

    /// Whether this exact extension implements internal prefill, proposal and
    /// replay hooks. This does not establish public collector or transport support.
    fn supports_internal_observations(&self) -> bool {
        false
    }

    /// Resolves complete phase hooks against the retained strategy. Backend
    /// binding forwards this fact without interpreting a family or phase name.
    fn activation_execution(
        &self,
        selected: &crate::SelectedSpeculativeRealization,
    ) -> Option<super::SpeculativeActivationExecution> {
        super::SpeculativeActivationExecution::selected(
            self.supports_internal_observations(),
            self.depth(),
            selected,
        )
    }

    /// Maximum target prefill span for a sequential predictor that supports causal
    /// streaming of hidden rows paired with the following original token. The shared
    /// driver preserves that pair across chunk boundaries. No declaration keeps the
    /// complete-input prefill contract, including for fused context builders.
    /// Declaring extensions also apply the selected capture's architecture-owned bounds.
    fn maximum_prefill_chunk_tokens(
        &self,
        _selected: &crate::SelectedSpeculativeRealization,
    ) -> Option<usize> {
        None
    }

    /// Physical rows consumed while seeding prediction state. Sequential
    /// prediction pairs each target hidden row with the following token; fused
    /// context builders may instead consume the complete accepted prefix.
    fn prefill_sequence_len(&self, target_sequence: usize) -> usize {
        crate::prediction_resources::PredictionExecutionMode::Sequential
            .prefill_sequence_len(target_sequence)
    }

    /// Clones the architecture-materialized lane-state prototype.
    fn new_state(&self) -> Self::LaneState;

    /// Maps one physical target capture into the ordered logical entry shapes
    /// declared by the selected architecture contract.
    fn logical_capture_shapes(
        &self,
        physical_shape: &[i32],
    ) -> Result<Vec<Vec<usize>>, crate::SpeculativeCaptureError> {
        let shape = physical_shape
            .iter()
            .copied()
            .map(|extent| {
                usize::try_from(extent)
                    .ok()
                    .filter(|extent| *extent > 0)
                    .ok_or(crate::SpeculativeCaptureError::ShapeMismatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![shape])
    }

    /// Validates a physical target capture against the exact selected lane
    /// before any extension state can be mutated.
    fn validate_capture(
        &self,
        selected: &crate::SelectedSpeculativeRealization,
        lane: &crate::SpeculativeLaneIdentity,
        physical_shape: &[i32],
    ) -> Result<(), crate::SpeculativeCaptureError> {
        validate_selected_capture_shapes(
            selected,
            lane,
            self.logical_capture_shapes(physical_shape)?,
        )
    }

    /// Advances extension state across a target-prefill observation.
    fn prefill<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, target_capture, hidden, tokens, lane, None)
    }

    /// Executes exactly one sequential proposal depth.
    fn logits<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.logits_observed::<S, I>(invoker, hidden, token, draft_index, lane, None)
    }

    /// Required prefill equation path, with optional internal hooks. Ordinary
    /// calls use the same path without creating diagnostic tensors.
    #[allow(clippy::too_many_arguments)]
    fn prefill_observed<S, I>(
        &mut self,
        invoker: &mut I,
        target_capture: &B::Tensor,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>;

    /// Required proposal equation path, with optional admitted internal hooks.
    /// Runtime support reports retain the exact selected observation capability.
    #[allow(clippy::too_many_arguments)]
    fn logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        token: &B::Tensor,
        draft_index: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(B::Tensor, B::Tensor), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>;

    /// Executes a fused proposal when selected by the architecture.
    fn fused_logits<S, I>(
        &mut self,
        _invoker: &mut I,
        _anchor: &B::Tensor,
        _capacity: usize,
        _lane: &Self::LaneState,
    ) -> Result<Option<B::Tensor>, I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        Ok(None)
    }

    /// Executes a fused proposal using the caller's temporary lane. The caller
    /// retains this lane through completion and discards it after the phase;
    /// speculative cache writes are never committed by this operation.
    #[allow(clippy::too_many_arguments)]
    fn fused_logits_observed<S, I>(
        &mut self,
        invoker: &mut I,
        anchor: &B::Tensor,
        capacity: usize,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<Option<B::Tensor>, I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        if observer.is_some() {
            return Err(I::invalid(
                "selected fused proposal has no internal observation path".into(),
            ));
        }
        self.fused_logits::<S, I>(invoker, anchor, capacity, lane)
    }

    /// Advances extension state after committed target execution.
    fn advance<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
    ) -> Result<(), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill::<S, I>(invoker, hidden, hidden, tokens, lane)
    }

    /// Replays retained target inputs with internal hooks when the selected
    /// extension supplies them. The caller retains replay/proposal attribution.
    fn advance_observed<S, I>(
        &mut self,
        invoker: &mut I,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        lane: &mut Self::LaneState,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    ) -> Result<(), I::Error>
    where
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
        I: PredictionOperationInvoker<A, B, S>,
    {
        self.prefill_observed::<S, I>(invoker, hidden, hidden, tokens, lane, observer)
    }
}
