//! RNNT neural prediction and bounded greedy decoding for one conversational stream.
//!
//! The streaming reference primes the prediction network with two zero inputs.
//! First-decision blank evidence is retained separately from the label-loop stop.
//! PCM processing, perception, punctuation bias and turn policy are separate owners.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_nn::RetainedTensorRoots;
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::{
    recurrent::{Lstm, LstmLayerSpec, LstmLayerState, LstmState},
    LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor,
};

use super::config::RnntConfig;

/// Geometry of the prediction network and additive joint network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnntDimensions {
    /// Nonblank vocabulary cardinality. Blank occupies the following row.
    pub vocab_size: i32,
    /// Embedding and recurrent width.
    pub predictor_width: i32,
    /// Number of unidirectional LSTM layers.
    pub predictor_layers: usize,
    /// Unprojected perception width.
    pub encoder_width: i32,
    /// Joint network width before ReLU and the output affine projection.
    pub joint_width: i32,
}
impl RnntDimensions {
    /// Projects an admitted VoiceChat configuration into neural geometry.
    pub fn from_config(config: &RnntConfig) -> Result<Self, RnntError> {
        if config.activation != "relu"
            || !config.blank_as_pad
            || config.blank_id != config.vocab_size
            || config.normalization_mode.is_some()
            || config.rnn_hidden_size != -1
            || config.num_extra_outputs != 0
        {
            return Err(RnntError::InvalidConfiguration(
                "RNNT configuration changes the released prediction/joint equations",
            ));
        }
        let convert = |value| {
            i32::try_from(value).map_err(|_| {
                RnntError::InvalidConfiguration("RNNT dimension exceeds tensor indexing range")
            })
        };
        let result = Self {
            vocab_size: convert(config.vocab_size)?,
            predictor_width: convert(config.pred_hidden)?,
            predictor_layers: config.pred_rnn_layers,
            encoder_width: convert(config.encoder_hidden)?,
            joint_width: convert(config.joint_hidden)?,
        };
        result.validate()?;
        Ok(result)
    }
    /// Validates finite tensor and recurrent geometry.
    pub fn validate(self) -> Result<(), RnntError> {
        if self.vocab_size <= 0
            || self.vocab_size.checked_add(1).is_none()
            || self.predictor_width <= 0
            || self.predictor_width.checked_mul(4).is_none()
            || self.predictor_layers == 0
            || self.predictor_layers > 256
            || self.encoder_width <= 0
            || self.joint_width <= 0
        {
            return Err(RnntError::InvalidConfiguration("invalid RNNT geometry"));
        }
        Ok(())
    }
}

/// Portable neural/decoding failures. A failed transition leaves its parent state unchanged.
#[derive(Debug, thiserror::Error)]
pub enum RnntError {
    /// Shared component declaration or invocation failed.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentDefinitionError),
    /// Portable execution graph, layout or ownership is invalid.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),
    /// Unit ordering is invalid.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),
    /// Runtime state geometry is invalid.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// State schedule is invalid.
    #[error(transparent)]
    Schedule(#[from] eredu_core::LayerScheduleError),
    /// Parameter geometry is invalid.
    #[error(transparent)]
    Parameters(#[from] eredu_runtime::ParallelPlanError),
    /// Parameter ownership is invalid.
    #[error(transparent)]
    Ownership(#[from] eredu_runtime::ArchitectureParameterError),
    /// State identity is invalid.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),
    /// Provider or submission failure retains its native source.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Invalid model or resource policy, rejected before advancing recurrent state.
    #[error("{0}")]
    InvalidConfiguration(&'static str),
    /// Input or restored state belongs to different geometry.
    #[error("{0}")]
    InvalidState(&'static str),
    /// Hypothesis retention has reached the caller's explicit capacity.
    #[error("RNNT hypothesis capacity {capacity} exhausted")]
    Capacity {
        /// Maximum retained token count supplied by the caller.
        capacity: usize,
    },
    /// The explicit pipeline invocation budget cannot reserve a complete frame.
    #[error("RNNT predictor pipeline needs {required} substeps, with {remaining} remaining")]
    PipelineCapacity {
        /// Worst-case predictor invocations for the frame.
        required: u64,
        /// Caller-owned remaining invocation budget.
        remaining: u64,
    },
    /// A recurrent substep did not observe its exact committed epoch.
    #[error("RNNT predictor expected committed epoch {expected:?}, observed {observed:?}")]
    PredictorCommit {
        /// Epoch reserved before any model reads.
        expected: eredu_core::DistributedCommitEpoch,
        /// Exact observed outcome, including indeterminate completion.
        observed: eredu_core::DistributedCommitOutcome,
    },
    /// A backend mechanism failed; the original error remains its source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Explicit bounds for one frame transition.
#[derive(Debug, Clone, Copy)]
pub struct RnntLimits {
    /// Maximum emissions on a single encoder frame. The pinned wrapper uses 10.
    pub max_symbols_per_frame: usize,
    /// Maximum complete retained hypothesis length, including existing tokens.
    pub hypothesis_capacity: usize,
}

/// Recurrent state owned by one predictor stage and its complete prediction readout.
/// The final-layer owner reads its last hidden tensor directly; other stages retain
/// one published readout without retaining remote recurrent cells.
#[derive(Debug, Clone)]
pub struct RnntPredictionState<T> {
    dimensions: RnntDimensions,
    global_layers: std::ops::Range<usize>,
    local: LstmState<T>,
    published_readout: Option<T>,
}
impl<T> RnntPredictionState<T> {
    /// Global model geometry retained with this stage's carry.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    /// Exact globally indexed recurrent layers owned by this stage.
    pub fn global_layers(&self) -> std::ops::Range<usize> {
        self.global_layers.clone()
    }
    /// Only this stage's hidden/cell pairs, in global layer order.
    pub fn local_state(&self) -> &LstmState<T> {
        &self.local
    }
    /// Complete final predictor output, independent of which stage owns it.
    pub fn output(&self) -> &T {
        self.published_readout.as_ref().unwrap_or_else(|| {
            &self
                .local
                .layers
                .last()
                .expect("validated final predictor owner")
                .hidden
        })
    }
}
impl<T: Tensor> RnntPredictionState<T> {
    /// Constructs exact stage-local carry. A stage owning the final global layer
    /// omits a separate readout; every other stage supplies the published row.
    pub fn new(
        dimensions: RnntDimensions,
        global_layers: std::ops::Range<usize>,
        local: LstmState<T>,
        published_readout: Option<T>,
    ) -> Result<Self, RnntError> {
        let state = Self {
            dimensions,
            global_layers,
            local,
            published_readout,
        };
        state.validate(dimensions, &state.global_layers)?;
        Ok(state)
    }
    /// Constructs ordinary or tensor-replicated state with all recurrent layers.
    pub fn full(dimensions: RnntDimensions, local: LstmState<T>) -> Result<Self, RnntError> {
        Self::new(dimensions, 0..dimensions.predictor_layers, local, None)
    }
    /// Checks selected stage ownership before any learned unit is acquired.
    pub fn validate(
        &self,
        dimensions: RnntDimensions,
        global_layers: &std::ops::Range<usize>,
    ) -> Result<(), RnntError> {
        dimensions.validate()?;
        validate_predictor_ownership(dimensions, global_layers)?;
        if self.dimensions != dimensions
            || &self.global_layers != global_layers
            || self.local.layers.len() != global_layers.len()
        {
            return Err(RnntError::InvalidState(
                "RNNT prediction belongs to different stage ownership or geometry",
            ));
        }
        let shape = [1, dimensions.predictor_width];
        let non_floating = |value: &T| {
            value.element_type().is_some_and(|dtype| {
                !matches!(
                    dtype,
                    eredu_nn::TensorElementType::F16
                        | eredu_nn::TensorElementType::Bf16
                        | eredu_nn::TensorElementType::F32
                        | eredu_nn::TensorElementType::F64
                )
            })
        };
        if self
            .local
            .layers
            .iter()
            .any(|layer| non_floating(&layer.hidden) || non_floating(&layer.cell))
            || self.published_readout.as_ref().is_some_and(non_floating)
        {
            return Err(RnntError::InvalidState(
                "RNNT predictor carry and readout must have floating precision",
            ));
        }
        if self
            .local
            .layers
            .iter()
            .any(|layer| layer.hidden.shape() != shape || layer.cell.shape() != shape)
        {
            return Err(RnntError::InvalidState(
                "RNNT predictor state has incompatible local layer geometry",
            ));
        }
        let final_owner = global_layers.contains(&(dimensions.predictor_layers - 1));
        if final_owner == self.published_readout.is_some()
            || self
                .published_readout
                .as_ref()
                .is_some_and(|value| value.shape() != shape)
        {
            return Err(RnntError::InvalidState(
                "RNNT prediction readout differs from final-layer ownership",
            ));
        }
        Ok(())
    }
}
impl<T> RetainedTensorRoots<T> for RnntPredictionState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.local.visit_retained_tensors(visitor);
        if let Some(readout) = &self.published_readout {
            visitor(readout);
        }
    }
}
impl<T: SnapshotValue> SnapshotValue for RnntPredictionState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let layer_slots = self
            .local
            .layers
            .capacity()
            .checked_mul(std::mem::size_of::<LstmLayerState<T>>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                BackendFailure::from_error(
                    eredu_core::execution_control::ExecutionControlError::Overflow,
                )
            })?;
        snapshot_value_estimate::<Self>(
            [
                Ok(SnapshotEstimate {
                    retained_bytes: layer_slots,
                    copy_bytes: 0,
                }),
                self.published_readout.estimate_snapshot_value(),
            ]
            .into_iter()
            .chain(self.local.layers.iter().map(|layer| {
                snapshot_value_estimate::<LstmLayerState<T>>([
                    layer.hidden.estimate_snapshot_value(),
                    layer.cell.estimate_snapshot_value(),
                ])
            })),
        )
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.local.layers.len())
            .map_err(BackendFailure::from_error)?;
        for layer in &self.local.layers {
            layers.push(LstmLayerState {
                hidden: layer.hidden.copy_snapshot_value(retention)?,
                cell: layer.cell.copy_snapshot_value(retention)?,
            });
        }
        Ok(Self {
            dimensions: self.dimensions,
            global_layers: self.global_layers.clone(),
            local: LstmState { layers },
            published_readout: self.published_readout.copy_snapshot_value(retention)?,
        })
    }
}
fn validate_predictor_ownership(
    dimensions: RnntDimensions,
    global_layers: &std::ops::Range<usize>,
) -> Result<(), RnntError> {
    if global_layers.start > global_layers.end || global_layers.end > dimensions.predictor_layers {
        return Err(RnntError::InvalidState(
            "RNNT predictor layer ownership exceeds global geometry",
        ));
    }
    Ok(())
}

/// Mutable neural state is separate from checkpoint parameters and host turn policy.
#[derive(Debug, Clone)]
pub struct RnntState<T> {
    dimensions: Option<RnntDimensions>,
    predictor: Option<RnntPredictionState<T>>,
    hypothesis: Vec<usize>,
    frames: u64,
}

impl<T: SnapshotValue> SnapshotValue for RnntState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.predictor.estimate_snapshot_value(),
            self.hypothesis.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            dimensions: self.dimensions,
            predictor: self.predictor.copy_snapshot_value(retention)?,
            hypothesis: self.hypothesis.copy_snapshot_value(retention)?,
            frames: self.frames,
        })
    }
}

impl<T> RetainedTensorRoots<T> for RnntState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(predictor) = &self.predictor {
            predictor.visit_retained_tensors(visitor);
        }
    }
}
impl<T> Default for RnntState<T> {
    fn default() -> Self {
        Self {
            dimensions: None,
            predictor: None,
            hypothesis: Vec::new(),
            frames: 0,
        }
    }
}
impl<T> RnntState<T> {
    /// Retained LSTM values, including the last-layer output used by the joint.
    pub fn predictor(&self) -> Option<&RnntPredictionState<T>> {
        self.predictor.as_ref()
    }
    /// Complete retained nonblank hypothesis.
    pub fn hypothesis(&self) -> &[usize] {
        &self.hypothesis
    }
    /// Encoder frames advanced; token-only prediction does not change this count.
    pub fn frames(&self) -> u64 {
        self.frames
    }
}

/// Unpublished result of one bounded encoder-frame transition.
#[derive(Debug, Clone)]
pub struct RnntFrame<T> {
    /// Candidate state; the caller owns completion and publication.
    pub state: RnntState<T>,
    /// Newly emitted nonblank tokens in order.
    pub emitted: Vec<usize>,
    /// Whether the first decision was blank, independent of the label-loop stop.
    pub first_blank: bool,
    /// Whether the final evaluated joint decision was blank.
    pub ended_blank: bool,
    /// The frame stopped with a nonblank decision at the symbol bound.
    pub hit_symbol_limit: bool,
}

/// VoiceChat RNNT prediction and joint networks using ordinary parameter operators.
pub struct Rnnt<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            super::rnnt_layered::RnntKernel<B>,
        >,
        B,
        eredu_runtime::StatelessComponentState,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for Rnnt<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Rnnt")
            .field(
                "dimensions",
                &self.runtime.architecture().kernel().dimensions(),
            )
            .finish_non_exhaustive()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for Rnnt<B> {
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, B::Tensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        for unit in self.runtime.units().iter().flatten() {
            unit.visit_parameters(visitor);
        }
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        for unit in self.runtime.units_mut().iter_mut().flatten() {
            unit.visit_parameters_mut(visitor);
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        for unit in self.runtime.units_mut().iter_mut().flatten() {
            unit.set_trainable(trainable);
        }
    }
}

pub(super) fn parameter(id: String, group: &str) -> ParameterSpec {
    let mut spec = ParameterSpec::trainable(id).expect("fixed RNNT parameter identity");
    spec.group = Some(group.into());
    spec
}
pub(super) fn affine(
    input: i32,
    output: i32,
    weight: String,
    bias: String,
    group: &str,
) -> LinearSpec {
    LinearSpec {
        input,
        output,
        weight: parameter(weight, group),
        bias: Some(parameter(bias, group)),
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
            .expect("dense RNNT layout"),
    }
}
impl<B: NeuralBackend> Rnnt<B> {
    /// Declares resident resources through the same typed kernel as bounded execution.
    pub fn new(
        dimensions: RnntDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, RnntError> {
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::new(
                super::rnnt_layered::RnntKernel::new(dimensions)?.into_component()?,
                context,
            )?,
        })
    }
}

fn greedy_symbol<T: Tensor>(logits: &T, context: &T::Context) -> Result<usize, RnntError> {
    let negative = logits.multiply_scalar(-1.0, context)?;
    let token = T::argmin_axis(&negative, -1, false, context)?.to_i32_vec(context)?;
    if token.len() != 1 || token[0] < 0 {
        return Err(RnntError::InvalidState(
            "RNNT greedy selection returned an invalid token",
        ));
    }
    Ok(token[0] as usize)
}

/// Independently materializable RNNT embedding and joint-network mechanisms.
/// Recurrent state and its transport belong to a separate predictor execution.
pub trait RnntProjectionExecution<B: NeuralBackend> {
    /// Exact shared RNNT geometry.
    fn dimensions(&self) -> RnntDimensions;
    /// One vocabulary row, including the blank used to derive typed zeros.
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// The encoder affine, evaluated once per audio frame.
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// The prediction affine over the published final recurrent readout.
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Vocabulary scores following the shared additive/ReLU equation.
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Greedy decision, invoked only by the distributed decision owner.
    fn select_symbol(
        &mut self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<usize, RnntError> {
        greedy_symbol(logits, context)
    }
}
/// An existing complete RNNT exposes its projection mechanisms through a borrowed view.
impl<B: NeuralBackend, R: RnntExecution<B>> RnntProjectionExecution<B> for &mut R {
    fn dimensions(&self) -> RnntDimensions {
        RnntExecution::dimensions(&**self)
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntExecution::embed_token(&mut **self, token, context)
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntExecution::encode_frame(&mut **self, input, context)
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntExecution::project_prediction(&mut **self, input, context)
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntExecution::project_output(&mut **self, input, context)
    }
    fn select_symbol(
        &mut self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<usize, RnntError> {
        RnntExecution::select_symbol(&mut **self, logits, context)
    }
}

/// Portable RNNT mechanisms with one shared priming and bounded greedy policy.
/// All returned recurrent state is unpublished; parent tensors are immutable.
pub trait RnntExecution<B: NeuralBackend> {
    /// Exact component geometry.
    fn dimensions(&self) -> RnntDimensions;
    /// Exact recurrent layer ownership; ordinary and tensor replicas own all layers.
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        0..self.dimensions().predictor_layers
    }
    /// Looks up one vocabulary row, including the blank row used for typed zeros.
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Applies every prediction layer; absent state starts with typed zeros.
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError>;
    /// Projects perception once in the encoder affine's actual parameter precision.
    fn encode_frame(
        &mut self,
        encoder: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Applies the independently materializable predictor joint affine.
    fn project_prediction(
        &mut self,
        prediction: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Applies the independently materializable vocabulary affine after ReLU.
    fn project_output(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError>;
    /// Selects one authoritative greedy symbol from complete logits.
    /// Distributed realizations override only this hook; the label loop stays shared.
    fn select_symbol(
        &mut self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<usize, RnntError> {
        greedy_symbol(logits, context)
    }
    /// Validates restored recurrent geometry before parameter acquisition.
    fn validate_state(&self, state: &RnntState<B::Tensor>) -> Result<(), RnntError> {
        validate_predictor_ownership(self.dimensions(), &self.predictor_ownership())?;
        if state.dimensions.is_some_and(|d| d != self.dimensions())
            || state.dimensions.is_some() != state.predictor.is_some()
        {
            return Err(RnntError::InvalidState(
                "RNNT state belongs to different model geometry",
            ));
        }
        if let Some(predictor) = &state.predictor {
            predictor.validate(self.dimensions(), &self.predictor_ownership())?;
        } else if !state.hypothesis.is_empty() || state.frames != 0 {
            return Err(RnntError::InvalidState(
                "unprimed RNNT state contains decoded history",
            ));
        }
        Ok(())
    }
    /// Validates one perception row before parameter acquisition.
    fn validate_frame(&self, encoder: &B::Tensor) -> Result<(), RnntError> {
        if encoder.shape() != [1, self.dimensions().encoder_width] {
            return Err(RnntError::InvalidState(
                "RNNT frame expects one unprojected encoder row [1, encoder_width]",
            ));
        }
        Ok(())
    }
    /// Runs the pinned streaming wrapper's two zero-input initialization steps.
    /// Existing primed state is retained unchanged; initialization emits no token.
    fn prime(
        &mut self,
        state: &RnntState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntState<B::Tensor>, RnntError> {
        self.validate_state(state)?;
        if state.predictor.is_some() {
            return Ok(state.clone());
        }
        let zero = self
            .embed_token(self.dimensions().vocab_size as usize, context)?
            .zeros_like(context)?;
        let first = self.predictor_step(&zero, None, context)?;
        first.validate(self.dimensions(), &self.predictor_ownership())?;
        let second = self.predictor_step(&zero, Some(&first), context)?;
        second.validate(self.dimensions(), &self.predictor_ownership())?;
        let mut next = state.clone();
        next.dimensions = Some(self.dimensions());
        next.predictor = Some(second);
        Ok(next)
    }
    /// Applies the additive joint equation to an already projected encoder row.
    fn joint_projected(
        &mut self,
        encoded: &B::Tensor,
        state: &RnntState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.validate_state(state)?;
        if encoded.shape() != [1, self.dimensions().joint_width] {
            return Err(RnntError::InvalidState(
                "RNNT projected encoder has incompatible row geometry",
            ));
        }
        let prediction = state
            .predictor
            .as_ref()
            .map(RnntPredictionState::output)
            .ok_or(RnntError::InvalidState("RNNT predictor is not primed"))?;
        let hidden = encoded
            .add(&self.project_prediction(prediction, context)?, context)?
            .maximum_scalar(0.0, context)?;
        self.project_output(&hidden, context)
    }
    /// Unnormalized joint scores for a primed state; greedy decoding is invariant
    /// to the reference's optional log-softmax and does not need it materialized.
    fn joint_logits(
        &mut self,
        encoder: &B::Tensor,
        state: &RnntState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.validate_frame(encoder)?;
        self.validate_state(state)?;
        if state.predictor.is_none() {
            return Err(RnntError::InvalidState("RNNT predictor is not primed"));
        }
        let encoded = self.encode_frame(encoder, context)?;
        self.joint_projected(&encoded, state, context)
    }
    /// Advances prediction through one nonblank token without advancing audio time.
    /// This is also the neural operation used by externally owned display punctuation.
    fn predict_token(
        &mut self,
        token: usize,
        state: &RnntState<B::Tensor>,
        capacity: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntState<B::Tensor>, RnntError> {
        self.validate_state(state)?;
        if token >= self.dimensions().vocab_size as usize {
            return Err(RnntError::InvalidState(
                "RNNT prediction requires a nonblank vocabulary token",
            ));
        }
        if state.hypothesis.len() >= capacity {
            return Err(RnntError::Capacity { capacity });
        }
        let mut next = self.prime(state, context)?;
        let embedded = self.embed_token(token, context)?;
        let predictor = self.predictor_step(&embedded, next.predictor.as_ref(), context)?;
        predictor.validate(self.dimensions(), &self.predictor_ownership())?;
        next.predictor = Some(predictor);
        next.hypothesis.push(token);
        Ok(next)
    }
    /// Validates the complete frame transition before acquiring parameters.
    /// Returns the exclusive candidate frame frontier for shared distributed admission.
    fn validate_decode_frame(
        &self,
        encoder: &B::Tensor,
        state: &RnntState<B::Tensor>,
        limits: RnntLimits,
    ) -> Result<u64, RnntError> {
        self.validate_frame(encoder)?;
        self.validate_decode_state(state, limits)
    }
    /// Cold state and resource preflight, shared before perception materialization.
    fn validate_decode_state(
        &self,
        state: &RnntState<B::Tensor>,
        limits: RnntLimits,
    ) -> Result<u64, RnntError> {
        self.validate_state(state)?;
        if limits.max_symbols_per_frame == 0 {
            return Err(RnntError::InvalidConfiguration(
                "RNNT symbol limit must be positive",
            ));
        }
        if state.hypothesis.len() > limits.hypothesis_capacity {
            return Err(RnntError::Capacity {
                capacity: limits.hypothesis_capacity,
            });
        }
        state
            .frames
            .checked_add(1)
            .ok_or(RnntError::InvalidState("RNNT frame counter exhausted"))
    }
    /// Greedy label loop over the shared perception result. Encoder projection is
    /// evaluated once. Blank preserves prediction state; a symbol bound advances
    /// to the next frame with every emitted token reflected in the retained state.
    /// Completion/publication, turn counters and punctuation bias remain external.
    fn decode_frame(
        &mut self,
        encoder: &B::Tensor,
        state: &RnntState<B::Tensor>,
        limits: RnntLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntFrame<B::Tensor>, RnntError> {
        let frames = self.validate_decode_frame(encoder, state, limits)?;
        let mut next = self.prime(state, context)?;
        let encoded = self.encode_frame(encoder, context)?;
        let logits = self.joint_projected(&encoded, &next, context)?;
        let mut token = self.select_symbol(&logits, context)?;
        let blank = self.dimensions().vocab_size as usize;
        let first_blank = token == blank;
        let mut emitted = Vec::new();
        while token != blank && emitted.len() < limits.max_symbols_per_frame {
            next = self.predict_token(token, &next, limits.hypothesis_capacity, context)?;
            emitted.push(token);
            let logits = self.joint_projected(&encoded, &next, context)?;
            token = self.select_symbol(&logits, context)?;
        }
        next.frames = frames;
        Ok(RnntFrame {
            state: next,
            emitted,
            first_blank,
            ended_blank: token == blank,
            hit_symbol_limit: token != blank,
        })
    }
}

// NVIDIA casts the perception output to the joint's actual parameter
// arithmetic type; generic linear promotion must not choose this policy.
pub(super) fn encoder_projection_input<B: NeuralBackend>(
    projection: &B::Linear,
    encoder: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, RnntError> {
    struct Dtype(Option<eredu_nn::TensorElementType>);
    impl<'a, T: Tensor> eredu_nn::ParameterVisitor<'a, T> for Dtype {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadata, tensor: &'a T) {
            if metadata.id.as_str() == "stt_model.rnnt_joint.enc.weight" {
                self.0 = tensor.element_type();
            }
        }
    }
    let mut dtype = Dtype(None);
    projection.visit_parameters(&mut dtype);
    let dtype = dtype
        .0
        .filter(|d| {
            matches!(
                d,
                eredu_nn::TensorElementType::F16
                    | eredu_nn::TensorElementType::Bf16
                    | eredu_nn::TensorElementType::F32
                    | eredu_nn::TensorElementType::F64
            )
        })
        .ok_or(RnntError::InvalidState(
            "RNNT joint parameter floating precision is unavailable",
        ))?;
    if encoder.element_type() == Some(dtype) {
        Ok(encoder.clone())
    } else {
        Ok(encoder.cast_float(dtype, context)?)
    }
}

fn predictor_layer_spec(
    dimensions: RnntDimensions,
    index: usize,
) -> Result<LstmLayerSpec, RnntError> {
    dimensions.validate()?;
    if index >= dimensions.predictor_layers {
        return Err(RnntError::InvalidConfiguration(
            "RNNT predictor layer exceeds global geometry",
        ));
    }
    let prefix = "stt_model.rnnt_decoder.prediction.dec_rnn.lstm";
    let group = format!("voicechat.rnnt.predictor.{index}");
    let matrix = |name: &str| {
        affine(
            dimensions.predictor_width,
            dimensions.predictor_width * 4,
            format!("{prefix}.weight_{name}_l{index}"),
            format!("{prefix}.bias_{name}_l{index}"),
            &group,
        )
    };
    Ok(LstmLayerSpec {
        input: matrix("ih"),
        recurrent: matrix("hh"),
    })
}
/// Constructs one canonical predictor layer for ordinary, leased or pipeline execution.
pub fn build_predictor_layer<B: NeuralBackend>(
    dimensions: RnntDimensions,
    index: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Lstm<B>, RnntError> {
    Ok(Lstm::new(
        vec![predictor_layer_spec(dimensions, index)?],
        context,
    )?)
}
/// Executes exactly one recurrent layer with immutable preceding carry.
/// Initialization always derives from the original prediction input's precision,
/// even when earlier layers promote their output to another arithmetic type.
pub fn step_predictor_layer<B: NeuralBackend>(
    layer: &mut Lstm<B>,
    input: &B::Tensor,
    previous: Option<&LstmLayerState<B::Tensor>>,
    zero_prototype: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<LstmLayerState<B::Tensor>, RnntError> {
    let previous = if let Some(previous) = previous {
        LstmState {
            layers: vec![previous.clone()],
        }
    } else {
        layer.zero_state_like(zero_prototype, context)?
    };
    if previous.layers.len() != 1 {
        return Err(RnntError::InvalidConfiguration(
            "RNNT predictor unit contains more than one recurrent layer",
        ));
    }
    let mut next = layer.step(input, &previous, context)?;
    Ok(next.layers.pop().expect("one validated recurrent layer"))
}

impl<B: NeuralBackend> RnntExecution<B> for Rnnt<B> {
    fn dimensions(&self) -> RnntDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime.forward(
            super::rnnt_layered::RnntInput::Embed(token),
            &mut eredu_runtime::StatelessComponentState::new()?,
            context,
        )
    }
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        self.runtime
            .forward_with_context(
                super::rnnt_layered::RnntInput::Predict { input, previous },
                &mut eredu_runtime::StatelessComponentState::new()?,
                context,
            )?
            .1
            .into_predictor()
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.validate_frame(input)?;
        self.runtime.forward(
            super::rnnt_layered::RnntInput::Encode(input),
            &mut eredu_runtime::StatelessComponentState::new()?,
            context,
        )
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime.forward(
            super::rnnt_layered::RnntInput::Prediction(input),
            &mut eredu_runtime::StatelessComponentState::new()?,
            context,
        )
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime.forward(
            super::rnnt_layered::RnntInput::Output(input),
            &mut eredu_runtime::StatelessComponentState::new()?,
            context,
        )
    }
}
