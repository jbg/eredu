//! One causal perception invocation supplies both RNNT and language-model features.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use super::{
    config::PerceptionConfig,
    conformer::{ConformerBlock, ConformerDimensions, ConformerError, ConformerState},
    subsampling::{CausalSubsampling, SubsamplingDimensions, SubsamplingError},
};
use eredu_nn::{LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor};

/// Geometry of the subsampler, independently materializable layers and final projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerceptionDimensions {
    /// Causal mel subsampling geometry.
    pub subsampling: SubsamplingDimensions,
    /// Geometry shared by the encoder layers.
    pub block: ConformerDimensions,
    /// Number of encoder layers.
    pub layers: usize,
    /// Language-model embedding width.
    pub output_width: i32,
}
impl PerceptionDimensions {
    pub(super) fn validate(self) -> Result<(), PerceptionError> {
        if self.layers == 0
            || self.layers > 256
            || self.output_width <= 0
            || self.subsampling.width != self.block.width
        {
            return Err(PerceptionError::Configuration(
                "invalid perception layer count or connected widths",
            ));
        }
        Ok(())
    }

    /// Resolves the released identity modality adapter and causal encoder policy.
    pub fn from_config(config: &PerceptionConfig) -> Result<Self, PerceptionError> {
        let encoder = &config.encoder;
        if encoder.xscaling
            || encoder.reduction.is_some()
            || encoder.feat_out != -1
            || encoder.global_tokens != 0
            || config.modality_adapter_dim != encoder.d_model
        {
            return Err(PerceptionError::Configuration(
                "perception changes the released identity adapter or encoder equations",
            ));
        }
        Ok(Self {
            subsampling: SubsamplingDimensions::from_config(encoder)?,
            block: ConformerDimensions::from_config(encoder)?,
            layers: encoder.n_layers,
            output_width: i32::try_from(config.output_dim).map_err(|_| {
                PerceptionError::Configuration("projection width exceeds tensor indexing")
            })?,
        })
    }

    /// Constructs the final projection independently of all perception layers.
    pub fn build_projection<B: NeuralBackend>(
        self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Linear, PerceptionError> {
        if self.block.width <= 0 || self.output_width <= 0 {
            return Err(PerceptionError::Configuration(
                "invalid perception projection widths",
            ));
        }
        let parameter = |name: &str| {
            let mut spec = ParameterSpec::trainable(format!("stt_model.perception.proj.{name}"))
                .expect("canonical projection identifier");
            spec.group = Some("voicechat.perception.projection".into());
            spec
        };
        Ok(B::linear(
            LinearSpec {
                input: self.block.width,
                output: self.output_width,
                weight: parameter("weight"),
                bias: Some(parameter("bias")),
                format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
            },
            context,
        )?)
    }
}

/// Invalid geometry, incompatible retained state or numerical failure.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionError {
    /// Shared component declaration or selected ownership is invalid.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentDefinitionError),
    /// Shared traversal, residency or native mechanism failure with its source preserved.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Portable execution graph declarations are inconsistent.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),
    /// Execution groups cannot be assigned exact unit ordinals.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),
    /// Architecture state geometry or a requested state slot is invalid.
    #[error(transparent)]
    StateLayout(#[from] eredu_runtime::StateError),
    /// Ordered layer state policy is invalid.
    #[error(transparent)]
    LayerSchedule(#[from] eredu_core::LayerScheduleError),
    /// Symbolic parameter group shapes or placement are invalid.
    #[error(transparent)]
    ParameterPlan(#[from] eredu_runtime::ParallelPlanError),
    /// Parameters do not have exact execution-unit ownership.
    #[error(transparent)]
    ParameterDescription(#[from] eredu_runtime::ArchitectureParameterError),
    /// Portable state identity or topology is invalid.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),
    /// The pinned streaming wrapper keeps perception parameters and activations in float32.
    #[error("{0}")]
    Precision(&'static str),
    /// Model geometry or equations are invalid.
    #[error("{0}")]
    Configuration(&'static str),
    /// A chunk or cache does not match the selected perception stack.
    #[error("{0}")]
    State(&'static str),
    /// Subsampling failed without publishing a state transition.
    #[error(transparent)]
    Subsampling(#[from] SubsamplingError),
    /// A causal encoder layer failed.
    #[error(transparent)]
    Conformer(#[from] ConformerError),
    /// An ordinary neural operation failed, preserving its original source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Encoder caches and the output-frame frontier, separate from PCM and tool clocks.
#[derive(Debug, Clone)]
pub struct PerceptionState<T> {
    pub(super) dimensions: Option<PerceptionDimensions>,
    pub(super) blocks: Vec<ConformerState<T>>,
    pub(super) frames: u64,
}

impl<T: SnapshotValue> SnapshotValue for PerceptionState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([self.blocks.estimate_snapshot_value()])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            blocks: self.blocks.copy_snapshot_value(retention)?,
            dimensions: self.dimensions,
            frames: self.frames,
        })
    }
}

impl<T> RetainedTensorRoots<T> for PerceptionState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for block in &self.blocks {
            block.visit_retained_tensors(visitor);
        }
    }
}
impl<T> Default for PerceptionState<T> {
    fn default() -> Self {
        Self {
            dimensions: None,
            blocks: Vec::new(),
            frames: 0,
        }
    }
}
impl<T> PerceptionState<T> {
    /// Retained layer caches, suitable for independent snapshot branches.
    pub fn blocks(&self) -> &[ConformerState<T>] {
        &self.blocks
    }
    /// Number of encoded output frames accepted into this candidate state.
    pub fn frames(&self) -> u64 {
        self.frames
    }
}

/// Unpublished outputs from the same encoder invocation.
#[derive(Debug, Clone)]
pub struct PerceptionOutput<T> {
    /// Unprojected `[batch, frames, encoder_width]` features for RNNT.
    pub encoder: T,
    /// Projected `[batch, frames, output_width]` features for Nemotron.
    pub projected: T,
    /// Candidate caches; completion and commitment remain the caller's responsibility.
    pub state: PerceptionState<T>,
}

/// Portable resident perception using the shared layered traversal.
pub struct Perception<B: NeuralBackend> {
    runtime: eredu_runtime::ResidentRuntime<
        eredu_runtime::ComponentAdapter<
            eredu_runtime::ComponentDefinition,
            super::perception_layered::PerceptionKernel<B>,
        >,
        B,
        super::perception_layered::PerceptionRuntimeState<B::Tensor>,
    >,
}
impl<B: NeuralBackend> std::fmt::Debug for Perception<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Perception")
            .field("dimensions", &self.dimensions())
            .finish_non_exhaustive()
    }
}
impl<B: NeuralBackend> Parameterized<B::Tensor> for Perception<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, B::Tensor>,
    {
        for group in self.runtime.units() {
            for unit in group {
                unit.visit_parameters(visitor);
            }
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>,
    {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.visit_parameters_mut(visitor);
            }
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        for group in self.runtime.units_mut() {
            for unit in group {
                unit.set_trainable(trainable);
            }
        }
    }
}
impl<B: NeuralBackend> Perception<B> {
    /// Constructs unloaded units using the ordinary resident runtime.
    pub fn new(
        dimensions: PerceptionDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, PerceptionError> {
        Self::new_with_precision(dimensions, Default::default(), context)
    }
    /// Constructs units with the selected portable projection precision.
    pub fn new_with_precision(
        dimensions: PerceptionDimensions,
        precision: super::precision::SpeechPrecision,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, PerceptionError> {
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::new(
                super::perception_layered::PerceptionKernel::new(dimensions)?
                    .with_precision(precision)?
                    .into_component()?,
                context,
            )?,
        })
    }
    /// Adopts independently materialized units without replacement allocations.
    pub(super) fn from_parts(
        dimensions: PerceptionDimensions,
        subsampling: CausalSubsampling<B>,
        blocks: Vec<ConformerBlock<B>>,
        projection: B::Linear,
    ) -> Result<Self, PerceptionError> {
        use super::perception_layered::{PerceptionKernel, PerceptionUnit};
        let architecture = PerceptionKernel::new(dimensions)?.into_component()?;
        let units = vec![
            vec![PerceptionUnit::Subsampling(subsampling)],
            blocks.into_iter().map(PerceptionUnit::Block).collect(),
            vec![PerceptionUnit::Projection(projection)],
        ];
        Ok(Self {
            runtime: eredu_runtime::ResidentRuntime::from_units(architecture, units).map_err(
                |error| match error {
                    eredu_runtime::ResidentUnitsError::Architecture(error) => error,
                    eredu_runtime::ResidentUnitsError::GroupCount { .. } => {
                        PerceptionError::Configuration(
                            "resident perception group count differs from its graph",
                        )
                    }
                    eredu_runtime::ResidentUnitsError::UnitCount { .. } => {
                        PerceptionError::Configuration(
                            "resident perception unit count differs from its graph",
                        )
                    }
                },
            )?,
        })
    }
    /// Connected perception geometry, independent of residency policy.
    pub fn dimensions(&self) -> PerceptionDimensions {
        self.runtime.architecture().kernel().dimensions()
    }
    /// Evaluates a supplied mel window through the same resident and bounded equations.
    /// Parent caches remain unchanged even if a later unit fails.
    pub fn forward(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError> {
        use super::perception_layered::{output, PerceptionInput, PerceptionRuntimeState};
        use eredu_runtime::LayeredInvocation;
        let mut state = PerceptionRuntimeState::from_state(self.dimensions(), previous)?;
        let (projected, forward) = self.runtime.invoke_with_context(
            PerceptionInput {
                mel,
                drop_preencoded,
            },
            &mut state,
            context,
        )?;
        output(projected, forward, state)
    }
}
impl<B: NeuralBackend> super::perception_layered::PerceptionExecution<B> for Perception<B> {
    fn dimensions(&self) -> PerceptionDimensions {
        self.dimensions()
    }
    fn forward(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionOutput<B::Tensor>, PerceptionError> {
        self.forward(mel, drop_preencoded, previous, context)
    }
}
