//! Independently leased RNNT embedding, recurrent layers and joint affines.
//!
//! Each invocation owns only an unpublished neural phase. The shared RNNT
//! execution trait owns priming, hypothesis capacity and the greedy label loop.

use super::rnnt::{
    encoder_projection_input, step_predictor_layer, RnntDimensions, RnntError, RnntExecution,
    RnntPredictionState,
};
use super::rnnt_specs::{Declaration, RnntDeclaration, UnitSpec, EMBEDDING, PREDICTOR};
use eredu_core::BackendFailure;
use eredu_nn::{
    recurrent::{Lstm, LstmState},
    DistributedNeuralBackend, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    LinearOperator, LinearSpec, NeuralBackend, Parameterized, RetainedTensorRoots, Tensor,
    VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentKernel, ComponentUnit,
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout,
    LayeredForwardState, LayeredInvocation, LayerwisePolicy, LayerwiseRuntime, LocalModelLayout,
    MemberSharding, OwnedParameterGroupSpec, ParallelComponentKernel, ParameterGroupOwner,
    ParameterRole, StatelessComponentState, SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const GROUPS: [&str; 5] = [
    "voicechat.rnnt.embedding",
    "voicechat.rnnt.predictor",
    "voicechat.rnnt.joint.enc",
    "voicechat.rnnt.joint.pred",
    "voicechat.rnnt.joint.joint_net.2",
];
#[derive(Debug, Clone, Copy)]
pub(crate) enum RnntPhase {
    Embedding,
    Predictor,
    Encoder,
    Prediction,
    Output,
}
impl RnntPhase {
    pub(super) const fn ordinal(self) -> usize {
        self as usize
    }
    pub(super) const fn id(self) -> &'static str {
        GROUPS[self.ordinal()]
    }
    pub(crate) fn units(self, dimensions: RnntDimensions) -> usize {
        if matches!(self, Self::Predictor) {
            dimensions.predictor_layers
        } else {
            1
        }
    }
    pub(super) fn parameter_group(self, index: usize) -> String {
        if matches!(self, Self::Predictor) {
            format!("{}.{index}", self.id())
        } else {
            self.id().into()
        }
    }
    fn vocabulary(self) -> bool {
        matches!(self, Self::Embedding | Self::Output)
    }
    pub(super) fn path(self, index: usize) -> String {
        match self {
            Self::Embedding => EMBEDDING.into(),
            Self::Predictor => format!("{PREDICTOR}.layer{index}"),
            Self::Encoder => "stt_model.rnnt_joint.enc".into(),
            Self::Prediction => "stt_model.rnnt_joint.pred".into(),
            Self::Output => "stt_model.rnnt_joint.joint_net.2".into(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum RnntSelection {
    Complete,
    Projections,
}
impl RnntSelection {
    pub(crate) fn phases(self) -> &'static [RnntPhase] {
        match self {
            Self::Complete => &[
                RnntPhase::Embedding,
                RnntPhase::Predictor,
                RnntPhase::Encoder,
                RnntPhase::Prediction,
                RnntPhase::Output,
            ],
            Self::Projections => &[
                RnntPhase::Embedding,
                RnntPhase::Encoder,
                RnntPhase::Prediction,
                RnntPhase::Output,
            ],
        }
    }
}
pub(super) fn rnnt_definition_for_selection(
    selection: RnntSelection,
    declaration: &Declaration,
) -> Result<ComponentDefinition, RnntError> {
    let phases = selection.phases();
    let parameters = declared_parameter_description(phases, declaration)?;
    let mut units = Vec::new();
    for phase in phases {
        units.extend(
            (0..declaration.phase(*phase)?.len()).map(|index| ComponentUnit {
                path: phase.path(index),
                state: 0..0,
            }),
        );
    }
    let transport = ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::Projector,
        first_owner_static_roles: vec![],
        last_owner_static_roles: vec![],
        merge_destination: ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: None,
        request_optional: false,
    };
    let primary = match selection {
        RnntSelection::Complete => RnntPhase::Embedding,
        RnntSelection::Projections => RnntPhase::Output,
    };
    Ok(ComponentDefinition::new(
        "voicechat.rnnt",
        parameters,
        units,
        vec![transport; phases.len()],
        primary.id(),
        vec![],
        StatelessComponentState::state_layout()?,
        ArchitectureStatePartitionPlan::new([]),
    )?)
}

#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum UnitModule<B: NeuralBackend> {
    Embedding(B::Embedding),
    Predictor(Lstm<B>),
    Projection(B::Linear),
}
/// One exact learned owner, including its immutable address and geometry.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RnntUnit<B: NeuralBackend> {
    #[parameter(skip)]
    dimensions: RnntDimensions,
    #[parameter(skip)]
    group: usize,
    #[parameter(skip)]
    index: usize,
    #[parameter(skip)]
    placement: Option<RnntLocalDimensions>,
    module: UnitModule<B>,
}

/// One neural phase selected before acquiring any parameter.
pub enum RnntInput<'a, T> {
    /// Learned embedding row, including blank for typed priming zeros.
    Embed(usize),
    /// One complete recurrent step through individually leased LSTM layers.
    Predict {
        /// Input row in embedding precision.
        input: &'a T,
        /// Immutable preceding recurrent state, or typed zero initialization.
        previous: Option<&'a RnntPredictionState<T>>,
    },
    /// Perception affine with released parameter-precision casting.
    Encode(&'a T),
    /// Predictor joint affine.
    Prediction(&'a T),
    /// Vocabulary joint affine after shared additive ReLU.
    Output(&'a T),
}
/// Parent and unpublished candidate roots remain live through exact lease completion.
pub struct RnntForward<T> {
    dimensions: RnntDimensions,
    group: usize,
    input: T,
    previous: Option<RnntPredictionState<T>>,
    candidate: LstmState<T>,
    completed: usize,
}
impl<T: Tensor> RnntForward<T> {
    /// Takes a completed recurrent candidate after every selected layer completes.
    pub fn into_predictor(self) -> Result<RnntPredictionState<T>, RnntError> {
        if self.group != 1 || self.candidate.layers.is_empty() {
            return Err(RnntError::InvalidState(
                "RNNT invocation has no recurrent candidate",
            ));
        }
        RnntPredictionState::full(self.dimensions, self.candidate)
    }
}

/// Coherent vocabulary ownership for both embedding and output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnntLocalDimensions {
    /// Inclusive first row in the full vocabulary, including the blank row.
    pub vocabulary_start: i32,
    /// Exclusive row limit.
    pub vocabulary_end: i32,
}
impl RnntLocalDimensions {
    fn range(self, dimensions: RnntDimensions) -> VocabularyParallelRange {
        VocabularyParallelRange {
            global_vocabulary: (dimensions.vocab_size + 1) as usize,
            local: self.vocabulary_start as usize..self.vocabulary_end as usize,
        }
    }
}
pub(crate) fn rnnt_local_dimensions_for_description(
    declaration: &Declaration,
    layout: &LocalModelLayout,
    description: &ArchitectureParameterDescription,
) -> Result<RnntLocalDimensions, RnntError> {
    let UnitSpec::Embedding(embedding) = declaration.unit(RnntPhase::Embedding, 0)? else {
        return Err(RnntError::InvalidConfiguration(
            "RNNT vocabulary owner is not an embedding",
        ));
    };
    let weight =
        layout
            .tensor(embedding.weight.id.as_str())
            .ok_or(RnntError::InvalidConfiguration(
                "RNNT layout omits vocabulary embedding",
            ))?;
    let (start, end) = match weight.placement() {
        TensorPlacement::Range {
            axis: 0,
            start,
            end,
        } => (*start, *end),
        _ => {
            return Err(RnntError::InvalidConfiguration(
                "RNNT embedding requires vocabulary row placement",
            ));
        }
    };
    let range = VocabularyParallelRange {
        global_vocabulary: embedding.vocabulary as usize,
        local: start..end,
    };
    range.validate()?;
    let local = RnntLocalDimensions {
        vocabulary_start: start as i32,
        vocabulary_end: end as i32,
    };
    eredu_runtime::validate_parameter_layout(
        description.groups().iter().map(|owned| owned.group()),
        layout,
        |group| {
            Ok::<_, RnntError>(if group.role() == ParameterRole::Vocabulary {
                eredu_runtime::ParameterGroupSelection::BalancedAxis {
                    axis: 0,
                    global_extent: range.global_vocabulary,
                    range: range.local.clone(),
                }
            } else {
                eredu_runtime::ParameterGroupSelection::Logical(None)
            })
        },
    )?;
    Ok(local)
}
// Function pointers preserve ordinary construction for neutral backends without
// requiring distributed mechanisms until parallel architecture selection.
struct ParallelRnnt<B: NeuralBackend> {
    local: RnntLocalDimensions,
    layout: LocalModelLayout,
    embedding: fn(
        EmbeddingSpec,
        VocabularyParallelRange,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Embedding, eredu_nn::Error>,
    output: fn(
        LinearSpec,
        VocabularyParallelRange,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Linear, eredu_nn::Error>,
    lookup: fn(
        &mut B::Embedding,
        &B::Tensor,
        EmbeddingLookupPolicy,
        &B::ParallelContext,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>,
    project: fn(
        &mut B::Linear,
        &B::Tensor,
        &B::ParallelContext,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>,
}

/// Canonical phase graph and independently materializable RNNT units.
/// `PREDICTOR` selects recurrent ownership and the available execution contract;
/// both profiles use the same phase equations and resource lifecycle.
pub struct RnntKernel<B: NeuralBackend, const PREDICTOR: bool = true> {
    dimensions: RnntDimensions,
    declaration: Declaration,
    definition: ComponentDefinition,
    parallel: Option<ParallelRnnt<B>>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend, const PREDICTOR: bool> RnntKernel<B, PREDICTOR> {
    /// Retains validated ownership without allocating any native parameter.
    pub fn new(declaration: RnntDeclaration<PREDICTOR>) -> Result<Self, RnntError> {
        let (dimensions, declaration, definition) = declaration.into_parts();
        Ok(Self {
            dimensions,
            declaration,
            definition,
            parallel: None,
            backend: PhantomData,
        })
    }
    /// Whether all primitive invocations require a matching collective context.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    /// Exact global RNNT geometry.
    pub fn dimensions(&self) -> RnntDimensions {
        self.dimensions
    }
    fn selection() -> RnntSelection {
        if PREDICTOR {
            RnntSelection::Complete
        } else {
            RnntSelection::Projections
        }
    }
    /// Couples selected family equations with one exact semantic declaration.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, RnntError> {
        Ok(ComponentAdapter::new(self.definition.clone(), self)?)
    }
    /// Declares only one requested embedding, recurrent layer or joint affine.
    fn build_resource(
        &self,
        phase: RnntPhase,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntUnit<B>, RnntError> {
        let group = phase.ordinal();
        let d = self.dimensions;
        let module = match self.declaration.unit(phase, index)?.clone() {
            UnitSpec::Embedding(spec) => {
                UnitModule::Embedding(if let Some(selected) = &self.parallel {
                    (selected.embedding)(spec, selected.local.range(d), context)?
                } else {
                    B::embedding(spec, context)?
                })
            }
            UnitSpec::Predictor(spec) => UnitModule::Predictor(Lstm::new(vec![spec], context)?),
            UnitSpec::Projection(spec) => {
                UnitModule::Projection(if matches!(phase, RnntPhase::Output) {
                    if let Some(selected) = &self.parallel {
                        (selected.output)(spec, selected.local.range(d), context)?
                    } else {
                        B::linear(spec, context)?
                    }
                } else {
                    B::linear(spec, context)?
                })
            }
        };
        Ok(RnntUnit {
            dimensions: d,
            group,
            index,
            placement: self.parallel.as_ref().map(|selected| selected.local),
            module,
        })
    }
    fn begin_input<'a>(
        &mut self,
        input: RnntInput<'a, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, RnntForward<B::Tensor>>, RnntError> {
        let (group, input, previous) = match input {
            RnntInput::Embed(token) => {
                if token > self.dimensions.vocab_size as usize {
                    return Err(RnntError::InvalidState(
                        "RNNT embedding token exceeds vocabulary",
                    ));
                }
                (
                    0,
                    B::Tensor::from_i32_slice(&[token as i32], &[1], context)?,
                    None,
                )
            }
            RnntInput::Predict { input, previous } => {
                if input.shape() != [1, self.dimensions.predictor_width] {
                    return Err(RnntError::InvalidState(
                        "RNNT predictor expects [1,predictor_width]",
                    ));
                }
                if let Some(previous) = previous {
                    previous.validate(self.dimensions, &(0..self.dimensions.predictor_layers))?;
                }
                (1, input.clone(), previous.cloned())
            }
            RnntInput::Encode(input) => (2, input.clone(), None),
            RnntInput::Prediction(input) => (3, input.clone(), None),
            RnntInput::Output(input) => (4, input.clone(), None),
        };
        let phase = Self::selection()
            .phases()
            .iter()
            .copied()
            .find(|phase| phase.ordinal() == group)
            .ok_or(RnntError::InvalidState(
                "RNNT operation is absent from the selected component",
            ))?;
        if let UnitSpec::Projection(spec) = self.declaration.unit(phase, 0)? {
            if input.shape() != [1, spec.input] {
                return Err(RnntError::InvalidState(
                    "RNNT joint input has incompatible row geometry",
                ));
            }
        }
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: RnntForward {
                dimensions: self.dimensions,
                group,
                input,
                previous,
                candidate: LstmState {
                    layers: Vec::with_capacity(self.dimensions.predictor_layers),
                },
                completed: 0,
            },
        })
    }

    fn execute_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut RnntUnit<B>,
        hidden: &B::Tensor,
        forward: &mut RnntForward<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        if parallel.is_some() != self.is_parallel() {
            return Err(RnntError::InvalidConfiguration(
                "RNNT invocation differs from selected collective execution",
            ));
        }
        if unit.dimensions != self.dimensions
            || unit.placement != self.parallel.as_ref().map(|selected| selected.local)
            || unit.group != group
            || unit.index != index
            || group != forward.group
            || index != forward.completed
        {
            return Err(RnntError::InvalidState(
                "leased RNNT unit differs from phase geometry or order",
            ));
        }
        let output = match &mut unit.module {
            UnitModule::Embedding(embedding) if group == 0 => {
                if let Some(parallel) = parallel {
                    (self
                        .parallel
                        .as_ref()
                        .expect("selected parallel RNNT")
                        .lookup)(
                        embedding,
                        hidden,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?
                } else {
                    embedding.forward(hidden, context)?
                }
            }
            UnitModule::Predictor(predictor) if group == 1 => {
                let previous = forward
                    .previous
                    .as_ref()
                    .map(|parent| &parent.local_state().layers[index]);
                let next =
                    step_predictor_layer(predictor, hidden, previous, &forward.input, context)?;
                let output = next.hidden.clone();
                forward.candidate.layers.push(next);
                output
            }
            UnitModule::Projection(projection) if group >= 2 => {
                let input = if group == 2 {
                    let UnitSpec::Projection(spec) =
                        self.declaration.unit(RnntPhase::Encoder, 0)?
                    else {
                        return Err(RnntError::InvalidState(
                            "RNNT encoder owner is not a projection",
                        ));
                    };
                    encoder_projection_input::<B>(projection, &spec.weight.id, hidden, context)?
                } else {
                    hidden.clone()
                };
                if group == 4 {
                    if let Some(parallel) = parallel {
                        (self
                            .parallel
                            .as_ref()
                            .expect("selected parallel RNNT")
                            .project)(projection, &input, parallel, context)?
                    } else {
                        projection.forward(&input, context)?
                    }
                } else {
                    projection.forward(&input, context)?
                }
            }
            _ => {
                return Err(RnntError::InvalidState(
                    "leased RNNT unit has incompatible operation",
                ));
            }
        };
        forward.completed += 1;
        Ok(output)
    }
}

fn graph(phases: &[RnntPhase]) -> Result<ExecutionGraph, RnntError> {
    Ok(ExecutionGraph::new(
        phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                if index == 0 {
                    ExecutionGroupSpec::root(phase.id())
                } else {
                    ExecutionGroupSpec::with_dependencies(phase.id(), [phases[index - 1].id()])
                }
            })
            .collect(),
        phases
            .last()
            .expect("nonempty semantic phase selection")
            .id(),
    )?)
}
pub(super) fn declared_parameter_description(
    phases: &[RnntPhase],
    declaration: &Declaration,
) -> Result<ArchitectureParameterDescription, RnntError> {
    let graph = graph(phases)?;
    let counts = phases
        .iter()
        .map(|phase| declaration.phase(*phase).map(|units| units.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = ExecutionUnitLayout::new(&graph, counts)?;
    let mut owned = Vec::new();
    for (local_group, phase) in phases.iter().enumerate() {
        let count = declaration.phase(*phase)?.len();
        for index in 0..count {
            let name = phase.parameter_group(index);
            let role = if phase.vocabulary() {
                ParameterRole::Vocabulary
            } else {
                ParameterRole::Replicated
            };
            let spec = eredu_runtime::logical_parameter_group(
                name,
                role,
                None,
                declaration.unit(*phase, index)?,
                |_| {
                    if phase.vocabulary() {
                        MemberSharding::Balanced { axis: 0 }
                    } else {
                        MemberSharding::Replicated
                    }
                },
            )?;
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout
                        .group_id(local_group)
                        .expect("known RNNT group")
                        .clone(),
                    index,
                ),
                spec,
            ));
        }
    }
    Ok(ArchitectureParameterDescription::from_owned(
        &graph, &layout, owned,
    )?)
}

impl<B: NeuralBackend, const PREDICTOR: bool> ComponentKernel<B> for RnntKernel<B, PREDICTOR> {
    type Input<'a> = RnntInput<'a, B::Tensor>;
    type Unit = RnntUnit<B>;
    type ForwardContext = RnntForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = RnntError;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, RnntError> {
        if state.global_layer_offset() != 0
            || *state.layout() != StatelessComponentState::state_layout()?
        {
            return Err(RnntError::InvalidState(
                "RNNT traversal state identity mismatch",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            match Self::selection() {
                RnntSelection::Complete => "voicechat_rnnt",
                RnntSelection::Projections => "rnnt_projections",
            },
            match Self::selection() {
                RnntSelection::Complete => format!(
                    "{:?}:{:?}",
                    self.dimensions,
                    self.parallel.as_ref().map(|selected| &selected.layout)
                ),
                RnntSelection::Projections => format!("{:?}", self.dimensions),
            },
            1,
            0,
            0,
            topology,
        )?)
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntUnit<B>, RnntError> {
        self.build_resource(
            Self::selection().phases()[address.group()],
            address.index(),
            context,
        )
    }

    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        Self::selection().phases()[group].ordinal() == forward.group
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut roots = vec![&forward.input];
        if let Some(previous) = &forward.previous {
            previous.visit_retained_tensors(&mut |value| roots.push(value));
        }
        for layer in &forward.candidate.layers {
            roots.extend([&layer.hidden, &layer.cell]);
        }
        roots.into_iter()
    }
}
impl<B: NeuralBackend, const PREDICTOR: bool> ComponentEquations<B, StatelessComponentState>
    for RnntKernel<B, PREDICTOR>
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut StatelessComponentState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        if self.is_parallel() {
            return Err(RnntError::InvalidConfiguration(
                "rank-local RNNT requires collective execution",
            ));
        }
        self.begin_input(input, context)
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut StatelessComponentState,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1..=4, [one]) => Ok((*one).clone()),
            _ => Err(RnntError::InvalidConfiguration(
                "RNNT graph dependencies mismatch",
            )),
        }
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut RnntUnit<B>,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.execute_unit(
            Self::selection().phases()[address.group()].ordinal(),
            address.index(),
            unit,
            hidden,
            forward,
            None,
            context,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        if forward.completed
            != if forward.group == RnntPhase::Predictor.ordinal() {
                self.dimensions.predictor_layers
            } else {
                1
            }
        {
            return Err(RnntError::InvalidState(
                "RNNT traversal omitted selected neural units",
            ));
        }
        Ok(hidden.clone())
    }
}

pub(crate) fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<RnntError, P>,
) -> RnntError {
    match error {
        eredu_runtime::LayerwiseRuntimeError::Architecture(error) => error,
        eredu_runtime::LayerwiseRuntimeError::State(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Layout(error) => error.into(),
        eredu_runtime::LayerwiseRuntimeError::Policy(error) => {
            BackendFailure::from_error(error).into()
        }
        error @ eredu_runtime::LayerwiseRuntimeError::Submission(_) => {
            BackendFailure::from_error(error).into()
        }
    }
}
impl<B, P> RnntExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RnntDimensions {
        self.architecture().kernel().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(
            RnntInput::Embed(token),
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)
    }
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        self.invoke_with_context(
            RnntInput::Predict { input, previous },
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)?
        .1
        .into_predictor()
    }
    fn encode_frame(
        &mut self,
        encoder: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(
            RnntInput::Encode(encoder),
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)
    }
    fn project_prediction(
        &mut self,
        prediction: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(
            RnntInput::Prediction(prediction),
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)
    }
    fn project_output(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(
            RnntInput::Output(hidden),
            &mut StatelessComponentState::new()?,
            context,
        )
        .map_err(runtime_error)
    }
}

impl<B: DistributedNeuralBackend, const PREDICTOR: bool> RnntKernel<B, PREDICTOR> {
    /// Selects exact vocabulary shards and recurrent replicas without native allocation.
    pub fn new_parallel(
        declaration: RnntDeclaration<PREDICTOR>,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntError> {
        let local = declaration.local_dimensions(layout)?;
        let mut architecture = Self::new(declaration)?;
        architecture.parallel = Some(ParallelRnnt {
            local,
            layout: layout.clone(),
            embedding: B::vocabulary_parallel_embedding,
            output: B::vocabulary_parallel_linear,
            lookup: B::vocabulary_parallel_lookup,
            project: B::vocabulary_parallel_project,
        });
        Ok(architecture)
    }
    /// Validates actual rank, degree, row ownership and all replicas before acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), RnntError> {
        let selected = self
            .parallel
            .as_ref()
            .ok_or(RnntError::InvalidConfiguration(
                "collective RNNT requires rank-local construction",
            ))?;
        selected
            .local
            .range(self.dimensions)
            .balanced_peer_widths(B::parallel_size(parallel), B::parallel_rank(parallel))?;
        eredu_runtime::validate_tensor_parallel_layout(
            "voicechat.rnnt",
            self.definition.parameters(),
            &selected.layout,
            B::parallel_size(parallel),
            B::parallel_rank(parallel),
        )?;
        Ok(())
    }
}
impl<B: DistributedNeuralBackend, const PREDICTOR: bool>
    ParallelComponentKernel<B, eredu_runtime::StatelessComponentState>
    for RnntKernel<B, PREDICTOR>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut StatelessComponentState,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        self.validate_parallel(parallel)?;
        self.begin_input(input, context)
    }
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.execute_unit(
            Self::selection().phases()[address.group()].ordinal(),
            address.index(),
            unit,
            hidden,
            forward,
            Some(parallel),
            context,
        )
    }
}
/// Tensor-parallel RNNT phases with complete recurrent ownership.
/// The typed invocation uses the shared resource and completion lifecycle.
pub trait ParallelRnntExecution<B: DistributedNeuralBackend>: RnntExecution<B> {
    /// Executes one selected phase and returns its unpublished request-local state.
    fn invoke_parallel(
        &mut self,
        input: RnntInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, RnntForward<B::Tensor>), RnntError>;
    /// Looks up and sums the selected vocabulary row across exact owners.
    fn embed_token_parallel(
        &mut self,
        token: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Embed(token), parallel, context)
            .map(|(value, _)| value)
    }
    /// Advances fully replicated recurrent layers through the same leased traversal.
    fn predictor_step_parallel(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        self.invoke_parallel(RnntInput::Predict { input, previous }, parallel, context)?
            .1
            .into_predictor()
    }
    /// Applies the replicated perception projection in its learned precision.
    fn encode_frame_parallel(
        &mut self,
        encoder: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Encode(encoder), parallel, context)
            .map(|(value, _)| value)
    }
    /// Applies the replicated predictor projection without another recurrent step.
    fn project_prediction_parallel(
        &mut self,
        prediction: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Prediction(prediction), parallel, context)
            .map(|(value, _)| value)
    }
    /// Gathers complete logits, including the blank row, from uneven vocabulary owners.
    fn project_output_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Output(hidden), parallel, context)
            .map(|(value, _)| value)
    }
}

impl<B, P> ParallelRnntExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, RnntKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn invoke_parallel(
        &mut self,
        input: RnntInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, RnntForward<B::Tensor>), RnntError> {
        self.forward_parallel_with_context_hook(
            input,
            &mut StatelessComponentState::new()?,
            parallel,
            context,
            |_, _, _| Ok(()),
        )
        .map_err(runtime_error)
    }
}
