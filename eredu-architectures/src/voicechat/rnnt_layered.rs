//! Independently leased RNNT embedding, recurrent layers and joint affines.
//!
//! Each invocation owns only an unpublished neural phase. The shared RNNT
//! execution trait owns priming, hypothesis capacity and the greedy label loop.

use super::rnnt::{
    affine, build_predictor_layer, encoder_projection_input, parameter, step_predictor_layer,
    RnntDimensions, RnntError, RnntExecution, RnntPredictionState,
};
use eredu_core::{cache::LayerCachePolicy, BackendFailure, LayerSchedule};
use eredu_nn::{
    recurrent::{Lstm, LstmState},
    DistributedNeuralBackend, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    LinearOperator, LinearSpec, NeuralBackend, Parameterized, RetainedTensorRoots, Tensor,
    VocabularyParallelRange,
};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredArchitecture, LayeredForwardState, LayeredInvocation,
    LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout, MemberSharding,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole, RuntimeState, StateError, StateLayout,
    StaticParameterVisitor, StaticParameterVisitorMut, SubmissionBackend, TensorPlacement,
};
use std::marker::PhantomData;

const GROUPS: [&str; 5] = [
    "voicechat.rnnt.embedding",
    "voicechat.rnnt.predictor",
    "voicechat.rnnt.joint.enc",
    "voicechat.rnnt.joint.pred",
    "voicechat.rnnt.joint.joint_net.2",
];
const EMBEDDING: &str = "stt_model.rnnt_decoder.prediction.embed.weight";
const PREDICTOR: &str = "stt_model.rnnt_decoder.prediction.dec_rnn.lstm";

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
/// No learned RNNT parameter is permanently resident outside the unit provider.
pub struct RnntStaticModules;
impl<T: 'static> Parameterized<T> for RnntStaticModules {
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, T>>(&'a self, _: &mut V) {}
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, T>>(&'a mut self, _: &mut V) {}
    fn set_trainable(&mut self, _: bool) {}
}

fn state_layout() -> Result<StateLayout, RnntError> {
    Ok(StateLayout::new(LayerSchedule::new(
        1,
        vec![LayerCachePolicy::NoState],
    )?)?)
}
/// Stateless traversal metadata; caller-owned `RnntState` retains recurrent history.
pub struct RnntRuntimeState {
    layout: StateLayout,
}
impl RnntRuntimeState {
    /// Creates host metadata without backend allocation.
    pub fn new() -> Result<Self, RnntError> {
        Ok(Self {
            layout: state_layout()?,
        })
    }
}
impl<B: NeuralBackend> RuntimeState<B> for RnntRuntimeState {
    type RetainedValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn optional_layout(&self) -> Option<&StateLayout> {
        None
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        if ordinal != 0 {
            return Err(StateError::UnknownLayer {
                layer: ordinal,
                count: 1,
            });
        }
        Ok(std::iter::empty())
    }
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
/// Validates complete physical vocabulary placement and every replicated member.
/// Uneven vocabulary ranges do not uniquely identify a communicator; invocation
/// independently validates the selected rank and degree.
pub fn rnnt_local_dimensions(
    dimensions: RnntDimensions,
    layout: &LocalModelLayout,
) -> Result<RnntLocalDimensions, RnntError> {
    rnnt_local_dimensions_for_description(
        dimensions,
        layout,
        &rnnt_parameter_description(dimensions)?,
    )
}
pub(crate) fn rnnt_local_dimensions_for_description(
    dimensions: RnntDimensions,
    layout: &LocalModelLayout,
    description: &ArchitectureParameterDescription,
) -> Result<RnntLocalDimensions, RnntError> {
    dimensions.validate()?;
    let weight = layout
        .tensor(EMBEDDING)
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
            ))
        }
    };
    let range = VocabularyParallelRange {
        global_vocabulary: (dimensions.vocab_size + 1) as usize,
        local: start..end,
    };
    range.validate()?;
    let local = RnntLocalDimensions {
        vocabulary_start: start as i32,
        vocabulary_end: end as i32,
    };
    let mut expected = LocalModelLayout::default();
    for owned in description.groups() {
        let group = owned.group();
        for member in group.members() {
            let mut shape = member.global_shape().to_vec();
            let placement = match member.sharding() {
                MemberSharding::Balanced { axis: 0 } if shape[0] == range.global_vocabulary => {
                    shape[0] = range.local.len();
                    TensorPlacement::Range {
                        axis: 0,
                        start,
                        end,
                    }
                }
                MemberSharding::Replicated => TensorPlacement::Replicated,
                _ => {
                    return Err(RnntError::InvalidConfiguration(
                        "RNNT member has incompatible vocabulary placement",
                    ))
                }
            };
            expected.insert(
                member.target().into(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    shape,
                    placement,
                    None,
                    None,
                    false,
                ),
            );
        }
    }
    if &expected != layout {
        return Err(RnntError::InvalidConfiguration(
            "RNNT physical layout differs from vocabulary and replica declarations",
        ));
    }
    Ok(local)
}
// Function pointers preserve ordinary construction for neutral backends without
// requiring distributed mechanisms until parallel architecture selection.
struct ParallelRnnt<B: NeuralBackend> {
    local: RnntLocalDimensions,
    layout: LocalModelLayout,
    description: ArchitectureParameterDescription,
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
pub struct RnntArchitecture<B: NeuralBackend> {
    dimensions: RnntDimensions,
    parallel: Option<ParallelRnnt<B>>,
    empty: RnntStaticModules,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> RnntArchitecture<B> {
    /// Validates geometry without allocating any native parameter.
    pub fn new(dimensions: RnntDimensions) -> Result<Self, RnntError> {
        dimensions.validate()?;
        Ok(Self {
            dimensions,
            parallel: None,
            empty: RnntStaticModules,
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
    /// Stable phase ordering; inactive phases pass their input through unchanged.
    pub fn execution_graph(&self) -> Result<ExecutionGraph, RnntError> {
        graph()
    }
    /// Each affine is one unit and each predictor layer is one unit.
    pub fn group_unit_count(&self, group: usize) -> Result<usize, RnntError> {
        match group {
            1 => Ok(self.dimensions.predictor_layers),
            0 | 2..=4 => Ok(1),
            _ => Err(RnntError::InvalidConfiguration(
                "unknown RNNT execution group",
            )),
        }
    }
    /// Canonical checkpoint identity of this independently leased owner.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, RnntError> {
        if index >= self.group_unit_count(group)? {
            return Err(RnntError::InvalidConfiguration("RNNT unit exceeds group"));
        }
        Ok(match group {
            0 => EMBEDDING.into(),
            1 => format!("{PREDICTOR}.layer{index}"),
            2 => "stt_model.rnnt_joint.enc".into(),
            3 => "stt_model.rnnt_joint.pred".into(),
            4 => "stt_model.rnnt_joint.joint_net.2".into(),
            _ => unreachable!(),
        })
    }
    /// Declares only one requested embedding, recurrent layer or joint affine.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntUnit<B>, RnntError> {
        self.unit_path(group, index)?;
        let d = self.dimensions;
        let module = match group {
            0 => {
                let spec = EmbeddingSpec {
                    vocabulary: d.vocab_size + 1,
                    dimensions: d.predictor_width,
                    weight: parameter(EMBEDDING.into(), GROUPS[0]),
                    format: eredu_nn::LinearFormatSpec::unscaled(
                        eredu_checkpoint::LinearFormat::Dense,
                    )
                    .expect("dense RNNT embedding"),
                };
                UnitModule::Embedding(if let Some(selected) = &self.parallel {
                    (selected.embedding)(spec, selected.local.range(d), context)?
                } else {
                    B::embedding(spec, context)?
                })
            }
            1 => UnitModule::Predictor(build_predictor_layer(d, index, context)?),
            2..=4 => {
                let (input, output) = projection_shape(d, group);
                let path = self.unit_path(group, index)?;
                let spec = affine(
                    input,
                    output,
                    format!("{path}.weight"),
                    format!("{path}.bias"),
                    GROUPS[group],
                );
                UnitModule::Projection(if group == 4 {
                    if let Some(selected) = &self.parallel {
                        (selected.output)(spec, selected.local.range(d), context)?
                    } else {
                        B::linear(spec, context)?
                    }
                } else {
                    B::linear(spec, context)?
                })
            }
            _ => unreachable!(),
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
        state: &mut RnntRuntimeState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, RnntForward<B::Tensor>>, RnntError> {
        if state.layout != state_layout()? {
            return Err(RnntError::InvalidState("RNNT traversal layout mismatch"));
        }
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
        if group >= 2 && input.shape() != [1, projection_shape(self.dimensions, group).0] {
            return Err(RnntError::InvalidState(
                "RNNT joint input has incompatible row geometry",
            ));
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
        self.unit_path(group, index)?;
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
                    encoder_projection_input::<B>(projection, hidden, context)?
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
                ))
            }
        };
        forward.completed += 1;
        Ok(output)
    }
}

fn projection_shape(d: RnntDimensions, group: usize) -> (i32, i32) {
    match group {
        2 => (d.encoder_width, d.joint_width),
        3 => (d.predictor_width, d.joint_width),
        4 => (d.joint_width, d.vocab_size + 1),
        _ => unreachable!(),
    }
}
fn graph() -> Result<ExecutionGraph, RnntError> {
    Ok(ExecutionGraph::new(
        GROUPS
            .iter()
            .enumerate()
            .map(|(index, name)| {
                if index == 0 {
                    ExecutionGroupSpec::root(*name)
                } else {
                    ExecutionGroupSpec::with_dependencies(*name, [GROUPS[index - 1]])
                }
            })
            .collect(),
        GROUPS[4],
    )?)
}
/// Cold exact geometry, canonical identities and separate ownership of all RNNT parameters.
pub fn rnnt_parameter_description(
    d: RnntDimensions,
) -> Result<ArchitectureParameterDescription, RnntError> {
    d.validate()?;
    let graph = graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [1, d.predictor_layers, 1, 1, 1])?;
    let mut specs = Vec::new();
    let mut owned = Vec::new();
    let member = |name: String, shape: Vec<usize>| {
        let vocabulary = name == EMBEDDING || name.starts_with("stt_model.rnnt_joint.joint_net.2.");
        ParameterMemberSpec::new(
            name,
            shape,
            if vocabulary {
                MemberSharding::Balanced { axis: 0 }
            } else {
                MemberSharding::Replicated
            },
        )
    };
    for (group, group_name) in GROUPS.iter().enumerate() {
        let count = if group == 1 { d.predictor_layers } else { 1 };
        for index in 0..count {
            let (name, members) = match group {
                0 => (
                    GROUPS[0].into(),
                    vec![member(
                        EMBEDDING.into(),
                        vec![(d.vocab_size + 1) as usize, d.predictor_width as usize],
                    )],
                ),
                1 => {
                    let mut members = Vec::new();
                    for matrix in ["ih", "hh"] {
                        members.push(member(
                            format!("{PREDICTOR}.weight_{matrix}_l{index}"),
                            vec![4 * d.predictor_width as usize, d.predictor_width as usize],
                        ));
                        members.push(member(
                            format!("{PREDICTOR}.bias_{matrix}_l{index}"),
                            vec![4 * d.predictor_width as usize],
                        ));
                    }
                    (format!("{}.{}", GROUPS[1], index), members)
                }
                _ => {
                    let (input, output) = projection_shape(d, group);
                    let path = match group {
                        2 => "enc",
                        3 => "pred",
                        4 => "joint_net.2",
                        _ => unreachable!(),
                    };
                    (
                        (*group_name).into(),
                        vec![
                            member(
                                format!("stt_model.rnnt_joint.{path}.weight"),
                                vec![output as usize, input as usize],
                            ),
                            member(
                                format!("stt_model.rnnt_joint.{path}.bias"),
                                vec![output as usize],
                            ),
                        ],
                    )
                }
            };
            let role = if group == 0 || group == 4 {
                ParameterRole::Vocabulary
            } else {
                ParameterRole::Replicated
            };
            let spec = ParameterGroupSpec::new(name, role, members)?;
            specs.push(spec.clone());
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    layout.group_id(group).expect("known RNNT group").clone(),
                    index,
                ),
                spec,
            ));
        }
    }
    Ok(ArchitectureParameterDescription::new(
        &graph, &layout, specs, owned,
    )?)
}
impl<B: NeuralBackend> ArchitectureParameters<B> for RnntArchitecture<B> {
    type DefinitionError = RnntError;
    fn state_layout(&self) -> Result<StateLayout, RnntError> {
        state_layout()
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, RnntError> {
        if state.global_layer_offset() != 0 || *state.layout() != state_layout()? {
            return Err(RnntError::InvalidState(
                "RNNT traversal state identity mismatch",
            ));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "nemotron_voicechat",
            "voicechat_rnnt",
            format!(
                "{:?}:{:?}",
                self.dimensions,
                self.parallel.as_ref().map(|selected| &selected.layout)
            ),
            1,
            0,
            0,
            topology,
        )?)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, RnntError> {
        rnnt_parameter_description(self.dimensions)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}
impl<B: NeuralBackend> LayeredArchitecture<B, RnntRuntimeState> for RnntArchitecture<B> {
    type Input<'a> = RnntInput<'a, B::Tensor>;
    type StaticModules = RnntStaticModules;
    type Unit = RnntUnit<B>;
    type ForwardContext = RnntForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = RnntError;
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Projector,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }
    fn primary_execution_group(&self) -> &str {
        GROUPS[0]
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, RnntError> {
        self.execution_graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, RnntError> {
        self.group_unit_count(group)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, RnntError> {
        self.unit_path(group, index)
    }
    fn static_modules(&self) -> &RnntStaticModules {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut RnntStaticModules {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntUnit<B>, RnntError> {
        self.build_unit(group, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntRuntimeState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        if self.is_parallel() {
            return Err(RnntError::InvalidConfiguration(
                "rank-local RNNT requires collective execution",
            ));
        }
        self.begin_input(input, state, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut RnntRuntimeState,
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
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == forward.group
    }
    fn retained_state_ordinals(&self, _: usize, _: usize, _: usize) -> std::ops::Range<usize> {
        0..0
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut RnntUnit<B>,
        hidden: &B::Tensor,
        _: &mut RnntRuntimeState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.execute_unit(group, index, unit, hidden, forward, None, context)
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut RnntRuntimeState,
        forward: &Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        if forward.completed != self.group_unit_count(forward.group)? {
            return Err(RnntError::InvalidState(
                "RNNT traversal omitted selected neural units",
            ));
        }
        Ok(hidden.clone())
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
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

/// RNNT neural mechanisms using the ordinary bounded parameter provider lifecycle.
pub struct LayerwiseRnnt<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
{
    runtime: LayerwiseRuntime<RnntArchitecture<B>, B, RnntRuntimeState, P>,
}
impl<B, P> LayerwiseRnnt<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::fmt::Display,
{
    /// Adopts an exact provider of independently materializable RNNT units.
    pub fn new(architecture: RnntArchitecture<B>, policy: P) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
        }
    }
    /// Shared runtime and its retained provider accounting.
    pub fn runtime(&self) -> &LayerwiseRuntime<RnntArchitecture<B>, B, RnntRuntimeState, P> {
        &self.runtime
    }
    /// Shared runtime for provider inspection and parameter operations.
    pub fn runtime_mut(
        &mut self,
    ) -> &mut LayerwiseRuntime<RnntArchitecture<B>, B, RnntRuntimeState, P> {
        &mut self.runtime
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
impl<B, P> RnntExecution<B> for LayerwiseRnnt<B, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn dimensions(&self) -> RnntDimensions {
        self.runtime.architecture().dimensions()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntInput::Embed(token),
                &mut RnntRuntimeState::new()?,
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
        self.runtime
            .invoke_with_context(
                RnntInput::Predict { input, previous },
                &mut RnntRuntimeState::new()?,
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
        self.runtime
            .invoke(
                RnntInput::Encode(encoder),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_prediction(
        &mut self,
        prediction: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntInput::Prediction(prediction),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
    fn project_output(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.runtime
            .invoke(
                RnntInput::Output(hidden),
                &mut RnntRuntimeState::new()?,
                context,
            )
            .map_err(runtime_error)
    }
}

impl<B: DistributedNeuralBackend> RnntArchitecture<B> {
    /// Selects exact vocabulary shards and recurrent replicas without native allocation.
    pub fn new_parallel(
        dimensions: RnntDimensions,
        layout: &LocalModelLayout,
    ) -> Result<Self, RnntError> {
        Self::new_parallel_for_description(
            dimensions,
            layout,
            rnnt_parameter_description(dimensions)?,
        )
    }
    pub(crate) fn new_parallel_for_description(
        dimensions: RnntDimensions,
        layout: &LocalModelLayout,
        description: ArchitectureParameterDescription,
    ) -> Result<Self, RnntError> {
        let local = rnnt_local_dimensions_for_description(dimensions, layout, &description)?;
        let mut architecture = Self::new(dimensions)?;
        architecture.parallel = Some(ParallelRnnt {
            local,
            layout: layout.clone(),
            description,
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
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| RnntError::InvalidConfiguration("invalid RNNT tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| RnntError::InvalidConfiguration("invalid RNNT tensor rank"))?;
        let expected =
            eredu_runtime::derive_partitioned_local_layout(&selected.description, topology)
                .map_err(|_| {
                    RnntError::InvalidConfiguration("RNNT tensor placement cannot be derived")
                })?;
        if expected != selected.layout {
            return Err(RnntError::InvalidConfiguration(
                "RNNT layout differs from selected communicator",
            ));
        }
        Ok(())
    }
}
impl<B: DistributedNeuralBackend> ParallelLayeredArchitecture<B, RnntRuntimeState>
    for RnntArchitecture<B>
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut RnntRuntimeState,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, RnntError> {
        self.validate_parallel(parallel)?;
        self.begin_input(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut RnntRuntimeState,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.execute_unit(group, index, unit, hidden, forward, Some(parallel), context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut RnntRuntimeState,
        forward: &Self::ForwardContext,
        _: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.finish_forward(hidden, state, forward, context)
    }
}
impl<B, P> LayerwiseRnnt<B, P>
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
        self.runtime
            .forward_parallel_with_context_hook(
                input,
                &mut RnntRuntimeState::new()?,
                parallel,
                context,
                |_, _, _| Ok(()),
            )
            .map_err(runtime_error)
    }
    /// Looks up and sums the selected vocabulary row across exact owners.
    pub fn embed_token_parallel(
        &mut self,
        token: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Embed(token), parallel, context)
            .map(|(value, _)| value)
    }
    /// Advances fully replicated recurrent layers through the same leased traversal.
    pub fn predictor_step_parallel(
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
    pub fn encode_frame_parallel(
        &mut self,
        encoder: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Encode(encoder), parallel, context)
            .map(|(value, _)| value)
    }
    /// Applies the replicated predictor projection without another recurrent step.
    pub fn project_prediction_parallel(
        &mut self,
        prediction: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Prediction(prediction), parallel, context)
            .map(|(value, _)| value)
    }
    /// Gathers complete logits, including the blank row, from uneven vocabulary owners.
    pub fn project_output_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke_parallel(RnntInput::Output(hidden), parallel, context)
            .map(|(value, _)| value)
    }
}
