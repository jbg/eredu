//! Independently leased residual codebooks over the shared layered traversal.
use crate::{
    prvq::{
        checked_rows, invalid, lookup_codes, quantize_residual_with_score_reduction, require_f32,
        Codebook, DistanceMetric, Geometry, Quantized,
    },
    Error,
};
use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, Index, NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ComponentAdapter, ComponentDefinition, ComponentEquations,
    ComponentKernel, ComponentUnit, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitAddress,
    ExecutionUnitLayout, LayeredForwardState, LayeredInvocation, LayeredTraversalHook,
    LayerwisePolicy, LayerwiseRuntime, LocalModelLayout, MemberSharding, OwnedParameterGroupSpec,
    ParallelComponentKernel, ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole, StatelessComponentState, SubmissionBackend,
};
use std::marker::PhantomData;
use std::ops::Range;

/// Coherent channel slice shared by every residual codebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerLocalDimensions {
    /// First globally indexed channel.
    pub channels_start: i32,
    /// Number of locally materialized channels.
    pub channels: i32,
}

/// Validates every physical mean and replicated variance against one exact channel slice.
pub fn quantizer_local_dimensions(
    geometry: Geometry,
    prefix: &str,
    layout: &LocalModelLayout,
) -> Result<QuantizerLocalDimensions, Error> {
    let description = quantizer_parameter_description(geometry, prefix)?;
    let range = layout
        .tensor(&format!("{prefix}.mus_list.0"))
        .ok_or_else(|| invalid("quantizer layout omits first codebook"))?
        .expanded_logical_range(geometry.channels as usize)?;
    if range.is_empty() || range.end > geometry.channels as usize {
        return Err(invalid("invalid quantizer channel partition"));
    }
    eredu_runtime::validate_parameter_layout(
        description.groups().iter().map(|owned| owned.group()),
        layout,
        |_| {
            Ok::<_, Error>(eredu_runtime::ParameterGroupSelection::Logical(Some(
                range.clone(),
            )))
        },
    )?;
    Ok(QuantizerLocalDimensions {
        channels_start: range.start as i32,
        channels: range.len() as i32,
    })
}

/// Ordered residual quantization with exact selected-depth parameter acquisition.
pub trait QuantizerExecution<B: NeuralBackend> {
    /// Verifies that calls either contain no collectives or agree every local
    /// failure before a dependent collective. Called before coordinated composition.
    fn validate_coordinated_execution(&self) -> Result<(), Error>;

    /// Geometry shared by the independently leased codebooks.
    fn geometry(&self) -> Geometry;
    /// Decodes only the given nonempty contiguous range, adding means in depth order.
    fn decode_range(
        &mut self,
        codes: &B::Tensor,
        range: Range<usize>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
    /// Encodes only the selected depths, starting with the supplied fresh residual.
    fn quantize_range(
        &mut self,
        latent: &B::Tensor,
        range: Range<usize>,
        metric: DistanceMetric,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error>;
    /// Decodes the prefix described by the code depth axis.
    fn decode(
        &mut self,
        codes: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if codes.shape().len() != 3 || codes.dim(2) <= 0 {
            return Err(invalid("codes must have shape [batch, frames, depth]"));
        }
        self.decode_range(codes, 0..codes.dim(2) as usize, context)
    }
    /// Encodes all codebooks using the complete squared-distance equation.
    fn quantize(
        &mut self,
        latent: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        self.quantize_range(
            latent,
            0..self.geometry().depth,
            DistanceMetric::SquaredEuclidean,
            context,
        )
    }
}
/// Invocation-local operation and selected contiguous codebook range.
pub enum QuantizerInput<'a, T> {
    /// Ordered sum of selected codebook vectors.
    Decode {
        /// Integer codes with the selected range's depth axis.
        codes: &'a T,
        /// Nonempty global depth range.
        range: Range<usize>,
    },
    /// Ordered residual nearest-mean decisions.
    Quantize {
        /// Fresh residual before any selected mean is subtracted.
        latent: &'a T,
        /// Nonempty global depth range.
        range: Range<usize>,
        /// Explicit nearest-mean scoring equation.
        metric: DistanceMetric,
    },
}
/// Immutable source and all temporary roots needed for exact unit completion.
pub struct QuantizerForward<T> {
    input: T,
    range: Range<usize>,
    metric: Option<DistanceMetric>,
    codes: Vec<T>,
    selected: Option<T>,
    result: Option<Quantized<T>>,
}
impl<T> QuantizerForward<T> {
    /// Returns all three outputs only after the selected traversal succeeds.
    pub fn into_quantized(self) -> Result<Quantized<T>, Error> {
        self.result
            .ok_or_else(|| invalid("quantizer traversal omitted residual result"))
    }
}
/// Codec-owned canonical codebook order and parameter namespaces.
pub struct QuantizerKernel<B: NeuralBackend> {
    geometry: Geometry,
    prefix: String,
    parallel: Option<(QuantizerLocalDimensions, LocalModelLayout)>,
    backend: PhantomData<fn() -> B>,
}
impl<B: NeuralBackend> QuantizerKernel<B> {
    /// Validates geometry and namespace without constructing backend resources.
    pub fn new(geometry: Geometry, prefix: &str) -> Result<Self, Error> {
        geometry.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty residual quantizer parameter prefix"));
        }
        Ok(Self {
            geometry,
            prefix: prefix.into(),
            parallel: None,
            backend: PhantomData,
        })
    }
    /// Constructs rank-local codebooks from a complete physical placement declaration.
    pub fn new_parallel(
        geometry: Geometry,
        prefix: &str,
        layout: &LocalModelLayout,
    ) -> Result<Self, Error> {
        let local = quantizer_local_dimensions(geometry, prefix, layout)?;
        let mut architecture = Self::new(geometry, prefix)?;
        architecture.parallel = Some((local, layout.clone()));
        Ok(architecture)
    }
    /// Whether construction selected tensor-parallel parameter placement.
    pub fn is_parallel(&self) -> bool {
        self.parallel.is_some()
    }
    fn local_geometry(&self) -> Geometry {
        Geometry {
            channels: self
                .parallel
                .as_ref()
                .map_or(self.geometry.channels, |(local, _)| local.channels),
            ..self.geometry
        }
    }
    /// Exact independently leased codebook geometry.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }
    /// Couples codec equations with the validated shared component definition.
    pub fn into_component(self) -> Result<ComponentAdapter<ComponentDefinition, Self>, Error> {
        Ok(ComponentAdapter::new(
            quantizer_component_definition(self.geometry, &self.prefix)?,
            self,
        )?)
    }
    /// Constructs only the requested unloaded codebook.
    pub fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Codebook<B::Tensor>, Error> {
        if group >= self.geometry.depth || index != 0 {
            return Err(invalid("unknown residual codebook unit"));
        }
        Codebook::unloaded(self.local_geometry(), &self.prefix, group, context)
    }
}
impl<B: NeuralBackend> QuantizerKernel<B> {
    fn begin_input(
        &self,
        input: QuantizerInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, QuantizerForward<B::Tensor>>, Error> {
        let (input, range, metric) = match input {
            QuantizerInput::Decode { codes, range } => (codes, range, None),
            QuantizerInput::Quantize {
                latent,
                range,
                metric,
            } => (latent, range, Some(metric)),
        };
        if range.start >= range.end || range.end > self.geometry.depth {
            return Err(invalid("invalid nonempty residual codebook range"));
        }
        if input.shape().len() != 3 {
            return Err(invalid("residual quantizer requires rank-three input"));
        }
        checked_rows(input.dim(0), input.dim(1))?;
        let hidden = if metric.is_some() {
            require_f32(input)?;
            if input.dim(2) != self.geometry.channels {
                return Err(invalid("latent width does not match residual quantizer"));
            }
            if let Some((local, _)) = &self.parallel {
                input.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::Range(local.channels_start, local.channels_start + local.channels),
                    ],
                    context,
                )?
            } else {
                input.clone()
            }
        } else {
            if input.dim(2) as usize != range.len()
                || !matches!(
                    input.element_type(),
                    Some(
                        TensorElementType::I32
                            | TensorElementType::U32
                            | TensorElementType::I64
                            | TensorElementType::U64
                    )
                )
            {
                return Err(invalid(
                    "code depth or integer dtype does not match selected codebook range",
                ));
            }
            B::Tensor::full_f32(
                0.0,
                &[input.dim(0), input.dim(1), self.local_geometry().channels],
                context,
            )?
        };
        let codes = Vec::with_capacity(if metric.is_some() { range.len() } else { 0 });
        Ok(LayeredForwardState {
            hidden,
            context: QuantizerForward {
                input: input.clone(),
                range,
                metric,
                codes,
                selected: None,
                result: None,
            },
        })
    }
}

impl<B: NeuralBackend> QuantizerKernel<B> {
    #[allow(clippy::too_many_arguments)]
    fn forward_selected(
        &self,
        group: usize,
        index: usize,
        unit: &mut Codebook<B::Tensor>,
        hidden: &B::Tensor,
        forward: &mut QuantizerForward<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        mut collective: impl FnMut(bool, Result<B::Tensor, Error>) -> Result<B::Tensor, Error>,
    ) -> Result<B::Tensor, Error> {
        let validated = (|| {
            if group >= self.geometry.depth || index != 0 {
                return Err(invalid("unknown residual codebook unit"));
            }
            if unit.geometry() != self.local_geometry()
                || unit.depth() != group
                || unit.prefix() != self.prefix
            {
                return Err(invalid("leased residual codebook identity mismatch"));
            }
            unit.validate_bound()
        })();
        if let Some(metric) = forward.metric {
            let stage = match validated {
                Ok(()) => quantize_residual_with_score_reduction(
                    hidden,
                    unit.means.as_ref(),
                    metric,
                    context,
                    |prepared| collective(false, prepared),
                ),
                Err(error) => collective(false, Err(error))
                    .and_then(|_| Err(invalid("collective accepted failed codebook preparation"))),
            };
            let residual = stage.map(|stage| {
                forward.codes.push(stage.codes);
                forward.selected = Some(stage.selected);
                stage.residual
            });
            if group + 1 == forward.range.end {
                let residual = collective(true, residual)?;
                forward.result = Some(Quantized {
                    codes: B::Tensor::stack(&forward.codes, -1, context)?,
                    quantized: forward.input.subtract(&residual, context)?,
                    residual: residual.clone(),
                });
                Ok(residual)
            } else {
                residual
            }
        } else {
            let output = (|| {
                validated?;
                let indices = forward.input.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::At((group - forward.range.start) as i32),
                    ],
                    context,
                )?;
                let selected = lookup_codes(unit.means.as_ref(), &indices, context)?;
                let output = hidden.add(&selected, context)?;
                forward.selected = Some(selected);
                Ok(output)
            })();
            if group + 1 == forward.range.end {
                collective(true, output)
            } else {
                output
            }
        }
    }
}

/// Stable execution group for one independently leased residual codebook depth.
pub fn codebook_group_name(depth: usize) -> String {
    format!("prvq.codebook.{depth}")
}
/// Validated selected-depth invocations over independently leased codebooks.
pub fn quantizer_component_definition(
    geometry: Geometry,
    prefix: &str,
) -> Result<ComponentDefinition, Error> {
    let parameters = quantizer_parameter_description(geometry, prefix)?;
    let transport = ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::Projector,
        first_owner_static_roles: vec![],
        last_owner_static_roles: vec![],
        merge_destination: ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: None,
        request_optional: false,
    };
    Ok(ComponentDefinition::new(
        "prvq",
        parameters,
        (0..geometry.depth)
            .map(|depth| ComponentUnit {
                path: format!("{prefix}.codebook.{depth}"),
                state: 0..0,
            })
            .collect(),
        vec![transport; geometry.depth],
        "prvq.codebook.0",
        vec![],
        StatelessComponentState::state_layout()?,
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::output_owner(0..1)]),
    )?)
}

fn quantizer_graph(geometry: Geometry) -> Result<ExecutionGraph, Error> {
    geometry.validate()?;
    let groups: Vec<_> = (0..geometry.depth)
        .map(|depth| {
            if depth == 0 {
                ExecutionGroupSpec::root(codebook_group_name(depth))
            } else {
                ExecutionGroupSpec::with_dependencies(
                    codebook_group_name(depth),
                    [codebook_group_name(depth - 1)],
                )
            }
        })
        .collect();
    Ok(ExecutionGraph::new(
        groups,
        codebook_group_name(geometry.depth - 1),
    )?)
}
/// Cold canonical means/scalar geometry with one independently owned group per depth.
pub fn quantizer_parameter_description(
    geometry: Geometry,
    prefix: &str,
) -> Result<ArchitectureParameterDescription, Error> {
    geometry.validate()?;
    if prefix.is_empty() {
        return Err(invalid("empty residual quantizer parameter prefix"));
    }
    let graph = quantizer_graph(geometry)?;
    let layout = ExecutionUnitLayout::new(&graph, vec![1; geometry.depth])?;
    let mut owned = Vec::with_capacity(geometry.depth);
    for depth in 0..geometry.depth {
        let spec = ParameterGroupSpec::partitioned(
            format!("{prefix}.codebook.{depth}"),
            ParameterRole::Channels,
            geometry.channels as usize,
            [
                ParameterMemberSpec::new(
                    format!("{prefix}.mus_list.{depth}"),
                    vec![geometry.cardinality as usize, geometry.channels as usize],
                    MemberSharding::Partitioned { axis: 1 },
                ),
                ParameterMemberSpec::new(
                    format!("{prefix}._variance_list.{depth}.variance"),
                    vec![],
                    MemberSharding::Replicated,
                ),
            ],
        )?;
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(
                layout.group_id(depth).expect("declared depth").clone(),
                0,
            ),
            spec,
        ));
    }
    Ok(ArchitectureParameterDescription::from_owned(
        &graph, &layout, owned,
    )?)
}

impl<B: NeuralBackend> ComponentKernel<B> for QuantizerKernel<B> {
    type Input<'a> = QuantizerInput<'a, B::Tensor>;
    type Unit = Codebook<B::Tensor>;
    type ForwardContext = QuantizerForward<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        if state.global_layer_offset() != 0
            || *state.layout() != StatelessComponentState::state_layout()?
        {
            return Err(invalid("residual quantizer stateless layout mismatch"));
        }
        Ok(eredu_runtime::ModelStateIdentity::new(
            "probabilistic_rvq",
            "quantizer",
            format!("{:?}:{}", self.geometry, self.prefix),
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
    ) -> Result<Self::Unit, Error> {
        self.build_unit(address.group(), address.index(), context)
    }

    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        forward.range.contains(&group)
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut roots = Vec::with_capacity(forward.codes.len() + 5);
        roots.push(&forward.input);
        roots.extend(forward.codes.iter());
        roots.extend(forward.selected.as_ref());
        if let Some(result) = &forward.result {
            roots.extend([&result.codes, &result.residual, &result.quantized]);
        }
        roots.into_iter()
    }
}
impl<B: NeuralBackend> ComponentEquations<B, StatelessComponentState> for QuantizerKernel<B> {
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut StatelessComponentState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        if self.is_parallel() {
            return Err(invalid("rank-local quantizer requires parallel execution"));
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
    ) -> Result<B::Tensor, Error> {
        if group == 0 && dependencies.is_empty() {
            Ok(initial.clone())
        } else if group > 0 && group < self.geometry.depth && dependencies.len() == 1 {
            Ok(dependencies[0].clone())
        } else {
            Err(invalid("residual quantizer graph dependency mismatch"))
        }
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.is_parallel() {
            return Err(invalid("rank-local quantizer requires parallel execution"));
        }
        self.forward_selected(
            address.group(),
            address.index(),
            unit,
            hidden,
            forward,
            context,
            |_, prepared| prepared,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _: &mut StatelessComponentState,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        // Final products are composed in the final active unit so its exact completion retains them.
        let _ = context;
        if forward.metric.is_some() && forward.result.is_none() {
            return Err(invalid(
                "residual quantization omitted final selected depth",
            ));
        }
        Ok(hidden.clone())
    }
}

impl<B: DistributedNeuralBackend> QuantizerKernel<B> {
    /// Rejects a mismatched communicator before parameter acquisition.
    pub fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), Error> {
        let (_, selected) = self
            .parallel
            .as_ref()
            .ok_or_else(|| invalid("parallel quantizer requires rank-local construction"))?;
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(B::parallel_size(parallel), 1, 1, 1)
                .map_err(|_| invalid("invalid quantizer tensor degree"))?,
            B::parallel_rank(parallel),
        )
        .map_err(|_| invalid("invalid quantizer tensor rank"))?;
        let description = quantizer_parameter_description(self.geometry, &self.prefix)?;
        let expected = eredu_runtime::derive_partitioned_local_layout(&description, topology)
            .map_err(|_| invalid("quantizer tensor placement cannot be derived"))?;
        if selected != &expected {
            return Err(invalid(
                "quantizer layout differs from its tensor communicator",
            ));
        }
        Ok(())
    }
    fn prepare_channels(
        &self,
        value: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let local = self
            .parallel
            .as_ref()
            .expect("validated parallel construction")
            .0;
        let (batch, time) = (value.dim(0), value.dim(1));
        let mut parts = Vec::with_capacity(3);
        if local.channels_start > 0 {
            parts.push(B::Tensor::full_f32(
                0.0,
                &[batch, time, local.channels_start],
                context,
            )?);
        }
        parts.push(value);
        let end = local.channels_start + local.channels;
        if end < self.geometry.channels {
            parts.push(B::Tensor::full_f32(
                0.0,
                &[batch, time, self.geometry.channels - end],
                context,
            )?);
        }
        Ok(B::Tensor::concatenate(&parts, -1, context)?)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_guarded(
        &self,
        group: usize,
        index: usize,
        unit: &mut Codebook<B::Tensor>,
        hidden: &B::Tensor,
        forward: &mut QuantizerForward<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        agree: &mut impl FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, Error>,
    ) -> Result<B::Tensor, Error> {
        self.forward_selected(
            group,
            index,
            unit,
            hidden,
            forward,
            context,
            |channels, prepared| {
                let prepared = if channels {
                    prepared.and_then(|value| self.prepare_channels(value, context))
                } else {
                    prepared
                };
                let mut descriptor = [
                    0x505256514c4f434c,
                    u64::from(channels),
                    group as u64,
                    index as u64,
                    0,
                    0,
                    0,
                    0,
                ];
                if let Ok(value) = &prepared {
                    descriptor[4] = value.shape().len() as u64;
                    for (slot, dim) in descriptor[5..].iter_mut().zip(value.shape()) {
                        *slot = *dim as u64;
                    }
                }
                let accepted = agree(descriptor, prepared.is_ok(), context)?;
                let value = prepared?;
                if !accepted {
                    return Err(invalid("a PRVQ participant failed collective preparation"));
                }
                Ok(B::sum_parallel(value, parallel, context)?)
            },
        )
    }
}
impl<B: DistributedNeuralBackend> ParallelComponentKernel<B, eredu_runtime::StatelessComponentState>
    for QuantizerKernel<B>
{
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut StatelessComponentState,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
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
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_guarded(
            address.group(),
            address.index(),
            unit,
            hidden,
            forward,
            parallel,
            context,
            &mut |_, success, _| Ok(success),
        )
    }
}

fn runtime_error<P: std::error::Error + Send + Sync + 'static>(
    error: eredu_runtime::LayerwiseRuntimeError<Error, P>,
) -> Error {
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

impl<B, P> QuantizerExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, QuantizerKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, Codebook<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn validate_coordinated_execution(&self) -> Result<(), Error> {
        if self.architecture().kernel().is_parallel() {
            Err(invalid(
                "parallel PRVQ requires guarded collective execution",
            ))
        } else {
            Ok(())
        }
    }
    fn geometry(&self) -> Geometry {
        self.architecture().kernel().geometry()
    }
    fn decode_range(
        &mut self,
        codes: &B::Tensor,
        range: Range<usize>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut state = StatelessComponentState::new()?;
        self.invoke(QuantizerInput::Decode { codes, range }, &mut state, context)
            .map_err(runtime_error)
    }
    fn quantize_range(
        &mut self,
        latent: &B::Tensor,
        range: Range<usize>,
        metric: DistanceMetric,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        let mut state = StatelessComponentState::new()?;
        self.invoke_with_context(
            QuantizerInput::Quantize {
                latent,
                range,
                metric,
            },
            &mut state,
            context,
        )
        .map_err(runtime_error)?
        .1
        .into_quantized()
    }
}

/// Tensor-parallel range operations over independently leased residual codebooks.
/// Implementors supply typed invocation and guarded channel-collective participation.
pub trait ParallelQuantizerExecution<B: DistributedNeuralBackend>: QuantizerExecution<B> {
    /// Invokes a selected depth range with request-local roots retained to completion.
    fn invoke_parallel(
        &mut self,
        input: QuantizerInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, QuantizerForward<B::Tensor>), Error>;

    /// Publishes full decoded channels while the final selected codebook stays leased.
    fn decode_range_parallel(
        &mut self,
        codes: &B::Tensor,
        range: Range<usize>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.invoke_parallel(QuantizerInput::Decode { codes, range }, parallel, context)
            .map(|(output, _)| output)
    }

    /// Reduces distance scores before each categorical decision and retains global results.
    fn quantize_range_parallel(
        &mut self,
        latent: &B::Tensor,
        range: Range<usize>,
        metric: DistanceMetric,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Quantized<B::Tensor>, Error> {
        self.invoke_parallel(
            QuantizerInput::Quantize {
                latent,
                range,
                metric,
            },
            parallel,
            context,
        )?
        .1
        .into_quantized()
    }

    /// Runs channel shards with shared acquisition and collective-preparation gates.
    fn invoke_parallel_guarded<H, F>(
        &mut self,
        input: QuantizerInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        agree: F,
    ) -> Result<(B::Tensor, QuantizerForward<B::Tensor>), Error>
    where
        H: LayeredTraversalHook<B, QuantizerForward<B::Tensor>, Error> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, Error>;
}

impl<B, P> ParallelQuantizerExecution<B>
    for LayerwiseRuntime<
        ComponentAdapter<ComponentDefinition, QuantizerKernel<B>>,
        B,
        StatelessComponentState,
        P,
    >
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, Codebook<B::Tensor>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn invoke_parallel(
        &mut self,
        input: QuantizerInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, QuantizerForward<B::Tensor>), Error> {
        self.forward_parallel_with_context_hook(
            input,
            &mut StatelessComponentState::new()?,
            parallel,
            context,
            |_, _, _| Ok(()),
        )
        .map_err(runtime_error)
    }
    /// Runs channel shards with shared acquisition and local collective-preparation gates.
    fn invoke_parallel_guarded<H, F>(
        &mut self,
        input: QuantizerInput<'_, B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
        mut agree: F,
    ) -> Result<(B::Tensor, QuantizerForward<B::Tensor>), Error>
    where
        H: LayeredTraversalHook<B, QuantizerForward<B::Tensor>, Error> + ?Sized,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, Error>,
    {
        let (mode, value, range) = match &input {
            QuantizerInput::Decode { codes, range } => (0, *codes, range),
            QuantizerInput::Quantize {
                latent,
                range,
                metric,
            } => (
                match metric {
                    DistanceMetric::SquaredEuclidean => 1,
                    DistanceMetric::CodebookNormMinusTwiceDot => 2,
                },
                *latent,
                range,
            ),
        };
        let mut descriptor = [
            0x50525651494e5054,
            mode,
            range.start as u64,
            range.end as u64,
            value.shape().len() as u64,
            0,
            0,
            0,
        ];
        for (slot, dim) in descriptor[5..].iter_mut().zip(value.shape()) {
            *slot = *dim as u64;
        }
        if !agree(descriptor, true, context)? {
            return Err(invalid("PRVQ participants selected different invocations"));
        }
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            &mut StatelessComponentState::new()?,
            parallel,
            context,
            |architecture, group, index, unit, hidden, _, forward, parallel, context| {
                architecture.kernel().forward_unit_guarded(
                    group, index, unit, hidden, forward, parallel, context, &mut agree,
                )
            },
            hook,
        )
        .map_err(runtime_error)
    }
}
