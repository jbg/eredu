//! A changing cadence and channel layout crosses real pipeline transport and publication.
use super::*;
use eredu_runtime::*;

#[derive(Clone)]
struct Empty;
impl Parameterized<NumericTensor> for Empty {
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, NumericTensor>>(&'a self, _: &mut V) {
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, NumericTensor>>(
        &'a mut self,
        _: &mut V,
    ) {
    }
    fn set_trainable(&mut self, _: bool) {}
}
struct State(StateLayout);
impl State {
    fn new() -> Self {
        Self(
            StateLayout::new(LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap())
                .unwrap(),
        )
    }
}
impl RuntimeState<NumericBackend> for State {
    type RetainedValues<'a> = std::iter::Empty<&'a NumericTensor>;
    fn layout(&self) -> &StateLayout {
        &self.0
    }
    fn retained_values(
        &self,
        _: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        Ok(std::iter::empty())
    }
}
#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Identity,
    Role,
    Width,
    OutputShape,
    FinalOutput,
}
struct Geometry {
    stage: usize,
    empty: Empty,
    fault: Fault,
}
#[derive(Clone, Copy)]
struct Boundary(usize);
impl ArchitectureBoundary for Boundary {
    type Boundary<T> = T;
    const IDENTITY: &'static str = "changing-geometry";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        use BoundaryTensorDimension::{Batch, Fixed, Sequence};
        BoundaryTensorSpec::new(
            "activation",
            if self.0 == 0 {
                vec![Batch, Sequence, Fixed(3)]
            } else {
                vec![Batch, Fixed(5), Sequence]
            },
            BoundaryTensorDtype::Activation,
        )
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![BoundaryTensorSpec::new(
            if self.0 == 0 {
                "first_cut"
            } else {
                "second_cut"
            },
            [BoundaryTensorDimension::Fixed(self.0 as i32 + 1)],
            BoundaryTensorDtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        value: T,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(vec![ArchitectureBoundaryValue::new(
            self.auxiliary_tensor_specs()[0].role(),
            value,
        )?])
    }
    fn decode<T>(&self, mut values: Vec<T>) -> Result<T, ArchitectureBoundaryError> {
        validate_boundary_tensor_count(self, &values)?;
        Ok(values.remove(0))
    }
}
impl ArchitectureParameters<NumericBackend> for Geometry {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        StateLayout::new(LayerSchedule::new(3, vec![LayerCachePolicy::NoState; 3]).unwrap())
            .map_err(Error::backend)
    }
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error> {
        ModelStateIdentity::new(
            "geometry",
            "geometry",
            "geometry",
            3,
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }
    fn parameter_description(
        &self,
        _: &NumericContext,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = self.execution_graph()?;
        let layout = ExecutionUnitLayout::new(&graph, [3]).map_err(Error::backend)?;
        ArchitectureParameterDescription::new(&graph, &layout, [], []).map_err(Error::backend)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<NumericBackend>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<NumericBackend>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}
impl LayeredArchitecture<NumericBackend, State> for Geometry {
    type Input<'a> = &'a NumericTensor;
    type StaticModules = Empty;
    type Unit = Empty;
    type ForwardContext = (i32, i32);
    type RetainedContextValues<'a> = std::iter::Empty<&'a NumericTensor>;
    type Error = Error;
    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec![],
            last_owner_static_roles: vec![],
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }
    fn primary_execution_group(&self) -> &str {
        "geometry"
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(0, 0..3)])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::new(vec![ExecutionGroupSpec::root("geometry")], "geometry")
            .map_err(Error::backend)
    }
    fn group_unit_count(&self, _: usize) -> Result<usize, Error> {
        Ok(3)
    }
    fn unit_path(&self, _: usize, index: usize) -> Result<String, Error> {
        Ok(format!("geometry.{index}"))
    }
    fn static_modules(&self) -> &Empty {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut Empty {
        &mut self.empty
    }
    fn build_unit(&self, _: usize, _: usize, _: &NumericContext) -> Result<Empty, Error> {
        Ok(Empty)
    }
    fn begin_forward<'a>(
        &mut self,
        input: &'a NumericTensor,
        _: &mut State,
        _: &NumericContext,
    ) -> Result<LayeredForwardState<NumericTensor, Self::ForwardContext>, Error> {
        let dimensions = self.partition_input_dimensions(&input)?;
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: dimensions,
        })
    }
    fn begin_execution_group(
        &mut self,
        _: usize,
        initial: &NumericTensor,
        _: &[&NumericTensor],
        _: &mut State,
        _: &mut Self::ForwardContext,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(initial.clone())
    }
    fn forward_unit(
        &mut self,
        _: usize,
        index: usize,
        _: &mut Empty,
        hidden: &NumericTensor,
        _: &mut State,
        forward: &mut Self::ForwardContext,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        assert_eq!(index, self.stage);
        let (batch, time) = *forward;
        let mut values = Vec::new();
        match index {
            0 => {
                for row in hidden.data.chunks_exact(2) {
                    for _ in 0..2 {
                        values.extend([row[0], row[1], row[0] + row[1]]);
                    }
                }
                Ok(NumericTensor::new([batch, time * 2, 3], values))
            }
            1 => {
                for lane in 0..batch as usize {
                    for channel in 0..5 {
                        for frame in 0..time as usize {
                            let offset = (lane * time as usize * 2 + frame * 2) * 3;
                            let (a, b) = (hidden.data[offset], hidden.data[offset + 1]);
                            let value = [a, b, a + b, a - b, 1.][channel];
                            values.extend([value; 3]);
                        }
                    }
                }
                Ok(NumericTensor::new([batch, 5, time * 3], values))
            }
            _ => {
                for lane in 0..batch as usize {
                    for frame in 0..time as usize * 3 {
                        let sum = (0..5)
                            .map(|channel| {
                                hidden.data[(lane * 5 + channel) * time as usize * 3 + frame]
                            })
                            .sum::<f32>();
                        values.extend([sum; 2]);
                    }
                }
                let shape = if matches!(self.fault, Fault::FinalOutput) {
                    [batch, time * 6, 1]
                } else {
                    [batch, 1, time * 6]
                };
                Ok(NumericTensor::new(shape, values))
            }
        }
    }
    fn finish_forward(
        &mut self,
        hidden: &NumericTensor,
        _: &mut State,
        _: &Self::ForwardContext,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(hidden.clone())
    }
    fn retained_context_values<'a>(
        &'a self,
        _: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }
}
impl ParallelLayeredArchitecture<NumericBackend, State> for Geometry {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: &'a NumericTensor,
        state: &mut State,
        _: &NumericParallelContext,
        context: &NumericContext,
    ) -> Result<LayeredForwardState<NumericTensor, Self::ForwardContext>, Error> {
        self.begin_forward(input, state, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Empty,
        hidden: &NumericTensor,
        state: &mut State,
        forward: &mut Self::ForwardContext,
        _: &NumericParallelContext,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &NumericTensor,
        state: &mut State,
        forward: &Self::ForwardContext,
        _: &NumericParallelContext,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.finish_forward(hidden, state, forward, context)
    }
}
impl PartitionedLayeredArchitecture<NumericBackend, State> for Geometry {
    type Boundary = Boundary;
    fn boundary_schema(&self) -> Result<Boundary, Error> {
        Ok(Boundary(0))
    }
    fn begin_partition<'a>(
        &mut self,
        _: LayeredPartitionInput<'a, NumericTensor, NumericTensor>,
        _: Option<&NumericTensor>,
        _: &mut State,
        _: &StateLayout,
        _: usize,
        _: &NumericContext,
    ) -> Result<LayeredForwardState<NumericTensor, Self::ForwardContext>, Error> {
        Err(Error::backend(
            "fixture preserves supplied request through pipeline ingress",
        ))
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, NumericTensor, NumericTensor>,
        mask: Option<&NumericTensor>,
        state: &mut State,
        expected: &StateLayout,
        first: usize,
        _: &NumericParallelContext,
        context: &NumericContext,
    ) -> Result<LayeredForwardState<NumericTensor, Self::ForwardContext>, Error> {
        self.begin_partition(input, mask, state, expected, first, context)
    }
    fn finish_partition(
        &mut self,
        hidden: &NumericTensor,
        _: &mut State,
        _: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&NumericParallelContext>,
        context: &NumericContext,
    ) -> Result<LayeredPartitionOutput<NumericTensor, NumericTensor>, Error> {
        if owns_output {
            Ok(LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: NumericTensor::from_i32_slice(
                    &vec![self.stage as i32 + 10; self.stage + 1],
                    &[self.stage as i32 + 1],
                    context,
                )?,
            })
        }
    }
}
impl PipelinePartitionArchitecture<NumericBackend, State> for Geometry {
    fn partition_input_dimensions(&self, input: &&NumericTensor) -> Result<(i32, i32), Error> {
        if !matches!(input.shape(), [batch,time,2] if *batch>0 && *time>0) {
            return Err(Error::backend("invalid fixture ingress"));
        }
        Ok((input.dim(0), input.dim(1)))
    }
    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: &'a NumericTensor,
        incoming: Option<(NumericTensor, NumericTensor)>,
        state: &mut State,
        _: Option<&StateLayout>,
        _: usize,
        _: Option<&NumericParallelContext>,
        context: &NumericContext,
        _: &mut O,
    ) -> Result<LayeredForwardState<NumericTensor, Self::ForwardContext>, Error>
    where
        O: ActivationObserver<NumericTensor, Error> + ?Sized,
    {
        let mut forward = self.begin_forward(input, state, context)?;
        if let Some((hidden, auxiliary)) = incoming {
            assert_eq!(
                auxiliary.to_i32_vec(context)?,
                vec![self.stage as i32 + 9; self.stage]
            );
            let (batch, time) = forward.context;
            assert_eq!(
                hidden.shape(),
                if self.stage == 1 {
                    vec![batch, time * 2, 3]
                } else {
                    vec![batch, 5, time * 3]
                }
            );
            forward.hidden = hidden;
        } else {
            assert_eq!(self.stage, 0);
        }
        Ok(forward)
    }
    fn partition_output_width(&self) -> i32 {
        1
    }
    fn partition_retained_state_ordinals(
        &self,
        _: usize,
        _: usize,
        _: usize,
    ) -> std::ops::Range<usize> {
        0..0
    }
    fn partition_boundary_schema(&self, route: &PartitionBoundaryRoute) -> Result<Boundary, Error> {
        Ok(Boundary(route.source_rank))
    }
    fn partition_resolved_boundary_schema(
        &self,
        route: &PartitionBoundaryRoute,
        batch: i32,
        sequence: i32,
    ) -> Result<ResolvedBoundaryWireSchema, Error> {
        let boundary = self.partition_boundary_schema(route)?;
        let mut primary = boundary.primary_tensor_spec();
        let mut auxiliary = boundary.auxiliary_tensor_specs();
        if matches!(self.fault, Fault::Width) {
            primary = BoundaryTensorSpec::new(
                "activation",
                [
                    BoundaryTensorDimension::Batch,
                    BoundaryTensorDimension::Sequence,
                    BoundaryTensorDimension::Fixed(99),
                ],
                BoundaryTensorDtype::Activation,
            );
        }
        if matches!(self.fault, Fault::Role) {
            auxiliary = vec![BoundaryTensorSpec::new(
                "wrong",
                [BoundaryTensorDimension::Fixed(route.source_rank as i32 + 1)],
                BoundaryTensorDtype::Int32,
            )];
        }
        BoundaryWireSchema::new(
            if matches!(self.fault, Fault::Identity) {
                "wrong"
            } else {
                Boundary::IDENTITY
            },
            primary,
            auxiliary,
        )
        .and_then(|schema| schema.resolve(batch, sequence * (route.source_rank as i32 + 2)))
        .map_err(Error::backend)
    }
    fn partition_output_shape(
        &self,
        _: &&NumericTensor,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Error> {
        Ok(if matches!(self.fault, Fault::OutputShape) {
            vec![batch, 0, sequence * 6]
        } else {
            vec![batch, 1, sequence * 6]
        })
    }
}

#[derive(Clone, Default)]
struct Allocator(Arc<Mutex<Vec<Vec<i32>>>>);
impl PartitionTensorAllocator<NumericBackend> for Allocator {
    fn tensor_to_wire(
        &mut self,
        tensor: NumericTensor,
        dtype: BoundaryTensorDtype,
        activation: PipelineActivationDtype,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        NumericPartitionTensorAllocator.tensor_to_wire(tensor, dtype, activation, context)
    }
    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        dtype: BoundaryTensorDtype,
        activation: PipelineActivationDtype,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.0.lock().unwrap().push(shape.to_vec());
        NumericPartitionTensorAllocator.tensor_placeholder(shape, dtype, activation, context)
    }
}
type Policy = NumericReplicatedPolicy<Empty>;
type Executor = PipelinePartitionExecutor<Geometry, NumericBackend, State, Policy, Allocator>;
type Runtime = PartitionedTextRuntime<
    Geometry,
    NumericBackend,
    State,
    Policy,
    Executor,
    u64,
    u64,
    NumericCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Engine = PartitionedTextExecution<
    Executor,
    u64,
    u64,
    NumericCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

fn routes() -> Vec<PartitionBoundaryRoute> {
    (0..2)
        .map(|rank| PartitionBoundaryRoute {
            source_group: 0,
            destination_group: 0,
            source_rank: rank,
            destination_rank: rank + 1,
            route: CommunicationRouteId::new(rank as u64 + 10),
        })
        .collect()
}
fn manifest(rank: usize) -> CommunicationManifest {
    let requirement = |operation, dtypes| {
        CommunicationOperationRequirement::tensors(
            operation,
            dtypes,
            CommunicationTensorLimits::new(2, 3, 1024, None).unwrap(),
            true,
        )
        .unwrap()
    };
    let groups = vec![CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(1),
        0,
        vec![0, 1, 2],
        Some(rank),
        CommunicationGroupRequirements::new([
            requirement(
                CommunicationOperation::Broadcast,
                vec![eredu_core::checkpoint::TensorDtype::F32],
            ),
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap()];
    let descriptors = routes()
        .into_iter()
        .map(|route| {
            let schema = Boundary(route.source_rank).wire_schema().unwrap();
            let roles = std::iter::once(schema.primary())
                .chain(schema.auxiliary())
                .map(|spec| {
                    BoundaryRoleContract::symbolic(
                        spec.role(),
                        if spec.dtype() == BoundaryTensorDtype::Int32 {
                            eredu_core::checkpoint::TensorDtype::I32
                        } else {
                            eredu_core::checkpoint::TensorDtype::F32
                        },
                        spec.shape()
                            .iter()
                            .map(|dim| match dim {
                                BoundaryTensorDimension::Fixed(n) => {
                                    BoundaryDimensionContract::Fixed(*n as usize)
                                }
                                BoundaryTensorDimension::Batch => {
                                    BoundaryDimensionContract::Variable { maximum: 2 }
                                }
                                BoundaryTensorDimension::Sequence => {
                                    BoundaryDimensionContract::Variable { maximum: 6 }
                                }
                                other => panic!("unexpected fixture dimension: {other:?}"),
                            })
                            .collect(),
                    )
                    .unwrap()
                });
            let contract = RoleExactBoundaryContract::new(schema.identity(), roles).unwrap();
            CommunicationRouteDescriptor::new(
                route.route,
                route.source_rank,
                route.source_rank,
                route.destination_rank,
                requirement(
                    CommunicationOperation::SendReceive,
                    vec![
                        eredu_core::checkpoint::TensorDtype::F32,
                        eredu_core::checkpoint::TensorDtype::I32,
                    ],
                ),
            )
            .unwrap()
            .with_boundary_contract(contract)
            .unwrap()
        })
        .collect();
    CommunicationManifest::new(3, rank, groups, descriptors)
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(10),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        )
}
fn executor(
    stage: usize,
    fault: Fault,
    allocator: Allocator,
) -> (
    Executor,
    LayeredPartitionDriver,
    ArchitectureParameterDescription,
    ArchitecturePartition<(), Boundary>,
) {
    let context = NumericContext::default();
    let architecture = Geometry {
        stage,
        empty: Empty,
        fault,
    };
    let description = architecture.parameter_description(&context).unwrap();
    let layout = architecture.state_layout().unwrap();
    let partition = ArchitecturePartition::from_description(
        &description,
        [("geometry", stage..stage + 1)],
        PartitionOwnership::new(stage == 0, stage == 2, std::iter::empty::<String>()).unwrap(),
        &layout,
        &architecture.state_partition_plan(&layout),
        (),
        Boundary(0),
    )
    .unwrap();
    let driver = LayeredPartitionDriver::new(&partition, 0, stage..stage + 1).unwrap();
    let executor = Executor::new(
        architecture,
        Policy::resident(vec![Empty]),
        partition.units().collect(),
        None,
        allocator,
        PipelineActivationDtype::Float32,
    )
    .unwrap();
    (executor, driver, description, partition)
}

fn run_rank(rank: usize, fault: Fault, world: Arc<NumericPartitionWorld>) {
    let context = NumericContext::with_partition(LocalModelLayout::default(), rank, world.clone());
    let allocations = Allocator::default();
    let (executor, driver, description, partition) = executor(rank, fault, allocations.clone());
    let manifest = manifest(rank);
    let plan = PartitionedExecutionPlan::new(
        description.graph().clone(),
        vec![(ArchitectureGroupKind::Decoder, false)],
        vec![Some(driver)],
        routes(),
        Some(PartitionOutputPublication {
            group: CollectiveGroupId::new(1),
            owner_rank: 2,
        }),
        Some(CollectiveGroupId::new(1)),
        PipelineWireContract::new(PipelineActivationDtype::Float32),
    )
    .unwrap();
    if rank == 1 && matches!(fault, Fault::None) {
        // The middle rank has distinct incoming and outgoing cuts.
        // A uniform declaration accepts the first route, then rejects
        // the second before any communication resources are realized.
        let mut visited = Vec::new();
        let wrong = validate_component_pipeline(
            &description,
            &partition,
            partition.units(),
            partition.state().map(|state| state.layout()),
            None,
            &manifest,
            &plan,
            None,
            PipelineActivationDtype::Float32,
            &[(ArchitectureGroupKind::Decoder, false)],
            0,
            64,
            |route| {
                visited.push(route.source());
                Ok(Boundary(0).wire_schema().unwrap())
            },
        );
        assert!(wrong.is_err());
        assert_eq!(visited, [0, 1]);
        validate_component_pipeline(
            &description,
            &partition,
            partition.units(),
            partition.state().map(|state| state.layout()),
            None,
            &manifest,
            &plan,
            None,
            PipelineActivationDtype::Float32,
            &[(ArchitectureGroupKind::Decoder, false)],
            0,
            64,
            |route| Ok(Boundary(route.source()).wire_schema().unwrap()),
        )
        .unwrap();
        assert!(allocations.0.lock().unwrap().is_empty());
    }
    world.realize_manifest(&manifest).unwrap();
    let communication = numeric_partition_communication(&world, manifest).unwrap();
    let mut runtime = Runtime::new(
        plan,
        executor,
        communication,
        context.clone(),
        OpaqueBoundaryTransport,
        OpaqueOutputPublisher,
        OpaqueFailureAgreement,
        ExecutionResidency::FullyResident,
        None,
    )
    .unwrap();
    let mut engine = Engine::new();
    let mut state = State::new();
    let requests = [
        (
            1,
            NumericTensor::new([1, 2, 2], vec![1., 2., 3., 4.]),
            vec![6., 6., 6., 6., 6., 6., 14., 14., 14., 14., 14., 14.],
        ),
        (
            2,
            NumericTensor::new([2, 1, 2], vec![5., 7., 11., 13.]),
            vec![23., 23., 23., 23., 23., 23., 47., 47., 47., 47., 47., 47.],
        ),
    ];
    for (epoch, input, expected) in requests {
        let mut observer = NoopObserver;
        let result = <Engine as ReplicatedTextExecutionStrategy<
            Geometry,
            NumericBackend,
            State,
            Policy,
            Policy,
        >>::forward_with_observer(
            &mut engine,
            &mut runtime,
            &input,
            &mut state,
            ExpertPass::Prefill,
            &context,
            &mut observer,
        );
        if matches!(fault, Fault::FinalOutput) {
            assert!(
                result.is_err(),
                "owner's wrong final layout must reject before output broadcast"
            );
            break;
        }
        let (output, _) = result.unwrap();
        let output = <Engine as ReplicatedTextExecutionStrategy<
            Geometry,
            NumericBackend,
            State,
            Policy,
            Policy,
        >>::publish_observed_output(&mut runtime, output, &context)
        .unwrap();
        assert_eq!(output.shape, [input.dim(0), 1, input.dim(1) * 6]);
        assert_eq!(output.data, expected);
        let epoch = eredu_core::DistributedCommitEpoch::new(epoch).unwrap();
        let committed = <Engine as ReplicatedTextExecutionStrategy<
            Geometry,
            NumericBackend,
            State,
            Policy,
            Policy,
        >>::commit_after_completion(&mut runtime, epoch, &context);
        assert_eq!(committed, DistributedCommitOutcome::Committed(epoch));
    }
    if matches!(fault, Fault::None) {
        let allocated = allocations.0.lock().unwrap();
        if rank < 2 {
            assert!(allocated.contains(&vec![1, 1, 12]));
            assert!(allocated.contains(&vec![2, 1, 6]));
        }
        if rank == 1 {
            assert!(allocated.contains(&vec![1, 4, 3]));
            assert!(allocated.contains(&vec![2, 2, 3]));
        }
        if rank == 2 {
            assert!(allocated.contains(&vec![1, 5, 6]));
            assert!(allocated.contains(&vec![2, 5, 3]));
        }
    }
}

#[test]
fn pipeline_geometry_transports_distinct_cadences_and_publishes_channel_first_output() {
    for fault in [Fault::None, Fault::FinalOutput] {
        let world = Arc::new(NumericPartitionWorld::default());
        std::thread::scope(|scope| {
            let workers = (0..3)
                .map(|rank| {
                    let world = world.clone();
                    scope.spawn(move || run_rank(rank, fault, world))
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        let trace = world.trace();
        assert!(trace
            .iter()
            .any(|event| event.operation == NumericOpaqueOperation::PointToPoint));
        assert_eq!(
            trace
                .iter()
                .any(|event| event.operation == NumericOpaqueOperation::Broadcast),
            matches!(fault, Fault::None),
        );
    }
}

#[test]
fn pipeline_geometry_rejects_resolved_contract_mismatches_before_receive_allocation() {
    let context = NumericContext::default();
    let input = NumericTensor::new([1, 2, 2], vec![1., 2., 3., 4.]);
    for fault in [
        Fault::Identity,
        Fault::Role,
        Fault::Width,
        Fault::OutputShape,
    ] {
        let allocator = Allocator::default();
        let (mut executor, _, _, _) = executor(1, fault, allocator.clone());
        let mut state = State::new();
        let result = <Executor as PartitionedGroupExecutor<
            Geometry,
            NumericBackend,
            State,
            u64,
            u64,
            NumericCommunicationMetadata,
        >>::begin(
            &mut executor,
            &input,
            &mut state,
            ExpertPass::Prefill,
            &context,
        );
        if matches!(fault, Fault::OutputShape) {
            assert!(result.is_err());
        } else {
            let pass = result.unwrap();
            assert!(<Executor as PartitionedGroupExecutor<
                Geometry,
                NumericBackend,
                State,
                u64,
                u64,
                NumericCommunicationMetadata,
            >>::boundary_schema(&executor, &pass, &routes()[0])
            .is_err());
        }
        assert!(allocator.0.lock().unwrap().is_empty());
    }
}
