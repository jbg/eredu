use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    convert::Infallible,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    },
    checkpoint::TensorDtype,
    AttentionPolicy, Completion, DistributedCommitEpoch, DistributedCommitOutcome,
    DistributedCommitPhase, InputModality, InputTensorIdentity, LayerSchedule, Submission,
    TokenFilter,
};
use eredu_nn::{
    AttentionCache, AttentionMask, AttentionRequest, EmbeddingOperator, EmbeddingSpec, Error,
    GatedProductPolicy, Index, LinearOperator, LinearSpec, NeuralBackend,
    NeuralOperatorCapabilities, NormalizationConstructionSpec, NormalizationOperator, PadMode,
    ParameterVisitor, ParameterVisitorMut, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, construct_replicated_text_session,
    construct_replicated_text_session_with_runtime, materialize_bindings,
    prepare_layered_text_contract, prepare_partitioned_session_runtime,
    prepare_replicated_text_contract, realize_architecture_state,
    select_replicated_text_realization, ArchitectureBoundary, ArchitectureGroupKind,
    ArchitectureGroupPlacement, ArchitectureGroupTransport, ArchitectureMergeDestination,
    ArchitectureParameterDescription, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStateFactory, BackendMechanismCapabilities, BarrierBackend, CacheResidencyPolicy,
    CollectiveBackend, CommunicationBackend, CommunicationGroupDescriptor, CommunicationGroupId,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationRouteDescriptor, CommunicationRouteId,
    CommunicationTensorLimits, CommunicationTensorMetadata, CompositeLayeredTraversalHook,
    DeviceState, DistributedExecutionPhase, ExecutionGraph, ExecutionGroupSpec, ExecutionResidency,
    ExecutionUnitAddress, ExecutionUnitLayout, FailureAgreementBackend, LayerWeightResidency,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionDriver, LayeredPartitionInput,
    LayeredPartitionOutput, LayeredTraversalHook, LayeredTraversalPoint, LayeredUnitAction,
    LayerwisePolicy, LayerwiseRuntime, NoAuxiliaryBoundary, NoAuxiliaryBoundarySchema,
    NoBoundaryTransport, NoCommitAgreement, NoOutputPublisher, OpaqueBoundaryTransport,
    OpaqueCommitAgreement, OpaqueFailureAgreement, ParallelLayeredArchitecture, ParameterBackend,
    PartitionBoundaryRoute, PartitionBoundaryTransport, PartitionCommitAgreement,
    PartitionCommunication, PartitionOutputPublication, PartitionOutputPublisher,
    PartitionOwnership, PartitionedExecutionPlan, PartitionedGroupExecutor,
    PartitionedLayeredArchitecture, PartitionedTextExecution, PartitionedTextRuntime,
    PenaltyConfig, PipelineActivationDtype, PipelineWireContract, PointToPointBackend,
    PredictionDirective, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
    RealizedCommunicationGroup, RealizedCommunicationRoute, ReplicatedTextArchitecture,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputSelection, ReplicatedTextParameterOwner,
    ReplicatedTextParameterPresence, ReplicatedTextParameterRequirement,
    ReplicatedTextParameterRole, ReplicatedTextPhysicalSource, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, ReplicatedTextSession, ReplicatedTextSessionMechanisms,
    ReplicatedTextStateAccess, ResettableRuntimeLayerState, ResettableRuntimeState,
    ResidentRuntime, RuntimeLayerState, RuntimeState, RuntimeStateComponents, Sampler,
    SamplingBackend, SequentialDecisionBoundary, SequentialDecisionDriver, SequentialDecisionError,
    SequentialDecisionPlan, SequentialDecisionSource, SequentialDecisionTraversal,
    StateComponentMechanism, StateComponentPlacement, StateError, StateLayout,
    StateMechanismCapabilities, StateSegmentId, StateSegmentLifetime, StateSegmentSpec,
    StaticParameterVisitor, StaticParameterVisitorMut, SubmissionBackend, SumReductionBackend,
    TokenDomain, TransactionalPromptCacheMechanisms, TransferBackend, UnevenGatherBackend,
    WeightBinding, WeightLoweringCapability, WeightLoweringKind, WeightResidencyMechanism,
};

#[derive(Debug, Clone, Eq, PartialEq)]
struct FakeTensor(Vec<i32>);

impl Tensor for FakeTensor {
    type Context = ();

    fn shape(&self) -> &[i32] {
        &self.0
    }

    fn unloaded_f32(shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn from_f32_slice(_: &[f32], shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn add(&self, _: &Self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn subtract(&self, _: &Self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn multiply(&self, _: &Self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn multiply_scalar(&self, _: f32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn divide(&self, _: &Self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn square(&self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn maximum_scalar(&self, _: f32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn reshape(&self, shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn transpose_axes(&self, _: &[i32], _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn swap_axes(&self, _: i32, _: i32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn transpose(&self, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn expand_dims(&self, _: i32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn squeeze_axes(&self, _: &[i32], _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn index(&self, _: &[Index], _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn take_axis(&self, _: &Self, _: i32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn concatenate(values: &[Self], _: i32, _: &()) -> Result<Self, Error> {
        Ok(values[0].clone())
    }
    fn stack(values: &[Self], _: i32, _: &()) -> Result<Self, Error> {
        Ok(values[0].clone())
    }
    fn matmul(lhs: &Self, _: &Self, _: &()) -> Result<Self, Error> {
        Ok(lhs.clone())
    }
    fn sum_axis(value: &Self, _: i32, _: bool, _: &()) -> Result<Self, Error> {
        Ok(value.clone())
    }
    fn argmin_axis(value: &Self, _: i32, _: bool, _: &()) -> Result<Self, Error> {
        Ok(value.clone())
    }
    fn pad(value: &Self, _: &[(i32, i32)], _: PadMode, _: &()) -> Result<Self, Error> {
        Ok(value.clone())
    }
    fn conv1d(
        input: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &(),
    ) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn conv_transpose1d(
        input: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &(),
    ) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn linear(input: &Self, _: &Self, _: Option<&Self>, _: &()) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn layer_norm(
        input: &Self,
        _: Option<&Self>,
        _: Option<&Self>,
        _: f32,
        _: &(),
    ) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn gelu(input: &Self, _: &()) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn elu(input: &Self, _: f32, _: &()) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn rope(input: &Self, _: i32, _: bool, _: f32, _: f32, _: i32, _: &()) -> Result<Self, Error> {
        Ok(input.clone())
    }
    fn scaled_dot_product_attention(
        queries: &Self,
        _: &Self,
        _: &Self,
        _: f32,
        _: AttentionMask<'_, Self>,
        _: &(),
    ) -> Result<Self, Error> {
        Ok(queries.clone())
    }
}

#[derive(Debug, Clone)]
struct FakeOperator;

struct FakeModule {
    weight: FakeTensor,
}

impl Parameterized<FakeTensor> for FakeModule {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, FakeTensor>,
    {
        visitor.visit(
            eredu_nn::ParameterMetadata {
                id: eredu_nn::ParameterId::new("weight").unwrap(),
                trainable: true,
                alias_of: None,
                group: None,
                linear_companion: None,
                linear_companion_of: None,
            },
            &self.weight,
        );
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, FakeTensor>,
    {
        visitor.visit_mut(
            eredu_nn::ParameterMetadata {
                id: eredu_nn::ParameterId::new("weight").unwrap(),
                trainable: true,
                alias_of: None,
                group: None,
                linear_companion: None,
                linear_companion_of: None,
            },
            &mut self.weight,
        );
    }

    fn set_trainable(&mut self, _trainable: bool) {}
}

impl Parameterized<FakeTensor> for FakeOperator {
    fn visit_parameters<'a, V>(&'a self, _visitor: &mut V)
    where
        V: ParameterVisitor<'a, FakeTensor>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, FakeTensor>,
    {
    }
    fn set_trainable(&mut self, _trainable: bool) {}
}

impl LinearOperator<FakeTensor> for FakeOperator {
    fn forward(&mut self, input: &FakeTensor, _: &()) -> Result<FakeTensor, Error> {
        Ok(input.clone())
    }
}
impl EmbeddingOperator<FakeTensor> for FakeOperator {
    fn forward(&mut self, input: &FakeTensor, _: &()) -> Result<FakeTensor, Error> {
        Ok(input.clone())
    }
    fn as_linear(&mut self, input: &FakeTensor, _: &()) -> Result<FakeTensor, Error> {
        Ok(input.clone())
    }
}
impl NormalizationOperator<FakeTensor> for FakeOperator {
    fn forward(&mut self, input: &FakeTensor, _: &()) -> Result<FakeTensor, Error> {
        Ok(input.clone())
    }
}
impl RotaryOperator<FakeTensor> for FakeOperator {
    fn forward(
        &mut self,
        input: &FakeTensor,
        _: RotaryPosition<'_, FakeTensor>,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        Ok(input.clone())
    }
}

struct FakeBackend;

impl NeuralBackend for FakeBackend {
    type Tensor = FakeTensor;
    type Linear = FakeOperator;
    type Embedding = FakeOperator;
    type Normalization = FakeOperator;
    type Rotary = FakeOperator;
    type ParallelContext = ();

    fn linear(_: LinearSpec, _: &()) -> Result<Self::Linear, Error> {
        Ok(FakeOperator)
    }
    fn embedding(_: EmbeddingSpec, _: &()) -> Result<Self::Embedding, Error> {
        Ok(FakeOperator)
    }
    fn normalization(
        _: NormalizationConstructionSpec,
        _: &(),
    ) -> Result<Self::Normalization, Error> {
        Ok(FakeOperator)
    }
    fn rotary(_: RotarySpec, _: &()) -> Result<Self::Rotary, Error> {
        Ok(FakeOperator)
    }
    fn silu(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
    }
    fn gated_product(
        gate: Self::Tensor,
        _: Self::Tensor,
        _: GatedProductPolicy,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(gate)
    }
    fn attention(
        queries: Self::Tensor,
        _: Self::Tensor,
        _: Self::Tensor,
        _: f32,
        _: Option<&Self::Tensor>,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(queries)
    }
    fn sliding_window_attention(
        queries: Self::Tensor,
        _: Self::Tensor,
        _: Self::Tensor,
        _: f32,
        _: i32,
        _: i32,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(queries)
    }
    fn causal_mask(sequence: i32, _: i32, _: Option<i32>, _: &()) -> Result<Self::Tensor, Error> {
        Ok(FakeTensor(vec![sequence, sequence]))
    }
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        LinearOperator::forward(linear, input, context)
    }
}

struct PartitionCollectiveBackend;

#[derive(Clone)]
struct PartitionCollectiveGroup {
    members: Vec<usize>,
    local_rank: usize,
    trace: Rc<RefCell<Vec<PartitionCollectiveCall>>>,
}

#[derive(Debug, Eq, PartialEq)]
struct PartitionCollectiveCall {
    local_rank: usize,
    members: Vec<usize>,
    value: Vec<i32>,
}

impl NeuralBackend for PartitionCollectiveBackend {
    type Tensor = FakeTensor;
    type Linear = FakeOperator;
    type Embedding = FakeOperator;
    type Normalization = FakeOperator;
    type Rotary = FakeOperator;
    type ParallelContext = ();

    fn linear(_: LinearSpec, _: &()) -> Result<Self::Linear, Error> {
        Ok(FakeOperator)
    }
    fn embedding(_: EmbeddingSpec, _: &()) -> Result<Self::Embedding, Error> {
        Ok(FakeOperator)
    }
    fn normalization(
        _: NormalizationConstructionSpec,
        _: &(),
    ) -> Result<Self::Normalization, Error> {
        Ok(FakeOperator)
    }
    fn rotary(_: RotarySpec, _: &()) -> Result<Self::Rotary, Error> {
        Ok(FakeOperator)
    }
    fn silu(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
    }
    fn gated_product(
        gate: Self::Tensor,
        _: Self::Tensor,
        _: GatedProductPolicy,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(gate)
    }
    fn attention(
        queries: Self::Tensor,
        _: Self::Tensor,
        _: Self::Tensor,
        _: f32,
        _: Option<&Self::Tensor>,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(queries)
    }
    fn sliding_window_attention(
        queries: Self::Tensor,
        _: Self::Tensor,
        _: Self::Tensor,
        _: f32,
        _: i32,
        _: i32,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(queries)
    }
    fn causal_mask(sequence: i32, _: i32, _: Option<i32>, _: &()) -> Result<Self::Tensor, Error> {
        Ok(FakeTensor(vec![sequence, sequence]))
    }
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        LinearOperator::forward(linear, input, context)
    }
}

impl SamplingBackend for FakeBackend {
    type Logits = FakeTensor;
    type Token = FakeTensor;
    type RandomState = i32;
    type Context = ();
    type Error = Error;

    fn error(message: String) -> Self::Error {
        Error::backend(message)
    }

    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        token
            .0
            .first()
            .copied()
            .and_then(|token| usize::try_from(token).ok())
            .filter(|token| *token < domain.cardinality())
            .map(|_| token.clone())
            .ok_or_else(|| Error::backend("token is outside its decision domain"))
    }

    fn scale_temperature(
        logits: &Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_penalties(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_top_k(
        logits: Self::Logits,
        _: i32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_top_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_min_p(
        logits: Self::Logits,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits)
    }

    fn apply_token_filter(
        logits: &Self::Logits,
        _: &TokenFilter,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn apply_mirostat(
        logits: &Self::Logits,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn sample_raw(
        logits: &Self::Logits,
        _: f32,
        random: Option<&mut Self::RandomState>,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        if let Some(random) = random {
            *random += 1;
        }
        Ok(logits.clone())
    }

    fn sample_processed(
        logits: &Self::Logits,
        temperature: f32,
        random: Option<&mut Self::RandomState>,
        context: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        Self::sample_raw(logits, temperature, random, context)
    }

    fn token_id(token: &Self::Token, _: &Self::Context) -> Result<u32, Self::Error> {
        token
            .0
            .first()
            .copied()
            .ok_or_else(|| Error::backend("fixture token is empty"))
            .and_then(|token| {
                u32::try_from(token).map_err(|error| Error::backend(error.to_string()))
            })
    }

    fn token_probability(_: &Self::Logits, _: u32, _: &Self::Context) -> Result<f32, Self::Error> {
        Ok(1.0)
    }
}

#[derive(Debug)]
struct Done;

#[derive(Debug)]
struct CommunicationDone;

#[derive(Debug, Clone, Copy)]
enum FakeCommunicationError {
    Submission,
    CorruptBoundaryHeader,
}

impl std::fmt::Display for FakeCommunicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Submission => "injected communication submission failure",
            Self::CorruptBoundaryHeader => "received boundary frame header was corrupted",
        })
    }
}

impl std::error::Error for FakeCommunicationError {}

#[derive(Debug, Clone, Copy)]
enum FakeCompletionOutcome {
    Completed,
    DeadlineExceeded,
    CorruptBoundaryHeader,
}

thread_local! {
    static FORK_COUNT: Cell<usize> = const { Cell::new(0) };
    static SUBMIT_COUNT: Cell<usize> = const { Cell::new(0) };
    static ORDER_COUNT: Cell<usize> = const { Cell::new(0) };
    static POINT_TO_POINT_COUNT: Cell<usize> = const { Cell::new(0) };
    static LOCAL_DEPENDENCY_SUBMISSION_COUNT: Cell<usize> = const { Cell::new(0) };
    static CORRUPT_GATHER: Cell<bool> = const { Cell::new(false) };
    static CORRUPT_GATHER_DTYPE: Cell<bool> = const { Cell::new(false) };
    static UNEVEN_GATHER_CALLS: Cell<usize> = const { Cell::new(0) };
    static COMMUNICATION_COMPLETION_SCRIPT: RefCell<VecDeque<FakeCompletionOutcome>> = const { RefCell::new(VecDeque::new()) };
    static BOUNDED_COMMUNICATION_WAIT_COUNT: Cell<usize> = const { Cell::new(0) };
    static FAILURE_AGREEMENT_COUNT: Cell<usize> = const { Cell::new(0) };
    static FAIL_NEXT_COMMUNICATION_SUBMISSION: Cell<bool> = const { Cell::new(false) };
}

thread_local! {
    static PARTITION_COMMIT_TRACE: RefCell<Option<Rc<RefCell<Vec<&'static str>>>>> = const { RefCell::new(None) };
    static PARTITION_PUBLICATION_VALUES: RefCell<Option<Rc<RefCell<Vec<FakeTensor>>>>> = const { RefCell::new(None) };
}

impl Completion for Done {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl eredu_core::BoundedCompletion for Done {
    fn wait_bounded(
        self,
        _policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        Ok(eredu_core::BoundedCompletionOutcome::Completed)
    }
}

impl Completion for CommunicationDone {
    type Error = FakeCommunicationError;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl eredu_core::BoundedCompletion for CommunicationDone {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        BOUNDED_COMMUNICATION_WAIT_COUNT.set(BOUNDED_COMMUNICATION_WAIT_COUNT.get() + 1);
        Ok(
            match COMMUNICATION_COMPLETION_SCRIPT.with(|script| script.borrow_mut().pop_front()) {
                Some(FakeCompletionOutcome::DeadlineExceeded) => {
                    eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                        cancellation: policy.cancellation(),
                    }
                }
                Some(FakeCompletionOutcome::Completed) | None => {
                    eredu_core::BoundedCompletionOutcome::Completed
                }
                Some(FakeCompletionOutcome::CorruptBoundaryHeader) => {
                    return Err(FakeCommunicationError::CorruptBoundaryHeader)
                }
            },
        )
    }
}

impl SubmissionBackend for FakeBackend {
    type Executor = ();
    type OwnedExecutor = ();
    type Completion = Done;
    fn fork_executors(_: &(), count: usize) -> Result<Vec<Self::OwnedExecutor>, Infallible> {
        FORK_COUNT.set(FORK_COUNT.get() + 1);
        Ok(vec![(); count])
    }
    fn submit<'a, I>(_: &(), _: I) -> Result<Self::Completion, Infallible>
    where
        FakeTensor: 'a,
        I: IntoIterator<Item = &'a FakeTensor>,
    {
        SUBMIT_COUNT.set(SUBMIT_COUNT.get() + 1);
        Ok(Done)
    }
    fn order_after(_: &Done, _: &()) -> Result<(), Infallible> {
        ORDER_COUNT.set(ORDER_COUNT.get() + 1);
        Ok(())
    }
    fn retain_until_complete<T: Send + 'static>(_: &(), _: &Done, _: T) -> Result<(), Infallible> {
        Ok(())
    }
}

impl CommunicationBackend for FakeBackend {
    type CommunicationGroup = ();
    type CommunicationRoute = ();
    type CommunicationCompletion = CommunicationDone;
    type CommunicationError = FakeCommunicationError;

    fn submit_local_dependencies<'a, I>(
        values: I,
        _: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        let _ = values.into_iter().count();
        LOCAL_DEPENDENCY_SUBMISSION_COUNT.set(LOCAL_DEPENDENCY_SUBMISSION_COUNT.get() + 1);
        if FAIL_NEXT_COMMUNICATION_SUBMISSION.replace(false) {
            return Err(FakeCommunicationError::Submission);
        }
        Ok(Submission {
            output: (),
            completion: CommunicationDone,
        })
    }
}

impl BarrierBackend for FakeBackend {
    fn barrier(
        _: &Self::CommunicationGroup,
        _: &Self::Executor,
    ) -> Result<Self::CommunicationCompletion, Self::CommunicationError> {
        PARTITION_COMMIT_TRACE.with(|trace| {
            if let Some(trace) = trace.borrow().as_ref() {
                trace.borrow_mut().push("commit");
            }
        });
        Ok(CommunicationDone)
    }
}

impl PointToPointBackend for FakeBackend {
    fn send_receive(
        values: Vec<eredu_runtime::RoleExactBoundaryValue<Self::Tensor>>,
        _: &Self::CommunicationRoute,
        _: &Self::Executor,
    ) -> Result<
        Submission<Vec<Self::Tensor>, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        POINT_TO_POINT_COUNT.set(POINT_TO_POINT_COUNT.get() + 1);
        if FAIL_NEXT_COMMUNICATION_SUBMISSION.replace(false) {
            return Err(FakeCommunicationError::Submission);
        }
        Ok(Submission {
            output: values
                .into_iter()
                .map(|value| value.into_parts().1)
                .collect(),
            completion: CommunicationDone,
        })
    }
}

impl FailureAgreementBackend for FakeBackend {
    type FailureAgreementOutput = bool;

    fn agree_success(
        local_success: bool,
        _: &Self::CommunicationGroup,
        _: &Self::Executor,
    ) -> Result<Submission<bool, Self::CommunicationCompletion>, Self::CommunicationError> {
        FAILURE_AGREEMENT_COUNT.set(FAILURE_AGREEMENT_COUNT.get() + 1);
        if FAIL_NEXT_COMMUNICATION_SUBMISSION.replace(false) {
            return Err(FakeCommunicationError::Submission);
        }
        Ok(Submission {
            output: local_success,
            completion: CommunicationDone,
        })
    }

    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError> {
        Ok(output)
    }
}

impl UnevenGatherBackend for FakeBackend {
    fn all_gather_uneven(
        _: Self::Tensor,
        counts: &[usize],
        _: usize,
        _: &Self::CommunicationGroup,
        _: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        UNEVEN_GATHER_CALLS.with(|calls| calls.set(calls.get() + 1));
        let mut rows: usize = counts.iter().sum();
        CORRUPT_GATHER.with(|corrupt| {
            if corrupt.get() {
                rows = rows.saturating_sub(1);
            }
        });
        let mut output = vec![0; rows];
        CORRUPT_GATHER_DTYPE.with(|corrupt| {
            if corrupt.get() && !output.is_empty() {
                output[0] = i32::MIN;
            }
        });
        Ok(Submission {
            output: FakeTensor(output),
            completion: CommunicationDone,
        })
    }
}

#[derive(Clone, Copy)]
struct FakeTensorMetadata;

impl CommunicationTensorMetadata<FakeBackend> for FakeTensorMetadata {
    fn dtype(&self, tensor: &FakeTensor) -> TensorDtype {
        if tensor.0.first() == Some(&i32::MIN) {
            TensorDtype::Bf16
        } else {
            TensorDtype::F32
        }
    }

    fn shape(&self, tensor: &FakeTensor) -> Vec<usize> {
        vec![tensor.0.len()]
    }
}

#[derive(Clone, Copy)]
struct PipelineTensorMetadata;

fn test_completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
    eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_millis(10),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap()
}

impl CommunicationTensorMetadata<FakeBackend> for PipelineTensorMetadata {
    fn dtype(&self, _: &FakeTensor) -> TensorDtype {
        TensorDtype::F32
    }

    fn shape(&self, _: &FakeTensor) -> Vec<usize> {
        vec![1, 1, 1]
    }
}

#[test]
fn uneven_gather_validates_distinct_input_and_completed_result_limits() {
    UNEVEN_GATHER_CALLS.with(|calls| calls.set(0));
    let group_id = CommunicationGroupId::new(1);
    let requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllGatherUneven,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 2, None)
            .unwrap()
            .with_output_tensor_elements(4)
            .unwrap(),
        true,
    )
    .unwrap();
    let descriptor = CommunicationGroupDescriptor::new(
        group_id,
        0,
        vec![0, 1],
        Some(0),
        CommunicationGroupRequirements::new([requirement.clone()]).unwrap(),
    )
    .unwrap();
    let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
        CommunicationManifest::new(2, 0, vec![descriptor], Vec::new())
            .unwrap()
            .with_completion_policy(test_completion_policy()),
        vec![RealizedCommunicationGroup::new(group_id, ())],
        Vec::new(),
        FakeTensorMetadata,
    )
    .unwrap();

    assert_eq!(
        communication
            .all_gather_uneven(FakeTensor(vec![1, 2]), &[2, 2], 0, group_id, &())
            .unwrap(),
        FakeTensor(vec![0, 0, 0, 0])
    );
    CORRUPT_GATHER.with(|corrupt| corrupt.set(true));
    let corrupt = communication
        .all_gather_uneven(FakeTensor(vec![1, 2]), &[2, 2], 0, group_id, &())
        .unwrap_err();
    CORRUPT_GATHER.with(|corrupt| corrupt.set(false));
    assert!(matches!(
        corrupt,
        eredu_runtime::PartitionExecutionError::CommunicationOutputShape {
            expected,
            actual,
        } if expected == [4] && actual == [3]
    ));
    assert_eq!(UNEVEN_GATHER_CALLS.get(), 2);
    let retry = communication
        .all_gather_uneven(FakeTensor(vec![1, 2]), &[2, 2], 0, group_id, &())
        .unwrap_err();
    assert!(matches!(
        retry,
        eredu_runtime::PartitionExecutionError::CommunicationPoisoned { .. }
    ));
    assert_eq!(UNEVEN_GATHER_CALLS.get(), 2);

    let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
        CommunicationManifest::new(
            2,
            0,
            vec![CommunicationGroupDescriptor::new(
                group_id,
                0,
                vec![0, 1],
                Some(0),
                CommunicationGroupRequirements::new([requirement]).unwrap(),
            )
            .unwrap()],
            Vec::new(),
        )
        .unwrap()
        .with_completion_policy(test_completion_policy()),
        vec![RealizedCommunicationGroup::new(group_id, ())],
        Vec::new(),
        FakeTensorMetadata,
    )
    .unwrap();
    CORRUPT_GATHER_DTYPE.with(|corrupt| corrupt.set(true));
    let corrupt = communication
        .all_gather_uneven(FakeTensor(vec![1, 2]), &[2, 2], 0, group_id, &())
        .unwrap_err();
    CORRUPT_GATHER_DTYPE.with(|corrupt| corrupt.set(false));
    assert!(matches!(
        corrupt,
        eredu_runtime::PartitionExecutionError::TensorDtype {
            dtype: TensorDtype::Bf16
        }
    ));
    assert_eq!(UNEVEN_GATHER_CALLS.get(), 3);
    let retry = communication
        .all_gather_uneven(FakeTensor(vec![1, 2]), &[2, 2], 0, group_id, &())
        .unwrap_err();
    assert!(matches!(
        retry,
        eredu_runtime::PartitionExecutionError::CommunicationPoisoned { .. }
    ));
    assert_eq!(UNEVEN_GATHER_CALLS.get(), 3);
}

#[test]
fn partition_communication_rejects_swapped_native_resource_identities() {
    let requirement = || {
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
            true,
        )
        .unwrap()])
        .unwrap()
    };
    let first = CommunicationGroupId::new(1);
    let second = CommunicationGroupId::new(2);
    let manifest = CommunicationManifest::new(
        1,
        0,
        vec![
            CommunicationGroupDescriptor::new(first, 0, vec![0], Some(0), requirement()).unwrap(),
            CommunicationGroupDescriptor::new(second, 1, vec![0], Some(0), requirement()).unwrap(),
        ],
        Vec::new(),
    )
    .unwrap()
    .with_completion_policy(test_completion_policy());
    let result = PartitionCommunication::<FakeBackend, (), (), _>::new(
        manifest,
        vec![
            RealizedCommunicationGroup::new(second, ()),
            RealizedCommunicationGroup::new(first, ()),
        ],
        Vec::new(),
        FakeTensorMetadata,
    );

    assert!(matches!(
        result,
        Err(eredu_runtime::PartitionExecutionError::ResourceIdentity {
            expected: 1,
            actual: 2,
        })
    ));
}

fn route_test_communication(
    rank: usize,
    with_failure_agreement: bool,
) -> PartitionCommunication<FakeBackend, (), (), PipelineTensorMetadata> {
    let route_id = CommunicationRouteId::new(9);
    let requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 3, 1, None).unwrap(),
        true,
    )
    .unwrap();
    let group_id = CommunicationGroupId::new(1);
    let groups = with_failure_agreement
        .then(|| {
            CommunicationGroupDescriptor::new(
                group_id,
                0,
                vec![0, 1],
                Some(rank),
                CommunicationGroupRequirements::new([
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap()
        })
        .into_iter()
        .collect::<Vec<_>>();
    let manifest = CommunicationManifest::new(
        2,
        rank,
        groups,
        vec![test_boundary_route(route_id, 0, 0, 1, requirement)],
    )
    .unwrap()
    .with_completion_policy(test_completion_policy());
    PartitionCommunication::new(
        manifest,
        with_failure_agreement
            .then(|| RealizedCommunicationGroup::new(group_id, ()))
            .into_iter()
            .collect(),
        vec![RealizedCommunicationRoute::new(route_id, ())],
        PipelineTensorMetadata,
    )
    .unwrap()
}

fn test_boundary_route(
    id: CommunicationRouteId,
    order: usize,
    source: usize,
    destination: usize,
    requirement: CommunicationOperationRequirement,
) -> CommunicationRouteDescriptor {
    let role = eredu_runtime::BoundaryRoleContract::symbolic(
        "hidden",
        TensorDtype::F32,
        vec![
            eredu_runtime::BoundaryDimensionContract::Variable { maximum: 1 },
            eredu_runtime::BoundaryDimensionContract::Variable { maximum: 1 },
            eredu_runtime::BoundaryDimensionContract::Fixed(1),
        ],
    )
    .unwrap();
    CommunicationRouteDescriptor::new(id, order, source, destination, requirement)
        .unwrap()
        .with_boundary_contract(
            eredu_runtime::RoleExactBoundaryContract::new("none", [role]).unwrap(),
        )
        .unwrap()
}

fn transfer_test_boundary(
    communication: &PartitionCommunication<FakeBackend, (), (), PipelineTensorMetadata>,
) -> Result<Vec<FakeTensor>, eredu_runtime::PartitionExecutionError> {
    let schema = NoAuxiliaryBoundarySchema::new(1)
        .wire_schema()
        .unwrap()
        .resolve(1, 1)
        .unwrap();
    OpaqueBoundaryTransport.transfer(
        communication,
        CommunicationRouteId::new(9),
        vec![eredu_runtime::ArchitectureBoundaryValue::new("hidden", FakeTensor(vec![1])).unwrap()],
        &schema,
        PipelineWireContract::new(PipelineActivationDtype::Float32),
        &(),
    )
}

#[test]
fn two_rank_deadline_poison_is_stable_and_retry_submits_no_native_work() {
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let outcomes = std::thread::scope(|scope| {
        (0..2)
            .map(|rank| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    COMMUNICATION_COMPLETION_SCRIPT.with(|script| {
                        script
                            .borrow_mut()
                            .push_back(FakeCompletionOutcome::DeadlineExceeded);
                    });
                    POINT_TO_POINT_COUNT.set(0);
                    let communication = route_test_communication(rank, false);
                    barrier.wait();
                    let first = transfer_test_boundary(&communication).unwrap_err();
                    let calls_after_deadline = POINT_TO_POINT_COUNT.get();
                    let retry = transfer_test_boundary(&communication).unwrap_err();
                    (
                        first,
                        retry,
                        calls_after_deadline,
                        POINT_TO_POINT_COUNT.get(),
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    for (first, retry, calls_after_deadline, calls_after_retry) in outcomes {
        assert!(matches!(
            first,
            eredu_runtime::PartitionExecutionError::CommunicationDeadlineExceeded {
                operation: CommunicationOperation::SendReceive,
                phase: DistributedExecutionPhase::Execution,
                route: Some(route),
                ..
            } if route == CommunicationRouteId::new(9)
        ));
        assert!(matches!(
            retry,
            eredu_runtime::PartitionExecutionError::CommunicationPoisoned {
                operation: CommunicationOperation::SendReceive,
                phase: DistributedExecutionPhase::Execution,
                route: Some(route),
                ..
            } if route == CommunicationRouteId::new(9)
        ));
        assert_eq!(calls_after_deadline, 1);
        assert_eq!(calls_after_retry, 1);
    }
}

#[test]
fn agreement_deadline_after_completed_transfer_poisons_later_route_use() {
    let communication = route_test_communication(0, true);
    COMMUNICATION_COMPLETION_SCRIPT.with(|script| {
        script.borrow_mut().extend([
            FakeCompletionOutcome::Completed,
            FakeCompletionOutcome::DeadlineExceeded,
        ]);
    });
    POINT_TO_POINT_COUNT.set(0);
    FAILURE_AGREEMENT_COUNT.set(0);
    transfer_test_boundary(&communication).unwrap();
    let error = OpaqueFailureAgreement
        .agree_phase(
            &communication,
            CommunicationGroupId::new(1),
            DistributedExecutionPhase::BoundarySourceReady(CommunicationRouteId::new(9)),
            true,
            &(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        eredu_runtime::PartitionExecutionError::CommunicationDeadlineExceeded {
            operation: CommunicationOperation::FailureAgreement,
            phase: DistributedExecutionPhase::BoundarySourceReady(route),
            ..
        } if route == CommunicationRouteId::new(9)
    ));
    assert!(matches!(
        transfer_test_boundary(&communication),
        Err(
            eredu_runtime::PartitionExecutionError::CommunicationPoisoned {
                operation: CommunicationOperation::FailureAgreement,
                ..
            }
        )
    ));
    assert_eq!(POINT_TO_POINT_COUNT.get(), 1);
    assert_eq!(FAILURE_AGREEMENT_COUNT.get(), 1);
}

#[test]
fn recovery_agreement_cannot_bypass_a_healthy_communication_authority() {
    let communication = route_test_communication(0, true);
    FAILURE_AGREEMENT_COUNT.set(0);
    let error = OpaqueFailureAgreement
        .agree_phase_after_prior_failure(
            &communication,
            CommunicationGroupId::new(1),
            DistributedExecutionPhase::Execution,
            true,
            &(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        eredu_runtime::PartitionExecutionError::RecoveryAgreementWithoutFailure {
            phase: DistributedExecutionPhase::Execution,
        }
    ));
    assert_eq!(FAILURE_AGREEMENT_COUNT.get(), 0);
}

#[test]
fn final_agreement_submission_and_completion_failures_are_indeterminate_and_poisoned() {
    for (submission_failure, expected_phase) in [
        (true, DistributedCommitPhase::DecisionSubmission),
        (false, DistributedCommitPhase::DecisionCompletion),
    ] {
        let communication = route_test_communication(0, true);
        FAILURE_AGREEMENT_COUNT.set(0);
        if submission_failure {
            FAIL_NEXT_COMMUNICATION_SUBMISSION.set(true);
        } else {
            COMMUNICATION_COMPLETION_SCRIPT.with(|script| {
                script
                    .borrow_mut()
                    .push_back(FakeCompletionOutcome::DeadlineExceeded);
            });
        }
        let first = OpaqueFailureAgreement.commit(
            &communication,
            CommunicationGroupId::new(1),
            DistributedCommitEpoch::FIRST,
            &(),
        );
        assert_eq!(
            first,
            DistributedCommitOutcome::Indeterminate {
                epoch: DistributedCommitEpoch::FIRST,
                phase: expected_phase,
            }
        );
        let calls = FAILURE_AGREEMENT_COUNT.get();
        let retry_epoch = DistributedCommitEpoch::FIRST.next().unwrap();
        assert!(OpaqueFailureAgreement
            .commit(
                &communication,
                CommunicationGroupId::new(1),
                retry_epoch,
                &(),
            )
            .is_indeterminate());
        assert_eq!(FAILURE_AGREEMENT_COUNT.get(), calls);
    }
}

#[test]
fn synchronous_submission_failure_poisons_retry_before_backend_call() {
    let communication = route_test_communication(0, false);
    POINT_TO_POINT_COUNT.set(0);
    FAIL_NEXT_COMMUNICATION_SUBMISSION.set(true);
    assert!(matches!(
        transfer_test_boundary(&communication),
        Err(eredu_runtime::PartitionExecutionError::CommunicationSubmissionFailed {
            operation: CommunicationOperation::SendReceive,
            route: Some(route),
            ..
        }) if route == CommunicationRouteId::new(9)
    ));
    assert_eq!(POINT_TO_POINT_COUNT.get(), 1);
    assert!(matches!(
        transfer_test_boundary(&communication),
        Err(eredu_runtime::PartitionExecutionError::CommunicationPoisoned { .. })
    ));
    assert_eq!(POINT_TO_POINT_COUNT.get(), 1);
}

impl SubmissionBackend for PartitionCollectiveBackend {
    type Executor = ();
    type OwnedExecutor = ();
    type Completion = Done;

    fn fork_executors(_: &(), count: usize) -> Result<Vec<Self::OwnedExecutor>, Infallible> {
        Ok(vec![(); count])
    }

    fn submit<'a, I>(_: &(), _: I) -> Result<Self::Completion, Infallible>
    where
        FakeTensor: 'a,
        I: IntoIterator<Item = &'a FakeTensor>,
    {
        Ok(Done)
    }

    fn order_after(_: &Done, _: &()) -> Result<(), Infallible> {
        Ok(())
    }

    fn retain_until_complete<T: Send + 'static>(_: &(), _: &Done, _: T) -> Result<(), Infallible> {
        Ok(())
    }
}

impl CollectiveBackend for PartitionCollectiveBackend {
    type Group = PartitionCollectiveGroup;
    type CollectiveError = Infallible;

    fn all_reduce(
        value: Self::Tensor,
        _: &Self::Group,
        _: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        Ok(value)
    }

    fn all_gather(
        value: Self::Tensor,
        _: &Self::Group,
        _: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        Ok(value)
    }

    fn all_to_all(
        value: Self::Tensor,
        group: &Self::Group,
        _: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError> {
        group.trace.borrow_mut().push(PartitionCollectiveCall {
            local_rank: group.local_rank,
            members: group.members.clone(),
            value: value.0.clone(),
        });
        Ok(value)
    }
}

impl CommunicationBackend for PartitionCollectiveBackend {
    type CommunicationGroup = PartitionCollectiveGroup;
    type CommunicationRoute = ();
    type CommunicationCompletion = Done;
    type CommunicationError = Infallible;

    fn submit_local_dependencies<'a, I>(
        _: I,
        _: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        Ok(Submission {
            output: (),
            completion: Done,
        })
    }
}

impl SumReductionBackend for PartitionCollectiveBackend {
    fn all_reduce_sum(
        value: Self::Tensor,
        group: &Self::CommunicationGroup,
        _: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        group.trace.borrow_mut().push(PartitionCollectiveCall {
            local_rank: group.local_rank,
            members: group.members.clone(),
            value: value.0.clone(),
        });
        Ok(Submission {
            output: value,
            completion: Done,
        })
    }
}

impl ParameterBackend for FakeBackend {
    type Parameter = FakeTensor;
    type MaterializedWeight = FakeTensor;
    type MaterializationContext = ();
    type Materialization = FakeTensor;
    type ParameterError = Infallible;
    fn materialize(
        lease: eredu_checkpoint::store::CheckpointLease,
        _: &(),
    ) -> Result<Self::Materialization, Infallible> {
        use eredu_checkpoint::store::EncodedTensorLease;
        Ok(FakeTensor(
            lease
                .output_shape()
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap_or(i32::MAX))
                .collect(),
        ))
    }
    fn materialize_recipe(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        _: &(),
    ) -> Result<Self::Materialization, Infallible> {
        let metadata = recipe
            .infer(source)
            .expect("fake-backend recipes are validated by the neutral catalog");
        Ok(FakeTensor(
            metadata
                .shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap_or(i32::MAX))
                .collect(),
        ))
    }
    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight {
        materialization
    }
    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Infallible> {
        Ok(materialization)
    }
    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Infallible> {
        Ok(weight.clone())
    }
    fn validate_bind(
        _parameter: &Self::Parameter,
        _weight: &Self::MaterializedWeight,
    ) -> Result<(), Infallible> {
        Ok(())
    }
    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Infallible> {
        *parameter = weight;
        Ok(())
    }
}

impl TransferBackend for FakeBackend {
    type HostBuffer = FakeTensor;
    type Transfer = Done;
    type TransferError = Infallible;
    fn promote(
        _: &(),
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Done), Infallible> {
        Ok((host.clone(), Done))
    }
    fn demote(
        _: &(),
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Done), Infallible> {
        Ok((weight.clone(), Done))
    }
}

#[test]
fn minimal_text_backend_compiles_without_optional_execution_extensions() {
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: ReplicatedSessionCounters::default(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let layout = architecture.state_layout().unwrap();
    let mut state =
        DeviceState::create(layout, |_, _| Ok::<_, Infallible>(FakeLayerState(0))).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();
    let tokens = FakeTensor(vec![9]);
    let input = <OrdinaryTextFixture as eredu_runtime::ReplicatedTextArchitecture<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >>::text_input(&tokens, None);
    let output = runtime.forward(input, &mut state, &()).unwrap();
    assert_eq!(output, FakeTensor(vec![9, 5]));
    assert_eq!(state.as_ref()[0].0, 1);
    assert_eq!(runtime.architecture().trace, ["input", "unit", "output"]);
}

#[test]
fn neutral_loader_materializes_and_binds_the_fake_backend() {
    use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("model.safetensors");
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap(),
        )],
        None,
        &file,
    )
    .unwrap();
    let source = SafetensorsWeightStore::open(&file).unwrap();
    let binding = WeightBinding::new("weight", "weight", TensorSelection::Full, 16).unwrap();
    let unit = materialize_bindings::<FakeBackend>(&source, &[binding], &()).unwrap();
    let id = eredu_nn::ParameterId::new("weight").unwrap();
    assert!(unit.contains(&id));

    let mut module = FakeModule {
        weight: FakeTensor(vec![0]),
    };
    bind_materialized_unit::<FakeBackend, _>(&mut module, unit).unwrap();
    assert_eq!(module.weight, FakeTensor(vec![2, 2]));
}

struct CountingSource {
    inner: eredu_checkpoint::store::MemoryWeightStore,
    leases: Arc<AtomicUsize>,
}

impl eredu_checkpoint::store::CheckpointSource for CountingSource {
    fn source_keys(&self) -> Vec<String> {
        eredu_checkpoint::store::CheckpointSource::source_keys(&self.inner)
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        eredu_checkpoint::store::CheckpointSource::source_metadata(&self.inner, key)
    }

    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        self.leases.fetch_add(1, Ordering::SeqCst);
        eredu_checkpoint::store::CheckpointSource::acquire_lease(&self.inner, request)
    }

    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        eredu_checkpoint::store::CheckpointSource::source_diagnostics(&self.inner)
    }
}

fn counting_source() -> (CountingSource, Arc<AtomicUsize>) {
    let leases = Arc::new(AtomicUsize::new(0));
    let inner = eredu_checkpoint::store::MemoryWeightStore::from_safetensors([(
        "owner".to_owned(),
        safetensors::Dtype::F32,
        vec![2, 2],
        vec![0; 16],
    )])
    .unwrap();
    (
        CountingSource {
            inner,
            leases: Arc::clone(&leases),
        },
        leases,
    )
}

#[test]
fn aliases_share_one_fake_backend_materialization() {
    use eredu_checkpoint::store::TensorSelection;

    let (source, leases) = counting_source();
    let bindings = vec![
        WeightBinding::new("owner", "owner", TensorSelection::Full, 16).unwrap(),
        WeightBinding::alias("alias-a", "owner", 16).unwrap(),
        WeightBinding::alias("alias-b", "alias-a", 16).unwrap(),
    ];
    let unit = materialize_bindings::<FakeBackend>(&source, &bindings, &()).unwrap();
    assert_eq!(unit.len(), 3);
    assert_eq!(leases.load(Ordering::SeqCst), 1);
}

#[test]
fn invalid_alias_graph_fails_before_any_physical_read() {
    use eredu_checkpoint::store::TensorSelection;

    let (source, leases) = counting_source();
    let bindings = vec![
        WeightBinding::new("owner", "owner", TensorSelection::Full, 16).unwrap(),
        WeightBinding::alias("alias", "missing", 16).unwrap(),
    ];
    assert!(materialize_bindings::<FakeBackend>(&source, &bindings, &()).is_err());
    assert_eq!(leases.load(Ordering::SeqCst), 0);
}

#[derive(Debug, Clone)]
struct FakeLayerState(i32);

impl RuntimeLayerState<FakeBackend> for FakeLayerState {
    type RetainedValues<'a> = std::iter::Empty<&'a FakeTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

impl ResettableRuntimeLayerState<FakeBackend> for FakeLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        self.0 = 0;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RetainedOrdinalLayerState(FakeTensor);

impl RuntimeLayerState<FakeBackend> for RetainedOrdinalLayerState {
    type RetainedValues<'a> = std::iter::Once<&'a FakeTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::once(&self.0)
    }
}

#[test]
fn device_state_retention_uses_local_ordinal_with_nonzero_partition_offset() {
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap();
    let local =
        StateLayout::new(LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap()).unwrap();
    let partition = eredu_runtime::PartitionState::new(local.clone(), 5).unwrap();
    let state = DeviceState::<FakeBackend, RetainedOrdinalLayerState>::create(
        partition.layout().clone(),
        |ordinal, _| {
            Ok::<_, Infallible>(RetainedOrdinalLayerState(FakeTensor(vec![i32::try_from(
                ordinal,
            )
            .unwrap()])))
        },
    )
    .unwrap();
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [7]).unwrap();
    let global_address = units.address(6).unwrap();

    assert_eq!(partition.global_layer_offset(), 5);
    let retained = RuntimeState::retained_values(&state, 1, global_address)
        .unwrap()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(retained, [FakeTensor(vec![1])]);
    assert_eq!(global_address.index(), 6);
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
struct ReplicatedSessionCounts {
    materializations: usize,
    resident_policies: usize,
    bounded_policies: usize,
    unit_constructions: usize,
    state_allocations: usize,
    forward_calls: usize,
    completion_attempts: usize,
    publications: usize,
}

#[derive(Clone, Default)]
struct ReplicatedSessionCounters(Rc<Cell<ReplicatedSessionCounts>>);

impl ReplicatedSessionCounters {
    fn snapshot(&self) -> ReplicatedSessionCounts {
        self.0.get()
    }

    fn update(&self, update: impl FnOnce(&mut ReplicatedSessionCounts)) {
        let mut counts = self.snapshot();
        update(&mut counts);
        self.0.set(counts);
    }
}

struct OrdinaryTextFixture {
    static_modules: FakeOperator,
    trace: Vec<&'static str>,
    counters: ReplicatedSessionCounters,
    inconsistent_transport: bool,
    inconsistent_identity: bool,
}

impl ArchitectureParameters<FakeBackend> for OrdinaryTextFixture {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap()],
            )
            .unwrap(),
        )
        .map_err(Error::backend)
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        eredu_runtime::ModelStateIdentity::new(
            "ordinary-text-fixture",
            "ordinary-text-fixture",
            if self.inconsistent_identity {
                "different-architecture"
            } else {
                "ordinary-text-fixture"
            },
            1,
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }

    fn parameter_description(
        &self,
        _: &(),
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        let graph = self.execution_graph()?;
        let layout = ExecutionUnitLayout::new(&graph, [1]).map_err(Error::backend)?;
        let group = eredu_runtime::ParameterGroupSpec::new(
            "decoder.weight",
            eredu_runtime::ParameterRole::Replicated,
            [eredu_runtime::ParameterMemberSpec::new(
                "decoder.weight",
                [1, 1],
                eredu_runtime::MemberSharding::Replicated,
            )],
        )
        .map_err(Error::backend)?;
        let owner = eredu_runtime::ParameterGroupOwner::execution_unit(
            layout.group_id(0).expect("decoder group").clone(),
            0,
        );
        ArchitectureParameterDescription::new(
            &graph,
            &layout,
            [group.clone()],
            [eredu_runtime::OwnedParameterGroupSpec::new(owner, group)],
        )
        .map_err(Error::backend)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<FakeBackend>,
    {
        visitor.visit("embedding", &self.static_modules)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<FakeBackend>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules)
    }
}

impl LayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for OrdinaryTextFixture
{
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = ();
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec!["embedding".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: self.inconsistent_transport,
        }
    }

    fn primary_execution_group(&self) -> &str {
        "decoder"
    }

    fn state_partition_plan(
        &self,
        _: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..1),
        ])
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder")
            .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        (group == 0)
            .then_some(1)
            .ok_or_else(|| Error::backend("unknown ordinary text group"))
    }

    fn unit_path(&self, _: usize, _: usize) -> Result<String, Self::Error> {
        Ok("decoder.unit.0".into())
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, _: usize, _: usize, _: &()) -> Result<Self::Unit, Self::Error> {
        self.counters
            .update(|counts| counts.unit_constructions += 1);
        Ok(FakeUnit { marker: 5 })
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.trace.push("input");
        Ok(LayeredForwardState {
            hidden: input.clone(),
            context: (),
        })
    }

    fn begin_execution_group(
        &mut self,
        _: usize,
        initial: &FakeTensor,
        _: &[&FakeTensor],
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        _: usize,
        _: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.counters.update(|counts| counts.forward_calls += 1);
        state.as_mut()[0].0 += 1;
        self.trace.push("unit");
        let mut output = hidden.clone();
        output.0.push(unit.marker);
        Ok(output)
    }

    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.trace.push("output");
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

impl
    eredu_runtime::ReplicatedTextArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for OrdinaryTextFixture
{
    fn text_input<'a>(tokens: &'a FakeTensor, _: Option<&'a FakeTensor>) -> Self::Input<'a> {
        tokens
    }
}

struct PredictionCaptureFixture(OrdinaryTextFixture);

struct PredictionStateOperation {
    fail: bool,
}

impl
    eredu_runtime::PredictionTargetOperation<
        PredictionCaptureFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for PredictionStateOperation
{
    type Output = i32;

    fn apply(
        self,
        _: &mut PredictionCaptureFixture,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: Option<&()>,
        _: &(),
    ) -> Result<Self::Output, Error> {
        state.as_mut()[0].0 += 10;
        if self.fail {
            Err(Error::backend("injected prediction extension failure"))
        } else {
            Ok(state.as_ref()[0].0)
        }
    }
}

impl ArchitectureParameters<FakeBackend> for PredictionCaptureFixture {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.0.state_layout()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        self.0.state_identity(state, topology)
    }

    fn parameter_description(
        &self,
        context: &(),
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.0.parameter_description(context)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<FakeBackend>,
    {
        self.0.visit_static_parameters(visitor)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<FakeBackend>,
    {
        self.0.visit_static_parameters_mut(visitor)
    }
}

impl LayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for PredictionCaptureFixture
{
    type Input<'a> = &'a FakeTensor;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = FakeTensor;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.0.group_transport(group)
    }

    fn primary_execution_group(&self) -> &str {
        self.0.primary_execution_group()
    }

    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.0.state_partition_plan(layout)
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        self.0.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.0.group_unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.0.unit_path(group, index)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self.0.static_modules()
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.0.static_modules_mut()
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &(),
    ) -> Result<Self::Unit, Self::Error> {
        self.0.build_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        let forward = self.0.begin_forward(input, state, context)?;
        Ok(LayeredForwardState {
            context: forward.hidden.clone(),
            hidden: forward.hidden,
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _forward: &mut Self::ForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.0
            .begin_execution_group(group, initial, dependencies, state, &mut (), context)
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &mut Self::ForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        let output = self
            .0
            .forward_unit(group, index, unit, hidden, state, &mut (), context)?;
        forward.clone_from(&output);
        Ok(output)
    }

    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _forward: &Self::ForwardContext,
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.0.finish_forward(hidden, state, &(), context)
    }

    fn retained_context_values<'a>(
        &'a self,
        _forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }

    fn prediction_target_capture(context: &Self::ForwardContext) -> Option<&FakeTensor> {
        Some(context)
    }
}

impl
    eredu_runtime::ReplicatedTextArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for PredictionCaptureFixture
{
    fn text_input<'a>(tokens: &'a FakeTensor, _: Option<&'a FakeTensor>) -> Self::Input<'a> {
        tokens
    }
}

impl ParallelLayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for OrdinaryTextFixture
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward(input, state, context)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &mut Self::ForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &Self::ForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.finish_forward(hidden, state, forward, context)
    }
}

impl PartitionedLayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for OrdinaryTextFixture
{
    type Boundary = NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(NoAuxiliaryBoundarySchema::new(1))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, FakeTensor, NoAuxiliaryBoundary>,
        _: Option<&FakeTensor>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &StateLayout,
        _: usize,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => tokens.clone(),
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        Ok(LayeredForwardState {
            hidden,
            context: (),
        })
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, FakeTensor, NoAuxiliaryBoundary>,
        mask: Option<&FakeTensor>,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        expected: &StateLayout,
        first_state_ordinal: usize,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
    }

    fn finish_partition(
        &mut self,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&()>,
        context: &(),
    ) -> Result<LayeredPartitionOutput<FakeTensor, NoAuxiliaryBoundary>, Self::Error> {
        let output = self.finish_forward(hidden, state, forward, context)?;
        Ok(if owns_output {
            LayeredPartitionOutput::Final {
                output,
                retained: None,
            }
        } else {
            LayeredPartitionOutput::Boundary {
                hidden: output,
                auxiliary: NoAuxiliaryBoundary,
            }
        })
    }
}

#[derive(Clone)]
struct HybridLayerState {
    position: i32,
    attention_keys: Option<FakeTensor>,
    attention_values: Option<FakeTensor>,
    recurrent: Option<FakeTensor>,
    convolution: Option<FakeTensor>,
}

impl RuntimeLayerState<FakeBackend> for HybridLayerState {
    type RetainedValues<'a> = std::vec::IntoIter<&'a FakeTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.recurrent
            .iter()
            .chain(self.convolution.iter())
            .chain(self.attention_keys.iter())
            .chain(self.attention_values.iter())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl AttentionCache<FakeTensor> for HybridLayerState {
    fn offset(&self) -> i32 {
        self.position
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_for_attention(
        &mut self,
        keys: FakeTensor,
        values: FakeTensor,
        _: &(),
    ) -> Result<(FakeTensor, FakeTensor), Error> {
        let cached_keys = self
            .attention_keys
            .as_mut()
            .ok_or_else(|| Error::backend("hybrid fixture omitted key state"))?;
        let cached_values = self
            .attention_values
            .as_mut()
            .ok_or_else(|| Error::backend("hybrid fixture omitted value state"))?;
        cached_keys.0.extend(keys.0);
        cached_values.0.extend(values.0);
        Ok((cached_keys.clone(), cached_values.clone()))
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, FakeTensor>,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        let _ = self.update_for_attention(request.keys, request.values, &())?;
        Ok(request.queries)
    }
}

impl RuntimeStateComponents<FakeBackend> for HybridLayerState {
    fn position(&self) -> i32 {
        self.position
    }

    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<FakeTensor>, StateError> {
        match role {
            StateTensorRole::Recurrent => Ok(&mut self.recurrent),
            StateTensorRole::Convolution { slot: 0 } => Ok(&mut self.convolution),
            _ => Err(StateError::UnknownComponent { role }),
        }
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        self.position = self
            .position
            .checked_add(tokens)
            .ok_or_else(|| StateError::InvalidAdvance("fixture position overflow".into()))?;
        Ok(())
    }
}

fn hybrid_policy() -> LayerCachePolicy {
    let fixed = |role, residency| {
        StateTensorPolicy::new(
            role,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(4).unwrap(),
            ],
            StateTensorDtype::Floating,
            residency,
        )
        .unwrap()
    };
    LayerCachePolicy::key_value_with_fixed_state(
        AttentionPolicy::Full,
        1,
        4,
        vec![
            fixed(
                StateTensorRole::Recurrent,
                MutableStateResidency::LayerScopedOffloadable,
            ),
            fixed(
                StateTensorRole::Convolution { slot: 0 },
                MutableStateResidency::AlwaysDeviceMutable,
            ),
        ],
    )
    .unwrap()
}

struct HybridFixture {
    static_modules: FakeOperator,
    trace: Vec<&'static str>,
}

struct HybridStateFactory {
    calls: Rc<Cell<usize>>,
}

impl ArchitectureStateFactory<FakeBackend> for HybridStateFactory {
    type State = DeviceState<FakeBackend, HybridLayerState>;
    type Error = Infallible;

    fn realize(&mut self, layout: &StateLayout) -> Result<Self::State, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        DeviceState::create(layout.clone(), |_, policy| {
            assert_eq!(policy.fixed_state().len(), 2);
            assert_eq!(policy.components().len(), 4);
            Ok::<_, Infallible>(HybridLayerState {
                position: 0,
                attention_keys: Some(FakeTensor(vec![0])),
                attention_values: Some(FakeTensor(vec![0])),
                recurrent: Some(FakeTensor(vec![1, 4])),
                convolution: Some(FakeTensor(vec![1, 4])),
            })
        })
    }
}

impl ArchitectureParameters<FakeBackend> for HybridFixture {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        StateLayout::new(LayerSchedule::new(1, vec![hybrid_policy()]).unwrap())
            .map_err(Error::backend)
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        eredu_runtime::ModelStateIdentity::new(
            "hybrid-fixture",
            "hybrid-fixture",
            "hybrid-fixture",
            1,
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }

    fn parameter_description(
        &self,
        _: &(),
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        let graph = self.execution_graph()?;
        let layout = ExecutionUnitLayout::new(&graph, [1]).map_err(Error::backend)?;
        ArchitectureParameterDescription::new(&graph, &layout, [], []).map_err(Error::backend)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<FakeBackend>,
    {
        visitor.visit("embedding", &self.static_modules)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<FakeBackend>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules)
    }
}

impl LayeredArchitecture<FakeBackend, DeviceState<FakeBackend, HybridLayerState>>
    for HybridFixture
{
    type Input<'a> = i32;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = ();
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

    fn group_transport(&self, _: usize) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: ArchitectureGroupPlacement::Pipeline,
            kind: ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec!["embedding".into()],
            last_owner_static_roles: Vec::new(),
            merge_destination: ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }
    }

    fn primary_execution_group(&self) -> &str {
        "hybrid"
    }

    fn state_partition_plan(
        &self,
        _: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..1),
        ])
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(vec![ExecutionGroupSpec::root("hybrid")], "hybrid")
            .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        (group == 0)
            .then_some(1)
            .ok_or_else(|| Error::backend("unknown hybrid fixture group"))
    }

    fn unit_path(&self, _: usize, _: usize) -> Result<String, Self::Error> {
        Ok("hybrid.unit.0".into())
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, _: usize, _: usize, _: &()) -> Result<Self::Unit, Self::Error> {
        Ok(FakeUnit { marker: 7 })
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut DeviceState<FakeBackend, HybridLayerState>,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.trace.push("input");
        Ok(LayeredForwardState {
            hidden: FakeTensor(vec![input]),
            context: (),
        })
    }

    fn begin_execution_group(
        &mut self,
        _: usize,
        initial: &FakeTensor,
        _: &[&FakeTensor],
        _: &mut DeviceState<FakeBackend, HybridLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        _: usize,
        _: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, HybridLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        let layer = eredu_runtime::LayerRuntimeState::layer(state, 0).map_err(Error::backend)?;
        let input = hidden.0[0];
        let (keys, values) = AttentionCache::update_for_attention(
            layer,
            FakeTensor(vec![1, input]),
            FakeTensor(vec![1, unit.marker]),
            &(),
        )?;
        RuntimeStateComponents::<FakeBackend>::fixed_component(layer, StateTensorRole::Recurrent)
            .map_err(|error| Error::backend(error.to_string()))?
            .as_mut()
            .unwrap()
            .0
            .push(input);
        RuntimeStateComponents::<FakeBackend>::fixed_component(
            layer,
            StateTensorRole::Convolution { slot: 0 },
        )
        .map_err(|error| Error::backend(error.to_string()))?
        .as_mut()
        .unwrap()
        .0
        .push(unit.marker);
        RuntimeStateComponents::<FakeBackend>::advance_fixed(layer, 2)
            .map_err(|error| Error::backend(error.to_string()))?;
        self.trace.push("unit");
        Ok(FakeTensor(vec![
            input,
            layer.position,
            i32::try_from(keys.0.len()).unwrap(),
            i32::try_from(values.0.len()).unwrap(),
        ]))
    }

    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        _: &mut DeviceState<FakeBackend, HybridLayerState>,
        _: &Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.trace.push("output");
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

#[test]
fn heterogeneous_state_uses_the_component_extension_contract() {
    let architecture = HybridFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let calls = Rc::new(Cell::new(0));
    let mut factory = HybridStateFactory {
        calls: Rc::clone(&calls),
    };
    let mut state = realize_architecture_state::<FakeBackend, _, _>(&architecture, &mut factory)
        .expect("architecture-selected heterogeneous state realization");
    assert_eq!(state.layout().layer(0).unwrap().fixed_state().len(), 2);
    assert_eq!(calls.get(), 1);
    let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();
    let output = runtime.forward(11, &mut state, &()).unwrap();
    let layer = eredu_runtime::LayerRuntimeState::layer(&mut state, 0).unwrap();
    assert_eq!(RuntimeStateComponents::<FakeBackend>::position(layer), 2);
    assert_eq!(
        RuntimeStateComponents::<FakeBackend>::fixed_component(
            layer,
            StateTensorRole::Convolution { slot: 0 },
        )
        .unwrap()
        .as_ref()
        .unwrap(),
        &FakeTensor(vec![1, 4, 7])
    );
    assert_eq!(layer.recurrent, Some(FakeTensor(vec![1, 4, 11])));
    assert_eq!(layer.attention_keys, Some(FakeTensor(vec![0, 1, 11])));
    assert_eq!(layer.attention_values, Some(FakeTensor(vec![0, 1, 7])));
    assert_eq!(output, FakeTensor(vec![11, 2, 3, 3]));
    assert_eq!(runtime.architecture().trace, ["input", "unit", "output"]);
}

#[test]
fn named_frame_local_segment_resets_without_touching_persistent_state() {
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::segmented(
        LayerSchedule::new(4, policies).unwrap(),
        [
            StateSegmentSpec::new("temporal", 0..2, StateSegmentLifetime::Persistent, 0).unwrap(),
            StateSegmentSpec::new("depth", 2..4, StateSegmentLifetime::FrameLocal, 0).unwrap(),
        ],
    )
    .unwrap();
    let mut state = DeviceState::<FakeBackend, FakeLayerState>::create(layout, |layer, _| {
        Ok::<_, Infallible>(FakeLayerState(i32::try_from(layer).unwrap() + 1))
    })
    .unwrap();

    state
        .reset_segment(&StateSegmentId::new("depth").unwrap())
        .unwrap();
    assert_eq!(
        state
            .as_ref()
            .iter()
            .map(|layer| layer.0)
            .collect::<Vec<_>>(),
        [1, 2, 0, 0]
    );

    let error = state
        .reset_segment(&StateSegmentId::new("missing").unwrap())
        .unwrap_err();
    assert!(matches!(
        error,
        eredu_runtime::StateError::UnknownSegment { .. }
    ));
    assert_eq!(
        state
            .as_ref()
            .iter()
            .map(|layer| layer.0)
            .collect::<Vec<_>>(),
        [1, 2, 0, 0]
    );
}

#[derive(Debug)]
struct FakeUnit {
    marker: i32,
}

struct RecordingLease {
    ordinal: usize,
    unit: FakeUnit,
}

impl std::ops::Deref for RecordingLease {
    type Target = FakeUnit;

    fn deref(&self) -> &Self::Target {
        &self.unit
    }
}

impl std::ops::DerefMut for RecordingLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit
    }
}

struct RecordingPolicy {
    units: Vec<Option<FakeUnit>>,
    addresses: Vec<(usize, usize, usize)>,
    forward_active: bool,
    aborts: usize,
}

impl RecordingPolicy {
    fn new(units: Vec<FakeUnit>) -> Self {
        Self {
            units: units.into_iter().map(Some).collect(),
            addresses: Vec::new(),
            forward_active: false,
            aborts: 0,
        }
    }

    fn bounded(unit_count: usize) -> Self {
        Self {
            units: (0..unit_count).map(|_| None).collect(),
            addresses: Vec::new(),
            forward_active: false,
            aborts: 0,
        }
    }
}

impl LayerwisePolicy<FakeBackend, FakeUnit> for RecordingPolicy {
    type Lease = RecordingLease;
    type Error = &'static str;

    fn begin(&mut self, _: &FakeTensor, _: &()) -> Result<(), Self::Error> {
        if self.forward_active {
            return Err("fixture forward remained active");
        }
        self.forward_active = true;
        Ok(())
    }

    fn abort(&mut self, active: Option<(usize, ExecutionUnitAddress, Self::Lease)>, _: &()) {
        if let Some((ordinal, _, lease)) = active {
            assert_eq!(lease.ordinal, ordinal);
            assert!(self.units[ordinal].replace(lease.unit).is_none());
        }
        self.forward_active = false;
        self.aborts += 1;
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &(),
    ) -> Result<Self::Lease, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&()) -> Result<FakeUnit, E>,
    {
        self.addresses
            .push((ordinal, address.group(), address.index()));
        let slot = self
            .units
            .get_mut(ordinal)
            .ok_or("invalid fixture acquisition")
            .map_err(eredu_runtime::LayerwiseAcquireError::Policy)?;
        let unit = match slot.take() {
            Some(unit) => unit,
            None => build(context).map_err(eredu_runtime::LayerwiseAcquireError::Architecture)?,
        };
        Ok(RecordingLease { ordinal, unit })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        _: ExecutionUnitAddress,
        lease: Self::Lease,
        _: &'a FakeTensor,
        _: StateValues,
        _: ContextValues,
        _: &(),
    ) -> Result<(), Self::Error>
    where
        FakeTensor: 'a,
        StateValues: Iterator<Item = &'a FakeTensor>,
        ContextValues: Iterator<Item = &'a FakeTensor>,
    {
        if lease.ordinal != ordinal {
            return Err("fixture completion ordinal drifted");
        }
        self.units[ordinal] = Some(lease.unit);
        Ok(())
    }

    fn finish(&mut self, _: &FakeTensor, _: &()) -> Result<(), Self::Error> {
        self.forward_active = false;
        Ok(())
    }
}

type ReferencePromptCache = Rc<
    RefCell<
        Option<(
            DeviceState<FakeBackend, FakeLayerState>,
            eredu_core::cache::PromptCacheManifest,
        )>,
    >,
>;

struct ReferencePromptCacheSaveTransaction {
    cache: ReferencePromptCache,
    previous: Option<(
        DeviceState<FakeBackend, FakeLayerState>,
        eredu_core::cache::PromptCacheManifest,
    )>,
    candidate: (
        DeviceState<FakeBackend, FakeLayerState>,
        eredu_core::cache::PromptCacheManifest,
    ),
    published: bool,
}

struct ReferenceTextMechanisms {
    tasks: Rc<RefCell<Vec<ReplicatedTextMaterializationTask>>>,
    completions: Rc<RefCell<Vec<FakeTensor>>>,
    counters: ReplicatedSessionCounters,
    fail_completion: Rc<Cell<bool>>,
    fail_checkpoint: bool,
    prompt_cache: Option<ReferencePromptCache>,
}

impl<A> ReplicatedTextSessionMechanisms<A, FakeBackend> for ReferenceTextMechanisms
where
    A: LayeredArchitecture<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        StaticModules = FakeOperator,
        Unit = FakeUnit,
        Error = Error,
    >,
{
    type State = DeviceState<FakeBackend, FakeLayerState>;
    type PolicyError = &'static str;
    type ResidentPolicy = RecordingPolicy;
    type BoundedPolicy = RecordingPolicy;
    type StateCheckpoint = DeviceState<FakeBackend, FakeLayerState>;
    type StateReport = Vec<i32>;
    type ExecutionReport = (LayerWeightResidency, bool);
    type Error = &'static str;

    fn prepare_materialization(
        &mut self,
        _: &mut A,
        _: &eredu_runtime::ExecutionUnitLayout,
        _: &mut [FakeUnit],
        _: Option<&mut A>,
        _: Option<&mut [FakeUnit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        _: &[String],
        _: &(),
    ) -> Result<(), Self::Error> {
        self.counters.update(|counts| counts.materializations += 1);
        self.tasks.borrow_mut().extend_from_slice(tasks);
        Ok(())
    }

    fn realize_state(
        &mut self,
        selected: &eredu_runtime::SelectedStateRealization,
        _: &(),
    ) -> Result<Self::State, Self::Error> {
        self.counters.update(|counts| counts.state_allocations += 1);
        DeviceState::create(selected.layout().clone(), |_, _| {
            Ok::<_, &'static str>(FakeLayerState(0))
        })
    }

    fn resident_policy(
        &mut self,
        _: &mut A,
        units: Vec<FakeUnit>,
        _: &eredu_runtime::SelectedReplicatedTextRealization,
        _: &(),
    ) -> Result<Self::ResidentPolicy, Self::Error> {
        self.counters.update(|counts| counts.resident_policies += 1);
        Ok(RecordingPolicy::new(units))
    }

    fn bounded_policy(
        &mut self,
        _: &mut A,
        selected: &eredu_runtime::SelectedReplicatedTextRealization,
        _: &(),
    ) -> Result<Self::BoundedPolicy, Self::Error> {
        self.counters.update(|counts| counts.bounded_policies += 1);
        Ok(RecordingPolicy::bounded(
            selected.requirements().execution_units().len(),
        ))
    }

    fn index_text_output(
        &mut self,
        output: FakeTensor,
        sequence_index: i32,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        assert_eq!(sequence_index, -1);
        output
            .0
            .last()
            .copied()
            .map(|value| FakeTensor(vec![value]))
            .ok_or("empty text output")
    }

    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        _: &(),
    ) -> Result<Self::StateCheckpoint, Self::Error> {
        if self.fail_checkpoint {
            Err("injected state checkpoint failure")
        } else {
            Ok(state.clone())
        }
    }

    fn restore_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        _: &(),
    ) -> Result<(), Self::Error> {
        *state = checkpoint;
        Ok(())
    }

    fn load_prompt_cache(
        &mut self,
        _: &std::path::Path,
        _: &eredu_core::cache::PromptCacheDescriptor,
        _: &eredu_core::cache::PromptCacheModelIdentity,
        _: &[u32],
        _: &eredu_runtime::SelectedStateRealization,
        _: &(),
    ) -> Result<(Self::State, eredu_core::cache::PromptCacheManifest), Self::Error> {
        self.prompt_cache
            .as_ref()
            .and_then(|cache| cache.borrow().clone())
            .ok_or("prompt cache is not selected by this fixture")
    }

    fn save_prompt_cache(
        &mut self,
        state: &mut Self::State,
        _: &std::path::Path,
        descriptor: eredu_core::cache::PromptCacheDescriptor,
        prefix: &[u32],
        _: &eredu_core::cache::PromptCacheOptions,
        _: &(),
    ) -> Result<eredu_core::cache::PromptCacheManifest, Self::Error> {
        let cache = self
            .prompt_cache
            .as_ref()
            .ok_or("prompt cache is not selected by this fixture")?;
        let manifest = reference_prompt_cache_manifest(&descriptor, prefix)?;
        *cache.borrow_mut() = Some((state.clone(), manifest.clone()));
        Ok(manifest)
    }

    fn state_report(&self, state: &Self::State) -> Result<Self::StateReport, Self::Error> {
        if state.optional_layout().is_none() {
            return Ok(Vec::new());
        }
        Ok(state.as_ref().iter().map(|layer| layer.0).collect())
    }

    fn execution_report(
        &self,
        residency: LayerWeightResidency,
        bounded: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Self::Error> {
        Ok((residency, bounded.is_some()))
    }

    fn complete(
        &mut self,
        output: &FakeTensor,
        _: &Self::State,
        _: &(),
    ) -> Result<(), Self::Error> {
        PARTITION_COMMIT_TRACE.with(|trace| {
            if let Some(trace) = trace.borrow().as_ref() {
                trace.borrow_mut().push("complete");
            }
        });
        COMPOSITE_FORWARD_RESOURCE.with(|resource| {
            if let Some(dropped) = resource.borrow().as_ref() {
                assert!(
                    !dropped.get(),
                    "architecture forward resources were dropped before exact completion"
                );
            }
        });
        self.counters
            .update(|counts| counts.completion_attempts += 1);
        if self.fail_completion.get() {
            return Err("fixture completion failed");
        }
        self.completions.borrow_mut().push(output.clone());
        self.counters.update(|counts| counts.publications += 1);
        Ok(())
    }
}

impl<A> TransactionalPromptCacheMechanisms<A, FakeBackend> for ReferenceTextMechanisms
where
    A: LayeredArchitecture<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        StaticModules = FakeOperator,
        Unit = FakeUnit,
        Error = Error,
    >,
{
    type PromptCacheSaveTransaction = ReferencePromptCacheSaveTransaction;

    fn prepare_prompt_cache_save(
        &mut self,
        state: &mut Self::State,
        _: &std::path::Path,
        descriptor: eredu_core::cache::PromptCacheDescriptor,
        prefix: &[u32],
        options: &eredu_core::cache::PromptCacheOptions,
        _: &(),
    ) -> Result<Self::PromptCacheSaveTransaction, Self::Error> {
        let cache = self
            .prompt_cache
            .as_ref()
            .ok_or("prompt cache is not selected by this fixture")?;
        if cache.borrow().is_some() && !options.replace_existing() {
            return Err("prompt cache replacement was not selected");
        }
        Ok(ReferencePromptCacheSaveTransaction {
            cache: Rc::clone(cache),
            previous: cache.borrow().clone(),
            candidate: (
                state.clone(),
                reference_prompt_cache_manifest(&descriptor, prefix)?,
            ),
            published: false,
        })
    }

    fn prepared_prompt_cache_manifest(
        transaction: &Self::PromptCacheSaveTransaction,
    ) -> &eredu_core::cache::PromptCacheManifest {
        &transaction.candidate.1
    }

    fn publish_prompt_cache_save(
        &mut self,
        transaction: &mut Self::PromptCacheSaveTransaction,
    ) -> Result<(), Self::Error> {
        if transaction.published {
            return Err("prompt cache transaction was already published");
        }
        *transaction.cache.borrow_mut() = Some(transaction.candidate.clone());
        transaction.published = true;
        Ok(())
    }

    fn commit_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        assert!(transaction.published);
    }

    fn rollback_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction) {
        if transaction.published {
            *transaction.cache.borrow_mut() = transaction.previous;
        }
    }
}

fn reference_prompt_cache_manifest(
    descriptor: &eredu_core::cache::PromptCacheDescriptor,
    prefix: &[u32],
) -> Result<eredu_core::cache::PromptCacheManifest, &'static str> {
    let batch = i32::try_from(descriptor.batch_size()).map_err(|_| "invalid cache batch")?;
    let tokens = i32::try_from(prefix.len()).map_err(|_| "invalid cache prefix")?;
    let logical_bytes = u64::try_from(i64::from(batch) * i64::from(tokens) * 8)
        .map_err(|_| "invalid cache bytes")?;
    let blocks = (descriptor.global_layer_start()..descriptor.global_layer_end())
        .map(|layer| eredu_core::cache::PromptCacheBlock {
            global_layer: layer,
            representation: eredu_core::cache::CacheRepresentation::KeyValue,
            start: 0,
            end: i64::from(tokens),
            rank: None,
            shard: format!("blocks/layer-{layer}.safetensors"),
            first_array: "keys".into(),
            second_array: "values".into(),
            first_shape: vec![batch, 1, tokens, 1],
            second_shape: vec![batch, 1, tokens, 1],
            first_dtype: "Float32".into(),
            second_dtype: "Float32".into(),
            logical_bytes,
            payload_sha256: "0".repeat(64),
        })
        .collect();
    Ok(eredu_core::cache::PromptCacheManifest {
        schema_version: eredu_core::cache::PROMPT_CACHE_SCHEMA_VERSION,
        model_family: descriptor.model_family().into(),
        effective_model_type: descriptor.effective_model_type().into(),
        checkpoint_fingerprint: descriptor.checkpoint_fingerprint().into(),
        prefix_content_fingerprint: descriptor.prefix_content_fingerprint().into(),
        architecture_fingerprint: descriptor.architecture_fingerprint().into(),
        layer_count: descriptor.layer_count(),
        global_layer_start: descriptor.global_layer_start(),
        global_layer_end: descriptor.global_layer_end(),
        block_size_tokens: tokens,
        batch_size: descriptor.batch_size(),
        total_prefix_tokens: prefix.len(),
        prefix_sha256: eredu_core::cache::prompt_cache_token_fingerprint(prefix),
        layer_layout: descriptor.layer_layout().clone(),
        layer_prefix_offsets: descriptor.layer_prefix_offsets().to_vec(),
        state_segments: descriptor.state_segments().to_vec(),
        sink_tokens: descriptor.sink_tokens(),
        topology: descriptor.topology().clone(),
        distributed_commit: descriptor.distributed_commit(),
        application_namespace: None,
        blocks,
        state_tensors: Vec::new(),
    })
}

fn reference_prompt_cache_snapshot(
    cache: &ReferencePromptCache,
) -> Option<(Vec<i32>, eredu_core::cache::PromptCacheManifest)> {
    cache.borrow().as_ref().map(|(state, manifest)| {
        (
            state.as_ref().iter().map(|layer| layer.0).collect(),
            manifest.clone(),
        )
    })
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DeniedReferenceMechanism {
    Storage,
    State,
    Observation,
    Persistence,
    Completion,
}

fn try_select_reference_text(
    architecture: &OrdinaryTextFixture,
    residency: LayerWeightResidency,
    denied: Option<DeniedReferenceMechanism>,
) -> Result<
    eredu_runtime::SelectedReplicatedTextRealization,
    eredu_runtime::ReplicatedTextSelectionError,
> {
    let graph = architecture.execution_graph().unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let state_layout = architecture.state_layout().unwrap();
    let source =
        eredu_checkpoint::SourceTensorEncoding::Safetensors(eredu_checkpoint::StoredDtype::F32);
    let parameter = ReplicatedTextParameterRequirement::new(
        "decoder.weight",
        vec!["decoder.weight".into()],
        vec![ReplicatedTextPhysicalSource::new(
            "decoder.weight",
            "model.safetensors",
            "decoder.weight",
        )
        .unwrap()],
        vec!["decoder.weight.alias".into()],
        Some(source.clone()),
        Some(vec![1, 1]),
        vec![1, 1],
        eredu_checkpoint::LinearFormat::Dense,
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::ExecutionUnit {
            group: "decoder".into(),
            unit: 0,
        },
        ReplicatedTextParameterPresence::Required,
        eredu_runtime::ParameterTransformConstraint::None,
    )
    .unwrap();
    let lowering = WeightLoweringCapability::new(
        parameter
            .lowering_descriptor(eredu_checkpoint::LinearFormat::Dense)
            .unwrap(),
        WeightLoweringKind::Direct,
    );
    let requirements = ReplicatedTextRequirements::new(
        "ordinary-text-fixture",
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![architecture.group_transport(0)],
        state_layout.clone(),
        ReplicatedTextStateAccess::KeyValue,
        vec![parameter],
    )
    .unwrap();
    let component_mechanisms = if denied == Some(DeniedReferenceMechanism::State) {
        Vec::new()
    } else {
        state_layout
            .layers()
            .iter()
            .enumerate()
            .flat_map(|(layer, policy)| {
                policy
                    .components()
                    .into_iter()
                    .map(move |component| {
                        StateComponentMechanism::new(
                            layer,
                            component,
                            Some(StateComponentPlacement::Device),
                            None,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let state_mechanisms = StateMechanismCapabilities::new(component_mechanisms)
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(denied != Some(DeniedReferenceMechanism::Persistence))
        .with_observation_retention(denied != Some(DeniedReferenceMechanism::Observation));
    let weight_residencies = if denied == Some(DeniedReferenceMechanism::Storage) {
        vec![WeightResidencyMechanism::Windowed]
    } else {
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
        ]
    };
    let session_capabilities = if denied == Some(DeniedReferenceMechanism::Observation) {
        eredu_core::SessionCapabilities::default()
    } else {
        eredu_core::SessionCapabilities::new(false, true, true)
    };
    let capabilities = BackendMechanismCapabilities::new(
        NeuralOperatorCapabilities::NONE,
        vec![lowering],
        weight_residencies,
        state_mechanisms,
    )
    .with_session(session_capabilities)
    .with_prompt_cache(denied != Some(DeniedReferenceMechanism::Persistence))
    .with_exact_completion(denied != Some(DeniedReferenceMechanism::Completion));
    let request = ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true)
        .with_prompt_cache(true);
    select_replicated_text_realization(&requirements, &request, &capabilities)
}

fn selected_reference_text(
    architecture: &OrdinaryTextFixture,
    residency: LayerWeightResidency,
) -> eredu_runtime::SelectedReplicatedTextRealization {
    try_select_reference_text(architecture, residency, None).unwrap()
}

struct FinalOutputReplacement;

impl eredu_runtime::ActivationObserver<FakeTensor, Error> for FinalOutputReplacement {
    fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Error> {
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            PARTITION_COMMIT_TRACE.with(|trace| {
                if let Some(trace) = trace.borrow().as_ref() {
                    trace.borrow_mut().push("observe");
                }
            });
        }
        Ok(())
    }

    fn intervene(&mut self, path: &str, _: &FakeTensor) -> Result<Option<FakeTensor>, Error> {
        Ok((path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH).then(|| FakeTensor(vec![99])))
    }
}

struct CommitOrderingObserver;

impl eredu_runtime::ActivationObserver<FakeTensor, Error> for CommitOrderingObserver {
    fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Error> {
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            PARTITION_COMMIT_TRACE.with(|trace| {
                if let Some(trace) = trace.borrow().as_ref() {
                    trace.borrow_mut().push("observe");
                }
            });
        }
        Ok(())
    }
}

struct ReferencePartitionPass {
    hidden: FakeTensor,
    output: Option<FakeTensor>,
}

struct ReferencePartitionExecutor {
    architecture: OrdinaryTextFixture,
    unit: FakeUnit,
    fail_after_state: Rc<Cell<bool>>,
}

impl
    PartitionedGroupExecutor<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        (),
        (),
        FakeTensorMetadata,
    > for ReferencePartitionExecutor
{
    type Pass<'a> = ReferencePartitionPass;

    fn begin<'a>(
        &mut self,
        input: <OrdinaryTextFixture as LayeredArchitecture<
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >>::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: eredu_runtime::ExpertPass,
        _: &(),
    ) -> Result<Self::Pass<'a>, Error> {
        Ok(ReferencePartitionPass {
            hidden: input.clone(),
            output: None,
        })
    }

    fn request_group_active(&self, _: &Self::Pass<'_>, _: usize) -> Result<bool, Error> {
        Ok(false)
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &LayeredPartitionDriver,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &PartitionCommunication<FakeBackend, (), (), FakeTensorMetadata>,
        _: &(),
        context: &(),
        _: &mut O,
    ) -> Result<(), Error> {
        let input = driver
            .input(LayeredPartitionInput::<FakeTensor, NoAuxiliaryBoundary>::Tokens(&pass.hidden))
            .map_err(|error| Error::backend(error.to_string()))?;
        let mut forward = driver
            .begin::<FakeBackend, _, _>(&mut self.architecture, input, None, state, None, context)
            .map_err(|error| Error::backend(error.to_string()))?;
        forward.hidden = self.architecture.forward_unit(
            driver.group_index(),
            driver.range().start,
            &mut self.unit,
            &forward.hidden,
            state,
            &mut forward.context,
            context,
        )?;
        if self.fail_after_state.get() {
            return Err(Error::backend("injected partition failure"));
        }
        pass.output = Some(
            match driver.finish::<FakeBackend, _, _>(
                &mut self.architecture,
                &forward.hidden,
                state,
                &mut forward.context,
                None,
                context,
            )? {
                LayeredPartitionOutput::Final { output, .. } => output,
                LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
            },
        );
        Ok(())
    }

    fn boundary_values(
        &mut self,
        _: &mut Self::Pass<'_>,
        _: &eredu_runtime::PartitionBoundaryRoute,
        _: &eredu_runtime::ResolvedBoundaryWireSchema,
        _: bool,
        _: &(),
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<FakeTensor>>, Error> {
        Err(Error::backend("reference path has no pipeline boundary"))
    }

    fn boundary_schema(
        &self,
        _: &Self::Pass<'_>,
        _: &eredu_runtime::PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, Error> {
        Err(Error::backend("reference path has no pipeline boundary"))
    }

    fn accept_boundary(
        &mut self,
        _: &mut Self::Pass<'_>,
        _: &eredu_runtime::PartitionBoundaryRoute,
        _: Vec<FakeTensor>,
    ) -> Result<(), Error> {
        Err(Error::backend("reference path has no pipeline boundary"))
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
    ) -> Result<(FakeTensor, ()), Error> {
        Ok((pass.output.unwrap_or(pass.hidden), ()))
    }

    fn prediction_target_capture(&mut self, _: &(), _: &()) -> Result<Option<FakeTensor>, Error> {
        Ok(Some(FakeTensor(vec![7])))
    }
}

struct PipelineDestinationExecutor {
    architecture: OrdinaryTextFixture,
    unit: FakeUnit,
    trace: Rc<RefCell<Vec<&'static str>>>,
    boundary: eredu_runtime::ResolvedBoundaryWireSchema,
    send_boundary: bool,
    fail_boundary_values: bool,
    swap_auxiliary_roles: bool,
}

impl
    PartitionedGroupExecutor<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        (),
        (),
        PipelineTensorMetadata,
    > for PipelineDestinationExecutor
{
    type Pass<'a> = ReferencePartitionPass;

    fn begin<'a>(
        &mut self,
        input: <OrdinaryTextFixture as LayeredArchitecture<
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >>::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: eredu_runtime::ExpertPass,
        _: &(),
    ) -> Result<Self::Pass<'a>, Error> {
        Ok(ReferencePartitionPass {
            hidden: input.clone(),
            output: None,
        })
    }

    fn request_group_active(&self, _: &Self::Pass<'_>, _: usize) -> Result<bool, Error> {
        Ok(false)
    }

    fn execute_group<O: eredu_runtime::ActivationObserver<FakeTensor, Error> + ?Sized>(
        &mut self,
        pass: &mut Self::Pass<'_>,
        driver: &LayeredPartitionDriver,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &PartitionCommunication<FakeBackend, (), (), PipelineTensorMetadata>,
        _: &(),
        context: &(),
        _: &mut O,
    ) -> Result<(), Error> {
        self.trace.borrow_mut().push("execute");
        if pass.hidden != FakeTensor(vec![77]) {
            return Err(Error::backend(
                "pipeline group executed before receiving its boundary",
            ));
        }
        let input = driver
            .input(LayeredPartitionInput::Hidden {
                hidden: pass.hidden.clone(),
                auxiliary: NoAuxiliaryBoundary,
            })
            .map_err(|error| Error::backend(error.to_string()))?;
        let mut forward = driver
            .begin::<FakeBackend, _, _>(&mut self.architecture, input, None, state, None, context)
            .map_err(|error| Error::backend(error.to_string()))?;
        forward.hidden = self.architecture.forward_unit(
            driver.group_index(),
            driver.range().start,
            &mut self.unit,
            &forward.hidden,
            state,
            &mut forward.context,
            context,
        )?;
        pass.output = Some(
            match driver.finish::<FakeBackend, _, _>(
                &mut self.architecture,
                &forward.hidden,
                state,
                &mut forward.context,
                None,
                context,
            )? {
                LayeredPartitionOutput::Final { output, .. } => output,
                LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
            },
        );
        Ok(())
    }

    fn boundary_values(
        &mut self,
        pass: &mut Self::Pass<'_>,
        _: &PartitionBoundaryRoute,
        _: &eredu_runtime::ResolvedBoundaryWireSchema,
        source: bool,
        _: &(),
    ) -> Result<Vec<eredu_runtime::ArchitectureBoundaryValue<FakeTensor>>, Error> {
        if source {
            if !self.send_boundary {
                return Err(Error::backend("destination fixture was asked to send"));
            }
            self.trace.borrow_mut().push("send");
            return Ok(vec![eredu_runtime::ArchitectureBoundaryValue::new(
                "hidden",
                pass.output
                    .clone()
                    .ok_or_else(|| Error::backend("middle-stage fixture has no output"))?,
            )
            .map_err(|error| Error::backend(error.to_string()))?]);
        }
        self.trace.borrow_mut().push("receive");
        if self.fail_boundary_values {
            return Err(Error::backend("injected destination placeholder failure"));
        }
        if self.swap_auxiliary_roles {
            return Ok(vec![
                eredu_runtime::ArchitectureBoundaryValue::new("hidden", FakeTensor(vec![77]))
                    .unwrap(),
                eredu_runtime::ArchitectureBoundaryValue::new("second", FakeTensor(vec![77]))
                    .unwrap(),
                eredu_runtime::ArchitectureBoundaryValue::new("first", FakeTensor(vec![77]))
                    .unwrap(),
            ]);
        }
        Ok(vec![eredu_runtime::ArchitectureBoundaryValue::new(
            "hidden",
            FakeTensor(vec![77]),
        )
        .map_err(|error| Error::backend(error.to_string()))?])
    }

    fn boundary_schema(
        &self,
        _: &Self::Pass<'_>,
        _: &PartitionBoundaryRoute,
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, Error> {
        Ok(self.boundary.clone())
    }

    fn accept_boundary(
        &mut self,
        pass: &mut Self::Pass<'_>,
        _: &PartitionBoundaryRoute,
        mut values: Vec<FakeTensor>,
    ) -> Result<(), Error> {
        self.trace.borrow_mut().push("accept");
        pass.hidden = values
            .pop()
            .ok_or_else(|| Error::backend("pipeline boundary was empty"))?;
        Ok(())
    }

    fn finish(
        &mut self,
        pass: Self::Pass<'_>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
    ) -> Result<(FakeTensor, ()), Error> {
        Ok((pass.output.unwrap_or(pass.hidden), ()))
    }
}

#[test]
fn production_replicated_text_constructor_executes_reference_mechanisms() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let counters = ReplicatedSessionCounters::default();
        let architecture = OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: Vec::new(),
            counters: counters.clone(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        };
        let selected = selected_reference_text(&architecture, residency);
        let expected_identity = selected.requirements().architecture_identity().to_owned();
        let contract = prepare_replicated_text_contract::<
            _,
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >(&architecture, None, selected, &expected_identity, &())
        .unwrap();
        let tasks = Rc::new(RefCell::new(Vec::new()));
        let completions = Rc::new(RefCell::new(Vec::new()));
        let fail_completion = Rc::new(Cell::new(false));
        let mechanisms = ReferenceTextMechanisms {
            tasks: Rc::clone(&tasks),
            completions: Rc::clone(&completions),
            counters: counters.clone(),
            fail_completion: Rc::clone(&fail_completion),
            fail_checkpoint: false,
            prompt_cache: None,
        };
        let mut session = construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            contract,
            mechanisms,
            &(),
        )
        .unwrap();

        let expected_policy_counts = if residency.is_fully_resident() {
            (1, 0)
        } else {
            (0, 1)
        };
        assert_eq!(
            counters.snapshot(),
            ReplicatedSessionCounts {
                materializations: 1,
                resident_policies: expected_policy_counts.0,
                bounded_policies: expected_policy_counts.1,
                unit_constructions: 1,
                state_allocations: 1,
                ..ReplicatedSessionCounts::default()
            }
        );

        assert_eq!(tasks.borrow().len(), 1);
        let task = &tasks.borrow()[0];
        assert_eq!(task.name(), "decoder.weight");
        assert_eq!(task.sources(), ["decoder.weight"]);
        assert_eq!(task.physical_sources()[0].output(), "decoder.weight");
        assert_eq!(task.logical_shape(), [1, 1]);
        assert_eq!(task.physical_shape(), [1, 1]);
        assert_eq!(task.lowering(), WeightLoweringKind::Direct);

        let prompt = FakeTensor(vec![1, 2]);
        let input = <OrdinaryTextFixture as ReplicatedTextArchitecture<
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >>::text_input(&prompt, None);
        assert_eq!(
            session.prefill_input(input, &()).unwrap(),
            FakeTensor(vec![5])
        );
        assert_eq!(
            session.decode(&FakeTensor(vec![3]), &()).unwrap(),
            FakeTensor(vec![5])
        );
        assert_eq!(session.report().unwrap().state_report(), &[2]);
        assert_eq!(counters.snapshot().forward_calls, 2);
        assert_eq!(counters.snapshot().completion_attempts, 2);
        assert_eq!(counters.snapshot().publications, 2);

        let before_prediction = session.report().unwrap().state_report().to_vec();
        let prediction_error = session
            .prefill_prediction_target(&FakeTensor(vec![9, 10]), None, &())
            .expect_err("an architecture without a target capture must fail closed");
        assert!(prediction_error
            .to_string()
            .contains("did not retain its declared hidden capture"));
        assert_eq!(session.report().unwrap().state_report(), &before_prediction);
        assert_eq!(counters.snapshot().publications, 2);
        assert_eq!(counters.snapshot().completion_attempts, 2);

        let checkpoint = session.checkpoint(&()).unwrap();
        session.decode(&FakeTensor(vec![4]), &()).unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[3]);
        assert_eq!(counters.snapshot().publications, 3);
        session.rollback(checkpoint, &()).unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[2]);

        fail_completion.set(true);
        assert!(session.decode(&FakeTensor(vec![8]), &()).is_err());
        assert_eq!(session.report().unwrap().state_report(), &[2]);
        assert_eq!(counters.snapshot().forward_calls, 5);
        assert_eq!(counters.snapshot().completion_attempts, 4);
        assert_eq!(counters.snapshot().publications, 3);
        fail_completion.set(false);

        let observed = session
            .forward_with_observer(&FakeTensor(vec![7]), None, &(), &mut FinalOutputReplacement)
            .unwrap();
        assert_eq!(observed, FakeTensor(vec![99]));
        assert_eq!(completions.borrow().last(), Some(&FakeTensor(vec![99])));
        assert_eq!(counters.snapshot().forward_calls, 6);
        assert_eq!(counters.snapshot().completion_attempts, 5);
        assert_eq!(counters.snapshot().publications, 4);

        session.reset(&()).unwrap();
        let report = session.report().unwrap();
        assert_eq!(report.execution(), residency.execution_residency());
        assert_eq!(report.state_report(), &[0]);
        assert_eq!(
            report.execution_report(),
            &(residency, !residency.is_fully_resident())
        );
        assert_eq!(counters.snapshot().state_allocations, 2);
        assert_eq!(completions.borrow().len(), counters.snapshot().publications);
    }
}

#[test]
fn prediction_target_capture_uses_one_authoritative_transaction_and_publication() {
    let counters = ReplicatedSessionCounters::default();
    let architecture = PredictionCaptureFixture(OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    });
    let selected = selected_reference_text(&architecture.0, LayerWeightResidency::FullyResident);
    let expected_identity = selected.requirements().architecture_identity().to_owned();
    let contract = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(&architecture, None, selected, &expected_identity, &())
    .unwrap();
    let completions = Rc::new(RefCell::new(Vec::new()));
    let mut session = construct_replicated_text_session::<_, FakeBackend, _>(
        architecture,
        None,
        contract,
        ReferenceTextMechanisms {
            tasks: Rc::new(RefCell::new(Vec::new())),
            completions: Rc::clone(&completions),
            counters: counters.clone(),
            fail_completion: Rc::new(Cell::new(false)),
            fail_checkpoint: false,
            prompt_cache: None,
        },
        &(),
    )
    .unwrap();

    let (logits, capture) = session
        .prefill_prediction_target(&FakeTensor(vec![1, 2]), None, &())
        .unwrap();
    assert_eq!(logits, FakeTensor(vec![1, 2, 5]));
    assert_eq!(capture, FakeTensor(vec![1, 2, 5]));
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert_eq!(counters.snapshot().forward_calls, 1);
    assert_eq!(counters.snapshot().completion_attempts, 1);
    assert_eq!(counters.snapshot().publications, 1);
    assert_eq!(
        completions.borrow().as_slice(),
        &[FakeTensor(vec![1, 2, 5])]
    );

    let mut lane = session.prepare_prediction_target_state(&()).unwrap();
    session
        .exchange_prediction_target_state(&mut lane, &())
        .unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[0]);
    session
        .exchange_prediction_target_state(&mut lane, &())
        .unwrap();
    assert_eq!(session.report().unwrap().state_report(), &[1]);
    assert_eq!(lane.as_ref()[0].0, 0);

    assert_eq!(
        session
            .apply_prediction_target_operation(PredictionStateOperation { fail: false }, &())
            .unwrap(),
        11
    );
    let before_failure = session.report().unwrap().state_report().to_vec();
    assert!(session
        .apply_prediction_target_operation(PredictionStateOperation { fail: true }, &())
        .is_err());
    assert_eq!(session.report().unwrap().state_report(), &before_failure);
}

#[test]
fn production_partitioned_constructor_reuses_session_rollback_and_stateless_rank() {
    type Strategy = PartitionedTextExecution<
        ReferencePartitionExecutor,
        (),
        (),
        FakeTensorMetadata,
        NoBoundaryTransport,
        NoOutputPublisher,
        OpaqueCommitAgreement,
    >;

    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let fail_after_state = Rc::new(Cell::new(false));
    let commit_group = CommunicationGroupId::new(1);
    let commit_requirements =
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap();
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        architecture,
        selected,
        partition,
        CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                commit_group,
                0,
                vec![0],
                Some(0),
                commit_requirements,
            )
            .unwrap()],
            Vec::new(),
        )
        .unwrap()
        .with_completion_policy(test_completion_policy()),
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, partition, manifest, tasks) = input.into_parts();
            assert_eq!(tasks.len(), 1);
            let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![Some(driver)],
                Vec::new(),
                None,
                Some(commit_group),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                vec![RealizedCommunicationGroup::new(commit_group, ())],
                Vec::new(),
                FakeTensorMetadata,
            )
            .unwrap();
            let state = DeviceState::create(partition.state().unwrap().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(0))
            })
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                ReferencePartitionExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    fail_after_state: Rc::clone(&fail_after_state),
                },
                communication,
                (),
                NoBoundaryTransport,
                NoOutputPublisher,
                OpaqueCommitAgreement,
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((runtime, state))
        },
    )
    .unwrap();
    let fail_completion = Rc::new(Cell::new(false));
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters: counters.clone(),
        fail_completion: Rc::clone(&fail_completion),
        fail_checkpoint: false,
        prompt_cache: None,
    };
    let mut session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        Strategy,
    >(binding, mechanisms, Strategy::new())
    .unwrap();

    assert_eq!(
        session.decode(&FakeTensor(vec![3]), &()).unwrap(),
        FakeTensor(vec![5])
    );
    assert_eq!(session.report().unwrap().state_report(), &[1]);

    let commit_trace = Rc::new(RefCell::new(Vec::new()));
    PARTITION_COMMIT_TRACE.with(|trace| *trace.borrow_mut() = Some(Rc::clone(&commit_trace)));
    session
        .forward_with_observer(&FakeTensor(vec![4]), None, &(), &mut CommitOrderingObserver)
        .unwrap();
    PARTITION_COMMIT_TRACE.with(|trace| *trace.borrow_mut() = None);
    assert_eq!(*commit_trace.borrow(), ["observe", "complete", "commit"]);

    fail_completion.set(true);
    commit_trace.borrow_mut().clear();
    PARTITION_COMMIT_TRACE.with(|trace| *trace.borrow_mut() = Some(Rc::clone(&commit_trace)));
    assert!(session
        .forward_with_observer(&FakeTensor(vec![5]), None, &(), &mut CommitOrderingObserver,)
        .is_err());
    PARTITION_COMMIT_TRACE.with(|trace| *trace.borrow_mut() = None);
    fail_completion.set(false);
    assert_eq!(*commit_trace.borrow(), ["observe", "complete"]);

    fail_after_state.set(true);
    assert!(session.decode(&FakeTensor(vec![9]), &()).is_err());
    assert_eq!(session.report().unwrap().state_report(), &[2]);

    let stateless_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let stateless_selected =
        selected_reference_text(&stateless_architecture, LayerWeightResidency::FullyResident);
    let stateless_parameters = stateless_architecture.parameter_description(&()).unwrap();
    let stateless_partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        String,
    >(
        &stateless_architecture,
        std::iter::empty(),
        PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &stateless_parameters,
    )
    .unwrap();
    let invalid_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let invalid_parameters = invalid_architecture.parameter_description(&()).unwrap();
    let invalid_partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        String,
    >(
        &invalid_architecture,
        std::iter::empty(),
        PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &invalid_parameters,
    )
    .unwrap();
    let invalid = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        invalid_architecture,
        stateless_selected.clone(),
        invalid_partition,
        CommunicationManifest::new(2, 1, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(test_completion_policy()),
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |_input, selected, _context| {
            let state = DeviceState::create(selected.state().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(42))
            })
            .unwrap();
            Ok(((), state))
        },
    );
    assert!(matches!(
        invalid,
        Err(error) if error.to_string().contains("contains mutable state geometry")
    ));
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        stateless_architecture,
        stateless_selected,
        stateless_partition,
        CommunicationManifest::new(2, 1, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(test_completion_policy()),
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, _partition, manifest, tasks) = input.into_parts();
            assert!(tasks.is_empty());
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![None],
                Vec::new(),
                None,
                None,
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                Vec::new(),
                Vec::new(),
                FakeTensorMetadata,
            )
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                ReferencePartitionExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    fail_after_state: Rc::new(Cell::new(false)),
                },
                communication,
                (),
                NoBoundaryTransport,
                NoOutputPublisher,
                OpaqueCommitAgreement,
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((
                runtime,
                DeviceState::<FakeBackend, FakeLayerState>::stateless(),
            ))
        },
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters,
        fail_completion: Rc::new(Cell::new(false)),
        fail_checkpoint: false,
        prompt_cache: None,
    };
    let mut stateless_session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        Strategy,
    >(binding, mechanisms, Strategy::new())
    .unwrap();
    assert!(stateless_session
        .report()
        .unwrap()
        .state_report()
        .is_empty());
    stateless_session.reset(&()).unwrap();
    assert!(stateless_session
        .report()
        .unwrap()
        .state_report()
        .is_empty());
}

#[derive(Clone)]
struct ScriptedPhaseAgreement {
    failed_phase: DistributedExecutionPhase,
    calls: Rc<RefCell<Vec<(DistributedExecutionPhase, bool)>>>,
    commits: Rc<Cell<usize>>,
}

impl<I> PartitionCommitAgreement<FakeBackend, (), (), I> for ScriptedPhaseAgreement
where
    I: CommunicationTensorMetadata<FakeBackend>,
{
    const ENABLED: bool = true;
    const PHASE_FAILURE_AGREEMENT: bool = true;

    fn agree_phase(
        &mut self,
        _: &PartitionCommunication<FakeBackend, (), (), I>,
        _: CommunicationGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        _: &(),
    ) -> Result<bool, eredu_runtime::PartitionExecutionError> {
        self.calls.borrow_mut().push((phase, local_success));
        Ok(local_success && phase != self.failed_phase)
    }

    fn commit(
        &mut self,
        _: &PartitionCommunication<FakeBackend, (), (), I>,
        _: CommunicationGroupId,
        epoch: DistributedCommitEpoch,
        _: &(),
    ) -> DistributedCommitOutcome {
        PARTITION_COMMIT_TRACE.with(|trace| {
            if let Some(trace) = trace.borrow().as_ref() {
                trace.borrow_mut().push("commit");
            }
        });
        self.commits.set(self.commits.get() + 1);
        DistributedCommitOutcome::Committed(epoch)
    }
}

struct ConcurrentCheckpointCoordinator {
    rendezvous: std::sync::Barrier,
    failed: std::sync::atomic::AtomicBool,
    calls: std::sync::Mutex<Vec<(usize, DistributedExecutionPhase, bool)>>,
}

impl ConcurrentCheckpointCoordinator {
    fn new() -> Self {
        Self {
            rendezvous: std::sync::Barrier::new(2),
            failed: std::sync::atomic::AtomicBool::new(false),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[derive(Clone)]
struct ConcurrentCheckpointAgreement {
    rank: usize,
    coordinator: Arc<ConcurrentCheckpointCoordinator>,
}

enum TestPhaseAgreement {
    Scripted(ScriptedPhaseAgreement),
    Concurrent(ConcurrentCheckpointAgreement),
    Final {
        result: FinalDecisionResult,
        commits: Rc<Cell<usize>>,
    },
}

#[derive(Clone, Copy)]
enum FinalDecisionResult {
    Committed,
    Aborted,
    Indeterminate(DistributedCommitPhase),
}

impl<I> PartitionCommitAgreement<FakeBackend, (), (), I> for TestPhaseAgreement
where
    I: CommunicationTensorMetadata<FakeBackend>,
{
    const ENABLED: bool = true;
    const PHASE_FAILURE_AGREEMENT: bool = true;

    fn agree_phase(
        &mut self,
        communication: &PartitionCommunication<FakeBackend, (), (), I>,
        group: CommunicationGroupId,
        phase: DistributedExecutionPhase,
        local_success: bool,
        executor: &(),
    ) -> Result<bool, eredu_runtime::PartitionExecutionError> {
        match self {
            Self::Scripted(scripted) => {
                scripted.agree_phase(communication, group, phase, local_success, executor)
            }
            Self::Concurrent(concurrent) => {
                assert_eq!(phase, DistributedExecutionPhase::StateCheckpoint);
                concurrent.coordinator.calls.lock().unwrap().push((
                    concurrent.rank,
                    phase,
                    local_success,
                ));
                if !local_success {
                    concurrent.coordinator.failed.store(true, Ordering::SeqCst);
                }
                concurrent.coordinator.rendezvous.wait();
                let agreed = !concurrent.coordinator.failed.load(Ordering::SeqCst);
                concurrent.coordinator.rendezvous.wait();
                Ok(agreed)
            }
            Self::Final { .. } => Ok(local_success),
        }
    }

    fn commit(
        &mut self,
        communication: &PartitionCommunication<FakeBackend, (), (), I>,
        group: CommunicationGroupId,
        epoch: DistributedCommitEpoch,
        executor: &(),
    ) -> DistributedCommitOutcome {
        match self {
            Self::Scripted(scripted) => scripted.commit(communication, group, epoch, executor),
            Self::Concurrent(_) => panic!("checkpoint-failure fixture must not reach commit"),
            Self::Final { result, commits } => {
                commits.set(commits.get() + 1);
                match result {
                    FinalDecisionResult::Committed => DistributedCommitOutcome::Committed(epoch),
                    FinalDecisionResult::Aborted => DistributedCommitOutcome::Aborted(epoch),
                    FinalDecisionResult::Indeterminate(phase) => {
                        DistributedCommitOutcome::Indeterminate {
                            epoch,
                            phase: *phase,
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum LocalPartitionFailure {
    None,
    Checkpoint,
    Execution,
    Observation,
    Intervention,
    Publication,
    Completion,
}

#[derive(Clone, Copy)]
struct ScriptedOutputPublisher {
    fail: bool,
}

impl<I> PartitionOutputPublisher<FakeBackend, (), (), I> for ScriptedOutputPublisher
where
    I: CommunicationTensorMetadata<FakeBackend>,
{
    const ENABLED: bool = true;

    fn publish(
        &mut self,
        _: &PartitionCommunication<FakeBackend, (), (), I>,
        value: FakeTensor,
        _: PartitionOutputPublication,
        _: DistributedExecutionPhase,
        _: &(),
    ) -> Result<FakeTensor, eredu_runtime::PartitionExecutionError> {
        PARTITION_COMMIT_TRACE.with(|trace| {
            if let Some(trace) = trace.borrow().as_ref() {
                trace.borrow_mut().push("publish");
            }
        });
        PARTITION_PUBLICATION_VALUES.with(|values| {
            if let Some(values) = values.borrow().as_ref() {
                values.borrow_mut().push(value.clone());
            }
        });
        if self.fail {
            Err(eredu_runtime::PartitionExecutionError::Communication(
                "injected output publication failure".into(),
            ))
        } else {
            Ok(value)
        }
    }
}

struct FailingFinalOutputObserver;

impl eredu_runtime::ActivationObserver<FakeTensor, Error> for FailingFinalOutputObserver {
    fn observe(&mut self, path: &str, _: &FakeTensor) -> Result<(), Error> {
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            Err(Error::backend("injected final output observation failure"))
        } else {
            Ok(())
        }
    }
}

struct AgreementRun {
    failed: bool,
    state: Vec<i32>,
    completions: usize,
    output: Option<FakeTensor>,
    forwards: usize,
    outcome: Option<DistributedCommitOutcome>,
    retry_forwards: usize,
    cached_outcome: Option<DistributedCommitOutcome>,
}

type ReferencePartitionedStrategy = PartitionedTextExecution<
    ReferencePartitionExecutor,
    (),
    (),
    FakeTensorMetadata,
    NoBoundaryTransport,
    ScriptedOutputPublisher,
    TestPhaseAgreement,
>;

type ReferencePartitionedSession = ReplicatedTextSession<
    OrdinaryTextFixture,
    FakeBackend,
    ReferenceTextMechanisms,
    ReferencePartitionedStrategy,
>;

fn run_partitioned_agreement_rank(
    rank: usize,
    local_failure: LocalPartitionFailure,
    failed_phase: DistributedExecutionPhase,
    calls: Rc<RefCell<Vec<(DistributedExecutionPhase, bool)>>>,
    commits: Rc<Cell<usize>>,
) -> AgreementRun {
    run_partitioned_agreement_rank_with(
        rank,
        local_failure,
        TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
            failed_phase,
            calls,
            commits,
        }),
    )
}

fn run_partitioned_agreement_rank_with(
    rank: usize,
    local_failure: LocalPartitionFailure,
    agreement: TestPhaseAgreement,
) -> AgreementRun {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let prompt_model_identity = partition
        .state()
        .unwrap()
        .prompt_cache_identity::<FakeBackend, OrdinaryTextFixture>(
            &architecture,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap();
    let group = CommunicationGroupId::new(41);
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![CommunicationGroupDescriptor::new(
            group,
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 3, 8, None).unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        Vec::new(),
    )
    .unwrap()
    .with_completion_policy(test_completion_policy());
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        architecture,
        selected,
        partition,
        manifest,
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, partition, manifest, _) = input.into_parts();
            let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![Some(driver)],
                Vec::new(),
                Some(PartitionOutputPublication {
                    group,
                    owner_rank: 0,
                }),
                Some(group),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                vec![RealizedCommunicationGroup::new(group, ())],
                Vec::new(),
                FakeTensorMetadata,
            )
            .unwrap();
            let state = DeviceState::create(partition.state().unwrap().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(0))
            })
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                ReferencePartitionExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    fail_after_state: Rc::new(Cell::new(matches!(
                        local_failure,
                        LocalPartitionFailure::Execution
                    ))),
                },
                communication,
                (),
                NoBoundaryTransport,
                ScriptedOutputPublisher {
                    fail: matches!(local_failure, LocalPartitionFailure::Publication),
                },
                agreement,
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((runtime, state))
        },
    )
    .unwrap();
    let prompt_cache = Rc::new(RefCell::new(None));
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters: counters.clone(),
        fail_completion: Rc::new(Cell::new(matches!(
            local_failure,
            LocalPartitionFailure::Completion
        ))),
        fail_checkpoint: matches!(local_failure, LocalPartitionFailure::Checkpoint),
        prompt_cache: Some(Rc::clone(&prompt_cache)),
    };
    let mut session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        ReferencePartitionedStrategy,
    >(binding, mechanisms, ReferencePartitionedStrategy::new())
    .unwrap();
    let result = match local_failure {
        LocalPartitionFailure::Observation => session.forward_with_observer(
            &FakeTensor(vec![3]),
            None,
            &(),
            &mut FailingFinalOutputObserver,
        ),
        LocalPartitionFailure::Intervention => session.forward_with_observer(
            &FakeTensor(vec![3]),
            None,
            &(),
            &mut FinalOutputReplacement,
        ),
        _ => session.decode(&FakeTensor(vec![3]), &()),
    };
    let output = result.as_ref().ok().cloned();
    let report = session.report().unwrap();
    let state = report.state_report().to_vec();
    let outcome = report.distributed_commit();
    let forward_calls = counters.snapshot().forward_calls;
    let retry_calls = if outcome.is_some_and(DistributedCommitOutcome::is_indeterminate) {
        assert!(session.decode(&FakeTensor(vec![4]), &()).is_err());
        let retry_calls = counters.snapshot().forward_calls - forward_calls;
        let allocations = counters.snapshot().state_allocations;
        assert!(session.reset(&()).is_err());
        assert_eq!(counters.snapshot().state_allocations, allocations);
        retry_calls
    } else {
        0
    };
    let cached_outcome = if outcome.is_some_and(DistributedCommitOutcome::is_indeterminate) {
        let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
            prompt_model_identity,
            "fixture-checkpoint",
            "fixture-prefix",
            1,
        )
        .unwrap();
        session
            .save_prompt_cache(
                std::path::Path::new("unused"),
                descriptor.clone(),
                &[3],
                &eredu_core::cache::PromptCacheOptions::default(),
                &(),
            )
            .unwrap();
        session
            .load_prompt_cache(std::path::Path::new("unused"), &descriptor, &[3], &())
            .unwrap();
        session.report().unwrap().distributed_commit()
    } else {
        None
    };
    AgreementRun {
        failed: result.is_err(),
        state,
        completions: counters.snapshot().completion_attempts,
        output,
        forwards: forward_calls,
        outcome,
        retry_forwards: retry_calls,
        cached_outcome,
    }
}

#[allow(clippy::type_complexity)]
fn partitioned_cache_control_session(
    rank: usize,
    failed_phase: DistributedExecutionPhase,
    prompt_cache: Option<ReferencePromptCache>,
    mechanism_available: bool,
) -> (
    ReferencePartitionedSession,
    eredu_core::cache::PromptCacheDescriptor,
    ReferencePromptCache,
    Rc<RefCell<Vec<(DistributedExecutionPhase, bool)>>>,
    ReplicatedSessionCounters,
) {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let identity = partition
        .state()
        .unwrap()
        .prompt_cache_identity::<FakeBackend, OrdinaryTextFixture>(
            &architecture,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap();
    let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
        identity,
        "fixture-checkpoint",
        "fixture-prefix",
        1,
    )
    .unwrap();
    let group = CommunicationGroupId::new(41);
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![CommunicationGroupDescriptor::new(
            group,
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 3, 8, None).unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        Vec::new(),
    )
    .unwrap()
    .with_completion_policy(test_completion_policy());
    let calls = Rc::new(RefCell::new(Vec::new()));
    let agreement = TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
        failed_phase,
        calls: Rc::clone(&calls),
        commits: Rc::new(Cell::new(0)),
    });
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        architecture,
        selected,
        partition,
        manifest,
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, partition, manifest, _) = input.into_parts();
            let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![Some(driver)],
                Vec::new(),
                Some(PartitionOutputPublication {
                    group,
                    owner_rank: 0,
                }),
                Some(group),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                vec![RealizedCommunicationGroup::new(group, ())],
                Vec::new(),
                FakeTensorMetadata,
            )
            .unwrap();
            let state = DeviceState::create(partition.state().unwrap().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(0))
            })
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                ReferencePartitionExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    fail_after_state: Rc::new(Cell::new(false)),
                },
                communication,
                (),
                NoBoundaryTransport,
                ScriptedOutputPublisher { fail: false },
                agreement,
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((runtime, state))
        },
    )
    .unwrap();
    let prompt_cache = prompt_cache.unwrap_or_else(|| Rc::new(RefCell::new(None)));
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters: counters.clone(),
        fail_completion: Rc::new(Cell::new(false)),
        fail_checkpoint: false,
        prompt_cache: mechanism_available.then(|| Rc::clone(&prompt_cache)),
    };
    let session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        ReferencePartitionedStrategy,
    >(binding, mechanisms, ReferencePartitionedStrategy::new())
    .unwrap();
    (session, descriptor, prompt_cache, calls, counters)
}

#[test]
fn distributed_prompt_cache_save_is_reversible_and_fences_one_rank_failures() {
    let replacement_options = eredu_core::cache::PromptCacheOptions::new(None, true).unwrap();
    let (mut local_failure, descriptor, local_store, local_calls, local_counters) =
        partitioned_cache_control_session(
            0,
            DistributedExecutionPhase::PromptCacheSavePreparation,
            None,
            false,
        );
    let error = local_failure
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[3],
            &replacement_options,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("mechanism failed"));
    assert!(local_store.borrow().is_none());
    assert_eq!(
        local_calls.borrow().as_slice(),
        &[
            (DistributedExecutionPhase::PromptCacheSavePreflight, true),
            (DistributedExecutionPhase::PromptCacheSavePreparation, false),
        ]
    );
    let call_count = local_calls.borrow().len();
    assert!(local_failure
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[3],
            &replacement_options,
            &(),
        )
        .is_err());
    assert_eq!(local_calls.borrow().len(), call_count);
    assert_eq!(local_counters.snapshot().forward_calls, 0);

    let (mut peer, descriptor, peer_store, peer_calls, _) = partitioned_cache_control_session(
        1,
        DistributedExecutionPhase::PromptCacheSavePreparation,
        None,
        true,
    );
    peer.save_prompt_cache(
        std::path::Path::new("unused"),
        descriptor.clone(),
        &[1],
        &eredu_core::cache::PromptCacheOptions::default(),
        &(),
    )
    .unwrap();
    let previous = reference_prompt_cache_snapshot(&peer_store);
    assert!(peer
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor,
            &[3],
            &replacement_options,
            &(),
        )
        .is_err());
    assert_eq!(reference_prompt_cache_snapshot(&peer_store), previous);
    assert_eq!(
        peer_calls.borrow().last(),
        Some(&(DistributedExecutionPhase::PromptCacheSavePreparation, true))
    );

    let (mut replacement, descriptor, replacement_store, replacement_calls, _) =
        partitioned_cache_control_session(
            1,
            DistributedExecutionPhase::PromptCacheSavePublication,
            None,
            true,
        );
    replacement
        .save_prompt_cache(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[1],
            &replacement_options,
            &(),
        )
        .unwrap();
    let previous = reference_prompt_cache_snapshot(&replacement_store);
    assert!(replacement
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor,
            &[3],
            &replacement_options,
            &(),
        )
        .is_err());
    assert_eq!(
        reference_prompt_cache_snapshot(&replacement_store),
        previous
    );
    assert_eq!(
        replacement_calls.borrow().last(),
        Some(&(DistributedExecutionPhase::PromptCacheSavePublication, true))
    );

    let (mut successful, descriptor, successful_store, _, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreflight,
        None,
        true,
    );
    successful
        .save_prompt_cache(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[1],
            &eredu_core::cache::PromptCacheOptions::default(),
            &(),
        )
        .unwrap();
    let old_prefix = successful_store
        .borrow()
        .as_ref()
        .unwrap()
        .1
        .prefix_sha256
        .clone();
    let manifest = successful
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor,
            &[3],
            &replacement_options,
            &(),
        )
        .unwrap()
        .unwrap();
    assert_ne!(manifest.prefix_sha256, old_prefix);
    assert_eq!(
        successful_store.borrow().as_ref().unwrap().1.prefix_sha256,
        manifest.prefix_sha256
    );
}

#[test]
fn distributed_prompt_cache_load_is_provisional_until_every_rank_succeeds() {
    let (mut local_failure, descriptor, store, calls, counters) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreparation,
        None,
        true,
    );
    assert!(local_failure
        .load_prompt_cache_distributed(std::path::Path::new("unused"), &descriptor, &[3], &(),)
        .is_err());
    assert_eq!(local_failure.report().unwrap().state_report(), &[0]);
    assert!(store.borrow().is_none());
    assert_eq!(
        calls.borrow().last(),
        Some(&(DistributedExecutionPhase::PromptCacheLoadPreparation, false))
    );
    let calls_before_retry = calls.borrow().len();
    assert!(local_failure
        .load_prompt_cache_distributed(std::path::Path::new("unused"), &descriptor, &[3], &(),)
        .is_err());
    assert_eq!(calls.borrow().len(), calls_before_retry);
    assert_eq!(counters.snapshot().forward_calls, 0);

    let (mut peer, descriptor, _, peer_calls, _) = partitioned_cache_control_session(
        1,
        DistributedExecutionPhase::PromptCacheLoadPreparation,
        None,
        true,
    );
    peer.save_prompt_cache(
        std::path::Path::new("unused"),
        descriptor.clone(),
        &[3],
        &eredu_core::cache::PromptCacheOptions::default(),
        &(),
    )
    .unwrap();
    peer.decode(&FakeTensor(vec![4]), &()).unwrap();
    assert_eq!(peer.report().unwrap().state_report(), &[1]);
    assert!(peer
        .load_prompt_cache_distributed(std::path::Path::new("unused"), &descriptor, &[3], &(),)
        .is_err());
    assert_eq!(peer.report().unwrap().state_report(), &[1]);
    assert_eq!(
        peer_calls.borrow().last(),
        Some(&(DistributedExecutionPhase::PromptCacheLoadPreparation, true))
    );

    let (mut successful, descriptor, _, _, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheSavePublication,
        None,
        true,
    );
    successful
        .save_prompt_cache(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[3],
            &eredu_core::cache::PromptCacheOptions::default(),
            &(),
        )
        .unwrap();
    successful.decode(&FakeTensor(vec![4]), &()).unwrap();
    assert_eq!(successful.report().unwrap().state_report(), &[1]);
    assert!(successful
        .load_prompt_cache_distributed(std::path::Path::new("unused"), &descriptor, &[3], &(),)
        .unwrap()
        .is_some());
    assert_eq!(successful.report().unwrap().state_report(), &[0]);
}

#[test]
fn distributed_prompt_cache_preflight_rejects_wrong_rank_and_input_before_io() {
    let (mut wrong_rank, descriptor, store, calls, counters) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreflight,
        None,
        true,
    );
    let wrong_topology =
        eredu_core::cache::PromptCacheTopology::new(Some((2, 1)), None, None, true).unwrap();
    let wrong_descriptor = descriptor.with_topology(wrong_topology).unwrap();
    assert!(
        wrong_rank
            .load_prompt_cache_distributed(
                std::path::Path::new("unused"),
                &wrong_descriptor,
                &[3],
                &(),
            )
            .is_err()
    );
    assert!(store.borrow().is_none());
    assert_eq!(
        calls.borrow().as_slice(),
        &[(DistributedExecutionPhase::PromptCacheLoadPreflight, false)]
    );
    assert_eq!(counters.snapshot().forward_calls, 0);

    let (mut wrong_input, descriptor, store, calls, _) = partitioned_cache_control_session(
        1,
        DistributedExecutionPhase::PromptCacheSavePreflight,
        None,
        true,
    );
    let input_identity = prepared_composite_input(99)
        .cache_identity(composite_content_fingerprint(99))
        .unwrap();
    assert!(wrong_input
        .save_prompt_cache_for_input_distributed(
            std::path::Path::new("unused"),
            descriptor.clone(),
            &[3],
            &eredu_core::cache::PromptCacheOptions::default(),
            &input_identity,
            &(),
        )
        .is_err());
    assert!(store.borrow().is_none());
    assert_eq!(
        calls.borrow().as_slice(),
        &[(DistributedExecutionPhase::PromptCacheSavePreflight, false)]
    );
    let agreement_count = calls.borrow().len();
    assert!(wrong_input
        .save_prompt_cache_distributed(
            std::path::Path::new("unused"),
            descriptor,
            &[3],
            &eredu_core::cache::PromptCacheOptions::default(),
            &(),
        )
        .is_err());
    assert_eq!(calls.borrow().len(), agreement_count);
    assert!(store.borrow().is_none());
}

#[test]
fn distributed_state_controls_prepare_on_all_ranks_and_fence_failed_retries() {
    let (mut checkpoint_failure, _, _, calls, counters) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::SessionCheckpoint,
        None,
        true,
    );
    assert!(checkpoint_failure
        .checkpoint_complete_distributed(&())
        .is_err());
    assert_eq!(
        calls.borrow().as_slice(),
        &[(DistributedExecutionPhase::SessionCheckpoint, true)]
    );
    assert!(checkpoint_failure
        .decode(&FakeTensor(vec![3]), &())
        .is_err());
    assert_eq!(counters.snapshot().forward_calls, 0);

    let (mut rollback_failure, _, _, calls, counters) = partitioned_cache_control_session(
        1,
        DistributedExecutionPhase::SessionRollbackPreparation,
        None,
        true,
    );
    let checkpoint = rollback_failure
        .checkpoint_complete_distributed(&())
        .unwrap();
    rollback_failure.decode(&FakeTensor(vec![3]), &()).unwrap();
    assert_eq!(rollback_failure.report().unwrap().state_report(), &[1]);
    let forwards = counters.snapshot().forward_calls;
    assert!(rollback_failure
        .rollback_complete_distributed(checkpoint, &())
        .is_err());
    assert_eq!(rollback_failure.report().unwrap().state_report(), &[1]);
    assert_eq!(
        calls.borrow().last(),
        Some(&(DistributedExecutionPhase::SessionRollbackPreparation, true))
    );
    assert!(rollback_failure.decode(&FakeTensor(vec![4]), &()).is_err());
    assert_eq!(counters.snapshot().forward_calls, forwards);

    let (mut reset_failure, _, _, calls, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::SessionResetPreparation,
        None,
        true,
    );
    reset_failure.decode(&FakeTensor(vec![3]), &()).unwrap();
    assert!(reset_failure.reset_distributed(&()).is_err());
    assert_eq!(reset_failure.report().unwrap().state_report(), &[1]);
    assert_eq!(
        calls.borrow().last(),
        Some(&(DistributedExecutionPhase::SessionResetPreparation, true))
    );

    let (mut successful, _, _, _, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PromptCacheLoadPreflight,
        None,
        true,
    );
    let complete = successful.checkpoint_complete_distributed(&()).unwrap();
    successful.decode(&FakeTensor(vec![3]), &()).unwrap();
    successful
        .rollback_complete_distributed(complete, &())
        .unwrap();
    assert_eq!(successful.report().unwrap().state_report(), &[0]);
    successful.decode(&FakeTensor(vec![4]), &()).unwrap();
    successful.reset_distributed(&()).unwrap();
    assert_eq!(successful.report().unwrap().state_report(), &[0]);

    let state = successful.checkpoint_distributed(&()).unwrap();
    successful.decode(&FakeTensor(vec![5]), &()).unwrap();
    successful.rollback_distributed(state, &()).unwrap();
    assert_eq!(successful.report().unwrap().state_report(), &[0]);
}

#[test]
fn partitioned_phase_agreement_propagates_execution_observation_and_completion_failures() {
    for (failed_phase, failing_site) in [
        (
            DistributedExecutionPhase::StateCheckpoint,
            LocalPartitionFailure::Checkpoint,
        ),
        (
            DistributedExecutionPhase::Execution,
            LocalPartitionFailure::Execution,
        ),
        (
            DistributedExecutionPhase::OutputObservation,
            LocalPartitionFailure::Observation,
        ),
        (
            DistributedExecutionPhase::OutputPublication,
            LocalPartitionFailure::Publication,
        ),
        (
            DistributedExecutionPhase::MechanismCompletion,
            LocalPartitionFailure::Completion,
        ),
    ] {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let commits = Rc::new(Cell::new(0));
        let failing = run_partitioned_agreement_rank(
            0,
            failing_site,
            failed_phase,
            Rc::clone(&calls),
            Rc::clone(&commits),
        );
        let peer = run_partitioned_agreement_rank(
            1,
            LocalPartitionFailure::None,
            failed_phase,
            Rc::clone(&calls),
            Rc::clone(&commits),
        );

        assert!(
            failing.failed && peer.failed,
            "both ranks must report phase failure"
        );
        assert_eq!(failing.state, [0]);
        assert_eq!(peer.state, [0]);
        let expected_completions =
            usize::from(matches!(failing_site, LocalPartitionFailure::Completion));
        assert_eq!(failing.completions, expected_completions);
        assert_eq!(peer.completions, expected_completions);
        if matches!(failing_site, LocalPartitionFailure::Checkpoint) {
            assert_eq!(
                failing.forwards, 0,
                "checkpoint failure must precede local execution"
            );
            assert_eq!(
                peer.forwards, 0,
                "checkpoint failure must suppress peer execution"
            );
        }
        assert_eq!(commits.get(), 0, "failed phases must not commit");
        let phase_calls = calls
            .borrow()
            .iter()
            .filter(|(phase, _)| *phase == failed_phase)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(phase_calls.len(), 2);
        assert!(phase_calls.iter().any(|(_, success)| !success));
        assert!(phase_calls.iter().any(|(_, success)| *success));
    }
}

#[test]
fn concurrent_checkpoint_failure_suppresses_execution_on_every_rank() {
    let coordinator = Arc::new(ConcurrentCheckpointCoordinator::new());
    let results = std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let coordinator = Arc::clone(&coordinator);
                scope.spawn(move || {
                    run_partitioned_agreement_rank_with(
                        rank,
                        if rank == 0 {
                            LocalPartitionFailure::Checkpoint
                        } else {
                            LocalPartitionFailure::None
                        },
                        TestPhaseAgreement::Concurrent(ConcurrentCheckpointAgreement {
                            rank,
                            coordinator,
                        }),
                    )
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert!(results.iter().all(|result| result.failed));
    assert!(results.iter().all(|result| result.state == [0]));
    assert!(results.iter().all(|result| result.completions == 0));
    assert!(results.iter().all(|result| result.output.is_none()));
    assert!(results.iter().all(|result| result.forwards == 0));
    let mut calls = coordinator.calls.lock().unwrap().clone();
    calls.sort_by_key(|(rank, _, _)| *rank);
    assert_eq!(
        calls,
        [
            (0, DistributedExecutionPhase::StateCheckpoint, false),
            (1, DistributedExecutionPhase::StateCheckpoint, true),
        ]
    );
}

#[test]
fn partitioned_final_output_observer_runs_only_on_publication_owner() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let commits = Rc::new(Cell::new(0));
    let result = run_partitioned_agreement_rank(
        1,
        LocalPartitionFailure::Observation,
        DistributedExecutionPhase::Commit,
        Rc::clone(&calls),
        Rc::clone(&commits),
    );

    assert!(
        !result.failed,
        "non-owner observer must not see final logits"
    );
    assert_eq!(result.state, [1]);
    assert_eq!(result.completions, 1);
    assert_eq!(commits.get(), 1);
    assert!(calls.borrow().iter().any(|(phase, success)| {
        *phase == DistributedExecutionPhase::OutputObservation && *success
    }));
}

#[test]
fn partitioned_owner_intervenes_before_publication_completion_and_commit() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let commits = Rc::new(Cell::new(0));
    let trace = Rc::new(RefCell::new(Vec::new()));
    let published = Rc::new(RefCell::new(Vec::new()));
    PARTITION_COMMIT_TRACE.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&trace)));
    PARTITION_PUBLICATION_VALUES.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&published)));
    let result = run_partitioned_agreement_rank(
        0,
        LocalPartitionFailure::Intervention,
        DistributedExecutionPhase::Commit,
        calls,
        Rc::clone(&commits),
    );
    PARTITION_COMMIT_TRACE.with(|slot| *slot.borrow_mut() = None);
    PARTITION_PUBLICATION_VALUES.with(|slot| *slot.borrow_mut() = None);

    assert!(!result.failed);
    assert_eq!(result.output, Some(FakeTensor(vec![99])));
    assert_eq!(*published.borrow(), [FakeTensor(vec![99])]);
    assert_eq!(
        *trace.borrow(),
        ["observe", "publish", "complete", "commit"]
    );
    assert_eq!(commits.get(), 1);
}

#[test]
fn partitioned_prediction_capture_agrees_before_publication_and_rolls_back_on_rejection() {
    let (mut session, _, _, calls, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::PredictionTargetCapture,
        None,
        true,
    );
    let published = Rc::new(RefCell::new(Vec::new()));
    PARTITION_PUBLICATION_VALUES.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&published)));
    let before = session.report().unwrap().state_report().to_vec();
    let error = session
        .prefill_prediction_target(&FakeTensor(vec![3]), None, &())
        .unwrap_err();
    PARTITION_PUBLICATION_VALUES.with(|slot| *slot.borrow_mut() = None);

    assert!(error.to_string().contains("another rank could not prepare"));
    assert_eq!(session.report().unwrap().state_report().to_vec(), before);
    assert!(
        published.borrow().is_empty(),
        "rejected capture published {:?}",
        *published.borrow()
    );
    assert!(calls.borrow().iter().any(|(phase, success)| {
        *phase == DistributedExecutionPhase::PredictionTargetCapture && *success
    }));
}

#[test]
fn distributed_commit_reports_asymmetric_observation_without_false_rollback() {
    for phase in [
        DistributedCommitPhase::DecisionSubmission,
        DistributedCommitPhase::DecisionCompletion,
        DistributedCommitPhase::DecisionObservation,
    ] {
        let committed_calls = Rc::new(Cell::new(0));
        let committed = run_partitioned_agreement_rank_with(
            0,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Final {
                result: FinalDecisionResult::Committed,
                commits: Rc::clone(&committed_calls),
            },
        );
        let uncertain_calls = Rc::new(Cell::new(0));
        let uncertain = run_partitioned_agreement_rank_with(
            1,
            LocalPartitionFailure::None,
            TestPhaseAgreement::Final {
                result: FinalDecisionResult::Indeterminate(phase),
                commits: Rc::clone(&uncertain_calls),
            },
        );

        assert!(!committed.failed);
        assert_eq!(committed.state, [1]);
        assert_eq!(committed.output, Some(FakeTensor(vec![5])));
        assert_eq!(
            committed.outcome,
            Some(DistributedCommitOutcome::Committed(
                DistributedCommitEpoch::FIRST
            ))
        );
        assert!(uncertain.failed);
        assert_eq!(
            uncertain.state,
            [1],
            "indeterminate state must not roll back"
        );
        assert!(
            uncertain.output.is_none(),
            "indeterminate output must not escape"
        );
        assert_eq!(
            uncertain.outcome,
            Some(DistributedCommitOutcome::Indeterminate {
                epoch: DistributedCommitEpoch::FIRST,
                phase,
            })
        );
        assert_eq!(
            uncertain.retry_forwards, 0,
            "poisoned retry must execute no unit"
        );
        assert_eq!(committed_calls.get(), 1);
        assert_eq!(uncertain_calls.get(), 1, "poisoned retry must not re-agree");
        assert_eq!(
            uncertain.cached_outcome, uncertain.outcome,
            "cache must retain the exact epoch"
        );
        assert!(!committed.outcome.unwrap().is_indeterminate());
        assert_ne!(
            committed.outcome,
            Some(DistributedCommitOutcome::Aborted(
                DistributedCommitEpoch::FIRST
            ))
        );
        assert_ne!(
            uncertain.outcome,
            Some(DistributedCommitOutcome::Aborted(
                DistributedCommitEpoch::FIRST
            ))
        );
    }
}

#[test]
fn globally_observed_final_abort_restores_state_and_withholds_output() {
    let commits = Rc::new(Cell::new(0));
    let aborted = run_partitioned_agreement_rank_with(
        0,
        LocalPartitionFailure::None,
        TestPhaseAgreement::Final {
            result: FinalDecisionResult::Aborted,
            commits: Rc::clone(&commits),
        },
    );
    assert!(aborted.failed);
    assert_eq!(aborted.state, [0]);
    assert!(aborted.output.is_none());
    assert_eq!(
        aborted.outcome,
        Some(DistributedCommitOutcome::Aborted(
            DistributedCommitEpoch::FIRST
        ))
    );
    assert_eq!(commits.get(), 1);
}

#[test]
fn partitioned_runtime_factory_rejects_independent_task_proof_before_factory_work() {
    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters,
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let factory_called = Rc::new(Cell::new(false));
    let called = Rc::clone(&factory_called);
    let error = prepare_partitioned_session_runtime::<
        _,
        FakeBackend,
        _,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
        Infallible,
        _,
    >(
        architecture,
        selected,
        partition,
        CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(test_completion_policy()),
        Some(&[]),
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        move |_input, _selected, _context| {
            called.set(true);
            Ok(((), DeviceState::stateless()))
        },
    )
    .err()
    .expect("independent task proof reached the partition runtime factory");
    assert!(error
        .to_string()
        .contains("precomputed local materialization tasks differ"));
    assert!(!factory_called.get());
}

#[test]
fn partitioned_runtime_rejects_publication_and_agreement_manifest_perturbations_before_calls() {
    type Runtime = PartitionedTextRuntime<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        RecordingPolicy,
        ReferencePartitionExecutor,
        (),
        (),
        FakeTensorMetadata,
        NoBoundaryTransport,
        ScriptedOutputPublisher,
        TestPhaseAgreement,
    >;

    let group = CommunicationGroupId::new(57);
    for (requirements, operation) in [
        (
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
            CommunicationOperation::Broadcast,
        ),
        (
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::Broadcast,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 3, 8, None).unwrap(),
                    true,
                )
                .unwrap(),
                CommunicationOperationRequirement::failure_agreement(false),
            ])
            .unwrap(),
            CommunicationOperation::FailureAgreement,
        ),
    ] {
        let counters = ReplicatedSessionCounters::default();
        let architecture = OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: Vec::new(),
            counters: counters.clone(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        };
        let graph = architecture.execution_graph().unwrap();
        let plan = PartitionedExecutionPlan::new(
            graph,
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(PartitionOutputPublication {
                group,
                owner_rank: 0,
            }),
            Some(group),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        let manifest = CommunicationManifest::new(
            2,
            1,
            vec![
                CommunicationGroupDescriptor::new(group, 0, vec![0, 1], Some(1), requirements)
                    .unwrap(),
            ],
            Vec::new(),
        )
        .unwrap()
        .with_completion_policy(test_completion_policy());
        let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
            manifest,
            vec![RealizedCommunicationGroup::new(group, ())],
            Vec::new(),
            FakeTensorMetadata,
        )
        .unwrap();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let commits = Rc::new(Cell::new(0));
        let error = Runtime::new(
            plan,
            ReferencePartitionExecutor {
                architecture,
                unit: FakeUnit { marker: 5 },
                fail_after_state: Rc::new(Cell::new(false)),
            },
            communication,
            (),
            NoBoundaryTransport,
            ScriptedOutputPublisher { fail: false },
            TestPhaseAgreement::Scripted(ScriptedPhaseAgreement {
                failed_phase: DistributedExecutionPhase::Commit,
                calls: Rc::clone(&calls),
                commits: Rc::clone(&commits),
            }),
            ExecutionResidency::FullyResident,
            None,
        )
        .err()
        .expect("perturbed manifest reached runtime execution");

        assert!(matches!(
            error,
            eredu_runtime::PartitionExecutionError::OperationNotSelected { operation: actual, .. }
                | eredu_runtime::PartitionExecutionError::InexactOperationRequirement { operation: actual, .. }
                if actual == operation
        ));
        assert_eq!(counters.snapshot().forward_calls, 0);
        assert!(calls.borrow().is_empty());
        assert_eq!(commits.get(), 0);
    }
}

#[test]
fn production_partitioned_middle_stage_completes_source_before_send_and_fences_deadline_retry() {
    type Strategy = PartitionedTextExecution<
        PipelineDestinationExecutor,
        (),
        (),
        PipelineTensorMetadata,
        OpaqueBoundaryTransport,
        NoOutputPublisher,
        NoCommitAgreement,
    >;

    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let boundary = NoAuxiliaryBoundarySchema::new(1)
        .wire_schema()
        .unwrap()
        .resolve(1, 1)
        .unwrap();
    let route_id = CommunicationRouteId::new(0);
    let route_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 3, 1, None).unwrap(),
        true,
    )
    .unwrap();
    let outgoing_route_id = CommunicationRouteId::new(1);
    let descriptor = test_boundary_route(route_id, 0, 0, 1, route_requirement.clone());
    let outgoing_descriptor = test_boundary_route(outgoing_route_id, 1, 1, 2, route_requirement);
    let trace = Rc::new(RefCell::new(Vec::new()));
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        architecture,
        selected,
        partition,
        CommunicationManifest::new(3, 1, Vec::new(), vec![descriptor, outgoing_descriptor])
            .unwrap()
            .with_completion_policy(test_completion_policy()),
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, partition, manifest, tasks) = input.into_parts();
            assert_eq!(tasks.len(), 1);
            let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![Some(driver)],
                vec![
                    PartitionBoundaryRoute {
                        source_group: 0,
                        destination_group: 0,
                        source_rank: 0,
                        destination_rank: 1,
                        route: route_id,
                    },
                    PartitionBoundaryRoute {
                        source_group: 0,
                        destination_group: 0,
                        source_rank: 1,
                        destination_rank: 2,
                        route: outgoing_route_id,
                    },
                ],
                None,
                None,
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                Vec::new(),
                vec![
                    RealizedCommunicationRoute::new(route_id, ()),
                    RealizedCommunicationRoute::new(outgoing_route_id, ()),
                ],
                PipelineTensorMetadata,
            )
            .unwrap();
            let state = DeviceState::create(partition.state().unwrap().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(0))
            })
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                PipelineDestinationExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    trace: Rc::clone(&trace),
                    boundary,
                    send_boundary: true,
                    fail_boundary_values: false,
                    swap_auxiliary_roles: false,
                },
                communication,
                (),
                OpaqueBoundaryTransport,
                NoOutputPublisher,
                NoCommitAgreement,
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((runtime, state))
        },
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters,
        fail_completion: Rc::new(Cell::new(false)),
        fail_checkpoint: false,
        prompt_cache: None,
    };
    let mut session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        Strategy,
    >(binding, mechanisms, Strategy::new())
    .unwrap();

    assert_eq!(
        session.decode(&FakeTensor(vec![3]), &()).unwrap(),
        FakeTensor(vec![5])
    );
    assert_eq!(*trace.borrow(), ["receive", "accept", "execute", "send"]);

    POINT_TO_POINT_COUNT.set(0);
    LOCAL_DEPENDENCY_SUBMISSION_COUNT.set(0);
    BOUNDED_COMMUNICATION_WAIT_COUNT.set(0);
    COMMUNICATION_COMPLETION_SCRIPT.with(|script| {
        script.borrow_mut().extend([
            FakeCompletionOutcome::Completed,
            FakeCompletionOutcome::DeadlineExceeded,
        ]);
    });
    let state_before_deadline = session.report().unwrap().state_report().to_vec();
    assert!(session.decode(&FakeTensor(vec![4]), &()).is_err());
    assert_eq!(
        session.report().unwrap().state_report().as_slice(),
        state_before_deadline.as_slice()
    );
    assert_eq!(LOCAL_DEPENDENCY_SUBMISSION_COUNT.get(), 1);
    assert_eq!(
        POINT_TO_POINT_COUNT.get(),
        1,
        "source dependency deadline must prevent the outgoing route submission",
    );
    assert_eq!(BOUNDED_COMMUNICATION_WAIT_COUNT.get(), 2);
    assert!(session.decode(&FakeTensor(vec![6]), &()).is_err());
    assert_eq!(LOCAL_DEPENDENCY_SUBMISSION_COUNT.get(), 1);
    assert_eq!(POINT_TO_POINT_COUNT.get(), 1);
}

#[derive(Clone, Copy)]
enum DestinationPreparationFailure {
    Placeholder,
    MalformedSchema,
    SwappedAuxiliaryRoles,
    CorruptReceivedHeader,
}

fn assert_destination_preparation_failure_is_agreed_before_transfer(
    failure: DestinationPreparationFailure,
) {
    type Strategy = PartitionedTextExecution<
        PipelineDestinationExecutor,
        (),
        (),
        PipelineTensorMetadata,
        OpaqueBoundaryTransport,
        NoOutputPublisher,
        ScriptedPhaseAgreement,
    >;

    let counters = ReplicatedSessionCounters::default();
    let architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected = selected_reference_text(&architecture, LayerWeightResidency::FullyResident);
    let parameters = architecture.parameter_description(&()).unwrap();
    let partition = ArchitecturePartition::from_architecture::<
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        _,
        _,
    >(
        &architecture,
        [("decoder", 0..1)],
        PartitionOwnership::new(false, true, std::iter::empty::<String>()).unwrap(),
        (),
        NoAuxiliaryBoundarySchema::new(1),
        &parameters,
    )
    .unwrap();
    let boundary = match failure {
        DestinationPreparationFailure::Placeholder => {
            NoAuxiliaryBoundarySchema::new(1).wire_schema().unwrap()
        }
        DestinationPreparationFailure::MalformedSchema => {
            NoAuxiliaryBoundarySchema::new(2).wire_schema().unwrap()
        }
        DestinationPreparationFailure::CorruptReceivedHeader => {
            NoAuxiliaryBoundarySchema::new(1).wire_schema().unwrap()
        }
        DestinationPreparationFailure::SwappedAuxiliaryRoles => {
            eredu_runtime::BoundaryWireSchema::new(
                "swapped.fixture",
                eredu_runtime::BoundaryTensorSpec::primary_activation(1),
                [
                    eredu_runtime::BoundaryTensorSpec::new(
                        "first",
                        [
                            eredu_runtime::BoundaryTensorDimension::Batch,
                            eredu_runtime::BoundaryTensorDimension::Sequence,
                            eredu_runtime::BoundaryTensorDimension::Fixed(1),
                        ],
                        eredu_runtime::BoundaryTensorDtype::Activation,
                    ),
                    eredu_runtime::BoundaryTensorSpec::new(
                        "second",
                        [
                            eredu_runtime::BoundaryTensorDimension::Batch,
                            eredu_runtime::BoundaryTensorDimension::Sequence,
                            eredu_runtime::BoundaryTensorDimension::Fixed(1),
                        ],
                        eredu_runtime::BoundaryTensorDtype::Activation,
                    ),
                ],
            )
            .unwrap()
        }
    }
    .resolve(1, 1)
    .unwrap();
    let route_id = CommunicationRouteId::new(0);
    let route_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 3, 1, None).unwrap(),
        true,
    )
    .unwrap();
    let group = CommunicationGroupId::new(1);
    let manifest = CommunicationManifest::new(
        2,
        1,
        vec![CommunicationGroupDescriptor::new(
            group,
            0,
            vec![0, 1],
            Some(1),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        vec![test_boundary_route(route_id, 0, 0, 1, route_requirement)],
    )
    .unwrap()
    .with_completion_policy(test_completion_policy());
    let trace = Rc::new(RefCell::new(Vec::new()));
    let calls = Rc::new(RefCell::new(Vec::new()));
    let commits = Rc::new(Cell::new(0));
    let binding = prepare_partitioned_session_runtime::<_, FakeBackend, _, _, _, _, Infallible, _>(
        architecture,
        selected,
        partition,
        manifest,
        None,
        eredu_core::cache::PromptCacheTopology::default(),
        ReplicatedTextOutputSelection::LastSequencePosition,
        &(),
        |input, _selected, _context| {
            let (architecture, partition, manifest, _) = input.into_parts();
            let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
            let plan = PartitionedExecutionPlan::new(
                architecture.execution_graph().unwrap(),
                vec![(ArchitectureGroupKind::Decoder, false)],
                vec![Some(driver)],
                vec![PartitionBoundaryRoute {
                    source_group: 0,
                    destination_group: 0,
                    source_rank: 0,
                    destination_rank: 1,
                    route: route_id,
                }],
                None,
                Some(group),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
                manifest,
                vec![RealizedCommunicationGroup::new(group, ())],
                vec![RealizedCommunicationRoute::new(route_id, ())],
                PipelineTensorMetadata,
            )
            .unwrap();
            let state = DeviceState::create(partition.state().unwrap().layout().clone(), |_, _| {
                Ok::<_, Infallible>(FakeLayerState(0))
            })
            .unwrap();
            let runtime = PartitionedTextRuntime::new(
                plan,
                PipelineDestinationExecutor {
                    architecture,
                    unit: FakeUnit { marker: 5 },
                    trace: Rc::clone(&trace),
                    boundary,
                    send_boundary: false,
                    fail_boundary_values: matches!(
                        failure,
                        DestinationPreparationFailure::Placeholder
                    ),
                    swap_auxiliary_roles: matches!(
                        failure,
                        DestinationPreparationFailure::SwappedAuxiliaryRoles
                    ),
                },
                communication,
                (),
                OpaqueBoundaryTransport,
                NoOutputPublisher,
                ScriptedPhaseAgreement {
                    failed_phase: DistributedExecutionPhase::Commit,
                    calls: Rc::clone(&calls),
                    commits: Rc::clone(&commits),
                },
                ExecutionResidency::FullyResident,
                None::<RecordingPolicy>,
            )
            .unwrap();
            Ok((runtime, state))
        },
    )
    .unwrap();
    let mechanisms = ReferenceTextMechanisms {
        tasks: Rc::new(RefCell::new(Vec::new())),
        completions: Rc::new(RefCell::new(Vec::new())),
        counters,
        fail_completion: Rc::new(Cell::new(false)),
        fail_checkpoint: false,
        prompt_cache: None,
    };
    let mut session = construct_replicated_text_session_with_runtime::<
        OrdinaryTextFixture,
        FakeBackend,
        _,
        Strategy,
    >(binding, mechanisms, Strategy::new())
    .unwrap();

    POINT_TO_POINT_COUNT.set(0);
    if matches!(
        failure,
        DestinationPreparationFailure::CorruptReceivedHeader
    ) {
        COMMUNICATION_COMPLETION_SCRIPT.with(|script| {
            script
                .borrow_mut()
                .push_back(FakeCompletionOutcome::CorruptBoundaryHeader);
        });
    }
    assert!(session.decode(&FakeTensor(vec![3]), &()).is_err());
    assert_eq!(trace.borrow().as_slice(), ["receive"]);
    assert_eq!(
        POINT_TO_POINT_COUNT.get(),
        usize::from(matches!(
            failure,
            DestinationPreparationFailure::CorruptReceivedHeader
        ))
    );
    assert_eq!(session.report().unwrap().state_report(), &[0]);
    assert_eq!(commits.get(), 0);
    let boundary_ready = matches!(
        failure,
        DestinationPreparationFailure::CorruptReceivedHeader
    );
    let mut expected_phases = vec![
        (DistributedExecutionPhase::StateCheckpoint, true),
        (
            DistributedExecutionPhase::BoundarySourceCompletion(route_id),
            boundary_ready,
        ),
    ];
    if boundary_ready {
        expected_phases.push((
            DistributedExecutionPhase::BoundarySourceReady(route_id),
            true,
        ));
    }
    expected_phases.push((DistributedExecutionPhase::Execution, false));
    assert_eq!(calls.borrow().as_slice(), expected_phases);
    if matches!(
        failure,
        DestinationPreparationFailure::CorruptReceivedHeader
    ) {
        let calls_before_retry = POINT_TO_POINT_COUNT.get();
        assert!(session.decode(&FakeTensor(vec![4]), &()).is_err());
        assert_eq!(POINT_TO_POINT_COUNT.get(), calls_before_retry);
    }
}

#[test]
fn destination_placeholder_failure_is_agreed_before_native_boundary_transfer() {
    assert_destination_preparation_failure_is_agreed_before_transfer(
        DestinationPreparationFailure::Placeholder,
    );
}

#[test]
fn malformed_boundary_schema_is_agreed_before_native_boundary_transfer() {
    assert_destination_preparation_failure_is_agreed_before_transfer(
        DestinationPreparationFailure::MalformedSchema,
    );
}

#[test]
fn same_shape_auxiliary_role_swap_is_agreed_before_submission_and_rolled_back() {
    assert_destination_preparation_failure_is_agreed_before_transfer(
        DestinationPreparationFailure::SwappedAuxiliaryRoles,
    );
}

#[test]
fn corrupt_received_role_header_is_agreed_by_all_ranks_and_rolled_back() {
    assert_destination_preparation_failure_is_agreed_before_transfer(
        DestinationPreparationFailure::CorruptReceivedHeader,
    );
}

#[test]
fn replicated_text_contract_mismatch_fails_before_backend_work() {
    let counters = ReplicatedSessionCounters::default();
    let selected_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected =
        selected_reference_text(&selected_architecture, LayerWeightResidency::FullyResident);
    let expected_identity = selected.requirements().architecture_identity().to_owned();
    let constructed_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: true,
        inconsistent_identity: false,
    };
    let error = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(
        &constructed_architecture,
        None,
        selected,
        &expected_identity,
        &(),
    )
    .err()
    .expect("inconsistent architecture entered backend construction");
    assert!(error.contains("architecture execution group 0 differs from selection"));
    assert_eq!(counters.snapshot(), ReplicatedSessionCounts::default());
}

#[test]
fn replicated_text_identity_mismatch_fails_before_backend_work() {
    let counters = ReplicatedSessionCounters::default();
    let selected_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: false,
    };
    let selected =
        selected_reference_text(&selected_architecture, LayerWeightResidency::FullyResident);
    let expected_identity = selected.requirements().architecture_identity().to_owned();
    let constructed_architecture = OrdinaryTextFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
        counters: counters.clone(),
        inconsistent_transport: false,
        inconsistent_identity: true,
    };
    let error = prepare_replicated_text_contract::<
        _,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    >(
        &constructed_architecture,
        None,
        selected,
        &expected_identity,
        &(),
    )
    .err()
    .expect("inconsistent identity entered backend construction");
    assert!(error.contains("prompt-cache identity differs from selection"));
    assert_eq!(counters.snapshot(), ReplicatedSessionCounts::default());
}

#[test]
fn replicated_text_selection_denies_each_required_mechanism_before_backend_work() {
    for (denied, expected) in [
        (
            DeniedReferenceMechanism::Storage,
            "weight residency Resident",
        ),
        (DeniedReferenceMechanism::State, "state component"),
        (DeniedReferenceMechanism::Observation, "observation"),
        (
            DeniedReferenceMechanism::Persistence,
            "prompt-cache persistence",
        ),
        (
            DeniedReferenceMechanism::Completion,
            "exact completion ownership",
        ),
    ] {
        let counters = ReplicatedSessionCounters::default();
        let architecture = OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: Vec::new(),
            counters: counters.clone(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        };
        let error = try_select_reference_text(
            &architecture,
            LayerWeightResidency::FullyResident,
            Some(denied),
        )
        .expect_err("missing mechanism was selected");
        assert!(
            error.issues().iter().any(|issue| issue.contains(expected)),
            "{denied:?}: {error}"
        );
        assert_eq!(counters.snapshot(), ReplicatedSessionCounts::default());
    }
}

impl Parameterized<FakeTensor> for FakeUnit {
    fn visit_parameters<'a, V>(&'a self, _visitor: &mut V)
    where
        V: ParameterVisitor<'a, FakeTensor>,
    {
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, _visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, FakeTensor>,
    {
    }

    fn set_trainable(&mut self, _trainable: bool) {}
}

struct GroupedFixture {
    static_modules: FakeOperator,
    trace: Vec<(usize, usize)>,
}

thread_local! {
    static COMPOSITE_FORWARD_RESOURCE: RefCell<Option<Rc<Cell<bool>>>> = const { RefCell::new(None) };
}

#[derive(Debug, Default)]
struct GroupedForwardContext {
    trace: Vec<(usize, usize)>,
    dropped: Option<Rc<Cell<bool>>>,
}

impl PartialEq for GroupedForwardContext {
    fn eq(&self, other: &Self) -> bool {
        self.trace == other.trace
    }
}

impl Eq for GroupedForwardContext {}

impl GroupedForwardContext {
    fn new(trace: Vec<(usize, usize)>) -> Self {
        let dropped = COMPOSITE_FORWARD_RESOURCE.with(|resource| resource.borrow().clone());
        if let Some(dropped) = &dropped {
            dropped.set(false);
        }
        Self { trace, dropped }
    }

    fn push(&mut self, value: (usize, usize)) {
        self.trace.push(value);
    }
}

impl PartialEq<Vec<(usize, usize)>> for GroupedForwardContext {
    fn eq(&self, other: &Vec<(usize, usize)>) -> bool {
        &self.trace == other
    }
}

impl Drop for GroupedForwardContext {
    fn drop(&mut self) {
        if let Some(dropped) = &self.dropped {
            dropped.set(true);
        }
    }
}

impl ArchitectureParameters<FakeBackend> for GroupedFixture {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        let policies = (0..4)
            .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
            .collect();
        StateLayout::new(LayerSchedule::new(4, policies).unwrap()).map_err(Error::backend)
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        eredu_runtime::ModelStateIdentity::new(
            "fixture",
            "fixture",
            "fixture",
            4,
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }

    fn parameter_description(
        &self,
        _: &(),
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["vision", "audio"]),
            ],
            "text",
        )
        .map_err(Error::backend)?;
        let layout = ExecutionUnitLayout::new(&graph, [1, 1, 2]).map_err(Error::backend)?;
        ArchitectureParameterDescription::new(&graph, &layout, [], []).map_err(Error::backend)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<FakeBackend>,
    {
        visitor.visit("embedding", &self.static_modules)
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<FakeBackend>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules)
    }
}

impl LayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>> for GroupedFixture {
    type Input<'a> = Option<&'a PreparedModelInput<FakeTensor>>;
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = GroupedForwardContext;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        match group {
            0 => ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::VisionEncoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: None,
                request_optional: false,
            },
            1 => ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::AudioEncoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: None,
                request_optional: false,
            },
            _ => ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            },
        }
    }

    fn primary_execution_group(&self) -> &str {
        "text"
    }

    fn state_partition_plan(
        &self,
        _: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(2, 0..2),
            eredu_runtime::ArchitectureStatePartitionRule::output_owner(2..4),
        ])
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text", ["vision", "audio"]),
            ],
            "text",
        )
        .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        [1, 1, 2]
            .get(group)
            .copied()
            .ok_or_else(|| Error::backend(format!("unknown fixture group {group}")))
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        Ok(format!("group.{group}.unit.{index}"))
    }

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 2).then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok(match group {
            0 => Some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()),
            1 => Some(eredu_core::AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()),
            _ => None,
        })
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, group: usize, index: usize, _: &()) -> Result<Self::Unit, Self::Error> {
        Ok(FakeUnit {
            marker: i32::try_from(group * 10 + index).unwrap(),
        })
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        let hidden = match input {
            None => FakeTensor(vec![0]),
            Some(input) => {
                let expected = [
                    InputModality::Text,
                    InputModality::Image,
                    InputModality::Audio,
                ];
                if input.len() != expected.len()
                    || input
                        .parts()
                        .iter()
                        .zip(expected)
                        .any(|(part, modality)| part.modality() != modality)
                {
                    return Err(Error::backend(
                        "composite fixture requires ordered text, image, and audio parts",
                    ));
                }
                FakeTensor(
                    input
                        .parts()
                        .iter()
                        .map(|part| {
                            part.payload()
                                .value()
                                .0
                                .first()
                                .copied()
                                .ok_or_else(|| Error::backend("composite payload is empty"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: GroupedForwardContext::new(Vec::new()),
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &FakeTensor,
        dependencies: &[&FakeTensor],
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        if initial.0.len() == 3 {
            return match group {
                0 if dependencies.is_empty() => Ok(FakeTensor(vec![initial.0[1]])),
                1 if dependencies.is_empty() => Ok(FakeTensor(vec![initial.0[2]])),
                2 if dependencies.len() == 2 => Ok(FakeTensor(vec![
                    initial.0[0] + dependencies[0].0[0] + dependencies[1].0[0],
                ])),
                _ => Err(Error::backend("invalid composite dependency inputs")),
            };
        }
        match group {
            0 if dependencies.is_empty() => Ok(FakeTensor(vec![10])),
            1 if dependencies.is_empty() => Ok(FakeTensor(vec![20])),
            2 if dependencies == [&FakeTensor(vec![10, 0]), &FakeTensor(vec![20, 10])] => {
                Ok(FakeTensor(vec![30]))
            }
            _ => Err(Error::backend("invalid fixture dependency inputs")),
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.trace.push((group, index));
        if group == 2 && state.as_ref().len() >= 4 {
            state.as_mut()[2 + index].0 += 1;
        }
        let mut output = hidden.clone();
        output.0.push(unit.marker);
        Ok(output)
    }

    fn finish_forward(
        &mut self,
        hidden: &FakeTensor,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
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

impl ParallelLayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for GroupedFixture
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward(input, state, context)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &mut Self::ForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &FakeTensor,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &Self::ForwardContext,
        _: &(),
        context: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.finish_forward(hidden, state, forward, context)
    }
}

impl PartitionedLayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>>
    for GroupedFixture
{
    type Boundary = NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(NoAuxiliaryBoundarySchema::new(8))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, FakeTensor, NoAuxiliaryBoundary>,
        _: Option<&FakeTensor>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        expected: &StateLayout,
        first_state_ordinal: usize,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        assert!(!expected.is_empty());
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => tokens.clone(),
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        Ok(LayeredForwardState {
            hidden,
            context: GroupedForwardContext::new(vec![(usize::MAX, first_state_ordinal)]),
        })
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, FakeTensor, NoAuxiliaryBoundary>,
        mask: Option<&FakeTensor>,
        state: &mut DeviceState<FakeBackend, FakeLayerState>,
        expected: &StateLayout,
        first_state_ordinal: usize,
        _: &(),
        context: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
    }

    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &FakeTensor,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        forward: &mut Self::ForwardContext,
        _: Option<&()>,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        forward.push((group, usize::MAX));
        Ok(initial.clone())
    }

    fn finish_partition(
        &mut self,
        hidden: &FakeTensor,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &Self::ForwardContext,
        owns_output: bool,
        _: Option<&()>,
        _: &(),
    ) -> Result<LayeredPartitionOutput<FakeTensor, NoAuxiliaryBoundary>, Self::Error> {
        Ok(if owns_output {
            LayeredPartitionOutput::Final {
                output: hidden.clone(),
                retained: None,
            }
        } else {
            LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: NoAuxiliaryBoundary,
            }
        })
    }
}

#[test]
fn partition_extension_uses_stable_groups_ownership_and_boundary_schema() {
    type FixtureState = DeviceState<FakeBackend, FakeLayerState>;

    let mut architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let parameters = architecture
        .parameter_description(&())
        .expect("fixture parameter description");
    let partitions = [
        ArchitecturePartition::<(), NoAuxiliaryBoundarySchema>::from_architecture::<
            FakeBackend,
            FixtureState,
            _,
            _,
        >(
            &architecture,
            [("text", 0..1)],
            PartitionOwnership::new(true, false, ["embedding"]).unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(8),
            &parameters,
        ),
        ArchitecturePartition::<(), NoAuxiliaryBoundarySchema>::from_architecture::<
            FakeBackend,
            FixtureState,
            _,
            _,
        >(
            &architecture,
            [("text", 1..2)],
            PartitionOwnership::new(false, true, std::iter::empty::<String>()).unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(8),
            &parameters,
        ),
    ]
    .map(|partition| partition.expect("partition derives its canonical graph"));
    for partition in &partitions {
        partition
            .validate_architecture::<FakeBackend, FixtureState, _>(&architecture)
            .expect("derived partition revalidates");
    }
    let drivers = [
        LayeredPartitionDriver::new(&partitions[0], 2, 0..1).unwrap(),
        LayeredPartitionDriver::new(&partitions[1], 2, 1..2).unwrap(),
    ];
    assert_eq!(drivers[0].state_layout().len(), 1);
    assert_eq!(drivers[1].state_layout().len(), 3);
    let schema = eredu_runtime::ArchitectureBoundary::wire_schema(partitions[0].boundary_schema())
        .unwrap()
        .resolve(1, 1)
        .unwrap();
    assert_eq!(schema.primary().shape(), [1, 1, 8]);
    assert_eq!(
        schema.primary().dtype(),
        eredu_runtime::BoundaryTensorDtype::Activation
    );

    let mut states = drivers.each_ref().map(|driver| {
        DeviceState::<FakeBackend, FakeLayerState>::create(
            driver.state_layout().clone(),
            |layer, _| Ok::<_, Infallible>(FakeLayerState(layer as i32)),
        )
        .unwrap()
    });
    let token = FakeTensor(vec![7]);
    let first_input = drivers[0]
        .input(LayeredPartitionInput::<FakeTensor, NoAuxiliaryBoundary>::Tokens(&token))
        .unwrap();
    let mut first = drivers[0]
        .begin::<FakeBackend, _, _>(
            &mut architecture,
            first_input,
            None,
            &mut states[0],
            None,
            &(),
        )
        .unwrap();
    for index in drivers[0].range() {
        let mut unit = architecture.build_unit(2, index, &()).unwrap();
        first.hidden = architecture
            .forward_unit(
                2,
                index,
                &mut unit,
                &first.hidden,
                &mut states[0],
                &mut first.context,
                &(),
            )
            .unwrap();
    }
    let LayeredPartitionOutput::Boundary {
        hidden: first_boundary,
        auxiliary,
    } = drivers[0]
        .finish::<FakeBackend, _, _>(
            &mut architecture,
            &first.hidden,
            &mut states[0],
            &mut first.context,
            None,
            &(),
        )
        .unwrap()
    else {
        panic!("non-output partition must produce a boundary")
    };
    let trace = Rc::new(RefCell::new(Vec::new()));
    let groups = [0, 1].map(|local_rank| PartitionCollectiveGroup {
        members: vec![0, 1],
        local_rank,
        trace: Rc::clone(&trace),
    });
    let first_exchange = drivers[0]
        .exchange_boundary::<PartitionCollectiveBackend>(first_boundary, &groups[0], &())
        .unwrap();
    let second_input = drivers[1]
        .input(LayeredPartitionInput::Hidden {
            hidden: first_exchange,
            auxiliary,
        })
        .unwrap();
    let mut second = drivers[1]
        .begin::<FakeBackend, _, _>(
            &mut architecture,
            second_input,
            None,
            &mut states[1],
            None,
            &(),
        )
        .unwrap();
    assert_eq!(
        second.context,
        vec![(usize::MAX, 0), (2, usize::MAX)],
        "a nonzero global unit range must address its rank-local state from ordinal zero"
    );
    for index in drivers[1].range() {
        let mut unit = architecture.build_unit(2, index, &()).unwrap();
        second.hidden = architecture
            .forward_unit(
                2,
                index,
                &mut unit,
                &second.hidden,
                &mut states[1],
                &mut second.context,
                &(),
            )
            .unwrap();
    }
    let LayeredPartitionOutput::Final { output: merged, .. } = drivers[1]
        .finish::<FakeBackend, _, _>(
            &mut architecture,
            &second.hidden,
            &mut states[1],
            &mut second.context,
            None,
            &(),
        )
        .unwrap()
    else {
        panic!("output partition must produce logits")
    };

    assert_eq!(merged, FakeTensor(vec![7, 20, 21]));
    assert_eq!(
        *trace.borrow(),
        [PartitionCollectiveCall {
            local_rank: 0,
            members: vec![0, 1],
            value: vec![7, 20],
        },]
    );
    assert_eq!(architecture.trace, [(2, 0), (2, 1)]);
    assert!(partitions[0].ownership().owns_input());
    assert!(partitions[1].ownership().owns_output());
}

#[test]
fn complete_state_partition_derives_identity_through_architecture_contract() {
    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let state = eredu_runtime::PartitionState::new(
        architecture.state_layout().expect("fixture state layout"),
        0,
    )
    .expect("complete replicated state partition");
    let topology = eredu_core::cache::PromptCacheTopology::default();

    let identity = state
        .prompt_cache_identity::<FakeBackend, _>(&architecture, topology.clone())
        .expect("architecture derives prompt-cache identity");

    assert_eq!(identity.architecture_fingerprint(), "fixture");
    assert_eq!(identity.global_layer_start(), 0);
    assert_eq!(identity.layer_count(), 4);
    assert_eq!(identity.topology(), &topology);
}

#[test]
fn neutral_layerwise_runtime_executes_dependency_groups_in_stable_order() {
    FORK_COUNT.set(0);
    SUBMIT_COUNT.set(0);
    ORDER_COUNT.set(0);
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::new(LayerSchedule::new(4, policies).unwrap()).unwrap();
    let mut state = DeviceState::<FakeBackend, FakeLayerState>::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState(0))
    })
    .unwrap();
    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let units = [0, 10, 20, 21]
        .into_iter()
        .map(|marker| FakeUnit { marker })
        .collect();
    let mut runtime =
        LayerwiseRuntime::<_, FakeBackend, _, _>::new(architecture, RecordingPolicy::new(units));

    let (output, context_trace) = runtime
        .forward_with_context_hook(None, &mut state, &(), |group, index, context| {
            context.push((group, index));
            Ok(())
        })
        .unwrap();

    assert_eq!(output, FakeTensor(vec![30, 20, 21]));
    assert_eq!(
        runtime.architecture().trace,
        vec![(0, 0), (1, 0), (2, 0), (2, 1)]
    );
    assert_eq!(context_trace, runtime.architecture().trace);
    assert_eq!(
        runtime.policy().addresses,
        vec![(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 2, 1)]
    );
    assert_eq!(FORK_COUNT.get(), 1);
    assert_eq!(SUBMIT_COUNT.get(), 4);
    assert_eq!(ORDER_COUNT.get(), 5);

    let repeated = runtime.forward(None, &mut state, &()).unwrap();
    assert_eq!(repeated, FakeTensor(vec![30, 20, 21]));
    assert_eq!(
        FORK_COUNT.get(),
        1,
        "one runtime must retain its graph executors across decode steps"
    );
    assert_eq!(SUBMIT_COUNT.get(), 8);
    assert_eq!(ORDER_COUNT.get(), 10);
}

fn prepared_composite_input(image: i32) -> PreparedModelInput<FakeTensor> {
    PreparedModelInput::new(
        vec![
            PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(FakeTensor(vec![7])),
                [],
            )
            .unwrap(),
            PreparedInputPart::new(
                InputModality::Image,
                PreparedInputPayload::Tensor(FakeTensor(vec![image])),
                [],
            )
            .unwrap(),
            PreparedInputPart::new(
                InputModality::Audio,
                PreparedInputPayload::Tensor(FakeTensor(vec![5])),
                [],
            )
            .unwrap(),
        ],
        |value| InputTensorIdentity::new(TensorDtype::I32, vec![value.0.len()]),
    )
    .unwrap()
}

fn composite_content_fingerprint(image: u8) -> String {
    eredu_core::MultimodalRequest::new(vec![
        eredu_core::MultimodalSegment::TokenIds(vec![7]),
        eredu_core::MultimodalSegment::Media(eredu_core::Media::Image(
            eredu_core::RgbImage::new(vec![image; 3], 1, 1).unwrap(),
        )),
        eredu_core::MultimodalSegment::Media(eredu_core::Media::Audio(
            eredu_core::Audio::new(vec![0.5], 16_000).unwrap(),
        )),
    ])
    .unwrap()
    .tokenize::<std::convert::Infallible>(|_| unreachable!())
    .unwrap()
    .semantic_content_fingerprint()
}

fn selected_reference_composite(
    architecture: &GroupedFixture,
    residency: LayerWeightResidency,
) -> eredu_runtime::SelectedReplicatedTextRealization {
    let graph = architecture.execution_graph().unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1, 1, 2]).unwrap();
    let state_layout = architecture.state_layout().unwrap();
    let requirements = ReplicatedTextRequirements::new(
        "composite-fixture",
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        (0..3)
            .map(|group| architecture.group_transport(group))
            .collect::<Vec<_>>(),
        state_layout.clone(),
        ReplicatedTextStateAccess::KeyValue,
        Vec::new(),
    )
    .unwrap();
    let state = StateMechanismCapabilities::new(
        state_layout
            .layers()
            .iter()
            .enumerate()
            .flat_map(|(layer, policy)| {
                policy.components().into_iter().map(move |component| {
                    StateComponentMechanism::new(
                        layer,
                        component,
                        Some(StateComponentPlacement::Device),
                        None,
                    )
                })
            })
            .collect::<Vec<_>>(),
    )
    .with_transactions(true, true)
    .with_reset(true)
    .with_prompt_cache(true);
    let capabilities = BackendMechanismCapabilities::new(
        NeuralOperatorCapabilities::NONE,
        Vec::new(),
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
        ],
        state,
    )
    .with_prompt_cache(true)
    .with_exact_completion(true);
    let request = ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
        .with_prompt_cache(true)
        .with_exact_completion(true);
    select_replicated_text_realization(&requirements, &request, &capabilities).unwrap()
}

struct CompositeCausalObserver {
    values: Vec<(String, FakeTensor)>,
    replace_vision: bool,
    fail_path: Option<&'static str>,
}

impl eredu_runtime::ActivationObserver<FakeTensor, Error> for CompositeCausalObserver {
    fn observe(&mut self, path: &str, value: &FakeTensor) -> Result<(), Error> {
        self.values.push((path.to_owned(), value.clone()));
        if self.fail_path == Some(path) {
            return Err(Error::backend("composite observer failure"));
        }
        Ok(())
    }

    fn intervene(&mut self, path: &str, _: &FakeTensor) -> Result<Option<FakeTensor>, Error> {
        Ok(
            (self.replace_vision && path == eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
                .then(|| FakeTensor(vec![30, 0])),
        )
    }
}

impl CompositeCausalObserver {
    fn value(&self, path: &str) -> &FakeTensor {
        &self
            .values
            .iter()
            .find(|(candidate, _)| candidate == path)
            .unwrap_or_else(|| panic!("missing composite observation {path}"))
            .1
    }
}

#[test]
fn production_composite_input_reuses_the_shared_session_lifecycle() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let architecture = GroupedFixture {
            static_modules: FakeOperator,
            trace: Vec::new(),
        };
        let selected = selected_reference_composite(&architecture, residency);
        let contract = prepare_layered_text_contract::<
            _,
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >(
            &architecture,
            None,
            selected,
            "fixture",
            eredu_runtime::ReplicatedTextOutputSelection::LastSequencePosition,
            &(),
        )
        .unwrap();
        let prompt_model_identity = contract.prompt_cache_identity().clone();
        let counters = ReplicatedSessionCounters::default();
        let completions = Rc::new(RefCell::new(Vec::new()));
        let fail_completion = Rc::new(Cell::new(false));
        let prompt_cache = Rc::new(RefCell::new(None));
        let mechanisms = ReferenceTextMechanisms {
            tasks: Rc::new(RefCell::new(Vec::new())),
            completions: Rc::clone(&completions),
            counters: counters.clone(),
            fail_completion: Rc::clone(&fail_completion),
            fail_checkpoint: false,
            prompt_cache: Some(Rc::clone(&prompt_cache)),
        };
        let mut session = construct_replicated_text_session::<_, FakeBackend, _>(
            architecture,
            None,
            contract,
            mechanisms,
            &(),
        )
        .unwrap();

        let resource_dropped = Rc::new(Cell::new(true));
        COMPOSITE_FORWARD_RESOURCE
            .with(|resource| *resource.borrow_mut() = Some(Rc::clone(&resource_dropped)));
        let prepared = prepared_composite_input(3);
        let input_identity = prepared
            .cache_identity(composite_content_fingerprint(3))
            .unwrap();
        assert_eq!(
            session
                .prefill_input_with_cache_identity(Some(&prepared), input_identity.clone(), &(),)
                .unwrap(),
            FakeTensor(vec![21])
        );
        assert!(resource_dropped.get());
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 1, 1]);
        assert_eq!(
            session.committed_prompt_input_identity(),
            Some(&input_identity)
        );

        let checkpoint = session.checkpoint_complete(&()).unwrap();
        let decode = prepared_composite_input(4);
        let mut baseline_observer = CompositeCausalObserver {
            values: Vec::new(),
            replace_vision: false,
            fail_path: None,
        };
        session
            .decode_input_with_observer(Some(&decode), &(), &mut baseline_observer)
            .unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 2, 2]);
        session.rollback_complete(checkpoint, &()).unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 1, 1]);
        assert_eq!(
            session.committed_prompt_input_identity(),
            Some(&input_identity)
        );

        let checkpoint = session.checkpoint_complete(&()).unwrap();
        let mut causal_observer = CompositeCausalObserver {
            values: Vec::new(),
            replace_vision: true,
            fail_path: None,
        };
        session
            .decode_input_with_observer(Some(&decode), &(), &mut causal_observer)
            .unwrap();
        assert_eq!(
            baseline_observer.value(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH),
            &FakeTensor(vec![16])
        );
        assert_eq!(
            causal_observer.value(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH),
            &FakeTensor(vec![42])
        );
        assert_eq!(
            causal_observer.value("group.2.unit.0.input"),
            &FakeTensor(vec![42])
        );
        session.rollback_complete(checkpoint, &()).unwrap();

        let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
            prompt_model_identity,
            "fixture-checkpoint",
            input_identity.prefix_content_fingerprint(),
            1,
        )
        .unwrap();
        let different_media = prepared_composite_input(99)
            .cache_identity(composite_content_fingerprint(99))
            .unwrap();
        let before_identity_mismatch = session.report().unwrap().state_report().clone();
        let mismatch = session
            .save_prompt_cache_for_input(
                std::path::Path::new("unused"),
                descriptor.clone(),
                &[7],
                &eredu_core::cache::PromptCacheOptions::default(),
                &different_media,
                &(),
            )
            .unwrap_err();
        assert!(mismatch
            .to_string()
            .contains("content identity differs from the prepared input"));
        assert_eq!(
            session.report().unwrap().state_report(),
            &before_identity_mismatch
        );

        session
            .save_prompt_cache_for_input(
                std::path::Path::new("unused"),
                descriptor.clone(),
                &[7],
                &eredu_core::cache::PromptCacheOptions::default(),
                &input_identity,
                &(),
            )
            .unwrap();
        session.reset(&()).unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 0, 0]);
        session
            .load_prompt_cache_for_input(
                std::path::Path::new("unused"),
                &descriptor,
                &[7],
                input_identity.clone(),
                &(),
            )
            .unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 1, 1]);
        assert_eq!(
            session.committed_prompt_input_identity(),
            Some(&input_identity)
        );

        let mut failing_observer = CompositeCausalObserver {
            values: Vec::new(),
            replace_vision: false,
            fail_path: Some("group.2.unit.0.output"),
        };
        assert!(session
            .decode_input_with_observer(Some(&decode), &(), &mut failing_observer)
            .is_err());
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 1, 1]);

        fail_completion.set(true);
        assert!(session.decode_input(Some(&decode), &()).is_err());
        fail_completion.set(false);
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 1, 1]);
        assert_eq!(
            session.committed_prompt_input_identity(),
            Some(&input_identity)
        );

        session.reset(&()).unwrap();
        assert_eq!(session.report().unwrap().state_report(), &[0, 0, 0, 0]);
        assert!(session.committed_prompt_input_identity().is_none());
        assert_eq!(completions.borrow().len(), counters.snapshot().publications);
        assert_eq!(
            counters.snapshot().completion_attempts,
            counters.snapshot().publications + 1
        );
        COMPOSITE_FORWARD_RESOURCE.with(|resource| resource.borrow_mut().take());
    }
}

#[test]
fn composite_prepared_parts_drive_dependency_group_output() {
    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let mut runtime =
        ResidentRuntime::<_, FakeBackend, DeviceState<_, _>>::new(architecture, &()).unwrap();
    let mut state = fixture_state();
    let first = prepared_composite_input(3);
    let first_output = runtime.forward(Some(&first), &mut state, &()).unwrap();
    assert_eq!(first_output, FakeTensor(vec![15, 20, 21]));
    assert_eq!(
        runtime.architecture().trace,
        [(0, 0), (1, 0), (2, 0), (2, 1)]
    );

    runtime.architecture_mut().trace.clear();
    let changed_media = prepared_composite_input(11);
    let changed_output = runtime
        .forward(Some(&changed_media), &mut state, &())
        .unwrap();
    assert_eq!(changed_output, FakeTensor(vec![23, 20, 21]));
    assert_ne!(first_output, changed_output);
    assert_eq!(
        first.parts()[0].payload().value(),
        changed_media.parts()[0].payload().value(),
        "text input remains fixed while media changes"
    );
    assert_eq!(
        runtime.architecture().trace,
        [(0, 0), (1, 0), (2, 0), (2, 1)]
    );
}

#[test]
fn layerwise_runtime_aborts_active_lease_and_forward_after_observer_error() {
    let mut state = fixture_state();
    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let units = [0, 10, 20, 21]
        .into_iter()
        .map(|marker| FakeUnit { marker })
        .collect();
    let mut runtime =
        LayerwiseRuntime::<_, FakeBackend, _, _>::new(architecture, RecordingPolicy::new(units));

    let error = runtime
        .forward_with_context_hook(None, &mut state, &(), |_, _, _| {
            Err(Error::backend("controlled observer failure"))
        })
        .unwrap_err();
    assert!(error.to_string().contains("controlled observer failure"));
    assert_eq!(runtime.policy().aborts, 1);
    assert!(!runtime.policy().forward_active);
    assert!(runtime.policy().units.iter().all(Option::is_some));

    let output = runtime.forward(None, &mut state, &()).unwrap();
    assert_eq!(output, FakeTensor(vec![30, 20, 21]));
    assert_eq!(runtime.policy().aborts, 1);
    assert!(!runtime.policy().forward_active);
}

#[test]
fn neutral_layerwise_runtime_accepts_a_static_unit_executor() {
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::new(LayerSchedule::new(4, policies).unwrap()).unwrap();
    let mut state = DeviceState::<FakeBackend, FakeLayerState>::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState(0))
    })
    .unwrap();
    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let units = [0, 10, 20, 21]
        .into_iter()
        .map(|marker| FakeUnit { marker })
        .collect();
    let mut runtime =
        LayerwiseRuntime::<_, FakeBackend, _, _>::new(architecture, RecordingPolicy::new(units));
    let mut trace = Vec::new();

    let output = runtime
        .forward_with_unit_executor(
            None,
            &mut state,
            &(),
            |_architecture, group, index, unit, hidden, _state, _forward, _context| {
                trace.push((group, index));
                let mut output = hidden.clone();
                output.0.push(unit.marker);
                Ok(output)
            },
        )
        .unwrap();

    assert_eq!(output, FakeTensor(vec![30, 20, 21]));
    assert_eq!(trace, vec![(0, 0), (1, 0), (2, 0), (2, 1)]);
    assert!(runtime.architecture().trace.is_empty());
}

#[derive(Clone)]
struct FixtureDecisionSampler(i32);

impl Sampler<FakeBackend> for FixtureDecisionSampler {
    fn sample(
        &mut self,
        logits: &FakeTensor,
        _: f32,
        random: Option<&mut i32>,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        if let Some(random) = random {
            *random += 1;
        }
        let mut token = logits.clone();
        token.0.push(self.0);
        Ok(token)
    }
}

#[derive(Default)]
struct FixtureDecisionBoundary {
    accepted: Vec<(usize, LayeredTraversalPoint, FakeTensor)>,
}

impl SequentialDecisionBoundary<FakeBackend, GroupedForwardContext, Error>
    for FixtureDecisionBoundary
{
    fn prediction_at(
        &self,
        point: LayeredTraversalPoint,
        _: &GroupedForwardContext,
    ) -> Option<usize> {
        match point {
            LayeredTraversalPoint::Group { group: 0 } => Some(0),
            LayeredTraversalPoint::Unit { group: 2, index } => Some(index + 1),
            _ => None,
        }
    }

    fn logits(
        &mut self,
        _: usize,
        _: LayeredTraversalPoint,
        value: &FakeTensor,
        _: &mut GroupedForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        Ok(value.clone())
    }

    fn token_domain(
        &mut self,
        _: usize,
        _: LayeredTraversalPoint,
        _: &GroupedForwardContext,
    ) -> Result<TokenDomain, Error> {
        Ok(TokenDomain::new(10_000))
    }

    fn accept(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        token: &FakeTensor,
        forward: &mut GroupedForwardContext,
        _: &(),
    ) -> Result<(), Error> {
        let token_marker = token
            .0
            .first()
            .copied()
            .ok_or_else(|| Error::backend("fixture decision token is empty"))?;
        forward.push((
            prediction,
            usize::try_from(token_marker).map_err(|error| Error::backend(error.to_string()))?,
        ));
        self.accepted.push((prediction, point, token.clone()));
        Ok(())
    }

    fn decision_error(&mut self, error: SequentialDecisionError<Error>) -> Error {
        Error::backend(error.to_string())
    }
}

fn fixture_decision_driver() -> SequentialDecisionDriver<FakeBackend, FixtureDecisionSampler> {
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Sample,
            PredictionDirective::Force(FakeTensor(vec![81])),
            PredictionDirective::Force(FakeTensor(vec![82])),
        ],
        false,
        true,
    )
    .unwrap();
    SequentialDecisionDriver::new(
        plan,
        vec![FixtureDecisionSampler(100); 3],
        vec![0.0; 3],
        Some(7),
    )
    .unwrap()
}

fn fixture_state() -> DeviceState<FakeBackend, FakeLayerState> {
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::new(LayerSchedule::new(4, policies).unwrap()).unwrap();
    DeviceState::create(layout, |_, _| Ok::<_, Infallible>(FakeLayerState(0))).unwrap()
}

#[test]
fn sequential_decision_driver_is_shared_by_resident_and_layerwise_traversal() {
    let resident_architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let mut resident =
        ResidentRuntime::<_, FakeBackend, DeviceState<_, _>>::new(resident_architecture, &())
            .unwrap();
    let mut resident_state = fixture_state();
    let mut resident_driver = fixture_decision_driver();
    let mut resident_boundary = FixtureDecisionBoundary::default();
    let (resident_output, resident_context) = {
        let mut hook =
            SequentialDecisionTraversal::new(&mut resident_driver, &mut resident_boundary);
        resident
            .forward_with_traversal_hook(None, &mut resident_state, &(), &mut hook)
            .unwrap()
    };

    resident_driver.finish().unwrap();
    assert_eq!(resident_output, FakeTensor(vec![30]));
    assert_eq!(resident.architecture().trace, vec![(0, 0), (1, 0)]);
    assert_eq!(resident_context, vec![(0, 10), (1, 81), (2, 82)]);
    assert_eq!(resident_driver.random_state(), Some(&8));
    assert_eq!(
        resident_driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );

    let layerwise_architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let units = [0, 10, 20, 21]
        .into_iter()
        .map(|marker| FakeUnit { marker })
        .collect();
    let mut layerwise = LayerwiseRuntime::<_, FakeBackend, _, _>::new(
        layerwise_architecture,
        RecordingPolicy::new(units),
    );
    let mut layerwise_state = fixture_state();
    let mut layerwise_driver = fixture_decision_driver();
    let mut layerwise_boundary = FixtureDecisionBoundary::default();
    let (layerwise_output, layerwise_context) = {
        let mut hook =
            SequentialDecisionTraversal::new(&mut layerwise_driver, &mut layerwise_boundary);
        layerwise
            .forward_with_traversal_hook(None, &mut layerwise_state, &(), &mut hook)
            .unwrap()
    };

    layerwise_driver.finish().unwrap();
    assert_eq!(layerwise_output, resident_output);
    assert_eq!(
        layerwise.architecture().trace,
        resident.architecture().trace
    );
    assert_eq!(layerwise_context, resident_context);
    assert_eq!(layerwise.policy().addresses, vec![(0, 0, 0), (1, 1, 0)]);
    assert_eq!(
        layerwise_driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        resident_driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>()
    );
    assert_eq!(layerwise_boundary.accepted, resident_boundary.accepted);
}

struct RecordingTraversalHook {
    name: &'static str,
    skip: bool,
    events: Rc<RefCell<Vec<String>>>,
}

impl LayeredTraversalHook<FakeBackend, (), Error> for RecordingTraversalHook {
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        _: usize,
        _: &mut FakeTensor,
        _: &mut (),
        _: &(),
    ) -> Result<LayeredUnitAction, Error> {
        self.events
            .borrow_mut()
            .push(format!("{}.before.{group}.{index}", self.name));
        Ok(if self.skip {
            LayeredUnitAction::SkipRemainingGroup
        } else {
            LayeredUnitAction::Execute
        })
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        _: &mut FakeTensor,
        _: &mut (),
        _: &(),
    ) -> Result<(), Error> {
        self.events
            .borrow_mut()
            .push(format!("{}.unit.{group}.{index}", self.name));
        Ok(())
    }

    fn after_group(
        &mut self,
        group: usize,
        _: &mut FakeTensor,
        _: &mut (),
        _: &(),
    ) -> Result<(), Error> {
        self.events
            .borrow_mut()
            .push(format!("{}.group.{group}", self.name));
        Ok(())
    }
}

#[test]
fn composite_traversal_hook_preserves_order_and_combines_tail_skip() {
    let events = Rc::new(RefCell::new(Vec::new()));
    let left = RecordingTraversalHook {
        name: "left",
        skip: false,
        events: Rc::clone(&events),
    };
    let right = RecordingTraversalHook {
        name: "right",
        skip: true,
        events: Rc::clone(&events),
    };
    let mut hook = CompositeLayeredTraversalHook::new(left, right);
    let mut value = FakeTensor(vec![1]);
    assert_eq!(
        hook.before_unit(1, 2, 3, &mut value, &mut (), &()).unwrap(),
        LayeredUnitAction::SkipRemainingGroup
    );
    hook.after_unit(1, 2, &mut value, &mut (), &()).unwrap();
    hook.after_group(1, &mut value, &mut (), &()).unwrap();
    assert_eq!(
        events.borrow().as_slice(),
        [
            "left.before.1.2",
            "right.before.1.2",
            "left.unit.1.2",
            "right.unit.1.2",
            "left.group.1",
            "right.group.1",
        ]
    );
}

#[test]
fn opaque_communication_projection_drives_backend_independent_callbacks() {
    let topology = eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap();
    let tensor_limits = eredu_runtime::CommunicationTensorLimits::new(1, 3, 4096, None).unwrap();
    let tensor_requirement = eredu_runtime::CommunicationOperationRequirement::tensors(
        eredu_runtime::CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],
        tensor_limits,
        true,
    )
    .unwrap();
    let group_requirements =
        eredu_runtime::CommunicationGroupRequirements::new([tensor_requirement]).unwrap();
    let route_requirement = eredu_runtime::CommunicationOperationRequirement::tensors(
        eredu_runtime::CommunicationOperation::SendReceive,
        [TensorDtype::F32],
        eredu_runtime::CommunicationTensorLimits::new(2, 3, 4096, None).unwrap(),
        true,
    )
    .unwrap();
    let plan = eredu_runtime::TopologyCommunicationPlan::new()
        .with_tensor_groups(group_requirements)
        .with_pipeline_routes(route_requirement)
        .unwrap();
    let manifest = eredu_runtime::project_all_communication_manifests(topology, &plan)
        .unwrap()
        .remove(0);

    let local_groups = manifest
        .try_create_groups(|descriptor| {
            Ok::<_, Infallible>(
                descriptor
                    .collective_descriptor()
                    .map(|group| (group.id(), group.members().to_vec(), group.local_rank())),
            )
        })
        .unwrap();
    assert_eq!(
        local_groups.into_iter().flatten().collect::<Vec<_>>(),
        [(eredu_core::CollectiveGroupId::new(1), vec![0, 1], 0)]
    );

    let routes = manifest
        .try_create_routes(|descriptor| {
            Ok::<_, Infallible>((
                descriptor.id(),
                descriptor.source(),
                descriptor.destination(),
                descriptor.requirement().operation(),
            ))
        })
        .unwrap();
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().all(|(_, source, destination, operation)| {
        source != destination && *operation == eredu_runtime::CommunicationOperation::SendReceive
    }));
}

#[test]
fn communication_extensions_require_only_the_selected_operation() {
    let trace = Rc::new(RefCell::new(Vec::new()));
    let group = PartitionCollectiveGroup {
        members: vec![2, 5],
        local_rank: 1,
        trace: Rc::clone(&trace),
    };
    let submission =
        PartitionCollectiveBackend::all_reduce_sum(FakeTensor(vec![7, 9]), &group, &()).unwrap();
    assert_eq!(submission.wait().unwrap(), FakeTensor(vec![7, 9]));
    assert_eq!(
        trace.borrow().as_slice(),
        [PartitionCollectiveCall {
            local_rank: 1,
            members: vec![2, 5],
            value: vec![7, 9],
        }]
    );
}
