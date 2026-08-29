use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use eredu_core::{
    cache::LayerCachePolicy, AttentionPolicy, Completion, LayerSchedule, TokenFilter,
};
use eredu_nn::{
    AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, GatedProductPolicy, Index,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, PadMode, ParameterVisitor, ParameterVisitorMut, Parameterized,
    RotaryOperator, RotaryPosition, RotarySpec, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, ArchitectureGroupKind,
    ArchitectureGroupPlacement, ArchitectureGroupTransport, ArchitectureMergeDestination,
    ArchitectureParameterDescription, ArchitectureParameters, ArchitecturePartition,
    CollectiveBackend, CompositeLayeredTraversalHook, DeviceState, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, ExecutionUnitLayout, LayeredArchitecture,
    LayeredForwardState, LayeredTraversalHook, LayeredTraversalPoint, LayeredUnitAction,
    LayerwisePolicy, LayerwiseRuntime, ParameterBackend, PartitionOwnership, PenaltyConfig,
    PredictionDirective, ResettableRuntimeLayerState, ResettableRuntimeState, ResidentRuntime,
    RuntimeLayerState, Sampler, SamplingBackend, SequentialDecisionBoundary,
    SequentialDecisionDriver, SequentialDecisionError, SequentialDecisionPlan,
    SequentialDecisionSource, SequentialDecisionTraversal, StateError, StateLayout, StateSegmentId,
    StateSegmentLifetime, StateSegmentSpec, StaticParameterVisitor, StaticParameterVisitorMut,
    SubmissionBackend, TokenDomain, TransferBackend, WeightBinding,
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

thread_local! {
    static FORK_COUNT: Cell<usize> = const { Cell::new(0) };
    static SUBMIT_COUNT: Cell<usize> = const { Cell::new(0) };
    static ORDER_COUNT: Cell<usize> = const { Cell::new(0) };
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

impl CollectiveBackend for FakeBackend {
    type Group = ();
    type CollectiveError = Infallible;
    fn all_reduce(value: Self::Tensor, _: &(), _: &()) -> Result<Self::Tensor, Infallible> {
        Ok(value)
    }
    fn all_gather(value: Self::Tensor, _: &(), _: &()) -> Result<Self::Tensor, Infallible> {
        Ok(value)
    }
    fn all_to_all(value: Self::Tensor, _: &(), _: &()) -> Result<Self::Tensor, Infallible> {
        Ok(value)
    }
}

#[test]
fn runtime_capabilities_compile_and_run_without_mlx() {
    let host = FakeTensor(vec![2, 3]);
    let (mut parameter, transfer) = FakeBackend::promote(&(), &host).unwrap();
    assert_eq!(parameter, host);
    FakeBackend::bind(&mut parameter, FakeTensor(vec![3, 2])).unwrap();
    FakeBackend::retain_until_complete(&(), &transfer, parameter).unwrap();
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

#[derive(Debug)]
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
        _: F,
        _: &(),
    ) -> Result<Self::Lease, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&()) -> Result<FakeUnit, E>,
    {
        self.addresses
            .push((ordinal, address.group(), address.index()));
        let unit = self
            .units
            .get_mut(ordinal)
            .and_then(Option::take)
            .ok_or("invalid fixture acquisition")
            .map_err(eredu_runtime::LayerwiseAcquireError::Policy)?;
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
        Ok(eredu_runtime::ModelStateIdentity {
            model_family: "fixture".into(),
            effective_model_type: "fixture".into(),
            architecture_fingerprint: "fixture".into(),
            layer_count: 4,
            global_layer_start: state.global_layer_offset(),
            sink_tokens: 0,
            topology,
        })
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
    type Input<'a> = ();
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = Vec<(usize, usize)>;
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

    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::output_owner(0..layout.len()),
        ])
    }

    fn model_identity(&self) -> &str {
        "grouped-fixture"
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
        _: Self::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &(),
    ) -> Result<LayeredForwardState<FakeTensor, Self::ForwardContext>, Self::Error> {
        Ok(LayeredForwardState {
            hidden: FakeTensor(vec![0]),
            context: Vec::new(),
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        _: &FakeTensor,
        dependencies: &[&FakeTensor],
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
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
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: &mut Self::ForwardContext,
        _: &(),
    ) -> Result<FakeTensor, Self::Error> {
        self.trace.push((group, index));
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

#[test]
fn architecture_partition_is_derived_from_and_revalidates_neutral_topology() {
    type FixtureState = DeviceState<FakeBackend, FakeLayerState>;

    let architecture = GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    };
    let ownership = PartitionOwnership::new(true, false, ["embedding"]).expect("valid ownership");
    let parameters = architecture
        .parameter_description(&())
        .expect("fixture parameter description");
    let partition =
        ArchitecturePartition::<(), eredu_runtime::NoAuxiliaryBoundarySchema>::from_architecture::<
            FakeBackend,
            FixtureState,
            _,
            _,
        >(
            &architecture,
            [("vision", 0..1), ("text", 0..2)],
            ownership,
            (),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            &parameters,
        )
        .expect("partition derives its canonical graph");
    partition
        .validate_architecture::<FakeBackend, FixtureState, _>(&architecture)
        .expect("derived partition revalidates");
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
        .forward_with_context_hook((), &mut state, &(), |group, index, context| {
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

    let repeated = runtime.forward((), &mut state, &()).unwrap();
    assert_eq!(repeated, FakeTensor(vec![30, 20, 21]));
    assert_eq!(
        FORK_COUNT.get(),
        1,
        "one runtime must retain its graph executors across decode steps"
    );
    assert_eq!(SUBMIT_COUNT.get(), 8);
    assert_eq!(ORDER_COUNT.get(), 10);
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
        .forward_with_context_hook((), &mut state, &(), |_, _, _| {
            Err(Error::backend("controlled observer failure"))
        })
        .unwrap_err();
    assert!(error.to_string().contains("controlled observer failure"));
    assert_eq!(runtime.policy().aborts, 1);
    assert!(!runtime.policy().forward_active);
    assert!(runtime.policy().units.iter().all(Option::is_some));

    let output = runtime.forward((), &mut state, &()).unwrap();
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
            (),
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

impl SequentialDecisionBoundary<FakeBackend, Vec<(usize, usize)>, Error>
    for FixtureDecisionBoundary
{
    fn prediction_at(
        &self,
        point: LayeredTraversalPoint,
        _: &Vec<(usize, usize)>,
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
        _: &mut Vec<(usize, usize)>,
        _: &(),
    ) -> Result<FakeTensor, Error> {
        Ok(value.clone())
    }

    fn token_domain(
        &mut self,
        _: usize,
        _: LayeredTraversalPoint,
        _: &Vec<(usize, usize)>,
    ) -> Result<TokenDomain, Error> {
        Ok(TokenDomain::new(10_000))
    }

    fn accept(
        &mut self,
        prediction: usize,
        point: LayeredTraversalPoint,
        token: &FakeTensor,
        forward: &mut Vec<(usize, usize)>,
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
            .forward_with_traversal_hook((), &mut resident_state, &(), &mut hook)
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
            .forward_with_traversal_hook((), &mut layerwise_state, &(), &mut hook)
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
        _: &FakeTensor,
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
        _: &FakeTensor,
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
        _: &FakeTensor,
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
    let value = FakeTensor(vec![1]);
    assert_eq!(
        hook.before_unit(1, 2, 3, &value, &mut (), &()).unwrap(),
        LayeredUnitAction::SkipRemainingGroup
    );
    hook.after_unit(1, 2, &value, &mut (), &()).unwrap();
    hook.after_group(1, &value, &mut (), &()).unwrap();
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
