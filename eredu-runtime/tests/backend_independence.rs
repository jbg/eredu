use std::{cell::Cell, convert::Infallible};

use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, Completion, LayerSchedule};
use eredu_nn::{
    AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationOperator, NormalizationSpec, PadMode, ParameterVisitor,
    ParameterVisitorMut, Parameterized, RotaryOperator, RotarySpec, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, CollectiveBackend, DeviceState, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitAddress, LayeredArchitecture, LayeredForwardState,
    LayerwisePolicy, LayerwiseRuntime, ParameterBackend, RuntimeLayerState, StateLayout,
    SubmissionBackend, TransferBackend, WeightBinding,
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
    fn forward(&mut self, input: &FakeTensor, _: i32, _: &()) -> Result<FakeTensor, Error> {
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
    fn rms_norm(_: NormalizationSpec, _: &()) -> Result<Self::Normalization, Error> {
        Ok(FakeOperator)
    }
    fn rotary(_: RotarySpec<'_>, _: &()) -> Result<Self::Rotary, Error> {
        Ok(FakeOperator)
    }
    fn silu(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
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

#[derive(Debug)]
struct FakeLayerState;

impl RuntimeLayerState<FakeBackend> for FakeLayerState {
    type RetainedValues<'a> = std::iter::Empty<&'a FakeTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
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
}

impl RecordingPolicy {
    fn new(units: Vec<FakeUnit>) -> Self {
        Self {
            units: units.into_iter().map(Some).collect(),
            addresses: Vec::new(),
        }
    }
}

impl LayerwisePolicy<FakeBackend, FakeUnit> for RecordingPolicy {
    type Lease = RecordingLease;
    type Error = &'static str;

    fn begin(&mut self, _: &FakeTensor, _: &()) -> Result<(), Self::Error> {
        Ok(())
    }

    fn acquire(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        _: &(),
    ) -> Result<Self::Lease, Self::Error> {
        self.addresses
            .push((ordinal, address.group(), address.index()));
        let unit = self
            .units
            .get_mut(ordinal)
            .and_then(Option::take)
            .ok_or("invalid fixture acquisition")?;
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

impl LayeredArchitecture<FakeBackend, DeviceState<FakeBackend, FakeLayerState>> for GroupedFixture {
    type Input<'a> = ();
    type StaticModules = FakeOperator;
    type Unit = FakeUnit;
    type ForwardContext = Vec<(usize, usize)>;
    type RetainedContextValues<'a> = std::iter::Empty<&'a FakeTensor>;
    type Error = Error;

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
fn neutral_layerwise_runtime_executes_dependency_groups_in_stable_order() {
    FORK_COUNT.set(0);
    SUBMIT_COUNT.set(0);
    ORDER_COUNT.set(0);
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::new(LayerSchedule::new(4, policies).unwrap()).unwrap();
    let mut state = DeviceState::<FakeBackend, FakeLayerState>::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState)
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
}

#[test]
fn neutral_layerwise_runtime_accepts_a_static_unit_executor() {
    let policies = (0..4)
        .map(|_| LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap())
        .collect();
    let layout = StateLayout::new(LayerSchedule::new(4, policies).unwrap()).unwrap();
    let mut state = DeviceState::<FakeBackend, FakeLayerState>::create(layout, |_, _| {
        Ok::<_, Infallible>(FakeLayerState)
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
