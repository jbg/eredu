use std::convert::Infallible;

use eredu_core::Completion;
use eredu_nn::{
    AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationOperator, NormalizationSpec, PadMode, ParameterVisitor,
    ParameterVisitorMut, Parameterized, RotaryOperator, RotarySpec, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, CollectiveBackend, ParameterBackend,
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
    type Completion = Done;
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
