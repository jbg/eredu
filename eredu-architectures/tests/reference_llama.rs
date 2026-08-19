use eredu_architectures::llama::{self, LayeredInput, ModelArgs};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, AttentionMask, EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator,
    LinearSpec, NeuralBackend, NormalizationOperator, NormalizationSpec, PadMode,
    ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized, RotaryOperator,
    RotarySpec, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, DeviceState, LayerwiseRuntime, ParameterBackend,
    ResidentRuntime, ResidentUnitWindow, RuntimeLayerState, RuntimeState, WeightBinding,
};

#[derive(Debug, Clone, Eq, PartialEq)]
struct ReferenceTensor(Vec<i32>);

impl ReferenceTensor {
    fn with_last(&self, dimension: i32) -> Self {
        let mut shape = self.0.clone();
        *shape.last_mut().expect("reference tensors are ranked") = dimension;
        Self(shape)
    }
}

impl Tensor for ReferenceTensor {
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
        let mut shape = shape.to_vec();
        if let Some(inferred) = shape.iter().position(|dimension| *dimension == -1) {
            let elements = self.0.iter().product::<i32>();
            let known = shape
                .iter()
                .filter(|dimension| **dimension != -1)
                .product::<i32>();
            shape[inferred] = elements / known;
        }
        Ok(Self(shape))
    }
    fn transpose_axes(&self, axes: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(
            axes.iter().map(|axis| self.0[*axis as usize]).collect(),
        ))
    }
    fn swap_axes(&self, left: i32, right: i32, _: &()) -> Result<Self, Error> {
        let mut shape = self.0.clone();
        shape.swap(left as usize, right as usize);
        Ok(Self(shape))
    }
    fn transpose(&self, _: &()) -> Result<Self, Error> {
        let mut shape = self.0.clone();
        shape.reverse();
        Ok(Self(shape))
    }
    fn expand_dims(&self, axis: i32, _: &()) -> Result<Self, Error> {
        let mut shape = self.0.clone();
        shape.insert(axis as usize, 1);
        Ok(Self(shape))
    }
    fn squeeze_axes(&self, axes: &[i32], _: &()) -> Result<Self, Error> {
        let mut shape = self.0.clone();
        for axis in axes.iter().rev() {
            shape.remove(*axis as usize);
        }
        Ok(Self(shape))
    }
    fn index(&self, _: &[Index], _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn take_axis(&self, _: &Self, _: i32, _: &()) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn concatenate(values: &[Self], axis: i32, _: &()) -> Result<Self, Error> {
        let mut shape = values[0].0.clone();
        shape[axis as usize] = values.iter().map(|value| value.0[axis as usize]).sum();
        Ok(Self(shape))
    }
    fn stack(values: &[Self], axis: i32, _: &()) -> Result<Self, Error> {
        let mut shape = values[0].0.clone();
        shape.insert(axis as usize, values.len() as i32);
        Ok(Self(shape))
    }
    fn matmul(lhs: &Self, rhs: &Self, _: &()) -> Result<Self, Error> {
        Ok(lhs.with_last(*rhs.0.last().expect("ranked right operand")))
    }
    fn sum_axis(value: &Self, axis: i32, keep_dims: bool, _: &()) -> Result<Self, Error> {
        let mut shape = value.0.clone();
        if keep_dims {
            shape[axis as usize] = 1;
        } else {
            shape.remove(axis as usize);
        }
        Ok(Self(shape))
    }
    fn argmin_axis(value: &Self, axis: i32, keep_dims: bool, context: &()) -> Result<Self, Error> {
        Self::sum_axis(value, axis, keep_dims, context)
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
    fn linear(input: &Self, weight: &Self, _: Option<&Self>, _: &()) -> Result<Self, Error> {
        Ok(input.with_last(weight.0[0]))
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

fn visit_parameter<'a, V>(metadata: &ParameterMetadata, value: &'a ReferenceTensor, visitor: &mut V)
where
    V: ParameterVisitor<'a, ReferenceTensor>,
{
    visitor.visit(metadata.clone(), value);
}

fn visit_parameter_mut<'a, V>(
    metadata: &ParameterMetadata,
    value: &'a mut ReferenceTensor,
    visitor: &mut V,
) where
    V: ParameterVisitorMut<'a, ReferenceTensor>,
{
    visitor.visit_mut(metadata.clone(), value);
}

#[derive(Debug, Clone)]
struct ReferenceLinear {
    output: i32,
    weight: ReferenceTensor,
    metadata: ParameterMetadata,
}

impl Parameterized<ReferenceTensor> for ReferenceLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
        visit_parameter(&self.metadata, &self.weight, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
        visit_parameter_mut(&self.metadata, &mut self.weight, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl LinearOperator<ReferenceTensor> for ReferenceLinear {
    fn forward(&mut self, input: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        Ok(input.with_last(self.output))
    }
}

#[derive(Debug, Clone)]
struct ReferenceEmbedding {
    vocabulary: i32,
    dimensions: i32,
    weight: ReferenceTensor,
    metadata: ParameterMetadata,
}

impl Parameterized<ReferenceTensor> for ReferenceEmbedding {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
        visit_parameter(&self.metadata, &self.weight, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
        visit_parameter_mut(&self.metadata, &mut self.weight, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl EmbeddingOperator<ReferenceTensor> for ReferenceEmbedding {
    fn forward(&mut self, input: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        let mut shape = input.0.clone();
        shape.push(self.dimensions);
        Ok(ReferenceTensor(shape))
    }
    fn as_linear(&mut self, input: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        Ok(input.with_last(self.vocabulary))
    }
}

#[derive(Debug, Clone)]
struct ReferenceNorm {
    weight: ReferenceTensor,
    metadata: ParameterMetadata,
}

impl Parameterized<ReferenceTensor> for ReferenceNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
        visit_parameter(&self.metadata, &self.weight, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
        visit_parameter_mut(&self.metadata, &mut self.weight, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl NormalizationOperator<ReferenceTensor> for ReferenceNorm {
    fn forward(&mut self, input: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
    }
}

#[derive(Debug, Clone)]
struct ReferenceRotary;

impl Parameterized<ReferenceTensor> for ReferenceRotary {
    fn visit_parameters<'a, V>(&'a self, _: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
    }
    fn set_trainable(&mut self, _: bool) {}
}

impl RotaryOperator<ReferenceTensor> for ReferenceRotary {
    fn forward(
        &mut self,
        input: &ReferenceTensor,
        _: i32,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
    }
}

struct ReferenceBackend;

impl NeuralBackend for ReferenceBackend {
    type Tensor = ReferenceTensor;
    type Linear = ReferenceLinear;
    type Embedding = ReferenceEmbedding;
    type Normalization = ReferenceNorm;
    type Rotary = ReferenceRotary;
    type ParallelContext = ();

    fn linear(spec: LinearSpec, _: &()) -> Result<Self::Linear, Error> {
        Ok(ReferenceLinear {
            output: spec.output,
            weight: ReferenceTensor(vec![spec.output, spec.input]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }
    fn embedding(spec: EmbeddingSpec, _: &()) -> Result<Self::Embedding, Error> {
        Ok(ReferenceEmbedding {
            vocabulary: spec.vocabulary,
            dimensions: spec.dimensions,
            weight: ReferenceTensor(vec![spec.vocabulary, spec.dimensions]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }
    fn rms_norm(spec: NormalizationSpec, _: &()) -> Result<Self::Normalization, Error> {
        Ok(ReferenceNorm {
            weight: ReferenceTensor(vec![spec.dimensions]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }
    fn rotary(_: RotarySpec<'_>, _: &()) -> Result<Self::Rotary, Error> {
        Ok(ReferenceRotary)
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
        Ok(ReferenceTensor(vec![sequence, sequence]))
    }
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        linear.forward(input, context)
    }
}

impl ParameterBackend for ReferenceBackend {
    type Parameter = ReferenceTensor;
    type MaterializedWeight = ReferenceTensor;
    type MaterializationContext = ();
    type Materialization = ReferenceTensor;
    type ParameterError = std::convert::Infallible;

    fn materialize(
        lease: eredu_checkpoint::store::CheckpointLease,
        _: &(),
    ) -> Result<Self::Materialization, Self::ParameterError> {
        use eredu_checkpoint::store::EncodedTensorLease;
        Ok(ReferenceTensor(
            lease
                .output_shape()
                .iter()
                .map(|dimension| *dimension as i32)
                .collect(),
        ))
    }

    fn materialize_recipe(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        _: &(),
    ) -> Result<Self::Materialization, Self::ParameterError> {
        let metadata = recipe.infer(source).expect("validated reference recipe");
        Ok(ReferenceTensor(
            metadata
                .shape
                .iter()
                .map(|dimension| *dimension as i32)
                .collect(),
        ))
    }

    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight {
        materialization
    }

    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(materialization)
    }

    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        *parameter = weight;
        Ok(())
    }
}

#[derive(Debug)]
struct ReferenceCache {
    offset: i32,
    window: Option<i32>,
}

impl AttentionCache<ReferenceTensor> for ReferenceCache {
    fn offset(&self) -> i32 {
        self.offset
    }
    fn max_size(&self) -> Option<i32> {
        self.window
    }
    fn update_for_attention(
        &mut self,
        keys: ReferenceTensor,
        values: ReferenceTensor,
        _: &(),
    ) -> Result<(ReferenceTensor, ReferenceTensor), Error> {
        self.offset += keys.dim(2);
        Ok((keys, values))
    }
    fn attention(
        &mut self,
        queries: ReferenceTensor,
        _: ReferenceTensor,
        _: ReferenceTensor,
        _: f32,
        _: Option<&ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(queries)
    }
}

impl RuntimeLayerState<ReferenceBackend> for ReferenceCache {
    type RetainedValues<'a> = std::iter::Empty<&'a ReferenceTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

fn tiny_args() -> ModelArgs {
    ModelArgs {
        model_type: "llama".into(),
        hidden_size: 8,
        num_hidden_layers: 2,
        intermediate_size: 16,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 32,
        num_key_value_heads: 1,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        rope_traditional: false,
        head_dim: 4,
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        rope_scaling: None,
        attention_schedule: LayerSchedule::new(2, vec![AttentionPolicy::Full; 2]).unwrap(),
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    }
}

fn topology<M: Parameterized<ReferenceTensor>>(module: &M) -> Vec<(String, Vec<usize>)> {
    struct Collector(Vec<(String, Vec<usize>)>);
    impl<'a> ParameterVisitor<'a, ReferenceTensor> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a ReferenceTensor) {
            self.0.push((
                metadata.id.to_string(),
                value
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect(),
            ));
        }
    }
    let mut collector = Collector(Vec::new());
    module.visit_parameters(&mut collector);
    collector.0
}

fn load_module<M: Parameterized<ReferenceTensor>>(
    module: &mut M,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) {
    let bindings = topology(module)
        .into_iter()
        .map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() as u64 * 4;
            WeightBinding::new(
                name.clone(),
                name,
                eredu_checkpoint::store::TensorSelection::Full,
                bytes,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let unit = materialize_bindings::<ReferenceBackend>(source, &bindings, &()).unwrap();
    bind_materialized_unit::<ReferenceBackend, _>(module, unit).unwrap();
}

#[test]
fn shared_llama_runs_prefill_and_decode_without_mlx() {
    let args = tiny_args();
    let layout = llama::state_layout(&args).unwrap();
    let state = DeviceState::<ReferenceBackend, ReferenceCache>::create(layout, |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
        })
    })
    .unwrap();
    let architecture = llama::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, state, &()).unwrap();

    let mut catalog = topology(runtime.architecture().static_modules());
    for unit in runtime.units() {
        catalog.extend(topology(unit));
    }
    let storage = catalog
        .into_iter()
        .map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * 4];
            (name, shape, bytes)
        })
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("model.safetensors");
    safetensors::tensor::serialize_to_file(
        storage.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        }),
        None,
        &checkpoint,
    )
    .unwrap();
    let source = eredu_checkpoint::store::SafetensorsWeightStore::open(&checkpoint).unwrap();
    load_module(runtime.architecture_mut().static_modules_mut(), &source);
    for unit in runtime.units_mut() {
        load_module(unit, &source);
    }

    let prefill = ReferenceTensor(vec![1, 3]);
    let logits = runtime
        .forward(
            LayeredInput {
                tokens: &prefill,
                mask: None,
            },
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 3, 32]);
    assert_eq!(runtime.state_mut().layer(0).unwrap().offset(), 3);
    assert_eq!(runtime.state_mut().layer(1).unwrap().offset(), 3);

    let decode = ReferenceTensor(vec![1, 1]);
    let logits = runtime
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(runtime.state_mut().layer(0).unwrap().offset(), 4);
    assert_eq!(runtime.state_mut().layer(1).unwrap().offset(), 4);

    let (architecture, units, state) = runtime.into_parts();
    let mut layerwise = LayerwiseRuntime::new(architecture, state, ResidentUnitWindow::new(units));
    let logits = layerwise
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(layerwise.state_mut().layer(0).unwrap().offset(), 5);
    assert_eq!(layerwise.state_mut().layer(1).unwrap().offset(), 5);
}
