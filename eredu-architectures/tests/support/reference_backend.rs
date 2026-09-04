#[derive(Debug, Clone, Eq, PartialEq)]
struct ReferenceTensor(Vec<i32>);

#[derive(Debug, Clone, Default, Eq, PartialEq)]
struct ReferenceTrace {
    linear_outputs: Vec<(String, Vec<i32>)>,
    embedding_lookups: Vec<(String, Vec<i32>)>,
    rotary_offsets: Vec<i32>,
    sliding_attention: Vec<(i32, i32)>,
    causal_masks: Vec<(i32, i32, Option<i32>)>,
}

thread_local! {
    static REFERENCE_TRACE: RefCell<ReferenceTrace> = RefCell::new(ReferenceTrace::default());
}

fn clear_reference_trace() {
    REFERENCE_TRACE.with(|trace| *trace.borrow_mut() = ReferenceTrace::default());
}

fn reference_trace() -> ReferenceTrace {
    REFERENCE_TRACE.with(|trace| trace.borrow().clone())
}

impl ReferenceTensor {
    fn with_last(&self, dimension: i32) -> Self {
        let mut shape = self.0.clone();
        *shape.last_mut().expect("reference tensors are ranked") = dimension;
        Self(shape)
    }
}

impl eredu_nn::Tensor for ReferenceTensor {
    type Context = ();

    fn shape(&self) -> &[i32] {
        &self.0
    }

    fn unloaded_f32(shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn unloaded_i32(shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn from_f32_slice(_: &[f32], shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn from_i32_slice(_: &[i32], shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn full_f32(_: f32, shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
    }
    fn full_i32(_: i32, shape: &[i32], _: &()) -> Result<Self, Error> {
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
    fn tanh(&self, _: &()) -> Result<Self, Error> {
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
    fn broadcast_to(&self, shape: &[i32], _: &()) -> Result<Self, Error> {
        Ok(Self(shape.to_vec()))
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
    fn rope_with_frequencies(
        &self,
        _: i32,
        _: bool,
        _: i32,
        _: &Self,
        _: &(),
    ) -> Result<Self, Error> {
        Ok(self.clone())
    }
    fn concatenate(values: &[Self], axis: i32, _: &()) -> Result<Self, Error> {
        let mut shape = values[0].0.clone();
        let axis = if axis < 0 {
            usize::try_from(shape.len() as i32 + axis).map_err(Error::backend)?
        } else {
            usize::try_from(axis).map_err(Error::backend)?
        };
        shape[axis] = values.iter().map(|value| value.0[axis]).sum();
        Ok(Self(shape))
    }
    fn stack(values: &[Self], axis: i32, _: &()) -> Result<Self, Error> {
        let mut shape = values[0].0.clone();
        let axis = if axis < 0 {
            usize::try_from(shape.len() as i32 + axis + 1).map_err(Error::backend)?
        } else {
            usize::try_from(axis).map_err(Error::backend)?
        };
        shape.insert(axis, values.len() as i32);
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
    bias: Option<(ReferenceTensor, ParameterMetadata)>,
    expert_spec: Option<GroupedGatedProductSpec>,
    relu2_spec: Option<GroupedRelu2Spec>,
}

impl Parameterized<ReferenceTensor> for ReferenceLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
        visit_parameter(&self.metadata, &self.weight, visitor);
        if let Some((bias, metadata)) = &self.bias {
            visit_parameter(metadata, bias, visitor);
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
        visit_parameter_mut(&self.metadata, &mut self.weight, visitor);
        if let Some((bias, metadata)) = &mut self.bias {
            visit_parameter_mut(metadata, bias, visitor);
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.metadata.trainable = trainable;
    }
}

impl LinearOperator<ReferenceTensor> for ReferenceLinear {
    fn forward(&mut self, input: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        let output = input.with_last(self.output);
        REFERENCE_TRACE.with(|trace| {
            trace
                .borrow_mut()
                .linear_outputs
                .push((self.metadata.id.as_str().to_owned(), output.0.clone()));
        });
        Ok(output)
    }
}

impl GroupSelectionOperator<ReferenceTensor> for ReferenceLinear {
    fn select(
        &mut self,
        input: &ReferenceTensor,
        _: &(),
    ) -> Result<GroupSelection<ReferenceTensor>, Error> {
        let tokens = input.shape()[..input.shape().len() - 1]
            .iter()
            .product::<i32>();
        let routes = ReferenceTensor(vec![tokens, self.output]);
        Ok(GroupSelection::new(routes.clone(), routes.clone(), routes))
    }

    fn select_indices(
        &mut self,
        _input: &ReferenceTensor,
        group_indices: &ReferenceTensor,
        _: &(),
    ) -> Result<GroupSelection<ReferenceTensor>, Error> {
        Ok(GroupSelection::new(
            group_indices.clone(),
            group_indices.clone(),
            group_indices.clone(),
        ))
    }
}

impl GroupedGatedProductOperator<ReferenceTensor> for ReferenceLinear {
    fn spec(&self) -> &GroupedGatedProductSpec {
        self.expert_spec
            .as_ref()
            .expect("reference expert bank retains its construction spec")
    }

    fn forward_grouped(
        &mut self,
        input: &ReferenceTensor,
        _: &GroupSelection<ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
    }
}

impl TensorParallelGroupedGatedProductOperator<ReferenceTensor> for ReferenceLinear {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &ReferenceTensor,
        _: &GroupSelection<ReferenceTensor>,
        _: usize,
        _: &(),
    ) -> Result<TensorParallelGroupedOutput<ReferenceTensor>, Error> {
        Ok(TensorParallelGroupedOutput::new(input.clone(), None))
    }
}

impl GroupedRelu2Operator<ReferenceTensor> for ReferenceLinear {
    fn spec(&self) -> &GroupedRelu2Spec {
        self.relu2_spec
            .as_ref()
            .expect("reference ReLU-squared bank retains its construction spec")
    }

    fn forward_grouped(
        &mut self,
        input: &ReferenceTensor,
        _: &GroupSelection<ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
    }
}

impl TensorParallelGroupedRelu2Operator<ReferenceTensor> for ReferenceLinear {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &ReferenceTensor,
        _: &GroupSelection<ReferenceTensor>,
        _: usize,
        _: &(),
    ) -> Result<TensorParallelGroupedOutput<ReferenceTensor>, Error> {
        Ok(TensorParallelGroupedOutput::new(input.clone(), None))
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
    fn lookup(
        &mut self,
        input: &ReferenceTensor,
        policy: EmbeddingLookupPolicy,
        context: &(),
    ) -> Result<ReferenceTensor, Error> {
        policy.validate()?;
        REFERENCE_TRACE.with(|trace| {
            trace
                .borrow_mut()
                .embedding_lookups
                .push((self.metadata.id.as_str().to_owned(), input.0.clone()));
        });
        self.forward(input, context)
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
        position: RotaryPosition<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        if let RotaryPosition::Offset(offset) = position {
            REFERENCE_TRACE.with(|trace| trace.borrow_mut().rotary_offsets.push(offset));
        }
        Ok(input.clone())
    }
}

#[derive(Clone, Copy)]
struct ReferenceBackend;

#[derive(Clone, Debug)]
struct ReferenceHyperConnection {
    parameters: Vec<(ReferenceTensor, ParameterMetadata)>,
}

#[derive(Clone, Debug)]
struct ReferenceHyperHead {
    parameters: Vec<(ReferenceTensor, ParameterMetadata)>,
}

macro_rules! impl_reference_hyper_parameters {
    ($name:ty) => {
        impl Parameterized<ReferenceTensor> for $name {
            fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
            where
                V: ParameterVisitor<'a, ReferenceTensor>,
            {
                for (value, metadata) in &self.parameters {
                    visitor.visit(metadata.clone(), value);
                }
            }

            fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
            where
                V: ParameterVisitorMut<'a, ReferenceTensor>,
            {
                for (value, metadata) in &mut self.parameters {
                    visitor.visit_mut(metadata.clone(), value);
                }
            }

            fn set_trainable(&mut self, trainable: bool) {
                for (_, metadata) in &mut self.parameters {
                    metadata.trainable = trainable;
                }
            }
        }
    };
}

impl_reference_hyper_parameters!(ReferenceHyperConnection);
impl_reference_hyper_parameters!(ReferenceHyperHead);

impl eredu_nn::HyperConnectionOperator<ReferenceTensor> for ReferenceHyperConnection {
    fn collapse(
        &mut self,
        residual: &ReferenceTensor,
        _: f32,
        _: &(),
    ) -> Result<eredu_nn::HyperConnectionState<ReferenceTensor>, Error> {
        let shape = residual.shape();
        let (batch, tokens, streams, hidden) = (shape[0], shape[1], shape[2], shape[3]);
        Ok(eredu_nn::HyperConnectionState {
            collapsed: ReferenceTensor(vec![batch, tokens, hidden]),
            pre: ReferenceTensor(vec![batch, tokens, streams]),
            post: ReferenceTensor(vec![batch, tokens, streams]),
            combination: ReferenceTensor(vec![batch, tokens, streams, streams]),
        })
    }

    fn expand(
        &mut self,
        _: &ReferenceTensor,
        residual: &ReferenceTensor,
        _: &eredu_nn::HyperConnectionState<ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(residual.clone())
    }
}

impl eredu_nn::HyperHeadOperator<ReferenceTensor> for ReferenceHyperHead {
    fn forward(&mut self, residual: &ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        let shape = residual.shape();
        Ok(ReferenceTensor(vec![shape[0], shape[1], shape[3]]))
    }
}

impl eredu_nn::HyperNeuralBackend for ReferenceBackend {
    type HyperConnection = ReferenceHyperConnection;
    type HyperHead = ReferenceHyperHead;

    fn hyper_connection(
        spec: eredu_nn::HyperConnectionSpec,
        _: &(),
    ) -> Result<Self::HyperConnection, Error> {
        spec.validate()?;
        let streams = spec.streams;
        let hidden = spec.hidden_size;
        Ok(ReferenceHyperConnection {
            parameters: vec![
                (
                    ReferenceTensor(vec![(2 + streams) * streams, streams * hidden]),
                    ParameterMetadata::from_spec(&spec.function, spec.function.trainable),
                ),
                (
                    ReferenceTensor(vec![(2 + streams) * streams]),
                    ParameterMetadata::from_spec(&spec.base, spec.base.trainable),
                ),
                (
                    ReferenceTensor(vec![3]),
                    ParameterMetadata::from_spec(&spec.scale, spec.scale.trainable),
                ),
            ],
        })
    }

    fn hyper_head(spec: eredu_nn::HyperHeadSpec, _: &()) -> Result<Self::HyperHead, Error> {
        spec.validate()?;
        Ok(ReferenceHyperHead {
            parameters: vec![
                (
                    ReferenceTensor(vec![spec.streams, spec.streams * spec.hidden_size]),
                    ParameterMetadata::from_spec(&spec.function, spec.function.trainable),
                ),
                (
                    ReferenceTensor(vec![spec.streams]),
                    ParameterMetadata::from_spec(&spec.base, spec.base.trainable),
                ),
                (
                    ReferenceTensor(vec![1]),
                    ParameterMetadata::from_spec(&spec.scale, spec.scale.trainable),
                ),
            ],
        })
    }
}

impl NeuralBackend for ReferenceBackend {
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        eredu_architectures::operator_requirements::KIMI_LINEAR
            .union(eredu_architectures::operator_requirements::INKLING)
            .union(eredu_architectures::operator_requirements::GEMMA4)
            .union(eredu_architectures::operator_requirements::MUSE_GLIMMER)
            .union(eredu_architectures::operator_requirements::QWEN_VL)
            .union(eredu_architectures::operator_requirements::QWEN_HYBRID)
            .union(eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL)
            .union(eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS)
            .union(eredu_nn::NeuralOperatorCapabilities::INDEXED_ATTENTION)
            .union(eredu_nn::NeuralOperatorCapabilities::POOLED_ATTENTION)
            .union(eredu_nn::NeuralOperatorCapabilities::POOLED_POSITION_SELECTION)
            .union(eredu_nn::NeuralOperatorCapabilities::POOLED_MASK_GATHER)
            .union(eredu_nn::NeuralOperatorCapabilities::GROUPED_LINEAR)
            .union(eredu_nn::NeuralOperatorCapabilities::ROPE_WITH_FREQUENCIES);

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
            bias: spec.bias.map(|bias| {
                (
                    ReferenceTensor(vec![spec.output]),
                    ParameterMetadata::from_spec(&bias, bias.trainable),
                )
            }),
            expert_spec: None,
            relu2_spec: None,
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
    fn normalization(
        spec: NormalizationConstructionSpec,
        _: &(),
    ) -> Result<Self::Normalization, Error> {
        spec.validate()?;
        let weight = match spec.scale {
            NormalizationScale::Learned(weight)
            | NormalizationScale::LearnedOffset { weight, .. } => weight,
            NormalizationScale::Unit => {
                return Err(Error::backend(
                    "reference backend does not model parameterless normalization",
                ));
            }
        };
        Ok(ReferenceNorm {
            weight: ReferenceTensor(vec![spec.dimensions]),
            metadata: ParameterMetadata::from_spec(&weight, weight.trainable),
        })
    }
    fn rotary(_: RotarySpec, _: &()) -> Result<Self::Rotary, Error> {
        Ok(ReferenceRotary)
    }
    fn silu(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
    }
    fn sigmoid(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
    }
    fn softplus(input: Self::Tensor, _: &()) -> Result<Self::Tensor, Error> {
        Ok(input)
    }

    fn rms_norm_without_weight(
        input: &Self::Tensor,
        _: f32,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(input.clone())
    }

    fn gated_product(
        gate: Self::Tensor,
        _up: Self::Tensor,
        _policy: eredu_nn::GatedProductPolicy,
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
    fn attention_with_sinks(
        request: AttentionRequest<'_, Self::Tensor>,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        request.validate()?;
        Self::attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            request.mask,
            context,
        )
    }
    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, Self::Tensor>,
        window: i32,
        position_offset: i32,
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        request.validate()?;
        Self::sliding_window_attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            window,
            position_offset,
            context,
        )
    }
    fn sliding_window_attention(
        queries: Self::Tensor,
        _: Self::Tensor,
        _: Self::Tensor,
        _: f32,
        window: i32,
        position_offset: i32,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        REFERENCE_TRACE.with(|trace| {
            trace
                .borrow_mut()
                .sliding_attention
                .push((window, position_offset));
        });
        Ok(queries)
    }
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        REFERENCE_TRACE.with(|trace| {
            trace
                .borrow_mut()
                .causal_masks
                .push((sequence, offset, window));
        });
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

impl eredu_nn::DistributedNeuralBackend for ReferenceBackend {
    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        _: &(),
    ) -> Result<Self::Embedding, Error> {
        range.validate()?;
        Ok(ReferenceEmbedding {
            vocabulary: spec.vocabulary,
            dimensions: spec.dimensions,
            weight: ReferenceTensor(vec![range.local.len() as i32, spec.dimensions]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
        })
    }
    fn vocabulary_parallel_linear(
        spec: LinearSpec,
        range: VocabularyParallelRange,
        _: &(),
    ) -> Result<Self::Linear, Error> {
        range.validate()?;
        Ok(ReferenceLinear {
            output: spec.output,
            weight: ReferenceTensor(vec![range.local.len() as i32, spec.input]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            bias: None,
            expert_spec: None,
            relu2_spec: None,
        })
    }
    fn vocabulary_parallel_lookup(
        embedding: &mut Self::Embedding,
        input: &Self::Tensor,
        policy: EmbeddingLookupPolicy,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        embedding.lookup(input, policy, context)
    }
    fn vocabulary_parallel_project(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        linear.forward(input, context)
    }
    fn vocabulary_parallel_embedding_project(
        embedding: &mut Self::Embedding,
        input: &Self::Tensor,
        _: &(),
        context: &(),
    ) -> Result<Self::Tensor, Error> {
        embedding.as_linear(input, context)
    }
    fn sum_parallel(
        value: Self::Tensor,
        _: &Self::ParallelContext,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        Ok(value)
    }
}

impl eredu_nn::BlockwiseAttentionBackend for ReferenceBackend {
    type BlockwiseAccumulator = ReferenceTensor;

    fn begin_blockwise_attention(
        spec: eredu_nn::BlockwiseAttentionSpec<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<Self::BlockwiseAccumulator, Error> {
        Ok(spec.queries.clone())
    }

    fn accumulate_blockwise_attention(
        _: &mut Self::BlockwiseAccumulator,
        _: i64,
        _: i64,
        _: ReferenceTensor,
        _: ReferenceTensor,
        _: &(),
    ) -> Result<u64, Error> {
        Ok(0)
    }

    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(accumulator)
    }
}

impl GroupedNeuralBackend for ReferenceBackend {
    type Selector = ReferenceLinear;
    type GatedProductGroups = ReferenceLinear;
    type Relu2Groups = ReferenceLinear;

    fn grouped_linear(
        _: &mut Self::Linear,
        input: &Self::Tensor,
        groups: i32,
        output_per_group: i32,
        _: &(),
    ) -> Result<Self::Tensor, Error> {
        let [batch, input_groups, tokens, _] = input.shape() else {
            return Err(Error::backend(
                "structural grouped-linear input must have rank four",
            ));
        };
        if *input_groups != groups {
            return Err(Error::backend("structural grouped-linear group mismatch"));
        }
        Ok(ReferenceTensor(vec![
            *batch,
            groups,
            *tokens,
            output_per_group,
        ]))
    }

    fn joint_group_selection(
        input: eredu_nn::JointGroupSelectionInput<'_, Self::Tensor>,
        _: &(),
    ) -> Result<eredu_nn::JointGroupSelection<Self::Tensor>, Error> {
        input.validate()?;
        let tokens = input.hidden().shape()[..input.hidden().shape().len() - 1]
            .iter()
            .product::<i32>();
        Ok(eredu_nn::JointGroupSelection::new(
            ReferenceTensor(vec![tokens, input.top_k()]),
            ReferenceTensor(vec![tokens, input.top_k()]),
            ReferenceTensor(vec![tokens, input.always_on_groups()]),
        ))
    }

    fn top_k_group_selector(spec: TopKGroupSelectorSpec, _: &()) -> Result<Self::Selector, Error> {
        Ok(ReferenceLinear {
            output: spec.selection().top_k(),
            weight: ReferenceTensor(vec![
                spec.selection().group_count(),
                spec.input_dimensions(),
            ]),
            metadata: ParameterMetadata::from_spec(spec.weight(), spec.weight().trainable),
            bias: None,
            expert_spec: None,
            relu2_spec: None,
        })
    }

    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        _: &(),
    ) -> Result<Self::GatedProductGroups, Error> {
        let weight = match &spec.layout() {
            eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. } => gate_up.weight().clone(),
            eredu_nn::GatedProductGroupLayout::Independent(experts) => {
                experts[0].gate().weight().clone()
            }
            _ => return Err(Error::backend("unsupported grouped parameter layout")),
        };
        Ok(ReferenceLinear {
            output: spec.output_dimensions(),
            weight: ReferenceTensor(vec![
                spec.group_count(),
                2 * spec.intermediate_dimensions(),
                spec.input_dimensions(),
            ]),
            metadata: ParameterMetadata::from_spec(&weight, weight.trainable),
            bias: None,
            expert_spec: Some(spec),
            relu2_spec: None,
        })
    }

    fn grouped_relu2(spec: GroupedRelu2Spec, _: &()) -> Result<Self::Relu2Groups, Error> {
        spec.validate()?;
        Ok(ReferenceLinear {
            output: spec.hidden_dimensions(),
            weight: ReferenceTensor(vec![
                spec.group_count(),
                spec.intermediate_dimensions(),
                spec.hidden_dimensions(),
            ]),
            metadata: ParameterMetadata::from_spec(
                spec.up().weight(),
                spec.up().weight().trainable,
            ),
            bias: None,
            expert_spec: None,
            relu2_spec: Some(spec),
        })
    }
}

#[derive(Debug)]
struct ReferenceCompletion;

impl Completion for ReferenceCompletion {
    type Error = std::convert::Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SubmissionBackend for ReferenceBackend {
    type Executor = ();
    type OwnedExecutor = ();
    type Completion = ReferenceCompletion;

    fn fork_executors(
        _: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, std::convert::Infallible> {
        Ok(vec![(); count])
    }

    fn submit<'a, I>(_: &Self::Executor, _: I) -> Result<Self::Completion, std::convert::Infallible>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        Ok(ReferenceCompletion)
    }

    fn order_after(
        _: &Self::Completion,
        _: &Self::Executor,
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
    }

    fn retain_until_complete<T: Send + 'static>(
        _: &Self::Executor,
        _: &Self::Completion,
        _: T,
    ) -> Result<(), std::convert::Infallible> {
        Ok(())
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

    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(weight.clone())
    }

    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        assert_eq!(parameter.0, weight.0);
        Ok(())
    }

    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        *parameter = weight;
        Ok(())
    }
}

impl SamplingBackend for ReferenceBackend {
    type Logits = ReferenceTensor;
    type Token = ReferenceTensor;
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
            .ok_or_else(|| Error::backend("reference token is outside its decision domain"))
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
        _: &Self::Logits,
        _: f32,
        random: Option<&mut Self::RandomState>,
        _: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        if let Some(random) = random {
            *random += 1;
        }
        Ok(ReferenceTensor(vec![1, 1]))
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
        Ok(token.0.first().copied().unwrap_or_default() as u32)
    }

    fn token_probability(_: &Self::Logits, _: u32, _: &Self::Context) -> Result<f32, Self::Error> {
        Ok(1.0)
    }
}

#[derive(Clone)]
struct ReferenceSampler;

impl Sampler<ReferenceBackend> for ReferenceSampler {
    fn sample(
        &mut self,
        logits: &ReferenceTensor,
        temperature: f32,
        random: Option<&mut i32>,
        context: &(),
    ) -> Result<ReferenceTensor, Error> {
        ReferenceBackend::sample_raw(logits, temperature, random, context)
    }
}

#[derive(Clone, Debug)]
struct ReferenceCache {
    offset: i32,
    window: Option<i32>,
    resets: usize,
    fixed: Option<ReferenceTensor>,
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
        request: AttentionRequest<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(request.queries)
    }
}

impl eredu_nn::AuxiliaryConvolutionState<ReferenceTensor> for ReferenceCache {
    fn convolution_state(&mut self, _slot: u32) -> Result<&mut Option<ReferenceTensor>, Error> {
        Ok(&mut self.fixed)
    }
}

impl eredu_nn::CompressedAttentionCache<ReferenceTensor> for ReferenceCache {
    type Checkpoint = (i32, Option<ReferenceTensor>);

    fn offset(&self) -> i32 {
        self.offset
    }

    fn is_paged(&self) -> bool {
        false
    }

    fn append(
        &mut self,
        state: eredu_nn::CompressedAttentionState<ReferenceTensor>,
        _: &(),
    ) -> Result<eredu_nn::CompressedAttentionView<ReferenceTensor>, Error> {
        self.offset += state.latent.dim(1);
        Ok(eredu_nn::CompressedAttentionView::Resident(state))
    }

    fn visit_blocks<F>(
        &mut self,
        _: i32,
        _: &(),
        _: F,
    ) -> Result<eredu_nn::CompressedAttentionScan, Error>
    where
        F: FnMut(eredu_nn::CompressedAttentionBlock<ReferenceTensor>) -> Result<u64, Error>,
    {
        Err(Error::backend(
            "structural reference cache is not block-addressable",
        ))
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        (self.offset, self.fixed.clone())
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, _: &()) -> Result<(), Error> {
        self.offset = checkpoint.0;
        self.fixed.clone_from(&checkpoint.1);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.offset = 0;
        Ok(())
    }
}

impl eredu_nn::PoolingAttentionCache<ReferenceTensor> for ReferenceCache {
    type Checkpoint = Self;

    fn offset(&self) -> i32 {
        self.offset
    }

    fn pooling_ratio(&self, _stream: u32) -> Option<i32> {
        None
    }

    fn append_local(&mut self, keys: ReferenceTensor, _: &()) -> Result<ReferenceTensor, Error> {
        self.offset += keys.dim(1);
        Ok(keys)
    }

    fn local_mask(&self, query_tokens: i32, _: i32, _: &()) -> Result<ReferenceTensor, Error> {
        Ok(ReferenceTensor(vec![query_tokens, self.offset]))
    }

    fn accumulate_pooling_windows(
        &mut self,
        _stream: u32,
        values: ReferenceTensor,
        gates: ReferenceTensor,
        absolute_offset: i32,
        _: &(),
    ) -> Result<eredu_nn::PoolingWindows<ReferenceTensor>, Error> {
        Ok(eredu_nn::PoolingWindows {
            values,
            gates,
            base_position: absolute_offset,
        })
    }

    fn replace_pooling_overlap(
        &mut self,
        _stream: u32,
        _values: ReferenceTensor,
        _gates: ReferenceTensor,
    ) -> Result<eredu_nn::PoolingOverlap<ReferenceTensor>, Error> {
        Ok(eredu_nn::PoolingOverlap {
            values: None,
            gates: None,
        })
    }

    fn append_pooled(
        &mut self,
        _stream: u32,
        values: ReferenceTensor,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(values)
    }

    fn pooling_mask(
        &self,
        _stream: u32,
        _query_tokens: i32,
        _offset: i32,
        _: &(),
    ) -> Result<Option<ReferenceTensor>, Error> {
        Ok(None)
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn restore(&mut self, checkpoint: &Self::Checkpoint, _: &()) -> Result<(), Error> {
        self.clone_from(checkpoint);
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Error> {
        self.offset = 0;
        Ok(())
    }
}

impl RuntimeLayerState<ReferenceBackend> for ReferenceCache {
    type RetainedValues<'a> = std::iter::Empty<&'a ReferenceTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

impl RuntimeStateComponents<ReferenceBackend> for ReferenceCache {
    fn position(&self) -> i32 {
        self.offset
    }

    fn fixed_component(
        &mut self,
        _role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<ReferenceTensor>, StateError> {
        Ok(&mut self.fixed)
    }

    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        self.offset += tokens;
        Ok(())
    }
}

impl ResettableRuntimeLayerState<ReferenceBackend> for ReferenceCache {
    fn reset(&mut self) -> Result<(), StateError> {
        self.offset = 0;
        self.resets += 1;
        Ok(())
    }
}
