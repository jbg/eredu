//! Multi-family structural and portable-lifecycle tests using a shape-only backend.

use std::{cell::RefCell, collections::BTreeMap};

use eredu_architectures::{
    decoder::{self, AttentionProjectionLayout, GatedProjectionLayout, TransformerBlock},
    gemma4, gpt_oss,
    llama::{self, LayeredInput, ModelArgs},
    moshi, muse_glimmer, qwen,
};
use eredu_core::{AttentionPolicy, Completion, LayerSchedule, TokenFilter};
use eredu_nn::{
    validate_parameter_topology, AttentionCache, AttentionMask, AttentionRequest,
    EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, GatedProductExpertBankOperator,
    GatedProductExpertBankSpec, Index, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, PadMode,
    ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized,
    Relu2ExpertBankOperator, Relu2ExpertBankSpec, RotaryOperator, RotaryPosition, RotarySpec,
    RoutedNeuralBackend, RoutingOperator, RoutingResult, Tensor, TopKRouterSpec,
    VocabularyParallelRange,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_bindings, ArchitectureParameters, DeviceState, ExpertPass,
    LayerRuntimeState, LayeredArchitecture, LayerwiseRuntime, LocalModelLayout, LocalTensorLayout,
    ParameterBackend, ParameterGroupSpec, PenaltyConfig, PredictionDirective,
    ResettableRuntimeLayerState, ResidentRuntime, ResidentUnitWindow, RoutedExpertProvider,
    RoutedExpertRequest, RuntimeLayerState, RuntimeState, Sampler, SamplingBackend,
    SequentialDecisionDriver, SequentialDecisionMode, SequentialDecisionPlan,
    SequentialDecisionSource, SequentialDecisionTraversal, StateError, StaticParameterVisitor,
    SubmissionBackend, TensorPlacement, TokenDomain, WeightBinding,
};

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

impl Tensor for ReferenceTensor {
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
    expert_spec: Option<GatedProductExpertBankSpec>,
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

impl RoutingOperator<ReferenceTensor> for ReferenceLinear {
    fn route(
        &mut self,
        input: &ReferenceTensor,
        _: &(),
    ) -> Result<RoutingResult<ReferenceTensor>, Error> {
        let tokens = input.shape()[..input.shape().len() - 1]
            .iter()
            .product::<i32>();
        let routes = ReferenceTensor(vec![tokens, self.output]);
        Ok(RoutingResult {
            expert_ids: routes.clone(),
            selected_scores: routes.clone(),
            route_weights: routes,
        })
    }

    fn route_selected(
        &mut self,
        _input: &ReferenceTensor,
        expert_ids: &ReferenceTensor,
        _: &(),
    ) -> Result<RoutingResult<ReferenceTensor>, Error> {
        Ok(RoutingResult {
            expert_ids: expert_ids.clone(),
            selected_scores: expert_ids.clone(),
            route_weights: expert_ids.clone(),
        })
    }
}

impl GatedProductExpertBankOperator<ReferenceTensor> for ReferenceLinear {
    fn spec(&self) -> &GatedProductExpertBankSpec {
        self.expert_spec
            .as_ref()
            .expect("reference expert bank retains its construction spec")
    }

    fn forward_routed(
        &mut self,
        input: &ReferenceTensor,
        _: &RoutingResult<ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
    }
}

impl Relu2ExpertBankOperator<ReferenceTensor> for ReferenceLinear {
    fn forward_routed(
        &mut self,
        input: &ReferenceTensor,
        _: &RoutingResult<ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Error> {
        Ok(input.clone())
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
            expert_spec: None,
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
            expert_spec: None,
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

impl RoutedNeuralBackend for ReferenceBackend {
    type Router = ReferenceLinear;
    type GatedProductExpertBank = ReferenceLinear;
    type Relu2ExpertBank = ReferenceLinear;

    fn top_k_router(spec: TopKRouterSpec, _: &()) -> Result<Self::Router, Error> {
        Ok(ReferenceLinear {
            output: spec.routing.top_k(),
            weight: ReferenceTensor(vec![spec.routing.expert_count(), spec.input_dimensions]),
            metadata: ParameterMetadata::from_spec(&spec.weight, spec.weight.trainable),
            expert_spec: None,
        })
    }

    fn gated_product_expert_bank(
        spec: GatedProductExpertBankSpec,
        _: &(),
    ) -> Result<Self::GatedProductExpertBank, Error> {
        let weight = match &spec.layout {
            eredu_nn::GatedProductExpertLayout::Packed { gate_up, .. } => gate_up.weight.clone(),
            eredu_nn::GatedProductExpertLayout::Independent(experts) => {
                experts[0].gate.weight.clone()
            }
        };
        Ok(ReferenceLinear {
            output: spec.output_dimensions,
            weight: ReferenceTensor(vec![
                spec.expert_count,
                2 * spec.intermediate_dimensions,
                spec.input_dimensions,
            ]),
            metadata: ParameterMetadata::from_spec(&weight, weight.trainable),
            expert_spec: Some(spec),
        })
    }

    fn relu2_expert_bank(
        spec: Relu2ExpertBankSpec,
        _: &(),
    ) -> Result<Self::Relu2ExpertBank, Error> {
        spec.validate()?;
        Ok(ReferenceLinear {
            output: spec.hidden_dimensions,
            weight: ReferenceTensor(vec![
                spec.expert_count,
                spec.intermediate_dimensions,
                spec.hidden_dimensions,
            ]),
            metadata: ParameterMetadata::from_spec(&spec.up.weight, spec.up.weight.trainable),
            expert_spec: None,
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

#[derive(Debug)]
struct ReferenceCache {
    offset: i32,
    window: Option<i32>,
    resets: usize,
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

impl RuntimeLayerState<ReferenceBackend> for ReferenceCache {
    type RetainedValues<'a> = std::iter::Empty<&'a ReferenceTensor>;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        std::iter::empty()
    }
}

impl ResettableRuntimeLayerState<ReferenceBackend> for ReferenceCache {
    fn reset(&mut self) -> Result<(), StateError> {
        self.offset = 0;
        self.resets += 1;
        Ok(())
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
        quantized_weights: None,
        quantized_weight_configs: None,
    }
}

fn llama_parallel_layout(
    args: &ModelArgs,
    vocabulary: std::ops::Range<usize>,
    local_query_heads: i32,
    local_key_value_heads: i32,
) -> LocalModelLayout {
    let architecture = llama::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let static_modules = architecture.static_modules();
    let mut groups = llama::static_parallel_parameter_groups::<ReferenceBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
        "model",
    )
    .unwrap();
    for layer in 0..args.num_hidden_layers as usize {
        let block = TransformerBlock::<ReferenceBackend>::new(args, layer, &()).unwrap();
        groups.extend(
            llama::layer_parallel_parameter_groups::<ReferenceBackend>(&block, args, layer)
                .unwrap(),
        );
    }

    let query_width = usize::try_from(local_query_heads * args.head_dim).unwrap();
    let key_value_width = usize::try_from(local_key_value_heads * args.head_dim).unwrap();
    let mut layout = LocalModelLayout::default();
    for group in groups {
        for member in group.members() {
            let target = member.target();
            let mut local_shape = member.global_shape().to_vec();
            let (placement, logical_range) = if group.role()
                == eredu_runtime::ParameterRole::Vocabulary
            {
                local_shape[0] = vocabulary.len();
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: vocabulary.start,
                        end: vocabulary.end,
                    },
                    Some(vocabulary.clone()),
                )
            } else if target.contains(".self_attn.q_proj") {
                local_shape[0] = query_width;
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: 0,
                        end: query_width,
                    },
                    Some(0..usize::try_from(local_query_heads).unwrap()),
                )
            } else if target.contains(".self_attn.k_proj") || target.contains(".self_attn.v_proj") {
                local_shape[0] = key_value_width;
                (
                    TensorPlacement::Range {
                        axis: 0,
                        start: 0,
                        end: key_value_width,
                    },
                    Some(0..usize::try_from(local_key_value_heads).unwrap()),
                )
            } else {
                (
                    TensorPlacement::Replicated,
                    group.partition_units().map(|units| 0..units),
                )
            };
            layout.insert(
                target.to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local_shape,
                    placement,
                    group.partition_units(),
                    logical_range,
                    false,
                ),
            );
        }
    }
    layout
}

fn replicated_parallel_layout(groups: &[ParameterGroupSpec]) -> LocalModelLayout {
    let mut layout = LocalModelLayout::default();
    for group in groups {
        for member in group.members() {
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    member.global_shape().to_vec(),
                    TensorPlacement::Replicated,
                    group.partition_units(),
                    group.partition_units().map(|units| 0..units),
                    false,
                ),
            );
        }
    }
    layout
}

fn assert_geometry_identity_error<T>(result: Result<T, Error>, family: &str) {
    match result {
        Err(error) => assert!(
            error
                .to_string()
                .contains("different normalized configuration"),
            "unexpected {family} geometry error: {error}"
        ),
        Ok(_) => panic!("{family} accepted geometry derived from a different configuration"),
    }
}

#[test]
fn shared_decoder_parallel_geometry_rejects_cross_config_reuse() {
    let llama_args = tiny_args();
    let llama_layout = llama_parallel_layout(&llama_args, 0..32, 2, 1);
    let llama_geometry = llama::local_geometry(&llama_args, &llama_layout).unwrap();
    let mut changed_llama = llama_args;
    changed_llama.rope_theta = 20_000.0;
    assert_geometry_identity_error(
        llama::LayeredModel::<ReferenceBackend>::new_parallel(changed_llama, llama_geometry, &()),
        "Llama",
    );

    let qwen_args = qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let qwen_model = qwen::LayeredModel::<ReferenceBackend>::new(qwen_args.clone(), &()).unwrap();
    let mut qwen_groups = qwen::static_parallel_parameter_groups::<ReferenceBackend>(
        &qwen_model.static_modules().embeddings,
        &qwen_model.static_modules().norm,
        qwen_model.static_modules().lm_head.as_ref(),
        &qwen_args.parameter_root,
    )
    .unwrap();
    let qwen_unit = qwen_model.construct_unit(0, &()).unwrap();
    qwen_groups.extend(qwen::layer_parallel_parameter_groups(&qwen_unit, &qwen_args, 0).unwrap());
    let qwen_geometry =
        qwen::local_geometry(&qwen_args, &replicated_parallel_layout(&qwen_groups)).unwrap();
    let mut changed_qwen = qwen_args;
    changed_qwen.rms_norm_eps = 0.00002;
    assert_geometry_identity_error(
        qwen::LayeredModel::<ReferenceBackend>::new_parallel(changed_qwen, qwen_geometry, &()),
        "Qwen",
    );

    let gpt_args = gpt_oss::model_args_from_config_value(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 0.00001,
        "sliding_window": 4,
        "max_position_embeddings": 64,
        "rope_theta": 150000.0,
        "layer_types": ["full_attention"],
        "quantization_config": {"quant_method": "mxfp4"},
        "swiglu_limit": 7.0
    }))
    .unwrap();
    let error = match gpt_oss::LayeredModel::<ReferenceBackend>::new(gpt_args, &()) {
        Ok(_) => panic!("GPT-OSS unexpectedly composed without attention-sink support"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("requires unsupported backend operators: attention_sinks"),
        "unexpected capability error: {error}"
    );
}

struct StaticTopologyCollector(BTreeMap<String, Vec<String>>);

impl StaticParameterVisitor<ReferenceBackend> for StaticTopologyCollector {
    type Error = Error;

    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<ReferenceTensor>,
    {
        let parameters = validate_parameter_topology(module)
            .map_err(Error::backend)?
            .into_iter()
            .map(|metadata| metadata.id.as_str().to_owned())
            .collect();
        assert!(self.0.insert(role.to_owned(), parameters).is_none());
        Ok(())
    }
}

#[test]
fn shared_decoder_exposes_architecture_owned_static_parameter_bindings() {
    let args = qwen::model_args_from_config_value(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let model = qwen::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let mut visitor = StaticTopologyCollector(BTreeMap::new());
    model.visit_static_parameters(&mut visitor).unwrap();

    assert_eq!(
        visitor.0,
        BTreeMap::from([
            ("embedding".into(), vec!["model.embed_tokens.weight".into()]),
            ("norm".into(), vec!["model.norm.weight".into()]),
            ("output".into(), vec!["lm_head.weight".into()]),
        ])
    );
}

#[test]
fn neutral_llama_parallel_geometry_owns_uneven_tied_and_untied_vocabularies() {
    let mut tied_args = tiny_args();
    tied_args.vocab_size = 7;
    let tied_layout = llama_parallel_layout(&tied_args, 4..7, 2, 1);
    let tied_geometry = llama::local_geometry(&tied_args, &tied_layout).unwrap();
    assert_eq!(tied_geometry.embedding_range().global_vocabulary, 7);
    assert_eq!(tied_geometry.embedding_range().local, 4..7);
    assert_eq!(tied_geometry.output_range(), None);

    let replicated: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new(tied_args.clone(), &()).unwrap();
    let parallel: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new_parallel(tied_args.clone(), tied_geometry, &()).unwrap();
    assert!(replicated.parallel_geometry().is_none());
    assert!(parallel.parallel_geometry().is_some());
    assert_eq!(parallel.static_modules().embeddings.weight.0, [3, 8]);
    assert!(parallel.static_modules().lm_head.is_none());

    let mut untied_args = tied_args;
    untied_args.tie_word_embeddings = false;
    let untied_layout = llama_parallel_layout(&untied_args, 4..7, 2, 1);
    let untied_geometry = llama::local_geometry(&untied_args, &untied_layout).unwrap();
    assert_eq!(
        untied_geometry
            .output_range()
            .map(|range| range.local.clone()),
        Some(4..7)
    );
    let untied: llama::LayeredModel<ReferenceBackend> =
        llama::LayeredModel::new_parallel(untied_args, untied_geometry, &()).unwrap();
    assert_eq!(untied.static_modules().embeddings.weight.0, [3, 8]);
    assert_eq!(
        untied.static_modules().lm_head.as_ref().unwrap().weight.0,
        [3, 8]
    );
}

#[test]
fn neutral_llama_parallel_geometry_drives_local_gqa_and_cache_shape() {
    let mut args = tiny_args();
    args.num_attention_heads = 4;
    args.num_key_value_heads = 2;
    args.head_dim = 2;
    let layout = llama_parallel_layout(&args, 0..16, 2, 1);
    let geometry = llama::local_geometry(&args, &layout).unwrap();

    assert!(geometry
        .blocks()
        .iter()
        .all(|block| { block.num_attention_heads == 2 && block.num_key_value_heads == 1 }));
    for layer in 0..args.num_hidden_layers as usize {
        match geometry.state_layout().layer(layer).unwrap() {
            eredu_core::cache::LayerCachePolicy::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            } => {
                assert_eq!(num_key_value_heads.get(), 1);
                assert_eq!(head_dim.get(), 2);
            }
            policy => panic!("unexpected local Llama cache policy {policy:?}"),
        }
    }

    let architecture =
        llama::LayeredModel::<ReferenceBackend>::new_parallel(args, geometry, &()).unwrap();
    let block = <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        DeviceState<ReferenceBackend, ReferenceCache>,
    >>::build_unit(&architecture, 0, 0, &())
    .unwrap();
    assert_eq!(block.self_attention.query_heads, 2);
    assert_eq!(block.self_attention.key_value_heads, 1);
}

#[test]
fn neutral_llama_tensor_parallel_size_one_matches_replicated_lifecycle() {
    let args = tiny_args();
    let state = || {
        DeviceState::<ReferenceBackend, ReferenceCache>::create(
            llama::state_layout(&args).unwrap(),
            |_, policy| {
                Ok::<_, std::convert::Infallible>(ReferenceCache {
                    offset: 0,
                    window: policy
                        .attention()
                        .and_then(|attention| attention.window())
                        .map(|window| window.get() as i32),
                    resets: 0,
                })
            },
        )
        .unwrap()
    };

    let replicated = llama::LayeredModel::<ReferenceBackend>::new(args.clone(), &()).unwrap();
    let units = (0..args.num_hidden_layers as usize)
        .map(|index| {
            <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            >>::build_unit(&replicated, 0, index, &())
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut replicated = LayerwiseRuntime::new(replicated, ResidentUnitWindow::new(units));
    let mut replicated_state = state();

    let layout = llama_parallel_layout(
        &args,
        0..args.vocab_size as usize,
        args.num_attention_heads,
        args.num_key_value_heads,
    );
    let geometry = llama::local_geometry(&args, &layout).unwrap();
    let parallel =
        llama::LayeredModel::<ReferenceBackend>::new_parallel(args.clone(), geometry, &()).unwrap();
    let units = (0..parallel.args().num_hidden_layers as usize)
        .map(|index| {
            <llama::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                DeviceState<ReferenceBackend, ReferenceCache>,
            >>::build_unit(&parallel, 0, index, &())
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut parallel = LayerwiseRuntime::new(parallel, ResidentUnitWindow::new(units));
    let mut parallel_state = state();
    let tokens = ReferenceTensor(vec![1, 3]);

    let replicated_logits = replicated
        .forward(
            LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut replicated_state,
            &(),
        )
        .unwrap();
    let parallel_logits = parallel
        .forward_parallel(
            LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            &mut parallel_state,
            &(),
            &(),
        )
        .unwrap();

    assert_eq!(parallel_logits, replicated_logits);
    assert_eq!(parallel_state.layer(0).unwrap().offset(), 3);
    assert_eq!(parallel_state.layer(1).unwrap().offset(), 3);
}

#[test]
fn neutral_llama_local_geometry_rejects_bad_vocabulary_companion_selection() {
    let mut args = tiny_args();
    args.vocab_size = 7;
    let mut layout = llama_parallel_layout(&args, 4..7, 2, 1);
    layout.insert(
        "model.embed_tokens.scales".into(),
        LocalTensorLayout::new(
            "model.embed_tokens",
            eredu_runtime::ParameterRole::Vocabulary,
            vec![7, 1],
            vec![3, 1],
            TensorPlacement::Range {
                axis: 0,
                start: 0,
                end: 3,
            },
            None,
            Some(0..3),
            false,
        ),
    );
    let error = llama::local_geometry(&args, &layout).unwrap_err();
    assert!(error
        .to_string()
        .contains("inconsistent companion selections"));

    let mut malformed = llama_parallel_layout(&args, 4..7, 2, 1);
    malformed.insert(
        "model.embed_tokens.weight".into(),
        LocalTensorLayout::new(
            "model.embed_tokens",
            eredu_runtime::ParameterRole::Vocabulary,
            vec![7, 8],
            vec![7, 3],
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 3,
            },
            None,
            Some(0..3),
            false,
        ),
    );
    let error = llama::local_geometry(&args, &malformed).unwrap_err();
    assert!(error.to_string().contains("non-row placement"));
}

struct ProjectionLayoutConfig {
    args: ModelArgs,
    fused: bool,
    empty_field: bool,
    alternate_fields: bool,
}

impl decoder::Config for ProjectionLayoutConfig {
    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "reference_projection_layout_decoder",
            [
                (
                    "base",
                    decoder::Config::architecture_fingerprint(&self.args),
                ),
                ("fused", self.fused.to_string()),
                ("empty_field", self.empty_field.to_string()),
                ("alternate_fields", self.alternate_fields.to_string()),
                ("attention_bias", "false".into()),
                ("mlp_bias", "false".into()),
                ("weight_quantization", "dense".into()),
            ],
        )
    }

    fn validate_config(&self) -> Result<(), Error> {
        Ok(())
    }

    fn block_parameter_fields(&self) -> decoder::BlockParameterFields<'_> {
        if self.alternate_fields {
            decoder::BlockParameterFields {
                attention_output: "out_proj",
                feed_forward: "gating",
                feed_forward_output: "linear_out",
                input_norm: "norm1",
                post_attention_norm: "norm2",
                ..decoder::BlockParameterFields::default()
            }
        } else {
            decoder::BlockParameterFields::default()
        }
    }

    fn hidden_size(&self) -> i32 {
        self.args.hidden_size
    }

    fn num_hidden_layers(&self) -> i32 {
        self.args.num_hidden_layers
    }

    fn intermediate_size(&self) -> i32 {
        self.args.intermediate_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.args.num_attention_heads
    }

    fn num_key_value_heads(&self) -> i32 {
        self.args.num_key_value_heads
    }

    fn head_dim(&self) -> i32 {
        self.args.head_dim
    }

    fn rms_norm_epsilon(&self) -> f32 {
        self.args.rms_norm_eps
    }

    fn vocabulary_size(&self) -> i32 {
        self.args.vocab_size
    }

    fn attention_bias(&self, _: decoder::AttentionProjection) -> bool {
        false
    }

    fn attention_projection_layout(&self) -> AttentionProjectionLayout<'_> {
        if self.fused {
            AttentionProjectionLayout::Fused {
                field: if self.empty_field { "" } else { "in_proj" },
            }
        } else {
            AttentionProjectionLayout::Split
        }
    }

    fn mlp_bias(&self) -> bool {
        false
    }

    fn gated_projection_layout(&self) -> GatedProjectionLayout<'_> {
        if self.fused {
            GatedProjectionLayout::Fused {
                field: if self.empty_field {
                    ""
                } else if self.alternate_fields {
                    "linear_in"
                } else {
                    "gate_up_proj"
                },
            }
        } else {
            GatedProjectionLayout::Split
        }
    }

    fn tie_word_embeddings(&self) -> bool {
        self.args.tie_word_embeddings
    }

    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.args.attention_schedule
    }

    fn weight_quantization(&self, _: &str) -> Option<eredu_checkpoint::WeightQuantization> {
        None
    }

    fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
        RotarySpec {
            dimensions,
            base: self.args.rope_theta,
            traditional: self.args.rope_traditional,
            algorithm: eredu_nn::RotaryAlgorithm::Default,
        }
    }
}

#[test]
fn tied_vocabulary_parallel_projection_preserves_embedding_storage_and_global_logits() {
    let spec = EmbeddingSpec {
        vocabulary: 7,
        dimensions: 4,
        weight: eredu_nn::ParameterSpec::trainable("model.embed_tokens.weight").unwrap(),
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_nn::LinearFormat::Dense).unwrap(),
    };
    let mut embedding = ReferenceBackend::vocabulary_parallel_embedding(
        spec,
        VocabularyParallelRange {
            global_vocabulary: 7,
            local: 0..4,
        },
        &(),
    )
    .unwrap();
    assert_eq!(embedding.weight.0, [4, 4]);

    let hidden = ReferenceTensor(vec![1, 2, 4]);
    let logits =
        ReferenceBackend::vocabulary_parallel_embedding_project(&mut embedding, &hidden, &(), &())
            .unwrap();

    assert_eq!(embedding.weight.0, [4, 4]);
    assert_eq!(logits.0, [1, 2, 7]);
}

#[test]
fn fused_and_split_decoder_blocks_publish_equivalent_projection_geometry() {
    let split = TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: false,
            empty_field: false,
            alternate_fields: false,
        },
        0,
        &(),
    )
    .unwrap();
    let fused = TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: true,
            empty_field: false,
            alternate_fields: false,
        },
        0,
        &(),
    )
    .unwrap();
    let split = topology(&split);
    let fused = topology(&fused);
    let elements = |topology: &[(String, Vec<usize>)]| {
        topology
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>()
    };
    assert_eq!(elements(&split), elements(&fused));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.q_proj.weight") && shape == &[8, 8]));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.k_proj.weight") && shape == &[4, 8]));
    assert!(split
        .iter()
        .any(|(name, shape)| name.ends_with("mlp.gate_proj.weight") && shape == &[16, 8]));
    assert!(fused
        .iter()
        .any(|(name, shape)| name.ends_with("self_attn.in_proj.weight") && shape == &[16, 8]));
    assert!(fused
        .iter()
        .any(|(name, shape)| name.ends_with("mlp.gate_up_proj.weight") && shape == &[32, 8]));
    assert!(!fused.iter().any(|(name, _)| {
        name.ends_with("q_proj.weight")
            || name.ends_with("k_proj.weight")
            || name.ends_with("gate_proj.weight")
    }));
}

#[test]
fn alternate_block_fields_drive_parameter_topology_and_parallel_groups() {
    let config = ProjectionLayoutConfig {
        args: tiny_args(),
        fused: true,
        empty_field: false,
        alternate_fields: true,
    };
    let block = TransformerBlock::<ReferenceBackend>::new(&config, 0, &()).unwrap();
    let expected = std::collections::BTreeSet::from([
        "model.layers.0.gating.linear_in.weight".to_string(),
        "model.layers.0.gating.linear_out.weight".to_string(),
        "model.layers.0.norm1.weight".to_string(),
        "model.layers.0.norm2.weight".to_string(),
        "model.layers.0.self_attn.in_proj.weight".to_string(),
        "model.layers.0.self_attn.out_proj.weight".to_string(),
    ]);
    let topology_names = topology(&block)
        .into_iter()
        .map(|(name, _)| name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(topology_names, expected);

    let groups = decoder::layer_parallel_parameter_groups(&block, &config, 0).unwrap();
    let group_names = groups
        .iter()
        .map(|group| group.logical_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        group_names,
        std::collections::BTreeSet::from([
            "model.layers.0.gating.projections",
            "model.layers.0.norm1",
            "model.layers.0.norm2",
            "model.layers.0.self_attn.projections",
        ])
    );
    let member_names = groups
        .iter()
        .flat_map(|group| group.members())
        .map(|member| member.target().to_string())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(member_names, expected);
}

fn tiny_moshi_config() -> moshi::MoshiConfig {
    moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 4,
            "dep_q": 3,
            "generated_audio_codebooks": 2,
            "card": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 2,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 2, 1]
        }"#,
    )
    .unwrap()
}

fn minimal_moshi_config() -> moshi::MoshiConfig {
    moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 2,
            "dep_q": 1,
            "generated_audio_codebooks": 1,
            "card": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 1,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1]
        }"#,
    )
    .unwrap()
}

type ReferenceMoshiState = DeviceState<ReferenceBackend, ReferenceCache>;

fn reference_moshi_state(config: &moshi::MoshiConfig) -> ReferenceMoshiState {
    let layout = moshi::state_layout(config).unwrap();
    DeviceState::create(layout, |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
            resets: 0,
        })
    })
    .unwrap()
}

fn reference_decision_driver(
    retain_diagnostics: bool,
) -> SequentialDecisionDriver<ReferenceBackend, ReferenceSampler> {
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Sample,
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        retain_diagnostics,
        true,
    )
    .unwrap();
    SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 4], vec![0.0; 4], Some(0)).unwrap()
}

fn execute_reference_moshi_frame(
    config: &moshi::MoshiConfig,
    directives: Vec<PredictionDirective<ReferenceTensor>>,
    retain_diagnostics: bool,
    allow_tail_skip: bool,
    random: Option<i32>,
) -> (
    ReferenceTensor,
    moshi::ForwardContext<ReferenceTensor>,
    SequentialDecisionDriver<ReferenceBackend, ReferenceSampler>,
    ReferenceMoshiState,
    ReferenceTrace,
) {
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let decision_count = architecture.decision_count();
    assert_eq!(directives.len(), decision_count);
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut state = reference_moshi_state(config);
    let plan =
        SequentialDecisionPlan::new(directives, retain_diagnostics, allow_tail_skip).unwrap();
    let mut driver = SequentialDecisionDriver::new(
        plan,
        vec![ReferenceSampler; decision_count],
        vec![0.0; decision_count],
        random,
    )
    .unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = (0..config.frame_schedule().total_audio_codebooks())
        .map(|_| ReferenceTensor(vec![1, 1]))
        .collect::<Vec<_>>();
    let audio = audio_values.iter().collect::<Vec<_>>();
    clear_reference_trace();
    let mut boundary = moshi::DecisionBoundary::new(config).unwrap();
    let (logits, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    driver.finish().unwrap();
    (logits, forward, driver, state, reference_trace())
}

fn replicated_moshi_parallel_layout(config: &moshi::MoshiConfig) -> LocalModelLayout {
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut groups = architecture.static_parameter_groups().unwrap();
    for group in 0..2 {
        let count = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
            ReferenceBackend,
            ReferenceMoshiState,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            let unit = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                ReferenceBackend,
                ReferenceMoshiState,
            >>::build_unit(&architecture, group, index, &())
            .unwrap();
            groups.extend(moshi::unit_parameter_groups(&unit, config, group, index).unwrap());
        }
    }
    let mut layout = LocalModelLayout::default();
    for group in groups {
        let logical_range = group.partition_units().map(|units| 0..units);
        for member in group.members() {
            layout.insert(
                member.target().to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    member.global_shape().to_vec(),
                    TensorPlacement::Replicated,
                    group.partition_units(),
                    logical_range.clone(),
                    false,
                ),
            );
        }
    }
    layout
}

#[test]
fn one_portable_moshi_model_runs_replicated_and_parallel_lifecycles() {
    let config = minimal_moshi_config();
    let replicated = execute_reference_moshi_frame(
        &config,
        vec![
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
        None,
    );

    let layout = replicated_moshi_parallel_layout(&config);
    let geometry = moshi::local_geometry(&config, &layout, std::iter::empty()).unwrap();
    let architecture =
        moshi::LayeredModel::<ReferenceBackend>::new_parallel(config.clone(), geometry, &())
            .unwrap();
    let mut units = Vec::new();
    for group in 0..2 {
        let count = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
            ReferenceBackend,
            ReferenceMoshiState,
        >>::group_unit_count(&architecture, group)
        .unwrap();
        for index in 0..count {
            units.push(
                <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
                    ReferenceBackend,
                    ReferenceMoshiState,
                >>::build_unit(&architecture, group, index, &())
                .unwrap(),
            );
        }
    }
    let mut runtime = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let mut state = reference_moshi_state(&config);
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 2], vec![0.0; 2], None).unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = (0..config.frame_schedule().total_audio_codebooks())
        .map(|_| ReferenceTensor(vec![1, 1]))
        .collect::<Vec<_>>();
    let audio = audio_values.iter().collect::<Vec<_>>();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let (parallel_logits, _) = {
        let mut traversal = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_parallel_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &(),
                &mut traversal,
            )
            .unwrap()
    };
    driver.finish().unwrap();

    assert_eq!(parallel_logits, replicated.0);
    assert_eq!(driver.decisions(), replicated.2.decisions());
    assert_eq!(
        state.layout(),
        &runtime.architecture().state_layout().unwrap()
    );
}

#[test]
fn moshi_parallel_geometry_rejects_policy_compatible_cross_config_reuse() {
    let config = minimal_moshi_config();
    let layout = replicated_moshi_parallel_layout(&config);
    let geometry = moshi::local_geometry(&config, &layout, std::iter::empty()).unwrap();
    let changed = moshi::MoshiConfig::from_json(
        r#"{
            "model_type": "moshi",
            "dim": 32,
            "text_card": 101,
            "n_q": 2,
            "dep_q": 1,
            "generated_audio_codebooks": 1,
            "card": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dim_feedforward": 48,
            "causal": true,
            "context": 7,
            "max_period": 20000.0,
            "positional_embedding": "rope",
            "depformer_dim": 24,
            "depformer_dim_feedforward": 36,
            "depformer_num_heads": 4,
            "depformer_num_layers": 1,
            "depformer_context": 3,
            "depformer_max_period": 10000.0,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1]
        }"#,
    )
    .unwrap();
    match moshi::LayeredModel::<ReferenceBackend>::new_parallel(changed, geometry, &()) {
        Err(error) => assert!(
            error
                .to_string()
                .contains("different normalized configuration"),
            "unexpected Moshi geometry error: {error}"
        ),
        Ok(_) => panic!("Moshi accepted geometry derived from a different configuration"),
    }
}

#[test]
fn tiny_moshi_executes_one_temporal_block_and_one_depth_slice_with_exact_logit_geometry() {
    let config = minimal_moshi_config();
    let forced_text = ReferenceTensor(vec![1, 1]);
    let forced_audio = ReferenceTensor(vec![1, 1]);
    let (text_logits, forward, driver, state, trace) = execute_reference_moshi_frame(
        &config,
        vec![
            PredictionDirective::Force(forced_text),
            PredictionDirective::Force(forced_audio.clone()),
        ],
        true,
        true,
        None,
    );

    assert_eq!(driver.plan().mode(), SequentialDecisionMode::TeacherForced);
    assert_eq!(text_logits, ReferenceTensor(vec![1, 1, 101]));
    assert_eq!(
        driver
            .diagnostics()
            .iter()
            .map(|diagnostic| (diagnostic.prediction(), diagnostic.logits().clone()))
            .collect::<Vec<_>>(),
        [
            (0, ReferenceTensor(vec![1, 1, 101])),
            (1, ReferenceTensor(vec![1, 1, 64])),
        ]
    );
    assert_eq!(forward.previous_depth_token(), Some(&forced_audio));
    assert_eq!(state.as_ref()[0].offset, 1);
    assert_eq!(state.as_ref()[1].offset, 1);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 1);
    assert!(trace
        .linear_outputs
        .contains(&("text_linear.weight".into(), vec![1, 1, 101])));
    assert!(trace.linear_outputs.contains(&(
        "depformer.slices.0.linear_out.weight".into(),
        vec![1, 1, 64]
    )));
}

#[test]
fn moshi_frame_decisions_cover_greedy_seeded_partial_forced_and_tail_skip() {
    let config = tiny_moshi_config();
    let count = config.frame_schedule().depth_audio_codebooks() + 1;
    let sampled = vec![PredictionDirective::Sample; count];

    let (_, _, greedy, _, _) =
        execute_reference_moshi_frame(&config, sampled.clone(), true, true, None);
    assert_eq!(greedy.plan().mode(), SequentialDecisionMode::Autoregressive);
    assert!(greedy.random_state().is_none());
    assert!(greedy
        .decisions()
        .iter()
        .all(|decision| decision.source() == SequentialDecisionSource::Sampled));

    let (_, _, seeded, _, _) =
        execute_reference_moshi_frame(&config, sampled, true, true, Some(17));
    assert_eq!(seeded.random_state(), Some(&21));
    assert_eq!(seeded.diagnostics().len(), count);

    let partial = vec![
        PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        PredictionDirective::Sample,
        PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        PredictionDirective::Sample,
    ];
    let (_, _, partial, _, _) =
        execute_reference_moshi_frame(&config, partial, true, true, Some(4));
    assert_eq!(
        partial.plan().mode(),
        SequentialDecisionMode::PartiallyForced
    );
    assert_eq!(partial.random_state(), Some(&6));
    assert_eq!(
        partial
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Sampled,
        ]
    );

    let forced = (0..count)
        .map(|_| PredictionDirective::Force(ReferenceTensor(vec![1, 1])))
        .collect::<Vec<_>>();
    let (_, _, forced, state, _) =
        execute_reference_moshi_frame(&config, forced, false, true, None);
    assert_eq!(forced.plan().mode(), SequentialDecisionMode::TeacherForced);
    assert_eq!(
        forced
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );
    assert_eq!(state.as_ref()[2].offset, 0);
    assert_eq!(state.as_ref()[3].offset, 0);
    assert_eq!(state.as_ref()[2].resets, 1);
    assert_eq!(state.as_ref()[3].resets, 1);
}

#[test]
fn moshi_depth_codebooks_receive_the_immediately_preceding_decision() {
    let config = tiny_moshi_config();
    let symbolic_tokens = [
        ReferenceTensor(vec![1, 2]),
        ReferenceTensor(vec![1, 3]),
        ReferenceTensor(vec![1, 4]),
        ReferenceTensor(vec![1, 5]),
    ];
    let directives = symbolic_tokens
        .iter()
        .cloned()
        .map(PredictionDirective::Force)
        .collect::<Vec<_>>();
    let (_, forward, _, _, trace) =
        execute_reference_moshi_frame(&config, directives, true, false, None);
    let depth_lookups = trace
        .embedding_lookups
        .iter()
        .filter(|(name, _)| name.starts_with("depformer.slices."))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        depth_lookups,
        [
            ("depformer.slices.0.emb.weight".into(), vec![1, 2]),
            ("depformer.slices.1.emb.weight".into(), vec![1, 3]),
            ("depformer.slices.2.emb.weight".into(), vec![1, 4]),
        ]
    );
    assert_eq!(forward.previous_depth_token(), Some(&symbolic_tokens[3]));
}

#[test]
fn moshi_depth_resets_once_per_frame_and_temporal_rope_uses_absolute_offset() {
    let config = minimal_moshi_config();
    assert_eq!(config.temporal().context(), 7);
    assert_eq!(config.temporal().attention_window(), 8);
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut state = reference_moshi_state(&config);
    assert_eq!(state.as_ref()[0].window, Some(8));
    assert_eq!(state.as_ref()[1].window, Some(3));

    let prefill_text = ReferenceTensor(vec![1, 3]);
    let prefill_audio_values = [ReferenceTensor(vec![1, 3]), ReferenceTensor(vec![1, 3])];
    let prefill_audio = prefill_audio_values.iter().collect::<Vec<_>>();
    let prefill_plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut prefill_driver =
        SequentialDecisionDriver::new(prefill_plan, vec![ReferenceSampler; 2], vec![0.0; 2], None)
            .unwrap();
    clear_reference_trace();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    {
        let mut hook = SequentialDecisionTraversal::new(&mut prefill_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &prefill_text,
                    audio: &prefill_audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap();
    }
    prefill_driver.finish().unwrap();
    let prefill_trace = reference_trace();
    assert_eq!(prefill_trace.rotary_offsets, [0, 0]);
    assert_eq!(prefill_trace.sliding_attention, [(8, 0), (3, 0)]);
    assert_eq!(prefill_trace.causal_masks, [(3, 0, None), (3, 0, None)]);
    assert_eq!(state.as_ref()[0].offset, 3);
    assert_eq!(state.as_ref()[1].offset, 3);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 1);

    let decode_text = ReferenceTensor(vec![1, 1]);
    let decode_audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 1])];
    let decode_audio = decode_audio_values.iter().collect::<Vec<_>>();
    let decode_plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        true,
    )
    .unwrap();
    let mut decode_driver =
        SequentialDecisionDriver::new(decode_plan, vec![ReferenceSampler; 2], vec![0.0; 2], None)
            .unwrap();
    clear_reference_trace();
    {
        let mut hook = SequentialDecisionTraversal::new(&mut decode_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &decode_text,
                    audio: &decode_audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap();
    }
    decode_driver.finish().unwrap();
    let decode_trace = reference_trace();
    assert_eq!(decode_trace.rotary_offsets, [3, 3]);
    assert!(decode_trace.sliding_attention.is_empty());
    assert!(decode_trace.causal_masks.is_empty());
    assert_eq!(state.as_ref()[0].offset, 4);
    assert_eq!(state.as_ref()[1].offset, 1);
    assert_eq!(state.as_ref()[0].resets, 0);
    assert_eq!(state.as_ref()[1].resets, 2);
}

#[test]
fn moshi_rejects_mismatched_cache_layout_and_depth_block_drift() {
    let config = minimal_moshi_config();
    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 1])];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let malformed_audio_values = [ReferenceTensor(vec![1, 1]), ReferenceTensor(vec![1, 2])];
    let malformed_audio = malformed_audio_values.iter().collect::<Vec<_>>();
    let mut valid_state = reference_moshi_state(&config);
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &malformed_audio,
                mask: None,
            },
            &mut valid_state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("token shape"));
    assert!(valid_state.as_ref().iter().all(|cache| cache.resets == 0));

    let mut wrong_state = reference_moshi_state(&tiny_moshi_config());
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &audio,
                mask: None,
            },
            &mut wrong_state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("state layout mismatch"));
    assert!(wrong_state.as_ref().iter().all(|cache| cache.resets == 0));

    let moshi::Unit::Depth(depth) = &mut runtime.units_mut()[1][0] else {
        panic!("depth group owns depth units");
    };
    depth.blocks.pop();
    let mut state = reference_moshi_state(&config);
    let plan = SequentialDecisionPlan::new(
        [
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
            PredictionDirective::Force(ReferenceTensor(vec![1, 1])),
        ],
        true,
        false,
    )
    .unwrap();
    let mut driver =
        SequentialDecisionDriver::new(plan, vec![ReferenceSampler; 2], vec![0.0; 2], None).unwrap();
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let error = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .err()
            .expect("drifted depth block count must fail")
    };
    assert!(error.to_string().contains("depth block count drifted"));
}

#[test]
fn portable_moshi_topology_state_and_decision_order_are_backend_independent() {
    let config = tiny_moshi_config();
    let layout = moshi::state_layout(&config).unwrap();
    assert_eq!(layout.layers().len(), 4);
    assert_eq!(layout.segments().len(), 2);
    assert_eq!(layout.segments()[0].id().as_str(), "temporal");
    assert_eq!(layout.segments()[0].layers(), 0..2);
    assert_eq!(layout.segments()[1].id().as_str(), "depth");
    assert_eq!(layout.segments()[1].layers(), 2..4);

    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    assert_eq!(runtime.units().len(), 2);
    assert_eq!(runtime.units()[0].len(), 2);
    assert_eq!(runtime.units()[1].len(), 3);
    assert!(runtime.units()[0].iter().all(|unit| match unit {
        moshi::Unit::Temporal(block) => block.self_attention.rotary.is_some(),
        moshi::Unit::Depth(_) => false,
    }));
    assert!(runtime.units()[1].iter().all(|unit| match unit {
        moshi::Unit::Depth(slice) => slice
            .blocks
            .iter()
            .all(|block| block.self_attention.rotary.is_none()),
        moshi::Unit::Temporal(_) => false,
    }));
    let graph = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        ReferenceMoshiState,
    >>::execution_graph(runtime.architecture())
    .unwrap();
    assert_eq!(graph.execution_order(), &[0, 1]);
    assert_eq!(
        graph.groups()[1].dependencies(),
        &["temporal_transformer".to_string()]
    );
    let retained = <moshi::LayeredModel<ReferenceBackend> as LayeredArchitecture<
        ReferenceBackend,
        ReferenceMoshiState,
    >>::retained_state_ordinals(runtime.architecture(), 1, 2, 4);
    assert_eq!(retained, 2..4);

    let mut names = topology(runtime.architecture().static_modules());
    for group in runtime.units() {
        for unit in group {
            names.extend(topology(unit));
        }
    }
    let names = names
        .into_iter()
        .map(|(name, _)| name)
        .collect::<std::collections::BTreeSet<_>>();
    for expected in [
        "text_emb.weight",
        "audio_embs.3.weight",
        "out_norm.weight",
        "text_linear.weight",
        "transformer.layers.0.norm1.weight",
        "transformer.layers.0.self_attn.in_proj.weight",
        "transformer.layers.0.self_attn.out_proj.weight",
        "transformer.layers.0.gating.linear_in.weight",
        "transformer.layers.0.gating.linear_out.weight",
        "depformer.slices.2.emb.weight",
        "depformer.slices.2.linear_in.weight",
        "depformer.slices.2.linear_out.weight",
        "depformer.slices.2.transformer.layers.1.norm2.weight",
    ] {
        assert!(names.contains(expected), "missing {expected}");
    }
    assert!(!names.iter().any(|name| {
        name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.contains(".mlp.")
            || name.contains(".o_proj.")
    }));

    let static_groups = runtime.architecture().static_parameter_groups().unwrap();
    assert_eq!(static_groups.len(), 7);
    let depth_groups = moshi::unit_parameter_groups(&runtime.units()[1][0], &config, 1, 0).unwrap();
    assert!(depth_groups
        .iter()
        .any(|group| group.logical_name() == "depformer.slices.0.linear_in"));
    assert!(depth_groups
        .iter()
        .any(|group| group.logical_name() == "depformer.slices.0.transformer.layers.1.norm2"));
    for name in ["depformer.slices.0.emb", "depformer.slices.0.linear_out"] {
        let group = depth_groups
            .iter()
            .find(|group| group.logical_name() == name)
            .unwrap();
        assert_eq!(group.role(), eredu_runtime::ParameterRole::Vocabulary);
        assert!(group.members().iter().all(|member| matches!(
            member.sharding(),
            eredu_runtime::MemberSharding::Balanced { axis: 0 }
        )));
    }
    let linear_in = depth_groups
        .iter()
        .find(|group| group.logical_name() == "depformer.slices.0.linear_in")
        .unwrap();
    assert_eq!(linear_in.role(), eredu_runtime::ParameterRole::Replicated);
    assert!(linear_in
        .members()
        .iter()
        .all(|member| member.sharding() == &eredu_runtime::MemberSharding::Replicated));

    let text = ReferenceTensor(vec![1, 1]);
    let audio_values = [
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
        ReferenceTensor(vec![1, 1]),
    ];
    let audio = audio_values.iter().collect::<Vec<_>>();
    let mut state = reference_moshi_state(&config);
    state.as_mut()[2].offset = 9;
    state.as_mut()[3].offset = 9;
    let error = runtime
        .forward(
            moshi::Input {
                text: &text,
                audio: &audio[..3],
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("3 audio codebooks, expected 4"));
    assert_eq!(state.as_ref()[2].offset, 9);
    assert_eq!(state.as_ref()[3].offset, 9);

    let mut driver = reference_decision_driver(true);
    let mut boundary = moshi::DecisionBoundary::new(&config).unwrap();
    let (_, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    driver.finish().unwrap();
    assert_eq!(driver.diagnostics().len(), 4);
    assert_eq!(driver.random_state(), Some(&1));
    assert_eq!(
        driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Forced,
            SequentialDecisionSource::Forced,
        ]
    );
    assert_eq!(
        forward.previous_depth_token(),
        Some(&ReferenceTensor(vec![1, 1]))
    );
    assert_eq!(state.as_ref()[2].offset, 3);
    assert_eq!(state.as_ref()[3].offset, 3);
    assert_eq!(state.as_ref()[2].resets, 1);
    assert_eq!(state.as_ref()[3].resets, 1);

    let architecture = moshi::LayeredModel::<ReferenceBackend>::new(config.clone(), &()).unwrap();
    let mut runtime =
        ResidentRuntime::<_, ReferenceBackend, ReferenceMoshiState>::new(architecture, &())
            .unwrap();
    let mut tail_driver = reference_decision_driver(false);
    let (_, forward) = {
        let mut hook = SequentialDecisionTraversal::new(&mut tail_driver, &mut boundary);
        runtime
            .forward_with_traversal_hook(
                moshi::Input {
                    text: &text,
                    audio: &audio,
                    mask: None,
                },
                &mut state,
                &(),
                &mut hook,
            )
            .unwrap()
    };
    tail_driver.finish().unwrap();
    assert_eq!(
        tail_driver
            .decisions()
            .iter()
            .map(|decision| decision.source())
            .collect::<Vec<_>>(),
        [
            SequentialDecisionSource::Sampled,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
            SequentialDecisionSource::ForcedTailSkipped,
        ]
    );
    assert_eq!(
        forward.previous_depth_token(),
        Some(&ReferenceTensor(vec![1, 1]))
    );
    assert_eq!(state.as_ref()[2].offset, 0);
    assert_eq!(state.as_ref()[3].offset, 0);
    assert_eq!(state.as_ref()[2].resets, 2);
    assert_eq!(state.as_ref()[3].resets, 2);
    assert_eq!(state.as_ref()[0].offset, 2);
    assert_eq!(state.as_ref()[1].offset, 2);
}

#[test]
fn fused_decoder_projection_fields_must_be_named() {
    let error = match TransformerBlock::<ReferenceBackend>::new(
        &ProjectionLayoutConfig {
            args: tiny_args(),
            fused: true,
            empty_field: true,
            alternate_fields: false,
        },
        0,
        &(),
    ) {
        Ok(_) => panic!("empty fused projection fields must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("field must not be empty"));
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
    let mut state = DeviceState::<ReferenceBackend, ReferenceCache>::create(layout, |_, policy| {
        Ok::<_, std::convert::Infallible>(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|window| window.get() as i32),
            resets: 0,
        })
    })
    .unwrap();
    let architecture = llama::LayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
    let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();

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
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 3, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 3);
    assert_eq!(state.layer(1).unwrap().offset(), 3);

    let decode = ReferenceTensor(vec![1, 1]);
    let logits = runtime
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 4);
    assert_eq!(state.layer(1).unwrap().offset(), 4);

    let (architecture, units) = runtime.into_parts();
    let mut layerwise = LayerwiseRuntime::new(architecture, ResidentUnitWindow::new(units));
    let logits = layerwise
        .forward(
            LayeredInput {
                tokens: &decode,
                mask: None,
            },
            &mut state,
            &(),
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, 32]);
    assert_eq!(state.layer(0).unwrap().offset(), 5);
    assert_eq!(state.layer(1).unwrap().offset(), 5);
}

#[test]
fn shared_decoder_runs_qwen2_and_qwen3_without_mlx() {
    for model_type in ["qwen2", "qwen3", "qwen3_moe"] {
        let mut config = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false
        });
        if model_type == "qwen3_moe" {
            config["intermediate_size"] = 0.into();
            config["moe_intermediate_size"] = 8.into();
            config["num_experts"] = 4.into();
            config["num_experts_per_tok"] = 2.into();
            config["norm_topk_prob"] = true.into();
        } else if model_type == "qwen2" {
            config["use_sliding_window"] = true.into();
            config["sliding_window"] = 4.into();
            config["max_window_layers"] = 1.into();
        }
        let args = qwen::model_args_from_config_value(&config).unwrap();
        let layout = qwen::state_layout(&args).unwrap();
        let mut state =
            DeviceState::<ReferenceBackend, ReferenceCache>::create(layout, |_, policy| {
                Ok::<_, std::convert::Infallible>(ReferenceCache {
                    offset: 0,
                    window: policy
                        .attention()
                        .and_then(|attention| attention.window())
                        .map(|window| window.get() as i32),
                    resets: 0,
                })
            })
            .unwrap();
        let architecture = qwen::RoutedLayeredModel::<ReferenceBackend>::new(args, &()).unwrap();
        let mut runtime = ResidentRuntime::new(architecture, &()).unwrap();
        let parameters = runtime
            .units()
            .iter()
            .flat_map(topology)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        if model_type == "qwen2" {
            assert!(!parameters
                .iter()
                .any(|name| name.ends_with("q_norm.weight")));
        } else {
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("q_norm.weight")));
        }
        if model_type == "qwen3_moe" {
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("mlp.gate.weight")));
            assert!(parameters
                .iter()
                .any(|name| name.ends_with("mlp.experts.gate_up_proj")));
        }
        let logits = runtime
            .forward(
                LayeredInput {
                    tokens: &ReferenceTensor(vec![1, 3]),
                    mask: None,
                },
                &mut state,
                &(),
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
        assert_eq!(state.layer(0).unwrap().offset(), 3);
        assert_eq!(state.layer(1).unwrap().offset(), 3);
        if model_type == "qwen2" {
            assert_eq!(state.layer(0).unwrap().max_size(), None);
            assert_eq!(state.layer(1).unwrap().max_size(), Some(4));
        }
        let logits = runtime
            .forward(
                LayeredInput {
                    tokens: &ReferenceTensor(vec![1, 1]),
                    mask: None,
                },
                &mut state,
                &(),
            )
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(state.layer(0).unwrap().offset(), 4);
        assert_eq!(state.layer(1).unwrap().offset(), 4);
    }
}

#[derive(Default)]
struct ProbeExpertProvider {
    calls: Vec<(usize, ExpertPass, Vec<i32>, Vec<i32>)>,
}

impl RoutedExpertProvider<ReferenceBackend> for ProbeExpertProvider {
    type Error = std::convert::Infallible;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut ReferenceLinear,
        request: RoutedExpertRequest<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Self::Error> {
        self.calls.push((
            request.layer,
            request.pass,
            request.input.shape().to_vec(),
            request.routes.expert_ids.shape().to_vec(),
        ));
        Ok(request.input.clone())
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut ReferenceLinear,
        request: RoutedExpertRequest<'_, ReferenceTensor>,
        _: &(),
    ) -> Result<ReferenceTensor, Self::Error> {
        self.calls.push((
            request.layer,
            request.pass,
            request.input.shape().to_vec(),
            request.routes.expert_ids.shape().to_vec(),
        ));
        Ok(request.input.clone())
    }
}

#[derive(Default)]
struct ProbeObserver {
    route_shapes: Vec<(Vec<i32>, Vec<i32>, i32)>,
}

impl eredu_runtime::ActivationObserver<ReferenceTensor, Error> for ProbeObserver {
    fn observe(&mut self, _: &str, _: &ReferenceTensor) -> Result<(), Error> {
        Ok(())
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, ReferenceTensor>,
    ) -> Result<(), Error> {
        self.route_shapes.push((
            routing.selected_experts.shape().to_vec(),
            routing.route_weights.shape().to_vec(),
            routing.expert_count,
        ));
        Ok(())
    }
}

#[test]
fn qwen_routed_execution_uses_the_runtime_provider_and_observer_contract() {
    let config = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 0,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": true,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false
    });
    let args = qwen::model_args_from_config_value(&config).unwrap();
    let mut policy = qwen::FeedForward::<ReferenceBackend>::new(&args, 0, &()).unwrap();
    let input = ReferenceTensor(vec![3, 8]);
    let mut provider = ProbeExpertProvider::default();
    let output = policy
        .forward_with_provider(0, ExpertPass::Prefill, &input, &(), &mut provider)
        .unwrap();
    assert_eq!(output.shape(), input.shape());
    assert_eq!(
        provider.calls,
        vec![(0, ExpertPass::Prefill, vec![3, 8], vec![3, 2])]
    );

    let mut observer = ProbeObserver::default();
    let point = args.routed_observation_point("model.layers.0", 0).unwrap();
    let mut observed_provider =
        eredu_runtime::ObservedExpertProvider::new(&mut provider, &mut observer, point);
    let output = policy
        .forward_with_provider(0, ExpertPass::Decode, &input, &(), &mut observed_provider)
        .unwrap();
    drop(observed_provider);
    assert_eq!(output.shape(), input.shape());
    assert_eq!(observer.route_shapes, vec![(vec![3, 2], vec![3, 2], 4)]);
    assert_eq!(provider.calls[1].1, ExpertPass::Decode);
}

#[test]
fn released_moshi_profiles_share_one_portable_model_contract() {
    let native = moshi::MoshiConfig::native_v0_1().unwrap();
    let persona =
        moshi::MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
    for config in [&native, &persona] {
        assert_eq!(config.family(), "moshi");
        assert_eq!(config.temporal().parameter_root(), "transformer");
        assert_eq!(
            config.temporal().attention_window(),
            config.temporal().context() + 1
        );
        assert_eq!(
            config.depth_template().attention_window(),
            config.depth_template().context()
        );
        assert_eq!(
            config.depth_transformer(0).unwrap().parameter_root(),
            "depformer.slices.0.transformer"
        );
        let layout = moshi::state_layout(config).unwrap();
        assert_eq!(layout.segments()[0].id().as_str(), "temporal");
        assert_eq!(layout.segments()[1].id().as_str(), "depth");
    }
    assert_ne!(
        native.architecture_fingerprint(),
        persona.architecture_fingerprint()
    );
}

#[test]
fn moshi_decision_domains_include_exact_released_padding_rows() {
    for config in [
        moshi::MoshiConfig::native_v0_1().unwrap(),
        moshi::MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap(),
    ] {
        let boundary = moshi::DecisionBoundary::new(&config).unwrap();
        assert_eq!(
            boundary.text_token_domain().cardinality(),
            config.text_vocabulary_size() as usize + 1
        );
        assert_eq!(
            boundary.audio_token_domain().cardinality(),
            config.audio_vocabulary_size() as usize + 1
        );
    }
}

#[test]
fn external_assistant_safetensors_schemas_equal_neutral_parameter_topology() {
    let gemma_config = gemma4::AssistantConfig::from_json(
        br#"{
          "model_type":"gemma4_assistant","backbone_hidden_size":32,
          "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
          "text_config":{"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
            "vocab_size":32,"max_position_embeddings":128,
            "tie_word_embeddings":false,"attention_k_eq_v":false,
            "layer_types":["full_attention"]}
        }"#,
    )
    .unwrap();
    let gemma = gemma4::Assistant::<ReferenceBackend>::new(gemma_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &gemma,
        gemma4::assistant_safetensors_plan(&gemma_config).unwrap(),
    );
    let mut tied_value = serde_json::from_slice::<serde_json::Value>(
        br#"{
          "model_type":"gemma4_assistant","backbone_hidden_size":32,
          "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
          "text_config":{"model_type":"gemma4_text","hidden_size":32,
            "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
            "vocab_size":32,"max_position_embeddings":128,
            "tie_word_embeddings":false,"attention_k_eq_v":false,
            "layer_types":["full_attention"]}
        }"#,
    )
    .unwrap();
    tied_value["tie_word_embeddings"] = true.into();
    let tied_config =
        gemma4::AssistantConfig::from_json(&serde_json::to_vec(&tied_value).unwrap()).unwrap();
    let tied = gemma4::Assistant::<ReferenceBackend>::new(tied_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &tied,
        gemma4::assistant_safetensors_plan(&tied_config).unwrap(),
    );
    tied_value["tie_word_embeddings"] = false.into();
    tied_value["use_ordered_embeddings"] = true.into();
    tied_value["num_centroids"] = 4.into();
    tied_value["centroid_intermediate_top_k"] = 2.into();
    let ordered_config =
        gemma4::AssistantConfig::from_json(&serde_json::to_vec(&tied_value).unwrap()).unwrap();
    let ordered = gemma4::Assistant::<ReferenceBackend>::new(ordered_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &ordered,
        gemma4::assistant_safetensors_plan(&ordered_config).unwrap(),
    );
    let dflash_config = muse_glimmer::DFlashConfig::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
          "model_type":"muse_glimmer_assistant","hidden_size":6656,
          "intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,
          "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
          "max_position_embeddings":131072,"sliding_window":2048,"block_size":16,
          "mask_token_id":201818,"target_layer_ids":[1,13,25,37,49],
          "layer_types":["sliding_attention","sliding_attention","sliding_attention",
            "sliding_attention","sliding_attention"],
          "hidden_act":"silu","attention_dropout":0.0,
          "rope_parameters":{"rope_theta":500000.0}
        }))
        .unwrap(),
    )
    .unwrap();
    let dflash = muse_glimmer::DFlash::<ReferenceBackend>::new(dflash_config.clone(), &()).unwrap();
    assert_assistant_plan_matches_topology(
        &dflash,
        muse_glimmer::dflash_safetensors_plan(&dflash_config).unwrap(),
    );
}

fn assert_assistant_plan_matches_topology<M: Parameterized<ReferenceTensor>>(
    module: &M,
    plan: eredu_checkpoint::schema::SafetensorsCheckpointPlan,
) {
    assert!(plan.layout_groups.is_empty());
    let declared = plan
        .common_tensors
        .into_iter()
        .map(|tensor| (tensor.key, tensor.shape))
        .collect::<BTreeMap<_, _>>();
    let actual = topology(module).into_iter().collect::<BTreeMap<_, _>>();
    assert_eq!(declared, actual);
}
