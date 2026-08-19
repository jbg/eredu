//! MLX implementation of backend-neutral neural operators.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
};

use eredu_core::Completion;
use eredu_nn::{
    validate_parameter_topology, AttentionCache, EmbeddingOperator, EmbeddingSpec,
    Error as ComputeError, LinearOperator, LinearSpec, NeuralBackend, NormalizationOperator,
    NormalizationSpec, ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut,
    Parameterized, RopeValue, RotaryOperator, RotarySpec,
};
use eredu_runtime::{ParameterBackend, SubmissionBackend, TransferBackend};
use safemlx::{
    builder::Builder,
    distributed::Group,
    fast::ScaledDotProductAttentionMask,
    module::{Module, ModuleParam, ModuleParamMut, ModuleParamRef, ModuleParameters},
    nested::NestedValue,
    nn,
    quantization::MaybeQuantized,
    Array, Dtype, Event, HostTransferBuffer, HostTransferPolicy, ImmutableHostTransferBuffer,
    Stream,
};

use crate::backend::mlx::{
    nn::{self as common, tensor::rope::RopeVariant},
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
};

fn compute<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, ComputeError> {
    result.map_err(ComputeError::backend)
}

fn companion_spec(weight: &ParameterSpec, component: &str) -> ParameterSpec {
    let prefix = weight
        .id
        .as_str()
        .strip_suffix(".weight")
        .unwrap_or_else(|| weight.id.as_str());
    ParameterSpec {
        id: eredu_nn::ParameterId::new(format!("{prefix}.{component}"))
            .expect("authoritative weight identity produces a non-empty companion identity"),
        trainable: weight.trainable,
        alias_of: None,
        group: Some(weight.id.as_str().to_owned()),
    }
}

fn parameter_topology(
    module: &impl ModuleParameters,
    weight: ParameterSpec,
    bias: Option<ParameterSpec>,
) -> Result<BTreeMap<String, ParameterSpec>, ComputeError> {
    module
        .parameters()
        .flatten()
        .into_keys()
        .map(|local| {
            let spec = match local.as_ref() {
                "weight" | "inner.weight" => weight.clone(),
                "bias" | "inner.bias" => bias.clone().ok_or_else(|| {
                    ComputeError::backend(format!(
                        "backend operator exposed unexpected bias parameter {local:?}"
                    ))
                })?,
                "scales" => companion_spec(&weight, "scales"),
                "biases" => companion_spec(&weight, "biases"),
                name => {
                    return Err(ComputeError::backend(format!(
                        "backend operator exposed unknown parameter slot {name:?}"
                    )))
                }
            };
            Ok((local.to_string(), spec))
        })
        .collect()
}

fn visit_module_parameters<'a, M, V>(
    module: &'a M,
    topology: &BTreeMap<String, ParameterSpec>,
    visitor: &mut V,
) where
    M: ModuleParameters,
    V: ParameterVisitor<'a, Array>,
{
    let trainable = module
        .trainable_parameters()
        .flatten()
        .into_keys()
        .map(|name| name.to_string())
        .collect::<BTreeSet<_>>();
    for (local, value) in module.parameters().flatten() {
        let spec = topology
            .get(local.as_ref())
            .expect("validated backend parameter topology covers every native slot");
        visitor.visit(
            ParameterMetadata::from_spec(spec, trainable.contains(local.as_ref())),
            value,
        );
    }
}

fn visit_module_parameters_mut<'a, M, V>(
    module: &'a mut M,
    topology: &BTreeMap<String, ParameterSpec>,
    visitor: &mut V,
) where
    M: ModuleParameters,
    V: ParameterVisitorMut<'a, Array>,
{
    let trainable = module
        .trainable_parameters()
        .flatten()
        .into_keys()
        .map(|name| name.to_string())
        .collect::<BTreeSet<_>>();
    for (local, value) in module.parameters_mut().flatten() {
        let spec = topology
            .get(local.as_ref())
            .expect("validated backend parameter topology covers every native slot");
        visitor.visit_mut(
            ParameterMetadata::from_spec(spec, trainable.contains(local.as_ref())),
            value,
        );
    }
}

fn set_module_trainable(module: &mut impl ModuleParameters, trainable: bool) {
    if trainable {
        module.unfreeze_parameters(true);
    } else {
        module.freeze_parameters(true);
    }
}

struct ParameterRefCollector<'a> {
    parameters: ModuleParamRef<'a>,
    trainable_only: bool,
}

impl<'a> ParameterVisitor<'a, Array> for ParameterRefCollector<'a> {
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a Array) {
        if !self.trainable_only || metadata.trainable {
            self.parameters
                .insert(Rc::from(metadata.id.as_str()), NestedValue::Value(value));
        }
    }
}

struct ParameterMutCollector<'a> {
    parameters: ModuleParamMut<'a>,
}

impl<'a> ParameterVisitorMut<'a, Array> for ParameterMutCollector<'a> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut Array) {
        self.parameters
            .insert(Rc::from(metadata.id.as_str()), NestedValue::Value(value));
    }
}

struct ParameterStateCollector {
    states: Vec<bool>,
}

impl<'a> ParameterVisitor<'a, Array> for ParameterStateCollector {
    fn visit(&mut self, metadata: ParameterMetadata, _value: &'a Array) {
        self.states.push(metadata.trainable);
    }
}

pub(crate) fn neutral_parameter_refs<M: Parameterized<Array>>(
    module: &M,
    trainable_only: bool,
) -> ModuleParamRef<'_> {
    validate_parameter_topology(module).expect("backend-neutral parameter topology is valid");
    let mut collector = ParameterRefCollector {
        parameters: ModuleParamRef::new(),
        trainable_only,
    };
    module.visit_parameters(&mut collector);
    collector.parameters
}

pub(crate) fn neutral_parameter_refs_mut<M: Parameterized<Array>>(
    module: &mut M,
) -> ModuleParamMut<'_> {
    validate_parameter_topology(&*module).expect("backend-neutral parameter topology is valid");
    let mut collector = ParameterMutCollector {
        parameters: ModuleParamMut::new(),
    };
    module.visit_parameters_mut(&mut collector);
    collector.parameters
}

pub(crate) fn neutral_parameter_states<M: Parameterized<Array>>(module: &M) -> Vec<bool> {
    validate_parameter_topology(module).expect("backend-neutral parameter topology is valid");
    let mut collector = ParameterStateCollector { states: Vec::new() };
    module.visit_parameters(&mut collector);
    collector.states
}

/// SafeMLX module view over any backend-neutral parameterized value.
///
/// This is a reusable backend capability: architecture types retain their
/// neutral parameter identities while legacy SafeMLX loading utilities can
/// traverse the same native slots without rebuilding a parameter tree.
#[derive(Debug, Clone)]
pub struct MlxModule<M> {
    /// Backend-neutral module specialized to MLX operators.
    pub(crate) inner: M,
}

impl<M> MlxModule<M> {
    /// Wraps a neutral module without changing its storage.
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M> std::ops::Deref for MlxModule<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M> std::ops::DerefMut for MlxModule<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M: Parameterized<Array>> ModuleParameters for MlxModule<M> {
    fn num_parameters(&self) -> usize {
        neutral_parameter_refs(&self.inner, false).entries.len()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(&self.inner, false)
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        neutral_parameter_refs_mut(&mut self.inner)
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(&self.inner, true)
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        self.inner.set_trainable(false);
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        self.inner.set_trainable(true);
    }

    fn all_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().all(|trainable| !trainable))
    }

    fn any_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(&self.inner);
        (!states.is_empty()).then(|| states.iter().any(|trainable| !trainable))
    }
}

macro_rules! delegate_parameters {
    ($type:ty, $field:tt) => {
        impl ModuleParameters for $type {
            fn num_parameters(&self) -> usize {
                self.$field.num_parameters()
            }
            fn parameters(&self) -> ModuleParamRef<'_> {
                self.$field.parameters()
            }
            fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
                self.$field.parameters_mut()
            }
            fn trainable_parameters(&self) -> ModuleParamRef<'_> {
                self.$field.trainable_parameters()
            }
            fn update(&mut self, parameters: ModuleParam) {
                self.$field.update(parameters);
            }
            fn freeze_parameters(&mut self, recursive: bool) {
                self.$field.freeze_parameters(recursive);
            }
            fn unfreeze_parameters(&mut self, recursive: bool) {
                self.$field.unfreeze_parameters(recursive);
            }
            fn all_frozen(&self) -> Option<bool> {
                self.$field.all_frozen()
            }
            fn any_frozen(&self) -> Option<bool> {
                self.$field.any_frozen()
            }
        }
    };
}

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear {
    module: MaybeQuantized<nn::Linear>,
    topology: BTreeMap<String, ParameterSpec>,
}

impl Deref for MlxLinear {
    type Target = MaybeQuantized<nn::Linear>;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}
impl DerefMut for MlxLinear {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}
delegate_parameters!(MlxLinear, module);

impl LinearOperator<Array> for MlxLinear {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.module.forward(input, context))
    }
}

impl Parameterized<Array> for MlxLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        visit_module_parameters_mut(&mut self.module, &self.topology, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.module, trainable);
    }
}

/// MLX dense-or-quantized token embedding.
#[derive(Debug, Clone)]
pub struct MlxEmbedding {
    module: MaybeQuantized<nn::Embedding>,
    topology: BTreeMap<String, ParameterSpec>,
}

impl Deref for MlxEmbedding {
    type Target = MaybeQuantized<nn::Embedding>;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}
impl DerefMut for MlxEmbedding {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}
delegate_parameters!(MlxEmbedding, module);

impl EmbeddingOperator<Array> for MlxEmbedding {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.module.forward(input, context))
    }

    fn as_linear(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(match &mut self.module {
            MaybeQuantized::Original(embedding) => embedding.as_linear(input, context),
            MaybeQuantized::Quantized(embedding) => embedding.as_linear(input, context),
        })
    }
}

impl Parameterized<Array> for MlxEmbedding {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        visit_module_parameters_mut(&mut self.module, &self.topology, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.module, trainable);
    }
}

/// MLX fused RMS normalization.
#[derive(Debug, Clone)]
pub struct MlxRmsNorm {
    module: nn::RmsNorm,
    topology: BTreeMap<String, ParameterSpec>,
}

impl Deref for MlxRmsNorm {
    type Target = nn::RmsNorm;
    fn deref(&self) -> &Self::Target {
        &self.module
    }
}
impl DerefMut for MlxRmsNorm {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.module
    }
}
delegate_parameters!(MlxRmsNorm, module);

impl NormalizationOperator<Array> for MlxRmsNorm {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.module.forward(input, context))
    }
}

impl Parameterized<Array> for MlxRmsNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        visit_module_parameters_mut(&mut self.module, &self.topology, visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.module, trainable);
    }
}

/// MLX RoPE variant selected from model metadata.
#[derive(Debug, Clone)]
pub struct MlxRotary(pub RopeVariant);

impl Deref for MlxRotary {
    type Target = RopeVariant;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MlxRotary {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
delegate_parameters!(MlxRotary, 0);

impl RotaryOperator<Array> for MlxRotary {
    fn forward(
        &mut self,
        input: &Array,
        offset: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let rope_input = nn::RopeInputBuilder::new(input)
            .offset(offset)
            .build()
            .map_err(ComputeError::backend)?;
        compute(self.0.forward(rope_input, context))
    }
}

impl Parameterized<Array> for MlxRotary {
    fn visit_parameters<'a, V>(&'a self, _visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
    }
    fn set_trainable(&mut self, _trainable: bool) {}
}

/// Zero-sized MLX backend selector. All calls are statically dispatched.
#[derive(Debug, Clone, Copy)]
pub struct MlxBackend;

/// Exact MLX completion retaining every Rust-side submission resource.
pub struct MlxSubmissionCompletion {
    event: Event,
    retained: RefCell<Vec<Box<dyn Send>>>,
}

impl MlxSubmissionCompletion {
    fn new(event: Event) -> Self {
        Self {
            event,
            retained: RefCell::new(Vec::new()),
        }
    }

    fn retain<T: Send + 'static>(&self, value: T) {
        self.retained.borrow_mut().push(Box::new(value));
    }

    /// Orders a consumer stream after this exact completion without blocking.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), safemlx::error::Exception> {
        self.event.wait_on(stream)
    }
}

impl Completion for MlxSubmissionCompletion {
    type Error = safemlx::error::Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        let complete = self.event.is_complete()?;
        if complete {
            self.retained.borrow_mut().clear();
        }
        Ok(complete)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize()?;
        self.retained.borrow_mut().clear();
        Ok(())
    }
}

impl Drop for MlxSubmissionCompletion {
    fn drop(&mut self) {
        if !matches!(self.event.is_complete(), Ok(true)) {
            let _ = self.event.synchronize();
        }
        self.retained.get_mut().clear();
    }
}

impl SubmissionBackend for MlxBackend {
    type Executor = Stream;
    type Completion = MlxSubmissionCompletion;

    fn retain_until_complete<T: Send + 'static>(
        _executor: &Self::Executor,
        completion: &Self::Completion,
        value: T,
    ) -> Result<(), <Self::Completion as Completion>::Error> {
        completion.retain(value);
        Ok(())
    }
}

impl TransferBackend for MlxBackend {
    type HostBuffer = Arc<ImmutableHostTransferBuffer>;
    type Transfer = MlxSubmissionCompletion;
    type TransferError = safemlx::error::Exception;

    fn promote(
        executor: &Self::Executor,
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Self::Transfer), Self::TransferError> {
        let submitted = host.copy_to_array(executor)?;
        let (weight, event) = submitted.into_parts();
        let completion = MlxSubmissionCompletion::new(event);
        completion.retain(Arc::clone(host));
        completion.retain(weight.clone());
        Ok((weight, completion))
    }

    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError> {
        let submitted =
            HostTransferBuffer::copy_from_array(weight, HostTransferPolicy::Transfer, executor)?;
        let (host, event) = submitted.into_parts();
        let host = Arc::new(host.freeze());
        let completion = MlxSubmissionCompletion::new(event);
        completion.retain(weight.clone());
        completion.retain(Arc::clone(&host));
        Ok((host, completion))
    }
}

/// MLX failure while lowering a neutral checkpoint lease or recipe.
#[derive(Debug, thiserror::Error)]
pub enum MlxParameterError {
    /// Encoded lease acquisition or MLX submission failed.
    #[error(transparent)]
    Store(#[from] crate::backend::mlx::runtime::checkpoint::store::WeightStoreError),
    /// A neutral derived-weight recipe could not be lowered.
    #[error(transparent)]
    Recipe(#[from] crate::backend::mlx::runtime::checkpoint::recipe::WeightRecipeError),
    /// Final stream-to-stream weight copy failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

impl ParameterBackend for MlxBackend {
    type Parameter = Array;
    type MaterializedWeight = Array;
    type MaterializationContext =
        crate::backend::mlx::runtime::checkpoint::store::MlxParameterMaterializationContext;
    type Materialization = crate::backend::mlx::runtime::checkpoint::store::WeightMaterialization;
    type ParameterError = MlxParameterError;

    fn materialize(
        lease: eredu_checkpoint::store::CheckpointLease,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError> {
        Ok(context
            .weight_lease(lease)?
            .materialize(context.source_stream(), context.execution_stream())?)
    }

    fn materialize_recipe(
        recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        context: &Self::MaterializationContext,
    ) -> Result<Self::Materialization, Self::ParameterError> {
        use crate::backend::mlx::runtime::checkpoint::recipe::MlxWeightRecipeExt;

        let pending = recipe.prepare_materialization(source, context)?;
        let (output, sources) = pending.into_parts();
        let output = if context.source_stream() == context.execution_stream() {
            output
        } else {
            output.copy(context.execution_stream())?
        };
        Ok(
            crate::backend::mlx::runtime::checkpoint::store::WeightMaterialization::submit_retained(
                output, sources,
            )?,
        )
    }

    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight {
        materialization.output()
    }

    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(materialization.synchronize()?)
    }

    fn bind(
        parameter: &mut Self::Parameter,
        weight: Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        *parameter = weight;
        Ok(())
    }
}

impl NeuralBackend for MlxBackend {
    type Tensor = Array;
    type Linear = MlxLinear;
    type Embedding = MlxEmbedding;
    type Normalization = MlxRmsNorm;
    type Rotary = MlxRotary;
    type ParallelContext = Group;

    fn linear(spec: LinearSpec, context: &Stream) -> Result<MlxLinear, ComputeError> {
        let module = compute(common::linear::unloaded_maybe_quantized_linear(
            spec.input,
            spec.output,
            spec.bias.is_some(),
            spec.quantization,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias)?;
        Ok(MlxLinear { module, topology })
    }

    fn embedding(spec: EmbeddingSpec, context: &Stream) -> Result<MlxEmbedding, ComputeError> {
        let module = compute(common::linear::unloaded_maybe_quantized_embedding(
            spec.vocabulary,
            spec.dimensions,
            spec.quantization,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None)?;
        Ok(MlxEmbedding { module, topology })
    }

    fn rms_norm(spec: NormalizationSpec, context: &Stream) -> Result<MlxRmsNorm, ComputeError> {
        let module = compute(nn::RmsNorm::unloaded(
            spec.dimensions,
            spec.epsilon,
            Dtype::Float32,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None)?;
        Ok(MlxRmsNorm { module, topology })
    }

    fn rotary(spec: RotarySpec<'_>, context: &Stream) -> Result<MlxRotary, ComputeError> {
        let scaling = spec.scaling.map(|values| {
            values
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        RopeValue::Float(value) => RopeValue::Float(*value),
                        RopeValue::String(value) => RopeValue::String(value.clone()),
                        RopeValue::Bool(value) => RopeValue::Bool(*value),
                    };
                    (key.clone(), value)
                })
                .collect()
        });
        compute(crate::backend::mlx::nn::tensor::rope::initialize_rope(
            spec.dimensions,
            spec.base,
            spec.traditional,
            &scaling,
            spec.max_positions,
            context,
        ))
        .map(MlxRotary)
    }

    fn silu(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(common::layers::silu(input, context))
    }

    fn attention(
        queries: Array,
        keys: Array,
        values: Array,
        scale: f32,
        mask: Option<&Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            None,
            context,
        ))
    }

    fn sliding_window_attention(
        queries: Array,
        keys: Array,
        values: Array,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let batch = queries.dim(0);
        let sequence = queries.dim(2);
        compute(common::attention::sliding_window_prefill_attention(
            queries,
            keys,
            values,
            scale,
            window,
            position_offset,
            batch,
            sequence,
            context,
        ))
    }

    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(crate::backend::mlx::nn::tensor::create_causal_mask(
            sequence,
            Some(offset),
            window,
            None,
            context,
        ))
    }

    fn row_parallel_linear(
        linear: &mut MlxLinear,
        input: &Array,
        parallel: &Group,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut linear.module,
            input,
            parallel,
            context,
        ))
    }
}

macro_rules! impl_attention_cache {
    ($type:ty) => {
        impl AttentionCache<Array> for $type {
            fn offset(&self) -> i32 {
                KeyValueCache::offset(self)
            }
            fn max_size(&self) -> Option<i32> {
                KeyValueCache::max_size(self)
            }
            fn update_for_attention(
                &mut self,
                keys: Array,
                values: Array,
                context: &Stream,
            ) -> Result<(Array, Array), ComputeError> {
                compute(KeyValueCache::update_for_attention(
                    self, keys, values, context,
                ))
            }
            fn attention(
                &mut self,
                queries: Array,
                keys: Array,
                values: Array,
                scale: f32,
                mask: Option<&Array>,
                context: &Stream,
            ) -> Result<Array, ComputeError> {
                if let Some(output) = compute(KeyValueCache::paged_attention(
                    self, &queries, scale, mask, None, context,
                ))? {
                    return Ok(output);
                }
                compute(
                    crate::backend::mlx::nn::tensor::scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        Some(self),
                        scale,
                        mask,
                        context,
                    ),
                )
            }
        }
    };
}

impl_attention_cache!(ConcatKeyValueCache);
impl_attention_cache!(SlidingKeyValueCache);
impl_attention_cache!(PagedKeyValueCache);
