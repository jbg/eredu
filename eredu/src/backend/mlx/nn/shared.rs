//! MLX implementation of backend-neutral neural operators.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
};

use eredu_checkpoint::WeightQuantization;
use eredu_core::Completion;
use eredu_nn::{
    validate_parameter_topology, AttentionCache, AttentionRequest, BlockwiseAttentionBackend,
    BlockwiseAttentionSpec, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error as ComputeError, GatedDeltaScanInput, GatedDeltaScanOutput,
    GatedProductExpertBankOperator, GatedProductExpertBankSpec, GatedProductExpertLayout,
    GatedProductPolicy, HyperConnectionOperator, HyperConnectionSpec, HyperConnectionState,
    HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend, IndexedAttentionInput,
    JointExpertRoutingInput, JointExpertRoutingResult, LinearFormat, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, NormalizationScale,
    NormalizationSpec, ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut,
    Parameterized, PooledAttentionInput, PooledPositionInput, RelativeAttentionInput,
    Relu2ExpertBankOperator, Relu2ExpertBankSpec, RopeValue, RotaryOperator, RotaryPosition,
    RotarySpec, RoutedNeuralBackend, RoutingOperator, RoutingResult, RoutingScoring,
    SegmentedAttentionInput, SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput, Tensor,
    TensorParallelExpertOutput, TopKRouterSpec, VocabularyParallelRange,
};
use eredu_runtime::{ParameterBackend, SubmissionBackend, TransferBackend};
use safemlx::ops::{
    arange, argpartition_axis, broadcast_to, clip, concatenate_axis, einsum,
    indexing::{take_along_axis, NewAxis, TryIndexOp},
    matmul, maximum, r#where, sigmoid, softmax_axis,
};
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
    nn::{
        self as common,
        tensor::{rope::RopeVariant, validate_token_domain},
    },
    runtime::cache::{
        BlockwiseAttentionAccumulator, ConcatKeyValueCache, KeyValueAttentionBlock, KeyValueCache,
        PagedKeyValueCache, SlidingKeyValueCache,
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

fn packed_expert_companion_spec(weight: &ParameterSpec, component: &str) -> ParameterSpec {
    ParameterSpec {
        id: eredu_nn::ParameterId::new(format!("{}_{}", weight.id.as_str(), component))
            .expect("authoritative expert identity produces a non-empty companion identity"),
        trainable: weight.trainable,
        alias_of: None,
        group: Some(weight.id.as_str().to_owned()),
    }
}

impl BlockwiseAttentionBackend for MlxBackend {
    type BlockwiseAccumulator = BlockwiseAttentionAccumulator;

    fn begin_blockwise_attention(
        spec: BlockwiseAttentionSpec<'_, Array>,
        context: &Stream,
    ) -> Result<Self::BlockwiseAccumulator, ComputeError> {
        compute(BlockwiseAttentionAccumulator::new(
            spec.queries,
            spec.scale,
            spec.mask,
            spec.query_start,
            spec.sliding_window,
            spec.prefix_tokens,
            spec.sinks,
            spec.context_end,
            context,
        ))
    }

    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: Array,
        values: Array,
        context: &Stream,
    ) -> Result<u64, ComputeError> {
        let scratch = keys.nbytes() as u64 + values.nbytes() as u64;
        let block = KeyValueAttentionBlock::unleased(start, end, keys, values);
        compute(accumulator.accumulate(&block, context))?;
        compute(accumulator.submit())?;
        Ok(scratch)
    }

    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(accumulator.finish(context))
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
                "weight_scale_inv" => companion_spec(&weight, "weight_scale_inv"),
                "e_score_correction_bias" => bias.clone().ok_or_else(|| {
                    ComputeError::backend(format!(
                        "backend operator exposed unexpected correction-bias parameter {local:?}"
                    ))
                })?,
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

fn exact_parameter_topology(
    module: &impl ModuleParameters,
    specs: impl IntoIterator<Item = (&'static str, ParameterSpec)>,
) -> Result<BTreeMap<String, ParameterSpec>, ComputeError> {
    let expected = specs
        .into_iter()
        .map(|(local, spec)| (local.to_owned(), spec))
        .collect::<BTreeMap<_, _>>();
    let actual = module
        .parameters()
        .flatten()
        .into_keys()
        .map(|local| local.to_string())
        .collect::<BTreeSet<_>>();
    let wanted = expected.keys().cloned().collect::<BTreeSet<_>>();
    if actual != wanted {
        return Err(ComputeError::backend(format!(
            "backend operator parameter slots differ: expected {wanted:?}, got {actual:?}"
        )));
    }
    Ok(expected)
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
/// Architecture types retain their neutral parameter identities while MLX
/// loading utilities traverse the same native slots without rebuilding a
/// parameter tree.
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

impl<M> AsMut<M> for MlxModule<M> {
    fn as_mut(&mut self) -> &mut M {
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

/// Borrowed SafeMLX parameter view over a backend-neutral module.
///
/// This lets residency/loading policy populate one architecture-owned static
/// component without cloning it or giving the architecture a SafeMLX trait.
pub(crate) struct MlxModuleRef<'a, M> {
    inner: &'a mut M,
}

impl<'a, M> MlxModuleRef<'a, M> {
    pub(crate) const fn new(inner: &'a mut M) -> Self {
        Self { inner }
    }
}

impl<M: Parameterized<Array>> ModuleParameters for MlxModuleRef<'_, M> {
    fn num_parameters(&self) -> usize {
        neutral_parameter_refs(&*self.inner, false).entries.len()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(&*self.inner, false)
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        neutral_parameter_refs_mut(&mut *self.inner)
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(&*self.inner, true)
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        self.inner.set_trainable(false);
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        self.inner.set_trainable(true);
    }

    fn all_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(&*self.inner);
        (!states.is_empty()).then(|| states.iter().all(|trainable| !trainable))
    }

    fn any_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(&*self.inner);
        (!states.is_empty()).then(|| states.iter().any(|trainable| !trainable))
    }
}

/// Native MLX module exposed through stable neutral parameter identities.
#[derive(Debug, Clone)]
pub(crate) struct MlxNamedModule<M> {
    inner: M,
    topology: BTreeMap<String, ParameterSpec>,
}

impl<M: ModuleParameters> MlxNamedModule<M> {
    /// Attaches one authoritative weight identity and optional bias identity.
    pub(crate) fn new(
        inner: M,
        weight: ParameterSpec,
        bias: Option<ParameterSpec>,
    ) -> Result<Self, ComputeError> {
        let topology = parameter_topology(&inner, weight, bias)?;
        Ok(Self { inner, topology })
    }

    fn with_exact_topology(
        inner: M,
        specs: impl IntoIterator<Item = (&'static str, ParameterSpec)>,
    ) -> Result<Self, ComputeError> {
        let topology = exact_parameter_topology(&inner, specs)?;
        Ok(Self { inner, topology })
    }
}

impl<M> std::ops::Deref for MlxNamedModule<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M> std::ops::DerefMut for MlxNamedModule<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M: ModuleParameters> Parameterized<Array> for MlxNamedModule<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        visit_module_parameters(&self.inner, &self.topology, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        visit_module_parameters_mut(&mut self.inner, &self.topology, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.inner, trainable);
    }
}

/// Generic neutral parameter view over an existing native MLX module tree.
///
/// The tree is inspected once during model construction and its complete
/// checkpoint-facing identities are then frozen for all loading and residency
/// operations. This capability is architecture-agnostic and never runs in a
/// forward loop.
#[derive(Debug, Clone)]
pub(crate) struct MlxParameterTree<M> {
    inner: M,
    topology: BTreeMap<String, ParameterSpec>,
}

impl<M: ModuleParameters> MlxParameterTree<M> {
    /// Captures one native module's complete stable parameter topology.
    pub(crate) fn new(inner: M, prefix: &str) -> Result<Self, ComputeError> {
        Self::new_filtered(inner, prefix, |_| true)
    }

    /// Captures a stable subset of one native module's parameter topology.
    ///
    /// This is used when a reusable execution unit contains parameters whose
    /// residency is owned by another policy, such as independently cached
    /// experts. The predicate is evaluated once on checkpoint-facing names;
    /// hot-path visitation remains statically dispatched over the retained
    /// native slots.
    pub(crate) fn new_filtered(
        inner: M,
        prefix: &str,
        include: impl Fn(&str) -> bool,
    ) -> Result<Self, ComputeError> {
        let mut topology = BTreeMap::new();
        for local in inner.parameters().flatten().into_keys() {
            let id = if prefix.is_empty() {
                local.to_string()
            } else {
                format!("{prefix}.{local}")
            };
            if !include(&id) {
                continue;
            }
            let group = ["scales", "biases"].into_iter().find_map(|component| {
                id.strip_suffix(&format!(".{component}"))
                    .map(|base| format!("{base}.weight"))
            });
            topology.insert(
                local.to_string(),
                ParameterSpec {
                    id: eredu_nn::ParameterId::new(id).map_err(ComputeError::backend)?,
                    trainable: true,
                    alias_of: None,
                    group,
                },
            );
        }
        Ok(Self { inner, topology })
    }

    /// Decomposes the cold-path view without cloning native parameter handles.
    #[cfg(test)]
    pub(crate) fn into_inner(self) -> M {
        self.inner
    }
}

impl<M> std::ops::Deref for MlxParameterTree<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M> std::ops::DerefMut for MlxParameterTree<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M: ModuleParameters> Parameterized<Array> for MlxParameterTree<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        let trainable = self
            .inner
            .trainable_parameters()
            .flatten()
            .into_keys()
            .map(|name| name.to_string())
            .collect::<BTreeSet<_>>();
        for (local, value) in self.inner.parameters().flatten() {
            let Some(spec) = self.topology.get(local.as_ref()) else {
                continue;
            };
            visitor.visit(
                ParameterMetadata::from_spec(spec, trainable.contains(local.as_ref())),
                value,
            );
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        let trainable = self
            .inner
            .trainable_parameters()
            .flatten()
            .into_keys()
            .map(|name| name.to_string())
            .collect::<BTreeSet<_>>();
        for (local, value) in self.inner.parameters_mut().flatten() {
            let Some(spec) = self.topology.get(local.as_ref()) else {
                continue;
            };
            visitor.visit_mut(
                ParameterMetadata::from_spec(spec, trainable.contains(local.as_ref())),
                value,
            );
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.inner, trainable);
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
    module: common::linear::PhysicalLinear,
    topology: BTreeMap<String, ParameterSpec>,
    vocabulary_range: Option<VocabularyParallelRange>,
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
    vocabulary: i32,
    vocabulary_range: Option<VocabularyParallelRange>,
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

    fn lookup(
        &mut self,
        input: &Array,
        policy: EmbeddingLookupPolicy,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        policy.validate()?;
        let sentinel = match policy {
            EmbeddingLookupPolicy::Strict => None,
            EmbeddingLookupPolicy::ZeroSentinel(sentinel) => Some(sentinel),
        };
        let input = compute(validate_token_domain(
            input,
            self.vocabulary,
            sentinel,
            context,
        ))?;
        let Some(sentinel) = sentinel else {
            return compute(self.module.forward(&input, context));
        };
        let sentinel_mask = input.equal_i32(sentinel, context)?;
        let nonnegative = compute(input.ge(Array::from_int(0), context))?;
        let below_vocabulary = compute(input.lt(Array::from_int(self.vocabulary), context))?;
        let ordinary_mask = compute(nonnegative.logical_and(&below_vocabulary, context))?;
        let safe_tokens =
            Array::where_condition(&ordinary_mask, &input, &input.zeros_like(context)?, context)?;
        let embedded = compute(self.module.forward(&safe_tokens, context))?;
        let output_mask = compute(sentinel_mask.expand_dims(-1, context))?;
        Array::where_condition(
            &output_mask,
            &embedded.zeros_like(context)?,
            &embedded,
            context,
        )
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

/// MLX RMS normalization with backend-native learned, learned-offset, or unit
/// scale construction.
#[derive(Debug, Clone)]
pub struct MlxRmsNorm {
    module: Option<nn::RmsNorm>,
    topology: BTreeMap<String, ParameterSpec>,
    offset: Option<f32>,
    dimensions: i32,
    epsilon: f32,
}

impl MlxRmsNorm {
    /// Applies the configured normalization through the neutral operator
    /// contract.
    pub fn forward(
        &mut self,
        input: &Array,
        context: &Stream,
    ) -> Result<Array, safemlx::error::Exception> {
        <Self as NormalizationOperator<Array>>::forward(self, input, context)
            .map_err(|error| safemlx::error::Exception::custom(error.to_string()))
    }
}

impl NormalizationOperator<Array> for MlxRmsNorm {
    fn forward(&mut self, input: &Array, context: &Stream) -> Result<Array, ComputeError> {
        if input.shape().last().copied() != Some(self.dimensions) {
            return Err(ComputeError::backend(format!(
                "RMS normalization expects final width {}, got {:?}",
                self.dimensions,
                input.shape()
            )));
        }
        match (&mut self.module, self.offset) {
            (Some(module), None) => compute(module.forward(input, context)),
            (Some(module), Some(offset)) => {
                let scale = compute(module.weight.as_ref().add(Array::from_f32(offset), context))?;
                compute(safemlx::fast::rms_norm(
                    input,
                    &scale,
                    self.epsilon,
                    context,
                ))
            }
            (None, None) => mlx_weightless_rms_norm(input, self.epsilon, context),
            (None, Some(_)) => unreachable!("validated normalization construction"),
        }
    }
}

impl Parameterized<Array> for MlxRmsNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        if let Some(module) = &self.module {
            visit_module_parameters(module, &self.topology, visitor);
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        if let Some(module) = &mut self.module {
            visit_module_parameters_mut(module, &self.topology, visitor);
        }
    }
    fn set_trainable(&mut self, trainable: bool) {
        if let Some(module) = &mut self.module {
            set_module_trainable(module, trainable);
        }
    }
}

impl ModuleParameters for MlxRmsNorm {
    fn num_parameters(&self) -> usize {
        neutral_parameter_refs(self, false).entries.len()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(self, false)
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        neutral_parameter_refs_mut(self)
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        neutral_parameter_refs(self, true)
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        self.set_trainable(false);
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        self.set_trainable(true);
    }

    fn all_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(self);
        (!states.is_empty()).then(|| states.iter().all(|trainable| !trainable))
    }

    fn any_frozen(&self) -> Option<bool> {
        let states = neutral_parameter_states(self);
        (!states.is_empty()).then(|| states.iter().any(|trainable| !trainable))
    }
}

fn mlx_weightless_rms_norm(
    input: &Array,
    epsilon: f32,
    context: &Stream,
) -> Result<Array, ComputeError> {
    let dtype = input.dtype();
    let variance = compute(input.square(context))?;
    let variance = compute(variance.mean_axis(-1, true, context))?;
    let denominator = compute(variance.add(Array::from_f32(epsilon), context))?;
    let denominator = compute(denominator.rsqrt(context))?;
    compute(input.multiply(denominator, context))?
        .as_dtype(dtype, context)
        .map_err(ComputeError::backend)
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
        position: RotaryPosition<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        match position {
            RotaryPosition::Offset(offset) => {
                let rope_input = nn::RopeInputBuilder::new(input)
                    .offset(offset)
                    .build()
                    .map_err(ComputeError::backend)?;
                compute(self.0.forward(rope_input, context))
            }
            RotaryPosition::Embeddings { cosine, sine } => compute(
                common::attention::apply_rotary_embeddings(input, cosine, sine, context),
            ),
        }
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

/// MLX implementation of neutral multi-stream residual mixing.
#[derive(Debug, Clone)]
pub struct MlxHyperConnection {
    module: common::hyper_connections::HyperConnection,
    topology: BTreeMap<String, ParameterSpec>,
}

delegate_parameters!(MlxHyperConnection, module);

impl HyperConnectionOperator<Array> for MlxHyperConnection {
    fn collapse(
        &mut self,
        residual: &Array,
        norm_epsilon: f32,
        context: &Stream,
    ) -> Result<HyperConnectionState<Array>, ComputeError> {
        let (collapsed, split) =
            compute(self.module.collapse_split(residual, norm_epsilon, context))?;
        Ok(HyperConnectionState {
            collapsed,
            pre: split.pre,
            post: split.post,
            combination: split.combination,
        })
    }

    fn expand(
        &mut self,
        sublayer: &Array,
        residual: &Array,
        state: &HyperConnectionState<Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(common::hyper_connections::expand(
            sublayer,
            residual,
            &state.post,
            &state.combination,
            context,
        ))
    }
}

impl Parameterized<Array> for MlxHyperConnection {
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

/// MLX implementation of the neutral final hyper-head collapse.
#[derive(Debug, Clone)]
pub struct MlxHyperHead {
    module: common::hyper_connections::HyperHead,
    topology: BTreeMap<String, ParameterSpec>,
}

delegate_parameters!(MlxHyperHead, module);

impl HyperHeadOperator<Array> for MlxHyperHead {
    fn forward(&mut self, residual: &Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(self.module.forward(residual, context))
    }
}

impl Parameterized<Array> for MlxHyperHead {
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

/// MLX implementation of the backend-neutral learned top-k router.
#[derive(Debug, Clone)]
pub struct MlxTopKRouter {
    module: MlxNamedModule<common::moe::TopKRouter>,
}

impl Parameterized<Array> for MlxTopKRouter {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        self.module.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        self.module.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl RoutingOperator<Array> for MlxTopKRouter {
    fn route(
        &mut self,
        input: &Array,
        context: &Stream,
    ) -> Result<RoutingResult<Array>, ComputeError> {
        let output = compute(
            self.module
                .forward_routes_with_selection_bias(input, None, context),
        )?;
        Ok(RoutingResult {
            expert_ids: output.indices,
            selected_scores: output.scores,
            route_weights: output.weights,
        })
    }

    fn route_selected(
        &mut self,
        input: &Array,
        expert_ids: &Array,
        context: &Stream,
    ) -> Result<RoutingResult<Array>, ComputeError> {
        let output = compute(
            self.module
                .forward_routes_with_routing_indices(input, expert_ids, context),
        )?;
        Ok(RoutingResult {
            expert_ids: output.indices,
            selected_scores: output.scores,
            route_weights: output.weights,
        })
    }
}

/// MLX packed execution bank for backend-neutral routed gated-product experts.
#[derive(Debug, Clone)]
pub struct MlxGatedProductExpertBank {
    module: MlxNamedModule<common::moe::PackedGatedProductExperts>,
}

impl Parameterized<Array> for MlxGatedProductExpertBank {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        self.module.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        self.module.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl GatedProductExpertBankOperator<Array> for MlxGatedProductExpertBank {
    fn forward_routed(
        &mut self,
        input: &Array,
        routes: &RoutingResult<Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward(
            &flattened,
            &routes.expert_ids,
            &routes.route_weights,
            context,
        ))?;
        compute(output.reshape(input.shape(), context))
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        input: &Array,
        routes: &RoutingResult<Array>,
        partitions: usize,
        context: &Stream,
    ) -> Result<TensorParallelExpertOutput<Array>, ComputeError> {
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward_tensor_parallel(
            &flattened,
            &routes.expert_ids,
            &routes.route_weights,
            partitions,
            context,
        ))?;
        Ok(TensorParallelExpertOutput {
            reducible: compute(output.reducible.reshape(input.shape(), context))?,
            post_reduce: output
                .post_reduce
                .map(|bias| compute(bias.reshape(input.shape(), context)))
                .transpose()?,
        })
    }
}

/// MLX packed execution bank for backend-neutral routed ReLU2 experts.
#[derive(Debug, Clone)]
pub struct MlxRelu2ExpertBank {
    module: MlxParameterTree<common::moe::PackedRelu2Experts>,
}

impl Parameterized<Array> for MlxRelu2ExpertBank {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        self.module.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        self.module.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl Relu2ExpertBankOperator<Array> for MlxRelu2ExpertBank {
    fn forward_routed(
        &mut self,
        input: &Array,
        routes: &RoutingResult<Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let shape = input.shape();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward(
            &flattened,
            &routes.expert_ids,
            &routes.route_weights,
            context,
        ))?;
        compute(output.reshape(shape, context))
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        input: &Array,
        routes: &RoutingResult<Array>,
        partitions: usize,
        context: &Stream,
    ) -> Result<TensorParallelExpertOutput<Array>, ComputeError> {
        let shape = input.shape();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward_tensor_parallel(
            &flattened,
            &routes.expert_ids,
            &routes.route_weights,
            partitions,
            context,
        ))?;
        Ok(TensorParallelExpertOutput {
            reducible: compute(output.reducible.reshape(shape, context))?,
            post_reduce: output
                .post_reduce
                .map(|bias| compute(bias.reshape(shape, context)))
                .transpose()?,
        })
    }
}

impl crate::backend::mlx::runtime::distributed::expert::LocalExpertBank for MlxRelu2ExpertBank {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, crate::backend::mlx::error::Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = crate::backend::mlx::runtime::distributed::expert::unit_route_weights(
            hidden.dim(0),
            hidden.dtype(),
            stream,
        )?;
        Ok(self.module.forward(hidden, &ids, &weights, stream)?)
    }
}

impl crate::backend::mlx::runtime::distributed::expert::LocalExpertBank
    for MlxGatedProductExpertBank
{
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, crate::backend::mlx::error::Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = crate::backend::mlx::runtime::distributed::expert::unit_route_weights(
            hidden.dim(0),
            hidden.dtype(),
            stream,
        )?;
        Ok(self.module.forward(hidden, &ids, &weights, stream)?)
    }
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
    type OwnedExecutor = Stream;
    type Completion = MlxSubmissionCompletion;

    fn fork_executors(
        executor: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, safemlx::error::Exception> {
        if count <= 1 {
            return Ok((0..count).map(|_| executor.clone()).collect());
        }
        let device = executor.get_device()?;
        Ok((0..count)
            .map(|_| Stream::new_with_device(&device))
            .collect())
    }

    fn submit<'a, I>(
        _executor: &Self::Executor,
        values: I,
    ) -> Result<Self::Completion, safemlx::error::Exception>
    where
        Array: 'a,
        I: IntoIterator<Item = &'a Array>,
    {
        Ok(MlxSubmissionCompletion::new(
            safemlx::transforms::async_eval_with_event(values)?,
        ))
    }

    fn order_after(
        completion: &Self::Completion,
        executor: &Self::Executor,
    ) -> Result<(), safemlx::error::Exception> {
        completion.wait_on(executor)
    }

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

    fn share_materialized_weight(
        weight: &Self::MaterializedWeight,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(weight.clone())
    }

    fn validate_bind(
        parameter: &Self::Parameter,
        weight: &Self::MaterializedWeight,
    ) -> Result<(), Self::ParameterError> {
        if parameter.shape() != weight.shape() {
            return Err(MlxParameterError::Mlx(safemlx::error::Exception::custom(
                format!(
                    "parameter shape {:?} does not match materialized weight {:?}",
                    parameter.shape(),
                    weight.shape()
                ),
            )));
        }
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

impl NeuralBackend for MlxBackend {
    type Tensor = Array;
    type Linear = MlxLinear;
    type Embedding = MlxEmbedding;
    type Normalization = MlxRmsNorm;
    type Rotary = MlxRotary;
    type ParallelContext = Group;

    fn linear(spec: LinearSpec, context: &Stream) -> Result<MlxLinear, ComputeError> {
        let module = compute(common::linear::PhysicalLinear::unloaded(
            spec.input,
            spec.output,
            spec.bias.is_some(),
            spec.format,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: None,
        })
    }

    fn embedding(spec: EmbeddingSpec, context: &Stream) -> Result<MlxEmbedding, ComputeError> {
        let module = compute(common::linear::unloaded_maybe_quantized_embedding(
            spec.vocabulary,
            spec.dimensions,
            spec.quantization,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None)?;
        Ok(MlxEmbedding {
            module,
            topology,
            vocabulary: spec.vocabulary,
            vocabulary_range: None,
        })
    }

    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        context: &Stream,
    ) -> Result<MlxEmbedding, ComputeError> {
        range.validate_global_rows(spec.vocabulary)?;
        let global = i32::try_from(range.global_vocabulary).map_err(ComputeError::backend)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend)?;
        let module = compute(common::linear::unloaded_maybe_quantized_embedding(
            local,
            spec.dimensions,
            spec.quantization,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None)?;
        Ok(MlxEmbedding {
            module,
            topology,
            vocabulary: global,
            vocabulary_range: Some(range),
        })
    }

    fn vocabulary_parallel_linear(
        spec: LinearSpec,
        range: VocabularyParallelRange,
        context: &Stream,
    ) -> Result<MlxLinear, ComputeError> {
        range.validate_global_rows(spec.output)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend)?;
        let module = compute(common::linear::PhysicalLinear::unloaded(
            spec.input,
            local,
            spec.bias.is_some(),
            spec.format,
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: Some(range),
        })
    }

    fn vocabulary_parallel_lookup(
        embedding: &mut MlxEmbedding,
        input: &Array,
        policy: EmbeddingLookupPolicy,
        parallel: &Group,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        policy.validate()?;
        let range = embedding
            .vocabulary_range
            .as_ref()
            .ok_or_else(|| ComputeError::backend("embedding has no vocabulary ownership"))?;
        let sentinel = match policy {
            EmbeddingLookupPolicy::Strict => None,
            EmbeddingLookupPolicy::ZeroSentinel(sentinel) => Some(sentinel),
        };
        let input = compute(validate_token_domain(
            input,
            embedding.vocabulary,
            sentinel,
            context,
        ))?;
        let start =
            Array::from_int(i32::try_from(range.local.start).map_err(ComputeError::backend)?);
        let end = Array::from_int(i32::try_from(range.local.end).map_err(ComputeError::backend)?);
        let valid = compute(input.ge(&start, context))?
            .logical_and(&compute(input.lt(&end, context))?, context)
            .map_err(ComputeError::backend)?;
        let local = compute(input.subtract(&start, context))?;
        let safe = compute(safemlx::ops::r#where(
            &valid,
            &local,
            Array::from_int(0),
            context,
        ))?;
        let value = compute(embedding.module.forward(&safe, context))?;
        let mask = compute(valid.expand_dims(-1, context))?;
        let value = compute(safemlx::ops::r#where(
            &mask,
            &value,
            value.zeros_like(context)?,
            context,
        ))?;
        compute(safemlx::distributed::all_sum(&value, parallel, context))
    }

    fn vocabulary_parallel_project(
        linear: &mut MlxLinear,
        input: &Array,
        parallel: &Group,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let range = linear
            .vocabulary_range
            .as_ref()
            .ok_or_else(|| ComputeError::backend("projection has no vocabulary ownership"))?;
        let local = compute(linear.module.forward(input, context))?;
        let widths = (0..parallel.size())
            .map(|rank| {
                eredu_core::balanced_contiguous_range(
                    range.global_vocabulary,
                    parallel.size(),
                    rank,
                    false,
                )
                .map(|range| range.len())
                .map_err(ComputeError::backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        compute(safemlx::distributed::all_gather_uneven_axis(
            &local, -1, &widths, parallel, context,
        ))
    }

    fn rms_norm(spec: NormalizationSpec, context: &Stream) -> Result<MlxRmsNorm, ComputeError> {
        Self::normalization(spec.into(), context)
    }

    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &Stream,
    ) -> Result<MlxRmsNorm, ComputeError> {
        spec.validate()?;
        let (weight, offset) = match spec.scale {
            NormalizationScale::Learned(weight) => (Some(weight), None),
            NormalizationScale::LearnedOffset { weight, offset } => (Some(weight), Some(offset)),
            NormalizationScale::Unit => (None, None),
        };
        let (module, topology) = match weight {
            Some(weight) => {
                let module = compute(nn::RmsNorm::unloaded(
                    spec.dimensions,
                    spec.epsilon,
                    Dtype::Float32,
                    context,
                ))?;
                let topology = parameter_topology(&module, weight, None)?;
                (Some(module), topology)
            }
            None => (None, BTreeMap::new()),
        };
        Ok(MlxRmsNorm {
            module,
            topology,
            offset,
            dimensions: spec.dimensions,
            epsilon: spec.epsilon,
        })
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

    fn gelu_approximate(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(nn::gelu_approximate(input, context))
    }

    fn sigmoid(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(safemlx::ops::sigmoid(input, context))
    }

    fn softplus(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(nn::softplus(input, context))
    }

    fn exp(input: Array, context: &Stream) -> Result<Array, ComputeError> {
        compute(safemlx::ops::exp(input, context))
    }

    fn gated_group_rms_norm(
        input: &Array,
        gate: &Array,
        weight: &Array,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let dtype = input.dtype();
        let shape = input.shape().to_vec();
        let width = *shape
            .last()
            .ok_or_else(|| ComputeError::backend("gated RMS input has no feature axis"))?;
        if groups <= 0 || width % groups != 0 || gate.shape() != shape || weight.shape() != [width]
        {
            return Err(ComputeError::backend(
                "invalid gated grouped RMS normalization geometry",
            ));
        }
        let input = compute(input.as_dtype(Dtype::Float32, context))?;
        let gate = compute(gate.as_dtype(Dtype::Float32, context))?;
        let gate =
            compute(gate.multiply(compute(safemlx::ops::sigmoid(&gate, context))?, context))?;
        let gated = compute(input.multiply(&gate, context))?;
        let grouped = compute(gated.reshape(&[-1, groups, width / groups], context))?;
        let variance = compute(safemlx::ops::mean_axis(
            compute(grouped.square(context))?,
            -1,
            true,
            context,
        ))?;
        let scale = compute(safemlx::ops::rsqrt(
            compute(variance.add(Array::from_f32(epsilon), context))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(&shape, context))?;
        let normalized = compute(normalized.as_dtype(dtype, context))?;
        compute(normalized.multiply(weight, context))
    }

    fn l2_normalize(input: &Array, epsilon: f32, context: &Stream) -> Result<Array, ComputeError> {
        if input.shape().last().is_none() || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ComputeError::backend("invalid L2 normalization geometry"));
        }
        let squared = compute(input.square(context))?;
        let sum = compute(safemlx::ops::sum_axis(&squared, -1, true, context))?;
        let denominator = compute(sum.add(Array::from_f32(epsilon), context))?;
        compute(input.multiply(compute(denominator.rsqrt(context))?, context))
    }

    fn silu_gated_group_rms_norm(
        input: &Array,
        gate: &Array,
        weight: &Array,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let dtype = input.dtype();
        let shape = input.shape().to_vec();
        let width = *shape
            .last()
            .ok_or_else(|| ComputeError::backend("gated RMS input has no feature axis"))?;
        if groups <= 0
            || width % groups != 0
            || gate.shape() != shape
            || weight.shape() != [width]
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(ComputeError::backend(
                "invalid SiLU-gated grouped RMS normalization geometry",
            ));
        }
        let input = compute(input.as_dtype(Dtype::Float32, context))?;
        let grouped = compute(input.reshape(&[-1, groups, width / groups], context))?;
        let variance = compute(safemlx::ops::mean_axis(
            compute(grouped.square(context))?,
            -1,
            true,
            context,
        ))?;
        let scale = compute(safemlx::ops::rsqrt(
            compute(variance.add(Array::from_f32(epsilon), context))?,
            context,
        ))?;
        let normalized = compute(grouped.multiply(&scale, context))?;
        let normalized = compute(normalized.reshape(&shape, context))?;
        let normalized = compute(normalized.multiply(weight, context))?;
        let gate = compute(gate.as_dtype(Dtype::Float32, context))?;
        let gate =
            compute(gate.multiply(compute(safemlx::ops::sigmoid(&gate, context))?, context))?;
        compute(normalized.multiply(&gate, context))?
            .as_dtype(dtype, context)
            .map_err(ComputeError::backend)
    }

    fn segmented_attention(
        input: SegmentedAttentionInput<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        input.validate()?;
        let heads = input.queries.dim(1);
        let query_dimensions = input.queries.dim(2);
        let value_dimensions = input.values.dim(2);
        let mut outputs = Vec::with_capacity(input.segment_lengths.len());
        let mut start = 0i32;
        for &length in input.segment_lengths {
            let end = start + length;
            let prepare = |value: &Array, dimensions: i32| -> Result<Array, ComputeError> {
                let value = compute(value.try_index_device((start..end, .., ..), context))?;
                let value = compute(value.transpose_axes(&[1, 0, 2], context))?;
                compute(value.reshape(&[1, heads, length, dimensions], context))
            };
            let queries = prepare(input.queries, query_dimensions)?;
            let keys = prepare(input.keys, query_dimensions)?;
            let values = prepare(input.values, value_dimensions)?;
            let output = compute(safemlx::fast::scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                input.scale,
                Option::<ScaledDotProductAttentionMask<'_>>::None,
                Option::<&Array>::None,
                context,
            ))?;
            let output = compute(output.reshape(&[heads, length, value_dimensions], context))?;
            outputs.push(compute(output.transpose_axes(&[1, 0, 2], context))?);
            start = end;
        }
        compute(concatenate_axis(&outputs, 0, context))
    }

    fn add_residual(
        residual: &Array,
        branch: &Array,
        fp32: bool,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        if fp32 {
            let residual = compute(residual.as_dtype(Dtype::Float32, context))?;
            let branch = compute(branch.as_dtype(Dtype::Float32, context))?;
            compute(residual.add(branch, context))
        } else {
            compute(residual.add(branch, context))
        }
    }

    fn gated_delta_scan(
        input: GatedDeltaScanInput<'_, Array>,
        context: &Stream,
    ) -> Result<GatedDeltaScanOutput<Array>, ComputeError> {
        let (state, output) = compute(crate::backend::mlx::nn::gated_delta::gated_delta_scan(
            input.query,
            input.key,
            input.value,
            input.log_decay,
            input.beta,
            input.initial_state.cloned(),
            context,
        ))?;
        Ok(GatedDeltaScanOutput { state, output })
    }

    fn selective_state_space_scan(
        input: SelectiveStateSpaceScanInput<'_, Array>,
        context: &Stream,
    ) -> Result<SelectiveStateSpaceScanOutput<Array>, ComputeError> {
        let shape = input.values.shape();
        if shape.len() != 4 || input.chunk_size == 0 {
            return Err(ComputeError::backend(
                "selective state-space scan expects rank-four values and nonzero chunks",
            ));
        }
        let [batch, sequence, heads, head_dimensions] =
            <[i32; 4]>::try_from(shape).expect("validated rank-four shape");
        let state_dimensions = input.input_state.dim(3);
        let mut state = match input.initial_state {
            Some(state) => compute(state.as_dtype(Dtype::Float32, context))?,
            None => compute(safemlx::ops::zeros::<f32>(
                &[batch, heads, head_dimensions, state_dimensions],
                context,
            ))?,
        };
        let values = compute(input.values.as_dtype(Dtype::Float32, context))?;
        let input_state = compute(input.input_state.as_dtype(Dtype::Float32, context))?;
        let output_state = compute(input.output_state.as_dtype(Dtype::Float32, context))?;
        let transition = compute(safemlx::ops::exp(
            compute(input.transition_log.as_dtype(Dtype::Float32, context))?,
            context,
        ))?;
        let transition = compute(transition.multiply(Array::from_f32(-1.0), context))?
            .reshape(&[1, heads, 1, 1], context)
            .map_err(ComputeError::backend)?;
        let skip = compute(input.skip.as_dtype(Dtype::Float32, context))?
            .reshape(&[1, heads, 1], context)
            .map_err(ComputeError::backend)?;
        let bias = compute(input.time_step_bias.as_dtype(Dtype::Float32, context))?
            .reshape(&[1, 1, heads], context)
            .map_err(ComputeError::backend)?;
        let mut outputs = Vec::with_capacity(sequence as usize);
        let chunk = i32::try_from(input.chunk_size).unwrap_or(i32::MAX).max(1);
        let mut chunk_start = 0;
        while chunk_start < sequence {
            let chunk_end = (chunk_start + chunk).min(sequence);
            for token in chunk_start..chunk_end {
                let value = compute(values.try_index_device((.., token, .., ..), context))?;
                let b = compute(input_state.try_index_device((.., token, .., ..), context))?;
                let c = compute(output_state.try_index_device((.., token, .., ..), context))?;
                let dt = compute(
                    input
                        .time_step
                        .try_index_device((.., token..token + 1, ..), context),
                )?;
                let dt = compute(dt.add(&bias, context))?;
                let dt = compute(nn::softplus(dt, context))?;
                let floor = Array::from_f32(input.time_step_floor);
                let dt = compute(maximum(dt, floor, context))?;
                let dt = compute(dt.as_dtype(Dtype::Float32, context))?
                    .reshape(&[batch, heads], context)
                    .map_err(ComputeError::backend)?;
                let dt_transition = compute(dt.reshape(&[batch, heads, 1, 1], context))?;
                let decay = compute(safemlx::ops::exp(
                    compute(dt_transition.multiply(&transition, context))?,
                    context,
                ))?;
                let dt_input = compute(dt.reshape(&[batch, heads, 1], context))?;
                let discretized_b = compute(dt_input.multiply(&b, context))?;
                let value_column = compute(value.try_index_device((.., .., .., NewAxis), context))?;
                let input_row =
                    compute(discretized_b.try_index_device((.., .., NewAxis, ..), context))?;
                let update = compute(value_column.multiply(&input_row, context))?;
                state = compute(compute(state.multiply(&decay, context))?.add(&update, context))?;
                let output_row = compute(c.try_index_device((.., .., NewAxis, ..), context))?;
                let projected = compute(safemlx::ops::sum_axis(
                    compute(state.multiply(&output_row, context))?,
                    -1,
                    false,
                    context,
                ))?;
                let output =
                    compute(projected.add(&compute(value.multiply(&skip, context))?, context))?;
                outputs.push(compute(
                    output.try_index_device((.., NewAxis, .., ..), context),
                )?);
            }
            chunk_start = chunk_end;
        }
        let output = compute(safemlx::ops::concatenate_axis(&outputs, 1, context))?;
        let output = compute(output.as_dtype(input.values.dtype(), context))?;
        Ok(SelectiveStateSpaceScanOutput { state, output })
    }

    fn gated_product(
        mut gate: Array,
        mut up: Array,
        policy: GatedProductPolicy,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        policy.validate()?;
        if let Some(bound) = policy.gate_upper_bound() {
            gate = compute(safemlx::ops::minimum(gate, Array::from_f32(bound), context))?;
        }
        if let Some(bound) = policy.up_absolute_bound() {
            up = compute(safemlx::ops::clip(up, (-bound, bound), context))?;
        }
        if policy.up_offset() != 0.0 {
            up = compute(up.add(Array::from_f32(policy.up_offset()), context))?;
        }
        let gate = match policy.activation() {
            eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
                compute(common::layers::silu(gate, context))?
            }
            eredu_nn::GatedProductActivation::Silu => {
                let scaled =
                    compute(gate.multiply(Array::from_f32(policy.sigmoid_multiplier()), context))?;
                let probability = compute(sigmoid(scaled, context))?;
                compute(gate.multiply(probability, context))?
            }
            eredu_nn::GatedProductActivation::GeluApproximate => {
                compute(nn::gelu_approximate(gate, context))?
            }
        };
        compute(gate.multiply(up, context))
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

    fn relative_attention(
        input: RelativeAttentionInput<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        input.validate()?;
        let query_shape = input.queries.shape();
        let key_shape = input.keys.shape();
        let batch = query_shape[0];
        let heads = query_shape[1];
        let query_len = query_shape[2];
        let dimensions = query_shape[3];
        let kv_heads = key_shape[1];
        let key_len = key_shape[2];
        let repeats = heads / kv_heads;
        let repeat_kv = |value: &Array| -> Result<Array, ComputeError> {
            if repeats == 1 {
                return Ok(value.clone());
            }
            let expanded =
                compute(value.reshape(&[batch, kv_heads, 1, key_len, dimensions], context))?;
            let expanded = compute(broadcast_to(
                &expanded,
                &[batch, kv_heads, repeats, key_len, dimensions],
                context,
            ))?;
            compute(expanded.reshape(&[batch, heads, key_len, dimensions], context))
        };
        let keys = repeat_kv(input.keys)?;
        let values = repeat_kv(input.values)?;
        let query_positions = compute(arange::<i32, i32>(
            input.query_offset,
            input.query_offset + query_len,
            1,
            context,
        ))?;
        let query_positions = compute(query_positions.try_index_device((.., NewAxis), context))?;
        let key_positions = compute(arange::<i32, i32>(
            input.key_offset,
            input.key_offset + key_len,
            1,
            context,
        ))?;
        let key_positions = compute(key_positions.try_index_device((NewAxis, ..), context))?;
        let distances = compute(query_positions.subtract(key_positions, context))?;
        let mut valid = compute(distances.ge(Array::from_int(0), context))?;
        if let Some(window) = input.window {
            valid = compute(valid.logical_and(
                &compute(distances.lt(Array::from_int(window), context))?,
                context,
            ))?;
        }
        let extent = input.profiles.dim(3);
        let gather = compute(clip(&distances, (0, extent - 1), context))?;
        let gather = compute(gather.as_dtype(Dtype::Int32, context))?;
        let gather = compute(gather.try_index_device((NewAxis, NewAxis, .., ..), context))?;
        let gather = compute(broadcast_to(
            &gather,
            &[batch, heads, query_len, key_len],
            context,
        ))?;
        let mut bias = compute(take_along_axis(input.profiles, &gather, -1, context))?;
        let relative_valid = compute(
            compute(distances.ge(Array::from_int(0), context))?.logical_and(
                &compute(distances.lt(Array::from_int(extent), context))?,
                context,
            ),
        )?;
        bias = compute(r#where(
            &relative_valid,
            bias,
            Array::from_f32(0.0),
            context,
        ))?;
        let mut queries = input.queries.clone();
        if input.window.is_none() {
            if let Some(floor) = input.log_scaling_floor {
                let positions = compute(arange::<i32, i32>(
                    input.query_offset + 1,
                    input.query_offset + query_len + 1,
                    1,
                    context,
                ))?;
                let positions = compute(positions.as_dtype(Dtype::Float32, context))?;
                let ratio = compute(positions.divide(Array::from_f32(floor as f32), context))?;
                let ratio = compute(maximum(ratio, Array::from_f32(1.0), context))?;
                let tau = compute(ratio.log(context))?;
                let tau = compute(tau.multiply(Array::from_f32(input.log_scaling_alpha), context))?;
                let tau = compute(tau.add(Array::from_f32(1.0), context))?;
                let tau = compute(tau.reshape(&[1, 1, query_len, 1], context))?;
                queries = compute(queries.multiply(&tau, context))?;
                bias = compute(bias.multiply(&tau, context))?;
            }
        }
        let scaled = compute(queries.multiply(Array::from_f32(1.0 / dimensions as f32), context))?;
        let scores = compute(matmul(
            &scaled,
            &compute(keys.swap_axes(-1, -2, context))?,
            context,
        ))?;
        let scores = compute(scores.add(bias, context))?;
        let scores = compute(r#where(
            &valid,
            scores,
            Array::from_f32(f32::NEG_INFINITY),
            context,
        ))?;
        let probabilities = compute(softmax_axis(scores, -1, true, context))?;
        compute(matmul(probabilities, values, context))
    }

    fn joint_expert_routing(
        input: JointExpertRoutingInput<'_, Array>,
        context: &Stream,
    ) -> Result<JointExpertRoutingResult<Array>, ComputeError> {
        input.validate()?;
        let hidden_width = input.hidden.dim(-1);
        let flat = compute(input.hidden.reshape(&[-1, hidden_width], context))?;
        let logits = compute(matmul(
            &flat,
            &compute(input.weight.transpose(context))?,
            context,
        ))?;
        let routed = compute(logits.try_index_device((.., ..input.routed_experts), context))?;
        let shared = compute(logits.try_index_device((.., input.routed_experts..), context))?;
        let choice = compute(sigmoid(&routed, context))?;
        let choice = compute(choice.add(input.correction_bias, context))?;
        let routed_ids = compute(argpartition_axis(choice, -input.top_k, -1, context))?;
        let routed_ids = compute(routed_ids.try_index_device((.., -input.top_k..), context))?;
        let selected_logits = compute(take_along_axis(&routed, &routed_ids, -1, context))?;
        let all_logits = compute(concatenate_axis(&[selected_logits, shared], -1, context))?;
        let weights = compute(nn::log_sigmoid(all_logits, context))?;
        let weights = compute(softmax_axis(weights, -1, true, context))?;
        let weights = compute(weights.multiply(Array::from_f32(input.route_scale), context))?;
        let weights = compute(weights.multiply(input.global_scale, context))?;
        let routed_weights = compute(weights.try_index_device((.., ..input.top_k), context))?;
        let shared_weights = compute(weights.try_index_device((.., input.top_k..), context))?;
        Ok(JointExpertRoutingResult {
            routed_ids,
            routed_weights,
            shared_weights,
        })
    }

    fn indexed_attention(
        input: IndexedAttentionInput<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        input.validate()?;
        compute(common::attention::indexed_sparse_attention(
            input.queries,
            input.local_keys,
            input.local_values,
            input.pooled_keys,
            input.pooled_values,
            input.selected_positions,
            input.scale,
            input.local_mask,
            input.pooled_mask,
            input.sinks,
            context,
        ))
    }

    fn pooled_attention(
        input: PooledAttentionInput<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let query_tokens = input.queries.dim(2);
        let local_tokens = input.local.dim(1);
        let pooled_tokens = input.pooled.dim(1);
        let local = compute(input.local.expand_dims(1, context))?;
        let pooled = compute(input.pooled.expand_dims(1, context))?;
        let keys = compute(safemlx::ops::concatenate_axis(&[local, pooled], 2, context))?;
        let mask = if input.local_mask.is_none() && input.pooled_mask.is_none() {
            None
        } else {
            let local = match input.local_mask {
                Some(mask) => mask.clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, local_tokens], context))?,
            };
            let pooled = match input.pooled_mask {
                Some(mask) => mask.clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, pooled_tokens], context))?,
            };
            Some(compute(safemlx::ops::concatenate_axis(
                &[local, pooled],
                -1,
                context,
            ))?)
        };
        compute(safemlx::fast::scaled_dot_product_attention(
            input.queries,
            &keys,
            &keys,
            input.scale,
            mask.as_ref().map(ScaledDotProductAttentionMask::Array),
            input.sinks,
            context,
        ))
    }

    fn select_pooled_positions(
        input: PooledPositionInput<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        let query_shape = input.queries.shape();
        let pooled_shape = input.pooled_keys.shape();
        let weight_shape = input.head_weights.shape();
        if query_shape.len() != 4
            || pooled_shape.len() != 3
            || weight_shape.len() != 3
            || query_shape[0] != pooled_shape[0]
            || query_shape[0] != weight_shape[0]
            || query_shape[1] != weight_shape[2]
            || query_shape[2] != weight_shape[1]
            || query_shape[3] != pooled_shape[2]
            || input.top_k <= 0
            || !input.scale.is_finite()
            || input.scale <= 0.0
            || !input.head_scale.is_finite()
            || input.head_scale <= 0.0
        {
            return Err(ComputeError::backend(format!(
                "invalid pooled-position geometry: queries={query_shape:?} pooled={pooled_shape:?} weights={weight_shape:?} top_k={}",
                input.top_k
            )));
        }
        let scores = compute(einsum(
            "bhld,bpd->bhlp",
            [
                &compute(input.queries.as_dtype(Dtype::Float32, context))?,
                &compute(input.pooled_keys.as_dtype(Dtype::Float32, context))?,
            ],
            context,
        ))?;
        let scores = compute(maximum(scores, Array::from_f32(0.0), context))?;
        let scores = compute(scores.multiply(Array::from_f32(input.scale), context))?;
        let weights = compute(input.head_weights.as_dtype(Dtype::Float32, context))?;
        let weights = compute(weights.multiply(Array::from_f32(input.head_scale), context))?;
        let weights = compute(weights.transpose_axes(&[0, 2, 1], context))?;
        let weights = compute(weights.expand_dims(-1, context))?;
        let mut scores = compute(scores.multiply(weights, context))?;
        scores = compute(scores.sum_axis(1, false, context))?;
        if let Some(mask) = input.mask {
            scores = compute(safemlx::ops::r#where(
                mask,
                scores,
                Array::from_f32(f32::NEG_INFINITY),
                context,
            ))?;
        }
        let top_k = input.top_k.min(pooled_shape[1]);
        let indices = compute(argpartition_axis(&scores, -top_k, -1, context))?;
        let mut indexes = vec![eredu_nn::Index::Full; indices.ndim()];
        let last = indexes.len() - 1;
        indexes[last] = eredu_nn::Index::Range(indices.dim(-1) - top_k, indices.dim(-1));
        indices.index(&indexes, context)
    }

    fn gather_pooled_mask(
        mask: &Array,
        selected_positions: &Array,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        if mask.ndim() != 2 || selected_positions.ndim() != 3 {
            return Err(ComputeError::backend(format!(
                "pooled mask gathering expects [query, pool] and [batch, query, selected], got {:?} and {:?}",
                mask.shape(),
                selected_positions.shape()
            )));
        }
        let expanded = compute(mask.expand_dims(0, context))?;
        let expanded = compute(broadcast_to(
            &expanded,
            &[
                selected_positions.dim(0),
                selected_positions.dim(1),
                mask.dim(1),
            ],
            context,
        ))?;
        let selected = compute(take_along_axis(&expanded, selected_positions, 2, context))?;
        compute(selected.expand_dims(1, context))
    }

    fn attention_with_sinks(
        request: AttentionRequest<'_, Array>,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        request.validate()?;
        compute(safemlx::fast::scaled_dot_product_attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            request.mask.map(ScaledDotProductAttentionMask::Array),
            request.sinks,
            context,
        ))
    }

    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, Array>,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        request.validate()?;
        let batch = request.queries.dim(0);
        let sequence = request.queries.dim(2);
        compute(common::attention::sliding_window_prefill_attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            window,
            position_offset,
            batch,
            sequence,
            request.sinks,
            context,
        ))
    }

    fn rms_norm_without_weight(
        input: &Array,
        epsilon: f32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        mlx_weightless_rms_norm(input, epsilon, context)
    }

    fn grouped_linear(
        linear: &mut MlxLinear,
        input: &Array,
        groups: i32,
        output_per_group: i32,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        if input.ndim() != 4 || input.dim(1) != groups || groups <= 0 || output_per_group <= 0 {
            return Err(ComputeError::backend(format!(
                "grouped linear expects [batch, {groups}, tokens, input] and positive output width, got {:?}",
                input.shape()
            )));
        }
        let projected = compute(linear.module.forward(input, context))?;
        if projected.dim(-1) != groups * output_per_group {
            return Err(ComputeError::backend(format!(
                "grouped linear produced width {}, expected {}",
                projected.dim(-1),
                groups * output_per_group
            )));
        }
        let mut pieces = Vec::with_capacity(groups as usize);
        for group in 0..groups {
            let selected = projected.index(
                &[
                    eredu_nn::Index::Full,
                    eredu_nn::Index::At(group),
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Range(
                        group * output_per_group,
                        (group + 1) * output_per_group,
                    ),
                ],
                context,
            )?;
            pieces.push(compute(selected.expand_dims(1, context))?);
        }
        Array::concatenate(&pieces, 1, context)
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
            None,
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
        compute(linear.module.forward_row_parallel(input, parallel, context))
    }

    fn sum_parallel(
        value: Array,
        parallel: &Group,
        context: &Stream,
    ) -> Result<Array, ComputeError> {
        compute(safemlx::distributed::all_sum(&value, parallel, context))
    }

    fn parallel_size(parallel: &Group) -> usize {
        parallel.size()
    }
}

impl HyperNeuralBackend for MlxBackend {
    type HyperConnection = MlxHyperConnection;
    type HyperHead = MlxHyperHead;

    fn hyper_connection(
        spec: HyperConnectionSpec,
        context: &Stream,
    ) -> Result<Self::HyperConnection, ComputeError> {
        spec.validate()?;
        let module = compute(common::hyper_connections::HyperConnection::unloaded(
            spec.streams,
            spec.hidden_size,
            spec.sinkhorn_iterations,
            spec.epsilon,
            context,
        ))?;
        let topology = exact_parameter_topology(
            &module,
            [
                ("function", spec.function),
                ("base", spec.base),
                ("scale", spec.scale),
            ],
        )?;
        Ok(MlxHyperConnection { module, topology })
    }

    fn hyper_head(spec: HyperHeadSpec, context: &Stream) -> Result<Self::HyperHead, ComputeError> {
        spec.validate()?;
        let module = compute(common::hyper_connections::HyperHead::unloaded(
            spec.streams,
            spec.hidden_size,
            spec.norm_epsilon,
            spec.epsilon,
            context,
        ))?;
        let topology = exact_parameter_topology(
            &module,
            [
                ("function", spec.function),
                ("base", spec.base),
                ("scale", spec.scale),
            ],
        )?;
        Ok(MlxHyperHead { module, topology })
    }
}

impl RoutedNeuralBackend for MlxBackend {
    type Router = MlxTopKRouter;
    type GatedProductExpertBank = MlxGatedProductExpertBank;
    type Relu2ExpertBank = MlxRelu2ExpertBank;

    fn top_k_router(spec: TopKRouterSpec, context: &Stream) -> Result<Self::Router, ComputeError> {
        spec.validate()?;
        let score_function = match spec.routing.scoring() {
            RoutingScoring::Softmax => common::moe::TopKRouterScoreFunction::Softmax,
            RoutingScoring::SelectedSoftmax => {
                common::moe::TopKRouterScoreFunction::SelectedSoftmax
            }
            RoutingScoring::Sigmoid => common::moe::TopKRouterScoreFunction::Sigmoid,
            RoutingScoring::SqrtSoftplus => common::moe::TopKRouterScoreFunction::SqrtSoftplus,
        };
        let module = compute(common::moe::TopKRouter::new_with_quantization(
            common::moe::TopKRouterConfig {
                top_k: spec.routing.top_k(),
                num_experts: spec.routing.expert_count(),
                hidden_size: spec.input_dimensions,
                score_function,
                norm_topk_prob: spec.routing.normalize_selected(),
                normalization_epsilon: spec.routing.normalization_epsilon(),
                routed_scaling_factor: spec.routing.routed_scaling(),
                n_group: spec.routing.expert_groups(),
                topk_group: spec.routing.selected_groups(),
                projection_bias: spec.bias.is_some(),
                score_correction_bias: spec.correction_bias.is_some(),
                input_rms_epsilon: spec
                    .input_transform
                    .as_ref()
                    .map(|transform| transform.epsilon),
                input_inverse_sqrt_dimensions: spec
                    .input_transform
                    .as_ref()
                    .is_some_and(|transform| transform.inverse_sqrt_dimensions),
                route_scale: spec.route_scale.is_some(),
            },
            spec.quantization,
            context,
        ))?;
        let weight = spec.weight;
        let mut topology = vec![("weight", weight.clone())];
        if let Some(bias) = spec.bias {
            topology.push(("bias", bias));
        }
        if let Some(quantization) = spec
            .quantization
            .filter(|format| !matches!(format, WeightQuantization::GgufIQuant { .. }))
        {
            topology.push(("scales", companion_spec(&weight, "scales")));
            if quantization.has_biases() {
                topology.push(("biases", companion_spec(&weight, "biases")));
            }
        }
        if let Some(correction_bias) = spec.correction_bias {
            topology.push(("e_score_correction_bias", correction_bias));
        }
        if let Some(transform) = spec.input_transform {
            topology.push(("input_scale", transform.scale));
        }
        if let Some(route_scale) = spec.route_scale {
            topology.push(("route_scale", route_scale));
        }
        Ok(MlxTopKRouter {
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn gated_product_expert_bank(
        spec: GatedProductExpertBankSpec,
        context: &Stream,
    ) -> Result<Self::GatedProductExpertBank, ComputeError> {
        spec.validate()?;
        if spec.input_dimensions != spec.output_dimensions {
            return Err(ComputeError::backend(
                "MLX packed gated-product experts require equal input and output dimensions",
            ));
        }
        let policy = spec.policy;
        let GatedProductExpertLayout::Packed { gate_up, down } = spec.layout else {
            return Err(ComputeError::backend(
                "independent expert units must be acquired through a runtime expert provider",
            ));
        };
        let prefix = gate_up
            .weight
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or_else(|| {
                ComputeError::backend("packed gate/up identity must end in .gate_up_proj")
            })?;
        let expected_down = format!("{prefix}.down_proj");
        if down.weight.id.as_str() != expected_down {
            return Err(ComputeError::backend(format!(
                "packed down identity {:?} does not match {expected_down:?}",
                down.weight.id.as_str()
            )));
        }
        let native_fp8 = match (gate_up.format, down.format) {
            (LinearFormat::E4M3BlockFp8(gate), LinearFormat::E4M3BlockFp8(down))
                if gate == down
                    && gate.scale_encoding == eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 =>
            {
                true
            }
            (LinearFormat::E4M3BlockFp8(_), LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "MLX packed block-FP8 experts require matching UE8M0 formats",
                ));
            }
            (LinearFormat::E4M3BlockFp8(_), _) | (_, LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "packed expert projections must use one physical format",
                ));
            }
            _ => false,
        };
        let mut module = compute(common::moe::PackedGatedProductExperts::new(
            spec.expert_count,
            spec.input_dimensions,
            spec.intermediate_dimensions,
            gate_up.format.weight_quantization(),
            down.format.weight_quantization(),
            [gate_up.bias.is_some(), down.bias.is_some()],
            context,
        ))?;
        module = compute(module.with_policy(policy))?;
        if native_fp8 {
            module = compute(module.with_native_fp8_e8m0(context))?;
        }
        let mut topology = vec![
            ("gate_up_proj", gate_up.weight.clone()),
            ("down_proj", down.weight.clone()),
        ];
        if let Some(bias) = gate_up.bias {
            topology.push(("gate_up_proj_bias", bias));
        }
        if let Some(bias) = down.bias {
            topology.push(("down_proj_bias", bias));
        }
        if native_fp8
            || gate_up
                .format
                .weight_quantization()
                .is_some_and(|format| !matches!(format, WeightQuantization::GgufIQuant { .. }))
        {
            topology.push((
                "gate_up_proj_scales",
                packed_expert_companion_spec(&gate_up.weight, "scales"),
            ));
        }
        if gate_up
            .format
            .weight_quantization()
            .is_some_and(WeightQuantization::has_biases)
        {
            topology.push((
                "gate_up_proj_biases",
                packed_expert_companion_spec(&gate_up.weight, "biases"),
            ));
        }
        if native_fp8
            || down
                .format
                .weight_quantization()
                .is_some_and(|format| !matches!(format, WeightQuantization::GgufIQuant { .. }))
        {
            topology.push((
                "down_proj_scales",
                packed_expert_companion_spec(&down.weight, "scales"),
            ));
        }
        if down
            .format
            .weight_quantization()
            .is_some_and(WeightQuantization::has_biases)
        {
            topology.push((
                "down_proj_biases",
                packed_expert_companion_spec(&down.weight, "biases"),
            ));
        }
        Ok(MlxGatedProductExpertBank {
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn relu2_expert_bank(
        spec: Relu2ExpertBankSpec,
        context: &Stream,
    ) -> Result<Self::Relu2ExpertBank, ComputeError> {
        spec.validate()?;
        let prefix = spec
            .up
            .weight
            .id
            .as_str()
            .strip_suffix(".up_proj")
            .ok_or_else(|| {
                ComputeError::backend("packed ReLU2 up identity must end in .up_proj")
            })?;
        if spec.down.weight.id.as_str() != format!("{prefix}.down_proj") {
            return Err(ComputeError::backend(
                "packed ReLU2 projections must share one parameter prefix",
            ));
        }
        let module = compute(common::moe::PackedRelu2Experts::new(
            spec.expert_count,
            spec.hidden_dimensions,
            spec.intermediate_dimensions,
            [
                spec.up.format.weight_quantization(),
                spec.down.format.weight_quantization(),
            ],
            context,
        ))?;
        Ok(MlxRelu2ExpertBank {
            module: MlxParameterTree::new(module, prefix)?,
        })
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
                request: AttentionRequest<'_, Array>,
                context: &Stream,
            ) -> Result<Array, ComputeError> {
                request.validate()?;
                if let Some(output) = compute(KeyValueCache::paged_attention(
                    self,
                    &request.queries,
                    request.scale,
                    request.mask,
                    request.sinks,
                    context,
                ))? {
                    return Ok(output);
                }
                compute(safemlx::fast::scaled_dot_product_attention(
                    request.queries,
                    request.keys,
                    request.values,
                    request.scale,
                    request.mask.map(ScaledDotProductAttentionMask::Array),
                    request.sinks,
                    context,
                ))
            }
        }
    };
}

impl_attention_cache!(ConcatKeyValueCache);
impl_attention_cache!(SlidingKeyValueCache);
impl_attention_cache!(PagedKeyValueCache);
impl_attention_cache!(crate::backend::mlx::runtime::cache::state::MlxKeyValueLayerState);

impl eredu_nn::AuxiliaryConvolutionState<Array>
    for crate::backend::mlx::runtime::cache::state::MlxHybridLayerState
{
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<Array>, ComputeError> {
        eredu_runtime::RuntimeStateComponents::fixed_component(
            self,
            eredu_core::cache::StateTensorRole::Convolution { slot },
        )
        .map_err(ComputeError::backend)
    }
}

#[cfg(test)]
mod neutral_semantic_operator_tests {
    use eredu_architectures::decoder::{MultiTableEmbedding, NamedEmbeddingSpec};
    use eredu_checkpoint::{AffineQuantization, LinearFormat, WeightQuantization};
    use eredu_nn::{
        reference_expand_heads, reference_segmented_attention, EmbeddingLookupPolicy,
        EmbeddingOperator, EmbeddingSpec, FusedProjectionLayout, FusedProjectionSegment,
        HeadExpansion, JointExpertRoutingInput, LinearOperator, LinearSpec, NeuralBackend,
        NormalizationConstructionSpec, NormalizationScale, ParameterSpec, RelativeAttentionInput,
        SegmentedAttentionInput, Tensor,
    };
    use safemlx::{
        ops::{quantize_with_mode, QuantizationMode},
        quantization::MaybeQuantized,
        transforms::async_eval_with_event,
        Array, Device, DeviceType, ExecutionContext,
    };

    use crate::backend::mlx::nn::tensor::TokenValidationScope;

    use super::{MlxBackend, MlxEmbedding, MlxLinear};

    fn close(actual: &Array, expected: &[f32], tolerance: f32) {
        let actual = actual.evaluated().unwrap();
        assert_eq!(actual.as_slice::<f32>().len(), expected.len());
        assert!(actual
            .as_slice::<f32>()
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= tolerance));
    }

    fn parameter(name: &str) -> ParameterSpec {
        ParameterSpec::trainable(name).unwrap()
    }

    fn bind_linear(
        linear: &mut MlxLinear,
        weight: Array,
        format: LinearFormat,
        stream: &safemlx::Stream,
    ) {
        match format.weight_quantization() {
            None => linear.module.weight.value = weight,
            Some(quantization) => {
                let mode = match quantization {
                    WeightQuantization::Affine(_) => QuantizationMode::Affine,
                    WeightQuantization::MxFp4 => QuantizationMode::MxFp4,
                    WeightQuantization::GgufIQuant { .. } => {
                        panic!("test helper does not synthesize GGUF blocks")
                    }
                };
                let arrays = quantize_with_mode(
                    &weight,
                    quantization.group_size(),
                    quantization.bits(),
                    mode,
                    stream,
                )
                .unwrap();
                linear.module.weight.value = arrays.weight;
                linear.module.scales.value = Some(arrays.scales);
                linear.module.biases.value = arrays.biases;
            }
        }
    }

    fn linear(
        name: &str,
        input: i32,
        output: i32,
        format: LinearFormat,
        weight: &[f32],
        stream: &safemlx::Stream,
    ) -> MlxLinear {
        let mut linear = <MlxBackend as NeuralBackend>::linear(
            LinearSpec {
                input,
                output,
                weight: parameter(name),
                bias: None,
                format,
            },
            stream,
        )
        .unwrap();
        bind_linear(
            &mut linear,
            Array::from_slice(weight, &[output, input]),
            format,
            stream,
        );
        linear
    }

    fn bind_embedding(embedding: &mut MlxEmbedding, weight: Array, stream: &safemlx::Stream) {
        match &mut embedding.module {
            MaybeQuantized::Original(embedding) => embedding.weight.value = weight,
            MaybeQuantized::Quantized(embedding) => {
                let arrays = quantize_with_mode(
                    &weight,
                    embedding.group_size,
                    embedding.bits,
                    embedding.mode,
                    stream,
                )
                .unwrap();
                embedding.inner.weight.value = arrays.weight;
                embedding.scales.value = arrays.scales;
                embedding.biases.value = arrays.biases;
            }
        }
    }

    fn embedding(
        name: &str,
        vocabulary: i32,
        dimensions: i32,
        quantization: Option<WeightQuantization>,
        weight: &[f32],
        stream: &safemlx::Stream,
    ) -> MlxEmbedding {
        let mut embedding = <MlxBackend as NeuralBackend>::embedding(
            EmbeddingSpec {
                vocabulary,
                dimensions,
                weight: parameter(name),
                quantization,
            },
            stream,
        )
        .unwrap();
        bind_embedding(
            &mut embedding,
            Array::from_slice(weight, &[vocabulary, dimensions]),
            stream,
        );
        embedding
    }

    fn supported_embedding_formats() -> [Option<WeightQuantization>; 3] {
        [
            None,
            Some(WeightQuantization::Affine(
                AffineQuantization::new(32, 4).unwrap(),
            )),
            Some(WeightQuantization::MxFp4),
        ]
    }

    #[test]
    fn hot_path_api_construction_keeps_values_backend_native() {
        fn lookup(
            embedding: &mut MlxEmbedding,
            tokens: &Array,
            stream: &safemlx::Stream,
        ) -> Result<Array, eredu_nn::Error> {
            embedding.lookup(tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
        }
        fn project(
            linear: &mut MlxLinear,
            input: &Array,
            stream: &safemlx::Stream,
        ) -> Result<Array, eredu_nn::Error> {
            linear.forward(input, stream)
        }
        fn sum(
            embeddings: &mut MultiTableEmbedding<MlxBackend>,
            tokens: &[&Array],
            stream: &safemlx::Stream,
        ) -> Result<Array, eredu_nn::Error> {
            embeddings.forward(tokens, stream)
        }

        let _: fn(&mut MlxEmbedding, &Array, &safemlx::Stream) -> Result<Array, eredu_nn::Error> =
            lookup;
        let _: fn(&mut MlxLinear, &Array, &safemlx::Stream) -> Result<Array, eredu_nn::Error> =
            project;
        let _: fn(
            &mut MultiTableEmbedding<MlxBackend>,
            &[&Array],
            &safemlx::Stream,
        ) -> Result<Array, eredu_nn::Error> = sum;
    }

    fn assert_fused_split_equivalence(format: LinearFormat) {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let input_width = 32;
        let segment_width = 2;
        let output_width = 6;
        let input = (0..input_width)
            .map(|index| (index as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let weight = (0..output_width * input_width)
            .map(|index| ((index % 19) as f32 - 9.0) / 32.0)
            .collect::<Vec<_>>();
        let input_array = Array::from_slice(&input, &[1, input_width]);
        let mut fused = linear(
            "fused.weight",
            input_width,
            output_width,
            format,
            &weight,
            stream,
        );
        let fused_output = fused.forward(&input_array, stream).unwrap();

        let mut split_outputs = Vec::new();
        for segment in 0..3 {
            let start = segment * segment_width * input_width;
            let end = start + segment_width * input_width;
            let mut split = linear(
                &format!("split.{segment}.weight"),
                input_width,
                segment_width,
                format,
                &weight[start as usize..end as usize],
                stream,
            );
            split_outputs.push(split.forward(&input_array, stream).unwrap());
        }
        let split_output = Array::concatenate(&split_outputs, -1, stream).unwrap();
        let fused_evaluated = fused_output.evaluated().unwrap();
        let split_evaluated = split_output.evaluated().unwrap();
        assert_eq!(
            fused_evaluated.as_slice::<f32>(),
            split_evaluated.as_slice::<f32>()
        );

        let qkv = FusedProjectionLayout::new([
            FusedProjectionSegment::new("query", 2).unwrap(),
            FusedProjectionSegment::new("key", 2).unwrap(),
            FusedProjectionSegment::new("value", 2).unwrap(),
        ])
        .unwrap();
        assert_eq!(qkv.split(&fused_output, stream).unwrap().len(), 3);
        let gate_up = FusedProjectionLayout::new([
            FusedProjectionSegment::new("gate", 3).unwrap(),
            FusedProjectionSegment::new("up", 3).unwrap(),
        ])
        .unwrap();
        assert_eq!(gate_up.split(&fused_output, stream).unwrap().len(), 2);

        if format == LinearFormat::Dense {
            let expected = weight
                .chunks_exact(input_width as usize)
                .map(|row| {
                    row.iter()
                        .zip(&input)
                        .map(|(left, right)| left * right)
                        .sum()
                })
                .collect::<Vec<f32>>();
            close(&fused_output, &expected, 1e-6);
        }
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_dense_fused_projection_equivalence() {
        assert_fused_split_equivalence(LinearFormat::Dense);
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_affine_fused_projection_equivalence() {
        assert_fused_split_equivalence(LinearFormat::Affine(
            AffineQuantization::new(32, 4).unwrap(),
        ));
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_mxfp4_fused_projection_equivalence() {
        assert_fused_split_equivalence(LinearFormat::MxFp4);
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_sentinel_embedding_validation() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let dimensions = 32;
        let vocabulary = 4;
        let weight = (0..vocabulary * dimensions)
            .map(|index| (index as f32 + 1.0) / 64.0)
            .collect::<Vec<_>>();
        for quantization in supported_embedding_formats() {
            let mut embedding = embedding(
                "embedding.weight",
                vocabulary,
                dimensions,
                quantization,
                &weight,
                stream,
            );
            let valid = Array::from_slice(&[-1_i32, 0, 3], &[3]);
            let scope = TokenValidationScope::begin().unwrap();
            let output = embedding
                .lookup(&valid, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                .unwrap();
            let output = output.as_type::<f32>(stream).unwrap();
            let validations = scope.finish();
            let event = async_eval_with_event(std::iter::once(&output).chain(validations.arrays()))
                .unwrap();
            event.synchronize().unwrap();
            validations.validate_completed().unwrap();
            let output = output.evaluated().unwrap();
            assert!(output.as_slice::<f32>()[..dimensions as usize]
                .iter()
                .all(|value| value.to_bits() == 0));

            for invalid in [-2_i32, vocabulary] {
                let tokens = Array::from_slice(&[invalid], &[1]);
                let scope = TokenValidationScope::begin().unwrap();
                let output = embedding
                    .lookup(&tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                    .expect("lazy lookup must not synchronize while building the graph");
                let validations = scope.finish();
                let event =
                    async_eval_with_event(std::iter::once(&output).chain(validations.arrays()))
                        .unwrap();
                event.synchronize().unwrap();
                assert!(
                    validations.validate_completed().is_err(),
                    "embedding accepted invalid token {invalid} under {quantization:?}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_multi_table_embedding_sum_is_ordered_and_sentinel_safe() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let dimensions = 32;
        for quantization in supported_embedding_formats() {
            let specs = (0..3)
                .map(|table| NamedEmbeddingSpec {
                    name: format!("stream-{table}"),
                    embedding: EmbeddingSpec {
                        vocabulary: 4,
                        dimensions,
                        weight: parameter(&format!("tables.{table}.weight")),
                        quantization,
                    },
                    lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
                })
                .collect::<Vec<_>>();
            let mut embeddings = MultiTableEmbedding::<MlxBackend>::new(specs, stream).unwrap();
            for (table, named) in embeddings.tables.iter_mut().enumerate() {
                let weight = (0..4 * dimensions)
                    .map(|index| (table as f32 + 1.0) * (index as f32 + 1.0) / 128.0)
                    .collect::<Vec<_>>();
                bind_embedding(
                    &mut named.embedding,
                    Array::from_slice(&weight, &[4, dimensions]),
                    stream,
                );
            }
            let first = Array::from_slice(&[0_i32], &[1]);
            let sentinel = Array::from_slice(&[-1_i32], &[1]);
            let third = Array::from_slice(&[2_i32], &[1]);
            let scope = TokenValidationScope::begin().unwrap();
            let output = embeddings
                .forward(&[&first, &sentinel, &third], stream)
                .unwrap();
            assert_eq!(output.shape(), &[1, dimensions]);
            let output = output.as_type::<f32>(stream).unwrap();
            let validations = scope.finish();
            let event = async_eval_with_event(std::iter::once(&output).chain(validations.arrays()))
                .unwrap();
            event.synchronize().unwrap();
            validations.validate_completed().unwrap();
            let output = output.evaluated().unwrap();
            assert!(output
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn relative_attention_gathers_causal_distance_profiles() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let queries = Array::from_slice(&[0.0_f32], &[1, 1, 1, 1]);
        let keys = Array::from_slice(&[1.0_f32, 2.0], &[1, 1, 2, 1]);
        let values = Array::from_slice(&[10.0_f32, 20.0], &[1, 1, 2, 1]);
        let profiles = Array::from_slice(&[0.0_f32, 3.0_f32.ln()], &[1, 1, 1, 2]);
        let output = <MlxBackend as NeuralBackend>::relative_attention(
            RelativeAttentionInput {
                queries: &queries,
                keys: &keys,
                values: &values,
                profiles: &profiles,
                query_offset: 1,
                key_offset: 0,
                window: None,
                log_scaling_floor: None,
                log_scaling_alpha: 0.0,
            },
            stream,
        )
        .unwrap();
        let output = output.evaluated().unwrap();
        assert!((output.as_slice::<f32>()[0] - 12.5).abs() < 1e-5);
    }

    #[test]
    fn joint_routing_selects_with_bias_but_weights_unbiased_logits() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let hidden = Array::from_slice(&[1.0_f32], &[1, 1]);
        let weight = Array::from_slice(&[0.0_f32, 1.0, 2.0], &[3, 1]);
        let correction = Array::from_slice(&[10.0_f32, 0.0], &[2]);
        let global = Array::from_slice(&[0.5_f32], &[1]);
        let routes = <MlxBackend as NeuralBackend>::joint_expert_routing(
            JointExpertRoutingInput {
                hidden: &hidden,
                weight: &weight,
                correction_bias: &correction,
                global_scale: &global,
                routed_experts: 2,
                shared_experts: 1,
                top_k: 1,
                route_scale: 2.0,
            },
            stream,
        )
        .unwrap();
        let ids = routes.routed_ids.evaluated().unwrap();
        let routed = routes.routed_weights.evaluated().unwrap();
        let shared = routes.shared_weights.evaluated().unwrap();
        assert_eq!(ids.as_slice::<u32>(), &[0]);
        let expected = 0.5 / (0.5 + 1.0 / (1.0 + (-2.0_f32).exp()));
        assert!((routed.as_slice::<f32>()[0] - expected).abs() < 1e-5);
        assert!((routed.as_slice::<f32>()[0] + shared.as_slice::<f32>()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "explicit MLX normalization parity; run outside the sandbox"]
    fn mlx_general_normalization_matches_scalar_references() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let values = [1.0_f32, -2.0, 3.0, -4.0];
        let input = Array::from_slice(&values, &[1, 4]);
        let epsilon = 1e-5;

        let mut normalization = <MlxBackend as NeuralBackend>::normalization(
            NormalizationConstructionSpec {
                dimensions: 4,
                epsilon,
                scale: NormalizationScale::Unit,
            },
            stream,
        )
        .unwrap();
        let rms = (values.iter().map(|value| value * value).sum::<f32>() / 4.0 + epsilon).sqrt();
        let expected_rms = values.map(|value| value / rms);
        close(
            &normalization.forward(&input, stream).unwrap(),
            &expected_rms,
            1e-5,
        );

        let l2 = (values.iter().map(|value| value * value).sum::<f32>() + epsilon).sqrt();
        let expected_l2 = values.map(|value| value / l2);
        close(
            &<MlxBackend as NeuralBackend>::l2_normalize(&input, epsilon, stream).unwrap(),
            &expected_l2,
            1e-5,
        );
    }

    #[test]
    #[ignore = "explicit MLX grouped-normalization parity; run outside the sandbox"]
    fn mlx_silu_gated_group_norm_matches_scalar_reference() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let values = [1.0_f32, -2.0, 3.0, -4.0];
        let gates = [0.5_f32, -1.0, 1.5, -0.25];
        let weights = [1.0_f32, 2.0, 0.5, -1.0];
        let epsilon = 1e-5;
        let input = Array::from_slice(&values, &[1, 4]);
        let gate = Array::from_slice(&gates, &[1, 4]);
        let weight = Array::from_slice(&weights, &[4]);
        let mut expected = [0.0_f32; 4];
        for group in 0..2 {
            let start = group * 2;
            let variance =
                (values[start] * values[start] + values[start + 1] * values[start + 1]) / 2.0;
            let scale = (variance + epsilon).sqrt().recip();
            for index in start..start + 2 {
                let silu_gate = gates[index] / (1.0 + (-gates[index]).exp());
                expected[index] = values[index] * scale * weights[index] * silu_gate;
            }
        }
        let actual = <MlxBackend as NeuralBackend>::silu_gated_group_rms_norm(
            &input, &gate, &weight, 2, epsilon, stream,
        )
        .unwrap();
        close(&actual, &expected, 1e-5);
    }

    #[test]
    #[ignore = "explicit MLX head-expansion parity; run outside the sandbox"]
    fn mlx_head_expansion_matches_scalar_reference() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let input = Array::from_slice(&values, &[1, 2, 2]);
        let expansion = HeadExpansion {
            axis: 1,
            source_heads: 2,
            target_heads: 4,
        };
        let actual =
            <MlxBackend as NeuralBackend>::expand_heads(&input, expansion, stream).unwrap();
        let (expected, shape) = reference_expand_heads(&values, &[1, 2, 2], 1, 4).unwrap();
        let shape = shape
            .into_iter()
            .map(|dimension| i32::try_from(dimension).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual.shape(), shape.as_slice());
        close(&actual, &expected, 1e-5);
    }

    #[test]
    #[ignore = "explicit MLX segmented-attention parity; run outside the sandbox"]
    fn mlx_segmented_attention_matches_scalar_reference() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let queries = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let keys = [1.0_f32, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0];
        let values = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let query = Array::from_slice(&queries, &[4, 1, 2]);
        let key = Array::from_slice(&keys, &[4, 1, 2]);
        let value = Array::from_slice(&values, &[4, 1, 2]);
        let segments = [2, 2];
        let scale = 2.0_f32.sqrt().recip();
        let actual = <MlxBackend as NeuralBackend>::segmented_attention(
            SegmentedAttentionInput {
                queries: &query,
                keys: &key,
                values: &value,
                segment_lengths: &segments,
                scale,
            },
            stream,
        )
        .unwrap();
        let expected =
            reference_segmented_attention(4, 1, 2, 2, &queries, &keys, &values, &segments, scale)
                .unwrap();
        close(&actual, &expected, 2e-5);
    }
}
