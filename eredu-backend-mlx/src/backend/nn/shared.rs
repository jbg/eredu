//! MLX implementation of backend-neutral neural operators.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

use eredu_checkpoint::LinearFormat;
use eredu_core::{checkpoint::TensorDtype, Completion, Submission};
use eredu_nn::{
    validate_parameter_topology, AttentionCache, AttentionRequest, BlockwiseAttentionBackend,
    BlockwiseAttentionSpec, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec,
    Error as ComputeError, GatedDeltaScanInput, GatedDeltaScanOutput, GatedProductGroupLayout,
    GatedProductPolicy, GroupScoring, GroupSelection, GroupSelectionOperator,
    GroupedGatedProductOperator, GroupedGatedProductSpec, GroupedNeuralBackend,
    GroupedRelu2Operator, GroupedRelu2Spec, HyperConnectionOperator, HyperConnectionSpec,
    HyperConnectionState, HyperHeadOperator, HyperHeadSpec, HyperNeuralBackend,
    IndexedAttentionInput, JointGroupSelection, JointGroupSelectionInput, LinearFormatSpec,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterMetadata, ParameterSpec, ParameterVisitor,
    ParameterVisitorMut, Parameterized, PooledAttentionInput, PooledPositionInput,
    RelativeAttentionInput, RotaryOperator, RotaryPosition, RotarySpec, SegmentedAttentionInput,
    SelectiveStateSpaceScanInput, SelectiveStateSpaceScanOutput, Tensor,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedOutput,
    TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    BarrierBackend, BroadcastBackend, CommunicationBackend, CommunicationPeerCounts,
    EvenGatherBackend, FailureAgreementBackend, ParameterBackend, PointToPointBackend,
    RoleExactBoundaryValue, SubmissionBackend, SumReductionBackend, TransferBackend,
    UnevenGatherBackend, VariableAllToAllBackend,
};
use ref_cast::RefCast;
use safemlx::ops::{
    arange, argpartition_axis, broadcast_to, clip, concatenate_axis, einsum,
    indexing::{take_along_axis, NewAxis, TryIndexOp},
    matmul, maximum, r#where, sigmoid, softmax_axis, zeros_dtype,
};
use safemlx::{
    fast::ScaledDotProductAttentionMask, Array, Dtype, Event, HostTransferBuffer,
    HostTransferPolicy, ImmutableHostTransferBuffer, Stream,
};

use crate::backend::runtime::distributed::{
    completion::{MlxCommunicationCompletion, MlxFailureAgreement},
    recv_like, send,
    topology::CommunicationRouteRealization,
    Group,
};
use crate::backend::{
    nn::{
        self as common,
        rope::{self, RopeVariant},
        tensor::validate_token_domain,
    },
    runtime::cache::kv::{
        BlockwiseAttentionAccumulator, ConcatKeyValueCache, KeyValueAttentionBlock, KeyValueCache,
        PagedKeyValueCache,
    },
};
use crate::MlxTensor;
use crate::{
    module::{Module, ModuleParamMut, ModuleParamRef, PhysicalParameters},
    nested::NestedValue,
    nn,
};

#[cfg(test)]
fn trace_partition_collective(operation: &str, input: &Array, group: &Group, detail: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    if std::env::var_os("EREDU_TEST_PARTITION_COLLECTIVE_TRACE").is_some() {
        static ORDINAL: AtomicUsize = AtomicUsize::new(0);
        eprintln!(
            "partition-schedule rank={} ordinal={} operation={} subgroup={}/{} shape={:?} {}",
            std::env::var("MLX_RANK").unwrap_or_else(|_| "?".into()),
            ORDINAL.fetch_add(1, Ordering::Relaxed),
            operation,
            group.rank(),
            group.size(),
            input.shape(),
            detail,
        );
    }
}

fn bind_linear_companion(weight: &ParameterSpec, mut companion: ParameterSpec) -> ParameterSpec {
    companion.linear_companion_of = Some(weight.id.clone());
    companion
}

fn compute<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, ComputeError> {
    result.map_err(ComputeError::backend)
}

fn compute_tensor(
    result: Result<Array, safemlx::error::Exception>,
) -> Result<MlxTensor, ComputeError> {
    compute(result).map(MlxTensor::from_array)
}

impl BlockwiseAttentionBackend for MlxNeuralBackend {
    type BlockwiseAccumulator = BlockwiseAttentionAccumulator;

    fn begin_blockwise_attention(
        spec: BlockwiseAttentionSpec<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<Self::BlockwiseAccumulator, ComputeError> {
        compute(BlockwiseAttentionAccumulator::new(
            spec.queries.as_array(),
            spec.scale,
            spec.mask.map(MlxTensor::as_array),
            spec.query_start,
            spec.sliding_window,
            spec.prefix_tokens,
            spec.sinks.map(MlxTensor::as_array),
            spec.context_end,
            context,
        ))
    }

    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: MlxTensor,
        values: MlxTensor,
        context: &Stream,
    ) -> Result<u64, ComputeError> {
        let scratch = keys.as_array().nbytes() as u64 + values.as_array().nbytes() as u64;
        let block =
            KeyValueAttentionBlock::unleased(start, end, keys.into_array(), values.into_array());
        compute(accumulator.accumulate(&block, context))?;
        compute(accumulator.submit())?;
        Ok(scratch)
    }

    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(accumulator.finish(context))
    }
}

fn parameter_topology(
    module: &impl PhysicalParameters,
    weight: ParameterSpec,
    bias: Option<ParameterSpec>,
    format: &LinearFormatSpec,
) -> Result<BTreeMap<String, ParameterSpec>, ComputeError> {
    format.validate_for_weight(&weight)?;
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
                "scales" | "weight_scale_inv" => bind_linear_companion(
                    &weight,
                    format.scale().cloned().ok_or_else(|| {
                        ComputeError::backend(format!(
                            "backend operator exposed scale slot {local:?} but the architecture declared none"
                        ))
                    })?,
                ),
                "biases" => bind_linear_companion(
                    &weight,
                    format.affine_bias().cloned().ok_or_else(|| {
                        ComputeError::backend(
                            "backend operator exposed affine-bias slot but the architecture declared none",
                        )
                    })?,
                ),
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
    module: &impl PhysicalParameters,
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
    M: PhysicalParameters,
    V: ParameterVisitor<'a, MlxTensor>,
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
            MlxTensor::ref_cast(value),
        );
    }
}

fn visit_module_parameters_mut<'a, M, V>(
    module: &'a mut M,
    topology: &BTreeMap<String, ParameterSpec>,
    visitor: &mut V,
) where
    M: PhysicalParameters,
    V: ParameterVisitorMut<'a, MlxTensor>,
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
            MlxTensor::ref_cast_mut(value),
        );
    }
}

fn set_module_trainable(module: &mut impl PhysicalParameters, trainable: bool) {
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

impl<'a> ParameterVisitor<'a, MlxTensor> for ParameterRefCollector<'a> {
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a MlxTensor) {
        if !self.trainable_only || metadata.trainable {
            self.parameters.insert(
                Rc::from(metadata.id.as_str()),
                NestedValue::Value(value.as_array()),
            );
        }
    }
}

struct ParameterMutCollector<'a> {
    parameters: ModuleParamMut<'a>,
}

impl<'a> ParameterVisitorMut<'a, MlxTensor> for ParameterMutCollector<'a> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        self.parameters.insert(
            Rc::from(metadata.id.as_str()),
            NestedValue::Value(value.as_array_mut()),
        );
    }
}

/// Collects immutable MLX parameter references from a neutral module.
pub(crate) fn neutral_parameter_refs<M: Parameterized<MlxTensor>>(
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

/// Collects mutable MLX parameter references from a neutral module.
pub(crate) fn neutral_parameter_refs_mut<M: Parameterized<MlxTensor>>(
    module: &mut M,
) -> ModuleParamMut<'_> {
    validate_parameter_topology(&*module).expect("backend-neutral parameter topology is valid");
    let mut collector = ParameterMutCollector {
        parameters: ModuleParamMut::new(),
    };
    module.visit_parameters_mut(&mut collector);
    collector.parameters
}

/// MLX module view over any backend-neutral parameterized value.
///
/// Architecture types retain their neutral parameter identities while MLX
/// loading utilities traverse the same native slots without rebuilding a
/// parameter tree.
#[derive(Debug, Clone)]
pub struct MlxModule<M> {
    /// Backend-neutral module specialized to MLX operators.
    pub inner: M,
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

impl<M: Parameterized<MlxTensor>> Parameterized<MlxTensor> for MlxModule<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        self.inner.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        self.inner.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.inner.set_trainable(trainable);
    }
}

/// Native MLX module exposed through stable neutral parameter identities.
#[derive(Debug, Clone)]
struct MlxNamedModule<M> {
    inner: M,
    topology: BTreeMap<String, ParameterSpec>,
}

impl<M: PhysicalParameters> MlxNamedModule<M> {
    fn with_exact_topology(
        inner: M,
        specs: impl IntoIterator<Item = (&'static str, ParameterSpec)>,
    ) -> Result<Self, ComputeError> {
        let topology = exact_parameter_topology(&inner, specs)?;
        Ok(Self { inner, topology })
    }

    fn local_parameter_names(&self) -> Vec<String> {
        self.topology.keys().cloned().collect()
    }

    fn bind_local_parameters(
        &mut self,
        mut bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        let expected = self.topology.keys().cloned().collect::<BTreeSet<_>>();
        let actual = bindings.keys().cloned().collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(ComputeError::backend(format!(
                "compact grouped bindings differ: missing {:?}, unexpected {:?}",
                expected.difference(&actual).collect::<Vec<_>>(),
                actual.difference(&expected).collect::<Vec<_>>()
            )));
        }
        for (local, parameter) in self.inner.parameters_mut().flatten() {
            let value = bindings
                .remove(local.as_ref())
                .expect("equal compact binding sets contain every native parameter");
            if parameter.shape() != value.shape() {
                return Err(ComputeError::backend(format!(
                    "compact grouped binding {local:?} has shape {:?}, expected {:?}",
                    value.shape(),
                    parameter.shape()
                )));
            }
            *parameter = value;
        }
        Ok(())
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

impl<M: PhysicalParameters> Parameterized<MlxTensor> for MlxNamedModule<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        visit_module_parameters(&self.inner, &self.topology, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        visit_module_parameters_mut(&mut self.inner, &self.topology, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.inner, trainable);
    }
}

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear {
    module: common::linear::PhysicalLinear,
    topology: BTreeMap<String, ParameterSpec>,
    vocabulary_range: Option<VocabularyParallelRange>,
}

impl LinearOperator<MlxTensor> for MlxLinear {
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.forward(input.as_array(), context))
    }
}

impl Parameterized<MlxTensor> for MlxLinear {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
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
    module: common::linear::PhysicalEmbedding,
    topology: BTreeMap<String, ParameterSpec>,
    vocabulary: i32,
    vocabulary_range: Option<VocabularyParallelRange>,
}

impl EmbeddingOperator<MlxTensor> for MlxEmbedding {
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.forward(input.as_array(), context))
    }

    fn lookup(
        &mut self,
        input: &MlxTensor,
        policy: EmbeddingLookupPolicy,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        policy.validate()?;
        let sentinel = match policy {
            EmbeddingLookupPolicy::Strict => None,
            EmbeddingLookupPolicy::ZeroSentinel(sentinel) => Some(sentinel),
        };
        let input = compute(validate_token_domain(
            input.as_array(),
            self.vocabulary,
            sentinel,
            context,
        ))?;
        let Some(sentinel) = sentinel else {
            return compute_tensor(self.module.forward(&input, context));
        };
        let sentinel_mask = compute(input.eq(Array::from_int(sentinel), context))?;
        let nonnegative = compute(input.ge(Array::from_int(0), context))?;
        let below_vocabulary = compute(input.lt(Array::from_int(self.vocabulary), context))?;
        let ordinary_mask = compute(nonnegative.logical_and(&below_vocabulary, context))?;
        let zero_tokens = compute(safemlx::ops::zeros_like(&input, context))?;
        let safe_tokens = compute(safemlx::ops::r#where(
            &ordinary_mask,
            &input,
            &zero_tokens,
            context,
        ))?;
        let embedded = compute(self.module.forward(&safe_tokens, context))?;
        let output_mask = compute(sentinel_mask.expand_dims(-1, context))?;
        let zero_embeddings = compute(safemlx::ops::zeros_like(&embedded, context))?;
        compute_tensor(safemlx::ops::r#where(
            &output_mask,
            &zero_embeddings,
            &embedded,
            context,
        ))
    }

    fn as_linear(
        &mut self,
        input: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.as_linear(input.as_array(), context))
    }
}

impl Parameterized<MlxTensor> for MlxEmbedding {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
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
        <Self as NormalizationOperator<MlxTensor>>::forward(
            self,
            &MlxTensor::from_array(input.clone()),
            context,
        )
        .map(MlxTensor::into_array)
        .map_err(|error| safemlx::error::Exception::custom(error.to_string()))
    }
}

impl NormalizationOperator<MlxTensor> for MlxRmsNorm {
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        if input.shape().last().copied() != Some(self.dimensions) {
            return Err(ComputeError::backend(format!(
                "RMS normalization expects final width {}, got {:?}",
                self.dimensions,
                input.shape()
            )));
        }
        match (&mut self.module, self.offset) {
            (Some(module), None) => compute_tensor(module.forward(input, context)),
            (Some(module), Some(offset)) => {
                let scale = compute(module.weight.as_ref().add(Array::from_f32(offset), context))?;
                compute_tensor(safemlx::fast::rms_norm(
                    input,
                    &scale,
                    self.epsilon,
                    context,
                ))
            }
            (None, None) => {
                mlx_weightless_rms_norm(input, self.epsilon, context).map(MlxTensor::from_array)
            }
            (None, Some(_)) => unreachable!("validated normalization construction"),
        }
    }
}

impl Parameterized<MlxTensor> for MlxRmsNorm {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        if let Some(module) = &self.module {
            visit_module_parameters(module, &self.topology, visitor);
        }
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
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
pub struct MlxRotary(RopeVariant);

impl RotaryOperator<MlxTensor> for MlxRotary {
    fn forward(
        &mut self,
        input: &MlxTensor,
        position: RotaryPosition<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        match position {
            RotaryPosition::Offset(offset) => {
                let rope_input = nn::RopeInputBuilder::new(input.as_array())
                    .offset(offset)
                    .build()
                    .map_err(ComputeError::backend)?;
                compute_tensor(self.0.forward(rope_input, context))
            }
            RotaryPosition::Embeddings { cosine, sine } => {
                compute_tensor(common::attention::apply_rotary_embeddings(
                    input.as_array(),
                    cosine.as_array(),
                    sine.as_array(),
                    context,
                ))
            }
        }
    }
}

impl Parameterized<MlxTensor> for MlxRotary {
    fn visit_parameters<'a, V>(&'a self, _visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
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

impl HyperConnectionOperator<MlxTensor> for MlxHyperConnection {
    fn collapse(
        &mut self,
        residual: &MlxTensor,
        norm_epsilon: f32,
        context: &Stream,
    ) -> Result<HyperConnectionState<MlxTensor>, ComputeError> {
        let (collapsed, split) = compute(self.module.collapse_split(
            residual.as_array(),
            norm_epsilon,
            context,
        ))?;
        Ok(HyperConnectionState {
            collapsed: MlxTensor::from_array(collapsed),
            pre: MlxTensor::from_array(split.pre),
            post: MlxTensor::from_array(split.post),
            combination: MlxTensor::from_array(split.combination),
        })
    }

    fn expand(
        &mut self,
        sublayer: &MlxTensor,
        residual: &MlxTensor,
        state: &HyperConnectionState<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(common::hyper_connections::expand(
            sublayer.as_array(),
            residual.as_array(),
            state.post.as_array(),
            state.combination.as_array(),
            context,
        ))
    }
}

impl Parameterized<MlxTensor> for MlxHyperConnection {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
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

impl HyperHeadOperator<MlxTensor> for MlxHyperHead {
    fn forward(
        &mut self,
        residual: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.forward(residual.as_array(), context))
    }
}

impl Parameterized<MlxTensor> for MlxHyperHead {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        visit_module_parameters(&self.module, &self.topology, visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        visit_module_parameters_mut(&mut self.module, &self.topology, visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        set_module_trainable(&mut self.module, trainable);
    }
}

/// MLX implementation of the backend-neutral learned top-k selector.
#[derive(Debug, Clone)]
pub struct MlxTopKGroupSelector {
    module: MlxNamedModule<common::grouped::TopKGroupSelector>,
}

impl Parameterized<MlxTensor> for MlxTopKGroupSelector {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        self.module.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl GroupSelectionOperator<MlxTensor> for MlxTopKGroupSelector {
    fn select(
        &mut self,
        input: &MlxTensor,
        context: &Stream,
    ) -> Result<GroupSelection<MlxTensor>, ComputeError> {
        let output = compute(self.module.select_with_selection_bias(
            input.as_array(),
            None,
            context,
        ))?;
        Ok(GroupSelection::new(
            MlxTensor::from_array(output.indices),
            MlxTensor::from_array(output.scores),
            MlxTensor::from_array(output.weights),
        ))
    }

    fn select_indices(
        &mut self,
        input: &MlxTensor,
        group_indices: &MlxTensor,
        context: &Stream,
    ) -> Result<GroupSelection<MlxTensor>, ComputeError> {
        let output = compute(self.module.select_indices(
            input.as_array(),
            group_indices.as_array(),
            context,
        ))?;
        Ok(GroupSelection::new(
            MlxTensor::from_array(output.indices),
            MlxTensor::from_array(output.scores),
            MlxTensor::from_array(output.weights),
        ))
    }
}

/// MLX packed execution bank for backend-neutral grouped gated-product groups.
#[derive(Debug, Clone)]
pub struct MlxGroupedGatedProduct {
    spec: GroupedGatedProductSpec,
    module: MlxNamedModule<common::grouped::PackedGatedProductGroups>,
}

impl MlxGroupedGatedProduct {
    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        self.module.local_parameter_names()
    }

    pub(crate) fn bind_local_parameters(
        &mut self,
        bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        self.module.bind_local_parameters(bindings)
    }
}

impl Parameterized<MlxTensor> for MlxGroupedGatedProduct {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        self.module.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl GroupedGatedProductOperator<MlxTensor> for MlxGroupedGatedProduct {
    fn spec(&self) -> &GroupedGatedProductSpec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            context,
        ))?;
        compute_tensor(output.reshape(input.shape(), context))
    }
}

impl TensorParallelGroupedGatedProductOperator<MlxTensor> for MlxGroupedGatedProduct {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        let input = input.as_array();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward_tensor_parallel(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            partitions,
            context,
        ))?;
        let (reducible, post_reduce) = output.into_parts();
        Ok(TensorParallelGroupedOutput::new(
            compute_tensor(reducible.reshape(input.shape(), context))?,
            post_reduce
                .map(|bias| compute_tensor(bias.reshape(input.shape(), context)))
                .transpose()?,
        ))
    }
}

/// MLX packed execution bank for backend-neutral grouped ReLU2 groups.
#[derive(Debug, Clone)]
pub struct MlxGroupedRelu2 {
    spec: GroupedRelu2Spec,
    module: MlxNamedModule<common::grouped::PackedRelu2Groups>,
}

impl MlxGroupedRelu2 {
    /// Returns the architecture-owned specification used to realize this bank.
    pub const fn spec(&self) -> &GroupedRelu2Spec {
        &self.spec
    }

    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        self.module.local_parameter_names()
    }

    pub(crate) fn bind_local_parameters(
        &mut self,
        bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        self.module.bind_local_parameters(bindings)
    }
}

impl Parameterized<MlxTensor> for MlxGroupedRelu2 {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, MlxTensor>,
    {
        self.module.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.module.set_trainable(trainable);
    }
}

impl GroupedRelu2Operator<MlxTensor> for MlxGroupedRelu2 {
    fn spec(&self) -> &GroupedRelu2Spec {
        &self.spec
    }

    fn forward_grouped(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let shape = input.shape();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            context,
        ))?;
        compute_tensor(output.reshape(shape, context))
    }
}

impl TensorParallelGroupedRelu2Operator<MlxTensor> for MlxGroupedRelu2 {
    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        let input = input.as_array();
        let shape = input.shape();
        let flattened = compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = compute(self.module.forward_tensor_parallel(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            partitions,
            context,
        ))?;
        let (reducible, post_reduce) = output.into_parts();
        Ok(TensorParallelGroupedOutput::new(
            compute_tensor(reducible.reshape(shape, context))?,
            post_reduce
                .map(|bias| compute_tensor(bias.reshape(shape, context)))
                .transpose()?,
        ))
    }
}

/// Zero-sized MLX backend selector. All calls are statically dispatched.
#[derive(Debug, Clone, Copy)]
pub struct MlxNeuralBackend;

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

impl SubmissionBackend for MlxNeuralBackend {
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
        MlxTensor: 'a,
        I: IntoIterator<Item = &'a MlxTensor>,
    {
        Ok(MlxSubmissionCompletion::new(
            safemlx::transforms::async_eval_with_event(
                values.into_iter().map(MlxTensor::as_array),
            )?,
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

const MLX_COMMUNICATION_MAX_ELEMENTS: usize = i32::MAX as usize;

fn communication_dtype(dtype: Dtype) -> TensorDtype {
    match dtype {
        Dtype::Bool => TensorDtype::Bool,
        Dtype::Uint8 => TensorDtype::U8,
        Dtype::Uint16 => TensorDtype::U16,
        Dtype::Uint32 => TensorDtype::U32,
        Dtype::Uint64 => TensorDtype::U64,
        Dtype::Int8 => TensorDtype::I8,
        Dtype::Int16 => TensorDtype::I16,
        Dtype::Int32 => TensorDtype::I32,
        Dtype::Int64 => TensorDtype::I64,
        Dtype::Float16 => TensorDtype::F16,
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float64 => TensorDtype::F64,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        Dtype::Complex64 => TensorDtype::Complex64,
    }
}

#[derive(Clone, Copy)]
enum MlxCommunicationDtypes {
    Floating,
    FloatingAndI32,
    FloatingI32AndU32,
}

impl MlxCommunicationDtypes {
    fn admits(self, dtype: Dtype) -> bool {
        let floating = matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
        match self {
            Self::Floating => floating,
            Self::FloatingAndI32 => floating || dtype == Dtype::Int32,
            Self::FloatingI32AndU32 => floating || matches!(dtype, Dtype::Int32 | Dtype::Uint32),
        }
    }
}

fn validate_communication_tensor(
    value: &Array,
    dtypes: MlxCommunicationDtypes,
) -> Result<(), safemlx::error::Exception> {
    if !dtypes.admits(value.dtype()) {
        return Err(safemlx::error::Exception::custom(format!(
            "MLX communication does not advertise dtype {:?}",
            value.dtype()
        )));
    }
    if value.ndim() > i32::MAX as usize || value.size() > MLX_COMMUNICATION_MAX_ELEMENTS {
        return Err(safemlx::error::Exception::custom(
            "MLX communication tensor exceeds advertised rank or element limits",
        ));
    }
    Ok(())
}

fn validate_route_bundle(
    values: &[MlxTensor],
    route: &CommunicationRouteRealization,
) -> Result<(), safemlx::error::Exception> {
    let requirement = route.descriptor().requirement();
    let limits = requirement.limits().ok_or_else(|| {
        safemlx::error::Exception::custom("point-to-point route has no tensor limits")
    })?;
    if values.is_empty() || values.len() > limits.max_tensors() {
        return Err(safemlx::error::Exception::custom(format!(
            "point-to-point route bundle has {} tensors, expected 1..={}",
            values.len(),
            limits.max_tensors()
        )));
    }
    for value in values {
        let array = value.as_array();
        validate_communication_tensor(array, MlxCommunicationDtypes::FloatingI32AndU32)?;
        let dtype = communication_dtype(array.dtype());
        if !requirement.dtypes().contains(&dtype) {
            return Err(safemlx::error::Exception::custom(format!(
                "point-to-point route does not admit dtype {dtype:?}"
            )));
        }
        if array.ndim() > limits.max_tensor_rank() || array.size() > limits.max_tensor_elements() {
            return Err(safemlx::error::Exception::custom(format!(
                "point-to-point placeholder shape {:?} exceeds route limits",
                array.shape()
            )));
        }
    }
    Ok(())
}

fn collective_completion(
    input: Array,
    output: &Array,
    group: &Group,
    executor: &Stream,
    count_buffers: Vec<Vec<usize>>,
) -> Result<MlxCommunicationCompletion, safemlx::error::Exception> {
    MlxCommunicationCompletion::submit(
        [output],
        vec![input, output.clone()],
        count_buffers,
        vec![group.clone()],
        Vec::new(),
        vec![executor.clone()],
    )
}

impl CommunicationBackend for MlxNeuralBackend {
    type CommunicationGroup = Group;
    type CommunicationRoute = CommunicationRouteRealization;
    type CommunicationCompletion = MlxCommunicationCompletion;
    type CommunicationError = safemlx::error::Exception;

    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        let outputs = values
            .into_iter()
            .map(|value| value.as_array().clone())
            .collect::<Vec<_>>();
        #[cfg(test)]
        if std::env::var_os("EREDU_TEST_PARTITION_COLLECTIVE_TRACE").is_some() {
            eprintln!(
                "partition-schedule rank={} operation=local-dependencies shapes={:?}",
                std::env::var("MLX_RANK").unwrap_or_else(|_| "?".into()),
                outputs.iter().map(Array::shape).collect::<Vec<_>>(),
            );
        }
        let retained = outputs.clone();
        let completion = MlxCommunicationCompletion::submit(
            outputs.iter(),
            retained,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![executor.clone()],
        )?;
        Ok(Submission {
            output: (),
            completion,
        })
    }
}

/// Mechanical tensor metadata used by the backend-neutral partition driver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MlxCommunicationTensorMetadata;

impl eredu_runtime::CommunicationTensorMetadata<MlxNeuralBackend>
    for MlxCommunicationTensorMetadata
{
    fn dtype(&self, tensor: &MlxTensor) -> TensorDtype {
        communication_dtype(tensor.as_array().dtype())
    }

    fn shape(&self, tensor: &MlxTensor) -> Vec<usize> {
        tensor
            .as_array()
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or(usize::MAX))
            .collect()
    }
}

impl SumReductionBackend for MlxNeuralBackend {
    fn all_reduce_sum(
        value: MlxTensor,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, MlxCommunicationCompletion>, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("sum", &input, group, "");
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        let output = crate::backend::runtime::distributed::all_sum(&input, group, executor)?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }
}

impl EvenGatherBackend for MlxNeuralBackend {
    fn all_gather_even(
        value: MlxTensor,
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, MlxCommunicationCompletion>, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("gather", &input, group, &format!("axis={axis}"));
        validate_communication_tensor(&input, MlxCommunicationDtypes::FloatingAndI32)?;
        let axis = i32::try_from(axis).map_err(|_| {
            safemlx::error::Exception::custom("all-gather axis does not fit in i32")
        })?;
        let output = crate::backend::distributed::all_gather_axis(&input, axis, group, executor)?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }
}

impl UnevenGatherBackend for MlxNeuralBackend {
    fn all_gather_uneven(
        value: MlxTensor,
        counts: &[usize],
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, MlxCommunicationCompletion>, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective(
            "gather_uneven",
            &input,
            group,
            &format!("axis={axis} counts={counts:?}"),
        );
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        let axis = i32::try_from(axis).map_err(|_| {
            safemlx::error::Exception::custom("uneven all-gather axis does not fit in i32")
        })?;
        let output = crate::backend::distributed::all_gather_uneven_axis(
            &input, axis, counts, group, executor,
        )?;
        let completion =
            collective_completion(input, &output, group, executor, vec![counts.to_vec()])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }
}

impl VariableAllToAllBackend for MlxNeuralBackend {
    fn variable_all_to_all(
        value: MlxTensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, MlxCommunicationCompletion>, Self::CommunicationError> {
        #[cfg(test)]
        crate::composition::mlx::path_instrumentation::variable_all_to_all_submission();
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective(
            "exchange",
            &input,
            group,
            &format!(
                "axis={axis} send={:?} receive={:?}",
                counts.send(),
                counts.receive()
            ),
        );
        validate_communication_tensor(&input, MlxCommunicationDtypes::FloatingAndI32)?;
        if counts.group_size() != group.size() {
            return Err(safemlx::error::Exception::custom(format!(
                "variable all-to-all has {} peer counts for group size {}",
                counts.group_size(),
                group.size()
            )));
        }
        if counts
            .send()
            .iter()
            .chain(counts.receive())
            .any(|count| *count > i32::MAX as usize)
        {
            return Err(safemlx::error::Exception::custom(
                "variable all-to-all peer count exceeds advertised i32 limit",
            ));
        }
        let all_zero = counts
            .send()
            .iter()
            .chain(counts.receive())
            .all(|count| *count == 0);
        if all_zero && group.size() > 1 {
            // A zero-element MLX result may be considered already evaluated,
            // which would omit the native collective and let this subgroup
            // advance ahead of non-empty subgroups in the same world wave.
            // Submit one private self-directed sentinel per member while
            // preserving the selected logical zero-count result exactly.
            let mut sentinel_shape = input.shape().to_vec();
            sentinel_shape[axis] = 1;
            let sentinel = zeros_dtype(&sentinel_shape, input.dtype(), executor)?;
            let mut sentinel_counts = vec![0usize; group.size()];
            sentinel_counts[group.rank()] = 1;
            let sentinel_output = crate::backend::distributed::all_to_all_v_axis(
                &sentinel,
                axis,
                &sentinel_counts,
                &sentinel_counts,
                group,
                executor,
            )?;
            let completion = collective_completion(
                sentinel,
                &sentinel_output,
                group,
                executor,
                vec![sentinel_counts],
            )?;
            return Ok(Submission {
                output: MlxTensor::from_array(input),
                completion,
            });
        }
        let output = crate::backend::distributed::all_to_all_v_axis(
            &input,
            axis,
            counts.send(),
            counts.receive(),
            group,
            executor,
        )?;
        let completion = collective_completion(
            input,
            &output,
            group,
            executor,
            vec![counts.send().to_vec(), counts.receive().to_vec()],
        )?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }
}

impl BroadcastBackend for MlxNeuralBackend {
    fn broadcast(
        value: MlxTensor,
        root: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, MlxCommunicationCompletion>, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        if root >= group.size() {
            return Err(safemlx::error::Exception::custom(format!(
                "broadcast root {root} is outside group size {}",
                group.size()
            )));
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("broadcast", &input, group, &format!("root={root}"));
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        // Preserve every rank's lazy predecessor graph in the submitted event.
        // A zero-valued expression keeps non-roots in the same dependency order
        // without performing an unbounded host synchronization before returning
        // the exact communication completion.
        let contribution = if group.rank() == root {
            input.clone()
        } else {
            input.multiply(Array::from_f32(0.0), executor)?
        };
        let output = crate::backend::runtime::distributed::all_sum_for(
            eredu_runtime::CommunicationOperation::Broadcast,
            &contribution,
            group,
            executor,
        )?;
        let completion = MlxCommunicationCompletion::submit(
            [&output],
            vec![input, contribution, output.clone()],
            Vec::new(),
            vec![group.clone()],
            Vec::new(),
            vec![executor.clone()],
        )?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion,
        })
    }
}

impl BarrierBackend for MlxNeuralBackend {
    fn barrier(
        group: &Group,
        executor: &Stream,
    ) -> Result<MlxCommunicationCompletion, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let token = zeros_dtype(&[], Dtype::Float32, executor)?;
        let completed = crate::backend::runtime::distributed::payload_free_all_sum_for(
            eredu_runtime::CommunicationOperation::Barrier,
            &token,
            group,
            executor,
        )?;
        MlxCommunicationCompletion::submit(
            [&completed],
            vec![token, completed.clone()],
            Vec::new(),
            vec![group.clone()],
            Vec::new(),
            vec![executor.clone()],
        )
    }
}

impl FailureAgreementBackend for MlxNeuralBackend {
    type FailureAgreementOutput = MlxFailureAgreement;

    fn agree_success(
        local_success: bool,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxFailureAgreement, MlxCommunicationCompletion>, Self::CommunicationError>
    {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let member_count = i32::try_from(group.size()).map_err(|_| {
            safemlx::error::Exception::custom(
                "failure-agreement group size exceeds the advertised i32 status count",
            )
        })?;
        let input = Array::from_slice(&[i32::from(local_success)], &[1]);
        #[cfg(test)]
        trace_partition_collective(
            "agreement",
            &input,
            group,
            &format!("success={local_success}"),
        );
        let output = crate::backend::runtime::distributed::payload_free_all_sum_for(
            eredu_runtime::CommunicationOperation::FailureAgreement,
            &input,
            group,
            executor,
        )?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        let (agreement, completion) = completion.with_failure_agreement(output, member_count);
        Ok(Submission {
            output: agreement,
            completion,
        })
    }

    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError> {
        output.resolve()
    }
}

impl PointToPointBackend for MlxNeuralBackend {
    fn send_receive(
        values: Vec<RoleExactBoundaryValue<MlxTensor>>,
        route: &CommunicationRouteRealization,
        executor: &Stream,
    ) -> Result<Submission<Vec<MlxTensor>, MlxCommunicationCompletion>, Self::CommunicationError>
    {
        let group = route.group().ok_or_else(|| {
            safemlx::error::Exception::custom(format!(
                "world rank is not an endpoint of communication route {}",
                route.descriptor().id().value()
            ))
        })?;
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let descriptor = route.descriptor();
        let receiving = matches!(
            route.endpoint(),
            Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination)
        );
        let peer_rank = route.peer_rank().ok_or_else(|| {
            safemlx::error::Exception::custom(format!(
                "communication route {} endpoint has no local peer rank",
                descriptor.id().value()
            ))
        })?;
        let logical = values
            .iter()
            .map(|value| value.tensor().clone())
            .collect::<Vec<_>>();
        validate_route_bundle(&logical, route)?;
        let mut inputs = Vec::with_capacity(values.len());
        let mut frames = Vec::with_capacity(values.len());
        let mut expected_headers = Vec::with_capacity(values.len());
        for value in values {
            let (header, tensor) = value.into_parts();
            let input = tensor.into_array();
            let bytes = input
                .view_dtype(Dtype::Uint8, executor)?
                .reshape(&[-1], executor)?;
            let header_len = i32::try_from(header.len()).map_err(|_| {
                safemlx::error::Exception::custom(
                    "boundary frame header length exceeds MLX dimensions",
                )
            })?;
            let header_array = Array::from_slice(&header, &[header_len]);
            let frame = safemlx::ops::concatenate(&[&header_array, &bytes], executor)?;
            inputs.push(input);
            frames.push(frame);
            expected_headers.push(header);
        }
        let (submitted, outputs, received_headers) = if !receiving {
            let submitted = frames
                .iter()
                .map(|frame| send(frame, peer_rank, group, executor))
                .collect::<Result<Vec<_>, _>>()?;
            (submitted, inputs.clone(), Vec::new())
        } else {
            let received = frames
                .iter()
                .map(|placeholder| recv_like(placeholder, peer_rank, group, executor))
                .collect::<Result<Vec<_>, _>>()?;
            let mut outputs = Vec::with_capacity(received.len());
            let mut received_headers = Vec::with_capacity(received.len());
            for ((input, expected), received) in inputs.iter().zip(&expected_headers).zip(&received)
            {
                let header_len = i32::try_from(expected.len()).map_err(|_| {
                    safemlx::error::Exception::custom(
                        "boundary frame header length exceeds MLX indexing",
                    )
                })?;
                let received_header = received.try_index_device(0..header_len, executor)?;
                let received_payload = received.try_index_device(header_len.., executor)?;
                let payload = received_payload
                    .view_dtype(input.dtype(), executor)?
                    .reshape(input.shape(), executor)?;
                outputs.push(payload);
                received_headers.push((received_header, expected.clone()));
            }
            (received.clone(), outputs, received_headers)
        };
        let mut retained = inputs;
        retained.extend(frames);
        retained.extend(outputs.iter().cloned());
        retained.extend(submitted.iter().cloned());
        let completion_outputs = submitted
            .iter()
            .chain(outputs.iter())
            .chain(received_headers.iter().map(|(header, _)| header))
            .collect::<Vec<_>>();
        let completion = MlxCommunicationCompletion::submit(
            completion_outputs,
            retained,
            Vec::new(),
            vec![group.clone()],
            vec![route.clone()],
            vec![executor.clone()],
        )?
        .with_boundary_headers(received_headers);
        Ok(Submission {
            output: outputs.into_iter().map(MlxTensor::from_array).collect(),
            completion,
        })
    }
}

impl TransferBackend for MlxNeuralBackend {
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
        Ok((MlxTensor::from_array(weight), completion))
    }

    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError> {
        let submitted = HostTransferBuffer::copy_from_array(
            weight.as_array(),
            HostTransferPolicy::Transfer,
            executor,
        )?;
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
    /// Neutral lease conversion or MLX checkpoint submission failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// A neutral derived-weight recipe could not be lowered.
    #[error(transparent)]
    Recipe(#[from] crate::backend::runtime::checkpoint::recipe::WeightRecipeError),
    /// Final stream-to-stream weight copy failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

impl ParameterBackend for MlxNeuralBackend {
    type Parameter = MlxTensor;
    type MaterializedWeight = MlxTensor;
    type MaterializationContext =
        crate::backend::runtime::checkpoint::store::MlxParameterMaterializationContext;
    type Materialization = crate::backend::runtime::checkpoint::store::WeightMaterialization;
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
        use crate::backend::runtime::checkpoint::recipe::MlxWeightRecipeExt;

        let pending = recipe.prepare_materialization(source, context)?;
        let (output, sources) = pending.into_parts();
        let output = if context.source_stream() == context.execution_stream() {
            output
        } else {
            output.copy(context.execution_stream())?
        };
        Ok(
            crate::backend::runtime::checkpoint::store::WeightMaterialization::submit_retained(
                output, sources,
            )?,
        )
    }

    fn materialized_weight(materialization: &Self::Materialization) -> &Self::MaterializedWeight {
        MlxTensor::ref_cast(materialization.output())
    }

    fn finish_materialization(
        materialization: Self::Materialization,
    ) -> Result<Self::MaterializedWeight, Self::ParameterError> {
        Ok(MlxTensor::from_array(materialization.synchronize()?))
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
        if parameter.as_array().shape() != weight.as_array().shape() {
            return Err(MlxParameterError::Mlx(safemlx::error::Exception::custom(
                format!(
                    "parameter shape {:?} does not match materialized weight {:?}",
                    parameter.as_array().shape(),
                    weight.as_array().shape()
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

impl NeuralBackend for MlxNeuralBackend {
    const OPERATOR_CAPABILITIES: eredu_nn::NeuralOperatorCapabilities =
        eredu_nn::NeuralOperatorCapabilities::ALL;

    type Tensor = MlxTensor;
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
            spec.format.encoding(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias, &spec.format)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: None,
        })
    }

    fn embedding(spec: EmbeddingSpec, context: &Stream) -> Result<MlxEmbedding, ComputeError> {
        let module = compute(common::linear::unloaded_embedding(
            spec.vocabulary,
            spec.dimensions,
            spec.format.encoding().weight_quantization(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None, &spec.format)?;
        Ok(MlxEmbedding {
            module,
            topology,
            vocabulary: spec.vocabulary,
            vocabulary_range: None,
        })
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
                let topology = exact_parameter_topology(&module, [("weight", weight)])?;
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

    fn rotary(spec: RotarySpec, context: &Stream) -> Result<MlxRotary, ComputeError> {
        compute(rope::initialize_rope(
            spec.dimensions,
            spec.base,
            spec.traditional,
            spec.algorithm,
            context,
        ))
        .map(MlxRotary)
    }

    fn silu(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(common::layers::silu(input.into_array(), context))
    }

    fn gelu_approximate(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(nn::gelu_approximate(input.into_array(), context))
    }

    fn sigmoid(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(safemlx::ops::sigmoid(input.into_array(), context))
    }

    fn softplus(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(nn::softplus(input.into_array(), context))
    }

    fn exp(input: MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(safemlx::ops::exp(input.into_array(), context))
    }

    fn gated_group_rms_norm(
        input: &MlxTensor,
        gate: &MlxTensor,
        weight: &MlxTensor,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let gate = gate.as_array();
        let weight = weight.as_array();
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
        compute_tensor(normalized.multiply(weight, context))
    }

    fn l2_normalize(
        input: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        if input.shape().last().is_none() || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ComputeError::backend("invalid L2 normalization geometry"));
        }
        let squared = compute(input.square(context))?;
        let sum = compute(safemlx::ops::sum_axis(&squared, -1, true, context))?;
        let denominator = compute(sum.add(Array::from_f32(epsilon), context))?;
        compute_tensor(input.multiply(compute(denominator.rsqrt(context))?, context))
    }

    fn silu_gated_group_rms_norm(
        input: &MlxTensor,
        gate: &MlxTensor,
        weight: &MlxTensor,
        groups: i32,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
        let gate = gate.as_array();
        let weight = weight.as_array();
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
            .map(MlxTensor::from_array)
            .map_err(ComputeError::backend)
    }

    fn segmented_attention(
        input: SegmentedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
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
            let queries = prepare(input.queries.as_array(), query_dimensions)?;
            let keys = prepare(input.keys.as_array(), query_dimensions)?;
            let values = prepare(input.values.as_array(), value_dimensions)?;
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
        compute_tensor(concatenate_axis(&outputs, 0, context))
    }

    fn add_residual(
        residual: &MlxTensor,
        branch: &MlxTensor,
        fp32: bool,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let residual = residual.as_array();
        let branch = branch.as_array();
        if fp32 {
            let residual = compute(residual.as_dtype(Dtype::Float32, context))?;
            let branch = compute(branch.as_dtype(Dtype::Float32, context))?;
            compute_tensor(residual.add(branch, context))
        } else {
            compute_tensor(residual.add(branch, context))
        }
    }

    fn gated_delta_scan(
        input: GatedDeltaScanInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<GatedDeltaScanOutput<MlxTensor>, ComputeError> {
        let (state, output) = compute(crate::backend::nn::gated_delta::gated_delta_scan(
            input.query.as_array(),
            input.key.as_array(),
            input.value.as_array(),
            input.log_decay.as_array(),
            input.beta.as_array(),
            input.initial_state.map(|state| state.as_array().clone()),
            context,
        ))?;
        Ok(GatedDeltaScanOutput {
            state: MlxTensor::from_array(state),
            output: MlxTensor::from_array(output),
        })
    }

    fn selective_state_space_scan(
        input: SelectiveStateSpaceScanInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<SelectiveStateSpaceScanOutput<MlxTensor>, ComputeError> {
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
            Some(state) => compute(state.as_array().as_dtype(Dtype::Float32, context))?,
            None => compute(safemlx::ops::zeros::<f32>(
                &[batch, heads, head_dimensions, state_dimensions],
                context,
            ))?,
        };
        let values = compute(input.values.as_array().as_dtype(Dtype::Float32, context))?;
        let input_state = compute(
            input
                .input_state
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let output_state = compute(
            input
                .output_state
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let transition = compute(safemlx::ops::exp(
            compute(
                input
                    .transition_log
                    .as_array()
                    .as_dtype(Dtype::Float32, context),
            )?,
            context,
        ))?;
        let transition = compute(transition.multiply(Array::from_f32(-1.0), context))?
            .reshape(&[1, heads, 1, 1], context)
            .map_err(ComputeError::backend)?;
        let skip = compute(input.skip.as_array().as_dtype(Dtype::Float32, context))?
            .reshape(&[1, heads, 1], context)
            .map_err(ComputeError::backend)?;
        let bias = compute(
            input
                .time_step_bias
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?
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
                        .as_array()
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
        let output = compute(output.as_dtype(input.values.as_array().dtype(), context))?;
        Ok(SelectiveStateSpaceScanOutput {
            state: MlxTensor::from_array(state),
            output: MlxTensor::from_array(output),
        })
    }

    fn gated_product(
        gate: MlxTensor,
        up: MlxTensor,
        policy: GatedProductPolicy,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let mut gate = gate.into_array();
        let mut up = up.into_array();
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
            _ => {
                return Err(ComputeError::backend(
                    "unsupported grouped gated-product activation",
                ))
            }
        };
        compute_tensor(gate.multiply(up, context))
    }

    fn attention(
        queries: MlxTensor,
        keys: MlxTensor,
        values: MlxTensor,
        scale: f32,
        mask: Option<&MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(safemlx::fast::scaled_dot_product_attention(
            queries.into_array(),
            keys.into_array(),
            values.into_array(),
            scale,
            mask.map(|mask| ScaledDotProductAttentionMask::Array(mask.as_array())),
            None,
            context,
        ))
    }

    fn relative_attention(
        input: RelativeAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
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
        let keys = repeat_kv(input.keys.as_array())?;
        let values = repeat_kv(input.values.as_array())?;
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
        let mut bias = compute(take_along_axis(
            input.profiles.as_array(),
            &gather,
            -1,
            context,
        ))?;
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
        let mut queries = input.queries.as_array().clone();
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
        compute_tensor(matmul(probabilities, values, context))
    }

    fn indexed_attention(
        input: IndexedAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        input.validate()?;
        compute_tensor(common::attention::indexed_sparse_attention(
            input.queries.as_array(),
            input.local_keys.as_array(),
            input.local_values.as_array(),
            input.pooled_keys.as_array(),
            input.pooled_values.as_array(),
            input.selected_positions.as_array(),
            input.scale,
            input.local_mask.map(MlxTensor::as_array),
            input.pooled_mask.map(MlxTensor::as_array),
            input.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn pooled_attention(
        input: PooledAttentionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let query_tokens = input.queries.dim(2);
        let local_tokens = input.local.dim(1);
        let pooled_tokens = input.pooled.dim(1);
        let local = compute(input.local.as_array().expand_dims(1, context))?;
        let pooled = compute(input.pooled.as_array().expand_dims(1, context))?;
        let keys = compute(safemlx::ops::concatenate_axis(&[local, pooled], 2, context))?;
        let mask = if input.local_mask.is_none() && input.pooled_mask.is_none() {
            None
        } else {
            let local = match input.local_mask {
                Some(mask) => mask.as_array().clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, local_tokens], context))?,
            };
            let pooled = match input.pooled_mask {
                Some(mask) => mask.as_array().clone(),
                None => compute(Array::ones::<bool>(&[query_tokens, pooled_tokens], context))?,
            };
            Some(compute(safemlx::ops::concatenate_axis(
                &[local, pooled],
                -1,
                context,
            ))?)
        };
        compute_tensor(safemlx::fast::scaled_dot_product_attention(
            input.queries.as_array(),
            &keys,
            &keys,
            input.scale,
            mask.as_ref().map(ScaledDotProductAttentionMask::Array),
            input.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn select_pooled_positions(
        input: PooledPositionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
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
                &compute(input.queries.as_array().as_dtype(Dtype::Float32, context))?,
                &compute(
                    input
                        .pooled_keys
                        .as_array()
                        .as_dtype(Dtype::Float32, context),
                )?,
            ],
            context,
        ))?;
        let scores = compute(maximum(scores, Array::from_f32(0.0), context))?;
        let scores = compute(scores.multiply(Array::from_f32(input.scale), context))?;
        let weights = compute(
            input
                .head_weights
                .as_array()
                .as_dtype(Dtype::Float32, context),
        )?;
        let weights = compute(weights.multiply(Array::from_f32(input.head_scale), context))?;
        let weights = compute(weights.transpose_axes(&[0, 2, 1], context))?;
        let weights = compute(weights.expand_dims(-1, context))?;
        let mut scores = compute(scores.multiply(weights, context))?;
        scores = compute(scores.sum_axis(1, false, context))?;
        if let Some(mask) = input.mask {
            scores = compute(safemlx::ops::r#where(
                mask.as_array(),
                scores,
                Array::from_f32(f32::NEG_INFINITY),
                context,
            ))?;
        }
        let top_k = input.top_k.min(pooled_shape[1]);
        let indices = compute(argpartition_axis(&scores, -top_k, -1, context))?;
        let start = indices.dim(-1) - top_k;
        compute_tensor(indices.try_index_device((.., .., start..), context))
    }

    fn gather_pooled_mask(
        mask: &MlxTensor,
        selected_positions: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let mask = mask.as_array();
        let selected_positions = selected_positions.as_array();
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
        compute_tensor(selected.expand_dims(1, context))
    }

    fn attention_with_sinks(
        request: AttentionRequest<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        request.validate()?;
        compute_tensor(safemlx::fast::scaled_dot_product_attention(
            request.queries.as_array().clone(),
            request.keys.as_array().clone(),
            request.values.as_array().clone(),
            request.scale,
            request
                .mask
                .map(|mask| ScaledDotProductAttentionMask::Array(mask.as_array())),
            request.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, MlxTensor>,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        request.validate()?;
        let batch = request.queries.dim(0);
        let sequence = request.queries.dim(2);
        compute_tensor(common::attention::sliding_window_prefill_attention(
            request.queries.as_array().clone(),
            request.keys.as_array().clone(),
            request.values.as_array().clone(),
            request.scale,
            window,
            position_offset,
            batch,
            sequence,
            request.sinks.map(MlxTensor::as_array),
            context,
        ))
    }

    fn rms_norm_without_weight(
        input: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        mlx_weightless_rms_norm(input.as_array(), epsilon, context).map(MlxTensor::from_array)
    }

    fn rms_norm_with_weight(
        input: &MlxTensor,
        weight: &MlxTensor,
        epsilon: f32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let output = compute(safemlx::fast::rms_norm(
            input.as_array(),
            weight.as_array(),
            epsilon,
            context,
        ))?;
        compute_tensor(output.as_dtype(input.as_array().dtype(), context))
    }

    fn sliding_window_attention(
        queries: MlxTensor,
        keys: MlxTensor,
        values: MlxTensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let batch = queries.dim(0);
        let sequence = queries.dim(2);
        compute_tensor(common::attention::sliding_window_prefill_attention(
            queries.as_array().clone(),
            keys.as_array().clone(),
            values.as_array().clone(),
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
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(crate::backend::nn::tensor::create_causal_mask(
            sequence,
            Some(offset),
            window,
            None,
            context,
        ))
    }

    fn row_parallel_linear(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(
            linear
                .module
                .forward_row_parallel(input.as_array(), parallel, context),
        )
    }

    fn parallel_size(parallel: &Group) -> usize {
        parallel.size()
    }
}

impl eredu_nn::DistributedNeuralBackend for MlxNeuralBackend {
    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        context: &Stream,
    ) -> Result<MlxEmbedding, ComputeError> {
        range.validate_global_rows(spec.vocabulary)?;
        let global = i32::try_from(range.global_vocabulary).map_err(ComputeError::backend)?;
        let local = i32::try_from(range.local.len()).map_err(ComputeError::backend)?;
        let module = compute(common::linear::unloaded_embedding(
            local,
            spec.dimensions,
            spec.format.encoding().weight_quantization(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, None, &spec.format)?;
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
            spec.format.encoding(),
            context,
        ))?;
        let topology = parameter_topology(&module, spec.weight, spec.bias, &spec.format)?;
        Ok(MlxLinear {
            module,
            topology,
            vocabulary_range: Some(range),
        })
    }

    fn vocabulary_parallel_lookup(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        policy: EmbeddingLookupPolicy,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
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
            input.as_array(),
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
        let zero_value = compute(safemlx::ops::zeros_like(&value, context))?;
        let value = compute(safemlx::ops::r#where(&mask, &value, &zero_value, context))?;
        compute_tensor(crate::backend::runtime::distributed::all_sum(
            &value, parallel, context,
        ))
    }

    fn vocabulary_parallel_project(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let range = linear
            .vocabulary_range
            .as_ref()
            .ok_or_else(|| ComputeError::backend("projection has no vocabulary ownership"))?;
        let local = compute(linear.module.forward(input.as_array(), context))?;
        let widths = range
            .balanced_peer_widths(parallel.size(), parallel.rank())
            .map_err(ComputeError::backend)?;
        compute_tensor(
            crate::backend::distributed::all_gather_uneven_axis(
                &local, -1, &widths, parallel, context,
            )
            .map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "vocabulary projection gather failed for local range {:?}, shape {:?}, and widths {widths:?}: {error}",
                    range.local,
                    local.shape(),
                ))
            }),
        )
    }

    fn vocabulary_parallel_embedding_project(
        embedding: &mut MlxEmbedding,
        input: &MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let range = embedding
            .vocabulary_range
            .clone()
            .ok_or_else(|| ComputeError::backend("embedding has no vocabulary ownership"))?;
        let local = compute(embedding.module.as_linear(input.as_array(), context))?;
        let widths = range
            .balanced_peer_widths(parallel.size(), parallel.rank())
            .map_err(ComputeError::backend)?;
        compute_tensor(
            crate::backend::distributed::all_gather_uneven_axis(
                &local, -1, &widths, parallel, context,
            )
            .map_err(|error| {
                safemlx::error::Exception::custom(format!(
                    "tied vocabulary projection gather failed for local range {:?}, shape {:?}, and widths {widths:?}: {error}",
                    range.local,
                    local.shape(),
                ))
            }),
        )
    }

    fn sum_parallel(
        value: MlxTensor,
        parallel: &Group,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(crate::backend::runtime::distributed::all_sum(
            value.as_array(),
            parallel,
            context,
        ))
    }
}

impl HyperNeuralBackend for MlxNeuralBackend {
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

impl GroupedNeuralBackend for MlxNeuralBackend {
    type Selector = MlxTopKGroupSelector;
    type GatedProductGroups = MlxGroupedGatedProduct;
    type Relu2Groups = MlxGroupedRelu2;

    fn grouped_linear(
        linear: &mut MlxLinear,
        input: &MlxTensor,
        groups: i32,
        output_per_group: i32,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let input = input.as_array();
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
            let selected = compute(projected.try_index_device(
                (
                    ..,
                    group,
                    ..,
                    group * output_per_group..(group + 1) * output_per_group,
                ),
                context,
            ))?;
            pieces.push(compute(selected.expand_dims(1, context))?);
        }
        compute_tensor(concatenate_axis(&pieces, 1, context))
    }

    fn joint_group_selection(
        input: JointGroupSelectionInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<JointGroupSelection<MlxTensor>, ComputeError> {
        input.validate()?;
        let hidden_width = input.hidden().as_array().dim(-1);
        let flat = compute(
            input
                .hidden()
                .as_array()
                .reshape(&[-1, hidden_width], context),
        )?;
        let logits = compute(matmul(
            &flat,
            &compute(input.weight().as_array().transpose(context))?,
            context,
        ))?;
        let primary = compute(logits.try_index_device((.., ..input.selectable_groups()), context))?;
        let always_on =
            compute(logits.try_index_device((.., input.selectable_groups()..), context))?;
        let choice = compute(sigmoid(&primary, context))?;
        let choice = compute(choice.add(input.correction_bias().as_array(), context))?;
        let primary_indices = compute(argpartition_axis(choice, -input.top_k(), -1, context))?;
        let primary_indices =
            compute(primary_indices.try_index_device((.., -input.top_k()..), context))?;
        let selected_logits = compute(take_along_axis(&primary, &primary_indices, -1, context))?;
        let all_logits = compute(concatenate_axis(&[selected_logits, always_on], -1, context))?;
        let coefficients = compute(nn::log_sigmoid(all_logits, context))?;
        let coefficients = compute(softmax_axis(coefficients, -1, true, context))?;
        let coefficients =
            compute(coefficients.multiply(Array::from_f32(input.coefficient_scale()), context))?;
        let coefficients =
            compute(coefficients.multiply(input.global_scale().as_array(), context))?;
        let primary_coefficients =
            compute(coefficients.try_index_device((.., ..input.top_k()), context))?;
        let always_on_coefficients =
            compute(coefficients.try_index_device((.., input.top_k()..), context))?;
        Ok(JointGroupSelection::new(
            MlxTensor::from_array(primary_indices),
            MlxTensor::from_array(primary_coefficients),
            MlxTensor::from_array(always_on_coefficients),
        ))
    }

    fn top_k_group_selector(
        spec: TopKGroupSelectorSpec,
        context: &Stream,
    ) -> Result<Self::Selector, ComputeError> {
        spec.validate()?;
        let selection = spec.selection();
        let score_function = match selection.scoring() {
            GroupScoring::Softmax => common::grouped::TopKGroupScoring::Softmax,
            GroupScoring::SelectedSoftmax => common::grouped::TopKGroupScoring::SelectedSoftmax,
            GroupScoring::Sigmoid => common::grouped::TopKGroupScoring::Sigmoid,
            GroupScoring::SqrtSoftplus => common::grouped::TopKGroupScoring::SqrtSoftplus,
            _ => return Err(ComputeError::backend("unsupported group scoring policy")),
        };
        let module = compute(common::grouped::TopKGroupSelector::new_with_quantization(
            compute(common::grouped::TopKGroupSelectorConfig::new(
                selection.top_k(),
                selection.group_count(),
                spec.input_dimensions(),
                score_function,
                selection.normalize_selected(),
                selection.normalization_epsilon(),
                selection.coefficient_scale(),
                selection.selection_partitions(),
                selection.selected_groups(),
                spec.bias().is_some(),
                spec.correction_bias().is_some(),
                spec.input_transform().map(|transform| transform.epsilon()),
                spec.input_transform()
                    .is_some_and(|transform| transform.inverse_sqrt_dimensions()),
                spec.coefficient_scale().is_some(),
            ))?,
            spec.format().encoding().weight_quantization(),
            context,
        ))?;
        let weight = spec.weight().clone();
        let mut topology = vec![("weight", weight.clone())];
        if let Some(bias) = spec.bias() {
            topology.push(("bias", bias.clone()));
        }
        if let Some(scale) = spec.format().scale() {
            topology.push(("scales", bind_linear_companion(&weight, scale.clone())));
        }
        if let Some(bias) = spec.format().affine_bias() {
            topology.push(("biases", bind_linear_companion(&weight, bias.clone())));
        }
        if let Some(correction_bias) = spec.correction_bias() {
            topology.push(("e_score_correction_bias", correction_bias.clone()));
        }
        if let Some(transform) = spec.input_transform() {
            topology.push(("input_scale", transform.scale().clone()));
        }
        if let Some(learned_coefficient_scale) = spec.coefficient_scale() {
            topology.push((
                "learned_coefficient_scale",
                learned_coefficient_scale.clone(),
            ));
        }
        Ok(MlxTopKGroupSelector {
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        context: &Stream,
    ) -> Result<Self::GatedProductGroups, ComputeError> {
        spec.validate()?;
        if spec.input_dimensions() != spec.output_dimensions() {
            return Err(ComputeError::backend(
                "MLX packed gated-product groups require equal input and output dimensions",
            ));
        }
        let policy = spec.policy();
        let GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
            return Err(ComputeError::backend(
                "independent group units must be acquired through a runtime group provider",
            ));
        };
        let native_fp8 = match (gate_up.format().encoding(), down.format().encoding()) {
            (LinearFormat::E4M3BlockFp8(gate), LinearFormat::E4M3BlockFp8(down))
                if gate == down
                    && gate.scale_encoding == eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 =>
            {
                true
            }
            (LinearFormat::E4M3BlockFp8(_), LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "MLX packed block-FP8 groups require matching UE8M0 formats",
                ));
            }
            (LinearFormat::E4M3BlockFp8(_), _) | (_, LinearFormat::E4M3BlockFp8(_)) => {
                return Err(ComputeError::backend(
                    "packed group projections must use one physical format",
                ));
            }
            _ => false,
        };
        let mut module = compute(common::grouped::PackedGatedProductGroups::new(
            spec.group_count(),
            spec.input_dimensions(),
            spec.intermediate_dimensions(),
            gate_up.format().encoding().weight_quantization(),
            down.format().encoding().weight_quantization(),
            [gate_up.bias().is_some(), down.bias().is_some()],
            context,
        ))?;
        module = compute(module.with_policy(policy))?;
        if native_fp8 {
            module = compute(module.with_native_fp8_e8m0(context))?;
        }
        let mut topology = vec![
            ("gate_up_proj", gate_up.weight().clone()),
            ("down_proj", down.weight().clone()),
        ];
        if let Some(bias) = gate_up.bias() {
            topology.push(("gate_up_proj_bias", bias.clone()));
        }
        if let Some(bias) = down.bias() {
            topology.push(("down_proj_bias", bias.clone()));
        }
        if let Some(scale) = gate_up.format().scale() {
            topology.push((
                "gate_up_proj_scales",
                bind_linear_companion(gate_up.weight(), scale.clone()),
            ));
        }
        if let Some(bias) = gate_up.format().affine_bias() {
            topology.push((
                "gate_up_proj_biases",
                bind_linear_companion(gate_up.weight(), bias.clone()),
            ));
        }
        if let Some(scale) = down.format().scale() {
            topology.push((
                "down_proj_scales",
                bind_linear_companion(down.weight(), scale.clone()),
            ));
        }
        if let Some(bias) = down.format().affine_bias() {
            topology.push((
                "down_proj_biases",
                bind_linear_companion(down.weight(), bias.clone()),
            ));
        }
        Ok(MlxGroupedGatedProduct {
            spec,
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }

    fn grouped_relu2(
        spec: GroupedRelu2Spec,
        context: &Stream,
    ) -> Result<Self::Relu2Groups, ComputeError> {
        spec.validate()?;
        if spec.up().bias().is_some() || spec.down().bias().is_some() {
            return Err(ComputeError::backend(
                "MLX packed ReLU2 groups do not support ordinary projection biases",
            ));
        }
        let module = compute(common::grouped::PackedRelu2Groups::new(
            spec.group_count(),
            spec.hidden_dimensions(),
            spec.intermediate_dimensions(),
            [
                spec.up().format().encoding().weight_quantization(),
                spec.down().format().encoding().weight_quantization(),
            ],
            context,
        ))?;
        let mut topology = vec![
            ("up_proj", spec.up().weight().clone()),
            ("down_proj", spec.down().weight().clone()),
        ];
        if let Some(scale) = spec.up().format().scale() {
            topology.push((
                "up_proj_scales",
                bind_linear_companion(spec.up().weight(), scale.clone()),
            ));
        }
        if let Some(bias) = spec.up().format().affine_bias() {
            topology.push((
                "up_proj_biases",
                bind_linear_companion(spec.up().weight(), bias.clone()),
            ));
        }
        if let Some(scale) = spec.down().format().scale() {
            topology.push((
                "down_proj_scales",
                bind_linear_companion(spec.down().weight(), scale.clone()),
            ));
        }
        if let Some(bias) = spec.down().format().affine_bias() {
            topology.push((
                "down_proj_biases",
                bind_linear_companion(spec.down().weight(), bias.clone()),
            ));
        }
        Ok(MlxGroupedRelu2 {
            spec,
            module: MlxNamedModule::with_exact_topology(module, topology)?,
        })
    }
}

macro_rules! impl_attention_cache {
    ($type:ty) => {
        impl AttentionCache<MlxTensor> for $type {
            fn offset(&self) -> i32 {
                KeyValueCache::offset(self)
            }
            fn max_size(&self) -> Option<i32> {
                KeyValueCache::max_size(self)
            }
            fn update_for_attention(
                &mut self,
                keys: MlxTensor,
                values: MlxTensor,
                context: &Stream,
            ) -> Result<(MlxTensor, MlxTensor), ComputeError> {
                compute(KeyValueCache::update_for_attention(
                    self,
                    keys.into_array(),
                    values.into_array(),
                    context,
                ))
                .map(|(keys, values)| (MlxTensor::from_array(keys), MlxTensor::from_array(values)))
            }
            fn attention(
                &mut self,
                request: AttentionRequest<'_, MlxTensor>,
                context: &Stream,
            ) -> Result<MlxTensor, ComputeError> {
                request.validate()?;
                if let Some(output) = compute(KeyValueCache::paged_attention(
                    self,
                    request.queries.as_array(),
                    request.scale,
                    request.mask.map(MlxTensor::as_array),
                    request.sinks.map(MlxTensor::as_array),
                    context,
                ))? {
                    return Ok(MlxTensor::from_array(output));
                }
                compute_tensor(safemlx::fast::scaled_dot_product_attention(
                    request.queries.as_array(),
                    request.keys.as_array(),
                    request.values.as_array(),
                    request.scale,
                    request
                        .mask
                        .map(|mask| ScaledDotProductAttentionMask::Array(mask.as_array())),
                    request.sinks.map(MlxTensor::as_array),
                    context,
                ))
            }
        }
    };
}

impl_attention_cache!(ConcatKeyValueCache);
impl_attention_cache!(PagedKeyValueCache);
impl_attention_cache!(crate::backend::runtime::cache::state::MlxKeyValueLayerState);

impl eredu_nn::AuxiliaryConvolutionState<MlxTensor>
    for crate::backend::runtime::cache::state::MlxHybridLayerState
{
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<MlxTensor>, ComputeError> {
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
    use eredu_architectures::operator_requirements;
    use eredu_checkpoint::{AffineQuantization, LinearFormat, WeightQuantization};
    use eredu_core::Completion as _;
    use eredu_nn::{
        reference_expand_heads, reference_segmented_attention, EmbeddingLookupPolicy,
        EmbeddingOperator, EmbeddingSpec, FusedProjectionLayout, FusedProjectionSegment,
        GatedProductGroupLayout, GroupedGatedProductSpec, GroupedNeuralBackend,
        GroupedProjectionSpec, GroupedRelu2Spec, HeadExpansion, JointGroupSelectionInput,
        JointGroupSelectionSpec, LinearOperator, LinearSpec, NeuralBackend,
        NormalizationConstructionSpec, NormalizationScale, ParameterSpec, RelativeAttentionInput,
        SegmentedAttentionInput, Tensor,
    };
    use eredu_runtime::{
        BarrierBackend, BroadcastBackend, CommunicationBackend, CommunicationPeerCounts,
        EvenGatherBackend, SumReductionBackend, UnevenGatherBackend, VariableAllToAllBackend,
    };
    use safemlx::{
        ops::{quantize_with_mode, QuantizationMode},
        transforms::async_eval_with_event,
        Array, Device, DeviceType, Dtype,
    };

    use crate::backend::{
        nn::{linear::PhysicalEmbedding, tensor::TokenValidationScope},
        ExecutionContext,
    };

    use super::{MlxEmbedding, MlxLinear, MlxNeuralBackend, MlxTensor};

    fn singleton_communication() -> (crate::backend::runtime::distributed::Group, safemlx::Stream) {
        let native =
            safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let group = crate::backend::runtime::distributed::Group::uncontracted(&native);
        let stream = safemlx::Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        (group, stream)
    }

    #[test]
    fn local_dependency_submission_retains_exact_outputs_and_executor_under_bound() {
        use eredu_core::BoundedCompletion as _;

        let (_, stream) = singleton_communication();
        let values = [
            MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 2])),
            MlxTensor::from_array(Array::from_slice(&[3.0_f32], &[1, 1])),
        ];
        crate::backend::runtime::distributed::completion::force_next_communication_pending();
        let submission = <MlxNeuralBackend as CommunicationBackend>::submit_local_dependencies(
            values.iter(),
            &stream,
        )
        .unwrap();
        assert_eq!(submission.completion.submitted_outputs(), values.len());
        assert_eq!(submission.completion.retained_arrays(), values.len());
        assert_eq!(submission.completion.retained_streams(), 1);
        assert_eq!(submission.completion.retained_groups(), 0);
        assert_eq!(submission.completion.retained_routes(), 0);
        let policy = eredu_core::BoundedCompletionWait::new(
            std::time::Duration::from_millis(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        assert_eq!(
            submission.completion.wait_bounded(policy).unwrap(),
            eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            }
        );
        crate::backend::runtime::distributed::completion::release_forced_pending_orphans();
    }

    #[test]
    fn fine_grained_collectives_retain_exact_singleton_resources() {
        let (group, stream) = singleton_communication();

        let reduced = <MlxNeuralBackend as SumReductionBackend>::all_reduce_sum(
            MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 2])),
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(reduced.completion.retained_arrays(), 2);
        assert_eq!(reduced.completion.retained_count_buffers(), 0);
        assert_eq!(reduced.completion.retained_groups(), 1);
        assert_eq!(reduced.completion.retained_routes(), 0);
        assert_eq!(reduced.completion.retained_streams(), 1);
        assert_eq!(
            reduced
                .wait()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[1.0, 2.0]
        );

        let gathered = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
            MlxTensor::from_array(Array::from_slice(&[3.0_f32, 4.0], &[1, 2])),
            1,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(gathered.completion.retained_arrays(), 2);
        let gathered = gathered.wait().unwrap();
        assert_eq!(gathered.as_array().shape(), &[1, 2]);

        let uneven = <MlxNeuralBackend as UnevenGatherBackend>::all_gather_uneven(
            MlxTensor::from_array(Array::from_slice(&[5.0_f32, 6.0], &[1, 2])),
            &[2],
            1,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(uneven.completion.retained_count_buffers(), 1);
        let uneven = uneven.wait().unwrap();
        assert_eq!(uneven.as_array().shape(), &[1, 2]);

        let counts = CommunicationPeerCounts::new(vec![2], vec![2], 1).unwrap();
        let exchanged = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
            MlxTensor::from_array(Array::from_slice(&[7.0_f32, 8.0, 9.0, 10.0], &[2, 2])),
            &counts,
            1,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(exchanged.completion.retained_arrays(), 2);
        assert_eq!(exchanged.completion.retained_count_buffers(), 2);
        assert_eq!(
            exchanged
                .wait()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[7.0, 8.0, 9.0, 10.0]
        );

        let broadcast = <MlxNeuralBackend as BroadcastBackend>::broadcast(
            MlxTensor::from_array(Array::from_slice(&[11.0_f32, 12.0], &[2])),
            0,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(broadcast.completion.retained_arrays(), 3);
        assert_eq!(broadcast.completion.retained_groups(), 1);
        assert_eq!(broadcast.completion.retained_streams(), 1);
        assert_eq!(
            broadcast
                .wait()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            &[11.0, 12.0]
        );

        let barrier = <MlxNeuralBackend as BarrierBackend>::barrier(&group, &stream).unwrap();
        assert_eq!(barrier.retained_arrays(), 2);
        assert_eq!(barrier.retained_groups(), 1);
        assert_eq!(barrier.retained_streams(), 1);
        barrier.wait().unwrap();
    }

    #[test]
    fn fine_grained_collectives_admit_i32_only_for_count_gather_and_variable_exchange() {
        let (group, stream) = singleton_communication();

        let gathered = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
            MlxTensor::from_array(Array::from_slice(&[2_i32, 0, 3], &[3])),
            0,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(gathered.completion.retained_arrays(), 2);
        assert_eq!(
            gathered
                .wait()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[2, 0, 3]
        );

        let reduction_error = <MlxNeuralBackend as SumReductionBackend>::all_reduce_sum(
            MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
            &group,
            &stream,
        )
        .expect_err("integer reduction is not advertised");
        assert!(reduction_error.what().contains("does not advertise dtype"));

        let uneven_error = <MlxNeuralBackend as UnevenGatherBackend>::all_gather_uneven(
            MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
            &[1],
            0,
            &group,
            &stream,
        )
        .expect_err("integer uneven gather is not advertised");
        assert!(uneven_error.what().contains("does not advertise dtype"));

        let counts = CommunicationPeerCounts::new(vec![1], vec![1], 1).unwrap();
        let exchanged = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
            MlxTensor::from_array(Array::from_slice(&[1_i32], &[1])),
            &counts,
            0,
            &group,
            &stream,
        )
        .unwrap();
        assert_eq!(exchanged.completion.retained_arrays(), 2);
        assert_eq!(exchanged.completion.retained_count_buffers(), 2);
        assert_eq!(exchanged.completion.retained_groups(), 1);
        assert_eq!(exchanged.completion.retained_streams(), 1);
        assert_eq!(
            exchanged
                .wait()
                .unwrap()
                .as_array()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1]
        );

        let unsigned_error = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
            MlxTensor::from_array(Array::from_slice(&[1_u32], &[1])),
            0,
            &group,
            &stream,
        )
        .expect_err("unsigned count gather is outside the exact admitted set");
        assert!(unsigned_error.what().contains("does not advertise dtype"));

        let unsigned_exchange = <MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
            MlxTensor::from_array(Array::from_slice(&[1_u32], &[1])),
            &counts,
            0,
            &group,
            &stream,
        )
        .expect_err("unsigned variable exchange is outside the exact admitted set");
        assert!(unsigned_exchange
            .what()
            .contains("does not advertise dtype"));
    }

    #[test]
    fn mlx_declares_every_supported_architecture_operator_set() {
        let declared = <MlxNeuralBackend as NeuralBackend>::OPERATOR_CAPABILITIES;
        for required in [
            operator_requirements::KIMI_LINEAR,
            operator_requirements::QWEN_HYBRID,
            operator_requirements::NEMOTRON_H,
            operator_requirements::QWEN_VISION,
            operator_requirements::QWEN_VL,
            operator_requirements::DEEPSEEK_V3,
            operator_requirements::DEEPSEEK_V4,
            operator_requirements::INKLING,
            operator_requirements::GEMMA4,
            operator_requirements::MUSE_GLIMMER,
            eredu_nn::NeuralOperatorCapabilities::ATTENTION_SINKS,
            eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
        ] {
            assert!(declared.contains(required));
        }
    }

    fn close(actual: &MlxTensor, expected: &[f32], tolerance: f32) {
        let actual = actual.as_array().evaluated().unwrap();
        assert_eq!(actual.as_slice::<f32>().len(), expected.len());
        assert!(actual
            .as_slice::<f32>()
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= tolerance));
    }

    #[test]
    #[ignore = "explicit MLX dtype regression; run outside the sandbox"]
    fn mlx_weighted_rms_norm_preserves_bfloat16_input_dtype() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let input = MlxTensor::from_array(
            Array::from_slice(&[1.0_f32, -2.0, 3.0, -4.0], &[1, 4])
                .as_dtype(Dtype::Bfloat16, stream)
                .unwrap(),
        );
        let weight = MlxTensor::from_array(Array::from_slice(&[1.0_f32; 4], &[4]));
        let output = <MlxNeuralBackend as NeuralBackend>::rms_norm_with_weight(
            &input, &weight, 1e-5, stream,
        )
        .unwrap();
        assert_eq!(output.as_array().dtype(), Dtype::Bfloat16);
        output.as_array().evaluated().unwrap();
    }

    fn parameter(name: &str) -> ParameterSpec {
        ParameterSpec::trainable(name).unwrap()
    }

    fn test_format(weight: &str, format: LinearFormat) -> eredu_nn::LinearFormatSpec {
        let prefix = weight.strip_suffix(".weight").unwrap_or(weight);
        match format {
            LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => {
                eredu_nn::LinearFormatSpec::unscaled(format).unwrap()
            }
            LinearFormat::E4M3BlockFp8(_) => eredu_nn::LinearFormatSpec::scaled(
                format,
                parameter(&format!("{prefix}.weight_scale_inv")),
            )
            .unwrap(),
            LinearFormat::MxFp4 => {
                eredu_nn::LinearFormatSpec::scaled(format, parameter(&format!("{prefix}.scales")))
                    .unwrap()
            }
            LinearFormat::Affine(_) => eredu_nn::LinearFormatSpec::affine(
                format,
                parameter(&format!("{prefix}.scales")),
                parameter(&format!("{prefix}.biases")),
            )
            .unwrap(),
        }
    }

    fn affine_group_projection(weight: &str, scales: &str, biases: &str) -> GroupedProjectionSpec {
        GroupedProjectionSpec::new(
            parameter(weight),
            None,
            eredu_nn::LinearFormatSpec::affine(
                LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
                parameter(scales),
                parameter(biases),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_quantized_topologies_use_literal_neutral_identities() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let affine = LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap());
        let linear = <MlxNeuralBackend as NeuralBackend>::linear(
            LinearSpec {
                input: 32,
                output: 32,
                weight: parameter("arbitrary.linear.matrix"),
                bias: None,
                format: eredu_nn::LinearFormatSpec::affine(
                    affine,
                    parameter("arbitrary.linear.scale"),
                    parameter("arbitrary.linear.affine"),
                )
                .unwrap(),
            },
            stream,
        )
        .unwrap();
        let linear_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&linear)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            linear_ids,
            [
                "arbitrary.linear.affine",
                "arbitrary.linear.matrix",
                "arbitrary.linear.scale",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect()
        );

        let embedding = <MlxNeuralBackend as NeuralBackend>::embedding(
            EmbeddingSpec {
                vocabulary: 32,
                dimensions: 32,
                weight: parameter("arbitrary.embedding.table"),
                format: eredu_nn::LinearFormatSpec::affine(
                    affine,
                    parameter("arbitrary.embedding.scale"),
                    parameter("arbitrary.embedding.affine"),
                )
                .unwrap(),
            },
            stream,
        )
        .unwrap();
        let embedding_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&embedding)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            embedding_ids,
            [
                "arbitrary.embedding.affine",
                "arbitrary.embedding.scale",
                "arbitrary.embedding.table",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect()
        );

        let gate = affine_group_projection(
            "arbitrary.gated.matrix_a",
            "arbitrary.gated.scale_a",
            "arbitrary.gated.affine_a",
        );
        let down = affine_group_projection(
            "arbitrary.gated.matrix_b",
            "arbitrary.gated.scale_b",
            "arbitrary.gated.affine_b",
        );
        let gated = <MlxNeuralBackend as GroupedNeuralBackend>::grouped_gated_product(
            GroupedGatedProductSpec::new(
                2,
                32,
                32,
                32,
                eredu_nn::GatedProductPolicy::ordinary_silu(),
                GatedProductGroupLayout::Packed {
                    gate_up: gate,
                    down,
                },
            )
            .unwrap(),
            stream,
        )
        .unwrap();
        let gated_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&gated)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            gated_ids,
            [
                "arbitrary.gated.affine_a",
                "arbitrary.gated.affine_b",
                "arbitrary.gated.matrix_a",
                "arbitrary.gated.matrix_b",
                "arbitrary.gated.scale_a",
                "arbitrary.gated.scale_b",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect()
        );

        let relu = <MlxNeuralBackend as GroupedNeuralBackend>::grouped_relu2(
            GroupedRelu2Spec::new(
                2,
                32,
                32,
                affine_group_projection(
                    "arbitrary.relu.matrix_a",
                    "arbitrary.relu.scale_a",
                    "arbitrary.relu.affine_a",
                ),
                affine_group_projection(
                    "arbitrary.relu.matrix_b",
                    "arbitrary.relu.scale_b",
                    "arbitrary.relu.affine_b",
                ),
            )
            .unwrap(),
            stream,
        )
        .unwrap();
        let relu_ids = eredu_nn::validate_parameter_topology::<MlxTensor, _>(&relu)
            .unwrap()
            .into_iter()
            .map(|parameter| parameter.id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            relu_ids,
            [
                "arbitrary.relu.affine_a",
                "arbitrary.relu.affine_b",
                "arbitrary.relu.matrix_a",
                "arbitrary.relu.matrix_b",
                "arbitrary.relu.scale_a",
                "arbitrary.relu.scale_b",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect()
        );
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
        let mut linear = <MlxNeuralBackend as NeuralBackend>::linear(
            LinearSpec {
                input,
                output,
                weight: parameter(name),
                bias: None,
                format: test_format(name, format),
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
            PhysicalEmbedding::Dense(embedding) => embedding.weight.value = weight,
            PhysicalEmbedding::Quantized(embedding) => {
                let arrays = quantize_with_mode(
                    &weight,
                    embedding.group_size,
                    embedding.bits,
                    embedding.mode,
                    stream,
                )
                .unwrap();
                embedding.inner.weight.value = arrays.weight;
                embedding.scales.value = Some(arrays.scales);
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
        let mut embedding = <MlxNeuralBackend as NeuralBackend>::embedding(
            EmbeddingSpec {
                vocabulary,
                dimensions,
                weight: parameter(name),
                format: test_format(name, quantization.into()),
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
            tokens: &MlxTensor,
            stream: &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> {
            embedding.lookup(tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
        }
        fn project(
            linear: &mut MlxLinear,
            input: &MlxTensor,
            stream: &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> {
            linear.forward(input, stream)
        }
        fn sum(
            embeddings: &mut MultiTableEmbedding<MlxNeuralBackend>,
            tokens: &[&MlxTensor],
            stream: &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> {
            embeddings.forward(tokens, stream)
        }

        let _: fn(
            &mut MlxEmbedding,
            &MlxTensor,
            &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> = lookup;
        let _: fn(
            &mut MlxLinear,
            &MlxTensor,
            &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> = project;
        let _: fn(
            &mut MultiTableEmbedding<MlxNeuralBackend>,
            &[&MlxTensor],
            &safemlx::Stream,
        ) -> Result<MlxTensor, eredu_nn::Error> = sum;
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
        let input_array = MlxTensor::from_array(Array::from_slice(&input, &[1, input_width]));
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
        let split_output = safemlx::ops::concatenate_axis(
            &split_outputs
                .iter()
                .map(MlxTensor::as_array)
                .collect::<Vec<_>>(),
            -1,
            stream,
        )
        .unwrap();
        let fused_evaluated = fused_output.as_array().evaluated().unwrap();
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
            let valid = MlxTensor::from_array(Array::from_slice(&[-1_i32, 0, 3], &[3]));
            let scope = TokenValidationScope::begin().unwrap();
            let output = embedding
                .lookup(&valid, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                .unwrap();
            let output = output.as_array().as_type::<f32>(stream).unwrap();
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
                let tokens = MlxTensor::from_array(Array::from_slice(&[invalid], &[1]));
                let scope = TokenValidationScope::begin().unwrap();
                let output = embedding
                    .lookup(&tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                    .expect("lazy lookup must not synchronize while building the graph");
                let validations = scope.finish();
                let event = async_eval_with_event(
                    std::iter::once(output.as_array()).chain(validations.arrays()),
                )
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
                        format: test_format(&format!("tables.{table}.weight"), quantization.into()),
                    },
                    lookup: EmbeddingLookupPolicy::ZeroSentinel(-1),
                })
                .collect::<Vec<_>>();
            let mut embeddings =
                MultiTableEmbedding::<MlxNeuralBackend>::new(specs, stream).unwrap();
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
            let first = MlxTensor::from_array(Array::from_slice(&[0_i32], &[1]));
            let sentinel = MlxTensor::from_array(Array::from_slice(&[-1_i32], &[1]));
            let third = MlxTensor::from_array(Array::from_slice(&[2_i32], &[1]));
            let scope = TokenValidationScope::begin().unwrap();
            let output = embeddings
                .forward(&[&first, &sentinel, &third], stream)
                .unwrap();
            assert_eq!(output.shape(), &[1, dimensions]);
            let output = output.as_array().as_type::<f32>(stream).unwrap();
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
        let queries = MlxTensor::from_array(Array::from_slice(&[0.0_f32], &[1, 1, 1, 1]));
        let keys = MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[1, 1, 2, 1]));
        let values = MlxTensor::from_array(Array::from_slice(&[10.0_f32, 20.0], &[1, 1, 2, 1]));
        let profiles =
            MlxTensor::from_array(Array::from_slice(&[0.0_f32, 3.0_f32.ln()], &[1, 1, 1, 2]));
        let output = <MlxNeuralBackend as NeuralBackend>::relative_attention(
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
        let output = output.as_array().evaluated().unwrap();
        assert!((output.as_slice::<f32>()[0] - 12.5).abs() < 1e-5);
    }

    #[test]
    fn joint_group_selection_selects_with_bias_but_weights_unbiased_logits() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let hidden = MlxTensor::from_array(Array::from_slice(&[1.0_f32], &[1, 1]));
        let weight = MlxTensor::from_array(Array::from_slice(&[0.0_f32, 1.0, 2.0], &[3, 1]));
        let correction = MlxTensor::from_array(Array::from_slice(&[10.0_f32, 0.0], &[2]));
        let global = MlxTensor::from_array(Array::from_slice(&[0.5_f32], &[1]));
        let selections = <MlxNeuralBackend as GroupedNeuralBackend>::joint_group_selection(
            JointGroupSelectionInput::new(
                &hidden,
                &weight,
                &correction,
                &global,
                JointGroupSelectionSpec::new(2, 1, 1, 2.0).unwrap(),
            )
            .unwrap(),
            stream,
        )
        .unwrap();
        let ids = selections.primary_indices().as_array().evaluated().unwrap();
        let grouped = selections
            .primary_coefficients()
            .as_array()
            .evaluated()
            .unwrap();
        let shared = selections
            .always_on_coefficients()
            .as_array()
            .evaluated()
            .unwrap();
        assert_eq!(ids.as_slice::<u32>(), &[0]);
        let expected = 0.5 / (0.5 + 1.0 / (1.0 + (-2.0_f32).exp()));
        assert!((grouped.as_slice::<f32>()[0] - expected).abs() < 1e-5);
        assert!((grouped.as_slice::<f32>()[0] + shared.as_slice::<f32>()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "explicit MLX normalization parity; run outside the sandbox"]
    fn mlx_general_normalization_matches_scalar_references() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = execution.stream();
        let values = [1.0_f32, -2.0, 3.0, -4.0];
        let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 4]));
        let epsilon = 1e-5;

        let mut normalization = <MlxNeuralBackend as NeuralBackend>::normalization(
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
            &MlxTensor::from_array(normalization.forward(input.as_array(), stream).unwrap()),
            &expected_rms,
            1e-5,
        );

        let l2 = (values.iter().map(|value| value * value).sum::<f32>() + epsilon).sqrt();
        let expected_l2 = values.map(|value| value / l2);
        close(
            &<MlxNeuralBackend as NeuralBackend>::l2_normalize(&input, epsilon, stream).unwrap(),
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
        let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 4]));
        let gate = MlxTensor::from_array(Array::from_slice(&gates, &[1, 4]));
        let weight = MlxTensor::from_array(Array::from_slice(&weights, &[4]));
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
        let actual = <MlxNeuralBackend as NeuralBackend>::silu_gated_group_rms_norm(
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
        let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2, 2]));
        let expansion = HeadExpansion {
            axis: 1,
            source_heads: 2,
            target_heads: 4,
        };
        let actual =
            <MlxNeuralBackend as NeuralBackend>::expand_heads(&input, expansion, stream).unwrap();
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
        let query = MlxTensor::from_array(Array::from_slice(&queries, &[4, 1, 2]));
        let key = MlxTensor::from_array(Array::from_slice(&keys, &[4, 1, 2]));
        let value = MlxTensor::from_array(Array::from_slice(&values, &[4, 1, 2]));
        let segments = [2, 2];
        let scale = 2.0_f32.sqrt().recip();
        let actual = <MlxNeuralBackend as NeuralBackend>::segmented_attention(
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
