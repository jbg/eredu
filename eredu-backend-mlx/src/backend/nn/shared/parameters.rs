use super::*;
use eredu_nn::{ParameterMetadataView, ParameterSourceError, ParameterSourceVisitor};
mod sources;
mod topology;
pub(super) use topology::{NativeParameterTopology, NamedParameterTopology, PreparedParameterTopology};
pub(super) mod compact;
pub(crate) use compact::PreparedCompactBindings;
pub(super) use compact::CompactBindingValues;
pub(super) use sources::visit_module_parameter_sources;
mod count;
pub(crate) use count::{
    add_parameter_count, visit_parameter_map, ParameterCountError, ParameterSourceCounter,
    ParameterSourceCounts,
};

pub(super) trait NativeParameterSourceVisitor<'a> {
    fn parameter(&mut self, local: &'static str, value: &'a Array, trainable: bool);
    fn retained(&mut self, value: &'a Array);
}

pub(super) trait NativeParameterSourceVisitorMut<'a> {
    fn parameter(&mut self, local: &'static str, value: &'a mut Array, trainable: bool);
}

/// Cold numerical inventory of an audited physical leaf. Parameter traversal
/// alone is not evidence that an arbitrary native module has no other storage.
pub(super) trait NativeRetainedValues: PhysicalParameters {
    /// Exact currently present named fields, or unsupported. This is distinct
    /// from the maximum numerical slot bound and grants no storage authority.
    fn native_parameter_source_count(&self) -> Option<usize> {
        None
    }
    fn visit_native_parameter_sources<'a>(
        &'a self,
        _visitor: &mut dyn NativeParameterSourceVisitor<'a>,
    ) -> Result<(), ParameterSourceError> {
        Err(ParameterSourceError::Unavailable)
    }
    fn visit_native_parameter_sources_mut<'a>(
        &'a mut self,
        _visitor: &mut dyn NativeParameterSourceVisitorMut<'a>,
    ) -> Result<(), ParameterSourceError> {
        Err(ParameterSourceError::Unavailable)
    }
    fn native_retained_value_slot_bound(&self) -> Option<usize> {
        None
    }
    fn visit_native_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool;
}

fn visit_native_array(value: &Array, visitor: &mut dyn FnMut(&MlxTensor)) {
    visitor(MlxTensor::ref_cast(value));
}

fn visit_native_optional_array(value: &Option<Array>, visitor: &mut dyn FnMut(&MlxTensor)) {
    if let Some(value) = value {
        visit_native_array(value, visitor);
    }
}

// These audited leaves store numerical values only in the listed fields.
// Borrow the actual fields rather than constructing parameter trees, flattened
// maps, or names. Optional companions remain independent: absence of one is
// not evidence that another is absent. Callers own callback storage and errors.
macro_rules! parameter_complete_native_leaf {
    ($leaf:ty, [$($required:ident),* $(,)?], [$($optional:ident),* $(,)?]) => {
        impl NativeRetainedValues for $leaf {
            fn native_parameter_source_count(&self) -> Option<usize> {
                let mut count = 0usize;
                $(let _ = &self.$required; count = count.checked_add(1)?;)*
                $(count = count.checked_add(usize::from(self.$optional.value.is_some()))?;)*
                Some(count)
            }
            fn visit_native_parameter_sources<'a>(&'a self, visitor: &mut dyn NativeParameterSourceVisitor<'a>)
                -> Result<(), ParameterSourceError> {
                $(visitor.parameter(stringify!($required), &self.$required.value, self.$required.is_trainable());)*
                $(if let Some(value) = &self.$optional.value {
                    visitor.parameter(stringify!($optional), value, self.$optional.is_trainable());
                })*
                Ok(())
            }
            fn visit_native_parameter_sources_mut<'a>(&'a mut self, visitor: &mut dyn NativeParameterSourceVisitorMut<'a>)
                -> Result<(), ParameterSourceError> {
                $(let trainable = self.$required.is_trainable();
                  visitor.parameter(stringify!($required), &mut self.$required.value, trainable);)*
                $(let trainable = self.$optional.is_trainable();
                  if let Some(value) = &mut self.$optional.value {
                    visitor.parameter(stringify!($optional), value, trainable);
                  })*
                Ok(())
            }
            fn native_retained_value_slot_bound(&self) -> Option<usize> {
                let mut count = 0usize;
                $(let _ = &self.$required; count = count.checked_add(1)?;)*
                $(let _ = &self.$optional; count = count.checked_add(1)?;)*
                Some(count)
            }
            fn visit_native_retained_values(
                &self,
                visitor: &mut dyn FnMut(&MlxTensor),
            ) -> bool {
                $(visit_native_array(&self.$required.value, visitor);)*
                $(visit_native_optional_array(&self.$optional.value, visitor);)*
                true
            }
        }
    };
}

parameter_complete_native_leaf!(
    common::linear::PhysicalLinear,
    [weight],
    [weight_scale_inv, scales, biases, bias]
);
parameter_complete_native_leaf!(nn::RmsNorm, [weight], []);
parameter_complete_native_leaf!(
    common::hyper_connections::HyperConnection,
    [function, base, scale],
    []
);
parameter_complete_native_leaf!(
    common::hyper_connections::HyperHead,
    [function, base, scale],
    []
);
parameter_complete_native_leaf!(
    common::grouped::TopKGroupSelector,
    [weight],
    [
        bias,
        scales,
        biases,
        e_score_correction_bias,
        input_scale,
        learned_coefficient_scale
    ]
);
parameter_complete_native_leaf!(
    common::grouped::PackedGatedProductGroups,
    [gate_up_proj, down_proj],
    [
        gate_up_proj_bias,
        gate_up_proj_scales,
        gate_up_proj_biases,
        down_proj_bias,
        down_proj_scales,
        down_proj_biases
    ]
);
parameter_complete_native_leaf!(
    common::grouped::PackedRelu2Groups,
    [up_proj, down_proj],
    [
        up_proj_scales,
        up_proj_biases,
        down_proj_scales,
        down_proj_biases
    ]
);

impl NativeRetainedValues for common::linear::PhysicalEmbedding {
    fn native_parameter_source_count(&self) -> Option<usize> {
        Some(match self {
            Self::Dense(_) => 1,
            Self::Quantized(value) => {
                1 + usize::from(value.scales.value.is_some())
                    + usize::from(value.biases.value.is_some())
            }
        })
    }
    fn visit_native_parameter_sources<'a>(
        &'a self,
        visitor: &mut dyn NativeParameterSourceVisitor<'a>,
    ) -> Result<(), ParameterSourceError> {
        match self {
            Self::Dense(value) => {
                visitor.parameter("weight", &value.weight.value, value.weight.is_trainable())
            }
            Self::Quantized(value) => {
                visitor.parameter(
                    "inner.weight",
                    &value.inner.weight.value,
                    value.inner.weight.is_trainable(),
                );
                if let Some(array) = &value.scales.value {
                    visitor.parameter("scales", array, value.scales.is_trainable());
                }
                if let Some(array) = &value.biases.value {
                    visitor.parameter("biases", array, value.biases.is_trainable());
                }
                if let Some(native) = &value.native {
                    visitor.retained(native.retained_packed_array());
                }
            }
        }
        Ok(())
    }
    fn visit_native_parameter_sources_mut<'a>(
        &'a mut self,
        visitor: &mut dyn NativeParameterSourceVisitorMut<'a>,
    ) -> Result<(), ParameterSourceError> {
        match self {
            Self::Dense(value) => {
                let trainable = value.weight.is_trainable();
                visitor.parameter("weight", &mut value.weight.value, trainable);
            }
            Self::Quantized(value) => {
                let trainable = value.inner.weight.is_trainable();
                visitor.parameter("inner.weight", &mut value.inner.weight.value, trainable);
                let trainable = value.scales.is_trainable();
                if let Some(array) = &mut value.scales.value {
                    visitor.parameter("scales", array, trainable);
                }
                let trainable = value.biases.is_trainable();
                if let Some(array) = &mut value.biases.value {
                    visitor.parameter("biases", array, trainable);
                }
            }
        }
        Ok(())
    }
    fn native_retained_value_slot_bound(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => Some(1),
            Self::Quantized(_) => Some(4),
        }
    }

    fn visit_native_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        match self {
            Self::Dense(embedding) => {
                visit_native_array(&embedding.weight.value, visitor);
                true
            }
            Self::Quantized(embedding) => {
                visit_native_array(&embedding.inner.weight.value, visitor);
                visit_native_optional_array(&embedding.scales.value, visitor);
                visit_native_optional_array(&embedding.biases.value, visitor);
                if let Some(native) = &embedding.native {
                    visit_native_array(native.retained_packed_array(), visitor);
                }
                true
            }
        }
    }
}

impl MlxNeuralBackend {
    pub(crate) fn prepared_binding_visit_control_bytes(rows: usize) -> Option<usize> {
        // One immutable validation and two mutable traversals. Every selected
        // physical leaf has at least one binding; direct neutral Parameter
        // leaves need less than this physical adapter envelope.
        sources::binding_visit_control_bytes()?
            .checked_mul(rows)?
            .checked_mul(3)
    }
}

pub(super) fn bind_linear_companion(
    weight: &ParameterSpec,
    mut companion: ParameterSpec,
) -> ParameterSpec {
    companion.linear_companion_of = Some(weight.id.clone());
    companion
}

pub(super) fn compute<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, ComputeError> {
    result.map_err(ComputeError::backend_source)
}

pub(super) fn compute_tensor(
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

    fn begin_blockwise_attention_with_options(
        spec: BlockwiseAttentionSpec<'_, MlxTensor>, options:eredu_nn::BlockwiseAttentionOptions,
        context:&Stream,
    ) -> Result<Self::BlockwiseAccumulator,ComputeError> {
        let mut accumulator=Self::begin_blockwise_attention(spec,context)?;
        compute(accumulator.set_options(options))?;
        Ok(accumulator)
    }
    fn begin_blockwise_value_pass(accumulator:&mut Self::BlockwiseAccumulator,_context:&Stream)->Result<(),ComputeError>{
        compute(accumulator.begin_value_pass())
    }

    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: MlxTensor,
        values: MlxTensor,
        context: &Stream,
    ) -> Result<u64, ComputeError> {
        Self::accumulate_blockwise_attention_with_bias(accumulator,start,end,keys,values,None,context)
    }
    fn accumulate_blockwise_attention_with_bias(
        accumulator:&mut Self::BlockwiseAccumulator,start:i64,end:i64,keys:MlxTensor,values:MlxTensor,
        bias:Option<&MlxTensor>,context:&Stream,
    )->Result<u64,ComputeError>{
        let scratch=keys.as_array().nbytes() as u64+values.as_array().nbytes() as u64;
        let block=KeyValueAttentionBlock::unleased(start,end,keys.into_array(),values.into_array());
        compute(accumulator.accumulate_with_bias(&block,bias.map(MlxTensor::as_array),context))?;
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

pub(super) fn parameter_topology(
    module: &impl PhysicalParameters,
    mut weight: ParameterSpec,
    bias: Option<ParameterSpec>,
    format: &LinearFormatSpec,
) -> Result<BTreeMap<String, ParameterSpec>, ComputeError> {
    weight.linear_row_layout = format.row_layout();
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

pub(super) fn exact_parameter_topology(
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

pub(super) fn visit_module_parameters<'a, M, V>(
    module: &'a M,
    topology: &'a dyn NativeParameterTopology,
    visitor: &mut V,
) where
    M: NativeRetainedValues,
    V: ParameterVisitor<'a, MlxTensor>,
{
    if module.native_parameter_source_count().is_none() {
        if visitor.requires_borrowed_metadata() {
            visitor.borrowed_metadata_unavailable();
            return;
        }
        visit_module_parameters_legacy(module, topology, visitor);
        return;
    }
    struct Owned<'v, V>(&'v mut V);
    impl<'a, V: ParameterVisitor<'a, MlxTensor>> ParameterSourceVisitor<'a, MlxTensor>
        for Owned<'_, V>
    {
        fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a MlxTensor) {
            self.0.visit_borrowed(metadata, value);
        }
        fn retained(&mut self, _: &'a MlxTensor) {}
    }
    visit_module_parameter_sources(module, topology, &mut Owned(visitor))
        .expect("validated backend parameter topology covers actual physical fields");
}

fn visit_module_parameters_legacy<'a, M, V>(
    module: &'a M,
    topology: &dyn NativeParameterTopology,
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

pub(super) fn visit_module_parameters_mut<'a, M, V>(
    module: &'a mut M,
    topology: &dyn NativeParameterTopology,
    visitor: &mut V,
) where
    M: NativeRetainedValues,
    V: ParameterVisitorMut<'a, MlxTensor>,
{
    if visitor.requires_borrowed_metadata() {
        sources::visit_module_parameters_mut_borrowed(module, topology, visitor);
        return;
    }
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

pub(super) fn set_module_trainable(module: &mut impl PhysicalParameters, trainable: bool) {
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

#[cfg(test)]
struct ParameterMutCollector<'a> {
    parameters: ModuleParamMut<'a>,
}

#[cfg(test)]
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
#[cfg(test)]
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        self.inner.visit_parameter_sources(visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.inner.retained_value_slot_bound()
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        self.inner.visit_retained_values(visitor)
    }

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
pub(super) struct MlxNamedModule<M> {
    pub(super) inner: M,
    pub(super) topology: NamedParameterTopology,
}

impl<M: PhysicalParameters> MlxNamedModule<M> {
    pub(super) fn with_exact_topology(
        inner: M,
        specs: impl IntoIterator<Item = (&'static str, ParameterSpec)>,
    ) -> Result<Self, ComputeError> {
        let topology = exact_parameter_topology(&inner, specs)?;
        Ok(Self { inner, topology: NamedParameterTopology::Ordinary(topology) })
    }

    pub(super) fn with_prepared_topology(inner:M, topology:PreparedParameterTopology)
        ->Result<Self,ComputeError> where M:NativeRetainedValues {
        topology.validate(&inner)?;
        Ok(Self { inner, topology:NamedParameterTopology::Prepared(topology) })
    }

    pub(super) fn local_parameter_names(&self) -> Vec<String> {
        self.topology.keys().map(str::to_owned).collect()
    }

    pub(super) fn bind_local_parameters_from<V: CompactBindingValues + ?Sized>(
        &mut self,
        bindings: &mut V,
        floating_shapes: &[(&str, [i32; 3])],
    ) -> Result<(), ComputeError>
    where M: NativeRetainedValues {
        compact::bind_named(&mut self.inner, &self.topology, bindings, floating_shapes)
    }

}

/// Packed placeholders retain source encoding geometry. A promoted floating
/// weight instead has the exact logical geometry supplied by its operator spec.
pub(super) fn validate_compact_binding(
    _name: &str,
    placeholder: &Array,
    value: &Array,
    floating_shape: Option<&[i32]>,
) -> Result<(), ComputeError> {
    compact::validate(placeholder, value, floating_shape).map_err(compact::binding_error)
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

impl<M: NativeRetainedValues> Parameterized<MlxTensor> for MlxNamedModule<M> {
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        visit_module_parameter_sources(&self.inner, &self.topology, visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.inner.native_retained_value_slot_bound()
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        self.inner.visit_native_retained_values(visitor)
    }

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

#[cfg(test)]
mod retained_tests;

#[cfg(test)]
mod direct_retained_tests;
