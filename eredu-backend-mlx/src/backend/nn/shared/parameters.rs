use super::*;

pub(super) fn bind_linear_companion(
    weight: &ParameterSpec,
    mut companion: ParameterSpec,
) -> ParameterSpec {
    companion.linear_companion_of = Some(weight.id.clone());
    companion
}

pub(super) fn compute<T>(result: Result<T, safemlx::error::Exception>) -> Result<T, ComputeError> {
    result.map_err(ComputeError::backend)
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

pub(super) fn parameter_topology(
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

pub(super) fn visit_module_parameters_mut<'a, M, V>(
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
    pub(super) topology: BTreeMap<String, ParameterSpec>,
}

impl<M: PhysicalParameters> MlxNamedModule<M> {
    pub(super) fn with_exact_topology(
        inner: M,
        specs: impl IntoIterator<Item = (&'static str, ParameterSpec)>,
    ) -> Result<Self, ComputeError> {
        let topology = exact_parameter_topology(&inner, specs)?;
        Ok(Self { inner, topology })
    }

    pub(super) fn local_parameter_names(&self) -> Vec<String> {
        self.topology.keys().cloned().collect()
    }

    pub(super) fn bind_local_parameters(
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
        for (local, parameter) in self.inner.parameters().flatten() {
            let value = bindings
                .get(local.as_ref())
                .expect("equal compact binding sets contain every native parameter");
            if parameter.shape() != value.shape() {
                return Err(ComputeError::backend(format!(
                    "compact grouped binding {local:?} has shape {:?}, expected {:?}",
                    value.shape(),
                    parameter.shape()
                )));
            }
        }
        for (local, parameter) in self.inner.parameters_mut().flatten() {
            let value = bindings
                .remove(local.as_ref())
                .expect("equal compact binding sets contain every native parameter");
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
