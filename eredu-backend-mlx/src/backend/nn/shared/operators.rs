use super::parameters::*;
use super::*;

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear {
    pub(super) module: common::linear::PhysicalLinear,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
    pub(super) vocabulary_range: Option<VocabularyParallelRange>,
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
    pub(super) module: common::linear::PhysicalEmbedding,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
    pub(super) vocabulary: i32,
    pub(super) vocabulary_range: Option<VocabularyParallelRange>,
}

impl EmbeddingOperator<MlxTensor> for MlxEmbedding {
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        self.lookup(input, EmbeddingLookupPolicy::Strict, context)
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
        let Some(sentinel) = sentinel else {
            return Ok(MlxTensor::from_array(embedded));
        };
        let sentinel_mask = compute(input.eq(Array::from_int(sentinel), context))?;
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
    pub(super) module: Option<nn::RmsNorm>,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
    pub(super) offset: Option<f32>,
    pub(super) dimensions: i32,
    pub(super) epsilon: f32,
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

pub(super) fn mlx_weightless_rms_norm(
    input: &Array,
    epsilon: f32,
    context: &Stream,
) -> Result<Array, ComputeError> {
    eredu_nn::operation_geometry::NormalizationGeometry::new(input.shape(), epsilon)?;
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
pub struct MlxRotary(pub(super) RopeVariant);

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
    pub(super) module: common::hyper_connections::HyperConnection,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
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
    pub(super) module: common::hyper_connections::HyperHead,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
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
    pub(super) module: MlxNamedModule<common::grouped::TopKGroupSelector>,
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
    fn select_intervened(
        &mut self,
        input: &MlxTensor,
        control: &eredu_nn::routing_intervention::GroupSelectionControl,
        context: &Stream,
    ) -> Result<eredu_nn::routing_intervention::IntervenedGroupSelection<MlxTensor>, ComputeError>
    {
        let (original, effective) = compute(self.module.select_intervened(
            input.as_array(),
            control,
            context,
        ))?;
        let convert = |output: super::super::grouped::GroupSelectionOutput| {
            GroupSelection::new(
                MlxTensor::from_array(output.indices),
                MlxTensor::from_array(output.scores),
                MlxTensor::from_array(output.weights),
            )
        };
        Ok(eredu_nn::routing_intervention::IntervenedGroupSelection {
            original: original.map(convert),
            effective: convert(effective),
        })
    }

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
    pub(super) spec: GroupedGatedProductSpec,
    pub(super) module: MlxNamedModule<common::grouped::PackedGatedProductGroups>,
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
    pub(super) spec: GroupedRelu2Spec,
    pub(super) module: MlxNamedModule<common::grouped::PackedRelu2Groups>,
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
