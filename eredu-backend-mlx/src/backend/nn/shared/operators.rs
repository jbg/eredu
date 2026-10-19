use super::parameters::*;
use super::*;
use safemlx::error::Exception;
mod grouped_units;

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear {
    pub(super) arithmetic: eredu_nn::LinearArithmetic,
    pub(super) module: common::linear::PhysicalLinear,
    pub(super) topology: BTreeMap<String, ParameterSpec>,
    pub(super) vocabulary_range: Option<VocabularyParallelRange>,
}

impl LinearOperator<MlxTensor> for MlxLinear {
    fn parameter_element_type(&self) -> Option<eredu_nn::TensorElementType> {
        floating_element(self.module.weight.dtype()).or_else(|| {
            // Block-scale bytes are not an activation dtype. Ordinary affine
            // parameters carry floating scales; native block formats use F32
            // for a standalone host-input projection.
            if self.module.gguf.is_none()
                && self.module.mode == safemlx::ops::QuantizationMode::Affine
            {
                self.module
                    .scales
                    .value
                    .as_ref()
                    .and_then(|value| floating_element(value.dtype()))
            } else {
                Some(eredu_nn::TensorElementType::F32)
            }
        })
    }
    fn set_compute_element(
        &mut self,
        element: Option<eredu_nn::TensorElementType>,
    ) -> Result<(), ComputeError> {
        self.arithmetic =
            eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?
                .with_compute_element(element)?
                .arithmetic();
        Ok(())
    }
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        self.forward_observed_input(input, None, context, None)
    }
    fn forward_with_input_observer(
        &mut self,
        input: &MlxTensor,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        self.forward_observed_input(input, None, context, observer)
    }
}

struct NativeInputObserver<'a> {
    inner: &'a mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>,
    failure: Option<ComputeError>,
}
impl NativeInputObserver<'_> {
    fn result(&mut self, result: Result<(), ComputeError>) -> Result<(), Exception> {
        result.map_err(|error| {
            let native = Exception::custom(error.to_string());
            self.failure = Some(error);
            native
        })
    }
}
impl common::linear::NativeProjectionInputObserver for NativeInputObserver<'_> {
    fn observe(&mut self, input: &Array) -> Result<(), Exception> {
        let result = self.inner.observe(MlxTensor::ref_cast(input));
        self.result(result)
    }
    fn observe_generated(
        &mut self,
        prototype: &Array,
        source: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception> {
        let result =
            self.inner
                .observe_generated(MlxTensor::ref_cast(prototype), source, &mut || {
                    generate()
                        .map(MlxTensor::from_array)
                        .map_err(ComputeError::backend_source)
                });
        self.result(result)
    }
}
impl MlxLinear {
    fn forward_row_affine(
        &mut self,
        input: &MlxTensor,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let original = input
            .element_type()
            .ok_or_else(|| ComputeError::backend("unknown affine precision"))?;
        let wide = input.cast_float(eredu_nn::TensorElementType::F32, context)?;
        if let Some(observer) = observer {
            observer.observe(&wide)?;
        }
        if let Some(output) = compute(super::super::matrix::bf16_affine_projection(
            input.as_array(),
            self.module.weight.as_ref(),
            self.module.bias.as_ref().as_ref(),
            context,
        ))? {
            return Ok(MlxTensor::from_array(output));
        }
        compute_tensor(self.module.forward(wide.as_array(), context))?.cast_float(original, context)
    }

    fn forward_channel_major(
        &mut self,
        input: &MlxTensor,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let shape = input.shape();
        if shape.len() != 3 || shape.iter().any(|&n| n <= 0) {
            return Err(ComputeError::backend(
                "channel-major affine requires [batch, time, channel]",
            ));
        }
        if shape[1] == 1 || shape[2] == 1 {
            return self.forward_row_affine(input, context, observer);
        }
        if let Some(observer) = observer {
            observer.observe(input)?;
        }
        let value = compute(super::super::matrix::bf16_column_projection(
            input.as_array(),
            self.module.weight.as_ref(),
            context,
        ))?;
        match value {
            Some(value) => compute_tensor(match self.module.bias.as_ref() {
                Some(bias) => value.add(bias, context),
                None => Ok(value),
            }),
            None => compute_tensor(self.module.forward(input.as_array(), context)),
        }
    }

    pub(super) fn forward_observed_input(
        &mut self,
        input: &MlxTensor,
        parallel: Option<&Group>,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        if self.arithmetic == eredu_nn::LinearArithmetic::ChannelMajorAffine {
            if parallel.is_some() {
                return Err(ComputeError::backend(
                    "channel-major affine arithmetic requires a replicated projection",
                ));
            }
            return self.forward_channel_major(input, context, observer);
        }
        if self.arithmetic == eredu_nn::LinearArithmetic::RowMajorAffine && parallel.is_none() {
            return self.forward_row_affine(input, context, observer);
        }
        let compute_element = match self.arithmetic {
            eredu_nn::LinearArithmetic::RowMajorAffine => Some(eredu_nn::TensorElementType::F32),
            arithmetic => arithmetic.compute_element(),
        };
        let original = input
            .element_type()
            .ok_or_else(|| ComputeError::backend("unknown projection input precision"))?;
        let cast = compute_element
            .filter(|_| parallel.is_none())
            .map(|e| input.cast_float(e, context))
            .transpose()?;
        let input = cast.as_ref().unwrap_or(input);
        let mut adapter = observer.map(|inner| NativeInputObserver {
            inner,
            failure: None,
        });
        let observer = adapter
            .as_mut()
            .map(|adapter| adapter as &mut dyn common::linear::NativeProjectionInputObserver);
        let result = match parallel {
            Some(group) => self.module.forward_row_parallel_with_compute(
                input.as_array(),
                group,
                context,
                observer,
                compute_element.map(|e| match e {
                    eredu_nn::TensorElementType::F16 => safemlx::Dtype::Float16,
                    eredu_nn::TensorElementType::Bf16 => safemlx::Dtype::Bfloat16,
                    _ => safemlx::Dtype::Float32,
                }),
            ),
            None => self
                .module
                .forward_with_input_observer(input.as_array(), context, observer),
        };
        match adapter.and_then(|adapter| adapter.failure) {
            Some(error) => Err(error),
            None => {
                let output = compute_tensor(result)?;
                if compute_element.is_some() {
                    output.cast_float(original, context)
                } else {
                    Ok(output)
                }
            }
        }
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
    fn output_element_type(&self) -> Option<eredu_nn::TensorElementType> {
        match &self.module {
            common::linear::PhysicalEmbedding::Dense(value) => {
                floating_element(value.weight.dtype())
            }
            common::linear::PhysicalEmbedding::Quantized(value) => {
                // Ordinary restored dense tables retain their own dtype;
                // packed row lookup explicitly publishes F32 activations.
                floating_element(value.inner.weight.dtype())
                    .or(Some(eredu_nn::TensorElementType::F32))
            }
        }
    }
    fn weighted_sum(
        &mut self,
        coefficients: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        let (coefficients, rows) = match &self.vocabulary_range {
            Some(range) => {
                if coefficients.shape().last().copied() != Some(range.global_vocabulary as i32) {
                    return Err(ComputeError::backend(
                        "partitioned weighted embedding requires full vocabulary coefficients",
                    ));
                }
                let shape = coefficients.shape();
                let flat = coefficients.reshape(&[-1, range.global_vocabulary as i32], context)?;
                let local = compute(flat.as_array().try_index_device(
                    (.., range.local.start as i32..range.local.end as i32),
                    context,
                ))?;
                let mut shape = shape.to_vec();
                *shape.last_mut().unwrap() = range.local.len() as i32;
                (
                    compute(local.reshape(&shape, context))?,
                    range.local.len() as i32,
                )
            }
            None => (coefficients.as_array().clone(), self.vocabulary),
        };
        compute_tensor(self.module.weighted_sum(&coefficients, rows, context))
    }
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
        if let Some(range) = &self.vocabulary_range {
            let start =
                Array::from_int(i32::try_from(range.local.start).map_err(ComputeError::backend)?);
            let end =
                Array::from_int(i32::try_from(range.local.end).map_err(ComputeError::backend)?);
            let owned = compute(input.ge(&start, context))?
                .logical_and(&compute(input.lt(&end, context))?, context)
                .map_err(ComputeError::backend)?;
            let local = compute(input.subtract(&start, context))?;
            let safe = compute(safemlx::ops::r#where(
                &owned,
                &local,
                Array::from_int(0),
                context,
            ))?;
            let value = compute(self.module.forward(&safe, context))?;
            let mask = compute(owned.expand_dims(-1, context))?;
            let zero = compute(safemlx::ops::zeros_like(&value, context))?;
            return compute_tensor(safemlx::ops::r#where(&mask, &value, &zero, context));
        }
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

fn floating_element(dtype: safemlx::Dtype) -> Option<eredu_nn::TensorElementType> {
    use eredu_nn::TensorElementType;
    match dtype {
        safemlx::Dtype::Float16 => Some(TensorElementType::F16),
        safemlx::Dtype::Bfloat16 => Some(TensorElementType::Bf16),
        safemlx::Dtype::Float32 => Some(TensorElementType::F32),
        safemlx::Dtype::Float64 => Some(TensorElementType::F64),
        _ => None,
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
    pub(super) reduction: eredu_nn::NormalizationReduction,
    pub(super) groups: Option<i32>,
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
        if self.reduction == eredu_nn::NormalizationReduction::Cascaded {
            return compute_tensor(super::super::normalization::cascaded_rms(
                input,
                self.module.as_ref().map(|module| module.weight.as_ref()),
                self.offset,
                self.epsilon,
                context,
            ));
        }
        if self.reduction == eredu_nn::NormalizationReduction::ChannelMajor {
            if let Some(value) = compute(super::super::normalization::channel_major_rms(
                input,
                self.module.as_ref().map(|module| module.weight.as_ref()),
                self.offset,
                self.epsilon,
                context,
            ))? {
                return Ok(MlxTensor::from_array(value));
            }
        }
        if self.groups.is_some() || self.offset.is_some() {
            let groups = self.groups.unwrap_or(1);
            if let Some(output) = compute(super::super::normalization::f32_grouped_rms(
                input,
                self.module.as_ref().map(|module| module.weight.as_ref()),
                groups,
                self.epsilon,
                self.offset,
                context,
            ))? {
                return Ok(MlxTensor::from_array(output));
            }
            let mut shape = input.shape().to_vec();
            shape.pop();
            shape.extend([groups, self.dimensions / groups]);
            let wide = compute(input.as_dtype(Dtype::Float32, context))?;
            let grouped = compute(wide.reshape(&shape, context))?;
            let normalized = mlx_weightless_rms_norm(&grouped, self.epsilon, context)?;
            let mut normalized = compute(normalized.reshape(input.shape(), context))?;
            if let Some(module) = &self.module {
                let mut scale = compute(module.weight.as_ref().as_dtype(Dtype::Float32, context))?;
                if let Some(offset) = self.offset {
                    scale = compute(scale.add(Array::from_f32(offset), context))?;
                }
                normalized = compute(normalized.multiply(scale, context))?;
            }
            return compute_tensor(normalized.as_dtype(input.dtype(), context));
        }
        match &mut self.module {
            Some(module) => compute_tensor(module.forward(input, context)),
            None => {
                mlx_weightless_rms_norm(input, self.epsilon, context).map(MlxTensor::from_array)
            }
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
    compute(super::super::normalization::weightless_rms(
        input, epsilon, context,
    ))
}

/// MLX RoPE variant selected from model metadata.
#[derive(Debug, Clone)]
pub struct MlxRotary {
    pub(super) arithmetic: eredu_nn::RotaryArithmetic,
    pub(super) dimensions: i32,
    pub(super) traditional: bool,
    pub(super) native: RopeVariant,
    pub(super) explicit: Option<rope::ElementwiseRotary>,
}

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
                match &self.explicit {
                    Some(rotary) => {
                        compute_tensor(rotary.forward(input.as_array(), offset, context))
                    }
                    None => compute_tensor(self.native.forward(rope_input, context)),
                }
            }
            RotaryPosition::Embeddings { cosine, sine } => {
                compute_tensor(common::attention::apply_rotary_embeddings_with_arithmetic(
                    input.as_array(),
                    cosine.as_array(),
                    sine.as_array(),
                    self.dimensions,
                    self.traditional,
                    self.arithmetic,
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

    fn forward_with_coefficients_observer(
        &mut self,
        residual: &MlxTensor,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::TensorValueObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let Some(observer) = observer else {
            return self.forward(residual, context);
        };
        let mut failure = None;
        let result = self.module.forward_with_coefficients_observer(
            residual.as_array(),
            context,
            Some(&mut |coefficients| {
                observer
                    .observe(MlxTensor::ref_cast(coefficients))
                    .map_err(|error| {
                        let native = Exception::custom(error.to_string());
                        failure = Some(error);
                        native
                    })
            }),
        );
        match failure {
            Some(error) => Err(error),
            None => compute_tensor(result),
        }
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
        self.module.bind_local_parameters(
            bindings,
            &[
                (
                    "gate_up_proj",
                    [
                        self.spec.group_count(),
                        self.spec
                            .intermediate_dimensions()
                            .checked_mul(2)
                            .ok_or_else(|| ComputeError::backend("grouped read width overflow"))?,
                        self.spec.input_dimensions(),
                    ],
                ),
                (
                    "down_proj",
                    [
                        self.spec.group_count(),
                        self.spec.output_dimensions(),
                        self.spec.intermediate_dimensions(),
                    ],
                ),
            ],
        )
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

    fn forward_uncombined(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        self.forward_uncombined_units(input, selections, context, observer)
    }

    fn combine_outputs(
        &self,
        outputs: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.combine_outputs(
            outputs.as_array(),
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            context,
        ))
    }

    fn forward_grouped(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
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

    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        self.forward_units(input, selections, context, observer)
    }
}

impl TensorParallelGroupedGatedProductOperator<MlxTensor> for MlxGroupedGatedProduct {
    fn forward_uncombined_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        self.forward_uncombined_units_tensor_parallel(
            input, selections, partitions, context, observer,
        )
    }

    fn forward_grouped_tensor_parallel(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
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

    fn forward_grouped_tensor_parallel_with_unit_observer(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        self.forward_units_tensor_parallel(input, selections, partitions, context, observer)
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
        self.module.bind_local_parameters(
            bindings,
            &[
                (
                    "up_proj",
                    [
                        self.spec.group_count(),
                        self.spec.intermediate_dimensions(),
                        self.spec.hidden_dimensions(),
                    ],
                ),
                (
                    "down_proj",
                    [
                        self.spec.group_count(),
                        self.spec.hidden_dimensions(),
                        self.spec.intermediate_dimensions(),
                    ],
                ),
            ],
        )
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
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Relu2,
            context,
        )?;
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

    fn forward_grouped_with_unit_observer(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        self.forward_units(input, selections, context, observer)
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
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Relu2,
            context,
        )?;
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

    fn forward_grouped_tensor_parallel_with_unit_observer(
        &mut self,
        input: &MlxTensor,
        selections: &GroupSelection<MlxTensor>,
        partitions: usize,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::GroupedUnitObserver<MlxTensor>>,
    ) -> Result<TensorParallelGroupedOutput<MlxTensor>, ComputeError> {
        self.forward_units_tensor_parallel(input, selections, partitions, context, observer)
    }
}

#[cfg(test)]
#[path = "operators/observation_errors.rs"]
mod observation_errors;
