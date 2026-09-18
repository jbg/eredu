use super::parameters::*;
use super::*;
use eredu_nn::{ParameterSourceError, ParameterSourceVisitor};
use safemlx::error::Exception;
mod grouped_units;
mod observation_transport;

pub(crate) fn projection_observation_control_bytes() -> Option<usize> {
    observation_transport::control_bytes()
}

/// MLX dense-or-quantized affine projection.
#[derive(Debug, Clone)]
pub struct MlxLinear {
    pub(super) module: common::linear::PhysicalLinear,
    pub(super) topology: NativeParameterTable,
    pub(super) vocabulary_range: Option<VocabularyParallelRange>,
}

impl LinearOperator<MlxTensor> for MlxLinear {
    fn forward(&mut self, input: &MlxTensor, context: &Stream) -> Result<MlxTensor, ComputeError> {
        compute_tensor(self.module.forward(input.as_array(), context))
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

// Retained factories cross this boundary without duplicating an arbitrary
// native/observer diagnostic. The original failure remains in the source chain.
#[derive(Debug)]
struct RetainedInputFailure<E>(E);
impl<E> std::fmt::Display for RetainedInputFailure<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("retained generated input observation failed")
    }
}
impl<E: std::error::Error + 'static> std::error::Error for RetainedInputFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
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
                        .map_err(ComputeError::backend_retained_source)
                });
        self.result(result)
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        prototype: &Array,
        source: &eredu_nn::GeneratedTensorSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<Array, Exception>,
    ) -> Result<(), Exception> {
        let mut mapped = observation_transport::Factory::new(
            factory,
            MlxTensor::ref_cast,
            MlxTensor::from_array,
            observation_transport::native,
            observation_transport::signal,
        );
        let result = self.inner.observe_generated_retained(
            MlxTensor::ref_cast(prototype),
            source,
            &mut mapped,
        );
        result.map_err(|error| {
            let signal = observation_transport::signal(&error);
            self.failure = Some(error);
            signal
        })
    }
}
impl MlxLinear {
    pub(super) fn forward_observed_input(
        &mut self,
        input: &MlxTensor,
        parallel: Option<&Group>,
        context: &Stream,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<MlxTensor>>,
    ) -> Result<MlxTensor, ComputeError> {
        let retained = observer.is_some() && self.module.generates_projection_input();
        let mut adapter = observer.map(|inner| NativeInputObserver {
            inner,
            failure: None,
        });
        let observer = adapter
            .as_mut()
            .map(|adapter| adapter as &mut dyn common::linear::NativeProjectionInputObserver);
        // Defer native error wrapping until after the observation adapter's
        // own error wins, exactly as in the ordinary path.
        enum ForwardFailure { Native(Exception), Reduction(ComputeError) }
        let result = match parallel {
            Some(group) => self.module.forward_row_parallel_with_reducer(
                input.as_array(),context,observer,
                |partial,stream| {
                    if group.has_original_parallel() { group.sum_model(partial,stream).map_err(ForwardFailure::Reduction) }
                    else { crate::backend::runtime::distributed::all_sum(partial,group,stream).map_err(ForwardFailure::Native) }
                },ForwardFailure::Native),
            None => self.module.forward_with_input_observer(input.as_array(),context,observer).map_err(ForwardFailure::Native),
        };
        match adapter.and_then(|adapter| adapter.failure) {
            Some(error) if retained => Err(observation_transport::callback(error)),
            Some(error) => Err(error),
            None => result.map(MlxTensor::from_array).map_err(|cause|match cause {
                ForwardFailure::Native(cause) if retained=>observation_transport::source(cause),
                ForwardFailure::Native(cause)=>ComputeError::backend_retained_source(cause),
                ForwardFailure::Reduction(cause)=>cause,
            }),
        }
    }
}

impl Parameterized<MlxTensor> for MlxLinear {
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        visit_module_parameter_sources(&self.module, &self.topology, visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.native_retained_value_slot_bound()
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
    pub(super) topology: NativeParameterTable,
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
        let nonnegative = compute(input.ge(
            Array::try_from_int(0).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
        let below_vocabulary = compute(input.lt(
            Array::try_from_int(self.vocabulary).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
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
        let sentinel_mask = compute(input.eq(
            Array::try_from_int(sentinel).map_err(ComputeError::backend_retained_source)?,
            context,
        ))?;
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        visit_module_parameter_sources(&self.module, &self.topology, visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.native_retained_value_slot_bound()
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
    pub(super) groups: Option<i32>,
    pub(super) module: Option<nn::RmsNorm>,
    pub(super) topology: NativeParameterTable,
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
            MlxTensor::ref_cast(input),
            context,
        )
        .map(MlxTensor::into_array)
        .map_err(safemlx::error::Exception::from_source)
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
        if let Some(groups) = self.groups {
            let mut shape = Vec::with_capacity(input.ndim() + 1);
            shape.extend_from_slice(&input.shape()[..input.ndim() - 1]);
            shape.extend([groups, self.dimensions / groups]);
            let wide = compute(input.as_dtype(Dtype::Float32, context))?;
            let grouped = compute(wide.reshape(&shape, context))?;
            let normalized = mlx_weightless_rms_norm(&grouped, self.epsilon, context)?;
            let mut normalized = compute(normalized.reshape(input.shape(), context))?;
            if let Some(module) = &self.module {
                let mut scale = compute(module.weight.as_ref().as_dtype(Dtype::Float32, context))?;
                if let Some(offset) = self.offset {
                    scale = compute(scale.add(
                        Array::try_from_f32(offset).map_err(ComputeError::backend_retained_source)?,
                        context,
                    ))?;
                }
                normalized = compute(normalized.multiply(scale, context))?;
            }
            return compute_tensor(normalized.as_dtype(input.dtype(), context));
        }
        match (&mut self.module, self.offset) {
            (Some(module), None) => compute_tensor(module.forward(input, context)),
            (Some(module), Some(offset)) => {
                let scale = compute(module.weight.as_ref().add(
                    Array::try_from_f32(offset).map_err(ComputeError::backend_retained_source)?,
                    context,
                ))?;
                compute_tensor(super::super::normalization::input_precision_rms(
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        match &self.module {
            Some(module) => visit_module_parameter_sources(module, &self.topology, visitor),
            None if self.topology.is_empty() => Ok(()),
            None => Err(ParameterSourceError::TopologyMismatch { slot: 0 }),
        }
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        Some(1)
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
    if let Some(output) = compute(super::super::normalization::f32_weightless_rms(
        input, epsilon, context,
    ))? {
        return Ok(output);
    }
    let dtype = input.dtype();
    let variance = compute(input.square(context))?;
    let variance = compute(variance.mean_axis(-1, true, context))?;
    let denominator = compute(variance.add(
        Array::try_from_f32(epsilon).map_err(ComputeError::backend_retained_source)?,
        context,
    ))?;
    let denominator = compute(denominator.rsqrt(context))?;
    compute(input.multiply(denominator, context))?
        .as_dtype(dtype, context)
        .map_err(ComputeError::backend_retained_source)
}

/// MLX RoPE variant selected from model metadata.
#[derive(Debug, Clone)]
pub struct MlxRotary {
    pub(super) native: RopeVariant,
    pub(super) explicit: Option<rope::ElementwiseRotary>,
    pub(super) dimensions: i32,
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
                let shape = input.as_array().shape();
                if shape.len() >= 2 && shape[shape.len() - 2] == 0 {
                    if self.dimensions <= 0
                        || self.dimensions % 2 != 0
                        || self.dimensions > shape[shape.len() - 1]
                    {
                        return Err(ComputeError::backend(
                            "empty rotary input has incompatible feature width",
                        ));
                    }
                    // No positions exist yet in a partially filled pooling
                    // stream. Inferred reshapes cannot represent this case;
                    // the elementwise rotation is the identity on empty data.
                    return Ok(input.clone());
                }
                let rope_input = nn::RopeInputBuilder::new(input.as_array())
                    .offset(offset)
                    .build()
                    .map_err(ComputeError::backend_retained_source)?;
                match &self.explicit {
                    Some(rotary) => {
                        compute_tensor(rotary.forward(input.as_array(), offset, context))
                    }
                    None => compute_tensor(self.native.forward(rope_input, context)),
                }
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        let mut arrays = |value: &'a Array| visitor.retained(MlxTensor::ref_cast(value));
        self.native.visit_retained_arrays(&mut arrays);
        if let Some(explicit) = &self.explicit {
            explicit.visit_retained_arrays(&mut arrays);
        }
        Ok(())
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        Some(2)
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
    pub(super) topology: NativeParameterTable,
}

fn hyper_native_error(cause: Exception) -> ComputeError {
    match safemlx::OriginalScopeObserver::try_current() {
        Ok(None) => ComputeError::backend_retained_source(cause),
        _ => ComputeError::backend_retained_source(cause),
    }
}
fn hyper_compute<T>(result: Result<T, Exception>) -> Result<T, ComputeError> {
    result.map_err(hyper_native_error)
}

pub(crate) fn hyper_observation_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        observation_transport::grouped_callback_control_bytes()?,
        size_of::<Option<ComputeError>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<MlxTensor, ComputeError>>(),
        size_of::<Result<(), ComputeError>>(),
        size_of::<&mut dyn eredu_nn::TensorValueObserver<MlxTensor>>(),
        size_of::<Option<&mut dyn FnMut(&Array) -> Result<(), Exception>>>(),
        size_of::<bool>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

impl HyperConnectionOperator<MlxTensor> for MlxHyperConnection {
    fn collapse(
        &mut self,
        residual: &MlxTensor,
        norm_epsilon: f32,
        context: &Stream,
    ) -> Result<HyperConnectionState<MlxTensor>, ComputeError> {
        let (collapsed, split) = hyper_compute(self.module.collapse_split(
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
        hyper_compute(common::hyper_connections::expand(
            sublayer.as_array(),
            residual.as_array(),
            state.post.as_array(),
            state.combination.as_array(),
            context,
        ))
        .map(MlxTensor::from_array)
    }
}

impl Parameterized<MlxTensor> for MlxHyperConnection {
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        visit_module_parameter_sources(&self.module, &self.topology, visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.native_retained_value_slot_bound()
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
    pub(super) topology: NativeParameterTable,
}

impl HyperHeadOperator<MlxTensor> for MlxHyperHead {
    fn forward(
        &mut self,
        residual: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ComputeError> {
        hyper_compute(self.module.forward(residual.as_array(), context)).map(MlxTensor::from_array)
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
        let original = safemlx::OriginalScopeObserver::try_current()
            .map_err(hyper_native_error)?
            .is_some();
        let mut failure = None;
        let result = self.module.forward_with_coefficients_observer(
            residual.as_array(),
            context,
            Some(&mut |coefficients| {
                observer
                    .observe(MlxTensor::ref_cast(coefficients))
                    .map_err(|error| {
                        let (error, signal) = if original {
                            let error = observation_transport::callback(error);
                            let signal = observation_transport::signal(&error);
                            (error, signal)
                        } else {
                            let signal = Exception::custom(error.to_string());
                            (error, signal)
                        };
                        failure = Some(error);
                        signal
                    })
            }),
        );
        match failure {
            Some(error) => Err(error),
            None => hyper_compute(result).map(MlxTensor::from_array),
        }
    }
}

impl Parameterized<MlxTensor> for MlxHyperHead {
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        visit_module_parameter_sources(&self.module, &self.topology, visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.native_retained_value_slot_bound()
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameter_sources(visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.retained_value_slot_bound()
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
    pub(crate) fn original_fp8_control_bytes() -> Option<usize> {
        super::selected_linear::original::control_bytes()?
            .checked_add(std::mem::size_of::<[Array; 3]>())?
            .checked_add(std::mem::size_of::<Result<MlxTensor, ComputeError>>())
    }

    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        self.module.local_parameter_names()
    }

    pub(crate) fn bind_local_parameters(
        &mut self,
        mut bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        self.bind_compact_values(&mut bindings)
    }

    pub(crate) fn bind_prepared_local_parameters(
        &mut self, bindings: &mut super::parameters::PreparedCompactBindings<'_>,
    ) -> Result<(), ComputeError> {
        self.bind_compact_values(bindings)
    }

    fn bind_compact_values<V: super::parameters::CompactBindingValues + ?Sized>(
        &mut self, bindings: &mut V,
    ) -> Result<(), ComputeError> {
        self.module.bind_local_parameters_from(
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameter_sources(visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.retained_value_slot_bound()
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
        #[cfg(test)]
        crate::tests::support::provider_failure::check(
            crate::tests::support::provider_failure::Operator::Gated,
            context,
        )?;
        let transport = super::selected_linear::original::Transport::new(matches!(
            self.spec.layout(), GatedProductGroupLayout::Packed { gate_up, .. }
                if matches!(gate_up.format().encoding(), LinearFormat::E4M3BlockFp8(_))
        ));
        let input = input.as_array();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = transport.compute(self.module.forward(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            context,
        ))?;
        transport.tensor(output.reshape(input.shape(), context))
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
        let transport = super::selected_linear::original::Transport::new(true);
        let input = input.as_array();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = transport.compute(self.module.forward_tensor_parallel(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            partitions,
            context,
        ))?;
        let (reducible, post_reduce) = output.into_parts();
        Ok(TensorParallelGroupedOutput::new(
            transport.tensor(reducible.reshape(input.shape(), context))?,
            post_reduce
                .map(|bias| transport.tensor(bias.reshape(input.shape(), context)))
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
    pub(crate) fn original_control_bytes() -> Option<usize> {
        super::selected_linear::original::control_bytes()?
            .checked_add(std::mem::size_of::<[Array; 4]>())?
            .checked_add(std::mem::size_of::<(i32, i32, usize)>())?
            .checked_add(std::mem::size_of::<
                Option<&mut dyn common::grouped::NativeGroupedUnitObserver>,
            >())?
            .checked_add(std::mem::size_of::<Result<MlxTensor, ComputeError>>())
    }

    /// Returns the architecture-owned specification used to realize this bank.
    pub const fn spec(&self) -> &GroupedRelu2Spec {
        &self.spec
    }

    pub(crate) fn local_parameter_names(&self) -> Vec<String> {
        self.module.local_parameter_names()
    }

    pub(crate) fn bind_local_parameters(
        &mut self,
        mut bindings: BTreeMap<String, Array>,
    ) -> Result<(), ComputeError> {
        self.bind_compact_values(&mut bindings)
    }

    pub(crate) fn bind_prepared_local_parameters(
        &mut self, bindings: &mut super::parameters::PreparedCompactBindings<'_>,
    ) -> Result<(), ComputeError> {
        self.bind_compact_values(bindings)
    }

    fn bind_compact_values<V: super::parameters::CompactBindingValues + ?Sized>(
        &mut self, bindings: &mut V,
    ) -> Result<(), ComputeError> {
        self.module.bind_local_parameters_from(
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
    fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), ParameterSourceError>
    where
        V: ParameterSourceVisitor<'a, MlxTensor>,
    {
        self.module.visit_parameter_sources(visitor)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        self.module.retained_value_slot_bound()
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
        let transport = super::selected_linear::original::Transport::new(true);
        let input = input.as_array();
        let shape = input.shape();
        let flattened = transport.compute(input.reshape(&[-1, input.dim(-1)], context))?;
        let output = transport.compute(self.module.forward(
            &flattened,
            selections.group_indices().as_array(),
            selections.coefficients().as_array(),
            context,
        ))?;
        transport.tensor(output.reshape(shape, context))
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

#[cfg(test)]
mod retained_tests;

#[cfg(test)]
mod generated_factory_tests;
