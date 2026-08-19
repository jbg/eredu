//! Backend-neutral neural computation contracts.
//!
//! Architecture implementations use opaque backend tensors through this
//! interface. Tensor storage, physical layout, laziness, device placement, and
//! synchronization remain owned by the selected backend.

#![warn(missing_docs)]

extern crate self as eredu_nn;

use std::fmt::Debug;

use eredu_checkpoint::WeightQuantization;

pub use eredu_nn_macros::Parameterized;

#[cfg(feature = "mlx")]
mod mlx;

/// Backend operation failure.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    message: String,
}

impl Error {
    /// Creates a backend operation failure without exposing backend-native
    /// exception types through architecture code.
    pub fn backend(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// One axis of a backend-neutral tensor view.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Index {
    /// Retains the complete axis.
    Full,
    /// Selects one element and removes the axis.
    At(i32),
    /// Retains the half-open interval `[start, end)`.
    Range(i32, i32),
}

/// Padding behavior for convolutional architecture components.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PadMode {
    /// Pads with zeroes.
    Constant,
    /// Repeats the nearest edge value.
    Edge,
}

/// Attention masking selected by architecture code.
#[derive(Debug, Clone, Copy)]
pub enum AttentionMask<'a, T> {
    /// No attention mask.
    None,
    /// Standard causal mask.
    Causal,
    /// Backend tensor added to attention logits.
    Tensor(&'a T),
}

/// Stable authoritative identity of one logical model parameter.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParameterId(String);

impl ParameterId {
    /// Creates a non-empty parameter identity.
    pub fn new(id: impl Into<String>) -> Result<Self, ParameterTopologyError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ParameterTopologyError::EmptyId);
        }
        Ok(Self(id))
    }

    /// Returns the stable checkpoint-facing identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ParameterId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Complete logical declaration for one parameter slot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterSpec {
    /// Stable authoritative identity.
    pub id: ParameterId,
    /// Whether optimizers may update this parameter by default.
    pub trainable: bool,
    /// Authoritative destination when this slot aliases a tied parameter.
    pub alias_of: Option<ParameterId>,
    /// Optional atomic encoding or sharding group.
    pub group: Option<String>,
}

impl ParameterSpec {
    /// Declares an ordinary trainable parameter.
    pub fn trainable(id: impl Into<String>) -> Result<Self, ParameterTopologyError> {
        Ok(Self {
            id: ParameterId::new(id)?,
            trainable: true,
            alias_of: None,
            group: None,
        })
    }
}

/// Parameter metadata observed during traversal.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterMetadata {
    /// Stable authoritative identity.
    pub id: ParameterId,
    /// Current trainable state.
    pub trainable: bool,
    /// Authoritative destination for a tied alias.
    pub alias_of: Option<ParameterId>,
    /// Optional atomic parameter group.
    pub group: Option<String>,
}

impl ParameterMetadata {
    /// Creates traversal metadata from a construction specification.
    pub fn from_spec(spec: &ParameterSpec, trainable: bool) -> Self {
        Self {
            id: spec.id.clone(),
            trainable,
            alias_of: spec.alias_of.clone(),
            group: spec.group.clone(),
        }
    }
}

/// Invalid stable parameter topology.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParameterTopologyError {
    /// A parameter identity is empty.
    #[error("parameter identity must not be empty")]
    EmptyId,
    /// The same stable identity was visited more than once.
    #[error("parameter identity {0} is duplicated")]
    DuplicateId(ParameterId),
    /// A tied alias points to an identity that is not in the topology.
    #[error("parameter alias {alias} points to missing destination {destination}")]
    MissingAliasDestination {
        /// Identity of the aliasing slot.
        alias: ParameterId,
        /// Missing authoritative destination.
        destination: ParameterId,
    },
    /// A tied alias points to another alias instead of an authoritative slot.
    #[error("parameter alias {alias} points to non-authoritative alias {destination}")]
    AliasTargetsAlias {
        /// Identity of the aliasing slot.
        alias: ParameterId,
        /// Non-authoritative destination.
        destination: ParameterId,
    },
}

/// Immutable statically dispatched parameter visitor.
pub trait ParameterVisitor<'a, T: 'a> {
    /// Visits one authoritative parameter slot.
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a T);
}

/// Mutable statically dispatched parameter visitor.
pub trait ParameterVisitorMut<'a, T: 'a> {
    /// Visits one authoritative mutable parameter slot.
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut T);
}

/// Backend-neutral parameter topology for a module or operator.
pub trait Parameterized<T: 'static> {
    /// Visits every parameter exactly once using stable identities.
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>;

    /// Mutably visits every parameter exactly once using stable identities.
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>;

    /// Updates whether all parameters in this module are trainable.
    fn set_trainable(&mut self, trainable: bool);
}

/// Collects and validates the stable parameter topology exposed by a module.
pub fn validate_parameter_topology<T: 'static, M>(
    module: &M,
) -> Result<Vec<ParameterMetadata>, ParameterTopologyError>
where
    M: Parameterized<T>,
{
    struct Collector(Vec<ParameterMetadata>);
    impl<'a, T: 'a> ParameterVisitor<'a, T> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, _value: &'a T) {
            self.0.push(metadata);
        }
    }

    let mut collector = Collector(Vec::new());
    module.visit_parameters(&mut collector);
    let mut topology = std::collections::BTreeMap::new();
    for metadata in &collector.0 {
        if topology.insert(metadata.id.clone(), metadata).is_some() {
            return Err(ParameterTopologyError::DuplicateId(metadata.id.clone()));
        }
    }
    for metadata in &collector.0 {
        let Some(destination) = &metadata.alias_of else {
            continue;
        };
        let Some(target) = topology.get(destination) else {
            return Err(ParameterTopologyError::MissingAliasDestination {
                alias: metadata.id.clone(),
                destination: destination.clone(),
            });
        };
        if target.alias_of.is_some() {
            return Err(ParameterTopologyError::AliasTargetsAlias {
                alias: metadata.id.clone(),
                destination: destination.clone(),
            });
        }
    }
    Ok(collector.0)
}

/// Shape and checkpoint identity for an affine projection.
#[derive(Debug, Clone)]
pub struct LinearSpec {
    /// Input feature count.
    pub input: i32,
    /// Output feature count.
    pub output: i32,
    /// Stable weight slot.
    pub weight: ParameterSpec,
    /// Optional stable bias slot.
    pub bias: Option<ParameterSpec>,
    /// Physical checkpoint encoding selected for this parameter.
    pub quantization: Option<WeightQuantization>,
}

/// Complete construction specification for a token embedding table.
#[derive(Debug, Clone)]
pub struct EmbeddingSpec {
    /// Vocabulary row count.
    pub vocabulary: i32,
    /// Embedding width.
    pub dimensions: i32,
    /// Stable embedding weight slot.
    pub weight: ParameterSpec,
    /// Physical checkpoint encoding selected for this parameter.
    pub quantization: Option<WeightQuantization>,
}

/// Complete construction specification for normalization.
#[derive(Debug, Clone)]
pub struct NormalizationSpec {
    /// Normalized feature count.
    pub dimensions: i32,
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// Stable scale parameter slot.
    pub weight: ParameterSpec,
}

/// One backend-neutral RoPE configuration value.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum RopeValue {
    /// Floating-point metadata.
    Float(f32),
    /// String metadata.
    String(String),
    /// Boolean metadata.
    Bool(bool),
}

/// Canonical metadata tag for piecewise wavelength-based rotary scaling.
pub const FREQUENCY_SCALED_ROPE_TYPE: &str = "llama3";

/// Complete backend-neutral rotary-position construction specification.
#[derive(Debug, Clone, Copy)]
pub struct RotarySpec<'a> {
    /// Rotated head dimensions.
    pub dimensions: i32,
    /// Base frequency.
    pub base: f32,
    /// Whether adjacent pairs are rotated instead of split halves.
    pub traditional: bool,
    /// Maximum configured position count.
    pub max_positions: i32,
    /// Optional normalized scaling metadata.
    pub scaling: Option<&'a std::collections::HashMap<String, RopeValue>>,
}

/// Backend-native affine projection used by shared architectures.
pub trait LinearOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies the projection without host materialization.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native token embedding used by shared architectures.
pub trait EmbeddingOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Looks up token embeddings.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
    /// Projects hidden states through the transposed embedding table.
    fn as_linear(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native normalization operator.
pub trait NormalizationOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies normalization.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native rotary-position operator.
pub trait RotaryOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies rotary positions at the supplied sequence offset.
    fn forward(&mut self, input: &T, offset: i32, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native key/value cache operations required by attention.
pub trait AttentionCache<T: Tensor> {
    /// Current absolute sequence offset.
    fn offset(&self) -> i32;
    /// Maximum retained history for a sliding cache.
    fn max_size(&self) -> Option<i32>;
    /// Appends projected keys and values and returns the tensors used by attention.
    fn update_for_attention(
        &mut self,
        keys: T,
        values: T,
        context: &T::Context,
    ) -> Result<(T, T), Error>;
    /// Runs cache-aware attention, including paged or quantized kernels where applicable.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        queries: T,
        keys: T,
        values: T,
        scale: f32,
        mask: Option<&T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// General neural-operator family selected by a shared architecture.
///
/// Associated concrete types make calls statically dispatched. Implementations
/// retain ownership of tensor storage, fusion, quantization, and collectives.
pub trait NeuralBackend: Sized + 'static {
    /// Backend tensor handle.
    type Tensor: Tensor;
    /// Native affine projection, including packed quantized variants.
    type Linear: LinearOperator<Self::Tensor>;
    /// Native embedding table.
    type Embedding: EmbeddingOperator<Self::Tensor>;
    /// Native normalization operator.
    type Normalization: NormalizationOperator<Self::Tensor>;
    /// Native rotary-position operator.
    type Rotary: RotaryOperator<Self::Tensor>;
    /// Backend collective context used by tensor-parallel execution.
    type ParallelContext: ?Sized;

    /// Builds one affine projection.
    fn linear(
        spec: LinearSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Linear, Error>;
    /// Builds one token embedding table.
    fn embedding(
        spec: EmbeddingSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Embedding, Error>;
    /// Builds one RMS normalization operator.
    fn rms_norm(
        spec: NormalizationSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Normalization, Error>;
    /// Builds the model's rotary-position operator.
    fn rotary(
        spec: RotarySpec<'_>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Rotary, Error>;
    /// Applies SiLU using a backend-native implementation.
    fn silu(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Runs un-cached scaled dot-product attention.
    fn attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Runs causal sliding-window prefill attention without a square mask.
    #[allow(clippy::too_many_arguments)]
    fn sliding_window_attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Builds the backend-native boolean causal mask used for a prefill.
    ///
    /// The returned value remains a lazy/backend-owned tensor. Calling this
    /// method must not synchronize or materialize mask contents on the host.
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Applies a row-parallel projection and its collective reduction.
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
}

/// Opaque tensor handle and the neural operations required by shared Eredu
/// architectures.
///
/// Implementations must preserve backend-native execution semantics. None of
/// these operations imply host materialization or synchronization.
pub trait Tensor: Clone + Debug + Sized + 'static {
    /// Backend execution context, such as an MLX stream.
    type Context: ?Sized;

    /// Logical tensor shape maintained without materializing tensor values.
    fn shape(&self) -> &[i32];

    /// Returns one logical dimension.
    fn dim(&self, axis: usize) -> i32 {
        self.shape()[axis]
    }

    /// Allocates an unloaded floating-point parameter tensor.
    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Creates a floating-point tensor from host initialization data.
    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Elementwise addition.
    fn add(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise subtraction.
    fn subtract(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise multiplication.
    fn multiply(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise multiplication by a floating-point scalar.
    fn multiply_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise division.
    fn divide(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise square.
    fn square(&self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise maximum with a scalar.
    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error>;

    /// Reshapes without changing logical element order.
    fn reshape(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Permutes axes.
    fn transpose_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Swaps two axes.
    fn swap_axes(&self, left: i32, right: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Reverses the axes of a rank-two tensor.
    fn transpose(&self, context: &Self::Context) -> Result<Self, Error>;
    /// Inserts one unit dimension.
    fn expand_dims(&self, axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Removes unit dimensions.
    fn squeeze_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Creates a tensor view using backend-neutral axis indexes.
    fn index(&self, indexes: &[Index], context: &Self::Context) -> Result<Self, Error>;
    /// Takes rows along one axis using a backend index tensor.
    fn take_axis(&self, indexes: &Self, axis: i32, context: &Self::Context) -> Result<Self, Error>;

    /// Concatenates tensors along an axis.
    fn concatenate(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Stacks tensors along a new axis.
    fn stack(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Matrix multiplication.
    fn matmul(lhs: &Self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Reduction sum.
    fn sum_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Reduction argmin.
    fn argmin_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Pads a tensor.
    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        context: &Self::Context,
    ) -> Result<Self, Error>;

    /// One-dimensional convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// One-dimensional transposed convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Affine linear projection. Backends may use fused or quantized paths.
    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Layer normalization.
    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Gaussian error linear unit.
    fn gelu(input: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Exponential linear unit.
    fn elu(input: &Self, alpha: f32, context: &Self::Context) -> Result<Self, Error>;
    /// Rotary positional encoding.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Scaled dot-product attention. Backends retain control over fusion.
    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error>;
}

/// One trainable parameter owned by a generic architecture module.
#[derive(Debug, Clone)]
pub struct Parameter<T> {
    spec: ParameterSpec,
    trainable: bool,
    value: T,
}

impl<T> Parameter<T> {
    /// Wraps a backend tensor as an authoritative parameter slot.
    pub fn new(spec: ParameterSpec, value: T) -> Self {
        let trainable = spec.trainable;
        Self {
            spec,
            trainable,
            value,
        }
    }
    /// Borrows the backend tensor.
    pub const fn as_ref(&self) -> &T {
        &self.value
    }
    /// Replaces the backend tensor.
    pub fn replace(&mut self, value: T) {
        self.value = value;
    }
}

impl<T: Tensor> Parameter<T> {
    /// Creates an unloaded floating-point parameter.
    pub fn unloaded(
        spec: ParameterSpec,
        shape: &[i32],
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self::new(spec, T::unloaded_f32(shape, context)?))
    }
}

impl<T: 'static> Parameterized<T> for Parameter<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        visitor.visit(
            ParameterMetadata::from_spec(&self.spec, self.trainable),
            &self.value,
        );
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        visitor.visit_mut(
            ParameterMetadata::from_spec(&self.spec, self.trainable),
            &mut self.value,
        );
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
    }
}

impl<T: 'static, M: Parameterized<T>> Parameterized<T> for Vec<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        for module in self {
            module.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        for module in self {
            module.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        for module in self {
            module.set_trainable(trainable);
        }
    }
}

impl<T: 'static, M: Parameterized<T>> Parameterized<T> for Option<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        if let Some(module) = self {
            module.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        if let Some(module) = self {
            module.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(module) = self {
            module.set_trainable(trainable);
        }
    }
}

/// Generic affine projection.
#[derive(Debug, Clone)]
pub struct Linear<T> {
    /// Projection weights shaped `[output, input]`.
    pub weight: Parameter<T>,
    /// Optional output bias.
    pub bias: Option<Parameter<T>>,
}

impl<T: Tensor> Linear<T> {
    /// Creates unloaded projection parameters.
    pub fn unloaded(spec: LinearSpec, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(spec.weight, &[spec.output, spec.input], context)?,
            bias: spec
                .bias
                .map(|bias| Parameter::unloaded(bias, &[spec.output], context))
                .transpose()?,
        })
    }

    /// Applies the projection without materializing backend tensors.
    pub fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        T::linear(
            input,
            self.weight.as_ref(),
            self.bias.as_ref().map(Parameter::as_ref),
            context,
        )
    }
}

impl<T: 'static> Parameterized<T> for Linear<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        self.weight.visit_parameters(visitor);
        if let Some(bias) = &self.bias {
            bias.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        self.weight.visit_parameters_mut(visitor);
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight.set_trainable(trainable);
        if let Some(bias) = &mut self.bias {
            bias.set_trainable(trainable);
        }
    }
}

/// Generic affine layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNorm<T> {
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// Optional trainable scale.
    pub weight: Option<Parameter<T>>,
    /// Optional trainable bias.
    pub bias: Option<Parameter<T>>,
}

impl<T: Tensor> LayerNorm<T> {
    /// Creates an unloaded affine layer normalization.
    pub fn unloaded(
        dimensions: i32,
        epsilon: f32,
        weight: Option<ParameterSpec>,
        bias: Option<ParameterSpec>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            epsilon,
            weight: weight
                .map(|weight| Parameter::unloaded(weight, &[dimensions], context))
                .transpose()?,
            bias: bias
                .map(|bias| Parameter::unloaded(bias, &[dimensions], context))
                .transpose()?,
        })
    }

    /// Applies layer normalization through the selected backend.
    pub fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        T::layer_norm(
            input,
            self.weight.as_ref().map(Parameter::as_ref),
            self.bias.as_ref().map(Parameter::as_ref),
            self.epsilon,
            context,
        )
    }
}

impl<T: 'static> Parameterized<T> for LayerNorm<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        if let Some(weight) = &self.weight {
            weight.visit_parameters(visitor);
        }
        if let Some(bias) = &self.bias {
            bias.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        if let Some(weight) = &mut self.weight {
            weight.visit_parameters_mut(visitor);
        }
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(weight) = &mut self.weight {
            weight.set_trainable(trainable);
        }
        if let Some(bias) = &mut self.bias {
            bias.set_trainable(trainable);
        }
    }
}

#[cfg(test)]
mod parameter_topology_tests {
    use super::*;

    #[derive(Parameterized)]
    #[parameterized(tensor = "i32")]
    struct DerivedModule {
        first: Parameter<i32>,
        second: Option<Parameter<i32>>,
        #[parameter(skip)]
        label: &'static str,
    }

    #[derive(Parameterized)]
    #[parameterized(tensor = "i32")]
    enum DerivedChoice {
        Present(Parameter<i32>),
        Empty,
    }

    fn parameter(id: &str, value: i32) -> Parameter<i32> {
        Parameter::new(ParameterSpec::trainable(id).unwrap(), value)
    }

    #[test]
    fn derive_recurses_through_structs_options_and_enums() {
        let mut module = DerivedModule {
            first: parameter("first.weight", 1),
            second: Some(parameter("second.weight", 2)),
            label: "not a parameter",
        };
        assert_eq!(module.label, "not a parameter");
        let metadata = validate_parameter_topology::<i32, _>(&module).unwrap();
        assert_eq!(
            metadata
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["first.weight", "second.weight"]
        );

        module.set_trainable(false);
        assert!(validate_parameter_topology::<i32, _>(&module)
            .unwrap()
            .iter()
            .all(|entry| !entry.trainable));

        let choice = DerivedChoice::Present(parameter("choice.weight", 3));
        assert_eq!(
            validate_parameter_topology::<i32, _>(&choice).unwrap()[0]
                .id
                .as_str(),
            "choice.weight"
        );
        assert!(validate_parameter_topology::<i32, _>(&DerivedChoice::Empty)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn validation_rejects_duplicates_and_invalid_aliases() {
        let duplicate = vec![parameter("same.weight", 1), parameter("same.weight", 2)];
        assert!(matches!(
            validate_parameter_topology::<i32, _>(&duplicate),
            Err(ParameterTopologyError::DuplicateId(id)) if id.as_str() == "same.weight"
        ));

        let alias = Parameter::new(
            ParameterSpec {
                id: ParameterId::new("alias.weight").unwrap(),
                trainable: true,
                alias_of: Some(ParameterId::new("missing.weight").unwrap()),
                group: None,
            },
            1,
        );
        assert!(matches!(
            validate_parameter_topology::<i32, _>(&alias),
            Err(ParameterTopologyError::MissingAliasDestination { .. })
        ));
    }
}

/// Generic rotary positional encoding configuration.
#[derive(Debug, Clone, Copy)]
pub struct Rope {
    dimensions: i32,
    traditional: bool,
    base: f32,
    scale: f32,
}

impl Rope {
    /// Creates a rotary positional encoding.
    pub const fn new(dimensions: i32, traditional: bool, base: f32, scale: f32) -> Self {
        Self {
            dimensions,
            traditional,
            base,
            scale,
        }
    }

    /// Applies rotary positional encoding through the selected backend.
    pub fn forward<T: Tensor>(
        &self,
        input: &T,
        offset: i32,
        context: &T::Context,
    ) -> Result<T, Error> {
        T::rope(
            input,
            self.dimensions,
            self.traditional,
            self.base,
            self.scale,
            offset,
            context,
        )
    }
}
