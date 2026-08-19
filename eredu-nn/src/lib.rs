//! Backend-neutral neural computation contracts.
//!
//! Architecture implementations use opaque backend tensors through this
//! interface. Tensor storage, physical layout, laziness, device placement, and
//! synchronization remain owned by the selected backend.

#![warn(missing_docs)]

use std::fmt::Debug;

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
    /// Standard causal mask.
    Causal,
    /// Backend tensor added to attention logits.
    Tensor(&'a T),
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
    value: T,
}

impl<T> Parameter<T> {
    /// Wraps a backend tensor as a parameter.
    pub const fn new(value: T) -> Self {
        Self { value }
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
    pub fn unloaded(shape: &[i32], context: &T::Context) -> Result<Self, Error> {
        Ok(Self::new(T::unloaded_f32(shape, context)?))
    }
}

/// Recursive parameter traversal implemented by shared architecture modules.
pub trait ModuleParameters<T> {
    /// Visits every parameter using stable dot-separated checkpoint names.
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T));
    /// Mutably visits every parameter.
    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    );
}

impl<T> ModuleParameters<T> for Parameter<T> {
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T)) {
        visitor(prefix, &self.value);
    }

    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    ) {
        visitor(prefix, self);
    }
}

impl<T, M: ModuleParameters<T>> ModuleParameters<T> for Vec<M> {
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T)) {
        for (index, module) in self.iter().enumerate() {
            module.visit_parameters(&parameter_name(prefix, &index.to_string()), visitor);
        }
    }

    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    ) {
        for (index, module) in self.iter_mut().enumerate() {
            module.visit_parameters_mut(&parameter_name(prefix, &index.to_string()), visitor);
        }
    }
}

impl<T, M: ModuleParameters<T>> ModuleParameters<T> for Option<M> {
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T)) {
        if let Some(module) = self {
            module.visit_parameters(prefix, visitor);
        }
    }

    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    ) {
        if let Some(module) = self {
            module.visit_parameters_mut(prefix, visitor);
        }
    }
}

/// Joins one checkpoint parameter path component.
pub fn parameter_name(prefix: &str, field: &str) -> String {
    if prefix.is_empty() {
        field.to_owned()
    } else {
        format!("{prefix}.{field}")
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
    pub fn unloaded(
        input: i32,
        output: i32,
        bias: bool,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(&[output, input], context)?,
            bias: bias
                .then(|| Parameter::unloaded(&[output], context))
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

impl<T> ModuleParameters<T> for Linear<T> {
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T)) {
        self.weight
            .visit_parameters(&parameter_name(prefix, "weight"), visitor);
        if let Some(bias) = &self.bias {
            bias.visit_parameters(&parameter_name(prefix, "bias"), visitor);
        }
    }

    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    ) {
        self.weight
            .visit_parameters_mut(&parameter_name(prefix, "weight"), visitor);
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(&parameter_name(prefix, "bias"), visitor);
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
    pub fn unloaded(dimensions: i32, epsilon: f32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            epsilon,
            weight: Some(Parameter::unloaded(&[dimensions], context)?),
            bias: Some(Parameter::unloaded(&[dimensions], context)?),
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

impl<T> ModuleParameters<T> for LayerNorm<T> {
    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &T)) {
        if let Some(weight) = &self.weight {
            weight.visit_parameters(&parameter_name(prefix, "weight"), visitor);
        }
        if let Some(bias) = &self.bias {
            bias.visit_parameters(&parameter_name(prefix, "bias"), visitor);
        }
    }

    fn visit_parameters_mut(
        &mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &mut Parameter<T>),
    ) {
        if let Some(weight) = &mut self.weight {
            weight.visit_parameters_mut(&parameter_name(prefix, "weight"), visitor);
        }
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(&parameter_name(prefix, "bias"), visitor);
        }
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
