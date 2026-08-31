//! Traits for quantization

use safemlx::Stream;

use super::module::{Module, PhysicalParameters};

/// Trait for quantization of modules.
pub trait Quantizable {
    /// The default group size for quantization.
    const DEFAULT_GROUP_SIZE: i32 = 64;

    /// The default number of bits for quantization.
    const DEFAULT_BITS: i32 = 4;

    /// The quantized type.
    type Quantized;

    /// The error type for quantization.
    type QuantizationError;

    /// Quantize the module with the specified group size and number of bits.
    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError>;
}

impl<M> Quantizable for Vec<M>
where
    M: Quantizable,
{
    type Quantized = Vec<M::Quantized>;

    type QuantizationError = M::QuantizationError;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        self.into_iter()
            .map(|m| m.try_into_quantized(group_size, bits, stream))
            .collect()
    }
}

impl<M> Quantizable for Box<M>
where
    M: Quantizable,
{
    type Quantized = Box<M::Quantized>;

    type QuantizationError = M::QuantizationError;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        (*self)
            .try_into_quantized(group_size, bits, stream)
            .map(Box::new)
    }
}

impl<M> Quantizable for Option<M>
where
    M: Quantizable,
{
    type Quantized = Option<M::Quantized>;

    type QuantizationError = M::QuantizationError;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        match self {
            Some(m) => m.try_into_quantized(group_size, bits, stream).map(Some),
            None => Ok(None),
        }
    }
}

/// A wrapper for a quantizable module.
#[derive(Debug, Clone)]
pub enum MaybeQuantized<M>
where
    M: Quantizable,
{
    /// The original module.
    Original(M),

    /// The quantized version of the module.
    Quantized(M::Quantized),
}

impl<M> Quantizable for MaybeQuantized<M>
where
    M: Quantizable,
{
    type Quantized = Self;
    type QuantizationError = <M as Quantizable>::QuantizationError;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self, Self::QuantizationError> {
        match self {
            MaybeQuantized::Original(m) => {
                let quantized = m.try_into_quantized(group_size, bits, stream)?;
                Ok(MaybeQuantized::Quantized(quantized))
            }
            MaybeQuantized::Quantized(q) => Ok(MaybeQuantized::Quantized(q)),
        }
    }
}

impl<M> MaybeQuantized<M>
where
    M: Quantizable,
{
    /// Create a new [`MaybeQuantized`] from the original module.
    pub fn new(module: M) -> Self {
        MaybeQuantized::Original(module)
    }

    /// Quantize the module with a custom quantization function.
    ///
    /// This is useful if one would like to quantize with a custom group size or bit width.
    pub fn quantize_with(
        self,
        op: impl FnOnce(M) -> Result<M::Quantized, M::QuantizationError>,
    ) -> Result<Self, M::QuantizationError> {
        match self {
            MaybeQuantized::Original(m) => op(m).map(MaybeQuantized::Quantized),
            MaybeQuantized::Quantized(q) => Ok(MaybeQuantized::Quantized(q)),
        }
    }

    /// Check if the module is quantized.
    pub fn is_quantized(&self) -> bool {
        match self {
            MaybeQuantized::Original(_) => false,
            MaybeQuantized::Quantized(_) => true,
        }
    }
}

impl<M> PhysicalParameters for MaybeQuantized<M>
where
    M: Quantizable + PhysicalParameters,
    M::Quantized: PhysicalParameters,
{
    fn parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            MaybeQuantized::Original(m) => m.parameters(),
            MaybeQuantized::Quantized(q) => q.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> crate::module::ModuleParamMut<'_> {
        match self {
            MaybeQuantized::Original(m) => m.parameters_mut(),
            MaybeQuantized::Quantized(q) => q.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> crate::module::ModuleParamRef<'_> {
        match self {
            MaybeQuantized::Original(m) => m.trainable_parameters(),
            MaybeQuantized::Quantized(q) => q.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            MaybeQuantized::Original(m) => m.freeze_parameters(recursive),
            MaybeQuantized::Quantized(q) => q.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            MaybeQuantized::Original(m) => m.unfreeze_parameters(recursive),
            MaybeQuantized::Quantized(q) => q.unfreeze_parameters(recursive),
        }
    }
}

impl<M, Input> Module<Input> for MaybeQuantized<M>
where
    M: Quantizable + Module<Input>,
    M::Quantized:
        Module<Input, Output = <M as Module<Input>>::Output, Error = <M as Module<Input>>::Error>,
{
    type Output = <M as Module<Input>>::Output;

    type Error = <M as Module<Input>>::Error;

    fn forward(&mut self, x: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        match self {
            MaybeQuantized::Original(m) => m.forward(x, stream),
            MaybeQuantized::Quantized(q) => q.forward(x, stream),
        }
    }
}
