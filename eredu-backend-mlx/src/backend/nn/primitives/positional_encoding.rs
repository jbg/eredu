//! Backend-owned rotary-position module state.

use eredu_backend_mlx_macros::ModuleParameters;
use safemlx::{error::Exception, Array, Stream};

use crate::module::Module;

/// Rotary positional encoding configured for a backend operator.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct RotaryPositionalEncoding {
    /// Number of feature dimensions to rotate.
    pub dimensions: i32,
    /// Whether consecutive pairs use traditional ordering.
    pub traditional: bool,
    /// Frequency base.
    pub base: f32,
    /// Position scale.
    pub scale: f32,
}

/// Builder for backend RoPE configuration.
#[derive(Debug, Clone)]
pub struct RotaryPositionalEncodingBuilder {
    dimensions: i32,
    traditional: bool,
    base: f32,
    scale: f32,
}

impl RotaryPositionalEncodingBuilder {
    pub fn new(dimensions: i32) -> Self {
        Self {
            dimensions,
            traditional: RotaryPositionalEncoding::DEFAULT_TRADITIONAL,
            base: RotaryPositionalEncoding::DEFAULT_BASE,
            scale: RotaryPositionalEncoding::DEFAULT_SCALE,
        }
    }

    pub fn traditional(mut self, traditional: bool) -> Self {
        self.traditional = traditional;
        self
    }

    pub fn base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> Result<RotaryPositionalEncoding, std::convert::Infallible> {
        Ok(RotaryPositionalEncoding {
            dimensions: self.dimensions,
            traditional: self.traditional,
            base: self.base,
            scale: self.scale,
        })
    }
}

impl RotaryPositionalEncoding {
    pub const DEFAULT_TRADITIONAL: bool = false;
    pub const DEFAULT_BASE: f32 = 10_000.0;
    pub const DEFAULT_SCALE: f32 = 1.0;
}

/// Input and sequence offset for rotary encoding.
#[derive(Debug, Clone)]
pub struct RopeInput<'a> {
    /// Input tensor.
    pub x: &'a Array,
    /// Sequence offset.
    pub offset: i32,
}

/// Builder for a RoPE invocation.
#[derive(Debug, Clone)]
pub struct RopeInputBuilder<'a> {
    x: &'a Array,
    offset: i32,
}

impl<'a> RopeInputBuilder<'a> {
    pub fn new(x: &'a Array) -> Self {
        Self {
            x,
            offset: RopeInput::DEFAULT_OFFSET,
        }
    }

    pub fn offset(mut self, offset: i32) -> Self {
        self.offset = offset;
        self
    }

    pub fn build(self) -> Result<RopeInput<'a>, std::convert::Infallible> {
        Ok(RopeInput {
            x: self.x,
            offset: self.offset,
        })
    }
}

impl RopeInput<'_> {
    pub const DEFAULT_OFFSET: i32 = 0;
}

impl<'a> From<&'a Array> for RopeInput<'a> {
    fn from(x: &'a Array) -> Self {
        Self { x, offset: 0 }
    }
}

impl<'a> From<(&'a Array,)> for RopeInput<'a> {
    fn from((x,): (&'a Array,)) -> Self {
        Self { x, offset: 0 }
    }
}

impl<'a> From<(&'a Array, i32)> for RopeInput<'a> {
    fn from((x, offset): (&'a Array, i32)) -> Self {
        Self { x, offset }
    }
}

impl<'a, Input> Module<Input> for RotaryPositionalEncoding
where
    Input: Into<RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Array, Exception> {
        let RopeInput { x, offset } = input.into();
        safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            self.base,
            self.scale,
            offset,
            None,
            stream,
        )
    }

    fn training_mode(&mut self, _mode: bool) {}
}
