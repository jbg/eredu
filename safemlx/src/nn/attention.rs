use crate::{
    array,
    builder::Builder,
    error::{Exception, MultiHeadAttentionBuildError},
    module::Module,
    nn::{Linear, LinearBuilder},
    ops::{arange, expand_dims, matmul, softmax_axis},
    quantization::MaybeQuantized,
    Array, ArrayElement, FromScalar,
};
use num_traits::bounds::LowerBounded;
use safemlx_internal_macros::{generate_builder, Buildable, Builder};
use safemlx_macros::{ModuleParameters, Quantizable};

/// Builder for the [`MultiHeadAttention`] module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_multi_head_attention,
    err = MultiHeadAttentionBuildError,
)]
pub struct MultiHeadAttentionBuilder {
    /// Model dimensions and default for the other dimensions if they are not supplied.
    pub dims: i32,

    /// Number of attention heads.
    pub num_heads: i32,

    /// Input dimensions of queries.
    #[builder(optional, default = None)]
    pub query_input_dims: Option<i32>,

    /// Input dimensions of keys.
    #[builder(optional, default = None)]
    pub key_input_dims: Option<i32>,

    /// Input dimensions of values.
    #[builder(optional, default = None)]
    pub value_input_dims: Option<i32>,

    /// Dimensions of values after the projection.
    #[builder(optional, default = None)]
    pub value_dims: Option<i32>,

    /// Dimensions new values will be projected to.
    #[builder(optional, default = None)]
    pub value_output_dims: Option<i32>,

    /// If `true`, use a bias in the [`Linear`] layers.
    #[builder(optional, default = MultiHeadAttention::DEFAULT_BIAS)]
    pub bias: bool,
}

fn build_multi_head_attention(
    builder: MultiHeadAttentionBuilder,
) -> Result<MultiHeadAttention, MultiHeadAttentionBuildError> {
    if builder.dims % builder.num_heads != 0 {
        return Err(MultiHeadAttentionBuildError::InvalidNumHeads(
            builder.num_heads,
        ));
    }

    let dims = builder.dims;
    let bias = builder.bias;
    let query_input_dims = builder.query_input_dims.unwrap_or(builder.dims);
    let key_input_dims = builder.key_input_dims.unwrap_or(builder.dims);
    let value_input_dims = builder.value_input_dims.unwrap_or(builder.dims);
    let value_dims = builder.value_dims.unwrap_or(builder.dims);
    let value_output_dims = builder.value_output_dims.unwrap_or(builder.dims);

    let num_heads = builder.num_heads;

    let query_proj = LinearBuilder::new(query_input_dims, dims)
        .bias(bias)
        .build()?;
    let key_proj = LinearBuilder::new(key_input_dims, dims)
        .bias(bias)
        .build()?;
    let value_proj = LinearBuilder::new(value_input_dims, value_dims)
        .bias(bias)
        .build()?;
    let output_proj = LinearBuilder::new(value_dims, value_output_dims)
        .bias(bias)
        .build()?;

    Ok(MultiHeadAttention {
        num_heads,
        query_proj: MaybeQuantized::new(query_proj),
        key_proj: MaybeQuantized::new(key_proj),
        value_proj: MaybeQuantized::new(value_proj),
        output_proj: MaybeQuantized::new(output_proj),
    })
}

/// Implements scaled dot-product attention with multiple heads.
#[derive(Debug, Clone, ModuleParameters, Quantizable, Buildable)]
#[module(root = crate)]
#[quantizable(root = crate)]
#[buildable(root = crate)]
pub struct MultiHeadAttention {
    /// Number of attention heads.
    pub num_heads: i32,

    /// Query projection layer.
    #[quantizable]
    #[param]
    pub query_proj: MaybeQuantized<Linear>,

    /// Key projection layer.
    #[quantizable]
    #[param]
    pub key_proj: MaybeQuantized<Linear>,

    /// Value projection layer.
    #[quantizable]
    #[param]
    pub value_proj: MaybeQuantized<Linear>,

    /// Output projection layer.
    #[quantizable]
    #[param]
    pub output_proj: MaybeQuantized<Linear>,
}

impl MultiHeadAttention {
    /// Default value for the `bias` field.
    pub const DEFAULT_BIAS: bool = false;

    /// Creates an additive causal mask for use with [`MultiHeadAttention`].
    pub fn create_additive_causal_mask<T>(
        n: i32,
        stream: &crate::Stream,
    ) -> Result<Array, Exception>
    where
        T: ArrayElement + LowerBounded,
        Array: FromScalar<T>,
    {
        let indices = arange::<_, T>(0, n, 1, stream)?;
        let left = expand_dims(&indices, 1, stream)?;
        let right = expand_dims(&indices, 0, stream)?;
        let mask = left.lt(right, stream)?;
        mask.as_type::<T>(stream)?
            .multiply(array!(T::min_value()), stream)
    }
}

generate_builder! {
    /// Input to the [`MultiHeadAttention`] module.
    #[derive(Debug, Clone, Buildable)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct MultiHeadAttentionInput<'a> {
        /// Queries.
        pub queries: &'a Array,

        /// Keys.
        pub keys: &'a Array,

        /// Values.
        pub values: &'a Array,

        /// Optional additive attention mask.
        #[builder(optional, default = None)]
        pub mask: Option<&'a Array>,
    }
}

impl<'a> From<(&'a Array, &'a Array, &'a Array)> for MultiHeadAttentionInput<'a> {
    fn from((queries, keys, values): (&'a Array, &'a Array, &'a Array)) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask: None,
        }
    }
}

impl<'a> From<(&'a Array, &'a Array, &'a Array, &'a Array)> for MultiHeadAttentionInput<'a> {
    fn from((queries, keys, values, mask): (&'a Array, &'a Array, &'a Array, &'a Array)) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask: Some(mask),
        }
    }
}

impl<'a> From<(&'a Array, &'a Array, &'a Array, Option<&'a Array>)>
    for MultiHeadAttentionInput<'a>
{
    fn from(
        (queries, keys, values, mask): (&'a Array, &'a Array, &'a Array, Option<&'a Array>),
    ) -> Self {
        MultiHeadAttentionInput {
            queries,
            keys,
            values,
            mask,
        }
    }
}

impl<'a, Input> Module<Input> for MultiHeadAttention
where
    Input: Into<MultiHeadAttentionInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: Input,
        stream: &crate::Stream,
    ) -> Result<Self::Output, Self::Error> {
        let input = input.into();
        let queries = self.query_proj.forward(input.queries, stream)?;
        let keys = self.key_proj.forward(input.keys, stream)?;
        let values = self.value_proj.forward(input.values, stream)?;

        let B = queries.dim(0);
        let L = queries.dim(1);
        let S = keys.dim(1);

        let queries = queries
            .reshape(&[B, L, self.num_heads, -1], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let keys = keys
            .reshape(&[B, S, self.num_heads, -1], stream)?
            .transpose_axes(&[0, 2, 3, 1], stream)?;
        let values = values
            .reshape(&[B, S, self.num_heads, -1], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;

        // Dimensions are [batch x num_heads x sequence x hidden_dim].
        let scale = f32::sqrt(1.0 / queries.dim(-1) as f32);
        let scaled_queries = queries.multiply(array!(scale), stream)?;
        let mut scores = scaled_queries.matmul(&keys, stream)?;
        if let Some(mask) = input.mask {
            scores = scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?;
        }
        scores = softmax_axis(&scores, -1, None, stream)?;
        let value_hat = matmul(&scores, &values, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[B, L, -1], stream)?;

        self.output_proj.forward(&value_hat, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.query_proj.training_mode(mode);
        self.key_proj.training_mode(mode);
        self.value_proj.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}
