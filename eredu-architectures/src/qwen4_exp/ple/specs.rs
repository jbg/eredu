//! Lexical parameter facts shared by construction, admission and cold inspection.
use crate::qwen4_exp::config::Config;
use eredu_nn::{
    residual_streams::ResidualStreamGeometry, CausalDepthwiseConvolutionSpec,
    ConvolutionActivation, DeclareLogicalParameters, Error, LinearFormatSpec, LinearSpec,
    LogicalParameterDeclaration, NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
};

/// Owned neural slots for lexical injection; row tables and hash controls have separate providers.
#[derive(Debug, Clone)]
pub struct LexicalParameters {
    /// Complete streams retained across the invocation boundary.
    pub geometry: ResidualStreamGeometry,
    /// Shared n-gram features to one key per residual stream.
    pub key: LinearSpec,
    /// Shared n-gram features to the common injected value.
    pub value: LinearSpec,
    /// Per-stream normalization of projected keys.
    pub key_norm: NormalizationConstructionSpec,
    /// Per-stream normalization of the original residuals.
    pub query_norm: NormalizationConstructionSpec,
    /// Per-stream normalization before causal convolution.
    pub convolution_norm: NormalizationConstructionSpec,
    /// Bias-free dilated convolution followed by SiLU.
    pub convolution: CausalDepthwiseConvolutionSpec,
}
impl LexicalParameters {
    /// Declares an active lexical owner without requiring a table payload, hash literals or execution bounds.
    pub fn from_config(
        config: &Config,
        layer: usize,
        root: &str,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        if !config.ngram.layers.contains(&layer) || layer >= config.layers.len() {
            return Err(Error::backend(
                "lexical parameter owner is absent from the declared schedule",
            ));
        }
        let geometry = ResidualStreamGeometry::new(config.residual.streams, config.hidden_size)?;
        let root = format!("{root}.ple");
        let parameter = |suffix: &str| {
            let mut p = ParameterSpec::trainable(format!("{root}.{suffix}.weight"))
                .map_err(Error::backend)?;
            p.group = Some(root.clone());
            Ok::<_, Error>(p)
        };
        let mut linear = |suffix: &str, output| {
            let weight = parameter(suffix)?;
            Ok::<_, Error>(LinearSpec {
                input: config.ngram.embedding_dim,
                output,
                format: format(weight.id.as_str())?,
                weight,
                bias: None,
            })
        };
        let norm = |suffix: &str| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec {
                reduction: eredu_nn::NormalizationReduction::Native,
                dimensions: geometry.flattened_width(),
                groups: Some(geometry.streams()),
                epsilon: config.norm_epsilon,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(suffix)?,
                    offset: 1.,
                },
            })
        };
        let spec = Self {
            geometry,
            key: linear("key_proj", geometry.flattened_width())?,
            value: linear("value_proj", geometry.hidden_size())?,
            key_norm: norm("norm_key")?,
            query_norm: norm("norm_query")?,
            convolution_norm: norm("norm_conv")?,
            convolution: CausalDepthwiseConvolutionSpec {
                arithmetic: eredu_nn::ConvolutionArithmetic::Native,
                channels: geometry.flattened_width(),
                kernel_size: config.ngram.kernel,
                dilation: config.ngram.order,
                weight: parameter("conv1d")?,
                bias: None,
                activation: ConvolutionActivation::Silu,
            },
        };
        spec.validate()?;
        Ok(spec)
    }
    /// Rejects inconsistent slot geometry, policy, identities and sharing before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        let width = self.geometry.flattened_width();
        if self.key.input <= 0
            || self.value.input != self.key.input
            || self.key.output != width
            || self.value.output != self.geometry.hidden_size()
        {
            return Err(Error::backend(
                "lexical affine geometry differs from residual streams",
            ));
        }
        for p in [&self.key, &self.value] {
            if p.bias.is_some() {
                return Err(Error::backend("lexical projections must be bias-free"));
            }
            p.format.validate_for_weight(&p.weight)?;
        }
        for n in [&self.key_norm, &self.query_norm, &self.convolution_norm] {
            n.validate()?;
            if n.dimensions != width || n.groups != Some(self.geometry.streams()) {
                return Err(Error::backend(
                    "lexical normalization must reduce within each residual stream",
                ));
            }
        }
        self.convolution.validate()?;
        if self.convolution.channels != width
            || self.convolution.bias.is_some()
            || self.convolution.activation != ConvolutionActivation::Silu
        {
            return Err(Error::backend(
                "lexical convolution must preserve streams, with no bias and SiLU",
            ));
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for LexicalParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.key.logical_parameters()?;
        p.extend(self.value.logical_parameters()?);
        for n in [&self.key_norm, &self.query_norm, &self.convolution_norm] {
            p.extend(n.logical_parameters()?);
        }
        p.extend(self.convolution.logical_parameters()?);
        Ok(p)
    }
}
