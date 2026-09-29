//! Released residual identities and the architecture-owned shared decoder boundary.
use super::config::Config;
use crate::decoder::{ComponentInstrumentation, DecoderBoundary};
use eredu_nn::residual_streams::{GatedResidual, GatedResidualSpec, ResidualStreamGeometry};
use eredu_nn::{
    Error, LinearFormatSpec, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationScale, ParameterSpec, Tensor,
};

/// Constructs the exact released mixer namespace with caller-retained physical formats.
/// The same equation serves block ingress/injection and the target/prediction readout.
pub fn mixer_spec(
    config: &Config,
    root: &str,
    injection: bool,
    mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
) -> Result<GatedResidualSpec, Error> {
    let geometry = ResidualStreamGeometry::new(config.residual.streams, config.hidden_size)?;
    let width = geometry.flattened_width();
    let mut linear = |suffix: &str, input, output| -> Result<LinearSpec, Error> {
        let name = format!("{root}.{suffix}.weight");
        Ok(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
            bias: None,
            format: format(&name)?,
        })
    };
    let spec = GatedResidualSpec {
        geometry,
        normalization: NormalizationConstructionSpec {
            dimensions: width,
            groups: Some(geometry.streams()),
            epsilon: config.norm_epsilon,
            scale: NormalizationScale::LearnedOffset {
                weight: ParameterSpec::trainable(format!("{root}.hc_norm.weight"))
                    .map_err(Error::backend)?,
                offset: 1.0,
            },
        },
        down: linear("input_mix_weight_down", width, config.residual.rank)?,
        up: linear("input_mix_weight_up", config.residual.rank, width)?,
        injection: injection
            .then(|| linear("block_inject_weight", width, geometry.streams()))
            .transpose()?,
    };
    spec.validate()?;
    Ok(spec)
}

/// Expands ordinary/media embeddings and collapses complete residual streams using
/// the released gated mixer. No extra final RMSNorm is applied after this collapse.
#[derive(Debug, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ResidualBoundary<B: NeuralBackend> {
    #[parameter(skip)]
    geometry: ResidualStreamGeometry,
    mixer: GatedResidual<B>,
}
impl<B: NeuralBackend> Clone for ResidualBoundary<B> {
    fn clone(&self) -> Self {
        Self {
            geometry: self.geometry,
            mixer: self.mixer.clone(),
        }
    }
}
impl<B: NeuralBackend> ResidualBoundary<B> {
    /// Requires a final-collapse specification with no sublayer injection parameters.
    pub fn new(
        spec: GatedResidualSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if spec.injection.is_some() {
            return Err(Error::backend(
                "decoder residual readout must not declare a block injection",
            ));
        }
        B::require_operator_capabilities(
            "qwen4_exp residual boundary",
            eredu_nn::NeuralOperatorCapabilities::BROADCAST_TO,
        )?;
        Ok(Self {
            geometry: spec.geometry,
            mixer: GatedResidual::new(spec, context)?,
        })
    }
}
impl<B: NeuralBackend> DecoderBoundary<B> for ResidualBoundary<B> {
    fn embedding_width(&self) -> i32 {
        self.geometry.hidden_size()
    }
    fn expand(
        &self,
        embeddings: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.geometry.expand(embeddings, context)
    }
    fn collapse(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let residual = instrumentation.apply("residual", residual.clone())?;
        let collapsed = self.mixer.forward(&residual, context)?.mixed;
        instrumentation.apply("normalized", collapsed)
    }
}
