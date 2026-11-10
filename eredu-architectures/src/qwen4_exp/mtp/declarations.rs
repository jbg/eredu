//! Prediction parameter ownership independent of invocation limits and mutable state.
use super::PredictionSharedSpec;
use crate::qwen4_exp::{
    config::Config,
    feed_forward::FeedForwardSublayerSpec,
    target::declarations::{
        linear_formats, residual_formats, unit_formats, validate_format_projection,
        MixerParameters, UnitParameters,
    },
};
use eredu_nn::{
    DeclareLogicalParameters, Error, GroupedGatedProductSpec, LinearFormatSpec, ParameterSpec,
};

#[derive(Debug, Clone)]
pub(crate) struct PredictionParameters {
    pub(crate) shared: PredictionSharedSpec,
    pub(crate) units: Vec<UnitParameters>,
}
impl PredictionParameters {
    pub(crate) fn from_config(
        config: &Config,
        mut experts: impl FnMut(usize) -> Result<GroupedGatedProductSpec, Error>,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let prediction = config
            .prediction
            .as_ref()
            .filter(|p| !p.layers.is_empty())
            .ok_or_else(|| Error::backend("artifact has no prediction schedule"))?;
        let shared = PredictionSharedSpec::from_config(config, &mut format)?;
        let units = prediction
            .layers
            .iter()
            .enumerate()
            .map(|(depth, &kind)| {
                let root = format!("mtp.layers.{depth}");
                Ok(UnitParameters::Decoder {
                    layer: depth,
                    mixer: MixerParameters::from_config(config, kind, &root, &mut format)?,
                    feed_forward: FeedForwardSublayerSpec::from_config(
                        config,
                        depth,
                        &root,
                        experts(depth)?,
                        &mut format,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let parameters = Self { shared, units };
        parameters.logical_parameters()?;
        Ok(parameters)
    }
    pub(crate) fn with_formats(
        &self,
        mut format: impl FnMut(&ParameterSpec, &LinearFormatSpec) -> Result<LinearFormatSpec, Error>,
        mut experts: impl FnMut(
            usize,
            &GroupedGatedProductSpec,
        ) -> Result<GroupedGatedProductSpec, Error>,
    ) -> Result<Self, Error> {
        let mut projected = self.clone();
        linear_formats(
            &mut projected.shared.fusion.embedding_projection,
            &mut format,
        )?;
        linear_formats(&mut projected.shared.fusion.hidden_projection, &mut format)?;
        residual_formats(&mut projected.shared.readout, &mut format)?;
        unit_formats(&mut projected.units, &mut format, &mut experts)?;
        validate_format_projection(self, &projected)?;
        Ok(projected)
    }
}
impl DeclareLogicalParameters for PredictionParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        if self.units.is_empty() {
            return Err(Error::backend("prediction declaration has no depth owners"));
        }
        let mut parameters = self.shared.logical_parameters()?;
        for (depth, unit) in self.units.iter().enumerate() {
            if !matches!(unit, UnitParameters::Decoder { layer, .. } if *layer == depth) {
                return Err(Error::backend(
                    "prediction parameter owner differs from depth",
                ));
            }
            parameters.extend(unit.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}
