//! Immutable family parameter authority shared by configuration and source admission.
use super::{
    config::Config, mtp::declarations::PredictionParameters, target::declarations::TargetParameters,
};
use eredu_nn::{DeclareLogicalParameters, Error, LogicalParameterDeclaration};
use std::sync::Arc;

/// Target and embedded prediction ownership, independent of physical source layouts.
#[derive(Debug, Clone)]
pub struct Declaration {
    config: Config,
    pub(in crate::qwen4_exp) target_parameters: Arc<TargetParameters>,
    pub(in crate::qwen4_exp) prediction_parameters: Option<Arc<PredictionParameters>>,
}
impl Declaration {
    /// Declares validated typed slots once at the normalized family boundary.
    pub fn new(config: Config) -> Result<Self, Error> {
        let target = Arc::new(
            super::checkpoint::schema::target_parameters(&config).map_err(Error::backend)?,
        );
        let prediction = config
            .prediction
            .as_ref()
            .map(|_| {
                super::checkpoint::schema::prediction_parameters(&config)
                    .map(Arc::new)
                    .map_err(Error::backend)
            })
            .transpose()?;
        let declaration = Self {
            config,
            target_parameters: target,
            prediction_parameters: prediction,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declaration)
    }
    /// Immutable normalized equations and component policy of these parameter owners.
    pub fn configuration(&self) -> &Config {
        &self.config
    }
    pub(crate) fn target_parameters(&self) -> &TargetParameters {
        &self.target_parameters
    }
}
impl std::ops::Deref for Declaration {
    type Target = Config;
    fn deref(&self) -> &Config {
        &self.config
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = self.target_parameters.logical_parameters()?;
        if let Some(prediction) = &self.prediction_parameters {
            slots.extend(prediction.logical_parameters()?);
        }
        Ok(slots)
    }
}
