//! Prediction ownership retained from the same validated artifact as the target.
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};

#[path = "tensor_prediction.rs"]
mod tensor;
pub use tensor::PreparedTensorPrediction;

/// Shared prediction weights and one bounded owner per depth. Token embeddings
/// and output weights retain target identities and source provenance.
#[derive(Clone)]
pub struct PreparedPrediction {
    pub(super) spec: PredictionSpec,
    shared: PreparedParameters,
    vocabulary: PreparedParameters,
    units: Vec<PreparedParameters>,
    pub(super) artifact: SharedCheckpointSource,
    expert_banks: BTreeMap<String, Arc<PreparedExpertBank>>,
}
impl PreparedTarget {
    /// Prepares embedded prediction without reading any additional tensor payload.
    pub fn prediction(
        &self,
        limits: PredictionLimits,
    ) -> Result<PreparedPrediction, PreparationError> {
        if self.spec.config.prediction.is_none() {
            return Err(PreparationError::MissingPrediction);
        }
        let bank = eredu_runtime::RoutedBankId::new(u32::try_from(self.tables.len() + 1).map_err(
            |_| PreparationError::Contract("prediction bank identity exceeds u32".into()),
        )?);
        let spec = PredictionSpec::from_prepared(
            &self.spec.config,
            limits,
            bank,
            |depth| {
                expert_spec(
                    &self.spec.config,
                    &self.formats,
                    &format!("mtp.layers.{depth}.mlp.experts"),
                )
            },
            |name| self.formats.ordinary(name),
        )?;
        let (shared, recipes) = self
            .prediction_recipes
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        let vocabulary = self
            .static_parameters
            .recipes
            .iter()
            .filter(|(name, _)| {
                name.starts_with("model.embed_tokens.")
                    || (self.spec.head.is_some() && name.starts_with("lm_head."))
            })
            .map(|(name, recipe)| (name.clone(), recipe.clone()))
            .collect();
        let units = recipes
            .iter()
            .map(|recipes| PreparedParameters::new(self.artifact.clone(), recipes.clone()))
            .collect::<Result<_, _>>()?;
        Ok(PreparedPrediction {
            spec,
            shared: PreparedParameters::new(self.artifact.clone(), shared.clone())?,
            vocabulary: PreparedParameters::new(self.artifact.clone(), vocabulary)?,
            units,
            artifact: self.artifact.clone(),
            expert_banks: self
                .expert_banks
                .iter()
                .filter(|(n, _)| n.starts_with("mtp."))
                .map(|(n, r)| (n.clone(), r.clone()))
                .collect(),
        })
    }
}
impl PreparedPrediction {
    /// Exact prediction-only construction and mutable state geometry.
    pub fn spec(&self) -> &PredictionSpec {
        &self.spec
    }
    /// Shared residual fusion and prediction readout; no target readout or vocabulary.
    pub fn shared(&self) -> &PreparedParameters {
        &self.shared
    }
    /// Restricted references to the target's physical embedding/output parameters.
    pub fn vocabulary(&self) -> &PreparedParameters {
        &self.vocabulary
    }
    /// Ordinary parameters of one independently materializable depth.
    pub fn unit(&self, depth: usize) -> Result<&PreparedParameters, PreparationError> {
        self.units.get(depth).ok_or_else(|| {
            PreparationError::Contract("prediction depth is outside the artifact schedule".into())
        })
    }
    /// One bounded expert recipe, distinct from target decoder/table owners.
    pub fn expert(
        &self,
        depth: usize,
        expert: usize,
    ) -> Result<PreparedParameters, PreparationError> {
        self.unit(depth)?;
        let root = format!("mtp.layers.{depth}.mlp.experts");
        let recipes = select_expert(&self.expert_banks, &root, expert)?;
        PreparedParameters::new(
            self.artifact.clone(),
            recipes
                .into_iter()
                .map(|(name, recipe)| (format!("{root}.{name}"), recipe))
                .collect(),
        )
    }
}
