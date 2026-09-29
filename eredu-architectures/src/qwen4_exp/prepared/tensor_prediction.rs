//! Prediction tensor ranks over the original admitted parameter sources.
use super::*;
use crate::qwen4_exp::mtp::PredictionTensorPartition;
use crate::qwen4_exp::prepared::tensor_partition::partition_recipe;

/// Exact local prediction recipes, retaining shared fusion/readout and target
/// vocabulary references. Constructing a rank reads no parameter payload.
#[derive(Clone)]
pub struct PreparedTensorPrediction {
    source: PreparedPrediction,
    partition: PredictionTensorPartition,
    units: Vec<PreparedParameters>,
    expert_banks: BTreeMap<String, Arc<PreparedExpertBank>>,
}

impl PreparedPrediction {
    /// Selects physical tensor-rank ranges after the artifact's canonical recipes.
    /// Independent experts retain global IDs; EP ownership can select those IDs
    /// separately without duplicating or reinterpreting the tensor transforms.
    pub fn tensor_partition(
        &self,
        rank: usize,
        ranks: usize,
    ) -> Result<PreparedTensorPrediction, PreparationError> {
        let partition = self.spec.tensor_partition(rank, ranks)?;
        let units = self
            .units
            .iter()
            .map(|owner| {
                let recipes = owner
                    .recipes
                    .iter()
                    .map(|(name, recipe)| {
                        Ok((
                            name.clone(),
                            partition_recipe(
                                owner.source.as_ref(),
                                name,
                                recipe,
                                partition.parameter_selections(name),
                            )?,
                        ))
                    })
                    .collect::<Result<_, PreparationError>>()?;
                PreparedParameters::new(owner.source.clone(), recipes)
            })
            .collect::<Result<_, PreparationError>>()?;
        let expert_banks = self
            .expert_banks
            .iter()
            .map(|(root, bank)| {
                let recipes = bank
                    .recipes
                    .iter()
                    .map(|(name, recipe)| {
                        let canonical = format!("{root}.{name}");
                        Ok((
                            name.clone(),
                            partition_recipe(
                                self.artifact.as_ref(),
                                &canonical,
                                recipe,
                                partition.parameter_selections(&canonical),
                            )?,
                        ))
                    })
                    .collect::<Result<_, PreparationError>>()?;
                Ok((
                    root.clone(),
                    Arc::new(PreparedExpertBank::new(
                        self.artifact.as_ref(),
                        recipes,
                        bank.members.len(),
                    )?),
                ))
            })
            .collect::<Result<_, PreparationError>>()?;
        Ok(PreparedTensorPrediction {
            source: self.clone(),
            partition,
            units,
            expert_banks,
        })
    }
}

impl PreparedTensorPrediction {
    /// Retained local geometry and exact physical source selections.
    pub fn partition(&self) -> &PredictionTensorPartition {
        &self.partition
    }

    /// Independent local prediction state and execution geometry.
    pub fn spec(&self) -> &PredictionSpec {
        self.partition.local_spec()
    }

    /// Replicated fusion and readout from the original prepared owner.
    pub fn shared(&self) -> &PreparedParameters {
        self.source.shared()
    }

    /// Original target embedding/output references, without target collapse.
    pub fn vocabulary(&self) -> &PreparedParameters {
        self.source.vocabulary()
    }

    /// Tensor-local ordinary parameters for one independently acquired depth.
    pub fn unit(&self, depth: usize) -> Result<&PreparedParameters, PreparationError> {
        self.units.get(depth).ok_or_else(|| {
            PreparationError::Contract("prediction depth is outside tensor partition".into())
        })
    }

    /// Tensor-local expert rows addressed by checkpoint-global expert ID.
    pub fn expert(
        &self,
        depth: usize,
        expert: usize,
    ) -> Result<PreparedParameters, PreparationError> {
        self.unit(depth)?;
        let root = format!("mtp.layers.{depth}.mlp.experts");
        let recipes = select_expert(&self.expert_banks, &root, expert)?;
        PreparedParameters::new(
            self.source.artifact.clone(),
            recipes
                .into_iter()
                .map(|(name, recipe)| (format!("{root}.{name}"), recipe))
                .collect(),
        )
    }
}
