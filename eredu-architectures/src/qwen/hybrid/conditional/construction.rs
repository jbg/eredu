//! Initial target and prediction specs retained beside the actual effective config.
use super::*;
use super::super::HybridConfig;
use crate::replicated_text::SharedCompositeConfig;
use eredu_nn::workspace::WorkspaceMetadataError;
#[derive(Debug)]
struct Specifications {
    config: SharedCompositeConfig<ParsedHybridConfig>,
    geometry: Option<std::sync::Arc<ConditionalLocalGeometry>>,
    realization: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
    targets: Vec<super::super::block::TargetBlockSpec>,
    predictions: Vec<super::super::mtp::PredictionUnitSpec>,
}
#[derive(Clone, Debug)]
pub(crate) struct RetainedConditionalUnits(SharedCompositeConfig<Specifications>);
impl RetainedConditionalUnits {
    pub(super) fn validate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        model: &ConditionalLayeredModel<B>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
            &Self,
            &ConditionalLayeredModel<B>,
            bool,
        )>()?;
        let geometry_matches = match (&self.0.geometry, &model.parallel_geometry) {
            (None, None) => true,
            (Some(left), Some(right)) => std::sync::Arc::ptr_eq(left, right),
            _ => false,
        };
        if !std::ptr::eq(&*self.0.config, &*model.parsed)
            || !geometry_matches
            || self.0.targets.len() != model.target_layers
            || self.0.predictions.len() != model.prediction_steps
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(())
    }
    pub(super) fn matches_realization(
        &self,
        realization: &crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> bool {
        self.0.realization.as_ref() == Some(realization)
    }
    pub(super) fn targets(&self) -> &[super::super::block::TargetBlockSpec] {
        &self.0.targets
    }
    pub(super) fn predictions(&self) -> &[super::super::mtp::PredictionUnitSpec] {
        &self.0.predictions
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ConditionalLayeredModel<B> {
    pub(super) fn target_construction_config(&self, index: usize) -> &HybridConfig {
        self.parallel_geometry
            .as_ref()
            .and_then(|geometry| geometry.text().target(index))
            .unwrap_or(&self.parsed.text)
    }
    pub(crate) fn prepare_construction_units(
        &mut self,
        selected: Option<&crate::SelectedRoutedTextRealization>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        super::super::block::construction::require_source_compiler::<B>(context)?;
        let selected = selected
            .map(|selected| {
                if selected.banks().len() != 1 {
                    return Err(Error::from(WorkspaceMetadataError::Unqualified));
                }
                selected
                    .bank(eredu_runtime::RoutedBankId::new(0))
                    .and_then(|bank| bank.plan().gated())
                    .ok_or_else(|| Error::from(WorkspaceMetadataError::Unqualified))
            })
            .transpose()?;
        let mut targets = Vec::with_capacity(self.target_layers);
        for index in 0..self.target_layers {
            let realization = selected.or(self.expert_realization.as_ref());
            let spec = realization
                .map(|plan| {
                    plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, index)
                        .cloned()
                        .ok_or_else(|| {
                            Error::backend("conditional target source has no selected expert unit")
                        })
                })
                .transpose()?;
            targets.push(super::super::block::TargetBlockSpec::new(
                self.target_construction_config(index),
                index,
                spec,
            )?);
        }
        let mut predictions = Vec::with_capacity(self.prediction_steps);
        for depth in 0..self.prediction_steps {
            let config = self
                .parallel_geometry
                .as_ref()
                .and_then(|geometry| geometry.text().prediction(depth))
                .unwrap_or(&self.parsed.text);
            predictions.push(super::super::mtp::PredictionUnitSpec::new(config, depth)?);
        }
        let source = RetainedConditionalUnits(SharedCompositeConfig::new(
            Specifications {
                config: self.parsed.clone(),
                geometry: self.parallel_geometry.clone(),
                realization: selected.or(self.expert_realization.as_ref()).cloned(),
                targets,
                predictions,
            },
            B::construction_metadata(context),
        )?);
        self.install_construction_units(source, context)
    }
    pub(crate) fn construction_units(&self) -> Option<&RetainedConditionalUnits> {
        self.construction_units.as_ref()
    }
    pub(crate) fn install_construction_units(
        &mut self,
        source: RetainedConditionalUnits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        crate::decoder::ModuleMetadata::new::<B>(context)
            .controls::<(RetainedConditionalUnits, &mut Self)>()?;
        source.validate(self, context)?;
        self.construction_units = Some(source);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
