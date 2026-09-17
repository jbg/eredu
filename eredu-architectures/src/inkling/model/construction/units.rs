//! Actual initial graph, state and unit declarations, closed under their config owner.
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};

#[derive(Debug)]
pub(in super::super) struct PreparedUnits {
    config: SharedCompositeConfig<ModelArgs>,
    geometry: Option<Arc<LocalGeometry>>,
    pub(in super::super) layouts: InklingStateLayouts,
    pub(in super::super) global_state_layers: usize,
    pub(in super::super) description: ArchitectureParameterDescription,
    pub(in super::super) transports: [eredu_runtime::ArchitectureGroupTransport; 3],
    pub(in super::super) targets: Vec<super::super::super::text::DecoderLayerSpec>,
    pub(in super::super) vision: Vec<super::super::super::vision::VisionLayerSpec>,
}
impl PreparedUnits {
    pub(in super::super) fn matches_realization(
        &self,
        plan: &crate::ExpertRealizationPlan<crate::inkling::ExpertBankRealization>,
    ) -> bool {
        self.targets.iter().enumerate().all(|(index, unit)| {
            unit.matches_realization(plan.unit_spec(TEXT_EXECUTION_GROUP, index))
        })
    }
    pub(in super::super) fn validate<
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    >(
        &self,
        model: &LayeredModel<B>,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<(), Error> {
        metadata.controls::<(&Self, &LayeredModel<B>, bool)>()?;
        let geometry = match (&self.geometry, &model.parallel_geometry) {
            (None, None) => true,
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        };
        if !std::ptr::eq(&*self.config, &*model.args) || !geometry {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(())
    }
}
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    pub(crate) fn prepare_graph_units(
        &mut self,
        selected: Option<&crate::SelectedRoutedTextRealization>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
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
        let count =
            usize::try_from(self.args.text_config.num_hidden_layers).map_err(Error::backend)?;
        let mut targets = Vec::with_capacity(count);
        for index in 0..count {
            let text = self
                .parallel_geometry
                .as_ref()
                .and_then(|geometry| geometry.text_layer(index))
                .unwrap_or(&self.args.text_config);
            let policy = text
                .layer_policy(index)
                .ok_or_else(|| Error::backend("Inkling source has no target layer policy"))?;
            let shared_index = count
                .checked_add(index)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let realization =
                if policy.feed_forward == super::super::super::FeedForwardPolicy::SparseMoe {
                    match selected {
                        Some(plan) => Some(crate::inkling::ExpertBankRealization {
                            routed: plan
                                .unit_spec(TEXT_EXECUTION_GROUP, index)
                                .cloned()
                                .ok_or_else(|| {
                                    Error::backend("Inkling source has no routed expert unit")
                                })?,
                            shared: plan
                                .unit_spec(TEXT_EXECUTION_GROUP, shared_index)
                                .cloned()
                                .ok_or_else(|| {
                                    Error::backend("Inkling source has no shared expert unit")
                                })?,
                        }),
                        None => self
                            .expert_realization
                            .as_ref()
                            .map(|plan| {
                                plan.unit_spec(TEXT_EXECUTION_GROUP, index)
                                    .cloned()
                                    .ok_or_else(|| {
                                        Error::backend("Inkling source has no local expert unit")
                                    })
                            })
                            .transpose()?,
                    }
                } else {
                    None
                };
            targets.push(super::super::super::text::DecoderLayerSpec::new(
                text,
                policy,
                &format!("model.layers.{index}"),
                index,
                shared_index,
                realization,
            )?);
        }
        let mut vision = Vec::new();
        if let Some(config) = &self.args.vision_config {
            let layers = config.layer_specs();
            vision.reserve(layers.len());
            for (index, spec) in layers.into_iter().enumerate() {
                vision.push(super::super::super::vision::VisionLayerSpec::new(
                    config, index, spec,
                )?);
            }
        }
        let source = PreparedUnits {
            config: self.args.clone(),
            geometry: self.parallel_geometry.clone(),
            targets,
            vision,
            layouts: self.state_layouts()?,
            global_state_layers: state_layer_count(&self.args)?,
            description: self.parameter_description_impl(context)?,
            transports: std::array::from_fn(Self::canonical_group_transport),
        };
        self.construction.set_units(SharedCompositeConfig::new(
            source,
            B::construction_metadata(context),
        )?);
        Ok(())
    }
    pub(in super::super) fn checked_units(
        &self,
        context: &WorkspaceContext,
    ) -> Result<&PreparedUnits, Error> {
        let metadata = crate::decoder::identity::Metadata::new(Some(context));
        metadata.controls::<(&Self, Option<&PreparedUnits>)>()?;
        let source = self
            .construction
            .units()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        source.validate(self, metadata)?;
        Ok(source)
    }
}

#[cfg(test)]
mod tests;
