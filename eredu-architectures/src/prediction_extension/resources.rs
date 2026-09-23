//! Resource topology projected from ordinary invocation discovery and selection.
use eredu_core::speculative::{SpeculativeCaptureBinding, SpeculativeCaptureScope};
use eredu_runtime::prediction_resources::{EmbeddedPredictionTopology, PredictionExecutionMode};

impl crate::SelectedPreparation {
    /// Additional prediction invocations retained by cold selection. This reads
    /// neither payloads nor native resources and does not enable a fit verdict.
    pub fn embedded_prediction_topology(
        &self,
    ) -> Result<Option<EmbeddedPredictionTopology>, eredu_core::resources::ResourceDescriptionError>
    {
        let Some(extension) = self.prediction_extension() else {
            return Ok(None);
        };
        let selected = self
            .prediction_realization()
            .ok_or_else(|| invalid("prediction extension has no selected strategy"))?;
        let descriptor =
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor();
        let mut nodes = Vec::new();
        let mut invocations = Vec::new();
        let mut state_layers = Vec::new();
        for node in &descriptor.nodes {
            // Container-only prediction roots are not physical invocations.
            if node.kind == eredu_core::ArchitectureNodeKind::Prediction
                && !descriptor
                    .speculative_invocations
                    .iter()
                    .any(|binding| binding.node_id == node.id)
                && !descriptor
                    .component_scopes
                    .iter()
                    .any(|scope| scope.node_id == node.id)
                && node.observation_paths.is_empty()
            {
                continue;
            }
            let scope =
                crate::speculative_execution::speculative_capture_scope(&descriptor, &node.id)
                    .map_err(|e| invalid(e.to_string()))?;
            if scope == SpeculativeCaptureScope::Target {
                continue;
            }
            if let Some(layer) = node.layer_index {
                state_layers.push(layer);
            }
            nodes.push(node.clone());
            invocations.push(SpeculativeCaptureBinding {
                node_id: node.id.clone(),
                scope,
            });
        }
        state_layers.sort_unstable();
        state_layers.dedup();
        let ids = nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let edges = descriptor
            .edges
            .iter()
            .filter(|edge| ids.contains(edge.to.as_str()))
            .cloned()
            .collect();
        let mut groups = nodes
            .iter()
            .flat_map(|node| node.parameter_groups.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>();
        // Preserve every canonical sharing target. A semantic alias must not
        // cause another physical copy of target embedding/readout parameters.
        loop {
            let before = groups.len();
            for group in &descriptor.parameter_groups {
                if groups.contains(&group.id) {
                    if let Some(shared) = &group.shared_with {
                        groups.insert(shared.clone());
                    }
                }
            }
            if groups.len() == before {
                break;
            }
        }
        let parameters = descriptor
            .parameter_groups
            .iter()
            .filter(|group| groups.contains(&group.id))
            .cloned()
            .collect();
        let complete_state = super::prediction_extension_capability(extension)
            .map_err(|e| invalid(e.to_string()))?;
        let state = state_layers
            .iter()
            .map(|&layer| {
                let policy = complete_state
                    .state_layout()
                    .layer_layout()
                    .get(layer)
                    .ok_or_else(|| {
                        invalid(format!(
                            "prediction state layer {layer} is absent from ordinary capability"
                        ))
                    })?;
                Ok(eredu_runtime::prediction_resources::PredictionStateLayer {
                    layer,
                    policy: policy.clone(),
                    processed_token_offset: complete_state.state_layout().layer_prefix_offsets()
                        [layer],
                })
            })
            .collect::<Result<Vec<_>, eredu_core::resources::ResourceDescriptionError>>()?;
        Ok(Some(EmbeddedPredictionTopology {
            mode: PredictionExecutionMode::from_strategy(
                selected.requirements().strategy().class(),
            )?,
            proposal_capacity: selected.requirements().strategy().proposal_capacity().get(),
            nodes,
            edges,
            parameters,
            invocations,
            target_features: selected.requirements().capture().clone(),
            state,
        }))
    }
}
fn invalid(reason: impl Into<String>) -> eredu_core::resources::ResourceDescriptionError {
    eredu_core::resources::ResourceDescriptionError::Invalid(reason.into())
}
