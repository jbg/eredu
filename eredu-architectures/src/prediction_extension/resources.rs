//! Resource topology projected from ordinary invocation discovery and selection.
use eredu_core::speculative::{SpeculativeCaptureBinding, SpeculativeCaptureScope};
use eredu_runtime::prediction_resources::{EmbeddedPredictionTopology, PredictionExecutionMode};

impl crate::SelectedPreparation {
    /// Additional prediction invocations retained by cold selection. This reads
    /// neither payloads nor native resources. Missing mechanisms remain explicit.
    pub fn embedded_prediction_topology(
        &self,
    ) -> Result<Option<EmbeddedPredictionTopology>, eredu_core::resources::ResourceDescriptionError>
    {
        let Some(selected) = self.prediction_realization() else {
            return Ok(None);
        };
        let extension = self.prediction_extension();
        let retained = self
            .qwen4_construction()
            .map(|construction| construction.prediction_discovery())
            .transpose()
            .map_err(|error| invalid(error.to_string()))?
            .flatten();
        let descriptor = if let Some((descriptor, _)) = &retained {
            descriptor.clone()
        } else if let Some(extension) = extension {
            crate::processor_plan::ArtifactArchitecturePlan::from_safetensors_architecture(
                extension.complete_architecture().clone(),
            )
            .architecture_descriptor()
        } else {
            return Err(invalid(
                "selected prediction has no retained architecture discovery",
            ));
        };
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
        let complete_state = extension
            .map(super::prediction_extension_capability)
            .transpose()
            .map_err(|e| invalid(e.to_string()))?;
        let state = state_layers
            .iter()
            .map(|&layer| {
                let (policy, processed_token_offset) = if let Some((_, selected_state)) = &retained
                {
                    let layout = selected_state.layout();
                    (
                        layout.layers().get(layer),
                        layout.layer_prefix_offsets().get(layer).copied(),
                    )
                } else if let Some(complete_state) = &complete_state {
                    let layout = complete_state.state_layout();
                    (
                        layout.layer_layout().get(layer),
                        layout.layer_prefix_offsets().get(layer).copied(),
                    )
                } else {
                    (None, None)
                };
                let policy = policy.ok_or_else(|| {
                    invalid(format!(
                        "prediction state layer {layer} is absent from selected state"
                    ))
                })?;
                Ok(eredu_runtime::prediction_resources::PredictionStateLayer {
                    layer,
                    policy: policy.clone(),
                    processed_token_offset: processed_token_offset
                        .ok_or_else(|| invalid("prediction state prefix is missing"))?,
                })
            })
            .collect::<Result<Vec<_>, eredu_core::resources::ResourceDescriptionError>>()?;
        let (mut execution_topology, mut missing) = match extension.map(|extension| extension.complete_architecture().model()) {
            Some(crate::configuration::SafetensorsModelConfig::QwenHybrid(args)) => (
                Some(crate::qwen::hybrid::topology::prediction(&args.text).map_err(|e| invalid(e.to_string()))?),
                Vec::new(),
            ),
            Some(crate::configuration::SafetensorsModelConfig::DeepSeekV3(_)) => (None, vec!["prediction multi-head latent attention and routed/shared feed-forward invocation topology is unavailable".into()]),
            Some(crate::configuration::SafetensorsModelConfig::DeepSeekV4(_)) => (None, vec!["prediction pooling attention, compressed state and hyper-connection invocation topology is unavailable".into()]),
            Some(crate::configuration::SafetensorsModelConfig::Inkling(_)) => (None, vec!["prediction learned relative attention and auxiliary causal-convolution invocation topology is unavailable".into()]),
            Some(crate::configuration::SafetensorsModelConfig::NemotronH(_)) => (None, vec!["prediction patterned attention/routed-expert and fusion invocation topology is unavailable".into()]),
            None if retained.is_some() => (None, vec!["prediction recurrent/indexed attention, gated residual fusion and routed/shared feed-forward invocation topology is unavailable".into()]),
            _ => (None, vec!["selected prediction module invocation topology is unavailable".into()]),
        };
        if self.execution().parallel_topology().is_some() {
            execution_topology = None;
            missing.push("rank-local prediction invocation topology is unavailable".into());
        }
        if let Some(topology) = execution_topology.as_mut() {
            apply_selected_formats(topology, self.text_realization())?;
        }
        Ok(Some(EmbeddedPredictionTopology {
            execution_topology,
            missing,
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

// Actual materialization tasks override source/config encodings, including load-time
// quantization. Canonical names preserve the target's shared embedding/readout.
fn apply_selected_formats(
    topology: &mut eredu_runtime::execution_topology::TextExecutionTopology,
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
) -> Result<(), eredu_core::resources::ResourceDescriptionError> {
    use eredu_runtime::execution_topology::*;
    let tasks = selected
        .materialization_tasks()
        .iter()
        .chain(selected.auxiliary_materialization_tasks())
        .collect::<Vec<_>>();
    let formats = tasks
        .iter()
        .map(|task| (task.name(), task.executable()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let update = |projection: &mut ProjectionTopology| {
        if let Some(format) = formats.get(projection.parameter.as_str()) {
            projection.format = *format;
        }
    };
    update(&mut topology.output);
    for layer in &mut topology.layers {
        layer.input_projections.iter_mut().for_each(&update);
        match &mut layer.mixer {
            TokenMixerTopology::Attention { projections, .. }
            | TokenMixerTopology::GatedConvolution { projections, .. } => {
                projections.iter_mut().for_each(&update)
            }
            TokenMixerTopology::Unknown { .. } => {}
        }
        match &mut layer.feed_forward {
            FeedForwardTopology::Gated { projections, .. } => {
                projections.iter_mut().for_each(&update)
            }
            FeedForwardTopology::Routed {
                projections,
                router,
                ..
            } => {
                projections.iter_mut().for_each(&update);
                update(router);
            }
            FeedForwardTopology::Unknown { .. } => {}
        }
    }
    let wider = tasks.iter().any(|task| {
        let dtype = task
            .derived_output()
            .map(|o| o.dtype().clone())
            .or_else(|| task.source_encoding().scalar_dtype().map(Into::into));
        matches!(dtype, Some(eredu_checkpoint::recipe::RecipeDtype::F32))
    });
    if selected
        .state()
        .floating_dtype()
        .is_some_and(|dtype| dtype.bytes().get() < 4)
        && wider
    {
        let bytes =
            selected
                .auxiliary_materialization_tasks()
                .iter()
                .try_fold(0u64, |sum, task| {
                    let bytes = task.logical_shape().iter().try_fold(4u64, |bytes, &dim| {
                        bytes
                            .checked_mul(dim as u64)
                            .ok_or_else(|| invalid("prediction parameter promotion overflow"))
                    })?;
                    topology
                        .selected_parameter_promotion_payloads
                        .insert(task.name().to_owned(), bytes);
                    sum.checked_add(bytes)
                        .ok_or_else(|| invalid("prediction parameter promotion overflow"))
                })?;
        topology.selected_parameter_promotion_bytes = Some(bytes);
    }
    Ok(())
}
