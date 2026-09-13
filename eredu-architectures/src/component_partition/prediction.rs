//! Producer placement for separately invoked prepared prediction scopes.
use super::*;
use crate::prediction_extension::PreparedPredictionPlacement;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentExecutionScope, ComponentExecutionScopeKind, ComponentResidualBase},
};

impl ComponentPartitionLayouts {
    pub(crate) fn with_prediction(
        mut self,
        descriptor: &ArchitectureDescriptor,
        prepared: &PreparedPredictionPlacement,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
    ) -> Result<Self, ComponentPartitionError> {
        if prepared.topology().topology() != self.topology {
            return Err(CaptureError::Invalid(
                "prediction placement differs from target topology".into(),
            )
            .into());
        }
        let parameters = prepared.parameters().ok_or_else(|| {
            CaptureError::Unsupported(
                "prepared prediction has no declared parameter placement".into(),
            )
        })?;
        let mut tensors = BTreeMap::new();
        for rank in 0..self.layouts.len() {
            let topology = self.layouts[rank].topology;
            let tensor_rank = topology.tensor_parallel_rank();
            if let std::collections::btree_map::Entry::Vacant(entry) = tensors.entry(tensor_rank) {
                entry.insert(
                    prepared
                        .layout_for_rank(rank)
                        .map_err(ComponentPartitionError::ParameterLayout)?
                        .ok_or_else(|| {
                            CaptureError::Unsupported(
                                "prediction parameter placement is absent".into(),
                            )
                        })?,
                );
            }
            let local = &tensors[&tensor_rank];
            for scope in &descriptor.component_scopes {
                let invocation = match scope.kind {
                    ComponentExecutionScopeKind::Prediction { depth } => {
                        eredu_core::speculative::SpeculativeCaptureScope::Prediction { depth }
                    }
                    ComponentExecutionScopeKind::FusedPrediction => {
                        eredu_core::speculative::SpeculativeCaptureScope::FusedProposal
                    }
                };
                execution.validate_scope(invocation)?;
                // A sequential depth can be entirely static. Parameter ownership
                // below must prove every declared dependency belongs to those
                // roles; removing an actual execution group does not bypass it.
                let expected_groups = match scope.kind {
                    ComponentExecutionScopeKind::Prediction { .. }
                        if scope.execution_groups.is_empty()
                            && !scope.static_parameter_roles.is_empty() =>
                    {
                        0
                    }
                    ComponentExecutionScopeKind::Prediction { .. } => 1,
                    ComponentExecutionScopeKind::FusedPrediction => execution.depth,
                };
                if scope.execution_groups.len() != expected_groups
                    || scope
                        .execution_groups
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != expected_groups
                    || scope
                        .execution_groups
                        .iter()
                        .any(|id| !descriptor.layer_groups.iter().any(|group| &group.id == id))
                {
                    return Err(CaptureError::Invalid(
                        "prediction scope differs from its physical execution groups".into(),
                    )
                    .into());
                }
                let owns = |parameter: &str| {
                    let group = parameters
                        .groups()
                        .iter()
                        .find(|group| {
                            group
                                .members()
                                .iter()
                                .any(|member| member.target() == parameter)
                        })
                        .ok_or_else(|| ComponentPartitionError::MissingWeight(parameter.into()))?;
                    let owned = match group.owner() {
                        eredu_runtime::ParameterGroupOwner::ExecutionUnit { group, .. } => {
                            scope.execution_groups.iter().any(|id| id == group.as_str())
                        }
                        eredu_runtime::ParameterGroupOwner::StaticRole(role) => {
                            scope.static_parameter_roles.contains(role)
                        }
                        eredu_runtime::ParameterGroupOwner::StaticAnyOf(roles) => roles
                            .iter()
                            .any(|role| scope.static_parameter_roles.contains(role)),
                        eredu_runtime::ParameterGroupOwner::StaticUnitConsumers {
                            role,
                            consumers,
                        } => {
                            scope.static_parameter_roles.contains(role)
                                && consumers.iter().any(|(group, unit)| {
                                    scope.execution_groups.iter().any(|id| id == group.as_str())
                                        && descriptor.layer_groups.iter().any(|candidate| {
                                            candidate.id == group.as_str()
                                                && *unit < candidate.physical_layer_count
                                        })
                                })
                        }
                        _ => false,
                    };
                    if !owned {
                        return Err(CaptureError::Invalid(format!(
                            "prediction parameter belongs to another invocation: {parameter}"
                        ))
                        .into());
                    }
                    Ok(true)
                };
                for parameter in std::iter::once(&scope.readout.weight)
                    .chain(scope.readout.bias.iter())
                    .chain(scope.readout.normalization.gain.iter())
                    .chain(scope.readout.normalization.bias.iter())
                {
                    owns(parameter)?;
                }
                let mut projected = ComponentPartitionLayout::from_components(
                    descriptor,
                    &scope.components,
                    local,
                    topology,
                    &owns,
                )?;
                if let ComponentResidualBase::LinearFusion {
                    inputs,
                    weight,
                    bias,
                    ..
                } = &scope.residual_base
                {
                    owns(weight)?;
                    if let Some(bias) = bias {
                        owns(bias)?;
                    }
                    for input in inputs {
                        for parameter in input
                            .normalization
                            .gain
                            .iter()
                            .chain(input.normalization.bias.iter())
                        {
                            owns(parameter)?;
                        }
                        if let eredu_core::component::ComponentFusionSource::TokenEmbedding {
                            weight,
                            normalization,
                            ..
                        } = &input.source
                        {
                            owns(weight)?;
                            if let Some(normalization) = normalization {
                                for parameter in
                                    normalization.gain.iter().chain(normalization.bias.iter())
                                {
                                    owns(parameter)?;
                                }
                            }
                        }
                    }
                }
                if let ComponentResidualBase::ProjectedSum { inputs, .. } = &scope.residual_base {
                    for input in inputs {
                        for parameter in std::iter::once(&input.weight)
                            .chain(input.bias.iter())
                            .chain(input.normalization.gain.iter())
                            .chain(input.normalization.bias.iter())
                        {
                            owns(parameter)?;
                        }
                    }
                }
                for write in &scope.readout.score_writes {
                    for parameter in std::iter::once(&write.weight).chain(write.bias.iter()) {
                        owns(parameter)?;
                    }
                }
                projected.prediction_boundaries(descriptor, scope)?;
                transforms::register(
                    &mut projected.observations,
                    descriptor,
                    local,
                    invocation,
                    &owns,
                )?;
                register_routed_boundaries(
                    &mut projected.observations,
                    descriptor,
                    &scope.routed_components,
                    |node| {
                        let actual = crate::speculative_execution::speculative_capture_scope(
                            descriptor, node,
                        )?;
                        if actual != invocation {
                            return Err(CaptureError::Invalid(
                                "routed prediction input belongs to another invocation".into(),
                            )
                            .into());
                        }
                        Ok(true)
                    },
                )?;
                if let Some(streams) = &scope.readout.stream_residual {
                    streams::register(
                        &mut projected.observations,
                        descriptor,
                        streams,
                        &owns,
                        (true, ObservationHookSite::Unit),
                        (true, ObservationHookSite::Unit),
                    )?;
                }
                projected.routed = routed::placement::prediction_observations(
                    descriptor,
                    &scope.routed_components,
                    local,
                )?;
                self.layouts[rank].merge_prediction(projected)?;
            }
            // Fused context preparation executes on every prediction replica,
            // independently of the proposal's scored decoder invocation. It
            // owns only the declared cache-input seams, never proposal writes.
            for point in &descriptor.observations.points {
                if crate::speculative_execution::speculative_capture_scope(
                    descriptor,
                    &point.node_id,
                )? == eredu_core::speculative::SpeculativeCaptureScope::PredictionContext
                {
                    execution.validate_scope(
                        eredu_core::speculative::SpeculativeCaptureScope::PredictionContext,
                    )?;
                    replicated_observation(
                        &mut self.layouts[rank].observations,
                        descriptor,
                        &point.path,
                        "hidden",
                        true,
                        ObservationHookSite::Unit,
                    )?;
                }
            }
        }
        Ok(self)
    }
}
impl ComponentPartitionLayout {
    fn prediction_boundaries(
        &mut self,
        descriptor: &ArchitectureDescriptor,
        scope: &ComponentExecutionScope,
    ) -> Result<(), ComponentPartitionError> {
        let mut add = |path: &str, axis: &str| {
            replicated_observation(
                &mut self.observations,
                descriptor,
                path,
                axis,
                true,
                ObservationHookSite::Unit,
            )
        };
        // These are actual scope inputs/outputs, not target publication. The
        // typed prediction traversal runs each unit on every execution replica.
        for point in descriptor.observations.points.iter().filter(|point| {
            point.node_id == scope.node_id
                || (point.position == eredu_core::ObservationPosition::ReadOnly
                    && descriptor.nodes.iter().any(|node| {
                        node.id == point.node_id
                            && matches!(
                                node.kind,
                                eredu_core::ArchitectureNodeKind::Embedding
                                    | eredu_core::ArchitectureNodeKind::Normalization
                            )
                            && node.parent.as_deref() == Some(scope.node_id.as_str())
                    }))
        }) {
            if matches!(point.value_type, eredu_core::ObservationValueType::Tensor)
                && point
                    .axes
                    .as_ref()
                    .is_some_and(|axes| axes.iter().any(|axis| axis.name == "hidden"))
            {
                add(&point.path, "hidden")?;
            }
        }
        match &scope.residual_base {
            ComponentResidualBase::Source {
                input,
                expansion,
                output,
                effective_output,
                ..
            } => {
                validate_residual_source(descriptor, input, expansion, output, effective_output)?;
                let original = input.strip_suffix(".effective").ok_or_else(|| {
                    CaptureError::Invalid("residual source lacks its original seam".into())
                })?;
                for path in [original, input, output, effective_output] {
                    add(path, "hidden")?;
                }
            }
            ComponentResidualBase::ProjectedSum {
                inputs,
                output,
                effective_output,
            } => {
                validate_projected_sum(descriptor, inputs, output, effective_output)?;
                for input in inputs {
                    // The equation points to the consumed normalized value;
                    // admission also retains its original intervention seam.
                    let original =
                        input.normalized.strip_suffix(".effective").ok_or_else(|| {
                            CaptureError::Invalid(
                                "projected normalization has no original seam".into(),
                            )
                        })?;
                    add(original, "hidden")?;
                    for path in [
                        &input.input,
                        &input.normalized,
                        &input.projection_input,
                        &input.output,
                        &input.effective_output,
                    ] {
                        add(path, "hidden")?;
                    }
                }
                add(output, "hidden")?;
                add(effective_output, "hidden")?;
            }
            ComponentResidualBase::LinearFusion {
                inputs,
                projection_input,
                output,
                effective_output,
                ..
            } => {
                for input in inputs {
                    let original = input.output.strip_suffix(".effective").ok_or_else(|| {
                        CaptureError::Invalid("fusion normalization has no original seam".into())
                    })?;
                    add(original, "hidden")?;
                    if let eredu_core::component::ComponentFusionSource::Observation { path } =
                        &input.source
                    {
                        add(path, "hidden")?;
                    }
                }
                add(projection_input, "hidden")?;
                add(output, "hidden")?;
                add(effective_output, "hidden")?;
            }
        }
        let readout = &scope.readout;
        for path in [&readout.residual, &readout.normalized]
            .into_iter()
            .chain(readout.projection_input.iter())
        {
            add(path, "hidden")?;
        }
        for path in [&readout.linear_scores, &readout.logits] {
            add(path, "vocabulary")?;
        }
        for write in &readout.score_writes {
            validate_score_write(descriptor, write, &readout.logits)?;
            let original = write.input.strip_suffix(".effective").ok_or_else(|| {
                CaptureError::Invalid("score input lacks its original seam".into())
            })?;
            for path in [original, &write.input, &write.projection_input] {
                add(path, "hidden")?;
            }
            for path in [&write.output, &write.effective_output] {
                add(path, "vocabulary")?;
            }
        }
        for normalization in &readout.block_normalizations {
            add(&normalization.input, "hidden")?;
            add(&normalization.output, "hidden")?;
        }
        for write in &readout.other_writes {
            for path in [&write.output, &write.effective_output]
                .into_iter()
                .chain(write.input.iter())
            {
                add(path, "hidden")?;
            }
        }
        Ok(())
    }
    fn merge_prediction(&mut self, prediction: Self) -> Result<(), ComponentPartitionError> {
        for (key, value) in prediction.groups {
            if self.groups.insert(key.clone(), value).is_some() {
                return Err(ComponentPartitionError::DuplicateIdentity(key));
            }
        }
        for (key, value) in prediction.paths {
            if self.paths.insert(key.clone(), value).is_some() {
                return Err(ComponentPartitionError::DuplicateIdentity(key));
            }
        }
        for (key, value) in prediction.observations {
            if self.observations.insert(key.clone(), value).is_some() {
                return Err(ComponentPartitionError::DuplicateIdentity(key));
            }
        }
        for (key, value) in prediction.routed {
            if self.routed.insert(key.clone(), value).is_some() {
                return Err(ComponentPartitionError::DuplicateIdentity(key));
            }
        }
        Ok(())
    }
}

fn observation_axes(
    descriptor: &ArchitectureDescriptor,
    path: &str,
    position: eredu_core::ObservationPosition,
) -> Result<Vec<eredu_core::TensorAxis>, ComponentPartitionError> {
    let point = descriptor
        .observations
        .get(path)
        .ok_or_else(|| CaptureError::MissingPath(path.into()))?;
    if point.position != position {
        return Err(
            CaptureError::Invalid("component equation observation timing differs".into()).into(),
        );
    }
    point
        .axes
        .clone()
        .ok_or_else(|| CaptureError::Invalid("component equation has no tensor axes".into()).into())
}

fn validate_residual_source(
    descriptor: &ArchitectureDescriptor,
    input: &str,
    expansion: &eredu_core::component::ComponentFusionExpansion,
    output: &str,
    effective_output: &str,
) -> Result<(), ComponentPartitionError> {
    use eredu_core::{
        component::ComponentFusionExpansion, ObservationPosition as P, SymbolicDimension,
    };
    let invalid =
        || CaptureError::Invalid("expanded residual source has inconsistent geometry".into());
    let mut source = observation_axes(descriptor, input, P::AfterIntervention)?;
    let original = input.strip_suffix(".effective").ok_or_else(invalid)?;
    if source.is_empty() || source != observation_axes(descriptor, original, P::BeforeIntervention)?
    {
        return Err(invalid().into());
    }
    if let ComponentFusionExpansion::BroadcastAxis { axis, name, extent } = expansion {
        if *axis > source.len()
            || *extent == 0
            || name.trim().is_empty()
            || source.iter().any(|existing| &existing.name == name)
        {
            return Err(invalid().into());
        }
        source.insert(
            *axis,
            eredu_core::TensorAxis {
                name: name.clone(),
                dimension: SymbolicDimension::Known(*extent),
            },
        );
    }
    if source != observation_axes(descriptor, output, P::BeforeIntervention)?
        || source != observation_axes(descriptor, effective_output, P::AfterIntervention)?
    {
        return Err(invalid().into());
    }
    Ok(())
}

fn validate_score_write(
    descriptor: &ArchitectureDescriptor,
    write: &eredu_core::component::ComponentScoreWrite,
    logits: &str,
) -> Result<(), ComponentPartitionError> {
    use eredu_core::{ObservationPosition as P, SymbolicDimension};
    let invalid = || {
        CaptureError::Invalid("dynamic score write has inconsistent producer or geometry".into())
    };
    let input = observation_axes(descriptor, &write.input, P::AfterIntervention)?;
    let original = write.input.strip_suffix(".effective").ok_or_else(invalid)?;
    let output = observation_axes(descriptor, &write.output, P::BeforeIntervention)?;
    let scores = descriptor
        .observations
        .get(logits)
        .and_then(|point| point.axes.as_ref())
        .ok_or_else(invalid)?;
    if input.is_empty()
        || input.len() != output.len()
        || output.len() != scores.len()
        || input != observation_axes(descriptor, original, P::BeforeIntervention)?
        || input != observation_axes(descriptor, &write.projection_input, P::ReadOnly)?
        || output != observation_axes(descriptor, &write.effective_output, P::AfterIntervention)?
        || input[..input.len() - 1] != output[..output.len() - 1]
        || !matches!(input.last(), Some(eredu_core::TensorAxis { name, dimension: SymbolicDimension::Known(width) }) if name == "hidden" && *width > 0)
        || output.last() != scores.last()
        || !matches!(output.last(), Some(eredu_core::TensorAxis { name, dimension: SymbolicDimension::Known(width) }) if name == "vocabulary" && *width > 0)
        || descriptor
            .observations
            .get(&write.output)
            .is_none_or(|point| point.node_id != write.node_id)
        || write
            .broadcast_axes
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != write.broadcast_axes.len()
    {
        return Err(invalid().into());
    }
    for axis in &write.broadcast_axes {
        if !output.iter().any(|a| &a.name == axis) || axis == "vocabulary" {
            return Err(invalid().into());
        }
    }
    for (source, target) in output.iter().zip(scores) {
        if source.name != target.name
            || if write.broadcast_axes.contains(&source.name) {
                source.dimension != SymbolicDimension::Known(1)
            } else {
                source != target
            }
        {
            return Err(invalid().into());
        }
    }
    Ok(())
}

fn validate_projected_sum(
    descriptor: &ArchitectureDescriptor,
    inputs: &[eredu_core::component::ComponentProjectedFusionInput],
    output: &str,
    effective_output: &str,
) -> Result<(), ComponentPartitionError> {
    use eredu_core::{component::ComponentFusionExpansion, ObservationPosition, SymbolicDimension};
    let invalid = || {
        CaptureError::Invalid("projected residual sum has inconsistent geometry or timing".into())
    };
    let axes = |path: &str, position| {
        let point = descriptor
            .observations
            .get(path)
            .ok_or_else(|| CaptureError::MissingPath(path.into()))?;
        if point.position != position {
            return Err(invalid());
        }
        point.axes.clone().ok_or_else(invalid)
    };
    let target = axes(output, ObservationPosition::BeforeIntervention)?;
    if inputs.is_empty()
        || target != axes(effective_output, ObservationPosition::AfterIntervention)?
    {
        return Err(invalid().into());
    }
    for input in inputs {
        let normalized = axes(&input.normalized, ObservationPosition::AfterIntervention)?;
        let original = input
            .normalized
            .strip_suffix(".effective")
            .ok_or_else(invalid)?;
        if normalized != axes(original, ObservationPosition::BeforeIntervention)?
            || normalized != axes(&input.input, ObservationPosition::ReadOnly)?
        {
            return Err(invalid().into());
        }
        if normalized != axes(&input.projection_input, ObservationPosition::ReadOnly)? {
            return Err(invalid().into());
        }
        let mut projected = axes(&input.output, ObservationPosition::BeforeIntervention)?;
        if projected
            != axes(
                &input.effective_output,
                ObservationPosition::AfterIntervention,
            )?
            || normalized.is_empty()
            || projected.len() != normalized.len()
            || projected[..projected.len() - 1] != normalized[..normalized.len() - 1]
            || [&normalized, &projected].iter().any(|shape| {
                !matches!(shape.last(), Some(eredu_core::TensorAxis {
                    name, dimension: SymbolicDimension::Known(width)
                }) if name == "hidden" && *width > 0)
            })
        {
            return Err(invalid().into());
        }
        if let ComponentFusionExpansion::BroadcastAxis { axis, name, extent } = &input.expansion {
            if *axis > projected.len()
                || *extent == 0
                || name.trim().is_empty()
                || projected.iter().any(|existing| &existing.name == name)
            {
                return Err(invalid().into());
            }
            projected.insert(
                *axis,
                eredu_core::TensorAxis {
                    name: name.clone(),
                    dimension: SymbolicDimension::Known(*extent),
                },
            );
        }
        if projected != target {
            return Err(invalid().into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{ModelConfigurationResolver, ParallelRankTopology, ParallelTopology};
    use std::sync::Arc;

    fn v4_config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"deepseek_v4", "hidden_size":8, "moe_intermediate_size":8,
            "num_hidden_layers":1, "num_attention_heads":2, "num_key_value_heads":1,
            "head_dim":4, "qk_rope_head_dim":2, "q_lora_rank":4,
            "o_groups":2, "o_lora_rank":4, "vocab_size":16,
            "max_position_embeddings":64, "sliding_window":4, "compress_ratios":[0,0],
            "index_n_heads":2, "index_head_dim":4, "index_topk":1,
            "hc_mult":2, "hc_sinkhorn_iters":2, "n_routed_experts":2,
            "n_shared_experts":1, "num_experts_per_tok":1, "num_hash_layers":0,
            "num_nextn_predict_layers":1,
            "scoring_func":"sqrtsoftplus", "topk_method":"noaux_tc",
            "norm_topk_prob":true, "routed_scaling_factor":1.0, "swiglu_limit":4.0
        })
    }

    #[test]
    fn projected_prediction_sum_rejects_wrong_broadcast_and_evidence_geometry() {
        use eredu_core::component::ComponentFusionExpansion;
        let config = v4_config();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let ComponentResidualBase::ProjectedSum {
            inputs,
            output,
            effective_output,
        } = &descriptor.component_scopes[0].residual_base
        else {
            panic!("projected sum")
        };
        validate_projected_sum(&descriptor, inputs, output, effective_output).unwrap();
        for (axis, name, extent) in [
            (1, "stream", 2),
            (2, "stream", 3),
            (2, "hidden", 2),
            (5, "stream", 2),
            (2, "stream", 0),
        ] {
            let mut bad = inputs.clone();
            bad[0].expansion = ComponentFusionExpansion::BroadcastAxis {
                axis,
                name: name.into(),
                extent,
            };
            assert!(validate_projected_sum(&descriptor, &bad, output, effective_output).is_err());
        }
        let mut bad = inputs.clone();
        bad[1].projection_input = inputs[0].projection_input.clone();
        assert!(validate_projected_sum(&descriptor, &bad, output, effective_output).is_err());
        assert!(validate_projected_sum(&descriptor, inputs, effective_output, output).is_err());
        assert!(validate_projected_sum(&descriptor, &[], output, effective_output).is_err());
    }

    #[test]
    fn v4_prediction_placement_preserves_projected_and_stream_geometry_on_all_replicas() {
        let config = v4_config();
        let args = crate::deepseek::parse_v4_config(&config).unwrap();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let parameters =
            Arc::new(crate::deepseek::parallel::v4_parameter_description(&args).unwrap());
        let execution = crate::speculative_execution::SpeculativeActivationExecution {
            depth: 1,
            strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential,
        };
        for (tensor, pipeline, expert) in [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (1, 1, 2),
            (2, 1, 2),
            (2, 2, 2),
        ] {
            let topology = ParallelTopology::new(tensor, pipeline, expert, 1).unwrap();
            let local = Arc::new(
                crate::partitioned_execution::derive_partitioned_local_layout(
                    &parameters,
                    ParallelRankTopology::new(ParallelTopology::new(tensor, 1, 1, 1).unwrap(), 0)
                        .unwrap(),
                )
                .unwrap(),
            );
            let prepared = PreparedPredictionPlacement::from_prepared(
                ParallelRankTopology::new(topology, 0).unwrap(),
                Some(Arc::clone(&parameters)),
                Some(local),
            );
            let layouts = ComponentPartitionLayouts::new(
                topology,
                (0..topology.world_size())
                    .map(|rank| ComponentPartitionLayout {
                        topology: ParallelRankTopology::new(topology, rank).unwrap(),
                        groups: BTreeMap::new(),
                        paths: BTreeMap::new(),
                        observations: BTreeMap::new(),
                        routed: BTreeMap::new(),
                    })
                    .collect(),
            )
            .unwrap()
            .with_prediction(&descriptor, &prepared, &execution)
            .unwrap();
            let scope = &descriptor.component_scopes[0];
            let ComponentResidualBase::ProjectedSum {
                inputs,
                output,
                effective_output,
            } = &scope.residual_base
            else {
                panic!("sum")
            };
            let streams = scope.readout.stream_residual.as_ref().unwrap();
            let mut replicated = vec![
                output.as_str(),
                effective_output.as_str(),
                streams.head.input.as_str(),
                streams.head.coefficients.as_str(),
            ];
            for input in inputs {
                replicated.extend([
                    input.input.as_str(),
                    input.normalized.as_str(),
                    input.projection_input.as_str(),
                    input.output.as_str(),
                    input.effective_output.as_str(),
                ]);
                replicated.push(input.normalized.strip_suffix(".effective").unwrap());
            }
            for cycle in &streams.cycles {
                replicated.extend([
                    cycle.input.as_str(),
                    cycle.output.as_str(),
                    cycle.collapsed.as_str(),
                    cycle.pre.as_str(),
                    cycle.post.as_str(),
                    cycle.combination.as_str(),
                ]);
            }
            for component in &scope.routed_components {
                replicated.extend(component.input.iter().map(String::as_str));
            }
            for path in replicated {
                assert_eq!(
                    layouts.capture_hook_members(path).unwrap().len(),
                    topology.world_size(),
                    "{path}"
                );
                for rank in 0..topology.world_size() {
                    let placement = layouts.rank(rank).unwrap().observation(path).unwrap();
                    assert_eq!(
                        placement.combination,
                        PartitionCaptureCombination::Disjoint,
                        "{path}"
                    );
                }
            }
            for component in &scope.components {
                for rank in 0..topology.world_size() {
                    let local = layouts.rank(rank).unwrap();
                    let r = local.topology.tensor_parallel_rank();
                    assert_eq!(
                        local.groups[&component.id]
                            .coordinates()
                            .unwrap()
                            .contiguous_range(),
                        Some(r * component.count / tensor..(r + 1) * component.count / tensor)
                    );
                }
            }
        }
    }

    #[test]
    fn fused_prediction_placement_retains_multiblock_and_context_producers() {
        use eredu_core::component::ComponentFusionExpansion;
        let mut config = v4_config();
        config["num_hidden_layers"] = 2.into();
        config["num_nextn_predict_layers"] = 2.into();
        config["compress_ratios"] = serde_json::json!([0, 4, 0, 0]);
        config["dspark_block_size"] = 3.into();
        config["dspark_noise_token_id"] = 0.into();
        config["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
        config["dspark_markov_rank"] = 4.into();
        let args = crate::deepseek::parse_v4_config(&config).unwrap();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let parameters =
            Arc::new(crate::deepseek::parallel::v4_parameter_description(&args).unwrap());
        let execution = crate::speculative_execution::SpeculativeActivationExecution {
            depth: 2,
            strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedFused,
        };
        let scope = &descriptor.component_scopes[0];
        let ComponentResidualBase::Source {
            input,
            expansion,
            output,
            effective_output,
            ..
        } = &scope.residual_base
        else {
            panic!("embedding source")
        };
        validate_residual_source(&descriptor, input, expansion, output, effective_output).unwrap();
        assert!(validate_residual_source(
            &descriptor,
            input,
            &ComponentFusionExpansion::BroadcastAxis {
                axis: 2,
                name: "stream".into(),
                extent: 3
            },
            output,
            effective_output
        )
        .is_err());
        assert!(
            validate_residual_source(&descriptor, input, expansion, effective_output, output)
                .is_err()
        );
        let write = &scope.readout.score_writes[0];
        validate_score_write(&descriptor, write, &scope.readout.logits).unwrap();
        for invalid in 0..4 {
            let mut other = write.clone();
            match invalid {
                0 => other.broadcast_axes.clear(),
                1 => other.broadcast_axes.push("sequence".into()),
                2 => other.broadcast_axes = vec!["vocabulary".into()],
                _ => other.node_id = "output".into(),
            }
            assert!(validate_score_write(&descriptor, &other, &scope.readout.logits).is_err());
        }
        for (tensor, pipeline, expert) in [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
        ] {
            let topology = ParallelTopology::new(tensor, pipeline, expert, 1).unwrap();
            let local = Arc::new(
                crate::partitioned_execution::derive_partitioned_local_layout(
                    &parameters,
                    ParallelRankTopology::new(ParallelTopology::new(tensor, 1, 1, 1).unwrap(), 0)
                        .unwrap(),
                )
                .unwrap(),
            );
            let prepared = PreparedPredictionPlacement::from_prepared(
                ParallelRankTopology::new(topology, 0).unwrap(),
                Some(Arc::clone(&parameters)),
                Some(local),
            );
            let empty = || {
                ComponentPartitionLayouts::new(
                    topology,
                    (0..topology.world_size())
                        .map(|rank| ComponentPartitionLayout {
                            topology: ParallelRankTopology::new(topology, rank).unwrap(),
                            groups: BTreeMap::new(),
                            paths: BTreeMap::new(),
                            observations: BTreeMap::new(),
                            routed: BTreeMap::new(),
                        })
                        .collect(),
                )
                .unwrap()
            };
            let layouts = empty()
                .with_prediction(&descriptor, &prepared, &execution)
                .unwrap();
            let mut paths = vec![
                input.as_str(),
                output,
                effective_output,
                &write.input,
                &write.projection_input,
                &write.output,
                &write.effective_output,
                &scope.readout.logits,
            ];
            paths.extend(
                descriptor
                    .observations
                    .points
                    .iter()
                    .filter(|point| {
                        crate::speculative_execution::speculative_capture_scope(
                            &descriptor,
                            &point.node_id,
                        )
                        .unwrap()
                            == eredu_core::speculative::SpeculativeCaptureScope::PredictionContext
                    })
                    .map(|point| point.path.as_str()),
            );
            for path in paths {
                assert_eq!(
                    layouts.capture_hook_members(path).unwrap().len(),
                    topology.world_size(),
                    "{path}"
                );
                for rank in 0..topology.world_size() {
                    assert!(
                        layouts.rank(rank).unwrap().observation(path).is_some(),
                        "{path}"
                    );
                }
            }
            for component in &scope.components {
                for rank in 0..topology.world_size() {
                    let local = layouts.rank(rank).unwrap();
                    let r = local.topology.tensor_parallel_rank();
                    assert_eq!(
                        local.groups[&component.id]
                            .coordinates()
                            .unwrap()
                            .contiguous_range(),
                        Some(r * component.count / tensor..(r + 1) * component.count / tensor)
                    );
                }
            }
            for invalid in 0..3 {
                let mut other = descriptor.clone();
                match invalid {
                    0 => {
                        other.component_scopes[0].execution_groups.pop();
                    }
                    1 => {
                        other.component_scopes[0].execution_groups[1] = "mtp.0".into();
                    }
                    _ => {
                        other.component_scopes[0].kind =
                            ComponentExecutionScopeKind::Prediction { depth: 0 };
                    }
                }
                assert!(empty()
                    .with_prediction(&other, &prepared, &execution)
                    .is_err());
            }
        }
    }

    #[test]
    fn prediction_fusion_input_observations_keep_their_invocation_ownership() {
        let config = serde_json::json!({
            "model_type":"qwen3_5_text", "vocab_size":16, "hidden_size":8,
            "num_hidden_layers":2, "intermediate_size":12, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":2, "max_position_embeddings":64,
            "linear_conv_kernel_dim":3, "linear_key_head_dim":2, "linear_value_head_dim":2,
            "linear_num_key_heads":2, "linear_num_value_heads":2,
            "layer_types":["linear_attention","full_attention"], "mtp_num_hidden_layers":2
        });
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        for (depth, scope) in descriptor.component_scopes.iter().enumerate() {
            let topology =
                ParallelRankTopology::new(ParallelTopology::new(2, 2, 1, 1).unwrap(), 3).unwrap();
            let mut layout = ComponentPartitionLayout {
                topology,
                groups: BTreeMap::new(),
                paths: BTreeMap::new(),
                observations: BTreeMap::new(),
                routed: BTreeMap::new(),
            };
            layout.prediction_boundaries(&descriptor, scope).unwrap();
            for suffix in [
                "hidden",
                "embedding",
                "hidden.normalized",
                "hidden.normalized.effective",
                "embedding.normalized",
                "embedding.normalized.effective",
                "fusion.input",
                "fusion.output",
                "fusion.output.effective",
            ] {
                let path = format!("mtp.layers.{depth}.prediction.{suffix}");
                assert_eq!(
                    layout
                        .observation(&path)
                        .unwrap()
                        .coordinates()
                        .unwrap()
                        .contiguous_range(),
                    Some(0..if suffix == "fusion.input" { 16 } else { 8 })
                );
                let other = format!("mtp.layers.{}.prediction.{suffix}", 1 - depth);
                assert!(
                    layout.observation(&other).is_none(),
                    "other invocation is not admitted"
                );
            }
        }
    }

    #[test]
    fn prediction_components_use_tensor_shards_and_all_execution_replicas() {
        let config = serde_json::json!({
            "model_type": "deepseek_v3", "hidden_size": 8, "intermediate_size": 16,
            "moe_intermediate_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
            "vocab_size": 16, "max_position_embeddings": 64, "kv_lora_rank": 4,
            "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 2,
            "first_k_dense_replace": 1, "n_routed_experts": 4, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
            "num_nextn_predict_layers": 2, "tie_word_embeddings": false
        });
        let args = crate::deepseek::parse_v3_config(&config).unwrap();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let parameters =
            Arc::new(crate::deepseek::parallel::v3_parameter_description(&args).unwrap());
        let execution = crate::speculative_execution::SpeculativeActivationExecution {
            depth: 2,
            strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential,
        };
        for (tensor, pipeline, expert) in [(1, 2, 2), (2, 1, 2), (2, 2, 2)] {
            let topology = ParallelTopology::new(tensor, pipeline, expert, 1).unwrap();
            let local = Arc::new(
                crate::partitioned_execution::derive_partitioned_local_layout(
                    &parameters,
                    ParallelRankTopology::new(ParallelTopology::new(tensor, 1, 1, 1).unwrap(), 0)
                        .unwrap(),
                )
                .unwrap(),
            );
            let prepared = PreparedPredictionPlacement::from_prepared(
                ParallelRankTopology::new(topology, 0).unwrap(),
                Some(Arc::clone(&parameters)),
                Some(Arc::clone(&local)),
            );
            assert!(Arc::ptr_eq(
                &prepared.layout_for_rank(0).unwrap().unwrap(),
                &local
            ));
            assert!(prepared.layout_for_rank(topology.world_size()).is_err());
            let empty = || {
                ComponentPartitionLayouts::new(
                    topology,
                    (0..topology.world_size())
                        .map(|rank| ComponentPartitionLayout {
                            topology: ParallelRankTopology::new(topology, rank).unwrap(),
                            groups: BTreeMap::new(),
                            paths: BTreeMap::new(),
                            observations: BTreeMap::new(),
                            routed: BTreeMap::new(),
                        })
                        .collect(),
                )
                .unwrap()
            };
            let layouts = empty()
                .with_prediction(&descriptor, &prepared, &execution)
                .unwrap();
            for scope in &descriptor.component_scopes {
                for component in &scope.components {
                    assert_eq!(
                        layouts.capture_hook_members(&component.activation).unwrap(),
                        (0..topology.world_size()).collect::<Vec<_>>()
                    );
                    for rank in 0..topology.world_size() {
                        let projection = layouts.rank(rank).unwrap();
                        let index = projection.topology.tensor_parallel_rank();
                        let coordinates = projection.groups[&component.id].coordinates().unwrap();
                        assert_eq!(coordinates.global_count(), component.count);
                        assert_eq!(
                            coordinates.contiguous_range(),
                            Some(
                                index * component.count / tensor
                                    ..(index + 1) * component.count / tensor
                            )
                        );
                        assert_eq!(
                            projection
                                .observation(&component.input)
                                .unwrap()
                                .coordinates()
                                .unwrap()
                                .global_count(),
                            8
                        );
                    }
                }
                for component in &scope.routed_components {
                    for rank in 0..topology.world_size() {
                        let projection = layouts.rank(rank).unwrap();
                        let ownership = projection
                            .routed_observation(&component.activation)
                            .unwrap()
                            .ownership()
                            .unwrap();
                        assert_eq!(
                            ownership.source_peers, 1,
                            "prediction resident banks do not reuse target EP source exchange"
                        );
                        assert_eq!(ownership.source_peer, None);
                        let experts = ownership.coordinates.experts();
                        assert_eq!(
                            (0..experts.local_count())
                                .map(|index| experts.local_to_global(index).unwrap())
                                .collect::<Vec<_>>(),
                            vec![0, 1, 2, 3]
                        );
                        let index = projection.topology.tensor_parallel_rank();
                        assert_eq!(
                            ownership.coordinates.units().contiguous_range(),
                            Some(index * 8 / tensor..(index + 1) * 8 / tensor)
                        );
                    }
                }
                for path in [&scope.readout.linear_scores, &scope.readout.logits] {
                    assert_eq!(
                        layouts.capture_hook_members(path).unwrap().len(),
                        topology.world_size()
                    );
                    for rank in 0..topology.world_size() {
                        assert_eq!(
                            layouts
                                .rank(rank)
                                .unwrap()
                                .observation(path)
                                .unwrap()
                                .coordinates()
                                .unwrap()
                                .contiguous_range(),
                            Some(0..16)
                        );
                    }
                }
                let ComponentResidualBase::LinearFusion {
                    inputs,
                    projection_input,
                    output,
                    ..
                } = &scope.residual_base
                else {
                    panic!("V3 fixture declares concatenated linear fusion")
                };
                for input in inputs {
                    let original = input.output.strip_suffix(".effective").unwrap();
                    assert_eq!(
                        layouts.capture_hook_members(original),
                        layouts.capture_hook_members(&input.output)
                    );
                    assert_eq!(
                        layouts.capture_hook_members(original).unwrap().len(),
                        topology.world_size()
                    );
                }
                assert_eq!(
                    layouts
                        .rank(0)
                        .unwrap()
                        .observation(projection_input)
                        .unwrap()
                        .coordinates()
                        .unwrap()
                        .global_count(),
                    16
                );
                assert_eq!(
                    layouts
                        .rank(0)
                        .unwrap()
                        .observation(output)
                        .unwrap()
                        .coordinates()
                        .unwrap()
                        .global_count(),
                    8
                );
            }
            assert!(empty()
                .with_prediction(
                    &descriptor,
                    &prepared,
                    &crate::speculative_execution::SpeculativeActivationExecution {
                        depth: 1,
                        strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential
                    }
                )
                .is_err());
            // A pinned module may serve several prediction invocations. Its
            // physical placement is unchanged, but only declared consumers may use it.
            let ComponentResidualBase::LinearFusion { weight, .. } =
                &descriptor.component_scopes[0].residual_base
            else {
                unreachable!()
            };
            for valid in [true, false] {
                let owner = eredu_runtime::ParameterGroupOwner::static_unit_consumers(
                    "shared-prediction",
                    if valid {
                        vec![
                            (eredu_runtime::ExecutionGroupId::new("mtp.0").unwrap(), 0),
                            (eredu_runtime::ExecutionGroupId::new("mtp.1").unwrap(), 0),
                        ]
                    } else {
                        vec![(eredu_runtime::ExecutionGroupId::new("mtp.1").unwrap(), 0)]
                    },
                );
                let groups = parameters
                    .groups()
                    .iter()
                    .map(|group| {
                        if group
                            .members()
                            .iter()
                            .any(|member| member.target() == weight)
                        {
                            eredu_runtime::OwnedParameterGroupSpec::new(
                                owner.clone(),
                                group.group().clone(),
                            )
                        } else {
                            group.clone()
                        }
                    })
                    .collect::<Vec<_>>();
                let changed = Arc::new(
                    eredu_runtime::ArchitectureParameterDescription::new(
                        parameters.graph(),
                        parameters.unit_layout(),
                        parameters
                            .groups()
                            .iter()
                            .map(|group| group.group().clone()),
                        groups,
                    )
                    .unwrap(),
                );
                let placement = PreparedPredictionPlacement::from_prepared(
                    ParallelRankTopology::new(topology, 0).unwrap(),
                    Some(changed),
                    Some(Arc::clone(&local)),
                );
                let mut shared = descriptor.clone();
                for scope in &mut shared.component_scopes {
                    scope
                        .static_parameter_roles
                        .push("shared-prediction".into());
                }
                assert_eq!(
                    empty()
                        .with_prediction(&shared, &placement, &execution)
                        .is_ok(),
                    valid
                );
                shared.component_scopes[0]
                    .static_parameter_roles
                    .retain(|role| role != "shared-prediction");
                assert!(empty()
                    .with_prediction(&shared, &placement, &execution)
                    .is_err());
            }
            let mut wrong_fusion = descriptor.clone();
            let ComponentResidualBase::LinearFusion { weight, .. } =
                &mut wrong_fusion.component_scopes[0].residual_base
            else {
                unreachable!()
            };
            *weight = "model.norm.weight".into();
            assert!(empty()
                .with_prediction(&wrong_fusion, &prepared, &execution)
                .is_err());
            let mut stale = descriptor.clone();
            stale.component_scopes[0].execution_groups = vec!["target".into()];
            assert!(empty()
                .with_prediction(&stale, &prepared, &execution)
                .is_err());
        }
    }
}
