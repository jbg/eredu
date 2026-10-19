//! Producer placement for separately invoked prepared prediction scopes.
use super::*;
use crate::prediction_extension::PreparedPredictionPlacement;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentExecutionScope, ComponentExecutionScopeKind, ComponentResidualBase},
};

mod qwen4;

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
            if let Some(spec) = prepared.qwen4_spec() {
                qwen4::register(&mut self.layouts[rank], descriptor, spec, local, execution)?;
            }
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
                residual::register(&mut projected.observations, descriptor, &scope.readout.residual_cycles, local, &owns)?;
                transforms::register(
                    &mut projected.observations,
                    descriptor,
                    local,
                    invocation,
                    &owns,
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
