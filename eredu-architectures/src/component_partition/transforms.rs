//! Capture coordinates follow declared equations and retained parameter placement.
use super::*;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentTensorTransform, ComponentTensorTransformEquation as Equation},
    speculative::SpeculativeCaptureScope,
    ObservationPosition, SymbolicDimension,
};

pub(super) fn register(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    layout: &LocalModelLayout,
    scope: SpeculativeCaptureScope,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    let mut pending = descriptor
        .component_transforms
        .iter()
        .filter_map(|transform| {
            match crate::speculative_execution::speculative_capture_scope(
                descriptor,
                &transform.node_id,
            ) {
                Ok(actual) if actual == scope => Some(Ok(transform)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A whole residual write can establish the owner of a convolution or
    // post-normalization before a scalar group does (for example a reduced
    // routed/shared sum). Both preserve axes and invocation ownership, so their
    // declared input has the same placement. register_one still proves complete
    // normalization axes and exact gain/bias or kernel geometry and ownership.
    for transform in &pending {
        if observations.contains_key(&transform.input)
            || !matches!(
                transform.equation,
                Equation::CausalDepthwiseConvolution { .. } | Equation::Normalization { .. }
            )
        {
            continue;
        }
        let Some(output) = observations.get(&transform.output).cloned() else {
            continue;
        };
        let invalid = || {
            CaptureError::Invalid(format!(
                "invalid shape-preserving input boundary: {}",
                transform.id
            ))
        };
        let input = descriptor
            .observations
            .get(&transform.input)
            .ok_or_else(&invalid)?;
        let point = descriptor
            .observations
            .get(&transform.output)
            .ok_or_else(&invalid)?;
        if input.axes.is_none() || input.axes != point.axes {
            return Err(invalid().into());
        }
        match input.position {
            ObservationPosition::ReadOnly => {}
            ObservationPosition::AfterIntervention => {
                let original = transform
                    .input
                    .strip_suffix(".effective")
                    .ok_or_else(&invalid)?;
                let point = descriptor.observations.get(original).ok_or_else(&invalid)?;
                if point.axes != input.axes
                    || point.position != ObservationPosition::BeforeIntervention
                {
                    return Err(invalid().into());
                }
                insert_observation(observations, original, output.clone())?;
            }
            _ => return Err(invalid().into()),
        }
        insert_observation(observations, &transform.input, output)?;
    }
    // Equations need not be listed in dependency order. Only already established
    // inputs authorize propagation; cycles or missing sources cannot imply zero.
    while !pending.is_empty() {
        let mut remaining = Vec::new();
        let before = pending.len();
        for transform in pending {
            let Some(input) = observations.get(&transform.input).cloned() else {
                remaining.push(transform);
                continue;
            };
            register_one(observations, descriptor, layout, transform, input, owns)?;
        }
        if remaining.len() == before {
            return Err(CaptureError::Invalid(format!(
                "transform has no established input placement: {}",
                remaining[0].input
            ))
            .into());
        }
        pending = remaining;
    }
    Ok(())
}

fn register_one(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    layout: &LocalModelLayout,
    transform: &ComponentTensorTransform,
    input: PartitionedObservation,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    let invalid =
        || CaptureError::Invalid(format!("invalid transform placement: {}", transform.id));
    let axes = |path: &str| {
        descriptor
            .observations
            .get(path)
            .and_then(|point| point.axes.as_deref())
            .ok_or_else(&invalid)
    };
    let input_axes = axes(&transform.input)?;
    let output_axes = axes(&transform.output)?;
    let local = input.coordinates.is_some();
    let parameter = |name: &str| -> Result<Option<&LocalTensorLayout>, ComponentPartitionError> {
        if owns(name)? != local {
            return Err(invalid().into());
        }
        if local {
            Ok(Some(layout.tensor(name).ok_or_else(|| {
                ComponentPartitionError::MissingWeight(name.into())
            })?))
        } else {
            Ok(None)
        }
    };
    let replicated_parameter =
        |name: &str, shape: &[usize]| -> Result<(), ComponentPartitionError> {
            if let Some(tensor) = parameter(name)? {
                if tensor.global_shape() != shape
                    || tensor.local_shape() != shape
                    || !matches!(
                        tensor.placement(),
                        TensorPlacement::Replicated | TensorPlacement::Local
                    )
                    || !tensor.additional_placements().is_empty()
                {
                    return Err(invalid().into());
                }
            }
            Ok(())
        };
    let mut output = input.clone();
    match &transform.equation {
        Equation::ConstantScale { .. } => {
            if input_axes != output_axes {
                return Err(invalid().into());
            }
        }
        Equation::LearnedScale { scale } => {
            if input_axes != output_axes {
                return Err(invalid().into());
            }
            replicated_parameter(&scale.parameter, &[1])?;
            // The scalar equation's read-only scalar records are supplied by
            // that same architecture node, independently of the hidden width.
            for point in descriptor.observations.points.iter().filter(|point| {
                point.node_id == transform.node_id
                    && point.position == ObservationPosition::ReadOnly
            }) {
                if let Some([axis]) = point.axes.as_deref() {
                    if axis.dimension == SymbolicDimension::Known(1) {
                        replicated_observation(
                            observations,
                            descriptor,
                            &point.path,
                            &axis.name,
                            local,
                            input.site,
                        )?;
                    }
                }
            }
        }
        Equation::Normalization { normalization } => {
            if input_axes != output_axes {
                return Err(invalid().into());
            }
            let width = input_axes
                .last()
                .and_then(|axis| match axis.dimension {
                    SymbolicDimension::Known(width) => Some(width),
                    _ => None,
                })
                .ok_or_else(&invalid)?;
            // A normalization of the complete last axis requires that complete
            // axis on each invocation. Replication is independent of PP ownership.
            if input
                .coordinates
                .as_ref()
                .is_some_and(|coordinates| coordinates.contiguous_range() != Some(0..width))
            {
                return Err(invalid().into());
            }
            for name in normalization.gain.iter().chain(normalization.bias.iter()) {
                replicated_parameter(name, &[width])?;
            }
        }
        Equation::CausalDepthwiseConvolution {
            kernel,
            channels,
            taps,
            ..
        } => {
            if *channels == 0
                || *taps == 0
                || input_axes != output_axes
                || input_axes.len() != 3
                || input_axes[2].dimension != SymbolicDimension::Known(*channels)
            {
                return Err(invalid().into());
            }
            if let Some(tensor) = parameter(&kernel.parameter)? {
                if tensor.global_shape() != [*channels, 1, *taps]
                    || derive_tensor_axis_coordinates(&kernel.parameter, *channels, tensor, 0)?
                        != *input.coordinates.as_ref().expect("local input")
                {
                    return Err(invalid().into());
                }
            }
        }
        Equation::SharedGroupedProjection {
            weight,
            groups,
            input_width,
            output_width,
        } => {
            let width = groups
                .checked_mul(*input_width)
                .filter(|width| *width > 0)
                .ok_or_else(&invalid)?;
            if *output_width == 0
                || input_axes.len() != 3
                || output_axes.len() != 4
                || input_axes[..2] != output_axes[..2]
                || input_axes[2].dimension != SymbolicDimension::Known(width)
                || output_axes[2].dimension != SymbolicDimension::Known(*groups)
                || output_axes[3].dimension != SymbolicDimension::Known(*output_width)
            {
                return Err(invalid().into());
            }
            replicated_parameter(&weight.parameter, &[*input_width, *output_width])?;
            output.axis = output_axes[2].name.clone();
            output.coordinates = input
                .coordinates
                .as_ref()
                .map(|coordinates| complete_groups(coordinates, *groups, *input_width))
                .transpose()?;
        }
    }
    let point = descriptor
        .observations
        .get(&transform.output)
        .ok_or_else(&invalid)?;
    if point.position
        != if transform.effective_output.is_some() {
            ObservationPosition::BeforeIntervention
        } else {
            ObservationPosition::ReadOnly
        }
    {
        return Err(invalid().into());
    }
    insert_observation(observations, &transform.output, output.clone())?;
    if let Some(effective) = &transform.effective_output {
        if axes(effective)? != output_axes
            || descriptor
                .observations
                .get(effective)
                .is_none_or(|point| point.position != ObservationPosition::AfterIntervention)
        {
            return Err(invalid().into());
        }
        insert_observation(observations, effective, output)?;
    }
    Ok(())
}

fn complete_groups(
    coordinates: &ComponentCoordinateMap,
    groups: usize,
    width: usize,
) -> Result<ComponentCoordinateMap, ComponentPartitionError> {
    let invalid =
        || CaptureError::Invalid("projection placement splits a shared input group".into());
    if width == 0
        || groups.checked_mul(width) != Some(coordinates.global_count())
        || !coordinates.local_count().is_multiple_of(width)
    {
        return Err(invalid().into());
    }
    if let Some(range) = coordinates.contiguous_range() {
        if !range.start.is_multiple_of(width) || !range.end.is_multiple_of(width) {
            return Err(invalid().into());
        }
        return ComponentCoordinateMap::range(groups, range.start / width..range.end / width)
            .map_err(Into::into);
    }
    let mut selected = Vec::with_capacity(coordinates.local_count() / width);
    for local in (0..coordinates.local_count()).step_by(width) {
        let first = coordinates.local_to_global(local).ok_or_else(&invalid)?;
        if !first.is_multiple_of(width)
            || (0..width)
                .any(|offset| coordinates.local_to_global(local + offset) != Some(first + offset))
        {
            return Err(invalid().into());
        }
        selected.push(first / width);
    }
    ComponentCoordinateMap::indices(groups, selected).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{ModelConfigurationResolver, ParallelRankTopology, ParallelTopology};
    use eredu_runtime::{
        ExecutionGraph, ExecutionUnitLayout, OwnedParameterGroupSpec, ParameterGroupOwner,
    };
    use std::sync::Arc;

    fn fixture() -> (ArchitectureDescriptor, ArchitectureParameterDescription) {
        let config = serde_json::json!({
            "model_type":"inkling_mm_model", "image_token_id":5,
            "text_config":{"hidden_size":8, "num_hidden_layers":2, "vocab_size":16,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":2,
                "sliding_window_size":4, "layer_types":["full_attention","sliding_attention"],
                "mlp_layer_types":["dense","dense"], "sconv_kernel_size":3, "d_rel":2,
                "rel_extent":8, "intermediate_size":12, "dense_intermediate_size":12,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,"moe_intermediate_size":6},
            "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true}
        });
        let args =
            crate::inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let graph = ExecutionGraph::chain([crate::inkling::TEXT_EXECUTION_GROUP]).unwrap();
        let units = ExecutionUnitLayout::new(&graph, [2]).unwrap();
        let mut owned = crate::inkling::static_parameter_groups(&args)
            .unwrap()
            .into_iter()
            .zip(["embedding", "embedding_norm", "norm", "output"])
            .map(|(group, role)| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role(role), group)
            })
            .collect::<Vec<_>>();
        owned.extend(
            crate::inkling::mtp_parameter_groups(&args)
                .unwrap()
                .into_iter()
                .map(|group| {
                    OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("mtp"), group)
                }),
        );
        for layer in 0..2 {
            owned.extend(
                crate::inkling::layer_parameter_groups(&args, layer)
                    .unwrap()
                    .into_iter()
                    .map(|group| {
                        OwnedParameterGroupSpec::new(
                            ParameterGroupOwner::execution_unit(
                                units.group_id(0).unwrap().clone(),
                                layer,
                            ),
                            group,
                        )
                    }),
            );
        }
        let expected = owned
            .iter()
            .map(|group| group.group().clone())
            .collect::<Vec<_>>();
        let parameters =
            ArchitectureParameterDescription::new(&graph, &units, expected, owned).unwrap();
        (descriptor, parameters)
    }

    #[test]
    fn target_transforms_follow_real_tensor_shards_and_pipeline_owners() {
        let (descriptor, parameters) = fixture();
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        for rank in 0..topology.world_size() {
            let rank = ParallelRankTopology::new(topology, rank).unwrap();
            let layout =
                crate::partitioned_execution::derive_partitioned_local_layout(&parameters, rank)
                    .unwrap();
            let owns = |name: &str| {
                let owner = parameters
                    .groups()
                    .iter()
                    .find(|group| group.members().iter().any(|member| member.target() == name))
                    .unwrap()
                    .owner();
                Ok(match owner {
                    ParameterGroupOwner::ExecutionUnit { global_unit, .. } => {
                        *global_unit == rank.pipeline_parallel_rank()
                    }
                    _ => rank.pipeline_parallel_rank() == 1,
                })
            };
            let mut projected = ComponentPartitionLayout::from_components(
                &descriptor,
                &descriptor.components,
                &layout,
                rank,
                &owns,
            )
            .unwrap();
            let readout = descriptor.component_readout.as_ref().unwrap();
            replicated_observation(
                &mut projected.observations,
                &descriptor,
                &readout.normalized,
                "hidden",
                rank.pipeline_parallel_rank() == 1,
                ObservationHookSite::Readout,
            )
            .unwrap();
            // Reverse declaration order to prove dependencies, not vector order,
            // select the scalar-to-convolution propagation sequence.
            let mut reversed = descriptor.clone();
            reversed.component_transforms.reverse();
            register(
                &mut projected.observations,
                &reversed,
                &layout,
                SpeculativeCaptureScope::Target,
                &owns,
            )
            .unwrap();
            for layer in 0..2 {
                let local = layer == rank.pipeline_parallel_rank();
                for (suffix, axis, width) in [
                    ("attention.key.convolved", "hidden", 4),
                    ("attention.value.convolved", "hidden", 4),
                    ("attention.relative.profiles", "head", 4),
                ] {
                    let value = &projected.observations[&format!("model.layers.{layer}.{suffix}")];
                    assert_eq!(value.axis(), axis);
                    assert_eq!(
                        value.coordinates().map(|map| map.contiguous_range()),
                        local.then_some(Some(
                            rank.tensor_parallel_rank() * width / 2
                                ..(rank.tensor_parallel_rank() + 1) * width / 2
                        ))
                    );
                }
                for branch in ["attention", "feed_forward"] {
                    let path = format!("model.layers.{layer}.{branch}.contribution");
                    let original = &projected.observations[&path];
                    assert_eq!(
                        original,
                        &projected.observations[&format!("{path}.effective")]
                    );
                    assert_eq!(
                        original.coordinates().map(|map| map.contiguous_range()),
                        local.then_some(Some(0..8))
                    );
                }
                assert_eq!(
                    projected.observations
                        [&format!("model.layers.{layer}.feed_forward.global_scale")]
                        .coordinates()
                        .map(|map| map.contiguous_range()),
                    local.then_some(Some(0..1))
                );
            }
            assert!(projected
                .observations
                .keys()
                .all(|path| !path.starts_with("model.mtp.")));
            let scaled = &projected.observations["readout.scaled"];
            assert_eq!(scaled.site(), ObservationHookSite::Readout);
            assert_eq!(scaled.exports(), rank.pipeline_parallel_rank() == 1);
        }
    }

    #[test]
    fn static_prediction_transforms_and_sources_are_complete_on_every_replica() {
        let (descriptor, parameters) = fixture();
        let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
        let local = Arc::new(
            crate::partitioned_execution::derive_partitioned_local_layout(
                &parameters,
                ParallelRankTopology::new(topology, 0).unwrap(),
            )
            .unwrap(),
        );
        let prepared = crate::prediction_extension::PreparedPredictionPlacement::from_prepared(
            ParallelRankTopology::new(topology, 0).unwrap(),
            Some(Arc::new(parameters)),
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
        let execution = crate::speculative_execution::SpeculativeActivationExecution {
            depth: 2,
            strategy: eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential,
        };
        let layouts = empty()
            .with_prediction(&descriptor, &prepared, &execution)
            .unwrap();
        for depth in 0..2 {
            for (suffix, width) in [
                ("prediction.embedding", 8),
                ("prediction.hidden.first_normalized", 8),
                ("prediction.hidden.first_normalized.effective", 8),
                ("prediction.readout.scaled", 8),
                ("transformer_block.attention.relative.profiles", 4),
                ("transformer_block.feed_forward.global_scale", 1),
            ] {
                let path = format!("model.mtp.layers.{depth}.{suffix}");
                assert_eq!(
                    layouts.capture_hook_members(&path).unwrap().len(),
                    topology.world_size(),
                    "{path}"
                );
                for rank in 0..topology.world_size() {
                    assert_eq!(
                        layouts
                            .rank(rank)
                            .unwrap()
                            .observation(&path)
                            .unwrap()
                            .coordinates()
                            .unwrap()
                            .contiguous_range(),
                        Some(0..width),
                        "{path}"
                    );
                }
            }
        }
        let mut stale = descriptor.clone();
        stale.component_scopes[0]
            .static_parameter_roles
            .retain(|role| role != "mtp");
        assert!(empty()
            .with_prediction(&stale, &prepared, &execution)
            .is_err());
    }

    #[test]
    fn grouped_projection_rejects_partial_groups_and_preserves_permutations() {
        assert_eq!(
            complete_groups(
                &ComponentCoordinateMap::indices(8, vec![6, 7, 2, 3]).unwrap(),
                4,
                2
            )
            .unwrap(),
            ComponentCoordinateMap::indices(4, vec![3, 1]).unwrap()
        );
        for coordinates in [
            ComponentCoordinateMap::range(8, 1..5).unwrap(),
            ComponentCoordinateMap::indices(8, vec![0, 2, 1, 3]).unwrap(),
        ] {
            assert!(complete_groups(&coordinates, 4, 2).is_err());
        }
        assert!(complete_groups(&ComponentCoordinateMap::range(8, 0..8).unwrap(), 4, 0).is_err());
    }
}
