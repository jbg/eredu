//! Exact ownership and replicated scalar geometry of post-projection gates.
use super::*;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentReadRole, ComponentRowMapping},
    ObservationPosition as Position, SymbolicDimension, TensorAxis,
};

pub(super) fn insert_observations(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    component: &ComponentGroup,
    layout: &LocalModelLayout,
    local: bool,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    let gate = component
        .output_gate
        .as_ref()
        .expect("declared output gate");
    let invalid =
        || CaptureError::Invalid(format!("invalid scalar output gate for {}", component.id));
    let point = |path: &str, position: Option<Position>| {
        let point = descriptor.observations.get(path).ok_or_else(&invalid)?;
        if position.is_some_and(|position| point.position != position) {
            return Err(invalid());
        }
        point.axes.as_deref().ok_or_else(&invalid)
    };
    let input = point(&gate.input, None)?;
    if !descriptor
        .observations
        .get(&gate.input)
        .is_some_and(|point| {
            matches!(
                point.position,
                Position::ReadOnly | Position::AfterIntervention
            )
        })
        || input.len() != 3
        || input[0].name != "batch"
        || input[1].name != "sequence"
        || gate.read.role != ComponentReadRole::OutputGate
        || gate.read.head_normalization.is_some()
        || component.count == 0
        || input != point(&gate.projection_input, Some(Position::ReadOnly))?
    {
        return Err(invalid().into());
    }
    let TensorAxis {
        name,
        dimension: SymbolicDimension::Known(width),
    } = &input[2]
    else {
        return Err(invalid().into());
    };
    if name != "hidden" || *width == 0 {
        return Err(invalid().into());
    }
    // A scalar gate has one read row shared by every component. Validate the
    // complete mapping algebraically, without enumerating a large unit group.
    let shared_row = match gate.read.rows {
        ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width,
            read_head_width: 1,
            component_heads_per_read_head,
            ..
        } => component_head_width
            .checked_mul(component_heads_per_read_head)
            .is_some_and(|covered| covered >= component.count),
        ComponentRowMapping::Direct { offset: 0 } => component.count == 1,
        _ => false,
    };
    let output = point(&gate.output, Some(Position::BeforeIntervention))?;
    if !shared_row
        || output.len() != 3
        || output[..2] != input[..2]
        || output[2].name != "gate"
        || output[2].dimension != SymbolicDimension::Known(1)
        || output != point(&gate.effective_output, Some(Position::AfterIntervention))?
        || owns(&gate.read.weight)? != local
    {
        return Err(invalid().into());
    }
    for (parameter, shape) in std::iter::once((&gate.read.weight, vec![1, *width]))
        .chain(gate.read.bias.iter().map(|bias| (bias, vec![1])))
    {
        if owns(parameter)? != local {
            return Err(invalid().into());
        }
        if local {
            let tensor = layout
                .tensor(parameter)
                .ok_or_else(|| ComponentPartitionError::MissingWeight(parameter.clone()))?;
            if tensor.global_shape() != shape
                || tensor.local_shape() != shape
                || !matches!(
                    tensor.placement(),
                    TensorPlacement::Local | TensorPlacement::Replicated
                )
                || !tensor.additional_placements().is_empty()
            {
                return Err(ComponentPartitionError::InvalidPlacement(parameter.clone()));
            }
        }
    }
    for (path, axis) in [
        (&gate.input, "hidden"),
        (&gate.projection_input, "hidden"),
        (&gate.output, "gate"),
        (&gate.effective_output, "gate"),
    ] {
        replicated_observation(
            observations,
            descriptor,
            path,
            axis,
            local,
            ObservationHookSite::Unit,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;

    fn fixture() -> (ArchitectureDescriptor, ComponentGroup, LocalModelLayout) {
        let descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&serde_json::json!({
                "model_type":"qwen3_next", "vocab_size":16,"hidden_size":8,
                "num_hidden_layers":2,"intermediate_size":12,"num_attention_heads":4,
                "num_key_value_heads":2,"head_dim":2,"max_position_embeddings":64,
                "linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_value_head_dim":2,
                "linear_num_key_heads":2,"linear_num_value_heads":2,"num_experts":4,
                "moe_intermediate_size":6,"shared_expert_intermediate_size":8,
                "num_experts_per_tok":2,"norm_topk_prob":true,
                "layer_types":["linear_attention","full_attention"],"tie_word_embeddings":false
            }))
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let component = descriptor
            .components
            .iter()
            .find(|g| g.output_gate.is_some())
            .unwrap()
            .clone();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            component.output_gate.as_ref().unwrap().read.weight.clone(),
            tensor(vec![1, 8], TensorPlacement::Replicated),
        );
        (descriptor, component, layout)
    }

    fn tensor(local: Vec<usize>, placement: TensorPlacement) -> LocalTensorLayout {
        LocalTensorLayout::new(
            "gate",
            eredu_runtime::ParameterRole::Replicated,
            vec![1, 8],
            local,
            placement,
            None,
            None,
            false,
        )
    }

    #[test]
    fn scalar_gate_inputs_and_values_follow_invocation_owners() {
        let (descriptor, component, layout) = fixture();
        let gate = component.output_gate.as_ref().unwrap();
        for local in [false, true] {
            let mut observations = BTreeMap::new();
            insert_observations(
                &mut observations,
                &descriptor,
                &component,
                &layout,
                local,
                &|_| Ok(local),
            )
            .unwrap();
            for (path, width) in [
                (&gate.input, 8),
                (&gate.projection_input, 8),
                (&gate.output, 1),
                (&gate.effective_output, 1),
            ] {
                let observation = &observations[path];
                assert_eq!(observation.exports, local);
                assert_eq!(
                    observation.combination,
                    PartitionCaptureCombination::Disjoint
                );
                assert_eq!(
                    observation
                        .coordinates
                        .as_ref()
                        .map(|c| c.contiguous_range()),
                    local.then_some(Some(0..width))
                );
            }
        }
        assert!(insert_observations(
            &mut BTreeMap::new(),
            &descriptor,
            &component,
            &layout,
            true,
            &|_| Ok(false)
        )
        .is_err());
    }

    #[test]
    fn scalar_gate_rejects_partial_parameters_nonconstant_rows_and_false_geometry() {
        let (descriptor, component, layout) = fixture();
        let rejected = |descriptor: &ArchitectureDescriptor,
                        component: &ComponentGroup,
                        layout: &LocalModelLayout| {
            insert_observations(
                &mut BTreeMap::new(),
                descriptor,
                component,
                layout,
                true,
                &|_| Ok(true),
            )
            .is_err()
        };
        let mut wrong = component.clone();
        wrong.output_gate.as_mut().unwrap().read.rows = ComponentRowMapping::Direct { offset: 0 };
        assert!(rejected(&descriptor, &wrong, &layout));
        wrong.output_gate.as_mut().unwrap().read.rows = ComponentRowMapping::GroupedQuery {
            offset: 0,
            head_width: 2,
            queries_per_kv: 2,
        };
        assert!(rejected(&descriptor, &wrong, &layout));
        let mut wrong = component.clone();
        wrong.output_gate.as_mut().unwrap().read.role = ComponentReadRole::Value;
        assert!(rejected(&descriptor, &wrong, &layout));
        let mut wrong = component.clone();
        wrong.output_gate.as_mut().unwrap().effective_output =
            wrong.output_gate.as_ref().unwrap().output.clone();
        assert!(rejected(&descriptor, &wrong, &layout));
        let mut changed = descriptor.clone();
        let gate = component.output_gate.as_ref().unwrap();
        changed
            .observations
            .points
            .iter_mut()
            .find(|p| p.path == gate.output)
            .unwrap()
            .axes
            .as_mut()
            .unwrap()[2]
            .dimension = SymbolicDimension::Known(2);
        assert!(rejected(&changed, &component, &layout));
        for (shape, placement) in [
            (vec![2, 8], TensorPlacement::Replicated),
            (
                vec![1, 4],
                TensorPlacement::Shard {
                    axis: 1,
                    index: 0,
                    parts: 2,
                },
            ),
        ] {
            let mut changed = layout.clone();
            changed.insert(gate.read.weight.clone(), tensor(shape, placement));
            assert!(rejected(&descriptor, &component, &changed));
        }
        assert!(rejected(
            &descriptor,
            &component,
            &LocalModelLayout::default()
        ));
    }
}
