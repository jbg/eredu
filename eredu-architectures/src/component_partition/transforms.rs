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
        Equation::AffineProjection {
            weight,
            bias,
            input_width,
            output_width,
        } => {
            let token_axes = |axes: &[eredu_core::TensorAxis]| {
                matches!(axes, [token, _] if token.dimension == SymbolicDimension::TokenRows)
                    || matches!(axes, [batch, sequence, _] if batch.dimension == SymbolicDimension::Batch && sequence.dimension == SymbolicDimension::Sequence)
            };
            if *input_width == 0
                || *output_width == 0
                || !token_axes(input_axes)
                || !token_axes(output_axes)
                || input_axes.last().map(|axis| &axis.dimension)
                    != Some(&SymbolicDimension::Known(*input_width))
                || output_axes.last().map(|axis| &axis.dimension)
                    != Some(&SymbolicDimension::Known(*output_width))
                || input.coordinates.as_ref().is_some_and(|coordinates| {
                    coordinates.contiguous_range() != Some(0..*input_width)
                })
            {
                return Err(invalid().into());
            }
            replicated_parameter(&weight.parameter, &[*output_width, *input_width])?;
            if let Some(bias) = bias {
                replicated_parameter(&bias.parameter, &[*output_width])?;
            }
            output.axis = output_axes
                .last()
                .expect("validated projection axes")
                .name
                .clone();
            output.coordinates = local
                .then(|| ComponentCoordinateMap::range(*output_width, 0..*output_width))
                .transpose()?;
        }
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

pub(super) fn complete_groups(
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
