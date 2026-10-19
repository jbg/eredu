//! Complete residual values are replicated, even when their sublayer writes
//! were assembled from tensor-parallel partials. Ownership comes from the
//! declared coefficient parameters and execution boundaries.
use super::*;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentStreamBase, ComponentStreamCoefficients, ComponentStreamResidual},
    ObservationPosition, SymbolicDimension, TensorAxis,
};

pub(super) fn register(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    streams: &ComponentStreamResidual,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
    base: (bool, ObservationHookSite),
    head: (bool, ObservationHookSite),
) -> Result<(), ComponentPartitionError> {
    let invalid = || CaptureError::Invalid("invalid stream-residual declaration".into());
    if streams.streams == 0 {
        return Err(invalid().into());
    }
    let (base_path, broadcast) = match &streams.base {
        ComponentStreamBase::Broadcast { input } => (input, true),
        ComponentStreamBase::Streams { input } => (input, false),
    };
    let base_point = descriptor
        .observations
        .points
        .iter()
        .find(|p| p.path == *base_path)
        .ok_or_else(|| CaptureError::MissingPath(base_path.clone()))?;
    let hidden = match base_point.axes.as_deref().and_then(|axes| axes.last()) {
        Some(TensorAxis {
            name,
            dimension: SymbolicDimension::Known(width),
        }) if name == "hidden" && *width > 0 => *width,
        _ => return Err(invalid().into()),
    };
    hidden.checked_mul(streams.streams).ok_or_else(invalid)?;
    streams
        .streams
        .checked_mul(streams.streams)
        .ok_or_else(invalid)?;
    let mut vector = vec![
        TensorAxis {
            name: "batch".into(),
            dimension: SymbolicDimension::Batch,
        },
        TensorAxis {
            name: "sequence".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "hidden".into(),
            dimension: SymbolicDimension::Known(hidden),
        },
    ];
    let hidden_axes = vector.clone();
    vector[2] = TensorAxis {
        name: "stream".into(),
        dimension: SymbolicDimension::Known(streams.streams),
    };
    let mut residual = vector.clone();
    residual.push(hidden_axes[2].clone());
    let mut matrix = vector.clone();
    matrix[2].name = "input_stream".into();
    matrix.push(TensorAxis {
        name: "output_stream".into(),
        dimension: SymbolicDimension::Known(streams.streams),
    });
    let coefficient_owner = |parameters: &ComponentStreamCoefficients| {
        let local = owns(&parameters.function)?;
        if owns(&parameters.base)? != local || owns(&parameters.scale)? != local {
            return Err(invalid().into());
        }
        Ok::<_, ComponentPartitionError>(local)
    };
    let mut add = |path: &str, axes: &[TensorAxis], axis: &str, local, site| {
        // Effective paths are an explicit public observation convention. Check
        // their catalog timing and geometry before pairing them with originals.
        let point = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path == path)
            .ok_or_else(|| CaptureError::MissingPath(path.into()))?;
        if path.ends_with(".effective") && point.position != ObservationPosition::AfterIntervention
        {
            return Err(invalid().into());
        }
        let original = if point.position == ObservationPosition::AfterIntervention {
            path.strip_suffix(".effective").ok_or_else(invalid)?
        } else {
            path
        };
        if point.axes.as_deref() != Some(axes) {
            return Err(invalid().into());
        }
        let original_point = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path == original)
            .ok_or_else(|| CaptureError::MissingPath(original.into()))?;
        if original_point.axes.as_deref() != Some(axes)
            || (original != path
                && original_point.position != ObservationPosition::BeforeIntervention)
        {
            return Err(invalid().into());
        }
        if let Some(effective) = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path == format!("{original}.effective"))
        {
            if effective.axes.as_deref() != Some(axes)
                || effective.position != ObservationPosition::AfterIntervention
            {
                return Err(invalid().into());
            }
        }
        replicated_observation(observations, descriptor, original, axis, local, site)
    };
    add(
        base_path,
        if broadcast { &hidden_axes } else { &residual },
        "hidden",
        base.0,
        base.1,
    )?;
    for cycle in &streams.cycles {
        let local = coefficient_owner(&cycle.coefficients)?;
        for path in [&cycle.input, &cycle.output] {
            add(path, &residual, "hidden", local, ObservationHookSite::Unit)?;
        }
        for path in [&cycle.collapsed, &cycle.write] {
            add(
                path,
                &hidden_axes,
                "hidden",
                local,
                ObservationHookSite::Unit,
            )?;
        }
        for path in [&cycle.pre, &cycle.post] {
            add(path, &vector, "stream", local, ObservationHookSite::Unit)?;
        }
        add(
            &cycle.combination,
            &matrix,
            "input_stream",
            local,
            ObservationHookSite::Unit,
        )?;
    }
    if coefficient_owner(&streams.head.parameters)? != head.0 {
        return Err(invalid().into());
    }
    add(&streams.head.input, &residual, "hidden", head.0, head.1)?;
    add(
        &streams.head.coefficients,
        &vector,
        "stream",
        head.0,
        head.1,
    )?;
    Ok(())
}
