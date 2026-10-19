//! Replicated channelwise residual cycles on their invocation owners.
use super::*;
use eredu_core::{
    capture::CaptureError,
    component::{ComponentResidualCombination, ComponentResidualCycle},
    SymbolicDimension,
};

pub(super) fn register(
    observations: &mut BTreeMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    cycles: &[ComponentResidualCycle],
    layout: &LocalModelLayout,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
) -> Result<(), ComponentPartitionError> {
    for cycle in cycles {
        let invalid =
            || CaptureError::Invalid(format!("invalid residual cycle placement: {}", cycle.id));
        let parameters = match &cycle.combination {
            ComponentResidualCombination::LearnedVariancePreserving {
                gate,
                temperature,
                minimum_variance,
            } => {
                if !temperature.value().is_finite()
                    || temperature.value() <= 0.0
                    || !minimum_variance.value().is_finite()
                    || minimum_variance.value() <= 0.0
                {
                    return Err(invalid().into());
                }
                vec![gate]
            }
            ComponentResidualCombination::StoredCoefficients { alpha, beta } => vec![alpha, beta],
        };
        let local = owns(&parameters[0].parameter)?;
        let output = descriptor
            .observations
            .get(&cycle.output)
            .ok_or_else(&invalid)?;
        let axes = output.axes.as_deref().ok_or_else(&invalid)?;
        let [batch, sequence, hidden] = axes else {
            return Err(invalid().into());
        };
        let SymbolicDimension::Known(width) = hidden.dimension else {
            return Err(invalid().into());
        };
        if width == 0
            || batch.dimension != SymbolicDimension::Batch
            || sequence.dimension != SymbolicDimension::Sequence
        {
            return Err(invalid().into());
        }
        for parameter in parameters {
            if owns(&parameter.parameter)? != local {
                return Err(invalid().into());
            }
            if local {
                let tensor = layout.tensor(&parameter.parameter).ok_or_else(|| {
                    ComponentPartitionError::MissingWeight(parameter.parameter.clone())
                })?;
                if tensor.global_shape() != [width]
                    || tensor.local_shape() != [width]
                    || !matches!(
                        tensor.placement(),
                        TensorPlacement::Replicated | TensorPlacement::Local
                    )
                    || !tensor.additional_placements().is_empty()
                {
                    return Err(invalid().into());
                }
            }
        }
        for path in [
            &cycle.input,
            &cycle.write,
            &cycle.output,
            &cycle.effective_output,
        ] {
            let point = descriptor.observations.get(path).ok_or_else(&invalid)?;
            if point.axes.as_deref() != Some(axes) {
                return Err(invalid().into());
            }
            let original = if point.position == eredu_core::ObservationPosition::AfterIntervention {
                path.strip_suffix(".effective").ok_or_else(&invalid)?
            } else {
                path.as_str()
            };
            replicated_observation(
                observations,
                descriptor,
                original,
                &hidden.name,
                local,
                ObservationHookSite::Unit,
            )?;
        }
    }
    Ok(())
}
