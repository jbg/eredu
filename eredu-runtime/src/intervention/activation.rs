use super::*;

/// Executes the shared activation recipe. Direct callers receive the same runtime
/// validation as admitted runs; no primitive receives the public action enum.
pub fn apply_activation<B: InterventionBackend>(
    backend: &mut B,
    input: &B::Tensor,
    action: &InterventionAction,
    slice: &ResolvedCaptureSlice,
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    use CaptureExecutionError::Backend;
    let source = backend.shape(input).map_err(Backend)?;
    let dtype = backend.intervention_dtype(input).map_err(Backend)?;
    let rank = source.len();
    if [&slice.starts, &slice.ends, &slice.strides, &slice.shape]
        .iter()
        .any(|v| v.len() != rank)
    {
        return Err(CaptureError::Invalid("activation slice rank mismatch".into()).into());
    }
    for (axis, extent) in source.iter().enumerate() {
        let (start, end, stride) = (slice.starts[axis], slice.ends[axis], slice.strides[axis]);
        if stride == 0
            || start >= end
            || end > *extent
            || (end - start).div_ceil(stride) != slice.shape[axis]
        {
            return Err(CaptureError::Invalid("invalid activation region".into()).into());
        }
    }
    action.validate_activation_region(dtype, &slice.shape)?;
    if matches!(action, InterventionAction::MaskLogits { .. })
        && (slice.starts[rank - 1] != 0
            || slice.ends[rank - 1] != source[rank - 1]
            || slice.strides[rank - 1] != 1)
    {
        return Err(CaptureError::Invalid(
            "logit masks require the complete vocabulary axis".into(),
        )
        .into());
    }
    backend.validate_intervention_geometry(&source, slice)?;
    let selected = backend.select_region(input, slice).map_err(Backend)?;
    validate_value(backend, &selected, &slice.shape, dtype)?;
    let replacement = match action {
        InterventionAction::Zero { .. } => backend.zeros(&slice.shape, dtype).map_err(Backend)?,
        InterventionAction::Scale { factor, .. } => {
            backend.scale(&selected, *factor).map_err(Backend)?
        }
        InterventionAction::Mask { keep, .. } => {
            backend.fill_masked(&selected, keep, 0.0).map_err(Backend)?
        }
        InterventionAction::Replace { tensor } => {
            backend.realize_tensor(tensor).map_err(Backend)?
        }
        InterventionAction::Add { tensor } => {
            let delta = backend.realize_tensor(tensor).map_err(Backend)?;
            validate_value(backend, &delta, &slice.shape, dtype)?;
            backend.add(&selected, &delta).map_err(Backend)?
        }
        InterventionAction::MaskLogits { token_ids, .. } => backend
            .fill_columns(&selected, token_ids, f32::NEG_INFINITY)
            .map_err(Backend)?,
        _ => {
            return Err(
                CaptureError::Invalid("routing action cannot replace an activation".into()).into(),
            )
        }
    };
    validate_value(backend, &replacement, &slice.shape, dtype)?;
    let output = backend
        .update_region(input, slice, &replacement)
        .map_err(Backend)?;
    validate_value(backend, &output, &source, dtype)?;
    Ok(output)
}

fn validate_value<B: InterventionBackend>(
    backend: &B,
    value: &B::Tensor,
    shape: &[u64],
    dtype: InterventionDtype,
) -> Result<(), CaptureExecutionError<B::Error>> {
    if backend
        .shape(value)
        .map_err(CaptureExecutionError::Backend)?
        != shape
        || backend
            .intervention_dtype(value)
            .map_err(CaptureExecutionError::Backend)?
            != dtype
    {
        return Err(CaptureError::Invalid(
            "native activation primitive changed shape or dtype".into(),
        )
        .into());
    }
    Ok(())
}
