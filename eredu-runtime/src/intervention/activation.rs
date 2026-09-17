use super::*;

/// Lowers a global compact mask and its explicit token-region slice to one local
/// component axis. This performs no native work and grants no additional budget
/// or admission authority. Distributed callers must establish invocation
/// ownership and reserve work before passing the result to `apply_activation`.
pub fn localize_component_mask(
    action: &InterventionAction,
    slice: &ResolvedCaptureSlice,
    coordinates: &eredu_core::component::ComponentCoordinateMap,
) -> Result<(InterventionAction, ResolvedCaptureSlice), CaptureError> {
    let InterventionAction::MaskComponents {
        dtype,
        indices,
        keep_selected,
    } = action
    else {
        return Err(CaptureError::Invalid(
            "component partition requires a compact component mask".into(),
        ));
    };
    let rank = slice.shape.len();
    let global_count = u64::try_from(coordinates.global_count())
        .map_err(|_| CaptureError::Invalid("global component count exceeds u64".into()))?;
    if rank == 0
        || [&slice.starts, &slice.ends, &slice.strides]
            .iter()
            .any(|values| values.len() != rank)
        || slice.starts[rank - 1] != 0
        || slice.ends[rank - 1] != global_count
        || slice.strides[rank - 1] != 1
        || slice.shape[rank - 1] != global_count
    {
        return Err(CaptureError::Invalid(
            "component partition mask requires the complete global component axis".into(),
        ));
    }
    for axis in 0..rank {
        let (start, end, stride) = (slice.starts[axis], slice.ends[axis], slice.strides[axis]);
        if start >= end || stride == 0 || (end - start).div_ceil(stride) != slice.shape[axis] {
            return Err(CaptureError::Invalid(
                "invalid global component mask region".into(),
            ));
        }
    }
    let indices = coordinates
        .localize_indices(indices)
        .map_err(|error| CaptureError::Invalid(error.to_string()))?;
    let local_count = u64::try_from(coordinates.local_count())
        .map_err(|_| CaptureError::Invalid("local component count exceeds u64".into()))?;
    if local_count == 0 {
        return Err(CaptureError::Invalid(
            "component mask has no local axis to execute".into(),
        ));
    }
    let mut slice = slice.clone();
    slice.ends[rank - 1] = local_count;
    slice.shape[rank - 1] = local_count;
    Ok((
        InterventionAction::MaskComponents {
            dtype: *dtype,
            indices,
            keep_selected: *keep_selected,
        },
        slice,
    ))
}

/// Executes the shared activation recipe. Direct callers receive the same runtime
/// validation as admitted runs; no primitive receives the public action enum.
pub fn apply_activation<B: InterventionBackend>(
    backend: &mut B,
    input: &B::Tensor,
    action: &InterventionAction,
    slice: &ResolvedCaptureSlice,
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    apply_activation_checked(
        backend,
        input,
        action,
        slice,
        InterventionAction::validate_activation_region,
    )
}

pub(super) fn apply_partition_activation<B: InterventionBackend>(
    backend: &mut B,
    input: &B::Tensor,
    action: &InterventionAction,
    slice: &ResolvedCaptureSlice,
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    apply_activation_checked(
        backend,
        input,
        action,
        slice,
        super::partition::validate_local_action,
    )
}

fn apply_activation_checked<B: InterventionBackend>(
    backend: &mut B,
    input: &B::Tensor,
    action: &InterventionAction,
    slice: &ResolvedCaptureSlice,
    validate: impl Fn(&InterventionAction, InterventionDtype, &[u64]) -> Result<(), CaptureError>,
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    use CaptureExecutionError::Backend;
    let source = backend.shape(input).map_err(Backend)?;
    apply_activation_known_shape(backend, input, action, slice, &source, validate)
}

/// Executes the same validated activation recipe using caller-retained geometry.
/// Exact native source shape is checked before any primitive runs; this grants
/// no admission authority and preserves ordinary action dispatch and validation.
pub fn apply_activation_with_source_shape<B: InterventionBackend>(
    backend: &mut B, input: &B::Tensor, action: &InterventionAction,
    slice: &ResolvedCaptureSlice, source: &[u64],
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    apply_activation_known_shape(backend, input, action, slice, source,
        InterventionAction::validate_activation_region)
}

fn apply_activation_known_shape<B: InterventionBackend>(
    backend: &mut B, input: &B::Tensor, action: &InterventionAction,
    slice: &ResolvedCaptureSlice, source: &[u64],
    validate: impl Fn(&InterventionAction, InterventionDtype, &[u64]) -> Result<(), CaptureError>,
) -> Result<B::Tensor, CaptureExecutionError<B::Error>> {
    use CaptureExecutionError::Backend;
    if !backend.matches_intervention_shape(input, source).map_err(Backend)? {
        return Err(CaptureError::Invalid("activation source shape mismatch".into()).into());
    }
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
    validate(action, dtype, &slice.shape)?;
    if matches!(
        action,
        InterventionAction::MaskLogits { .. } | InterventionAction::MaskComponents { .. }
    ) && (slice.starts[rank - 1] != 0
        || slice.ends[rank - 1] != source[rank - 1]
        || slice.strides[rank - 1] != 1)
    {
        return Err(
            CaptureError::Invalid("column masks require the complete final axis".into()).into(),
        );
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
        InterventionAction::MaskComponents {
            indices,
            keep_selected,
            ..
        } => backend
            .mask_components(&selected, indices, *keep_selected)
            .map_err(Backend)?,
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
    if !backend
        .matches_intervention_shape(value, shape)
        .map_err(CaptureExecutionError::Backend)?
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
