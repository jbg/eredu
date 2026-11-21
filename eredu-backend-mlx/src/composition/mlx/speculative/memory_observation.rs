//! Read-only observations of native tensor backing.

/// Complete available native backing, deduplicated within this one survey.
/// Foreign or unavailable backing remains unknown; this never evaluates arrays.
pub(super) fn tensor_backing(
    tensors: &[&crate::MlxTensor],
) -> Result<Option<u64>, eredu_core::BackendFailure> {
    use eredu_nn::Tensor;
    let survey = crate::MlxTensor::inspect_storage(tensors)
        .map_err(eredu_core::BackendFailure::from_error)?;
    if survey.values.iter().any(Option::is_none) {
        return Ok(None);
    }
    Ok(survey.backings.iter().try_fold(0u64, |total, backing| {
        if !backing.allocator_owned {
            return None;
        }
        total.checked_add(backing.allocator_capacity_bytes?)
    }))
}
