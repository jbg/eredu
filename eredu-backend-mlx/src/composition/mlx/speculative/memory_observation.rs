//! Read-only native resource observations shared by ordinary and conditioned drafts.
use crate::composition::mlx::replicated_text::MlxPredictionTargetState;

pub(in crate::composition::mlx) fn model(
    state: &MlxPredictionTargetState,
    additional: u64,
    residency: Option<eredu_runtime::ResidencyReport>,
    banks: Option<
        crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport,
    >,
) -> Result<eredu_core::speculative::SpeculativeModelMemoryObservation, eredu_core::BackendFailure>
{
    let parameter_conversions = residency
        .as_ref()
        .and_then(|report| report.device_parameter_conversions().map(<[_]>::to_vec));
    let retained = state
        .control_estimate()
        .map(|estimate| estimate.retained_bytes);
    Ok(eredu_core::speculative::SpeculativeModelMemoryObservation {
        current_positions: state
            .generation()
            .map_err(eredu_core::BackendFailure::from_error)?,
        current_state_bytes: retained.and_then(|n| n.checked_add(state.control_growth(0)?)),
        peak_state_bytes: retained.and_then(|n| n.checked_add(state.control_growth(additional)?)),
        parameters: crate::composition::mlx::capability::static_memory_from_residency(
            residency, banks,
        )
        .map_err(eredu_core::BackendFailure::from_error)?,
        parameter_conversions,
        available: crate::composition::mlx::capability::available_memory()
            .map_err(eredu_core::BackendFailure::from_error)?,
        allocator_cache_limit: match crate::allocator_cache_policy() {
            Ok(policy) => eredu_core::Observed::exact(
                policy.limit_bytes,
                "current MLX allocator-cache policy",
            ),
            Err(error) => eredu_core::Observed::unavailable(error.to_string()),
        },
    })
}

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

#[cfg(test)]
mod tests {
    use super::tensor_backing;
    use crate::MlxTensor;
    use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Stream};

    #[test]
    fn tensor_backing_deduplicates_views_and_keeps_lazy_storage_unknown() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let root = MlxTensor::from_array(Array::from_slice(&[1f32, -2., 3., 4., 5., -6.], &[2, 3]));
        let transposed = MlxTensor::from_array(root.as_array().transpose(&stream).unwrap());
        let slice = MlxTensor::from_array(root.as_array().try_index_device(0..1, &stream).unwrap());
        transposed.as_array().evaluated().unwrap();
        slice.as_array().evaluated().unwrap();
        let capacity = tensor_backing(&[&root]).unwrap().unwrap();
        assert!(capacity >= 24);
        assert_eq!(
            tensor_backing(&[&root, &root, &transposed, &slice]).unwrap(),
            Some(capacity)
        );
        assert_eq!(tensor_backing(&[]).unwrap(), Some(0));

        let lazy = MlxTensor::from_array(root.as_array().add(root.as_array(), &stream).unwrap());
        assert!(!lazy.as_array().is_available().unwrap());
        assert_eq!(tensor_backing(&[&root, &lazy]).unwrap(), None);
        assert!(
            !lazy.as_array().is_available().unwrap(),
            "forecast observation evaluates lazy work"
        );
    }
}
