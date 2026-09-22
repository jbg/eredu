//! The ordinary model completion worker's caller allocations and transports.
use super::*;
use crate::backend::nn::workspace::OrdinaryCallControls;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

/// Both ordinary completion entry points use the same completed-root worker.
/// The retained-media variant owns C Array aliases; the plain variant borrows
/// them. Their union preserves a finite envelope for this shared producer.
/// Roots include every output, state and validation occurrence, without
/// deduplicating aliases. Native Eval is supplied by the retained equation.
pub(crate) fn ordinary_model_completion_call_controls(
    roots: usize,
    validations: usize,
) -> Option<OrdinaryCallControls> {
    if validations > roots {
        return None;
    }
    let mut controls = MlxNeuralBackend::ordinary_completion_call_controls(roots)?;
    let evaluation = safemlx::ops::OrdinaryRecipeCall::BorrowedEvaluation.control_bytes()?;
    if let Some(observed) = evaluation.observed_controls() {
        controls.observed.include(observed)?;
    }
    // These Vecs collect and append existing roots. On the qualified standard
    // library, doubling growth has cumulative requested capacity below four
    // times its final population plus two initial four-element buffers. Include
    // all replaced buffers so allocator overlap never relies on early refunds.
    let slots = roots.checked_mul(4)?.checked_add(8)?;
    let fields = [
        Layout::array::<Array>(slots).ok()?.size(),
        Array::ordinary_clone_control_bytes()?.checked_mul(roots)?,
        evaluation.metadata_bytes().checked_mul(roots)?,
        safemlx::Event::ordinary_synchronize_control_bytes()?,
        crate::backend::nn::tensor::ordinary_validation_completion_controls(validations)?,
        size_of::<Vec<Array>>(),
        size_of::<Vec<&Array>>(),
        size_of::<std::slice::Iter<'static, Array>>(),
        size_of::<std::slice::Iter<'static, &'static Array>>(),
        size_of::<safemlx::Event>(),
        size_of::<Result<safemlx::Event, Exception>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Option<&MlxTensor>>(),
        size_of::<&eredu_runtime::media_prefill::RetainedMediaRoots<'static, MlxTensor>>(),
        size_of::<OrdinaryCallControls>(),
        size_of::<Option<OrdinaryCallControls>>(),
    ];
    let metadata = fields
        .into_iter()
        .try_fold(size_of_val(&fields), usize::checked_add)?;
    controls.metadata_bytes = controls
        .metadata_bytes
        .checked_add(u64::try_from(metadata).ok()?)?;
    Some(controls)
}
