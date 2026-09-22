//! Controls of the active ordinary validation source at completed model roots.
use super::*;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
struct OrdinaryValidationFailure {
    cause: TokenValidationFailure,
    // The erased failure shell retires before this existing host account.
    _host: Option<eredu_core::HostPreparationAuthority>,
}
impl std::fmt::Display for OrdinaryValidationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(formatter)
    }
}
impl std::error::Error for OrdinaryValidationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

pub(super) fn failure(cause: TokenValidationFailure) -> Result<Exception, Exception> {
    let host = crate::backend::nn::shared::current_ordinary_execution_owner()?
        .map(|owner| owner.host().clone());
    Ok(Exception::from_retained_source(OrdinaryValidationFailure {
        cause,
        _host: host,
    }))
}

/// The actual active-validation clone vector and completed predicate reader.
/// Graph construction and its already-completed reductions are priced by the
/// enclosing equation. Repeated references retain separate C Array handles.
pub(crate) fn ordinary_validation_completion_controls(count: usize) -> Option<usize> {
    let evaluation = safemlx::ops::OrdinaryRecipeCall::BorrowedEvaluation.control_bytes()?;
    if evaluation.observed_controls().is_some() {
        return None;
    }
    let per_validation = [
        Array::ordinary_clone_control_bytes()?,
        evaluation.metadata_bytes(),
        safemlx::EvaluatedArray::iteration_control_bytes::<bool>()?,
        size_of::<&TokenValidation>(),
        size_of::<Option<&bool>>(),
        size_of::<Option<bool>>(),
        size_of::<Result<&[bool], safemlx::error::AsSliceError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let parts = [
        Layout::array::<Array>(count).ok()?.size(),
        per_validation.checked_mul(count)?,
        size_of::<Vec<Array>>(),
        size_of::<std::slice::Iter<'static, TokenValidation>>(),
        size_of::<std::cell::Ref<'static, Option<ActiveTokenValidations>>>(),
        size_of::<Option<&ActiveTokenValidations>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<Exception, Exception>>(),
        size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
        // A validation loop returns its first failure. Its typed source needs
        // one fixed erased shell; diagnostic formatting happens at the caller.
        if count == 0 {
            0
        } else {
            Exception::retained_source_control_bytes::<OrdinaryValidationFailure>()?
        },
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
