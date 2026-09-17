use super::Array;
use crate::{
    OriginalScopeObserver,
    error::Exception,
    utils::{guard::Guarded, runtime_lock},
};

/// Checked geometry for one eager I32 input with no numerical host staging.
///
/// The output repeats each integer in `0..groups` exactly `repeats` times.
/// Construction uses the same native descriptor, Data, buffer and original
/// scope ownership as [`Array::try_from_slice`]. This plan grants no admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatedI32InputPlan {
    groups: usize,
    repeats: usize,
    elements: usize,
}

impl RepeatedI32InputPlan {
    /// Validate the iterator range, I32 shape and requested byte extent without
    /// constructing native state. Empty ranges are valid and never divide.
    pub fn new(groups: usize, repeats: usize) -> Option<Self> {
        let maximum = i32::MAX as usize;
        if groups > maximum || repeats > maximum {
            return None;
        }
        let elements = groups.checked_mul(repeats)?;
        if elements > maximum || elements > isize::MAX as usize {
            return None;
        }
        elements.checked_mul(size_of::<i32>())?;
        Some(Self {
            groups,
            repeats,
            elements,
        })
    }

    /// Exact rank-one output length.
    pub fn elements(self) -> usize {
        self.elements
    }

    /// Bytes requested by the one eager native buffer, before allocator padding.
    /// The native allocation recipe must count that buffer exactly once.
    pub fn requested_bytes(self) -> usize {
        // Validated during construction; fields are private and immutable.
        self.elements * size_of::<i32>()
    }

    /// Named fixed safe/native producer controls. This excludes the separately
    /// accounted descriptor, Data, C shell and final native buffer, and does not
    /// certify unrelated ordinary diagnostics or allocator infrastructure.
    pub fn control_bytes() -> Option<usize> {
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<Array>(),
            size_of::<crate::utils::guard::MaybeUninitArray>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Option<OriginalScopeObserver>>(),
            size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<usize>() * 3,
            size_of::<i32>(),
        ];
        // SAFETY: pure native sizeof inventory; no runtime initialization.
        let native = unsafe { safemlx_sys::mlx_array_repeated_i32_control_bytes() };
        controls.into_iter().try_fold(
            native.checked_add(OriginalScopeObserver::control_bytes()?)?,
            usize::checked_add,
        )
    }

    /// Fill the final owned native allocation directly and publish the array
    /// only after successful construction. Original calls use the current
    /// authenticated scope and its existing fixed failure carrier.
    pub fn create(self) -> Result<Array, Exception> {
        Array::try_from_op(|output| {
            // SAFETY: private fields passed the same checked range contract as
            // the native factory. The destination guard owns publication and
            // cleanup; no iterator or borrowed storage escapes this call.
            unsafe { safemlx_sys::mlx_array_set_repeated_i32(output, self.groups, self.repeats) }
        })
    }
}
