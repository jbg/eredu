//! Eager token-ID upload in the already admitted original preparation scope.
use super::Array;
use crate::{
    error::{Exception, ScopedEvaluationCause},
    utils::runtime_lock,
    OriginalScopeObserver, PreparedInputRuntime,
};
use std::{fmt, mem::{size_of, size_of_val}, ptr};

/// Fixed refusal. The caller's original recovery scope retains any native cause;
/// this result does not settle or release that scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OriginalPromptInputCause {
    /// The token extent cannot be represented by the selected input constructor.
    Invalid,
    /// The retained allocator or native host layout has no complete bound.
    Unqualified,
}
impl fmt::Display for OriginalPromptInputCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => f.write_str("invalid original prompt input"),
            Self::Unqualified => f.write_str("original prompt input layout is unqualified"),
        }
    }
}
impl std::error::Error for OriginalPromptInputCause {}

/// Actual one-source physical capacity and Graph destinations. The fact itself
/// grants no storage; the selected request must prepay these contributions.
#[derive(Clone, Copy, Debug)]
pub struct OriginalPromptInputFacts {
    elements: usize,
    raw: safemlx_sys::mlx_original_prompt_input_layout,
}
impl OriginalPromptInputFacts {
    /// Query the exact eager four-byte integer input producer (U32 or I32)
    /// using retained allocator facts. Both share the same native worker/layout.
    /// This allocates no tensor and supplies no execution authority.
    pub fn inspect(
        runtime: &PreparedInputRuntime,
        elements: usize,
    ) -> Result<Self, OriginalPromptInputCause> {
        if elements == 0 || elements > i32::MAX as usize {
            return Err(OriginalPromptInputCause::Invalid);
        }
        let mut raw = safemlx_sys::mlx_original_prompt_input_layout {
            metadata_bytes: 0,
            backing_bytes: 0,
            controls: 0,
            ordinary_handle_bytes: 0,
        };
        // SAFETY: the pure query borrows the genuine retained allocator facts.
        let status = unsafe {
            safemlx_sys::mlx_original_prompt_input_layout_for(&mut raw, runtime.raw(), elements)
        };
        if status != 0 {
            return Err(OriginalPromptInputCause::Unqualified);
        }
        Ok(Self { elements, raw })
    }
    /// Number of token IDs copied into the final `[1,N]` tensor.
    pub fn elements(self) -> usize {
        self.elements
    }
    /// Physical capacity of the single allocator-owned native backing.
    pub fn mutable_bytes(self) -> usize {
        self.raw.backing_bytes
    }
    /// Graph allocation extents for descriptor, data, birth and output handle.
    pub fn graph_bytes(self) -> usize {
        self.raw.metadata_bytes
    }
    /// The eager constructor creates one physical buffer generation.
    pub fn maximum_births(self) -> usize {
        1
    }
    /// Physical C++ outer array storage created by one ordinary handle clone.
    pub fn ordinary_handle_bytes(self) -> usize {
        self.raw.ordinary_handle_bytes
    }
    /// Checked storage for the named native and safe constructor controls.
    pub fn control_bytes(self) -> Option<usize> {
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, OriginalPromptInputCause>>(),
            size_of::<OriginalPromptInputCause>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<Array>(),
            size_of::<safemlx_sys::mlx_original_prompt_input_layout>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<&PreparedInputRuntime>(),
            size_of::<&[u32]>(),
            size_of::<unsafe extern "C" fn(*mut safemlx_sys::mlx_array, *const u32, usize) -> u32>(),
            size_of::<(*const u32, usize)>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<usize>() * 2,
            size_of::<u32>() * 2,
        ];
        controls.into_iter().try_fold(
            self.raw
                .controls
                .checked_add(size_of_val(&controls))?
                .checked_add(OriginalScopeObserver::control_bytes()?)?
                .checked_add(size_of::<Exception>())?,
            usize::checked_add,
        )
    }
}
impl Array {
    /// Construct exactly U32[1,N] in the current authenticated original scope.
    /// The copy is eager, with no reshape, evaluation, submission or callback.
    pub fn try_from_original_prompt_ids(values: &[u32]) -> Result<Self, Exception> {
        original_token_ids(values, safemlx_sys::mlx_original_prompt_input_new)
    }

    /// Construct exactly I32[1,N] through the same eager original source worker.
    /// Values are copied without conversion, reshaping, evaluation or submission.
    pub fn try_from_original_prediction_ids(values: &[i32]) -> Result<Self, Exception> {
        original_token_ids(values, safemlx_sys::mlx_original_prediction_input_new)
    }
}

fn original_token_ids<T>(
    values: &[T],
    construct: unsafe extern "C" fn(*mut safemlx_sys::mlx_array, *const T, usize) -> u32,
) -> Result<Array, Exception> {
        if values.is_empty() || values.len() > i32::MAX as usize {
            return Err(Exception::from_scoped_evaluation(
                ScopedEvaluationCause::Invalid,
                None,
            ));
        }
        let observer = OriginalScopeObserver::require_current()?;
        let _loan = runtime_lock::try_enter_for_recovery().ok_or_else(|| observer.error(10))?;
        let mut output = safemlx_sys::mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        // SAFETY: the initialized slice lives throughout the synchronous copy.
        // Native validates the actual original Scope/Graph/budget, publishes an
        // owning prepared handle only on success, and otherwise keeps it null.
        let status = unsafe {
            construct(&mut output, values.as_ptr(), values.len())
        };
        if status != 0 {
            // The factory uses PreparedHostCopyCause; translate its fixed code
            // to the observer protocol while retaining the exact native carrier.
            let code = match status {
                2 => 10,
                3 => 3,
                4 => 2,
                6 => 4,
                7 => 5,
                9 => 7,
                _ => 1,
            };
            return Err(observer.error(code));
        }
        // SAFETY: successful native publication supplies the complete owned handle.
        Ok(unsafe { Array::from_ptr(output) })
}

#[cfg(test)]
mod tests;
