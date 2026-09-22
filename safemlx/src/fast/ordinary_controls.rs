//! Caller controls for the existing ordinary fast wrappers.
use super::*;
use std::mem::{size_of, size_of_val};

fn sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn native(kind: u32) -> Option<usize> {
    let mut bytes = 0;
    // SAFETY: the linked pure sizeof query writes only the initialized scalar.
    unsafe { safemlx_sys::mlx_ordinary_fast_wrapper_controls(&mut bytes, kind) }.then_some(bytes)
}

/// One ordinary fast SDPA call with an empty mask-mode string. Optional array
/// mask and sink inputs are borrowed. No ArrayVector is constructed by this wrapper;
/// native fallback graph, numerical backings and Eval are separate sources.
pub fn ordinary_sdpa_control_bytes() -> Option<usize> {
    sum([
        native(0)?,
        crate::ops::ordinary_array_result_guard_control_bytes()?,
        size_of::<(
            Array,
            Array,
            Array,
            f32,
            Option<ScaledDotProductAttentionMask<'static>>,
            Option<&Array>,
            &Stream,
        )>(),
        size_of::<(
            &Array,
            &Array,
            &Array,
            &f32,
            &Option<ScaledDotProductAttentionMask<'static>>,
            &Option<&Array>,
            &&Stream,
        )>(),
        size_of::<(&CStr, safemlx_sys::mlx_array)>(),
        size_of::<Option<ScaledDotProductAttentionMask<'static>>>(),
        size_of::<Option<&Array>>(),
        size_of::<safemlx_sys::mlx_array>() * 2,
        size_of::<Result<Array>>(),
    ])
}

/// One actual scalar-offset C RoPE invocation. The enclosing batch wrapper
/// separately retains its output Vec, input clones/slices and final join.
pub fn ordinary_rope_invocation_control_bytes() -> Option<usize> {
    sum([
        native(1)?,
        crate::ops::ordinary_array_result_guard_control_bytes()?,
        size_of::<(
            &Array,
            i32,
            bool,
            Option<f32>,
            f32,
            i32,
            Option<&Array>,
            &Stream,
        )>(),
        size_of::<(
            &Array,
            &i32,
            &bool,
            &Option<f32>,
            &f32,
            &i32,
            &Option<&Array>,
            &&Stream,
        )>(),
        size_of::<safemlx_sys::mlx_optional_float>(),
        size_of::<Option<&Array>>(),
        size_of::<safemlx_sys::mlx_array>(),
        size_of::<Result<Array>>(),
    ])
}
