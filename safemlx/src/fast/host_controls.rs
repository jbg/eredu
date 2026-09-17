//! Concrete one-output configuration and synchronous C-bridge owner layouts.
use super::*;
use std::{alloc::Layout, mem::size_of};

/// Per-call host owners for three borrowed inputs and a fresh one-output kernel configuration with
/// up to four named templates and an inline native output shape. The caller must
/// qualify Rust's fresh String/Vec/CString producers and separately cover native
/// Graph storage, kernel construction/cache lifetime and generated source text.
/// This pure query performs no construction and grants no storage authority.
pub fn three_input_kernel_control_bytes(names: &[&str], rank: usize) -> Option<usize> {
    if names.len() > 4 || names.iter().any(|name| name.as_bytes().contains(&0)) {
        return None;
    }
    let maximum_name = names.iter().map(|name| name.len()).max().unwrap_or(0);
    // SAFETY: this owning native query reads only scalar counts and allocates nothing.
    let native = unsafe {
        safemlx_sys::mlx_fast_metal_single_output_control_bytes(names.len(), rank, maximum_name)
    };
    if native == 0 {
        return None;
    }
    // Qualified Rust RawVec grows these non-byte elements to four slots on the
    // first push. Four names need no subsequent reallocation. The fallible
    // output collector likewise starts at four (its lower size hint is zero).
    let payload = Layout::array::<CustomKernelOutput>(4)
        .ok()?
        .size()
        .checked_add(
            Layout::array::<CustomKernelTemplateArg>(if names.is_empty() { 0 } else { 4 })
                .ok()?
                .size(),
        )?
        .checked_add(Layout::array::<i32>(rank).ok()?.size())?
        .checked_add(Layout::array::<Array>(4).ok()?.size())?;
    let names = names
        .iter()
        .try_fold(0usize, |n, name| n.checked_add(name.len()))?;
    // CString::new(&str) specializes to one len+1 allocation, with no shrink:
    // it reserves the terminator before copying. Only one name conversion is live.
    let c_name = maximum_name.checked_add(1)?;
    [
        payload,
        names,
        c_name,
        size_of::<CustomKernelConfig>(),
        size_of::<CustomKernelOutput>(),
        size_of::<CustomKernelTemplateArg>(),
        size_of::<RawMetalKernelConfig>(),
        size_of::<Result<RawMetalKernelConfig>>(),
        size_of::<Result<()>>(),
        size_of::<CString>(),
        size_of::<Result<CString>>(),
        size_of::<Vec<Array>>(),
        size_of::<Result<Vec<Array>>>(),
        size_of::<VectorArray>(),
        size_of::<Result<VectorArray>>(),
        size_of::<crate::utils::guard::MaybeUninitVectorArray>().checked_mul(2)?,
        size_of::<crate::utils::guard::MaybeUninitArray>(),
        size_of::<std::slice::Iter<'static, CustomKernelOutput>>(),
        size_of::<std::slice::Iter<'static, CustomKernelTemplateArg>>(),
        size_of::<std::array::IntoIter<&'static Array, 3>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::cell::Ref<'static, Option<MetalKernel>>>(),
        size_of::<std::cell::RefMut<'static, Option<MetalKernel>>>(),
        size_of::<&CustomKernelConfig>(),
        size_of::<&Stream>(),
        size_of::<safemlx_sys::mlx_fast_metal_kernel_config>(),
        size_of::<[i32; 3]>().checked_mul(2)?,
        crate::OriginalScopeObserver::control_bytes()?,
    ]
    .into_iter()
    .try_fold(native, usize::checked_add)
}
