//! Fixed-shape replay of an already constructed native graph.

use std::{
    ffi::{c_void, CString},
    ptr::NonNull,
};

use crate::{
    error::Result,
    utils::{guard::Guarded, runtime_lock, VectorArray},
    Array, Stream,
};

/// Immutable, bounded description of a pure MLX graph region.
#[derive(Debug)]
pub struct ReplayGraph(NonNull<c_void>);

// SAFETY: the tape owns immutable graph descriptors and MLX array handles.
// Applying it only creates new array descriptors and checks the original stream.
unsafe impl Send for ReplayGraph {}
// SAFETY: concurrent applications read the tape and allocate independent outputs.
unsafe impl Sync for ReplayGraph {}

impl ReplayGraph {
    /// Records a region whose changing inputs are explicit. The caller
    /// certifies that the named custom kernels only write declared outputs.
    pub fn record(
        inputs: &[&Array],
        outputs: &[&Array],
        stream: &Stream,
        replayable_kernels: &[&str],
    ) -> Result<Option<Self>> {
        let names = replayable_kernels
            .iter()
            .map(|&name| {
                CString::new(name)
                    .map_err(|error| crate::error::Exception::custom(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let name_ptrs: Vec<_> = names.iter().map(|name| name.as_ptr()).collect();
        let _guard = runtime_lock::enter_for_metadata();
        let inputs = VectorArray::try_from_iter(inputs.iter().copied())?;
        let outputs = VectorArray::try_from_iter(outputs.iter().copied())?;
        let mut result = std::ptr::null_mut();
        // SAFETY: the handles and writable out pointer stay live for the call.
        <()>::try_from_op(|_| unsafe {
            safemlx_sys::mlx_record_fixed_shape_graph(
                &mut result,
                inputs.as_ptr(),
                outputs.as_ptr(),
                stream.as_ptr(),
                name_ptrs.as_ptr(),
                name_ptrs.len(),
            )
        })?;
        Ok(NonNull::new(result).map(Self))
    }

    /// Constructs fresh output arrays, or declines incompatible bindings.
    pub fn apply(&self, inputs: &[&Array], stream: &Stream) -> Result<Option<Vec<Array>>> {
        let _guard = runtime_lock::enter_for_metadata();
        let inputs = VectorArray::try_from_iter(inputs.iter().copied())?;
        let mut matched = false;
        // SAFETY: the tape and its input handles stay live, and native code writes
        // only the initialized output vector and the borrowed match flag.
        let output = VectorArray::try_from_op(|result| unsafe {
            safemlx_sys::mlx_apply_fixed_shape_graph(
                result,
                &mut matched,
                self.0.as_ptr(),
                inputs.as_ptr(),
                stream.as_ptr(),
            )
        })?;
        if matched {
            output.try_into_values().map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Drop for ReplayGraph {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter_for_metadata();
        // SAFETY: this handle uniquely owns the tape.
        unsafe { safemlx_sys::mlx_free_fixed_shape_graph(self.0.as_ptr()) };
    }
}
