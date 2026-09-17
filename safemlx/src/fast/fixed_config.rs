//! Synchronous stack configuration; generated kernel/source owners remain separate.
use super::*;
use std::mem::size_of;

/// One output shape borrowed only until the kernel invocation returns.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedKernelOutput<'a> {
    /// Logical output shape; the fixed entry supports native inline ranks.
    pub shape: &'a [i32],
    /// Exact output scalar type.
    pub dtype: Dtype,
}
/// One template value with a borrowed NUL-terminated name, at most 22 bytes.
#[derive(Clone, Copy, Debug)]
pub enum BorrowedKernelTemplate<'a> {
    /// Signed integer specialization.
    Int(&'a CStr, i32),
    /// Boolean specialization.
    Bool(&'a CStr, bool),
    /// Scalar-type specialization.
    Dtype(&'a CStr, Dtype),
}
impl BorrowedKernelTemplate<'_> {
    pub(crate) fn raw(self) -> Option<safemlx_sys::mlx_fast_template_view> {
        let (name, kind, value) = match self {
            Self::Int(name, value) => (name, 0, value),
            Self::Bool(name, value) => (name, 1, i32::from(value)),
            Self::Dtype(name, value) => (name, 2, safemlx_sys::mlx_dtype::from(value) as i32),
        };
        (name.to_bytes().len() <= 22).then_some(safemlx_sys::mlx_fast_template_view {
            name: name.as_ptr(),
            kind,
            value,
        })
    }
}
const EMPTY: safemlx_sys::mlx_array = safemlx_sys::mlx_array {
    ctx: std::ptr::null_mut(),
    prepared_owner: std::ptr::null_mut(),
};
struct FixedOutputs<const N: usize>([safemlx_sys::mlx_array; N]);
impl<const N: usize> Drop for FixedOutputs<N> {
    fn drop(&mut self) {
        for output in &mut self.0 {
            // SAFETY: every nonempty slot is a unique owned C handle, including
            // a prefix installed before native failure. Empty slots are valid.
            unsafe { safemlx_sys::mlx_array_free(std::mem::replace(output, EMPTY)) };
        }
    }
}
fn unsupported() -> Exception {
    match crate::OriginalScopeObserver::try_current() {
        Ok(Some(observer)) => observer.capacity_error(),
        Err(cause) => cause,
        Ok(None) => Exception::custom("fixed Metal invocation configuration is unqualified"),
    }
}
impl MetalKernel {
    /// Fixed invocation/configuration controls for up to eight inputs, four
    /// outputs and four templates. This prices no generated source, kernel,
    /// cache, physical payload or Graph allocation, and grants no authority.
    pub fn fixed_control_bytes<const I: usize, const O: usize>(
        templates: usize,
        maximum_rank: usize,
    ) -> Option<usize> {
        // SAFETY: the pure native query reads counts only and allocates nothing.
        let native = unsafe {
            safemlx_sys::mlx_fast_metal_fixed_control_bytes(I, O, templates, maximum_rank)
        };
        if native == 0 {
            return None;
        }
        [
            size_of::<[&Array; I]>(),
            size_of::<[safemlx_sys::mlx_array; I]>(),
            size_of::<[BorrowedKernelOutput<'_>; O]>(),
            size_of::<[safemlx_sys::mlx_fast_output_view; O]>(),
            size_of::<[safemlx_sys::mlx_fast_template_view; 4]>(),
            size_of::<[BorrowedKernelTemplate<'_>; 4]>(),
            size_of::<FixedOutputs<O>>(),
            size_of::<[Array; O]>(),
            size_of::<Result<[Array; O]>>(),
            size_of::<[i32; 3]>() * 2,
            size_of::<Option<safemlx_sys::mlx_fast_template_view>>(),
            size_of::<std::slice::Iter<'_, BorrowedKernelTemplate<'_>>>(),
            size_of::<FixedKernel>(),
            size_of::<&MetalKernel>(),
            size_of::<&Stream>(),
            crate::OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }

    /// Applies through the ordinary native kernel worker with borrowed fixed
    /// configuration. Returned arrays own their normal native output descriptors.
    /// No config, template-name, output-shape vector or Rust result vector is
    /// allocated. Generated source and retained kernel/cache owners are unchanged.
    pub fn apply_fixed_device<const I: usize, const O: usize>(
        &self,
        inputs: [&Array; I],
        outputs: [BorrowedKernelOutput<'_>; O],
        templates: &[BorrowedKernelTemplate<'_>],
        grid: [i32; 3],
        thread_group: [i32; 3],
        stream: &Stream,
    ) -> Result<[Array; O]> {
        apply_fixed(
            FixedKernel::Ordinary(self.c_kernel),
            inputs,
            outputs,
            templates,
            grid,
            thread_group,
            stream,
        )
    }
}
// Both safe owners enter the same stack configuration/partial-output guard.
// These raw handles remain private and are borrowed from their owning caller.
pub(crate) enum FixedKernel {
    Ordinary(safemlx_sys::mlx_fast_metal_kernel),
    Prepared(safemlx_sys::mlx_fast_prepared_definition),
    Family(safemlx_sys::mlx_fast_kernel_family),
}
pub(crate) fn apply_fixed<const I: usize, const O: usize>(
    kernel: FixedKernel,
    inputs: [&Array; I],
    outputs: [BorrowedKernelOutput<'_>; O],
    templates: &[BorrowedKernelTemplate<'_>],
    grid: [i32; 3],
    thread_group: [i32; 3],
    stream: &Stream,
) -> Result<[Array; O]> {
    let rank = outputs
        .iter()
        .map(|output| output.shape.len())
        .max()
        .unwrap_or(0);
    MetalKernel::fixed_control_bytes::<I, O>(templates.len(), rank).ok_or_else(unsupported)?;
    let mut raw_templates = [safemlx_sys::mlx_fast_template_view {
        name: std::ptr::null(),
        kind: 0,
        value: 0,
    }; 4];
    for (target, source) in raw_templates.iter_mut().zip(templates) {
        *target = source.raw().ok_or_else(unsupported)?;
    }
    let raw_inputs = inputs.map(Array::as_ptr);
    let raw_outputs = outputs.map(|output| safemlx_sys::mlx_fast_output_view {
        shape: output.shape.as_ptr(),
        ndim: output.shape.len(),
        dtype: output.dtype.into(),
    });
    let mut result = FixedOutputs([EMPTY; O]);
    crate::error::ensure_mlx_error_handler();
    // SAFETY: all descriptors and names remain borrowed for this synchronous
    // call. Native validates fixed capacities and writes unique output slots.
    let status = unsafe {
        match kernel {
            FixedKernel::Ordinary(kernel) => safemlx_sys::mlx_fast_metal_kernel_apply_fixed(
                result.0.as_mut_ptr(),
                O,
                kernel,
                raw_inputs.as_ptr(),
                I,
                raw_outputs.as_ptr(),
                raw_templates.as_ptr(),
                templates.len(),
                grid.as_ptr(),
                thread_group.as_ptr(),
                stream.as_ptr(),
            ),
            FixedKernel::Prepared(kernel) => safemlx_sys::mlx_fast_metal_definition_apply_fixed(
                result.0.as_mut_ptr(),
                O,
                kernel,
                raw_inputs.as_ptr(),
                I,
                raw_outputs.as_ptr(),
                raw_templates.as_ptr(),
                templates.len(),
                grid.as_ptr(),
                thread_group.as_ptr(),
                stream.as_ptr(),
            ),
            FixedKernel::Family(kernel) => safemlx_sys::mlx_fast_kernel_family_apply_fixed(
                result.0.as_mut_ptr(),
                O,
                kernel,
                raw_inputs.as_ptr(),
                I,
                raw_outputs.as_ptr(),
                raw_templates.as_ptr(),
                templates.len(),
                grid.as_ptr(),
                thread_group.as_ptr(),
                stream.as_ptr(),
            ),
        }
    };
    check_status(status)?;
    Ok(std::array::from_fn(|i| {
        // SAFETY: successful native return initialized every slot exactly
        // once. Moving it out disables this guard's corresponding free.
        unsafe { Array::from_ptr(std::mem::replace(&mut result.0[i], EMPTY)) }
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;

    #[test]
    fn fixed_config_preserves_multiple_outputs_and_refuses_oversized_templates() {
        let qualified = MetalKernel::fixed_control_bytes::<1, 2>(3, 1).is_some();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_FIXED_KERNEL").is_some() {
            assert!(qualified, "pinned fixed invocation layout must qualify");
        }
        if !qualified {
            return;
        }
        let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
        let input =
            Array::try_from_slice(&[-3.0f32, -0.25, 0.5, 2.0, 4.0, 7.0, 9.0, 11.0], &[8]).unwrap();
        let kernel = MetalKernel::new("fixed_config_pair", ["input"], ["first", "second"],
            "uint i=thread_position_in_grid.x; first[i]=T(input[i]*GAIN); second[i]=FLIP ? -input[i] : input[i];",
            "", true, false).unwrap();
        let owned = CustomKernelConfig::new()
            .with_template_arg_dtype("T", Dtype::Float32)
            .with_template_arg_int("GAIN", 3)
            .with_template_arg_bool("FLIP", true)
            .with_grid([8, 1, 1])
            .with_thread_group([8, 1, 1])
            .with_output_arg([8], Dtype::Float32)
            .with_output_arg([8], Dtype::Float32);
        let reference = kernel.apply_device([&input], &owned, &stream).unwrap();
        let outputs = [BorrowedKernelOutput {
            shape: &[8],
            dtype: Dtype::Float32,
        }; 2];
        let templates = [
            BorrowedKernelTemplate::Dtype(c"T", Dtype::Float32),
            BorrowedKernelTemplate::Int(c"GAIN", 3),
            BorrowedKernelTemplate::Bool(c"FLIP", true),
        ];
        let actual = kernel
            .apply_fixed_device([&input], outputs, &templates, [8, 1, 1], [8, 1, 1], &stream)
            .unwrap();
        for (actual, expected) in actual.iter().zip(&reference) {
            assert_eq!(
                crate::array::eval_vec::<f32>(actual),
                crate::array::eval_vec::<f32>(expected)
            );
        }
        // The rejected call leaves this kernel/configuration usable, and does
        // not partially return an output array or enter source generation.
        let too_many = [BorrowedKernelTemplate::Int(c"N", 1); 5];
        assert!(kernel
            .apply_fixed_device([&input], outputs, &too_many, [8, 1, 1], [8, 1, 1], &stream)
            .is_err());
        let again = kernel
            .apply_fixed_device([&input], outputs, &templates, [8, 1, 1], [8, 1, 1], &stream)
            .unwrap();
        assert_eq!(
            crate::array::eval_vec::<f32>(&again[0]),
            &[-9.0, -0.75, 1.5, 6.0, 12.0, 21.0, 27.0, 33.0]
        );
    }
}
