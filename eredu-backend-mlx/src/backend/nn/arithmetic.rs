//! Native FP32 pointwise arithmetic with explicit exponential rounding.
use safemlx::{error::Exception, Array, Stream};
#[derive(Clone, Copy)]
pub(crate) enum Pointwise {
    Sigmoid = 1,
    Silu = 2,
}
// Invocation and lazy compiler transports. Immutable source/cache slots are
// separately funded once by the prepared pointwise family.
pub(crate) fn f32_pointwise_control_bytes(rank: usize) -> Option<usize> {
    use std::mem::size_of;
    [
        Stream::device_type_control_bytes()?,
        size_of::<Option<safemlx::fast::CustomKernelConfig>>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<std::cell::Ref<'static, Option<safemlx::fast::MetalKernel>>>(),
        size_of::<std::cell::RefMut<'static, Option<safemlx::fast::MetalKernel>>>(),
        size_of::<Pointwise>(),
        size_of::<i32>(),
        size_of::<bool>(),
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        size_of::<
            Option<
                &safemlx::fast::PreparedMetalKernelFamily<
                    eredu_runtime::working_memory::SharedNativeInitializationCustody,
                >,
            >,
        >(),
    ]
    .into_iter()
    .try_fold(
        safemlx::fast::PreparedMetalKernelFamily::<
            eredu_runtime::working_memory::SharedNativeInitializationCustody,
        >::control_bytes::<1, 1>(1, rank)?,
        usize::checked_add,
    )
}
// CPU exits before native kernel/TLS/source construction. Quote only this
// actual probe, never use GPU kernel storage to qualify the CPU equation.
pub(crate) fn cpu_pointwise_probe_control_bytes() -> Option<usize> {
    use std::mem::{size_of,size_of_val};
    let parts=[Stream::device_type_control_bytes()?,
        size_of::<(&Array,Pointwise,&Stream)>(),size_of::<Result<Option<Array>,Exception>>(),
        size_of::<safemlx::Dtype>(),size_of::<usize>(),size_of::<bool>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
pub(crate) fn f32_pointwise(
    input: &Array,
    operation: Pointwise,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || stream.device_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let size = i32::try_from(input.size())
            .map_err(|_| Exception::custom("pointwise size overflow"))?;
        let fixed = MetalKernel::fixed_control_bytes::<1, 1>(1, input.ndim()).is_some();
        let original = safemlx::OriginalScopeObserver::try_current()?;
        if let Some(observer) = &original {
            if !fixed || crate::backend::managed_memory::pointwise_kernel::definition().is_none() {
                return Err(observer.capacity_error());
            }
        }
        if fixed {
            if let Some(kernel) = crate::backend::managed_memory::pointwise_kernel::definition() {
                use safemlx::fast::{BorrowedKernelOutput, BorrowedKernelTemplate};
                let [output] = kernel.apply_fixed_device(
                    [input],
                    [BorrowedKernelOutput {
                        shape: input.shape(),
                        dtype: Dtype::Float32,
                    }],
                    &[BorrowedKernelTemplate::Int(c"OP", operation as i32)],
                    [size, 1, 1],
                    [256, 1, 1],
                    stream,
                )?;
                return Ok(Some(output));
            }
        }
        // Arbitrary ordinary ranks keep their historical owning config. The
        // closed inline route borrows the shape and static template names.
        let config = (!fixed).then(|| {
            CustomKernelConfig::new()
                .with_template_arg_int("OP", operation as i32)
                .with_grid([size, 1, 1])
                .with_thread_group([256, 1, 1])
                .with_output_arg(input.shape(), Dtype::Float32)
        });
        return KERNEL.with(|cell| -> Result<Option<Array>, Exception> {
            if cell.borrow().is_none() {
                let plan = &crate::backend::managed_memory::pointwise_kernel::PLAN;
                *cell.borrow_mut() = Some(MetalKernel::new(
                    plan.name,
                    plan.inputs,
                    plan.outputs,
                    plan.source,
                    plan.header,
                    plan.ensure_row_contiguous,
                    plan.atomic_outputs,
                )?);
            }
            let kernel = cell.borrow();
            let kernel = kernel.as_ref().expect("pointwise kernel initialized");
            if let Some(config) = &config {
                kernel
                    .apply_device([input], config, stream)
                    .map(|mut outputs| outputs.pop())
            } else {
                use safemlx::fast::{BorrowedKernelOutput, BorrowedKernelTemplate};
                let [output] = kernel.apply_fixed_device(
                    [input],
                    [BorrowedKernelOutput {
                        shape: input.shape(),
                        dtype: Dtype::Float32,
                    }],
                    &[BorrowedKernelTemplate::Int(c"OP", operation as i32)],
                    [size, 1, 1],
                    [256, 1, 1],
                    stream,
                )?;
                Ok(Some(output))
            }
        });
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, operation, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};
    #[test]
    fn native_sigmoid_and_coefficients_match_independent_rounding() {
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let fixture = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/f32_sigmoid.safetensors"
            ),
            &weights,
        )
        .unwrap();
        let scores = f32_pointwise(&fixture["input"], Pointwise::Sigmoid, &stream)
            .unwrap()
            .unwrap();
        let selected =
            safemlx::ops::indexing::take_along_axis(&scores, &fixture["ids"], -1, &stream).unwrap();
        let denominator = crate::backend::nn::normalization::f32_sum_last(&selected, &stream)
            .unwrap()
            .unwrap();
        let coefficients = selected
            .divide(denominator, &stream)
            .unwrap()
            .multiply(Array::from_f32(2.5), &stream)
            .unwrap()
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        for (name, values) in [("scores", scores), ("coefficients", coefficients)] {
            let output = values
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let expected = fixture[name]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let differences = output
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differences, 0, "{name}");
        }
    }
}

pub(crate) fn softmax_last(input: &Array, stream: &Stream) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use crate::backend::managed_memory::row_kernels::{self, RowKernel};
        use safemlx::{DeviceType, Dtype};
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || input.ndim() == 0
            || input.dim(-1) < 4
            || stream.device_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        row_kernels::validate_call(RowKernel::Softmax, input.ndim())?;
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("softmax row count overflow"))?;
        row_kernels::apply(RowKernel::Softmax, [input], input.shape(), rows, stream).map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, stream);
        Ok(None)
    }
}
