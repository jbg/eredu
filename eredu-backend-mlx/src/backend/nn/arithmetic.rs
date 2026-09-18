//! Native FP32 pointwise arithmetic with explicit exponential rounding.
use safemlx::{Array, Stream, error::Exception};
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
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<[i32; 1]>(),
        size_of::<safemlx::fast::BorrowedKernelOutput<'static>>(),
        size_of::<safemlx::fast::BorrowedKernelTemplate<'static>>(),
        size_of::<[i32; 3]>() * 2,
        size_of::<[Array; 1]>(),
        size_of::<Result<[Array; 1], Exception>>(),
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
    use std::mem::{size_of, size_of_val};
    let parts = [
        Stream::device_type_control_bytes()?,
        size_of::<(&Array, Pointwise, &Stream)>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<safemlx::Dtype>(),
        size_of::<usize>(),
        size_of::<bool>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(crate) fn f32_pointwise(
    input: &Array,
    operation: Pointwise,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        if input.dtype() != safemlx::Dtype::Float32
            || input.size() == 0
            || stream.device_type()? != safemlx::DeviceType::Gpu
        {
            return Ok(None);
        }
        let size = i32::try_from(input.size())
            .map_err(|_| Exception::custom("pointwise size overflow"))?;
        crate::backend::managed_memory::pointwise_kernel::apply(
            input,
            operation as i32,
            size,
            stream,
        )
        .map(Some)
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
    fn native_high_rank_pointwise_and_rows_preserve_logical_shape() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mut shape = [1; 33];
        shape[32] = 4;
        let input = Array::try_from_slice(&[-2.0_f32, -0.5, 1.0, 3.0], &shape).unwrap();
        let sigmoid = f32_pointwise(&input, Pointwise::Sigmoid, &stream)
            .unwrap()
            .unwrap();
        assert_eq!(sigmoid.shape(), &shape);
        let values = sigmoid.evaluated().unwrap();
        for (&actual, x) in values
            .as_slice::<f32>()
            .iter()
            .zip([-2.0_f64, -0.5, 1.0, 3.0])
        {
            assert!((f64::from(actual) - 1.0 / (1.0 + (-x).exp())).abs() < 1e-6);
        }
        let sum = super::super::normalization::f32_sum_last(&input, &stream)
            .unwrap()
            .unwrap();
        assert_eq!(sum.shape(), &[1; 33]);
        assert_eq!(sum.evaluated().unwrap().as_slice::<f32>(), &[1.5]);
        let rms = super::super::normalization::f32_weightless_rms(&input, 0.25, &stream)
            .unwrap()
            .unwrap();
        assert_eq!(rms.shape(), &shape);
        for (&actual, x) in rms
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .zip([-2.0_f32, -0.5, 1.0, 3.0])
        {
            assert!((actual - x / (14.25_f32 / 4.0 + 0.25).sqrt()).abs() < 1e-6);
        }
    }
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
