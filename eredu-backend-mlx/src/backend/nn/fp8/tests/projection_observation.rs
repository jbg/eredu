use super::*;
use crate::backend::nn::linear::{NativeProjectionInputObserver, PhysicalLinear};
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use safemlx::error::Exception;

struct Observer {
    selected: bool,
    borrowed: usize,
    offered: usize,
    generated: usize,
    value: Option<Array>,
    source: Option<eredu_nn::GeneratedTensorSource>,
}
impl Observer {
    fn new(selected: bool) -> Self {
        Self {
            selected,
            borrowed: 0,
            offered: 0,
            generated: 0,
            value: None,
            source: None,
        }
    }
}
impl NativeProjectionInputObserver for Observer {
    fn observe(&mut self, input: &Array) -> Result<(), Exception> {
        self.borrowed += 1;
        self.value = Some(input.clone());
        Ok(())
    }
    fn observe_generated(
        &mut self,
        prototype: &Array,
        source: &eredu_nn::GeneratedTensorSource,
        create: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception> {
        self.offered += 1;
        self.source = Some(*source);
        assert_eq!(
            *source,
            eredu_nn::BlockFp8InputReconstructionPlan::new(prototype.shape())
                .unwrap()
                .logical_capture_source()
                .unwrap()
        );
        if self.selected {
            let value = create()?;
            self.generated += 1;
            assert_eq!(value.dtype(), Dtype::Float32);
            assert_eq!(value.shape(), prototype.shape());
            self.value = Some(value)
        }
        Ok(())
    }
}
fn module(width: i32, fp8: bool, stream: &safemlx::Stream) -> PhysicalLinear {
    let format = if fp8 {
        LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
        )
    } else {
        LinearFormat::Dense
    };
    let mut module = PhysicalLinear::unloaded(width, 2, false, format, stream).unwrap();
    if fp8 {
        module.weight.value = Array::from_slice(&vec![0x38u8; 2 * width as usize], &[2, width]);
        module.weight_scale_inv.value = Some(Array::from_slice(
            &vec![1.0f32; (width as usize).div_ceil(128)],
            &[1, (width as u64).div_ceil(128) as i32],
        ));
    } else {
        module.weight.value = Array::from_slice(&vec![1.0f32; 2 * width as usize], &[2, width]);
    }
    module
}
#[test]
fn native_dense_and_deferred_fp8_selection_preserve_values_dtype_and_skipped_work() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    for width in [1, 128, 259] {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let values = (0..2 * width)
                .map(|i| ((i * 17 % 53) as f32 - 26.0) / 9.0)
                .collect::<Vec<_>>();
            let input = Array::from_slice(&values, &[1, 2, width])
                .as_dtype(dtype, stream)
                .unwrap();
            let original = input
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            for fp8 in [false, true] {
                let mut module = module(width, fp8, stream);
                if !fp8 {
                    // Dense matmul promotes mixed operands. Match the fixture
                    // weights to the input so each precision is actually tested.
                    module.weight.value = module.weight.value.as_dtype(dtype, stream).unwrap();
                }
                let reference = module
                    .forward(&input, stream)
                    .unwrap()
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
                    .to_vec();
                for selected in [false, true] {
                    let mut observer = Observer::new(selected);
                    let output = module
                        .forward_with_input_observer(&input, stream, Some(&mut observer))
                        .unwrap();
                    assert_eq!(output.dtype(), dtype);
                    assert_eq!(
                        output
                            .as_dtype(Dtype::Float32, stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .as_slice::<f32>(),
                        reference
                    );
                    assert_eq!(observer.borrowed, usize::from(!fp8));
                    assert_eq!(observer.offered, usize::from(fp8));
                    assert_eq!(observer.generated, usize::from(fp8 && selected));
                    if let Some(captured) = observer.value {
                        let expected = if fp8 {
                            host_fp8_input(&original, width as usize)
                        } else {
                            original.clone()
                        };
                        let actual = captured
                            .as_dtype(Dtype::Float32, stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .as_slice::<f32>()
                            .to_vec();
                        for (a, b) in actual.iter().zip(expected) {
                            assert!((a - b).abs() <= 1e-7 + b.abs() * 3e-7, "{a} != {b}")
                        }
                    }
                }
            }
            assert_eq!(
                input
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>(),
                original
            );
        }
    }
}
#[test]
fn shared_reconstruction_output_keeps_real_quantized_dependencies_after_source_owners_retire() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let context = ExecutionContext::new(Device::new(device, 0));
        let stream = context.stream();
        let values = (0..2 * 259)
            .map(|i| ((i * 7 % 89) as f32 - 44.0) / 17.0)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&values, &[2, 259]);
        let quantized = super::super::quantize_activations(&input, 2, 259, stream).unwrap();
        let output =
            super::super::dequantize_activations(&quantized, &[1, 2, 259], stream).unwrap();
        drop(quantized);
        drop(input);
        let expected = host_fp8_input(&values, 259);
        assert_eq!(output.dtype(), Dtype::Float32);
        assert_eq!(output.shape(), [1, 2, 259]);
        let actual = output.evaluated().unwrap().as_slice::<f32>().to_vec();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= 1e-7 + b.abs() * 3e-7)
        }
    }
}

mod retained;
