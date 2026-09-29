use super::*;
use eredu_checkpoint::{AffineQuantization, BlockFp8Format};
use safemlx::{Device, DeviceType};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy)]
enum Stop {
    Error,
    Panic,
}
struct Observer {
    stop: Option<Stop>,
    calls: usize,
}
impl NativeProjectionInputObserver for Observer {
    fn observe(&mut self, input: &Array) -> Result<(), Exception> {
        self.calls += 1;
        assert_eq!(input.shape(), &[2, 32]);
        match self.stop {
            Some(Stop::Error) => Err(Exception::custom("observer refused input")),
            Some(Stop::Panic) => panic!("observer unwound"),
            None => Ok(()),
        }
    }
    fn observe_generated(
        &mut self,
        _: &Array,
        _: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception> {
        self.observe(&generate()?)
    }
}

fn formats() -> [LinearFormat; 3] {
    [
        LinearFormat::Dense,
        LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
        LinearFormat::E4M3BlockFp8(
            BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
        ),
    ]
}

fn bound(format: LinearFormat, stream: &Stream) -> PhysicalLinear {
    let mut linear = PhysicalLinear::unloaded(32, 2, true, format, stream).unwrap();
    let weight = Array::from_slice(&[vec![0.25f32; 32], vec![-0.5; 32]].concat(), &[2, 32]);
    match format {
        LinearFormat::Dense => linear.weight.value = weight,
        LinearFormat::Affine(config) => {
            let packed = safemlx::ops::quantize_with_mode(
                &weight,
                config.group_size,
                config.bits,
                QuantizationMode::Affine,
                stream,
            )
            .unwrap();
            linear.weight.value = packed.weight;
            linear.scales.value = Some(packed.scales);
            linear.biases.value = packed.biases;
        }
        LinearFormat::E4M3BlockFp8(_) => {
            linear.weight.value = weight.to_fp8(stream).unwrap();
            linear.weight_scale_inv.value = Some(Array::from_slice(&[1.0f32], &[1, 1]));
        }
        _ => unreachable!(),
    }
    linear.bias.value = Some(Array::from_slice(&[0.75f32, -1.25], &[2]));
    linear
}

#[test]
fn row_parallel_observer_error_or_unwind_preserves_bias_and_retry() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = crate::backend::runtime::distributed::Group::uncontracted(&native);
    let input = Array::from_slice(&[vec![1.0f32; 32], vec![2.0; 32]].concat(), &[2, 32]);
    for format in formats() {
        for stop in [Stop::Error, Stop::Panic] {
            let mut linear = bound(format, &stream);
            let bias_id = linear.bias.as_ref().as_ref().unwrap().graph_identity();
            let weight_id = linear.weight.value.graph_identity();
            let mut observer = Observer {
                stop: Some(stop),
                calls: 0,
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                linear.forward_row_parallel_with_input_observer(
                    &input,
                    &group,
                    &stream,
                    Some(&mut observer),
                )
            }));
            match stop {
                Stop::Error => assert!(result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("observer refused input")),
                Stop::Panic => assert_eq!(
                    *result.unwrap_err().downcast::<&'static str>().unwrap(),
                    "observer unwound"
                ),
            }
            assert_eq!(observer.calls, 1);
            assert_eq!(
                linear.bias.as_ref().as_ref().unwrap().graph_identity(),
                bias_id
            );
            assert_eq!(linear.weight.value.graph_identity(), weight_id);
            let mut observer = Observer {
                stop: None,
                calls: 0,
            };
            let retried = linear
                .forward_row_parallel_with_input_observer(
                    &input,
                    &group,
                    &stream,
                    Some(&mut observer),
                )
                .unwrap();
            assert_eq!(observer.calls, 1);
            let expected = [8.75f32, -17.25, 16.75, -33.25];
            assert_eq!(
                retried.evaluated().unwrap().as_slice::<f32>(),
                &expected,
                "{format:?}"
            );
            let ordinary = linear.forward(&input, &stream).unwrap();
            assert_eq!(ordinary.evaluated().unwrap().as_slice::<f32>(), &expected);
        }
    }
}

#[test]
fn ordinary_and_row_parallel_observers_apply_bias_once_or_leave_absence_unchanged() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = crate::backend::runtime::distributed::Group::uncontracted(&native);
    let input = Array::from_slice(&[1.0f32; 64], &[2, 32]);
    for format in formats() {
        let mut linear = bound(format, &stream);
        for biased in [true, false] {
            if !biased {
                linear.bias.value = None;
            }
            let expected = if biased {
                [8.75, -17.25, 8.75, -17.25]
            } else {
                [8.0, -16.0, 8.0, -16.0]
            };
            let mut observer = Observer {
                stop: None,
                calls: 0,
            };
            let ordinary = linear
                .forward_with_input_observer(&input, &stream, Some(&mut observer))
                .unwrap();
            let parallel = linear
                .forward_row_parallel_with_input_observer(
                    &input,
                    &group,
                    &stream,
                    Some(&mut observer),
                )
                .unwrap();
            assert_eq!(observer.calls, 2);
            assert_eq!(
                ordinary.evaluated().unwrap().as_slice::<f32>(),
                &expected,
                "{format:?}"
            );
            assert_eq!(
                parallel.evaluated().unwrap().as_slice::<f32>(),
                &expected,
                "{format:?}"
            );
            assert_eq!(linear.bias.value.is_some(), biased);
        }
    }
}
