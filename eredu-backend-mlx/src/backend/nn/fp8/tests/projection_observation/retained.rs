use super::*;
use eredu_nn::{GeneratedTensorProgram, GeneratedTensorSourceRole, RetainedGeneratedTensorFactory};
struct Retaining {
    selected: bool,
    stop: Option<usize>,
    roots: Vec<Array>,
    outputs: usize,
}
impl NativeProjectionInputObserver for Retaining {
    fn observe(&mut self, _: &Array) -> Result<(), Exception> {
        panic!("actual GPU FP8 generated program")
    }
    fn observe_generated(
        &mut self,
        _: &Array,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception> {
        panic!("retained factory was erased")
    }
    fn observe_generated_retained(
        &mut self,
        prototype: &Array,
        source: &eredu_nn::GeneratedTensorSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<Array, Exception>,
    ) -> Result<(), Exception> {
        if !self.selected {
            return Ok(());
        }
        assert_eq!(source.element_type, Some(eredu_nn::TensorElementType::F32));
        let GeneratedTensorProgram::BlockFp8Input(plan) = factory.program();
        assert_eq!(plan.shape(), prototype.shape());
        let (values_shape, scales_shape) = (plan.values_shape(), plan.scales_shape());
        factory.visit_sources(&mut |role, value| {
            match role {
                GeneratedTensorSourceRole::CompactValues => {
                    assert_eq!(value.dtype(), Dtype::Uint8);
                    assert_eq!(value.shape(), values_shape);
                }
                GeneratedTensorSourceRole::BlockScales => {
                    assert_eq!(value.dtype(), Dtype::Float32);
                    assert_eq!(value.shape(), scales_shape);
                }
            }
            self.roots.push(value.clone());
            Ok(())
        })?;
        let output = factory.generate(&mut |value| {
            self.outputs += 1;
            self.roots.push(value.clone());
            if self.stop == Some(self.outputs) {
                Err(Exception::custom("retained output sentinel"))
            } else {
                Ok(())
            }
        })?;
        assert_eq!(output.shape(), prototype.shape());
        assert_eq!(output.dtype(), Dtype::Float32);
        Ok(())
    }
}
#[test]
fn actual_native_roots_survive_original_operator_and_selected_factory_failure() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    for stop in [None, Some(1), Some(4), Some(7)] {
        let width = 259;
        let input = Array::from_slice(
            &(0..2 * width)
                .map(|i| ((i * 17 % 53) as f32 - 26.0) / 9.0)
                .collect::<Vec<_>>(),
            &[1, 2, width],
        )
        .as_dtype(Dtype::Bfloat16, stream)
        .unwrap();
        let original = input
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let mut module = module(width, true, stream);
        let mut observer = Retaining {
            selected: true,
            stop,
            roots: vec![],
            outputs: 0,
        };
        let result = module.forward_with_input_observer(&input, stream, Some(&mut observer));
        assert_eq!(result.is_err(), stop.is_some());
        assert_eq!(observer.outputs, stop.unwrap_or(7));
        assert_eq!(observer.roots.len(), 2 + stop.unwrap_or(7));
        drop(result);
        drop(module);
        drop(input);
        if stop.is_none() {
            let output = observer.roots.pop().unwrap();
            observer.roots.clear();
            let expected = host_fp8_input(&original, width as usize);
            let evaluated = output.evaluated().unwrap();
            let actual = evaluated.as_slice::<f32>();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() <= 1e-7 + b.abs() * 3e-7);
            }
        } else {
            // Earlier actual lazy roots still own their dependency graph even
            // after the operator frame and prototype have retired.
            for root in &observer.roots {
                root.evaluated().unwrap();
            }
        }
    }
    let input = Array::from_slice(&[1.0f32; 259], &[1, 259]);
    let mut module = module(259, true, stream);
    let mut observer = Retaining {
        selected: false,
        stop: Some(1),
        roots: vec![],
        outputs: 0,
    };
    module
        .forward_with_input_observer(&input, stream, Some(&mut observer))
        .unwrap()
        .evaluated()
        .unwrap();
    assert_eq!(observer.outputs, 0);
    assert!(observer.roots.is_empty());
}
