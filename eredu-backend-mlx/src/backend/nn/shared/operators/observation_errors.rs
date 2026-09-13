use super::*;
use common::linear::NativeProjectionInputObserver;
use std::error::Error as _;

struct Generate;
impl eredu_nn::ProjectionInputObserver<MlxTensor> for Generate {
    fn observe(&mut self, _: &MlxTensor) -> Result<(), ComputeError> {
        Ok(())
    }
    fn observe_generated(
        &mut self,
        _: &MlxTensor,
        _: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, ComputeError>,
    ) -> Result<(), ComputeError> {
        generate().map(|_| ())
    }
}

#[test]
fn deferred_native_factory_failure_retains_the_original_exception() {
    let stream = crate::test_stream();
    let input = Array::from_slice(&[1.25f32, -2.5], &[1, 2]);
    let original = input.reshape(&[3], stream).unwrap_err();
    let expected = (original.what().to_owned(), original.location());
    let mut original = Some(original);
    let mut observer = NativeInputObserver {
        inner: &mut Generate,
        failure: None,
    };
    let source = eredu_nn::GeneratedTensorSource {
        creation_bytes: 64,
        element_type: Some(eredu_nn::TensorElementType::F32),
    };
    let result = observer.observe_generated(&input, &source, &mut || Err(original.take().unwrap()));
    assert!(result.is_err());
    assert!(original.is_none());
    let retained = observer
        .failure
        .take()
        .expect("factory failure reaches the neural caller");
    let cause = retained
        .source()
        .unwrap()
        .downcast_ref::<Exception>()
        .unwrap();
    assert_eq!(cause.what(), expected.0);
    assert_eq!(cause.location(), expected.1);
}

#[test]
fn native_neural_operation_failure_retains_the_original_exception() {
    let stream = crate::test_stream();
    let input = Array::from_slice(&[1.25f32, -2.5], &[1, 2]);
    let original = input.reshape(&[3], stream).unwrap_err();
    let expected = (original.what().to_owned(), original.location());
    let retained = compute_tensor(Err(original)).unwrap_err();
    let cause = retained
        .source()
        .unwrap()
        .downcast_ref::<Exception>()
        .unwrap();
    assert_eq!(cause.what(), expected.0);
    assert_eq!(cause.location(), expected.1);
}
