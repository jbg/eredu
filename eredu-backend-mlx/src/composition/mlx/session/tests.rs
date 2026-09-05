use crate::backend::ExecutionContext;
use safemlx::{Array, Device, DeviceType};

use crate::backend::nn::tensor::validate_token_domain;

use super::*;

#[test]
fn owned_input_preserves_multimodal_parts_and_metadata() {
    let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
    let image = Array::from_slice(&[0.0_f32; 8], &[1, 2, 4]);
    let grid = Array::from_slice(&[1_i32, 2, 2], &[1, 3]);
    let parts = [
        input::input_part(
            InputModality::Text,
            input::InputPayload::TokenIds(tokens),
            [],
            [],
        )
        .unwrap(),
        input::input_part(
            InputModality::Image,
            input::InputPayload::Tensor(image),
            [(eredu_core::InputMetadataKey::PatchGrid, grid)],
            [],
        )
        .unwrap(),
    ];

    let owned = MlxModelInput::from(input::ModelInput::new(&parts));
    owned.with_borrowed(|borrowed| {
        assert_eq!(borrowed.parts.len(), 2);
        assert_eq!(borrowed.parts[0].modality(), InputModality::Text);
        assert_eq!(borrowed.parts[1].modality(), InputModality::Image);
        assert_eq!(
            borrowed.parts[1]
                .metadata_value(eredu_core::InputMetadataKey::PatchGrid)
                .expect("grid metadata")
                .shape(),
            &[1, 3]
        );
    });
}

#[test]
fn model_session_is_the_backend_session_implementation() {
    fn assert_session<T: BackendSession<MlxBackend<'static>>>() {}
    fn assert_inspectable<T: InspectableBackendSession<MlxBackend<'static>>>() {}
    assert_session::<MlxModelSession>();
    assert_inspectable::<MlxModelSession>();
}

#[test]
fn completed_model_submission_validates_tokens_and_releases_its_gate() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let submit = |token| {
        let scope = TokenValidationScope::begin().unwrap();
        validate_token_domain(&Array::from_int(token), 4, None, stream).unwrap();
        let validations = scope.finish();
        safemlx::transforms::async_eval_with_event(validations.arrays())
            .unwrap()
            .synchronize()
            .unwrap();
        let gate = Rc::new(Cell::new(Some(1)));
        let submission = model_submission(
            Array::from_int(0),
            validations,
            true,
            SessionSubmissionLease {
                owner: gate.clone(),
                ticket: 1,
            },
        );
        (submission, gate)
    };

    let (valid, valid_gate) = submit(3);
    valid.completion.wait().unwrap();
    assert_eq!(valid_gate.get(), None);
    valid_gate.set(Some(2));
    drop(valid.completion);
    assert_eq!(valid_gate.get(), Some(2));
    let (invalid, invalid_gate) = submit(4);
    let invalid = invalid.completion;
    let error = invalid.wait().unwrap_err();
    assert!(error.to_string().contains("outside 0..4"));
    assert_eq!(invalid_gate.get(), None);
    let error = invalid.is_complete().unwrap_err();
    assert!(error.to_string().contains("outside 0..4"));
}

#[test]
fn inspection_collector_filters_and_materializes_portable_values() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let requested = ObservationRequest::selected([eredu_core::ObservationSelector::Exact(
        "model.layers.1.output".into(),
    )]);
    let mut collector = InspectionCollector::new(&requested);
    collector.capture(
        "model.layers.0.output",
        &MlxTensor::from_array(Array::from_slice(&[1.0f32], &[1])),
    );
    collector.capture(
        "model.layers.1.output",
        &MlxTensor::from_array(Array::from_slice(&[2.0f32, 3.0], &[1, 2])),
    );
    let observations = collector.materialize(stream).unwrap();
    assert_eq!(observations.len(), 1);
    let Some(ObservationValue::Tensor(tensor)) = observations.get("model.layers.1.output") else {
        panic!("selected activation must be a tensor");
    };
    assert_eq!(tensor.shape(), [1, 2]);
    assert_eq!(tensor.data(), &TensorObservationData::F32(vec![2.0, 3.0]));
}
