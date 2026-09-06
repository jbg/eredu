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
        let mut gate = SessionAuthority::new();
        let lease = gate.begin_submission().unwrap();
        let submission = model_submission(Array::from_int(0), validations, true, lease);
        (submission, gate)
    };

    let (valid, mut valid_gate) = submit(3);
    valid.completion.wait().unwrap();
    assert_eq!(valid_gate.require_idle(), Ok(()));
    let next = valid_gate.begin_submission().unwrap();
    drop(valid.completion);
    assert!(valid_gate.require_idle().is_err());
    drop(next);
    let (invalid, invalid_gate) = submit(4);
    let invalid = invalid.completion;
    let error = invalid.wait().unwrap_err();
    assert!(error.to_string().contains("outside 0..4"));
    assert_eq!(invalid_gate.require_idle(), Ok(()));
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

#[test]
fn text_completion_keeps_authority_through_pending_and_failed_native_observation() {
    use std::cell::Cell;

    struct NativeToken<'a> {
        authority: &'a SessionAuthority,
        state: Cell<u8>,
        waited: Cell<bool>,
        failure_is_terminal: bool,
    }
    impl Completion for NativeToken<'_> {
        type Error = Error;
        fn resources_releasable(&self) -> bool {
            self.state.get() == 1 || (self.state.get() == 2 && self.failure_is_terminal)
        }
        fn is_complete(&self) -> Result<bool, Error> {
            assert!(self.authority.require_idle().is_err());
            match self.state.get() {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(Error::ArchitectureModel("native token failure".into())),
            }
        }
        fn wait(&self) -> Result<(), Error> {
            assert!(self.authority.require_idle().is_err());
            self.waited.set(true);
            if self.state.get() == 2 {
                Err(Error::ArchitectureModel("native token failure".into()))
            } else {
                self.state.set(1);
                Ok(())
            }
        }
    }

    for (state, poll, terminal) in [
        (0, true, false),
        (1, true, false),
        (2, true, false),
        (0, false, false),
        (2, false, false),
        (2, true, true),
        (2, false, true),
    ] {
        let mut authority = SessionAuthority::new();
        let lease = authority.begin_submission().unwrap();
        let model = model_submission(
            Array::from_int(0),
            TokenValidationBatch::default(),
            true,
            lease,
        )
        .completion;
        let token = NativeToken {
            authority: &authority,
            state: Cell::new(state),
            waited: Cell::new(false),
            failure_is_terminal: terminal,
        };
        if poll {
            let result = output_completion::token_then_model_is_complete(&token, &model);
            match state {
                0 => {
                    assert!(!result.unwrap());
                    assert!(authority.require_idle().is_err());
                    assert!(!token.waited.get());
                    output_completion::token_then_model_wait(&token, &model).unwrap();
                }
                1 => assert!(result.unwrap()),
                _ => {
                    assert!(result.is_err());
                    assert!(
                        !token.waited.get(),
                        "polling failure must never start a cleanup wait"
                    );
                    assert!(authority.require_idle().is_err());
                    if terminal {
                        assert!(output_completion::token_then_model_wait(&token, &model).is_err());
                    }
                }
            }
        } else {
            let result = output_completion::token_then_model_wait(&token, &model);
            assert_eq!(result.is_err(), state == 2);
            assert!(token.waited.get());
        }
        if state == 2 && !terminal {
            assert!(authority.require_idle().is_err());
            assert!(output_completion::token_then_model_wait(&token, &model).is_err());
            assert!(authority.require_idle().is_err());
            token.state.set(1);
            output_completion::token_then_model_wait(&token, &model).unwrap();
        }
        assert_eq!(authority.require_idle(), Ok(()));
    }
}

#[test]
fn dropping_settled_text_completion_releases_model_authority() {
    let mut authority = SessionAuthority::new();
    let lease = authority.begin_submission().unwrap();
    let model = model_submission(
        Array::from_int(0),
        TokenValidationBatch::default(),
        true,
        lease,
    )
    .completion;
    let sampled = MlxCompletion::submission(Array::from_slice(&[17_u32], &[1])).unwrap();
    let completion = MlxTextCompletion {
        model,
        token: sampled.completion,
        recovery: std::cell::RefCell::new(None),
    };
    assert!(authority.require_idle().is_err());
    drop(completion);
    assert_eq!(authority.require_idle(), Ok(()));
    assert_eq!(sampled.output.evaluated().unwrap().as_slice::<u32>(), &[17]);
}
