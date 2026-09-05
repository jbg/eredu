use eredu_core::Completion;
use safemlx::{Array, Device, DeviceType, Stream};

use super::{array_probability_at, MlxSpeculativeCompletion};

#[test]
fn probability_lookup_rejects_tokens_outside_the_i32_vocabulary_before_indexing() {
    let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
    let probabilities = Array::from_slice(&[0.25_f32, 0.75], &[1, 2]);

    let error = array_probability_at(&probabilities, u32::MAX, &stream).unwrap_err();
    assert!(error.to_string().contains("exceeds vocabulary size 2"));
}

#[test]
fn native_speculative_completion_retains_every_submitted_array_handle() {
    let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
    let logits = Array::from_slice(&[1.0_f32], &[1]);
    let capture = Array::from_slice(&[3.0_f32], &[1]);
    let completion = MlxSpeculativeCompletion::submit([&logits, &capture]).unwrap();
    drop(logits);
    drop(capture);

    assert_eq!(completion.retained().len(), 2);
    completion.wait().unwrap();
    assert!(completion.is_complete().unwrap());
    assert_eq!(completion.retained()[0].clone().item::<f32>(&stream), 1.0);
    assert_eq!(completion.retained()[1].clone().item::<f32>(&stream), 3.0);
}
