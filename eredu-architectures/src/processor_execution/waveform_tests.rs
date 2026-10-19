use super::*;
use eredu_core::{MultimodalRequest, MultimodalSegment};
use std::convert::Infallible;

#[test]
fn muse_waveform_framing_preserves_samples_and_independent_padding() {
    let source = include_bytes!("../../../validation/fixtures/muse-spark/processor-config.json");
    let plan = MuseProcessorPlan::from_hf_json(source).unwrap();
    let first: Vec<f32> = (0..1921)
        .map(|index| index as f32 / 4096.0 - 0.25)
        .collect();
    let second = vec![-0.375; 1920];
    let request = MultimodalRequest::new(vec![
        MultimodalSegment::Media(Media::Audio(
            eredu_core::Audio::new(first.clone(), 24000).unwrap(),
        )),
        MultimodalSegment::Media(Media::Audio(
            eredu_core::Audio::new(second.clone(), 24000).unwrap(),
        )),
    ])
    .unwrap()
    .tokenize::<Infallible>(|_| unreachable!())
    .unwrap();
    let encode = &mut |text: &str| {
        Ok::<_, Infallible>(vec![match text {
            "<|soft_audio_start|>" => 17,
            "<|soft_audio_end|>" => 18,
            _ => panic!("unexpected framing text {text}"),
        }])
    };
    let prepared = prepare_muse(&plan, &request, &mut HostProcessorMechanisms, encode).unwrap();
    assert_eq!(prepared.len(), 6);
    for (index, expected) in [(0, 17), (2, 18), (3, 17), (5, 18)] {
        assert_eq!(
            prepared.parts()[index].payload().value(),
            &HostTensor::U32 {
                values: vec![expected],
                shape: vec![1, 1]
            }
        );
    }
    let first_part = &prepared.parts()[1];
    let HostTensor::F32 { values, shape } = first_part.payload().value() else {
        panic!("waveform must remain floating PCM")
    };
    assert_eq!(shape, &[1, 3840]);
    assert_eq!(&values[..1921], first);
    assert!(values[1921..].iter().all(|value| *value == 0.0));
    assert_eq!(first_part.extents(), [InputExtent::AudioValidSamples(1921)]);
    assert_eq!(
        prepared.parts()[4].payload().value(),
        &HostTensor::F32 {
            values: second,
            shape: vec![1, 1920]
        }
    );
    assert_eq!(
        prepared.parts()[4].extents(),
        [InputExtent::AudioValidSamples(1920)]
    );
    let wrong_rate = MultimodalRequest::new(vec![MultimodalSegment::Media(Media::Audio(
        eredu_core::Audio::new(vec![0.25; 1920], 16000).unwrap(),
    ))])
    .unwrap()
    .tokenize::<Infallible>(|_| unreachable!())
    .unwrap();
    assert!(prepare_muse(&plan, &wrong_rate, &mut HostProcessorMechanisms, encode).is_err());
}
