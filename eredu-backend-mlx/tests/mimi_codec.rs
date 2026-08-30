#![cfg(feature = "codec")]

use eredu_backend_mlx::{codec::mimi::load, native::ExecutionContext, MlxTensor};
use eredu_codec::AudioTokenizer;
use eredu_nn::Tensor;
use safemlx::{ops::indexing::TryIndexOp, transforms::eval, Array, Device, DeviceType};

#[test]
#[ignore = "requires EREDU_MIMI_PATH with a released Mimi safetensors checkpoint and Metal"]
fn local_mimi_checkpoint_encode_decode_smoke() {
    let path = std::env::var("EREDU_MIMI_PATH")
        .expect("EREDU_MIMI_PATH must point to a Mimi safetensors checkpoint");
    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let mut mimi = load(path, Some(8), stream).unwrap();
    let cfg = mimi.config();
    assert_eq!(cfg.codebooks, 8);
    assert_eq!(cfg.cardinality, 2_048);

    let codes = MlxTensor::from_array(Array::zeros::<i32>(&[1, 8, 2], stream).unwrap());
    let latent = mimi.decode_latent(&codes, stream).unwrap();
    assert_eq!(latent.shape(), &[1, 512, 2]);
    let recoded = mimi.encode_latent(&latent, stream).unwrap();
    assert_eq!(recoded.shape(), &[1, 8, 2]);
    let pcm = mimi.decode(&codes, stream).unwrap();
    assert_eq!(pcm.shape(), &[1, 1, 3840]);
    let alternate_codes = MlxTensor::from_array(Array::ones::<i32>(&[1, 8, 2], stream).unwrap());
    let alternate_pcm = mimi.decode(&alternate_codes, stream).unwrap();
    eval([pcm.as_array(), alternate_pcm.as_array()]).unwrap();
    stream.synchronize().unwrap();
    let pcm_values = pcm.as_array().evaluated().unwrap();
    let alternate_values = alternate_pcm.as_array().evaluated().unwrap();
    let difference = pcm_values
        .as_slice::<f32>()
        .iter()
        .zip(alternate_values.as_slice::<f32>())
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>();
    assert!(difference > 1e-3, "Mimi decode ignored token values");
    let encoded = mimi.encode(&pcm, stream).unwrap();
    assert_eq!(encoded.shape(), &[1, 8, 2]);

    // PyTorch Mimi oracle for x[n] = ((n mod 17) - 8) / 64. This catches
    // architecture drift that a shape-only checkpoint smoke test cannot.
    let parity_pcm = (0..7680)
        .map(|sample| ((sample % 17) as f32 - 8.0) / 64.0)
        .collect::<Vec<_>>();
    let parity_pcm = MlxTensor::from_array(
        Array::from_slice(&parity_pcm, &[1, 1, 7680])
            .copy(stream)
            .unwrap(),
    );
    let actual_codes = mimi.encode(&parity_pcm, stream).unwrap();
    let expected_codes = Array::from_slice(
        &[
            1049, 605, 1964, 1964, 74, 712, 712, 712, 1441, 1441, 1441, 1441, 1820, 1820, 1820,
            1820, 1711, 1711, 1711, 1711, 1386, 818, 818, 1418, 127, 755, 755, 127, 130, 1228,
            1228, 1115,
        ],
        &[1, 8, 4],
    )
    .copy(stream)
    .unwrap();
    assert!(
        actual_codes
            .as_array()
            .all_close(&expected_codes, 0.0, 0.0, None, stream)
            .unwrap()
            .item::<bool>(stream),
        "Mimi encode tokens differ from the released PyTorch checkpoint oracle"
    );

    mimi.reset_encode_state();
    let encoded_first = mimi
        .encode_step(
            &MlxTensor::from_array(
                pcm.as_array()
                    .try_index_device((.., .., 0..1920), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap()
        .expect("first PCM frame should encode to one Mimi frame");
    let encoded_second = mimi
        .encode_step(
            &MlxTensor::from_array(
                pcm.as_array()
                    .try_index_device((.., .., 1920..3840), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap()
        .expect("second PCM frame should encode to one Mimi frame");
    assert_eq!(encoded_first.shape(), &[1, 8]);
    assert_eq!(encoded_second.shape(), &[1, 8]);

    mimi.reset_decode_state();
    let first = mimi
        .decode_step(
            &MlxTensor::from_array(
                codes
                    .as_array()
                    .try_index_device((.., .., 0), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap();
    let second = mimi
        .decode_step(
            &MlxTensor::from_array(
                codes
                    .as_array()
                    .try_index_device((.., .., 1), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap();
    assert_eq!(first.shape(), &[1, 1, 1920]);
    assert_eq!(second.shape(), &[1, 1, 1920]);
    let streamed = MlxTensor::concatenate(&[first, second], 2, stream).unwrap();
    assert_eq!(streamed.shape(), pcm.shape());
}
