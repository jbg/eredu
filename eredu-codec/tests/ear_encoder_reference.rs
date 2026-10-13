mod support {
    #[allow(dead_code)]
    pub mod numeric_backend;
}
use eredu_codec::{
    ear::{AudioEncoder, AudioEncoderConfig},
    ear_encoder::EncoderConfig,
    ear_spectral::SpectralConfig,
    prvq::{Geometry, ResidualQuantizer},
    spectral::PeriodicHannStft,
};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use serde_json::Value;
use support::numeric_backend::{
    activate, Context, FailAt, NumericTensor, ReferenceBackend as Backend,
};
// Evaluation owns this artifact; importing its crate would create a dev cycle.
const REFERENCE: &str = eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON;
fn tensor(v: &Value) -> NumericTensor {
    let shape = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    NumericTensor::f32(
        v["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_f64().unwrap() as f32)
            .collect(),
        &shape,
    )
}
fn geometry(v: &Value) -> EncoderConfig {
    let i = |key: &str| v[key].as_i64().unwrap() as i32;
    EncoderConfig {
        latent_size: i("latent_size"),
        base_hidden_size: i("base_hidden_size"),
        channel_mult: serde_json::from_value(v["channel_mult"].clone()).unwrap(),
        rates: serde_json::from_value(v["rates"].clone()).unwrap(),
        blocks_per_stage: i("num_blocks") as usize,
        kernel_size: i("kernel_size"),
        fft_size: i("n_fft"),
        groups: i("groups"),
    }
}
struct Bind<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(
            output.shape(),
            replacement.shape(),
            "{}",
            metadata.id.as_str()
        );
        *output = replacement;
    }
}
fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value, ctx: &Context) {
    assert_eq!(actual.shape(), tensor(expected).shape());
    let a = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let r = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (i, (value, expected)) in actual
        .to_f32_vec(ctx)
        .unwrap()
        .iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        let expected = expected.as_f64().unwrap() as f32;
        assert!(
            (value - expected).abs() <= a + r * expected.abs(),
            "{i}: {value} vs NVIDIA {expected}"
        );
    }
}
#[test]
fn ear_encoder_matches_original_spectral_silent_carrier_and_codes() {
    let fixture: Value = serde_json::from_str(REFERENCE).unwrap();
    let ctx = Context::new(FailAt::None);
    activate(&ctx);
    for case in fixture["spectral"].as_array().unwrap() {
        let stft = PeriodicHannStft::<NumericTensor>::new(
            case["n_fft"].as_i64().unwrap() as i32,
            case["hop_length"].as_i64().unwrap() as i32,
            &ctx,
        )
        .unwrap();
        let (real, imag) = stft.analyze(&tensor(&case["pcm"]), &ctx).unwrap();
        compare(&real, &case["real"], &fixture, &ctx);
        compare(&imag, &case["imag"], &fixture, &ctx);
    }
    for name in ["compact", "released_rates"] {
        let case = &fixture["encoders"][name];
        let config = geometry(&case["geometry"]);
        let mut encoder = AudioEncoder::<Backend>::new(
            AudioEncoderConfig {
                neural: config.clone(),
                spectral: SpectralConfig {
                    fft_size: config.fft_size,
                    hop_length: case["geometry"]["hop_length"].as_i64().unwrap() as i32,
                },
                quantizer: Geometry {
                    depth: 3,
                    cardinality: 5,
                    channels: 4,
                },
            },
            "tts_model.audio_codec",
            &ctx,
        )
        .unwrap();
        encoder
            .neural
            .visit_parameters_mut(&mut Bind(&case["parameters"]));
        let stft = PeriodicHannStft::<NumericTensor>::new(
            config.fft_size,
            case["geometry"]["hop_length"].as_i64().unwrap() as i32,
            &ctx,
        )
        .unwrap();
        let (real, imag) = stft.analyze(&tensor(&case["pcm"]), &ctx).unwrap();
        compare(&real, &case["spectrum"]["real"], &fixture, &ctx);
        compare(&imag, &case["spectrum"]["imag"], &fixture, &ctx);
        let packed = NumericTensor::concatenate(&[real, imag], 1, &ctx).unwrap();
        compare(&packed, &case["packed"], &fixture, &ctx);
        let latent = encoder.latent(&tensor(&case["pcm"]), &ctx).unwrap();
        compare(&latent, &case["latent"], &fixture, &ctx);
        if let Some(stages) = case["stages"].as_array() {
            let mut hidden = packed.clone();
            for (unit, expected) in encoder.neural.units().iter().zip(stages) {
                hidden = unit.forward(&hidden, &ctx).unwrap();
                compare(&hidden, expected, &fixture, &ctx);
            }
        }
        let mut quantizer = ResidualQuantizer::<Backend>::new(
            Geometry {
                depth: 3,
                cardinality: 5,
                channels: 4,
            },
            "tts_model.audio_codec.prvq",
            &ctx,
        )
        .unwrap();
        quantizer.visit_parameters_mut(&mut Bind(&case["parameters"]));
        let encoded = encoder
            .encode(&tensor(&case["pcm"]), &mut quantizer, &ctx)
            .unwrap();
        assert_eq!(
            encoded.to_i32_vec(&ctx).unwrap(),
            case["codes"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>()
        );
        compare(
            &quantizer.decode(&encoded, &ctx).unwrap(),
            &case["decoded"],
            &fixture,
            &ctx,
        );
        assert_eq!(
            case["pcm"]["shape"][2].as_i64().unwrap() / i64::from(latent.dim(1)),
            case["samples_per_code"].as_i64().unwrap()
        );
    }
}
