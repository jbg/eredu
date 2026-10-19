//! Released family acceptance against external pinned references.
use super::*;
use crate::composition::mlx::MlxBackend;
use eredu_core::{ModelRuntime, TextGenerationBackend};
use safemlx::Dtype;

fn execution_streams() -> (Stream, Stream) {
    let execution_device = if cfg!(feature = "metal") {
        safemlx::Device::new(safemlx::DeviceType::Gpu, 0)
    } else {
        safemlx::Device::new(safemlx::DeviceType::Cpu, 0)
    };
    let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    (
        Stream::new_with_device(&execution_device),
        Stream::new_with_device(&weights_device),
    )
}

/// Runs only with the pinned, externally stored release and independent fixture.
#[test]
#[ignore = "requires the pinned 50 GB released checkpoint and independent reference"]
fn released_diffusion_public_loader_matches_cached_reference() {
    released_public_loader_reference(false);
}

#[test]
#[ignore = "requires the pinned published GGUF and independent dequantized reference"]
fn released_diffusion_gguf_public_loader_matches_cached_reference() {
    released_public_loader_reference(true);
}

fn released_public_loader_reference(gguf: bool) {
    use eredu_nn::Tensor;
    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_DIFFUSION_CHECKPOINT").expect("checkpoint path"),
    );
    let reference_path = std::path::PathBuf::from(
        std::env::var_os("EREDU_DIFFUSION_REFERENCE").expect("reference path"),
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("source_manifest.json")).unwrap()).unwrap();
    let reference_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_path.with_extension("json")).unwrap())
            .unwrap();
    assert_eq!(manifest["id"], "google/diffusiongemma-26B-A4B-it");
    assert_eq!(manifest["sha"], "f7f5b7f5fa82ffc52addd066915886d497f5517b");
    assert_eq!(reference_manifest["model_revision"], manifest["sha"]);
    assert_eq!(
        reference_manifest["reference_revision"],
        "55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e"
    );
    let (stream, weights) = execution_streams();
    let reference = Array::load_safetensors(reference_path, &stream).unwrap();
    let backend = crate::native::backend(&stream, &weights);
    let request = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                Default::default(),
            ),
        ),
    );
    let path = if gguf {
        std::path::PathBuf::from(std::env::var_os("EREDU_DIFFUSION_GGUF").expect("GGUF path"))
    } else {
        root
    };
    if gguf {
        assert_eq!(
            reference_manifest["quantized_source"]["sha256"],
            "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
        );
    }
    let model = eredu_core::load_model(&backend, &path, request).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    println!("Loaded pinned released checkpoint through public admission and native construction");
    fn compare(
        label: &str,
        actual: &Array,
        expected: &Array,
        stream: &Stream,
        tolerance: f32,
        relative_rms_limit: Option<f64>,
    ) {
        assert_eq!(actual.shape(), expected.shape(), "{label}");
        let actual = actual
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .into_evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = expected
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .into_evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let maximum = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let reference_rms = (expected.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        let relative_rms = rms / reference_rms.max(f64::MIN_POSITIVE);
        println!(
            "{label}: max_absolute_error={maximum} rms_error={rms} relative_rms_error={relative_rms}"
        );
        assert!(actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert!(maximum <= tolerance, "{label}: {maximum}");
        if label.starts_with("logits.") || label.starts_with("shared_feedback.") {
            let width = expected.len() / 2;
            let argmax = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            for (actual, expected) in actual.chunks_exact(width).zip(expected.chunks_exact(width)) {
                assert_eq!(
                    argmax(actual),
                    argmax(expected),
                    "{label}: changed prediction"
                );
            }
        }
        if let Some(limit) = relative_rms_limit {
            assert!(
                relative_rms <= limit,
                "{label}: relative rms={relative_rms}"
            );
        }
    }
    let cases: [(&[u32], &[i32]); 3] = [
        (&[2, 100, 200], &[101, 211]),
        (&[17], &[421, 653]),
        (&[31, 29], &[1019, 2047]),
    ];
    let mut feedback = None;
    for (step, (tokens, canvas)) in cases.into_iter().enumerate() {
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), tokens.to_vec()).unwrap();
        let encoded = runtime
            .prefill(prompt)
            .unwrap()
            .wait()
            .unwrap()
            .into_logits()
            .unwrap();
        compare(
            &format!("encode.{step}"),
            encoded.as_array(),
            &reference[&format!("encode.{step}")],
            &stream,
            if gguf { 1e-3 } else { 2e-5 },
            None,
        );
        let canvas = Array::from_slice(canvas, &[1, canvas.len() as i32]);
        if gguf && step == 1 {
            let shared_feedback = &reference["prior.0"];
            let probe = runtime
                .session_mut()
                .with_model_operation(|model| {
                    let output = model.erased_mut().attention_forward_with_observer(
                        &canvas,
                        Some(shared_feedback),
                        eredu_core::decoding::AttentionInvocation::DIFFUSION,
                        None,
                        false,
                        &stream,
                        &mut eredu_runtime::NoopObserver,
                    )?;
                    output.evaluated()?;
                    Ok(output)
                })
                .unwrap();
            compare(
                "shared_feedback.1",
                &probe,
                &reference["logits.1"],
                &stream,
                1e-2,
                Some(2e-4),
            );
        }
        let decoded = runtime
            .session_mut()
            .with_model_operation(|model| {
                let output = model.erased_mut().attention_forward_with_observer(
                    &canvas,
                    feedback.as_ref(),
                    eredu_core::decoding::AttentionInvocation::DIFFUSION,
                    None,
                    false,
                    &stream,
                    &mut eredu_runtime::NoopObserver,
                )?;
                output.evaluated()?;
                Ok(output)
            })
            .unwrap();
        compare(
            &format!("logits.{step}"),
            &decoded,
            &reference[&format!("logits.{step}")],
            &stream,
            if gguf { 1e-2 } else { 2e-5 },
            gguf.then_some(2e-4),
        );
        feedback = Some(
            MlxTensor::from_array(decoded)
                .divide(
                    &MlxTensor::from_f32_slice(&[0.8], &[1], &stream).unwrap(),
                    &stream,
                )
                .unwrap()
                .cast_float(
                    if gguf {
                        eredu_nn::TensorElementType::F32
                    } else {
                        eredu_nn::TensorElementType::Bf16
                    },
                    &stream,
                )
                .unwrap()
                .into_array(),
        );
        assert_eq!(
            runtime
                .session_mut()
                .with_model_operation(|model| Ok(model.erased().forecast_state_offset()?))
                .unwrap(),
            Some([3, 4, 6][step])
        );
    }
}

mod nemotron;
