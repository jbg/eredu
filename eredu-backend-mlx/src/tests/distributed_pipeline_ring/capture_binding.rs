//! Native complete captures use retained target ownership and shared rollback.
use super::*;
use eredu_architectures::composite_execution::{
    ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation as Op,
};

#[test]
#[ignore = "spawns local CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_partitioned_target_capture_binding() {
    assert!(distributed::is_available(Backend::Ring));
    for (axes, residency) in [
        ("tp", WorkerResidency::FullyResident),
        ("pp", WorkerResidency::LayerwiseHost),
    ] {
        let checkpoint = tempfile::tempdir().unwrap();
        write_muse_glimmer_component_fixture(checkpoint.path(), false, false);
        let path = checkpoint.path().to_path_buf();
        run_ring_pipeline_processes(
            residency,
            FixtureFamily::MuseGlimmer,
            WorkerMode::OpaqueExternalTargetCapture,
            checkpoint,
            path,
            Some(axes),
        );
    }
}

#[test]
#[ignore = "spawns local CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_spark_partitioned_target_capture_binding() {
    assert!(distributed::is_available(Backend::Ring));
    for dtype in [Dtype::F32, Dtype::BF16] {
        for (axes, residency) in [
            ("tp", WorkerResidency::FullyResident),
            ("pp", WorkerResidency::LayerwiseHost),
            ("ep", WorkerResidency::DenseDiskStream),
            ("tp-pp-ep", WorkerResidency::DenseDiskStream),
        ] {
            let checkpoint = tempfile::tempdir().unwrap();
            execution_plan_target::write_spark_fixture(checkpoint.path(), dtype);
            let path = checkpoint.path().to_path_buf();
            run_ring_pipeline_processes(
                residency,
                FixtureFamily::MuseSpark,
                WorkerMode::OpaqueExternalTargetCapture,
                checkpoint,
                path,
                Some(axes),
            );
        }
    }
}

pub(super) fn verify(
    mut session: MlxModelSession,
    checkpoint: &Path,
    stream: &Stream,
    rank: usize,
) {
    let backend = crate::native::backend(stream, stream);
    let model = load_model(&backend, checkpoint, MlxLoadRequest::default())
        .unwrap()
        .into_inner();
    let mut reference = MlxModelSession::from_model(
        model,
        eredu_core::SessionCapabilities::new(true, true, true),
    )
    .unwrap();
    let target = session
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .unwrap();
    let ordinary = reference
        .neutral_prediction_target_mut()
        .unwrap()
        .external_prediction_mut()
        .unwrap();
    let mut cache = target.prepare_external_prediction_target_cache().unwrap();
    let mut reference_cache = ordinary.prepare_external_prediction_target_cache().unwrap();
    let request = ExternalPredictionCaptureRequest::Muse {
        target_layers: vec![0, 1].into_boxed_slice(),
        target_paths: vec![
            "model.layers.0.output".into(),
            "model.layers.1.output".into(),
        ]
        .into_boxed_slice(),
    };
    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [text_input_part(&tokens)];
    let input = || crate::backend::runtime::media::input::ModelInput::new(&parts);
    let actual = target
        .prefill_external_prediction_target(input(), &request, &mut cache)
        .unwrap();
    let expected = ordinary
        .prefill_external_prediction_target(input(), &request, &mut reference_cache)
        .unwrap();
    compare(actual, expected, true, stream);
    {
        let prefix = cache.generation().unwrap();
        let ids = MlxTensor::from_array(Array::from_slice(&[0u32, 31, 63], &[1, 3]));
        let embeddings = ordinary
            .apply_external_prediction_target_operation(Op::RawTokenEmbeddings(&ids))
            .unwrap();
        let width = usize::try_from(embeddings.as_array().shape()[2]).unwrap();
        let hidden = MlxTensor::from_array(Array::from_slice(
            &(0..2 * width)
                .map(|i| (i as f32 - 5.) * 0.37)
                .collect::<Vec<_>>(),
            &[1, 2, i32::try_from(width).unwrap()],
        ));
        for operation in [
            Op::RawTokenEmbeddings(&ids),
            Op::TokenEmbeddings(&ids),
            Op::RawVocabularyProjection(&hidden),
            Op::ProjectLogits(&hidden),
        ] {
            let expected = ordinary
                .apply_external_prediction_target_operation(operation)
                .unwrap();
            let actual = target
                .apply_external_prediction_target_operation(operation)
                .unwrap();
            close(&actual, &expected, true, stream);
            assert_eq!(cache.generation().unwrap(), prefix);
        }
        let invalid = MlxTensor::from_array(Array::from_slice(&[1u32], &[1, 1, 1]));
        assert!(target
            .apply_external_prediction_target_operation(Op::RawTokenEmbeddings(if rank == 1 {
                &invalid
            } else {
                &ids
            }))
            .is_err());
        assert_eq!(cache.generation().unwrap(), prefix);
        let actual = target
            .apply_external_prediction_target_operation(Op::RawTokenEmbeddings(&ids))
            .unwrap();
        let expected = ordinary
            .apply_external_prediction_target_operation(Op::RawTokenEmbeddings(&ids))
            .unwrap();
        close(&actual, &expected, true, stream);
    }
    let prefix = cache.generation().unwrap();
    let invalid = if rank == 1 {
        ExternalPredictionCaptureRequest::Muse {
            target_layers: vec![0, 1].into_boxed_slice(),
            target_paths: vec!["missing.0".into(), "missing.1".into()].into_boxed_slice(),
        }
    } else {
        request.clone()
    };
    let token = MlxTensor::from_array(Array::from_slice(&[6u32], &[1, 1]));
    assert!(target
        .verify_external_prediction_target(&token, &invalid, &mut cache)
        .is_err());
    assert_eq!(cache.generation().unwrap(), prefix);
    for token in [3u32, 4, 2] {
        let token = MlxTensor::from_array(Array::from_slice(&[token], &[1, 1]));
        let actual = target
            .verify_external_prediction_target(&token, &request, &mut cache)
            .unwrap();
        let expected = ordinary
            .verify_external_prediction_target(&token, &request, &mut reference_cache)
            .unwrap();
        compare(actual, expected, false, stream);
        assert_eq!(
            cache.generation().unwrap(),
            reference_cache.generation().unwrap()
        );
    }
}
fn compare(
    actual: (MlxTensor, ExternalPredictionTargetCapture<MlxTensor>),
    expected: (MlxTensor, ExternalPredictionTargetCapture<MlxTensor>),
    require_nonzero: bool,
    stream: &Stream,
) {
    close(&actual.0, &expected.0, require_nonzero, stream);
    let (
        ExternalPredictionTargetCapture::Muse {
            target_states: actual,
        },
        ExternalPredictionTargetCapture::Muse {
            target_states: expected,
        },
    ) = (actual.1, expected.1)
    else {
        panic!("expected Muse captures")
    };
    assert_eq!(actual.len(), 2);
    assert_eq!(expected.len(), 2);
    let pipeline =
        std::env::var(CARTESIAN_AXES).is_ok_and(|axes| axes.split('-').any(|axis| axis == "pp"));
    for (layer, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        // This two-layer fixture cuts the pipeline between layers. The second
        // producer consumes the declared F32 wire, while the first preserves
        // the checkpoint compute precision. Publication preserves each producer.
        let precision = if pipeline && layer == 1 {
            MlxDtype::Float32
        } else {
            expected.as_array().dtype()
        };
        assert_eq!(
            actual.as_array().dtype(),
            precision,
            "capture layer {layer} must preserve its producer precision"
        );
        close(actual, expected, require_nonzero, stream);
    }
}

fn close(actual: &MlxTensor, expected: &MlxTensor, require_nonzero: bool, stream: &Stream) {
    assert_eq!(actual.as_array().shape(), expected.as_array().shape());
    // BF16 partial products can round before their TP reduction. Preserve the
    // F32 bound and allow two BF16 epsilons at the observed fixture scale.
    let bf16 = expected.as_array().dtype() == MlxDtype::Bfloat16;
    let actual_array = actual
        .as_array()
        .as_dtype(MlxDtype::Float32, stream)
        .unwrap();
    let expected_array = expected
        .as_array()
        .as_dtype(MlxDtype::Float32, stream)
        .unwrap();
    let actual = actual_array.evaluated().unwrap();
    let expected = expected_array.evaluated().unwrap();
    if require_nonzero {
        assert!(expected.as_slice::<f32>().iter().any(|x| x.abs() > 1e-5));
    }
    let error = actual
        .as_slice::<f32>()
        .iter()
        .zip(expected.as_slice::<f32>())
        .map(|(a, e)| {
            assert!(
                a.is_finite() && e.is_finite(),
                "capture comparison requires finite values"
            );
            (a - e).abs()
        })
        .fold(0f32, f32::max);
    let scale = expected
        .as_slice::<f32>()
        .iter()
        .fold(1.0f32, |maximum, value| maximum.max(value.abs()));
    let tolerance = if bf16 { scale * (2.0 / 128.0) } else { 2e-4 };
    assert!(
        error <= tolerance,
        "native partition target capture maximum absolute error {error}"
    );
}
