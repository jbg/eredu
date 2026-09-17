use super::*;
use eredu_core::{capture::*, InputExtent, InputModality, PendingTextInput, TextGenerationBackend};
use eredu_runtime::{
    capture::CaptureSession, execution_control::TextSnapshotBackend,
    working_memory::WorkingMemoryPool,
};
use safemlx::{Device, DeviceType};

fn settle(stream: &Stream, pool: &WorkingMemoryPool, expected: usize) {
    stream.synchronize().unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::try_retire_completed_submissions();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == expected
    });
}
#[test]
fn missing_ordinary_attachment_error_alone_retains_actual_copied_source_lease() {
    let (stream, pool, error) = {
        let root = tempfile::tempdir().unwrap();
        crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
            root.path(),
            false,
            false,
        );
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let backend = MlxBackend::new(&stream, &stream).with_memory_pool(pool.clone());
        let model = eredu_core::load_model(&backend, root.path(), crate::MlxLoadRequest::default())
            .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let text = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let pixels = Array::from_slice(
            &(0..16 * 12)
                .map(|i| (i as f32 - 73.) / 193.)
                .collect::<Vec<_>>(),
            &[16, 12],
        );
        let grid = Array::from_slice(&[1_i32, 4, 4], &[1, 3]);
        let parts = [
            input::token_ids_part(&text).unwrap(),
            input::input_part(
                InputModality::Image,
                input::InputPayload::Tensor(pixels),
                [(eredu_core::InputMetadataKey::PatchGrid, grid)],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 4,
                    width: 4,
                }],
            )
            .unwrap(),
        ];
        let raw = MlxModelInput::from(input::ModelInput::new(&parts))
            .with_semantic_content_fingerprint("actual missing attachment copy")
            .unwrap()
            .with_prefill_chunk_positions(2.try_into().unwrap());
        let input = MlxBackend::prepare_control_input(&runtime, raw).unwrap();
        let a = input.attribution();
        let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
        let usage = CaptureUsage {
            captures: 16,
            retained_bytes: 1 << 20,
            host_bytes: 1 << 20,
            encoded_bytes: 1 << 20,
        };
        let plan = CapturePlan {
            schema_version: 1,
            selections: vec![],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: a.batch,
                prompt_tokens: a.decoder_positions,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: a.opening_position,
            },
        )
        .unwrap();
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let input = MlxBackend::bind_control_input_capture(
            &runtime,
            input,
            config,
            SharedCapturePlan::new(plan),
        )
        .unwrap();
        let prompt = MlxBackend::consume_control_input(&runtime, input)
            .unwrap()
            .0;
        let host = HostPreparationAuthority::retain(
            prompt
                .memory_owner
                .as_ref()
                .unwrap()
                .unquoted_lease()
                .unwrap(),
        );
        let run = CaptureSession::with_ordinary_prefill(
            prompt.prepared_capture.as_ref().unwrap().clone(),
            &host,
        )
        .unwrap();
        let saved = run.checkpoint(&discovery).unwrap();
        let mut pending =
            MlxBackend::copy_pending_input(&mut runtime, Some(PendingTextInput::Prefill(&prompt)))
                .unwrap();
        let Some(PendingTextInput::Prefill(copied)) = &mut pending else {
            panic!("actual copied prompt")
        };
        assert!(copied.prepared_capture.take().is_some());
        let error =
            MlxBackend::rebind_pending_capture(&runtime, Some(&saved), Some(&run), &mut pending)
                .unwrap_err();
        assert!(matches!(&error, Error::StorageSource(_)));
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        loop {
            if let Some(rejection) = cause.downcast_ref::<Rejection>() {
                assert!(matches!(rejection, Rejection::SourceMismatch));
                break;
            }
            cause = cause.source().expect("preserved typed source");
        }
        drop((pending, saved, run, prompt, host, runtime, parts, text));
        (stream, pool, error)
    };
    settle(&stream, &pool, 1);
    drop(error);
    settle(&stream, &pool, 0);
}
