//! Real retained source/parameter routes, with no capture installation or grant.
use super::*;
use crate::composition::mlx::session::model_session::{
    disk_layerwise_tests as disk, host_layerwise_tests as host,
};
use eredu_core::{capture::*, TextGenerationBackend, TokenFilter};

fn source(runtime: &ModelRuntime<MlxBackend<'_>>, cached: u64, prefill: bool) -> SharedCapturePlan {
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let usage = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let raw = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "real-logits".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                prefill,
                ..Default::default()
            },
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 3 },
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: cached,
            },
        )
        .unwrap(),
    )
}
fn quiescent(runtime: &ModelRuntime<MlxBackend<'_>>) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
    runtime.session().ensure_no_submission_in_flight().unwrap();
}
#[test]
fn real_resident_host_and_disk_quotes_share_paths_and_preserve_populated_decoder() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for route in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, _artifact) = match route {
            0 => host::runtime(&stream, &pool, None),
            1 => host::runtime(&stream, &pool, Some(1)),
            _ => disk::load_runtime(&stream, &pool, true),
        };
        let paths = paths(&runtime);
        for cached in [0, 8] {
            quiescent(&runtime);
            let capture = source(&runtime, cached, false);
            let executable = &runtime.session().payload.model;
            let shape = eredu_core::InferenceGeometry {
                batch_size: 1,
                cached_positions: cached,
                input_positions: 3,
                max_output_tokens: 4,
                prefill_chunk_positions: 2,
                output: eredu_core::OutputDemand::LastPosition,
            };
            let config = disk::config(0.0, 2, u64::MAX);
            let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
            let frontier = executable.erased().state_snapshot();
            let plain = executable
                .quote_replicated_resident_text_with_sampling(shape, config, &TokenFilter::All)
                .unwrap();
            let (observed, h) = executable
                .quote_replicated_resident_text_with_sampling_and_capture(
                    shape,
                    config,
                    &TokenFilter::All,
                    &capture,
                )
                .unwrap();
            let (registered, pin, rh) = executable
                .quote_registered_resident_text_with_sampling_and_capture(
                    shape,
                    config,
                    &TokenFilter::All,
                    &pool,
                    &capture,
                )
                .unwrap();
            assert!(observed.equations.transient().bytes().is_some());
            assert!(registered
                .equations
                .residual_workspace()
                .unwrap()
                .peak_bytes()
                .is_some());
            assert_eq!(
                observed.equations.completed_spans(),
                plain.equations.completed_spans()
            );
            assert_eq!(
                h.initialization_peak_bytes(),
                rh.initialization_peak_bytes()
            );
            assert!(h.initialization_peak_bytes() > 0);
            assert!(std::ptr::eq(h.source(), &capture));
            assert!(paths.same_storage(executable.erased().shared_observation_paths().unwrap()));
            assert_eq!(executable.erased().state_snapshot(), frontier);
            assert_eq!(
                (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
                before
            );
            let selected_prefill = source(&runtime, cached, true);
            let rejected = executable
                .quote_replicated_resident_text_with_sampling_and_capture(
                    shape,
                    config,
                    &TokenFilter::All,
                    &selected_prefill,
                )
                .unwrap_err();
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&rejected);
            let mut found = false;
            while let Some(error) = cause {
                found |= matches!(
                    error.downcast_ref::<eredu_runtime::capture::CaptureProtocolError>(),
                    Some(eredu_runtime::capture::CaptureProtocolError::PrefillAttribution)
                );
                cause = error.source();
            }
            assert!(
                found,
                "original typed prefill attribution cause must survive"
            );
            drop((plain, observed, registered, pin, h, rh));
            drop((selected_prefill, capture));
            if cached == 0 {
                let ids = disk::tokens();
                let (capacity, _) = disk::exact_capacity(&runtime, &pool, &ids, 0.0, 2);
                let output =
                    disk::outputs(&mut runtime, ids, disk::config(0.0, 2, capacity), false);
                assert_eq!(disk::token_ids(&output).len(), 4);
                drop(output);
            }
        }
        runtime.synchronize().unwrap();
        let retained = paths.capacity_bytes().unwrap();
        drop(runtime);
        stream.synchronize().unwrap();
        safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
            .unwrap()
            .synchronize()
            .unwrap();
        settle(&pool, retained);
        drop(paths);
        settle(&pool, 0);
    }
}
