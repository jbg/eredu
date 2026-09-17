//! Saved pooling state must feed the same fresh ordinary/controlled driver.
use super::super::super::tests::{cause, reclaim, settle, words, AllTokens, Runtime};
use super::tests::{required_capacity, saved_source, source_config};
use super::*;
use crate::composition::mlx::replicated_text::tests::{
    routed_deepseek_v4_config, tiny_heterogeneous_artifact,
};
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, GenerationCancellationToken, TextGeneration, TokenOutput,
};
use eredu_runtime::working_memory::WorkingMemoryPool;

fn pooling_runtime(pool: &WorkingMemoryPool) -> (Runtime, tempfile::TempDir) {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &weights).with_memory_pool(pool.clone());
    let artifact = tiny_heterogeneous_artifact(routed_deepseek_v4_config());
    let prepared =
        eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
            .unwrap();
    let runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.unquoted_owner_count().unwrap() == 0
    });
    assert!(runtime.session().payload.model.has_published_idle_storage());
    (runtime, artifact)
}

#[derive(Debug, PartialEq)]
struct SavedValues {
    sampler: String,
    key: Vec<u32>,
    pending: Vec<u32>,
    arrays: Vec<(Vec<i32>, Vec<f32>)>,
}

fn values(saved: &MlxSavedTextComponents) -> SavedValues {
    let source = saved.funded_source().unwrap();
    let mut arrays = Vec::new();
    source
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            arrays.push((
                array.shape().to_vec(),
                array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            ));
        });
    SavedValues {
        sampler: format!("{:?}", source.sampling.sampler.as_sampler()),
        key: words(source.sampling.arrays.key.as_ref().unwrap()),
        pending: words(source.sampling.arrays.pending.as_ref().unwrap()),
        arrays,
    }
}

#[test]
fn pooling_partial_windows_resume_with_identical_ordinary_and_controlled_future() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = pooling_runtime(&pool);
        // F6 has a completed ratio-four pool and a partial next window. The
        // three resumed predictions reach F9, crossing the next pool boundary.
        let (saved, expected) = saved_source(&mut runtime, adaptive);
        let original = values(&saved);
        assert!(original
            .arrays
            .iter()
            .flat_map(|(_, v)| v)
            .any(|v| *v != 0.0));
        let (required, first_capacity) =
            required_capacity(&runtime, &saved, source_config(adaptive, 3, u64::MAX));
        let capacity = first_capacity
            .checked_add(required.checked_mul(3).unwrap())
            .unwrap();
        let config = source_config(adaptive, 3, capacity);
        let prior_peak = pool.peak_bytes().unwrap();
        let cancellation = GenerationCancellationToken::new();
        let ordinary = TextGeneration::resume_saved(&mut runtime, &saved, config, &cancellation)
            .unwrap()
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ordinary, expected);
        reclaim();
        let controlled = {
            let mut run = ControlledTextGeneration::resume_saved(
                &mut runtime,
                &saved,
                config,
                AllTokens,
                &cancellation,
            )
            .unwrap()
            .unwrap();
            let mut ids = Vec::new();
            while let Some(token) = run.next_cancellable(&cancellation) {
                let token = token.unwrap();
                assert_eq!(
                    token.output().step_receipt().unwrap().attempt(),
                    ids.len() as u64
                );
                ids.push(token.token_id());
            }
            ids
        };
        assert_eq!(controlled, expected);
        assert_eq!(values(&saved), original);
        let source = saved.funded_source().unwrap();
        assert_eq!(
            (source.sampling.frontier(), source.sampling.next_prediction),
            (6, 2)
        );
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .admission()
                .unwrap()
                .position(),
            9
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        assert!(pool.peak_bytes().unwrap() <= prior_peak.max(capacity));
        drop((saved, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn pooling_saved_resume_exact_capacity_and_one_short_use_actual_copied_state_quote() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = pooling_runtime(&pool);
    let (saved, expected) = saved_source(&mut runtime, true);
    let original = values(&saved);
    let (_, capacity) = required_capacity(&runtime, &saved, source_config(true, 3, u64::MAX));
    let before = (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        paths::snapshot(),
    );
    let state = runtime.session().payload.model.erased().state_snapshot();
    let cancellation = GenerationCancellationToken::new();
    let error = TextGeneration::resume_saved(
        &mut runtime,
        &saved,
        source_config(true, 3, capacity - 1),
        &cancellation,
    )
    .err()
    .expect("one byte below the complete bound must reject");
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(
        (
            pool.used_bytes().unwrap(),
            pool.peak_bytes().unwrap(),
            paths::snapshot()
        ),
        before
    );
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    assert_eq!(values(&saved), original);
    let tokens = TextGeneration::resume_saved(
        &mut runtime,
        &saved,
        source_config(true, 3, capacity),
        &cancellation,
    )
    .unwrap()
    .unwrap()
    .map(|token| token.unwrap().token_id().unwrap())
    .collect::<Vec<_>>();
    assert_eq!(tokens, expected);
    assert_eq!(pool.effective_capacity().unwrap(), capacity);
    assert_eq!(pool.peak_bytes().unwrap(), before.1.max(capacity));
    assert_eq!(values(&saved), original);
    drop((saved, runtime, artifact));
    settle(&pool, 0);
}
