//! Actual recurrent and compressed families through the same fresh-resume driver.
use super::super::super::tests::{reclaim, settle, words, AllTokens, Runtime};
use super::tests::{required_capacity, saved_source, source_config};
use super::*;
use crate::composition::mlx::{
    replicated_text::tests::{routed_deepseek_v3_config, tiny_heterogeneous_artifact},
    session::model_session::host_layerwise_tests::family_artifact,
};
use eredu_core::{
    cache::LayerCachePolicy, GenerationCancellationToken, TextGeneration, TextGenerationDriver,
    TokenOutput,
};
use eredu_runtime::working_memory::WorkingMemoryPool;

#[derive(Clone, Copy, Debug)]
enum Family {
    Recurrent,
    Compressed,
}
#[derive(Debug, PartialEq)]
enum Numbers {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
}
#[derive(Debug, PartialEq)]
struct NativeValue {
    shape: Vec<i32>,
    numbers: Numbers,
}
fn read(
    plan: &crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
) -> Vec<NativeValue> {
    let mut result = Vec::new();
    plan.visit_operands(&mut |array| {
        let evaluated = array.evaluated().unwrap();
        let numbers = match array.dtype() {
            safemlx::Dtype::Float32 => Numbers::F32(evaluated.try_to_vec::<f32>().unwrap()),
            safemlx::Dtype::Int32 => Numbers::I32(evaluated.try_to_vec::<i32>().unwrap()),
            safemlx::Dtype::Uint32 => Numbers::U32(evaluated.try_to_vec::<u32>().unwrap()),
            dtype => panic!("unexpected fixture state dtype {dtype:?}"),
        };
        result.push(NativeValue {
            shape: array.shape().to_vec(),
            numbers,
        });
    });
    result
}
fn current(runtime: &Runtime) -> Vec<NativeValue> {
    read(
        &runtime
            .session()
            .payload
            .model
            .erased()
            .prepare_resident_decoder_copy()
            .unwrap(),
    )
}
fn settle_runtime(runtime: &Runtime) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
}

fn close(actual: &[NativeValue], expected: &[NativeValue]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.shape, expected.shape);
        match (&actual.numbers, &expected.numbers) {
            (Numbers::F32(actual), Numbers::F32(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(expected) {
                    assert!(
                        (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                        "state value {a} != {b}"
                    );
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }
}
fn parity(family: Family) {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxBackend::new(&stream, &weights).with_memory_pool(pool.clone());
        let artifact = match family {
            Family::Recurrent => family_artifact("qwen3_5_text"),
            Family::Compressed => tiny_heterogeneous_artifact(routed_deepseek_v3_config()),
        };
        let prepared =
            eredu_core::load_model(&backend, artifact.path(), crate::MlxLoadRequest::default())
                .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
        crate::backend::submission_recovery::wait_for_retirement(|| {
            reclaim();
            pool.unquoted_owner_count().unwrap() == 0
        });
        let (saved, expected) = saved_source(&mut runtime, adaptive);
        let expected_state = current(&runtime);
        let source = saved.funded_source().unwrap();
        let source_plan = source.decoder.native.prepare_copy().unwrap();
        let layout = source_plan.shared_layout().unwrap().clone();
        assert!((0..layout.layout().len()).any(|index| match family {
            Family::Recurrent => !layout
                .layout()
                .layer(index)
                .unwrap()
                .fixed_state()
                .is_empty(),
            Family::Compressed => matches!(
                layout.layout().layer(index),
                Some(LayerCachePolicy::CompressedLatentRotary { .. })
            ),
        }));
        let before = read(&source_plan);
        drop(source_plan);
        assert!(before.iter().any(
            |value| matches!(&value.numbers,Numbers::F32(values) if values.iter().any(|v|*v!=0.0))
        ));
        let key = words(source.sampling.arrays.key.as_ref().unwrap());
        let pending = words(source.sampling.arrays.pending.as_ref().unwrap());
        let sampler = format!("{:?}", source.sampling.sampler.as_sampler());
        let (required, exact) =
            required_capacity(&runtime, &saved, source_config(adaptive, 3, u64::MAX));
        let capacity = exact.checked_add(required.checked_mul(3).unwrap()).unwrap();
        let config = source_config(adaptive, 3, capacity);
        let cancellation = GenerationCancellationToken::new();
        let ordinary = {
            let mut run = TextGeneration::resume_saved(&mut runtime, &saved, config, &cancellation)
                .unwrap_or_else(|error| panic!("{family:?} ordinary resume: {error}"))
                .unwrap();
            run.by_ref()
                .map(|token| token.unwrap().token_id().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(ordinary, expected, "{family:?} ordinary future");
        settle_runtime(&runtime);
        close(&current(&runtime), &expected_state);
        reclaim();
        let controlled = {
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut state = driver
                .resume_saved(&saved, config, AllTokens, &cancellation)
                .unwrap_or_else(|error| panic!("{family:?} controlled resume: {error}"))
                .unwrap();
            {
                let model = driver.runtime().session().payload.model.erased();
                let mut all = model.retained_decoder_state_storage().unwrap();
                all.merge(model.retained_idle_auxiliary_storage().unwrap())
                    .unwrap();
                drop(all.pin_registered(&pool).expect(
                    "all actual child tables and native roots published before first prediction",
                ));
            }
            let mut ids = Vec::new();
            while let Some(output) = driver.advance(&mut state).unwrap() {
                assert_eq!(
                    output.output().step_receipt().unwrap().attempt(),
                    ids.len() as u64
                );
                ids.push(output.token_id());
                assert!(driver.take_completed_step(&mut state).unwrap().is_none());
            }
            ids
        };
        assert_eq!(controlled, expected, "{family:?} controlled future");
        settle_runtime(&runtime);
        close(&current(&runtime), &expected_state);
        assert_eq!(read(&source.decoder.native.prepare_copy().unwrap()), before);
        assert_eq!(words(source.sampling.arrays.key.as_ref().unwrap()), key);
        assert_eq!(
            words(source.sampling.arrays.pending.as_ref().unwrap()),
            pending
        );
        assert_eq!(
            format!("{:?}", source.sampling.sampler.as_sampler()),
            sampler
        );
        assert_eq!(
            (source.sampling.frontier(), source.sampling.next_prediction),
            (6, 2)
        );
        assert!(runtime
            .session()
            .payload
            .model
            .erased()
            .state_snapshot()
            .iter()
            .all(|(position, _)| *position == 9));
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        drop((layout, saved, runtime, artifact));
        settle(&pool, 0);
    }
}
#[test]
fn recurrent_qwen_saved_resume_preserves_numeric_state_and_stochastic_future() {
    parity(Family::Recurrent)
}
#[test]
fn compressed_routed_deepseek_saved_resume_preserves_numeric_state_and_stochastic_future() {
    parity(Family::Compressed)
}
