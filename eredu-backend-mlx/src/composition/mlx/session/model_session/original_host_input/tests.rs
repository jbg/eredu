use super::*;
use eredu_core::{Completion, InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use safemlx::{Device, DeviceType};
use std::cell::Cell;
thread_local! {
    static SLOTS:Cell<usize>=const{Cell::new(0)};
    static FAIL:Cell<Option<(usize,bool)>>=const{Cell::new(None)};
    static ESCAPED:RefCell<Option<Array>>=const{RefCell::new(None)};
}
#[derive(Debug, thiserror::Error)]
#[error("injected original host upload failure after actual slot")]
struct UploadFault;
pub(super) fn after_slot(n: usize, array: &Array) -> Result<(), Error> {
    SLOTS.set(n);
    if let Some((at, unwind)) = FAIL.get() {
        if n == at {
            FAIL.set(None);
            ESCAPED.with_borrow_mut(|s| *s = Some(array.clone()));
            assert!(!unwind, "upload unwind after real array");
            return Err(Error::Other(Box::new(UploadFault)));
        }
    }
    Ok(())
}
struct Hooks;
impl Drop for Hooks {
    fn drop(&mut self) {
        FAIL.set(None);
        ESCAPED.with_borrow_mut(|s| {
            s.take();
        });
    }
}
fn parts<R>(hidden: usize, f: impl FnOnce(&[HostInputPart<'_>]) -> R) -> R {
    let text = [1_u32, 2];
    let tail = [3_u32];
    let patches = (0..192)
        .map(|n| (n as f32 - 73.) / 193.)
        .collect::<Vec<_>>();
    let grid = [1_i32, 4, 4];
    let embeddings = (0..hidden * 2)
        .map(|n| (n as f32 - 51.) / 127.)
        .collect::<Vec<_>>();
    let meta = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::I32(&grid),
        },
    )];
    let extents = [InputExtent::PatchGrid {
        time: 1,
        height: 4,
        width: 4,
    }];
    f(&[
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::U32(&text),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[16, 12],
                values: HostTensorValues::F32(&patches),
            },
            metadata: &meta,
            extents: &extents,
        },
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::Embeddings,
            payload: HostTensorView {
                shape: &[1, 2, hidden],
                values: HostTensorValues::F32(&embeddings),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 1],
                values: HostTensorValues::U32(&tail),
            },
            metadata: &[],
            extents: &[],
        },
    ])
}
pub(super) fn source(pool: &MemoryLedger, hidden: usize) -> OriginalPreparedHostInput {
    parts(hidden, |p| {
        pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(p).unwrap())
            .unwrap()
    })
}
pub(super) fn settle(pool: &MemoryLedger, expected: usize, bytes: u64) {
    submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::memory::clear_cache();
        safemlx::reclaim_allocation_owners();

        pool.unquoted_owner_count().unwrap() == expected
            && pool.fixture_host_charge().unwrap() == bytes
    });
    assert_eq!(pool.unquoted_owner_count().unwrap(), expected);
}
pub(super) fn array_values(value: &Array) -> Vec<f64> {
    let value = value.evaluated().unwrap();
    match value.as_array().dtype() {
        safemlx::Dtype::Float32 => value.try_iter::<f32>().unwrap().map(f64::from).collect(),
        safemlx::Dtype::Uint32 => value.try_iter::<u32>().unwrap().map(f64::from).collect(),
        safemlx::Dtype::Int32 => value.try_iter::<i32>().unwrap().map(f64::from).collect(),
        other => panic!("fixture dtype {other:?}"),
    }
}
#[test]
fn ordinary_upload_preserves_every_slot_and_identity_but_foreign_pool_does_no_work() {
    let _hooks = Hooks;
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let source = source(&pool, 64);
    let bytes = source.original_bytes();
    let foreign = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    SLOTS.set(0);
    assert!(matches!(
        upload(&foreign, &source),
        Err(MlxHostInputUploadError::Admission(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(SLOTS.get(), 0);
    assert_eq!(foreign.fixture_host_charge().unwrap(), 0);
    assert_eq!(foreign.unquoted_owner_count().unwrap(), 0);
    let output = upload(&pool, &source).unwrap();
    assert_eq!(SLOTS.get(), 5);
    assert!(!output.has_original_input_custody());
    output.with_borrowed(|input| {
        assert_eq!(input.parts.len(), source.parts().len());
        for (actual, expected) in input.parts.iter().zip(source.parts()) {
            assert_eq!(actual.modality(), expected.modality());
            assert_eq!(actual.payload().kind(), expected.kind());
            assert_eq!(actual.extents(), expected.extents());
            for (a, b) in std::iter::once(actual.payload().value())
                .chain(actual.metadata().values())
                .zip(std::iter::once(expected.payload()).chain(expected.metadata().map(|(_, v)| v)))
            {
                assert_eq!(
                    a.shape().iter().map(|d| *d as usize).collect::<Vec<_>>(),
                    b.shape
                );
                let expected: Vec<f64> = match b.values {
                    HostTensorValues::U32(v) => v.iter().map(|v| f64::from(*v)).collect(),
                    HostTensorValues::I32(v) => v.iter().map(|v| f64::from(*v)).collect(),
                    HostTensorValues::F32(v) => v.iter().map(|v| f64::from(*v)).collect(),
                    _ => unreachable!(),
                };
                assert_eq!(array_values(a), expected);
            }
        }
    });
    let fingerprint = source
        .content_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(
        output
            .cache_identity()
            .unwrap()
            .semantic_content_fingerprint(),
        fingerprint
    );
    let refused = parts(64, |p| {
        pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(p).unwrap())
    })
    .unwrap_err();
    assert!(matches!(
        refused.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(refused.retained_bytes(), 0);
    // Independently uploaded prompt/metadata aliases retain ordinary exclusion.
    let identity = output.shared_cache_identity().unwrap().clone();
    drop(output);
    settle(&pool, 1, bytes);
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    drop(identity);
    settle(&pool, 0, bytes);
    drop(source);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}
#[test]
fn partial_upload_error_and_unwind_preserve_original_and_escaped_native_owners() {
    for unwind in [false, true] {
        let _hooks = Hooks;
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let source = source(&pool, 64);
        let bytes = source.original_bytes();
        SLOTS.set(0);
        FAIL.set(Some((2, unwind)));
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| upload(&pool, &source)));
        drop(source);
        assert_eq!(SLOTS.get(), 2);
        if unwind {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().unwrap_err();
            settle(&pool, 1, bytes);
            assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
            assert!(matches!(error, MlxHostInputUploadError::Operation(_)));
            drop(error);
        }
        settle(&pool, 1, 0);
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
        let escaped = ESCAPED.with_borrow_mut(Option::take).unwrap();
        assert!(!array_values(&escaped).is_empty());
        drop(escaped);
        settle(&pool, 0, 0);
    }
}
fn weights(mode: usize) -> crate::MlxLoadRequest {
    use eredu_runtime::{DenseDiskStreamLoadOptions, LayerwiseLoadOptions, WeightResidency};
    let residency = match mode {
        0 => WeightResidency::fully_resident(),
        1 => WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 26), Some(1 << 26), 1).unwrap(),
        )),
        2 => WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(1 << 26, 1 << 26, 1, 1).unwrap(),
        ),
        _ => unreachable!(),
    };
    crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
    )
}
fn baseline(pool: &MemoryLedger, hidden: usize) -> MlxModelInput {
    let owner = NativeMemoryOwner::acquire_typed(pool).unwrap();
    parts(hidden, |parts| {
        let one = |v: HostTensorView<'_>| {
            let shape = v.shape.iter().map(|n| *n as i32).collect::<Vec<_>>();
            let a = match v.values {
                HostTensorValues::U32(v) => Array::from_slice(v, &shape),
                HostTensorValues::I32(v) => Array::from_slice(v, &shape),
                HostTensorValues::F32(v) => Array::from_slice(v, &shape),
                _ => unreachable!(),
            };
            a.evaluated().unwrap();
            owner.retain_array(&a).unwrap();
            a
        };
        let values = parts
            .iter()
            .map(|p| {
                let value = one(p.payload);
                let payload = match p.kind {
                    InputPayloadKind::TokenIds => input::InputPayload::TokenIds(value),
                    InputPayloadKind::Tensor => input::InputPayload::Tensor(value),
                    InputPayloadKind::Embeddings => input::InputPayload::Embeddings(value),
                    _ => unreachable!(),
                };
                input::input_part(
                    p.modality,
                    payload,
                    p.metadata.iter().map(|(k, v)| (*k, one(*v))),
                    p.extents.iter().copied(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        MlxModelInput::from(input::ModelInput::new(&values).with_memory_owner(&owner))
            .with_semantic_content_fingerprint("independent full ordinary host-source reference")
            .unwrap()
    })
}
fn check_upload_mode(path: &std::path::Path, mode: usize, source_mode: usize, hidden: usize) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    // Original source construction is before any ordinary native fixture owner.
    let source = (source_mode != 0).then(|| source(&pool, hidden));
    let backend = admitted::backend(&pool);
    let config = cold_config(&backend, path, mode);
    let semantics = if source_mode >= 2 {
        Some(
            config
                .prepared_sources()
                .plan_original_media_semantics(source.as_ref().unwrap())
                .unwrap()
                .compile(&pool)
                .unwrap(),
        )
    } else {
        None
    };
    let native = if source_mode == 3 {
        Some(
            MlxPreparedInputMaterializer::prepare()
                .unwrap()
                .plan(source.as_ref().unwrap())
                .unwrap()
                .materialize(&pool)
                .unwrap(),
        )
    } else {
        None
    };
    let full = if source_mode == 4 {
        Some(
            MlxPreparedInputMaterializer::prepare()
                .unwrap()
                .model_input_plan(semantics.as_ref().unwrap())
                .unwrap()
                .materialize(&pool)
                .unwrap(),
        )
    } else {
        None
    };
    let model = backend.prepare_model_borrowed(&config).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    if full.is_some() {
        runtime
            .session()
            .payload
            .model
            .erased()
            .prepare_completed_media_binding_fixture()
            .unwrap();
    }
    let prompt = match (source.as_ref(), semantics) {
        (_, Some(semantics)) => {
            let prompt = if let Some(full) = full {
                full.bind(&runtime, semantics).unwrap()
            } else {
                match native {
                    Some(native) => MlxModelInput::from_original_native_input_with_semantics(
                        &runtime, native, semantics,
                    ),
                    None => {
                        MlxModelInput::from_original_host_input_with_semantics(&runtime, semantics)
                    }
                }
                .unwrap()
            };
            prompt.with_prefill_chunk_positions(2.try_into().unwrap())
        }
        (Some(source), None) => MlxModelInput::from_original_host_input(&runtime, source)
            .unwrap()
            .with_prefill_chunk_positions(2.try_into().unwrap()),
        (None, None) => baseline(&pool, hidden),
    };
    let reference = baseline(&pool, hidden);
    let values = |prompt: &MlxModelInput| {
        prompt.with_borrowed(|view| {
            view.parts
                .iter()
                .flat_map(|part| {
                    std::iter::once(part.payload().value()).chain(part.metadata().values())
                })
                .map(|array| (array.shape().to_vec(), array_values(array)))
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(values(&prompt), values(&reference));
    admitted::rejects_upload(&mut runtime, &prompt);
    drop((reference, prompt, runtime, config, source));
    submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::memory::clear_cache();
        safemlx::reclaim_allocation_owners();
        pool.unquoted_owner_count().unwrap() == 0
    });
}
fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 3e-4 + 3e-4 * b.abs(), "{a} != {b}");
    }
}
fn same(path: &std::path::Path, hidden: usize) {
    same_mode(path, hidden, 1);
}
pub(super) fn same_mode(path: &std::path::Path, hidden: usize, source_mode: usize) {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    for mode in 0..3 {
        if source_mode < 4 {
            check_upload_mode(path, mode, source_mode, hidden);
            continue;
        }
        let (expected, states) = admitted::run(path, mode, hidden, true, true);
        let (actual, actual_states) = admitted::run(path, mode, hidden, false, true);
        for (a, b) in actual.iter().zip(&expected) {
            close(a, b);
        }
        assert_eq!(actual_states.len(), states.len());
        for ((a, af), (b, bf)) in actual_states.iter().zip(&states) {
            assert_eq!(a.len(), b.len());
            for ((ashape, av), (bshape, bv)) in a.iter().zip(b) {
                assert_eq!(ashape, bshape);
                close(
                    &av.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                    &bv.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                );
            }
            assert_eq!(af.len(), bf.len());
            for ((al, ar, ashape, av), (bl, br, bshape, bv)) in af.iter().zip(bf) {
                assert_eq!((al, ar, ashape), (bl, br, bshape));
                close(
                    &av.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                    &bv.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                );
            }
        }
    }
}
#[test]
fn original_host_qwen_vl_upload_matches_source_and_refuses_execution_across_residencies() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    same(root.path(), 64);
}
#[test]
fn original_host_conditional_qwen_upload_matches_source_and_refuses_execution_across_residencies() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    same(root.path(), 16);
}
#[test]
fn original_source_upload_requires_complete_execution_producer() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    core_driver(false);
}
fn core_driver(semantic: bool) {
    core_driver_mode(usize::from(semantic));
}
pub(super) fn core_driver_mode(source_mode: usize) {
    core_driver_family(source_mode, false);
}
pub(super) fn core_driver_family(source_mode: usize, conditional: bool) {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    if conditional {
        crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
            root.path(),
            false,
        );
    } else {
        crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
            root.path(),
            false,
            false,
        );
    }
    let hidden = if conditional { 16 } else { 64 };
    for mode in 0..3 {
        if source_mode < 3 {
            check_upload_mode(root.path(), mode, source_mode + 1, hidden);
            continue;
        }
        let (expected, expected_states) = admitted::run(root.path(), mode, hidden, false, false);
        let (actual, actual_states) = admitted::run(root.path(), mode, hidden, false, true);
        assert_eq!(expected.len(), actual.len());
        for (a, b) in actual.iter().zip(&expected) {
            close(a, b);
        }
        assert_eq!(actual_states.last(), expected_states.last());
    }
}
#[path = "tests/admitted.rs"]
pub(super) mod admitted;

pub(super) fn cold_config(
    backend: &MlxBackend<'_>,
    path: &std::path::Path,
    mode: usize,
) -> crate::composition::mlx::loading::MlxModelConfig {
    use eredu_core::ModelLoadingBackend;
    let inspection = eredu_core::inspect_artifact(path, backend.configuration_resolver()).unwrap();
    eredu_core::prepare_inspected_model_config(backend, inspection, weights(mode)).unwrap()
}
#[path = "tests/semantics.rs"]
pub(super) mod semantics;

#[test]
#[cfg(target_vendor = "apple")]
fn original_native_pending_copy_uses_independent_slots_and_clears_the_compiled_packet() {
    if !crate::composition::mlx::session::model_session::original_host_input::tests::admitted::enter(
    ) {
        return;
    }
    semantics::pending_copy(true);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
