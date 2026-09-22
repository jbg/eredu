use super::*;
use crate::input::host::{
    HostInputPart, HostInputPartView, HostInputPlanError, HostTensorValues, HostTensorView,
};
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use std::error::Error as _;
fn parts<R>(f: impl FnOnce(&[HostInputPart<'_>]) -> R) -> R {
    let tokens = vec![1_u32, 2, 3];
    let values = vec![0.5_f32, -1., 2., 3.];
    let grid = [1_i32, 2, 2];
    let metadata = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::I32(&grid),
        },
    )];
    let extents = [InputExtent::PatchGrid {
        time: 1,
        height: 2,
        width: 2,
    }];
    f(&[
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 3],
                values: HostTensorValues::U32(&tokens),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[4, 1],
                values: HostTensorValues::F32(&values),
            },
            metadata: &metadata,
            extents: &extents,
        },
    ])
}
fn required() -> u64 {
    parts(|p| {
        MemoryLedger::prepared_host_input_required_bytes(
            &PreparedHostInputPlan::prepare(p).unwrap(),
        )
        .unwrap()
    })
}
fn source(pool: &MemoryLedger) -> OriginalPreparedHostInput {
    parts(|p| {
        pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(p).unwrap())
            .unwrap()
    })
}
fn values(source: &OriginalPreparedHostInput) {
    let mut parts = source.parts();
    let tokens = parts.next().unwrap();
    assert!(matches!(tokens.payload().values,HostTensorValues::U32(v) if v==[1,2,3]));
    let image = parts.next().unwrap();
    assert!(matches!(image.payload().values,HostTensorValues::F32(v) if v==[0.5,-1.,2.,3.]));
    let (key, grid) = image.metadata().next().unwrap();
    assert_eq!(key, InputMetadataKey::PatchGrid);
    assert!(matches!(grid.values,HostTensorValues::I32(v) if v==[1,2,2]));
    assert_eq!(
        image.extents(),
        [InputExtent::PatchGrid {
            time: 1,
            height: 2,
            width: 2
        }]
    );
    assert!(parts.next().is_none());
}
#[test]
fn exact_source_charge_precedes_all_reserves_and_outlives_borrowed_values() {
    let bytes = required();
    let short = crate::working_memory::memory_fixture::host_ledger(bytes - 1, 0).unwrap();
    let error = parts(|p| {
        short.compile_prepared_host_input_with(
            PreparedHostInputPlan::prepare(p).unwrap(),
            || panic!("short source reached constructor"),
            Storage::compile,
        )
    })
    .unwrap_err();
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes==bytes && (limit_bytes - existing_bytes)==bytes-1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let output = parts(|p| {
        pool.compile_prepared_host_input_with(
            PreparedHostInputPlan::prepare(p).unwrap(),
            || {
                assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
            },
            Storage::compile,
        )
    })
    .unwrap();
    values(&output);
    assert_eq!(output.slot_count(), 3);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn every_failed_destination_retains_actual_prefix_through_closed_error_retirement() {
    let bytes = required();
    let mut previous = 0;
    for at in 0..8 {
        let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
        let error = parts(|p| {
            pool.compile_prepared_host_input_with(
                PreparedHostInputPlan::prepare(p).unwrap(),
                || {},
                |p| Storage::compile_failing(p, at),
            )
        })
        .unwrap_err();
        assert_eq!(error.failed_buffer(), Some(at));
        assert_eq!(error.retained_bytes(), bytes);
        let heap = error.failed_prefix_bytes();
        assert_eq!(heap == 0, at == 0);
        if at > 0 {
            assert!(heap > previous);
        }
        previous = heap;
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
        let error = BackendFailure::from_error(error);
        let concrete = error
            .source()
            .unwrap()
            .downcast_ref::<OriginalPreparedHostInputError>()
            .unwrap();
        assert_eq!(concrete.failed_prefix_bytes(), heap);
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        drop(error);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
#[test]
fn equal_content_is_not_source_authority_and_all_aliases_retire_the_same_charge() {
    let bytes = required();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes * 2, 0).unwrap();
    let first = source(&pool);
    let second = source(&pool);
    assert_eq!(first.content_digest(), second.content_digest());
    assert!(!first.same_source(&second));
    let aliases = (0..8).map(|_| first.clone()).collect::<Vec<_>>();
    assert!(aliases.iter().all(|v| v.same_source(&first)));
    let foreign = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    assert!(matches!(
        first.validate_pool(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    drop(first);
    std::thread::scope(|scope| {
        for alias in aliases {
            scope.spawn(move || {
                values(&alias);
                drop(alias);
            });
        }
    });
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    values(&second);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn unquoted_predecessor_and_unwind_preserve_real_exclusion_without_adoption() {
    let bytes = required();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let prior = pool.acquire_unquoted().unwrap();
    let error = parts(|p| {
        pool.compile_prepared_host_input_with(
            PreparedHostInputPlan::prepare(p).unwrap(),
            || panic!("unquoted input reached compiler"),
            Storage::compile,
        )
    })
    .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(error.retained_bytes(), 0);
    drop(prior);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parts(|p| pool
            .compile_prepared_host_input_with(
                PreparedHostInputPlan::prepare(p).unwrap(),
                || panic!("after source admission"),
                Storage::compile
            ))))
        .is_err()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let value = source(&pool);
    values(&value);
    drop(value);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn malformed_shapes_overflow_and_metadata_are_rejected_without_owned_preflight() {
    parts(|parts| {
        let mut input = parts.to_vec();
        input[0].payload.shape = &[usize::MAX, 2];
        assert!(matches!(
            PreparedHostInputPlan::prepare(&input),
            Err(HostInputPlanError::Overflow)
        ));
        input[0] = parts[0];
        input[0].payload.shape = &[1, 4];
        assert!(matches!(
            PreparedHostInputPlan::prepare(&input),
            Err(HostInputPlanError::Shape { part: 0 })
        ));
        input[0] = parts[0];
        let duplicate = [parts[1].metadata[0], parts[1].metadata[0]];
        input[1].metadata = &duplicate;
        assert!(matches!(
            PreparedHostInputPlan::prepare(&input),
            Err(HostInputPlanError::Structure { part: 1 })
        ));
        input[1] = parts[1];
    });
}
#[test]
fn poisoned_settlement_retains_completed_or_partial_sources_without_false_refund() {
    for partial in [false, true] {
        let bytes = required();
        let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
        let error = parts(|p| {
            pool.compile_prepared_host_input_with(
                PreparedHostInputPlan::prepare(p).unwrap(),
                || {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _guard = pool.0.usage.lock().unwrap();
                        panic!("poison source settlement");
                    }));
                },
                |p| {
                    if partial {
                        Storage::compile_failing(p, 4)
                    } else {
                        Storage::compile(p)
                    }
                },
            )
        })
        .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), bytes);
        assert_eq!(error._completed.is_some(), !partial);
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reserved, bytes);
        assert_eq!(usage.reservations, 1);
    }
}
#[test]
fn concurrent_real_compilers_hold_independent_original_capacity_until_terminal_source_drop() {
    let bytes = required();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes * 2, 0).unwrap();
    let (first, second, held, arrivals) = std::thread::scope(|scope| {
        let (notify, notified) = std::sync::mpsc::channel();
        let (release_a, resume_a) = std::sync::mpsc::channel::<()>();
        let (release_b, resume_b) = std::sync::mpsc::channel::<()>();
        let make = |notify: std::sync::mpsc::Sender<bool>,
                    resume: std::sync::mpsc::Receiver<()>| {
            let mut accepted = false;
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parts(|p| {
                    pool.compile_prepared_host_input_with(
                        PreparedHostInputPlan::prepare(p).unwrap(),
                        || {
                            accepted = true;
                            let _ = notify.send(true);
                            let _ = resume.recv();
                        },
                        Storage::compile,
                    )
                })
            }));
            if !accepted {
                let _ = notify.send(false);
            }
            out
        };
        let other = notify.clone();
        let a = scope.spawn(move || make(other, resume_a));
        let b = scope.spawn(move || make(notify, resume_b));
        let arrivals = [notified.recv(), notified.recv()];
        let held = (pool.payload_used_bytes(), pool.acquire_unquoted().err());
        drop(release_a);
        drop(release_b);
        (a.join(), b.join(), held, arrivals)
    });
    assert!(arrivals.iter().all(|a| matches!(a, Ok(true))));
    assert_eq!(held.0.unwrap(), bytes * 2);
    assert!(matches!(
        held.1,
        Some(WorkingMemoryError::ReservedWorkActive)
    ));
    let first = first.unwrap().unwrap().unwrap();
    let second = second.unwrap().unwrap().unwrap();
    assert!(!first.same_source(&second));
    values(&first);
    values(&second);
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parts(|p| pool
            .compile_prepared_host_input_with(
                PreparedHostInputPlan::prepare(p).unwrap(),
                || {},
                |p| {
                    let completed = Storage::compile(p)?;
                    assert_eq!(completed.slot_count(), 3);
                    panic!("completed source before publication");
                }
            ))))
        .is_err()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn metadata_key_order_is_canonical_but_actual_value_changes_and_source_identity_remain_distinct() {
    parts(|base| {
        let positions = [7_i32, 11];
        let shifted = [7_i32, 12];
        let grid = base[1].metadata[0];
        let position = (
            InputMetadataKey::PatchPositions,
            HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::I32(&positions),
            },
        );
        let a = [position, grid];
        let b = [grid, position];
        let c = [
            grid,
            (
                InputMetadataKey::PatchPositions,
                HostTensorView {
                    shape: &[1, 2],
                    values: HostTensorValues::I32(&shifted),
                },
            ),
        ];
        let mut left = base.to_vec();
        left[1].metadata = &a;
        let mut right = base.to_vec();
        right[1].metadata = &b;
        let mut changed = base.to_vec();
        changed[1].metadata = &c;
        let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
        let first = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&left).unwrap())
            .unwrap();
        let second = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&right).unwrap())
            .unwrap();
        let third = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&changed).unwrap())
            .unwrap();
        assert_eq!(first.content_digest(), second.content_digest());
        assert_ne!(first.content_digest(), third.content_digest());
        assert!(!first.same_source(&second));
        let image = first.parts().nth(1).unwrap();
        assert_eq!(
            image.metadata().map(|(key, _)| key).collect::<Vec<_>>(),
            [
                InputMetadataKey::PatchGrid,
                InputMetadataKey::PatchPositions
            ]
        );
        let mut scalar = base.to_vec();
        scalar[0].payload.shape = &[];
        assert!(matches!(
            PreparedHostInputPlan::prepare(&scalar),
            Err(HostInputPlanError::Shape { part: 0 })
        ));
    });
}

#[test]
fn boolean_audio_mask_source_copies_values_and_retains_exact_original_charge() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let source = {
        let frames = [0.25_f32, -0.5, 0.75, 1.0, -1.25];
        let flags = [true, true, false, true, false];
        let metadata = [(
            InputMetadataKey::AudioMask,
            HostTensorView {
                shape: &[1, 5],
                values: HostTensorValues::Bool(&flags),
            },
        )];
        let parts = [HostInputPart {
            modality: InputModality::Audio,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[1, 5, 1],
                values: HostTensorValues::F32(&frames),
            },
            metadata: &metadata,
            extents: &[InputExtent::AudioValidFrames(3)],
        }];
        let plan = PreparedHostInputPlan::prepare(&parts).unwrap();
        let required = MemoryLedger::prepared_host_input_required_bytes(&plan).unwrap();
        let source = pool.compile_prepared_host_input(plan).unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), required);
        source
    };
    let retained = pool.payload_used_bytes().unwrap();
    let alias = source.clone();
    assert!(source.same_source(&alias));
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), retained);
    assert_eq!(alias.slot_count(), 2);
    let part = alias.parts().next().unwrap();
    let (key, mask) = part.metadata().next().unwrap();
    assert_eq!(key, InputMetadataKey::AudioMask);
    assert_eq!(mask.shape, &[1, 5]);
    assert!(
        matches!(mask.values, HostTensorValues::Bool(values) if values == [true,true,false,true,false])
    );
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
