use super::*;

#[test]
fn original_preview_controls_reject_one_short_and_zero_result_keeps_exact_hold() {
    for maximum in [0, 5] {
        let source = admitted(
            vec![SymbolicDimension::Sequence, SymbolicDimension::Known(4)],
            CaptureTransform::Preview {
                max_elements: maximum,
            },
            vec![],
        );
        let payload_bytes = plan(&source).retained_payload_bytes();
        let required = plan(&source).initialization_peak_bytes();
        for bytes in [required - 1, required] {
            let pool = WorkingMemoryPool::new(required, 0).unwrap();
            let (reservation, run) = fresh(&pool, bytes);
            let before = ledger(&pool);
            let allocations = ALLOCATIONS.get();
            let result = run.prepare_capture_tensor(&reservation, plan(&source));
            if bytes < required {
                assert!(matches!(result,
                    Err(CaptureTensorConstructionError::Memory(
                        WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }
                    )) if required_bytes == required && available_bytes == bytes
                ));
                assert_eq!(ledger(&pool), before);
                assert_eq!(ALLOCATIONS.get(), allocations);
                drop((run, reservation));
            } else {
                let output = fill(result.unwrap());
                assert_eq!(output.shape(), &[maximum as usize]);
                // This diagnostic keeps its preexisting DTO/shape/data meaning.
                assert_eq!(output.retained_payload_bytes(), Some(payload_bytes));
                assert_eq!(
                    output.as_observation().retained_payload_bytes(),
                    Some(payload_bytes)
                );
                assert_eq!(ledger(&pool).2, required);
                let alias = output.clone();
                drop((output, run, reservation));
                assert_eq!(pool.used_bytes().unwrap(), required);
                let TensorObservationData::F32(values) = alias.data() else {
                    unreachable!()
                };
                assert_eq!(values.len(), maximum as usize);
                assert_eq!(values.capacity(), maximum as usize);
                assert_eq!(
                    values,
                    &(0..maximum).map(|n| n as f32 + 0.25).collect::<Vec<_>>()
                );
                drop(alias);
            }
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn incomplete_and_rejected_preview_keep_original_controls_until_error_retirement() {
    for close_parent in [false, true] {
        let source = admitted(
            vec![SymbolicDimension::Sequence, SymbolicDimension::Known(4)],
            CaptureTransform::Preview { max_elements: 5 },
            vec![],
        );
        let required = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let (reservation, run) = fresh(&pool, required);
        let mut run = Some(run);
        let mut builder = run
            .as_ref()
            .unwrap()
            .prepare_capture_tensor(&reservation, plan(&source))
            .unwrap();
        builder.push_f32(7.5).unwrap();
        let pointer = builder.data.as_ptr();
        if close_parent {
            drop(run.take());
        }
        let error = builder.finish().unwrap_err();
        assert_eq!(pool.used_bytes().unwrap(), required);
        if close_parent {
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<WorkingMemoryError>()
                .is_some_and(|e| matches!(e, WorkingMemoryError::ExecutionFenced)));
            drop((run, reservation));
            assert_eq!(pool.used_bytes().unwrap(), required);
            drop(error);
        } else {
            let builder = error.into_builder().unwrap();
            assert_eq!(builder.data.as_ptr(), pointer);
            assert_eq!(builder.initialized_count(), 1);
            assert_eq!(builder.data, [7.5]);
            drop((run, reservation));
            assert_eq!(pool.used_bytes().unwrap(), required);
            drop(builder);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
        drop(pool.acquire_unquoted().unwrap());
    }
}
