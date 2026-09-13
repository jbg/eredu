use super::*;
use eredu_core::execution_control::SnapshotEstimate;

pub(super) fn estimate(length: usize) -> SnapshotEstimate {
    let bytes = (length * std::mem::size_of::<i32>()) as u64;
    SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    }
}

#[test]
fn complete_embedded_snapshot_keeps_both_prediction_replicas_and_is_reusable() {
    let mut strategy = Strategy {
        fused_rows: None,
        corrupt_capture: false,
        failure: Failure::None,
    };
    let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
    let mut live = Cache {
        target: vec![2, 3],
        prediction: vec![5, 7, 11],
    };
    let seed = EmbeddedPredictionTargetState {
        capture: Tensor(vec![13, 17]),
        prediction_cache: vec![19, 23, 29, 31],
    };
    let cost = executor.control_snapshot_estimate(&live, &seed).unwrap();
    assert_eq!(
        cost.retained_bytes,
        11 * 4
            + std::mem::size_of_val(&seed) as u64
            + std::mem::size_of::<EmbeddedPredictionCheckpoint<Cache>>() as u64
    );
    assert_eq!(cost.copy_bytes, cost.retained_bytes);
    let (saved, saved_seed) = executor
        .control_snapshot(&live, &seed, ())
        .unwrap()
        .unwrap();
    for _ in 0..3 {
        live.target.push(101);
        live.prediction[0] = 103;
        let mut restored = executor
            .restore_control_snapshot(&mut live, &saved, &saved_seed, ())
            .unwrap()
            .unwrap();
        assert_eq!(live, saved.cache);
        assert_eq!(restored.capture, seed.capture);
        assert_eq!(restored.prediction_cache, seed.prediction_cache);
        restored.prediction_cache[0] = 107;
        restored.capture.0[0] = 109;
    }
    assert_eq!(saved.cache.prediction, [5, 7, 11]);
    assert_eq!(saved_seed.prediction_cache, [19, 23, 29, 31]);
}

#[test]
fn incomplete_estimate_prevents_copies_and_late_failure_preserves_live_cache_and_source() {
    let original = Cache {
        target: vec![2],
        prediction: vec![3],
    };
    let saved = EmbeddedPredictionCheckpoint {
        cache: Cache {
            target: vec![5],
            prediction: vec![7],
        },
        activations: None,
    };
    let seed = EmbeddedPredictionTargetState {
        capture: Tensor(vec![11]),
        prediction_cache: vec![13],
    };
    for unbounded in [true, false] {
        let mut strategy = Strategy {
            fused_rows: None,
            corrupt_capture: unbounded,
            failure: Failure::Advance,
        };
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
        let mut live = original.clone();
        let result = executor.restore_control_snapshot(&mut live, &saved, &seed, ());
        if unbounded {
            assert!(result.unwrap().is_none());
        } else {
            let error = result.err().unwrap();
            let SpeculativeControlError::Backend(source) = error else {
                panic!("lost backend cause")
            };
            assert_eq!(
                std::error::Error::source(&source)
                    .unwrap()
                    .downcast_ref::<TestError>(),
                Some(&TestError("seed copy failed".into()))
            );
        }
        assert_eq!(live, original);
    }
}
