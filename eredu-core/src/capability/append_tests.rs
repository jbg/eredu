use super::*;
use crate::cache::AppendStreamPolicy;

fn layout(offset: i32) -> StateMemoryLayout {
    let layer = LayerCachePolicy::key_value_with_state(
        AttentionPolicy::Full,
        2,
        3,
        vec![StateTensorPolicy::new(
            StateTensorRole::IntegerHistory { slot: 0 },
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(5).unwrap(),
            ],
            StateTensorDtype::Int32,
            crate::cache::MutableStateResidency::AlwaysDeviceMutable,
        )
        .unwrap()],
        vec![
            AppendStreamPolicy::new(0, 5, StateTensorDtype::Floating, 4).unwrap(),
            AppendStreamPolicy::new(1, 4, StateTensorDtype::Int32, 4).unwrap(),
            AppendStreamPolicy::new(2, 3, StateTensorDtype::Float32, 3).unwrap(),
        ],
    )
    .unwrap();
    StateMemoryLayout::new(
        LayerSchedule::new(1, vec![layer]).unwrap(),
        vec![offset],
        8,
        1,
        EstimationCompleteness::PersistentStateOnly,
    )
    .unwrap()
}

#[test]
fn stream_estimates_include_completed_records_and_exact_integer_widths() {
    for bytes in [2, 4] {
        for positions in [0, 2, 3, 4, 7, 8, 17] {
            let estimate = estimate_runtime_state(
                &layout(0),
                InputTokenCount::text(positions),
                0,
                2,
                NonZeroU8::new(bytes).unwrap(),
            )
            .unwrap();
            // Independent arithmetic: K/V plus three independently advancing streams.
            let record_bytes = 5 * u64::from(bytes) + 4 * 4;
            let expected = 2
                * (positions * 12 * u64::from(bytes)
                    + (positions / 4) * record_bytes
                    + (positions / 3) * 3 * 4);
            assert_eq!(estimate.fixed_state_bytes, 40);
            assert_eq!(estimate.context_state_bytes, expected);
            assert_eq!(estimate.requested_state_bytes, 40 + expected);
            assert_eq!(
                estimate.bytes_per_position_per_batch,
                12 * u64::from(bytes) + (5 * u64::from(bytes)).div_ceil(4) + 4 + 4
            );
            let minimum = estimate_runtime_state_payload_lower_bound(
                &layout(0),
                InputTokenCount::text(positions),
                2,
                NonZeroU8::new(bytes).unwrap(),
            )
            .unwrap();
            assert_eq!(
                minimum,
                40 + 2 * positions * 12 * u64::from(bytes),
                "optional streams can remain empty under padding"
            );
        }
    }
}

#[test]
fn streams_honor_prefix_offsets_and_check_overflow() {
    let dtype = NonZeroU8::new(2).unwrap();
    let delayed =
        estimate_runtime_state(&layout(-3), InputTokenCount::text(5), 6, 2, dtype).unwrap();
    let ordinary =
        estimate_runtime_state(&layout(0), InputTokenCount::text(8), 0, 2, dtype).unwrap();
    assert_eq!(delayed.context_state_bytes, ordinary.context_state_bytes);
    let stream_only = LayerCachePolicy::key_value_with_state(
        AttentionPolicy::Sliding {
            window: std::num::NonZeroU32::new(1).unwrap(),
        },
        1,
        1,
        vec![],
        vec![AppendStreamPolicy::new(0, i32::MAX, StateTensorDtype::Int32, 1).unwrap()],
    )
    .unwrap();
    let huge = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![stream_only]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::PersistentStateOnly,
    )
    .unwrap();
    assert!(matches!(
        estimate_runtime_state(&huge, InputTokenCount::text(u64::MAX / 2), 0, 1, dtype),
        Err(CapabilityError::ArithmeticOverflow {
            operation: "append stream lane bytes"
        })
    ));
}
