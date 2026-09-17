use super::*;
use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions};

#[test]
#[ignore = "requires native CPU attention"]
fn settled_block_submission_reuses_completion_and_revokes_readiness_on_failure() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let query = Array::from_slice(&[1.0f32, 0.5], &[1, 1, 1, 2]);
    let keys = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 2, 2]);
    let values = Array::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[1, 1, 2, 2]);
    let invalid = KeyValueAttentionBlock::unleased(
        0,
        2,
        Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]),
        values.clone(),
    );
    for arithmetic in [AttentionArithmetic::Fused, AttentionArithmetic::InputScores] {
        let options = BlockwiseAttentionOptions {
            arithmetic,
            softcap: None,
        };
        let mut accumulator =
            BlockwiseAttentionAccumulator::new(&query, 1.0, None, 1, None, 0, None, 2, &stream)
                .unwrap();
        accumulator.set_options(options).unwrap();
        assert!(!accumulator.recurrence_completed);
        for pass in 0..options.passes() {
            if pass == 1 {
                accumulator.begin_value_pass().unwrap();
            }
            let block = KeyValueAttentionBlock::unleased(0, 2, keys.clone(), values.clone());
            accumulator.accumulate(&block, &stream).unwrap();
            assert!(accumulator.recurrence_completed);
            accumulator.submit().unwrap();
            accumulator.submit().unwrap();
            assert!(accumulator.recurrence_completed);
        }
        assert!(accumulator.accumulate(&invalid, &stream).is_err());
        assert!(!accumulator.recurrence_completed);
        let output = accumulator.finish(&stream).unwrap();
        let output = output.evaluated().unwrap();
        let p = 2.0f32.exp() / (1.0 + 2.0f32.exp());
        let expected = [2.0 * p + 6.0 * (1.0 - p), 4.0 * p + 8.0 * (1.0 - p)];
        for (actual, expected) in output.as_slice::<f32>().iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{arithmetic:?}: {actual} vs {expected}"
            );
        }
    }
}
