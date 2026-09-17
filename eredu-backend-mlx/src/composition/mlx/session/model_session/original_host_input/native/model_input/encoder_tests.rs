use super::super::super::tests::same_mode;

#[test]
fn original_vl_encoder_tables_are_consumed_once_with_full_state_and_three_decode_residency_parity()
{
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen3_vl_component_fixture(
        root.path(),
        false,
        false,
    );
    crate::tensor::reset_prepared_rotary_calls();
    same_mode(root.path(), 64, 4);
    // Each real selected resident/host/disk run executes its tower once. The
    // independent ordinary reference and cached decoder use the ordinary route.
    assert_eq!(crate::tensor::prepared_rotary_calls(), 3);
}

#[test]
fn original_conditional_encoder_tables_are_consumed_once_with_full_state_and_three_decode_residency_parity(
) {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_qwen35_conditional_component_fixture(
        root.path(),
        false,
    );
    crate::tensor::reset_prepared_rotary_calls();
    same_mode(root.path(), 16, 4);
    assert_eq!(crate::tensor::prepared_rotary_calls(), 3);
}

#[test]
fn original_encoder_cancellation_before_future_and_inside_media_preserves_prefix_and_three_decode_parity(
) {
    super::super::super::tests::semantics::compare_cancellation(4, &[0, 1, 2], &[0, 1, 2]);
}
