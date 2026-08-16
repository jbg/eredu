use std::any::TypeId;

#[test]
fn moved_attention_types_keep_facade_paths() {
    assert_eq!(
        TypeId::of::<safemlx_lm::AttentionPolicy>(),
        TypeId::of::<safemlx_lm::core::attention::AttentionPolicy>(),
    );
    let schedule = safemlx_lm::LayerSchedule::all_full(2).unwrap();
    assert_eq!(schedule.len(), 2);
    assert_eq!(
        TypeId::of::<safemlx_lm::RequestId>(),
        TypeId::of::<safemlx_lm::core::scheduler::RequestId>(),
    );
    assert_eq!(
        TypeId::of::<safemlx_lm::SchedulerLimits>(),
        TypeId::of::<safemlx_lm::core::scheduler::SchedulerLimits>(),
    );
}
