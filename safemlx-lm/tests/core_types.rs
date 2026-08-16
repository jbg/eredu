use std::any::TypeId;

#[test]
fn facade_exports_are_the_canonical_core_types() {
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
    assert_eq!(
        TypeId::of::<safemlx_lm::SchedulerReport>(),
        TypeId::of::<safemlx_lm::core::scheduler::SchedulerReport>(),
    );
    assert_eq!(
        TypeId::of::<safemlx_lm::SchedulerError>(),
        TypeId::of::<safemlx_lm::core::scheduler::SchedulerError>(),
    );
}
