#[test]
fn empty_stats_have_zero_acceptance_rate() {
    assert_eq!(SpeculativeStats::default().accept_rate(), 0.0);
}

#[test]
fn component_timings_accumulate_without_overwriting_scheduler_stats() {
    let mut stats = SpeculativeStats::default();
    stats.add_scheduler_rounds(7);
    stats.add_component_timings(
        Duration::from_millis(2),
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    SpeculativeComponentTimings {
        draft_context: Duration::from_millis(3),
        draft_assistant: Duration::from_millis(5),
        draft_head: Duration::from_millis(7),
        target_verification: Duration::from_millis(11),
    }
    .add_to(&mut stats);

    assert_eq!(stats.rounds(), 7);
    assert_eq!(stats.draft_context_time(), Duration::from_millis(5));
    assert_eq!(stats.draft_assistant_time(), Duration::from_millis(5));
    assert_eq!(stats.draft_head_time(), Duration::from_millis(7));
    assert_eq!(stats.target_verification_time(), Duration::from_millis(11));
}

#[test]
fn component_timing_guard_is_scoped_and_nested() {
    assert!(!component_timing_enabled());
    {
        let _outer = SpeculativeComponentTimingGuard::enable();
        assert!(component_timing_enabled());
        {
            let _inner = SpeculativeComponentTimingGuard::enable();
            assert!(component_timing_enabled());
        }
        assert!(component_timing_enabled());
    }
    assert!(!component_timing_enabled());
}

#[test]
fn adaptive_lookahead_uses_deterministic_reuse_accounting() {
    let options = SpeculativeSchedulerOptions {
        adaptive_lookahead_min_blocks: 4,
        ..SpeculativeSchedulerOptions::default()
    };
    let mut profitable = SpeculativeStats::default();
    profitable.record_optimistic_accounting(4, 3, 2);
    profitable.update_adaptive_lookahead(options);
    assert!(!profitable.adaptive_lookahead_disabled());

    let mut unprofitable = SpeculativeStats::default();
    unprofitable.record_optimistic_accounting(4, 1, 2);
    unprofitable.update_adaptive_lookahead(options);
    assert!(unprofitable.adaptive_lookahead_disabled());

    let mut no_reuse = SpeculativeStats::default();
    no_reuse.record_optimistic_accounting(4, 0, 0);
    no_reuse.update_adaptive_lookahead(options);
    assert!(no_reuse.adaptive_lookahead_disabled());

    let mut disabled_policy = unprofitable.clone();
    disabled_policy.reset_adaptive_lookahead_decision();
    disabled_policy.update_adaptive_lookahead(SpeculativeSchedulerOptions {
        adaptive_lookahead: false,
        ..options
    });
    assert!(!disabled_policy.adaptive_lookahead_disabled());
}
