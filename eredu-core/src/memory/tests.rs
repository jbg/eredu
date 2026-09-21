use super::*;

const POLICIES: [MemoryOverheadPolicy; 2] = [
    MemoryOverheadPolicy::AllowUnknownOverhead,
    MemoryOverheadPolicy::RequireFiniteEstimates,
];

fn accounted(bytes: u64) -> MemoryContribution<'static> {
    MemoryContribution::Accounted {
        source: "controlled buffer",
        bytes,
    }
}

fn estimated(lower: u64, upper: u64) -> MemoryContribution<'static> {
    MemoryContribution::Estimated {
        source: "parser",
        range: FiniteMemoryEstimate::new(lower, upper).unwrap(),
        basis: "bounded input size and configurable parser estimate",
    }
}

fn unknown(source: &str) -> MemoryContribution<'_> {
    MemoryContribution::Unknown {
        source,
        reason: "no finite estimate for internal bookkeeping",
    }
}

#[test]
fn default_policy_permits_unknown_with_a_visible_source() {
    assert_eq!(MemoryOverheadPolicy::default(), POLICIES[0]);
    let rows = [unknown("driver")];
    let evaluation = evaluate_memory_requirements(&rows, Default::default(), Some(0), 0).unwrap();
    assert_eq!(
        evaluation.policy(),
        MemoryOverheadPolicy::AllowUnknownOverhead
    );
    assert_eq!(evaluation.budget_bytes(), Some(0));
    assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
    assert_eq!(evaluation.report().budget_charge_bytes(), 0);
    assert_eq!(evaluation.report().accounted_bytes(), 0);
    assert_eq!(evaluation.report().estimated_allowance_bytes(), 0);
    assert_eq!(evaluation.report().unknown_count(), 1);
    assert_eq!(evaluation.report().contributions(), &rows);
    let warning = evaluation.warnings().next().unwrap();
    assert_eq!(warning.contribution_index, 0);
    assert_eq!(warning.source, "driver");
    assert_eq!(
        warning.reason,
        "no finite estimate for internal bookkeeping"
    );
    assert_eq!(evaluation.warnings().count(), 1);
}

#[test]
fn accounted_capacity_uses_exact_budget_boundaries_under_both_policies() {
    let rows = [accounted(11), accounted(17)];
    for policy in POLICIES {
        for budget in [None, Some(28), Some(29)] {
            let evaluation = evaluate_memory_requirements(&rows, policy, budget, 0).unwrap();
            assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
            assert_eq!(evaluation.report().accounted_bytes(), 28);
            assert_eq!(evaluation.report().budget_charge_bytes(), 28);
            assert_eq!(evaluation.warnings().count(), 0);
        }
        let rejected = evaluate_memory_requirements(&rows, policy, Some(27), 0).unwrap();
        assert_eq!(
            rejected.decision(),
            MemoryPolicyDecision::BudgetExceeded {
                required_bytes: 28,
                budget_bytes: 27,
            }
        );
        assert_eq!(rejected.report().contributions(), &rows);
    }
}

#[test]
fn finite_estimates_keep_their_ranges_and_reserve_upper_endpoints() {
    let rows = [accounted(13), estimated(3, 17), estimated(5, 23)];
    for policy in POLICIES {
        let evaluation = evaluate_memory_requirements(&rows, policy, Some(60), 7).unwrap();
        assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
        let report = evaluation.report();
        assert_eq!(report.accounted_bytes(), 13);
        assert_eq!(
            report.estimated_range(),
            FiniteMemoryEstimate::new(8, 40).unwrap()
        );
        assert_eq!(report.estimated_allowance_bytes(), 40);
        assert_eq!(report.headroom_bytes(), 7);
        assert_eq!(report.budget_charge_bytes(), 60);
        assert_eq!(report.unknown_count(), 0);
        assert_eq!(report.contributions(), &rows);
        assert_eq!(evaluation.warnings().count(), 0);
        let rejected = evaluate_memory_requirements(&rows, policy, Some(59), 7).unwrap();
        assert_eq!(
            rejected.decision(),
            MemoryPolicyDecision::BudgetExceeded {
                required_bytes: 60,
                budget_bytes: 59,
            }
        );
        assert_eq!(rejected.report(), report);
    }
}

#[test]
fn strict_refusal_preserves_all_unknowns_with_or_without_budget() {
    let source = String::from("driver compiler");
    let rows = [
        unknown(&source),
        accounted(13),
        estimated(3, 17),
        unknown("allocator"),
    ];
    for budget in [None, Some(37)] {
        let allowed = evaluate_memory_requirements(&rows, POLICIES[0], budget, 7).unwrap();
        let strict = evaluate_memory_requirements(&rows, POLICIES[1], budget, 7).unwrap();
        assert_eq!(allowed.decision(), MemoryPolicyDecision::Permitted);
        assert_eq!(
            strict.decision(),
            MemoryPolicyDecision::FiniteEstimateRequired
        );
        assert_eq!(allowed.report(), strict.report());
        assert_eq!(strict.report().contributions(), &rows);
        assert_eq!(strict.report().budget_charge_bytes(), 37);
        assert_eq!(strict.report().unknown_count(), 2);
        let warnings: Vec<_> = allowed.warnings().collect();
        assert_eq!(
            warnings
                .iter()
                .map(|w| w.contribution_index)
                .collect::<Vec<_>>(),
            [0, 3]
        );
        assert_eq!(
            warnings.iter().map(|w| w.source).collect::<Vec<_>>(),
            [source.as_str(), "allocator"]
        );
        assert_eq!(
            strict.report().unknown_contributions().collect::<Vec<_>>(),
            warnings
        );
        assert_eq!(strict.warnings().count(), 0);
    }
}

#[test]
fn headroom_cannot_make_unknown_overhead_finite() {
    let rows = [unknown("driver")];
    for headroom in [0, 1024, u64::MAX] {
        let allowed =
            evaluate_memory_requirements(&rows, POLICIES[0], Some(headroom), headroom).unwrap();
        let strict =
            evaluate_memory_requirements(&rows, POLICIES[1], Some(headroom), headroom).unwrap();
        assert_eq!(allowed.decision(), MemoryPolicyDecision::Permitted);
        assert_eq!(
            strict.decision(),
            MemoryPolicyDecision::FiniteEstimateRequired
        );
        assert_eq!(allowed.report(), strict.report());
        assert_eq!(strict.report().headroom_bytes(), headroom);
        assert_eq!(
            strict.report().estimated_range(),
            FiniteMemoryEstimate::new(0, 0).unwrap()
        );
        assert_eq!(strict.report().unknown_contributions().count(), 1);
        assert_eq!(allowed.warnings().count(), 1);
    }
}

#[test]
fn exhausted_budget_precedes_unknown_refusal_and_keeps_the_report() {
    let rows = [unknown("driver"), estimated(2, 5), accounted(7)];
    for policy in POLICIES {
        let rejected = evaluate_memory_requirements(&rows, policy, Some(14), 3).unwrap();
        assert_eq!(
            rejected.decision(),
            MemoryPolicyDecision::BudgetExceeded {
                required_bytes: 15,
                budget_bytes: 14,
            }
        );
        assert_eq!(rejected.report().contributions(), &rows);
        assert_eq!(
            rejected
                .report()
                .unknown_contributions()
                .next()
                .unwrap()
                .source,
            "driver"
        );
        assert_eq!(rejected.warnings().count(), 0);
    }
}

#[test]
fn ranges_are_ordered_and_maximum_finite_values_are_not_unknown() {
    assert_eq!(
        FiniteMemoryEstimate::new(2, 1),
        Err(MemoryContractError::InvalidEstimateRange {
            lower_bytes: 2,
            upper_bytes: 1,
        })
    );
    for (lower, upper) in [(0, 0), (0, u64::MAX), (1, 1), (u64::MAX, u64::MAX)] {
        let range = FiniteMemoryEstimate::new(lower, upper).unwrap();
        assert_eq!(range.lower_bytes(), lower);
        assert_eq!(range.upper_bytes(), upper);
        let rows = [estimated(lower, upper)];
        let evaluation = evaluate_memory_requirements(&rows, POLICIES[1], Some(upper), 0).unwrap();
        assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
        assert_eq!(evaluation.report().estimated_allowance_bytes(), upper);
        assert_eq!(evaluation.report().unknown_count(), 0);
        assert!(matches!(
            evaluation.report().contributions()[0],
            MemoryContribution::Estimated { .. }
        ));
    }
}

#[test]
fn missing_descriptions_are_errors_even_after_unknown_entries() {
    for empty in ["", " \t\n", "\u{2003}"] {
        let invalid = [
            (
                MemoryContribution::Accounted {
                    source: empty,
                    bytes: 0,
                },
                "source",
            ),
            (
                MemoryContribution::Estimated {
                    source: empty,
                    range: FiniteMemoryEstimate::new(0, 0).unwrap(),
                    basis: "empty input",
                },
                "source",
            ),
            (unknown(empty), "source"),
            (
                MemoryContribution::Estimated {
                    source: "parser",
                    range: FiniteMemoryEstimate::new(0, 0).unwrap(),
                    basis: empty,
                },
                "basis",
            ),
            (
                MemoryContribution::Unknown {
                    source: "driver",
                    reason: empty,
                },
                "reason",
            ),
        ];
        for (invalid, field) in invalid {
            let rows = [unknown("other mechanism"), invalid];
            for policy in POLICIES {
                assert_eq!(
                    evaluate_memory_requirements(&rows, policy, None, 0),
                    Err(MemoryContractError::MissingDescription {
                        contribution_index: 1,
                        field
                    })
                );
            }
        }
    }
}

#[test]
fn finite_sum_overflow_remains_an_error_even_with_unknown_overhead() {
    let cases = [
        ([accounted(u64::MAX), accounted(1)], 0, "accounted bytes"),
        (
            [estimated(u64::MAX, u64::MAX), estimated(1, 1)],
            0,
            "lower estimated bytes",
        ),
        (
            [estimated(0, u64::MAX), estimated(0, 1)],
            0,
            "upper estimated bytes",
        ),
        (
            [accounted(1), estimated(0, u64::MAX)],
            0,
            "accounted bytes plus estimated allowances",
        ),
        (
            [accounted(u64::MAX), estimated(0, 0)],
            1,
            "budget charge including headroom",
        ),
        (
            [accounted(0), estimated(0, u64::MAX)],
            1,
            "budget charge including headroom",
        ),
    ];
    for (finite, headroom, operation) in cases {
        for policy in POLICIES {
            for budget in [None, Some(0), Some(u64::MAX)] {
                for rows in [
                    [unknown("driver"), finite[0], finite[1]],
                    [finite[0], finite[1], unknown("driver")],
                ] {
                    assert_eq!(
                        evaluate_memory_requirements(&rows, policy, budget, headroom),
                        Err(MemoryContractError::ArithmeticOverflow { operation })
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_source_labels_neither_deduplicate_charges_nor_warnings() {
    let rows = [
        accounted(3),
        accounted(5),
        estimated(1, 2),
        estimated(4, 8),
        unknown("driver"),
        unknown("driver"),
    ];
    let evaluation = evaluate_memory_requirements(&rows, POLICIES[0], Some(18), 0).unwrap();
    assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
    assert_eq!(evaluation.report().accounted_bytes(), 8);
    assert_eq!(
        evaluation.report().estimated_range(),
        FiniteMemoryEstimate::new(5, 10).unwrap()
    );
    assert_eq!(evaluation.report().budget_charge_bytes(), 18);
    assert_eq!(evaluation.report().unknown_count(), 2);
    assert_eq!(
        evaluation
            .warnings()
            .map(|w| w.contribution_index)
            .collect::<Vec<_>>(),
        [4, 5]
    );
}

#[test]
fn empty_and_explicit_zero_requirements_keep_their_meaning() {
    let zero = [accounted(0), estimated(0, 0)];
    for rows in [&[][..], &zero[..]] {
        for policy in POLICIES {
            let evaluation = evaluate_memory_requirements(rows, policy, Some(0), 0).unwrap();
            assert_eq!(evaluation.decision(), MemoryPolicyDecision::Permitted);
            assert_eq!(evaluation.report().contributions(), rows);
            assert_eq!(evaluation.report().budget_charge_bytes(), 0);
            assert_eq!(evaluation.report().unknown_count(), 0);
            assert_eq!(evaluation.warnings().count(), 0);
            let headroom = evaluate_memory_requirements(rows, policy, Some(1), 2).unwrap();
            assert_eq!(headroom.report().budget_charge_bytes(), 2);
            assert_eq!(
                headroom.decision(),
                MemoryPolicyDecision::BudgetExceeded {
                    required_bytes: 2,
                    budget_bytes: 1,
                }
            );
        }
    }
}

#[test]
fn headroom_changes_the_charge_without_reclassifying_estimates() {
    let rows = [accounted(11), estimated(3, 7)];
    for policy in POLICIES {
        let ordinary = evaluate_memory_requirements(&rows, policy, None, 0).unwrap();
        let padded = evaluate_memory_requirements(&rows, policy, None, 5).unwrap();
        assert_eq!(ordinary.report().budget_charge_bytes(), 18);
        assert_eq!(padded.report().budget_charge_bytes(), 23);
        assert_eq!(
            ordinary.report().estimated_range(),
            padded.report().estimated_range()
        );
        assert_eq!(
            ordinary.report().contributions(),
            padded.report().contributions()
        );
        assert_eq!(
            ordinary.report().accounted_bytes(),
            padded.report().accounted_bytes()
        );
    }
}
