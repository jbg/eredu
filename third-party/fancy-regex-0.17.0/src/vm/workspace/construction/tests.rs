use super::*;

fn spans(source: &Source, text: &str) -> Vec<(usize, usize)> {
    let mut workspace = source.plan().unwrap().prepare().unwrap();
    workspace
        .find_iter(text)
        .map(|m| {
            let m = m.unwrap();
            (m.start(), m.end())
        })
        .collect()
}

#[test]
fn all_five_fresh_sources_match_existing_source_and_ordinary_iterators() {
    for recipe in SOURCES {
        let owned = String::from(recipe.pattern);
        let plan = Plan::new(&owned).unwrap();
        assert_eq!(plan.source().as_ptr(), owned.as_ptr());
        assert!(plan.required_bytes() > 0);
        let actual = plan.prepare().unwrap();
        drop(owned);
        let existing = Source::new(recipe.pattern).unwrap();
        let ordinary = crate::Regex::new(recipe.pattern).unwrap();
        assert_eq!(actual.options.pattern, recipe.pattern);
        for text in [
            "",
            "I'm café 123456 --!\r\n",
            "世\u{0301}界😀\t \n",
            " x y z 23.25\u{00a0}",
            "\u{10ffff}\u{0000}",
        ] {
            let a = spans(&actual, text);
            let b = spans(&existing, text);
            let c: Vec<_> = ordinary
                .find_iter(text)
                .map(|m| {
                    let m = m.unwrap();
                    (m.start(), m.end())
                })
                .collect();
            assert_eq!(a, b, "{} / {:?}", recipe.pattern, text);
            assert_eq!(a, c, "{} / {:?}", recipe.pattern, text);
        }
        for (a, b) in actual.delegates().zip(existing.delegates()) {
            assert_eq!(a.get_nfa().states(), b.get_nfa().states());
            assert_ne!(a.get_nfa().states().as_ptr(), b.get_nfa().states().as_ptr());
        }
    }
}

#[test]
fn actual_pattern_program_and_each_literal_frontier_retains_prefix() {
    let recipe = &SOURCES[0];
    for fault in [Fault::Pattern, Fault::Instructions] {
        let error = Plan::new(recipe.pattern)
            .unwrap()
            .prepare_inner(Some(fault))
            .unwrap_err();
        assert!(matches!(&error.cause, Cause::Reserve(_)));
        assert_eq!(
            error.storage.pattern.capacity(),
            if fault == Fault::Pattern {
                0
            } else {
                recipe.pattern.len()
            }
        );
        assert!(error.storage.scratch.is_none());
    }
    let BodyRecipe::Fancy { instructions, .. } = recipe.body else {
        panic!("emitted fancy source")
    };
    for (index, instruction) in instructions.iter().enumerate() {
        if !matches!(instruction, InsnRecipe::Lit(_)) {
            continue;
        }
        let error = Plan::new(recipe.pattern)
            .unwrap()
            .prepare_inner(Some(Fault::Literal(index)))
            .unwrap_err();
        assert!(matches!(&error.cause, Cause::Reserve(_)));
        assert_eq!(error.storage.instructions.len(), index);
        assert_eq!(error.storage.literal.capacity(), 0);
        assert_eq!(error.storage.pattern, recipe.pattern);
        assert!(error.storage.scratch.is_some());
    }
}

#[test]
fn nested_real_last_transition_failure_retains_completed_delegates() {
    for recipe in SOURCES {
        let BodyRecipe::Fancy { instructions, .. } = recipe.body else {
            panic!("emitted fancy source")
        };
        for (index, instruction) in instructions.iter().enumerate() {
            let InsnRecipe::Pike { ordinal, .. } = instruction else {
                continue;
            };
            let error = Plan::new(recipe.pattern)
                .unwrap()
                .prepare_inner(Some(Fault::Delegate(*ordinal)))
                .unwrap_err();
            assert!(matches!(&error.cause, Cause::Delegate(_)));
            assert_eq!(error.storage.instructions.len(), index);
            assert_eq!(
                error
                    .storage
                    .instructions
                    .iter()
                    .filter(|i| matches!(i, Insn::PikeDelegate(_)))
                    .count(),
                *ordinal
            );
            assert!(error.storage.scratch.is_some());
            assert_eq!(error.storage.pattern, recipe.pattern);
            #[cfg(feature = "std")]
            {
                use std::error::Error;
                let mut cause: &dyn Error = &error;
                while let Some(next) = cause.source() {
                    cause = next;
                }
                assert!(cause.downcast_ref::<TryReserveError>().is_some());
            }
        }
    }
}

#[test]
fn completed_rejection_keeps_real_source_and_shared_scratch() {
    for recipe in SOURCES {
        let error = Plan::new(recipe.pattern)
            .unwrap()
            .prepare_inner(Some(Fault::Completed))
            .unwrap_err();
        assert!(matches!(&error.cause, Cause::Profile(_)));
        assert!(error.storage.scratch.is_some());
        let source = error.storage.source.as_ref().unwrap();
        assert_eq!(source.options.pattern, recipe.pattern);
        assert!(!spans(source, "actual nonzero 123 café").is_empty());
        assert!(error.storage.instructions.is_empty());
        assert!(error.storage.pattern.is_empty());
    }
}

#[test]
fn exact_source_profile_and_checked_layout_errors_precede_storage() {
    for pattern in ["", "a+", "(?i:a)", "\\s+"] {
        assert_eq!(Plan::new(pattern).unwrap_err(), PlanError::Profile);
    }
    assert_eq!(array::<Insn>(usize::MAX).unwrap_err(), PlanError::Overflow);
    assert_eq!(add(usize::MAX, 1).unwrap_err(), PlanError::Overflow);
    let altered = alloc::format!("(?:{})", SOURCES[0].pattern);
    assert_eq!(Plan::new(&altered).unwrap_err(), PlanError::Profile);
    assert!(Plan::new(SOURCES[0].pattern).unwrap().prepare().is_ok());
}

#[cfg(all(feature = "workspace-test-support", feature = "std"))]
#[test]
fn development_selectors_fail_actual_source_scratch_and_late_delegate_targets() {
    use std::error::Error as _;
    let source = patterns().next().unwrap();
    for target in [
        ConstructionFailure::Pattern,
        ConstructionFailure::Instructions,
        ConstructionFailure::FirstLiteral,
        ConstructionFailure::LastDelegate,
        ConstructionFailure::Scratch(ScratchFailureBuffer::Stack),
        ConstructionFailure::Scratch(ScratchFailureBuffer::Dense),
        ConstructionFailure::Scratch(ScratchFailureBuffer::Sparse),
    ] {
        let error = Plan::new(source)
            .unwrap()
            .prepare_failing(target)
            .unwrap_err();
        let mut cause = error.source();
        let mut actual_reserve = false;
        while let Some(error) = cause {
            actual_reserve |= error.is::<alloc::collections::TryReserveError>();
            cause = error.source();
        }
        assert!(actual_reserve, "{target:?}: {error:?}");
        if !matches!(target, ConstructionFailure::Pattern) {
            assert!(!error.storage.pattern.is_empty());
        }
    }
    let error = Plan::new(source)
        .unwrap()
        .prepare_failing(ConstructionFailure::Completed)
        .unwrap_err();
    assert!(error.storage.source.is_some());
    assert!(matches!(
        error.cause,
        Cause::Profile(WorkspacePlanError::Geometry)
    ));
    // This is an explicit injected completed-source rejection, not a natural failure.
    let no_literals = patterns().nth(2).unwrap();
    let error = Plan::new(no_literals)
        .unwrap()
        .prepare_failing(ConstructionFailure::FirstLiteral)
        .unwrap_err();
    assert!(matches!(
        error.cause,
        Cause::Profile(WorkspacePlanError::Geometry)
    ));
    assert_eq!(error.storage.pattern.capacity(), 0);
}
