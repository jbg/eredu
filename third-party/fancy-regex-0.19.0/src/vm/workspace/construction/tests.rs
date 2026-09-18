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
fn all_fresh_sources_match_existing_source_and_ordinary_iterators() {
    for recipe in SOURCES {
        let owned = String::from(recipe.pattern);
        let plan = Plan::new(&owned).unwrap();
        assert_eq!(plan.source().as_ptr(), owned.as_ptr());
        assert!(plan.required_bytes() > 0);
        let actual = plan.prepare().unwrap();
        drop(owned);
        let existing = Source::new(recipe.pattern).unwrap();
        let ordinary = crate::Regex::new(recipe.pattern).unwrap();
        assert_eq!(actual.pattern, recipe.pattern);
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
    }
    let instructions = recipe.instructions;
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
    }
}

#[test]
fn nested_real_last_table_failure_retains_completed_delegates() {
    for recipe in SOURCES {
        let instructions = recipe.instructions;
        for (index, instruction) in instructions.iter().enumerate() {
            let InsnRecipe::Regular { ordinal, .. } = instruction else {
                continue;
            };
            let error = Plan::new(recipe.pattern)
                .unwrap()
                .prepare_inner(Some(Fault::Delegate(*ordinal)))
                .unwrap_err();
            assert!(matches!(&error.cause, Cause::Dfa(_)));
            assert_eq!(error.storage.instructions.len(), index);
            assert_eq!(
                error
                    .storage
                    .instructions
                    .iter()
                    .filter(|i| matches!(i, Insn::DfaDelegate(_)))
                    .count(),
                *ordinal
            );
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
fn completed_rejection_keeps_real_source_tables() {
    for recipe in SOURCES {
        let error = Plan::new(recipe.pattern)
            .unwrap()
            .prepare_inner(Some(Fault::Completed))
            .unwrap_err();
        assert!(matches!(&error.cause, Cause::Profile(_)));
        let source = error.storage.source.as_ref().unwrap();
        assert_eq!(source.pattern, recipe.pattern);
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

#[test]
fn every_character_class_reserve_retains_the_completed_source_prefix() {
    let mut reached = 0;
    for recipe in SOURCES {
        for (index, instruction) in recipe.instructions.iter().enumerate() {
            if !matches!(
                instruction,
                InsnRecipe::CharClassCodepoint(_) | InsnRecipe::CharClassByte(_)
            ) {
                continue;
            }
            reached += 1;
            let failure = Plan::new(recipe.pattern)
                .unwrap()
                .prepare_inner(Some(Fault::Class(index)))
                .unwrap_err();
            assert!(matches!(failure.cause, Cause::Reserve(_)));
            assert_eq!(failure.storage.instructions.len(), index);
            assert_eq!(failure.storage.codepoint_ranges.capacity(), 0);
            assert_eq!(failure.storage.byte_ranges.capacity(), 0);
            assert_eq!(failure.storage.pattern, recipe.pattern);
        }
    }
    assert!(reached > 0);
}

#[test]
fn fresh_source_heap_geometry_includes_character_ranges_and_all_tables() {
    for recipe in SOURCES {
        let plan = Plan::new(recipe.pattern).unwrap();
        let planned = plan.heap_bytes();
        let source = plan.prepare().unwrap();
        let Body::Fancy(prog) = &source.body else {
            panic!("fixed source program")
        };
        assert!(prog.scratch_pool.is_none());
        let mut heap = source.pattern.capacity()
            + prog.body.capacity() * mem::size_of::<Insn>()
            + prog.seek_pattern.capacity();
        for instruction in &prog.body {
            heap += match instruction {
                Insn::Lit(literal) => literal.capacity(),
                Insn::CharClass(crate::vm::CharClassMatcher::Codepoint(ranges)) => {
                    mem::size_of_val(&**ranges)
                }
                Insn::CharClass(crate::vm::CharClassMatcher::Byte(ranges)) => {
                    mem::size_of_val(&**ranges)
                }
                Insn::DfaDelegate(dfa) => dfa.memory_usage() + mem::size_of_val(&**dfa),
                _ => 0,
            };
        }
        assert_eq!(heap, planned, "{}", recipe.pattern);
    }
}
