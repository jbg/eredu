use super::*;

#[cfg(not(feature = "internal-instrument-pikevm"))]
mod enabled {
    use super::*;
    use crate::{Anchored, MatchKind};
    use alloc::vec;

    #[test]
    fn repeated_unicode_captures_share_existing_worker_without_growth() {
        let source =
            PikeVM::new(r"(?P<word>(?:ab|a|αβ|[A-Z]|[0-9])+)(?:\s+(β|b))?")
                .unwrap();
        let plan = source.workspace_plan().unwrap();
        let requirements = plan.requirements();
        let mut workspace = plan.prepare().unwrap();
        assert!(core::ptr::eq(workspace.source, &source));
        let before = workspace.capacities();
        for (index, buffer) in Buffer::ALL.iter().copied().enumerate() {
            assert!(before[index] >= requirements.capacity(buffer));
        }
        let mut ordinary = source.create_cache();
        for _ in 0..5 {
            for text in
                ["!abαβZ12 β", "ab b", "no match!", "", "αβ", "aaaaaab"]
            {
                for width in [
                    requirements.minimum_slots(),
                    requirements.maximum_slots(),
                ] {
                    let mut expected = vec![None; width];
                    let mut actual = vec![None; width];
                    let input = Input::new(text);
                    let matched = source.search_slots(
                        &mut ordinary,
                        &input,
                        &mut expected,
                    );
                    assert_eq!(
                        workspace.search_slots(&input, &mut actual).unwrap(),
                        matched
                    );
                    assert_eq!(actual, expected);
                    assert_eq!(workspace.capacities(), before);
                }
            }
        }
        let mut slots = vec![None; requirements.maximum_slots()];
        assert!(workspace
            .search_slots(&Input::new("!abαβZ12 β"), &mut slots)
            .unwrap()
            .is_some());
        assert_eq!(slots[0].unwrap().get(), 1);
        assert_eq!(slots[1].unwrap().get(), "!abαβZ12 β".len());
    }

    #[test]
    fn empty_utf8_and_invalid_slot_widths_preserve_destinations() {
        let source = PikeVM::new(r"(?:a|)").unwrap();
        let mut workspace =
            source.workspace_plan().unwrap().prepare().unwrap();
        let before = workspace.capacities();
        let marker = Some(NonMaxUsize::new(17).unwrap());
        for width in [0, 1, workspace.requirements().maximum_slots() + 1] {
            let mut slots = vec![marker; width];
            let cache_before = workspace.cache.memory_usage();
            assert_eq!(
                workspace.search_slots(&Input::new("☃"), &mut slots),
                Err(SearchError::SlotCount {
                    minimum: 2,
                    maximum: workspace.requirements().maximum_slots(),
                    supplied: width,
                })
            );
            assert_eq!(slots, vec![marker; width]);
            assert_eq!(workspace.capacities(), before);
            assert_eq!(workspace.cache.memory_usage(), cache_before);
        }
        let mut ordinary = source.create_cache();
        for start in 0..="☃a".len() {
            // Include UTF-8 continuation-byte starts, where the existing
            // worker must skip an empty match splitting a code point.
            let input = Input::new("☃a").span(start.."☃a".len());
            let mut expected = [None; 2];
            let mut actual = [None; 2];
            let matched =
                source.search_slots(&mut ordinary, &input, &mut expected);
            assert_eq!(
                workspace.search_slots(&input, &mut actual).unwrap(),
                matched
            );
            assert_eq!(actual, expected);
            if let Some(offset) = actual[0] {
                assert!("☃a".is_char_boundary(offset.get()));
            }
            assert_eq!(workspace.capacities(), before);
        }
    }

    #[test]
    fn every_actual_reserve_failure_retains_its_partial_prefix() {
        let source = PikeVM::new(r"((?:ab|αβ|a)+)(β?)").unwrap();
        let mut retained = Vec::new();
        for (frontier, buffer) in Buffer::ALL.iter().copied().enumerate() {
            let mut plan = source.workspace_plan().unwrap();
            let requirements = plan.requirements();
            plan.fail = Some(buffer);
            let error = plan.prepare().unwrap_err();
            assert_eq!(error.buffer(), buffer);
            assert_eq!(error.requirements(), requirements);
            assert_eq!(
                error.source_state_count(),
                source.get_nfa().states().len()
            );
            assert!(core::ptr::eq(error.source, &source));
            for (index, prior) in Buffer::ALL.iter().copied().enumerate() {
                if index < frontier {
                    assert!(
                        error.capacities()[index]
                            >= requirements.capacity(prior)
                    );
                } else {
                    assert_eq!(error.capacities()[index], 0);
                }
            }
            #[cfg(feature = "std")]
            {
                use std::error::Error;
                let cause = Error::source(&error)
                    .unwrap()
                    .downcast_ref::<TryReserveError>()
                    .unwrap();
                assert!(core::ptr::eq(cause, error.cause()));
            }
            retained.push(error);
        }
        // These are seven independent, consumed plans. There is no method to
        // retry any retained failure. A fresh plan still binds this source.
        let mut workspace =
            source.workspace_plan().unwrap().prepare().unwrap();
        let mut slots = [None; 2];
        assert!(workspace
            .search_slots(&Input::new("abαββ"), &mut slots)
            .unwrap()
            .is_some());
        assert_eq!(retained.len(), 7);
        drop(workspace);
        drop(retained);
        let mut ordinary = source.create_cache();
        assert!(source
            .search_slots(&mut ordinary, &Input::new("ab"), &mut slots)
            .is_some());
    }

    #[test]
    fn actual_multipattern_empty_source_rejects_before_prepare() {
        let source = PikeVM::new_many(&["", "(?P<a>a+)"]).unwrap();
        assert_eq!(
            source.workspace_plan().unwrap_err(),
            PlanError::MultiplePatterns
        );
        // The ordinary API is unchanged and still supports this source.
        let mut ordinary = source.create_cache();
        let mut slots = vec![None; source.get_nfa().group_info().slot_len()];
        assert!(source
            .search_slots(&mut ordinary, &Input::new("☃aa"), &mut slots)
            .is_some());
        assert_eq!(slots[0].unwrap().get(), 0);
        assert_eq!(slots[1].unwrap().get(), 0);
    }

    #[cfg(feature = "perf-literal")]
    #[test]
    fn actual_prefilter_is_rejected_without_changing_legacy_search() {
        use crate::util::prefilter::Prefilter;
        let prefilter =
            Prefilter::new(MatchKind::LeftmostFirst, &["foo", "bar"]);
        assert!(prefilter.is_some());
        let source = PikeVM::builder()
            .configure(PikeVM::config().prefilter(prefilter))
            .build(r"(foo|bar)[a-z]+")
            .unwrap();
        assert_eq!(source.workspace_plan().unwrap_err(), PlanError::Prefilter);
        let mut ordinary = source.create_cache();
        let mut slots = [None; 2];
        assert!(source
            .search_slots(&mut ordinary, &Input::new("!barfox"), &mut slots)
            .is_some());
        assert_eq!(slots[0].unwrap().get(), 1);
        assert_eq!(slots[1].unwrap().get(), 7);
    }

    #[test]
    fn anchored_and_match_kind_inputs_keep_ordinary_semantics() {
        for kind in [MatchKind::LeftmostFirst, MatchKind::All] {
            let source = PikeVM::builder()
                .configure(PikeVM::config().match_kind(kind))
                .build(r"(a|ab|abc)+")
                .unwrap();
            let mut workspace =
                source.workspace_plan().unwrap().prepare().unwrap();
            let before = workspace.capacities();
            let mut ordinary = source.create_cache();
            for anchored in [
                Anchored::No,
                Anchored::Yes,
                Anchored::Pattern(PatternID::must(0)),
                Anchored::Pattern(PatternID::must(1)),
            ] {
                for earliest in [false, true] {
                    let input = Input::new("!abcab")
                        .span(1..6)
                        .anchored(anchored)
                        .earliest(earliest);
                    let mut expected = [None; 4];
                    let mut actual = [None; 4];
                    let matched = source.search_slots(
                        &mut ordinary,
                        &input,
                        &mut expected,
                    );
                    assert_eq!(
                        workspace.search_slots(&input, &mut actual).unwrap(),
                        matched
                    );
                    assert_eq!(actual, expected);
                    assert_eq!(workspace.capacities(), before);
                }
            }
        }
    }

    #[test]
    fn checked_geometry_rejects_unrepresentable_layouts_before_reserve() {
        // These call the same pure checked geometry used after actual NFA
        // inspection. They are overflow proofs, not invented source authority.
        assert_eq!(
            Requirements::checked(usize::MAX, 2, 2, 1),
            Err(PlanError::CapacityOverflow)
        );
        assert_eq!(
            Requirements::checked(2, usize::MAX, 2, 1),
            Err(PlanError::CapacityOverflow)
        );
        assert_eq!(
            Requirements::checked(2, 2, 2, usize::MAX),
            Err(PlanError::CapacityOverflow)
        );
        assert_eq!(
            Requirements::checked(2, 1, 2, 1),
            Err(PlanError::CapacityOverflow)
        );
        let source = PikeVM::new(r"([a-z]+)\s+(\p{Greek}+)").unwrap();
        let requirements = source.workspace_plan().unwrap().requirements();
        assert!(requirements.required_bytes() > requirements.buffer_bytes());
        assert!(requirements.control_bytes() > 0);
        assert!(source.workspace_plan().unwrap().prepare().is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn scoped_workers_keep_independent_workspaces_on_one_source() {
        let source = PikeVM::new(r"(αβ|ab)+\s*(z?)").unwrap();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..4 {
                let borrowed = &source;
                handles.push(scope.spawn(move || {
                    let mut workspace =
                        borrowed.workspace_plan().unwrap().prepare().unwrap();
                    let before = workspace.capacities();
                    let mut ordinary = borrowed.create_cache();
                    for _ in 0..16 {
                        let input = Input::new("!abαβ z");
                        let mut actual = [None; 6];
                        let mut expected = [None; 6];
                        let matched = borrowed.search_slots(
                            &mut ordinary,
                            &input,
                            &mut expected,
                        );
                        assert_eq!(
                            workspace
                                .search_slots(&input, &mut actual)
                                .unwrap(),
                            matched
                        );
                        assert_eq!(actual, expected);
                        assert!(matched.is_some());
                        assert_eq!(workspace.capacities(), before);
                    }
                    before
                }));
            }
            // No barriers or pending callbacks: an assertion failure cannot
            // strand another worker. Join every handle before asserting.
            let results: Vec<_> =
                handles.into_iter().map(|h| h.join()).collect();
            assert!(results.iter().all(|r| r.is_ok()));
        });
    }
}

#[cfg(feature = "internal-instrument-pikevm")]
#[test]
fn allocating_instrumentation_rejects_workspace_but_preserves_legacy_api() {
    let source = PikeVM::new(r"(ab)+").unwrap();
    assert_eq!(
        source.workspace_plan().unwrap_err(),
        PlanError::Instrumentation
    );
    let mut ordinary = source.create_cache();
    let mut slots = [None; 2];
    assert!(source
        .search_slots(&mut ordinary, &Input::new("!abab"), &mut slots)
        .is_some());
    assert_eq!(slots[0].unwrap().get(), 1);
    assert_eq!(slots[1].unwrap().get(), 5);
}
