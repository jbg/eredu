use super::*;

#[test]
fn every_emitted_source_reconstructs_actual_states_properties_and_search() {
    let layout = NFAS
        .iter()
        .map(|r| Plan::new(r.pattern).unwrap().scratch_layout())
        .reduce(ScratchLayout::union)
        .unwrap();
    let mut scratch = layout.prepare().unwrap();
    let capacities =
        (scratch.stack.capacity(), scratch.seen.workspace_capacities());
    for recipe in NFAS {
        let plan = Plan::new(recipe.pattern).unwrap();
        assert!(plan.required_bytes() > 0);
        let actual = plan.prepare(&mut scratch).unwrap();
        let ordinary = PikeVM::new(recipe.pattern).unwrap();
        let a = actual.get_nfa();
        let b = ordinary.get_nfa();
        assert_eq!(a.states(), b.states(), "{}", recipe.pattern);
        assert_ne!(a.states().as_ptr(), b.states().as_ptr());
        assert_eq!(a.start_anchored(), b.start_anchored());
        assert_eq!(a.start_unanchored(), b.start_unanchored());
        assert_eq!(a.has_empty(), b.has_empty());
        assert_eq!(a.look_set_any(), b.look_set_any());
        assert_eq!(a.look_set_prefix_any(), b.look_set_prefix_any());
        assert_eq!(a.group_info().slot_len(), b.group_info().slot_len());
        assert_eq!(
            a.group_info().slots(PatternID::ZERO, 0),
            b.group_info().slots(PatternID::ZERO, 0)
        );
        let mut ac = actual.create_cache();
        let mut bc = ordinary.create_cache();
        for text in [
            "",
            "a",
            " café 世界123\r\n",
            "I'm 123456 --!\t\n",
            "\u{0301}\u{200d}😀 \u{00a0}",
        ] {
            let am: Vec<_> = actual.find_iter(&mut ac, text).collect();
            let bm: Vec<_> = ordinary.find_iter(&mut bc, text).collect();
            assert_eq!(am, bm, "{} / {:?}", recipe.pattern, text);
        }
        assert_eq!(
            (scratch.stack.capacity(), scratch.seen.workspace_capacities()),
            capacities
        );
        assert!(scratch.stack.is_empty());
    }
}

#[test]
fn actual_state_start_capture_and_nested_failures_retain_prefixes() {
    for recipe in NFAS {
        let ordinary = PikeVM::new(recipe.pattern).unwrap();
        let mut scratch = Plan::new(recipe.pattern)
            .unwrap()
            .scratch_layout()
            .prepare()
            .unwrap();
        let mut faults = alloc::vec![Fault::States, Fault::Starts];
        for buffer in [
            groups::Buffer::Ranges,
            groups::Buffer::Maps,
            groups::Buffer::Rows,
            groups::Buffer::Names,
        ] {
            faults.push(Fault::Groups(buffer));
        }
        for (index, state) in recipe.states.iter().enumerate() {
            match state {
                StateRecipe::Sparse(_) => {
                    faults.push(Fault::Transitions(index))
                }
                StateRecipe::Union(_) => faults.push(Fault::Alternates(index)),
                _ => {}
            }
        }
        for fault in faults {
            let error = Plan::new(recipe.pattern)
                .unwrap()
                .prepare_inner(&mut scratch, Some(fault))
                .unwrap_err();
            match fault {
                Fault::States => {
                    assert_eq!(error.storage.states.capacity(), 0);
                    assert!(matches!(&error.cause, Cause::Reserve(_)));
                }
                Fault::Starts => {
                    assert_eq!(
                        error.storage.states.capacity(),
                        recipe.states.len()
                    );
                    assert_eq!(error.storage.starts.capacity(), 0);
                    assert!(matches!(&error.cause, Cause::Reserve(_)));
                }
                Fault::Groups(_) => {
                    assert_eq!(
                        error.storage.states.capacity(),
                        recipe.states.len()
                    );
                    assert_eq!(error.storage.starts.capacity(), 1);
                    assert!(matches!(&error.cause, Cause::Groups(_)));
                }
                Fault::Transitions(index) | Fault::Alternates(index) => {
                    assert_eq!(
                        error.storage.inner.as_ref().unwrap().states.len(),
                        index
                    );
                    assert_eq!(
                        error
                            .storage
                            .inner
                            .as_ref()
                            .unwrap()
                            .states
                            .capacity(),
                        recipe.states.len()
                    );
                    assert!(matches!(&error.cause, Cause::Reserve(_)));
                    assert_eq!(
                        error
                            .storage
                            .inner
                            .as_ref()
                            .unwrap()
                            .states
                            .as_slice(),
                        &ordinary.get_nfa().states()[..index]
                    );
                }
            }
            let pointers = (
                error.storage.states.as_ptr(),
                error.storage.starts.as_ptr(),
                error.storage.inner.as_ref().map(|inner| {
                    (inner.states.as_ptr(), inner.start_pattern.as_ptr())
                }),
            );
            let error = core::hint::black_box(error);
            assert_eq!(
                (
                    error.storage.states.as_ptr(),
                    error.storage.starts.as_ptr(),
                    error.storage.inner.as_ref().map(|inner| (
                        inner.states.as_ptr(),
                        inner.start_pattern.as_ptr()
                    ))
                ),
                pointers
            );
            drop(error);
        }
        // Failure is terminal for each consumed plan; a new actual plan may
        // still use the same unmodified, fully prepared scratch.
        assert!(Plan::new(recipe.pattern)
            .unwrap()
            .prepare(&mut scratch)
            .is_ok());
    }
}

#[test]
fn each_real_scratch_reserve_failure_keeps_previous_capacity() {
    let layout = Plan::new(NFAS[0].pattern).unwrap().scratch_layout();
    for buffer in
        [ScratchBuffer::Stack, ScratchBuffer::Dense, ScratchBuffer::Sparse]
    {
        let error = layout.prepare_inner(Some(buffer)).unwrap_err();
        assert!(error.cause.is_some());
        let (dense, sparse) = error.scratch.seen.workspace_capacities();
        assert_eq!(
            error.scratch.stack.capacity(),
            if buffer == ScratchBuffer::Stack { 0 } else { layout.stack }
        );
        assert_eq!(
            dense,
            if buffer == ScratchBuffer::Sparse { layout.states } else { 0 }
        );
        assert_eq!(sparse, 0);
    }
}

#[test]
fn wrong_scratch_and_foreign_source_reject_without_construction() {
    assert_eq!(
        Plan::new("a source outside the exact inventory").unwrap_err(),
        PlanError::Profile
    );
    let small = NFAS.iter().min_by_key(|r| r.states.len()).unwrap();
    let large = NFAS.iter().max_by_key(|r| r.states.len()).unwrap();
    let mut scratch =
        Plan::new(small.pattern).unwrap().scratch_layout().prepare().unwrap();
    let error =
        Plan::new(large.pattern).unwrap().prepare(&mut scratch).unwrap_err();
    assert!(matches!(error.cause, Cause::Scratch));
    assert_eq!(error.storage.states.capacity(), 0);
    assert!(error.storage.inner.is_none());
    assert!(Plan::new(small.pattern).unwrap().prepare(&mut scratch).is_ok());
}

const fn syntax() -> SyntaxRecipe {
    SyntaxRecipe {
        unicode: true,
        case_insensitive: false,
        multi_line: false,
        dot_matches_new_line: false,
        crlf: false,
        line_terminator: 10,
        swap_greed: false,
        ignore_whitespace: false,
        utf8: true,
        nest_limit: 250,
        octal: false,
    }
}
const EMPTY: NfaRecipe = NfaRecipe {
    pattern: "",
    syntax: syntax(),
    anchored: 2,
    unanchored: 0,
    utf8: true,
    reverse: false,
    line_terminator: 10,
    starts: &[2],
    groups: &[&[(0, 1)]],
    states: &[
        StateRecipe::BinaryUnion(2, 1),
        StateRecipe::ByteRange(0, 255, 0),
        StateRecipe::Capture { next: 3, pattern: 0, group: 0, slot: 0 },
        StateRecipe::Capture { next: 4, pattern: 0, group: 0, slot: 1 },
        StateRecipe::Match(0),
    ],
    expected: PropertiesRecipe {
        has_empty: true,
        has_capture: true,
        look_any: &[],
        look_prefix: &[],
        byte_classes: &[0; 256],
    },
};
const START: NfaRecipe = NfaRecipe {
    pattern: "^",
    syntax: syntax(),
    anchored: 2,
    unanchored: 0,
    utf8: true,
    reverse: false,
    line_terminator: 10,
    starts: &[2],
    groups: &[&[(0, 1)]],
    states: &[
        StateRecipe::BinaryUnion(2, 1),
        StateRecipe::ByteRange(0, 255, 0),
        StateRecipe::Capture { next: 3, pattern: 0, group: 0, slot: 0 },
        StateRecipe::Look(Look::Start, 4),
        StateRecipe::Capture { next: 5, pattern: 0, group: 0, slot: 1 },
        StateRecipe::Match(0),
    ],
    expected: PropertiesRecipe {
        has_empty: true,
        has_capture: true,
        look_any: &[Look::Start],
        look_prefix: &[Look::Start],
        byte_classes: &[0; 256],
    },
};
const fn ab_classes() -> [u8; 256] {
    let mut values = [0; 256];
    let mut index = 0;
    while index < 256 {
        values[index] = if index < 97 {
            0
        } else if index == 97 {
            1
        } else if index == 98 {
            2
        } else {
            3
        };
        index += 1;
    }
    values
}
const ORDERED: NfaRecipe = NfaRecipe {
    pattern: "a|ab",
    syntax: syntax(),
    anchored: 2,
    unanchored: 0,
    utf8: true,
    reverse: false,
    line_terminator: 10,
    starts: &[2],
    groups: &[&[(0, 1)]],
    states: &[
        StateRecipe::BinaryUnion(2, 1),
        StateRecipe::ByteRange(0, 255, 0),
        StateRecipe::Capture { next: 3, pattern: 0, group: 0, slot: 0 },
        StateRecipe::Union(&[4, 6]),
        StateRecipe::ByteRange(b'a', b'a', 5),
        StateRecipe::Capture { next: 8, pattern: 0, group: 0, slot: 1 },
        StateRecipe::ByteRange(b'a', b'a', 7),
        StateRecipe::ByteRange(b'b', b'b', 5),
        StateRecipe::Match(0),
    ],
    expected: PropertiesRecipe {
        has_empty: false,
        has_capture: true,
        look_any: &[],
        look_prefix: &[],
        byte_classes: &ab_classes(),
    },
};

#[test]
fn private_empty_look_and_ordered_recipes_preserve_utf8_iteration() {
    for recipe in [&EMPTY, &START, &ORDERED] {
        let plan = Plan::from_recipe(recipe.pattern, recipe).unwrap();
        let mut scratch = plan.scratch_layout().prepare().unwrap();
        let actual = plan.prepare(&mut scratch).unwrap();
        let ordinary = PikeVM::new(recipe.pattern).unwrap();
        let mut a = actual.create_cache();
        let mut b = ordinary.create_cache();
        for text in ["", "é😀", "ab ab a", "\n世a"] {
            let aa: Vec<_> = actual.find_iter(&mut a, text).collect();
            let bb: Vec<_> = ordinary.find_iter(&mut b, text).collect();
            assert_eq!(aa, bb, "{} / {:?}", recipe.pattern, text);
            assert!(aa.iter().all(|m| text.is_char_boundary(m.start())
                && text.is_char_boundary(m.end())));
        }
    }
}

#[test]
fn malformed_private_geometry_and_layout_overflow_reject_before_storage() {
    let mut bad = EMPTY;
    bad.anchored = usize::MAX;
    assert_eq!(validate(&bad).unwrap_err(), PlanError::Recipe);
    let mut bad = EMPTY;
    bad.states = &[StateRecipe::ByteRange(7, 1, 0)];
    assert_eq!(validate(&bad).unwrap_err(), PlanError::Recipe);
    let mut bad = EMPTY;
    bad.groups = &[&[(0, 7)]];
    assert_eq!(validate(&bad).unwrap_err(), PlanError::Recipe);
    assert_eq!(
        groups::array::<State>(usize::MAX).unwrap_err(),
        groups::Overflow
    );
    assert_eq!(add(usize::MAX, 1).unwrap_err(), PlanError::Overflow);
}
