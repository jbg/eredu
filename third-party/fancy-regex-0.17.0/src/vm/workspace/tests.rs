use super::*;
use alloc::{vec, vec::Vec};

fn spans(regex: &crate::Regex, text: &str) -> Vec<(usize, usize)> {
    regex
        .find_iter(text)
        .map(|found| {
            let found = found.unwrap();
            (found.start(), found.end())
        })
        .collect()
}
fn prepared_spans(workspace: &mut Workspace<'_>, text: &str) -> Vec<(usize, usize)> {
    workspace
        .find_iter(text)
        .map(|found| {
            let found = found.unwrap();
            (found.start(), found.end())
        })
        .collect()
}
fn storage_identity(workspace: &Workspace<'_>) -> ([usize; 5], [usize; 5], Vec<[usize; 7]>) {
    let storage = &workspace.storage;
    (
        workspace.capacities(),
        [
            storage.delegates.as_ptr() as usize,
            storage.state.saves.as_ptr() as usize,
            storage.state.stack.as_ptr() as usize,
            storage.state.oldsave.as_ptr() as usize,
            storage.slots.as_ptr() as usize,
        ],
        storage
            .delegates
            .iter()
            .map(|workspace| workspace.capacities())
            .collect(),
    )
}

#[test]
fn direct_wrap_and_actual_vm_share_nonzero_iteration_and_capture_fixup() {
    for pattern in [
        r"\w+",
        r"\s+(?!\S)|\s+|\w+",
        r"(a+)(?!b)|β+",
        r"a\Kb+",
        r"(?=ab)ab|c",
    ] {
        let ordinary = crate::Regex::new(pattern).unwrap();
        let source = Source::new(pattern).unwrap();
        let mut workspace = source.plan().unwrap().prepare().unwrap();
        let identity = storage_identity(&workspace);
        for text in [
            "abbb ββ aaaa  \n ",
            "ab aaaβ c",
            "",
            "nothing  ",
            "ββ🙂 aaa",
        ] {
            assert_eq!(
                prepared_spans(&mut workspace, text),
                spans(&ordinary, text),
                "{pattern}: {text:?}"
            );
            assert_eq!(storage_identity(&workspace), identity);
            for pos in text
                .char_indices()
                .map(|(i, _)| i)
                .chain(core::iter::once(text.len()))
            {
                let expected = ordinary
                    .find_from_pos(text, pos)
                    .unwrap()
                    .map(|m| (m.start(), m.end()));
                let actual = workspace
                    .find_from_pos(text, pos)
                    .unwrap()
                    .map(|m| (m.start(), m.end()));
                assert_eq!(actual, expected, "{pattern}: {text:?} at {pos}");
            }
        }
    }
}

#[test]
fn empty_matches_utf8_end_and_skipped_empty_use_the_shared_progression() {
    for pattern in ["", r"(?=.)", r"a*", r"(?!x)", r"(?m:$)|a"] {
        let ordinary = crate::Regex::new(pattern).unwrap();
        let source = Source::new(pattern).unwrap();
        let mut workspace = source.plan().unwrap().prepare().unwrap();
        for text in ["", "é🙂a\n", "aaa", "xéx", "\r\n"] {
            assert_eq!(prepared_spans(&mut workspace, text), spans(&ordinary, text));
        }
        let before = storage_identity(&workspace);
        assert!(matches!(
            workspace.find_from_pos("é", 1),
            Err(Error::RuntimeError(RuntimeError::WorkspaceInputBoundary))
        ));
        assert!(matches!(
            workspace.find_from_pos("é", 3),
            Err(Error::RuntimeError(RuntimeError::WorkspaceInputBoundary))
        ));
        assert_eq!(storage_identity(&workspace), before);
        assert_eq!(prepared_spans(&mut workspace, "éa"), spans(&ordinary, "éa"));
    }
}

#[test]
fn checked_profile_rejects_allocating_instruction_families_and_legacy_remains_available() {
    for pattern in [
        r"(a+)\1",
        r"(?>a+)b",
        r"a++b",
        r"(?<=a)b",
        r"(?<!a+)b",
        r"\Ga",
        r"(a)?(?(1)b|c)",
    ] {
        assert!(crate::Regex::new(pattern).is_ok(), "ordinary: {pattern}");
        assert!(
            matches!(
                Source::new(pattern),
                Err(SourceError::Profile(PlanError::Instruction))
                    | Err(SourceError::Compile(Error::CompileError(_)))
            ),
            "{pattern}"
        );
    }
    let error = Source::new("[").unwrap_err();
    assert!(matches!(error, SourceError::Compile(Error::ParseError(..))));
}

#[test]
fn all_five_outer_real_target_failures_keep_actual_completed_prefixes() {
    let source = Source::new(r"(a+)(?!b)|\s+").unwrap();
    let requirements = source.plan().unwrap().requirements();
    for (failed, buffer) in Buffer::ALL.iter().copied().enumerate() {
        let mut plan = source.plan().unwrap();
        plan.fail = Some(buffer);
        let error = plan.prepare().unwrap_err();
        assert_eq!(error.buffer(), Some(buffer));
        assert!(error.reserve_error().is_some());
        assert_eq!(error.completed_delegates(), 0);
        assert!(core::ptr::eq(error.source, &source));
        assert_eq!(error.requirements(), requirements);
        for (index, &capacity) in error.capacities().iter().enumerate() {
            if index < failed {
                assert!(capacity >= requirements.capacity(Buffer::ALL[index]));
            } else {
                assert_eq!(capacity, 0);
            }
        }
    }
    let mut workspace = source.plan().unwrap().prepare().unwrap();
    assert_eq!(workspace.find("aaaa ").unwrap().unwrap().as_str(), "aaaa");
}

#[test]
fn each_nested_real_partial_frontier_keeps_completed_siblings_and_outer_storage() {
    let source = Source::new(r"(a+)(?![bc])|\s+").unwrap();
    let requirements = source.plan().unwrap().requirements();
    assert!(requirements.delegates >= 2);
    let buffers = [
        pike::Buffer::Epsilon,
        pike::Buffer::CurrentDense,
        pike::Buffer::CurrentSparse,
        pike::Buffer::NextDense,
        pike::Buffer::NextSparse,
        pike::Buffer::CurrentSlots,
        pike::Buffer::NextSlots,
    ];
    for (failed, buffer) in buffers.iter().copied().enumerate() {
        let mut plan = source.plan().unwrap();
        plan.delegate_fail = Some((1, buffer));
        let error = plan.prepare().unwrap_err();
        assert_eq!(error.buffer(), None);
        assert_eq!(error.completed_delegates(), 1);
        assert!(core::ptr::eq(error.source, &source));
        let (ordinal, nested) = error.delegate_error().unwrap();
        assert_eq!(ordinal, 1);
        assert_eq!(nested.buffer(), buffer);
        assert!(core::ptr::eq(
            error.reserve_error().unwrap(),
            nested.cause()
        ));
        assert_eq!(
            nested.source_state_count(),
            source.delegates().nth(1).unwrap().get_nfa().states().len()
        );
        for (i, &capacity) in error.capacities().iter().enumerate() {
            assert!(capacity >= requirements.capacity(Buffer::ALL[i]));
        }
        for (i, &capacity) in nested.capacities().iter().enumerate() {
            if i < failed {
                assert!(capacity >= nested.requirements().capacity(buffers[i]));
            } else {
                assert_eq!(capacity, 0);
            }
        }
    }
}

#[test]
fn actual_runtime_limit_is_terminal_for_iterator_and_same_workspace_is_reusable() {
    let mut builder = RegexBuilder::new(r"(?:a|aa)+(?!b)");
    // `aa ` needs three backtracks: try the second alternative, leave
    // the greedy repeat, then succeed through the negative lookahead.
    // The longer failing input needs another backtrack and still exceeds 3.
    builder.backtrack_limit(3);
    let source = builder.build_workspace_source().unwrap();
    let ordinary = builder.build().unwrap();
    let mut workspace = source.plan().unwrap().prepare().unwrap();
    let identity = storage_identity(&workspace);
    let text = "aaaaaaaaaaaab";
    assert!(matches!(
        ordinary.find(text),
        Err(Error::RuntimeError(RuntimeError::BacktrackLimitExceeded))
    ));
    let mut iter = workspace.find_iter(text);
    assert!(matches!(
        iter.next(),
        Some(Err(Error::RuntimeError(
            RuntimeError::BacktrackLimitExceeded
        )))
    ));
    assert!(iter.next().is_none());
    drop(iter);
    assert_eq!(storage_identity(&workspace), identity);
    let expected = ordinary.find("aa ").unwrap().unwrap();
    assert_eq!(
        (expected.start(), expected.end(), expected.as_str()),
        (0, 2, "aa")
    );
    let actual = workspace.find("aa ").unwrap().unwrap();
    assert_eq!(
        (actual.start(), actual.end()),
        (expected.start(), expected.end())
    );
    assert_eq!(actual.as_str(), expected.as_str());
    assert_eq!(storage_identity(&workspace), identity);
}

#[test]
fn concrete_geometry_and_checked_layout_overflow_reject_before_reserve() {
    let source = Source::new(r"a+(?!b)|\s+").unwrap();
    let plan = source.plan().unwrap();
    let r = plan.requirements();
    assert_eq!(r.undo, r.saves * (MAX_STACK + 1));
    assert_eq!(
        r.required_bytes(),
        r.buffer_bytes() + r.control_bytes() + r.delegate_bytes()
    );
    assert!(matches!(
        Requirements::checked(usize::MAX, 1, 1, 0),
        Err(PlanError::CapacityOverflow)
    ));
    assert!(matches!(
        Requirements::checked(0, usize::MAX, 1, 0),
        Err(PlanError::CapacityOverflow)
    ));
    assert!(matches!(
        Requirements::checked(0, 0, 1, usize::MAX),
        Err(PlanError::CapacityOverflow)
    ));
    let assertion = Prog::new(
        vec![Insn::Assertion(crate::Assertion::WordBoundary), Insn::End],
        2,
    );
    assert_eq!(
        validate(&assertion),
        regex_automata::util::look::UnicodeWordBoundaryError::check()
            .map_err(|_| PlanError::UnicodeUnavailable)
    );
    let bad = Prog::new(vec![Insn::Save(2), Insn::End], 2);
    assert_eq!(validate(&bad), Err(PlanError::Geometry));
    let bad = Prog::new(vec![Insn::Jmp(2), Insn::End], 2);
    assert_eq!(validate(&bad), Err(PlanError::Geometry));
}

#[test]
fn actual_build_limit_retains_typed_pikevm_error() {
    let mut builder = RegexBuilder::new(r"[a-z]{10000}(?!x)");
    builder.delegate_size_limit(1);
    let error = builder.build_workspace_source().unwrap_err();
    assert!(
        matches!(&error, SourceError::Compile(Error::CompileError(cause))
        if matches!(cause.as_ref(), crate::CompileError::PikeVmBuildError(_)))
    );
    #[cfg(feature = "std")]
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<regex_automata::nfa::thompson::BuildError>()
        .is_some());
}

#[test]
fn independent_workspaces_keep_their_exact_source_and_storage() {
    let first = Source::new(r"a+(?!b)").unwrap();
    let second = Source::new(r"β+(?!γ)").unwrap();
    let mut a = first.plan().unwrap().prepare().unwrap();
    let mut b = second.plan().unwrap().prepare().unwrap();
    assert!(core::ptr::eq(a.source, &first));
    assert!(core::ptr::eq(b.source, &second));
    let before_a = storage_identity(&a);
    let before_b = storage_identity(&b);
    for _ in 0..4 {
        assert_eq!(a.find("ββ aaa ").unwrap().unwrap().as_str(), "aaa");
        assert_eq!(b.find("ββ aaa ").unwrap().unwrap().as_str(), "ββ");
    }
    assert_eq!(storage_identity(&a), before_a);
    assert_eq!(storage_identity(&b), before_b);
}

#[test]
#[cfg(feature = "unicode")]
fn five_released_pattern_programs_match_current_engine_ranges() {
    // Exact patterns copied from the pinned five-family source manifest. This
    // exercises split spans only, not full HF pipeline or tokenizer support.
    for pattern in RELEASED_PATTERNS {
        let source = Source::new(pattern).unwrap();
        assert!(matches!(source.body, Body::Fancy(_)));
        assert!(source.plan().unwrap().requirements().delegates > 0);
        let ordinary = crate::Regex::new(pattern).unwrap();
        let mut workspace = source.plan().unwrap().prepare().unwrap();
        let identity = storage_identity(&workspace);
        for text in [
            "We'LL test 123456789\r\n\t  ",
            "é\u{301} ΖΩή 漢字 १२३🙂\u{200c}\u{200d} ",
            "a\0b\u{85}\u{a0}\u{2028}c\r\n",
            "    x    ",
            "\\r\\n 's I'M",
        ] {
            let expected = spans(&ordinary, text);
            assert!(!expected.is_empty());
            assert_eq!(
                prepared_spans(&mut workspace, text),
                expected,
                "{pattern}: {text:?}"
            );
        }
        let alphabet = ["a", "1", " ", "\n", "é", "🙂", "'", "\u{200d}"];
        for a in alphabet {
            for b in alphabet {
                for c in alphabet {
                    let text = [a, b, c].concat();
                    assert_eq!(
                        prepared_spans(&mut workspace, &text),
                        spans(&ordinary, &text),
                        "{pattern}: {text:?}"
                    );
                }
            }
        }
        assert_eq!(storage_identity(&workspace), identity);
    }
}

#[cfg(feature = "unicode")]
const RELEASED_PATTERNS: [&str; 5] = [
    r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"###,
    r###"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
    r###"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
    r###"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
    r###"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
];

#[test]
#[cfg(feature = "std")]
fn shared_immutable_source_has_independent_concurrent_search_storage() {
    let source = Source::new(r"a+(?!b)|β+").unwrap();
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            let mut workspace = source.plan().unwrap().prepare().unwrap();
            for _ in 0..4 {
                assert_eq!(workspace.find("aaa β").unwrap().unwrap().as_str(), "aaa");
            }
            workspace.capacities()
        });
        let b = scope.spawn(|| {
            let mut workspace = source.plan().unwrap().prepare().unwrap();
            for _ in 0..4 {
                assert_eq!(workspace.find("ββ aaa").unwrap().unwrap().as_str(), "ββ");
            }
            workspace.capacities()
        });
        let a = a.join().unwrap();
        let b = b.join().unwrap();
        let required = source.plan().unwrap().requirements();
        for (index, buffer) in Buffer::ALL.iter().copied().enumerate() {
            assert!(a[index] >= required.capacity(buffer));
            assert!(b[index] >= required.capacity(buffer));
        }
    });
}

#[cfg(feature = "workspace-test-support")]
#[test]
fn public_development_selector_checks_exact_delegate_before_attempt() {
    let source = Source::new(r"(a+)(?![bc])|\s+").unwrap();
    let count = source
        .plan()
        .unwrap()
        .requirements()
        .capacity(Buffer::Delegates);
    let invalid = source
        .plan()
        .unwrap()
        .fail_reservation(PrepareFailure::Delegate {
            ordinal: count,
            buffer: DelegateBuffer::NextSlots,
        });
    assert!(matches!(invalid, Err(PlanError::Geometry)));
    let error = source
        .plan()
        .unwrap()
        .fail_reservation(PrepareFailure::Delegate {
            ordinal: count - 1,
            buffer: DelegateBuffer::NextSlots,
        })
        .unwrap()
        .prepare()
        .unwrap_err();
    assert_eq!(error.completed_delegates(), count - 1);
    assert_eq!(error.delegate_error().unwrap().0, count - 1);
    assert!(error.reserve_error().is_some());
    let mut valid = source.plan().unwrap().prepare().unwrap();
    assert_eq!(valid.find("aaaa ").unwrap().unwrap().as_str(), "aaaa");
}
