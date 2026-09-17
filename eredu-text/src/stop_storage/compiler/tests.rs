use super::*;
use crate::stop_storage::{StopProgress, StopStreamLayout};
use std::error::Error;

#[test]
fn borrowed_concrete_inputs_pack_only_first_nonempty_exact_utf8_occurrences() {
    let mut strings: Vec<String> = ["", "ENDtail", "é!", "END", "ENDtail", "\0", "É!", "é!", ""]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let plan = StopCompilePlan::prepare(&strings).unwrap();
    let facts = *plan.requirements();
    assert_eq!(facts.entries(), 5);
    assert_eq!(facts.packed_bytes(), "ENDtailé!END\0É!".len());
    assert_eq!(facts.longest_stop_bytes(), 7);
    let source = plan.compile().unwrap();
    strings.fill(String::from("replaced after compilation"));
    drop(strings);
    assert_eq!(
        source.stops().collect::<Vec<_>>(),
        ["ENDtail", "é!", "END", "\0", "É!"]
    );
    assert_eq!(source.longest_stop_bytes(), 7);
    let refs = ["", "ENDtail", "é!", "END", "ENDtail", "\0", "É!", "é!", ""];
    let refs_source = StopCompilePlan::prepare_refs(&refs)
        .unwrap()
        .compile()
        .unwrap();
    assert!(source.stops().eq(refs_source.stops()));
    assert_eq!(source.storage_bytes(), refs_source.storage_bytes());
    assert!(source.storage_bytes().unwrap() >= facts.buffer_bytes());
}

#[test]
fn compiled_program_preserves_full_frontiers_source_order_and_case() {
    for (stops, matched) in [(["ENDtail", "END"], "ENDtail"), (["END", "ENDtail"], "END")] {
        let source = StopCompilePlan::prepare_refs(&stops)
            .unwrap()
            .compile()
            .unwrap();
        let layout = StopStreamLayout::for_source(&source, 16).unwrap();
        let mut destination = vec![0; layout.destination_bytes()];
        let mut progress = StopProgress::default();
        let out = layout
            .step(&mut progress, &mut destination, "é🌍ENDtail!")
            .unwrap();
        assert_eq!((out.visible, out.matched), ("é🌍", Some(matched)));
    }
    for (input, stops, expected, matched) in [
        ("é🌍ENDtail!", vec!["END", "ENDtail"], "é🌍", Some("END")),
        ("é!doneÉ!tail", vec!["É!"], "é!done", Some("É!")),
        ("a\0tail", vec!["\0", "\0"], "a", Some("\0")),
        ("prefixあ", vec!["あい"], "prefixあ", None),
    ] {
        let strings: Vec<_> = stops.into_iter().map(str::to_owned).collect();
        let source = StopCompilePlan::prepare(&strings)
            .unwrap()
            .compile()
            .unwrap();
        let boundaries: Vec<_> = input
            .char_indices()
            .map(|(i, _)| i)
            .chain([input.len()])
            .collect();
        for &a in &boundaries {
            for &b in boundaries.iter().filter(|&&b| b >= a) {
                let layout = StopStreamLayout::for_source(&source, input.len()).unwrap();
                let mut destination = vec![0; layout.destination_bytes()];
                let address = destination.as_ptr();
                let mut progress = StopProgress::default();
                let mut visible = String::new();
                let mut selected = None;
                for chunk in [&input[..a], &input[a..b], &input[b..]] {
                    let out = layout.step(&mut progress, &mut destination, chunk).unwrap();
                    visible.push_str(out.visible);
                    if out.matched.is_some() {
                        selected = out.matched.map(str::to_owned);
                    }
                    assert_eq!(address, destination.as_ptr());
                }
                visible.push_str(layout.finish(&mut progress, &destination).unwrap());
                assert_eq!(visible, expected);
                assert_eq!(selected.as_deref(), matched);
                assert!(progress.pending(&destination).unwrap().is_empty());
            }
        }
    }
}

#[test]
fn empty_concrete_program_has_no_packing_or_destination_allocation() {
    for strings in [vec![], vec![String::new(), String::new()]] {
        let plan = StopCompilePlan::prepare(&strings).unwrap();
        assert_eq!(
            (
                plan.requirements().entries(),
                plan.requirements().packed_bytes(),
                plan.requirements().longest_stop_bytes(),
                plan.requirements().buffer_bytes()
            ),
            (0, 0, 0, 0)
        );
        let source = plan.compile().unwrap();
        assert_eq!((source.entries.capacity(), source.bytes.capacity()), (0, 0));
        let layout = StopStreamLayout::for_source(&source, 4).unwrap();
        assert_eq!(layout.destination_bytes(), 0);
        let input = String::from("é!");
        let mut progress = StopProgress::default();
        let mut destination = [];
        let out = layout
            .step(&mut progress, &mut destination, &input)
            .unwrap();
        assert_eq!(out.visible.as_ptr(), input.as_ptr());
        assert_eq!(out.matched, None);
    }
}

#[test]
fn both_real_reserve_failures_keep_the_actual_prefix_and_original_cause() {
    let inputs = ["first", "é!", "first", ""];
    let mut prefix = 0;
    for stage in 0..2 {
        let failure = StopCompilePlan::prepare_refs(&inputs)
            .unwrap()
            .fail_reservation(stage)
            .compile()
            .unwrap_err();
        assert!(matches!(failure.cause(), StopSourceError::Allocation(_)));
        assert!(failure
            .source()
            .unwrap()
            .downcast_ref::<StopSourceError>()
            .is_some());
        if let StopSourceError::Allocation(cause) = failure.cause() {
            assert!(!cause.to_string().is_empty());
        }
        assert!(failure.partial.entries.is_empty());
        assert!(failure.partial.bytes.is_empty());
        assert_eq!(failure.partial.bytes.capacity(), 0);
        if stage == 0 {
            assert_eq!(failure.retained_buffer_bytes(), 0);
        } else {
            assert!(failure.retained_buffer_bytes() > prefix);
            assert!(failure.partial.entries.capacity() >= 2);
        }
        prefix = failure.retained_buffer_bytes();
    }
}

#[test]
fn checked_packing_layout_rejects_overflow_before_any_reserve() {
    for (entries, bytes, longest) in [
        (usize::MAX, 0, 0),
        (0, usize::MAX, 0),
        (
            isize::MAX as usize / size_of::<Entry>(),
            isize::MAX as usize,
            isize::MAX as usize,
        ),
    ] {
        assert!(matches!(
            requirements(entries, bytes, longest),
            Err(StopSourceError::Overflow)
        ));
    }
    let facts = requirements(2, 7, 4).unwrap();
    assert_eq!(facts.buffer_bytes(), 2 * size_of::<Entry>() + 7);
    assert_eq!(
        facts.required_bytes(),
        facts.buffer_bytes() + facts.control_bytes()
    );
}
