use super::*;

#[test]
fn fixed_literal_stops_preserve_unicode_ties_and_frontiers_at_every_split() {
    for (input, stops, expected, matched) in [
        (
            "é🌍ENDtail",
            vec!["END", "ENDtail", "", "END"],
            "é🌍",
            Some("END"),
        ),
        ("ababa!", vec!["aba", "bab"], "", Some("aba")),
        ("あいうend", vec!["いう", "う"], "あ", Some("いう")),
        ("prefixé", vec!["é!"], "prefixé", None),
        ("nothing", vec![""], "nothing", None),
    ] {
        let boundaries: Vec<_> = input
            .char_indices()
            .map(|(i, _)| i)
            .chain([input.len()])
            .collect();
        for &a in &boundaries {
            for &b in boundaries.iter().filter(|&&b| b >= a) {
                let source = PreparedStopSource::prepare(stops.iter().copied()).unwrap();
                let layout = StopStreamLayout::for_source(&source, input.len()).unwrap();
                let mut memory = vec![0; layout.destination_bytes()];
                let address = memory.as_ptr();
                let mut progress = StopProgress::default();
                let mut visible = String::new();
                let mut selected = None;
                for chunk in [&input[..a], &input[a..b], &input[b..]] {
                    let out = layout.step(&mut progress, &mut memory, chunk).unwrap();
                    visible.push_str(out.visible);
                    if let Some(stop) = out.matched {
                        selected = Some(stop.to_owned());
                    }
                    assert_eq!(memory.as_ptr(), address);
                    let pending = progress.pending(&memory).unwrap();
                    assert!(pending.is_empty() || source.stops().any(|stop| stop.starts_with(pending) && stop.len()>pending.len()));
                    if selected.is_some() {
                        assert!(progress.is_matched());
                        assert!(pending.is_empty());
                    }
                }
                visible.push_str(layout.finish(&mut progress, &memory).unwrap());
                assert_eq!(
                    (visible.as_str(), selected.as_deref()),
                    (expected, matched),
                    "{input:?} cuts {a}/{b}"
                );
                assert!(progress.pending(&memory).unwrap().is_empty());
                assert_eq!(layout.finish(&mut progress, &memory).unwrap(), "");
            }
        }
    }
}

#[test]
fn stop_destinations_reject_exact_minus_one_and_oversize_before_mutation() {
    let source = PreparedStopSource::prepare(["stop"]).unwrap();
    let layout = StopStreamLayout::for_source(&source, 4).unwrap();
    assert_eq!(layout.destination_bytes(), 7); // four incoming bytes + proper prefix "sto"
    let mut progress = StopProgress::default();
    let mut short = [42; 6];
    assert_eq!(
        layout.step(&mut progress, &mut short, "sto"),
        Err(StopStorageError::InvalidStorage)
    );
    assert_eq!(short, [42; 6]);
    assert_eq!(progress, StopProgress::default());
    let mut memory = [0; 7];
    assert_eq!(
        layout
            .step(&mut progress, &mut memory, "sto")
            .unwrap()
            .visible,
        ""
    );
    assert_eq!(progress.pending(&memory).unwrap(), "sto");
    let before = (progress.clone(), memory);
    assert_eq!(
        layout.step(&mut progress, &mut memory, "12345"),
        Err(StopStorageError::InputLimit)
    );
    assert_eq!((progress.clone(), memory), before);
    let out = layout.step(&mut progress, &mut memory, "p!!!").unwrap();
    assert_eq!((out.visible, out.matched), ("", Some("stop")));
    assert!(progress.pending(&memory).unwrap().is_empty());
    assert!(StopStreamLayout::for_source(&source, usize::MAX).is_err());
}

#[test]
fn empty_program_borrows_input_and_nonterminal_calls_reuse_only_lookbehind() {
    let source = PreparedStopSource::prepare(["", ""]).unwrap();
    let mut storage = OwnedStopStorage::new(source, 5).unwrap();
    storage.prepare_destination().unwrap();
    let input = String::from("hé!");
    let out = storage.step(&input).unwrap();
    assert_eq!(out.visible.as_ptr(), input.as_ptr());
    assert!(storage.destinations.destination.is_empty());
    let source = PreparedStopSource::prepare(["xxx!"]).unwrap();
    let mut storage = OwnedStopStorage::new(source, 4).unwrap();
    storage.prepare_destination().unwrap();
    let address = storage.destinations.destination.as_ptr();
    let mut total = 0;
    for _ in 0..100 {
        let out = storage.step("axxx").unwrap();
        total += out.visible.len();
        assert_eq!(storage.destinations.destination.as_ptr(), address);
        assert_eq!(
            storage
                .destinations
                .progress
                .pending(&storage.destinations.destination)
                .unwrap(),
            "xxx"
        );
    }
    total += storage.finish().unwrap().len();
    assert_eq!(total, 400); // total output exceeds the single seven-byte working buffer
    assert_eq!(storage.destinations.destination.len(), 7);
}

#[test]
fn packed_stop_source_preserves_first_entry_and_partial_preparation_cannot_retry() {
    let source = PreparedStopSource::prepare(["é!", "x", "é!", "", "xy"]).unwrap();
    assert_eq!(source.stops().collect::<Vec<_>>(), ["é!", "x", "xy"]);
    let mut storage = OwnedStopStorage::new(source, 8).unwrap();
    let error = storage
        .prepare_with(|| Err(Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err()))
        .unwrap_err();
    assert!(matches!(error, StopPreparationError::Allocation(_)));
    let address = storage.destinations.destination.as_ptr();
    assert_eq!(storage.destinations.destination.len(), 10);
    assert_eq!(
        storage.source.stops().collect::<Vec<_>>(),
        ["é!", "x", "xy"]
    );
    assert!(matches!(
        storage.prepare_destination(),
        Err(StopPreparationError::AlreadyAttempted)
    ));
    assert_eq!(storage.destinations.destination.as_ptr(), address);
    assert_eq!(storage.step("x"), Err(StopStorageError::Unprepared));
}

#[test]
fn pointer_free_stop_destinations_share_frontier_with_the_concrete_owner() {
    for stops in [vec![], vec!["é!"], vec!["END", "ENDtail"]] {
        let source = StopCompilePlan::prepare_refs(&stops)
            .unwrap()
            .compile()
            .unwrap();
        let mut direct = StopDestinations::new(&source, 8).unwrap();
        let mut owned = OwnedStopStorage::new(
            StopCompilePlan::prepare_refs(&stops)
                .unwrap()
                .compile()
                .unwrap(),
            8,
        )
        .unwrap();
        assert_eq!(direct.step(&source, "x"), Err(StopStorageError::Unprepared));
        direct.prepare_destination().unwrap();
        owned.prepare_destination().unwrap();
        let address = direct.destination.as_ptr();
        for chunk in ["prefix", "é", "!END", "tail"] {
            assert_eq!(
                direct.step(&source, chunk).unwrap(),
                owned.step(chunk).unwrap()
            );
            assert_eq!(direct.progress, owned.destinations.progress);
            assert_eq!(direct.destination, owned.destinations.destination);
            assert_eq!(direct.destination.as_ptr(), address);
        }
        assert_eq!(direct.finish(&source).unwrap(), owned.finish().unwrap());
    }
}
#[test]
fn pointer_free_stop_failure_and_wrong_extent_preserve_existing_storage() {
    let source = PreparedStopSource::prepare(["é!"]).unwrap();
    let mut storage = StopDestinations::new(&source, 8).unwrap();
    let error = storage
        .prepare_with(|| Err(Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err()))
        .unwrap_err();
    assert!(matches!(error, StopPreparationError::Allocation(_)));
    assert_eq!(storage.destination.len(), 10);
    let address = storage.destination.as_ptr();
    assert!(matches!(
        storage.prepare_destination(),
        Err(StopPreparationError::AlreadyAttempted)
    ));
    assert_eq!(storage.destination.as_ptr(), address);
    let mut ready = StopDestinations::new(&source, 8).unwrap();
    ready.prepare_destination().unwrap();
    ready.step(&source, "é").unwrap();
    let before = (ready.progress.clone(), ready.destination.clone());
    let wrong = PreparedStopSource::prepare(["longer!"]).unwrap();
    assert_eq!(
        ready.step(&wrong, "!"),
        Err(StopStorageError::InvalidStorage)
    );
    assert_eq!((ready.progress.clone(), ready.destination.clone()), before);
    assert_eq!(ready.step(&source, "!").unwrap().matched, Some("é!"));
}
