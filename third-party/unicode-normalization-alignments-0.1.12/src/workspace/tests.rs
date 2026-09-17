use super::*;
use UnicodeNormalization;
#[path = "reference/mod.rs"]
mod reference;

fn pairs(workspace: &mut Workspace, range: Range<usize>) {
    let input = workspace.input.get(range).unwrap();
    workspace.storage.decomposition.entries.clear();
    workspace.storage.recomposition.entries.clear();
    let mut actual = Recomposing {
        iter: Decomposing {
            iter: input.chars().fuse(),
            ready: 0..0,
            storage: &mut workspace.storage.decomposition,
        },
        state: RecompositionState::Composing,
        storage: &mut workspace.storage.recomposition,
        composee: None,
        last_ccc: None,
    };
    let mut prior = reference::recompose::new_canonical(input.chars());
    loop {
        let a = actual.next();
        let b = prior.next();
        assert_eq!(a, b, "{input:?}");
        if a.is_none() {
            break;
        }
    }
}
fn pointers(workspace: &Workspace) -> (*const Record, *const (char, isize), *const u8) {
    (
        workspace.storage.decomposition.entries.as_ptr(),
        workspace.storage.recomposition.entries.as_ptr(),
        workspace.storage.text.as_ptr(),
    )
}
#[test]
fn every_scalar_preserves_pre_refactor_transformations_and_utf8_bound() {
    let input: String = (0..=0x10ffff).filter_map(::std::char::from_u32).collect();
    let mut workspace = Plan::new(&input).unwrap().prepare().unwrap();
    let ptr = pointers(&workspace);
    let cap = workspace.capacities();
    for (start, c) in input.char_indices() {
        let range = start..start + c.len_utf8();
        pairs(&mut workspace, range.clone());
        let actual = workspace.normalize(range.clone()).unwrap();
        assert!(actual
            .chars()
            .eq(reference::recompose::new_canonical(input[range].chars()).map(|x| x.0)));
    }
    assert_eq!(workspace.capacities(), cap);
    assert_eq!(pointers(&workspace), ptr);
}
#[test]
fn nonstarter_order_blocking_hangul_expansion_and_repeated_subranges_use_same_storage() {
    let mut input=String::from("head|e\u{301}|\u{344}|\u{1100}\u{1161}\u{11a8}|\u{301}a\u{315}\u{300}\u{315}|\u{200c}\u{200d}|\u{212b}\u{2126}|");
    input.push_str(&"a\u{315}\u{300}\u{301}\u{0323}\u{0301}".repeat(96));
    input.push('|');
    input.push_str(&"\u{315}\u{300}\u{301}\u{0323}".repeat(1024));
    input.push('|');
    let mut workspace = Plan::new(&input).unwrap().prepare().unwrap();
    let ptr = pointers(&workspace);
    let cap = workspace.capacities();
    let mut start = 0;
    for (end, c) in input.char_indices().filter(|&(_, c)| c == '|') {
        let range = start..end;
        let _ = c;
        for _ in 0..2 {
            pairs(&mut workspace, range.clone());
            assert!(workspace.normalize(range.clone()).unwrap().chars().eq(
                reference::recompose::new_canonical(input[range.clone()].chars()).map(|x| x.0)
            ));
        }
        start = end + 1;
    }
    pairs(&mut workspace, 0..input.len());
    assert_eq!(workspace.capacities(), cap);
    assert_eq!(pointers(&workspace), ptr);
    let _ = workspace.normalize(0..4).unwrap();
    let last = workspace.storage.text.clone();
    for range in [usize::MAX..usize::MAX, 4..3, 6..7] {
        assert_eq!(workspace.normalize(range), Err(InvalidRange));
        assert_eq!(workspace.storage.text, last);
    }
    assert_eq!(pointers(&workspace), ptr);
}
#[test]
fn each_actual_reserve_failure_keeps_earlier_buffers_and_real_cause() {
    let input = String::from("a\u{315}\u{300}\u{301}\u{0323}\u{344}");
    for (stage, buffer) in [Buffer::Decomposition, Buffer::Recomposition, Buffer::Text]
        .iter()
        .enumerate()
    {
        let failure = Plan::new(&input)
            .unwrap()
            .fail_reservation(*buffer)
            .prepare()
            .unwrap_err();
        assert_eq!(failure.buffer(), *buffer);
        assert!(std::error::Error::source(&failure)
            .unwrap()
            .is::<TryReserveError>());
        let caps = failure.capacities();
        assert!(caps[..stage].iter().all(|&n| n > 0));
        assert!(caps[stage..].iter().all(|&n| n == 0));
    }
    let failure = Plan::new(&input)
        .unwrap()
        .fail_reservation(Buffer::Text)
        .prepare()
        .unwrap_err();
    drop(input);
    assert!(failure.capacities()[0] > 4 && failure.capacities()[1] > 4);
}
#[test]
fn empty_overflow_and_exact_decomposition_geometry_are_distinct() {
    let mut empty = Plan::new("").unwrap().prepare().unwrap();
    assert_eq!(empty.capacities(), [0; 3]);
    assert_eq!(empty.normalize(0..0).unwrap(), "");
    let plan = Plan::new("\u{344}").unwrap();
    assert_eq!(plan.requirements().scalar_capacity(), 2);
    assert_eq!(plan.requirements().text_capacity(), 4);
    let mut workspace = plan.prepare().unwrap();
    assert_eq!(workspace.normalize(0..2).unwrap(), "\u{308}\u{301}");
    assert!(requirements(usize::MAX, 1).is_err());
    assert!(requirements(1, usize::MAX).is_err());
    assert!(Plan::new("")
        .unwrap()
        .fail_reservation(Buffer::Text)
        .prepare()
        .is_err());
}
#[test]
fn retirement_strips_source_without_reallocating_or_releasing_destinations() {
    let input = String::from("e\u{301} and \u{344}");
    let mut workspace = Plan::new(&input).unwrap().prepare().unwrap();
    let ptr = pointers(&workspace);
    let cap = workspace.capacities();
    assert!(!workspace.normalize(0..input.len()).unwrap().is_empty());
    let retired = workspace.retire();
    drop(input);
    assert_eq!(retired.capacities(), cap);
    assert_eq!(retired.decomposition.entries.as_ptr(), ptr.0);
    assert_eq!(retired.recomposition.entries.as_ptr(), ptr.1);
    assert_eq!(retired.text.as_ptr(), ptr.2);
}
#[test]
fn ordinary_clone_display_and_compatible_workers_keep_pre_refactor_pairs() {
    for text in [
        "e\u{301}\u{315}",
        "\u{fb01}\u{2163}\u{344}",
        "\u{1100}\u{1161}\u{11a8}",
        "",
    ] {
        assert!(text
            .nfd()
            .eq(reference::decompose::new_canonical(text.chars())));
        assert!(text
            .nfkd()
            .eq(reference::decompose::new_compatible(text.chars())));
        assert!(text
            .nfkc()
            .eq(reference::recompose::new_compatible(text.chars())));
        let mut current = text.nfc();
        let _ = current.next();
        let mut prior = reference::recompose::new_canonical(text.chars());
        let _ = prior.next();
        assert!(current.clone().eq(prior.clone()));
        assert_eq!(format!("{}", current), format!("{}", prior));
    }
}
