//! Pinned wrapper/lower-engine comparison. Requires current/reference regex
//! dependencies and a current_automata alias; see bounded-grammar-construction.md.
use std::{hint::black_box, sync::Arc, time::Instant};
fn compare(pattern: &str, texts: &[&str]) -> usize {
    let actual =
        current::RegexBuilder::new_with_allocations(pattern, &current::allocation::Unenforced)
            .unwrap()
            .build_with_allocations(&current::allocation::Unenforced);
    let expected = reference::Regex::new(pattern);
    let mut count = 0;
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(
                actual.capture_names().collect::<Vec<_>>(),
                expected.capture_names().collect::<Vec<_>>()
            );
            let actual = Arc::new(actual);
            let mut scoped = current::OwnedSearchWorkspace::new_with_allocations(
                actual.clone(),
                &current::allocation::Unenforced,
            )
            .unwrap();
            for text in texts {
                assert_eq!(
                    actual
                        .find_iter(text)
                        .map(|m| m.range())
                        .collect::<Vec<_>>(),
                    expected
                        .find_iter(text)
                        .map(|m| m.range())
                        .collect::<Vec<_>>(),
                    "{pattern:?} {text:?}"
                );
                assert_eq!(
                    actual
                        .captures_iter(text)
                        .map(|c| c.iter().map(|m| m.map(|m| m.range())).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    expected
                        .captures_iter(text)
                        .map(|c| c.iter().map(|m| m.map(|m| m.range())).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual.replace_all(text, "[$0][$1]"),
                    expected.replace_all(text, "[$0][$1]")
                );
                assert_eq!(
                    scoped
                        .is_match(text, &current::allocation::Unenforced)
                        .unwrap(),
                    expected.is_match(text)
                );
                count += 4;
            }
        }
        (Err(a), Err(b)) => {
            assert_eq!(a.to_string(), b.to_string(), "{pattern:?}");
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            count += 2;
        }
        (a, b) => panic!("acceptance {pattern:?}: {a:?} vs {b:?}"),
    }
    match (
        current::bytes::Regex::new(pattern),
        reference::bytes::Regex::new(pattern),
    ) {
        (Ok(a), Ok(b)) => {
            for text in texts
                .iter()
                .map(|s| s.as_bytes())
                .chain([&b"\xffa\x80\0z"[..]])
            {
                assert_eq!(
                    a.find_iter(text).map(|m| m.range()).collect::<Vec<_>>(),
                    b.find_iter(text).map(|m| m.range()).collect::<Vec<_>>()
                );
                assert_eq!(
                    a.captures_iter(text)
                        .map(|c| c.iter().map(|m| m.map(|m| m.range())).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    b.captures_iter(text)
                        .map(|c| c.iter().map(|m| m.map(|m| m.range())).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                );
                count += 2;
            }
        }
        (Err(a), Err(b)) => {
            assert_eq!(a.to_string(), b.to_string());
            count += 1;
        }
        (a, b) => panic!("byte acceptance {pattern:?}: {a:?} vs {b:?}"),
    }
    count
}
fn best(mut operation: impl FnMut(), repeats: usize) -> u128 {
    (0..7)
        .map(|_| {
            let time = Instant::now();
            for _ in 0..repeats {
                operation();
            }
            time.elapsed().as_nanos()
        })
        .min()
        .unwrap()
}
fn main() {
    let texts = [
        "", "a", "aaa", "abab", "abc123", "a\r\nb", "élève", "éÉ", "kKK", "ſSs", "水🙂", "αβ12",
        "a\u{301}", "123", " a \n", "\0", "x-name", "b", "abcdefz",
    ];
    let mut patterns = vec![
        "(".into(),
        "[z-a]".into(),
        "a{3,1}".into(),
        "(?=a)".into(),
        r"(a)\1".into(),
        "".into(),
        r"(?P<name>[a-z]+)-(?P<n>[0-9]+)".into(),
        r"(?-u:\xFF)".into(),
    ];
    for atom in [
        "a",
        "é",
        "水",
        "[a-z]",
        "[α-ω]",
        r"\d",
        r"\w",
        r"\p{Greek}",
        ".",
        "(?:ab|a)",
        "[[:alpha:]]",
    ] {
        for tail in ["", "?", "*", "+", "{0,3}", "+?", "{2,4}"] {
            for flags in ["", "(?i)", "(?m)", "(?s)"] {
                patterns.push(format!("{flags}({atom}{tail})(a|[0-9])?"));
            }
        }
    }
    let mut comparisons = 0;
    for pattern in &patterns {
        comparisons += compare(pattern, &texts);
    }
    for patterns in [
        ["^foo", "[0-9]{2,4}$", "α|β"],
        ["", "^a", "a$"],
        ["(?i)abc", "水+", "[a-z]+z$"],
    ] {
        let a = current::RegexSet::new(patterns).unwrap();
        let b = reference::RegexSet::new(patterns).unwrap();
        let c = current::bytes::RegexSet::new(patterns).unwrap();
        let d = reference::bytes::RegexSet::new(patterns).unwrap();
        for text in &texts {
            assert_eq!(
                a.matches(text).into_iter().collect::<Vec<_>>(),
                b.matches(text).into_iter().collect::<Vec<_>>()
            );
            assert_eq!(
                c.matches(text.as_bytes()).into_iter().collect::<Vec<_>>(),
                d.matches(text.as_bytes()).into_iter().collect::<Vec<_>>()
            );
            comparisons += 2;
        }
    }
    println!(
        "patterns={} exact_comparisons={comparisons}",
        patterns.len()
    );
    for (name, pattern, text) in [
        ("identifier", r"^[a-z][a-z0-9_-]{1,32}$", "property_name_42"),
        ("unicode", r"[α-ω]+[0-9]+", "prefix αβγ123 suffix"),
        (
            "literals",
            "alpha|beta|gamma|delta|epsilon",
            "the epsilon needle",
        ),
    ] {
        let a = current::Regex::new(pattern).unwrap();
        let b = reference::Regex::new(pattern).unwrap();
        assert_eq!(a.is_match(text), b.is_match(text));
        let local_meta = current_automata::meta::Regex::builder()
            .configure(
                current_automata::meta::Regex::config()
                    .nfa_size_limit(Some(10 * (1 << 20)))
                    .hybrid_cache_capacity(2 * (1 << 20))
                    .utf8_empty(true),
            )
            .build(pattern)
            .unwrap();
        let input = current_automata::Input::new(text).earliest(true);
        black_box(local_meta.search_half(&input));
        let direct = best(
            || {
                black_box(local_meta.search_half(black_box(&input)));
            },
            20000,
        );
        println!("{name} direct_local_meta_ns={direct}");
        let current = best(
            || {
                black_box(a.is_match(black_box(text)));
            },
            20000,
        );
        let pristine = best(
            || {
                black_box(b.is_match(black_box(text)));
            },
            20000,
        );
        let construct_current = best(
            || {
                black_box(current::Regex::new(black_box(pattern)).unwrap());
            },
            200,
        );
        let construct_pristine = best(
            || {
                black_box(reference::Regex::new(black_box(pattern)).unwrap());
            },
            200,
        );
        println!("{name} bytes={} search_current_ns={current} search_pristine_ns={pristine} ratio={:.3} construction_current_ns={construct_current} construction_pristine_ns={construct_pristine} ratio={:.3}",text.len(),current as f64/pristine as f64,construct_current as f64/construct_pristine as f64);
    }
}
