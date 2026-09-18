use std::{
    cell::Cell,
    hint::black_box,
    time::{Duration, Instant},
};
struct Funding {
    calls: Cell<usize>,
    bytes: Cell<usize>,
}
impl Funding {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            bytes: Cell::new(0),
        }
    }
}
impl selected::util::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), selected::util::allocation::AllocationError> {
        self.calls.set(self.calls.get() + 1);
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}
fn elapsed(mut f: impl FnMut()) -> f64 {
    for _ in 0..4 {
        f();
    }
    let start = Instant::now();
    let mut count = 0u64;
    while start.elapsed() < Duration::from_millis(300) {
        f();
        count += 1;
    }
    start.elapsed().as_secs_f64() * 1e6 / count as f64
}
fn main() {
    let atoms = [
        "a",
        "abc",
        "a|ab|abc",
        "ab|a|b",
        "a?",
        "(?:a|)",
        "[a-c]",
        "[é-ê]",
        "(?:a|b)c",
        "(?P<inner>a?)",
        "^a",
        r"\b[a-z]",
        "(?:ab){2,4}",
        r"[\u0080-\u08ff]",
        "",
    ];
    let repeats = [
        "", "?", "??", "*", "*?", "+", "+?", "{0}", "{1}", "{2}", "{0,3}", "{2,4}?", "{2,}",
        "{2,}?",
    ];
    let haystacks = [
        "", "a", "abc", "ababc", "zz", "éêz", "aaz", "xyz", "baaa", "é", "\n", "123",
    ];
    let mut constructions = 0;
    let mut searches = 0;
    for atom in atoms {
        for repeat in repeats {
            for pattern in [
                format!("(?P<outer>(?:{atom}){repeat})(?:z|zz)?"),
                format!("(?:{atom}){repeat}|xyz|xy|x"),
                format!("(?:a?)(?:{atom}){repeat}(?:b*)"),
            ] {
                let funding = Funding::new();
                let actual = selected::hybrid::regex::Builder::new()
                    .build_many_with_allocations(&[&pattern], &funding);
                let expected = reference::hybrid::regex::Regex::new(&pattern);
                assert_eq!(
                    actual.as_ref().map(|_| ()).map_err(ToString::to_string),
                    expected.as_ref().map(|_| ()).map_err(ToString::to_string),
                    "{pattern}"
                );
                constructions += 1;
                if let (Ok(actual), Ok(expected)) = (actual, expected) {
                    let mut ac =
                        selected::hybrid::regex::Cache::new_with_allocations(&actual, &funding)
                            .unwrap();
                    let mut ec = expected.create_cache();
                    for text in haystacks {
                        for start in 0..=text.len() {
                            for earliest in [false, true] {
                                let a = actual.try_search_with_allocations(
                                    &mut ac,
                                    &selected::Input::new(text).range(start..).earliest(earliest),
                                    &funding,
                                );
                                let e = expected.try_search(
                                    &mut ec,
                                    &reference::Input::new(text)
                                        .range(start..)
                                        .earliest(earliest),
                                );
                                assert_eq!(
                                    format!("{a:?}"),
                                    format!("{e:?}"),
                                    "{pattern} on {text:?} at {start} earliest={earliest}"
                                );
                                searches += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut overlaps = 0;
    for patterns in [
        vec!["a", "ab", "a+"],
        vec!["", "é", "é+", "z"],
        vec!["abc", "bc", "c"],
    ] {
        let actual = selected::hybrid::dfa::Builder::new()
            .configure(selected::hybrid::dfa::Config::new().match_kind(selected::MatchKind::All))
            .build_many(&patterns)
            .unwrap();
        let expected = reference::hybrid::dfa::Builder::new()
            .configure(reference::hybrid::dfa::Config::new().match_kind(reference::MatchKind::All))
            .build_many(&patterns)
            .unwrap();
        for text in haystacks {
            let mut ac = actual.create_cache();
            let mut ec = expected.create_cache();
            let mut ast = selected::hybrid::dfa::OverlappingState::start();
            let mut est = reference::hybrid::dfa::OverlappingState::start();
            loop {
                actual
                    .try_search_overlapping_fwd_with_allocations(
                        &mut ac,
                        &selected::Input::new(text),
                        &mut ast,
                        &Funding::new(),
                    )
                    .unwrap();
                expected
                    .try_search_overlapping_fwd(&mut ec, &reference::Input::new(text), &mut est)
                    .unwrap();
                assert_eq!(
                    format!("{:?}", ast.get_match()),
                    format!("{:?}", est.get_match())
                );
                overlaps += 1;
                if ast.get_match().is_none() {
                    break;
                }
            }
        }
    }
    println!("{constructions} pristine hybrid construction/error, {searches} exact search/range/earliest, {overlaps} overlapping comparisons passed");
    for (name, pattern, text, capacity) in [
        ("assignment", r"([a-zA-Z_][a-zA-Z0-9_]*) *= *([0-9]+)", "! sample_name = 12345 !".repeat(1024), 2 * 1024 * 1024),
        ("unicode", r"([α-ω]+):([0-9]+)", "! αβγδεζηθ:123456789 !".repeat(1024), 2 * 1024 * 1024),
        ("long_run", r"(a+)(b+)", format!("{}{}", "a".repeat(32768), "b".repeat(32768)), 2 * 1024 * 1024),
        ("cache_churn", r"[01]*1[01]{10}", "01000110110100111000001010111000101111010111010101100111000010100101011111000010110100101001100010101".repeat(64), 4096),
    ] {
        let builder = selected::hybrid::regex::Builder::new().dfa(selected::hybrid::dfa::Config::new().cache_capacity(capacity)).clone();
        let pristine = reference::hybrid::regex::Builder::new().dfa(reference::hybrid::dfa::Config::new().cache_capacity(capacity)).clone();
        let source = Funding::new(); let scratch = Funding::new();
        let actual = builder.build_many_with_allocations(&[pattern], &source).unwrap();
        let expected = pristine.build(pattern).unwrap();
        let mut ac = selected::hybrid::regex::Cache::new_with_allocations(&actual, &scratch).unwrap(); let mut ec = expected.create_cache();
        let input = selected::Input::new(&text); let einput = reference::Input::new(&text);
        assert_eq!(format!("{:?}", actual.try_search_with_allocations(&mut ac, &input, &scratch).unwrap()), format!("{:?}", expected.try_search(&mut ec, &einput).unwrap()));
        let es = elapsed(|| { black_box(expected.try_search(&mut ec, black_box(&einput)).unwrap()); });
        let as_ = elapsed(|| { black_box(actual.try_search(&mut ac, black_box(&input)).unwrap()); });
        let ecold = elapsed(|| { let mut cache = expected.create_cache(); black_box(expected.try_search(&mut cache, black_box(&einput)).unwrap()); });
        let acold = elapsed(|| { let mut cache = actual.create_cache(); black_box(actual.try_search(&mut cache, black_box(&input)).unwrap()); });
        let eb = elapsed(|| { black_box(pristine.build(pattern).unwrap()); });
        let ab = elapsed(|| { black_box(builder.build(pattern).unwrap()); });
        println!("{name}: source_bytes={} input_bytes={} pristine_warm_us={es:.3} current_warm_us={as_:.3} warm_ratio={:.3} pristine_cold_us={ecold:.3} current_cold_us={acold:.3} cold_ratio={:.3} pristine_build_us={eb:.3} current_build_us={ab:.3} build_ratio={:.3} cache_heuristic_bytes={} source_cumulative_bytes={} first_search_cumulative_bytes={} source_destinations={} first_search_destinations={}", pattern.len(), text.len(), as_/es, acold/ecold, ab/eb, ac.memory_usage(), source.bytes.get(), scratch.bytes.get(), source.calls.get(), scratch.calls.get());
    }
}
