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
    let mut count = 0;
    while start.elapsed() < Duration::from_millis(300) {
        f();
        count += 1;
    }
    start.elapsed().as_secs_f64() * 1e6 / count as f64
}
fn main() {
    let atoms = [
        "",
        "a",
        "a|ab|abc",
        "(?:ab|a)+",
        "[a-c]",
        "[^a]",
        "(a+)(b+)",
        r"\d+",
        r"\w+",
        r"\b\w+\b",
        "αβ+",
        "[α-ω]+",
        r"\d+XYZ",
        r"\d+@!\w+",
        r"[a-z]+\s+END",
        "(?P<one>a)(?P<two>b)",
        "(",
        "[",
        "a{3,1}",
    ];
    let mut texts: Vec<String> = [
        "",
        "a",
        "aaa",
        "abc",
        "abcc",
        "aabb",
        "αβ γδ",
        "123XYZ",
        "12@!abc",
        "abc abc END",
        "é水🙂",
        "a\r\nb",
        "\0",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let alphabet = [
        'a', 'b', 'c', 'Z', 'α', 'β', 'é', '水', '🙂', '0', '१', ' ', '\n', '@', '!',
    ];
    let mut seed = 0x34abcde1u32;
    for _ in 0..64 {
        let mut text = String::new();
        for _ in 0..32 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            text.push(alphabet[seed as usize % alphabet.len()]);
        }
        texts.push(text);
    }
    let mut construction = 0;
    let mut comparisons = 0;
    for profile in 0..4 {
        for atom in atoms {
            for suffix in ["", "$", "?", "|z"] {
                let pattern = format!("{atom}{suffix}");
                let ac = match profile {
                    1 => selected::meta::Config::new()
                        .dfa(false)
                        .onepass(false)
                        .backtrack(false),
                    2 => selected::meta::Config::new()
                        .dfa(false)
                        .hybrid(false)
                        .onepass(false),
                    3 => selected::meta::Config::new()
                        .dfa(false)
                        .hybrid(false)
                        .onepass(false)
                        .backtrack(false),
                    _ => selected::meta::Config::new(),
                };
                let bc = match profile {
                    1 => reference::meta::Config::new()
                        .dfa(false)
                        .onepass(false)
                        .backtrack(false),
                    2 => reference::meta::Config::new()
                        .dfa(false)
                        .hybrid(false)
                        .onepass(false),
                    3 => reference::meta::Config::new()
                        .dfa(false)
                        .hybrid(false)
                        .onepass(false)
                        .backtrack(false),
                    _ => reference::meta::Config::new(),
                };
                let paid = Funding::new();
                let a = selected::meta::Builder::new()
                    .configure(ac)
                    .build_with_allocations(&pattern, &paid);
                let b = reference::meta::Builder::new()
                    .configure(bc)
                    .build(&pattern);
                construction += 1;
                match (a, b) {
                    (Ok(a), Ok(b)) => {
                        let scratch = Funding::new();
                        let mut ca = a.create_cache_with_allocations(&scratch).unwrap();
                        let mut cb = b.create_cache();
                        let mut acaps = a.create_captures();
                        let mut bcaps = b.create_captures();
                        for text in &texts {
                            for range in [
                                0..text.len(),
                                text.char_indices().nth(1).map_or(text.len(), |(i, _)| i)
                                    ..text.len(),
                            ] {
                                for anchored in [false, true] {
                                    let ai = selected::Input::new(text)
                                        .range(range.clone())
                                        .anchored(if anchored {
                                            selected::Anchored::Yes
                                        } else {
                                            selected::Anchored::No
                                        });
                                    let bi = reference::Input::new(text)
                                        .range(range.clone())
                                        .anchored(if anchored {
                                            reference::Anchored::Yes
                                        } else {
                                            reference::Anchored::No
                                        });
                                    let af = a
                                        .search_with_allocations(&mut ca, &ai, &scratch)
                                        .unwrap()
                                        .map(|m| (m.pattern().as_usize(), m.start(), m.end()));
                                    let bf = b
                                        .search_with(&mut cb, &bi)
                                        .map(|m| (m.pattern().as_usize(), m.start(), m.end()));
                                    assert_eq!(af, bf, "{profile} {pattern:?} {text:?}");
                                    a.search_captures_with_allocations(
                                        &mut ca, &ai, &mut acaps, &scratch,
                                    )
                                    .unwrap();
                                    b.search_captures_with(&mut cb, &bi, &mut bcaps);
                                    assert_eq!(acaps.is_match(), bcaps.is_match());
                                    if acaps.is_match() {
                                        let av: Vec<_> = acaps
                                            .iter()
                                            .map(|s| s.map(|s| (s.start, s.end)))
                                            .collect();
                                        let bv: Vec<_> = bcaps
                                            .iter()
                                            .map(|s| s.map(|s| (s.start, s.end)))
                                            .collect();
                                        assert_eq!(
                                            av, bv,
                                            "captures {profile} {pattern:?} {text:?}"
                                        );
                                    }
                                    assert_eq!(
                                        a.is_match_with_allocations(&mut ca, &ai, &scratch)
                                            .unwrap(),
                                        b.is_match(bi.clone())
                                    );
                                    comparisons += 3;
                                }
                            }
                        }
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("acceptance differs {profile} {pattern}"),
                }
            }
        }
    }
    println!(
        "construction={construction} independent_search_capture_boolean_comparisons={comparisons}"
    );
    for (name, pattern, text) in [
        ("literal", "needle", "hay hay hay needle".repeat(1024)),
        (
            "assignment",
            r"(?P<key>[a-z]+)\s*=\s*(?P<value>[0-9]+)",
            "prefix alpha = 123 tail\n".repeat(1024),
        ),
        (
            "unicode",
            r"(\p{Greek}+)([0-9]+)",
            "prefix αβγ123 tail\n".repeat(1024),
        ),
        ("long_run", "[a-z]+Z", "a".repeat(65536)),
        (
            "reverse_suffix",
            r"\w+\s+END",
            "word word ".repeat(4096) + "END",
        ),
    ] {
        let source = Funding::new();
        let a = selected::meta::Builder::new()
            .build_with_allocations(pattern, &source)
            .unwrap();
        let b = reference::meta::Regex::new(pattern).unwrap();
        let scratch = Funding::new();
        let mut ca = a.create_cache_with_allocations(&scratch).unwrap();
        let mut cb = b.create_cache();
        let ai = selected::Input::new(&text);
        let bi = reference::Input::new(&text);
        assert_eq!(
            a.search_with_allocations(&mut ca, &ai, &scratch)
                .unwrap()
                .map(|m| (m.start(), m.end())),
            b.search_with(&mut cb, &bi).map(|m| (m.start(), m.end()))
        );
        let b_us = elapsed(|| {
            black_box(b.search_with(&mut cb, &bi));
        });
        let a_us = elapsed(|| {
            black_box(
                a.search_with_allocations(&mut ca, &ai, &selected::util::allocation::Unenforced)
                    .unwrap(),
            );
        });
        let b_cold = elapsed(|| {
            black_box(b.search_with(&mut b.create_cache(), &bi));
        });
        let a_cold = elapsed(|| {
            let mut cache = a
                .create_cache_with_allocations(&selected::util::allocation::Unenforced)
                .unwrap();
            black_box(
                a.search_with_allocations(&mut cache, &ai, &selected::util::allocation::Unenforced)
                    .unwrap(),
            );
        });
        let b_build = elapsed(|| {
            black_box(reference::meta::Regex::new(pattern).unwrap());
        });
        let a_build = elapsed(|| {
            black_box(selected::meta::Regex::new(pattern).unwrap());
        });
        println!("{name} source_bytes={} input_bytes={} warm_us={b_us:.3}/{a_us:.3} ratio={:.3} cold_us={b_cold:.3}/{a_cold:.3} build_us={b_build:.3}/{a_build:.3} source_requests={} source_cumulative_bytes={} scratch_requests={} scratch_cumulative_bytes={} declared_cache_memory={}",pattern.len(),text.len(),a_us/b_us,source.calls.get(),source.bytes.get(),scratch.calls.get(),scratch.bytes.get(),ca.memory_usage());
    }
}
