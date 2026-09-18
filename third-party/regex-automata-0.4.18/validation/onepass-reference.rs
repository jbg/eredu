use std::{cell::Cell, hint::black_box, time::{Duration, Instant}};

struct Funding { calls: Cell<usize>, bytes: Cell<usize> }
impl selected::util::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), selected::util::allocation::AllocationError> {
        self.calls.set(self.calls.get() + 1);
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}
fn elapsed(mut f: impl FnMut()) -> f64 {
    for _ in 0..4 { f(); }
    let start = Instant::now();
    let mut count = 0u64;
    while start.elapsed() < Duration::from_millis(300) { f(); count += 1; }
    start.elapsed().as_secs_f64() * 1e6 / count as f64
}
fn main() {
    let atoms = ["a", "abc", "a|ab|abc", "ab|a|b", "a?", "(?:a|)", "[a-c]", "[é-ê]", "(?:a|b)c", "(?P<inner>a?)", "^a", r"\b[a-z]", "(?:ab){2,4}", r"[\u0080-\u08ff]", ""];
    let repeats = ["", "?", "??", "*", "*?", "+", "+?", "{0}", "{1}", "{2}", "{0,3}", "{2,4}?", "{2,}", "{2,}?"];
    let haystacks = ["", "a", "abc", "ababc", "zz", "éêz", "aaz", "xyz", "baaa", "é", "\n", "123"];
    let mut constructions = 0;
    let mut captures = 0;
    for atom in atoms {
        for repeat in repeats {
            for pattern in [format!("(?P<outer>(?:{atom}){repeat})(?:z|zz)?"), format!("(?:{atom}){repeat}|xyz|xy|x"), format!("(?:a?)(?:{atom}){repeat}(?:b*)")] {
                let funding = Funding { calls: Cell::new(0), bytes: Cell::new(0) };
                let nfa = selected::nfa::thompson::NFA::compiler().build_many_with_allocations(&[&pattern], &funding).unwrap();
                let actual = selected::dfa::onepass::Builder::new().build_from_nfa_with_allocations(nfa, &funding);
                let expected = reference::dfa::onepass::DFA::new(&pattern);
                assert_eq!(actual.as_ref().map(|_| ()).map_err(ToString::to_string), expected.as_ref().map(|_| ()).map_err(ToString::to_string), "{pattern}");
                constructions += 1;
                if let (Ok(actual), Ok(expected)) = (actual, expected) {
                    let mut ac = selected::dfa::onepass::Cache::new_with_allocations(&actual, &funding).unwrap();
                    let mut ec = expected.create_cache();
                    let mut av = selected::util::captures::Captures::all_with_allocations(actual.get_nfa().group_info().clone(), &funding).unwrap();
                    let mut ev = expected.create_captures();
                    for text in haystacks {
                        actual.captures(&mut ac, text, &mut av);
                        expected.captures(&mut ec, text, &mut ev);
                        assert_eq!(format!("{av:?}"), format!("{ev:?}"), "{pattern} on {text:?}");
                        captures += 1;
                    }
                }
            }
        }
    }
    println!("{constructions} pristine one-pass construction/error comparisons and {captures} exact capture comparisons passed");
    for (name, pattern, text) in [
        ("assignment", r"([a-zA-Z_][a-zA-Z0-9_]*) *= *([0-9]+)", "sample_name = 12345".to_owned()),
        ("unicode", r"([α-ω]+):([0-9]+)", "αβγδεζηθ:123456789".to_owned()),
        ("long_run", r"(a+)(b+)", format!("{}{}", "a".repeat(32768), "b".repeat(32768))),
    ] {
        let funding = Funding { calls: Cell::new(0), bytes: Cell::new(0) };
        let nfa = selected::nfa::thompson::NFA::compiler().build_many_with_allocations(&[pattern], &funding).unwrap();
        let actual = selected::dfa::onepass::Builder::new().build_from_nfa_with_allocations(nfa, &funding).unwrap();
        let expected = reference::dfa::onepass::DFA::new(pattern).unwrap();
        let mut ac = selected::dfa::onepass::Cache::new_with_allocations(&actual, &funding).unwrap();
        let mut ec = expected.create_cache();
        assert_eq!(format!("{:?}", actual.find(&mut ac, &text)), format!("{:?}", expected.find(&mut ec, &text)));
        let es = elapsed(|| { black_box(expected.find(&mut ec, black_box(&text))); });
        let as_ = elapsed(|| { black_box(actual.find(&mut ac, black_box(&text))); });
        let eb = elapsed(|| { black_box(reference::dfa::onepass::DFA::new(pattern).unwrap()); });
        let ab = elapsed(|| { black_box(selected::dfa::onepass::DFA::new(pattern).unwrap()); });
        println!("{name}: source_bytes={} input_bytes={} pristine_search_us={es:.3} current_search_us={as_:.3} search_ratio={:.3} pristine_build_us={eb:.3} current_build_us={ab:.3} build_ratio={:.3} dfa_table_bytes={} explicit_cache_bytes={} admitted_destinations={} admitted_cumulative_bytes={}", pattern.len(), text.len(), as_/es, ab/eb, actual.memory_usage(), ac.memory_usage(), funding.calls.get(), funding.bytes.get());
    }
}
