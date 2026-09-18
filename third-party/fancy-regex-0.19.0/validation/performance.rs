//! Release comparison with the independently built, pinned 0.19 source.
use fancy_regex as current;
use fancy_regex_reference as reference;
use std::hint::black_box;
use std::{cell::Cell, collections::HashSet};
struct Funding(Cell<usize>);
impl current::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), current::allocation::AllocationError> {
        self.0.set(self.0.get().checked_add(bytes).unwrap());
        Ok(())
    }
}
use std::time::{Duration, Instant};

fn measure(mut work: impl FnMut() -> usize) -> f64 {
    for _ in 0..4 {
        black_box(work());
    }
    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_millis(400) {
        black_box(work());
        iterations += 1;
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}
fn compare(label: &str, pattern: &str, text: &str, compile: bool) {
    let funding = Funding(Cell::new(0));
    let actual = current::RegexOptionsBuilder::new()
        .build_with_allocations(pattern, &funding)
        .unwrap();
    let mut seen = HashSet::new();
    let mut retained = 0usize;
    actual
        .visit_source_storage(&mut |identity: *const (), bytes| {
            if seen.insert(identity as usize) {
                retained += bytes;
                true
            } else {
                false
            }
        })
        .unwrap();
    println!("source {label}: pattern_bytes={} retained_bytes={retained} retained_groups={} construction_cumulative_bytes={}", pattern.len(), seen.len(), funding.0.get());
    let expected = reference::Regex::new(pattern).unwrap();
    let spans_a: Vec<_> = actual
        .find_iter(text)
        .map(|m| {
            let m = m.unwrap();
            (m.start(), m.end())
        })
        .collect();
    let spans_b: Vec<_> = expected
        .find_iter(text)
        .map(|m| {
            let m = m.unwrap();
            (m.start(), m.end())
        })
        .collect();
    assert_eq!(spans_a, spans_b, "{label}");
    let baseline = measure(|| {
        expected
            .find_iter(black_box(text))
            .map(|m| {
                let m = m.unwrap();
                m.start() ^ m.end()
            })
            .fold(0, usize::wrapping_add)
    });
    let current = measure(|| {
        actual
            .find_iter(black_box(text))
            .map(|m| {
                let m = m.unwrap();
                m.start() ^ m.end()
            })
            .fold(0, usize::wrapping_add)
    });
    println!("search {label}: bytes={} matches={} pristine_us={baseline:.3} current_us={current:.3} ratio={:.3}", text.len(), spans_a.len(), current / baseline);
    if compile {
        let scratch_funding = Funding(Cell::new(0));
        let mut scoped = actual
            .search_workspace_with_allocations(&scratch_funding)
            .unwrap();
        assert_eq!(
            scoped.is_match(text, &scratch_funding).unwrap(),
            expected.is_match(text).unwrap()
        );
        let scoped_first = scratch_funding.0.get();
        let ordinary_boolean = measure(|| expected.is_match(black_box(text)).unwrap() as usize);
        let scoped_boolean = measure(|| {
            scoped
                .is_match(black_box(text), &current::allocation::Unenforced)
                .unwrap() as usize
        });
        let scoped_cold = measure(|| {
            let mut scoped = actual
                .search_workspace_with_allocations(&current::allocation::Unenforced)
                .unwrap();
            scoped
                .is_match(black_box(text), &current::allocation::Unenforced)
                .unwrap() as usize
        });
        println!("scoped {label}: pristine_pooled_us={ordinary_boolean:.3} scoped_reused_us={scoped_boolean:.3} scoped_fresh_us={scoped_cold:.3} first_scratch_cumulative_bytes={scoped_first}");
        let baseline = measure(|| {
            black_box(reference::Regex::new(black_box(pattern)).unwrap())
                .as_str()
                .len()
        });
        let current = measure(|| {
            black_box(current::Regex::new(black_box(pattern)).unwrap())
                .as_str()
                .len()
        });
        println!("compile {label}: pattern_bytes={} pristine_us={baseline:.3} current_us={current:.3} ratio={:.3}", pattern.len(), current / baseline);
    }
}
fn main() {
    for ordinal in [0, 2] {
        let pattern = current::workspace::construction::patterns()
            .nth(ordinal)
            .unwrap();
        for (name, text) in [
            (
                "chat",
                "The assistant's answer has 12345 tokens.\n".repeat(512),
            ),
            ("unicode", "élève 東京🙂 १२३ e\u{301}\r\n".repeat(512)),
            ("spaces", format!("{}x", " ".repeat(32_768))),
        ] {
            compare(
                &format!("tokenizer-{ordinal}-{name}"),
                pattern,
                &text,
                name == "chat",
            );
        }
    }
    compare(
        "schema-lookahead",
        r"^(?=.*[A-Z])(?=.*[0-9])[A-Za-z0-9]{8,32}$",
        "Abcd12345xyz",
        true,
    );
    compare(
        "unicode-backref",
        r"(?i)([σςſKk]+)\1",
        &"σςſKkσσsKk ".repeat(256),
        true,
    );
    compare(
        "atomic-stack",
        r"(a)(?>\1*)b",
        &format!("{}b", "a".repeat(8192)),
        true,
    );
}
