//! Isolate scoped invocation construction from the identical engine's pooled search.
//! Compile against the pinned local fancy-regex release artifact. This is not an
//! upstream comparison and does not include schema traversal or source compilation.
use fancy_regex::{allocation::Unenforced, OwnedSearchWorkspace, RegexOptionsBuilder};
use std::{hint::black_box, sync::Arc, time::Instant};
fn main() {
    let repetitions = 20_000;
    for (pattern, input) in [
        (r"^[a-z][a-z0-9_-]{1,32}$", "property_name_42"),
        (r"[α-ω]+[0-9]+", "text αβγ123 trailing"),
        (r"(?=a+)(a+)\1", "aaaaaaaa"),
    ] {
        let regex = Arc::new(
            RegexOptionsBuilder::new()
                .build_with_allocations(pattern, &Unenforced)
                .unwrap(),
        );
        assert!(regex.is_match(input).unwrap());
        let pooled = (0..7)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..repetitions {
                    assert!(black_box(regex.is_match(black_box(input)).unwrap()));
                }
                start.elapsed().as_nanos()
            })
            .min()
            .unwrap();
        let scoped = (0..7)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..repetitions {
                    let mut workspace = regex
                        .search_workspace_with_allocations(&Unenforced)
                        .unwrap();
                    assert!(black_box(
                        workspace.is_match(black_box(input), &Unenforced).unwrap()
                    ));
                }
                start.elapsed().as_nanos()
            })
            .min()
            .unwrap();
        let mut workspace =
            OwnedSearchWorkspace::new_with_allocations(regex.clone(), &Unenforced).unwrap();
        assert!(workspace.is_match(input, &Unenforced).unwrap());
        let reused = (0..7)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..repetitions {
                    assert!(black_box(
                        workspace.is_match(black_box(input), &Unenforced).unwrap()
                    ));
                }
                start.elapsed().as_nanos()
            })
            .min()
            .unwrap();
        println!("pattern={pattern:?} bytes={} calls={repetitions} pooled_ns={pooled} scoped_ns={scoped} fresh_ratio={:.3} reused_ns={reused} reused_ratio={:.3}", input.len(), scoped as f64 / pooled as f64, reused as f64 / pooled as f64);
    }
}
