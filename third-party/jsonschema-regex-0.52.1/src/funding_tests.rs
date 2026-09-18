use super::*;
use std::cell::Cell;

struct Probe {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    refuse: Option<usize>,
}
impl Allocation for Probe {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.refuse == Some(call) {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get() + bytes);
        Ok(())
    }
}
fn each_refusal<T: std::fmt::Debug>(
    name: &str,
    run: impl Fn(&dyn Allocation) -> Result<T, AllocationError>,
) {
    let probe = Probe {
        calls: Cell::new(0),
        bytes: Cell::new(0),
        refuse: None,
    };
    run(&probe).unwrap();
    let reached = probe.calls.get();
    assert!(reached > 0, "{name}");
    eprintln!(
        "{name}: {reached} requests, {} cumulative bytes",
        probe.bytes.get()
    );
    for fail in 0..reached {
        let probe = Probe {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse: Some(fail),
        };
        assert!(
            matches!(run(&probe), Err(AllocationError::Refused)),
            "{name}, request {fail}"
        );
        assert_eq!(
            probe.calls.get(),
            fail + 1,
            "{name}, request after refusal {fail}"
        );
    }
}

#[test]
fn original_translation_refuses_ast_visitor_and_rewrite_growth() {
    for pattern in [r"([\d\w\s]+|\cA){2}", r"\cA\cB\cC", r"(?=a)\w", r"\a"] {
        each_refusal(pattern, |policy| {
            match to_rust_regex_with_allocations(pattern, policy) {
                Ok(value) => Ok(Some(value)),
                Err(TranslationError::Syntax) => Ok(None),
                Err(TranslationError::Allocation(error)) => Err(error),
            }
        });
    }
}

#[test]
fn original_syntax_refuses_group_name_copies_and_frame_growth() {
    for pattern in [
        r"(?<\u0061>\w+)(?<outer>(?<inner>a)|b)\k<a>",
        r"(?<x>a)|(?<x>b)",
        r"(((invalid",
    ] {
        each_refusal(pattern, |policy| {
            is_valid_ecma_regex_with_allocations(pattern, policy)
        });
    }
}

#[test]
fn original_literal_optimization_refuses_owned_spellings_and_alternatives() {
    for pattern in [r"^(abc|a\/b|second|abc)$", r"^\/escaped\$name$"] {
        each_refusal(pattern, |policy| {
            analyze_pattern_with_allocations(pattern, policy)
        });
    }
    let probe = Probe {
        calls: Cell::new(0),
        bytes: Cell::new(0),
        refuse: Some(0),
    };
    assert_eq!(
        analyze_pattern_with_allocations("^borrowed$", &probe).unwrap(),
        Some(PatternAnalysis::Exact(Cow::Borrowed("borrowed")))
    );
    assert_eq!(probe.calls.get(), 0);
}

#[test]
fn original_witness_refuses_parser_hir_and_output_growth() {
    each_refusal("witness", |policy| {
        pattern_witness_with_allocations(r"^(?:\d{2}|water)[a-z水]{3}$", policy)
    });
}
