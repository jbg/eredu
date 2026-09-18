//! Standalone parity probe against the pristine pinned upstream crate.
//! Compile with two `--extern` arguments as recorded in the allocation inventory.

use std::cell::Cell;

struct Funding(Cell<usize>);
impl regex_syntax::allocation::Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), regex_syntax::allocation::AllocationError> {
        self.0.set(self.0.get().checked_add(bytes).unwrap());
        Ok(())
    }
}

fn check(pattern: &str) {
    let expected_ast = regex_syntax_reference::ast::parse::Parser::new().parse(pattern);
    let funding = Funding(Cell::new(0));
    let mut parser = regex_syntax::ast::parse::Parser::new();
    let actual_ast = parser.parse_with_allocations(pattern, &funding);
    assert_eq!(
        format!("{actual_ast:?}"),
        format!("{expected_ast:?}"),
        "AST: {pattern}"
    );
    if let (Err(actual), Err(expected)) = (&actual_ast, &expected_ast) {
        assert_eq!(
            actual.to_string(),
            expected.to_string(),
            "AST diagnostic: {pattern}"
        );
    }
    drop(actual_ast);
    drop(parser);
    let expected = regex_syntax_reference::Parser::new().parse(pattern);
    let mut parser = regex_syntax::Parser::new();
    let actual = parser.parse_with_allocations(pattern, &funding);
    assert_eq!(
        format!("{actual:?}"),
        format!("{expected:?}"),
        "HIR: {pattern}"
    );
    match (&actual, &expected) {
        (Ok(actual), Ok(expected)) => {
            for suffix in [false, true] {
                for limit in [4, 250] {
                    let mut current = regex_syntax::hir::literal::Extractor::new();
                    let mut reference = regex_syntax_reference::hir::literal::Extractor::new();
                    current.limit_total(limit);
                    reference.limit_total(limit);
                    if suffix {
                        current.kind(regex_syntax::hir::literal::ExtractKind::Suffix);
                        reference.kind(regex_syntax_reference::hir::literal::ExtractKind::Suffix);
                    }
                    let mut a = current
                        .extract_with_allocations(
                            actual,
                            regex_syntax::allocation::Allocator::new(&funding),
                        )
                        .unwrap();
                    let mut b = reference.extract(expected);
                    assert_eq!(
                        format!("{a:?}"),
                        format!("{b:?}"),
                        "literal extraction: {pattern}"
                    );
                    if suffix {
                        a.optimize_for_suffix_by_preference_with_allocations(
                            regex_syntax::allocation::Allocator::new(&funding),
                        )
                        .unwrap();
                        b.optimize_for_suffix_by_preference();
                    } else {
                        a.optimize_for_prefix_by_preference_with_allocations(
                            regex_syntax::allocation::Allocator::new(&funding),
                        )
                        .unwrap();
                        b.optimize_for_prefix_by_preference();
                    }
                    assert_eq!(
                        format!("{a:?}"),
                        format!("{b:?}"),
                        "literal optimization: {pattern}"
                    );
                }
            }
            let a = actual.properties();
            let b = expected.properties();
            assert_eq!(
                (
                    a.minimum_len(),
                    a.maximum_len(),
                    a.is_utf8(),
                    a.explicit_captures_len(),
                    a.static_explicit_captures_len(),
                    a.is_literal()
                ),
                (
                    b.minimum_len(),
                    b.maximum_len(),
                    b.is_utf8(),
                    b.explicit_captures_len(),
                    b.static_explicit_captures_len(),
                    b.is_literal()
                ),
                "properties: {pattern}"
            );
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                actual.to_string(),
                expected.to_string(),
                "HIR diagnostic: {pattern}"
            );
        }
        _ => unreachable!(),
    }
}

fn main() {
    let atoms = [
        "a",
        "β",
        "[a-z]",
        "[0-9]",
        r"\w",
        r"\p{Greek}",
        r"\P{Letter}",
        r"[a-z&&[^b]]",
        r"(?-u:[A-Z])",
        ".",
        "^",
        "$",
        r"\b",
    ];
    let quantifiers = ["", "?", "*", "+", "{0}", "{2,4}", "+?"];
    let mut count = 0;
    for (i, a) in atoms.iter().enumerate() {
        for (j, b) in atoms.iter().enumerate() {
            for q in quantifiers {
                let pattern = format!(
                    "(?{})(?:{a}{q}|{b})(?:{a}|{b})",
                    if (i + j) % 2 == 0 { "i" } else { "-i" }
                );
                check(&pattern);
                count += 1;
            }
        }
    }
    // Mix malformed and valid concrete syntax; every run has the same corpus.
    let alphabet = [
        'a', 'b', 'β', '[', ']', '(', ')', '{', '}', '*', '+', '?', '|', '\\', 'x', '0', '2', '-',
        ':', '\n', '#',
    ];
    let mut state = 0x92d68ca2_u32;
    for ordinal in 0..4096 {
        let mut pattern = String::new();
        for _ in 0..ordinal % 47 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pattern.push(alphabet[state as usize % alphabet.len()]);
        }
        check(&pattern);
        count += 1;
    }
    for pattern in [
        "(?P<name>a)(?P<name>b)",
        "(?x) # comment\n (a\n",
        "a{\n",
        "(?i-i:a)",
        "(?x) [a-z # comment\n",
        "(?:cat[0-9]|cat[a-z])",
    ] {
        check(pattern);
        count += 1;
    }
    println!("{count} patterns match pinned upstream AST, HIR, properties, literal extraction/optimization and diagnostics");
}
