use super::*;
use crate::compiler::lexer::Tokenizer;
use crate::compiler::tokens::Token;
use crate::syntax::SyntaxConfig;
use std::fmt::Write as _;

mod reference_lexer;
mod reference_unescape;
include!("tests/fixtures.rs");

#[derive(Debug, PartialEq)]
struct Observation {
    kind: TokenKind,
    text: Option<String>,
    integer: Option<u128>,
    float: Option<u64>,
    span: Span,
}
fn observed(view: TokenView<'_, '_>) -> Observation {
    Observation {
        kind: view.kind(),
        text: view.text().map(str::to_owned),
        integer: view.integer(),
        float: view.float().map(f64::to_bits),
        span: view.span(),
    }
}
fn ordinary(token: Token<'_>, span: Span) -> Observation {
    let kind = kind(&token);
    let (text, integer, float) = match token {
        Token::TemplateData(s) | Token::Ident(s) | Token::Str(s) => {
            (Some(s.to_owned()), None, None)
        }
        Token::String(s) => (Some(s.into_string()), None, None),
        Token::Int(n) => (None, Some(n as u128), None),
        Token::Int128(n) => (None, Some(*n), None),
        Token::Float(n) => (None, None, Some(n.to_bits())),
        _ => (None, None, None),
    };
    Observation {
        kind,
        text,
        integer,
        float,
        span,
    }
}
fn error_matches(actual: &LexicalError, reference: &crate::Error) {
    assert_eq!(actual.kind(), reference.kind());
    assert_eq!(actual.detail(), reference.detail());
    assert_eq!(
        actual.span().map(|s| s.start_line as usize),
        reference.line()
    );
    assert_eq!(reference.name(), actual.span().map(|_| "fixture"));
    #[cfg(feature = "debug")]
    assert_eq!(actual.span(), reference.span());
}
fn compare(source: &str, in_expr: bool, whitespace: WhitespaceConfig) -> usize {
    let mut old = reference_lexer::Tokenizer::new(
        source,
        "fixture",
        in_expr,
        SyntaxConfig::default(),
        reference_lexer::WhitespaceConfig {
            keep_trailing_newline: whitespace.keep_trailing_newline,
            lstrip_blocks: whitespace.lstrip_blocks,
            trim_blocks: whitespace.trim_blocks,
        },
    );
    let mut current = Tokenizer::new(
        source,
        "fixture",
        in_expr,
        SyntaxConfig::default(),
        whitespace,
    );
    let mut expected = Vec::new();
    let error = loop {
        match (old.next_token(), current.next_token()) {
            (Ok(Some((a, sa))), Ok(Some((b, sb)))) => {
                let a = ordinary(a, sa);
                assert_eq!(a, ordinary(b, sb), "ordinary adapter: {source:?}");
                expected.push(a);
            }
            (Ok(None), Ok(None)) => break None,
            (Err(a), Err(b)) => {
                assert_eq!(a.kind(), b.kind());
                assert_eq!(a.detail(), b.detail());
                assert_eq!(a.name(), b.name());
                assert_eq!(a.line(), b.line());
                #[cfg(feature = "debug")]
                assert_eq!(a.span(), b.span());
                break Some(a);
            }
            pair => panic!("ordinary termination mismatch {source:?}: {pair:?}"),
        }
    };
    let plan = Plan::inspect(source, "fixture", in_expr, whitespace).unwrap();
    match (plan.construct(), error) {
        (Ok(tokens), None) => {
            assert_eq!(
                tokens.tokens().map(observed).collect::<Vec<_>>(),
                expected,
                "closed: {source:?}"
            );
            assert_eq!(tokens.source().as_ptr(), source.as_ptr());
        }
        (Err(error), Some(reference)) => {
            assert_eq!(
                error.partial().tokens().map(observed).collect::<Vec<_>>(),
                expected,
                "prefix: {source:?}"
            );
            let Cause::Lexical(actual) = error.cause() else {
                panic!("wrong cause: {error:?}")
            };
            error_matches(actual, &reference);
        }
        pair => panic!("closed termination mismatch {source:?}: {pair:?}"),
    }
    expected.len()
}

#[test]
fn all_existing_template_sources_match_pre_refactor_and_shared_ordinary_lexer() {
    assert_eq!(FIXTURES.len(), 32);
    let mut tokens = 0;
    for (_, source) in FIXTURES {
        for bits in 0..8 {
            tokens += compare(
                source,
                false,
                WhitespaceConfig {
                    keep_trailing_newline: bits & 1 != 0,
                    lstrip_blocks: bits & 2 != 0,
                    trim_blocks: bits & 4 != 0,
                },
            );
        }
    }
    assert!(tokens > 10_000);
}

#[test]
fn numeric_radix_underscores_wide_integer_rounding_and_errors_match() {
    for source in [
        "0 42 18446744073709551615 18446744073709551616 340282366920938463463374607431768211455",
        "0b1_010 0O7_7 0Xff_ff 0x_1 1__2 1_.2_3e+0_2",
        "1.0 1e999 1e-999 2.2250738585072012e-308 0.1000000000000000055511151231257827021181583404541015625",
        "1.foo 1.e+2 1.e+ 1e", "0x", "0b2", "0o8", "0x1_", "1_", "1e_", "1e+_1",
        "340282366920938463463374607431768211456", "1 + 1e + 'later'",
    ] { compare(source, true, WhitespaceConfig::default()); }
    let source = format!("1.{}1e-1_0", "0".repeat(900));
    compare(&source, true, WhitespaceConfig::default());
    let tokens = Plan::inspect(
        "1e999 18446744073709551616",
        "fixture",
        true,
        WhitespaceConfig::default(),
    )
    .unwrap()
    .construct()
    .unwrap();
    assert_eq!(tokens.tokens().next().unwrap().float(), Some(f64::INFINITY));
    assert_eq!(tokens.tokens().nth(1).unwrap().kind(), TokenKind::Int128);
}

#[test]
fn escape_bytes_unicode_surrogates_and_radix_plus_preserve_reference_semantics() {
    for n in 0..=255u16 {
        compare(
            &format!("'\\x{n:02x}' '\\{n:03o}'"),
            true,
            WhitespaceConfig::default(),
        );
    }
    for raw in [
        r"\u+123",
        r"\x+f",
        r"\u-123",
        r"\ud83d\udca9",
        r"\udc00\ud800",
        r"\ud800x",
        r"\u2603雪🙂",
        r"\q\8\9",
        r"\x雪a",
        r"\u雪a00",
        r"\x",
        r"\u123",
        r"\400",
        r"\777",
        r"\378",
        r"\0x",
    ] {
        compare(&format!("'{raw}'"), true, WhitespaceConfig::default());
    }
    for n in [0xd7ff, 0xd800, 0xdbff, 0xdc00, 0xdfff, 0xe000, 0xffff] {
        compare(
            &format!("'prefix\\u{n:04x}'"),
            true,
            WhitespaceConfig::default(),
        );
    }
    assert_eq!(
        crate::utils::unescape(r"\u+123\x+f").unwrap(),
        "\u{123}\u{f}"
    );
}

#[test]
fn errors_keep_ordinary_order_and_real_completed_and_partial_literal_prefixes() {
    for source in [
        "{{ 1e 'bad\\x' }}",
        "{{ 'good\\n' 12_3 'kept\\xQZ' 1e }}",
        "{{ 'good\\n' 1e }}",
        "hello {# unfinished",
        "{% raw %}unfinished",
        "{{ 'unterminated",
        "{{ ? }}",
        "{{ 0x_ 'bad\\x' }}",
    ] {
        compare(source, false, WhitespaceConfig::default());
    }
    let error = Plan::inspect(
        "'good\\n' 12_3 'kept\\xQZ'",
        "fixture",
        true,
        WhitespaceConfig::default(),
    )
    .unwrap()
    .construct()
    .unwrap_err();
    assert_eq!(error.partial().len(), 2);
    assert_eq!(error.partial().literals.as_slice(), b"good\nkept");
    assert_eq!(error.partial().numeric.as_slice(), b"123");
    assert!(matches!(error.cause(), Cause::Lexical(error) if error.kind() == ErrorKind::BadEscape));
}

#[test]
fn three_real_reserve_failures_keep_actual_preceding_destinations() {
    let source = "'hello\\n' 123_456 340282366920938463463374607431768211455";
    for (index, buffer) in [Buffer::Tokens, Buffer::Literals, Buffer::NumericScratch]
        .into_iter()
        .enumerate()
    {
        let plan = Plan::inspect(source, "fixture", true, WhitespaceConfig::default()).unwrap();
        let required = plan.requirements();
        assert!(
            required.tokens() > 0 && required.literal_bytes() > 0 && required.numeric_bytes() > 0
        );
        let error = plan.construct_inner(Some(buffer)).unwrap_err();
        assert!(
            matches!(error.cause(), Cause::Reserve { buffer: actual, .. } if *actual == buffer)
        );
        assert!(std::error::Error::source(&error).is_some());
        let capacity = error.partial().capacities();
        for (i, need) in [
            required.tokens(),
            required.literal_bytes(),
            required.numeric_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            if i < index {
                assert!(capacity[i] >= need);
            } else {
                assert_eq!(capacity[i], 0);
            }
        }
        assert!(error.partial().is_empty());
        assert_eq!(error.partial().source().as_ptr(), source.as_ptr());
    }
}

#[test]
fn each_one_short_destination_stops_before_growth_and_keeps_prefix() {
    let source = "'a\\n' 1_2 'b\\t' 3_4";
    for buffer in [Buffer::Tokens, Buffer::Literals, Buffer::NumericScratch] {
        let mut plan = Plan::inspect(source, "fixture", true, WhitespaceConfig::default()).unwrap();
        // Private corruption tests guard finite materialization; no public API
        // accepts capacities or lets a caller manufacture this altered plan.
        match buffer {
            Buffer::Tokens => plan.requirements.tokens -= 1,
            Buffer::Literals => plan.requirements.literal_bytes -= 1,
            Buffer::NumericScratch => plan.requirements.numeric_bytes -= 1,
        }
        let requirements = plan.requirements();
        let error = plan.construct().unwrap_err();
        assert!(matches!(error.cause(), Cause::Capacity(actual) if *actual == buffer));
        assert!(error.partial().records.len() <= requirements.tokens());
        assert!(error.partial().literals.len() <= requirements.literal_bytes());
        assert!(error.partial().numeric.len() <= requirements.numeric_bytes());
        assert!(!error.partial().is_empty());
    }
}

#[test]
fn empty_zero_storage_geometry_and_saturating_line_spans_remain_explicit() {
    for source in ["", "\n", "\r\n"] {
        let owner = Plan::inspect(source, "fixture", false, WhitespaceConfig::default())
            .unwrap()
            .construct()
            .unwrap();
        assert!(owner.is_empty());
        assert_eq!(owner.capacities(), [0; 3]);
        assert_eq!(owner.requirements().heap_bytes(), 0);
    }
    compare("\n{{雪}}\n\r", false, WhitespaceConfig::default());
    let long_lines = format!("{}{{{{ 7 }}}}", "\n".repeat(65_540));
    compare(&long_lines, false, WhitespaceConfig::default());
    let long_column = "x".repeat(u16::MAX as usize);
    assert!(matches!(
        Plan::inspect(&long_column, "fixture", false, WhitespaceConfig::default()),
        Err(PlanError::Geometry)
    ));
    for (t, b, n) in [(usize::MAX, 0, 0), (0, usize::MAX, 0), (0, 0, usize::MAX)] {
        assert!(matches!(requirements(t, b, n), Err(PlanError::Geometry)));
    }
}

#[test]
fn source_identity_and_parallel_construction_do_not_share_mutable_storage() {
    let left = String::from("a{{ 'x\\n' + 1_0 }}z");
    let right = left.clone();
    assert_ne!(left.as_ptr(), right.as_ptr());
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            Plan::inspect(&left, "left", false, WhitespaceConfig::default())
                .unwrap()
                .construct()
                .unwrap()
        });
        let b = scope.spawn(|| {
            Plan::inspect(&right, "right", false, WhitespaceConfig::default())
                .unwrap()
                .construct()
                .unwrap()
        });
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(a.source().as_ptr(), left.as_ptr());
    assert_eq!(b.source().as_ptr(), right.as_ptr());
    assert_ne!(a.records.as_ptr(), b.records.as_ptr());
    assert_ne!(a.literals.as_ptr(), b.literals.as_ptr());
    assert_ne!(a.numeric.as_ptr(), b.numeric.as_ptr());
    assert_eq!(
        a.tokens().map(observed).collect::<Vec<_>>(),
        b.tokens().map(observed).collect::<Vec<_>>()
    );
    drop(a);
    assert_eq!(b.tokens().filter_map(TokenView::integer).sum::<u128>(), 10);
}

#[test]
fn ordinary_parser_consumes_shared_literals_with_strict_undefined_and_custom_formatter() {
    let mut env = crate::Environment::new();
    env.set_undefined_behavior(crate::UndefinedBehavior::Strict);
    env.set_formatter(|out, _state, value| {
        write!(out, "[{value}]").map_err(|_| crate::Error::from(ErrorKind::InvalidOperation))
    });
    env.add_template(
        "literals",
        "{{ 'a\\n' }}{{ 18446744073709551616 }}{{ 1_2 + 3 }}",
    )
    .unwrap();
    assert_eq!(
        env.get_template("literals").unwrap().render(()).unwrap(),
        "[a\n][18446744073709551616][15]"
    );
    env.add_template("missing", "{{missing}}").unwrap();
    assert_eq!(
        env.get_template("missing")
            .unwrap()
            .render(())
            .unwrap_err()
            .kind(),
        ErrorKind::UndefinedError
    );
}

#[cfg(feature = "custom_syntax")]
#[test]
fn ordinary_custom_delimiters_and_line_statements_keep_reference_scanning() {
    let syntax = SyntaxConfig::builder()
        .block_delimiters("<%", "%>")
        .variable_delimiters("<<", ">>")
        .line_statement_prefix("#")
        .line_comment_prefix("##")
        .build()
        .unwrap();
    let source = "## comment\n# set x = 1_2\n<<x>> <% raw %> {{ untouched }} <% endraw %>";
    let mut old = reference_lexer::Tokenizer::new(
        source,
        "fixture",
        false,
        syntax.clone(),
        reference_lexer::WhitespaceConfig::default(),
    );
    let mut current = Tokenizer::new(
        source,
        "fixture",
        false,
        syntax,
        WhitespaceConfig::default(),
    );
    loop {
        match (old.next_token().unwrap(), current.next_token().unwrap()) {
            (Some((a, sa)), Some((b, sb))) => assert_eq!(ordinary(a, sa), ordinary(b, sb)),
            (None, None) => break,
            pair => panic!("{pair:?}"),
        }
    }
    // The default-only closed entry does not accept or silently consume a custom config.
    compare("<< 12 >>", false, WhitespaceConfig::default());
}

fn kind(token: &Token<'_>) -> TokenKind {
    match token {
        Token::TemplateData(..) => TokenKind::TemplateData,
        Token::VariableStart => TokenKind::VariableStart,
        Token::VariableEnd => TokenKind::VariableEnd,
        Token::BlockStart => TokenKind::BlockStart,
        Token::BlockEnd => TokenKind::BlockEnd,
        Token::Ident(..) => TokenKind::Ident,
        Token::Str(..) => TokenKind::Str,
        Token::String(..) => TokenKind::String,
        Token::Int(..) => TokenKind::Int,
        Token::Int128(..) => TokenKind::Int128,
        Token::Float(..) => TokenKind::Float,
        Token::Plus => TokenKind::Plus,
        Token::Minus => TokenKind::Minus,
        Token::Mul => TokenKind::Mul,
        Token::Div => TokenKind::Div,
        Token::FloorDiv => TokenKind::FloorDiv,
        Token::Pow => TokenKind::Pow,
        Token::Mod => TokenKind::Mod,
        Token::Dot => TokenKind::Dot,
        Token::Comma => TokenKind::Comma,
        Token::Colon => TokenKind::Colon,
        Token::Tilde => TokenKind::Tilde,
        Token::Assign => TokenKind::Assign,
        Token::Pipe => TokenKind::Pipe,
        Token::Eq => TokenKind::Eq,
        Token::Ne => TokenKind::Ne,
        Token::Gt => TokenKind::Gt,
        Token::Gte => TokenKind::Gte,
        Token::Lt => TokenKind::Lt,
        Token::Lte => TokenKind::Lte,
        Token::BracketOpen => TokenKind::BracketOpen,
        Token::BracketClose => TokenKind::BracketClose,
        Token::ParenOpen => TokenKind::ParenOpen,
        Token::ParenClose => TokenKind::ParenClose,
        Token::BraceOpen => TokenKind::BraceOpen,
        Token::BraceClose => TokenKind::BraceClose,
    }
}

#[test]
fn closed_expression_end_rejects_while_ordinary_keeps_its_existing_panic() {
    let source = "1 }} trailing";
    let old = std::panic::catch_unwind(|| {
        let mut scanner = reference_lexer::Tokenizer::new(
            source,
            "fixture",
            true,
            SyntaxConfig::default(),
            reference_lexer::WhitespaceConfig::default(),
        );
        while scanner.next_token().unwrap().is_some() {}
    });
    let current = std::panic::catch_unwind(|| {
        let mut scanner = Tokenizer::new(
            source,
            "fixture",
            true,
            SyntaxConfig::default(),
            WhitespaceConfig::default(),
        );
        while scanner.next_token().unwrap().is_some() {}
    });
    assert!(old.is_err() && current.is_err());
    let error = Plan::inspect(source, "fixture", true, WhitespaceConfig::default())
        .unwrap()
        .construct()
        .unwrap_err();
    assert!(matches!(error.cause(), Cause::ExpressionEnd));
    assert_eq!(error.partial().len(), 2);
}
