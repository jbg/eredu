// Copyright 2026 The Fancy Regex Authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! Translation of easy `Expr` subtrees directly into `regex_syntax::hir::Hir`.
//!
//! Delegating a subtree to regex-automata used to mean re-serializing it with
//! [`Expr::to_str`] and letting the engine parse the string all over again —
//! a full AST parse plus Hir translation per delegated engine. Building the
//! `Hir` directly from the `Expr` tree skips that second parse.
//!
//! The translation is defined to produce **exactly** the `Hir` that
//! `regex_syntax` would produce for the `to_str` output (there is an oracle
//! test asserting this equality). Node kinds outside the easy subset that
//! `to_str` handles — and any fragment whose semantics depend on parsing (a
//! `Delegate`'s inner string, a case-insensitive literal) that fails its
//! (fragment-sized) parse — return `None`, and the caller falls back to the
//! string path, preserving both behavior and error reporting.

use crate::allocation::{AllocationError, Context};
use crate::{Assertion, Expr};
use alloc::string::String;
use alloc::vec::Vec;
use core::convert::TryFrom;
use regex_syntax::hir::{Capture, Dot, Hir, Look, Repetition};

/// Syntax options and capture counter shared by direct HIR translations.
pub(crate) struct HirCtx {
    unicode: bool,
    utf8: bool,
    next_group: u32,
}
impl HirCtx {
    pub(crate) fn new(unicode: bool, utf8: bool) -> Self {
        Self {
            unicode,
            utf8,
            next_group: 1,
        }
    }
}

pub(crate) fn expr_to_hir(expr: &Expr, ctx: &mut HirCtx) -> Option<Hir> {
    expr_to_hir_with_allocations(expr, ctx, Context::unenforced())
        .expect("expression HIR allocation")
}

/// Semantic fallback remains `None`; an allocation refusal must never select a fallback.
pub(crate) fn expr_to_hir_with_allocations(
    expr: &Expr,
    ctx: &mut HirCtx,
    allocation: Context<'_>,
) -> Result<Option<Hir>, AllocationError> {
    let a = allocation.storage;
    // Preserve early semantic fallback without hiding allocation failure.
    macro_rules! supported {
        ($value:expr) => {
            match $value {
                Some(value) => value,
                None => return Ok(None),
            }
        };
    }
    Ok(Some(match *expr {
        Expr::Empty | Expr::DefineGroup { .. } => Hir::empty_with_allocations(a)?,
        Expr::Any { newline, crlf } => {
            let dot = match (newline, crlf, ctx.unicode, ctx.utf8) {
                (true, _, true, _) => Dot::AnyChar,
                (true, _, false, false) => Dot::AnyByte,
                (false, false, true, _) => Dot::AnyCharExceptLF,
                (false, false, false, false) => Dot::AnyByteExceptLF,
                (false, true, true, _) => Dot::AnyCharExceptCRLF,
                (false, true, false, false) => Dot::AnyByteExceptCRLF,
                (_, _, false, true) => return Ok(None),
            };
            Hir::dot_with_allocations(dot, a)?
        }
        Expr::Literal { ref val, casei } => {
            if !casei {
                Hir::literal_with_allocations(a.copy_slice(val.as_bytes())?, a)?
            } else {
                let mut cooked = String::new();
                a.push_str(&mut cooked, "(?i:")?;
                crate::push_quoted_with_allocations(&mut cooked, val, allocation)?;
                a.push_char(&mut cooked, ')')?;
                supported!(parse_fragment(&cooked, ctx, allocation)?)
            }
        }
        Expr::Assertion(assertion) => Hir::look_with_allocations(
            match assertion {
                Assertion::StartText => Look::Start,
                Assertion::EndText => Look::End,
                Assertion::StartLine { crlf: false }
                | Assertion::StartLineOniguruma { crlf: false } => Look::StartLF,
                Assertion::EndLine { crlf: false } => Look::EndLF,
                Assertion::StartLine { crlf: true }
                | Assertion::StartLineOniguruma { crlf: true } => Look::StartCRLF,
                Assertion::EndLine { crlf: true } => Look::EndCRLF,
                _ => return Ok(None),
            },
            a,
        )?,
        Expr::Concat(ref children) | Expr::Alt(ref children) => {
            let mut subs = Vec::new();
            a.grow(&mut subs, children.len())?;
            for child in children {
                subs.push(supported!(expr_to_hir_with_allocations(
                    child, ctx, allocation
                )?));
            }
            if matches!(expr, Expr::Concat(_)) {
                Hir::concat_with_allocations(subs, a)?
            } else {
                Hir::alternation_with_allocations(subs, a)?
            }
        }
        Expr::Group(ref child) => {
            let index = ctx.next_group;
            ctx.next_group += 1;
            let sub = supported!(expr_to_hir_with_allocations(child, ctx, allocation)?);
            Hir::capture_with_allocations(
                Capture {
                    index,
                    name: None,
                    sub: a.boxed(sub)?,
                },
                a,
            )?
        }
        Expr::Repeat {
            ref child,
            lo,
            hi,
            greedy,
        } => {
            let min = supported!(u32::try_from(lo).ok());
            let max = if hi == usize::MAX {
                None
            } else {
                Some(supported!(u32::try_from(hi).ok()))
            };
            let sub = supported!(expr_to_hir_with_allocations(child, ctx, allocation)?);
            Hir::repetition_with_allocations(
                Repetition {
                    min,
                    max,
                    greedy,
                    sub: a.boxed(sub)?,
                },
                a,
            )?
        }
        Expr::Delegate { ref inner, casei } => {
            if casei {
                let mut cooked = String::new();
                a.push_str(&mut cooked, "(?i:")?;
                a.push_str(&mut cooked, inner)?;
                a.push_char(&mut cooked, ')')?;
                supported!(parse_fragment(&cooked, ctx, allocation)?)
            } else {
                supported!(parse_fragment(inner, ctx, allocation)?)
            }
        }
        _ => return Ok(None),
    }))
}

fn parse_fragment(
    fragment: &str,
    ctx: &HirCtx,
    allocation: Context<'_>,
) -> Result<Option<Hir>, AllocationError> {
    match regex_syntax::ParserBuilder::new()
        .utf8(ctx.utf8)
        .unicode(ctx.unicode)
        .build()
        .parse_with_allocations(fragment, allocation.policy)
    {
        Ok(hir) => Ok(Some(hir)),
        Err(regex_syntax::Error::Parse(error)) => match error.kind() {
            regex_syntax::ast::ErrorKind::Allocation(error) => Err(*error),
            _ => Ok(None),
        },
        Err(regex_syntax::Error::Translate(error)) => match error.kind() {
            regex_syntax::hir::ErrorKind::Allocation(error) => Err(*error),
            _ => Ok(None),
        },
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{analyze, AnalyzeContext};
    use crate::{BytesMode, RegexOptions};
    use alloc::string::ToString;

    #[test]
    fn each_direct_and_parsed_hir_destination_preserves_refusal() {
        use crate::allocation::Allocation;
        use core::cell::Cell;
        struct Funding {
            calls: Cell<usize>,
            refuse: usize,
        }
        impl Allocation for Funding {
            fn reserve(&self, _: usize) -> Result<(), AllocationError> {
                let call = self.calls.get();
                self.calls.set(call + 1);
                if call >= self.refuse {
                    Err(AllocationError::Refused)
                } else {
                    Ok(())
                }
            }
        }
        for pattern in [
            "abc",
            "a|b",
            "(a|bc){2,4}",
            "(?i)ſΣ",
            r"[a-z\d]+",
            "(?R:.)",
            r"\p{UnknownProperty}",
        ] {
            let tree = Expr::parse_tree(pattern).unwrap();
            let expected = expr_to_hir(&tree.expr, &mut HirCtx::new(true, true));
            let funding = Funding {
                calls: Cell::new(0),
                refuse: usize::MAX,
            };
            let actual = expr_to_hir_with_allocations(
                &tree.expr,
                &mut HirCtx::new(true, true),
                Context::new(&funding),
            )
            .unwrap();
            assert_eq!(actual, expected, "{}", pattern);
            let requests = funding.calls.get();
            assert!(requests > 0, "{}", pattern);
            drop(actual);
            for refuse in 0..requests {
                let funding = Funding {
                    calls: Cell::new(0),
                    refuse,
                };
                assert_eq!(
                    expr_to_hir_with_allocations(
                        &tree.expr,
                        &mut HirCtx::new(true, true),
                        Context::new(&funding)
                    ),
                    Err(AllocationError::Refused),
                    "{} at {}",
                    pattern,
                    refuse
                );
                assert_eq!(funding.calls.get(), refuse + 1);
            }
        }
    }
    /// Patterns covering the easy subset: literals (plain, quoted, casei,
    /// non-ASCII), dots in all newline/CRLF modes, anchors in all line modes,
    /// concat/alt/group nesting, all quantifier shapes, classes and other
    /// delegated fragments, and mixed inline flags.
    const PATTERNS: &[&str] = &[
        "a",
        "abc",
        "a.c",
        ".",
        "(?s).",
        "(?s:.)x",
        "^abc$",
        "(?m)^abc$",
        "(?m:^)foo",
        "(?Rm)^x$",
        "(?i)abc",
        "(?i)aBc(?-i)d",
        "(?i)δΔ",
        "αβγ",
        "a+b*c?",
        "a{2,5}",
        "a{3}",
        "a{2,}?",
        "a+?b??",
        "(a)(b(c))",
        "(?:ab|cd)e",
        "a|b|",
        "(a|)",
        "()",
        "[a-z]+",
        r"\d\w\s",
        "[^a-c]{2,3}",
        r"\p{L}+",
        "(?i)[a-k]x",
        r"\x61\n\t",
        "(a+)(?:b|c)*",
        r"\.\*\+#",
        r"(?i)ſ",
        // Hard patterns must be skipped by the harness (info.hard).
        r"\bword\b",
        r"(a)\1",
    ];

    /// The translator must produce exactly the Hir the parser produces for
    /// the `to_str` serialization — or `None` where the parser errors, so the
    /// string path can report the error.
    fn check_oracle(bytes_mode: BytesMode) {
        for pattern in PATTERNS {
            let options = RegexOptions {
                bytes_mode,
                ..RegexOptions::default()
            };
            let Ok(tree) = crate::Expr::parse_tree_with_flags(pattern, options.compute_flags())
            else {
                continue;
            };
            let Ok(info) = analyze(&tree, AnalyzeContext::default()) else {
                continue;
            };
            if info.hard {
                continue;
            }
            let mut cooked = String::new();
            tree.expr.to_str(&mut cooked, 0);
            let unicode =
                options.syntaxc.get_unicode() && !matches!(options.bytes_mode, BytesMode::Ascii);
            let utf8 = matches!(options.bytes_mode, BytesMode::Unicode);
            let expected = regex_syntax::ParserBuilder::new()
                .utf8(utf8)
                .unicode(unicode)
                .build()
                .parse(&cooked);
            let got = expr_to_hir(&tree.expr, &mut HirCtx::new(unicode, utf8));
            match expected {
                Ok(hir) => assert_eq!(
                    Some(hir),
                    got,
                    "Hir mismatch for {:?} (cooked {:?}) in {:?} mode",
                    pattern,
                    cooked,
                    bytes_mode
                ),
                Err(_) => assert_eq!(
                    None, got,
                    "expected fallback for {:?} (cooked {:?}) in {:?} mode: the parser \
                     rejects it, so the translator must not accept it",
                    pattern, cooked, bytes_mode
                ),
            }
        }
    }

    #[test]
    fn oracle_unicode_mode() {
        check_oracle(BytesMode::Unicode);
    }

    #[test]
    fn oracle_unicode_bytes_mode() {
        check_oracle(BytesMode::UnicodeBytes);
    }

    #[test]
    fn oracle_ascii_mode() {
        check_oracle(BytesMode::Ascii);
    }

    #[test]
    fn group_numbering_matches_textual_order() {
        let tree = crate::Expr::parse_tree("(a)((b)(c))").unwrap();
        let hir = expr_to_hir(&tree.expr, &mut HirCtx::new(true, true)).unwrap();
        let cooked = {
            let mut s = String::new();
            tree.expr.to_str(&mut s, 0);
            s
        };
        let expected = regex_syntax::ParserBuilder::new().build().parse(&cooked);
        assert_eq!(expected.unwrap().to_string(), hir.to_string());
    }
}
