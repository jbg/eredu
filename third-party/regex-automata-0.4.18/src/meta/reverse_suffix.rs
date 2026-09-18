/*!
Routines specific to the reverse suffix optimization.
*/

use alloc::vec::Vec;

use regex_syntax::hir::{literal::Literal, Hir, HirKind, Repetition};

use crate::{
    meta::prefix,
    util::allocation::{Allocation, AllocationError, Allocator},
};

/// Returns true when it's impossible for an earlier match to be detected after
/// a literal candidate (corresponding to `suffix`) has been found.
///
/// Specifically, that there is no earlier match than what a reverse scan of
/// `hirs` after a match of `suffix` reports.
///
/// At present, this always returns `false` when `hirs` has any length except
/// `1`. That is, this optimization does not apply to multi-regex.
pub(super) fn has_no_earlier_match_with_allocations(
    hirs: &[&Hir],
    suffix: &[u8],
    funding: &dyn Allocation,
) -> Result<bool, AllocationError> {
    if hirs.len() != 1 || suffix.is_empty() {
        return Ok(false);
    }
    let Some(prefix) =
        strip_literal_suffix_with_allocations(hirs[0], suffix, funding)?
    else {
        return Ok(false);
    };
    if prefix::hir_has_fixed_length(&prefix) {
        return Ok(true);
    }
    let allocation = Allocator::new(funding);
    let syntax = regex_syntax::allocation::Allocator::new(&allocation);
    let mut literal = Literal::exact_with_allocations(suffix, syntax)?;
    literal.make_inexact();
    if prefix::has_disjoint_class_separator_with_allocations(
        &prefix,
        &[literal],
        funding,
    )? {
        return Ok(true);
    }
    Ok(!prefix::hir_can_contain_literal_with_allocations(
        &prefix, suffix, funding,
    )?)
}

/// Strip `suffix` from the end of `hir` and return the HIR that remains.
///
/// This is conservative. It returns `None` when the structure of the HIR does
/// not make the suffix straightforward to remove, even when `suffix` might be
/// required by the pattern.
#[cfg(test)]
fn strip_literal_suffix(hir: &Hir, suffix: &[u8]) -> Option<Hir> {
    strip_literal_suffix_with_allocations(hir, suffix, &crate::util::allocation::Unenforced)
        .expect("ordinary suffix copy allocation")
}

/// Run the original right-to-left suffix removal with paid continuations.
/// Untouched prefix children are copied once, after stripping succeeds.
fn strip_literal_suffix_with_allocations<'a>(
    hir: &'a Hir,
    suffix: &[u8],
    funding: &dyn Allocation,
) -> Result<Option<Hir>, AllocationError> {
    enum Frame<'a> {
        Visit(&'a Hir, usize),
        Concat {
            children: &'a [Hir],
            index: usize,
            before: usize,
            tail: Vec<Hir>,
        },
        Repeat {
            rep: &'a Repetition,
            min: u32,
            max: Option<u32>,
            before: usize,
            tail: Vec<Hir>,
        },
    }
    let allocation = Allocator::new(funding);
    let syntax = regex_syntax::allocation::Allocator::new(&allocation);
    let mut stack = Vec::new();
    allocation.push(&mut stack, Frame::Visit(hir, suffix.len()))?;
    let mut result = None;
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Visit(mut hir, cursor) => {
                if cursor == 0 {
                    result = Some((hir.clone_with_allocations(syntax)?, 0));
                    continue;
                }
                while let HirKind::Capture(capture) = hir.kind() {
                    hir = &capture.sub;
                }
                match hir.kind() {
                    HirKind::Literal(lit) => {
                        let bytes = &lit.0;
                        let mut len = 0;
                        while len < bytes.len()
                            && len < cursor
                            && bytes[bytes.len() - len - 1]
                                == suffix[cursor - len - 1]
                        {
                            len += 1;
                        }
                        if len == 0 || (len < bytes.len() && len < cursor) {
                            return Ok(None);
                        }
                        result = Some((
                            Hir::literal_with_allocations(
                                syntax
                                    .copy_slice(&bytes[..bytes.len() - len])?,
                                syntax,
                            )?,
                            cursor - len,
                        ));
                    }
                    HirKind::Concat(children) => {
                        let index = children.len() - 1;
                        allocation.push(
                            &mut stack,
                            Frame::Concat {
                                children,
                                index,
                                before: cursor,
                                tail: Vec::new(),
                            },
                        )?;
                        allocation.push(
                            &mut stack,
                            Frame::Visit(&children[index], cursor),
                        )?;
                    }
                    HirKind::Repetition(rep) => {
                        if rep.min == 0 {
                            return Ok(None);
                        }
                        allocation.push(
                            &mut stack,
                            Frame::Repeat {
                                rep,
                                min: rep.min,
                                max: rep.max,
                                before: cursor,
                                tail: Vec::new(),
                            },
                        )?;
                        allocation.push(
                            &mut stack,
                            Frame::Visit(&rep.sub, cursor),
                        )?;
                    }
                    _ => return Ok(None),
                }
            }
            Frame::Concat {
                children,
                index,
                before,
                mut tail,
            } => {
                let (stripped, after) =
                    result.take().expect("completed suffix child");
                if after == before {
                    return Ok(None);
                }
                allocation.push(&mut tail, stripped)?;
                if after > 0 && index > 0 {
                    allocation.push(
                        &mut stack,
                        Frame::Concat {
                            children,
                            index: index - 1,
                            before: after,
                            tail,
                        },
                    )?;
                    allocation.push(
                        &mut stack,
                        Frame::Visit(&children[index - 1], after),
                    )?;
                } else {
                    let mut prefix = Vec::new();
                    allocation.grow(
                        &mut prefix,
                        index
                            .checked_add(tail.len())
                            .ok_or(AllocationError::SizeOverflow)?,
                    )?;
                    for child in &children[..index] {
                        prefix.push(child.clone_with_allocations(syntax)?);
                    }
                    prefix.extend(tail.into_iter().rev());
                    result = Some((
                        Hir::concat_with_allocations(prefix, syntax)?,
                        after,
                    ));
                }
            }
            Frame::Repeat {
                rep,
                min,
                max,
                before,
                mut tail,
            } => {
                let (stripped, after) =
                    result.take().expect("completed repetition suffix");
                if after == before {
                    return Ok(None);
                }
                let min = min.saturating_sub(1);
                let max = max.map(|max| max.saturating_sub(1));
                if !matches!(stripped.kind(), HirKind::Empty) {
                    allocation.push(&mut tail, stripped)?;
                }
                if after > 0 {
                    if min == 0 {
                        return Ok(None);
                    }
                    allocation.push(
                        &mut stack,
                        Frame::Repeat {
                            rep,
                            min,
                            max,
                            before: after,
                            tail,
                        },
                    )?;
                    allocation
                        .push(&mut stack, Frame::Visit(&rep.sub, after))?;
                } else {
                    let rest = Hir::repetition_with_allocations(
                        Repetition {
                            min,
                            max,
                            greedy: rep.greedy,
                            sub: syntax.boxed(
                                rep.sub.clone_with_allocations(syntax)?,
                            )?,
                        },
                        syntax,
                    )?;
                    let mut prefix = Vec::new();
                    allocation.grow(
                        &mut prefix,
                        tail.len()
                            .checked_add(1)
                            .ok_or(AllocationError::SizeOverflow)?,
                    )?;
                    if !matches!(rest.kind(), HirKind::Empty) {
                        prefix.push(rest);
                    }
                    prefix.extend(tail.into_iter().rev());
                    result = Some((
                        Hir::concat_with_allocations(prefix, syntax)?,
                        0,
                    ));
                }
            }
        }
    }
    let (prefix, after) = result.expect("completed suffix root");
    Ok(if after == 0 { Some(prefix) } else { None })
}

// We only test when we have `unicode-perl` here since some regexes require
// that.
#[cfg(all(feature = "unicode-perl", not(miri),))]
#[cfg(test)]
mod tests {
    use crate::util::syntax;

    use super::*;

    #[track_caller]
    fn assert_strip(pattern: &str, suffix: &str, expected: Option<&str>) {
        let hir = syntax::parse(pattern).unwrap();
        let got = strip_literal_suffix(&hir, suffix.as_bytes());
        let expected = expected.map(|pattern| syntax::parse(pattern).unwrap());
        assert_eq!(expected, got);
    }

    #[test]
    fn literal() {
        assert_strip("foobar", "bar", Some("foo"));
        assert_strip("foobar", "foobar", Some(""));
        assert_strip("foobar", "", Some("foobar"));
    }

    #[test]
    fn capture() {
        assert_strip(r".+foo(bar)", "foobar", Some(r".+"));
    }

    #[test]
    fn concat() {
        assert_strip(r".+(?:ab){2}cd", "ababcd", Some(r".+"));
        assert_strip(r"a(?:)cd", "acd", Some(r""));
        assert_strip(r"az{0}cd", "acd", Some(r""));
        assert_strip(r"az{2}cd", "acd", None);
        assert_strip(r"az{0,2}cd", "acd", None);
    }

    #[test]
    fn repetition() {
        assert_strip(r"x(?:ab){3}", "bab", Some("xaba"));
        assert_strip(r"x(?:ab){2,}", "bab", Some(r"x(?:ab)*a"));
    }

    #[test]
    fn mismatch() {
        assert_strip("foobar", "baz", None);
        assert_strip("foobar", "xfoobar", None);
        assert_strip(r"foo[a-z]", "a", None);
    }

    #[test]
    fn suffix_must_be_adjacent() {
        assert_strip(r"foo(xbar)", "foobar", None);
    }

    #[test]
    fn repetition_must_be_required() {
        assert_strip(r"x(?:ab)*cd", "abcd", None);
    }

    #[track_caller]
    fn assert_fixed_length_prefix(yes: bool, pattern: &str, suffix: &[u8]) {
        let hir = syntax::parse(pattern).unwrap();
        assert_eq!(
            yes,
            strip_literal_suffix(&hir, suffix)
                .map_or(false, |hir| prefix::hir_has_fixed_length(&hir)),
        );
    }

    #[track_caller]
    fn assert_internal_suffix(yes: bool, pattern: &str, suffix: &[u8]) {
        let hir = syntax::parse(pattern).unwrap();
        assert_eq!(
            yes,
            strip_literal_suffix(&hir, suffix).map_or(false, |hir| {
                prefix::hir_can_contain_literal(&hir, suffix)
            }),
        );
    }

    #[test]
    fn reverse_suffix_accepts_prefix_without_internal_suffix() {
        assert_internal_suffix(false, r"\d+XYZ", b"XYZ");
        assert_internal_suffix(false, r"[a-q][^u-z]{13}x", b"x");
    }

    #[test]
    fn reverse_suffix_accepts_fixed_length_prefix() {
        assert_fixed_length_prefix(true, r"(?:ab|cd)XYZ", b"XYZ");
        assert_internal_suffix(true, r"[A-Z][0-9]XYZ", b"XYZ");
        assert_fixed_length_prefix(true, r"[A-Z][0-9]XYZ", b"XYZ");
    }

    #[test]
    fn reverse_suffix_fixed_length_prefix_is_conservative() {
        assert_fixed_length_prefix(false, r"a{1,3}yy", b"yy");
        assert_fixed_length_prefix(false, r"a*yy", b"yy");
    }
}
