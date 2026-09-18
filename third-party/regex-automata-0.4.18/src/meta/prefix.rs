/*!
Cheap HIR proofs that literal candidate order is compatible with regex match
order.

The reverse suffix and reverse inner strategies both scan for a required
literal before confirming the surrounding regex. These proofs are all
conservative. If they cannot establish that the first confirmed candidate is
the match the regex engine would report, then the corresponding reverse
strategy is not used.
*/

use crate::util::allocation::{
    Allocation, AllocationError, Allocator,
};
use alloc::vec::Vec;
use regex_syntax::hir::{literal::Literal, Class, Hir, HirKind};

/// Return true when `hir` can match some string containing every byte in
/// `lit`.
///
/// This is deliberately low precision. If every byte in a literal can be
/// consumed somewhere in the HIR, this gives up and reports that the
/// literal might occur internally. That is, this returning true does not
/// necessarily mean that the `Hir` provided definitively matches `lit`.
#[cfg(test)]
pub(super) fn hir_can_contain_literal(hir: &Hir, lit: &[u8]) -> bool {
    hir_can_contain_literal_with_allocations(hir, lit, &crate::util::allocation::Unenforced)
        .expect("ordinary HIR proof allocation")
}

pub(super) fn hir_can_contain_literal_with_allocations(
    hir: &Hir,
    lit: &[u8],
    funding: &dyn Allocation,
) -> Result<bool, AllocationError> {
    let allocation = Allocator::new(funding);
    let mut stack = Vec::new();
    for &byte in lit {
        stack.clear();
        allocation.push(&mut stack, hir)?;
        let mut consumes = false;
        while let Some(node) = stack.pop() {
            let yes = match node.kind() {
                HirKind::Empty | HirKind::Look(_) => false,
                HirKind::Literal(lit) => lit.0.contains(&byte),
                HirKind::Class(Class::Bytes(cls)) => {
                    byte_class_contains(cls, byte)
                }
                // Preserve the original conservative non-ASCII-byte proof.
                HirKind::Class(Class::Unicode(cls)) => {
                    byte > 0x7F
                        || unicode_class_contains(cls, char::from(byte))
                }
                _ => {
                    for sub in node.kind().subs().iter().rev() {
                        allocation.push(&mut stack, sub)?;
                    }
                    false
                }
            };
            if yes {
                consumes = true;
                break;
            }
        }
        if !consumes {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Returns true when the given `Hir` has a fixed length.
///
/// That is, when its minimum and maximum lengths are both finite and
/// equivalent.
pub(super) fn hir_has_fixed_length(hir: &Hir) -> bool {
    let props = hir.properties();
    props
        .minimum_len()
        .and_then(|min| props.maximum_len().map(|max| min == max))
        .unwrap_or(false)
}

/// Return true when an edge of `prefix` provides a required separator between
/// the start of a match and each literal candidate.
///
/// More precisely, this looks at the first and last consuming children of a
/// top-level concatenation. One of those children must be either a character
/// class or a repetition of a character class that matches at least once. Let
/// that child be `S`. This returns true when both of the following hold:
///
/// * `S` is disjoint from everything consumed by the other children in
///   `prefix`.
/// * `S` is disjoint from every literal in `literals`.
///
/// In that case, `S` acts as a separator whose characters cannot be mistaken
/// for characters belonging to either the rest of the prefix or a literal
/// candidate. A reverse search therefore cannot slide `S` across either one
/// and produce a later match start from an earlier literal candidate.
///
/// For example, consider this prefix and literal:
///
/// ```text
/// prefix  = \w+\s+
/// literal = Holmes
/// ```
///
/// The trailing `\s+` is required, is disjoint from `\w+` and cannot match
/// anything in `Holmes`. Thus this proves the reverse suffix optimization safe
/// for `\w+\s+Holmes`. It also works with multiple literals, such as `Holmes`
/// and `Watson`, provided the separator is disjoint from all of them.
///
/// The separator may instead be the first consuming child:
///
/// ```text
/// prefix  = \s[A-Za-z]{0,12}
/// literal = ing
/// ```
///
/// Here, the leading `\s` is disjoint from both `[A-Za-z]{0,12}` and `ing`.
///
/// This returns false when the possible separator is optional, overlaps
/// another prefix component or can match a character in any literal. For
/// example, neither `\s*` in `[A-Za-z]*\s*` nor `\w+` in `\w+\w+` provides
/// the required separator.
pub(super) fn has_disjoint_class_separator_with_allocations(
    prefix: &Hir,
    literals: &[Literal],
    funding: &dyn Allocation,
) -> Result<bool, AllocationError> {
    let hirs = match uncapture(prefix).kind() {
        HirKind::Concat(hirs) if hirs.len() >= 2 => hirs,
        _ => return Ok(false),
    };
    let Some(first) = hirs.iter().position(|hir| !hir_matches_empty_only(hir))
    else {
        return Ok(false);
    };
    let last = hirs
        .iter()
        .rposition(|hir| !hir_matches_empty_only(hir))
        .unwrap();
    if has_disjoint_class_separator_at(hirs, last, literals, funding)? {
        return Ok(true);
    }
    if first != last {
        return has_disjoint_class_separator_at(
            hirs, first, literals, funding,
        );
    }
    Ok(false)
}

fn has_disjoint_class_separator_at(
    hirs: &[Hir],
    separator: usize,
    literals: &[Literal],
    funding: &dyn Allocation,
) -> Result<bool, AllocationError> {
    let Some(separator_class) = required_class(&hirs[separator]) else {
        return Ok(false);
    };
    if !class_is_disjoint_from_literals(separator_class, literals) {
        return Ok(false);
    }
    let allocation = Allocator::new(funding);
    let mut stack = Vec::new();
    for (i, hir) in hirs.iter().enumerate() {
        if i == separator {
            continue;
        }
        allocation.push(&mut stack, hir)?;
        while let Some(node) = stack.pop() {
            let disjoint = match node.kind() {
                HirKind::Empty | HirKind::Look(_) => true,
                HirKind::Literal(lit) => {
                    class_is_disjoint_from_literal(separator_class, &lit.0)
                }
                HirKind::Class(cls) => {
                    classes_are_disjoint(cls, separator_class)
                }
                _ => {
                    for sub in node.kind().subs().iter().rev() {
                        allocation.push(&mut stack, sub)?;
                    }
                    true
                }
            };
            if !disjoint {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn hir_matches_empty_only(hir: &Hir) -> bool {
    hir.properties().maximum_len() == Some(0)
}

fn required_class(hir: &Hir) -> Option<&Class> {
    let hir = uncapture(hir);
    match hir.kind() {
        HirKind::Class(cls) => Some(cls),
        HirKind::Repetition(rep) if rep.min > 0 => {
            match uncapture(&rep.sub).kind() {
                HirKind::Class(cls) => Some(cls),
                _ => None,
            }
        }
        _ => None,
    }
}

fn uncapture(mut hir: &Hir) -> &Hir {
    while let HirKind::Capture(capture) = hir.kind() {
        hir = &capture.sub;
    }
    hir
}

fn classes_are_disjoint(left: &Class, right: &Class) -> bool {
    // Both canonical classes contain sorted, disjoint ranges. A merge walk
    // answers the same intersection-emptiness proof without copying a class.
    fn disjoint<T: Ord + Copy>(
        mut left: impl Iterator<Item = (T, T)>,
        mut right: impl Iterator<Item = (T, T)>,
    ) -> bool {
        let (mut l, mut r) = (left.next(), right.next());
        while let (Some((ls, le)), Some((rs, re))) = (l, r) {
            if le < rs {
                l = left.next();
            } else if re < ls {
                r = right.next();
            } else {
                return false;
            }
        }
        true
    }
    match (left, right) {
        (Class::Bytes(l), Class::Bytes(r)) => disjoint(
            l.ranges().iter().map(|x| (x.start(), x.end())),
            r.ranges().iter().map(|x| (x.start(), x.end())),
        ),
        (Class::Unicode(l), Class::Unicode(r)) => disjoint(
            l.ranges().iter().map(|x| (x.start(), x.end())),
            r.ranges().iter().map(|x| (x.start(), x.end())),
        ),
        _ => false,
    }
}

fn class_is_disjoint_from_literals(cls: &Class, literals: &[Literal]) -> bool {
    literals
        .iter()
        .all(|lit| class_is_disjoint_from_literal(cls, lit.as_bytes()))
}

fn class_is_disjoint_from_literal(cls: &Class, lit: &[u8]) -> bool {
    match cls {
        Class::Bytes(cls) => {
            lit.iter().all(|&byte| !byte_class_contains(cls, byte))
        }
        Class::Unicode(cls) => core::str::from_utf8(lit)
            .map_or(false, |lit| {
                lit.chars().all(|ch| !unicode_class_contains(cls, ch))
            }),
    }
}

fn byte_class_contains(cls: &regex_syntax::hir::ClassBytes, byte: u8) -> bool {
    cls.ranges()
        .iter()
        .any(|range| range.start() <= byte && byte <= range.end())
}

fn unicode_class_contains(
    cls: &regex_syntax::hir::ClassUnicode,
    ch: char,
) -> bool {
    cls.ranges()
        .iter()
        .any(|range| range.start() <= ch && ch <= range.end())
}
