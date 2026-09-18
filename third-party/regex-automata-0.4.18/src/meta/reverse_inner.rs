/*!
A module dedicated to plucking inner literals out of a regex pattern, and
then constructing a prefilter for them. We also include a regex pattern
"prefix" that corresponds to the bits of the regex that need to match before
the literals do. The reverse inner optimization then proceeds by looking for
matches of the inner literal(s), and then doing a reverse search of the prefix
from the start of the literal match to find the overall start position of the
match.

The essential invariant we want to uphold here is that the literals we return
reflect a set where *at least* one of them must match in order for the overall
regex to match. We also need to maintain the invariant that the regex prefix
returned corresponds to the entirety of the regex up until the literals we
return.

This somewhat limits what we can do. That is, if we a regex like
`\w+(@!|%%)\w+`, then we can pluck the `{@!, %%}` out and build a prefilter
from it. Then we just need to compile `\w+` in reverse. No fuss no muss. But if
we have a regex like \d+@!|\w+%%`, then we get kind of stymied. Technically,
we could still extract `{@!, %%}`, and it is true that at least of them must
match. But then, what is our regex prefix? Again, in theory, that could be
`\d+|\w+`, but that's not quite right, because the `\d+` only matches when `@!`
matches, and `\w+` only matches when `%%` matches.

All of that is technically possible to do, but it seemingly requires a lot of
sophistication and machinery. Probably the way to tackle that is with some kind
of formalism and approach this problem more generally.

For now, the code below basically just looks for a top-level concatenation.
And if it can find one, it looks for literals in each of the direct child
sub-expressions of that concatenation. If some good ones are found, we return
those and a concatenation of the Hir expressions seen up to that point.
*/

use alloc::vec::Vec;

use regex_syntax::hir::{
    self,
    literal::{self, Literal},
    Hir, HirKind,
};

use crate::{
    meta::prefix,
    util::{
        allocation::{Allocation, AllocationError, Allocator},
        prefilter::Prefilter,
    },
    MatchKind,
};

/// Returns true when it's impossible for an earlier match to be detected after
/// a literal candidate (corresponding to anything in `literals`) has
/// been found.
///
/// Specifically, that there is no earlier match than what a reverse scan of
/// `concat_prefix` after a match of `literals` reports.
///
/// Since this requires a single `Hir`, this implies the reverse inner optimization
/// only works with a single regex.
pub(super) fn has_no_earlier_match_with_allocations(
    concat_prefix: &Hir,
    literals: &[Literal],
    funding: &dyn Allocation,
) -> Result<bool, AllocationError> {
    // let literals = prefix::LiteralSet::many(literals);
    if literals.is_empty() || literals.iter().any(|lit| lit.is_empty()) {
        debug!(
            "reverse inner is not early return safe because \
                 no non-empty inner literals were found"
        );
        return Ok(false);
    }
    // With one literal, an occurrence crossing the prefix boundary must
    // overlap another occurrence of that same literal. Such an overlap
    // requires the prefix to consume every distinct byte in the literal.
    // This reasoning does not apply when one extracted literal can cross
    // the boundary into a different extracted literal.
    if literals.len() == 1 {
        let prefix_may_contain =
            prefix::hir_can_contain_literal_with_allocations(
                concat_prefix,
                literals[0].as_bytes(),
                funding,
            )?;
        debug!(
            "reverse inner prefix can contain inner literals? \
             {prefix_may_contain}"
        );
        if !prefix_may_contain {
            return Ok(true);
        }
    }

    let fixed_length = prefix::hir_has_fixed_length(concat_prefix);
    debug!("reverse inner has fixed length prefix? {fixed_length}");
    if fixed_length {
        return Ok(true);
    }

    let class_separator =
        prefix::has_disjoint_class_separator_with_allocations(
            concat_prefix,
            &literals,
            funding,
        )?;
    debug!("reverse inner has disjoint class separator? {class_separator}");
    if class_separator {
        return Ok(true);
    }

    // We couldn't prove that the reverse inner optimization
    // was safe, so bail out.
    Ok(false)
}

/// This attempts to extract an "inner" prefilter from the given HIR
/// expressions. If one was found, then a concatenation of the HIR expressions
/// that precede it is returned.
///
/// The idea here is that the prefilter returned can be used to find
/// candidate matches. And then the HIR returned can be used to build a
/// reverse regex matcher, which will find the start of the candidate
/// match. Finally, the match still has to be confirmed with a normal
/// anchored forward scan to find the end position of the match.
///
/// Note that this assumes leftmost-first match semantics, so callers must
/// not call this otherwise.
#[derive(Debug)]
pub(crate) struct InnerPrefilter {
    pub(crate) prefix: Hir,
    /// The prefilter generated from `literals`.
    pub(crate) pre: Prefilter,
    /// The actual literals extracted and used to build `pre`.
    ///
    /// These are used by the meta strategy to prove that the inner prefilter
    /// can return after the first confirmed candidate. If that proof fails, we
    /// could try extracting a different set of literals. But we don't
    /// currently do that.
    pub(crate) literals: Vec<Literal>,
}

impl InnerPrefilter {
    pub(crate) fn new_with_allocations(
        hirs: &[&Hir],
        funding: &dyn Allocation,
    ) -> Result<Option<InnerPrefilter>, AllocationError> {
        if hirs.len() != 1 {
            debug!(
                "skipping reverse inner optimization since it only \
                 supports 1 pattern, {} were given",
                hirs.len(),
            );
            return Ok(None);
        }
        let mut concat = match top_concat_with_allocations(hirs[0], funding)? {
            Some(concat) => concat,
            None => {
                debug!(
                    "skipping reverse inner optimization because a top-level \
                     concatenation could not found",
                );
                return Ok(None);
            }
        };
        // We skip the first HIR because if it did have a prefix prefilter in
        // it, we probably wouldn't be here looking for an inner prefilter.
        for i in 1..concat.len() {
            let hir = &concat[i];
            let (pre, lits) = match prefilter_with_literals_with_allocations(
                hir, funding,
            )? {
                None => continue,
                Some(pre) => pre,
            };
            // Even if we got a prefilter, if it isn't consider "fast," then
            // we probably don't want to bother with it. Namely, since the
            // reverse inner optimization requires some overhead, it likely
            // only makes sense if the prefilter scan itself is (believed) to
            // be much faster than the regex engine.
            if !pre.is_fast() {
                debug!(
                    "skipping extracted inner prefilter because \
                     it probably isn't fast"
                );
                continue;
            }
            let allocation = Allocator::new(funding);
            let syntax = regex_syntax::allocation::Allocator::new(&allocation);
            let mut suffix = Vec::new();
            allocation.grow(&mut suffix, concat.len() - i)?;
            suffix.extend(concat.drain(i..));
            let concat_suffix = Hir::concat_with_allocations(suffix, syntax)?;
            let concat_prefix = Hir::concat_with_allocations(concat, syntax)?;
            // Look for a prefilter again. Why? Because above we only looked
            // for a prefilter on the individual 'hir', but we might be able
            // to find something better and more discriminatory by looking at
            // the entire suffix. We don't do this above to avoid making this
            // loop worst case quadratic in the length of 'concat'.
            let (preinner, inner_literals) =
                match prefilter_with_literals_with_allocations(
                    &concat_suffix,
                    funding,
                )? {
                    None => (pre, lits),
                    Some((pre2, lits2)) => {
                        if pre2.is_fast() {
                            (pre2, lits2)
                        } else {
                            (pre, lits)
                        }
                    }
                };
            return Ok(Some(InnerPrefilter {
                prefix: concat_prefix,
                pre: preinner,
                literals: inner_literals,
            }));
        }
        debug!(
            "skipping reverse inner optimization because a top-level \
             sub-expression with a fast prefilter could not be found"
        );
        Ok(None)
    }
}

/// Attempt to extract a prefilter from an HIR expression.
///
/// We do a little massaging here to do our best that the prefilter we get out
/// of this is *probably* fast. Basically, the false positive rate has a much
/// higher impact for things like the reverse inner optimization because more
/// work needs to potentially be done for each candidate match.
///
/// Note that this assumes leftmost-first match semantics, so callers must
/// not call this otherwise.
fn prefilter_with_literals_with_allocations(
    hir: &Hir,
    funding: &dyn Allocation,
) -> Result<Option<(Prefilter, Vec<Literal>)>, AllocationError> {
    let allocation = Allocator::new(funding);
    let syntax = regex_syntax::allocation::Allocator::new(&allocation);
    let mut extractor = literal::Extractor::new();
    extractor.kind(literal::ExtractKind::Prefix);
    let mut prefixes = extractor.extract_with_allocations(hir, syntax)?;
    prefixes.make_inexact();
    prefixes.optimize_for_prefix_by_preference_with_allocations(syntax)?;
    let Some(lits) = prefixes.literals() else {
        return Ok(None);
    };
    let Some(pre) = Prefilter::new_with_allocations(
        MatchKind::LeftmostFirst,
        lits,
        funding,
    )?
    else {
        return Ok(None);
    };
    let mut copied = Vec::new();
    allocation.grow(&mut copied, lits.len())?;
    for lit in lits {
        copied.push(lit.clone_with_allocations(syntax)?);
    }
    Ok(Some((pre, copied)))
}

/// Looks for a "top level" HirKind::Concat item in the given HIR. This will
/// try to return one even if it's embedded in a capturing group, but is
/// otherwise pretty conservative in what is returned.
///
/// The HIR returned is a complete copy of the concat with all capturing
/// groups removed. In effect, the concat returned is "flattened" with respect
/// to capturing groups. This makes the detection logic above for prefixes
/// a bit simpler, and it works because 1) capturing groups never influence
/// whether a match occurs or not and 2) capturing groups are not used when
/// doing the reverse inner search to find the start of the match.
fn top_concat_with_allocations(
    mut hir: &Hir,
    funding: &dyn Allocation,
) -> Result<Option<Vec<Hir>>, AllocationError> {
    let allocation = Allocator::new(funding);
    let syntax = regex_syntax::allocation::Allocator::new(&allocation);
    loop {
        hir = match hir.kind() {
            HirKind::Capture(capture) => &capture.sub,
            HirKind::Concat(subs) => {
                let mut copied = Vec::new();
                allocation.grow(&mut copied, subs.len())?;
                for sub in subs {
                    copied.push(flatten_with_allocations(sub, funding)?);
                }
                return Ok(
                    match Hir::concat_with_allocations(copied, syntax)?
                        .into_kind()
                    {
                        HirKind::Concat(children) => Some(children),
                        _ => None,
                    },
                );
            }
            _ => return Ok(None),
        };
    }
}

/// Rebuild the same capture-free HIR through its canonical smart constructors,
/// using paid postorder controls instead of recursive Rust frames.
fn flatten_with_allocations(
    hir: &Hir,
    funding: &dyn Allocation,
) -> Result<Hir, AllocationError> {
    enum Frame<'a> {
        Visit(&'a Hir),
        Finish(&'a Hir),
    }
    let allocation = Allocator::new(funding);
    let syntax = regex_syntax::allocation::Allocator::new(&allocation);
    let mut stack = Vec::new();
    let mut values = Vec::new();
    allocation.push(&mut stack, Frame::Visit(hir))?;
    while let Some(frame) = stack.pop() {
        let source = match frame {
            Frame::Visit(mut source) => {
                while let HirKind::Capture(capture) = source.kind() {
                    source = &capture.sub;
                }
                allocation.push(&mut stack, Frame::Finish(source))?;
                for sub in source.kind().subs().iter().rev() {
                    allocation.push(&mut stack, Frame::Visit(sub))?;
                }
                continue;
            }
            Frame::Finish(source) => source,
        };
        let copied = match source.kind() {
            HirKind::Empty => Hir::empty_with_allocations(syntax)?,
            HirKind::Literal(literal) => Hir::literal_with_allocations(
                syntax.copy_slice(&literal.0)?,
                syntax,
            )?,
            HirKind::Class(class) => Hir::class_with_allocations(
                class.clone_with_allocations(syntax)?,
                syntax,
            )?,
            HirKind::Look(look) => Hir::look_with_allocations(*look, syntax)?,
            HirKind::Capture(_) => {
                unreachable!("capture stripped before postorder visit")
            }
            HirKind::Repetition(rep) => Hir::repetition_with_allocations(
                hir::Repetition {
                    min: rep.min,
                    max: rep.max,
                    greedy: rep.greedy,
                    sub: syntax.boxed(
                        values.pop().expect("postorder repetition child"),
                    )?,
                },
                syntax,
            )?,
            HirKind::Concat(children) | HirKind::Alternation(children) => {
                let start = values.len() - children.len();
                let mut subs = Vec::new();
                allocation.grow(&mut subs, children.len())?;
                subs.extend(values.drain(start..));
                if matches!(source.kind(), HirKind::Concat(_)) {
                    Hir::concat_with_allocations(subs, syntax)?
                } else {
                    Hir::alternation_with_allocations(subs, syntax)?
                }
            }
        };
        allocation.push(&mut values, copied)?;
    }
    Ok(values.pop().expect("one transformed root"))
}
