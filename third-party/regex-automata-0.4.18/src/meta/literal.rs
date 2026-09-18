use alloc::{vec, vec::Vec};

use regex_syntax::hir::Hir;

use crate::{meta::regex::RegexInfo, util::search::MatchKind};

/// Pull out an alternation of literals from the given sequence of HIR
/// expressions.
///
/// There are numerous ways for this to fail. Generally, this only applies
/// to regexes of the form 'foo|bar|baz|...|quux'. It can also fail if there
/// are "too few" alternates, in which case, the regex engine is likely faster.
///
/// And currently, this only returns something when 'hirs.len() == 1'.
pub(crate) fn alternation_literals_with_allocations(
    info: &RegexInfo,
    hirs: &[&Hir],
    funding: &dyn crate::util::allocation::Allocation,
) -> Result<Option<Vec<Vec<u8>>>, crate::util::allocation::AllocationError> {
    let allocation = crate::util::allocation::Allocator::new(funding);
    use regex_syntax::hir::{HirKind, Literal};

    // Might as well skip the work below if we know we can't build an
    // Aho-Corasick searcher.
    if !cfg!(feature = "perf-literal-multisubstring") {
        return Ok(None);
    }
    // This is pretty hacky, but basically, if `is_alternation_literal` is
    // true, then we can make several assumptions about the structure of our
    // HIR. This is what justifies the `unreachable!` statements below.
    if hirs.len() != 1
        || !info.props()[0].look_set().is_empty()
        || info.props()[0].explicit_captures_len() > 0
        || !info.props()[0].is_alternation_literal()
        || info.config().get_match_kind() != MatchKind::LeftmostFirst
    {
        return Ok(None);
    }
    let hir = &hirs[0];
    let alts = match *hir.kind() {
        HirKind::Alternation(ref alts) => alts,
        _ => return Ok(None), // one literal isn't worth it
    };

    let mut lits = vec![];
    for alt in alts {
        let mut lit = vec![];
        match *alt.kind() {
            HirKind::Literal(Literal(ref bytes)) => allocation.extend_copy(&mut lit, bytes)?,
            HirKind::Concat(ref exprs) => {
                for e in exprs {
                    match *e.kind() {
                        HirKind::Literal(Literal(ref bytes)) => {
                            allocation.extend_copy(&mut lit, bytes)?;
                        }
                        _ => unreachable!("expected literal, got {e:?}"),
                    }
                }
            }
            _ => unreachable!("expected literal or concat, got {alt:?}"),
        }
        allocation.push(&mut lits, lit)?;
    }
    // Why do this? Well, when the number of literals is small, it's likely
    // that we'll use the lazy DFA which is in turn likely to be faster than
    // Aho-Corasick in such cases. Primarily because Aho-Corasick doesn't have
    // a "lazy DFA" but either a contiguous NFA or a full DFA. We rarely use
    // the latter because it is so hungry (in time and space), and the former
    // is decently fast, but not as fast as a well oiled lazy DFA.
    //
    // However, once the number starts getting large, the lazy DFA is likely
    // to start thrashing because of the modest default cache size. When
    // exactly does this happen? Dunno. But at whatever point that is (we make
    // a guess below based on ad hoc benchmarking), we'll want to cut over to
    // Aho-Corasick, where even the contiguous NFA is likely to do much better.
    if lits.len() < 3000 {
        debug!("skipping Aho-Corasick because there are too few literals");
        return Ok(None);
    }
    Ok(Some(lits))
}
