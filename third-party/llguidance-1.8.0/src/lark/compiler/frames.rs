//! Concrete reached emitter frames; recursive children borrow one live scope.
use super::*;
use derivre::prepared_funding::{Frame, FrameError, PreparedFunding, Scope};
use std::{mem::size_of, ops::RangeInclusive};
#[cfg(test)]
mod tests;

pub(super) type FundingScope<'a> = Scope<'a, ParserAllocationFunding>;
pub(super) fn failure(error: FrameError<derivre::ParserAllocationFailure>, funding: &ParserAllocationFunding) -> ParserError {
    match error { FrameError::Overflow => funding.storage_overflow().into(), FrameError::Funding(error) => error.into() }
}
pub(super) fn enter<'a, T>(scope: &'a FundingScope<'_>, funding: &ParserAllocationFunding) -> Result<Frame<'a>> {
    if let Some(error) = funding.failure() { return Err(error.into()); }
    let bytes = size_of::<T>().checked_add(size_of::<(
        &FundingScope<'_>, &ParserAllocationFunding, usize, Result<Frame<'_>>,
    )>()).ok_or_else(|| funding.storage_overflow())?;
    scope.frame(bytes).map_err(|error| failure(error, funding))
}

pub(super) type Public<'a, 't> = (
    GrammarBuilder<'t>, &'a str, ParserAllocationFunding, ParsedLark,
    GrammarResult<'t>, Result<GrammarResult<'t>>, Result<ParsedLark>,
);
pub(super) type Compile<'a, 't, 'f> = (
    GrammarBuilder<'t>, ParsedLark, &'a FundingScope<'f>, Compiler<'t, 'f>,
    GrammarResult<'t>, Result<GrammarResult<'t>>,
);
pub(super) type Token<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a str, TokenDef, RegexId, Option<&'a RegexId>,
    Result<RegexId>, (&'a Compiler<'t, 'f>, &'a str),
);
pub(super) type Regex<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a str, String, Result<RegexId>,
    (&'a Compiler<'t, 'f>, &'a str, &'a String),
);
pub(super) type TokenAtom<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, Atom, Expansions, Box<Atom>, Value,
    (String, String, char, char, RegexId), Result<RegexId>,
    std::str::Chars<'a>, Escaped<'a>, EscapedChar,
);
pub(super) type TokenExpr<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, Expr, RegexId, &'a (i32, i32), &'a Op,
    Option<u32>, Result<RegexId>,
);
pub(super) type TokenExpansions<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, Expansions, Vec<RegexId>, Vec<RegexId>, Vec<RegexId>,
    Alias, Expansion, Expr, RegexId, Result<RegexId>,
    std::vec::IntoIter<Alias>, std::vec::IntoIter<Expansion>, std::vec::IntoIter<Expr>,
    &'a Location,
);
pub(super) type Lift<'a, 't, 'f> = (&'a mut Compiler<'t, 'f>, RegexId, Result<NodeRef>);
pub(super) type Nested<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a Location, Value, Option<f32>, NodeProps,
    PendingGrammar, String, GenGrammarOptions, NodeRef, Result<NodeRef>,
    (NodeRef, Location, PendingGrammar),
);
pub(super) type AtomFrame<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a Location, Atom, Expansions, Value,
    RegexId, NodeRef, Result<NodeRef>, Option<ParamExpr>, Option<f32>, NodeProps,
    &'a str, &'a str, &'a str, bool, Vec<RangeInclusive<u32>>, RangeInclusive<u32>,
    std::str::Split<'a, &'a str>, std::str::Split<'a, char>,
    std::iter::Map<std::str::Split<'a, char>, fn(&str)->&str>,
    Option<&'a str>, u32, u32, Result<u32, std::num::ParseIntError>,
    &'a Compiler<'t, 'f>,
);
pub(super) type ExprFrame<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a Location, Expr, NodeRef, i32, i32, Option<usize>,
    &'a Op, Result<NodeRef>,
);
pub(super) type ExpansionsFrame<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, Expansions, Location, Vec<ParamCond>, Vec<NodeRef>, Vec<NodeRef>,
    bool, Alias, Expr, NodeRef, NodeProps, Result<NodeRef>, &'a Location,
    std::slice::Iter<'a, Alias>, std::vec::IntoIter<Alias>, std::vec::IntoIter<Expr>,
);
pub(super) type Rule<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a str, Option<ParamExpr>, Option<&'a NodeRef>,
    NodeRef, bool, Result<NodeRef>, String,
);
pub(super) type GenGrammar<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a str, Option<f32>, NodeProps, String,
    GenGrammarOptions, NodeRef, Result<NodeRef>, Result<usize, std::num::ParseIntError>,
);
pub(super) type RuleCore<'a, 't, 'f> = (
    &'a mut Compiler<'t, 'f>, &'a str, super::Rule, NodeProps, bool, NodeRef,
    (&'a Compiler<'t, 'f>, &'a str), &'a Compiler<'t, 'f>,
    bool, bool, bool, Atom, RegexId, RegexId, GenOptions,
    Option<&'a Atom>, Option<Atom>, Value, usize, [NodeRef; 1], Result<NodeRef>,
);
pub(super) type Execute<'a, 't, 'f> = (
    Compiler<'t, 'f>, Grammar, Item, Location, &'a str, Vec<Expansions>, LarkLLGuidanceOptions,
    Vec<RegexAst>, RegexAst, Expansions, SkipSpec, NodeRef, crate::earley::SymIdx,
    GrammarBuilder<'t>, PendingGrammar, GrammarResult<'t>, Result<GrammarResult<'t>>,
    JsonCompileOptions, ParsedLark, &'a Location,
    std::vec::IntoIter<Item>, std::vec::IntoIter<Expansions>,
    std::vec::IntoIter<(NodeRef, Location, PendingGrammar)>,
);
pub(super) type ApplyOptions<'a> = (
    &'a mut LarkLLGuidanceOptions, serde_json::Value, &'a ParserAllocationFunding,
    serde_json::Map<String, serde_json::Value>, serde_json::map::IntoIter,
    String, serde_json::Value, &'a mut bool, bool, Result<()>,
);
pub(super) type AddToken<'a> = (
    &'a mut Grammar, &'a Location, String, &'a str, &'a ParserAllocationFunding,
    Vec<Expr>, Vec<Expansion>, Vec<Alias>, Expr, Atom, Value, Expansion, Alias,
    TokenDef, Expansions, Location, Result<()>,
);
pub(super) type StatementFrame<'a> = (
    &'a mut Grammar, &'a Location, Statement, &'a ParserAllocationFunding,
    Expansions, String, Option<String>, Vec<String>, String, &'a str,
    std::vec::IntoIter<String>, std::str::Split<'a, char>, Result<()>,
);
pub(super) type ItemFrame<'a> = (
    &'a mut Grammar, Item, &'a ParserAllocationFunding, super::Rule, TokenDef,
    Location, Statement, Result<()>,
);
pub(super) type ExtendedRegex<'a, 't> = (
    &'a mut GrammarBuilder<'t>, RegexExt, [&'static str; 3], usize,
    &'a mut derivre::RegexBuilder, RegexId, String, Vec<String>, Result<RegexId>,
    std::slice::Iter<'a, String>,
);
