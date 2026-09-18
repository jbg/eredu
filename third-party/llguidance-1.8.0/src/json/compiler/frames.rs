//! Fixed controls for the existing JSON-to-grammar emission worker.
//! A borrowed invocation scope retains recursive overlap; heap requests still
//! go directly to the original cumulative compiler account.
use super::*;
use derivre::prepared_funding::{Frame, FrameError, PreparedFunding, Scope};
use derivre::ParserAllocationFailure;
use std::mem::size_of;

#[cfg(test)]
mod tests;

pub(super) fn failure(error: FrameError<ParserAllocationFailure>, funding: &ParserAllocationFunding) -> ParserError {
    match error {
        FrameError::Overflow => funding.storage_overflow().into(),
        FrameError::Funding(error) => error.into(),
    }
}
pub(super) fn enter<'f, T>(scope: &'f Scope<'_, ParserAllocationFunding>,
    funding: &ParserAllocationFunding) -> Result<Frame<'f>> {
    // A previously refused heap request cannot be hidden by reusable capacity.
    if let Some(error) = funding.failure() { return Err(error.into()); }
    let bytes = size_of::<T>().checked_add(size_of::<(
        &Scope<'_, ParserAllocationFunding>, &ParserAllocationFunding,
        usize, Result<Frame<'_>>, Option<ParserAllocationFailure>,
    )>()).ok_or_else(|| funding.storage_overflow())?;
    scope.frame(bytes).map_err(|error| failure(error, funding))
}

pub(super) type Invocation<'a, 't> = (
    &'a JsonCompileOptions, GrammarBuilder<'t>, Value, bool,
    ParserAllocationFunding, Compiler<'t, 'a>, JsonCompileOptions,
    Result<JsonCompileOptions>, Result<Compiler<'t, 'a>>, Result<GrammarResult<'t>>,
);
pub(super) type Execute<'a, 't, 'f> = (
    Compiler<'t, 'f>, Value, RegexAst, LLGuidanceOptions, SkipSpec,
    super::super::shared_context::BuiltSchema, Result<super::super::shared_context::BuiltSchema>,
    std::vec::IntoIter<String>, String, NodeRef, &'a Schema,
    Option<(String, NodeRef)>, Result<NodeRef>, Result<GrammarResult<'t>>,
    // Missing-definition and contextual-error closures borrow the same path
    // beside the actual error being annotated.
    (&'a Compiler<'t, 'f>, &'a String),
    (&'a Compiler<'t, 'f>, &'a String), ParserError,
);
pub(super) type Alternatives<'a> = (
    &'a [Schema], Vec<RegexAst>, Vec<NodeRef>, Option<ParserError>,
    std::slice::Iter<'a, Schema>, &'a Schema, ParserError, RegexAst, NodeRef,
    Result<()>, Result<NodeRef>,
);
pub(super) type Integer<'a> = (
    &'a NumberSchema, Option<String>, Option<i64>, Option<i64>, Option<f64>,
    f64, String, RegexAst, &'a super::super::numeric::Decimal,
    [Result<RegexAst>; 2], std::array::IntoIter<Result<RegexAst>, 2>, Result<RegexAst>,
    (&'a Compiler<'a, 'a>, &'a Option<i64>, &'a Option<i64>),
    ParserError, UnsatisfiableSchemaError,
);
pub(super) type Number<'a> = (
    &'a NumberSchema, Option<String>, Option<f64>, bool, Option<f64>, bool,
    String, RegexAst, &'a super::super::numeric::Decimal,
    [Result<RegexAst>; 2], std::array::IntoIter<Result<RegexAst>, 2>, Result<RegexAst>,
    (&'a Compiler<'a, 'a>, &'a Option<f64>, &'a Option<f64>),
    ParserError, UnsatisfiableSchemaError,
);
pub(super) type Any = ( Option<NodeRef>, NodeRef, RegexAst, ExprRef, [NodeRef; 6], NodeRef,
    ArraySchema, ObjectSchema, Box<Schema>, Result<NodeRef>,
);
pub(super) type SequenceCache<'a> = HashMap<(&'a [(NodeRef, bool)], bool), NodeRef>;
pub(super) type Sequence<'a> = (
    &'a [(NodeRef, bool)], bool, &'a mut SequenceCache<'a>,
    Option<&'a NodeRef>, NodeRef, NodeRef, bool, &'a [(NodeRef, bool)],
    NodeRef, NodeRef, NodeRef, NodeRef, NodeRef, NodeRef, [NodeRef; 3], [NodeRef; 2],
    Result<NodeRef>,
);
pub(super) type Object<'a> = (
    (&'a ObjectSchema, Vec<String>, Vec<&'a str>, Vec<(NodeRef, bool)>, NodeRef),
    (usize, usize, usize, Option<usize>, &'a String, &'a Schema, bool, String),
    (indexmap::map::Keys<'a, String, Schema>, indexmap::set::Iter<'a, String>, &'a ObjectSchema),
    (NodeRef, NodeRef, NodeRef, [NodeRef; 3], Result<NodeRef>, ParserError, UnsatisfiableSchemaError),
    (Vec<Vec<(NodeRef, bool)>>, Vec<(NodeRef, bool)>, usize, usize, &'a bool, &'a NodeRef),
    (std::iter::Enumerate<std::slice::Iter<'a, (NodeRef, bool)>>,
        std::iter::Enumerate<std::slice::Iter<'a, (NodeRef, bool)>>,
        std::vec::IntoIter<(NodeRef, bool)>, std::slice::Iter<'a, Vec<(NodeRef, bool)>>,
        ParserAllocationFunding, Vec<NodeRef>, &'a mut Compiler<'a, 'a>),
    (Vec<ExprRef>, std::slice::Iter<'a, String>, Vec<NodeRef>,
        indexmap::map::Iter<'a, String, Schema>, &'a String, &'a Schema),
    (ExprRef, NodeRef, Vec<ExprRef>, std::iter::Enumerate<std::slice::Iter<'a, &'a str>>,
        usize, &'a &'a str, bool, ExprRef, ExprRef, [ExprRef; 2]),
    (NodeRef, NodeRef, ExprRef, ExprRef, RegexAst, ExprRef, ExprRef,
        NodeRef, bool, NodeRef, &'static str),
);
pub(super) type Array<'a> = (
    (&'a ArraySchema, Option<usize>, usize, Option<usize>, Option<NodeRef>),
    (Vec<NodeRef>, Vec<NodeRef>, usize, std::ops::Range<usize>, usize, NodeRef),
    (Vec<NodeRef>, NodeRef, std::slice::Iter<'a, NodeRef>, &'a NodeRef),
    (NodeRef, NodeRef, std::iter::Rev<std::iter::Skip<std::vec::IntoIter<NodeRef>>>,
        NodeRef, NodeRef, NodeRef, NodeRef, [NodeRef; 3], [NodeRef; 2]),
    (ParserError, UnsatisfiableSchemaError, Result<NodeRef>),
);
pub(super) type StringValue<'a> = (
    StringSchema, usize, Option<usize>, Option<RegexAst>, RegexAst, bool,
    &'a String, usize, std::str::Chars<'a>, [Result<RegexAst>; 2],
    std::array::IntoIter<Result<RegexAst>, 2>, derivre::RegexBuilder, ExprRef,
    derivre::Regex, Result<derivre::Regex>, Result<RegexAst>,
    (&'a Compiler<'a, 'a>, &'a RegexAst), ParserError, UnsatisfiableSchemaError,
);
pub(super) type Unicode<'a> = (
    usize, Option<usize>, (usize, Option<usize>), Option<&'a ExprRef>, ExprRef,
    &'a str, String, bool, std::str::Chars<'a>, char, String, ExprRef,
    u32, u32, Option<u32>, std::num::TryFromIntError, RegexAst, ExprRef,
    [Result<RegexAst>; 3], std::array::IntoIter<Result<RegexAst>, 3>,
    Result<RegexAst>,
    // The formatted escape suffix exists while both accumulated strings live.
    String, Result<String>,
    // Integer conversion's outer map and each reached map_err closure retain
    // the actual funding/length borrows until their diagnostic returns.
    (&'a Compiler<'a, 'a>, &'a usize), &'a Compiler<'a, 'a>,
    (&'a Compiler<'a, 'a>, &'a usize), usize, ParserError,
    Result<u32>, Result<Option<u32>>,
);
