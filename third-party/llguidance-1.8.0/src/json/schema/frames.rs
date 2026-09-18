//! Reached schema frames share the compiler's one private high-water scope.
//! Heap producers still use the unchanged cumulative allocation account.
use super::*;
use derivre::prepared_funding::{Frame, FrameError, PreparedFunding};
use derivre::{ParserAllocationFailure, ParserAllocationFunding};
use std::mem::size_of;
#[cfg(test)]
mod tests;

pub(in crate::json) fn failure(
    error: FrameError<ParserAllocationFailure>,
    funding: &ParserAllocationFunding,
) -> ParserError {
    match error {
        FrameError::Overflow => funding.storage_overflow().into(),
        FrameError::Funding(error) => error.into(),
    }
}

/// `T` names this reached function's inputs, locals, iterators and result.
/// The guard is retained by the caller until its last recursive child returns.
pub(super) fn enter<'a, T>(ctx: &'a Context<'_>) -> Result<Frame<'a>> {
    if let Some(error) = ctx.funding.failure() {
        return Err(error.into());
    }
    let bytes = size_of::<T>()
        .checked_add(size_of::<(&Context, usize, Result<Frame<'_>>)>())
        .ok_or_else(|| ctx.funding.storage_overflow())?;
    ctx.frames
        .frame(bytes)
        .map_err(|error| failure(error, &ctx.funding))
}

// Each branch owns its concrete schema values only while that branch runs;
// do not multiply the largest frame by a configured maximum depth.
pub(super) type Intersection<'a> = (
    (Schema, Schema, &'a Context<'a>, usize),
    (Schema, Schema, Result<Schema>),
    (NumberSchema, NumberSchema, StringSchema, StringSchema),
    (
        Option<Decimal>,
        Option<Decimal>,
        Option<RegexAst>,
        Option<RegexAst>,
    ),
    (std::vec::IntoIter<Schema>, Vec<Schema>, Schema, &'a Schema),
    // The collecting map retains these captures beside the dispatch inputs.
    (&'a Context<'a>, usize),
    (Option<bool>, Option<bool>, bool),
);
pub(super) type Arrays<'a> = (
    (ArraySchema, ArraySchema, &'a Context<'a>, usize),
    (ArraySchema, Result<Schema>, usize, usize, Option<usize>),
    (
        std::iter::Zip<std::vec::IntoIter<Schema>, std::vec::IntoIter<Schema>>,
        Vec<Schema>,
    ),
    (Schema, Schema, Option<Box<Schema>>, Option<Box<Schema>>),
    (&'a Context<'a>, usize),
);
pub(super) type Objects<'a> = (
    (ObjectSchema, ObjectSchema, &'a Context<'a>, usize),
    (ObjectSchema, Result<Schema>, usize, Option<usize>),
    (
        IndexMap<String, Schema>,
        IndexMap<String, Schema>,
        IndexSet<String>,
    ),
    (
        indexmap::map::IntoIter<String, Schema>,
        indexmap::set::IntoIter<String>,
    ),
    (
        String,
        Schema,
        &'a Schema,
        Option<Box<Schema>>,
        Option<Box<Schema>>,
    ),
);
pub(super) type Normalize<'a> = (
    (Schema, &'a Context<'a>),
    (Vec<Schema>, Vec<Schema>, Schema, Result<Schema>),
    (
        bool,
        Option<&'static str>,
        std::vec::IntoIter<Schema>,
        std::vec::IntoIter<Schema>,
    ),
    (
        std::iter::Enumerate<std::slice::Iter<'a, Schema>>,
        std::iter::Skip<std::slice::Iter<'a, Schema>>,
    ),
    (usize, &'a Schema, &'a Schema),
);
pub(super) type Disjoint<'a> = (
    (&'a Schema, &'a Schema, &'a Context<'a>, Result<bool>),
    (
        std::slice::Iter<'a, Schema>,
        indexmap::set::Union<'a, String, std::collections::hash_map::RandomState>,
    ),
    (
        &'a Schema,
        &'a Schema,
        &'a String,
        &'a ObjectSchema,
        Result<&'a Schema>,
    ),
    (&'a Context<'a>, &'a String),
);
pub(super) type Patterns<'a> = (
    (
        IndexMap<String, Schema>,
        IndexMap<String, Schema>,
        &'a Option<Box<Schema>>,
        &'a Option<Box<Schema>>,
        &'a Context<'a>,
        usize,
    ),
    (
        IndexMap<String, Schema>,
        IndexMap<String, Schema>,
        Result<IndexMap<String, Schema>>,
    ),
    (
        indexmap::map::IntoIter<String, Schema>,
        indexmap::map::IntoIter<String, Schema>,
    ),
    (
        String,
        Schema,
        Option<Schema>,
        &'a Schema,
        Vec<&'a String>,
        indexmap::map::Keys<'a, String, Schema>,
    ),
);
pub(super) type Reference<'a> = (
    (&'a Context<'a>, &'a str, Schema, bool, usize),
    (Schema, Option<Schema>, Result<Schema>),
);
pub(super) type Apply<'a> = (
    (Schema, (&'a str, &'a Value), &'a Context<'a>),
    (Schema, Schema, Result<Schema>, String, Vec<Schema>),
    (&'a [Value], std::slice::Iter<'a, Value>, &'a Value, &'a str),
    &'a Context<'a>,
);
pub(super) type Contents<'a> = (
    (&'a Context<'a>, &'a Value, Result<Schema>, bool),
    (
        &'a serde_json::Map<String, Value>,
        IndexMap<&'a str, &'a Value>,
        serde_json::map::Iter<'a>,
    ),
    (&'a String, &'a Value),
);
pub(super) type ContentsMap<'a> = (
    (
        &'a Context<'a>,
        IndexMap<&'a str, &'a Value>,
        Result<Schema>,
    ),
    (
        Vec<&'a &'a str>,
        String,
        Option<Value>,
        Option<Value>,
        Value,
    ),
    (
        serde_json::Map<String, Value>,
        Vec<Value>,
        serde_json::map::Keys<'a>,
        std::slice::Iter<'a, Value>,
    ),
    (Schema, Schema, HashMap<&'a str, &'a Value>, [&'a str; 6]),
    (
        indexmap::map::Keys<'a, &'a str, &'a Value>,
        indexmap::map::Iter<'a, &'a str, &'a Value>,
    ),
    (&'a str, &'a Value, &'a Value),
);
pub(super) type SimpleContents<'a> = (
    (&'a Context<'a>, HashMap<&'a str, &'a Value>, Result<Schema>),
    (
        Option<&'a &'a Value>,
        Vec<&'a str>,
        std::slice::Iter<'a, Value>,
        &'a Value,
    ),
    &'a Context<'a>,
);
pub(super) type Constant<'a> = (
    (&'a Context<'a>, &'a Value, Result<Schema>),
    (f64, NumberSchema, StringSchema, ArraySchema, ObjectSchema),
    (
        Vec<Schema>,
        IndexMap<String, Schema>,
        IndexSet<String>,
        Schema,
        String,
    ),
    (
        std::slice::Iter<'a, Value>,
        serde_json::map::Iter<'a>,
        &'a String,
        &'a Value,
    ),
    &'a Context<'a>,
);
pub(super) type StringValue<'a> = (
    (
        &'a Context<'a>,
        &'a HashMap<&'a str, &'a Value>,
        Result<Schema>,
    ),
    (Option<&'a Value>, Option<&'a Value>, usize, Option<usize>),
    (
        Option<RegexAst>,
        Option<RegexAst>,
        Option<RegexAst>,
        RegexAst,
        RegexAst,
    ),
    (
        std::array::IntoIter<Result<RegexAst>, 2>,
        Vec<RegexAst>,
        StringSchema,
        String,
        &'a str,
    ),
);
pub(super) type ArrayValue<'a> = (
    (
        &'a Context<'a>,
        &'a HashMap<&'a str, &'a Value>,
        Result<Schema>,
    ),
    (usize, Option<usize>, [Option<&'a Value>; 5]),
    (
        Vec<Schema>,
        Option<Box<Schema>>,
        ArraySchema,
        Schema,
        std::slice::Iter<'a, Value>,
        &'a Value,
    ),
    &'a Context<'a>,
);
pub(super) type PropertyMap<'a> = (
    (
        &'a Context<'a>,
        &'a str,
        Option<&'a Value>,
        Result<IndexMap<String, Schema>>,
    ),
    (
        IndexMap<String, Schema>,
        serde_json::map::Iter<'a>,
        &'a String,
        &'a Value,
        Schema,
        String,
    ),
);
pub(super) type ObjectValue<'a> = (
    (
        &'a Context<'a>,
        &'a HashMap<&'a str, &'a Value>,
        Result<Schema>,
    ),
    ([Option<&'a Value>; 4], usize, Option<usize>, ObjectSchema),
    (
        IndexMap<String, Schema>,
        IndexMap<String, Schema>,
        Vec<&'a String>,
        IndexSet<String>,
    ),
    (
        indexmap::map::Keys<'a, String, Schema>,
        indexmap::map::IterMut<'a, String, Schema>,
        indexmap::map::Iter<'a, String, Schema>,
    ),
    (
        &'a String,
        &'a Schema,
        &'a mut Schema,
        Schema,
        Option<Box<Schema>>,
    ),
    (std::slice::Iter<'a, Value>, &'a Value, &'a str, String),
);
