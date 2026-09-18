//! Actual retained builder copy, sharing expression/memo storage workers.
use super::{QuoteCache, RegexBuilder};
use crate::{
    ExprRef, RandomState,
    ast::{ExprSet, ExprSetCopyFailure, ExprSetCopyPlan},
};
use regex_syntax::ParserBuilder;
use std::{
    fmt,
    hash::BuildHasher,
    mem::{size_of, size_of_val},
};
/// Actual builder-copy destinations and fixed frames, without regex compilation.
#[derive(Clone, Copy, Debug)]
pub struct RegexBuilderCopyRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl RegexBuilderCopyRequirements {
    /// Independent buffer population, including the retained JSON memo table.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed source/copy frames and exact child frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local constructor requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Exact source loan; no name-based reconstruction or new source parsing.
pub struct RegexBuilderCopyPlan<'a> {
    source: &'a RegexBuilder,
    exprset: ExprSetCopyPlan<'a>,
    requirements: RegexBuilderCopyRequirements,
    table_bytes: usize,
}
impl fmt::Debug for RegexBuilderCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexBuilderCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    Expressions(ExprSetCopyFailure),
    Table(hashbrown::TryReserveError),
}
/// Owns actual expression and table destinations, including failed child prefixes.
pub struct RegexBuilderCopyFailure {
    cause: Cause,
    exprset: Option<ExprSet>,
    cache: Option<QuoteCache>,
}
impl fmt::Debug for RegexBuilderCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexBuilderCopyFailure")
            .field("cause", &self.cause)
            .field("retains_expression_set", &self.exprset.is_some())
            .field("retains_memo", &self.cache.is_some())
            .finish()
    }
}
impl fmt::Display for RegexBuilderCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("builder copy geometry overflow"),
            Cause::Capacity => f.write_str("builder copy destination differs from actual source"),
            Cause::Expressions(e) => fmt::Display::fmt(e, f),
            Cause::Table(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for RegexBuilderCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Expressions(e) => Some(e),
            Cause::Table(e) => Some(e),
            _ => None,
        }
    }
}
impl RegexBuilder {
    /// Existing builder/source inspection frames; no source traversal occurs.
    pub fn source_inspection_control_bytes() -> Option<usize> {
        RegexBuilderCopyPlan::wrapper_controls()?
            .checked_add(ExprSet::prepared_source_inspection_control_bytes()?)
    }
    /// Actual expression and memo capacities owned by this compiler instance.
    /// Parser configuration is inline; constructing a syntax parser is separate.
    pub fn retained_capacity_bytes(&self) -> Option<usize> {
        self.exprset.retained_capacity_bytes()?.checked_add(self.json_quote_cache.allocation_size())
    }

    /// Loans the exact compiled expression owner and parser configuration. The
    /// parser-builder configuration is inline (flags/limits), not a built parser.
    pub fn source_copy_plan(&self) -> Result<RegexBuilderCopyPlan<'_>, RegexBuilderCopyFailure> {
        RegexBuilderCopyPlan::prepare(self).map_err(|cause| RegexBuilderCopyFailure {
            cause,
            exprset: None,
            cache: None,
        })
    }
}
impl<'a> RegexBuilderCopyPlan<'a> {
    fn wrapper_controls() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<RegexBuilder>(),
            size_of::<RegexBuilderCopyRequirements>(),
            size_of::<RegexBuilderCopyFailure>(),
            size_of::<Cause>(),
            size_of::<Option<ExprSet>>(),
            size_of::<Option<QuoteCache>>(),
            size_of::<QuoteCache>(),
            size_of::<ParserBuilder>(),
            size_of::<RandomState>(),
            size_of::<<RandomState as BuildHasher>::Hasher>(),
            size_of::<hashbrown::hash_map::Iter<'_, ExprRef, ExprRef>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, RegexBuilderCopyFailure>>(),
            size_of::<Result<RegexBuilder, RegexBuilderCopyFailure>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
            size_of::<Option<ExprRef>>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn prepare(source: &'a RegexBuilder) -> Result<Self, Cause> {
        let exprset = source
            .exprset
            .source_copy_plan()
            .map_err(Cause::Expressions)?;
        let table_bytes = source.json_quote_cache.allocation_size();
        let buffers = exprset
            .requirements()
            .buffer_bytes()
            .checked_add(table_bytes)
            .ok_or(Cause::Overflow)?;
        let controls = Self::wrapper_controls()
            .and_then(|n| n.checked_add(exprset.requirements().control_bytes()))
            .ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            source,
            exprset,
            table_bytes,
            requirements: RegexBuilderCopyRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
    /// The actual complete local-copy population.
    pub fn requirements(&self) -> RegexBuilderCopyRequirements {
        self.requirements
    }
    /// Same constructor used by ordinary Clone. No syntax parser is built and no
    /// expression transformation runs during this copy.
    pub fn compile(self) -> Result<RegexBuilder, RegexBuilderCopyFailure> {
        let source = self.source;
        let exprset = self
            .exprset
            .compile()
            .map_err(|e| RegexBuilderCopyFailure {
                cause: Cause::Expressions(e),
                exprset: None,
                cache: None,
            })?;
        let mut cache = QuoteCache::with_hasher(source.json_quote_cache.hasher().clone());
        if let Err(e) = cache.try_reserve(source.json_quote_cache.capacity()) {
            return Err(RegexBuilderCopyFailure {
                cause: Cause::Table(e),
                exprset: Some(exprset),
                cache: Some(cache),
            });
        }
        if cache.capacity() != source.json_quote_cache.capacity()
            || cache.allocation_size() != self.table_bytes
        {
            return Err(RegexBuilderCopyFailure {
                cause: Cause::Capacity,
                exprset: Some(exprset),
                cache: Some(cache),
            });
        }
        for (&key, &value) in &source.json_quote_cache {
            if cache.len() == cache.capacity() {
                return Err(RegexBuilderCopyFailure {
                    cause: Cause::Capacity,
                    exprset: Some(exprset),
                    cache: Some(cache),
                });
            }
            if cache.insert(key, value).is_some() {
                return Err(RegexBuilderCopyFailure {
                    cause: Cause::Capacity,
                    exprset: Some(exprset),
                    cache: Some(cache),
                });
            }
        }
        Ok(RegexBuilder {
            parser_builder: source.parser_builder.clone(),
            exprset,
            json_quote_cache: cache,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JsonQuoteOptions, RegexAst};
    #[test]
    fn exact_builder_copy_preserves_unicode_json_configuration_and_failed_prefix() {
        let mut source = RegexBuilder::new(crate::ParserAllocationFunding::unenforced()).unwrap();
        source.case_insensitive(true).ignore_whitespace(true);
        let unicode = source.mk_regex("[α-ω]+ | [Ж-Я]+").unwrap();
        let literal = source.mk(&RegexAst::Literal("a\n\"".into())).unwrap();
        let options = JsonQuoteOptions::regular();
        let quoted = source.json_quote(literal, &options).unwrap();
        assert!(!source.exprset.unicode_cache.is_empty());
        assert!(!source.json_quote_cache.is_empty());
        let plan = source.source_copy_plan().unwrap();
        let quote = plan.requirements();
        let mut copy = plan.compile().unwrap();
        let mut ordinary = source.clone();
        assert!(quote.buffer_bytes() > source.json_quote_cache.allocation_size());
        assert!(quote.required_bytes() > quote.buffer_bytes());
        assert_eq!(
            copy.exprset.expr_to_string(unicode),
            source.exprset.expr_to_string(unicode)
        );
        assert_eq!(copy.exprset.unicode_cache, source.exprset.unicode_cache);
        assert_eq!(copy.json_quote_cache, source.json_quote_cache);
        assert_eq!(copy.json_quote(literal, &options).unwrap(), quoted);
        assert_eq!(
            copy.mk_regex(" a b ").unwrap(),
            source.mk_regex(" a b ").unwrap()
        );
        for (expression, text, expected) in [
            (unicode, "Αβ", true),
            (unicode, "ЖЯ", true),
            (unicode, "ab", false),
            (quoted, "\"a\\n\\\"\"", true),
        ] {
            assert_eq!(copy.to_regex(expression).unwrap().is_match(text).unwrap(), expected);
            assert_eq!(ordinary.to_regex(expression).unwrap().is_match(text).unwrap(), expected);
        }
        let mut fail = source.source_copy_plan().unwrap();
        fail.table_bytes = 0;
        let failure = match fail.compile() {
            Err(e) => e,
            Ok(_) => panic!("wrong table layout accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        assert!(failure.cache.as_ref().unwrap().allocation_size() > 0);
        drop(source);
        drop(copy);
        drop(ordinary);
        assert!(failure.exprset.as_ref().unwrap().unicode_cache.len() > 0);
        assert!(
            failure
                .exprset
                .as_ref()
                .unwrap()
                .expr_to_string(unicode)
                .len()
                > 0
        );
    }
}
