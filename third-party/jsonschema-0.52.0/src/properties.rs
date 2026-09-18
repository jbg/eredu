use std::{borrow::Cow, sync::Arc};

use crate::{
    compiler,
    node::SchemaNode,
    paths::{LazyEvaluationPath, Location},
    regex::{analyze_pattern, is_ecma_whitespace, LiteralMatchError, PatternOptimization},
    validator::Validate as _,
    Json, Object, SerdeJson, ValidationContext,
};

use serde_json::{Map, Value};

use crate::ValidationError;

/// A compiled pattern that can be a literal optimized match or a full regex.
#[derive(Debug, Clone)]
pub(crate) enum CompiledPattern<R> {
    /// Simple prefix match using `starts_with()`.
    Prefix(Arc<str>),
    /// Exact match using `==` - for `^...$` patterns.
    Exact(Arc<str>),
    /// `^(a|b|c)$` — linear scan over a small sorted array of alternatives.
    Alternation(Arc<[String]>),
    /// `^\S*$` — no ECMA-262 whitespace characters.
    NoWhitespace,
    /// Full regex pattern.
    Regex(R),
}

impl<R: crate::regex::RegexEngine> crate::regex::RegexEngine for CompiledPattern<R> {
    type Error = LiteralMatchError;
    fn original_source<F: Json>(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        match self {
            Self::Prefix(text) | Self::Exact(text) => source.arc(text).map(|_| ()),
            Self::Alternation(values) => {
                if source.arc(values)? {
                    for text in values.iter() {
                        source.string(text)?;
                    }
                }
                Ok(())
            }
            Self::NoWhitespace => Ok(()),
            Self::Regex(regex) => regex.original_source(source),
        }
    }
    fn original_controls(&self) -> Result<usize, crate::validator::workspace::Error> {
        let own = std::mem::size_of::<(
            &Self,
            &str,
            std::str::Chars<'_>,
            std::slice::Iter<'_, String>,
        )>();
        own.checked_add(match self {
            Self::Regex(regex) => regex.original_controls()?,
            _ => 0,
        })
        .ok_or(crate::validator::workspace::Error::Overflow)
    }

    #[inline]
    fn is_match(&self, text: &str, ctx: &mut ValidationContext) -> Result<bool, Self::Error> {
        match self {
            CompiledPattern::Prefix(prefix) => Ok(text.starts_with(prefix.as_ref())),
            CompiledPattern::Exact(exact) => Ok(text == exact.as_ref()),
            CompiledPattern::Alternation(alts) => Ok(alts.iter().any(|a| a.as_str() == text)),
            CompiledPattern::NoWhitespace => Ok(!text.chars().any(is_ecma_whitespace)),
            // Treat regex errors as non-match for compatibility
            CompiledPattern::Regex(re) => Ok(re.is_match(text, ctx).unwrap_or(false)),
        }
    }
}

pub(crate) type FancyRegexValidators<F = SerdeJson> =
    Vec<(CompiledPattern<Arc<fancy_regex::Regex>>, SchemaNode<F>)>;
pub(crate) type RegexValidators<F = SerdeJson> =
    Vec<(CompiledPattern<Arc<regex::Regex>>, SchemaNode<F>)>;

/// A value that can look up property validators by name.
pub(crate) trait PropertiesValidatorsMap<F: Json = SerdeJson>: Send + Sync {
    fn original_source(
        &self,
        _: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        Err(crate::validator::workspace::Error::Unqualified(
            crate::validator::workspace::Component::Source("compiled property hash table"),
        ))
    }
    /// Fixed lookup transport; implementations share the actual ordinary worker.
    fn original_lookup_controls(&self) -> Option<usize>;
    fn get_validator(&self, property: &str) -> Option<&SchemaNode<F>>;
    fn get_key_validator(&self, property: &str) -> Option<(&str, &SchemaNode<F>)>;
}

/// Threshold for switching from linear scan to `HashMap`.
pub(crate) const HASHMAP_THRESHOLD: usize = 15;

pub(crate) type SmallValidatorsMap<F = SerdeJson> = Vec<(String, SchemaNode<F>)>;
pub(crate) type BigValidatorsMap<F = SerdeJson> =
    hashbrown::HashMap<String, SchemaNode<F>, ahash::RandomState>;

impl<F: Json> PropertiesValidatorsMap<F> for SmallValidatorsMap<F> {
    fn original_lookup_controls(&self) -> Option<usize> {
        let parts = [
            std::mem::size_of::<std::slice::Iter<'_, (String, SchemaNode<F>)>>(),
            std::mem::size_of::<(&Self, &str)>(),
            std::mem::size_of::<Option<&SchemaNode<F>>>(),
            std::mem::size_of::<Option<(&str, &SchemaNode<F>)>>(),
            std::mem::size_of::<(&str, &str)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        source.vector(self)?;
        for (name, node) in self {
            source.string(name)?;
            source.node(node)?;
        }
        Ok(())
    }

    #[inline]
    fn get_validator(&self, property: &str) -> Option<&SchemaNode<F>> {
        for (prop, node) in self {
            if prop == property {
                return Some(node);
            }
        }
        None
    }
    #[inline]
    fn get_key_validator(&self, property: &str) -> Option<(&str, &SchemaNode<F>)> {
        for (prop, node) in self {
            if prop == property {
                return Some((prop.as_str(), node));
            }
        }
        None
    }
}

impl<F: Json> PropertiesValidatorsMap<F> for BigValidatorsMap<F> {
    fn original_lookup_controls(&self) -> Option<usize> {
        self.lookup_control_bytes("")
    }
    fn original_source(
        &self,
        source: &mut crate::validator::source::Inspector<F>,
    ) -> Result<(), crate::validator::workspace::Error> {
        // Exact reached table allocation from the same pinned ordinary map.
        source.add(self.allocation_size())?;
        for (name, node) in self {
            source.string(name)?;
            source.node(node)?;
        }
        Ok(())
    }

    #[inline]
    fn get_validator(&self, property: &str) -> Option<&SchemaNode<F>> {
        self.get(property)
    }

    #[inline]
    fn get_key_validator(&self, property: &str) -> Option<(&str, &SchemaNode<F>)> {
        self.get_key_value(property)
            .map(|(key, node)| (key.as_str(), node))
    }
}

pub(crate) fn compile_small_map<'a, F: Json>(
    ctx: &compiler::Context<F>,
    map: &'a Map<String, Value>,
) -> Result<SmallValidatorsMap<F>, crate::compilation::CompileError<'a>> {
    let mut properties = Vec::new();
    ctx.funding().grow(&mut properties, map.len())?;
    let kctx = ctx.new_at_location("properties")?;
    for (key, subschema) in map {
        let pctx = kctx.new_at_location(key.as_str())?;
        properties.push((
            ctx.funding().copy_str(key)?,
            compiler::compile(&pctx, pctx.as_resource_ref(subschema))?,
        ));
    }
    Ok(properties)
}

pub(crate) fn compile_big_map<'a, F: Json>(
    ctx: &compiler::Context<F>,
    map: &'a Map<String, Value>,
) -> Result<BigValidatorsMap<F>, crate::compilation::CompileError<'a>> {
    let mut properties = BigValidatorsMap::<F>::with_hasher(ctx.funding().random_state()?);
    ctx.funding().reserve_map(&mut properties, map.len())?;
    let kctx = ctx.new_at_location("properties")?;
    for (key, subschema) in map {
        let pctx = kctx.new_at_location(key.as_str())?;
        properties.insert(
            ctx.funding().copy_str(key)?,
            compiler::compile(&pctx, pctx.as_resource_ref(subschema))?,
        );
    }
    Ok(properties)
}

pub(crate) fn are_properties_valid<'i, F, M, O, C>(
    prop_map: &M,
    object: &O,
    ctx: &mut ValidationContext,
    check: C,
) -> bool
where
    F: Json,
    M: PropertiesValidatorsMap<F>,
    O: Object<'i, F, Node = F::Node<'i>>,
    C: Fn(&F::Node<'i>, &mut ValidationContext) -> bool,
{
    for (property, instance) in object.members() {
        if let Some(validator) = prop_map.get_validator(property.as_ref()) {
            if !validator.is_valid(&instance, ctx) {
                return false;
            }
        } else if !check(&instance, ctx) {
            return false;
        }
    }
    true
}

/// Create a vector of pattern-validators pairs.
/// Uses prefix optimization when patterns are simple `^prefix` patterns.
#[inline]
pub(crate) fn compile_fancy_regex_patterns<'a, F: Json>(
    ctx: &compiler::Context<F>,
    obj: &'a Map<String, Value>,
) -> Result<FancyRegexValidators<F>, crate::compilation::CompileError<'a>> {
    let kctx = ctx.new_at_location("patternProperties")?;
    let mut compiled_patterns = Vec::new();
    ctx.funding().grow(&mut compiled_patterns, obj.len())?;
    for (pattern, subschema) in obj {
        let pctx = kctx.new_at_location(pattern.as_str())?;
        let compiled_pattern =
            match crate::regex::analyze_pattern_with_funding(pattern, ctx.funding())? {
                Some(PatternOptimization::Prefix(prefix)) => {
                    CompiledPattern::Prefix(ctx.funding().arc_str(&prefix)?)
                }
                Some(PatternOptimization::Exact(exact)) => {
                    CompiledPattern::Exact(ctx.funding().arc_str(&exact)?)
                }
                Some(PatternOptimization::Alternation(alts)) => {
                    CompiledPattern::Alternation(ctx.funding().arc_slice(alts)?)
                }
                Some(PatternOptimization::NoWhitespace) => CompiledPattern::NoWhitespace,
                None => {
                    let regex = ctx.get_or_compile_regex(pattern).map_err(|error| {
                        error.diagnostic(kctx.funding(), kctx.location(), subschema)
                    })?;
                    CompiledPattern::Regex(regex)
                }
            };
        let node = compiler::compile(&pctx, pctx.as_resource_ref(subschema))?;
        compiled_patterns.push((compiled_pattern, node));
    }
    Ok(compiled_patterns)
}

/// Create a vector of pattern-validators pairs using standard regex.
/// Uses literal optimizations when patterns are simple prefix or exact-match patterns.
#[inline]
pub(crate) fn compile_regex_patterns<'a, F: Json>(
    ctx: &compiler::Context<F>,
    obj: &'a Map<String, Value>,
) -> Result<RegexValidators<F>, crate::compilation::CompileError<'a>> {
    let kctx = ctx.new_at_location("patternProperties")?;
    let mut compiled_patterns = Vec::new();
    ctx.funding().grow(&mut compiled_patterns, obj.len())?;
    for (pattern, subschema) in obj {
        let pctx = kctx.new_at_location(pattern.as_str())?;
        let compiled_pattern =
            match crate::regex::analyze_pattern_with_funding(pattern, ctx.funding())? {
                Some(PatternOptimization::Prefix(prefix)) => {
                    CompiledPattern::Prefix(ctx.funding().arc_str(&prefix)?)
                }
                Some(PatternOptimization::Exact(exact)) => {
                    CompiledPattern::Exact(ctx.funding().arc_str(&exact)?)
                }
                Some(PatternOptimization::Alternation(alts)) => {
                    CompiledPattern::Alternation(ctx.funding().arc_slice(alts)?)
                }
                Some(PatternOptimization::NoWhitespace) => CompiledPattern::NoWhitespace,
                None => {
                    let regex = ctx
                        .get_or_compile_standard_regex(pattern)
                        .map_err(|error| {
                            error.diagnostic(kctx.funding(), kctx.location(), subschema)
                        })?;
                    CompiledPattern::Regex(regex)
                }
            };
        let node = compiler::compile(&pctx, pctx.as_resource_ref(subschema))?;
        compiled_patterns.push((compiled_pattern, node));
    }
    Ok(compiled_patterns)
}

macro_rules! compile_dynamic_prop_map_validator {
    ($validator:tt, $properties:ident, $ctx:expr, $( $arg:expr ),* $(,)*) => {{
        if let Value::Object(map) = $properties {
            if map.len() < HASHMAP_THRESHOLD {
                Some($validator::<SmallValidatorsMap<F>>::compile(
                    map, $ctx, $($arg, )*
                ))
            } else {
                Some($validator::<BigValidatorsMap<F>>::compile(
                    map, $ctx, $($arg, )*
                ))
            }
        } else {
            let location = $ctx.location().clone();
            Some(Err(ValidationError::compile_error(
                location.clone(),
                location,
                Location::new(),
                std::borrow::Cow::Borrowed($properties),
                "Unexpected type",
            ).into()))
        }
    }};
}

pub(crate) use compile_dynamic_prop_map_validator;
