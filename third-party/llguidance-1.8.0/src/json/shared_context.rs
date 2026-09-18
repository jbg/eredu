use crate::{json::schema::OptSchemaExt, regex_to_lark};
use derivre::{SourceHashMap as HashMap, ParserAllocationFunding};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::{Regex, RegexAst, RegexBuilder};

use super::{
    context::Context,
    schema::{ObjectSchema, Schema, IMPLEMENTED, META_AND_ANNOTATIONS},
};

pub struct SharedContext {
    defs: HashMap<String, Schema>,
    seen: HashSet<String>,
    n_compiled: usize,
    pending_warnings: Vec<String>,
    pattern_cache: PatternPropertyCache,
    funding: ParserAllocationFunding,
}

pub struct PatternPropertyCache {
    inner: HashMap<String, Regex>,
    funding: ParserAllocationFunding,
}

type HashSet<T> = hashbrown::HashSet<T, derivre::RandomState>;

const CHECK_LIMIT: u64 = 10_000;

impl PatternPropertyCache {
    pub fn new(funding: ParserAllocationFunding) -> Self { Self { inner: HashMap::default(), funding } }
    pub fn is_match(&mut self, regex: &str, value: &str) -> Result<bool> {
        let lark_regex = self.funding.try_format(format_args!("{}", regex_to_lark(regex, "dw")))?;
        if let Some(cached_regex) = self.inner.get_mut(lark_regex.as_str()) {
            return cached_regex.is_match(value);
        }

        let mut builder = RegexBuilder::new(self.funding.clone())?;
        let eref = builder.mk_regex_for_serach(lark_regex.as_str())?;
        let mut rx = builder.to_regex_limited(eref, CHECK_LIMIT)?;
        let res = rx.is_match(value)?;
        self.funding.try_insert(&mut self.inner, lark_regex, rx)?;
        Ok(res)
    }

    pub fn check_disjoint(&mut self, regexes: &[&String]) -> Result<()> {
        // TODO cache something?
        let mut builder = RegexBuilder::new(self.funding.clone())?;
        let mut erefs = Vec::new();
        for regex in regexes {
            let regex = self.funding.try_format(format_args!("{}", regex_to_lark(regex, "dw")))?;
            let expr = builder.mk_regex_for_serach(&regex)?;
            self.funding.try_push(&mut erefs, expr)?;
        }
        for (ai, a) in erefs.iter().enumerate() {
            for (bi, b) in erefs.iter().enumerate() {
                if ai >= bi {
                    continue;
                }
                let mut args = Vec::new();
                self.funding.try_push(&mut args, RegexAst::ExprRef(*a))?;
                self.funding.try_push(&mut args, RegexAst::ExprRef(*b))?;
                let intersect = builder.mk(&RegexAst::And(args))?;
                let mut rx = builder
                    .to_regex_limited(intersect, CHECK_LIMIT)
                    .map_err(|error| {
                        if crate::earley::is_grammar_storage_failure(&error) { return error; }
                        anyhow!(&self.funding,
                            "can't determine if patternProperty regexes /{}/ and /{}/ are disjoint",
                            regex_to_lark(regexes[ai], ""),
                            regex_to_lark(regexes[bi], "")
                        )
                    })?;
                if !rx.always_empty() {
                    return Err(anyhow!(&self.funding,
                        "patternProperty regexes /{}/ and /{}/ are not disjoint",
                        regex_to_lark(regexes[ai], ""),
                        regex_to_lark(regexes[bi], "")
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn property_schema<'a>(&mut self, obj: &'a ObjectSchema, prop: &str) -> Result<&'a Schema> {
        if let Some(schema) = obj.properties.get(prop) {
            return Ok(schema);
        }

        for (key, schema) in obj.pattern_properties.iter() {
            if self.is_match(key, prop)? {
                return Ok(schema);
            }
        }

        Ok(obj.additional_properties.schema_ref())
    }
}

impl SharedContext {
    pub fn new(funding: ParserAllocationFunding) -> Self {
        SharedContext {
            defs: HashMap::default(),
            seen: HashSet::default(),
            n_compiled: 0,
            pending_warnings: Vec::new(),
            pattern_cache: PatternPropertyCache::new(funding.clone()),
            funding,
        }
    }
}

impl Context<'_> {
    pub fn insert_ref(&self, uri: &str, schema: Schema) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        self.funding.try_insert(&mut shared.defs, self.funding.try_copy_str(uri)?, schema)?;
        Ok(())
    }

    pub fn get_ref_cloned(&self, uri: &str) -> Result<Option<Schema>> {
        self.shared.borrow().defs.get(uri).map(|schema| schema.copy_with_funding(&self.funding)).transpose()
    }

    pub fn mark_seen(&self, uri: &str) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        self.funding.try_insert_set(&mut shared.seen, self.funding.try_copy_str(uri)?)?;
        Ok(())
    }

    pub fn been_seen(&self, uri: &str) -> bool {
        self.shared.borrow().seen.contains(uri)
    }

    pub fn is_valid_keyword(&self, keyword: &str) -> bool {
        if !self.draft.is_known_keyword(keyword)
            || IMPLEMENTED.contains(&keyword)
            || META_AND_ANNOTATIONS.contains(&keyword)
        {
            return true;
        }
        false
    }

    pub fn increment(&self) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        shared.n_compiled += 1;
        if shared.n_compiled > self.options.max_size {
            bail!(&self.funding, "schema too large");
        }
        Ok(())
    }

    pub fn record_warning(&self, msg: String) -> Result<()> {
        self.funding.try_push(&mut self.shared.borrow_mut().pending_warnings, msg)?;
        Ok(())
    }

    pub fn property_schema<'a>(&self, obj: &'a ObjectSchema, prop: &str) -> Result<&'a Schema> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .property_schema(obj, prop)
    }

    pub fn property_schema_matches(&self, pattern: &str, name: &str) -> Result<bool> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .is_match(pattern, name)
    }

    pub fn check_disjoint_pattern_properties(&self, regexes: &[&String]) -> Result<()> {
        self.shared
            .borrow_mut()
            .pattern_cache
            .check_disjoint(regexes)
    }

    pub fn into_result(self, schema: Schema) -> BuiltSchema {
        let mut shared = self.shared.borrow_mut();
        BuiltSchema {
            schema,
            definitions: std::mem::take(&mut shared.defs),
            warnings: std::mem::take(&mut shared.pending_warnings),
            pattern_cache: std::mem::replace(&mut shared.pattern_cache, PatternPropertyCache::new(self.funding.clone())),
        }
    }
}

pub struct BuiltSchema {
    pub schema: Schema,
    pub definitions: HashMap<String, Schema>,
    pub warnings: Vec<String>,
    pub pattern_cache: PatternPropertyCache,
}

impl BuiltSchema {
    pub fn simple(schema: Schema, funding: ParserAllocationFunding) -> Self {
        BuiltSchema {
            schema,
            definitions: HashMap::default(),
            warnings: Vec::new(),
            pattern_cache: PatternPropertyCache::new(funding),
        }
    }
}
