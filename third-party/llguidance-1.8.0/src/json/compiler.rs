mod frames;
use crate::api::{LLGuidanceOptions, SkipSpec};
use crate::grammar_builder::GrammarResult;
use crate::json::schema::{NumberSchema, StringSchema};
use crate::regex_to_lark;
use crate::allocation::CollectFunded;
use derivre::{ParserAllocationFunding, SourceHashMap as HashMap};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::{ExprRef, JsonQuoteOptions, RegexAst};
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::numeric::{check_number_bounds, rx_float_range, rx_int_range};
use super::schema::{build_schema, ArraySchema, ObjectSchema, OptSchemaExt, Schema};
use super::shared_context::PatternPropertyCache;
use super::RetrieveWrapper;

use crate::{GrammarBuilder, NodeRef};

// TODO: grammar size limit
// TODO: array maxItems etc limits
// TODO: schemastore/src/schemas/json/BizTalkServerApplicationSchema.json - this breaks 1M fuel on lexer, why?!

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct JsonCompileOptions {
    pub item_separator: String,
    pub key_separator: String,
    pub whitespace_flexible: bool,
    pub whitespace_pattern: Option<String>,
    pub coerce_one_of: bool,
    pub lenient: bool,
    /// Allowed escape letters after '\' when quoting JSON strings.
    /// Defaults to full JSON set: nrbtf"u\
    /// For example, set to nrbtf"\ to disallow \uXXXX escapes.
    pub json_allowed_escapes: Option<String>,
    /// Allow printable Unicode escapes in strings and object keys without regex constraints.
    /// Paired surrogate escapes count as one character for string length limits.
    pub json_allow_general_unicode_escapes: bool,
    #[serde(skip)]
    pub retriever: Option<RetrieveWrapper>,
}

#[derive(Debug)]
struct UnsatisfiableSchemaError {
    message: String,
}

impl std::error::Error for UnsatisfiableSchemaError {}
impl std::fmt::Display for UnsatisfiableSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsatisfiable schema: {}", self.message)
    }
}

// Keep both JSON string paths aligned with Derivre's regular escape policy.
const DEFAULT_JSON_ALLOWED_ESCAPES: &str = "nrbtf\\\"u";

// Match one Unicode scalar: a non-surrogate BMP escape or a high surrogate
// immediately followed by a low surrogate. Keeping each pair in one alternative
// makes string length limits count supplementary-plane characters correctly.
const UNICODE_SCALAR_ESCAPE_REGEX: &str = concat!(
    r"\\u(?:[0-9a-ce-fA-CE-F][0-9a-fA-F]{3}",
    r"|[dD][0-7][0-9a-fA-F]{2}",
    r"|[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})"
);

struct Compiler<'t, 'f> {
    frames: &'f derivre::prepared_funding::Scope<'f, ParserAllocationFunding>,
    builder: GrammarBuilder<'t>,
    options: JsonCompileOptions,
    definitions: HashMap<String, NodeRef>,
    pending_definitions: Vec<(String, NodeRef)>,
    pattern_cache: PatternPropertyCache,

    any_cache: Option<NodeRef>,
    string_cache: Option<NodeRef>,
    /// Reuse the decoded Unicode-scalar expression across all string length bounds.
    general_unicode_scalar_cache: Option<ExprRef>,
    /// Reuse compiled Unicode-string expressions for identical decoded length bounds.
    general_unicode_string_cache: HashMap<(usize, Option<usize>), ExprRef>,
    item_separator_cache: Option<NodeRef>,
    key_separator_cache: Option<NodeRef>,
}

impl Default for JsonCompileOptions {
    fn default() -> Self {
        Self::new(&ParserAllocationFunding::unenforced()).expect("unenforced JSON option storage")
    }
}

impl JsonCompileOptions {
    pub fn new(funding: &ParserAllocationFunding) -> Result<Self> {
        Ok(Self {
            item_separator: funding.try_copy_str(",")?,
            key_separator: funding.try_copy_str(":")?,
            whitespace_pattern: None,
            whitespace_flexible: true,
            coerce_one_of: false,
            lenient: false,
            json_allowed_escapes: None,
            json_allow_general_unicode_escapes: false,
            retriever: None,
        })
    }

    fn copy_with_funding(&self, funding: &ParserAllocationFunding) -> Result<Self> {
        Ok(Self {
            item_separator: funding.try_copy_str(&self.item_separator)?,
            key_separator: funding.try_copy_str(&self.key_separator)?,
            whitespace_pattern: self.whitespace_pattern.as_deref().map(|s| funding.try_copy_str(s)).transpose()?,
            whitespace_flexible: self.whitespace_flexible,
            coerce_one_of: self.coerce_one_of,
            lenient: self.lenient,
            json_allowed_escapes: self.json_allowed_escapes.as_deref().map(|s| funding.try_copy_str(s)).transpose()?,
            json_allow_general_unicode_escapes: self.json_allow_general_unicode_escapes,
            retriever: self.retriever.clone(),
        })
    }

    /// Consume the existing object strings; defaults alone construct new text.
    fn from_owned(value: Value, funding: &ParserAllocationFunding) -> Result<Self> {
        let Value::Object(fields) = value else { bail!(funding, "expected JSON compile option object"); };
        let mut options = Self::new(funding)?;
        for (name, value) in fields {
            match name.as_str() {
                "item_separator" | "key_separator" => {
                    let Value::String(value) = value else { bail!(funding, "expected string for {name}"); };
                    if name == "item_separator" { options.item_separator = value; } else { options.key_separator = value; }
                }
                "whitespace_pattern" | "json_allowed_escapes" => {
                    let value = match value { Value::Null => None, Value::String(s) => Some(s), _ => bail!(funding, "expected string or null for {name}") };
                    if name == "whitespace_pattern" { options.whitespace_pattern = value; } else { options.json_allowed_escapes = value; }
                }
                "whitespace_flexible" | "coerce_one_of" | "lenient" | "json_allow_general_unicode_escapes" => {
                    let Value::Bool(value) = value else { bail!(funding, "expected boolean for {name}"); };
                    match name.as_str() {
                        "whitespace_flexible" => options.whitespace_flexible = value,
                        "coerce_one_of" => options.coerce_one_of = value,
                        "lenient" => options.lenient = value,
                        _ => options.json_allow_general_unicode_escapes = value,
                    }
                }
                _ => bail!(funding, "unknown JSON compile option: {name}"),
            }
        }
        Ok(options)
    }

    pub fn json_to_llg_with_overrides<'t>(
        &self,
        builder: GrammarBuilder<'t>,
        mut schema: Value,
    ) -> Result<GrammarResult<'t>> {
        builder.funding.reserve(std::mem::size_of::<(
            &Self, GrammarBuilder<'t>, Value, Option<Value>, Value, Self,
            Result<Self>, Result<GrammarResult<'t>>,
        )>())?;
        if let Some(x_guidance) = schema.as_object_mut().and_then(|object| object.remove("x-guidance")) {
            let opts = Self::from_owned(x_guidance, &builder.funding)?;
            opts.json_to_llg(builder, schema)
        } else {
            self.json_to_llg(builder, schema)
        }
    }

    pub fn json_to_llg<'t>(
        &self,
        builder: GrammarBuilder<'t>,
        schema: Value,
    ) -> Result<GrammarResult<'t>> {
        self.compile_with_scope(builder, schema, true)
    }

    pub fn json_to_llg_no_validate<'t>(
        &self,
        builder: GrammarBuilder<'t>,
        schema: Value,
    ) -> Result<GrammarResult<'t>> {
        self.compile_with_scope(builder, schema, false)
    }

    // Both public entries use this one emitter and its exact invocation payer.
    // The scope is borrowed by the compiler and cannot escape in GrammarResult.
    fn compile_with_scope<'t>(
        &self, builder: GrammarBuilder<'t>, schema: Value, validate: bool,
    ) -> Result<GrammarResult<'t>> {
        let funding = builder.funding.clone();
        let scope = derivre::prepared_funding::Scope::new(&funding)
            .map_err(|error| frames::failure(error, &funding))?;
        let _frame = frames::enter::<frames::Invocation<'_, 't>>(&scope, &funding)?;
        let compiler = Compiler::new(self.copy_with_funding(&funding)?, builder, &scope)?;
        #[cfg(feature = "jsonschema_validation")]
        if validate { super::super::json_validation::validate_schema(&schema)?; }
        #[cfg(not(feature = "jsonschema_validation"))]
        let _ = validate;
        compiler.execute(schema)
    }

    pub fn apply_to(&self, schema: &mut Value, funding: &ParserAllocationFunding) -> Result<()> {
        let text = super::source_text::serialize(self, funding)?;
        let value = serde_json::bounded_events::from_slice_with_allocations(text.as_bytes(), &crate::allocation::CompilerAllocation(funding))
            .map_err(|error| ParserError::cause(error, funding))?;
        schema.as_object_mut().unwrap().try_insert_with_allocations(funding.try_copy_str("x-guidance")?, value, &crate::allocation::CompilerAllocation(funding))
            .map_err(|error| ParserError::cause(error, funding))?;
        Ok(())
    }

}

impl<'t, 'f> Compiler<'t, 'f> {
    fn frame<T>(&self) -> Result<derivre::prepared_funding::Frame<'f>> {
        frames::enter::<(&Self, T)>(self.frames, &self.builder.funding)
    }
    pub fn new(options: JsonCompileOptions, builder: GrammarBuilder<'t>,
        frames: &'f derivre::prepared_funding::Scope<'f, ParserAllocationFunding>) -> Result<Self> {
        let _frame = frames::enter::<(JsonCompileOptions, GrammarBuilder<'t>, Self,
            PatternPropertyCache, Result<Self>)>(frames, &builder.funding)?;
        let pattern_cache = PatternPropertyCache::new(builder.funding.clone());
        Ok(Self {
            frames,
            builder,
            options,
            definitions: HashMap::default(),
            pending_definitions: vec![],
            any_cache: None,
            string_cache: None,
            general_unicode_scalar_cache: None,
            general_unicode_string_cache: HashMap::default(),
            item_separator_cache: None,
            key_separator_cache: None,
            pattern_cache,
        })
    }

    pub fn execute(mut self, schema: Value) -> Result<GrammarResult<'t>> {
        let _frame = self.frame::<frames::Execute<'_ , 't, 'f>>()?;
        let skip = if let Some(pattern) = &self.options.whitespace_pattern {
            RegexAst::Regex(self.builder.funding.try_copy_str(pattern)?)
        } else if self.options.whitespace_flexible {
            RegexAst::Regex(self.builder.funding.try_copy_str(r"[\x20\x0A\x0D\x09]+")?)
        } else {
            RegexAst::NoMatch
        };
        let id = self
            .builder
            .add_grammar_with_skip(LLGuidanceOptions::default(), SkipSpec::once(skip))?;

        let built = build_schema(schema, &self.options, &self.builder.funding)?;
        self.pattern_cache = built.pattern_cache;

        for w in built.warnings {
            self.builder.add_warning(&w)?;
        }

        let root = self.gen_json(&built.schema)?;
        self.builder.set_start_node(root)?;

        while let Some((path, pl)) = self.pending_definitions.pop() {
            let schema = built
                .definitions
                .get(&path)
                .ok_or_else(|| anyhow!(&self.builder.funding, "Definition not found: {}", path))?;
            let compiled = self.gen_json(schema).map_err(|error|
                error.annotate("", format_args!("\n  while processing {path}"), &self.builder.funding))?;
            self.builder.set_placeholder(pl, compiled)?;
        }

        Ok(self.builder.finalize(id))
    }

    fn gen_json(&mut self, json_schema: &Schema) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, &Schema, Option<RegexAst>, Result<NodeRef>)>()?;
        if let Some(ast) = self.regex_compile(json_schema)? {
            return self.ast_lexeme(ast);
        }
        match json_schema {
            Schema::Any => self.gen_json_any(),
            Schema::Unsatisfiable(reason) => Err(ParserError::cause(UnsatisfiableSchemaError {
                message: self.builder.funding.try_copy_str(reason)?,
            }, &self.builder.funding)),

            Schema::Array(arr) => self.gen_json_array(arr),
            Schema::Object(obj) => self.gen_json_object(obj),
            Schema::AnyOf(options) => self.process_any_of(options),
            Schema::OneOf(options) => self.process_one_of(options),
            Schema::Ref(uri) => self.get_definition(uri),

            Schema::Null | Schema::Boolean(_) | Schema::String(_) | Schema::Number(_) => {
                unreachable!("should be handled in regex_compile()")
            }
        }
    }

    fn process_one_of(&mut self, options: &[Schema]) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, &[Schema], Result<NodeRef>)>()?;
        if self.options.coerce_one_of || self.options.lenient {
            self.builder
                .add_warning("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.")?;
            self.process_any_of(options)
        } else {
            Err(anyhow!(&self.builder.funding, "oneOf constraints are not supported. Enable 'coerce_one_of' option to approximate oneOf with anyOf"))
        }
    }

    fn process_option(
        &mut self,
        option: &Schema,
        regex_nodes: &mut Vec<RegexAst>,
        cfg_nodes: &mut Vec<NodeRef>,
    ) -> Result<()> {
        let _frame = self.frame::<(&Self, &Schema, &mut Vec<RegexAst>, &mut Vec<NodeRef>, Option<RegexAst>, Result<()>)>()?;
        match self.regex_compile(option)? {
            Some(c) => self.builder.funding.clone().try_push(regex_nodes, c)?,
            None => self.builder.funding.clone().try_push(cfg_nodes, self.gen_json(option)?)?,
        }
        Ok(())
    }

    fn process_any_of(&mut self, options: &[Schema]) -> Result<NodeRef> {
        let _frame = self.frame::<frames::Alternatives<'_>>()?;
        let mut regex_nodes = vec![];
        let mut cfg_nodes = vec![];
        let mut last_error = None;

        for option in options.iter() {
            if let Err(err) = self.process_option(option, &mut regex_nodes, &mut cfg_nodes) {
                match err.downcast_ref::<UnsatisfiableSchemaError>() {
                    Some(_) => last_error = Some(err),
                    None => return Err(err),
                }
            }
        }

        self.builder.check_limits()?;

        if !regex_nodes.is_empty() {
            let node = RegexAst::Or(regex_nodes);
            let lex = self.ast_lexeme(node)?;
            self.builder.funding.clone().try_push(&mut cfg_nodes, lex)?;
        }

        if !cfg_nodes.is_empty() {
            Ok(self.builder.select(&cfg_nodes)?)
        } else if let Some(e) = last_error {
            Err(ParserError::cause(UnsatisfiableSchemaError {
                message: self.builder.funding.try_copy_str("All options in anyOf are unsatisfiable")?,
            }, &self.builder.funding)
            .context(e, &self.builder.funding))
        } else {
            Err(ParserError::cause(UnsatisfiableSchemaError {
                message: self.builder.funding.try_copy_str("No options in anyOf")?,
            }, &self.builder.funding))
        }
    }

    fn json_int(&mut self, num: &NumberSchema) -> Result<RegexAst> {
        let _frame = self.frame::<frames::Integer<'_>>()?;
        if let Some(message) = check_number_bounds(num, &self.builder.funding)? {
            return Err(ParserError::cause(UnsatisfiableSchemaError { message }, &self.builder.funding));
        }
        let minimum = match num.get_minimum() {
            (Some(min_val), true) => {
                if min_val.fract() != 0.0 {
                    Some(min_val.ceil())
                } else {
                    Some(min_val + 1.0)
                }
            }
            (Some(min_val), false) => Some(min_val.ceil()),
            _ => None,
        }
        .map(|val| val as i64);
        let maximum = match num.get_maximum() {
            (Some(max_val), true) => {
                if max_val.fract() != 0.0 {
                    Some(max_val.floor())
                } else {
                    Some(max_val - 1.0)
                }
            }
            (Some(max_val), false) => Some(max_val.floor()),
            _ => None,
        }
        .map(|val| val as i64);
        let rx = rx_int_range(minimum, maximum, &self.builder.funding).map_err(|error| {
            error.context(format_args!("Failed to generate regex for integer range: min={minimum:?}, max={maximum:?}"), &self.builder.funding)
        })?;
        let mut ast = RegexAst::Regex(rx);
        if let Some(d) = num.multiple_of.as_ref() {
            ast = RegexAst::And([Ok::<_, derivre::ParserError>(ast), Ok(signed_multiple_of_ast(d.coef, d.exp, self.frames, &self.builder.funding)?)].into_iter().collect_with_funding(&self.builder.funding)?);
        }
        Ok(ast)
    }

    fn json_number(&mut self, num: &NumberSchema) -> Result<RegexAst> {
        let _frame = self.frame::<frames::Number<'_>>()?;
        if let Some(message) = check_number_bounds(num, &self.builder.funding)? {
            return Err(ParserError::cause(UnsatisfiableSchemaError { message }, &self.builder.funding));
        }
        let (minimum, exclusive_minimum) = num.get_minimum();
        let (maximum, exclusive_maximum) = num.get_maximum();
        let rx = rx_float_range(minimum, maximum, !exclusive_minimum, !exclusive_maximum, &self.builder.funding).map_err(|error| {
            error.context(format_args!("Failed to generate regex for float range: min={minimum:?}, max={maximum:?}"), &self.builder.funding)
        })?;
        let mut ast = RegexAst::Regex(rx);
        if let Some(d) = num.multiple_of.as_ref() {
            ast = RegexAst::And([Ok::<_, derivre::ParserError>(ast), Ok(signed_multiple_of_ast(d.coef, d.exp, self.frames, &self.builder.funding)?)].into_iter().collect_with_funding(&self.builder.funding)?);
        }
        Ok(ast)
    }

    fn ast_lexeme(&mut self, ast: RegexAst) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, RegexAst, ExprRef, Result<NodeRef>)>()?;
        let id = self.builder.regex.add_ast(ast)?;
        Ok(self.builder.lexeme(id)?)
    }

    fn json_simple_string(&mut self) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, Option<NodeRef>, RegexAst, NodeRef, Result<NodeRef>)>()?;
        if let Some(node) = self.string_cache {
            return Ok(node);
        }

        let ast = if self.options.json_allow_general_unicode_escapes {
            self.json_general_unicode_string(0, None)?
        } else {
            self.json_quote(RegexAst::Regex(self.builder.funding.try_copy_str("(?s:.*)")?))?
        };
        let node = self.ast_lexeme(ast)?;
        self.string_cache = Some(node);
        Ok(node)
    }

    fn item_separator(&mut self) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, Option<NodeRef>, ExprRef, NodeRef, Result<NodeRef>)>()?;
        if let Some(node) = self.item_separator_cache {
            return Ok(node);
        }
        let rx = self.builder.regex.regex(&self.options.item_separator)?;
        let node = self.builder.lexeme(rx)?;
        self.item_separator_cache = Some(node);
        Ok(node)
    }

    fn key_separator(&mut self) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, Option<NodeRef>, ExprRef, NodeRef, Result<NodeRef>)>()?;
        if let Some(node) = self.key_separator_cache {
            return Ok(node);
        }
        let rx = self.builder.regex.regex(&self.options.key_separator)?;
        let node = self.builder.lexeme(rx)?;
        self.key_separator_cache = Some(node);
        Ok(node)
    }

    fn get_definition(&mut self, reference: &str) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, &str, NodeRef, String, (String, NodeRef), Result<NodeRef>)>()?;
        if let Some(definition) = self.definitions.get(reference) {
            return Ok(*definition);
        }
        let r = self.builder.new_node(reference)?;
        self.builder.funding.try_insert(&mut self.definitions, self.builder.funding.try_copy_str(reference)?, r)?;
        self.builder.funding.clone().try_push(&mut self.pending_definitions, (self.builder.funding.try_copy_str(reference)?, r))?;
        Ok(r)
    }

    fn gen_json_any(&mut self) -> Result<NodeRef> {
        let _frame = self.frame::<frames::Any>()?;
        if let Some(json_any) = self.any_cache {
            return Ok(json_any);
        }

        let json_any = self.builder.new_node("json_any")?;
        self.any_cache = Some(json_any); // avoid infinite recursion
        let num = self.json_number(&NumberSchema::default())?;
        let tf = self.builder.regex.regex("true|false")?;
        let options = [
            self.builder.string("null")?,
            self.builder.lexeme(tf)?,
            self.ast_lexeme(num)?,
            self.json_simple_string()?,
            self.gen_json_array(&ArraySchema {
                min_items: 0,
                max_items: None,
                prefix_items: vec![],
                items: Some(self.builder.funding.try_box(Schema::Any)?),
            })?,
            self.gen_json_object(&ObjectSchema {
                properties: IndexMap::new(),
                additional_properties: Some(self.builder.funding.try_box(Schema::Any)?),
                required: IndexSet::new(),
                pattern_properties: IndexMap::new(),
                min_properties: 0,
                max_properties: None,
            })?,
        ];
        let inner = self.builder.select(&options)?;
        self.builder.set_placeholder(json_any, inner)?;
        Ok(json_any)
    }

    fn gen_json_object(&mut self, obj: &ObjectSchema) -> Result<NodeRef> {
        let _frame = self.frame::<frames::Object<'_>>()?;
        let mut taken_names: Vec<String> = vec![];
        let mut unquoted_taken_names: Vec<&str> = vec![];
        let mut items: Vec<(NodeRef, bool)> = vec![];

        let colon = self.key_separator()?;

        let mut num_required = 0;
        let mut num_optional = 0;

        for name in obj.properties.keys().chain(
            obj.required
                .iter()
                .filter(|n| !obj.properties.contains_key(n.as_str())),
        ) {
            let property_schema = self.pattern_cache.property_schema(obj, name)?;
            let is_required = obj.required.contains(name);
            if !obj.pattern_properties.is_empty() {
                self.builder.funding.clone().try_push(&mut unquoted_taken_names, name.as_str())?;
            }
            // Quote (and escape) the name
            let quoted_name = super::source_text::serialize(name, &self.builder.funding)?;
            let property = match self.gen_json(property_schema) {
                Ok(node) => node,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Property is optional; don't raise UnsatisfiableSchemaError but mark name as taken
                    Some(_) if !is_required => {
                        self.builder.funding.clone().try_push(&mut taken_names, quoted_name)?;
                        continue;
                    }
                    // Property is required; add context and propagate UnsatisfiableSchemaError
                    Some(_) => {
                        return Err(e.context(UnsatisfiableSchemaError {
                            message: self.builder.funding.try_format(format_args!("required property '{name}' is unsatisfiable"))?,
                        }, &self.builder.funding));
                    }
                },
            };
            let name = self.builder.string(&quoted_name)?;
            self.builder.funding.clone().try_push(&mut taken_names, quoted_name)?;
            let item = self.builder.join(&[name, colon, property])?;
            self.builder.funding.clone().try_push(&mut items, (item, is_required))?;
            if is_required {
                num_required += 1;
            } else {
                num_optional += 1;
            }
        }

        let min_properties = obj.min_properties.saturating_sub(num_required);
        let max_properties = obj.max_properties.map(|v| v.saturating_sub(num_required));

        if num_optional > 0 && (min_properties > 0 || max_properties.is_some()) {
            // special case for min/maxProperties == 1
            // this is sometimes used to indicate that at least one property is required
            if min_properties <= 1
                && max_properties.unwrap_or(1) == 1
                && obj.pattern_properties.is_empty()
                && obj
                    .additional_properties
                    .as_ref()
                    .map(|s| s.is_unsat())
                    .unwrap_or(false)
            {
                let mut options: Vec<Vec<(NodeRef, bool)>> = Vec::new();
                for (idx, (_, required)) in items.iter().enumerate() {
                    if *required { continue; }
                    let mut option = Vec::new();
                    for (other, (node, required)) in items.iter().enumerate() {
                        if max_properties == Some(1) {
                            if other == idx || *required { self.builder.funding.clone().try_push(&mut option, (*node, true))?; }
                        } else {
                            self.builder.funding.clone().try_push(&mut option, (*node, *required || other == idx))?;
                        }
                    }
                    self.builder.funding.clone().try_push(&mut options, option)?;
                }
                if max_properties == Some(1) && min_properties == 0 {
                    let mut option = Vec::new();
                    for item in items.into_iter().filter(|(_, required)| *required) {
                        self.builder.funding.clone().try_push(&mut option, item)?;
                    }
                    self.builder.funding.clone().try_push(&mut options, option)?;
                }
                let funding = self.builder.funding.clone();
                let sel_options = options.iter().map(|option| self.object_fields(option)).collect_with_funding(&funding)?;
                return Ok(self.builder.select(&sel_options)?);
            }

            let msg = "min/maxProperties only supported when all keys listed in \"properties\" are required";
            if self.options.lenient {
                self.builder.add_warning(msg)?;
            } else {
                bail!(&self.builder.funding, msg);
            }
        }

        let mut taken_name_ids = Vec::new();
        self.builder.funding.try_grow_vec(&mut taken_name_ids, taken_names.len())?;
        for name in &taken_names { taken_name_ids.push(self.builder.regex.literal(name)?); }

        let mut pattern_options = vec![];
        for (pattern, schema) in obj.pattern_properties.iter() {
            let regex = self
                .builder
                .regex
                .add_ast(self.json_quote(RegexAst::SearchRegex(self.builder.funding.try_format(format_args!("{}", regex_to_lark(pattern, "dw")))?))?)?;
            self.builder.funding.clone().try_push(&mut taken_name_ids, regex)?;

            let schema = match self.gen_json(schema) {
                Ok(r) => r,
                Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                    // If it's not an UnsatisfiableSchemaError, just propagate it normally
                    None => return Err(e),
                    // Property is optional; don't raise UnsatisfiableSchemaError but mark name as taken
                    Some(_) => continue,
                },
            };

            let mut exclude_names = Vec::new();
            for (index, name) in unquoted_taken_names.iter().enumerate() {
                let matches = match self.pattern_cache.is_match(pattern, name) {
                    Ok(value) => value,
                    Err(error) if crate::earley::is_grammar_storage_failure(&error) => return Err(error),
                    Err(_) => true,
                };
                if matches { self.builder.funding.clone().try_push(&mut exclude_names, taken_name_ids[index])?; }
            }
            let regex = if exclude_names.is_empty() {
                regex
            } else {
                let options = self.builder.regex.select(&exclude_names)?;
                let not_taken = self.builder.regex.not(options)?;
                self.builder.regex.and(&[regex, not_taken])?
            };

            let name = self.builder.lexeme(regex)?;
            self.builder.funding.clone().try_push(&mut pattern_options, self.builder.join(&[name, colon, schema])?)?;
        }

        match self.gen_json(obj.additional_properties.schema_ref()) {
            Err(e) => {
                if e.downcast_ref::<UnsatisfiableSchemaError>().is_none() {
                    // Propagate errors that aren't UnsatisfiableSchemaError
                    return Err(e);
                }
                // Ignore UnsatisfiableSchemaError for additionalProperties
            }
            Ok(property) => {
                let name = if taken_name_ids.is_empty() {
                    self.json_simple_string()?
                } else {
                    let taken = self.builder.regex.select(&taken_name_ids)?;
                    let not_taken = self.builder.regex.not(taken)?;
                    let valid_ast = self.json_general_unicode_string(0, None)?;
                    let valid = self.builder.regex.add_ast(valid_ast)?;
                    let valid_and_not_taken = self.builder.regex.and(&[valid, not_taken])?;
                    self.builder.lexeme(valid_and_not_taken)?
                };
                let item = self.builder.join(&[name, colon, property])?;
                self.builder.funding.clone().try_push(&mut pattern_options, item)?;
            }
        }

        if !pattern_options.is_empty() && max_properties != Some(0) {
            let pattern = self.builder.select(&pattern_options)?;
            let required = min_properties > 0;
            let seq = self.bounded_sequence(pattern, min_properties, max_properties)?;
            self.builder.funding.clone().try_push(&mut items, (seq, required))?;
        } else if min_properties > 0 {
            return Err(ParserError::cause(UnsatisfiableSchemaError {
                message: self.builder.funding.try_format(format_args!(
                    "minProperties ({min_properties}) is greater than number of properties ({num_required})"
                ))?,
            }, &self.builder.funding));
        }

        self.object_fields(&items)
    }

    fn object_fields(&mut self, items: &[(NodeRef, bool)]) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, &[(NodeRef, bool)], NodeRef, NodeRef, NodeRef, [NodeRef; 3], frames::SequenceCache<'_>, Result<NodeRef>)>()?;
        let opener = self.builder.string("{")?;
        let inner = self.ordered_sequence(items, false, &mut HashMap::default())?;
        let closer = self.builder.string("}")?;
        Ok(self.builder.join(&[opener, inner, closer])?)
    }

    #[allow(clippy::type_complexity)]
    fn ordered_sequence<'a>(
        &mut self,
        items: &'a [(NodeRef, bool)],
        prefixed: bool,
        cache: &mut HashMap<(&'a [(NodeRef, bool)], bool), NodeRef>,
    ) -> Result<NodeRef> {
        let _frame = self.frame::<frames::Sequence<'_>>()?;
        // Cache to reduce number of nodes from O(n^2) to O(n)
        if let Some(node) = cache.get(&(items, prefixed)) {
            return Ok(*node);
        }
        if items.is_empty() {
            return Ok(self.builder.string("")?);
        }
        let comma = self.item_separator()?;
        let (item, required) = items[0];
        let rest = &items[1..];

        let node = match (prefixed, required) {
            (true, true) => {
                // If we know we have preceeding elements, we can safely just add a (',' + e)
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[comma, item, rest_seq])?
            }
            (true, false) => {
                // If we know we have preceeding elements, we can safely just add an optional(',' + e)
                // TODO optimization: if the rest is all optional, we can nest the rest in the optional
                let comma_item = self.builder.join(&[comma, item])?;
                let optional_comma_item = self.builder.optional(comma_item)?;
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[optional_comma_item, rest_seq])?
            }
            (false, true) => {
                // No preceding elements, so we just add the element (no comma)
                let rest_seq = self.ordered_sequence(rest, true, cache)?;
                self.builder.join(&[item, rest_seq])?
            }
            (false, false) => {
                // No preceding elements, but our element is optional. If we add the element, the remaining
                // will be prefixed, else they are not.
                // TODO: same nested optimization as above
                let prefixed_rest = self.ordered_sequence(rest, true, cache)?;
                let unprefixed_rest = self.ordered_sequence(rest, false, cache)?;
                let opts = [self.builder.join(&[item, prefixed_rest])?, unprefixed_rest];
                self.builder.select(&opts)?
            }
        };
        self.builder.funding.try_insert(cache, (items, prefixed), node)?;
        Ok(node)
    }

    fn bounded_sequence(
        &mut self,
        item: NodeRef,
        min_elts: usize,
        max_elts: Option<usize>,
    ) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, NodeRef, usize, Option<usize>, usize, Option<usize>, NodeRef, NodeRef, NodeRef, [NodeRef; 3], Result<NodeRef>)>()?;
        let min_elts = min_elts.saturating_sub(1);
        let max_elts = max_elts.map(|v| v.saturating_sub(1));
        let comma = self.item_separator()?;
        let item_comma = self.builder.join(&[item, comma])?;
        let item_comma_rep = self.builder.repeat(item_comma, min_elts, max_elts)?;
        Ok(self.builder.join(&[item_comma_rep, item])?)
    }

    fn sequence(&mut self, item: NodeRef) -> Result<NodeRef> {
        let _frame = self.frame::<(&Self, NodeRef, NodeRef, NodeRef, NodeRef, [NodeRef; 2], Result<NodeRef>)>()?;
        let comma = self.item_separator()?;
        let item_comma = self.builder.join(&[item, comma])?;
        let item_comma_star = self.builder.zero_or_more(item_comma)?;
        Ok(self.builder.join(&[item_comma_star, item])?)
    }

    fn json_quote(&self, ast: RegexAst) -> Result<RegexAst> {
        let _frame = self.frame::<(&Self, RegexAst, String, JsonQuoteOptions, Box<RegexAst>, Result<RegexAst>)>()?;
        let allowed_escapes = self.builder.funding.try_copy_str(self.options.json_allowed_escapes.as_deref().unwrap_or(DEFAULT_JSON_ALLOWED_ESCAPES))?;
        Ok(RegexAst::JsonQuote(self.builder.funding.try_box(ast)?, JsonQuoteOptions { allowed_escapes, raw_mode: false }))
    }

    fn regex_compile(&mut self, schema: &Schema) -> Result<Option<RegexAst>> {
        let _frame = self.frame::<(&Self, &Schema, &Self, &str, Option<RegexAst>, Result<Option<RegexAst>>, StringSchema)>()?;
        let literal_regex = |rx: &str| -> Result<Option<RegexAst>> { Ok(Some(RegexAst::Literal(self.builder.funding.try_copy_str(rx)?))) };

        self.builder.check_limits()?;

        let r = match schema {
            Schema::Null => literal_regex("null")?,
            Schema::Boolean(None) => Some(RegexAst::Regex(self.builder.funding.try_copy_str("true|false")?)),
            Schema::Boolean(Some(value)) => literal_regex(if *value { "true" } else { "false" })?,

            Schema::Number(num) => Some(if num.integer {
                self.json_int(num)?
            } else {
                self.json_number(num)?
            }),

            Schema::String(opts) => return self.gen_json_string(opts.copy_with_funding(&self.builder.funding)?).map(Some),

            Schema::Any
            | Schema::Unsatisfiable(_)
            | Schema::Array(_)
            | Schema::Object(_)
            | Schema::AnyOf(_)
            | Schema::OneOf(_)
            | Schema::Ref(_) => None,
        };
        Ok(r)
    }

    fn length_regex(&self, min: usize, max: Option<usize>) -> Result<RegexAst> {
        let _frame = self.frame::<(&Self, usize, Option<usize>, String, Result<RegexAst>)>()?;
        Ok(RegexAst::Regex(match max {
            Some(max) => self.builder.funding.try_format(format_args!("(?s:.{{{min},{max}}})"))?,
            None => self.builder.funding.try_format(format_args!("(?s:.{{{min},}})"))?,
        }))
    }

    fn gen_json_string(&mut self, opts: StringSchema) -> Result<RegexAst> {
        let _frame = self.frame::<frames::StringValue<'_>>()?;
        let min_length = opts.min_length;
        let max_length = opts.max_length;
        if let Some(max_length) = max_length {
            if min_length > max_length {
                return Err(ParserError::cause(UnsatisfiableSchemaError {
                    message: self.builder.funding.try_format(format_args!(
                        "minLength ({min_length}) is greater than maxLength ({max_length})"
                    ))?,
                }, &self.builder.funding));
            }
        }
        if opts.regex.is_none() && self.options.json_allow_general_unicode_escapes {
            return self.json_general_unicode_string(min_length, max_length);
        }
        if min_length == 0 && max_length.is_none() && opts.regex.is_none() {
            return Ok(self.json_quote(RegexAst::Regex(self.builder.funding.try_copy_str("(?s:.*)")?))?);
        }
        if let Some(mut ast) = opts.regex {
            let mut positive = false;

            // special-case literals - the length is easy to check
            if let RegexAst::Literal(s) = &ast {
                let l = s.chars().count();

                if l < min_length || l > max_length.unwrap_or(usize::MAX) {
                    return Err(ParserError::cause(UnsatisfiableSchemaError {
                        message: self.builder.funding.try_format(format_args!("Constant {s:?} doesn't match length constraints"))?
                    }, &self.builder.funding));
                }

                positive = true;
            } else if min_length != 0 || max_length.is_some() {
                ast = RegexAst::And([Ok::<_, derivre::ParserError>(ast), Ok(self.length_regex(min_length, max_length)?)].into_iter().collect_with_funding(&self.builder.funding)?);
            } else {
                positive = always_non_empty(&ast, self.frames, &self.builder.funding)?;
                // eprintln!("positive:{} {}", positive, ast.display(1_000, None));
            }

            if !positive {
                // Check if the regex is empty
                let mut builder = derivre::RegexBuilder::new(self.builder.funding.clone())?;
                let expr = builder.mk(&ast)?;
                // if regex is not positive, do the more expensive non-emptiness check
                if !builder.exprset().is_positive(expr) {
                    // in JSB, 13 cases above 2000;
                    // 1 case above 5000:
                    // "format": "email",
                    // "pattern": "^\\w+([\\.-]?\\w+)*@\\w+([\\.-]?\\w+)*(\\.\\w{2,})+$",
                    // "minLength": 6,
                    //
                    // (excluding two handwritten examples with minLength:10000)
                    let mut regex = builder.to_regex_limited(expr, 10_000).map_err(|error| {
                        if crate::earley::is_grammar_storage_failure(&error) { return error; }
                        anyhow!(&self.builder.funding,
                            "Unable to determine if regex is empty: {}",
                            ast.display(1_000, None)
                        )
                    })?;
                    if regex.always_empty() {
                        return Err(ParserError::cause(UnsatisfiableSchemaError {
                            message: self.builder.funding.try_format(format_args!("Regex is empty: {}", ast.display(1_000, None)))?
                        }, &self.builder.funding));
                    }
                }
            }

            ast = self.json_quote(ast)?;

            Ok(ast)
        } else {
            self.json_quote(self.length_regex(min_length, max_length)?)
        }
    }

    /// Builds a JSON-string regex whose repetitions count decoded Unicode scalars.
    ///
    /// This is only valid for strings without regex constraints: printable escapes
    /// are not decoded before matching a schema pattern. Complete surrogate pairs
    /// are one repetition, while unpaired surrogates are rejected. The scalar
    /// expression is compiled once and reused across all string length bounds.
    fn json_general_unicode_string(
        &mut self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> Result<RegexAst> {
        let _frame = self.frame::<frames::Unicode<'_>>()?;
        let cache_key = (min_length, max_length);
        if let Some(expr) = self.general_unicode_string_cache.get(&cache_key) {
            return Ok(RegexAst::ExprRef(*expr));
        }

        let scalar = if let Some(expr) = self.general_unicode_scalar_cache {
            expr
        } else {
            let allowed_escapes = self
                .options
                .json_allowed_escapes
                .as_deref()
                .unwrap_or(DEFAULT_JSON_ALLOWED_ESCAPES);
            let mut short_escapes = String::new();
            let mut allow_unicode_escapes = false;

            for escape in allowed_escapes.chars() {
                match escape {
                    'u' => allow_unicode_escapes = true,
                    '\\' => self.builder.funding.try_push_str(&mut short_escapes, r"\\")?,
                    '"' | 'b' | 'f' | 'n' | 'r' | 't' => self.builder.funding.try_push_char(&mut short_escapes, escape)?,
                    _ => bail!(&self.builder.funding, "invalid escape character in allowed_escapes: {escape}"),
                }
            }

            let mut character_regex = self.builder.funding.try_copy_str(r#"[^"\\\x00-\x1F\x7F]"#)?;
            if !short_escapes.is_empty() {
                self.builder.funding.try_push_str(&mut character_regex, &self.builder.funding.try_format(format_args!(r"|\\[{short_escapes}]"))?)?;
            }
            if allow_unicode_escapes {
                self.builder.funding.try_push_char(&mut character_regex, '|')?;
                self.builder.funding.try_push_str(&mut character_regex, UNICODE_SCALAR_ESCAPE_REGEX)?;
            }

            let expr = self.builder.regex.regex(&character_regex)?;
            self.general_unicode_scalar_cache = Some(expr);
            expr
        };

        let min = u32::try_from(min_length)
            .map_err(|error| ParserError::cause(error, &self.builder.funding).context(format_args!("minLength ({min_length}) exceeds the supported range"), &self.builder.funding))?;
        let max = max_length
            .map(|max| {
                u32::try_from(max)
                    .map_err(|error| ParserError::cause(error, &self.builder.funding).context(format_args!("maxLength ({max}) exceeds the supported range"), &self.builder.funding))
            })
            .transpose()?
            .unwrap_or(u32::MAX);
        let ast = RegexAst::Concat([
            Ok::<_, derivre::ParserError>(RegexAst::Literal(self.builder.funding.try_copy_str("\"")?)),
            Ok(RegexAst::Repeat(self.builder.funding.try_box(RegexAst::ExprRef(scalar))?, min, max)),
            Ok(RegexAst::Literal(self.builder.funding.try_copy_str("\"")?)),
        ].into_iter().collect_with_funding(&self.builder.funding)?);
        let expr = self.builder.regex.add_ast(ast)?;
        self.builder.funding.try_insert(&mut self.general_unicode_string_cache, cache_key, expr)?;
        Ok(RegexAst::ExprRef(expr))
    }

    fn gen_json_array(&mut self, arr: &ArraySchema) -> Result<NodeRef> {
        let _frame = self.frame::<frames::Array<'_>>()?;
        let mut max_items = arr.max_items;
        let min_items = arr.min_items;

        if let Some(max_items) = max_items {
            if min_items > max_items {
                return Err(ParserError::cause(UnsatisfiableSchemaError {
                    message: self.builder.funding.try_format(format_args!(
                        "minItems ({min_items}) is greater than maxItems ({max_items})"
                    ))?,
                }, &self.builder.funding));
            }
        }

        let additional_item_grm = match self.gen_json(arr.items.schema_ref()) {
            Ok(node) => Some(node),
            Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                // If it's not an UnsatisfiableSchemaError, just propagate it normally
                None => return Err(e),
                // Item is optional; don't raise UnsatisfiableSchemaError
                Some(_) if arr.prefix_items.len() >= min_items => None,
                // Item is required; add context and propagate UnsatisfiableSchemaError
                Some(_) => {
                    return Err(e.context(UnsatisfiableSchemaError {
                        message: self.builder.funding.try_copy_str("required item is unsatisfiable")?,
                    }, &self.builder.funding));
                }
            },
        };

        let mut required_items = vec![];
        let mut optional_items = vec![];

        // If max_items is None, we can add an infinite tail of items later
        let n_to_add = max_items.map_or(arr.prefix_items.len().max(min_items), |max| max);

        for i in 0..n_to_add {
            let item = if i < arr.prefix_items.len() {
                match self.gen_json(&arr.prefix_items[i]) {
                    Ok(node) => node,
                    Err(e) => match e.downcast_ref::<UnsatisfiableSchemaError>() {
                        // If it's not an UnsatisfiableSchemaError, just propagate it normally
                        None => return Err(e),
                        // Item is optional; don't raise UnsatisfiableSchemaError.
                        // Set max_items to the current index, as we can't satisfy any more items.
                        Some(_) if i >= min_items => {
                            max_items = Some(i);
                            break;
                        }
                        // Item is required; add context and propagate UnsatisfiableSchemaError
                        Some(_) => {
                            return Err(e.context(UnsatisfiableSchemaError {
                                message: self.builder.funding.try_format(format_args!(
                                    "prefixItems[{i}] is unsatisfiable but minItems is {min_items}"
                                ))?,
                            }, &self.builder.funding));
                        }
                    },
                }
            } else if let Some(compiled) = &additional_item_grm {
                *compiled
            } else {
                break;
            };

            if i < min_items {
                self.builder.funding.clone().try_push(&mut required_items, item)?;
            } else {
                self.builder.funding.clone().try_push(&mut optional_items, item)?;
            }
        }

        if max_items.is_none() {
            if let Some(additional_item) = additional_item_grm {
                // Add an infinite tail of items
                self.builder.funding.clone().try_push(&mut optional_items, self.sequence(additional_item)?)?;
            }
        }

        let mut grammars: Vec<NodeRef> = Vec::new();
        self.builder.funding.clone().try_push(&mut grammars, self.builder.string("[")?)?;
        let comma = self.item_separator()?;

        if !required_items.is_empty() {
            self.builder.funding.clone().try_push(&mut grammars, required_items[0])?;
            for item in &required_items[1..] {
                self.builder.funding.clone().try_push(&mut grammars, comma)?;
                self.builder.funding.clone().try_push(&mut grammars, *item)?;
            }
        }

        if !optional_items.is_empty() {
            let first = optional_items[0];
            let mut tail = self.builder.empty()?;
            for item in optional_items.into_iter().skip(1).rev() {
                let joined = self.builder.join(&[comma, item, tail])?;
                tail = self.builder.optional(joined)?;
            }
            let tail = self.builder.join(&[first, tail])?;

            if !required_items.is_empty() {
                let j = self.builder.join(&[comma, tail])?;
                self.builder.funding.clone().try_push(&mut grammars, self.builder.optional(j)?)?;
            } else {
                self.builder.funding.clone().try_push(&mut grammars, self.builder.optional(tail)?)?;
            }
        }

        self.builder.funding.clone().try_push(&mut grammars, self.builder.string("]")?)?;
        Ok(self.builder.join(&grammars)?)
    }
}

/// Build the `multipleOf` regex constraint with sign handling.
///
/// derivre's `RegexAst::MultipleOf` matches only the unsigned numeric literal
/// (the digits, plus a decimal point for non-integer `multipleOf`); it has no
/// transition for a leading `-`. Allow an optional sign here, otherwise
/// negative multiples (e.g. `-6` for `multipleOf 3`) would be rejected.
/// Divisibility is independent of sign.
/// https://github.com/guidance-ai/llguidance/issues/222
fn signed_multiple_of_ast(coef: u32, exp: u32,
    scope: &derivre::prepared_funding::Scope<'_, ParserAllocationFunding>,
    funding: &ParserAllocationFunding) -> Result<RegexAst> {
    let _frame = frames::enter::<(u32, u32, &ParserAllocationFunding, RegexAst,
        [Result<RegexAst>; 2], std::array::IntoIter<Result<RegexAst>, 2>, Result<RegexAst>)>(scope, funding)?;
    Ok(RegexAst::Concat([
        Ok::<_, derivre::ParserError>(RegexAst::Regex(funding.try_copy_str("-?")?)),
        Ok(RegexAst::MultipleOf(coef, exp)),
    ].into_iter().collect_with_funding(funding)?))
}

fn always_non_empty(ast: &RegexAst,
    scope: &derivre::prepared_funding::Scope<'_, ParserAllocationFunding>,
    funding: &ParserAllocationFunding) -> Result<bool> {
    let _frame = frames::enter::<(&RegexAst, &ParserAllocationFunding,
        std::slice::Iter<'_, RegexAst>, &RegexAst, bool, Result<bool>)>(scope, funding)?;
    Ok(match ast {
        RegexAst::Or(asts) => {
            for ast in asts { if always_non_empty(ast, scope, funding)? { return Ok(true); } }
            false
        }
        RegexAst::Concat(asts) => {
            for ast in asts { if !always_non_empty(ast, scope, funding)? { return Ok(false); } }
            true
        }
        RegexAst::Repeat(ast, _, _) | RegexAst::JsonQuote(ast, _) | RegexAst::LookAhead(ast) => {
            always_non_empty(ast, scope, funding)?
        }

        RegexAst::EmptyString
        | RegexAst::Literal(_)
        | RegexAst::ByteLiteral(_)
        | RegexAst::Byte(_)
        | RegexAst::ByteSet(_)
        | RegexAst::MultipleOf(_, _) => true,

        RegexAst::And(_)
        | RegexAst::Not(_)
        | RegexAst::NoMatch
        | RegexAst::Regex(_)
        | RegexAst::SearchRegex(_)
        | RegexAst::ExprRef(_) => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ParserLimits;
    use serde_json::json;

    #[test]
    fn json_quote_default_allows_u_escape() {
        let funding = ParserAllocationFunding::unenforced();
        let scope = derivre::prepared_funding::Scope::new(&funding).unwrap();
        let compiler = Compiler::new(
            JsonCompileOptions::default(),
            GrammarBuilder::new(
                None,
                ParserLimits::default(),
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
            &scope,
        ).unwrap();
        let quoted = compiler.json_quote(RegexAst::EmptyString).unwrap();
        match quoted {
            RegexAst::JsonQuote(_, opts) => {
                assert_eq!(opts.allowed_escapes, DEFAULT_JSON_ALLOWED_ESCAPES)
            }
            _ => panic!("expected JsonQuote AST"),
        }
        assert_eq!(
            DEFAULT_JSON_ALLOWED_ESCAPES,
            JsonQuoteOptions::regular().allowed_escapes
        );
    }

    #[test]
    fn json_quote_uses_override_from_options() {
        let funding = ParserAllocationFunding::unenforced();
        let scope = derivre::prepared_funding::Scope::new(&funding).unwrap();
        let compiler = Compiler::new(
            JsonCompileOptions {
                json_allowed_escapes: Some("nrbtf\\\"".to_string()),
                ..Default::default()
            },
            GrammarBuilder::new(
                None,
                ParserLimits::default(),
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
            &scope,
        ).unwrap();
        let quoted = compiler.json_quote(RegexAst::EmptyString).unwrap();
        match quoted {
            RegexAst::JsonQuote(_, opts) => assert_eq!(opts.allowed_escapes, "nrbtf\\\""),
            _ => panic!("expected JsonQuote AST"),
        }
    }

    /// Reuses one Unicode-scalar expression while keeping different string bounds distinct.
    #[test]
    fn general_unicode_strings_reuse_compiled_regex() {
        let funding = ParserAllocationFunding::unenforced();
        let scope = derivre::prepared_funding::Scope::new(&funding).unwrap();
        let mut compiler = Compiler::new(
            JsonCompileOptions::default(),
            GrammarBuilder::new(
                None,
                ParserLimits::default(),
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
            &scope,
        ).unwrap();

        let unbounded = compiler.json_general_unicode_string(0, None).unwrap();
        let scalar = compiler.general_unicode_scalar_cache.unwrap();
        let repeated = compiler.json_general_unicode_string(0, None).unwrap();
        let bounded = compiler.json_general_unicode_string(1, Some(1)).unwrap();

        match (unbounded, repeated, bounded) {
            (RegexAst::ExprRef(first), RegexAst::ExprRef(second), RegexAst::ExprRef(third)) => {
                assert_eq!(first, second);
                assert_ne!(first, third);
            }
            _ => panic!("expected cached regex expression references"),
        }
        assert_eq!(compiler.general_unicode_scalar_cache, Some(scalar));
        assert_eq!(compiler.general_unicode_string_cache.len(), 2);
    }

    /// Returns invalid escape-policy errors for every schema shape instead of panicking.
    #[test]
    fn invalid_json_allowed_escapes_propagate_for_all_schema_shapes() {
        for allow_general_unicode_escapes in [false, true] {
            let options = JsonCompileOptions {
                json_allowed_escapes: Some("x".to_string()),
                json_allow_general_unicode_escapes: allow_general_unicode_escapes,
                ..JsonCompileOptions::default()
            };

            for schema in [
                json!({ "type": "string" }),
                json!({ "type": "object" }),
                json!({}),
            ] {
                let error = options
                    .json_to_llg(
                        GrammarBuilder::new(
                            None,
                            ParserLimits::default(),
                            derivre::ParserAllocationFunding::unenforced(),
                        )
                        .unwrap(),
                        schema,
                    )
                    .err()
                    .expect("invalid JSON escape settings should return an error");
                assert!(
                    error
                        .to_string()
                        .contains("invalid escape character in allowed_escapes: x"),
                    "unexpected error: {error}"
                );
            }
        }
    }

    #[test]
    fn x_guidance_accepts_json_allowed_escapes() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "x": { "type": "string" }
            },
            "required": ["x"],
            "additionalProperties": false,
            "x-guidance": {
                "json_allowed_escapes": "nrbtf\\\""
            }
        });

        let builder = GrammarBuilder::new(
            None,
            ParserLimits::default(),
            derivre::ParserAllocationFunding::unenforced(),
        )
        .unwrap();
        JsonCompileOptions::default()
            .json_to_llg_with_overrides(builder, schema.clone())
            .expect("schema with x-guidance override should compile");

        // Ensure callers can still stamp options back into schema.
        JsonCompileOptions {
            json_allowed_escapes: Some("nrbtf\\\"".to_string()),
            ..Default::default()
        }
        .apply_to(&mut schema, &ParserAllocationFunding::unenforced()).unwrap();
        assert_eq!(
            schema["x-guidance"]["json_allowed_escapes"],
            Value::String("nrbtf\\\"".to_string())
        );
    }
}
