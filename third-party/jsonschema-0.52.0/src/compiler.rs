use crate::{
    content_encoding::ContentEncodingSource,
    content_media_type::ContentMediaTypeSource,
    keywords::{
        self,
        custom::{CustomKeyword, KeywordFactory},
        format::Format,
        unevaluated_items::PendingItemsValidators,
        unevaluated_properties::PendingPropertyValidators,
        BoxedValidator, BuiltinKeyword, Keyword,
    },
    node::{PendingSchemaNode, SchemaNode},
    options::{PatternEngineOptions, ValidationOptions},
    paths::{Location, LocationSegment},
    types::{JsonType, JsonTypeSet},
    validator::Validate,
    Json, SerdeJson, ValidationError, Validator, ValidatorMap,
};
use ahash::{AHashMap, AHashSet};
use referencing::{
    uri, write_escaped_str, Draft, List, Registry, Resolved, Resolver, ResourceRef, Uri,
    Vocabulary, VocabularySet,
};
use serde_json::{Map, Value};
use std::{borrow::Cow, cell::RefCell, fmt, rc::Rc, sync::Arc};

const DEFAULT_SCHEME: &str = "json-schema";
pub(crate) const DEFAULT_BASE_URI: &str = "json-schema:///";

pub(crate) const fn formats_are_assertions_by_default(draft: Draft) -> bool {
    matches!(draft, Draft::Draft4 | Draft::Draft6 | Draft::Draft7)
}

/// Type alias for shared cache maps in compiler state.
type CompilerMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type CompilerSet<K> = hashbrown::HashSet<K, ahash::RandomState>;
type SharedCache<K, V> = Rc<RefCell<CompilerMap<K, V>>>;
/// Type alias for shared sets in compiler state.
type SharedSet<T> = Rc<RefCell<CompilerSet<T>>>;

pub(crate) trait CompilationOptions<F: Json> {
    fn validate_formats(&self) -> Option<bool>;
    fn declares_vocabulary(&self, uri: &str) -> bool;
    fn are_unknown_formats_ignored(&self) -> bool;
    fn get_content_media_type_check(&self, media_type: &str) -> Option<ContentMediaTypeSource>;
    fn get_content_encoding(&self, encoding: &str) -> Option<ContentEncodingSource>;
    fn get_keyword_factory(&self, name: &str) -> Option<&Arc<dyn KeywordFactory<F>>>;
    fn get_format(&self, format: &str) -> Option<(&String, &Arc<dyn Format>)>;
    fn pattern_options(&self) -> PatternEngineOptions;
    fn email_options(&self) -> Option<&email_address::Options>;
}

impl<R, F: Json> CompilationOptions<F> for ValidationOptions<'_, R, F> {
    fn validate_formats(&self) -> Option<bool> {
        ValidationOptions::validate_formats(self)
    }

    fn declares_vocabulary(&self, uri: &str) -> bool {
        ValidationOptions::declares_vocabulary(self, uri)
    }

    fn are_unknown_formats_ignored(&self) -> bool {
        ValidationOptions::are_unknown_formats_ignored(self)
    }

    fn get_content_media_type_check(&self, media_type: &str) -> Option<ContentMediaTypeSource> {
        ValidationOptions::get_content_media_type_check(self, media_type)
    }

    fn get_content_encoding(&self, encoding: &str) -> Option<ContentEncodingSource> {
        ValidationOptions::get_content_encoding(self, encoding)
    }

    fn get_keyword_factory(&self, name: &str) -> Option<&Arc<dyn KeywordFactory<F>>> {
        ValidationOptions::get_keyword_factory(self, name)
    }

    fn get_format(&self, format: &str) -> Option<(&String, &Arc<dyn Format>)> {
        ValidationOptions::get_format(self, format)
    }

    fn pattern_options(&self) -> PatternEngineOptions {
        ValidationOptions::compiler_pattern_options(self)
    }

    fn email_options(&self) -> Option<&email_address::Options> {
        ValidationOptions::compiler_email_options(self)
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) struct LocationCacheKey {
    pub(crate) base_uri: Arc<Uri<String>>,
    location: Arc<str>,
    dynamic_scope: List<Uri<String>>,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct PropertyValidatorsPendingKey {
    schema_ptr: usize,
}

impl PropertyValidatorsPendingKey {
    fn new(schema: &Map<String, Value>) -> Self {
        Self {
            schema_ptr: std::ptr::from_ref(schema) as usize,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct ItemsValidatorsPendingKey {
    schema_ptr: usize,
}

impl ItemsValidatorsPendingKey {
    fn new(schema: &Map<String, Value>) -> Self {
        Self {
            schema_ptr: std::ptr::from_ref(schema) as usize,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) struct AliasCacheKey {
    uri: Arc<Uri<String>>,
    dynamic_scope: List<Uri<String>>,
}

/// Shared caches reused across every `Context` derived from a schema root.
struct SharedContextState<F: Json = SerdeJson> {
    seen: SharedSet<Arc<Uri<String>>>,
    location_nodes: SharedCache<LocationCacheKey, SchemaNode<F>>,
    alias_nodes: SharedCache<AliasCacheKey, SchemaNode<F>>,
    pending_nodes: SharedCache<LocationCacheKey, PendingSchemaNode<F>>,
    alias_placeholders: SharedCache<Arc<Uri<String>>, PendingSchemaNode<F>>,
    pending_property_validators: SharedCache<LocationCacheKey, PendingPropertyValidators<F>>,
    pending_property_validators_by_schema:
        SharedCache<PropertyValidatorsPendingKey, PendingPropertyValidators<F>>,
    pending_items_validators: SharedCache<LocationCacheKey, PendingItemsValidators<F>>,
    pending_items_validators_by_schema:
        SharedCache<ItemsValidatorsPendingKey, PendingItemsValidators<F>>,
    pattern_cache: SharedCache<Arc<str>, PatternCacheEntry>,
    uri_buffer: Rc<RefCell<String>>,
}

impl<F: Json> Clone for SharedContextState<F> {
    fn clone(&self) -> Self {
        Self {
            seen: self.seen.clone(),
            location_nodes: self.location_nodes.clone(),
            alias_nodes: self.alias_nodes.clone(),
            pending_nodes: self.pending_nodes.clone(),
            alias_placeholders: self.alias_placeholders.clone(),
            pending_property_validators: self.pending_property_validators.clone(),
            pending_property_validators_by_schema: self
                .pending_property_validators_by_schema
                .clone(),
            pending_items_validators: self.pending_items_validators.clone(),
            pending_items_validators_by_schema: self.pending_items_validators_by_schema.clone(),
            pattern_cache: self.pattern_cache.clone(),
            uri_buffer: self.uri_buffer.clone(),
        }
    }
}

impl<F: Json> fmt::Debug for SharedContextState<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedContextState").finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
struct PatternCacheEntry {
    translated: Arc<str>,
    fancy: Option<Arc<fancy_regex::Regex>>,
    standard: Option<Arc<regex::Regex>>,
}

impl<F: Json> SharedContextState<F> {
    /// `capacity` pre-sizes the per-location node cache to avoid rehashing during a build.
    fn new(
        capacity: usize,
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        let mut location_nodes = CompilerMap::with_hasher(funding.random_state()?);
        funding.reserve_map(&mut location_nodes, capacity)?;
        Ok(Self {
            seen: funding.rc(RefCell::new(CompilerSet::with_hasher(
                funding.random_state()?,
            )))?,
            location_nodes: funding.rc(RefCell::new(location_nodes))?,
            alias_nodes: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            pending_nodes: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            alias_placeholders: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            pending_property_validators: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            pending_property_validators_by_schema: funding.rc(RefCell::new(
                CompilerMap::with_hasher(funding.random_state()?),
            ))?,
            pending_items_validators: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            pending_items_validators_by_schema: funding.rc(RefCell::new(
                CompilerMap::with_hasher(funding.random_state()?),
            ))?,
            pattern_cache: funding.rc(RefCell::new(CompilerMap::with_hasher(
                funding.random_state()?,
            )))?,
            uri_buffer: funding.rc(RefCell::new(String::new()))?,
        })
    }
}

/// Per-location view used while compiling schemas into validators.
pub(crate) struct Context<'a, F: Json = SerdeJson> {
    config: &'a dyn CompilationOptions<F>,
    resolver: Resolver<'a>,
    vocabularies: VocabularySet,
    location: Location,
    /// The location where the current resource starts.
    ///
    /// When compiling a schema reached via `$ref`, this is set to the `$ref` target location.
    /// Used to compute the "suffix" (path relative to resource root) for evaluation paths.
    ///
    /// # Example
    ///
    /// ```text
    /// Schema: { "$ref": "#/$defs/Item", "$defs": { "Item": { "type": "string" } } }
    ///
    /// When compiling the "type" keyword inside "Item":
    ///   location      = /$defs/Item/type
    ///   resource_base = /$defs/Item
    ///   suffix()      = /type
    /// ```
    resource_base: Location,
    pub(crate) draft: Draft,
    shared: SharedContextState<F>,
    funding: crate::compilation::Funding,
}

impl<'a, F: Json> Context<'a, F> {
    pub(crate) fn new(
        config: &'a dyn CompilationOptions<F>,
        resolver: Resolver<'a>,
        vocabularies: VocabularySet,
        draft: Draft,
        location: Location,
        capacity: usize,
        funding: crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        Ok(Context {
            config,
            resolver,
            resource_base: location.clone(),
            location,
            vocabularies,
            draft,
            shared: SharedContextState::new(capacity, &funding)?,
            funding,
        })
    }
    pub(crate) fn funding(&self) -> &crate::compilation::Funding {
        &self.funding
    }
    pub(crate) fn draft(&self) -> Draft {
        self.draft
    }
    pub(crate) fn config(&self) -> &dyn CompilationOptions<F> {
        self.config
    }

    /// Create a context for this schema.
    pub(crate) fn in_subresource(
        &'a self,
        resource: ResourceRef<'_>,
    ) -> Result<Context<'a, F>, crate::compilation::CompileError<'static>> {
        let resolver = self.resolver.in_subresource(resource)?;
        Ok(Context {
            config: self.config,
            resolver,
            vocabularies: self
                .vocabularies
                .try_clone_with_allocations(&self.funding)
                .map_err(|error| self.funding.reference_error(error))?,
            draft: resource.draft(),
            resource_base: self.resource_base.clone(),
            location: self.location.clone(),
            shared: self.shared.clone(),
            funding: self.funding.clone(),
        })
    }
    pub(crate) fn as_resource_ref<'r>(&'a self, contents: &'r Value) -> ResourceRef<'r> {
        self.draft.detect(contents).create_resource_ref(contents)
    }

    #[inline]
    pub(crate) fn new_at_location(
        &'a self,
        chunk: impl Into<LocationSegment<'a>>,
    ) -> Result<Self, crate::compilation::CompileError<'static>> {
        let location = self.location.join_with_funding(chunk, &self.funding)?;
        Ok(Context {
            config: self.config,
            resolver: self.resolver.clone(),
            vocabularies: self
                .vocabularies
                .try_clone_with_allocations(&self.funding)
                .map_err(|error| self.funding.reference_error(error))?,
            resource_base: self.resource_base.clone(),
            location,
            draft: self.draft,
            shared: self.shared.clone(),
            funding: self.funding.clone(),
        })
    }
    pub(crate) fn lookup(&'a self, reference: &str) -> Result<Resolved<'a>, referencing::Error> {
        self.resolver.lookup(reference)
    }

    pub(crate) fn location_cache_key(&self) -> LocationCacheKey {
        LocationCacheKey {
            base_uri: self.resolver.base_uri(),
            location: self.location.as_arc(),
            dynamic_scope: self.resolver.dynamic_scope(),
        }
    }

    fn alias_cache_key(&self, alias: Arc<Uri<String>>) -> AliasCacheKey {
        AliasCacheKey {
            uri: alias,
            dynamic_scope: self.resolver.dynamic_scope(),
        }
    }

    pub(crate) fn base_uri(&self) -> Option<Arc<Uri<String>>> {
        let base_uri = self.resolver.base_uri();
        if base_uri.scheme().as_str() == DEFAULT_SCHEME {
            None
        } else {
            Some(base_uri)
        }
    }

    pub(crate) fn absolute_location(
        &self,
        location: &Location,
    ) -> Result<Option<Arc<Uri<String>>>, crate::CompilationError> {
        let Some(base) = self.base_uri() else {
            return Ok(None);
        };
        let mut buffer = self.shared.uri_buffer.borrow_mut();
        buffer.clear();
        uri::encode_to_with_allocations(location.as_str(), &mut buffer, &self.funding)
            .map_err(|error| self.funding.reference_allocation(error))?;
        let resolved = base
            .with_fragment_with_allocations(
                Some(uri::EncodedString::new_or_panic(&buffer)),
                &self.funding,
            )
            .map_err(|error| self.funding.reference_allocation(error))?;
        buffer.clear();
        Ok(Some(self.funding.arc(resolved)?))
    }

    fn translated_pattern(
        &self,
        pattern: &str,
    ) -> Result<Arc<str>, crate::compilation::PatternError> {
        if let Some(entry) = self.shared.pattern_cache.borrow().get(pattern) {
            return Ok(Arc::clone(&entry.translated));
        }
        let translated = jsonschema_regex::to_rust_regex_with_allocations(pattern, &self.funding)
            .map_err(|error| match error {
            jsonschema_regex::allocation::TranslationError::Syntax => {
                crate::compilation::PatternError::Syntax
            }
            jsonschema_regex::allocation::TranslationError::Allocation(error) => {
                crate::compilation::PatternError::Storage(self.funding.syntax_allocation(error))
            }
        })?;
        let translated = self.funding.arc_str(translated.as_ref())?;
        let key = self.funding.arc_str(pattern)?;
        self.funding.insert(
            &mut self.shared.pattern_cache.borrow_mut(),
            key,
            PatternCacheEntry {
                translated: Arc::clone(&translated),
                fancy: None,
                standard: None,
            },
        )?;
        Ok(translated)
    }

    fn is_known_keyword(&self, keyword: &str) -> bool {
        self.draft.is_known_keyword(keyword)
    }
    pub(crate) fn supports_adjacent_validation(&self) -> bool {
        !matches!(self.draft, Draft::Draft4 | Draft::Draft6 | Draft::Draft7)
    }
    pub(crate) fn supports_integer_valued_numbers(&self) -> bool {
        !matches!(self.draft, Draft::Draft4)
    }
    pub(crate) fn validates_formats_by_default(&self) -> bool {
        self.config.validate_formats().unwrap_or_else(|| {
            self.asserts_formats_by_dialect() || formats_are_assertions_by_default(self.draft)
        })
    }
    pub(crate) fn are_unknown_formats_ignored(&self) -> bool {
        !self.asserts_formats_by_dialect() && self.config.are_unknown_formats_ignored()
    }
    /// The meta-schema requires `format` to be an assertion.
    pub(crate) fn asserts_formats_by_dialect(&self) -> bool {
        // Not `has_vocabulary`: that reports every vocabulary as present below Draft 2019-09.
        // Draft 2019-09 writes the requirement as its Format vocabulary declared `true`.
        self.vocabularies.contains(&Vocabulary::FormatAssertion)
            || self.vocabularies.contains(&Vocabulary::Format)
    }
    /// Enter a referenced resource, which may follow a meta-schema of its own.
    ///
    /// # Errors
    ///
    /// That meta-schema requires a vocabulary this crate does not implement.
    pub(crate) fn with_resolver_and_draft(
        &'a self,
        resolver: Resolver<'a>,
        draft: Draft,
        vocabularies: VocabularySet,
        resource_base: Location,
    ) -> Result<Context<'a, F>, crate::compilation::CompileError<'static>> {
        ensure_vocabularies_supported(self.config, &vocabularies, &self.funding)?;
        Ok(Context {
            config: self.config,
            resolver,
            draft,
            vocabularies,
            location: resource_base.clone(),
            resource_base,
            shared: self.shared.clone(),
            funding: self.funding.clone(),
        })
    }
    pub(crate) fn get_content_media_type_check(
        &self,
        media_type: &str,
    ) -> Option<ContentMediaTypeSource> {
        self.config.get_content_media_type_check(media_type)
    }
    pub(crate) fn get_content_encoding(&self, encoding: &str) -> Option<ContentEncodingSource> {
        self.config.get_content_encoding(encoding)
    }
    pub(crate) fn get_keyword_factory(&self, name: &str) -> Option<&Arc<dyn KeywordFactory<F>>> {
        self.config.get_keyword_factory(name)
    }
    pub(crate) fn get_format(&self, format: &str) -> Option<(&String, &Arc<dyn Format>)> {
        self.config.get_format(format)
    }
    pub(crate) fn is_circular_reference(
        &self,
        reference: &str,
    ) -> Result<bool, referencing::Error> {
        let uri = self
            .resolver
            .resolve_uri(&self.resolver.base_uri().borrow(), reference)?;
        Ok(self.shared.seen.borrow().contains(&*uri))
    }
    pub(crate) fn mark_seen(
        &self,
        reference: &str,
    ) -> Result<(), crate::compilation::CompileError<'static>> {
        let uri = self
            .resolver
            .resolve_uri(&self.resolver.base_uri().borrow(), reference)?;
        self.funding
            .insert_set(&mut self.shared.seen.borrow_mut(), uri)?;
        Ok(())
    }

    pub(crate) fn lookup_recursive_reference(&self) -> Result<Resolved<'_>, referencing::Error> {
        self.resolver.lookup_recursive_ref()
    }
    pub(crate) fn resolve_reference_uri(
        &self,
        reference: &str,
    ) -> Result<Arc<Uri<String>>, referencing::Error> {
        self.resolver
            .resolve_uri(&self.resolver.base_uri().borrow(), reference)
    }

    pub(crate) fn cached_location_node(&self, key: &LocationCacheKey) -> Option<SchemaNode<F>> {
        self.shared.location_nodes.borrow().get(key).cloned()
    }

    pub(crate) fn cache_location_node(
        &self,
        key: LocationCacheKey,
        node: SchemaNode<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding
            .insert(&mut self.shared.location_nodes.borrow_mut(), key, node)?;
        Ok(())
    }

    pub(crate) fn cached_alias_node(&self, key: &AliasCacheKey) -> Option<SchemaNode<F>> {
        self.shared.alias_nodes.borrow().get(key).cloned()
    }

    pub(crate) fn cache_alias_node(
        &self,
        key: AliasCacheKey,
        node: SchemaNode<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding
            .insert(&mut self.shared.alias_nodes.borrow_mut(), key, node)?;
        Ok(())
    }

    pub(crate) fn cached_pending_location_node(
        &self,
        key: &LocationCacheKey,
    ) -> Option<PendingSchemaNode<F>> {
        self.shared.pending_nodes.borrow().get(key).cloned()
    }

    pub(crate) fn cache_pending_location_node(
        &self,
        key: LocationCacheKey,
        node: PendingSchemaNode<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding
            .insert(&mut self.shared.pending_nodes.borrow_mut(), key, node)?;
        Ok(())
    }

    pub(crate) fn remove_pending_location_node(&self, key: &LocationCacheKey) {
        self.shared.pending_nodes.borrow_mut().remove(key);
    }

    pub(crate) fn get_pending_property_validators(
        &self,
        key: &LocationCacheKey,
    ) -> Option<PendingPropertyValidators<F>> {
        self.shared
            .pending_property_validators
            .borrow()
            .get(key)
            .cloned()
    }

    pub(crate) fn cache_pending_property_validators(
        &self,
        key: LocationCacheKey,
        pending: PendingPropertyValidators<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding.insert(
            &mut self.shared.pending_property_validators.borrow_mut(),
            key,
            pending,
        )?;
        Ok(())
    }

    pub(crate) fn remove_pending_property_validators(&self, key: &LocationCacheKey) {
        self.shared
            .pending_property_validators
            .borrow_mut()
            .remove(key);
    }

    fn property_schema_key(schema: &Map<String, Value>) -> PropertyValidatorsPendingKey {
        PropertyValidatorsPendingKey::new(schema)
    }

    pub(crate) fn get_pending_property_validators_for_schema(
        &self,
        schema: &Map<String, Value>,
    ) -> Option<PendingPropertyValidators<F>> {
        let key = Self::property_schema_key(schema);
        self.shared
            .pending_property_validators_by_schema
            .borrow()
            .get(&key)
            .cloned()
    }

    pub(crate) fn cache_pending_property_validators_for_schema(
        &self,
        schema: &Map<String, Value>,
        pending: PendingPropertyValidators<F>,
    ) -> Result<(), crate::CompilationError> {
        let key = Self::property_schema_key(schema);
        self.funding.insert(
            &mut self
                .shared
                .pending_property_validators_by_schema
                .borrow_mut(),
            key,
            pending,
        )?;
        Ok(())
    }

    pub(crate) fn remove_pending_property_validators_for_schema(
        &self,
        schema: &Map<String, Value>,
    ) {
        let key = Self::property_schema_key(schema);
        self.shared
            .pending_property_validators_by_schema
            .borrow_mut()
            .remove(&key);
    }

    fn items_schema_key(schema: &Map<String, Value>) -> ItemsValidatorsPendingKey {
        ItemsValidatorsPendingKey::new(schema)
    }

    pub(crate) fn get_pending_items_validators_for_schema(
        &self,
        schema: &Map<String, Value>,
    ) -> Option<PendingItemsValidators<F>> {
        let key = Self::items_schema_key(schema);
        self.shared
            .pending_items_validators_by_schema
            .borrow()
            .get(&key)
            .cloned()
    }

    pub(crate) fn get_pending_items_validators(
        &self,
        key: &LocationCacheKey,
    ) -> Option<PendingItemsValidators<F>> {
        self.shared
            .pending_items_validators
            .borrow()
            .get(key)
            .cloned()
    }

    pub(crate) fn cache_pending_items_validators(
        &self,
        key: LocationCacheKey,
        pending: PendingItemsValidators<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding.insert(
            &mut self.shared.pending_items_validators.borrow_mut(),
            key,
            pending,
        )?;
        Ok(())
    }

    pub(crate) fn remove_pending_items_validators(&self, key: &LocationCacheKey) {
        self.shared
            .pending_items_validators
            .borrow_mut()
            .remove(key);
    }

    pub(crate) fn cache_pending_items_validators_for_schema(
        &self,
        schema: &Map<String, Value>,
        pending: PendingItemsValidators<F>,
    ) -> Result<(), crate::CompilationError> {
        let key = Self::items_schema_key(schema);
        self.funding.insert(
            &mut self.shared.pending_items_validators_by_schema.borrow_mut(),
            key,
            pending,
        )?;
        Ok(())
    }

    pub(crate) fn remove_pending_items_validators_for_schema(&self, schema: &Map<String, Value>) {
        let key = Self::items_schema_key(schema);
        self.shared
            .pending_items_validators_by_schema
            .borrow_mut()
            .remove(&key);
    }

    pub(crate) fn cached_alias_placeholder(
        &self,
        alias: &Arc<Uri<String>>,
    ) -> Option<PendingSchemaNode<F>> {
        self.shared.alias_placeholders.borrow().get(alias).cloned()
    }

    pub(crate) fn set_alias_placeholder(
        &self,
        alias: Arc<Uri<String>>,
        node: PendingSchemaNode<F>,
    ) -> Result<(), crate::CompilationError> {
        self.funding.insert(
            &mut self.shared.alias_placeholders.borrow_mut(),
            alias,
            node,
        )?;
        Ok(())
    }

    pub(crate) fn remove_alias_placeholder(&self, alias: &Arc<Uri<String>>) {
        self.shared.alias_placeholders.borrow_mut().remove(alias);
    }

    /// Get a cached compiled regex, or compile and cache it if not present.
    pub(crate) fn get_or_compile_regex(
        &self,
        pattern: &str,
    ) -> Result<Arc<fancy_regex::Regex>, crate::compilation::PatternError> {
        let translated = self.translated_pattern(pattern)?;
        {
            let cache = self.shared.pattern_cache.borrow();
            if let Some(entry) = cache.get(pattern) {
                if let Some(regex) = &entry.fancy {
                    return Ok(Arc::clone(regex));
                }
            }
        }

        let (backtrack_limit, size_limit, dfa_size_limit) = match self.config.pattern_options() {
            PatternEngineOptions::FancyRegex {
                backtrack_limit,
                size_limit,
                dfa_size_limit,
            } => (backtrack_limit, size_limit, dfa_size_limit),
            PatternEngineOptions::Regex { .. } => (None, None, None),
        };

        let regex = self
            .funding
            .arc(crate::regex::build_fancy_regex_with_funding(
                translated.as_ref(),
                backtrack_limit,
                size_limit,
                dfa_size_limit,
                &self.funding,
            )?)?;

        if let Some(entry) = self.shared.pattern_cache.borrow_mut().get_mut(pattern) {
            entry.fancy = Some(Arc::clone(&regex));
        }

        Ok(regex)
    }

    /// Compile the default profile used by unevaluated-property tracking.
    pub(crate) fn compile_default_regex(
        &self,
        pattern: &str,
    ) -> Result<Arc<fancy_regex::Regex>, crate::compilation::PatternError> {
        let translated = self.translated_pattern(pattern)?;
        let regex = crate::regex::build_fancy_regex_with_funding(
            &translated,
            None,
            None,
            None,
            &self.funding,
        )?;
        Ok(self.funding.arc(regex)?)
    }

    /// Get a cached compiled standard regex, or compile and cache it if not present.
    pub(crate) fn get_or_compile_standard_regex(
        &self,
        pattern: &str,
    ) -> Result<Arc<regex::Regex>, crate::compilation::PatternError> {
        let translated = self.translated_pattern(pattern)?;
        {
            let cache = self.shared.pattern_cache.borrow();
            if let Some(entry) = cache.get(pattern) {
                if let Some(regex) = &entry.standard {
                    return Ok(Arc::clone(regex));
                }
            }
        }

        let (size_limit, dfa_size_limit) = match self.config.pattern_options() {
            PatternEngineOptions::Regex {
                size_limit,
                dfa_size_limit,
            } => (size_limit, dfa_size_limit),
            PatternEngineOptions::FancyRegex { .. } => (None, None),
        };

        let regex = self
            .funding
            .arc(crate::regex::build_standard_regex_with_funding(
                translated.as_ref(),
                size_limit,
                dfa_size_limit,
                &self.funding,
            )?)?;

        if let Some(entry) = self.shared.pattern_cache.borrow_mut().get_mut(pattern) {
            entry.standard = Some(Arc::clone(&regex));
        }

        Ok(regex)
    }

    /// Lookup a reference that is potentially recursive and return already
    /// compiled nodes when available.
    pub(crate) fn lookup_maybe_recursive(
        &self,
        reference: &str,
    ) -> Result<Option<Box<dyn Validate<F>>>, crate::compilation::CompileError<'static>> {
        if self.is_circular_reference(reference)? {
            let uri = self
                .resolve_reference_uri(reference)
                .map_err(|error| self.funding.reference_error(error))?;
            let key = self.alias_cache_key(Arc::clone(&uri));
            if let Some(node) = self.cached_alias_node(&key) {
                return Ok(Some(self.funding.boxed(node)?));
            }
            if let Some(node) = self.cached_alias_placeholder(&uri) {
                return Ok(Some(self.funding.boxed(node)?));
            }
        }
        Ok(None)
    }

    pub(crate) fn location(&self) -> &Location {
        &self.location
    }

    /// Returns the current location relative to the resource base.
    ///
    /// This "suffix" is used for evaluation path computation. When an error occurs
    /// inside a `$ref` target, we combine the `$ref` traversal chain (prefix) with
    /// this suffix to form the complete evaluation path.
    ///
    /// # Example
    ///
    /// ```text
    /// Schema:
    /// {
    ///   "properties": {
    ///     "user": { "$ref": "#/$defs/Person" }
    ///   },
    ///   "$defs": {
    ///     "Person": {
    ///       "properties": {
    ///         "age": { "type": "integer" }
    ///       }
    ///     }
    ///   }
    /// }
    ///
    /// When compiling the "type" keyword inside "Person":
    ///   location()      = /$defs/Person/properties/age/type
    ///   resource_base   = /$defs/Person
    ///   suffix()        = /properties/age/type
    ///
    /// At validation time, if reached via /properties/user/$ref:
    ///   tracker = /properties/user/$ref + /properties/age/type
    ///                   = /properties/user/$ref/properties/age/type
    /// ```
    pub(crate) fn suffix(&self) -> Result<Location, crate::CompilationError> {
        let suffix = self
            .location
            .as_str()
            .strip_prefix(self.resource_base.as_str())
            .expect("location must start with resource_base");
        Location::from_escaped_with_funding(suffix, &self.funding)
    }

    pub(crate) fn has_vocabulary(&self, vocabulary: &Vocabulary) -> bool {
        if self.draft() < Draft::Draft201909 || vocabulary == &Vocabulary::Core {
            true
        } else {
            self.vocabularies.contains(vocabulary)
        }
    }
}

pub(crate) fn build_registry<'a, F: Json>(
    config: &'a ValidationOptions<'a, Arc<dyn referencing::Retrieve>, F>,
    draft: Draft,
    resource: ResourceRef<'a>,
    schema_id: Option<&'a str>,
) -> Result<(referencing::Registry<'a>, referencing::Uri<String>), referencing::Error> {
    build_registry_with_allocations(
        config,
        draft,
        resource,
        schema_id,
        &referencing::allocation::Unenforced,
    )
}
fn build_registry_with_allocations<'a, F: Json>(
    config: &'a ValidationOptions<'a, Arc<dyn referencing::Retrieve>, F>,
    draft: Draft,
    resource: ResourceRef<'a>,
    schema_id: Option<&'a str>,
    funding: &'a dyn referencing::allocation::Allocation,
) -> Result<(Registry<'a>, Uri<String>), referencing::Error> {
    let base_uri = resolve_base_uri_with_allocations(config.base_uri.as_ref(), schema_id, funding)?;
    let registry = referencing::Registry::new_with_allocations(funding)?
        .retriever(config.retriever.clone())
        .draft(draft)
        .add(base_uri.as_str(), resource)?
        .prepare()?;
    Ok((registry, base_uri))
}

/// Compile `schema` into a validator over representation `F`, honoring `config`.
pub(crate) fn build_validator<F: Json>(
    config: &ValidationOptions<'_, Arc<dyn referencing::Retrieve>, F>,
    schema: &Value,
) -> Result<Validator<F>, crate::compilation::CompileError<'static>> {
    build_validator_with_funding(config, schema, &crate::compilation::Funding::default())
}
pub(crate) fn build_validator_with_funding<F: Json>(
    config: &ValidationOptions<'_, Arc<dyn referencing::Retrieve>, F>,
    schema: &Value,
    funding: &crate::compilation::Funding,
) -> Result<Validator<F>, crate::compilation::CompileError<'static>> {
    let draft = config.draft_for_with_funding(schema, funding)?;
    let resource = draft.create_resource_ref(schema);

    if config.validate_schema {
        validate_schema_with_funding(draft, schema, funding)?;
    }

    if let Some(registry) = config.registry {
        let base_uri =
            resolve_base_uri_with_allocations(config.base_uri.as_ref(), resource.id(), funding)
                .map_err(|error| funding.reference_error(error))?;
        let registry = registry
            .add_with_allocations(base_uri.as_str(), resource, funding)
            .map_err(|error| funding.reference_error(error))?
            .retriever(config.retriever.clone())
            .draft(draft)
            .prepare()
            .map_err(|error| funding.reference_error(error))?;
        return build_validator_with_registry(config, schema, draft, resource, &registry, funding);
    }
    let (registry, _) =
        build_registry_with_allocations(config, draft, resource, resource.id(), funding)
            .map_err(|error| funding.reference_error(error))?;
    build_validator_with_registry(config, schema, draft, resource, &registry, funding)
}

#[cfg(feature = "resolve-async")]
pub(crate) async fn build_registry_async<'a, F: Json>(
    config: &'a ValidationOptions<'a, Arc<dyn referencing::AsyncRetrieve>, F>,
    draft: Draft,
    resource: ResourceRef<'a>,
    schema_id: Option<&'a str>,
) -> Result<(referencing::Registry<'a>, referencing::Uri<String>), referencing::Error> {
    let base_uri = resolve_base_uri(config.base_uri.as_ref(), schema_id)?;
    let registry = referencing::Registry::new()
        .async_retriever(config.retriever.clone())
        .draft(draft)
        .add(base_uri.as_str(), resource)?
        .async_prepare()
        .await?;
    Ok((registry, base_uri))
}

#[cfg(feature = "resolve-async")]
pub(crate) async fn build_validator_async<F: Json>(
    config: &ValidationOptions<'_, Arc<dyn referencing::AsyncRetrieve>, F>,
    schema: &Value,
) -> Result<Validator<F>, crate::compilation::CompileError<'static>> {
    let draft = config.draft_for(schema).await?;
    let resource_ref = draft.create_resource_ref(schema); // single computation

    if config.validate_schema {
        validate_schema(draft, schema)?;
    }

    if let Some(registry) = config.registry {
        let base_uri = resolve_base_uri(config.base_uri.as_ref(), resource_ref.id())?;
        let registry = registry
            .add(base_uri.as_str(), resource_ref)?
            .async_retriever(config.retriever.clone())
            .draft(draft)
            .async_prepare()
            .await?;
        return build_validator_with_registry(
            config,
            schema,
            draft,
            resource_ref,
            &registry,
            &crate::compilation::Funding::default(),
        );
    }

    let (registry, _) =
        build_registry_async(config, draft, resource_ref, resource_ref.id()).await?;
    build_validator_with_registry(
        config,
        schema,
        draft,
        resource_ref,
        &registry,
        &crate::compilation::Funding::default(),
    )
}

/// Upper-bound object-node count in `schema`, used to pre-size the per-location node cache.
///
/// Over-counts (data objects in `enum`/`const` are not subschemas), but it only sizes a transient
/// build cache, so over-provisioning is cheap and under-provisioning would reintroduce rehashing.
fn estimate_subschema_count(schema: &Value) -> usize {
    match schema {
        Value::Object(map) => 1 + map.values().map(estimate_subschema_count).sum::<usize>(),
        Value::Array(items) => items.iter().map(estimate_subschema_count).sum(),
        _ => 0,
    }
}

/// A meta-schema that requires a vocabulary this crate does not implement cannot be honored,
/// so the schemas written against it are refused.
fn ensure_vocabularies_supported<F: Json>(
    config: &dyn CompilationOptions<F>,
    vocabularies: &VocabularySet,
    funding: &crate::compilation::Funding,
) -> Result<(), crate::compilation::CompileError<'static>> {
    for uri in vocabularies.custom() {
        if !config.declares_vocabulary(uri) {
            return Err(ValidationError::compile_error_with_funding(
                Location::new_with_funding(funding)?,
                Location::new_with_funding(funding)?,
                Location::new_with_funding(funding)?,
                Cow::Owned(Value::Null),
                funding.format(format_args!("Unknown vocabulary: '{uri}' is required by the meta-schema. Adjust configuration to declare support for it"))?,
                funding,
            )?.into());
        }
    }
    Ok(())
}

fn compile_root_with_registry<R, F: Json>(
    config: &ValidationOptions<'_, R, F>,
    schema: &Value,
    draft: Draft,
    resource: ResourceRef<'_>,
    registry: &Registry<'_>,
    funding: &crate::compilation::Funding,
) -> Result<SchemaNode<F>, crate::compilation::CompileError<'static>> {
    let requested_base_uri =
        resolve_base_uri_with_allocations(config.base_uri.as_ref(), resource.id(), funding)
            .map_err(|error| funding.reference_error(error))?;
    let base_uri = normalize_base_uri_with_allocations(registry, &requested_base_uri, funding)
        .map_err(|error| funding.reference_error(error))?;
    let vocabularies = registry
        .try_find_vocabularies(draft, schema)
        .map_err(|error| funding.reference_error(error))?;
    ensure_vocabularies_supported(config, &vocabularies, funding)?;
    let resolver = registry
        .try_resolver(base_uri)
        .map_err(|error| funding.reference_error(error))?;
    let capacity = estimate_subschema_count(schema);
    let ctx: Context<'_, F> = Context::new(
        config,
        resolver,
        vocabularies,
        draft,
        Location::new_with_funding(funding)?,
        capacity,
        funding.clone(),
    )?;
    compile(&ctx, resource).map_err(|error| error.to_owned_with_funding(funding))
}
fn build_validator_with_registry<R, F: Json>(
    config: &ValidationOptions<'_, R, F>,
    schema: &Value,
    draft: Draft,
    resource: ResourceRef<'_>,
    registry: &Registry<'_>,
    funding: &crate::compilation::Funding,
) -> Result<Validator<F>, crate::compilation::CompileError<'static>> {
    let root =
        compile_root_with_registry::<_, F>(config, schema, draft, resource, registry, funding)?;
    Ok(Validator {
        root,
        draft: config.draft(),
        funding: funding.clone(),
    })
}

pub(crate) fn normalize_base_uri(registry: &Registry<'_>, base_uri: &Uri<String>) -> Uri<String> {
    normalize_base_uri_with_allocations(registry, base_uri, &referencing::allocation::Unenforced)
        .expect("ordinary root URI normalization")
}
fn normalize_base_uri_with_allocations(
    registry: &Registry<'_>,
    base_uri: &Uri<String>,
    funding: &dyn referencing::allocation::Allocation,
) -> Result<Uri<String>, referencing::Error> {
    if registry.try_contains_resource(base_uri.as_str())? {
        return Ok(base_uri.to_owned_with_allocations(funding)?);
    }
    if base_uri
        .fragment()
        .is_some_and(|fragment| fragment.as_str().is_empty())
    {
        let mut normalized = base_uri.to_owned_with_allocations(funding)?;
        normalized.set_fragment_with_allocations(None, funding)?;
        if registry.try_contains_resource(normalized.as_str())? {
            return Ok(normalized);
        }
    }
    panic!("generated registry is missing root URI '{base_uri}'");
}
pub(crate) fn resolve_base_uri(
    base_uri: Option<&String>,
    schema_id: Option<&str>,
) -> Result<Uri<String>, referencing::Error> {
    resolve_base_uri_with_allocations(base_uri, schema_id, &referencing::allocation::Unenforced)
}
fn resolve_base_uri_with_allocations(
    base_uri: Option<&String>,
    schema_id: Option<&str>,
    funding: &dyn referencing::allocation::Allocation,
) -> Result<Uri<String>, referencing::Error> {
    uri::from_str_with_allocations(
        base_uri
            .map(String::as_str)
            .unwrap_or_else(|| schema_id.unwrap_or(DEFAULT_BASE_URI)),
        funding,
    )
}

/// Compile the exact bundled meta-schema through the ordinary source compiler.
/// The caller chooses whether to retain this graph in its source or in the
/// ordinary global cache; the graph itself never adopts another source's work.
pub(crate) fn compile_meta_validator(
    draft: Draft,
    funding: &crate::compilation::Funding,
) -> Result<Validator, crate::compilation::CompileError<'static>> {
    let schema = funding.json_source(referencing::meta::source_for_draft(draft))?;
    let options = ValidationOptions::default_with_funding(funding)?.without_schema_validation();
    build_validator_with_funding(&options, &schema, funding)
}

pub(crate) fn validate_schema(
    draft: Draft,
    schema: &Value,
) -> Result<(), crate::compilation::CompileError<'static>> {
    validate_schema_with_funding(draft, schema, &crate::compilation::Funding::default())
}
fn validate_schema_with_funding(
    draft: Draft,
    schema: &Value,
    funding: &crate::compilation::Funding,
) -> Result<(), crate::compilation::CompileError<'static>> {
    // Boolean schemas are always valid per the spec, skip validation
    if schema.is_boolean() {
        return Ok(());
    }

    // For objects, we can skip validation if they're empty (always valid)
    if let Some(obj) = schema.as_object() {
        if obj.is_empty() {
            return Ok(());
        }
    }

    #[cfg(feature = "macros")]
    {
        if funding.is_enforced() {
            return Err(funding
                .error(crate::CompilationAllocationError::Unqualified(
                    "generated meta-schema validation storage",
                ))
                .into());
        }
        // Preserve the explicitly selected generated ordinary meta validator.
        let validator = crate::meta::validator_for_draft(draft);
        return validator
            .validate(schema)
            .map_err(|error| error.to_owned().into());
    }
    use crate::validator::Validate;
    let owned;
    let cached;
    let validator = if funding.is_enforced() {
        owned = compile_meta_validator(draft, funding)?;
        &owned
    } else {
        cached = crate::meta::validator_for_draft(draft);
        cached.as_ref()
    };
    if let Err(error) = funding.with_validation_context::<SerdeJson, _>(|context| {
        validator
            .root
            .validate(&schema, &crate::paths::LazyLocation::new(), None, context)
    })? {
        return Err(error.to_owned_with_funding(funding)?.into());
    }

    Ok(())
}

/// Compile a JSON Schema instance to a tree of nodes.
pub(crate) fn compile<'a, F: Json>(
    ctx: &Context<F>,
    resource: ResourceRef<'a>,
) -> Result<SchemaNode<F>, crate::compilation::CompileError<'a>> {
    let ctx = ctx.in_subresource(resource)?;
    compile_with_internal(&ctx, resource, None)
}

pub(crate) fn compile_with_alias<'a, F: Json>(
    ctx: &Context<F>,
    resource: ResourceRef<'a>,
    alias: Arc<Uri<String>>,
) -> Result<SchemaNode<F>, crate::compilation::CompileError<'a>> {
    compile_with_internal(ctx, resource, Some(alias))
}

#[allow(clippy::needless_pass_by_value)]
fn compile_with_internal<'a, F: Json>(
    ctx: &Context<F>,
    resource: ResourceRef<'a>,
    alias: Option<Arc<Uri<String>>>,
) -> Result<SchemaNode<F>, crate::compilation::CompileError<'a>> {
    // Check if this alias already has a cached node
    if let Some(alias_key) = alias.as_ref() {
        let scoped_key = ctx.alias_cache_key(Arc::clone(alias_key));
        if let Some(existing_alias) = ctx.cached_alias_node(&scoped_key) {
            return Ok(existing_alias);
        }
    }

    // Check location-based cache
    let key = ctx.location_cache_key();
    if let Some(existing) = ctx.cached_location_node(&key) {
        return Ok(existing);
    }

    // Check if there's a pending node (circular reference being compiled)
    if let Some(pending) = ctx.cached_pending_location_node(&key) {
        // If the node has already been initialized, reuse it. Otherwise, we rely on the
        // in-flight compilation to finish initialization and continue compiling here.
        if let Some(node) = pending.get() {
            return Ok(node.clone());
        }
    }

    // Create placeholder for circular reference detection
    let placeholder = PendingSchemaNode::new(ctx.funding())?;
    ctx.cache_pending_location_node(key.clone(), placeholder.clone())?;
    if let Some(alias_key) = alias.as_ref() {
        ctx.set_alias_placeholder(Arc::clone(alias_key), placeholder.clone())?;
    }

    // Compile the schema
    match compile_without_cache(ctx, resource) {
        Ok(node) => {
            // Initialize the placeholder with the compiled node
            placeholder.initialize(&node);

            // Remove from pending cache and add to final cache
            ctx.remove_pending_location_node(&key);
            ctx.cache_location_node(key.clone(), node.clone())?;

            if let Some(alias_key) = alias.as_ref() {
                ctx.remove_alias_placeholder(alias_key);
                let scoped_key = ctx.alias_cache_key(Arc::clone(alias_key));
                ctx.cache_alias_node(scoped_key, node.clone())?;
            }
            Ok(node)
        }
        Err(err) => Err(err.into()),
    }
}

fn compile_without_cache<'a, F: Json>(
    ctx: &Context<F>,
    resource: ResourceRef<'a>,
) -> Result<SchemaNode<F>, crate::compilation::CompileError<'a>> {
    match resource.contents() {
        Value::Bool(value) => match value {
            true => Ok(SchemaNode::from_boolean(ctx, None)?),
            false => Ok(SchemaNode::from_boolean(
                ctx,
                Some(keywords::boolean::FalseValidator::compile(
                    ctx,
                    ctx.location().clone(),
                )?),
            )?),
        },
        Value::Object(schema) => {
            // A schema could contain validation keywords along with annotations and we need to
            // collect annotations separately
            if !ctx.supports_adjacent_validation() {
                // Older drafts ignore all other keywords if `$ref` is present
                if let Some(reference) = schema.get("$ref") {
                    // Treat all keywords other than `$ref` as annotations
                    let mut annotations = Map::new();
                    for (key, value) in schema.iter().filter(|(key, _)| key.as_str() != "$ref") {
                        ctx.funding.json_insert(&mut annotations, key, value)?;
                    }
                    let annotations = if annotations.is_empty() {
                        None
                    } else {
                        Some(ctx.funding.arc(Value::Object(annotations))?)
                    };
                    return if let Some(validator) =
                        keywords::ref_::compile_ref(ctx, schema, reference)
                    {
                        let mut validators = Vec::new();
                        ctx.funding
                            .push(&mut validators, (BuiltinKeyword::Ref.into(), validator?))?;
                        Ok(SchemaNode::from_keywords(ctx, validators, annotations)?)
                    } else {
                        // Infinite reference to the same location
                        Ok(SchemaNode::from_boolean(ctx, None)?)
                    };
                }
            }

            let mut validators = Vec::new();
            ctx.funding.grow(&mut validators, schema.len())?;
            let mut annotations = Map::new();
            for (keyword, value) in schema {
                // Check if this keyword is overridden, then check the standard definitions
                if let Some(factory) = ctx.get_keyword_factory(keyword) {
                    if ctx.funding.is_enforced() {
                        return Err(ctx
                            .funding
                            .error(crate::CompilationAllocationError::Unqualified(
                                "custom keyword construction",
                            ))
                            .into());
                    }
                    let path = ctx.location().join_with_funding(keyword, ctx.funding())?;
                    let validator = CustomKeyword::new(
                        factory.init(schema, value, path.clone(), keyword)?,
                        path,
                        keyword.clone(),
                    );
                    let validator: BoxedValidator<F> = Box::new(validator);
                    validators.push((Keyword::custom(keyword), validator));
                } else if let Some((keyword, validator)) = keywords::get_for_draft(ctx, keyword)
                    .and_then(|(keyword, f)| f(ctx, schema, value).map(|v| (keyword, v)))
                {
                    validators.push((
                        keyword,
                        validator.map_err(|error| error.to_owned_with_funding(ctx.funding()))?,
                    ));
                } else if !ctx.is_known_keyword(keyword) {
                    // Treat all non-validation keywords as annotations
                    ctx.funding.json_insert(&mut annotations, keyword, value)?;
                }
            }
            let annotations = if annotations.is_empty() {
                None
            } else {
                Some(ctx.funding.arc(Value::Object(annotations))?)
            };
            Ok(SchemaNode::from_keywords(ctx, validators, annotations)?)
        }
        _ => {
            let location = ctx.location().clone();
            Err(ValidationError::multiple_type_error_with_funding(
                location.clone(),
                location,
                Location::new_with_funding(ctx.funding())?,
                Cow::Borrowed(resource.contents()),
                JsonTypeSet::from(JsonType::Boolean).insert(JsonType::Object),
                ctx.funding(),
            )?
            .into())
        }
    }
}

/// Iteratively traverse a schema document and compile a [`Validator`] for every
/// reachable subschema, keyed by URI-fragment JSON pointer.
///
/// Each subschema is compiled with its own fresh [`Context`] so that caches from
/// sibling compilations do not interfere. Nodes that fail to compile (e.g.
/// unresolvable `$ref`) are silently skipped.
fn collect_validators<'a, F: Json>(
    config: &'a dyn CompilationOptions<F>,
    resolver: &Resolver<'a>,
    vocabularies: &VocabularySet,
    schema: &'a Value,
    draft: Draft,
) -> Result<AHashMap<String, Validator<F>>, crate::compilation::CompileError<'static>> {
    let mut validators: AHashMap<String, Validator<F>> = AHashMap::new();
    let mut stack: Vec<(&'a Value, String)> = vec![(schema, "#".to_string())];
    while let Some((current, pointer)) = stack.pop() {
        if matches!(current, Value::Object(_) | Value::Bool(_)) {
            let ctx: Context<'_, F> = Context::new(
                config,
                resolver.clone(),
                vocabularies.clone(),
                draft,
                Location::new(),
                estimate_subschema_count(current),
                crate::compilation::Funding::default(),
            )?;
            let resource_ref = ctx.as_resource_ref(current);
            match compile(&ctx, resource_ref) {
                Ok(root) => {
                    validators.insert(
                        pointer.clone(),
                        Validator {
                            root,
                            draft,
                            funding: crate::compilation::Funding::default(),
                        },
                    );
                }
                Err(crate::compilation::CompileError::Storage(error)) => return Err(error.into()),
                Err(_) => {}
            }
        }
        match current {
            Value::Object(obj) => {
                for (key, value) in obj {
                    let mut escaped = String::new();
                    write_escaped_str(&mut escaped, key);
                    stack.push((value, format!("{pointer}/{escaped}")));
                }
            }
            Value::Array(arr) => {
                for (idx, item) in arr.iter().enumerate() {
                    stack.push((item, format!("{pointer}/{idx}")));
                }
            }
            _ => {}
        }
    }
    Ok(validators)
}

fn build_validator_map_with_registry<R, F: Json>(
    config: &ValidationOptions<'_, R, F>,
    schema: &Value,
    draft: Draft,
    resource: ResourceRef<'_>,
    registry: &Registry<'_>,
) -> Result<ValidatorMap<F>, crate::compilation::CompileError<'static>> {
    let requested_base_uri = resolve_base_uri(config.base_uri.as_ref(), resource.id())?;
    let base_uri = normalize_base_uri(registry, &requested_base_uri);
    let vocabularies = registry.find_vocabularies(draft, schema);
    ensure_vocabularies_supported(
        config,
        &vocabularies,
        &crate::compilation::Funding::default(),
    )?;
    let resolver = registry.resolver(base_uri);
    let validators = collect_validators::<F>(config, &resolver, &vocabularies, schema, draft)?;
    Ok(ValidatorMap { validators })
}

/// Compile every reachable subschema into a validator over representation `F`, honoring `config`.
pub(crate) fn build_validator_map<F: Json>(
    config: &ValidationOptions<'_, Arc<dyn referencing::Retrieve>, F>,
    schema: &Value,
) -> Result<ValidatorMap<F>, crate::compilation::CompileError<'static>> {
    let draft = config.draft_for(schema)?;
    let resource = draft.create_resource_ref(schema);
    validate_schema(draft, schema)?;

    if let Some(registry) = config.registry {
        let base_uri = resolve_base_uri(config.base_uri.as_ref(), resource.id())?;
        let registry = registry
            .add(base_uri.as_str(), resource)?
            .retriever(config.retriever.clone())
            .draft(draft)
            .prepare()?;
        return build_validator_map_with_registry(config, schema, draft, resource, &registry);
    }

    let (registry, _) = build_registry(config, draft, resource, resource.id())?;
    build_validator_map_with_registry(config, schema, draft, resource, &registry)
}

#[cfg(feature = "resolve-async")]
pub(crate) async fn build_validator_map_async<F: Json>(
    config: &ValidationOptions<'_, Arc<dyn referencing::AsyncRetrieve>, F>,
    schema: &Value,
) -> Result<ValidatorMap<F>, crate::compilation::CompileError<'static>> {
    let draft = config.draft_for(schema).await?;
    let resource = draft.create_resource_ref(schema);

    validate_schema(draft, schema)?;

    if let Some(registry) = config.registry {
        let base_uri = resolve_base_uri(config.base_uri.as_ref(), resource.id())?;
        let registry = registry
            .add(base_uri.as_str(), resource)?
            .async_retriever(config.retriever.clone())
            .draft(draft)
            .async_prepare()
            .await?;
        return build_validator_map_with_registry(config, schema, draft, resource, &registry);
    }

    let (registry, _) = build_registry_async(config, draft, resource, resource.id()).await?;
    build_validator_map_with_registry(config, schema, draft, resource, &registry)
}
