use std::{
    fmt,
    sync::{Arc, LazyLock},
};

use fluent_uri::Uri;
use hashbrown::HashMap;
use serde_json::Value;

use crate::{
    allocation::{Allocation, Allocator, Unenforced},
    cache::{SharedUriCache, UriCache},
    meta, uri,
    vocabularies::{self, VocabularySet},
    Anchor, DefaultRetriever, Draft, Error, Resolver, ResourceRef, Retrieve,
};

mod build;
use build::{KnownResources, ResourceStore};

mod index;
use index::{Index, IndexedAnchor, IndexedResource};

mod input;
#[cfg(feature = "retrieve-async")]
pub(crate) use input::IntoAsyncRetriever;
pub use input::IntoRegistryResource;
pub(crate) use input::{IntoRetriever, PendingResource};

/// Pre-loaded registry containing all JSON Schema meta-schemas and their vocabularies
pub static SPECIFICATIONS: LazyLock<Registry<'static>> =
    LazyLock::new(|| Registry::from_meta_schemas(meta::META_SCHEMAS_ALL.as_slice()));

pub struct RegistryBuilder<'a> {
    allocation: Allocator<'a>,
    baseline: Option<&'a Registry<'a>>,
    pending: HashMap<Uri<String>, PendingResource<'a>>,
    retriever: Arc<dyn Retrieve>,
    #[cfg(feature = "retrieve-async")]
    async_retriever: Option<Arc<dyn crate::AsyncRetrieve>>,
    draft: Option<Draft>,
}

impl fmt::Debug for RegistryBuilder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistryBuilder")
            .field("has_baseline", &self.baseline.is_some())
            .field("pending_len", &self.pending.len())
            .field("draft", &self.draft)
            .finish()
    }
}

/// A registry of JSON Schema resources, each identified by their canonical URIs.
///
/// `Registry` is a prepared registry: add resources with [`Registry::new`] and
/// [`RegistryBuilder::add`], then call [`RegistryBuilder::prepare`] to build the
/// reusable registry. To resolve `$ref` references directly, create a [`Resolver`]
/// from the prepared registry:
///
/// ```rust
/// use referencing::Registry;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let schema = serde_json::json!({
///     "$schema": "https://json-schema.org/draft/2020-12/schema",
///     "$id": "https://example.com/root",
///     "$defs": { "item": { "type": "string" } },
///     "items": { "$ref": "#/$defs/item" }
/// });
///
/// let registry = Registry::new()
///     .add("https://example.com/root", schema)?
///     .prepare()?;
///
/// let resolver = registry.resolver(referencing::uri::from_str("https://example.com/root")?);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Registry<'a> {
    baseline: Option<&'a Registry<'a>>,
    resolution_cache: SharedUriCache<'a>,
    known_resources: KnownResources,
    index: Index<'a>,
}

impl<'a> RegistryBuilder<'a> {
    fn new() -> Self {
        Self::with_allocations(&Unenforced).unwrap()
    }

    fn with_allocations(allocation: &'a dyn Allocation) -> Result<Self, Error> {
        let allocation = Allocator(allocation);
        Ok(Self {
            allocation,
            baseline: None,
            pending: HashMap::new(),
            retriever: allocation.arc(DefaultRetriever)?,
            #[cfg(feature = "retrieve-async")]
            async_retriever: None,
            draft: None,
        })
    }

    fn from_registry(registry: &'a Registry<'a>) -> Result<Self, Error> {
        let mut builder = Self::with_allocations(registry.allocation().0)?;
        builder.baseline = Some(registry);
        Ok(builder)
    }

    #[must_use]
    pub fn draft(mut self, draft: Draft) -> Self {
        self.draft = Some(draft);
        self
    }

    #[must_use]
    pub fn retriever(self, retriever: impl IntoRetriever) -> Self {
        self.try_retriever(retriever).unwrap()
    }
    /// Installs a retriever, admitting a new shared owner when the input is not already shared.
    pub fn try_retriever(mut self, retriever: impl IntoRetriever) -> Result<Self, Error> {
        self.retriever = retriever.into_retriever(self.allocation.0)?;
        Ok(self)
    }

    #[cfg(feature = "retrieve-async")]
    #[must_use]
    pub fn async_retriever(self, retriever: impl IntoAsyncRetriever) -> Self {
        self.try_async_retriever(retriever).unwrap()
    }
    /// Installs an asynchronous retriever with prospective shared-owner admission.
    #[cfg(feature = "retrieve-async")]
    pub fn try_async_retriever(
        mut self,
        retriever: impl IntoAsyncRetriever,
    ) -> Result<Self, Error> {
        self.async_retriever = Some(retriever.into_retriever(self.allocation.0)?);
        Ok(self)
    }

    /// Add a resource to the registry builder.
    ///
    /// # Errors
    ///
    /// Returns an error if the URI is invalid.
    pub fn add<'b>(
        self,
        uri: impl AsRef<str>,
        resource: impl IntoRegistryResource<'b>,
    ) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
    {
        let parsed =
            uri::from_str_with_allocations(uri.as_ref().trim_end_matches('#'), self.allocation.0)?;
        let mut pending: HashMap<Uri<String>, PendingResource<'b>> = self.pending;
        self.allocation.insert(
            &mut pending,
            parsed,
            input::private::Sealed::into_pending(resource),
        )?;
        Ok(RegistryBuilder {
            allocation: self.allocation,
            baseline: self.baseline,
            pending,
            retriever: self.retriever,
            #[cfg(feature = "retrieve-async")]
            async_retriever: self.async_retriever,
            draft: self.draft,
        })
    }

    /// Add multiple resources to the registry builder.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid.
    pub fn extend<'b, I, U, T>(self, pairs: I) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
        I: IntoIterator<Item = (U, T)>,
        U: AsRef<str>,
        T: IntoRegistryResource<'b>,
    {
        let mut builder = RegistryBuilder {
            allocation: self.allocation,
            baseline: self.baseline,
            pending: self.pending,
            retriever: self.retriever,
            #[cfg(feature = "retrieve-async")]
            async_retriever: self.async_retriever,
            draft: self.draft,
        };
        for (uri, resource) in pairs {
            builder = builder.add(uri, resource)?;
        }
        Ok(builder)
    }

    /// Prepare the registry for reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if URI processing, retrieval, or custom meta-schema validation fails.
    pub fn prepare(self) -> Result<Registry<'a>, Error> {
        // When extending an existing registry, seed known resources from the baseline so the
        // retriever skips URIs already owned by the parent.
        let mut known_resources = KnownResources::new();
        if let Some(baseline) = self.baseline {
            for uri in &baseline.known_resources {
                self.allocation.insert_set(
                    &mut known_resources,
                    uri.to_owned_with_allocations(self.allocation.0)?,
                )?;
            }
        }
        let mut documents = ResourceStore::new();
        let mut resolution_cache = UriCache::with_allocations(self.allocation.0);
        let (custom_metaschemas, index_data) = build::index_resources(
            self.pending,
            &*self.retriever,
            &mut documents,
            &mut known_resources,
            &mut resolution_cache,
            self.draft,
        )?;
        build::validate_custom_metaschemas(&custom_metaschemas, &known_resources, self.allocation)?;
        Ok(Registry {
            baseline: self.baseline,
            resolution_cache: resolution_cache.into_shared(),
            known_resources,
            index: index_data,
        })
    }

    #[cfg(feature = "retrieve-async")]
    /// Prepare the registry for reuse with async retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if URI processing, retrieval, or custom meta-schema validation fails.
    pub async fn async_prepare(self) -> Result<Registry<'a>, Error> {
        let retriever: Arc<dyn crate::AsyncRetrieve> = match self.async_retriever {
            Some(retriever) => retriever,
            None => self.allocation.arc(DefaultRetriever)?,
        };
        let mut known_resources = KnownResources::new();
        if let Some(baseline) = self.baseline {
            for uri in &baseline.known_resources {
                self.allocation.insert_set(
                    &mut known_resources,
                    uri.to_owned_with_allocations(self.allocation.0)?,
                )?;
            }
        }
        let mut documents = ResourceStore::new();
        let mut resolution_cache = UriCache::with_allocations(self.allocation.0);
        let (custom_metaschemas, index_data) = build::index_resources_async(
            self.pending,
            &*retriever,
            &mut documents,
            &mut known_resources,
            &mut resolution_cache,
            self.draft,
        )
        .await?;
        build::validate_custom_metaschemas(&custom_metaschemas, &known_resources, self.allocation)?;
        Ok(Registry {
            baseline: self.baseline,
            resolution_cache: resolution_cache.into_shared(),
            known_resources,
            index: index_data,
        })
    }
}

impl<'a> Registry<'a> {
    #[allow(clippy::new_ret_no_self)]
    #[must_use]
    pub fn new<'b>() -> RegistryBuilder<'b> {
        RegistryBuilder::new()
    }
    /// Creates a builder whose storage requests borrow the supplied admission policy.
    pub fn new_with_allocations<'b>(
        allocation: &'b dyn Allocation,
    ) -> Result<RegistryBuilder<'b>, Error> {
        RegistryBuilder::with_allocations(allocation)
    }
    pub(crate) fn allocation(&self) -> Allocator<'a> {
        self.resolution_cache.allocation()
    }

    /// Add a resource to a prepared registry, returning a builder that must be prepared again.
    ///
    /// # Errors
    ///
    /// Returns an error if the URI is invalid.
    pub fn add<'b>(
        &'b self,
        uri: impl AsRef<str>,
        resource: impl IntoRegistryResource<'b>,
    ) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
    {
        self.add_with_allocations(uri, resource, self.allocation().0)
    }

    /// Extends this borrowed baseline using the supplied owner for new storage.
    pub fn add_with_allocations<'b>(
        &'b self,
        uri: impl AsRef<str>,
        resource: impl IntoRegistryResource<'b>,
        allocation: &'b dyn Allocation,
    ) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
    {
        let mut builder = RegistryBuilder::with_allocations(allocation)?;
        builder.baseline = Some(self);
        builder.add(uri, resource)
    }

    /// Add multiple resources to a prepared registry, returning a builder that
    /// must be prepared again.
    ///
    /// # Errors
    ///
    /// Returns an error if any URI is invalid.
    pub fn extend<'b, I, U, T>(&'b self, pairs: I) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
        I: IntoIterator<Item = (U, T)>,
        U: AsRef<str>,
        T: IntoRegistryResource<'b>,
    {
        self.extend_with_allocations(pairs, self.allocation().0)
    }

    /// Extend a borrowed immutable baseline with independently funded caches and indexes.
    /// Existing resource lookups remain borrowed; new resolution storage belongs to
    /// the supplied allocation policy and never enters the baseline's caches.
    pub fn extend_with_allocations<'b, I, U, T>(
        &'b self, pairs: I, allocation: &'b dyn Allocation,
    ) -> Result<RegistryBuilder<'b>, Error>
    where
        'a: 'b,
        I: IntoIterator<Item = (U, T)>,
        U: AsRef<str>,
        T: IntoRegistryResource<'b>,
    {
        let mut builder = RegistryBuilder::with_allocations(allocation)?;
        builder.baseline = Some(self);
        builder.extend(pairs)
    }

    /// Build a registry with all the given meta-schemas from specs.
    pub(crate) fn from_meta_schemas(schemas: &[(&'static str, &'static Value)]) -> Self {
        RegistryBuilder::new()
            .extend(schemas.iter().copied())
            .and_then(RegistryBuilder::prepare)
            .expect("built-in meta-schema registry must build")
    }
    /// Returns `true` if the registry contains a resource at the given URI.
    ///
    /// Returns `false` if the URI is malformed.
    #[must_use]
    pub fn contains_resource(&self, uri: &str) -> bool {
        self.try_contains_resource(uri).unwrap()
    }
    /// Queries resources, preserving storage refusals while malformed URIs still return false.
    pub fn try_contains_resource(&self, uri: &str) -> Result<bool, Error> {
        match uri::from_str_with_allocations(uri, self.allocation().0) {
            Ok(uri) => Ok(self.resource_by_uri(&uri).is_some()),
            Err(Error::Allocation(error)) => Err(Error::Allocation(error)),
            Err(_) => Ok(false),
        }
    }

    /// Creates a [`Resolver`] rooted at `base_uri`.
    ///
    /// The returned resolver borrows from this registry and cannot outlive it.
    #[must_use]
    pub fn resolver(&self, base_uri: Uri<String>) -> Resolver<'_> {
        self.try_resolver(base_uri).unwrap()
    }

    /// Creates a resolver after admitting its shared URI owner.
    pub fn try_resolver(&self, base_uri: Uri<String>) -> Result<Resolver<'_>, Error> {
        Ok(Resolver::new(self, self.allocation().arc(base_uri)?))
    }

    /// Returns the vocabulary set `draft` puts in effect for a schema with the given `contents`.
    ///
    /// A `$schema` naming a custom meta-schema is the one case where `contents` decides: its
    /// `$vocabulary` is read from the registry. Never errors.
    #[must_use]
    pub fn find_vocabularies(&self, draft: Draft, contents: &Value) -> VocabularySet {
        self.try_find_vocabularies(draft, contents).unwrap()
    }

    /// Finds vocabularies while propagating storage refusal from custom identifiers.
    pub fn try_find_vocabularies(
        &self,
        draft: Draft,
        contents: &Value,
    ) -> Result<VocabularySet, Error> {
        // Detection answers one question: whether `$schema` names a custom meta-schema, the
        // only kind that declares vocabularies of its own. Every other schema takes them from
        // `draft`, which `with_draft` may have set over the `$schema` declaration.
        Ok(match draft.detect(contents) {
            Draft::Unknown => {
                if let Some(specification) = contents
                    .as_object()
                    .and_then(|obj| obj.get("$schema"))
                    .and_then(|s| s.as_str())
                {
                    let uri = uri::from_str_with_allocations(specification, self.allocation().0);
                    if matches!(uri, Err(Error::Allocation(_))) {
                        return uri.map(|_| unreachable!());
                    }
                    if let Ok(mut uri) = uri {
                        uri.set_fragment(None);
                        if let Some(resource) = self.resource_by_uri(&uri) {
                            let found = vocabularies::find_with_allocations(
                                resource.contents(),
                                self.allocation().0,
                            );
                            match found {
                                Ok(Some(vocabularies)) => return Ok(vocabularies),
                                Err(Error::Allocation(error)) => {
                                    return Err(Error::Allocation(error))
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Draft::Unknown.default_vocabularies()
            }
            _ => draft.default_vocabularies(),
        })
    }

    /// Resolves `uri` against `base` and returns the resulting absolute URI.
    ///
    /// Results are cached. Returns an error if `base` has no scheme or if
    /// resolution fails.
    ///
    /// # Errors
    ///
    /// Returns an error if base has no schema or there is a fragment.
    pub fn resolve_uri(&self, base: &Uri<&str>, uri: &str) -> Result<Arc<Uri<String>>, Error> {
        self.resolution_cache.resolve_against(base, uri)
    }

    #[inline]
    pub(crate) fn resource_by_uri(&self, uri: &Uri<String>) -> Option<ResourceRef<'_>> {
        self.index
            .resources
            .get(uri)
            .and_then(IndexedResource::resolve)
            .or_else(|| {
                self.baseline
                    .and_then(|baseline| baseline.resource_by_uri(uri))
            })
    }

    pub(crate) fn anchor(&self, uri: &Uri<String>, name: &str) -> Result<Anchor<'_>, Error> {
        if let Some(anchor) = self.anchor_exact(uri, name) {
            return Ok(anchor);
        }

        if let Some(resource) = self.resource_by_uri(uri) {
            if let Some(id) = resource.id() {
                let canonical = uri::from_str_with_allocations(id, self.allocation().0)?;
                if let Some(anchor) = self.anchor_exact(&canonical, name) {
                    return Ok(anchor);
                }
            }
        }

        if name.contains('/') {
            Err(Error::invalid_anchor(self.allocation().copy_str(name)?))
        } else {
            Err(Error::no_such_anchor(self.allocation().copy_str(name)?))
        }
    }

    fn local_anchor_by_uri(&self, uri: &Uri<String>, name: &str) -> Option<Anchor<'_>> {
        self.index
            .anchors
            .get(uri)
            .and_then(|entries| entries.get(name))
            .and_then(IndexedAnchor::resolve)
    }

    fn anchor_exact(&self, uri: &Uri<String>, name: &str) -> Option<Anchor<'_>> {
        self.local_anchor_by_uri(uri, name).or_else(|| {
            self.baseline
                .and_then(|baseline| baseline.anchor_exact(uri, name))
        })
    }
}
