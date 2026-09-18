use derivre::{ParserResult as Result, ParserError};
use referencing::{Registry, Resolver};
use serde_json::Value;
use std::{cell::RefCell, rc::Rc};

use super::{schema::SchemaBuilderOptions, shared_context::SharedContext, RetrieveWrapper};

const DEFAULT_DRAFT: Draft = Draft::Draft202012;
const DEFAULT_ROOT_URI: &str = "json-schema:///";

pub use referencing::{Draft, ResourceRef};

fn draft_for(value: &Value) -> Draft {
    match DEFAULT_DRAFT.detect(value) { Draft::Unknown => DEFAULT_DRAFT, draft => draft }
}

pub struct PreContext<'a> {
    registry: Registry<'a>,
    draft: Draft,
    pub base_uri: String,
    funding: derivre::ParserAllocationFunding,
    frames: derivre::prepared_funding::Scope<'a, derivre::ParserAllocationFunding>,
}

pub struct Context<'a> {
    resolver: Resolver<'a>,
    pub draft: Draft,
    pub shared: Rc<RefCell<SharedContext>>,
    pub options: SchemaBuilderOptions,
    pub funding: derivre::ParserAllocationFunding,
    pub(super) frames: &'a derivre::prepared_funding::Scope<'a, derivre::ParserAllocationFunding>,
}

impl<'a> PreContext<'a> {
    pub fn new(contents: Value, retriever: Option<RetrieveWrapper>, allocation: &'a crate::allocation::CompilerAllocation<'_>) -> Result<Self> {
        let frames = derivre::prepared_funding::Scope::new(allocation.0)
            .map_err(|error| super::schema::frames::failure(error, allocation.0))?;
        let draft = draft_for(&contents);
        let resource = draft.create_resource(contents);
        let base_uri = allocation.0.try_copy_str(draft.create_resource_ref(resource.contents()).id().unwrap_or(DEFAULT_ROOT_URI))?;
        let mut registry = Registry::new_with_allocations(allocation).map_err(|error| derivre::ParserError::cause(error, allocation.0))?.draft(draft);
        if let Some(retriever) = retriever { registry = registry.try_retriever(retriever).map_err(|error| derivre::ParserError::cause(error, allocation.0))?; }
        let registry = registry.add(&base_uri, resource).map_err(|error| derivre::ParserError::cause(error, allocation.0))?.prepare().map_err(|error| derivre::ParserError::cause(error, allocation.0))?;
        Ok(Self { registry, draft, base_uri, funding: allocation.0.clone(), frames })
    }
}

impl<'a> Context<'a> {
    pub fn new(pre_context: &'a PreContext<'a>) -> Result<Self> {
        let base = referencing::uri::from_str_with_allocations(&pre_context.base_uri, &crate::allocation::CompilerAllocation(&pre_context.funding)).map_err(|error| derivre::ParserError::cause(error, &pre_context.funding))?;
        let resolver = pre_context.registry.try_resolver(base).map_err(|error| derivre::ParserError::cause(error, &pre_context.funding))?;
        let ctx = Context {
            resolver,
            draft: pre_context.draft,
            shared: pre_context.funding.try_rc(RefCell::new(SharedContext::new(pre_context.funding.clone())))?,
            funding: pre_context.funding.clone(),
            options: SchemaBuilderOptions::default(),
            frames: &pre_context.frames,
        };

        Ok(ctx)
    }

    pub fn in_subresource(&'a self, resource: ResourceRef) -> Result<Context<'a>> {
        let resolver = self.resolver.in_subresource(resource).map_err(|error| derivre::ParserError::cause(error, &self.funding))?;
        Ok(Context {
            resolver,
            draft: resource.draft(),
            shared: Rc::clone(&self.shared),
            options: self.options.clone(),
            funding: self.funding.clone(),
            frames: self.frames,
        })
    }

    pub fn as_resource_ref<'r>(&'a self, contents: &'r Value) -> ResourceRef<'r> {
        let draft = match self.draft.detect(contents) { Draft::Unknown => DEFAULT_DRAFT, draft => draft };
        draft.create_resource_ref(contents)
    }

    pub fn normalize_ref(&self, reference: &str) -> Result<String> {
        let uri = self.resolver.resolve_uri(&self.resolver.base_uri().borrow(), reference).map_err(|error| derivre::ParserError::cause(error, &self.funding))?;
        Ok(self.funding.try_copy_str(uri.as_str())?)
    }

    pub fn lookup_resource(&'a self, reference: &str) -> Result<ResourceRef<'a>> {
        let resolved = self.resolver.lookup(reference).map_err(|error| derivre::ParserError::cause(error, &self.funding))?;
        Ok(self.as_resource_ref(resolved.contents()))
    }
}

impl referencing::Retrieve for RetrieveWrapper {
    fn retrieve(
        &self,
        uri: &referencing::Uri<String>,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let value = self.0.retrieve(uri.as_str())?;
        Ok(value)
    }
}
