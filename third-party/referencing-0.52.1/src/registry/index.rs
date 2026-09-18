//! Lookup structures produced by the build pass.
use std::sync::Arc;

use fluent_uri::Uri;

use serde_json::Value;

use crate::{
    allocation::Allocator, anchor::Anchor, draft::Draft, pointer::ParsedPointer,
    small_map::SmallMap, Error, ResourceRef,
};

use super::build::StoredResource;

/// Lookup tables mapping canonical URIs to resources and anchors.
#[derive(Debug, Default)]
pub(super) struct Index<'a> {
    pub(super) resources: SmallMap<Arc<Uri<String>>, IndexedResource<'a>>,
    pub(super) anchors: SmallMap<Arc<Uri<String>>, SmallMap<String, IndexedAnchor<'a>>>,
}

impl<'a> Index<'a> {
    pub(super) fn register_document(
        &mut self,
        key: &Arc<Uri<String>>,
        document: &Arc<StoredResource<'a>>,
        allocation: Allocator<'_>,
    ) -> Result<(), Error> {
        if let Some(contents) = document.borrowed_contents() {
            self.register_borrowed_subresource(key, document.draft(), true, contents, allocation)
        } else {
            self.register_owned_subresource(
                key,
                document,
                &ParsedPointer::default(),
                document.draft(),
                true,
                document.contents(),
                allocation,
            )
        }
    }

    pub(super) fn register_borrowed_subresource(
        &mut self,
        key: &Arc<Uri<String>>,
        draft: Draft,
        has_id: bool,
        contents: &'a Value,
        allocation: Allocator<'_>,
    ) -> Result<(), Error> {
        if has_id {
            self.resources.try_insert(
                Arc::clone(key),
                IndexedResource::Borrowed(ResourceRef::new(contents, draft)),
                allocation,
            )?;
        }
        let anchors = self
            .anchors
            .try_get_or_insert_default(Arc::clone(key), allocation)?;
        for anchor in draft.anchors(contents) {
            anchors.try_insert(
                allocation.copy_str(anchor.name())?,
                IndexedAnchor::Borrowed(anchor),
                allocation,
            )?;
        }
        Ok(())
    }

    pub(super) fn register_owned_subresource(
        &mut self,
        key: &Arc<Uri<String>>,
        document: &Arc<StoredResource<'a>>,
        pointer: &ParsedPointer,
        draft: Draft,
        has_id: bool,
        contents: &Value,
        allocation: Allocator<'_>,
    ) -> Result<(), Error> {
        if has_id {
            self.resources.try_insert(
                Arc::clone(key),
                IndexedResource::Owned {
                    document: Arc::clone(document),
                    pointer: pointer.try_clone_with_allocations(allocation.0)?,
                    draft,
                },
                allocation,
            )?;
        }
        let anchors = self
            .anchors
            .try_get_or_insert_default(Arc::clone(key), allocation)?;
        for anchor in draft.anchors(contents) {
            let (name, kind) = match anchor {
                Anchor::Default { name, .. } => (name, AnchorKind::Default),
                Anchor::Dynamic { name, .. } => (name, AnchorKind::Dynamic),
            };
            anchors.try_insert(
                allocation.copy_str(name)?,
                IndexedAnchor::Owned {
                    document: Arc::clone(document),
                    pointer: pointer.try_clone_with_allocations(allocation.0)?,
                    draft,
                    kind,
                    name: allocation.copy_str(name)?,
                },
                allocation,
            )?;
        }
        Ok(())
    }
}

/// A schema resource in the index: either borrowed from the caller or owned by the registry.
#[derive(Debug)]
pub(super) enum IndexedResource<'a> {
    Borrowed(ResourceRef<'a>),
    Owned {
        document: Arc<StoredResource<'a>>,
        pointer: ParsedPointer,
        draft: Draft,
    },
}

impl IndexedResource<'_> {
    #[inline]
    pub(super) fn resolve(&self) -> Option<ResourceRef<'_>> {
        match self {
            IndexedResource::Borrowed(resource) => {
                Some(ResourceRef::new(resource.contents(), resource.draft()))
            }
            IndexedResource::Owned {
                document,
                pointer,
                draft,
            } => {
                let contents = pointer.lookup(document.contents())?;
                Some(ResourceRef::new(contents, *draft))
            }
        }
    }
}

/// An anchor in the index: either borrowed from the caller or owned by the registry.
#[derive(Debug)]
pub(super) enum IndexedAnchor<'a> {
    Borrowed(Anchor<'a>),
    Owned {
        document: Arc<StoredResource<'a>>,
        pointer: ParsedPointer,
        draft: Draft,
        kind: AnchorKind,
        name: String,
    },
}

impl IndexedAnchor<'_> {
    #[inline]
    pub(super) fn resolve(&self) -> Option<Anchor<'_>> {
        match self {
            IndexedAnchor::Borrowed(anchor) => Some(match anchor {
                Anchor::Default { name, resource } => Anchor::Default {
                    name,
                    resource: ResourceRef::new(resource.contents(), resource.draft()),
                },
                Anchor::Dynamic { name, resource } => Anchor::Dynamic {
                    name,
                    resource: ResourceRef::new(resource.contents(), resource.draft()),
                },
            }),
            IndexedAnchor::Owned {
                document,
                pointer,
                draft,
                kind,
                name,
            } => {
                let contents = pointer.lookup(document.contents())?;
                let resource = ResourceRef::new(contents, *draft);
                Some(match kind {
                    AnchorKind::Default => Anchor::Default { name, resource },
                    AnchorKind::Dynamic => Anchor::Dynamic { name, resource },
                })
            }
        }
    }
}

/// Whether an anchor is a plain anchor (`$anchor`) or a dynamic anchor (`$dynamicAnchor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnchorKind {
    Default,
    Dynamic,
}
