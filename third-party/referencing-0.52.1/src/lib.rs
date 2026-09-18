//! # referencing
//!
//! An implementation-agnostic JSON reference resolution library for Rust.
pub mod allocation;
mod anchor;
mod cache;
mod draft;
mod error;
mod list;
pub mod meta;
mod path;
mod pointer;
mod registry;
mod resolver;
mod resource;
mod retriever;
mod segments;
mod small_map;
mod spec;
pub mod uri;
mod vocabularies;

pub(crate) use anchor::Anchor;
pub use draft::Draft;
pub use error::{Error, UriError};
pub use fluent_uri::{Iri, IriRef, Uri, UriRef};
pub use list::List;
#[doc(hidden)]
pub use path::{write_escaped_str, write_index};
pub use path::{JsonPointerNode, JsonPointerSegment};
pub use pointer::{parse_index, pointer};
pub use registry::{IntoRegistryResource, Registry, RegistryBuilder, SPECIFICATIONS};
pub use resolver::{Resolved, Resolver};
pub use resource::{unescape_segment, unescape_segment_with_allocations, Resource, ResourceRef};
pub use retriever::{DefaultRetriever, RetrievalError, Retrieve};
pub(crate) use segments::Segments;
pub use vocabularies::{Vocabulary, VocabularySet};

#[cfg(feature = "retrieve-async")]
pub use retriever::AsyncRetrieve;
