use core::fmt;
use std::{num::ParseIntError, str::Utf8Error};

use fluent_uri::{resolve::ResolveError, ParseError, Uri};

/// Errors that can occur during reference resolution and resource handling.
#[derive(Debug)]
pub enum Error {
    /// A prospective storage request failed before mutation.
    Allocation(crate::allocation::AllocationError),
    /// An external producer has no prospective allocation contract.
    Unqualified(&'static str),
    /// The bundled source JSON producer failed, retaining its paid diagnostic.
    MetaSchema(serde_json::bounded_events::ValueError),
    /// A resource is not present in a registry and retrieving it failed.
    Unretrievable {
        uri: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A JSON Pointer leads to a part of a document that does not exist.
    PointerToNowhere { pointer: String },
    /// JSON Pointer contains invalid percent-encoded data.
    InvalidPercentEncoding { pointer: String, source: Utf8Error },
    /// Failed to parse array index in JSON Pointer.
    InvalidArrayIndex {
        pointer: String,
        index: String,
        source: ParseIntError,
    },
    /// An anchor does not exist within a particular resource.
    NoSuchAnchor { anchor: String },
    /// An anchor which could never exist in a resource was dereferenced.
    InvalidAnchor { anchor: String },
    /// An error occurred while parsing or manipulating a URI.
    InvalidUri(UriError),
    /// An unknown JSON Schema specification was encountered.
    UnknownSpecification { specification: String },
    /// A circular reference was detected in a meta-schema chain.
    CircularMetaschema { uri: String },
}

impl From<crate::allocation::AllocationError> for Error {
    fn from(error: crate::allocation::AllocationError) -> Self {
        Self::Allocation(error)
    }
}

impl Error {
    pub(crate) fn pointer_to_nowhere(pointer: impl Into<String>) -> Error {
        Error::PointerToNowhere {
            pointer: pointer.into(),
        }
    }
    pub(crate) fn invalid_percent_encoding(pointer: impl Into<String>, source: Utf8Error) -> Error {
        Error::InvalidPercentEncoding {
            pointer: pointer.into(),
            source,
        }
    }
    pub(crate) fn invalid_array_index(
        pointer: impl Into<String>,
        index: impl Into<String>,
        source: ParseIntError,
    ) -> Error {
        Error::InvalidArrayIndex {
            pointer: pointer.into(),
            index: index.into(),
            source,
        }
    }
    pub(crate) fn invalid_anchor(anchor: impl Into<String>) -> Error {
        Error::InvalidAnchor {
            anchor: anchor.into(),
        }
    }
    pub(crate) fn no_such_anchor(anchor: impl Into<String>) -> Error {
        Error::NoSuchAnchor {
            anchor: anchor.into(),
        }
    }

    pub fn unknown_specification(specification: impl Into<String>) -> Error {
        Error::UnknownSpecification {
            specification: specification.into(),
        }
    }

    pub fn circular_metaschema(uri: impl Into<String>) -> Error {
        Error::CircularMetaschema { uri: uri.into() }
    }

    pub fn unretrievable(
        uri: impl Into<String>,
        source: Box<dyn std::error::Error + Send + Sync>,
    ) -> Error {
        Error::Unretrievable {
            uri: uri.into(),
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Allocation(error) => error.fmt(f),
            Error::MetaSchema(error) => error.fmt(f),
            Error::Unqualified(producer) => write!(f, "reference allocation is unqualified: {producer}"),
            Error::Unretrievable { uri, source } => {
                f.write_fmt(format_args!("Resource '{uri}' is not present in a registry and retrieving it failed: {source}"))
            },
            Error::PointerToNowhere { pointer } => {
                f.write_fmt(format_args!("Pointer '{pointer}' does not exist"))
            }
            Error::InvalidPercentEncoding { pointer, .. } => {
                f.write_fmt(format_args!("Invalid percent encoding in pointer '{pointer}': the decoded bytes do not represent valid UTF-8"))
            }
            Error::InvalidArrayIndex { pointer, index, .. } => {
                f.write_fmt(format_args!("Failed to parse array index '{index}' in pointer '{pointer}'"))
            }
            Error::NoSuchAnchor { anchor } => {
                f.write_fmt(format_args!("Anchor '{anchor}' does not exist"))
            }
            Error::InvalidAnchor { anchor } => {
                f.write_fmt(format_args!("Anchor '{anchor}' is invalid"))
            }
            Error::InvalidUri(error) => error.fmt(f),
            Error::UnknownSpecification { specification } => {
                write!(f, "Unknown meta-schema: '{specification}'. Custom meta-schemas must be registered in the registry before use")
            }
            Error::CircularMetaschema { uri } => {
                write!(f, "Circular meta-schema reference detected at '{uri}'")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Allocation(error) => Some(error),
            Error::MetaSchema(error) => Some(error),
            Error::Unretrievable { source, .. } => Some(&**source),
            Error::InvalidUri(error) => Some(error),
            Error::InvalidPercentEncoding { source, .. } => Some(source),
            Error::InvalidArrayIndex { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Errors that can occur during URI handling.
#[derive(Debug)]
pub enum UriError {
    Normalize {
        uri: String,
        error: fluent_uri::normalize::NormalizeError,
    },
    Parse {
        uri: String,
        is_reference: bool,
        error: ParseError,
    },
    Resolve {
        uri: String,
        base: Uri<String>,
        error: ResolveError,
    },
}

impl fmt::Display for UriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UriError::Normalize { uri, error } => write!(f, "Failed to normalize '{uri}': {error}"),
            UriError::Parse {
                uri,
                is_reference,
                error,
            } => {
                if *is_reference {
                    f.write_fmt(format_args!("Invalid URI reference '{uri}': {error}"))
                } else {
                    f.write_fmt(format_args!("Invalid URI '{uri}': {error}"))
                }
            }
            UriError::Resolve { uri, base, error } => f.write_fmt(format_args!(
                "Failed to resolve '{uri}' against '{base}': {error}"
            )),
        }
    }
}

impl std::error::Error for UriError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UriError::Normalize { error, .. } => Some(error),
            UriError::Parse { error, .. } => Some(error),
            UriError::Resolve { error, .. } => Some(error),
        }
    }
}
