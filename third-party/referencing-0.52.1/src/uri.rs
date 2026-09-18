//! URI handling utilities for JSON Schema references.
use fluent_uri::{
    pct_enc::{encoder::Fragment, EStr, Encoder},
    Uri, UriRef,
};
use std::sync::LazyLock;

use crate::{
    allocation::{Allocation, AllocationError, Allocator, Unenforced},
    Error, UriError,
};
pub use fluent_uri::pct_enc::encoder::Path;

/// Resolves the URI reference against the given base URI and returns the target URI.
///
/// # Errors
///
/// Returns an error if `uri` is not a valid URI reference or cannot be resolved against `base`.
pub fn resolve_against(base: &Uri<&str>, uri: &str) -> Result<Uri<String>, Error> {
    resolve_against_with_allocations(base, uri, &Unenforced)
}

/// Resolve through the shared worker with prospective storage admission.
pub fn resolve_against_with_allocations(
    base: &Uri<&str>,
    uri: &str,
    allocation: &dyn Allocation,
) -> Result<Uri<String>, Error> {
    if uri.starts_with('#') && base.as_str().ends_with(uri) {
        return Ok(base.to_owned_with_allocations(allocation)?);
    }
    // RFC 3986, 5.2.1: the base URI's fragment is undefined and takes no part in resolution.
    // Drafts 4-7 allow `$id` to carry one, so drop it rather than reject the base.
    let without_fragment;
    let base = match base.fragment() {
        Some(fragment) => {
            let full = base.as_str();
            let end = full.len() - fragment.len() - "#".len();
            without_fragment = Uri::parse(&full[..end])
                .map_err(|error| parsing_error(full, false, error, allocation))?;
            &without_fragment
        }
        None => base,
    };
    Ok(UriRef::parse(uri)
        .map_err(|error| parsing_error(uri, true, error, allocation))?
        .resolve_against_with_allocations(base, allocation)
        .map_err(|error| resolving_error(uri, *base, error, allocation))?
        .normalize_with_allocations(allocation)
        .map_err(|error| normalizing_error(uri, error, allocation))?)
}

/// Parses a URI reference from a string into a [`crate::Uri`].
///
/// # Errors
///
/// Returns an error if the input string does not conform to URI-reference from RFC 3986.
pub fn from_str(uri: &str) -> Result<Uri<String>, Error> {
    from_str_with_allocations(uri, &Unenforced)
}

/// Parse and normalize through the shared worker with prospective storage admission.
pub fn from_str_with_allocations(
    uri: &str,
    allocation: &dyn Allocation,
) -> Result<Uri<String>, Error> {
    let uriref = UriRef::parse(uri)
        .map_err(|error| parsing_error(uri, true, error, allocation))?
        .normalize_with_allocations(allocation)
        .map_err(|error| normalizing_error(uri, error, allocation))?;
    if uriref.has_scheme() {
        // Move the already normalized owner. The checked scheme predicate guarantees this conversion.
        Ok(Uri::try_from(uriref).expect("URI reference with a scheme is a URI"))
    } else {
        Ok(uriref
            .resolve_against_with_allocations(&*DEFAULT_ROOT_URI, allocation)
            .map_err(|error| resolving_error(uri, *DEFAULT_ROOT_URI, error, allocation))?)
    }
}

fn parsing_error(
    uri: &str,
    is_reference: bool,
    error: fluent_uri::ParseError,
    allocation: &dyn Allocation,
) -> Error {
    match Allocator(allocation).copy_str(uri) {
        Ok(uri) => Error::InvalidUri(UriError::Parse {
            uri,
            is_reference,
            error,
        }),
        Err(error) => Error::Allocation(error),
    }
}
fn resolving_error(
    uri: &str,
    base: Uri<&str>,
    error: fluent_uri::resolve::ResolveError,
    allocation: &dyn Allocation,
) -> Error {
    if let fluent_uri::resolve::ResolveError::Allocation(error) = error {
        return Error::Allocation(error);
    }
    let result = (|| {
        Ok::<_, AllocationError>((
            Allocator(allocation).copy_str(uri)?,
            base.to_owned_with_allocations(allocation)?,
        ))
    })();
    match result {
        Ok((uri, base)) => Error::InvalidUri(UriError::Resolve { uri, base, error }),
        Err(error) => Error::Allocation(error),
    }
}
fn normalizing_error(
    uri: &str,
    error: fluent_uri::normalize::NormalizeError,
    allocation: &dyn Allocation,
) -> Error {
    if let fluent_uri::normalize::NormalizeError::Allocation(error) = error {
        return Error::Allocation(error);
    }
    match Allocator(allocation).copy_str(uri) {
        Ok(uri) => Error::InvalidUri(UriError::Normalize { uri, error }),
        Err(error) => Error::Allocation(error),
    }
}

pub(crate) static DEFAULT_ROOT_URI: LazyLock<Uri<&'static str>> =
    LazyLock::new(|| Uri::parse("json-schema:///").expect("Invalid URI"));

pub type EncodedString = EStr<Fragment>;

// Adapted from `https://github.com/yescallop/fluent-uri-rs/blob/main/src/encoding/table.rs#L153`
pub fn encode_to(input: &str, buffer: &mut String) {
    encode_to_with_allocations(input, buffer, &Unenforced).unwrap()
}

/// Percent-encode after admitting the exact output growth before modifying the buffer.
pub fn encode_to_with_allocations(
    input: &str,
    buffer: &mut String,
    allocation: &dyn Allocation,
) -> Result<(), AllocationError> {
    let additional = input.chars().try_fold(0usize, |len, ch| {
        let bytes = if Path::TABLE.allows(ch) {
            ch.len_utf8()
        } else {
            ch.len_utf8() * 3
        };
        len.checked_add(bytes).ok_or(AllocationError::SizeOverflow)
    })?;
    Allocator(allocation).grow_string(buffer, additional)?;
    const HEX_TABLE: [u8; 512] = {
        const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

        let mut i = 0;
        let mut table = [0; 512];
        while i < 256 {
            table[i * 2] = HEX_DIGITS[i >> 4];
            table[i * 2 + 1] = HEX_DIGITS[i & 0b1111];
            i += 1;
        }
        table
    };

    for ch in input.chars() {
        if Path::TABLE.allows(ch) {
            buffer.push(ch);
        } else {
            for x in ch.encode_utf8(&mut [0; 4]).bytes() {
                buffer.push('%');
                buffer.push(HEX_TABLE[x as usize * 2] as char);
                buffer.push(HEX_TABLE[x as usize * 2 + 1] as char);
            }
        }
    }
    Ok(())
}
