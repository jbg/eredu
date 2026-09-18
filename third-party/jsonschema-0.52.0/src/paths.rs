//! Facilities for working with paths within schemas or validated instances.
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    fmt,
    sync::{Arc, OnceLock},
};

use referencing::{
    unescape_segment, write_escaped_str, write_index, JsonPointerNode, JsonPointerSegment,
};

use crate::keywords::Keyword;

/// A location segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LocationSegment<'a> {
    /// Property name within a JSON object.
    Property(Cow<'a, str>),
    /// JSON Schema keyword.
    Index(usize),
}

impl fmt::Display for LocationSegment<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocationSegment::Property(property) => f.write_str(property),
            LocationSegment::Index(idx) => f.write_str(itoa::Buffer::new().format(*idx)),
        }
    }
}

/// A lazily constructed location within a JSON instance.
///
/// [`LazyLocation`] builds a path incrementally during JSON Schema validation without allocating
/// memory until required by storing each segment on the stack.
pub type LazyLocation<'a, 'b> = JsonPointerNode<'a, 'b>;

/// Cached empty location - very common for root-level errors.
static EMPTY_LOCATION: OnceLock<Location> = OnceLock::new();

thread_local! {
    static LOCATION_BUFFER: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Build an `Arc<str>` in a single allocation.
///
/// `Arc::from(String)` would allocate a second time and copy, so format into a reused
/// buffer instead. `write` must not build a location itself - that panics on the borrow.
#[inline]
pub(crate) fn build_arc_str(capacity: usize, write: impl FnOnce(&mut String)) -> Arc<str> {
    LOCATION_BUFFER.with_borrow_mut(|buffer| {
        buffer.clear();
        buffer.reserve(capacity);
        write(buffer);
        Arc::from(buffer.as_str())
    })
}

impl<'a> From<&'a LazyLocation<'_, '_>> for Location {
    fn from(value: &'a LazyLocation<'_, '_>) -> Self {
        Self::from_lazy_with_funding(value, &crate::compilation::Funding::default())
            .expect("ordinary instance location")
    }
}
impl Location {
    pub(crate) fn from_lazy_with_funding(
        value: &LazyLocation<'_, '_>,
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        const STACK_CAPACITY: usize = 16;
        let mut count = 0usize;
        let mut length = 0usize;
        let mut head = value;
        while let Some(parent) = head.parent() {
            count = count
                .checked_add(1)
                .ok_or_else(|| funding.error(crate::CompilationAllocationError::Overflow))?;
            let segment = match head.segment() {
                JsonPointerSegment::Key(key) => key.len().checked_add(
                    key.bytes()
                        .filter(|byte| matches!(*byte, b'/' | b'~'))
                        .count(),
                ),
                JsonPointerSegment::Index(index) => Some(itoa::Buffer::new().format(*index).len()),
            };
            length = segment
                .and_then(|bytes| bytes.checked_add(1))
                .and_then(|bytes| length.checked_add(bytes))
                .ok_or_else(|| funding.error(crate::CompilationAllocationError::Overflow))?;
            head = parent;
        }
        let mut buffer = funding.string(length)?;
        let mut write = |segment: &JsonPointerSegment<'_>| {
            buffer.push('/');
            match segment {
                JsonPointerSegment::Key(key) => write_escaped_str(&mut buffer, key),
                JsonPointerSegment::Index(index) => write_index(&mut buffer, *index),
            }
        };
        head = value;
        if count <= STACK_CAPACITY {
            let mut segments = [None; STACK_CAPACITY];
            for slot in segments.iter_mut().take(count) {
                *slot = Some(head.segment());
                head = head.parent().expect("counted instance path");
            }
            for segment in segments[..count].iter().rev().flatten() {
                write(segment);
            }
        } else {
            let mut segments = Vec::new();
            funding.grow(&mut segments, count)?;
            while let Some(parent) = head.parent() {
                segments.push(head.segment());
                head = parent;
            }
            for segment in segments.iter().rev() {
                write(segment);
            }
        }
        Self::from_escaped_with_funding(&buffer, funding)
    }
}

/// Tracks `$ref` traversals during validation for evaluation path computation.
///
/// This is a stack-allocated linked list that gets pushed when crossing `$ref` boundaries.
/// Each entry stores the **suffix** of the `$ref` location (path relative to its resource base),
/// which is precomputed at compile time.
///
/// Use `Option<&RefTracker>` throughout the validation code:
/// - `None` means no `$ref` has been traversed yet
/// - `Some(&tracker)` means at least one `$ref` has been crossed
///
/// # Example: Single $ref
///
/// ```text
/// Schema:
/// {
///   "properties": {
///     "user": { "$ref": "#/$defs/Person" }
///   },
///   "$defs": {
///     "Person": { "type": "object" }
///   }
/// }
///
/// Instance: { "user": "not-an-object" }
///
/// At the "type" validator:
///   tracker.prefix() = /properties/user/$ref
///   validator.suffix = /type
///   tracker  = /properties/user/$ref/type
/// ```
///
/// # Example: Nested $refs
///
/// ```text
/// Schema:
/// {
///   "properties": {
///     "order": { "$ref": "#/$defs/Order" }
///   },
///   "$defs": {
///     "Order": {
///       "properties": {
///         "item": { "$ref": "#/$defs/Item" }
///       }
///     },
///     "Item": { "type": "string" }
///   }
/// }
///
/// Instance: { "order": { "item": 123 } }
///
/// At the "type" validator:
///   tracker has two entries:
///     1. suffix = /properties/order/$ref (from root resource)
///     2. suffix = /properties/item/$ref  (from $defs/Order resource)
///   tracker.prefix() = /properties/order/$ref/properties/item/$ref
///   validator.suffix = /type
///   tracker  = /properties/order/$ref/properties/item/$ref/type
/// ```
#[derive(Debug)]
pub(crate) struct RefTracker<'a> {
    /// Path of the `$ref` keyword relative to its resource base.
    /// E.g., `/properties/user/$ref` (not the full canonical path).
    suffix: &'a Location,
    /// The resource base of the `$ref` target.
    /// Used to compute validator suffixes at runtime.
    /// E.g., `/$defs/Person` when `$ref` points to `#/$defs/Person`.
    target_base: &'a Location,
    /// Parent tracker for nested `$ref`s. `None` for the first `$ref` in the chain.
    parent: Option<&'a RefTracker<'a>>,
    /// Cached joined prefix (computed once on first access).
    cached_prefix: std::sync::OnceLock<Location>,
    /// Identifier of the canonical prefix in the current evaluation cache. Zero means uncached.
    cached_prefix_identifier: Cell<usize>,
}

impl<'a> RefTracker<'a> {
    /// Create a new tracker for a `$ref` traversal.
    ///
    /// # Arguments
    /// - `suffix`: The `$ref` keyword's path relative to its resource base
    ///   (precomputed at compile time via `ctx.suffix()`)
    /// - `target_base`: The resource base of the `$ref` target
    ///   (e.g., `/$defs/Person` when `$ref` points to `#/$defs/Person`)
    /// - `parent`: The parent tracker, or `None` if this is the first `$ref`
    #[inline]
    #[must_use]
    pub(crate) fn new(
        suffix: &'a Location,
        target_base: &'a Location,
        parent: Option<&'a RefTracker<'a>>,
    ) -> Self {
        RefTracker {
            suffix,
            target_base,
            parent,
            cached_prefix: std::sync::OnceLock::new(),
            cached_prefix_identifier: Cell::new(0),
        }
    }

    /// Get the joined prefix of all `$ref` suffixes.
    ///
    /// Computed once on first access, then cached.
    #[inline]
    pub(crate) fn prefix(&self) -> &Location {
        self.prefix_with_funding(&crate::compilation::Funding::default())
            .expect("ordinary reference prefix")
    }

    pub(crate) fn prefix_with_funding(
        &self,
        funding: &crate::compilation::Funding,
    ) -> Result<&Location, crate::CompilationError> {
        if let Some(prefix) = self.cached_prefix.get() {
            return Ok(prefix);
        }
        let prefix = match self.parent {
            None => self.suffix.clone(),
            Some(parent) => parent
                .prefix_with_funding(funding)?
                .join_raw_suffix_with_funding(self.suffix.as_str(), funding)?,
        };
        let _ = self.cached_prefix.set(prefix);
        Ok(self
            .cached_prefix
            .get()
            .expect("reference prefix installed"))
    }

    /// Compute the suffix for a canonical location by stripping the `target_base`.
    ///
    /// E.g., for location `/$defs/Person/type` with `target_base` `/$defs/Person`,
    /// returns `/type`.
    #[inline]
    pub(crate) fn compute_suffix(&self, location: &Location) -> Location {
        let suffix = location
            .as_str()
            .strip_prefix(self.target_base.as_str())
            .unwrap_or(location.as_str());
        Location::from_escaped(suffix)
    }

    /// Compute the evaluation path for a validator with the given canonical location.
    ///
    /// This is: `prefix + (location - target_base)`
    #[inline]
    pub(crate) fn evaluation_path_with_prefix(
        &self,
        prefix: &Location,
        location: &Location,
    ) -> Location {
        let suffix = location
            .as_str()
            .strip_prefix(self.target_base.as_str())
            .unwrap_or(location.as_str());
        prefix.join_raw_suffix(suffix)
    }

    #[inline]
    pub(crate) fn parent(&self) -> Option<&RefTracker<'a>> {
        self.parent
    }

    #[inline]
    pub(crate) fn suffix(&self) -> &Location {
        self.suffix
    }

    #[inline]
    pub(crate) fn cached_prefix_identifier(&self) -> Option<usize> {
        match self.cached_prefix_identifier.get() {
            0 => None,
            identifier => Some(identifier),
        }
    }

    #[inline]
    pub(crate) fn cache_prefix_identifier(&self, identifier: usize) {
        debug_assert_ne!(identifier, 0);
        self.cached_prefix_identifier.set(identifier);
    }

    /// Capture evaluation path state at error creation time.
    ///
    /// Returns a lazy evaluation path that defers computation until needed.
    #[inline]
    pub(crate) fn capture(&self, location: &Location) -> LazyEvaluationPath {
        self.capture_with_funding(location, &crate::compilation::Funding::default())
            .expect("ordinary deferred evaluation path")
    }
    pub(crate) fn capture_with_funding(
        &self,
        location: &Location,
        funding: &crate::compilation::Funding,
    ) -> Result<LazyEvaluationPath, crate::CompilationError> {
        let prefix = self.prefix_with_funding(funding)?.clone();
        let suffix = location
            .as_str()
            .strip_prefix(self.target_base.as_str())
            .unwrap_or(location.as_str());
        let suffix = Location::from_escaped_with_funding(suffix, funding)?;
        let capacity = prefix
            .as_str()
            .len()
            .checked_add(suffix.as_str().len())
            .ok_or_else(|| funding.error(crate::CompilationAllocationError::Overflow))?;
        let buffer = std::alloc::Layout::array::<u8>(capacity)
            .map_err(|_| funding.error(crate::CompilationAllocationError::Overflow))?;
        let shared = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(buffer)
            .map_err(|_| funding.error(crate::CompilationAllocationError::Overflow))?
            .0
            .pad_to_align();
        // This immutable one-shot producer reserves both eventual destinations.
        // Shared aliases resolve the same cell, so cloning cannot spend another attempt.
        if capacity != 0 {
            funding.reserve(buffer.size())?;
        }
        funding.reserve(shared.size())?;
        Ok(LazyEvaluationPath::Deferred(funding.arc(
            DeferredEvaluationPath {
                prefix,
                suffix,
                capacity,
                cached: OnceLock::new(),
                funding: funding.clone(),
            },
        )?))
    }
}

/// Compute the evaluation path, using the tracker if present.
#[inline]
pub(crate) fn evaluation_path(
    tracker: Option<&RefTracker<'_>>,
    location: &Location,
    ctx: &mut crate::validator::ValidationContext,
) -> Location {
    match tracker {
        None => location.clone(),
        Some(t) => ctx.evaluation_path(t, location),
    }
}

/// Capture evaluation path state at error creation time.
#[inline]
pub(crate) fn capture_evaluation_path(
    tracker: Option<&RefTracker<'_>>,
    location: &Location,
) -> LazyEvaluationPath {
    match tracker {
        None => LazyEvaluationPath::SameAsSchemaPath,
        Some(t) => t.capture(location),
    }
}

pub(crate) fn capture_evaluation_path_with_funding(
    tracker: Option<&RefTracker<'_>>,
    location: &Location,
    funding: &crate::compilation::Funding,
) -> Result<LazyEvaluationPath, crate::CompilationError> {
    match tracker {
        None => Ok(LazyEvaluationPath::SameAsSchemaPath),
        Some(tracker) => tracker.capture_with_funding(location, funding),
    }
}

/// Lazily-computed evaluation path stored in validation errors.
///
/// # Why lazy?
///
/// Validators like `anyOf` collect errors from all branches but discard them
/// if any branch succeeds. Eagerly computing evaluation paths for discarded
/// errors wastes CPU cycles. We defer the string join until the error is
/// actually displayed.
///
/// # Example 1: No $ref traversal
///
/// ```text
/// Schema: { "properties": { "name": { "type": "string" } } }
/// Instance: { "name": 123 }
///
/// schema_path = tracker = /properties/name/type
///
/// We store: SameAsSchemaPath (zero extra allocation)
/// ```
///
/// # Example 2: Single $ref
///
/// ```text
/// Schema:
/// {
///   "properties": {
///     "user": { "$ref": "#/$defs/Person" }
///   },
///   "$defs": {
///     "Person": { "type": "object" }
///   }
/// }
/// Instance: { "user": "not-an-object" }
///
/// schema_path     = /$defs/Person/type  (canonical location, no $ref)
/// tracker = /properties/user/$ref/type (actual traversal path)
///
/// We store:
///   - prefix: /properties/user/$ref  (from RefTracker)
///   - suffix: /type                  (precomputed at compile time)
///
/// On access: tracker = prefix + suffix
/// ```
///
/// # Example 3: Nested $refs
///
/// ```text
/// Schema:
/// {
///   "properties": {
///     "order": { "$ref": "#/$defs/Order" }
///   },
///   "$defs": {
///     "Order": {
///       "properties": {
///         "item": { "$ref": "#/$defs/Item" }
///       }
///     },
///     "Item": {
///       "properties": {
///         "price": { "type": "number" }
///       }
///     }
///   }
/// }
/// Instance: { "order": { "item": { "price": "free" } } }
///
/// schema_path     = /$defs/Item/properties/price/type
/// tracker = /properties/order/$ref/properties/item/$ref/properties/price/type
///
/// We store:
///   - prefix: /properties/order/$ref/properties/item/$ref  (from RefTracker chain)
///   - suffix: /properties/price/type                       (computed at error creation)
/// ```
#[derive(Debug, Clone)]
pub(crate) enum LazyEvaluationPath {
    SameAsSchemaPath,
    Computed(Location),
    Deferred(Arc<DeferredEvaluationPath>),
}
#[derive(Debug)]
pub(crate) struct DeferredEvaluationPath {
    prefix: Location,
    suffix: Location,
    capacity: usize,
    cached: OnceLock<Location>,
    // All actual backing retires before the original source.
    funding: crate::compilation::Funding,
}
impl DeferredEvaluationPath {
    fn resolve(&self) -> &Location {
        self.cached.get_or_init(|| {
            let mut buffer = String::with_capacity(self.capacity);
            buffer.push_str(self.prefix.as_str());
            buffer.push_str(self.suffix.as_str());
            assert_eq!(
                buffer.len(),
                self.capacity,
                "immutable prepared path extent"
            );
            Location(Arc::from(buffer.as_str()))
        })
    }
}
impl From<Location> for LazyEvaluationPath {
    fn from(location: Location) -> Self {
        Self::Computed(location)
    }
}
impl LazyEvaluationPath {
    pub(crate) fn resolve<'a>(&'a self, schema_path: &'a Location) -> &'a Location {
        match self {
            Self::SameAsSchemaPath => schema_path,
            Self::Computed(location) => location,
            Self::Deferred(path) => path.resolve(),
        }
    }
    pub(crate) fn into_owned(self, schema_path: Location) -> Location {
        match self {
            Self::SameAsSchemaPath => schema_path,
            Self::Computed(location) => location,
            Self::Deferred(path) => path.resolve().clone(),
        }
    }
}

impl<'a> From<&'a Keyword> for LocationSegment<'a> {
    fn from(value: &'a Keyword) -> Self {
        match value {
            Keyword::Builtin(k) => LocationSegment::Property(k.as_str().into()),
            Keyword::Custom(s) => LocationSegment::Property(Cow::Borrowed(s)),
        }
    }
}

impl<'a> From<&'a str> for LocationSegment<'a> {
    #[inline]
    fn from(value: &'a str) -> LocationSegment<'a> {
        LocationSegment::Property(Cow::Borrowed(value))
    }
}

impl<'a> From<&'a String> for LocationSegment<'a> {
    #[inline]
    fn from(value: &'a String) -> LocationSegment<'a> {
        LocationSegment::Property(Cow::Borrowed(value))
    }
}

impl<'a> From<Cow<'a, str>> for LocationSegment<'a> {
    #[inline]
    fn from(value: Cow<'a, str>) -> LocationSegment<'a> {
        LocationSegment::Property(value)
    }
}

impl From<usize> for LocationSegment<'_> {
    #[inline]
    fn from(value: usize) -> Self {
        LocationSegment::Index(value)
    }
}

impl<'a> From<LocationSegment<'a>> for JsonPointerSegment<'a> {
    fn from(value: LocationSegment<'a>) -> Self {
        match value {
            LocationSegment::Property(property) => JsonPointerSegment::Key(property),
            LocationSegment::Index(idx) => JsonPointerSegment::Index(idx),
        }
    }
}

/// A cheap to clone JSON pointer that represents location with a JSON value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Location(Arc<str>);

impl serde::Serialize for Location {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl Location {
    /// Create a new, empty `Location`.
    ///
    /// Returns a cached instance to avoid allocation.
    #[must_use]
    pub fn new() -> Self {
        EMPTY_LOCATION.get_or_init(|| Self(Arc::from(""))).clone()
    }

    pub(crate) fn from_escaped(escaped: &str) -> Self {
        Self::from_escaped_with_funding(escaped, &crate::compilation::Funding::default())
            .expect("ordinary escaped location allocation")
    }
    pub(crate) fn from_escaped_with_funding(
        escaped: &str,
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        Ok(Self(funding.arc_str(escaped)?))
    }

    /// Append a raw JSON pointer suffix (already escaped).
    /// This is more efficient than multiple `join` calls when the suffix
    /// is already a valid JSON pointer path.
    #[must_use]
    pub(crate) fn join_raw_suffix(&self, suffix: &str) -> Self {
        self.join_raw_suffix_with_funding(suffix, &crate::compilation::Funding::default())
            .expect("ordinary joined evaluation path")
    }
    pub(crate) fn join_raw_suffix_with_funding(
        &self,
        suffix: &str,
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        if suffix.is_empty() {
            return Ok(self.clone());
        }
        let capacity = self
            .0
            .len()
            .checked_add(suffix.len())
            .ok_or_else(|| funding.error(crate::CompilationAllocationError::Overflow))?;
        let mut buffer = funding.string(capacity)?;
        buffer.push_str(&self.0);
        buffer.push_str(suffix);
        Self::from_escaped_with_funding(&buffer, funding)
    }

    #[must_use]
    pub fn join<'a>(&self, segment: impl Into<LocationSegment<'a>>) -> Self {
        self.join_with_funding(segment, &crate::compilation::Funding::default())
            .expect("ordinary location allocation")
    }

    pub(crate) fn new_with_funding(
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        Ok(Self(funding.arc_str("")?))
    }

    pub(crate) fn join_with_funding<'a>(
        &self,
        segment: impl Into<LocationSegment<'a>>,
        funding: &crate::compilation::Funding,
    ) -> Result<Self, crate::CompilationError> {
        let segment = segment.into();
        let mut index = itoa::Buffer::new();
        let escaped_len = match &segment {
            LocationSegment::Property(property) => property.len().checked_add(
                property
                    .bytes()
                    .filter(|byte| matches!(*byte, b'~' | b'/'))
                    .count(),
            ),
            LocationSegment::Index(value) => Some(index.format(*value).len()),
        }
        .and_then(|n| n.checked_add(self.0.len()))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| funding.error(crate::CompilationAllocationError::Overflow))?;
        let mut buffer = funding.string(escaped_len)?;
        buffer.push_str(&self.0);
        buffer.push('/');
        match segment {
            LocationSegment::Property(property) => write_escaped_str(&mut buffer, &property),
            LocationSegment::Index(value) => buffer.push_str(index.format(value)),
        }
        if buffer.len() != escaped_len {
            return Err(funding.error(crate::CompilationAllocationError::Capacity));
        }
        Ok(Self(funding.arc_str(&buffer)?))
    }
    /// Address of the backing string, stable while the owner is alive.
    #[must_use]
    pub(crate) fn as_ptr(&self) -> usize {
        Arc::as_ptr(&self.0).cast::<u8>() as usize
    }

    /// Get a clone of the inner `Arc<str>` representing the location.
    #[must_use]
    pub(crate) fn as_arc(&self) -> Arc<str> {
        Arc::clone(&self.0)
    }

    /// Get a string slice representing the location.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Whether this location points at the root of the document.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Get a byte slice representing the location.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    #[must_use]
    pub fn iter(&self) -> std::vec::IntoIter<LocationSegment<'_>> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl Default for Location {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'a> IntoIterator for &'a Location {
    type Item = LocationSegment<'a>;
    type IntoIter = std::vec::IntoIter<LocationSegment<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_str()
            .split('/')
            .filter(|p| !p.is_empty())
            .map(|p| {
                p.parse::<usize>().map_or(
                    LocationSegment::Property(unescape_segment(p)),
                    LocationSegment::Index,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a> FromIterator<LocationSegment<'a>> for Location {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = LocationSegment<'a>>,
    {
        fn inner<'a, 'b, 'c, I>(path_iter: &mut I, location: &'b LazyLocation<'b, 'a>) -> Location
        where
            I: Iterator<Item = LocationSegment<'c>>,
        {
            let Some(path) = path_iter.next() else {
                return location.into();
            };
            let location = location.push(path);
            inner(path_iter, &location)
        }

        let loc = LazyLocation::default();
        inner(&mut iter.into_iter(), &loc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use test_case::test_case;

    #[test]
    fn test_location_default() {
        let loc = Location::default();
        assert_eq!(loc.as_str(), "");
    }

    #[test]
    fn test_location_new() {
        let loc = Location::new();
        assert_eq!(loc.as_str(), "");
    }

    #[test]
    fn test_location_is_empty() {
        let root = Location::new();
        assert!(root.is_empty());
        assert!(!root.join("foo").is_empty());
        assert!(!root.join(0).is_empty());
    }

    #[test]
    fn test_location_join_property() {
        let loc = Location::new();
        let loc = loc.join("property");
        assert_eq!(loc.as_str(), "/property");
    }

    #[test]
    fn test_location_join_index() {
        let loc = Location::new();
        let loc = loc.join(0);
        assert_eq!(loc.as_str(), "/0");
    }

    #[test_case(0, "/0"; "cached index 0")]
    #[test_case(15, "/15"; "cached index 15")]
    #[test_case(16, "/16"; "uncached index 16")]
    #[test_case(100, "/100"; "uncached index 100")]
    fn test_lazy_location_single_index(idx: usize, expected: &str) {
        let root = LazyLocation::new();
        let loc = root.push(idx);
        let location: Location = (&loc).into();
        assert_eq!(location.as_str(), expected);
    }

    #[test]
    fn test_location_join_multiple() {
        let loc = Location::new();
        let loc = loc.join("property").join(0);
        assert_eq!(loc.as_str(), "/property/0");
    }

    #[test]
    fn test_as_bytes() {
        let loc = Location::new().join("test");
        assert_eq!(loc.as_bytes(), b"/test");
    }

    #[test]
    fn test_display_trait() {
        let loc = Location::new().join("property");
        assert_eq!(format!("{loc}"), "/property");
    }

    #[test_case("tilde~character", "/tilde~0character"; "escapes tilde")]
    #[test_case("slash/character", "/slash~1character"; "escapes slash")]
    #[test_case("combo~and/slash", "/combo~0and~1slash"; "escapes tilde and slash combined")]
    #[test_case("multiple~/escapes~", "/multiple~0~1escapes~0"; "multiple escapes")]
    #[test_case("first/segment", "/first~1segment"; "escapes slash in nested segment")]
    fn test_location_escaping(segment: &str, expected: &str) {
        let loc = Location::new().join(segment);
        assert_eq!(loc.as_str(), expected);
    }

    #[test_case("/a/b/c", &[LocationSegment::from("a"), LocationSegment::from("b"), LocationSegment::from("c")]; "location with properties")]
    #[test_case("/1/2/3", &[LocationSegment::Index(1), LocationSegment::Index(2), LocationSegment::Index(3)]; "location with indices")]
    #[test_case("/a/1/b/2", &[
        LocationSegment::from("a"),
        LocationSegment::Index(1),
        LocationSegment::from("b"),
        LocationSegment::Index(2)
    ]; "mixed properties and indices")]
    fn test_into_iter(location: &str, expected_segments: &[LocationSegment]) {
        let loc = Location(Arc::from(location.to_string()));
        assert_eq!(loc.into_iter().collect::<Vec<_>>(), expected_segments);
    }

    #[test_case(vec![LocationSegment::from("a"), LocationSegment::from("b")], "/a/b"; "properties only")]
    #[test_case(vec![LocationSegment::Index(1), LocationSegment::Index(2)], "/1/2"; "indices only")]
    #[test_case(vec![LocationSegment::from("a"), LocationSegment::Index(1)], "/a/1"; "mixed segments")]
    fn test_from_iter(segments: Vec<LocationSegment>, expected: &str) {
        assert_eq!(Location::from_iter(segments).as_str(), expected);
    }

    #[test]
    fn test_roundtrip_join_iter_rebuild_equals() {
        let loc = Location::new().join("a/b").join(2).join("x~y");

        let segments: Vec<_> = loc.into_iter().collect();

        let rebuilt = segments
            .into_iter()
            .fold(Location::new(), |acc, seg| match seg {
                LocationSegment::Property(p) => acc.join(p),
                LocationSegment::Index(i) => acc.join(i),
            });

        assert_eq!(loc, rebuilt);
    }

    #[test]
    fn test_validate_error_instance_path_traverses_instance() {
        let schema = json!({
            "type": "object",
            "properties": {
                "table-node": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": { "~": { "type": "string", "minLength": 1 } },
                        "required": ["~"],
                    }
                }
            },
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        });
        let instance = json!({
            "table-node": [
                { "~": "" },
                { "other-value": "" },
            ],
        });

        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        // Traverse instance using the `instance_path`` segments
        let mut current = &instance;
        for segment in error.instance_path() {
            match segment {
                LocationSegment::Property(property) => {
                    current = &current[property.as_ref()];
                }
                LocationSegment::Index(idx) => {
                    current = &current[idx];
                }
            }
        }
        assert_eq!(
            current,
            instance
                .pointer("/table-node/0/~0")
                .expect("Pointer is valid")
        );
    }

    #[test]
    fn test_deep_path_heap_allocation() {
        // Create a schema with >16 levels of nesting to exercise heap allocation path
        const DEPTH: usize = 20;

        let mut schema = json!({"type": "integer"});
        for i in (0..DEPTH).rev() {
            schema = json!({
                "type": "object",
                "properties": {
                    format!("level{i}"): schema
                }
            });
        }

        let mut instance = json!("not an integer");
        for i in (0..DEPTH).rev() {
            instance = json!({
                format!("level{i}"): instance
            });
        }

        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        // Verify instance_path has correct depth (>16 exercises heap allocation)
        let instance_path = error.instance_path();
        assert_eq!(instance_path.into_iter().count(), DEPTH);
        let expected_instance_path = (0..DEPTH).fold(String::new(), |mut acc, i| {
            use std::fmt::Write;
            write!(acc, "/level{i}").unwrap();
            acc
        });
        assert_eq!(instance_path.as_str(), expected_instance_path);

        // Verify schema_path also has correct depth
        let schema_path = error.schema_path();
        let expected_schema_path = (0..DEPTH).fold(String::new(), |mut acc, i| {
            use std::fmt::Write;
            write!(acc, "/properties/level{i}").unwrap();
            acc
        }) + "/type";
        assert_eq!(schema_path.as_str(), expected_schema_path);
    }

    #[test]
    fn test_dynamic_ref_evaluation_path() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$dynamicAnchor": "node",
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "child": {"$dynamicRef": "#node"}
            }
        });
        let instance = json!({
            "data": "parent",
            "child": {
                "data": 123
            }
        });

        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        // schema_path is canonical (anchor resolves to root)
        assert_eq!(error.schema_path().as_str(), "/properties/data/type");
        // evaluation_path includes $dynamicRef traversal
        assert_eq!(
            error.evaluation_path().as_str(),
            "/properties/child/$dynamicRef/properties/data/type"
        );
        assert_eq!(error.instance_path().as_str(), "/child/data");
    }

    #[test]
    fn test_nested_ref_evaluation_path() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {
                "order": { "$ref": "#/$defs/Order" }
            },
            "$defs": {
                "Order": {
                    "properties": {
                        "item": { "$ref": "#/$defs/Item" }
                    }
                },
                "Item": { "type": "string" }
            }
        });
        let instance = json!({ "order": { "item": 123 } });

        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        assert_eq!(error.schema_path().as_str(), "/$defs/Item/type");
        assert_eq!(
            error.evaluation_path().as_str(),
            "/properties/order/$ref/properties/item/$ref/type"
        );
        assert_eq!(error.instance_path().as_str(), "/order/item");
    }

    #[test]
    fn test_ref_to_boolean_schema_evaluation_path() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "deny": false
            },
            "$ref": "#/$defs/deny"
        });

        let instance = json!("anything");
        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        assert_eq!(error.schema_path().as_str(), "/$defs/deny");
        assert_eq!(error.evaluation_path().as_str(), "/$ref");
    }

    #[test]
    fn test_no_ref_evaluation_path_equals_schema_path() {
        let schema = json!({
            "properties": {
                "name": { "type": "string" }
            }
        });
        let instance = json!({ "name": 123 });

        let error = crate::validate(&schema, &instance).expect_err("Should fail");

        assert_eq!(error.schema_path().as_str(), "/properties/name/type");
        assert_eq!(
            error.evaluation_path().as_str(),
            error.schema_path().as_str()
        );
    }
}
