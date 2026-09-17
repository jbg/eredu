//! Owning text events may outlive their state, publisher and callback.
use crate::HostPreparationAuthority;
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Payload {
    bytes: Vec<u8>,
    // The last closed Arc is freed before this payload returns its funding.
    host: HostPreparationAuthority,
}
#[derive(Debug)]
enum Storage {
    Ordinary(String),
    Retained(Option<Arc<Payload>>),
}

/// Immutable text carried by semantic events. Ordinary strings retain their
/// existing copy behavior; prepared text clones only its closed paying owner.
/// Serialization remains a string. There is no managed String/Vec extraction.
#[derive(Debug)]
pub struct SemanticText(Storage);

/// A failed fresh text destination retains custody through its actual cause.
#[derive(Debug, thiserror::Error)]
#[error("semantic text destination allocation failed")]
pub struct SemanticTextAllocationError {
    #[source]
    cause: Option<std::collections::TryReserveError>,
    host: HostPreparationAuthority,
}
impl SemanticText {
    /// Exact UTF-8 contents, borrowed for no longer than this owner.
    pub fn as_str(&self) -> &str {
        match &self.0 {
            Storage::Ordinary(s) => s,
            Storage::Retained(s) => std::str::from_utf8(&s.as_ref().expect("live text").bytes)
                .expect("only an exact UTF-8 source can construct text"),
        }
    }
    /// Requested text buffer, closed Arc and actual constructor/error controls.
    /// The caller pays this before construction; H alone is lifetime custody.
    pub fn retained_control_bytes(bytes: usize) -> Option<usize> {
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            Layout::array::<u8>(bytes).ok()?.size(),
            arc,
            size_of::<Payload>(),
            size_of::<Self>(),
            size_of::<Storage>(),
            size_of::<Vec<u8>>(),
            size_of::<Arc<Payload>>(),
            size_of::<Option<Arc<Payload>>>(),
            size_of::<Option<Payload>>(),
            size_of::<SemanticTextAllocationError>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Result<Self, SemanticTextAllocationError>>(),
            size_of::<(&str, HostPreparationAuthority)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Copies the exact source into a new paid destination. Existing ordinary
    /// strings cannot be relabelled retained by this operation.
    pub fn try_copy_retained(
        text: &str,
        host: HostPreparationAuthority,
    ) -> Result<Self, SemanticTextAllocationError> {
        let mut bytes = Vec::new();
        if let Err(cause) = bytes.try_reserve_exact(text.len()) {
            drop(bytes);
            return Err(SemanticTextAllocationError {
                cause: Some(cause),
                host,
            });
        }
        if bytes.capacity() != text.len() {
            drop(bytes);
            return Err(SemanticTextAllocationError { cause: None, host });
        }
        bytes.extend_from_slice(text.as_bytes());
        Ok(Self(Storage::Retained(Some(Arc::new(Payload {
            bytes,
            host,
        })))))
    }
    /// Bytes copied by an ordinary snapshot. Prepared clones retain their
    /// original immutable allocation rather than allocating a second String.
    pub fn snapshot_copy_bytes(&self) -> usize {
        match &self.0 {
            Storage::Ordinary(s) => s.capacity(),
            Storage::Retained(_) => 0,
        }
    }
}
impl Clone for SemanticText {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            Storage::Ordinary(s) => Storage::Ordinary(s.clone()),
            Storage::Retained(s) => {
                Storage::Retained(Some(Arc::clone(s.as_ref().expect("live text"))))
            }
        })
    }
}
impl Drop for SemanticText {
    fn drop(&mut self) {
        if let Storage::Retained(s) = &mut self.0 {
            if let Some(owner) = s.take() {
                drop(Arc::into_inner(owner));
            }
        }
    }
}
impl From<String> for SemanticText {
    fn from(s: String) -> Self {
        Self(Storage::Ordinary(s))
    }
}
impl From<&str> for SemanticText {
    fn from(s: &str) -> Self {
        s.to_owned().into()
    }
}
impl Deref for SemanticText {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl AsRef<str> for SemanticText {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl fmt::Display for SemanticText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}
impl PartialEq for SemanticText {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
// Preserve the string comparisons used by existing semantic-event consumers.
impl PartialEq<str> for SemanticText {
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}
impl PartialEq<&str> for SemanticText {
    fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}
impl PartialEq<String> for SemanticText {
    fn eq(&self, other: &String) -> bool { self.as_str() == other.as_str() }
}
impl Eq for SemanticText {}
impl serde::Serialize for SemanticText {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}
impl<'de> serde::Deserialize<'de> for SemanticText {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <String as serde::Deserialize>::deserialize(d).map(Into::into)
    }
}
