use serde_json::Value;

use crate::{
    allocation::{Allocation, AllocationError, Allocator, Unenforced},
    path::{JsonPointerNode, JsonPointerSegment},
    resource::unescape_segment_with_allocations,
};

#[derive(Debug, Default)]
pub(crate) struct ParsedPointer {
    pub(crate) segments: Vec<ParsedPointerSegment>,
}

impl ParsedPointer {
    pub(crate) fn from_json_pointer_with_allocations(
        pointer: &str,
        allocation: &dyn Allocation,
    ) -> Result<Option<Self>, AllocationError> {
        if pointer.is_empty() {
            return Ok(Some(Self::default()));
        }
        if !pointer.starts_with('/') {
            return Ok(None);
        }

        let mut segments = Vec::new();
        let allocation = Allocator(allocation);
        for raw in pointer.split('/').skip(1) {
            let token = unescape_segment_with_allocations(raw, allocation.0)?;
            if let Some(index) = parse_index(&token) {
                allocation.push(&mut segments, ParsedPointerSegment::Index(index))?;
            } else {
                let key = match token {
                    std::borrow::Cow::Borrowed(key) => allocation.copy_str(key)?,
                    std::borrow::Cow::Owned(key) => key,
                };
                allocation.push(&mut segments, ParsedPointerSegment::Key(key))?;
            }
        }
        Ok(Some(Self { segments }))
    }

    pub(crate) fn from_pointer_node_with_allocations(
        path: &JsonPointerNode<'_, '_>,
        allocation: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        let allocation = Allocator(allocation);
        let mut segments = Vec::new();
        let mut head = path;

        while let Some(parent) = head.parent() {
            allocation.push(
                &mut segments,
                match head.segment() {
                    JsonPointerSegment::Key(key) => {
                        ParsedPointerSegment::Key(allocation.copy_str(key.as_ref())?)
                    }
                    JsonPointerSegment::Index(idx) => ParsedPointerSegment::Index(*idx),
                },
            )?;
            head = parent;
        }

        segments.reverse();
        Ok(Self { segments })
    }

    pub(crate) fn try_clone_with_allocations(
        &self,
        allocation: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        let mut segments = Vec::new();
        let allocation = Allocator(allocation);
        allocation.grow_vec(&mut segments, self.segments.len())?;
        for segment in &self.segments {
            segments.push(segment.try_clone_with_allocations(allocation.0)?);
        }
        Ok(Self { segments })
    }

    pub(crate) fn lookup<'a>(&self, document: &'a Value) -> Option<&'a Value> {
        self.segments
            .iter()
            .try_fold(document, |target, token| match token {
                ParsedPointerSegment::Key(key) => match target {
                    Value::Object(map) => map.get(&**key),
                    _ => None,
                },
                ParsedPointerSegment::Index(index) => match target {
                    Value::Array(list) => list.get(*index),
                    _ => None,
                },
            })
    }
}

#[derive(Debug)]
pub(crate) enum ParsedPointerSegment {
    Key(String),
    Index(usize),
}

impl ParsedPointerSegment {
    pub(crate) fn try_clone_with_allocations(
        &self,
        allocation: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        Ok(match self {
            Self::Index(index) => Self::Index(*index),
            Self::Key(key) => Self::Key(Allocator(allocation).copy_str(key)?),
        })
    }
}

/// Look up a value by a JSON Pointer.
///
/// **NOTE**: A slightly faster version of pointer resolution based on `Value::pointer` from `serde_json`.
pub fn pointer<'a>(document: &'a Value, pointer: &str) -> Option<&'a Value> {
    pointer_with_allocations(document, pointer, &Unenforced).unwrap()
}

pub(crate) fn pointer_with_allocations<'a>(
    document: &'a Value,
    pointer: &str,
    allocation: &dyn Allocation,
) -> Result<Option<&'a Value>, AllocationError> {
    if pointer.is_empty() {
        return Ok(Some(document));
    }
    if !pointer.starts_with('/') {
        return Ok(None);
    }
    let mut target = document;
    for raw in pointer.split('/').skip(1) {
        let token = unescape_segment_with_allocations(raw, allocation)?;
        let next = match target {
            Value::Object(map) => map.get(&*token),
            Value::Array(list) => parse_index(&token).and_then(|x| list.get(x)),
            _ => None,
        };
        let Some(next) = next else {
            return Ok(None);
        };
        target = next;
    }
    Ok(Some(target))
}

// Taken from `serde_json`.
#[must_use]
pub fn parse_index(s: &str) -> Option<usize> {
    if s.starts_with('+') || (s.starts_with('0') && s.len() != 1) {
        return None;
    }
    s.parse().ok()
}
