//! Move-only scalar planning around the ordinary JSON number deserializer.
//!
//! Planning borrows the original bytes and performs no allocation. Parsing must
//! be invoked only after the caller admits the returned storage requirement.
//! This module owns no memory grant. Additional numeric feature profiles remain
//! explicit refusals until their actual scratch/storage producer is closed.
use crate::{Error, Number};
use core::{fmt, mem::size_of, ops::Range};

/// Fixed planning refusal without an allocated serde error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Arbitrary-precision retained Number storage is a separate profile.
    Features,
    /// The slice is not a numeric spelling or exceeds the ordinary i32 counter.
    Input,
    /// A requested control layout overflowed.
    Overflow,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Features => "JSON scalar storage feature profile is unqualified",
            Self::Input => "JSON scalar input profile is unqualified",
            Self::Overflow => "JSON scalar storage layout overflow",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}

/// Concrete parser controls, synchronous scratch and one possible syntax error.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    failure: usize,
    temporary: usize,
    controls: usize,
}
impl Requirements {
    /// Actual private error Box payload. A returned Number owns no heap storage.
    pub fn failure_bytes(self) -> usize {
        self.failure
    }
    /// Synchronous parser scratch and at most two finite lexical limb buffers.
    /// These retire before success or a returned syntax error leaves parse.
    pub fn temporary_bytes(self) -> usize {
        self.temporary
    }
    /// Actual scalar parser, caller/return, and fixed planning controls.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
    /// Checked sum to admit before parsing.
    pub fn required_bytes(self) -> usize {
        self.failure + self.temporary + self.controls
    }
}

/// One numeric slice tied to the original complete immutable source.
#[derive(Debug)]
pub struct Plan<'a> {
    source: &'a str,
    range: Range<usize>,
    requirements: Requirements,
}
impl<'a> Plan<'a> {
    /// Borrow a numeric slice without deserializing or allocating. The spelling
    /// check only confines the ordinary parser to numeric/syntax-error branches;
    /// it does not replace JSON syntax or finite-range validation during parse.
    pub fn prepare(source: &'a str, range: Range<usize>) -> Result<Self, PlanError> {
        if cfg!(feature = "arbitrary_precision") {
            return Err(PlanError::Features);
        }
        let raw = tri!(source.get(range.clone()).ok_or(PlanError::Input));
        if raw.len() > i32::MAX as usize
            || !matches!(raw.as_bytes().first(), Some(b'-' | b'0'..=b'9'))
            || !raw
                .bytes()
                .all(|byte| matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            return Err(PlanError::Input);
        }
        let parts = [
            tri!(crate::de::bounded_number_control_bytes().ok_or(PlanError::Overflow)),
            size_of::<Self>(),
            size_of::<Result<Self, PlanError>>(),
            size_of::<Requirements>(),
            size_of::<PlanError>(),
            size_of::<(&str, Range<usize>, core::str::Bytes<'a>, bool)>(),
        ];
        let controls = tri!(parts
            .into_iter()
            .try_fold(size_of::<[usize; 6]>(), usize::checked_add)
            .ok_or(PlanError::Overflow));
        let failure = Error::bounded_number_storage_bytes();
        let temporary =
            tri!(crate::de::bounded_number_temporary_bytes(raw.len()).ok_or(PlanError::Overflow));
        tri!(controls
            .checked_add(failure)
            .and_then(|n| n.checked_add(temporary))
            .ok_or(PlanError::Overflow));
        Ok(Self {
            source,
            range,
            requirements: Requirements {
                failure,
                temporary,
                controls,
            },
        })
    }
    /// Exact known storage before the first ordinary parser invocation.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Consume this plan through the same deserializer as ordinary Number/Value.
    /// The original serde error retains its exact code and document position.
    pub fn parse(self) -> Result<Number, Error> {
        crate::de::from_bounded_number_slice(self.source[self.range.clone()].as_bytes())
            .map_err(|error: Error| error.relocate_number(&self.source[..self.range.start]))
    }
}
