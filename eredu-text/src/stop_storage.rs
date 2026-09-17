//! Literal UTF-8 stop matching with fixed destinations. Selection and generation
//! policy belong to callers. This utility grants no funding or execution right.
use std::{alloc::Layout, collections::TryReserveError, ops::Range};

mod compiler;
pub use compiler::{StopCompileFailure, StopCompilePlan, StopCompileRequirements};

/// The common byte-search/frontier decision used by owned legacy wrappers and
/// fixed storage. The input is complete UTF-8; stop entries are nonempty UTF-8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopDecision {
    /// Prefix that can become visible.
    pub visible: Range<usize>,
    /// Index of the first matching entry, resolving ties by source order.
    pub matched: Option<usize>,
    /// Retained proper-prefix suffix; empty after a match.
    pub pending: Range<usize>,
}
/// Finds the earliest literal match, or withholds the longest possible prefix.
/// No allocation, mutation, normalization or parser policy occurs here.
pub fn decide<'a>(input: &str, stops: impl Clone + Iterator<Item = &'a str>) -> StopDecision {
    let bytes = input.as_bytes();
    if let Some((position, index)) = (0..bytes.len()).find_map(|position| {
        stops
            .clone()
            .position(|stop| !stop.is_empty() && bytes[position..].starts_with(stop.as_bytes()))
            .map(|index| (position, index))
    }) {
        return StopDecision {
            visible: 0..position,
            matched: Some(index),
            pending: 0..0,
        };
    }
    let longest = stops
        .clone()
        .map(str::len)
        .max()
        .unwrap_or(0)
        .min(bytes.len());
    let tail = (1..=longest)
        .rev()
        .find(|&n| {
            stops.clone().any(|stop| {
                !stop.is_empty() && stop.as_bytes().starts_with(&bytes[bytes.len() - n..])
            })
        })
        .unwrap_or(0);
    let end = bytes.len() - tail;
    StopDecision {
        visible: 0..end,
        matched: None,
        pending: end..bytes.len(),
    }
}

/// Cold source construction failure. Cold temporary packing is caller-owned.
#[derive(Debug, thiserror::Error)]
pub enum StopSourceError {
    /// A source or requested destination extent overflowed.
    #[error("literal stop storage extent overflow")]
    Overflow,
    /// Cold source allocation failed.
    #[error("literal stop source allocation failed")]
    Allocation(#[from] TryReserveError),
}
#[derive(Debug)]
struct Entry {
    start: usize,
    end: usize,
}
/// One immutable packed source, preserving first occurrence order and omitting
/// empty/duplicate stops. No Clone, replacement or mutable table accessor exists.
#[derive(Debug)]
pub struct PreparedStopSource {
    entries: Vec<Entry>,
    bytes: Vec<u8>,
    longest: usize,
}
impl PreparedStopSource {
    /// Packs the actual ordered strings. Final extents are measured separately
    /// from the compatibility reference collection; this constructor makes no budget claim.
    pub fn prepare<'a>(stops: impl IntoIterator<Item = &'a str>) -> Result<Self, StopSourceError> {
        let mut unique = Vec::<&str>::new();
        for stop in stops {
            if !stop.is_empty() && !unique.contains(&stop) {
                unique.try_reserve(1)?;
                unique.push(stop);
            }
        }
        // Compatibility retains its existing fallible temporary references.
        // Original admission uses a borrowed concrete plan directly instead.
        StopCompilePlan::prepare_refs(&unique)?
            .compile()
            .map_err(StopCompileFailure::discard_partial)
    }
    /// Actual source entries in the selected order.
    pub fn stops(&self) -> impl Clone + Iterator<Item = &str> {
        self.entries.iter().map(|e| {
            std::str::from_utf8(&self.bytes[e.start..e.end]).expect("validated source UTF-8")
        })
    }
    /// Actual immutable longest entry length, zero for an empty program.
    pub fn longest_stop_bytes(&self) -> usize {
        self.longest
    }
    /// Actual retained table capacities plus the concrete source representation.
    /// Allocator overhead and cold packing temporaries are not included.
    pub fn storage_bytes(&self) -> Option<usize> {
        size_of::<Self>()
            .checked_add(self.entries.capacity().checked_mul(size_of::<Entry>())?)?
            .checked_add(self.bytes.capacity())
    }
}

/// Fixed transition rejection; no owned diagnostic or buffer growth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StopStorageError {
    /// Checked layout arithmetic failed.
    #[error("literal stop storage extent overflow")]
    Overflow,
    /// Destination or retained frontier violates the sealed geometry.
    #[error("invalid literal stop destination")]
    InvalidStorage,
    /// An incoming suffix exceeds the originally selected per-call extent.
    #[error("literal stop input exceeds its original extent")]
    InputLimit,
    /// Local destinations have not completed their one preparation.
    #[error("literal stop storage is not prepared")]
    Unprepared,
}
/// A complete visible UTF-8 range and the selected stop, borrowed until the next
/// mutation. The caller decides its semantic events and termination policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopOutput<'a> {
    /// Visible prefix, possibly empty.
    pub visible: &'a str,
    /// Matched immutable source entry, if this call found a stop.
    pub matched: Option<&'a str>,
}
/// Pointer-free mutable frontier. Pending bytes remain in the supplied destination.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StopProgress {
    pending: Range<usize>,
    matched: bool,
}
impl StopProgress {
    /// Complete retained suffix, for inspection/copy into separately owned storage.
    pub fn pending<'a>(&self, storage: &'a [u8]) -> Result<&'a str, StopStorageError> {
        std::str::from_utf8(
            storage
                .get(self.pending.clone())
                .ok_or(StopStorageError::InvalidStorage)?,
        )
        .map_err(|_| StopStorageError::InvalidStorage)
    }
    /// Whether a previous call matched; no subsequent text becomes visible.
    pub fn is_matched(&self) -> bool {
        self.matched
    }
}
/// Layout derived from actual stops and a selected per-call decoded-text extent.
/// The utility's caller must authenticate that extent before original admission.
#[derive(Debug, Clone, Copy)]
pub struct StopStreamLayout<'a> {
    source: &'a PreparedStopSource,
    maximum_input: usize,
    bytes: usize,
}
impl<'a> StopStreamLayout<'a> {
    /// Uses L + max(K-1,0) for nonempty programs; no program requires no scratch.
    pub fn for_source(
        source: &'a PreparedStopSource,
        maximum_input: usize,
    ) -> Result<Self, StopStorageError> {
        let bytes = if source.entries.is_empty() {
            0
        } else {
            maximum_input
                .checked_add(source.longest.saturating_sub(1))
                .ok_or(StopStorageError::Overflow)?
        };
        Layout::array::<u8>(bytes).map_err(|_| StopStorageError::Overflow)?;
        Ok(Self {
            source,
            maximum_input,
            bytes,
        })
    }
    /// Exact requested mutable byte extent.
    pub fn destination_bytes(&self) -> usize {
        self.bytes
    }
    /// Executes one common decision without growing storage. Every check precedes
    /// mutation. Visible/pending ranges coexist; next call compacts the old tail.
    pub fn step<'b>(
        &self,
        progress: &mut StopProgress,
        destination: &'b mut [u8],
        text: &'b str,
    ) -> Result<StopOutput<'b>, StopStorageError>
    where
        'a: 'b,
    {
        if destination.len() != self.bytes {
            return Err(StopStorageError::InvalidStorage);
        }
        if text.len() > self.maximum_input {
            return Err(StopStorageError::InputLimit);
        }
        let old = progress.pending(destination)?;
        if old.len() > self.source.longest.saturating_sub(1) {
            return Err(StopStorageError::InvalidStorage);
        }
        if progress.matched {
            return Ok(StopOutput {
                visible: "",
                matched: None,
            });
        }
        if self.source.entries.is_empty() {
            return Ok(StopOutput {
                visible: text,
                matched: None,
            });
        }
        let old_len = old.len();
        let used = old_len
            .checked_add(text.len())
            .filter(|n| *n <= self.bytes)
            .ok_or(StopStorageError::InvalidStorage)?;
        destination.copy_within(progress.pending.clone(), 0);
        destination[old_len..used].copy_from_slice(text.as_bytes());
        let decision = decide(
            std::str::from_utf8(&destination[..used]).expect("two complete UTF-8 fragments"),
            self.source.stops(),
        );
        progress.pending = decision.pending;
        progress.matched = decision.matched.is_some();
        Ok(StopOutput {
            visible: std::str::from_utf8(&destination[decision.visible])
                .expect("literal stop boundary"),
            matched: decision
                .matched
                .and_then(|index| self.source.stops().nth(index)),
        })
    }
    /// Flushes lookbehind only after the caller's decoder finish has succeeded.
    pub fn finish<'b>(
        &self,
        progress: &mut StopProgress,
        destination: &'b [u8],
    ) -> Result<&'b str, StopStorageError> {
        if destination.len() != self.bytes {
            return Err(StopStorageError::InvalidStorage);
        }
        let text = progress.pending(destination)?;
        if text.len() > self.source.longest.saturating_sub(1) {
            return Err(StopStorageError::InvalidStorage);
        }
        progress.pending = 0..0;
        Ok(if progress.matched { "" } else { text })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Dormant,
    Attempted,
    Ready,
}
/// Local preparation failure. Partial storage stays in the same owning object.
#[derive(Debug, thiserror::Error)]
pub enum StopPreparationError {
    /// A failed attempt cannot request another allocation.
    #[error("literal stop preparation already attempted")]
    AlreadyAttempted,
    /// The exact requested allocation could not be made.
    #[error("literal stop allocation failed")]
    Allocation(#[from] TryReserveError),
}
/// Pointer-free destinations and progress for the single literal-stop kernel.
/// Callers supply the same immutable source for every transition. Concrete
/// ownership wrappers establish source identity; this object grants no custody.
#[derive(Debug)]
pub struct StopDestinations {
    maximum_input: usize,
    requested: usize,
    destination: Vec<u8>,
    progress: StopProgress,
    phase: Phase,
}
impl StopDestinations {
    /// Borrows an actual source to derive dormant requested destinations.
    pub fn new(
        source: &PreparedStopSource,
        maximum_input: usize,
    ) -> Result<Self, StopStorageError> {
        let requested = StopStreamLayout::for_source(source, maximum_input)?.destination_bytes();
        Ok(Self {
            maximum_input,
            requested,
            destination: Vec::new(),
            progress: StopProgress::default(),
            phase: Phase::Dormant,
        })
    }
    /// Exact additional mutable destination extent.
    pub fn destination_bytes(&self) -> usize {
        self.requested
    }
    /// Named constructor/local preparation/transition controls. A containing
    /// provider separately counts its stored inline value and error erasure.
    pub fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Result<Self, StopStorageError>>(),
            size_of::<StopStreamLayout<'static>>(),
            size_of::<Result<StopStreamLayout<'static>, StopStorageError>>(),
            size_of::<StopDecision>(),
            size_of::<StopOutput<'static>>(),
            size_of::<Result<StopOutput<'static>, StopStorageError>>(),
            size_of::<Result<&'static str, StopStorageError>>(),
            size_of::<Result<(), StopPreparationError>>(),
            size_of::<StopPreparationError>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    fn validate_source(&self, source: &PreparedStopSource) -> Result<(), StopStorageError> {
        if StopStreamLayout::for_source(source, self.maximum_input)?.destination_bytes()
            != self.requested
        {
            return Err(StopStorageError::InvalidStorage);
        }
        Ok(())
    }
    /// One local attempt; no hot shrink or reallocation. Caller owns readiness.
    pub fn prepare_destination(&mut self) -> Result<(), StopPreparationError> {
        self.prepare_with(|| Ok(()))
    }
    // Production passes only a zero-sized no-op. Tests interrupt after actual
    // allocation to verify that failure retains its exact partial destination.
    fn prepare_with(
        &mut self,
        mut after: impl FnMut() -> Result<(), TryReserveError>,
    ) -> Result<(), StopPreparationError> {
        match self.phase {
            Phase::Ready => return Ok(()),
            Phase::Attempted => return Err(StopPreparationError::AlreadyAttempted),
            Phase::Dormant => self.phase = Phase::Attempted,
        }
        self.destination.try_reserve_exact(self.requested)?;
        self.destination.resize(self.requested, 0);
        after()?;
        self.phase = Phase::Ready;
        Ok(())
    }
    /// Lends visible text while excluding another mutation or destruction.
    pub fn step<'a>(
        &'a mut self,
        source: &'a PreparedStopSource,
        text: &'a str,
    ) -> Result<StopOutput<'a>, StopStorageError> {
        if self.phase != Phase::Ready {
            return Err(StopStorageError::Unprepared);
        }
        self.validate_source(source)?;
        let layout = StopStreamLayout {
            source,
            maximum_input: self.maximum_input,
            bytes: self.requested,
        };
        layout.step(&mut self.progress, &mut self.destination, text)
    }
    /// Finishes without allocating. A never-started zero-output owner has no tail.
    pub fn finish(&mut self, source: &PreparedStopSource) -> Result<&str, StopStorageError> {
        if self.phase == Phase::Dormant {
            return Ok("");
        }
        if self.phase != Phase::Ready {
            return Err(StopStorageError::Unprepared);
        }
        self.validate_source(source)?;
        StopStreamLayout {
            source,
            maximum_input: self.maximum_input,
            bytes: self.requested,
        }
        .finish(&mut self.progress, &self.destination)
    }
}

/// Concrete compatibility source owner around the same pointer-free kernel.
///
/// ```compile_fail
/// fn overlap(storage: &mut eredu_text::stop_storage::OwnedStopStorage) {
///     let text = storage.step("a").unwrap();
///     storage.step("b").unwrap();
///     println!("{}", text.visible);
/// }
/// ```
#[derive(Debug)]
pub struct OwnedStopStorage {
    destinations: StopDestinations,
    source: PreparedStopSource,
}
impl OwnedStopStorage {
    /// Consumes one actual source and derives its dormant destinations.
    pub fn new(source: PreparedStopSource, maximum_input: usize) -> Result<Self, StopStorageError> {
        let destinations = StopDestinations::new(&source, maximum_input)?;
        Ok(Self {
            destinations,
            source,
        })
    }
    /// Borrows the exact immutable source.
    pub fn source(&self) -> &PreparedStopSource {
        &self.source
    }
    /// Exact requested mutable byte extent.
    pub fn destination_bytes(&self) -> usize {
        self.destinations.destination_bytes()
    }
    /// Actual wrapper and shared kernel constructor/return controls.
    pub fn control_bytes() -> Option<usize> {
        StopDestinations::control_bytes()?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<PreparedStopSource>())?
            .checked_add(size_of::<Result<Self, StopStorageError>>())
    }
    /// One local allocation attempt through the same kernel.
    pub fn prepare_destination(&mut self) -> Result<(), StopPreparationError> {
        self.destinations.prepare_destination()
    }
    #[cfg(test)]
    fn prepare_with(
        &mut self,
        after: impl FnMut() -> Result<(), TryReserveError>,
    ) -> Result<(), StopPreparationError> {
        self.destinations.prepare_with(after)
    }
    /// Borrows visible text until the next mutation.
    pub fn step<'a>(&'a mut self, text: &'a str) -> Result<StopOutput<'a>, StopStorageError> {
        self.destinations.step(&self.source, text)
    }
    /// Finishes the same frontier without allocating.
    pub fn finish(&mut self) -> Result<&str, StopStorageError> {
        self.destinations.finish(&self.source)
    }
}

#[cfg(test)]
mod tests;

/// Source-bound copy of the current literal-stop destination and progress.
pub struct StopDestinationCopy<'a>(&'a StopDestinations);
#[derive(Debug, thiserror::Error)]
#[error("stop state copy allocation failed")]
pub struct StopDestinationCopyError {
    #[source]
    cause: TryReserveError,
    _partial: StopDestinations,
}
impl StopDestinations {
    /// Excludes transitions while a separately admitted copy is constructed.
    pub fn prepare_copy(&self) -> StopDestinationCopy<'_> {
        StopDestinationCopy(self)
    }
}
impl StopDestinationCopy<'_> {
    /// Actual requested backing and concrete copy/failure transports.
    pub fn required_bytes(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<StopDestinations>(),
            size_of::<StopDestinationCopyError>(),
            size_of::<Result<StopDestinations, StopDestinationCopyError>>(),
            size_of::<Result<(), TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(
            self.0.destination.capacity().max(self.0.requested),
            usize::checked_add,
        )
    }
    /// Preserves the pending range, matched stop and terminal/readiness state.
    pub fn copy(self) -> Result<StopDestinations, StopDestinationCopyError> {
        let s = self.0;
        let mut out = StopDestinations {
            maximum_input: s.maximum_input,
            requested: s.requested,
            destination: Vec::new(),
            progress: s.progress.clone(),
            phase: s.phase,
        };
        if let Err(cause) = out.destination.try_reserve_exact(s.destination.capacity()) {
            return Err(StopDestinationCopyError {
                cause,
                _partial: out,
            });
        }
        out.destination.extend_from_slice(&s.destination);
        Ok(out)
    }
}
