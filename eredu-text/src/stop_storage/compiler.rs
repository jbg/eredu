//! Concrete borrowed stop planning and one-attempt packing, without policy.
use super::{Entry, PreparedStopSource, StopSourceError};
use std::{alloc::Layout, mem::size_of};

#[derive(Debug)]
enum Input<'a> {
    Strings(&'a [String]),
    Refs(&'a [&'a str]),
}
impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Strings(s) => s.len(),
            Self::Refs(s) => s.len(),
        }
    }
    fn get(&self, i: usize) -> &str {
        match self {
            Self::Strings(s) => &s[i],
            Self::Refs(s) => s[i],
        }
    }
    fn is_first_nonempty(&self, i: usize) -> bool {
        let value = self.get(i);
        !value.is_empty() && (0..i).all(|j| self.get(j) != value)
    }
}
/// Exact checked packing requirements, obtained from actual immutable strings.
/// No allocator metadata, general stack, input ownership or budget is implied.
#[derive(Debug, Clone, Copy)]
pub struct StopCompileRequirements {
    entries: usize,
    bytes: usize,
    longest: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl StopCompileRequirements {
    /// Number of nonempty first occurrences.
    pub fn entries(&self) -> usize {
        self.entries
    }
    /// Total UTF-8 bytes of those entries.
    pub fn packed_bytes(&self) -> usize {
        self.bytes
    }
    /// Longest retained literal in bytes.
    pub fn longest_stop_bytes(&self) -> usize {
        self.longest
    }
    /// Simultaneous Entries and Bytes allocation requests.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named fixed construction/error/return representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked buffers plus fixed controls, not a grant.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// One borrowed immutable plan, consumed by one packing attempt.
/// Duplicate detection compares earlier actual strings without allocation. Its
/// O(n²) work is explicit; this is a memory fact, not a latency guarantee.
///
/// ```compile_fail
/// fn replay(plan: eredu_text::stop_storage::StopCompilePlan<'_>) {
///     let first = plan.compile();
///     let second = plan.compile();
/// }
/// ```
#[derive(Debug)]
pub struct StopCompilePlan<'a> {
    input: Input<'a>,
    requirements: StopCompileRequirements,
    #[cfg(any(test, feature = "stop-compiler-test-support"))]
    fail_at: Option<usize>,
}
impl<'a> StopCompilePlan<'a> {
    /// Plans directly from the facade's actual request slice, without cloning it.
    pub fn prepare(stops: &'a [String]) -> Result<Self, StopSourceError> {
        Self::new(Input::Strings(stops))
    }
    /// Plans from an existing immutable reference slice, with no collection.
    pub fn prepare_refs(stops: &'a [&'a str]) -> Result<Self, StopSourceError> {
        Self::new(Input::Refs(stops))
    }
    fn new(input: Input<'a>) -> Result<Self, StopSourceError> {
        let (mut count, mut bytes, mut longest) = (0usize, 0usize, 0usize);
        for i in 0..input.len() {
            if input.is_first_nonempty(i) {
                count = count.checked_add(1).ok_or(StopSourceError::Overflow)?;
                bytes = bytes
                    .checked_add(input.get(i).len())
                    .ok_or(StopSourceError::Overflow)?;
                longest = longest.max(input.get(i).len());
            }
        }
        Ok(Self {
            input,
            requirements: requirements(count, bytes, longest)?,
            #[cfg(any(test, feature = "stop-compiler-test-support"))]
            fail_at: None,
        })
    }
    /// Borrows this source-derived checked fact without compiling or granting.
    pub fn requirements(&self) -> &StopCompileRequirements {
        &self.requirements
    }
    /// Performs exactly the two checked reserve requests and fills in source order.
    /// No grow, boxing, shrinking, string clone, source callback or retry follows.
    pub fn compile(self) -> Result<PreparedStopSource, StopCompileFailure> {
        let mut partial = Partial::default();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(StopCompileFailure { cause, partial });
        }
        Ok(PreparedStopSource {
            entries: partial.entries,
            bytes: partial.bytes,
            longest: self.requirements.longest,
        })
    }
    fn fill(&self, partial: &mut Partial) -> Result<(), StopSourceError> {
        partial
            .entries
            .try_reserve_exact(self.requested(0, self.requirements.entries))?;
        partial
            .bytes
            .try_reserve_exact(self.requested(1, self.requirements.bytes))?;
        for i in 0..self.input.len() {
            if self.input.is_first_nonempty(i) {
                let start = partial.bytes.len();
                partial
                    .bytes
                    .extend_from_slice(self.input.get(i).as_bytes());
                partial.entries.push(Entry {
                    start,
                    end: partial.bytes.len(),
                });
            }
        }
        Ok(())
    }
    fn requested(&self, _stage: usize, capacity: usize) -> usize {
        #[cfg(any(test, feature = "stop-compiler-test-support"))]
        if self.fail_at == Some(_stage) {
            // Real target capacity overflow, never a substitute allocation.
            return usize::MAX;
        }
        capacity
    }
    /// Development-only capacity-overflow injection at Entries(0) or Bytes(1).
    /// No external allocator, callback or caller byte fact is accepted.
    #[doc(hidden)]
    #[cfg(any(test, feature = "stop-compiler-test-support"))]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 2);
        self.fail_at = Some(stage);
        self
    }
}
fn requirements(
    entries: usize,
    bytes: usize,
    longest: usize,
) -> Result<StopCompileRequirements, StopSourceError> {
    let entries_layout = Layout::array::<Entry>(entries).map_err(|_| StopSourceError::Overflow)?;
    let bytes_layout = Layout::array::<u8>(bytes).map_err(|_| StopSourceError::Overflow)?;
    let buffers = entries_layout
        .size()
        .checked_add(bytes_layout.size())
        .ok_or(StopSourceError::Overflow)?;
    let controls = [
        size_of::<usize>(), // actual target selector argument
        size_of::<usize>(), // requested target capacity argument/return
        size_of::<Input<'_>>(),
        size_of::<StopCompilePlan<'_>>(),
        size_of::<Result<StopCompilePlan<'_>, StopSourceError>>(),
        size_of::<StopCompileRequirements>(),
        size_of::<Partial>(),
        size_of::<Entry>(),
        size_of::<Result<(), StopSourceError>>(),
        size_of::<PreparedStopSource>(),
        size_of::<StopCompileFailure>(),
        size_of::<Result<PreparedStopSource, StopCompileFailure>>(),
        size_of::<Result<PreparedStopSource, StopSourceError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(StopSourceError::Overflow)?;
    let total = buffers
        .checked_add(controls)
        .ok_or(StopSourceError::Overflow)?;
    Ok(StopCompileRequirements {
        entries,
        bytes,
        longest,
        buffers,
        controls,
        total,
    })
}
#[derive(Debug, Default)]
struct Partial {
    entries: Vec<Entry>,
    bytes: Vec<u8>,
}
/// Terminal failure preserving both real partial allocations and original cause.
/// There is no consuming partial accessor, source reference or retry operation.
#[derive(Debug)]
pub struct StopCompileFailure {
    cause: StopSourceError,
    partial: Partial,
}
impl StopCompileFailure {
    /// Actual underlying allocation or checked-extent error.
    pub fn cause(&self) -> &StopSourceError {
        &self.cause
    }
    /// Actual retained capacities, excluding this value and allocator metadata.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.partial.entries.capacity() * size_of::<Entry>() + self.partial.bytes.capacity()
    }
    pub(super) fn discard_partial(self) -> StopSourceError {
        let Self { cause, partial } = self;
        drop(partial);
        cause
    }
}
impl std::fmt::Display for StopCompileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for StopCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[cfg(test)]
mod tests;
