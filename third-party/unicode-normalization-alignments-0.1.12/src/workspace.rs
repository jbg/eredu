//! One source-bound, once-reserved NFC destination with shared Unicode workers.
//!
//! ```compile_fail
//! use unicode_normalization_alignments::workspace::Plan;
//! let mut source = String::from("e\u{301}");
//! let plan = Plan::new(&source).unwrap();
//! source.clear();
//! let _ = plan.prepare();
//! ```
//! ```compile_fail
//! use unicode_normalization_alignments::workspace::Plan;
//! let plan = Plan::new("text").unwrap();
//! let _ = plan.prepare();
//! let _ = plan.prepare();
//! ```
//! ```compile_fail
//! use unicode_normalization_alignments::workspace::Plan;
//! let mut workspace = Plan::new("e\u{301}").unwrap().prepare().unwrap();
//! let output = workspace.normalize(0..3).unwrap();
//! let retired = workspace.retire();
//! println!("{} {:?}", output, retired);
//! ```
use decompose::{self, DecompositionBuffer, DecompositionType};
use recompose::{self, RecompositionBuffer, RecompositionState};
use std::{
    alloc::Layout, collections::TryReserveError, fmt, iter::Fuse, mem::size_of, ops::Range,
    str::Chars,
};

/// One actual destination, selected only by the development failure seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Canonical class/character/change/ordinal records.
    Decomposition,
    /// Pending recomposition characters and changes.
    Recomposition,
    /// Normalized UTF-8 output.
    Text,
}
/// Geometry was rejected before any allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overflow;
impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("NFC destination layout overflow")
    }
}
impl std::error::Error for Overflow {}
/// The requested range is not a UTF-8 subrange of the originally borrowed source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidRange;
impl fmt::Display for InvalidRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("NFC source range is invalid")
    }
}
impl std::error::Error for InvalidRange {}

#[derive(Clone, Copy, Debug)]
struct Record {
    class: u8,
    scalar: char,
    change: isize,
    ordinal: usize,
}
#[derive(Debug)]
struct Decomposition {
    entries: Vec<Record>,
    limit: usize,
}
impl DecompositionBuffer for Decomposition {
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn get(&self, i: usize) -> (u8, char, isize) {
        let r = self.entries[i];
        (r.class, r.scalar, r.change)
    }
    fn set(&mut self, i: usize, (class, scalar, change): (u8, char, isize)) {
        self.entries[i] = Record {
            class,
            scalar,
            change,
            ordinal: 0,
        };
    }
    fn push(&mut self, (class, scalar, change): (u8, char, isize)) {
        assert!(
            self.entries.len() < self.limit && self.entries.len() < self.entries.capacity(),
            "NFC decomposition bound"
        );
        self.entries.push(Record {
            class,
            scalar,
            change,
            ordinal: 0,
        });
    }
    fn truncate(&mut self, len: usize) {
        self.entries.truncate(len);
    }
    fn sort_pending(&mut self, start: usize) {
        let pending = &mut self.entries[start..];
        for (ordinal, item) in pending.iter_mut().enumerate() {
            item.ordinal = ordinal;
        }
        // Unique total key is exactly the ordinary stable class order. Renumber
        // after every compaction; never retain old ordinals as source authority.
        pending.sort_unstable_by_key(|r| (r.class, r.ordinal));
    }
}
#[derive(Debug)]
struct Recomposition {
    entries: Vec<(char, isize)>,
    limit: usize,
}
impl RecompositionBuffer for Recomposition {
    fn get(&self, i: usize) -> Option<(char, isize)> {
        self.entries.get(i).cloned()
    }
    fn push(&mut self, item: (char, isize)) {
        assert!(
            self.entries.len() < self.limit && self.entries.len() < self.entries.capacity(),
            "NFC recomposition bound"
        );
        self.entries.push(item);
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
}
struct Decomposing<'a, 'b> {
    iter: Fuse<Chars<'a>>,
    ready: Range<usize>,
    storage: &'b mut Decomposition,
}
impl Iterator for Decomposing<'_, '_> {
    type Item = (char, isize);
    fn next(&mut self) -> Option<Self::Item> {
        decompose::next(
            &DecompositionType::Canonical,
            &mut self.iter,
            self.storage,
            &mut self.ready,
        )
    }
}
struct Recomposing<'a, 'b> {
    iter: Decomposing<'a, 'b>,
    state: RecompositionState,
    storage: &'b mut Recomposition,
    composee: Option<(char, isize)>,
    last_ccc: Option<u8>,
}
impl Iterator for Recomposing<'_, '_> {
    type Item = (char, isize);
    fn next(&mut self) -> Option<Self::Item> {
        recompose::next(
            &mut self.iter,
            &mut self.state,
            self.storage,
            &mut self.composee,
            &mut self.last_ccc,
        )
    }
}
/// Exact checked requested layouts and named controls from the actual input.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    scalars: usize,
    bytes: usize,
    heap: usize,
    controls: usize,
    total: usize,
}
impl Requirements {
    /// Total canonical decomposition scalar population D.
    pub fn scalar_capacity(&self) -> usize {
        self.scalars
    }
    /// Total canonical decomposition UTF-8 bytes N, bounding recomposed output.
    pub fn text_capacity(&self) -> usize {
        self.bytes
    }
    /// Three actual destination layouts, without allocator metadata.
    pub fn buffer_bytes(&self) -> usize {
        self.heap
    }
    /// Named plan, iterator, return/error and fixed sorter-array controls.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Sum of actual destination layouts and named controls; no account grant.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Move-only plan tied to one immutable source; counts cannot be supplied by callers.
#[derive(Debug)]
pub struct Plan<'a> {
    input: &'a str,
    requirements: Requirements,
    #[cfg(any(test, feature = "workspace-test-support"))]
    failure: Option<Buffer>,
}
impl<'a> Plan<'a> {
    /// Scans static canonical decomposition tables without allocating.
    pub fn new(input: &'a str) -> Result<Self, Overflow> {
        let mut counts = Some((0usize, 0usize));
        for c in input.chars() {
            ::char::decompose_canonical(c, |d| {
                counts = counts
                    .and_then(|(n, b)| Some((n.checked_add(1)?, b.checked_add(d.len_utf8())?)));
            });
            if counts.is_none() {
                return Err(Overflow);
            }
        }
        let (scalars, bytes) = counts.ok_or(Overflow)?;
        Ok(Self {
            input,
            requirements: requirements(scalars, bytes)?,
            #[cfg(any(test, feature = "workspace-test-support"))]
            failure: None,
        })
    }
    /// Actual requirements of this source, not independent capacity authority.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Overflow one selected real reserve. This never changes a successful capacity.
    #[cfg(any(test, feature = "workspace-test-support"))]
    #[doc(hidden)]
    pub fn fail_reservation(mut self, buffer: Buffer) -> Self {
        self.failure = Some(buffer);
        self
    }
    fn request(&self, buffer: Buffer, n: usize) -> usize {
        #[cfg(any(test, feature = "workspace-test-support"))]
        if self.failure == Some(buffer) {
            return usize::MAX;
        }
        let _ = buffer;
        n
    }
    /// Attempts each actual destination once. Error owns every actual partial buffer.
    pub fn prepare(self) -> Result<Workspace<'a>, PrepareFailure> {
        let mut storage = Retired {
            decomposition: Decomposition {
                entries: Vec::new(),
                limit: self.requirements.scalars,
            },
            recomposition: Recomposition {
                entries: Vec::new(),
                limit: self.requirements.scalars,
            },
            text: String::new(),
        };
        macro_rules! reserve {
            ($buffer:expr,$target:expr,$n:expr) => {
                if let Err(cause) = $target.try_reserve_exact(self.request($buffer, $n)) {
                    return Err(PrepareFailure {
                        buffer: $buffer,
                        cause,
                        partial: storage,
                    });
                }
            };
        }
        reserve!(
            Buffer::Decomposition,
            storage.decomposition.entries,
            self.requirements.scalars
        );
        reserve!(
            Buffer::Recomposition,
            storage.recomposition.entries,
            self.requirements.scalars
        );
        reserve!(Buffer::Text, storage.text, self.requirements.bytes);
        Ok(Workspace {
            input: self.input,
            requirements: self.requirements,
            storage,
        })
    }
}
fn requirements(scalars: usize, bytes: usize) -> Result<Requirements, Overflow> {
    let heap = Layout::array::<Record>(scalars)
        .map_err(|_| Overflow)?
        .size()
        .checked_add(
            Layout::array::<(char, isize)>(scalars)
                .map_err(|_| Overflow)?
                .size(),
        )
        .and_then(|n| n.checked_add(Layout::array::<u8>(bytes).ok()?.size()))
        .ok_or(Overflow)?;
    let controls = [
        size_of::<Plan<'_>>(),
        size_of::<Result<Plan<'_>, Overflow>>(),
        size_of::<Requirements>(),
        size_of::<Option<(usize, usize)>>(),
        size_of::<Chars<'_>>(),
        size_of::<&mut Option<(usize, usize)>>(),
        size_of::<(&mut Decomposition, &mut Range<usize>, &mut bool)>(),
        size_of::<Workspace<'_>>(),
        size_of::<Result<Workspace<'_>, PrepareFailure>>(),
        size_of::<Retired>(),
        size_of::<PrepareFailure>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Range<usize>>(),
        size_of::<Result<&str, InvalidRange>>(),
        size_of::<Decomposing<'_, '_>>(),
        size_of::<Recomposing<'_, '_>>(),
        size_of::<Option<(char, isize)>>(),
        size_of::<[u8; 4]>(),
        size_of::<[Record; 48]>(),
        size_of::<(u8, usize)>(),
        size_of::<&Record>(),
        size_of::<Record>(),
    ]
    .iter()
    .try_fold(0usize, |sum, &n| sum.checked_add(n))
    .ok_or(Overflow)?;
    Ok(Requirements {
        scalars,
        bytes,
        heap,
        controls,
        total: heap.checked_add(controls).ok_or(Overflow)?,
    })
}
/// Reusable storage for checked subranges of the original source only.
#[derive(Debug)]
pub struct Workspace<'a> {
    input: &'a str,
    requirements: Requirements,
    storage: Retired,
}
impl Workspace<'_> {
    /// Normalize a UTF-8 subrange; invalid ranges reject before any destination mutation.
    pub fn normalize(&mut self, range: Range<usize>) -> Result<&str, InvalidRange> {
        let input = self.input.get(range).ok_or(InvalidRange)?;
        self.storage.decomposition.entries.clear();
        self.storage.recomposition.entries.clear();
        self.storage.text.clear();
        let chars = Recomposing {
            iter: Decomposing {
                iter: input.chars().fuse(),
                ready: 0..0,
                storage: &mut self.storage.decomposition,
            },
            state: RecompositionState::Composing,
            storage: &mut self.storage.recomposition,
            composee: None,
            last_ccc: None,
        };
        for (c, _) in chars {
            let next = self
                .storage
                .text
                .len()
                .checked_add(c.len_utf8())
                .expect("checked NFC bytes");
            assert!(
                next <= self.requirements.bytes && next <= self.storage.text.capacity(),
                "NFC text bound"
            );
            self.storage.text.push(c);
        }
        Ok(&self.storage.text)
    }
    /// Actual retained capacities in decomposition/recomposition/text order.
    pub fn capacities(&self) -> [usize; 3] {
        self.storage.capacities()
    }
    /// Consume source/cursor association, retaining the same destinations without refill APIs.
    pub fn retire(self) -> Retired {
        self.storage
    }
}
/// Pointer-free retired destinations; no source, refill, clone or raw storage escape.
#[derive(Debug)]
pub struct Retired {
    decomposition: Decomposition,
    recomposition: Recomposition,
    text: String,
}
impl Retired {
    /// Actual capacities, including partially prepared prefixes.
    pub fn capacities(&self) -> [usize; 3] {
        [
            self.decomposition.entries.capacity(),
            self.recomposition.entries.capacity(),
            self.text.capacity(),
        ]
    }
}
/// The real reserve error and all earlier partial destinations, without a source borrow.
#[derive(Debug)]
pub struct PrepareFailure {
    buffer: Buffer,
    cause: TryReserveError,
    partial: Retired,
}
impl PrepareFailure {
    /// The destination whose actual reserve failed.
    pub fn buffer(&self) -> Buffer {
        self.buffer
    }
    /// The original standard-library cause.
    pub fn reserve_error(&self) -> &TryReserveError {
        &self.cause
    }
    /// Actual retained prefix capacities.
    pub fn capacities(&self) -> [usize; 3] {
        self.partial.capacities()
    }
}
impl fmt::Display for PrepareFailure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PrepareFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[cfg(test)]
mod tests;
