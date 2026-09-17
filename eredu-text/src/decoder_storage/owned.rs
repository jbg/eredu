//! Move-only source and storage for original provider composition. This utility
//! grants no account/custody and performs no HF work after cold compilation.
use super::*;
use std::collections::TryReserveError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Dormant,
    Attempted,
    Ready,
}

/// A fixed owned stream cannot execute before its local destinations are ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OwnedDecodeStorageError {
    /// Local destination preparation has not succeeded.
    #[error("decoder destinations are not prepared")]
    Unprepared,
    /// The unchanged fixed kernel rejected this transition.
    #[error(transparent)]
    Storage(#[from] DecodeStorageError),
}

/// Failed local destination preparation; partial allocations stay in their owner.
#[derive(Debug, thiserror::Error)]
pub enum OwnedDecodePreparationError {
    /// A failed attempt cannot allocate another set of destinations.
    #[error("decoder destination preparation was already attempted")]
    AlreadyAttempted,
    /// A requested destination could not be allocated.
    #[error("decoder destination allocation failed")]
    Allocation(#[from] TryReserveError),
}

/// Pointer-free progress and four fixed destinations, without a source owner.
///
/// Construction derives N, skip policy and extents from an actual source. Each
/// transition explicitly borrows its program and checks those same extents.
/// This low-level mechanism establishes no immutable-source or account authority:
/// concrete owning wrappers must supply their retained source on every call.
/// Preparation/step/finish are the same implementation used by OwnedDecodeStorage.
#[derive(Debug)]
pub struct DecodeDestinations {
    calls: usize,
    skip_special: bool,
    raw_capacity: usize,
    text_capacity: usize,
    destination_bytes: usize,
    ids: Vec<u32>,
    raw: Vec<u8>,
    candidate: Vec<u8>,
    prefix: Vec<u8>,
    progress: DecodeProgress,
    phase: Phase,
}
impl DecodeDestinations {
    /// Derives fixed extents without taking, cloning or compiling the source.
    /// This constructor allocates no destination and establishes no admission.
    pub fn new(
        source: &PreparedDecodeSource,
        calls: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        let layout = DecodeStreamLayout::for_source(source, calls, skip_special)?;
        let (raw_capacity, text_capacity, destination_bytes) =
            (layout.raw, layout.text, layout.buffer_bytes);
        Ok(Self {
            calls,
            skip_special,
            raw_capacity,
            text_capacity,
            destination_bytes,
            ids: Vec::new(),
            raw: Vec::new(),
            candidate: Vec::new(),
            prefix: Vec::new(),
            progress: DecodeProgress::default(),
            phase: Phase::Dormant,
        })
    }

    /// Original successful-call ceiling, independent of compacted history.
    pub fn token_capacity(&self) -> usize {
        self.calls
    }
    /// Immutable skip policy selected before original admission.
    pub fn skip_special_tokens(&self) -> bool {
        self.skip_special
    }
    /// Four requested destination allocation extents. No source/control sum.
    pub fn destination_bytes(&self) -> usize {
        self.destination_bytes
    }
    /// Actual additional named constructor/preparation/transition controls.
    /// The containing provider measures this value's inline representation;
    /// no source payload is included. Allocator overhead, cold HF work, error
    /// erasure and arbitrary stacks are separate obligations.
    pub fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(), // constructor value / return overlap
            size_of::<Result<Self, DecodeStorageError>>(),
            size_of::<DecodeStreamLayout<'static>>(),
            size_of::<Result<DecodeStreamLayout<'static>, DecodeStorageError>>(),
            size_of::<Result<DecodeStreamLayout<'static>, OwnedDecodeStorageError>>(),
            size_of::<Result<(), OwnedDecodePreparationError>>(),
            size_of::<OwnedDecodePreparationError>(),
            size_of::<Result<Option<&'static str>, OwnedDecodeStorageError>>(),
            size_of::<Result<(), OwnedDecodeStorageError>>(),
            size_of::<OwnedDecodeStorageError>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// One local attempt before the existing caller readiness agreement.
    /// Every partial Vec remains here on error/unwind. No boxed-slice conversion
    /// or shrink/reallocation is performed after the initial exact request.
    pub fn prepare_destinations(&mut self) -> Result<(), OwnedDecodePreparationError> {
        self.prepare_with(|| Ok(()))
    }
    // Production instantiates only a zero-sized no-op; the child test injects a
    // real capacity-overflow TryReserveError after each completed destination.
    fn prepare_with(
        &mut self,
        mut after_destination: impl FnMut() -> Result<(), TryReserveError>,
    ) -> Result<(), OwnedDecodePreparationError> {
        match self.phase {
            Phase::Ready => return Ok(()),
            Phase::Attempted => return Err(OwnedDecodePreparationError::AlreadyAttempted),
            Phase::Dormant => self.phase = Phase::Attempted,
        }
        self.ids.try_reserve_exact(self.calls)?;
        self.ids.resize(self.calls, 0);
        after_destination()?;
        self.raw.try_reserve_exact(self.raw_capacity)?;
        self.raw.resize(self.raw_capacity, 0);
        after_destination()?;
        self.candidate.try_reserve_exact(self.text_capacity)?;
        self.candidate.resize(self.text_capacity, 0);
        after_destination()?;
        self.prefix.try_reserve_exact(self.text_capacity)?;
        self.prefix.resize(self.text_capacity, 0);
        after_destination()?;
        self.phase = Phase::Ready;
        Ok(())
    }

    fn layout<'s>(
        &self,
        source: &'s PreparedDecodeSource,
    ) -> Result<DecodeStreamLayout<'s>, OwnedDecodeStorageError> {
        let layout = DecodeStreamLayout::for_source(source, self.calls, self.skip_special)?;
        for (buffer, expected, actual) in [
            (DecodeBuffer::Raw, layout.raw, self.raw_capacity),
            (DecodeBuffer::Candidate, layout.text, self.text_capacity),
        ] {
            if actual != expected {
                return Err(DecodeStorageError::Extent {
                    buffer,
                    expected,
                    actual,
                }
                .into());
            }
        }
        Ok(layout)
    }

    /// Runs the same fixed transition and lends its suffix without allocating.
    /// A zero-call owner retains the kernel's CallLimit without preparation.
    pub fn step(
        &mut self,
        source: &PreparedDecodeSource,
        id: u32,
    ) -> Result<Option<&str>, OwnedDecodeStorageError> {
        if self.phase != Phase::Ready && self.calls != 0 {
            return Err(OwnedDecodeStorageError::Unprepared);
        }
        let layout = self.layout(source)?;
        self.progress
            .step(
                &layout,
                &mut self.ids,
                &mut self.raw,
                &mut self.candidate,
                &mut self.prefix,
                id,
            )
            .map_err(Into::into)
    }
    /// Same non-flushing residual check. Cancellation must skip this operation.
    pub fn finish(&mut self, source: &PreparedDecodeSource) -> Result<(), OwnedDecodeStorageError> {
        if self.phase != Phase::Ready && self.calls != 0 {
            return Err(OwnedDecodeStorageError::Unprepared);
        }
        let layout = self.layout(source)?;
        self.progress
            .finish(&layout, &self.ids, &mut self.raw, &mut self.candidate)
            .map_err(Into::into)
    }
    /// Exact retained kernel history, including failed appended transitions.
    pub fn retained_ids(&self) -> &[u32] {
        &self.ids[..self.progress.ids_len]
    }
    /// Exact retained prefix for the next transition.
    pub fn prefix(&self) -> &str {
        std::str::from_utf8(&self.prefix[..self.progress.prefix_len]).expect("decoded prefix")
    }
    /// Complete last candidate, including failed InvalidPrefix output.
    pub fn candidate(&self) -> &str {
        std::str::from_utf8(&self.candidate[..self.progress.candidate_len])
            .expect("decoded candidate")
    }
    /// Exact HF prefix index within retained history.
    pub fn prefix_index(&self) -> usize {
        self.progress.prefix_index
    }
    /// Number of successful transitions, not current history length.
    pub fn successful_calls(&self) -> usize {
        self.progress.successful_calls
    }
}

/// One actual compiled source plus four dormant fixed destinations.
///
/// No Clone, raw allocation/custody exit, source replacement or capacity setter
/// exists. Source compilation and this cold constructor have no funding claim;
/// an original provider must price and retain the complete value before calling
/// `prepare_destinations`. All transition logic is shared with DecodeStreamState.
/// Returned text borrows this owner and excludes mutation or retirement.
///
/// ```compile_fail
/// fn overlap(stream: &mut eredu_text::decoder_storage::OwnedDecodeStorage) {
///     let text = stream.step(1).unwrap();
///     stream.step(2).unwrap();
///     println!("{:?}", text);
/// }
/// ```
#[derive(Debug)]
pub struct OwnedDecodeStorage {
    source: Box<PreparedDecodeSource>,
    destinations: DecodeDestinations,
}
impl OwnedDecodeStorage {
    /// Takes the exact program and derives all destination facts once.
    /// Cold rejection drops that transferred source; no admission exists here.
    pub fn new(
        source: PreparedDecodeSource,
        calls: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        let destinations = DecodeDestinations::new(&source, calls, skip_special)?;
        Ok(Self {
            source: Box::new(source),
            destinations,
        })
    }
    /// Borrows the exact immutable source without transferring its owner.
    pub fn source(&self) -> &PreparedDecodeSource {
        &self.source
    }
    /// Original successful-call ceiling.
    pub fn token_capacity(&self) -> usize {
        self.destinations.token_capacity()
    }
    /// Immutable original skip policy.
    pub fn skip_special_tokens(&self) -> bool {
        self.destinations.skip_special_tokens()
    }
    /// Four destination payload extents, excluding the source.
    pub fn destination_bytes(&self) -> usize {
        self.destinations.destination_bytes()
    }
    /// Named concrete owner/constructor controls plus shared destination controls.
    pub fn control_bytes() -> Option<usize> {
        DecodeDestinations::control_bytes()?
            .checked_add(size_of::<Self>())?
            .checked_add(size_of::<Result<Self, DecodeStorageError>>())?
            .checked_add(size_of::<PreparedDecodeSource>())
    }
    /// The shared one-attempt destination preparation, preserving partial buffers.
    pub fn prepare_destinations(&mut self) -> Result<(), OwnedDecodePreparationError> {
        self.destinations.prepare_destinations()
    }
    /// Runs the shared kernel against this exact retained source.
    pub fn step(&mut self, id: u32) -> Result<Option<&str>, OwnedDecodeStorageError> {
        self.destinations.step(&self.source, id)
    }
    /// Runs the same non-flushing finish against this retained source.
    pub fn finish(&mut self) -> Result<(), OwnedDecodeStorageError> {
        self.destinations.finish(&self.source)
    }
    /// Exact retained history.
    pub fn retained_ids(&self) -> &[u32] {
        self.destinations.retained_ids()
    }
    /// Exact retained UTF-8 prefix.
    pub fn prefix(&self) -> &str {
        self.destinations.prefix()
    }
    /// Last complete candidate, including a failed appended transition.
    pub fn candidate(&self) -> &str {
        self.destinations.candidate()
    }
    /// Exact HF prefix index.
    pub fn prefix_index(&self) -> usize {
        self.destinations.prefix_index()
    }
    /// Count of successful transitions.
    pub fn successful_calls(&self) -> usize {
        self.destinations.successful_calls()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_partial_destination_survives_the_only_failed_attempt() {
        let tokenizer = crate::tokenizer::Tokenizer::from_bytes(br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},"model":{"type":"WordLevel","vocab":{"hello":0,"[UNK]":1},"unk_token":"[UNK]"}}"#).unwrap();
        for stop in 1..=4 {
            let source = PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap();
            let mut owner = OwnedDecodeStorage::new(source, 3, false).unwrap();
            let source = std::ptr::from_ref(owner.source());
            let mut completed = 0;
            let error = owner
                .destinations
                .prepare_with(|| {
                    completed += 1;
                    if completed == stop {
                        // Deterministic capacity overflow, NOT allocator OOM.
                        Err(Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err())
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert!(matches!(error, OwnedDecodePreparationError::Allocation(_)));
            assert_eq!(completed, stop);
            let lengths = [
                owner.destinations.ids.len(),
                owner.destinations.raw.len(),
                owner.destinations.candidate.len(),
                owner.destinations.prefix.len(),
            ];
            let capacities = [
                owner.destinations.ids.capacity(),
                owner.destinations.raw.capacity(),
                owner.destinations.candidate.capacity(),
                owner.destinations.prefix.capacity(),
            ];
            let pointers = [
                owner.destinations.ids.as_ptr() as usize,
                owner.destinations.raw.as_ptr() as usize,
                owner.destinations.candidate.as_ptr() as usize,
                owner.destinations.prefix.as_ptr() as usize,
            ];
            for i in 0..4 {
                assert_eq!(lengths[i] != 0, i < stop);
            }
            assert_eq!(owner.destinations.phase, Phase::Attempted);
            assert!(matches!(
                owner.prepare_destinations(),
                Err(OwnedDecodePreparationError::AlreadyAttempted)
            ));
            assert_eq!(owner.step(0), Err(OwnedDecodeStorageError::Unprepared));
            assert_eq!(owner.finish(), Err(OwnedDecodeStorageError::Unprepared));
            assert_eq!(
                capacities,
                [
                    owner.destinations.ids.capacity(),
                    owner.destinations.raw.capacity(),
                    owner.destinations.candidate.capacity(),
                    owner.destinations.prefix.capacity()
                ]
            );
            assert_eq!(
                pointers,
                [
                    owner.destinations.ids.as_ptr() as usize,
                    owner.destinations.raw.as_ptr() as usize,
                    owner.destinations.candidate.as_ptr() as usize,
                    owner.destinations.prefix.as_ptr() as usize
                ]
            );
            assert_eq!(source, std::ptr::from_ref(owner.source()));
            assert!(owner.retained_ids().is_empty());
        }
    }
}

/// Source-bound copy of all fixed decoder buffers and progress. This plan owns
/// no allowance; the runtime provider admits its destination before consuming it.
pub struct DecodeDestinationCopy<'a>(&'a DecodeDestinations);
#[derive(Debug, thiserror::Error)]
#[error("decoder state copy allocation failed")]
pub struct DecodeDestinationCopyError {
    #[source]
    cause: TryReserveError,
    _partial: DecodeDestinations,
}
impl DecodeDestinations {
    /// Borrows the actual mutable state, excluding further transitions during copy.
    pub fn prepare_copy(&self) -> DecodeDestinationCopy<'_> {
        DecodeDestinationCopy(self)
    }
}
impl DecodeDestinationCopy<'_> {
    /// Requested backing plus concrete copy and partial-error controls.
    pub fn required_bytes(&self) -> Option<usize> {
        let s = self.0;
        let mut bytes = 0usize;
        for (count, width) in [
            (s.ids.capacity(), size_of::<u32>()),
            (s.raw.capacity(), 1),
            (s.candidate.capacity(), 1),
            (s.prefix.capacity(), 1),
        ] {
            bytes = bytes.checked_add(count.checked_mul(width)?)?;
        }
        [
            size_of::<Self>(),
            size_of::<DecodeDestinations>(),
            size_of::<DecodeDestinationCopyError>(),
            size_of::<Result<DecodeDestinations, DecodeDestinationCopyError>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<[(&mut Vec<u8>, &Vec<u8>); 3]>(),
            size_of::<std::array::IntoIter<(&mut Vec<u8>, &Vec<u8>), 3>>(),
        ]
        .into_iter()
        .try_fold(bytes.max(s.destination_bytes), usize::checked_add)
    }
    /// Copies even attempted/failed progress exactly; it does not reset readiness,
    /// token/call ceilings, incomplete UTF-8, or a retained invalid-prefix frontier.
    pub fn copy(self) -> Result<DecodeDestinations, DecodeDestinationCopyError> {
        let s = self.0;
        let mut out = DecodeDestinations {
            calls: s.calls,
            skip_special: s.skip_special,
            raw_capacity: s.raw_capacity,
            text_capacity: s.text_capacity,
            destination_bytes: s.destination_bytes,
            ids: Vec::new(),
            raw: Vec::new(),
            candidate: Vec::new(),
            prefix: Vec::new(),
            progress: s.progress.clone(),
            phase: s.phase,
        };
        let result = (|| {
            out.ids.try_reserve_exact(s.ids.capacity())?;
            out.ids.extend_from_slice(&s.ids);
            for (to, from) in [
                (&mut out.raw, &s.raw),
                (&mut out.candidate, &s.candidate),
                (&mut out.prefix, &s.prefix),
            ] {
                to.try_reserve_exact(from.capacity())?;
                to.extend_from_slice(from);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(DecodeDestinationCopyError {
                cause,
                _partial: out,
            }),
        }
    }
}
