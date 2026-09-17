//! Borrowed planning and one-attempt construction of the immutable program.

use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

use tokenizers::tokenizer::DecodeVocabulary;

use super::{DecodeSourceError, Mode, PreparedDecodeSource, Record, byte_for_char, fallback_byte};
use crate::tokenizer::TokenizerSnapshot;

/// Checked requirements obtained from one borrowed, immutable HF snapshot.
///
/// Buffer bytes cover the simultaneous ID scratch, record destination and byte
/// destination. Control bytes describe the named Rust values listed by
/// [`DecodeCompilePlan::prepare`], not allocator metadata, general machine stack
/// use, HF residency or the construction of that snapshot. They grant no budget.
#[derive(Debug, Clone, Copy)]
pub struct DecodeCompileRequirements {
    ids: usize,
    pieces: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}

impl DecodeCompileRequirements {
    /// Model forward entries plus added reverse entries, including duplicate IDs.
    pub fn id_slots(&self) -> usize {
        self.ids
    }
    /// Upper bound on packed bytes, counting duplicate IDs before deduplication.
    pub fn piece_bytes(&self) -> usize {
        self.pieces
    }
    /// Checked simultaneous payload capacities of the three compiler buffers.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named fixed Rust representation and move/return overlap, excluding HF.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked sum of the buffer and named control requirements.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}

/// A non-allocating plan borrowing the exact snapshot used by compilation.
///
/// The plan cannot outlive or mutate its source, and is consumed by one attempt.
/// A failed attempt retains its allocated prefix in [`DecodeCompileFailure`].
/// Creating another plan is a separate construction; this type is not admission.
///
/// ```compile_fail
/// # use eredu_text::decoder_storage::DecodeCompilePlan;
/// # fn attempt(plan: DecodeCompilePlan<'_>) {
/// let first = plan.compile();
/// let replay = plan.compile();
/// # }
/// ```
#[derive(Debug)]
pub struct DecodeCompilePlan<'a> {
    vocabulary: DecodeVocabulary<'a>,
    mode: Mode,
    requirements: DecodeCompileRequirements,
    #[cfg(any(test, feature = "decode-compiler-test-support"))]
    fail_at: Option<usize>,
}

impl<'a> DecodeCompilePlan<'a> {
    /// Borrows decoder metadata and visits actual IDs without allocation.
    ///
    /// No vocabulary clone, serialization, token String, regex construction,
    /// cache mutation or dense maximum-ID traversal occurs here. All sums and
    /// array layouts are checked before compilation may reserve a buffer.
    ///
    /// Named controls include this plan and its result, the requirements return,
    /// partial buffers, one piece descriptor, reservation/fill results, source,
    /// owning failure, final compilation result and compatibility error result.
    /// These account for explicit representation/return overlaps; allocator and
    /// library call-stack overhead are outside this source-level byte fact.
    pub fn prepare(snapshot: &'a TokenizerSnapshot) -> Result<Self, DecodeSourceError> {
        Self::prepare_hf(snapshot)
    }

    // Exact fresh aggregate HF borrow; compatibility Snapshot entry delegates here.
    pub(crate) fn prepare_hf(
        tokenizer: &'a tokenizers::Tokenizer,
    ) -> Result<Self, DecodeSourceError> {
        let mode = Mode::from_decoder(tokenizer.get_decoder())
            .ok_or(DecodeSourceError::UnsupportedDecoder)?;
        let vocabulary = tokenizer.decode_vocabulary();
        let mut extents = Ok((0usize, 0usize));
        vocabulary.visit_ids(&mut |id| {
            if let Ok((count, bytes)) = &mut extents {
                let next = count.checked_add(1);
                let piece = vocabulary
                    .id_to_token(id)
                    .map(|token| Piece::new(mode, token));
                let added = piece.as_ref().map_or(0, |piece| piece.len());
                match (next, bytes.checked_add(added)) {
                    (Some(next), Some(next_bytes)) => {
                        *count = next;
                        *bytes = next_bytes;
                    }
                    _ => extents = Err(DecodeSourceError::Overflow),
                }
            }
        });
        let (ids, pieces) = extents?;
        let requirements = requirements(ids, pieces)?;
        Ok(Self {
            vocabulary,
            mode,
            requirements,
            #[cfg(any(test, feature = "decode-compiler-test-support"))]
            fail_at: None,
        })
    }

    /// Returns the actual source-derived requirements without retaining another owner.
    pub fn requirements(&self) -> DecodeCompileRequirements {
        self.requirements
    }

    /// Consumes the plan and attempts each destination allocation once.
    ///
    /// ID scratch is sorted in place and deduplicated; canonical spellings are
    /// then written directly into exact-reserved destinations. No shrink or
    /// boxed-slice conversion allocates during publication. Success moves those
    /// same immutable destinations to the existing decode kernel.
    pub fn compile(self) -> Result<PreparedDecodeSource, DecodeCompileFailure> {
        let mut partial = PartialCompilation::default();
        if let Err(cause) = self.fill(&mut partial) {
            return Err(DecodeCompileFailure { cause, partial });
        }
        let storage_bytes = match partial
            .records
            .capacity()
            .checked_mul(size_of::<Record>())
            .and_then(|bytes| bytes.checked_add(partial.bytes.capacity()))
            .and_then(|bytes| bytes.checked_add(size_of::<PreparedDecodeSource>()))
        {
            Some(bytes) => bytes,
            None => {
                return Err(DecodeCompileFailure {
                    cause: DecodeSourceError::Overflow,
                    partial,
                });
            }
        };
        let PartialCompilation {
            ids,
            records,
            bytes,
            max_piece,
        } = partial;
        // Scratch is retired before the standalone program is returned. The HF
        // view is borrowed and never becomes an owner in the resulting program.
        drop(ids);
        Ok(PreparedDecodeSource {
            mode: self.mode,
            records,
            bytes,
            max_piece,
            storage_bytes,
        })
    }

    fn fill(&self, partial: &mut PartialCompilation) -> Result<(), DecodeSourceError> {
        partial
            .ids
            .try_reserve_exact(self.requested_capacity(0, self.requirements.ids))?;
        self.vocabulary.visit_ids(&mut |id| partial.ids.push(id));
        partial.ids.sort_unstable();
        partial.ids.dedup();
        let mut bytes = 0usize;
        for id in &partial.ids {
            if let Some(token) = self.vocabulary.id_to_token(*id) {
                bytes = bytes
                    .checked_add(Piece::new(self.mode, token).len())
                    .ok_or(DecodeSourceError::Overflow)?;
            }
        }
        partial
            .records
            .try_reserve_exact(self.requested_capacity(1, partial.ids.len()))?;
        partial
            .bytes
            .try_reserve_exact(self.requested_capacity(2, bytes))?;
        for id in &partial.ids {
            let Some(token) = self.vocabulary.id_to_token(*id) else {
                continue;
            };
            let piece = Piece::new(self.mode, token);
            let start = partial.bytes.len();
            piece.write(&mut partial.bytes);
            let end = partial.bytes.len();
            partial.records.push(Record {
                id: *id,
                start,
                end,
                special: self.vocabulary.is_special_token(token),
                byte: piece.byte,
            });
            partial.max_piece = partial.max_piece.max(end - start);
        }
        Ok(())
    }

    fn requested_capacity(&self, stage: usize, requested: usize) -> usize {
        #[cfg(any(test, feature = "decode-compiler-test-support"))]
        if self.fail_at == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        requested
    }

    /// Development-only capacity-overflow injection on the actual target reserve.
    /// It never calls a caller allocator or changes the derived source extents.
    #[cfg(any(test, feature = "decode-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 3);
        self.fail_at = Some(stage);
        self
    }
}

pub(crate) fn requirements(
    ids: usize,
    pieces: usize,
) -> Result<DecodeCompileRequirements, DecodeSourceError> {
    let extent = |len: usize, size: usize| len.checked_mul(size).ok_or(DecodeSourceError::Overflow);
    Layout::array::<u32>(ids).map_err(|_| DecodeSourceError::Overflow)?;
    Layout::array::<Record>(ids).map_err(|_| DecodeSourceError::Overflow)?;
    Layout::array::<u8>(pieces).map_err(|_| DecodeSourceError::Overflow)?;
    let buffers = extent(ids, size_of::<u32>())?
        .checked_add(extent(ids, size_of::<Record>())?)
        .and_then(|bytes| bytes.checked_add(pieces))
        .ok_or(DecodeSourceError::Overflow)?;
    let control_parts = [
        Mode::control_bytes().ok_or(DecodeSourceError::Overflow)?,
        size_of::<DecodeCompilePlan<'_>>(),
        size_of::<Result<DecodeCompilePlan<'_>, DecodeSourceError>>(),
        size_of::<DecodeCompileRequirements>(),
        size_of::<PartialCompilation>(),
        size_of::<Piece<'_>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<(), DecodeSourceError>>(),
        size_of::<PreparedDecodeSource>(),
        size_of::<DecodeCompileFailure>(),
        size_of::<Result<PreparedDecodeSource, DecodeCompileFailure>>(),
        size_of::<Result<PreparedDecodeSource, DecodeSourceError>>(),
    ];
    let controls = control_parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&control_parts), usize::checked_add)
        .ok_or(DecodeSourceError::Overflow)?;
    let total = buffers
        .checked_add(controls)
        .ok_or(DecodeSourceError::Overflow)?;
    Ok(DecodeCompileRequirements {
        ids,
        pieces,
        buffers,
        controls,
        total,
    })
}

#[derive(Debug, Default)]
struct PartialCompilation {
    ids: Vec<u32>,
    records: Vec<Record>,
    bytes: Vec<u8>,
    max_piece: usize,
}

/// Terminal compilation failure retaining every already allocated buffer.
///
/// There is no consuming partial-buffer or retry accessor. The original HF
/// source was only borrowed; the error needs no tokenizer owner or source clone.
/// No account guard is manufactured here. An admitting caller must enclose this
/// failure under the same original construction authority as its plan.
#[derive(Debug)]
pub struct DecodeCompileFailure {
    cause: DecodeSourceError,
    partial: PartialCompilation,
}

impl DecodeCompileFailure {
    /// Borrows the original cause, including the actual `TryReserveError`.
    pub fn cause(&self) -> &DecodeSourceError {
        &self.cause
    }

    /// Actual live buffer capacities retained by this failed attempt.
    /// Excludes this inline error value and allocator bookkeeping.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.partial.ids.capacity() * size_of::<u32>()
            + self.partial.records.capacity() * size_of::<Record>()
            + self.partial.bytes.capacity()
    }

    pub(super) fn discard_partial(self) -> DecodeSourceError {
        let Self { cause, partial } = self;
        drop(partial);
        cause
    }
}

impl std::fmt::Display for DecodeCompileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for DecodeCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

struct Piece<'a> {
    token: &'a str,
    mapped: bool,
    replace: bool,
    byte: Option<u8>,
}
impl<'a> Piece<'a> {
    fn new(mode: Mode, token: &'a str) -> Self {
        // Replacing U+2581 cannot create or destroy an ASCII <0xXX> spelling.
        // Thus all three supported fallback orders share this token lowering;
        // their run/fusion differences stay in the existing execution kernel.
        let replace = matches!(mode, Mode::Fallback { .. });
        Self {
            token,
            replace,
            byte: if replace { fallback_byte(token) } else { None },
            mapped: mode == Mode::ByteLevel && token.chars().all(|c| byte_for_char(c).is_some()),
        }
    }
    fn len(&self) -> usize {
        if self.byte.is_some() {
            0
        } else if self.mapped {
            self.token.chars().count()
        } else if self.replace {
            self.token.len() - self.token.chars().filter(|c| *c == '▁').count() * 2
        } else {
            self.token.len()
        }
    }
    fn write(&self, bytes: &mut Vec<u8>) {
        if self.byte.is_some() {
            return;
        }
        if self.mapped {
            bytes.extend(
                self.token
                    .chars()
                    .map(|c| byte_for_char(c).expect("checked byte alphabet")),
            );
        } else if self.replace {
            for c in self.token.chars() {
                if c == '▁' {
                    bytes.push(b' ');
                } else {
                    let mut scalar = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut scalar).as_bytes());
                }
            }
        } else {
            bytes.extend_from_slice(self.token.as_bytes());
        }
    }
}

#[cfg(test)]
pub(super) fn overflow_requirements(
    ids: usize,
    pieces: usize,
) -> Result<DecodeCompileRequirements, DecodeSourceError> {
    requirements(ids, pieces)
}
