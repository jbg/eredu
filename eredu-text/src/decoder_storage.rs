//! Fixed-destination streaming decode for concrete HF decoder forms.
//!
//! Preparation allocates a standalone immutable vocabulary program. Execution
//! borrows that program and exact caller-owned buffers, and never calls HF or
//! allocates an output. These are storage requirements, not original admission,
//! a tokenizer-wide storage bound, or a facade/event delivery contract.

use std::mem::size_of;
#[cfg(test)]
use tokenizers::decoders::DecoderWrapper;

use crate::tokenizer::TokenizerSnapshot;

pub(crate) mod compiler;
pub use compiler::{DecodeCompileFailure, DecodeCompilePlan, DecodeCompileRequirements};

mod mode;
use mode::FallbackOrder;
pub(crate) use mode::Mode;

#[derive(Debug)]
struct Record {
    id: u32,
    start: usize,
    end: usize,
    special: bool,
    byte: Option<u8>,
}

/// Cold compilation could not produce a supported exact decoder program.
#[derive(Debug, thiserror::Error)]
pub enum DecodeSourceError {
    /// Distinct model spellings cannot share one decoding ID. Model/added-token
    /// aliases are permitted and keep upstream added-token precedence.
    #[error("model vocabulary contains duplicate token ID {0}")]
    DuplicateModelId(u32),
    /// This decoder form has no fixed-destination lowering in this module yet.
    #[error("unsupported fixed-destination decoder form")]
    UnsupportedDecoder,
    /// A required extent exceeds the host address space.
    #[error("decoder source extent overflow")]
    Overflow,
    /// A fallible allocation owned by this compiler failed.
    #[error("decoder source allocation failed")]
    Allocation(#[from] std::collections::TryReserveError),
}

/// Immutable, move-only decoding program compiled from one actual snapshot.
///
/// It owns the immutable record and byte vector capacities used by execution;
/// there is no shrink/reallocation during publication. It does not retain HF.
/// [`DecodeCompilePlan`] reports separate construction requirements and preserves
/// allocation failures; [`Self::storage_bytes`] reports only the final program.
/// Neither interface establishes original-account or tokenizer-wide admission.
#[derive(Debug)]
pub struct PreparedDecodeSource {
    mode: Mode,
    records: Vec<Record>,
    bytes: Vec<u8>,
    max_piece: usize,
    storage_bytes: usize,
}

impl PreparedDecodeSource {
    /// Compiles plain space joining, ByteLevel (direct or singleton Sequence),
    /// or the exact ordered ByteFallback/Fuse/Replace/Strip pipelines used by
    /// the supported GGUF constructors. Other decoder sequences reject.
    ///
    /// Added-token reverse IDs and model IDs are enumerated separately. Lookup
    /// then uses HF's added-first (including normalized-cache) spelling and its
    /// actual special-token membership. Unknown IDs remain absent.
    pub fn prepare(snapshot: &TokenizerSnapshot) -> Result<Self, DecodeSourceError> {
        // Ordinary compatibility entry: partial buffers retire before its old
        // non-owning error is returned. Admitted construction uses the consuming
        // plan directly, retaining its closed failure under original custody.
        DecodeCompilePlan::prepare(snapshot)?
            .compile()
            .map_err(DecodeCompileFailure::discard_partial)
    }

    /// Live program struct plus both retained vector capacities; excludes allocator
    /// bookkeeping, cold construction overlap and the original HF snapshot.
    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    /// Number of actual retained reverse-ID records (not a dense ID extent).
    pub fn token_count(&self) -> usize {
        self.records.len()
    }

    fn record(&self, id: u32, skip_special: bool) -> Option<&Record> {
        let row = &self.records[self.records.binary_search_by_key(&id, |r| r.id).ok()?];
        (!(skip_special && row.special)).then_some(row)
    }

    fn piece(&self, id: u32, skip_special: bool) -> Option<&[u8]> {
        let row = self.record(id, skip_special)?;
        Some(&self.bytes[row.start..row.end])
    }

    fn fallback_into(
        &self,
        ids: &[u32],
        skip_special: bool,
        replace_run: bool,
        run: &mut [u8],
        text: &mut [u8],
    ) -> usize {
        let (mut pending, mut len) = (0, 0);
        for id in ids {
            let Some(row) = self.record(*id, skip_special) else {
                continue;
            };
            if let Some(byte) = row.byte {
                run[pending] = byte;
                pending += 1;
            } else {
                len += flush_fallback(&run[..pending], replace_run, &mut text[len..]);
                pending = 0;
                // Even an empty ordinary record ends the preceding byte run.
                let piece = &self.bytes[row.start..row.end];
                text[len..len + piece.len()].copy_from_slice(piece);
                len += piece.len();
            }
        }
        len + flush_fallback(&run[..pending], replace_run, &mut text[len..])
    }

    // All destinations were checked together before construction, and all ID
    // slices have at most layout.calls elements. No allocating fallback exists.
    fn decode_into(
        &self,
        ids: &[u32],
        skip_special: bool,
        raw: &mut [u8],
        text: &mut [u8],
    ) -> usize {
        match self.mode {
            Mode::Join => {
                let mut len = 0;
                let mut seen = false;
                for id in ids {
                    if let Some(piece) = self.piece(*id, skip_special) {
                        if seen {
                            text[len] = b' ';
                            len += 1;
                        }
                        text[len..len + piece.len()].copy_from_slice(piece);
                        len += piece.len();
                        seen = true;
                    }
                }
                len
            }
            Mode::ByteLevel => {
                let mut len = 0;
                for id in ids {
                    if let Some(piece) = self.piece(*id, skip_special) {
                        raw[len..len + piece.len()].copy_from_slice(piece);
                        len += piece.len();
                    }
                }
                lossy_into(&raw[..len], text)
            }
            Mode::Metaspace {
                replacement,
                remove_first,
            } => {
                let mut len = 0;
                let mut first = true;
                for id in ids {
                    if let Some(piece) = self.piece(*id, skip_special) {
                        let piece =
                            std::str::from_utf8(piece).expect("original UTF-8 token spelling");
                        for mut c in piece.chars() {
                            if c == replacement {
                                if first && remove_first {
                                    continue;
                                }
                                c = ' ';
                            }
                            let end = len + c.len_utf8();
                            c.encode_utf8(&mut text[len..end]);
                            len = end;
                        }
                        first = false;
                    }
                }
                len
            }
            Mode::Fallback { order, strip } => {
                let len = if order == FallbackOrder::ByteLevelLast {
                    // The exact layout is N + F + F. Derive partitions from
                    // these checked extents, never from compacted history.
                    let fused_capacity = text.len() / 3;
                    let (run, rest) = raw.split_at_mut(raw.len() - 2 * fused_capacity);
                    let (fused, converted) = rest.split_at_mut(fused_capacity);
                    let n = self.fallback_into(ids, skip_special, true, run, fused);
                    let fused = std::str::from_utf8(&fused[..n]).expect("fallback produces UTF-8");
                    let raw_len = if fused.chars().all(|c| byte_for_char(c).is_some()) {
                        let mut len = 0;
                        for c in fused.chars() {
                            converted[len] =
                                byte_for_char(c).expect("checked whole fused alphabet");
                            len += 1;
                        }
                        len
                    } else {
                        converted[..n].copy_from_slice(fused.as_bytes());
                        n
                    };
                    lossy_into(&converted[..raw_len], text)
                } else {
                    self.fallback_into(
                        ids,
                        skip_special,
                        order == FallbackOrder::ReplaceLast,
                        raw,
                        text,
                    )
                };
                if strip && len != 0 && text[0] == b' ' {
                    text.copy_within(1..len, 0);
                    len - 1
                } else {
                    len
                }
            }
        }
    }
}

// Exactly HF 0.23.2's cold byte-token parser, including its accepted radix syntax.
fn fallback_byte(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

fn flush_fallback(run: &[u8], replace: bool, text: &mut [u8]) -> usize {
    match std::str::from_utf8(run) {
        Ok(valid) if replace => {
            let mut len = 0;
            for c in valid.chars() {
                let c = if c == '▁' { ' ' } else { c };
                let n = c.len_utf8();
                c.encode_utf8(&mut text[len..len + n]);
                len += n;
            }
            len
        }
        Ok(_) => {
            text[..run.len()].copy_from_slice(run);
            run.len()
        }
        Err(_) => {
            // ByteFallback invalidates the WHOLE run. from_utf8_lossy would
            // preserve valid prefixes and group invalid subsequences instead.
            for (index, _) in run.iter().enumerate() {
                text[index * 3..index * 3 + 3].copy_from_slice("�".as_bytes());
            }
            run.len() * 3
        }
    }
}

// The inverse of HF 0.23.2 bytes_char(), without its lazy global table. Only
// called during cold compilation and whole-fused ByteLevel conversion.
// Printable byte ranges map to themselves; all other bytes map, in byte order,
// to consecutive scalars starting at 256. There is no lazy table or allocation.
pub(crate) fn byte_for_char(c: char) -> Option<u8> {
    let value = u32::from(c);
    let direct =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    if direct(value) {
        return Some(value as u8);
    }
    let mut scalar = 256;
    for b in 0..=255u32 {
        if !direct(b) {
            if value == scalar {
                return Some(b as u8);
            }
            scalar += 1;
        }
    }
    None
}

// Matches String::from_utf8_lossy: one replacement for each invalid subsequence,
// including a single replacement for the final incomplete sequence.
fn lossy_into(mut raw: &[u8], text: &mut [u8]) -> usize {
    let mut out = 0;
    loop {
        match std::str::from_utf8(raw) {
            Ok(valid) => {
                text[out..out + valid.len()].copy_from_slice(valid.as_bytes());
                return out + valid.len();
            }
            Err(error) => {
                let valid = error.valid_up_to();
                text[out..out + valid].copy_from_slice(&raw[..valid]);
                out += valid;
                text[out..out + 3].copy_from_slice("�".as_bytes());
                out += 3;
                match error.error_len() {
                    Some(invalid) => raw = &raw[valid + invalid..],
                    None => return out,
                }
            }
        }
    }
}

/// One of the four independent execution destinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBuffer {
    /// Retained lookbehind IDs, separate from canonical generation history.
    History,
    /// ByteLevel bytes or exact byte-run/fused/conversion scratch regions;
    /// zero for plain join.
    Raw,
    /// Full decoded candidate; returned output borrows this storage.
    Candidate,
    /// Retained decoded prefix.
    Prefix,
}

/// Fixed, non-owning execution diagnostic. It does not allocate text or custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DecodeStorageError {
    /// A checked layout operation overflowed.
    #[error("decoder layout overflow")]
    Overflow,
    /// All four slice extents must exactly match the prepared layout.
    #[error("incorrect {buffer:?} extent: expected {expected}, got {actual}")]
    Extent {
        /// Destination that does not match.
        buffer: DecodeBuffer,
        /// Required element count (u32 for History, bytes otherwise).
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// The finite successful-step ceiling has been reached, even if IDs compacted.
    #[error("decoder successful-step limit reached")]
    CallLimit,
    /// Retained IDs filled the fixed destination after failed transitions.
    #[error("decoder history destination exhausted")]
    HistoryLimit,
    /// HF's candidate failed to extend the retained prefix. The actual appended
    /// ID and repaired prefix remain in state, exactly as for HF. Text is exposed
    /// by borrowed state accessors until the next operation, not owned here.
    #[error("invalid decoded prefix at token {token_id}")]
    InvalidPrefix {
        /// Appended token ID.
        token_id: u32,
        /// Expected prefix occupies 0..expected_bytes in prefix storage.
        expected_bytes: usize,
        /// Actual candidate occupies 0..actual_bytes in candidate storage.
        actual_bytes: usize,
    },
    /// Finish observed more decoded bytes than the retained prefix.
    #[error("incomplete byte sequence at decoder finish")]
    IncompleteByteSequence,
}

/// Exact source-associated execution requirements for a fresh stream.
///
/// N is a successful `step` ceiling, not the current compacted history length.
/// ByteLevel reserves N*B raw bytes and 3*N*B bytes in each UTF-8 destination.
/// Plain join reserves N*B + max(N-1,0) bytes in each UTF-8 destination.
/// Fallback pipelines use F=N*max(B,3), where B is the largest packed ordinary
/// piece. Replacing before fallback or after fusion uses N raw bytes and F per
/// UTF-8 destination. Appended ByteLevel
/// uses N+F+F raw bytes and 3*F per UTF-8 destination, retaining the whole fused
/// token before its all-or-nothing alphabet conversion. All extents are checked.
/// No allocation, registration or original-budget authority is carried here.
#[derive(Debug)]
pub struct DecodeStreamLayout<'source> {
    source: &'source PreparedDecodeSource,
    calls: usize,
    skip_special: bool,
    raw: usize,
    text: usize,
    buffer_bytes: usize,
}

impl<'source> DecodeStreamLayout<'source> {
    /// Derives every extent with checked arithmetic from this actual source.
    pub fn for_source(
        source: &'source PreparedDecodeSource,
        calls: usize,
        skip_special: bool,
    ) -> Result<Self, DecodeStorageError> {
        let pieces = calls
            .checked_mul(source.max_piece)
            .ok_or(DecodeStorageError::Overflow)?;
        let (raw, text) = match source.mode {
            Mode::Join => (
                0,
                pieces
                    .checked_add(calls.saturating_sub(1))
                    .ok_or(DecodeStorageError::Overflow)?,
            ),
            Mode::ByteLevel => (
                pieces,
                pieces.checked_mul(3).ok_or(DecodeStorageError::Overflow)?,
            ),
            Mode::Metaspace { .. } => (0, pieces),
            Mode::Fallback { order, .. } => {
                let fused = calls
                    .checked_mul(source.max_piece.max(3))
                    .ok_or(DecodeStorageError::Overflow)?;
                if order == FallbackOrder::ByteLevelLast {
                    let raw = calls
                        .checked_add(fused)
                        .and_then(|n| n.checked_add(fused))
                        .ok_or(DecodeStorageError::Overflow)?;
                    let text = fused.checked_mul(3).ok_or(DecodeStorageError::Overflow)?;
                    (raw, text)
                } else {
                    (calls, fused)
                }
            }
        };
        // A non-overflowing usize product can still exceed Rust's maximum
        // slice allocation extent. Each separately supplied destination must
        // be representable, even for an all-empty vocabulary.
        std::alloc::Layout::array::<u32>(calls).map_err(|_| DecodeStorageError::Overflow)?;
        std::alloc::Layout::array::<u8>(raw).map_err(|_| DecodeStorageError::Overflow)?;
        std::alloc::Layout::array::<u8>(text).map_err(|_| DecodeStorageError::Overflow)?;
        let buffer_bytes = calls
            .checked_mul(size_of::<u32>())
            .and_then(|v| v.checked_add(raw))
            .and_then(|v| v.checked_add(text))
            .and_then(|v| v.checked_add(text))
            .ok_or(DecodeStorageError::Overflow)?;
        Ok(Self {
            source,
            calls,
            skip_special,
            raw,
            text,
            buffer_bytes,
        })
    }

    /// Required u32 history slots and successful-step ceiling.
    pub fn token_capacity(&self) -> usize {
        self.calls
    }
    /// Required raw byte destination extent.
    pub fn raw_capacity(&self) -> usize {
        self.raw
    }
    /// Required extent of EACH candidate/prefix destination.
    pub fn text_capacity(&self) -> usize {
        self.text
    }
    /// Sum of the four variable destinations, with u32 history width included.
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }
}

/// Concrete fixed control representations, separate from variable destinations.
///
/// These named sizes are diagnostics, not a sum or a bound for arbitrary caller
/// stack frames, compiler temporaries, source compilation, errors rendered to
/// strings, repeated copies, or a future owner/provider. A future original plan
/// must describe its actual construction/return overlap and closed owners.
#[derive(Debug)]
pub struct DecodeControlBytes {
    /// The immutable source struct (also included in source.storage_bytes()).
    pub source: usize,
    /// One source-associated layout value.
    pub layout: usize,
    /// One mutable stream value, which contains its consumed layout.
    pub stream: usize,
    /// Actual constructor result representation.
    pub constructor_result: usize,
    /// Actual step result representation, including its borrowed output.
    pub step_result: usize,
    /// Actual finish result representation.
    pub finish_result: usize,
    /// One fixed diagnostic value.
    pub error: usize,
}

impl DecodeControlBytes {
    /// Reports actual concrete Rust type sizes; creates no storage authority.
    pub fn actual() -> Self {
        Self {
            source: size_of::<PreparedDecodeSource>(),
            layout: size_of::<DecodeStreamLayout<'static>>(),
            stream: size_of::<DecodeStreamState<'static, 'static>>(),
            constructor_result: size_of::<
                Result<DecodeStreamState<'static, 'static>, DecodeStorageError>,
            >(),
            step_result: size_of::<Result<Option<&'static str>, DecodeStorageError>>(),
            finish_result: size_of::<Result<(), DecodeStorageError>>(),
            error: size_of::<DecodeStorageError>(),
        }
    }
}

/// Mutable stream borrowing exact caller-owned storage. No Clone or owned output.
///
/// `finish` is the existing residual check, not a flush or a terminal-state
/// transition. Cancellation is a caller decision to skip both step and finish.
/// Repeated finish/diagnostic access produces no owned text. A failed HF prefix
/// transition preserves its mutated frontier; a capacity rejection mutates none.
///
/// A live output loan prevents another operation from overwriting the buffer:
/// ```compile_fail
/// use eredu_text::decoder_storage::DecodeStreamState;
/// fn overlapping(stream: &mut DecodeStreamState<'_, '_>) {
///     let output = stream.step(1).unwrap().unwrap();
///     stream.step(2).unwrap();
///     println!("{output}");
/// }
/// ```
#[derive(Debug)]
pub struct DecodeStreamState<'source, 'storage> {
    layout: DecodeStreamLayout<'source>,
    ids: &'storage mut [u32],
    raw: &'storage mut [u8],
    candidate: &'storage mut [u8],
    prefix: &'storage mut [u8],
    progress: DecodeProgress,
}

impl<'source, 'storage> DecodeStreamState<'source, 'storage> {
    /// Validates ALL exact extents before writing any caller byte or ID. Source
    /// substitution is impossible: the consumed layout contains the source loan.
    pub fn new(
        layout: DecodeStreamLayout<'source>,
        ids: &'storage mut [u32],
        raw: &'storage mut [u8],
        candidate: &'storage mut [u8],
        prefix: &'storage mut [u8],
    ) -> Result<Self, DecodeStorageError> {
        for (buffer, expected, actual) in [
            (DecodeBuffer::History, layout.calls, ids.len()),
            (DecodeBuffer::Raw, layout.raw, raw.len()),
            (DecodeBuffer::Candidate, layout.text, candidate.len()),
            (DecodeBuffer::Prefix, layout.text, prefix.len()),
        ] {
            if expected != actual {
                return Err(DecodeStorageError::Extent {
                    buffer,
                    expected,
                    actual,
                });
            }
        }
        Ok(Self {
            layout,
            ids,
            raw,
            candidate,
            prefix,
            progress: DecodeProgress::default(),
        })
    }

    /// Appends one token with HF streaming frontier semantics and lends at most
    /// one UTF-8 suffix. Neither a caller-skipped cancellation nor finish counts
    /// as a step. Failed prefix transitions do not count as successful calls;
    /// their appended IDs still consume history space until HF compaction.
    pub fn step(&mut self, id: u32) -> Result<Option<&str>, DecodeStorageError> {
        self.progress.step(
            &self.layout,
            self.ids,
            self.raw,
            self.candidate,
            self.prefix,
            id,
        )
    }

    /// Checks the same residual byte-length condition as the existing facade
    /// finish path. Produces no extra text, clears no state, consumes no call.
    pub fn finish(&mut self) -> Result<(), DecodeStorageError> {
        self.progress
            .finish(&self.layout, self.ids, self.raw, self.candidate)
    }

    /// Actual retained lookbehind IDs, including an invalid-prefix appended ID.
    pub fn retained_ids(&self) -> &[u32] {
        &self.ids[..self.progress.ids_len]
    }
    /// Actual retained prefix; also the expected text for an InvalidPrefix error.
    pub fn prefix(&self) -> &str {
        std::str::from_utf8(&self.prefix[..self.progress.prefix_len]).expect("decoded prefix")
    }
    /// Full candidate of the latest step/finish, including InvalidPrefix text.
    pub fn candidate(&self) -> &str {
        std::str::from_utf8(&self.candidate[..self.progress.candidate_len])
            .expect("decoded candidate")
    }
    /// Current HF prefix index within retained history.
    pub fn prefix_index(&self) -> usize {
        self.progress.prefix_index
    }
    /// Successful steps since construction, independent of history compaction.
    pub fn successful_calls(&self) -> usize {
        self.progress.successful_calls
    }
}

// Pointer-free progress shared by the existing borrowed state and the owning
// original-provider companion. No source/storage pointer or copied ID history.
#[derive(Debug, Default, Clone)]
struct DecodeProgress {
    ids_len: usize,
    candidate_len: usize,
    prefix_len: usize,
    prefix_index: usize,
    successful_calls: usize,
}
impl DecodeProgress {
    fn step<'output>(
        &mut self,
        layout: &DecodeStreamLayout<'_>,
        ids: &mut [u32],
        raw: &mut [u8],
        candidate: &'output mut [u8],
        prefix: &mut [u8],
        id: u32,
    ) -> Result<Option<&'output str>, DecodeStorageError> {
        // Both checks precede even empty-prefix repair.
        if self.successful_calls == layout.calls {
            return Err(DecodeStorageError::CallLimit);
        }
        if self.ids_len == ids.len() {
            return Err(DecodeStorageError::HistoryLimit);
        }
        if self.prefix_len == 0 && self.ids_len != 0 {
            let len = layout.source.decode_into(
                &ids[..self.ids_len],
                layout.skip_special,
                raw,
                candidate,
            );
            self.candidate_len = len;
            if !candidate[..len].ends_with("�".as_bytes()) {
                prefix[..len].copy_from_slice(&candidate[..len]);
                self.prefix_len = len;
                self.prefix_index = self.ids_len;
            }
        }
        ids[self.ids_len] = id;
        self.ids_len += 1;
        let len =
            layout
                .source
                .decode_into(&ids[..self.ids_len], layout.skip_special, raw, candidate);
        self.candidate_len = len;
        if len > self.prefix_len && !candidate[..len].ends_with("�".as_bytes()) {
            if !candidate[..len].starts_with(&prefix[..self.prefix_len]) {
                return Err(DecodeStorageError::InvalidPrefix {
                    token_id: id,
                    expected_bytes: self.prefix_len,
                    actual_bytes: len,
                });
            }
            let start = self.prefix_len;
            let new_prefix_index = self.ids_len - self.prefix_index;
            ids.copy_within(self.prefix_index..self.ids_len, 0);
            self.ids_len = new_prefix_index;
            self.prefix_len =
                layout
                    .source
                    .decode_into(&ids[..self.ids_len], layout.skip_special, raw, prefix);
            self.prefix_index = new_prefix_index;
            self.successful_calls += 1;
            Ok(Some(
                std::str::from_utf8(&candidate[start..len]).expect("decoded UTF-8 suffix"),
            ))
        } else {
            self.successful_calls += 1;
            Ok(None)
        }
    }
    fn finish(
        &mut self,
        layout: &DecodeStreamLayout<'_>,
        ids: &[u32],
        raw: &mut [u8],
        candidate: &mut [u8],
    ) -> Result<(), DecodeStorageError> {
        self.candidate_len =
            layout
                .source
                .decode_into(&ids[..self.ids_len], layout.skip_special, raw, candidate);
        if self.candidate_len > self.prefix_len {
            Err(DecodeStorageError::IncompleteByteSequence)
        } else {
            Ok(())
        }
    }
}

mod owned;
pub use owned::{
    DecodeDestinationCopy, DecodeDestinationCopyError, DecodeDestinations,
    OwnedDecodePreparationError, OwnedDecodeStorage, OwnedDecodeStorageError,
};

#[cfg(test)]
mod tests;
