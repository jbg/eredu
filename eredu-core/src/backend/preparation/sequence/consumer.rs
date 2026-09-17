//! Type-derived fixed consumer representations; never a grant or byte allocator.
use crate::{
    BackendFailure, FinishReason, GenerationOutput, GenerationTiming, GenerationTokenIds,
    RetainedGenerationSequence,
};
use std::{any::TypeId, error::Error, mem::size_of};

/// Immutable concrete consumer identity and named fixed control overlaps.
///
/// No constructor accepts bytes or custody. Transitive parser, source, event,
/// tokenizer and error payloads remain separately required bounds. Equality
/// identifies the specialization, not a funding account or completion proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationSequenceConsumerLayout {
    cursor: TypeId,
    step_error: TypeId,
    failure: TypeId,
    ordinary_output: Option<TypeId>,
    plain_text_output: bool,
    terminal_text_output: bool,
    terminal_source_controls: bool,
    bytes: usize,
}
impl GenerationSequenceConsumerLayout {
    /// Authenticates the concrete cursor whose fixed copy controls are priced.
    pub fn is_cursor<C: 'static>(&self) -> bool {
        self.cursor == TypeId::of::<C>()
    }

    /// Selects the concrete retained IDs/text result and borrowed plain delivery
    /// before admission. Its byte destination is derived separately from the
    /// actual decoder source, maximum and skip policy by the original provider.
    pub fn for_driver_with_terminal_text<
        C: 'static,
        E: 'static,
        F: Error + Send + Sync + 'static,
    >() -> Option<Self> {
        use crate::{GenerationPlainTextOutput, GenerationText};
        let bytes = [
            size_of::<GenerationPlainTextOutput>(),
            size_of::<Result<GenerationPlainTextOutput, C>>(),
            size_of::<Result<GenerationPlainTextOutput, GenerationTokenIds>>(),
            size_of::<GenerationTiming>(),
            size_of::<GenerationTiming>(),
            size_of::<GenerationTokenIds>(),
            size_of::<GenerationText>(),
            size_of::<Option<GenerationText>>(),
            size_of::<FinishReason>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        let mut layout = Self::for_layout::<C, E, F>(None, bytes)?.with_plain_text_output()?;
        layout.terminal_text_output = true;
        Some(layout)
    }
    /// Adds the actual private source adapter S, move-only session T and by-value
    /// startup failure E to a terminal-text consumer. This measures fixed Rust
    /// representations only: no bytes, attachment or source authority is accepted.
    /// Lifetimes in S/T may borrow the selected runtime; no TypeId erases them.
    pub fn with_terminal_source_types<S, T, E>(mut self) -> Option<Self> {
        if !self.terminal_text_output || self.terminal_source_controls {
            return None;
        }
        let controls = [
            size_of::<S>(),                         // source constructor/local
            size_of::<T>(),                         // retained session, including source and cursor
            size_of::<Option<T>>(),                 // pre-cancel or installed session
            size_of::<E>(),                         // by-value startup failure
            size_of::<Result<Option<T>, E>>(),      // startup return
            size_of::<Result<T, BackendFailure>>(), // consuming advancement
            size_of::<Result<crate::GenerationPlainTextOutput, BackendFailure>>(),
        ];
        self.bytes = controls
            .into_iter()
            .try_fold(self.bytes, usize::checked_add)?;
        self.terminal_source_controls = true;
        Some(self)
    }
    /// Measures concrete cursor C, step error E and consuming failure F, including
    /// their named return/retirement overlaps and one closed core error erasure.
    /// `None` means checked overflow. This performs no allocation or admission.
    pub fn for_driver_types<C: 'static, E: 'static, F: Error + Send + Sync + 'static>(
    ) -> Option<Self> {
        Self::for_layout::<C, E, F>(None, 0)
    }

    /// Adds only the closed ordinary retained terminal result to the same
    /// original consumer. Tuple-only admission cannot authorize this mode.
    /// No token/EOS buffer, provider, arbitrary result type or new owner is added.
    pub fn for_driver_with_ordinary_output<
        C: 'static,
        E: 'static,
        F: Error + Send + Sync + 'static,
    >() -> Option<Self> {
        type Output = GenerationOutput<(), GenerationTokenIds>;
        let output = [
            size_of::<Output>(),             // terminal constructor result/local
            size_of::<Result<Output, C>>(),  // consuming extraction return
            size_of::<GenerationTiming>(),   // terminal method input
            size_of::<GenerationTiming>(),   // constructor argument
            size_of::<GenerationTokenIds>(), // constructor owner argument
            size_of::<FinishReason>(),       // constructor terminal reason
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        Self::for_layout::<C, E, F>(Some(TypeId::of::<Output>()), output)
    }

    fn for_layout<C: 'static, E: 'static, F: Error + Send + Sync + 'static>(
        ordinary_output: Option<TypeId>,
        output_bytes: usize,
    ) -> Option<Self> {
        let parts = [
            size_of::<C>(),
            size_of::<E>(),
            size_of::<F>(),
            size_of::<Option<RetainedGenerationSequence>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<Option<()>, E>>(),
            size_of::<(GenerationTokenIds, FinishReason)>(),
            size_of::<Result<(GenerationTokenIds, FinishReason), C>>(),
            // Consuming construction/advancement and failure-local overlap.
            size_of::<Result<C, F>>(),
            // Association preflight returns an unchanged original sequence on
            // mismatch; no concrete consuming failure is created before proof.
            size_of::<Result<C, RetainedGenerationSequence>>(),
            BackendFailure::source_retention_peak_bytes::<F>()?,
            // Cold descriptor producer/result. Stored binding/provider copies are
            // included by those providers' concrete original control layouts.
            size_of::<Self>(),
            size_of::<Option<Self>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(output_bytes, usize::checked_add)?;
        Some(Self {
            cursor: TypeId::of::<C>(),
            step_error: TypeId::of::<E>(),
            failure: TypeId::of::<F>(),
            ordinary_output,
            plain_text_output: false,
            terminal_text_output: false,
            terminal_source_controls: false,
            bytes,
        })
    }
    /// Adds only the closed borrowed plain event/projection controls. Its source
    /// and fixed bytes belong to the original provider's separate concrete layout.
    /// No arbitrary bytes, type-erased payload, hold or event copy is accepted.
    pub fn with_plain_text_output(mut self) -> Option<Self> {
        if !self.plain_text_output {
            use crate::{
                GenerationPlainTextEvent, GenerationPlainTextEvents, GenerationPlainTextProjection,
            };
            let controls = [
                size_of::<GenerationPlainTextProjection>(),
                size_of::<GenerationPlainTextEvents<'static>>(), // returned batch
                size_of::<GenerationPlainTextEvents<'static>>(), // delivery iterator
                size_of::<GenerationPlainTextEvent<'static>>(),  // callback argument
                size_of::<GenerationPlainTextEvent<'static>>(),  // callback forwarding
                size_of::<(bool, bool)>(),
            ]; // matched/cancelled disposition
            self.bytes = controls
                .into_iter()
                .try_fold(self.bytes, usize::checked_add)?;
            self.plain_text_output = true;
        }
        Some(self)
    }
    /// Whether the original consumer explicitly includes borrowed plain output.
    pub const fn plain_text_output(&self) -> bool {
        self.plain_text_output
    }
    /// Whether this original descriptor selected the concrete retained text result.
    pub const fn terminal_text_output(&self) -> bool {
        self.terminal_text_output
    }
    /// Requested fixed consumer representations, excluding all transitive payloads.
    pub const fn retention_peak_bytes(&self) -> usize {
        self.bytes
    }
}
