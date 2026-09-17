//! Synchronous input transport and fixed transition diagnostics only.
use std::{any::Any, fmt::Debug};

/// Borrowed concrete decoder input supplied before original admission.
///
/// Implementations retain their actual immutable source association after a
/// unique cold take. The accepting runtime downcasts the concrete source owner;
/// this interface carries no arbitrary byte fact, allocation guard or tokenizer
/// policy. Core only forwards the same input and checks the returned provider.
pub trait GenerationDecoderInput: Debug + Send + Sync {
    /// Explicit original output requirement; legacy inputs request only suffixes.
    /// A plain request needs its dedicated original admission hook and provider.
    fn output_kind(&self) -> GenerationDecoderOutput {
        GenerationDecoderOutput::Suffix
    }
    /// Read-only concrete input, never an erased owning attachment.
    fn as_any(&self) -> &dyn Any;
}

/// Fixed provider-decoder diagnostics without allocating owned text/errors.
/// Borrowed candidate/prefix text remains private to the source-owning provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerationDecoderError {
    /// This provider has no originally admitted decoder.
    #[error("no original decoder source")]
    Unavailable,
    /// The existing first readiness preparation has not succeeded.
    #[error("decoder storage is not ready")]
    Unprepared,
    /// The originally admitted successful-call ceiling was reached.
    #[error("decoder call limit reached")]
    CallLimit,
    /// Failed transitions exhausted the retained history destination.
    #[error("decoder history limit reached")]
    HistoryLimit,
    /// The candidate did not extend the retained prefix; state is preserved.
    #[error("invalid decoded prefix at token {token_id}")]
    InvalidPrefix {
        /// Appended token identifier.
        token_id: u32,
        /// Retained expected prefix byte count.
        expected_bytes: usize,
        /// Complete candidate byte count.
        actual_bytes: usize,
    },
    /// The ordinary non-flushing finish check found incomplete output.
    #[error("incomplete byte sequence at decoder finish")]
    IncompleteByteSequence,
    /// A concrete provider violated its sealed source/storage geometry.
    #[error("invalid original decoder storage")]
    InvalidStorage,
}

/// Original decoder output contract, separate from tokenizer/parser selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationDecoderOutput {
    /// Ordinary decoded suffix; stop/parser storage remains with the caller.
    Suffix,
    /// Actual originally sealed literal-stop source and fixed visible destination.
    PlainText,
}
/// Provider-owned lexical output. No semantic publication or owned String.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationPlainText<'a> {
    /// Visible complete UTF-8 text, excluding stops and withheld lookbehind.
    pub visible: &'a str,
    /// This transition matched an original literal stop.
    pub stop_matched: bool,
}
/// Synchronous borrowed plain event; owning public events remain a separate API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationPlainTextEvent<'a> {
    /// Nonempty user-visible text.
    TextDelta(&'a str),
    /// Terminal semantic disposition.
    Finished { reason: crate::FinishReason },
}
/// Two inline event descriptors; no Vec/String, Clone, or owning payload exit.
/// Borrowed text cannot outlive its original provider loan.
///
/// ```compile_fail
/// fn overlap(sequence: &mut eredu_core::RetainedGenerationSequence) {
///     let mut projection = eredu_core::GenerationPlainTextProjection::default();
///     let events = projection.push(sequence.project_plain_text(0).unwrap());
///     sequence.project_plain_text(1).unwrap();
///     for event in events { println!("{:?}", event); }
/// }
/// ```
#[derive(Debug)]
pub struct GenerationPlainTextEvents<'a> {
    events: [Option<GenerationPlainTextEvent<'a>>; 2],
    next: usize,
}
impl<'a> GenerationPlainTextEvents<'a> {
    /// Common plain projection used by legacy owning sinks and borrowed delivery.
    /// Empty text is suppressed. Text precedes the optional terminal descriptor.
    pub fn new(text: &'a str, finished: Option<crate::FinishReason>) -> Self {
        Self {
            events: [
                (!text.is_empty()).then_some(GenerationPlainTextEvent::TextDelta(text)),
                finished.map(|reason| GenerationPlainTextEvent::Finished { reason }),
            ],
            next: 0,
        }
    }
}
impl<'a> Iterator for GenerationPlainTextEvents<'a> {
    type Item = GenerationPlainTextEvent<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.events.len() {
            let event = self.events[self.next].take();
            self.next += 1;
            if event.is_some() {
                return event;
            }
        }
        None
    }
}
impl std::iter::FusedIterator for GenerationPlainTextEvents<'_> {}
/// Plain terminal projection shared by synchronous consumers. This contains no
/// decoder, stop policy, funding owner or second generation loop.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerationPlainTextProjection {
    finished: bool,
}
impl GenerationPlainTextProjection {
    /// Whether this existing projection has already emitted its terminal event.
    pub const fn is_finished(&self)->bool { self.finished }
    /// Makes stop termination final before any callback observes the returned batch.
    pub fn push<'a>(&mut self, text: GenerationPlainText<'a>) -> GenerationPlainTextEvents<'a> {
        if self.finished {
            return GenerationPlainTextEvents::new("", None);
        }
        self.finished = text.stop_matched;
        GenerationPlainTextEvents::new(
            text.visible,
            text.stop_matched
                .then_some(crate::FinishReason::StopSequence),
        )
    }
    /// Flushes once after successful decoder finish, preserving the actual reason.
    pub fn finish<'a>(
        &mut self,
        text: &'a str,
        reason: crate::FinishReason,
    ) -> GenerationPlainTextEvents<'a> {
        if self.finished {
            return GenerationPlainTextEvents::new("", None);
        }
        self.finished = true;
        GenerationPlainTextEvents::new(text, Some(reason))
    }
    /// Cancels without asking the decoder or stop source to flush anything.
    pub fn cancel(&mut self) -> GenerationPlainTextEvents<'static> {
        self.finish("", crate::FinishReason::Cancelled)
    }
}
