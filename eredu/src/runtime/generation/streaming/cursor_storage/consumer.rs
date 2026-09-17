//! Private fixed-envelope transfer only. Public facade payload admission remains
//! closed; this does not bound the source, decoder, parser or delivered events.
use super::*;
mod snapshot;
use eredu_core::{BackendFailure, BackendFailureKind, GenerationSequenceConsumerLayout};
use std::{error::Error, marker::PhantomData};

type StepError<S, D> = CommittedGenerationError<S, D, RetainedSequencePreparationError>;
type Cursor = CommittedGenerationCursor<RetainedGenerationSequenceStorage>;

/// The specialization whose exact descriptor was consumed at original admission.
/// There is no mutable/raw cursor exit or Clone/snapshot implementation.
#[derive(Debug)]
pub(crate) struct RetainedConsumerCursor<
    S,
    D,
    const ORDINARY: bool = false,
    const PLAIN: bool = false,
    const TEXT: bool = false,
> {
    layout: GenerationSequenceConsumerLayout,
    types: PhantomData<fn() -> (S, D)>,
    // Retire fixed controls before the same original sequence owner.
    cursor: Cursor,
}
// Keep the actual owner last in BOTH paths. Preparation has moved the sole
// sequence into the cause; all other errors leave it in the cursor. This enum's
// declaration order also retires non-owning fixed controls before that owner.
#[derive(Debug)]
enum FailureOrder<C, E> {
    CursorFirst { cursor: C, cause: E },
    CauseFirst { cause: E, cursor: C },
}
impl<C, E> FailureOrder<C, E> {
    fn new(cause_owns_sequence: bool, cursor: C, cause: E) -> Self {
        if cause_owns_sequence {
            Self::CursorFirst { cursor, cause }
        } else {
            Self::CauseFirst { cause, cursor }
        }
    }
    fn cause(&self) -> &E {
        match self {
            Self::CursorFirst { cause, .. } | Self::CauseFirst { cause, .. } => cause,
        }
    }
}
/// Consuming failure with its sole sequence owner last. No owned cause extraction.
#[derive(Debug)]
pub(crate) struct RetainedCursorFailure<
    S,
    D,
    const ORDINARY: bool = false,
    const PLAIN: bool = false,
    const TEXT: bool = false,
> {
    order: FailureOrder<RetainedConsumerCursor<S, D, ORDINARY, PLAIN, TEXT>, StepError<S, D>>,
}
impl<S, D, const ORDINARY: bool, const PLAIN: bool, const TEXT: bool>
    RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>
{
    fn new(
        cause: StepError<S, D>,
        cursor: RetainedConsumerCursor<S, D, ORDINARY, PLAIN, TEXT>,
    ) -> Self {
        let cause_owns_sequence = matches!(&cause, CommittedGenerationError::Preparation(_));
        debug_assert!(!cause_owns_sequence || cursor.cursor.sequence.is_none());
        Self {
            order: FailureOrder::new(cause_owns_sequence, cursor, cause),
        }
    }
    #[cfg(test)]
    pub(super) fn cause_owns_sequence_for_test(&self) -> bool {
        matches!(&self.order, FailureOrder::CursorFirst { .. })
    }
    pub(crate) fn cause(&self) -> &StepError<S, D> {
        self.order.cause()
    }
}
impl<
        S: Error + Send + Sync + 'static,
        D: Error + Send + Sync + 'static,
        const ORDINARY: bool,
        const PLAIN: bool,
        const TEXT: bool,
    > RetainedConsumerCursor<S, D, ORDINARY, PLAIN, TEXT>
{
    /// Actual final concrete controls, completed before constructing the request.
    /// None is an unknown/overflow rejection, never a zero-byte contribution.
    pub(crate) fn layout() -> Option<GenerationSequenceConsumerLayout> {
        if TEXT {
            if !PLAIN || ORDINARY {
                return None;
            }
            return GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
                Self,
                StepError<S, D>,
                RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>,
            >();
        }
        let layout = if ORDINARY {
            GenerationSequenceConsumerLayout::for_driver_with_ordinary_output::<
                Self,
                StepError<S, D>,
                RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>,
            >()
        } else {
            GenerationSequenceConsumerLayout::for_driver_types::<
                Self,
                StepError<S, D>,
                RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>,
            >()
        }?;
        if PLAIN {
            layout.with_plain_text_output()
        } else {
            Some(layout)
        }
    }
    /// Checks the consumed provider association without copying tokens or custody.
    pub(crate) fn from_sequence(
        sequence: RetainedGenerationSequence,
    ) -> Result<Self, RetainedGenerationSequence> {
        Self::from_sequence_layout(sequence, Self::layout())
    }
    fn from_sequence_layout(
        sequence: RetainedGenerationSequence,
        layout: Option<GenerationSequenceConsumerLayout>,
    ) -> Result<Self, RetainedGenerationSequence> {
        if layout.is_none()
            || sequence.consumer_layout() != layout.as_ref()
            || (sequence.decoder_output() == eredu_core::GenerationDecoderOutput::PlainText)
                != PLAIN
        {
            return Err(sequence);
        }
        Ok(Self {
            cursor: Cursor::from_retained_sequence(sequence),
            layout: layout.expect("matched concrete descriptor"),
            types: PhantomData,
        })
    }
    pub(crate) fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    pub(crate) fn finish_reason(&self) -> Option<FinishReason> {
        self.cursor.finish_reason()
    }

    /// Keeps unsuccessful extraction intact; a successful result retains the same
    /// original raw owner through all immutable/iterator aliases.
    pub(crate) fn into_tokens(self) -> Result<(GenerationTokenIds, FinishReason), Self> {
        match self.cursor.into_retained_tokens() {
            Ok(value) => Ok(value),
            Err(cursor) => Err(Self {
                cursor,
                layout: self.layout,
                types: PhantomData,
            }),
        }
    }
}
impl<S: Error + Send + Sync + 'static, D: Error + Send + Sync + 'static, const PLAIN: bool>
    RetainedConsumerCursor<S, D, true, PLAIN>
{
    /// One originally selected ordinary terminal result, with no allocation or
    /// token copy. The tuple-only specialization has no such method.
    pub(crate) fn into_output(
        self,
        timing: eredu_core::GenerationTiming,
    ) -> Result<eredu_core::GenerationOutput<(), GenerationTokenIds>, Self> {
        match self.into_tokens() {
            Ok((tokens, reason)) => Ok(eredu_core::GenerationOutput::from_retained(
                tokens, reason, timing,
            )),
            Err(cursor) => Err(cursor),
        }
    }
}
impl<S: Error, D: Error, const ORDINARY: bool, const PLAIN: bool, const TEXT: bool> fmt::Display
    for RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cause() {
            CommittedGenerationError::Source(e) => fmt::Display::fmt(e, f),
            CommittedGenerationError::Pipeline(e) => fmt::Display::fmt(e, f),
            CommittedGenerationError::Preparation(e) => fmt::Display::fmt(e, f),
            CommittedGenerationError::Lifecycle(e) => fmt::Display::fmt(e, f),
            CommittedGenerationError::Delivery(e) => fmt::Display::fmt(e, f),
            CommittedGenerationError::MissingTerminalToken => {
                f.write_str("generation source stopped without a terminal token")
            }
        }
    }
}
impl<
        S: Error + 'static,
        D: Error + 'static,
        const ORDINARY: bool,
        const PLAIN: bool,
        const TEXT: bool,
    > Error for RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.cause() {
            CommittedGenerationError::Source(e) => Some(e),
            CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Decoder(e)) => Some(e),
            CommittedGenerationError::Pipeline(CommittedTokenPipelineError::OriginalDecoder(e)) => {
                Some(e)
            }
            CommittedGenerationError::Preparation(e) => Some(e),
            CommittedGenerationError::Lifecycle(e) => Some(e),
            CommittedGenerationError::Delivery(e) => Some(e),
            _ => None,
        }
    }
}
impl<
        S: Error + Send + Sync + 'static,
        D: Error + Send + Sync + 'static,
        const ORDINARY: bool,
        const PLAIN: bool,
        const TEXT: bool,
    > RetainedCursorFailure<S, D, ORDINARY, PLAIN, TEXT>
{
    /// One prepriced concrete error erasure; core deallocates its source Box
    /// before this cause and cursor retire. No native error wrapper is involved.
    pub(crate) fn into_backend_failure(self, kind: BackendFailureKind) -> BackendFailure {
        BackendFailure::new(kind, self)
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    struct Witness(&'static str, Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Witness {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }
    #[test]
    fn consumer_failure_order_retires_nonowning_controls_before_actual_owner() {
        for cause_owns_sequence in [false, true] {
            let events = Rc::new(RefCell::new(vec![]));
            let cursor = Witness("cursor", events.clone());
            let cause = Witness("cause", events.clone());
            drop(FailureOrder::new(cause_owns_sequence, cursor, cause));
            assert_eq!(
                *events.borrow(),
                if cause_owns_sequence {
                    vec!["cursor", "cause"]
                } else {
                    vec!["cause", "cursor"]
                }
            );
        }
    }
}

impl<S: Error + Send + Sync + 'static, D: Error + Send + Sync + 'static, const ORDINARY: bool>
    RetainedConsumerCursor<S, D, ORDINARY, false>
{
    /// Advances the SAME step algorithm and readiness calls. Failure consumes the
    /// cursor, including non-Preparation causes whose type owns no token storage.
    pub(crate) fn advance<T, Decoder>(
        mut self,
        source: &mut T,
        pipeline: &mut CommittedTokenPipeline<Decoder>,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<Self, RetainedCursorFailure<S, D, ORDINARY, false>>
    where
        T: CommittedTokenSource<Error = S>,
        Decoder: TokenDecoderBackend<Error = D>,
    {
        match self.cursor.step(source, pipeline, cancellation, emit) {
            Ok(()) => Ok(self),
            Err(cause) => Err(RetainedCursorFailure::new(cause, self)),
        }
    }
}
impl<S: Error + Send + Sync + 'static, const ORDINARY: bool, const TEXT: bool>
    RetainedConsumerCursor<S, eredu_core::GenerationDecoderError, ORDINARY, true, TEXT>
{
    /// One borrowed event route through the same cursor readiness/commit driver.
    /// No legacy allocating parser, structural map or owning event conversion is
    /// accepted here. Its exact mode was checked before constructing this cursor.
    pub(crate) fn advance_plain<T>(
        mut self,
        source: &mut T,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'a> FnMut(eredu_core::GenerationPlainTextEvent<'a>),
    ) -> Result<
        Self,
        RetainedCursorFailure<S, eredu_core::GenerationDecoderError, ORDINARY, true, TEXT>,
    >
    where
        T: CommittedTokenSource<Error = S>,
    {
        // Terminal projection state lasts through this entire step's callbacks
        // and settlement. A terminal cursor cannot be advanced again; any stop
        // lookbehind across nonterminal steps remains in the original provider.
        let mut projection = eredu_core::GenerationPlainTextProjection::default();
        match self
            .cursor
            .step_pipeline(source, &mut projection, cancellation, emit)
        {
            Ok(()) => Ok(self),
            Err(cause) => Err(RetainedCursorFailure::new(cause, self)),
        }
    }
}

impl<S: Error + Send + Sync + 'static>
    RetainedConsumerCursor<S, eredu_core::GenerationDecoderError, false, true, true>
{
    pub(crate) fn terminal_layout<Source, Session, StartError>(
    ) -> Option<GenerationSequenceConsumerLayout> {
        Self::layout()?.with_terminal_source_types::<Source, Session, StartError>()
    }
    pub(crate) fn from_terminal_sequence<Source, Session, StartError>(
        sequence: RetainedGenerationSequence,
    ) -> Result<Self, RetainedGenerationSequence> {
        Self::from_sequence_layout(
            sequence,
            Self::terminal_layout::<Source, Session, StartError>(),
        )
    }
    /// Freezes the same original payload into its concrete immutable text/ID views.
    pub(crate) fn into_text_output(
        self,
        timing: eredu_core::GenerationTiming,
    ) -> Result<eredu_core::GenerationPlainTextOutput, Self> {
        match self.into_tokens() {
            Ok((tokens, reason)) => Ok(eredu_core::GenerationPlainTextOutput::from_retained(
                tokens, reason, timing,
            )
            .unwrap_or_else(|_| unreachable!("original terminal descriptor and provider agree"))),
            Err(cursor) => Err(cursor),
        }
    }
}
