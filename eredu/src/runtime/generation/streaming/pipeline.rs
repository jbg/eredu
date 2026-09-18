//! Event/storage dispatch beneath the one existing commitment/readiness driver.
use super::*;
use eredu_core::{
    GenerationDecoderError, GenerationPlainTextEvent, GenerationPlainTextProjection,
    RetainedGenerationSequenceStorage,
};

pub(super) trait CursorPipeline<S: CursorStorage> {
    type DecoderError;
    type Event<'a>;
    fn push(
        &mut self,
        sequence: &mut GenerationSequence<S>,
        id: u32,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(bool, bool), CommittedTokenPipelineError<Self::DecoderError>>;
    fn finish(
        &mut self,
        sequence: &mut GenerationSequence<S>,
        reason: FinishReason,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<Self::DecoderError>>;
    fn cancel(
        &mut self,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<Self::DecoderError>>;
}
impl<S: CursorStorage, D: TokenDecoderBackend> CursorPipeline<S> for CommittedTokenPipeline<D> {
    type DecoderError = D::Error;
    type Event<'a> = SemanticEvent;
    fn push(
        &mut self,
        sequence: &mut GenerationSequence<S>,
        id: u32,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(bool, bool), CommittedTokenPipelineError<D::Error>> {
        match S::decode_token(sequence, id) {
            Some(decoded) => self.push_original_cancellable(id, decoded, cancellation, emit),
            None => self.push_cancellable(id, cancellation, emit),
        }
    }
    fn finish(
        &mut self,
        sequence: &mut GenerationSequence<S>,
        reason: FinishReason,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<D::Error>> {
        match S::finish_decoder(sequence) {
            Some(local) => self.finish_original(local, reason, emit),
            None => CommittedTokenPipeline::finish(self, reason, emit),
        }
    }
    fn cancel(
        &mut self,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<D::Error>> {
        CommittedTokenPipeline::cancel(self, emit);
        Ok(())
    }
}
impl CursorPipeline<RetainedGenerationSequenceStorage> for GenerationPlainTextProjection {
    type DecoderError = GenerationDecoderError;
    type Event<'a> = GenerationPlainTextEvent<'a>;
    fn push(
        &mut self,
        sequence: &mut GenerationSequence<RetainedGenerationSequenceStorage>,
        id: u32,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(bool, bool), CommittedTokenPipelineError<GenerationDecoderError>> {
        let text = sequence
            .project_plain_text(id)
            .map_err(CommittedTokenPipelineError::OriginalDecoder)?;
        let matched = text.stop_matched;
        // The projection makes terminal state final before lending any descriptor.
        for event in GenerationPlainTextProjection::push(self, text) {
            emit(event);
            if !matched && cancellation.is_cancelled() {
                break;
            }
        }
        Ok((matched, cancellation.is_cancelled()))
    }
    fn finish(
        &mut self,
        sequence: &mut GenerationSequence<RetainedGenerationSequenceStorage>,
        reason: FinishReason,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<GenerationDecoderError>> {
        let text = sequence
            .finish_plain_text()
            .map_err(CommittedTokenPipelineError::OriginalDecoder)?;
        for event in GenerationPlainTextProjection::finish(self, text, reason) {
            emit(event);
        }
        Ok(())
    }
    fn cancel(
        &mut self,
        emit: &mut impl for<'a> FnMut(Self::Event<'a>),
    ) -> Result<(), CommittedTokenPipelineError<GenerationDecoderError>> {
        for event in GenerationPlainTextProjection::cancel(self) {
            emit(event);
        }
        Ok(())
    }
}

impl<S: CursorStorage> CursorPipeline<S> for eredu_core::SemanticStateOwner {
    type DecoderError = eredu_core::SpeculativeOutputError;
    type Event<'a> = SemanticEvent;

    fn push(
        &mut self,
        _sequence: &mut GenerationSequence<S>,
        id: u32,
        cancellation: &GenerationCancellationToken,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<(bool, bool), CommittedTokenPipelineError<Self::DecoderError>> {
        let matched = self
            .push_token(id)
            .map_err(CommittedTokenPipelineError::Decoder)?;
        self.publish_events(emit);
        Ok((matched, cancellation.is_cancelled()))
    }

    fn finish(
        &mut self,
        _sequence: &mut GenerationSequence<S>,
        reason: FinishReason,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<(), CommittedTokenPipelineError<Self::DecoderError>> {
        eredu_core::SemanticStateOwner::finish(self, reason)
            .map_err(CommittedTokenPipelineError::Decoder)?;
        self.publish_events(emit);
        Ok(())
    }

    fn cancel(
        &mut self,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<(), CommittedTokenPipelineError<Self::DecoderError>> {
        eredu_core::SemanticStateOwner::cancel(self)
            .map_err(CommittedTokenPipelineError::Decoder)?;
        self.publish_events(emit);
        Ok(())
    }
}
