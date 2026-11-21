//! Type erasure and retention accounting for text prefill cursors.

use super::{ChunkedPrefillRequest, PrefillCursor, PrefillStep, RetainedPrefillCursor};
use crate::{composite_execution::CompositeArchitecture, RuntimeState};
use eredu_nn::{NeuralBackend, Tensor};
use std::{any::Any, ops::Range};

/// Progress or retention information unavailable for an erased prefill cursor.
#[derive(Debug, thiserror::Error)]
pub enum PrefillCursorError {
    /// The request has not reached its final decoder advance.
    #[error("retained prefill cursor is not ready for the final decoder step")]
    NotFinal,
    /// The cursor belongs to another prepared execution type.
    #[error("prefill cursor belongs to another architecture")]
    Foreign,
    /// The request cannot report its next step.
    #[error(transparent)]
    Progress(Box<dyn std::error::Error + Send + Sync>),
    /// The cursor's retained roots could not be measured.
    #[error(transparent)]
    Storage(#[from] eredu_nn::Error),
    /// An exact metadata size or tensor size was unavailable or overflowed.
    #[error("retained prefill cursor storage is unavailable")]
    RetentionUnavailable,
}

impl From<PrefillCursorError> for eredu_core::BackendFailure {
    fn from(error: PrefillCursorError) -> Self {
        match error {
            PrefillCursorError::NotFinal | PrefillCursorError::Foreign => {
                Self::from_error(error).with_model_state_preserved(true)
            }
            PrefillCursorError::Progress(source) => {
                Self::from_boxed_error(source).with_model_state_preserved(false)
            }
            error => Self::from_error(error),
        }
    }
}

type CursorResult<T> = Result<T, PrefillCursorError>;
type ChunkedCursor<A, B, S> = PrefillCursor<
    <A as CompositeArchitecture<B, S>>::PrefillRequest,
    <<A as CompositeArchitecture<B, S>>::PrefillRequest as ChunkedPrefillRequest<
        A, B, S,
    >>::Continuation,
>;

/// Owns architecture-erased request progress and retained tensor roots.
pub struct ErasedPrefillCursor(Box<dyn Cursor>);

impl Clone for ErasedPrefillCursor {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl ErasedPrefillCursor {
    /// Erases a resumable request and accounts for its distinct observed backings.
    pub fn new<C: RetainedPrefillCursor + 'static>(cursor: C) -> Self {
        Self(Box::new(TypedCursor {
            cursor,
            position: C::decoder_position,
            token_count: C::token_count,
            next_step: |cursor| {
                cursor
                    .next_step()
                    .map_err(|error| PrefillCursorError::Progress(Box::new(error)))
            },
            retained_bytes: physical_bytes,
        }))
    }

    /// Erases an ordinary chunked request and measures its retained logical tensor extents.
    pub fn chunked<A, B, S>(cursor: ChunkedCursor<A, B, S>) -> Self
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        A: CompositeArchitecture<B, S>,
        A::PrefillRequest: 'static,
        <A::PrefillRequest as ChunkedPrefillRequest<A, B, S>>::Continuation: 'static,
    {
        Self(Box::new(TypedCursor {
            cursor,
            position: PrefillCursor::position,
            token_count: PrefillCursor::token_count,
            next_step: |cursor| Ok(cursor.next_step()),
            retained_bytes: |cursor| {
                let request = cursor.request();
                let mut values = request.request_values();
                if let Some(continuation) = cursor.continuation() {
                    values.extend(request.retained_values(continuation));
                }
                values
                    .into_iter()
                    .try_fold(std::mem::size_of_val(cursor), |bytes, value| {
                        let logical = eredu_nn::tensor_storage::logical_bytes(
                            value.shape(),
                            value.element_type(),
                        )?;
                        logical
                            .and_then(|length| usize::try_from(length).ok())
                            .and_then(|length| bytes.checked_add(length))
                            .ok_or(PrefillCursorError::RetentionUnavailable)
                    })
            },
        }))
    }

    /// Borrows the exact cursor type expected by its executable.
    pub fn downcast_mut<C: 'static>(&mut self) -> Option<&mut C> {
        self.0.value_mut().downcast_mut()
    }

    /// Accesses the typed state of an ordinary chunked request.
    pub fn chunked_mut<A, B, S>(&mut self) -> CursorResult<&mut ChunkedCursor<A, B, S>>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        A: CompositeArchitecture<B, S>,
        A::PrefillRequest: 'static,
        <A::PrefillRequest as ChunkedPrefillRequest<A, B, S>>::Continuation: 'static,
    {
        self.downcast_mut().ok_or(PrefillCursorError::Foreign)
    }

    /// Number of committed decoder positions.
    pub fn decoder_position(&self) -> usize {
        self.0.decoder_position()
    }

    /// Total admitted decoder positions.
    pub fn token_count(&self) -> usize {
        self.0.token_count()
    }

    /// Next admitted ingress or decoder step.
    pub fn next_step(&self) -> CursorResult<PrefillStep> {
        self.0.next_step()
    }

    /// Exact remaining range when the next step is the final decoder advance.
    pub fn final_decoder_range(&self) -> CursorResult<Range<usize>> {
        match self.next_step()? {
            PrefillStep::Decoder(range) if range.end == self.token_count() => Ok(range),
            _ => Err(PrefillCursorError::NotFinal),
        }
    }

    /// Retention charged by the cursor's selected assessment.
    pub fn retained_bytes(&self) -> CursorResult<usize> {
        self.0.retained_bytes()
    }
}

fn physical_bytes<C: RetainedPrefillCursor>(cursor: &C) -> CursorResult<usize> {
    let mut values = Vec::new();
    cursor.visit_retained_values(|value| values.push(value));
    let survey = C::Tensor::inspect_storage(&values)?;
    survey.validate(values.len())?;
    if survey.values.iter().any(Option::is_none) {
        return Err(PrefillCursorError::RetentionUnavailable);
    }
    survey.backings.into_iter().try_fold(
        cursor
            .retained_metadata_bytes()
            .ok_or(PrefillCursorError::RetentionUnavailable)?,
        |bytes, backing| {
            backing
                .allocator_capacity_bytes
                .and_then(|capacity| usize::try_from(capacity).ok())
                .and_then(|capacity| bytes.checked_add(capacity))
                .ok_or(PrefillCursorError::RetentionUnavailable)
        },
    )
}

trait Cursor {
    fn clone_box(&self) -> Box<dyn Cursor>;
    fn value_mut(&mut self) -> &mut dyn Any;
    fn decoder_position(&self) -> usize;
    fn token_count(&self) -> usize;
    fn next_step(&self) -> CursorResult<PrefillStep>;
    fn retained_bytes(&self) -> CursorResult<usize>;
}

#[derive(Clone)]
struct TypedCursor<C> {
    cursor: C,
    position: fn(&C) -> usize,
    token_count: fn(&C) -> usize,
    next_step: fn(&C) -> CursorResult<PrefillStep>,
    retained_bytes: fn(&C) -> CursorResult<usize>,
}

impl<C: Clone + 'static> Cursor for TypedCursor<C> {
    fn clone_box(&self) -> Box<dyn Cursor> {
        Box::new(self.clone())
    }
    fn value_mut(&mut self) -> &mut dyn Any {
        &mut self.cursor
    }
    fn decoder_position(&self) -> usize {
        (self.position)(&self.cursor)
    }
    fn token_count(&self) -> usize {
        (self.token_count)(&self.cursor)
    }
    fn next_step(&self) -> CursorResult<PrefillStep> {
        (self.next_step)(&self.cursor)
    }
    fn retained_bytes(&self) -> CursorResult<usize> {
        let overhead = std::mem::size_of::<Self>() - std::mem::size_of::<C>();
        (self.retained_bytes)(&self.cursor)?
            .checked_add(overhead)
            .ok_or(PrefillCursorError::RetentionUnavailable)
    }
}
