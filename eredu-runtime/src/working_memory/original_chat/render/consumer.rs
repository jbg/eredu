//! A concrete terminal consumer is funded separately from reusable prompt bytes.
use super::*;

#[derive(Debug)]
struct Payload {
    render: OriginalRenderedChat,
    consumer: GenerationSequenceConsumerLayout,
    allowance: Allowance,
}

/// Originally funded association between one rendering and an exact consumer.
/// This carries source custody and storage evidence, never submission authority.
pub struct OriginalChatConsumer(Option<Arc<Payload>>);
impl OriginalChatConsumer {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live chat consumer")
    }
    /// The original rendering; association creates no prompt or tokenizer copy.
    pub fn render(&self) -> &OriginalRenderedChat {
        &self.payload().render
    }
    /// Exact consumer declaration, not compatibility inferred from byte sizes.
    pub fn accepts_consumer(&self, consumer: &GenerationSequenceConsumerLayout) -> bool {
        self.payload().consumer == *consumer
    }
    /// Actual association construction allowance, separate from render storage.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Authenticate this allowance and its original rendering in the same pool.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if !self.payload().allowance.pool().same_ledger(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.render().validate_pool(pool)
    }
    /// Prospective storage for this exact association and terminal consumer.
    pub fn required_bytes(
        consumer: &GenerationSequenceConsumerLayout,
    ) -> Result<u64, WorkingMemoryError> {
        let parts = [
            consumer.retention_peak_bytes(),
            arc_bytes::<Payload>()?,
            size_of::<Payload>(),
            size_of::<Self>(),
            size_of::<OriginalChatConsumerError>(),
            size_of::<GenerationSequenceConsumerLayout>(),
            size_of::<OriginalRenderedChat>(),
            size_of::<Arc<Payload>>(),
            size_of::<Option<Arc<Payload>>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<Result<Self, OriginalChatConsumerError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<Self>>(),
            size_of::<(&OriginalRenderedChat, GenerationSequenceConsumerLayout)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
}
impl Clone for OriginalChatConsumer {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live chat consumer"),
        )))
    }
}
impl Drop for OriginalChatConsumer {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for OriginalChatConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalChatConsumer")
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}

/// A failed association retains the genuine rendering and any admitted shell.
#[derive(Debug)]
pub struct OriginalChatConsumerError {
    cause: WorkingMemoryError,
    completed: Option<OriginalChatConsumer>,
    render: OriginalRenderedChat,
}
impl OriginalChatConsumerError {
    /// Actual allocation admission or settlement cause.
    pub fn accounting_failure(&self) -> &WorkingMemoryError {
        &self.cause
    }
    /// The genuine rendering retained through this association failure.
    pub fn render(&self) -> &OriginalRenderedChat {
        &self.render
    }
    /// Any completed association remains charged until this failure retires.
    pub fn retained_bytes(&self) -> u64 {
        self.completed
            .as_ref()
            .map_or(0, OriginalChatConsumer::original_bytes)
    }
}
impl fmt::Display for OriginalChatConsumerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for OriginalChatConsumerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl OriginalRenderedChat {
    /// Fund a concrete consumer while retaining these same immutable prompt bytes.
    /// Another consumer requires its own allowance; no existing allowance is
    /// refunded, transferred or reinterpreted by this operation.
    pub fn bind_consumer(
        &self,
        consumer: GenerationSequenceConsumerLayout,
    ) -> Result<OriginalChatConsumer, OriginalChatConsumerError> {
        let retain = |cause, completed| OriginalChatConsumerError {
            cause,
            completed,
            render: self.clone(),
        };
        let bytes =
            OriginalChatConsumer::required_bytes(&consumer).map_err(|cause| retain(cause, None))?;
        let allowance = self
            .payload()
            .allowance
            .pool()
            .admit_source_compiler(bytes)
            .map_err(|cause| retain(cause, None))?;
        let mut owner = Arc::new(Payload {
            render: self.clone(),
            consumer,
            allowance,
        });
        match Arc::get_mut(&mut owner)
            .expect("unpublished chat consumer")
            .allowance
            .end_compilation()
        {
            Ok(()) => Ok(OriginalChatConsumer(Some(owner))),
            Err(cause) => Err(retain(cause, Some(OriginalChatConsumer(Some(owner))))),
        }
    }
}
