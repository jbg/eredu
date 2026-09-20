//! Original outer source ownership, independent of its nested reader/catalog.
use super::{gguf_source::SourceAccount, qualified_storage, WorkingMemoryError, WorkingMemoryPool};
use eredu_checkpoint::{
    gguf_store::GgufWeightStore,
    store::{CompositeCheckpointSource, RetainedCheckpointSource, SafetensorsWeightStore, SourceErasureStorageRequest},
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
struct ErasureCustody(SourceAccount);

enum Input {
    Safetensors(SafetensorsWeightStore),
    Gguf(GgufWeightStore),
    Composite(CompositeCheckpointSource),
}
impl Input {
    fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Safetensors(source) => pool.validate_safetensors_source_controls(source),
            Self::Gguf(source) => pool.validate_gguf_source_controls(source),
            Self::Composite(source) => pool.validate_gguf_composite_controls(source),
        }
    }
    fn requested(&self) -> Result<u64, WorkingMemoryError> {
        match self {
            Self::Safetensors(_) => WorkingMemoryPool::safetensors_source_erasure_required_bytes(),
            Self::Gguf(_) => WorkingMemoryPool::gguf_source_erasure_required_bytes(),
            Self::Composite(_) => WorkingMemoryPool::gguf_composite_erasure_required_bytes(),
        }
    }
    fn retain(self, custody: ErasureCustody) -> RetainedCheckpointSource {
        match self {
            Self::Safetensors(source) => RetainedCheckpointSource::from_safetensors_with_custody(source, custody),
            Self::Gguf(source) => RetainedCheckpointSource::from_gguf_with_custody(source, custody),
            Self::Composite(source) => {
                RetainedCheckpointSource::from_composite_with_custody(source, custody)
            }
        }
    }
}

/// Exact uncalled source or completed root remains owned through refusal.
/// Nested owners and the outer allocation retire before independent accounting.
pub struct OriginalRetainedSourceError {
    cause: WorkingMemoryError,
    input: Option<Input>,
    completed: Option<RetainedCheckpointSource>,
    account: Option<SourceAccount>,
}
impl OriginalRetainedSourceError {
    /// Typed qualification, source-domain or admission refusal.
    pub fn accounting_failure(&self) -> &WorkingMemoryError {
        &self.cause
    }
    /// Borrow the original SafeTensors source retained by a refused erasure.
    pub fn rejected_safetensors(&self) -> Option<&SafetensorsWeightStore> {
        match self.input.as_ref()? {
            Input::Safetensors(source) => Some(source),
            _ => None,
        }
    }
    /// Borrow the original GGUF input when erasure was not invoked.
    pub fn rejected_gguf(&self) -> Option<&GgufWeightStore> {
        match self.input.as_ref()? {
            Input::Gguf(source) => Some(source),
            _ => None,
        }
    }
    /// Borrow the original composite input when erasure was not invoked.
    pub fn rejected_composite(&self) -> Option<&CompositeCheckpointSource> {
        match self.input.as_ref()? {
            Input::Composite(source) => Some(source),
            _ => None,
        }
    }
}
impl fmt::Debug for OriginalRetainedSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalRetainedSourceError")
            .field("cause", &self.cause)
            .field("input", &self.input.is_some())
            .field("completed", &self.completed.is_some())
            .field("account", &self.account)
            .finish()
    }
}
impl fmt::Display for OriginalRetainedSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalRetainedSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

fn required(request: Option<SourceErasureStorageRequest>) -> Result<u64, WorkingMemoryError> {
    if !qualified_storage::qualified() {
        return Err(WorkingMemoryError::UnknownBound);
    }
    let request = request.ok_or(WorkingMemoryError::Overflow)?;
    let controls = [
        size_of::<SourceErasureStorageRequest>(),
        size_of::<Option<SourceErasureStorageRequest>>(),
        size_of::<Input>(),
        size_of::<super::loaded_decode_source::Allowance>(),
        size_of::<SourceAccount>(),
        size_of::<ErasureCustody>(),
        size_of::<OriginalRetainedSourceError>(),
        size_of::<Result<RetainedCheckpointSource, OriginalRetainedSourceError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        size_of::<u64>(),
        size_of::<Option<u64>>(),
        size_of::<&WorkingMemoryPool>(),
        size_of::<&Input>(),
    ];
    let controls = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .and_then(|n| n.checked_add(request.control_bytes()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    let account = SourceAccount::storage_bytes()?;
    qualified_storage::shared_layout_bytes(request.source_body())?
        .checked_add(qualified_storage::shared_layout_bytes(
            request.custody_body(),
        )?)
        .and_then(|n| n.checked_add(account))
        .and_then(|n| n.checked_add(controls))
        .ok_or(WorkingMemoryError::Overflow)
}

impl WorkingMemoryPool {
    /// Concrete SafeTensors outer allocation, custody/account shells and fixed
    /// transports. Its discovery and header contributions remain independent.
    pub fn safetensors_source_erasure_required_bytes() -> Result<u64, WorkingMemoryError> {
        required(RetainedCheckpointSource::safetensors_storage_request::<ErasureCustody>())
    }
    /// Admit closed outer ownership of this pool's original SafeTensors source.
    /// Ordinary/custom-policy and foreign-pool sources are retained on refusal.
    /// Erasure does not prepare lazy headers, read payloads or reopen artifacts.
    pub fn retain_safetensors_source(
        &self,
        source: SafetensorsWeightStore,
    ) -> Result<RetainedCheckpointSource, OriginalRetainedSourceError> {
        self.retain_source(Input::Safetensors(source))
    }

    /// One exact GGUF outer allocation, control/account shells and named finite
    /// transports. Nested reader/catalog storage has its independent account.
    pub fn gguf_source_erasure_required_bytes() -> Result<u64, WorkingMemoryError> {
        required(RetainedCheckpointSource::gguf_storage_request::<
            ErasureCustody,
        >())
    }
    /// One exact composite outer allocation; legacy child Arcs, recipes and
    /// unconverted architecture/manager owners remain separate contributions.
    pub fn gguf_composite_erasure_required_bytes() -> Result<u64, WorkingMemoryError> {
        required(RetainedCheckpointSource::composite_storage_request::<
            ErasureCustody,
        >())
    }
    /// Admit before creating the private outer allocation. The supplied source
    /// must already carry this pool's original reader-construction custody.
    pub fn retain_gguf_source(
        &self,
        source: GgufWeightStore,
    ) -> Result<RetainedCheckpointSource, OriginalRetainedSourceError> {
        self.retain_source(Input::Gguf(source))
    }
    /// Same constructor path for a genuinely admitted built-in composite.
    pub fn retain_gguf_composite(
        &self,
        source: CompositeCheckpointSource,
    ) -> Result<RetainedCheckpointSource, OriginalRetainedSourceError> {
        self.retain_source(Input::Composite(source))
    }
    fn retain_source(
        &self,
        source: Input,
    ) -> Result<RetainedCheckpointSource, OriginalRetainedSourceError> {
        let refused = |source, cause| OriginalRetainedSourceError {
            cause,
            input: Some(source),
            completed: None,
            account: None,
        };
        if let Err(cause) = source.validate(self) {
            return Err(refused(source, cause));
        }
        let bytes = match source.requested() {
            Ok(bytes) => bytes,
            Err(cause) => return Err(refused(source, cause)),
        };
        let allowance = match self.admit_source_compiler(bytes) {
            Ok(allowance) => allowance,
            Err(cause) => return Err(refused(source, cause)),
        };
        let account = allowance.into_source_account();
        let completed = source.retain(ErasureCustody(account.share()));
        if let Err(cause) = account.finish() {
            return Err(OriginalRetainedSourceError {
                cause,
                input: None,
                completed: Some(completed),
                account: Some(account),
            });
        }
        Ok(completed)
    }
    /// Read-only identity validation. Wrapping an ordinary Arc or supplying a
    /// custom custody value cannot acquire this private original source origin.
    pub fn validate_retained_source_controls(
        &self,
        source: &RetainedCheckpointSource,
    ) -> Result<(), WorkingMemoryError> {
        let Some(custody) = source.constructor_control_owner::<ErasureCustody>() else {
            return self.validate_prepared_safetensors_controls(source);
        };
        if custody.0.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}

#[cfg(test)]
mod tests;
