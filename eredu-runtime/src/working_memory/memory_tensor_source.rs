//! Original encoded payload construction with separately estimated host metadata.

use super::{
    gguf_source::{SourceAccount, SourcePayloadCustody},
    qualified_storage, DependencyMemoryPolicy, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_checkpoint::store::{MemoryTensorBuffer, MemoryTensorBufferError};
use safetensors::tensor::Dtype;
use std::{alloc::Layout, fmt, mem::size_of};

/// Prospective reservation for one newly constructed encoded tensor.
/// Payload capacity is checked separately from estimated metadata headroom.
/// This describes no process-wide ceiling and grants no allocation authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryTensorBufferQuote {
    payload_bytes: u64,
    metadata_estimate_bytes: u64,
    constructor_control_bytes: u64,
    total_bytes: u64,
}
impl MemoryTensorBufferQuote {
    /// Qualified fresh payload capacity, without metadata or allocator overhead.
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    /// Configured estimate for metadata construction and catalog publication.
    /// Later recipe-cache growth and reader/selection controls remain separate.
    pub const fn metadata_estimate_bytes(&self) -> u64 {
        self.metadata_estimate_bytes
    }
    /// Original account allocation and fixed constructor/result controls.
    pub const fn constructor_control_bytes(&self) -> u64 {
        self.constructor_control_bytes
    }
    /// Complete reservation, retained until the last custody alias retires.
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// An admission or construction refusal retaining any actual constructor output.
/// Payload and metadata retire before the original source account is released.
pub struct OriginalMemoryTensorError {
    accounting: Option<WorkingMemoryError>,
    construction: Option<MemoryTensorBufferError>,
    completed: Option<MemoryTensorBuffer>,
    account: Option<SourceAccount>,
}
impl OriginalMemoryTensorError {
    fn refused(cause: WorkingMemoryError) -> Self {
        Self {
            accounting: Some(cause),
            construction: None,
            completed: None,
            account: None,
        }
    }
    /// Typed qualification, admission or settlement refusal, if any.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Actual checkpoint allocation/metadata error, retaining its custody.
    pub fn construction_failure(&self) -> Option<&MemoryTensorBufferError> {
        self.construction.as_ref()
    }
}
impl fmt::Debug for OriginalMemoryTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalMemoryTensorError")
            .field("accounting", &self.accounting)
            .field("construction", &self.construction)
            .field("completed", &self.completed.is_some())
            .field("account", &self.account.is_some())
            .finish()
    }
}
impl fmt::Display for OriginalMemoryTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(cause) = &self.construction {
            cause.fmt(f)
        } else {
            self.accounting.as_ref().expect("buffer refusal").fmt(f)
        }
    }
}
impl std::error::Error for OriginalMemoryTensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.construction {
            Some(cause) => Some(cause),
            None => self.accounting.as_ref().map(|cause| cause as _),
        }
    }
}

impl WorkingMemoryPool {
    /// Prices a fresh payload before construction. Metadata headroom covers
    /// standard-library container/sharing internals and catalog publication as an
    /// explicit estimate. The caller separately limits inputs and funds later
    /// cache entries, selections and readers. No existing payload can be adopted.
    pub fn memory_tensor_buffer_quote(
        name: &str,
        shape: &[usize],
        byte_len: u64,
        metadata_policy: DependencyMemoryPolicy,
    ) -> Result<MemoryTensorBufferQuote, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let payload = usize::try_from(byte_len).map_err(|_| WorkingMemoryError::Overflow)?;
        Layout::array::<u8>(payload).map_err(|_| WorkingMemoryError::Overflow)?;
        let metadata_input = MemoryTensorBuffer::metadata_estimate_input_bytes(name, shape)
            .and_then(|n| n.checked_add(size_of::<SourcePayloadCustody>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let metadata_estimate_bytes = metadata_policy
            .estimate(metadata_input)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = [
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<SourceAccount>(),
            size_of::<SourcePayloadCustody>(),
            size_of::<MemoryTensorBufferQuote>(),
            size_of::<OriginalMemoryTensorError>(),
            size_of::<Result<MemoryTensorBuffer, OriginalMemoryTensorError>>(),
            size_of::<Result<MemoryTensorBuffer, MemoryTensorBufferError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        let account_bytes = SourceAccount::storage_bytes()?;
        let constructor_control_bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_add(account_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        let total_bytes = byte_len
            .checked_add(metadata_estimate_bytes)
            .and_then(|n| n.checked_add(constructor_control_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(MemoryTensorBufferQuote {
            payload_bytes: byte_len,
            metadata_estimate_bytes,
            constructor_control_bytes,
            total_bytes,
        })
    }

    /// Admits and creates one unique writable tensor with runtime-owned custody.
    /// Publication and all payload aliases retain the same original reservation.
    /// The source account settles before returning; native conversion, streams
    /// and working buffers require their own admission after this operation.
    pub fn allocate_memory_tensor_buffer(
        &self,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        byte_len: u64,
        metadata_policy: DependencyMemoryPolicy,
    ) -> Result<MemoryTensorBuffer, OriginalMemoryTensorError> {
        let quote = Self::memory_tensor_buffer_quote(name, shape, byte_len, metadata_policy)
            .map_err(OriginalMemoryTensorError::refused)?;
        let account = self
            .admit_source_compiler(quote.total_bytes)
            .map_err(OriginalMemoryTensorError::refused)?
            .into_source_account();
        let custody = SourcePayloadCustody::new(account.share(), byte_len, byte_len);
        let buffer = match MemoryTensorBuffer::allocate_with_custody(
            name, dtype, shape, byte_len, custody,
        ) {
            Ok(buffer) => buffer,
            Err(cause) => {
                return Err(OriginalMemoryTensorError {
                    accounting: None,
                    construction: Some(cause),
                    completed: None,
                    account: Some(account),
                });
            }
        };
        // Fresh exact-reserve behavior was qualified before allocation. Reject
        // any contradictory observation without publishing or refunding it.
        let actual_bytes = buffer.payload_capacity() as u64;
        let settlement = if actual_bytes != byte_len {
            Err(WorkingMemoryError::StorageCapacityMismatch {
                expected_bytes: byte_len,
                actual_bytes,
            })
        } else {
            account.finish()
        };
        if let Err(cause) = settlement {
            return Err(OriginalMemoryTensorError {
                accounting: Some(cause),
                construction: None,
                completed: Some(buffer),
                account: Some(account),
            });
        }
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests;
