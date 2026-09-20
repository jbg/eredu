//! Runtime policy for lazy checkpoint headers; metadata remains an estimate.
use super::{
    DependencyMemoryPolicy, WorkingMemoryError, WorkingMemoryPool, gguf_source::SourceAccount,
    qualified_storage,
};
use eredu_checkpoint::{
    safetensors::{
        SafetensorsHeaderAdmission, SafetensorsHeaderFailure, SafetensorsHeaderRequest,
        SafetensorsHeaderReservation,
    },
    store::StoreError,
};
use std::{
    error::Error,
    fmt,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Separate encoded storage, metadata headroom and fixed reservation controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetensorsHeaderQuote {
    encoded_buffer_bytes: u64,
    metadata_estimate_bytes: u64,
    reservation_control_bytes: u64,
    total_bytes: u64,
}
impl SafetensorsHeaderQuote {
    /// Fresh encoded header buffer, including its eight-byte prefix.
    pub fn encoded_buffer_bytes(self) -> u64 {
        self.encoded_buffer_bytes
    }
    /// Configurable estimate for decoded metadata, paths and dependency work.
    pub fn metadata_estimate_bytes(self) -> u64 {
        self.metadata_estimate_bytes
    }
    /// Qualified shared account, custody and potential error wrapper storage.
    pub fn reservation_control_bytes(self) -> u64 {
        self.reservation_control_bytes
    }
    /// Contribution retained until the last header or failure alias retires.
    pub fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Refusal retaining the policy's prepaid diagnostic storage.
#[derive(Debug)]
pub struct SafetensorsHeaderPolicyError {
    memory: Option<WorkingMemoryError>,
    maximum_headers: Option<usize>,
    _account: SourceAccount,
}
impl SafetensorsHeaderPolicyError {
    /// Original pool, qualification or arithmetic failure.
    pub fn memory_failure(&self) -> Option<&WorkingMemoryError> {
        self.memory.as_ref()
    }
    /// Exhausted lifetime initialization allowance, when this is a limit refusal.
    pub fn maximum_headers(&self) -> Option<usize> {
        self.maximum_headers
    }
}
impl fmt::Display for SafetensorsHeaderPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = &self.memory {
            error.fmt(f)
        } else {
            write!(
                f,
                "SafeTensors header initialization limit {} exhausted",
                self.maximum_headers.unwrap()
            )
        }
    }
}
impl Error for SafetensorsHeaderPolicyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.memory.as_ref().map(|error| error as _)
    }
}
#[derive(Debug)]
struct HeaderPolicy {
    pool: Weak<super::Pool>,
    metadata: DependencyMemoryPolicy,
    maximum_headers: usize,
    attempted: AtomicUsize,
    exhausted: Arc<SafetensorsHeaderFailure>,
    account: SourceAccount,
}
impl SafetensorsHeaderAdmission for HeaderPolicy {
    fn reserve(
        &self,
        request: SafetensorsHeaderRequest,
    ) -> Result<SafetensorsHeaderReservation, Arc<SafetensorsHeaderFailure>> {
        // Every admitted attempt owns one prepaid refusal slot. Calls beyond the
        // finite source inventory reuse the same failure without allocating.
        if self
            .attempted
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n < self.maximum_headers {
                    n.checked_add(1)
                } else {
                    None
                }
            })
            .is_err()
        {
            return Err(self.exhausted.clone());
        }
        let reserve = || {
            let quote = WorkingMemoryPool::safetensors_header_quote(request, self.metadata)?;
            let pool = WorkingMemoryPool(
                self.pool
                    .upgrade()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            );
            let account = pool
                .admit_source_compiler(quote.total_bytes())?
                .into_source_account();
            Ok(SafetensorsHeaderReservation::with_completion(
                account,
                |account| {
                    account
                        .finish()
                        .map_err(|error| Arc::new(error) as Arc<dyn Error + Send + Sync>)
                },
            ))
        };
        reserve().map_err(|memory| {
            Arc::new(SafetensorsHeaderFailure::refused(Arc::new(
                SafetensorsHeaderPolicyError {
                    memory: Some(memory),
                    maximum_headers: None,
                    _account: self.account.share(),
                },
            )))
        })
    }
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, WorkingMemoryError> {
    values
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
}
impl WorkingMemoryPool {
    /// Price a header before reading its body. Metadata headroom is applied to
    /// JSON and path bytes; it is not a dependency-wide or process-wide ceiling.
    /// Source discovery, index/catalog storage and payload caches are separate.
    pub fn safetensors_header_quote(
        request: SafetensorsHeaderRequest,
        metadata: DependencyMemoryPolicy,
    ) -> Result<SafetensorsHeaderQuote, WorkingMemoryError> {
        let buffer = request
            .json_bytes
            .checked_add(8)
            .ok_or(WorkingMemoryError::Overflow)?;
        if request.buffer_bytes != buffer {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let input = request
            .json_bytes
            .checked_add(request.path_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let metadata_estimate_bytes = metadata
            .estimate(input)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let encoded_buffer_bytes =
            u64::try_from(buffer).map_err(|_| WorkingMemoryError::Overflow)?;
        let reservation_control_bytes = sum([
            SourceAccount::storage_bytes()?,
            qualified_storage::shared_layout_bytes(
                SafetensorsHeaderReservation::custody_layout::<SourceAccount>(),
            )?,
            qualified_storage::shared_bytes::<SafetensorsHeaderFailure>()?,
            qualified_storage::shared_bytes::<StoreError>()?,
            qualified_storage::shared_bytes::<WorkingMemoryError>()?,
        ])?;
        let total_bytes = sum([
            encoded_buffer_bytes,
            metadata_estimate_bytes,
            reservation_control_bytes,
        ])?;
        Ok(SafetensorsHeaderQuote {
            encoded_buffer_bytes,
            metadata_estimate_bytes,
            reservation_control_bytes,
            total_bytes,
        })
    }

    /// Fixed policy/account storage plus one diagnostic pair per allowed attempt
    /// and one shared exhausted-limit diagnostic. The limit counts lifetime
    /// initializations, including refusals, rather than simultaneous parses.
    pub fn safetensors_header_policy_required_bytes(
        maximum_headers: usize,
    ) -> Result<u64, WorkingMemoryError> {
        let failures = u64::try_from(maximum_headers)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(WorkingMemoryError::Overflow)?;
        let diagnostic = sum([
            qualified_storage::shared_bytes::<SafetensorsHeaderFailure>()?,
            qualified_storage::shared_bytes::<SafetensorsHeaderPolicyError>()?,
        ])?;
        sum([
            SourceAccount::storage_bytes()?,
            qualified_storage::shared_bytes::<HeaderPolicy>()?,
            diagnostic
                .checked_mul(failures)
                .ok_or(WorkingMemoryError::Overflow)?,
        ])
    }

    /// Create header admission in this pool for a finite source inventory.
    /// Set `maximum_headers` to cover all independent lazy header initializations
    /// using this policy. Aliases of an initialized header do not consume slots.
    /// Refusal storage is prepaid; accepted header bytes are charged on demand.
    /// Neither this policy nor retained errors keep the pool alive by themselves.
    pub fn safetensors_header_admission(
        &self,
        maximum_headers: usize,
        metadata: DependencyMemoryPolicy,
    ) -> Result<Arc<dyn SafetensorsHeaderAdmission>, WorkingMemoryError> {
        let bytes = Self::safetensors_header_policy_required_bytes(maximum_headers)?;
        let account = self.admit_source_compiler(bytes)?.into_source_account();
        let exhausted = Arc::new(SafetensorsHeaderFailure::refused(Arc::new(
            SafetensorsHeaderPolicyError {
                memory: None,
                maximum_headers: Some(maximum_headers),
                _account: account.share(),
            },
        )));
        let policy = Arc::new(HeaderPolicy {
            pool: Arc::downgrade(&self.0),
            metadata,
            maximum_headers,
            attempted: AtomicUsize::new(0),
            exhausted,
            account,
        });
        policy.account.finish()?;
        Ok(policy)
    }
}

#[cfg(test)]
mod tests;
