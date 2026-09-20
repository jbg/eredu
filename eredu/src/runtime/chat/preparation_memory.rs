//! Admission for facade-owned preparation buffers and estimated dependency work.
//! No dependency allocator is intercepted. Retaining this account keeps admitted
//! work charged until the last source, result or error owner retires.
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::DependencyMemoryPolicy;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

#[derive(Clone, Debug, Default)]
pub(crate) struct PreparationFunding {
    account: Option<HostMetadataFunding>,
    memory: DependencyMemoryPolicy,
}
impl PreparationFunding {
    /// Ordinary host preparation without a finite runtime account.
    pub(crate) fn unmanaged() -> Self {
        Self::default()
    }

    /// Borrow the existing neutral account; no callback shell or new account.
    pub(crate) fn from_metadata(account: &HostMetadataFunding) -> Self {
        Self {
            account: Some(account.clone()),
            memory: DependencyMemoryPolicy::default(),
        }
    }
    pub(crate) fn with_memory_policy(mut self, memory: DependencyMemoryPolicy) -> Self {
        self.memory = memory;
        self
    }
    pub(crate) fn memory_policy(&self) -> DependencyMemoryPolicy {
        self.memory
    }

    /// Cumulative admission before an Eredu-owned producer. This is a requested
    /// host footprint, excluding allocator bookkeeping and process residency.
    pub(crate) fn reserve(&self, bytes: usize) -> Result<(), PreparationFailure> {
        if let Some(account) = &self.account {
            account
                .reserve_metadata(bytes)
                .map_err(|cause| PreparationFailure {
                    cause,
                    funding: self.clone(),
                })?;
        }
        Ok(())
    }
    /// Separate headroom for stock dependency work. It does not enforce an
    /// allocator ceiling or observe dependency-private capacities.
    pub(crate) fn reserve_dependency(
        &self,
        input_bytes: usize,
    ) -> Result<usize, PreparationFailure> {
        let bytes = self
            .memory
            .estimate(input_bytes)
            .ok_or_else(|| PreparationFailure {
                cause: HostMetadataFundingError::Overflow,
                funding: self.clone(),
            })?;
        self.reserve(bytes)?;
        Ok(bytes)
    }
    pub(crate) fn storage_overflow(&self) -> StorageFailure {
        self.storage_failure(StorageCause::Overflow)
    }
    fn storage_failure(&self, cause: impl Into<StorageCause>) -> StorageFailure {
        StorageFailure {
            cause: cause.into(),
            funding: self.clone(),
        }
    }
    pub(crate) fn allocation_failure(&self, cause: TryReserveError) -> StorageFailure {
        self.storage_failure(cause)
    }
    /// Pay the full new requested allocation while the old buffer remains paid.
    /// The account is cumulative; replacing a buffer does not refund its history.
    pub(crate) fn try_grow_vec<T>(
        &self,
        values: &mut Vec<T>,
        required: usize,
    ) -> Result<(), StorageFailure> {
        if required <= values.capacity() {
            return Ok(());
        }
        let capacity = required
            .max(
                values
                    .capacity()
                    .checked_mul(2)
                    .ok_or_else(|| self.storage_overflow())?,
            )
            .max(8);
        let bytes = Layout::array::<T>(capacity)
            .map_err(|_| self.storage_overflow())?
            .size();
        self.reserve(bytes)
            .map_err(|error| self.storage_failure(error))?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|error| self.storage_failure(error))
    }
    pub(crate) fn try_push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), StorageFailure> {
        let next = values
            .len()
            .checked_add(1)
            .ok_or_else(|| self.storage_overflow())?;
        self.try_grow_vec(values, next)?;
        values.push(value);
        Ok(())
    }
    pub(crate) fn try_extend_copy<T: Copy>(
        &self,
        values: &mut Vec<T>,
        added: &[T],
    ) -> Result<(), StorageFailure> {
        let next = values
            .len()
            .checked_add(added.len())
            .ok_or_else(|| self.storage_overflow())?;
        self.try_grow_vec(values, next)?;
        values.extend_from_slice(added);
        Ok(())
    }
    pub(crate) fn try_copy_str(&self, input: &str) -> Result<String, StorageFailure> {
        self.reserve(input.len())
            .map_err(|error| self.storage_failure(error))?;
        let mut output = String::new();
        output
            .try_reserve_exact(input.len())
            .map_err(|error| self.storage_failure(error))?;
        output.push_str(input);
        Ok(output)
    }
    pub(crate) fn try_format(&self, args: fmt::Arguments<'_>) -> Result<String, StorageFailure> {
        use fmt::Write;
        struct Count(usize);
        impl Write for Count {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                self.0 = self.0.checked_add(text.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        let mut count = Count(0);
        count.write_fmt(args).map_err(|_| self.storage_overflow())?;
        self.reserve(
            count
                .0
                .checked_add(size_of::<String>())
                .ok_or_else(|| self.storage_overflow())?,
        )
        .map_err(|error| self.storage_failure(error))?;
        struct Destination {
            text: String,
            limit: usize,
        }
        impl Write for Destination {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                if text.len() > self.limit - self.text.len() {
                    return Err(fmt::Error);
                }
                self.text.push_str(text);
                Ok(())
            }
        }
        let mut output = Destination {
            text: String::new(),
            limit: count.0,
        };
        output
            .text
            .try_reserve_exact(count.0)
            .map_err(|error| self.storage_failure(error))?;
        output
            .write_fmt(args)
            .map_err(|_| self.storage_failure(StorageCause::Formatting))?;
        if output.text.len() != count.0 {
            return Err(self.storage_failure(StorageCause::Formatting));
        }
        Ok(output.text)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PreparationFailure {
    #[source]
    cause: HostMetadataFundingError,
    funding: PreparationFunding,
}
#[derive(Debug, thiserror::Error)]
enum StorageCause {
    #[error("preparation buffer extent overflow")]
    Overflow,
    #[error("formatted preparation text changed between sizing and writing")]
    Formatting,
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error(transparent)]
    Allocation(#[from] TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct StorageFailure {
    #[source]
    cause: StorageCause,
    funding: PreparationFunding,
}

/// Test accounts exercise the same neutral funding boundary as runtime callers.
#[cfg(test)]
pub(crate) fn test_funding(
    reserve: impl Fn(usize) -> Result<(), HostMetadataFundingError> + Send + Sync + 'static,
) -> Result<PreparationFunding, HostMetadataFundingError> {
    struct Account<F>(F);
    impl<F> fmt::Debug for Account<F> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("TestAccount")
        }
    }
    impl<F: Fn(usize) -> Result<(), HostMetadataFundingError> + Send + Sync + 'static>
        eredu_core::HostMetadataAccount for Account<F>
    {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            (self.0)(bytes)
        }
    }
    HostMetadataFunding::new(Account(reserve))
        .map(|account| PreparationFunding::from_metadata(&account))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[test]
    fn failed_growth_preserves_existing_buffer_and_cumulative_account() {
        let stop = Arc::new(AtomicBool::new(false));
        let paid = Arc::new(AtomicUsize::new(0));
        let funding = test_funding({
            let stop = stop.clone();
            let paid = paid.clone();
            move |bytes| {
                if stop.load(Ordering::SeqCst) {
                    return Err(HostMetadataFundingError::Capacity {
                        required: bytes as u64,
                        available: 0,
                    });
                }
                paid.fetch_add(bytes, Ordering::SeqCst);
                Ok(())
            }
        })
        .unwrap();
        let mut values = Vec::new();
        funding
            .try_extend_copy(&mut values, &[17u32, 29, 43])
            .unwrap();
        let capacity = values.capacity();
        let before = paid.load(Ordering::SeqCst);
        stop.store(true, Ordering::SeqCst);
        let error = funding.try_grow_vec(&mut values, capacity + 1).unwrap_err();
        assert_eq!(values, [17, 29, 43]);
        assert_eq!(values.capacity(), capacity);
        assert_eq!(paid.load(Ordering::SeqCst), before);
        assert!(matches!(error.cause, StorageCause::Funding(_)));
        // A funded buffer can be appended to without another growth request.
        funding.try_push(&mut values, 59).unwrap();
        assert_eq!(values, [17, 29, 43, 59]);
    }

    #[test]
    fn formatting_rejects_changed_extent_without_unfunded_growth() {
        struct Changing(std::cell::Cell<bool>);
        impl fmt::Display for Changing {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(if self.0.replace(true) { "longer" } else { "x" })
            }
        }
        let error = PreparationFunding::unmanaged()
            .try_format(format_args!("{}", Changing(std::cell::Cell::new(false))))
            .unwrap_err();
        assert!(matches!(error.cause, StorageCause::Formatting));
    }
}
