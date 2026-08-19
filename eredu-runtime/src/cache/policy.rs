//! Backend-neutral mutable-cache residency policy.

use std::{path::PathBuf, sync::Arc};

use eredu_core::residency::CacheEvictionPolicy;

use super::{CachePoolError, CachePoolLimits, CacheResidencyPool};

/// Selects fully resident mutable state or bounded block residency.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum CacheResidencyPolicy {
    /// Keep state entirely in backend execution memory.
    #[default]
    Device,
    /// Store sealed state in token-addressable blocks under finite budgets.
    Paged(PagedCacheOptions),
}

/// Controls optional disk backing for a live inference cache.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub enum LiveCacheDiskPolicy {
    /// Do not write live mutable state to disk.
    #[default]
    Disabled,
    /// Retain demoted sealed blocks in an explicit ephemeral directory.
    Enabled {
        /// Directory dedicated to this live cache.
        directory: PathBuf,
        /// Finite logical byte limit for live cache files.
        budget_bytes: u64,
        /// Bound on pending reader or writer requests.
        queue_capacity: usize,
    },
}

/// Validated finite limits for block-addressable mutable state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PagedCacheOptions {
    block_size_tokens: i32,
    device_budget_bytes: u64,
    host_budget_bytes: u64,
    recent_device_blocks: usize,
    eviction_policy: CacheEvictionPolicy,
    full_attention: bool,
    retain_discarded_for_persistence: bool,
    live_disk: LiveCacheDiskPolicy,
    sample_process: bool,
    pool: Option<Arc<CacheResidencyPool>>,
}

impl PagedCacheOptions {
    /// Creates paged-state limits. Every memory limit is finite and explicit.
    pub fn new(
        block_size_tokens: i32,
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        recent_device_blocks: usize,
    ) -> Result<Self, CacheResidencyConfigurationError> {
        if block_size_tokens <= 0 {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "cache block size must be positive".into(),
            ));
        }
        if device_budget_bytes == 0 {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "paged cache device budget must be nonzero".into(),
            ));
        }
        if recent_device_blocks == 0 {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "paged cache must protect at least one recent device block".into(),
            ));
        }
        Ok(Self {
            block_size_tokens,
            device_budget_bytes,
            host_budget_bytes,
            recent_device_blocks,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            full_attention: false,
            retain_discarded_for_persistence: false,
            live_disk: LiveCacheDiskPolicy::Disabled,
            sample_process: false,
            pool: None,
        })
    }

    /// Enables exact blockwise full-context attention.
    pub const fn with_full_attention(mut self, enabled: bool) -> Self {
        self.full_attention = enabled;
        self
    }

    /// Retains blocks older than a sliding window solely for later persistence.
    pub const fn with_persistence_retention(mut self, enabled: bool) -> Self {
        self.retain_discarded_for_persistence = enabled;
        self
    }

    /// Selects deterministic block eviction ordering.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Configures explicit live disk backing.
    pub fn with_live_disk(
        mut self,
        directory: impl Into<PathBuf>,
        budget_bytes: u64,
        queue_capacity: usize,
    ) -> Result<Self, CacheResidencyConfigurationError> {
        if budget_bytes == 0 {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "live cache disk budget must be nonzero".into(),
            ));
        }
        if queue_capacity == 0 {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "live cache disk queue capacity must be nonzero".into(),
            ));
        }
        let directory = directory.into();
        if directory.as_os_str().is_empty() {
            return Err(CacheResidencyConfigurationError::InvalidOptions(
                "live cache disk directory must not be empty".into(),
            ));
        }
        self.live_disk = LiveCacheDiskPolicy::Enabled {
            directory,
            budget_bytes,
            queue_capacity,
        };
        Ok(self)
    }

    /// Enables optional process-memory sampling in reports.
    pub const fn with_process_sampling(mut self, enabled: bool) -> Self {
        self.sample_process = enabled;
        self
    }

    /// Attaches this per-cache policy to an aggregate process pool.
    pub fn with_pool(
        mut self,
        pool: CacheResidencyPool,
    ) -> Result<Self, CacheResidencyConfigurationError> {
        if self.device_budget_bytes > pool.limits().device_bytes() {
            return Err(CacheResidencyConfigurationError::InvalidOptions(format!(
                "per-cache device budget {} exceeds cache pool budget {}",
                self.device_budget_bytes,
                pool.limits().device_bytes()
            )));
        }
        if self.host_budget_bytes > pool.limits().host_bytes() {
            return Err(CacheResidencyConfigurationError::InvalidOptions(format!(
                "per-cache host budget {} exceeds cache pool budget {}",
                self.host_budget_bytes,
                pool.limits().host_bytes()
            )));
        }
        if let LiveCacheDiskPolicy::Enabled { budget_bytes, .. } = &self.live_disk {
            if *budget_bytes > pool.limits().disk_bytes() {
                return Err(CacheResidencyConfigurationError::InvalidOptions(format!(
                    "per-cache disk budget {budget_bytes} exceeds cache pool budget {}",
                    pool.limits().disk_bytes()
                )));
            }
        }
        self.pool = Some(Arc::new(pool));
        Ok(self)
    }

    /// Returns the block size in tokens.
    pub const fn block_size_tokens(&self) -> i32 {
        self.block_size_tokens
    }

    /// Returns the finite logical device-cache budget.
    pub const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// Returns the finite physical host-transfer allocation budget.
    pub const fn host_budget_bytes(&self) -> u64 {
        self.host_budget_bytes
    }

    /// Returns the recent block count protected on the execution device per layer.
    pub const fn recent_device_blocks(&self) -> usize {
        self.recent_device_blocks
    }

    /// Returns deterministic block eviction ordering.
    pub const fn eviction_policy(&self) -> CacheEvictionPolicy {
        self.eviction_policy
    }

    /// Returns whether exact blockwise full attention is enabled.
    pub const fn full_attention_enabled(&self) -> bool {
        self.full_attention
    }

    /// Returns whether discarded sliding state is retained for persistence.
    pub const fn retains_discarded_for_persistence(&self) -> bool {
        self.retain_discarded_for_persistence
    }

    /// Returns the live disk policy.
    pub const fn live_disk_policy(&self) -> &LiveCacheDiskPolicy {
        &self.live_disk
    }

    /// Returns whether process-memory sampling is enabled.
    pub const fn process_sampling_enabled(&self) -> bool {
        self.sample_process
    }

    /// Returns the explicitly attached aggregate pool, if any.
    pub fn pool(&self) -> Option<&CacheResidencyPool> {
        self.pool.as_deref()
    }

    /// Creates the default aggregate ownership pool for this policy.
    pub fn create_pool(&self) -> Result<CacheResidencyPool, CacheResidencyConfigurationError> {
        let disk_bytes = match self.live_disk_policy() {
            LiveCacheDiskPolicy::Disabled => 0,
            LiveCacheDiskPolicy::Enabled { budget_bytes, .. } => *budget_bytes,
        };
        let transfer_bytes = self
            .device_budget_bytes()
            .max(self.host_budget_bytes())
            .saturating_mul(2)
            .max(1);
        Ok(CacheResidencyPool::new(CachePoolLimits::new(
            self.device_budget_bytes(),
            self.host_budget_bytes(),
            transfer_bytes,
            disk_bytes,
        )?))
    }
}

/// Invalid backend-neutral mutable-cache residency configuration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CacheResidencyConfigurationError {
    /// Paged options were contradictory or unbounded.
    #[error("invalid paged cache options: {0}")]
    InvalidOptions(String),
    /// Aggregate pool limits were invalid.
    #[error(transparent)]
    Pool(#[from] CachePoolError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_limits_and_live_disk_are_validated_without_a_backend() {
        assert!(PagedCacheOptions::new(0, 1, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 0, 1, 1).is_err());
        assert!(PagedCacheOptions::new(16, 1, 1, 0).is_err());
        let options = PagedCacheOptions::new(16, 8, 0, 1)
            .unwrap()
            .with_live_disk("cache", 32, 2)
            .unwrap();
        assert_eq!(options.create_pool().unwrap().limits().disk_bytes(), 32);
    }

    #[test]
    fn attached_pool_must_cover_every_enabled_residency_tier() {
        let pool = CacheResidencyPool::new(CachePoolLimits::new(8, 4, 8, 0).unwrap());
        let error = PagedCacheOptions::new(16, 16, 4, 1)
            .unwrap()
            .with_pool(pool)
            .unwrap_err();
        assert!(matches!(
            error,
            CacheResidencyConfigurationError::InvalidOptions(_)
        ));
    }
}
