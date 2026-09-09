//! Bounded JSON record accounting shared by every inspected generation path.

use eredu_core::capture::{CaptureBudget, CaptureError};
use serde::{Deserialize, Serialize};

/// Limits on the entire JSON trace, including prompt, decoded text, provenance,
/// capture data, and terminal records. Independent of capture storage/transfer bounds.
/// Measures compact JSON, excluding application framing or additional encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLimits {
    /// Maximum UTF-8 JSON bytes in one record.
    pub per_record_bytes: u64,
    /// Maximum sum of UTF-8 JSON record lengths for the run.
    pub total_bytes: u64,
}

/// Shared compact-JSON transport accounting. This owner is never snapshotted.
pub struct TraceBudget {
    limits: TraceLimits,
    emitted_bytes: u64,
}
impl TraceBudget {
    /// Starts a fresh non-rewindable transport budget.
    pub fn new(limits: TraceLimits) -> Self {
        Self {
            limits,
            emitted_bytes: 0,
        }
    }
    /// Bytes already delivered under this owner.
    pub fn emitted_bytes(&self) -> u64 {
        self.emitted_bytes
    }
    /// Counts compact JSON without allocating an encoded buffer, then charges it.
    pub fn charge(&mut self, record: &impl Serialize) -> Result<(), CaptureError> {
        let remaining = self.limits.total_bytes.saturating_sub(self.emitted_bytes);
        let mut sink = TraceCounter {
            used: 0,
            limit: remaining.min(self.limits.per_record_bytes),
        };
        serde_json::to_writer(&mut sink, record).map_err(|_| CaptureError::Limit {
            budget: CaptureBudget::Encoded,
            cumulative: remaining < self.limits.per_record_bytes,
        })?;
        self.emitted_bytes += sink.used;
        Ok(())
    }
}

struct TraceCounter {
    used: u64,
    limit: u64,
}
impl std::io::Write for TraceCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.used = self
            .used
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| std::io::Error::other("trace byte limit"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
