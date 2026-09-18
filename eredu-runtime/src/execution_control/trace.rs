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
    /// Exact controls of the shared allocation-free compact-JSON counter.
    /// The caller pays these before counting its closed, funded record type.
    #[doc(hidden)]
    pub fn counting_control_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<TraceCounter>(),
            std::mem::size_of::<serde_json::Serializer<&mut TraceCounter>>(),
            std::mem::size_of::<Result<(), serde_json::Error>>(),
            std::mem::size_of::<Result<(), CaptureError>>(),
            std::mem::size_of::<(u64, u64, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
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
            exceeded: false,
        };
        serde_json::to_writer(&mut sink, record).map_err(|_| CaptureError::Limit {
            budget: CaptureBudget::Encoded,
            cumulative: remaining < self.limits.per_record_bytes,
        })?;
        if sink.exceeded {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Encoded,
                cumulative: remaining < self.limits.per_record_bytes,
            });
        }
        self.emitted_bytes += sink.used;
        Ok(())
    }
}

struct TraceCounter {
    used: u64,
    limit: u64,
    exceeded: bool,
}
impl std::io::Write for TraceCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = u64::try_from(bytes.len())
            .ok()
            .and_then(|n| self.used.checked_add(n));
        // Finish the same counting pass without allocating an I/O or serde
        // error at a limit boundary. The fixed transport refusal follows below.
        self.exceeded |= next.is_none_or(|n| n > self.limit);
        self.used = next.unwrap_or(u64::MAX);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trace_matches_wire_limits_and_refusal_keeps_consumption() {
        #[derive(Serialize)]
        struct Record<'a> {
            text: &'a str,
            tokens: &'a [u32],
        }
        let record = Record {
            text: "quoted \" and \n λ",
            tokens: &[1, 29, 400],
        };
        let bytes = serde_json::to_vec(&record).unwrap().len() as u64;
        let limits = TraceLimits {
            per_record_bytes: bytes,
            total_bytes: 2 * bytes - 1,
        };
        let mut ordinary = TraceBudget::new(limits);
        ordinary.charge(&record).unwrap();
        assert_eq!(ordinary.emitted_bytes(), bytes);
        for result in [ordinary.charge(&record), ordinary.charge(&record)] {
            assert!(matches!(
                result,
                Err(CaptureError::Limit {
                    budget: CaptureBudget::Encoded,
                    cumulative: true
                })
            ));
        }
        assert_eq!(ordinary.emitted_bytes(), bytes);
        let mut small = TraceBudget::new(TraceLimits {
            per_record_bytes: bytes - 1,
            total_bytes: u64::MAX,
        });
        assert!(matches!(
            small.charge(&record),
            Err(CaptureError::Limit {
                budget: CaptureBudget::Encoded,
                cumulative: false
            })
        ));
        assert_eq!(small.emitted_bytes(), 0);
    }
}
