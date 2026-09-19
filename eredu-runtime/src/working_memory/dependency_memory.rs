//! Admission estimates for host work performed inside stock dependencies.

/// Configurable headroom for dependency work whose internal allocations are
/// not exposed by its public API. The caller supplies a separately enforced
/// input-byte limit and retains the reservation with the result or failure.
///
/// These estimates are planning policy, not measured retained bytes or an
/// enforceable dependency/process memory ceiling. Native tensors, model state,
/// and caller-owned fixed buffers use their own admission contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DependencyMemoryPolicy {
    /// Fixed overhead per independently owned operation/session.
    pub fixed_bytes: usize,
    /// Estimated bytes of allocation per allowed input byte.
    pub bytes_per_input_byte: usize,
}

impl Default for DependencyMemoryPolicy {
    fn default() -> Self {
        Self {
            fixed_bytes: 64 * 1024,
            bytes_per_input_byte: 128,
        }
    }
}

impl DependencyMemoryPolicy {
    /// Checked estimate for one operation or independently mutable session.
    pub fn estimate(self, input_bytes: usize) -> Option<usize> {
        input_bytes
            .checked_mul(self.bytes_per_input_byte)?
            .checked_add(self.fixed_bytes)
    }

    /// Estimates work over a JSON value from its serialized input size, without
    /// retaining a second JSON string or claiming to measure its heap layout.
    pub fn estimate_json(self, value: &serde_json::Value) -> Option<usize> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .ok_or(std::io::ErrorKind::FileTooLarge)?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, value).ok()?;
        self.estimate(count.0)
    }
}
