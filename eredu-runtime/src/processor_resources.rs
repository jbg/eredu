//! Optional coarse limits for portable media request preparation.

/// Explicit caller ceilings. Zero denies the corresponding resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorRequestBudget {
    /// Caller-owned decoded pixels and original token buffers retained by the request.
    pub decoded_input_bytes: u64,
    /// Coarse media/token buffer estimate, including outputs and preprocessing copies.
    /// Native allocations, validation metadata and tokenizer workspace are excluded.
    pub host_buffer_bytes: u64,
    /// Logical bytes supplied to tensor constructors; not a native allocation bound.
    pub output_tensor_bytes: u64,
    /// Coarse request-complexity limit on plan entries and retained token IDs.
    pub planning_items: u64,
    /// Decoder positions including media placeholders and framing.
    pub decoder_positions: u64,
}

/// Request-specific estimates for the selected preprocessing operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessorRequestResources {
    /// Bytes of caller-owned decoded buffers (not charged again to host scratch).
    pub decoded_input_bytes: u64,
    /// Coarse estimate for media/token buffers, excluding native allocations.
    pub host_buffer_bytes: u64,
    /// Total logical host outputs passed to native construction.
    pub output_tensor_bytes: u64,
    /// Coarse plan-entry and retained-token count; not an allocator byte estimate.
    pub planning_items: u64,
    /// Exact decoder sequence length after tokenization and media expansion.
    pub decoder_positions: u64,
}

/// Typed failure before the rejected preparation stage allocates data buffers.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessorResourceError {
    /// Checked shape arithmetic overflowed.
    #[error("processor resource arithmetic overflow")]
    Overflow,
    /// A declared ceiling was exceeded.
    #[error("processor {resource} requires {required}, exceeding budget {limit}")]
    Exhausted {
        /// Resource whose ceiling was exceeded.
        resource: &'static str,
        /// Required quantity.
        required: u64,
        /// Admitted ceiling.
        limit: u64,
    },
    /// No complete buffer plan exists for the selected processor mechanism.
    #[error("processor buffer planning is unavailable for this mechanism")]
    Unavailable,
}
impl ProcessorRequestBudget {
    /// Admits one derived report without changing it or allocating execution resources.
    pub fn admit(
        &self,
        required: &ProcessorRequestResources,
    ) -> Result<(), ProcessorResourceError> {
        for (resource, required, limit) in [
            (
                "decoded input bytes",
                required.decoded_input_bytes,
                self.decoded_input_bytes,
            ),
            (
                "host buffer bytes",
                required.host_buffer_bytes,
                self.host_buffer_bytes,
            ),
            (
                "output tensor bytes",
                required.output_tensor_bytes,
                self.output_tensor_bytes,
            ),
            (
                "planning items",
                required.planning_items,
                self.planning_items,
            ),
            (
                "decoder positions",
                required.decoder_positions,
                self.decoder_positions,
            ),
        ] {
            if required > limit {
                return Err(ProcessorResourceError::Exhausted {
                    resource,
                    required,
                    limit,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_buffer_ceilings_are_inclusive_and_fail_with_quantities() {
        let report = ProcessorRequestResources {
            decoded_input_bytes: 70,
            host_buffer_bytes: 200,
            output_tensor_bytes: 80,
            planning_items: 12,
            decoder_positions: 9,
        };
        let budget = ProcessorRequestBudget {
            decoded_input_bytes: 70,
            host_buffer_bytes: 200,
            output_tensor_bytes: 80,
            planning_items: 12,
            decoder_positions: 9,
        };
        budget.admit(&report).unwrap();
        let mut limited = budget;
        limited.planning_items = 11;
        assert_eq!(
            limited.admit(&report),
            Err(ProcessorResourceError::Exhausted {
                resource: "planning items",
                required: 12,
                limit: 11
            })
        );
        let mut limited = budget;
        limited.host_buffer_bytes = 199;
        assert_eq!(
            limited.admit(&report),
            Err(ProcessorResourceError::Exhausted {
                resource: "host buffer bytes",
                required: 200,
                limit: 199
            })
        );
    }
}
