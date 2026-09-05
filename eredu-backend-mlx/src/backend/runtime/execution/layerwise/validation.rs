//! Source-consumption and residency-budget validation.

use super::*;

/// Rejects checkpoint tensors that were neither consumed nor explicitly ignored.
pub fn validate_unused<F>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    consumed: &BTreeSet<String>,
    ignored: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    let unused = store
        .source_keys()
        .into_iter()
        .filter(|key| !consumed.contains(key))
        .filter(|key| !ignored(key))
        .collect::<Vec<_>>();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(LayerwiseModelError::UnexpectedCheckpointParameters { unused }.into())
    }
}

#[cfg(test)]
mod validate_unused_tests {
    use super::*;
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    };

    struct ResolvedTestSource;

    impl CheckpointSource for ResolvedTestSource {
        fn source_keys(&self) -> Vec<String> {
            vec!["claimed.weight".into()]
        }

        fn source_metadata(&self, _key: &str) -> Result<TensorMetadata, StoreError> {
            unreachable!("unused-key validation does not read tensor metadata")
        }

        fn acquire_lease(
            &self,
            _request: TensorReadRequest,
        ) -> Result<CheckpointLease, StoreError> {
            unreachable!("unused-key validation does not acquire tensor payloads")
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            unreachable!("unused-key validation does not inspect storage diagnostics")
        }

        fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
            vec!["schema.allowed.extra".into()]
        }

        fn is_checkpoint_contract_resolved(&self) -> bool {
            true
        }
    }

    #[test]
    fn schema_unclaimed_keys_are_not_backend_unused_parameters() {
        let consumed = BTreeSet::from(["claimed.weight".into()]);
        validate_unused(&ResolvedTestSource, &consumed, |_| false).unwrap();
    }
}

/// Validates that required host storage fits the configured offload budget.
pub fn validate_host_budget(config: OffloadConfig, required: u64) -> Result<(), Error> {
    if let Some(budget) = config.host_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::HostBudgetTooSmall { required, budget }.into());
        }
    }
    Ok(())
}

/// Validates that static and window storage fit the configured device budget.
pub fn validate_device_budget(
    config: OffloadConfig,
    static_bytes: u64,
    window_bytes: u64,
    depth: usize,
) -> Result<(), Error> {
    let required =
        static_bytes
            .checked_add(window_bytes)
            .ok_or(LayerwiseModelError::ArithmeticOverflow {
                context: "static plus device-window byte total",
            })?;
    if let Some(budget) = config.device_budget_bytes() {
        if required > budget {
            return Err(LayerwiseModelError::DeviceBudgetTooSmall {
                static_bytes,
                window_bytes,
                depth,
                required,
                budget,
            }
            .into());
        }
    }
    Ok(())
}

/// Structured failures produced by the generic layerwise execution engine.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseModelError {
    /// A non-alias binding omitted its architecture-owned parameter identity.
    #[error("parallel binding {binding:?} has no architecture-logical target")]
    MissingParallelBindingTarget {
        /// Resident-unit binding name.
        binding: String,
    },
    /// A binding's exact architecture-owned identity was absent from the local layout.
    #[error("parallel binding {binding:?} targets unknown architecture parameter {target:?}")]
    UnknownParallelBindingTarget {
        /// Resident-unit binding name.
        binding: String,
        /// Exact architecture-logical parameter identity.
        target: String,
    },
    /// A multi-input group did not define how to combine its dependencies.
    #[error(
        "execution group slot {group} has {inputs} dependency outputs but no merge implementation"
    )]
    UnmergedExecutionGroupInputs {
        /// Architecture group slot.
        group: usize,
        /// Number of ready dependency outputs.
        inputs: usize,
    },
    /// Adapter and configured execution-group counts differ.
    #[error("adapter declares {adapter} execution groups but {configured} were configured")]
    ExecutionGroupCount {
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// Adapter and configured group identities differ at one stable slot.
    #[error("execution group slot {slot} is {configured:?}, expected adapter group {adapter:?}")]
    ExecutionGroupIdentity {
        /// Architecture group slot.
        slot: usize,
        /// Adapter-declared identity.
        adapter: String,
        /// Configured residency identity.
        configured: String,
    },
    /// Adapter and configured unit counts differ for one execution group.
    #[error("execution group {group:?} has {configured} configured units but adapter declares {adapter}")]
    ExecutionGroupLength {
        /// Group id.
        group: String,
        /// Adapter-declared count.
        adapter: usize,
        /// Configured count.
        configured: usize,
    },
    /// A requested execution group does not exist.
    #[error("unknown resident execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// The configured ordered layer window was invalid.
    #[error("device layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Decoder layer count.
        layer_count: usize,
    },
    /// The protected host lookahead exceeds an execution group.
    #[error("host layer window depth {depth} must be between 1 and layer count {layer_count}")]
    InvalidHostLayerWindow {
        /// Requested depth.
        depth: usize,
        /// Available ordered units.
        layer_count: usize,
    },
    /// A dense transfer window contained an invalid or unordered unit index.
    #[error(
        "dense transfer window index {index} is out of order or outside {unit_count} planned units"
    )]
    InvalidDenseTransferWindow {
        /// Invalid unit index.
        index: usize,
        /// Available units.
        unit_count: usize,
    },
    /// Strict loading found unrelated checkpoint tensors.
    #[error("strict layerwise loading found unexpected checkpoint parameters: {unused:?}")]
    UnexpectedCheckpointParameters {
        /// Unexpected keys in stable order.
        unused: Vec<String>,
    },
    /// The host cannot retain the required decoder-layer allocation capacity.
    #[error("host budget {budget} bytes cannot contain {required} bytes of decoder host-transfer allocations")]
    HostBudgetTooSmall {
        /// Required charged host-allocation bytes.
        required: u64,
        /// Configured host budget.
        budget: u64,
    },
    /// The device cannot contain static weights plus the configured window.
    #[error("device budget {budget} bytes cannot contain {static_bytes} static bytes plus the depth-{depth} layer window ({window_bytes} bytes, {required} total)")]
    DeviceBudgetTooSmall {
        /// Pinned static device bytes.
        static_bytes: u64,
        /// Largest consecutive window bytes.
        window_bytes: u64,
        /// Configured layer count.
        depth: usize,
        /// Total required parameter bytes.
        required: u64,
        /// Configured device budget.
        budget: u64,
    },
    /// A cache vector had the wrong number of layers.
    #[error("layerwise cache has {actual} layers, expected {expected}")]
    CacheLengthMismatch {
        /// Model decoder count.
        expected: usize,
        /// Supplied cache count.
        actual: usize,
    },
    /// A cache entry was absent.
    #[error("layerwise cache entry {index} is missing")]
    MissingLayerCache {
        /// Missing decoder index.
        index: usize,
    },
    /// Checked byte or index arithmetic overflowed.
    #[error("layerwise model arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Module checkpoint binding failed.
    #[error(transparent)]
    ModuleBinding(#[from] ModuleBindingError),
    /// Residency execution failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}
