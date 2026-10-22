//! MLX residency for independently addressable parameter-bank entries.
//!
//! Each logical entry is an atomic disk-planned residency unit. Selection ids are
//! inspected once per grouped block, validated before acquisition, coalesced in
//! deterministic global-id order, and rewritten to a temporary compact bank.

use eredu_checkpoint::WeightQuantization;
#[cfg(test)]
use eredu_checkpoint::store::TensorSelection;
use eredu_nn::GroupedNeuralBackend;
pub use eredu_runtime::{AddressableBankCatalog, AddressableBankEntry};
#[cfg(test)]
use eredu_runtime::OffloadUnit;
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, ParameterBankAccess, ParameterBankAcquisition,
    ResidencyReport, WeightBinding, WeightMaterializationReport,
};

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, segment_sum},
    transforms::eval,
    Array, Dtype, Stream,
};

#[cfg(test)]
use crate::backend::runtime::residency::manager::ResidentUnitLease;
use crate::MlxTensor;
use crate::{
    backend::error::Error,
    backend::nn::shared::MlxNeuralBackend,
    backend::runtime::checkpoint::bounded_quantization::prepare_quantized_source,
    backend::runtime::residency::manager::{ResidencyError, ResidencyManager, ResidentTransfer},
};
#[cfg(test)]
use eredu_core::residency::{OffloadPlan, ResidencyPolicy};
use eredu_core::residency::{MemoryTier, OffloadConfig, OffloadUnitId, ResidencyLedgerError};

mod catalog;
#[cfg(test)]
pub(crate) use catalog::quantize_entry_catalog;
#[allow(unused_imports)]
pub(crate) use catalog::QuantizedParameterBankCatalog;
pub use catalog::{
    entries_from_selected_members, BankAccessClass, ParameterBankKey, ParameterBankOptions,
    ParameterBankOptionsError,
};
use catalog::quantize_selected_entry_catalog;

mod telemetry;
use telemetry::ParameterBankStatistics;
pub use telemetry::{
    BankPassStatistics, BankTierStatistics, ParameterBankResidencyReport,
    ParameterBanksResidencyReport, RowLookupPoolReport,
};

mod acquisition;
mod parameters;
pub use acquisition::{
    AcquiredParameters, AddressableParameterBank, SharedAddressableParameterBank,
};
pub(crate) use parameters::publish_bank_parameter_replacements;

mod movement;
pub use movement::MlxIndexedMovement;

/// Structured sparse entry cache failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AddressableParameterBankError {
    /// Portable demand ownership, use counts or compact resource admission failed.
    #[error(transparent)]
    Demand(#[from] eredu_runtime::AddressableBankDemandError),
    /// Grouped-entry placement controls were invalid.
    #[error(transparent)]
    Policy(#[from] ParameterBankOptionsError),
    /// Bounded entry materialisation failed before cache construction.
    #[error("bounded entry materialisation failed: {source}")]
    Transformation {
        /// Shared checkpoint transformation failure.
        #[source]
        source: Box<Error>,
    },
    /// Independent member ownership or combined residency budgets are invalid.
    #[error(transparent)]
    Preparation(#[from] eredu_runtime::AddressableBankResidencyError),
    /// No owned entry catalog exists for this namespace.
    #[error("sparse entry cache has no catalog for namespace {namespace}")]
    UnknownNamespace {
        /// Missing namespace identity.
        namespace: usize,
    },
    /// A namespace's global entry span cannot be represented by MLX indexing.
    #[error("namespace {namespace} global entry span {global_span} exceeds MLX i32 indexing")]
    EntryCountOverflow {
        /// Namespace identity.
        namespace: usize,
        /// Required global entry span.
        global_span: usize,
    },
    /// A dense replacement row cannot be represented by the native indexing mechanism.
    #[error("replacement row {row} for bank member {identity:?} exceeds MLX i32 indexing")]
    ReplacementRowOverflow {
        /// Original global physical storage identity.
        identity: ParameterBankKey,
        /// Dense row within the canonical replacement parameter.
        row: usize,
    },
    /// Device-side validation found one or more out-of-range selections.
    #[error("namespace {namespace} contains {invalid_count} grouped ids outside 0..{global_span}")]
    InvalidSelectionSet {
        /// Incrementalr namespace identity.
        namespace: usize,
        /// Number of invalid selection rows.
        invalid_count: usize,
        /// Valid global entry span.
        global_span: usize,
    },
    /// A selection id was negative or otherwise invalid.
    #[error("invalid grouped entry id {entry} for namespace {namespace}; this rank catalogs {known_owned_entries} owned entries")]
    InvalidEntryId {
        /// Namespace containing the selection.
        namespace: usize,
        /// Invalid signed selection value.
        entry: i64,
        /// Owned entries cataloged for diagnostics.
        known_owned_entries: usize,
    },
    /// Selectionr ids used an unsupported scalar type.
    #[error("grouped entry ids must use an integer dtype, got {actual:?}")]
    InvalidSelectionDtype {
        /// Unsupported selector-id scalar type.
        actual: Dtype,
    },
    /// A supplied selection shape had a negative dimension or overflowed.
    #[error("invalid grouped entry shape {0:?}")]
    InvalidSelectionShape(Vec<i32>),
    /// Selection shape and host values disagreed.
    #[error("grouped entry shape {shape:?} does not describe {elements} values")]
    SelectionShapeMismatch {
        /// Declared selection shape.
        shape: Vec<i32>,
        /// Supplied host value count.
        elements: usize,
    },
    /// Hidden rows, selection rows, and selection-weight rows did not align.
    #[error("grouped entry batch shapes do not align: hidden {hidden:?}, selections {selections:?}, weights {weights:?}")]
    GroupedBatchShapeMismatch {
        /// Grouped hidden-state shape.
        hidden: Vec<i32>,
        /// Grouped entry-id shape.
        selections: Vec<i32>,
        /// Grouped entry-weight shape.
        weights: Vec<i32>,
    },
    /// A caller-supplied compact bank returned the wrong row count.
    #[error("compact entry bank returned shape {actual:?}, expected {expected_rows} rows")]
    CompactBankOutputShapeMismatch {
        /// Required output rows.
        expected_rows: i32,
        /// Returned output shape.
        actual: Vec<i32>,
    },
    /// Cache statistics mutex was poisoned by a panic.
    #[error("sparse entry cache statistics are unavailable after a panic")]
    StatisticsPoisoned,
    /// A required compact binding had no selected source entries.
    #[error("compact entry binding {name:?} has no source arrays")]
    EmptyCompactBinding {
        /// Required binding name.
        name: String,
    },
    /// Optional companion presence differed across selected entries.
    #[error("compact entry companion {name:?} is missing from only part of the selected bank")]
    InconsistentCompanion {
        /// Inconsistent optional binding name.
        name: String,
    },
    /// Invalid offload plan configuration.
    #[error(transparent)]
    Offload(#[from] eredu_core::residency::OffloadError),
    /// Residency validation or materialization failed.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
    /// MLX evaluation, synchronization, or transfer failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
#[path = "parameter_bank/tests.rs"]
mod tests;

mod rows;
pub use rows::{MlxRowBank, MlxRowLookupSupport, MlxRowLookups};
#[cfg(test)]
mod rows_tests;
