//! Descriptive memory contracts for reusable neural mechanisms.
//!
//! Logical values are not allocations: a view or broadcast may share storage.
//! Backends add storage facts for their selected implementation. These facts do
//! not schedule, sum, or predict a live peak. Queries may allocate host records,
//! but must never allocate tensors, submit work, evaluate graphs, or change state.

use crate::{AttentionArithmetic, Error, TensorElementType};
use eredu_checkpoint::LinearFormat;

/// One invocation's typed logical geometry, derived from ordinary operator inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum MechanismInvocation {
    /// Dense or packed affine projection; physical parameter storage is described separately.
    Projection {
        /// Independent input rows.
        rows: u64,
        /// Complete dot-product width.
        input: u64,
        /// Locally owned output width.
        output: u64,
        /// Selected physical encoding.
        format: LinearFormat,
        /// Activation scalar representation.
        element: TensorElementType,
        /// Dense weight representation, when known (packed encodings use `format`).
        weight_element: Option<TensorElementType>,
        /// Ordinary output bias is present.
        bias: bool,
    },
    /// Grouped-query attention, before backend kernel/tile selection.
    Attention {
        /// Batch size.
        batch: u64,
        /// Query heads.
        query_heads: u64,
        /// Key/value heads.
        kv_heads: u64,
        /// Query rows per head.
        queries: u64,
        /// Visible key positions.
        keys: u64,
        /// Query/key width.
        key_width: u64,
        /// Value width.
        value_width: u64,
        /// Input scalar representation.
        element: TensorElementType,
        /// Required arithmetic/rounding policy.
        arithmetic: AttentionArithmetic,
        /// Score soft-capping is required.
        softcap: bool,
        /// Learned sink logits are present.
        sinks: bool,
    },
    /// Causal depthwise convolution including its retained history.
    Convolution {
        /// Batch size.
        batch: u64,
        /// Input token count.
        tokens: u64,
        /// Independent channels.
        channels: u64,
        /// Kernel width, including the current token.
        kernel: u64,
        /// Input scalar representation.
        element: TensorElementType,
    },
    /// Recurrent scan with a float32 state matrix.
    Recurrent {
        /// Selected portable recurrence.
        kind: RecurrentKind,
        /// Batch size.
        batch: u64,
        /// Input token count.
        tokens: u64,
        /// Independent heads.
        heads: u64,
        /// Output width per head.
        value_width: u64,
        /// State width (key width for gated delta).
        state_width: u64,
        /// Input scalar representation.
        element: TensorElementType,
        /// Explicit scan chunk limit, when part of the ordinary request.
        chunk_size: Option<u64>,
    },
    /// Selected affine expert dispatch and weighted reduction.
    ExpertDispatch {
        /// Input rows.
        rows: u64,
        /// Locally available expert count.
        experts: u64,
        /// Selected routes per row.
        selected: u64,
        /// Complete input width.
        input: u64,
        /// Locally owned output width.
        output: u64,
        /// Physical expert parameter encoding.
        format: LinearFormat,
        /// Input scalar representation.
        element: TensorElementType,
    },
    /// An append to logical key/value history. The cache owns physical policy.
    CacheUpdate {
        /// Batch size.
        batch: u64,
        /// Key/value heads.
        heads: u64,
        /// Retained positions before this append (not absolute position).
        previous: u64,
        /// New positions.
        appended: u64,
        /// Key width.
        key_width: u64,
        /// Value width.
        value_width: u64,
        /// Input scalar representation.
        element: TensorElementType,
    },
    /// Vocabulary sampling; policy is derived from the admitted generation config.
    Sampling {
        /// Simultaneously processed distributions.
        rows: u64,
        /// Vocabulary size.
        vocabulary: u64,
        /// Retained token history consumed by penalties.
        history: u64,
        /// Logits representation.
        element: TensorElementType,
        /// Selected sampling algorithm.
        mode: SamplingMode,
        /// Active top-k limit; zero disables it.
        top_k: u64,
        /// Nucleus filtering is active.
        top_p: bool,
        /// Minimum-probability filtering is active.
        min_p: bool,
        /// History-based penalties are active.
        penalties: bool,
    },
}

/// Portable recurrence whose state geometry is known without family dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentKind {
    /// Matrix gated-delta recurrence.
    GatedDelta,
    /// Selective state-space recurrence.
    SelectiveStateSpace,
}

/// Portable sampling selection, after resolving temperature and strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMode {
    /// Deterministic maximum logit.
    Greedy,
    /// Standard stochastic categorical selection.
    Categorical,
    /// Adaptive surprise cutoff with retained scalar state.
    MirostatV2,
}

/// Meaning of a logical value; none of these categories proves allocation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalValueKind {
    /// Borrowed input owned by an earlier invocation or parameter/state producer.
    Input,
    /// Result consumed by the next invocation.
    Output,
    /// Replacement mutable state.
    State,
}

/// Shape and physical scalar representation of one logical invocation value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalValue {
    /// Invocation-local semantic name.
    pub name: String,
    /// Logical axes; views may have larger logical extents than backing storage.
    pub shape: Vec<u64>,
    /// Scalar representation.
    pub element: TensorElementType,
    /// Input, output, or replacement state.
    pub kind: LogicalValueKind,
}

impl LogicalValue {
    /// Checked logical byte extent, not a backing allocation capacity.
    pub fn logical_bytes(&self) -> Result<u64, Error> {
        product(&self.shape)?
            .checked_mul(element_bytes(self.element))
            .ok_or_else(|| Error::backend("mechanism logical byte extent overflowed"))
    }
}

/// Explicit byte interval used only for one described storage object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MechanismBytes {
    /// Known minimum bytes.
    pub lower: u64,
    /// Known upper bound; omission means unknown, never zero.
    pub upper: Option<u64>,
}

impl MechanismBytes {
    /// Exactly sized logical allocation payload.
    pub const fn exact(bytes: u64) -> Self {
        Self {
            lower: bytes,
            upper: Some(bytes),
        }
    }
    /// Known minimum with an unbounded remainder.
    pub const fn unknown(lower: u64) -> Self {
        Self { lower, upper: None }
    }
    fn validate(self) -> Result<(), Error> {
        if self.upper.is_some_and(|upper| upper < self.lower) {
            return Err(Error::backend(
                "mechanism upper bytes are below lower bytes",
            ));
        }
        Ok(())
    }
}

/// Storage's role, independent of the time it is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismStorageRole {
    /// Produced tensor result.
    Output,
    /// Replacement state.
    State,
    /// Implementation workspace.
    Scratch,
    /// Converted parameter owned by its residency owner.
    ParameterConversion,
}

/// How far storage can remain reachable. This is not an execution schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageRetention {
    /// Kernel-local storage, released when native execution completes.
    NativeCompletion,
    /// Lazy graph dependencies remain live until evaluation completes.
    Evaluation,
    /// Returned value remains live until its consuming owner releases it.
    Returned,
    /// Conversion belongs to a retained parameter materialization.
    ParameterOwner,
    /// The implementation has not established retention.
    Unknown,
}

/// Placement relative to the invocation, resolved to physical pools by composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismPlacement {
    /// Native execution pool supplied by the caller/backend.
    Execution,
    /// Host memory (which can alias the execution pool on unified memory).
    Host,
    /// No pool has been established.
    Unknown,
}

/// Backing ownership must be explicit to avoid inventing sharing or residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MechanismBacking {
    /// Fresh storage local to this invocation; name is unique within it.
    Invocation,
    /// Storage identity is supplied by an existing owner (e.g. converted weight).
    Owner(String),
    /// A view/alias or optional cached conversion lacks an authoritative backing identity.
    Unknown,
}

/// One independently described storage object, never an aggregate live peak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MechanismStorage {
    /// Unique local name.
    pub name: String,
    /// Logical use.
    pub role: MechanismStorageRole,
    /// Actual representation payload (not hypothetical uncompressed parameters).
    pub payload: MechanismBytes,
    /// Allocation capacity including payload, alignment and spare space.
    pub capacity: MechanismBytes,
    /// Backing identity source.
    pub backing: MechanismBacking,
    /// Physical location relative to invocation.
    pub placement: MechanismPlacement,
    /// Declared retention boundary.
    pub retention: StorageRetention,
    /// Producer provenance or missing implementation fact.
    pub detail: String,
}

/// Reusable mechanism description; logical values and physical storage are distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MechanismMemoryContract {
    /// Checked logical inputs/results derived by portable mechanisms.
    pub values: Vec<LogicalValue>,
    /// Backend-described independent storage, including explicit unknown objects.
    pub storage: Vec<MechanismStorage>,
    /// Specific missing facts; empty means complete only for this invocation.
    pub missing: Vec<String>,
}

impl MechanismMemoryContract {
    /// Default for a backend that has not described its implementation.
    pub fn unknown(invocation: &MechanismInvocation) -> Result<Self, Error> {
        Ok(Self {
            values: invocation.logical_values()?,
            storage: Vec::new(),
            missing: vec!["implementation allocation, scratch, conversions, alignment and retention are undescribed".into()],
        })
    }

    /// Validates all records without allocating or inspecting execution resources.
    pub fn validate(&self) -> Result<(), Error> {
        let mut names = std::collections::BTreeSet::new();
        for value in &self.values {
            if value.name.trim().is_empty() || !names.insert(&value.name) {
                return Err(Error::backend(
                    "mechanism logical value names must be unique and nonempty",
                ));
            }
            value.logical_bytes()?;
        }
        names.clear();
        for storage in &self.storage {
            if storage.name.trim().is_empty()
                || storage.detail.trim().is_empty()
                || !names.insert(&storage.name)
            {
                return Err(Error::backend(
                    "mechanism storage needs unique names and provenance",
                ));
            }
            storage.payload.validate()?;
            storage.capacity.validate()?;
            if storage
                .capacity
                .upper
                .is_some_and(|upper| upper < storage.payload.lower)
            {
                return Err(Error::backend("mechanism capacity cannot contain payload"));
            }
            if matches!(&storage.backing, MechanismBacking::Owner(owner) if owner.trim().is_empty())
            {
                return Err(Error::backend("mechanism backing owner must be nonempty"));
            }
        }
        if self.missing.iter().any(|reason| reason.trim().is_empty()) {
            return Err(Error::backend("mechanism missing facts must be named"));
        }
        Ok(())
    }
}

/// Bytes in an ordinary neural scalar representation.
pub const fn element_bytes(element: TensorElementType) -> u64 {
    use TensorElementType::*;
    match element {
        Bool | I8 | U8 => 1,
        F16 | Bf16 | I16 | U16 => 2,
        F32 | I32 | U32 => 4,
        F64 | I64 | U64 | Complex64 => 8,
    }
}

pub(crate) fn product(shape: &[u64]) -> Result<u64, Error> {
    shape
        .iter()
        .try_fold(1u64, |total, width| total.checked_mul(*width))
        .ok_or_else(|| Error::backend("mechanism geometry overflowed"))
}

mod geometry;
mod producers;
pub use producers::cache_update_invocation;
#[cfg(test)]
mod tests;
