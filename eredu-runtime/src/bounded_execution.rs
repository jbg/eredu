//! Explicit finite policy for invocation, indexed selection, rows and append history.
//!
//! These limits carry no model geometry. Architecture preparation lowers them to
//! exact mechanisms and rejects insufficient byte allowances before construction.
use crate::{AppendStreamLimits, ParameterBankLoadOptions, RowLookupLimits};

/// Invalid or conflicting finite execution policy, before source acquisition.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BoundedExecutionPolicyError {
    /// A count or nonempty workspace is zero, negative or cannot be addressed.
    #[error("invalid bounded execution resource: {0}")]
    Invalid(&'static str),
    /// An omitted residency cap would leave a resource unbounded.
    #[error("bounded execution requires a finite {0} limit")]
    Unbounded(&'static str),
    /// Two independently declared limits cannot describe one execution.
    #[error("conflicting bounded execution limits: {0}")]
    Conflict(&'static str),
}

fn bytes(
    value: u64,
    resource: &'static str,
    allow_zero: bool,
) -> Result<(), BoundedExecutionPolicyError> {
    if !allow_zero && value == 0 {
        return Err(BoundedExecutionPolicyError::Invalid(resource));
    }
    Ok(())
}

/// Maximum batch, per-call chunk and complete sequence history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvocationLimits {
    batch: i32,
    chunk_tokens: i32,
    history_tokens: i32,
    invocation_tokens: usize,
}
impl InvocationLimits {
    /// Declares positive tensor-addressable capacities. History includes prefill
    /// and every committed decode token; it is not a sliding attention window.
    /// A complete prefill may span multiple chunks without increasing batch capacity.
    pub fn new(
        batch: i32,
        chunk_tokens: i32,
        history_tokens: i32,
    ) -> Result<Self, BoundedExecutionPolicyError> {
        if batch <= 0 || chunk_tokens <= 0 || history_tokens <= 0 {
            return Err(BoundedExecutionPolicyError::Invalid("invocation geometry"));
        }
        if chunk_tokens > history_tokens {
            return Err(BoundedExecutionPolicyError::Conflict(
                "chunk exceeds history",
            ));
        }
        let invocation_tokens = (batch as usize)
            .checked_mul(chunk_tokens as usize)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(BoundedExecutionPolicyError::Invalid(
                "invocation token count",
            ))?;
        Ok(Self {
            batch,
            chunk_tokens,
            history_tokens,
            invocation_tokens,
        })
    }
    /// Maximum simultaneous sequence lanes.
    pub const fn batch(self) -> i32 {
        self.batch
    }
    /// Maximum tokens per lane in one prefill or decode invocation.
    pub const fn chunk_tokens(self) -> i32 {
        self.chunk_tokens
    }
    /// Maximum retained tokens per lane across the complete sequence.
    pub const fn history_tokens(self) -> i32 {
        self.history_tokens
    }
    /// Checked maximum count of original token IDs in one invocation.
    pub const fn invocation_tokens(self) -> usize {
        self.invocation_tokens
    }
}

/// Tiled indexed-selection geometry and independent workspace allowances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledSelectionLimits {
    tile_entries: i32,
    selection_workspace_bytes: u64,
    invocation_workspace_bytes: u64,
}
impl TiledSelectionLimits {
    /// Selects a finite score tile and scratch bounds. Architecture admission
    /// accounts for exact heads, selected positions, and partial state.
    pub fn new(
        tile_entries: i32,
        selection_workspace_bytes: u64,
        invocation_workspace_bytes: u64,
    ) -> Result<Self, BoundedExecutionPolicyError> {
        if tile_entries <= 0 {
            return Err(BoundedExecutionPolicyError::Invalid("selection tile"));
        }
        bytes(selection_workspace_bytes, "selection workspace", false)?;
        bytes(invocation_workspace_bytes, "invocation workspace", false)?;
        if selection_workspace_bytes > invocation_workspace_bytes {
            return Err(BoundedExecutionPolicyError::Conflict(
                "selection workspace exceeds invocation workspace",
            ));
        }
        Ok(Self {
            tile_entries,
            selection_workspace_bytes,
            invocation_workspace_bytes,
        })
    }
    /// Maximum complete entries scored by one tile.
    pub const fn tile_entries(self) -> i32 {
        self.tile_entries
    }
    /// Workspace for one tiled selection query.
    pub const fn selection_workspace_bytes(self) -> u64 {
        self.selection_workspace_bytes
    }
    /// Workspace for the complete architecture invocation, including selection.
    pub const fn invocation_workspace_bytes(self) -> u64 {
        self.invocation_workspace_bytes
    }
}

/// Row lookup limits and one shared generic parameter-bank residency policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLookupLoadPolicy {
    limits: RowLookupLimits,
    bank: ParameterBankLoadOptions,
    retained_scalar_bytes: u64,
}
impl RowLookupLoadPolicy {
    /// Validates finite caches, transfer/conversion scratch and lookup buffers.
    /// Zero host residency and zero retained scalars are valid explicit choices.
    pub fn new(
        limits: RowLookupLimits,
        bank: ParameterBankLoadOptions,
        retained_scalar_bytes: u64,
    ) -> Result<Self, BoundedExecutionPolicyError> {
        limits
            .validate()
            .map_err(|_| BoundedExecutionPolicyError::Invalid("row lookup limits"))?;
        bank.validate()
            .map_err(|_| BoundedExecutionPolicyError::Invalid("parameter-bank policy"))?;
        if limits.rows_per_acquisition > i32::MAX as usize {
            return Err(BoundedExecutionPolicyError::Invalid("rows per acquisition"));
        }
        for (value, resource) in [
            (bank.offload().device_budget_bytes(), "row device cache"),
            (bank.offload().host_budget_bytes(), "row host cache"),
        ] {
            bytes(
                value.ok_or(BoundedExecutionPolicyError::Unbounded(resource))?,
                resource,
                true,
            )?;
        }
        for (value, resource) in [
            (limits.acquisition_bytes, "row acquisition bytes"),
            (limits.host_bytes, "row host planning bytes"),
            (limits.output_bytes, "row output bytes"),
            (bank.compact_bank_scratch_bytes(), "parameter-bank scratch"),
            (
                bank.prefill_compact_bank_target_bytes(),
                "parameter-bank prefill target",
            ),
        ] {
            bytes(value, resource, false)?;
        }
        bytes(retained_scalar_bytes, "retained row scalars", true)?;
        Ok(Self {
            limits,
            bank,
            retained_scalar_bytes,
        })
    }
    /// Exact portable row request, acquisition and output limits.
    pub const fn limits(self) -> RowLookupLimits {
        self.limits
    }
    /// Shared generic parameter-bank cache and scratch controls.
    pub const fn bank(self) -> ParameterBankLoadOptions {
        self.bank
    }
    /// Total retained scale-companion allowance across all row sources.
    pub const fn retained_scalar_bytes(self) -> u64 {
        self.retained_scalar_bytes
    }
}

/// Append-history controls for each declared stream and sequence lane.
///
/// Counts are records, not tokens. Architectures must prove that `entries`
/// covers the complete invocation history and multiply byte allowances by all
/// stream owners and lanes. Paged storage still uses the selected shared pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendStreamLoadPolicy {
    limits: AppendStreamLimits,
    payload_bytes: u64,
    scratch_bytes: u64,
    catalog_bytes: u64,
}
impl AppendStreamLoadPolicy {
    /// Validates finite per-stream/lane allowances independently of record width.
    pub fn new(
        limits: AppendStreamLimits,
        payload_bytes: u64,
        scratch_bytes: u64,
        catalog_bytes: u64,
    ) -> Result<Self, BoundedExecutionPolicyError> {
        limits
            .validate()
            .map_err(|_| BoundedExecutionPolicyError::Invalid("append stream limits"))?;
        for (value, resource) in [
            (payload_bytes, "append payload"),
            (scratch_bytes, "append scratch"),
            (catalog_bytes, "append catalog"),
        ] {
            bytes(value, resource, false)?;
        }
        Ok(Self {
            limits,
            payload_bytes,
            scratch_bytes,
            catalog_bytes,
        })
    }
    /// Record count, page geometry and compact read capacities.
    pub const fn limits(self) -> AppendStreamLimits {
        self.limits
    }
    /// Per-stream and per-lane resident payload allowance.
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }
    /// Per-stream and per-lane tail/range scratch allowance.
    pub const fn scratch_bytes(self) -> u64 {
        self.scratch_bytes
    }
    /// Per-stream and per-lane host catalog allowance.
    pub const fn catalog_bytes(self) -> u64 {
        self.catalog_bytes
    }
}

/// One explicit finite request for mechanisms needing indexed selection, row
/// acquisition and append-only history. No family-specific defaults are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedExecutionPolicy {
    invocation: InvocationLimits,
    selection: TiledSelectionLimits,
    rows: RowLookupLoadPolicy,
    append: AppendStreamLoadPolicy,
}
impl BoundedExecutionPolicy {
    /// Binds the independently validated resource controls atomically.
    pub fn new(
        invocation: InvocationLimits,
        selection: TiledSelectionLimits,
        rows: RowLookupLoadPolicy,
        append: AppendStreamLoadPolicy,
    ) -> Result<Self, BoundedExecutionPolicyError> {
        if append.limits().read_entries < selection.tile_entries() as usize {
            return Err(BoundedExecutionPolicyError::Conflict(
                "append reads cannot cover selection tile",
            ));
        }
        Ok(Self {
            invocation,
            selection,
            rows,
            append,
        })
    }
    /// Maximum invocation and complete history geometry.
    pub const fn invocation(self) -> InvocationLimits {
        self.invocation
    }
    /// Tiled selection and invocation workspace allowances.
    pub const fn selection(self) -> TiledSelectionLimits {
        self.selection
    }
    /// Generic row lookup and parameter-bank controls.
    pub const fn rows(self) -> RowLookupLoadPolicy {
        self.rows
    }
    /// Per-stream and per-lane append resource controls.
    pub const fn append(self) -> AppendStreamLoadPolicy {
        self.append
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommunicationCompletionPolicy, NormalizedLoadRequest, ParallelLoadRequest,
        PipelineActivationDtype, PipelineWireContract,
    };
    use eredu_core::{
        residency::OffloadConfig, CompletionCancellationMode, ParallelRankTopology,
        ParallelTopology,
    };

    fn bank(device: Option<u64>, host: Option<u64>) -> ParameterBankLoadOptions {
        ParameterBankLoadOptions::new(OffloadConfig::new(device, host, 2).unwrap(), 8192, 4096)
            .unwrap()
    }
    fn rows() -> RowLookupLimits {
        RowLookupLimits {
            requests: 12,
            rows_per_acquisition: 3,
            acquisition_bytes: 1024,
            host_bytes: 2048,
            output_bytes: 4096,
        }
    }
    fn append(read_entries: usize) -> AppendStreamLoadPolicy {
        AppendStreamLoadPolicy::new(
            AppendStreamLimits {
                entries: 100,
                page_entries: 8,
                read_entries,
            },
            8192,
            4096,
            2048,
        )
        .unwrap()
    }
    fn policy() -> BoundedExecutionPolicy {
        BoundedExecutionPolicy::new(
            InvocationLimits::new(2, 3, 200).unwrap(),
            TiledSelectionLimits::new(4, 1024, 8192).unwrap(),
            RowLookupLoadPolicy::new(rows(), bank(Some(1024), Some(0)), 0).unwrap(),
            append(4),
        )
        .unwrap()
    }

    #[test]
    fn invocation_rejects_nonpositive_inconsistent_and_unaddressable_counts() {
        for (batch, chunk, history) in [
            (0, 1, 2),
            (1, 0, 2),
            (-1, 1, 2),
            (1, 1, 0),
            (1, 3, 2),
            (i32::MAX, 2, 2),
        ] {
            assert!(InvocationLimits::new(batch, chunk, history).is_err());
        }
        assert!(InvocationLimits::new(1, i32::MAX, i32::MAX).is_ok());
    }

    #[test]
    fn longer_history_changes_request_identity_without_inflating_invocation_or_rows() {
        let base = policy();
        let make = |history| {
            BoundedExecutionPolicy::new(
                InvocationLimits::new(1, 3, history).unwrap(),
                base.selection(),
                base.rows(),
                base.append(),
            )
            .unwrap()
        };
        let short = make(32);
        let long = make(256);
        assert_eq!(short.invocation().invocation_tokens(), 3);
        assert_eq!(long.invocation().invocation_tokens(), 3);
        assert_eq!(short.rows(), long.rows());
        assert_eq!(short.selection(), long.selection());
        assert_ne!(
            NormalizedLoadRequest::default().with_bounded_execution(short),
            NormalizedLoadRequest::default().with_bounded_execution(long),
        );
    }

    #[test]
    fn byte_allowances_reject_empty_required_buffers_but_preserve_explicit_caps() {
        for (tile, selection, invocation) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 2, 1)] {
            assert!(TiledSelectionLimits::new(tile, selection, invocation).is_err());
        }
        let limits = append(4).limits();
        for (payload, scratch, catalog) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(AppendStreamLoadPolicy::new(limits, payload, scratch, catalog).is_err());
        }
        // Large explicitly supplied integer caps are still exact; no undocumented
        // sentinel interpretation is introduced by this policy.
        assert!(TiledSelectionLimits::new(1, u64::MAX, u64::MAX).is_ok());
    }

    #[test]
    fn row_policy_requires_explicit_tier_bounds_and_valid_existing_contracts() {
        for (device, host) in [(None, Some(0)), (Some(1), None), (None, None)] {
            assert!(matches!(
                RowLookupLoadPolicy::new(rows(), bank(device, host), 0),
                Err(BoundedExecutionPolicyError::Unbounded(_))
            ));
        }
        assert!(RowLookupLoadPolicy::new(rows(), bank(Some(0), Some(0)), 0).is_ok());
        for field in 0..5 {
            let mut limits = rows();
            match field {
                0 => limits.requests = 0,
                1 => limits.rows_per_acquisition = 0,
                2 => limits.acquisition_bytes = 0,
                3 => limits.host_bytes = 0,
                _ => limits.output_bytes = 0,
            }
            assert!(RowLookupLoadPolicy::new(limits, bank(Some(1024), Some(0)), 0).is_err());
        }
    }

    #[test]
    fn append_read_capacity_must_cover_selection_tile() {
        let policy = policy();
        assert!(matches!(
            BoundedExecutionPolicy::new(
                policy.invocation(),
                policy.selection(),
                policy.rows(),
                append(3)
            ),
            Err(BoundedExecutionPolicyError::Conflict(_))
        ));
        let mut invalid = append(4).limits();
        invalid.entries = 0;
        assert!(AppendStreamLoadPolicy::new(invalid, 1, 1, 1).is_err());
    }

    #[test]
    fn parallel_and_local_invocation_authority_must_match_in_either_builder_order() {
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap();
        let completion = CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        for (batch, chunk, admitted) in [(2, 3, true), (1, 3, false), (2, 4, false)] {
            let parallel = ParallelLoadRequest::new(
                topology,
                PipelineWireContract::new(PipelineActivationDtype::Float32),
                batch,
                chunk,
                completion,
            )
            .unwrap();
            for request in [
                NormalizedLoadRequest::default()
                    .with_parallel_execution(parallel)
                    .unwrap()
                    .with_bounded_execution(policy()),
                NormalizedLoadRequest::default()
                    .with_bounded_execution(policy())
                    .with_parallel_execution(parallel)
                    .unwrap(),
            ] {
                assert_eq!(request.validate_model_preparation().is_ok(), admitted);
            }
        }
    }
}
