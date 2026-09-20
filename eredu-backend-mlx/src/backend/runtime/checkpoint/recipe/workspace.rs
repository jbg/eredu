//! Capacity facts for the exact disk-read batch selected by the materializer.

use super::*;
use crate::backend::nn::workspace::NativeAllocationFacts;
use eredu_core::WorkspaceBound;

impl BindingReadBatch<'_> {
    /// Complete tensor payload for this batch through its enclosing transfer's
    /// completion. All batches submitted together must be added, since the
    /// transfer retains every source and output. Existing pinned bindings and
    /// host-residency transfers require separate charges.
    ///
    /// Direct encoded reads fill final MLX allocations without host payload
    /// staging. The selected MLX `Copy` primitive shares that same backing even
    /// across streams. Metadata, OS file caches and driver memory are outside this
    /// tensor/host-payload domain. Ordinary recipes remain explicitly unpriced.
    /// The caller supplies facts retained from the selected Metal realization;
    /// this method neither selects a device nor reads a checkpoint payload.
    pub(crate) fn workspace_bound(
        &self,
        allocation: NativeAllocationFacts,
    ) -> Result<WorkspaceBound, WeightRecipeError> {
        let Self::Direct { bindings, reads } = self else {
            return Ok(WorkspaceBound::Unknown {
                reason: "ordinary checkpoint materialization lacks complete selected native capacity, conversion and source-staging facts".into(),
            });
        };
        if bindings.len() != reads.len() {
            return Err(WeightRecipeError::Preflight(
                "direct materialization batch has different binding and read counts".into(),
            ));
        }
        direct_workspace_bound(
            bindings
                .iter()
                .zip(reads)
                .map(|(binding, read)| (*binding, read.read.output().byte_len)),
            allocation,
        )
    }
}

/// One shared physical-output rule for ephemeral batches and retained plans.
pub(super) fn validate_direct_output(
    binding: &eredu_runtime::WeightBinding,
    logical: u64,
) -> Result<(), WeightRecipeError> {
    if logical != binding.expected_bytes() {
        return Err(WeightRecipeError::Preflight(format!(
            "direct binding {} expects {} bytes, but its retained read produces {logical}",
            binding.name(),
            binding.expected_bytes()
        )));
    }
    Ok(())
}

pub(super) fn direct_output_capacity(
    binding: &eredu_runtime::WeightBinding,
    logical: u64,
    allocation: NativeAllocationFacts,
) -> Result<u64, WeightRecipeError> {
    validate_direct_output(binding, logical)?;
    allocation
        .buffer_capacity(logical)
        .map_err(|error| WeightRecipeError::Preflight(error.to_string()))
}

pub(super) fn direct_workspace_bound<'a>(
    outputs: impl IntoIterator<Item = (&'a eredu_runtime::WeightBinding, u64)>,
    allocation: NativeAllocationFacts,
) -> Result<WorkspaceBound, WeightRecipeError> {
    let mut bytes = 0_u64;
    let mut count = 0_usize;
    for (binding, logical) in outputs {
        bytes = bytes
            .checked_add(direct_output_capacity(binding, logical, allocation)?)
            .ok_or(WeightRecipeError::ArithmeticOverflow(
                "direct materialization capacity",
            ))?;
        count += 1;
    }
    Ok(WorkspaceBound::bounded(bytes, format!(
        "selected direct encoded read batch retains {count} unique output allocations through transfer completion; MLX Copy shares source backing across streams; Metal capacity rounding and oversized cache reuse included; direct vectored reads allocate no host payload staging; excludes metadata, existing residency, OS caches and driver memory"
    )))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
