//! Source-free physical topology for the shared Gemma vision equations.
use super::VisionConfig;
use eredu_runtime::{MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole};

/// Patch projection, position table and optional pooled standardization.
pub(crate) fn static_groups(
    config: &VisionConfig,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let spec = config
        .pinned(root)
        .map_err(|e| ParallelPlanError::InvalidTensor(e.to_string()))?;
    groups(spec, &spec.root)
}

/// One independently leased attention and gated feed-forward block.
pub(crate) fn layer_groups(
    config: &VisionConfig,
    root: &str,
    index: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let spec = config
        .layer(index, root)
        .map_err(|e| ParallelPlanError::InvalidTensor(e.to_string()))?;
    groups(spec, &spec.root)
}

fn groups(
    spec: &impl eredu_nn::DeclareLogicalParameters,
    root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    Ok(vec![eredu_runtime::logical_parameter_group(
        root,
        ParameterRole::Replicated,
        None,
        spec,
        |_| MemberSharding::Replicated,
    )?])
}
