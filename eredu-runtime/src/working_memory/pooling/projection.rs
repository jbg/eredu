//! Actual pooling projection populations, shared by its constructor and query.
use super::*;
use std::mem::size_of;

#[derive(Clone, Copy)]
pub(super) enum ProjectionFailure {
    Duplicate,
    Omitted,
    Presence,
    Shape,
    Undeclared,
}
impl ProjectionFailure {
    const ALL: [Self; 5] = [
        Self::Duplicate,
        Self::Omitted,
        Self::Presence,
        Self::Shape,
        Self::Undeclared,
    ];
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::Duplicate => "duplicate existing pooling role",
            Self::Omitted => "existing pooling role omitted",
            Self::Presence => "existing pooling presence differs from its frontier",
            Self::Shape => "existing pooling shape differs from its frontier",
            Self::Undeclared => "existing pooling inventory has undeclared roles",
        }
    }
}

fn geometry(policy: &LayerCachePolicy) -> Option<crate::state::PoolingAttentionGeometryPlan> {
    crate::state::pooling_attention_geometry_plan(0, policy, |_| {
        WorkspaceMetadataError::Unqualified.into()
    })
    .ok()
}

impl WorkspacePoolingStateFactory {
    /// Exact declared native-adapter output population. The same finite geometry
    /// validator checks this source before rows are constructed or counted.
    pub fn projection_component_count(policy: &LayerCachePolicy) -> Option<usize> {
        let geometry = geometry(policy)?;
        Some(
            geometry
                .stream_ratios()
                .iter()
                .enumerate()
                .map(|(stream, _)| {
                    COMPONENTS
                        .iter()
                        .filter(|&&component| {
                            policy.fixed_state().iter().any(|declaration| {
                                declaration.role
                                    == StateTensorRole::Pooling {
                                        stream: stream as u32,
                                        component,
                                    }
                            })
                        })
                        .count()
                })
                .sum(),
        )
    }

    /// Host constructors for this selected pooling projection. Native array
    /// observation/import and the outer state table have separate owners. Input
    /// rows must be the exact declared population emitted by the shared native
    /// adapter; this is not a bound for an arbitrary caller-supplied iterator.
    pub fn projection_control_bytes(policy: &LayerCachePolicy) -> Option<usize> {
        let geometry = geometry(policy)?;
        let declarations = policy.fixed_state().len();
        let rows = Self::projection_component_count(policy)?;
        let local = WorkspaceConcatStateFactory::fixed_role_constructor_storage_bytes(policy)?;
        let mut policy_bytes = WorkspaceContext::metadata_rc_bytes::<LayerCachePolicy>()?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<StateTensorPolicy>(
                declarations,
            )?)?;
        for declaration in policy.fixed_state() {
            policy_bytes = policy_bytes.checked_add(WorkspaceContext::metadata_vec_bytes::<
                StateTensorDimension,
            >(declaration.shape.len())?)?;
        }
        // One local is constructed by create_layer, then project_layer builds
        // its imported replacement. Each clears its freshly constructed fixed
        // rows; these are two actual cumulative constructor requests.
        let invalid_length = ProjectionFailure::ALL
            .into_iter()
            .map(|v| v.reason().len())
            .max()?;
        let invalid = WorkspaceContext::metadata_string_bytes(invalid_length)?
            .checked_add(WorkspaceContext::metadata_source_bytes::<StateError>()?)?;
        let typed = WorkspaceContext::metadata_source_bytes::<
            eredu_core::cache::StateTensorDimensionError,
        >()?
        .max(WorkspaceContext::metadata_source_bytes::<
            std::num::TryFromIntError,
        >()?);
        let stream = stream::validation_error_bytes()?;
        let parts = [
            Self::component_control_bytes::<Vec<(StateTensorRole, Option<WorkspaceTensor>)>>(),
            policy_bytes,
            local.checked_mul(2)?,
            WorkspaceContext::metadata_vec_bytes::<WorkspacePoolingStream>(
                geometry.stream_ratios().len(),
            )?,
            WorkspaceContext::metadata_rc_bytes::<Vec<WorkspacePoolingStream>>()?,
            WorkspaceContext::metadata_vec_bytes::<(
                StateTensorRole,
                Option<Option<WorkspaceTensor>>,
            )>(declarations)?,
            WorkspaceContext::metadata_vec_bytes::<(StateTensorRole, Option<WorkspaceTensor>)>(
                rows,
            )?,
            invalid.max(typed).max(stream),
            WorkspaceConcatStateFactory::projection_error_control_bytes(policy)?,
            size_of::<WorkspacePoolingLayerState>(),
            size_of::<WorkspacePoolingStream>(),
            size_of::<Result<WorkspacePoolingLayerState, StateError>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<usize>>(),
            size_of::<StateError>(),
            size_of::<ProjectionFailure>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<crate::state::PoolingAttentionGeometryPlan>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
