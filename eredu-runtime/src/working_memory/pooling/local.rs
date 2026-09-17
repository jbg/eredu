//! Distinct resident and paged local histories use their actual shared workers.
use super::*;
use crate::working_memory::{WorkspacePagedAppendState, WorkspacePagedBlock};
use std::mem::size_of;

#[derive(Debug, Clone)]
pub(super) enum Local {
    Resident(WorkspaceConcatLayerState),
    Paged(Rc<WorkspacePagedAppendState>),
}
impl Local {
    pub(super) fn paged(
        layer: usize,
        policy: &LayerCachePolicy,
        batch: i32,
        state: WorkspacePagedAppendState,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError> {
        context
            .charge_metadata(size_of::<(
                usize,
                &LayerCachePolicy,
                i32,
                WorkspacePagedAppendState,
                &WorkspaceContext,
                crate::state::PoolingAttentionGeometryPlan,
                super::super::WorkspacePagedGeometry,
                Result<Self, StateError>,
            )>())
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        let geometry = crate::state::pooling_attention_geometry_plan(layer, policy, |args| {
            context.metadata_error(args)
        })
        .map_err(StateError::WorkspaceConstruction)?;
        let actual = state.geometry();
        let width = match policy {
            LayerCachePolicy::KeyOnly { head_dim, .. }
            | LayerCachePolicy::KeyOnlyWithFixedState { head_dim, .. } => head_dim.get(),
            _ => unreachable!("validated key-only geometry"),
        };
        let width = i32::try_from(width)
            .map_err(|cause| StateError::WorkspaceConstruction(context.metadata_source(cause)))?;
        if !context.shares_trace(state.workspace_context())
            || !actual.key_only
            || actual.dimensions != [batch, 1, width]
            || actual.window != Some(geometry.sliding_window)
            || i32::try_from(actual.offset).is_err()
        {
            return Err(StateError::WorkspaceConstruction(context.metadata_error(
                format_args!("paged pooling local differs from selected source geometry"),
            )));
        }
        context
            .metadata_rc(state)
            .map(Self::Paged)
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))
    }
    pub(super) fn offset(&self) -> i32 {
        match self {
            Self::Resident(s) => s.offset(),
            Self::Paged(s) => s.geometry().offset as i32,
        }
    }
    pub(super) fn window(&self) -> i32 {
        match self {
            Self::Resident(s) => s.max_size().expect("local window"),
            Self::Paged(s) => s.geometry().window.expect("validated local window"),
        }
    }
    pub(super) fn logical_value_width(&self) -> i32 {
        match self {
            Self::Resident(_) => 1,
            Self::Paged(_) => 0,
        }
    }
    fn paged_mut<'a>(
        state: &'a mut Rc<WorkspacePagedAppendState>,
        context: &WorkspaceContext,
    ) -> Result<&'a mut WorkspacePagedAppendState, Error> {
        context.charge_metadata(size_of::<(
            &mut Rc<WorkspacePagedAppendState>,
            &WorkspaceContext,
            WorkspacePagedAppendState,
            Result<(), Error>,
        )>())?;
        if Rc::get_mut(state).is_none() {
            let copied = state.copy_metadata(context)?;
            *state = context.metadata_rc(copied)?;
        }
        Ok(Rc::get_mut(state).expect("unique paged metadata"))
    }
    pub(super) fn update(
        &mut self,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<[WorkspaceTensor; 2], Error> {
        context.charge_metadata(size_of::<(
            &mut Self,
            WorkspaceTensor,
            WorkspaceTensor,
            &WorkspaceContext,
            [WorkspaceTensor; 2],
            Result<[WorkspaceTensor; 2], Error>,
        )>())?;
        match self {
            Self::Resident(state) => state
                .update_for_attention(keys, values, context)
                .map(|(k, v)| [k, v]),
            Self::Paged(state) => {
                if keys.shape().len() != 4
                    || values.shape() != [keys.shape()[0], 1, keys.shape()[2], 0]
                    || i32::try_from(state.geometry().offset)
                        .ok()
                        .and_then(|n| n.checked_add(keys.shape()[2]))
                        .is_none()
                {
                    return Err(
                        context.metadata_error(format_args!("invalid paged pooling update"))
                    );
                }
                // The ordinary key-only producer receives a zero-width logical
                // value then materializes its one-channel persistence sentinel.
                let sentinel = WorkspaceTensor::full_f32(
                    0.,
                    &[keys.shape()[0], 1, keys.shape()[2], 1],
                    context,
                )?;
                drop(values);
                Self::paged_mut(state, context)?
                    .update_visible_normalized([keys, sentinel], context)
            }
        }
    }
    pub(super) fn reset(&mut self, context: &WorkspaceContext) -> Result<(), Error> {
        match self {
            Self::Resident(state) => state
                .reset()
                .map_err(|cause| cause.into_workspace_error(context)),
            Self::Paged(state) => {
                Self::paged_mut(state, context)?.reset_metadata();
                Ok(())
            }
        }
    }
    pub(super) fn values(&self) -> Values<'_> {
        match self {
            Self::Resident(state) => Values::Resident(state.retained_values()),
            Self::Paged(state) => Values::Paged {
                blocks: state.blocks().iter(),
                current: None,
                tail: state.tail().map(|values| values.iter()),
            },
        }
    }
}
pub(super) enum Values<'a> {
    Resident(
        <WorkspaceConcatLayerState as RuntimeLayerState<WorkspaceBackend>>::RetainedValues<'a>,
    ),
    Paged {
        blocks: std::slice::Iter<'a, WorkspacePagedBlock>,
        current: Option<std::slice::Iter<'a, WorkspaceTensor>>,
        tail: Option<std::slice::Iter<'a, WorkspaceTensor>>,
    },
}
impl<'a> Iterator for Values<'a> {
    type Item = &'a WorkspaceTensor;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Resident(values) => values.next(),
            Self::Paged {
                blocks,
                current,
                tail,
            } => loop {
                if let Some(value) = current.as_mut().and_then(Iterator::next) {
                    return Some(value);
                }
                if let Some(block) = blocks.next() {
                    *current = block.tensor_values().map(|values| values.iter());
                } else {
                    return tail.as_mut().and_then(Iterator::next);
                }
            },
        }
    }
}
