//! Paged layer access over exact block metadata and shared fixed-role storage.
use super::*;
use crate::working_memory::{WorkspacePagedAppendState, WorkspacePagedGeometry};
use eredu_nn::{BlockwiseAttentionOptions, BlockwiseAttentionSpec, workspace::WorkspaceDtype};
use std::rc::Rc;

/// Architecture access to an explicitly projected paged mechanism. Its shared
/// metadata snapshot copies on mutation under the same context; this neither
/// copies nor authorizes native cache state.
#[derive(Debug, Clone)]
pub struct WorkspacePagedLayerState {
    paged: Rc<WorkspacePagedAppendState>,
    fixed: FixedSlots,
    context: WorkspaceContext,
}
impl WorkspacePagedLayerState {
    /// Imports actual paged state and complete fixed-role values through the
    /// existing geometry/role builders. Composition retains original native
    /// storage and canonical pins beside the complete state until binding.
    pub fn project<I>(
        layer: usize,
        policy: &LayerCachePolicy,
        paged: WorkspacePagedAppendState,
        fixed: I,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError>
    where
        I: IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
    {
        context
            .charge_metadata(std::mem::size_of::<(
                Self,
                usize,
                &LayerCachePolicy,
                WorkspacePagedAppendState,
                I,
                &WorkspaceContext,
                WorkspacePagedGeometry,
                WorkspaceConcatStateFactory,
                WorkspaceConcatLayerState,
                Result<Self, StateError>,
            )>())
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        let invalid = |reason: &str| {
            StateError::workspace_invalid_layer(context, layer, format_args!("{reason}"))
        };
        if !context.shares_trace(paged.workspace_context()) {
            return Err(invalid("paged layer belongs to another metadata context"));
        }
        let geometry = paged.geometry();
        let batch = u32::try_from(geometry.dimensions[0])
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| invalid(projection::BATCH_EXTENT))?;
        let position = i32::try_from(geometry.offset)
            .map_err(|_| invalid("paged frontier exceeds tensor coordinate"))?;
        let factory = WorkspaceConcatStateFactory::new(batch, context)
            .map_err(StateError::WorkspaceConstruction)?;
        let mut declaration = factory.create_layer(layer, policy)?;
        projection::import_fixed(&mut declaration.fixed, fixed, context, &invalid)?;
        let Some(attention) = declaration.attention else {
            return Err(invalid("paged layer has no declared attention policy"));
        };
        if geometry.dimensions != [declaration.batch, attention.heads, attention.width]
            || geometry.window != attention.window
            || geometry.key_only != attention.key_only
        {
            return Err(invalid(
                "paged source differs from declared attention geometry",
            ));
        }
        context
            .validate_values(declaration.fixed.values().flatten())
            .map_err(StateError::WorkspaceConstruction)?;
        projection::validate_fixed(
            &declaration.fixed,
            policy,
            declaration.batch,
            position,
            &invalid,
        )?;
        Ok(Self {
            paged: context
                .metadata_rc(paged)
                .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?,
            fixed: declaration.fixed,
            context: context.clone(),
        })
    }
    /// Projects independently copied values through the same isolated-copy
    /// worker as native source tracing. It preserves the declared block/fixed
    /// geometry and validates the original trace before making any copy.
    pub fn copy_isolated(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.copy_values(context, false)
    }
    /// Copies mutable Device/fixed/tail values while retaining exact imported
    /// immutable Host roots for an independent resumed manager. This metadata
    /// operation supplies no native source, copy or execution authority.
    pub fn copy_for_resume(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.copy_values(context, true)
    }
    fn copy_values(&self, context: &WorkspaceContext, retain_host: bool) -> Result<Self, Error> {
        self.validate_context(context)?;
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &WorkspaceContext,
            Self,
            Result<Self, Error>,
            FixedSlots,
            eredu_nn::WorkspaceIsolatedCopy<'_>,
            WorkspacePagedAppendState,
            bool,
        )>())?;
        let paged = if retain_host {
            self.paged.copy_for_resume(context)?
        } else {
            self.paged.copy_isolated(context)?
        };
        let mut fixed = self.fixed.clone();
        for value in fixed.values_mut(context)?.iter_mut().flatten() {
            *value = eredu_nn::isolated_copy(eredu_nn::WorkspaceIsolatedCopy::new(
                value.clone(),
                context,
            ))?;
        }
        Ok(Self {
            paged: context.metadata_rc(paged)?,
            fixed,
            context: context.clone(),
        })
    }
    /// Actual selected geometry after successful appends/scans.
    pub fn geometry(&self) -> WorkspacePagedGeometry {
        self.paged.geometry()
    }
    pub(in crate::working_memory) fn workspace_context(&self) -> &WorkspaceContext {
        &self.context
    }
    pub(in crate::working_memory) fn validate_context(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !self.context.shares_trace(context) {
            return Err(
                context.metadata_error(format_args!("paged layer uses another metadata context"))
            );
        }
        Ok(())
    }
    fn paged_mut(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<&mut WorkspacePagedAppendState, Error> {
        self.validate_context(context)?;
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            &WorkspaceContext,
            WorkspacePagedAppendState,
            Rc<WorkspacePagedAppendState>,
        )>())?;
        if Rc::get_mut(&mut self.paged).is_none() {
            let copy = self.paged.copy_metadata(context)?;
            self.paged = context.metadata_rc(copy)?;
        }
        Ok(Rc::get_mut(&mut self.paged).expect("exclusive paged metadata"))
    }
    /// Retains immutable metadata and spending. Actual native manager history
    /// retention and copy receipts remain separate mechanisms.
    pub fn checkpoint_for_transaction(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        self.validate_context(context)?;
        context.charge_metadata(std::mem::size_of::<(
            &Self,
            &WorkspaceContext,
            Self,
            Result<Self, Error>,
        )>())?;
        Ok(self.clone())
    }
}
impl AttentionCache<WorkspaceTensor> for WorkspacePagedLayerState {
    fn uses_blockwise_attention(&self) -> bool {
        true
    }
    fn offset(&self) -> i32 {
        self.paged.geometry().offset as i32
    }
    fn max_size(&self) -> Option<i32> {
        self.paged.geometry().window
    }
    fn update_for_attention(
        &mut self,
        keys: WorkspaceTensor,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(WorkspaceTensor, WorkspaceTensor), Error> {
        self.validate_context(context)?;
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            WorkspaceTensor,
            WorkspaceTensor,
            &WorkspaceContext,
            (WorkspaceTensor, WorkspaceTensor),
            WorkspacePagedGeometry,
            [i32; 4],
            i32,
        )>())?;
        context.validate_values([&keys, &values])?;
        let shape = keys.shape();
        if shape.len() != 4 || shape[2] <= 0 || self.offset().checked_add(shape[2]).is_none() {
            return Err(
                context.metadata_error(format_args!("paged append exceeds tensor coordinate"))
            );
        }
        let submitted = (keys.clone(), values.clone());
        let values = if self.geometry().key_only {
            if values.shape() != [shape[0], shape[1], shape[2], 0]
                || values.layout().dtype() != keys.layout().dtype()
                || values.layout().dtype() != WorkspaceDtype::Float32
            {
                return Err(context.metadata_error(format_args!(
                    "paged key-only state requires a zero-width value input"
                )));
            }
            WorkspaceTensor::full_f32(0., &[shape[0], shape[1], shape[2], 1], context)?
        } else {
            values
        };
        self.paged_mut(context)?
            .append_normalized([keys, values], true, context)?;
        Ok(submitted)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            AttentionRequest<'_, WorkspaceTensor>,
            &WorkspaceContext,
            BlockwiseAttentionSpec<'_, WorkspaceTensor>,
            BlockwiseAttentionOptions,
            i64,
        )>())?;
        context.validate_values(
            [&request.queries, &request.keys, &request.values]
                .into_iter()
                .chain(request.mask)
                .chain(request.sinks),
        )?;
        let query_len = *request.queries.shape().get(2).ok_or_else(|| {
            context.metadata_error(format_args!("paged attention requires rank-four queries"))
        })?;
        let geometry = self.geometry();
        let spec = BlockwiseAttentionSpec {
            queries: &request.queries,
            scale: request.scale,
            mask: request.mask,
            query_start: geometry.offset - i64::from(query_len),
            context_end: geometry.offset,
            sliding_window: geometry.window,
            prefix_tokens: i64::from(geometry.prefix_tokens),
            sinks: request.sinks,
        };
        self.paged_mut(context)?.scan_attention(
            spec,
            BlockwiseAttentionOptions {
                arithmetic: request.arithmetic,
                softcap: request.softcap,
            },
            |_, _, _| Ok(None),
            context,
        )
    }
    fn relative_attention<B: NeuralBackend<Tensor = WorkspaceTensor>>(
        &mut self,
        _request: eredu_nn::RelativeAttentionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.validate_context(context)?;
        Err(context.metadata_error(format_args!(
            "paged relative attention requires its shared bias projection"
        )))
    }
}
/// Allocation-free borrowed traversal of sealed blocks, tail and fixed roles.
pub struct WorkspacePagedValues<'a> {
    state: &'a WorkspacePagedLayerState,
    block: usize,
    member: usize,
    tail: usize,
    fixed: std::iter::Flatten<std::slice::Iter<'a, Option<WorkspaceTensor>>>,
}
impl<'a> Iterator for WorkspacePagedValues<'a> {
    type Item = &'a WorkspaceTensor;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(block) = self.state.paged.blocks().get(self.block) {
            let Some(values) = block.tensor_values() else {
                self.member = 0;
                self.block += 1;
                continue;
            };
            let value = &values[self.member];
            self.member += 1;
            if self.member == 2 {
                self.member = 0;
                self.block += 1;
            }
            return Some(value);
        }
        if let Some(tail) = self.state.paged.tail() {
            if let Some(value) = tail.get(self.tail) {
                self.tail += 1;
                return Some(value);
            }
        }
        self.fixed.next()
    }
}
impl RuntimeLayerState<WorkspaceBackend> for WorkspacePagedLayerState {
    type RetainedValues<'a> = WorkspacePagedValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        WorkspacePagedValues {
            state: self,
            block: 0,
            member: 0,
            tail: 0,
            fixed: self.fixed.values().flatten(),
        }
    }
}
impl RuntimeStateComponents<WorkspaceBackend> for WorkspacePagedLayerState {
    fn position(&self) -> i32 {
        self.offset()
    }
    fn fixed_component(
        &mut self,
        role: StateTensorRole,
    ) -> Result<&mut Option<WorkspaceTensor>, StateError> {
        self.fixed
            .get_mut(&role, &self.context)
            .map_err(StateError::WorkspaceConstruction)?
            .ok_or(StateError::UnknownComponent { role })
    }
    fn advance_fixed(&mut self, _tokens: i32) -> Result<(), StateError> {
        Err(StateError::workspace_invalid_advance(
            &self.context,
            format_args!("paged attention advances through its block append"),
        ))
    }
}
impl AuxiliaryConvolutionState<WorkspaceTensor> for WorkspacePagedLayerState {
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<WorkspaceTensor>, Error> {
        let context = self.context.clone();
        self.fixed_component(StateTensorRole::Convolution { slot })
            .map_err(|cause| cause.into_workspace_error(&context))
    }
}
impl ResettableRuntimeLayerState<WorkspaceBackend> for WorkspacePagedLayerState {
    fn reset(&mut self) -> Result<(), StateError> {
        let context = &self.context;
        let mut fixed = self.fixed.clone();
        fixed
            .values_mut(context)
            .map_err(StateError::WorkspaceConstruction)?
            .fill(None);
        let mut paged = self
            .paged
            .copy_metadata(context)
            .map_err(StateError::WorkspaceConstruction)?;
        paged.reset_metadata();
        let paged = context
            .metadata_rc(paged)
            .map_err(|cause| StateError::WorkspaceConstruction(cause.into()))?;
        self.paged = paged;
        self.fixed = fixed;
        Ok(())
    }
}
