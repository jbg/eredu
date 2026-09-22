//! Metadata uses the same transient visibility and append order as native pooling.
use super::*;
use crate::cache::{
    PagedLocalUpdateMechanisms, PagedLocalUpdatePlan, PagedVisibleError, PagedVisibleMechanisms,
    PagedVisiblePlan,
};
use std::{mem::size_of, ops::Range};

impl WorkspacePagedAppendState {
    /// Returns the preceding local window plus all new input, while retaining
    /// independent paged blocks and tail. Source and native authority stay with
    /// their separate owners; accepted trace work is never refunded on failure.
    pub fn update_visible_normalized(
        &mut self,
        input: [WorkspaceTensor; 2],
        context: &WorkspaceContext,
    ) -> Result<[WorkspaceTensor; 2], Error> {
        context.charge_metadata(
            PagedLocalUpdatePlan::control_bytes::<Update<'_>>()
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        &mut Self,
                        [WorkspaceTensor; 2],
                        &WorkspaceContext,
                        i32,
                        PagedLocalUpdatePlan,
                        (i64, i64),
                        Result<PagedLocalUpdatePlan, PagedVisibleError>,
                    )>())
                })
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if !context.shares_trace(&self.context) {
            return Err(
                context.metadata_error(format_args!("paged visibility uses another context"))
            );
        }
        let tokens = *input[0]
            .shape()
            .get(2)
            .ok_or_else(|| context.metadata_source(PagedVisibleError::Extent))?;
        validate_pair(&self.geometry, &input, i64::from(tokens), context)?;
        let window = self
            .geometry
            .window
            .ok_or_else(|| context.metadata_source(PagedVisibleError::Extent))?;
        let plan = PagedLocalUpdatePlan::new(self.geometry.offset, window, tokens)
            .map_err(|cause| context.metadata_source(cause))?;
        let invocation = (
            self.geometry.offset,
            self.geometry
                .offset
                .checked_add(i64::from(tokens))
                .ok_or(WorkspaceMetadataError::Overflow)?,
        );
        self.prepare_host_sources(invocation.0, invocation.1, context)?;
        plan.run(&mut Update {
            invocation,
            state: self,
            input: Some(input),
            context,
        })
    }
}
struct Visible<'a> {
    state: &'a mut WorkspacePagedAppendState,
    plan: PagedVisiblePlan,
    invocation: (i64, i64),
    keys: Vec<WorkspaceTensor>,
    values: Vec<WorkspaceTensor>,
    context: &'a WorkspaceContext,
}
impl Visible<'_> {
    fn push(&mut self, pair: &[WorkspaceTensor; 2], range: Range<i32>) -> Result<(), Error> {
        self.context.charge_metadata(size_of::<(
            &Self,
            &[WorkspaceTensor; 2],
            Range<i32>,
            [Index; 4],
            Result<WorkspaceTensor, Error>,
        )>())?;
        let axes = [
            Index::Full,
            Index::Full,
            Index::Range(range.start, range.end),
            Index::Full,
        ];
        let keys = pair[0].index(&axes, self.context)?;
        self.context.retain_values(&[&keys])?;
        self.keys.push(keys);
        let values = pair[1].index(&axes, self.context)?;
        self.context.retain_values(&[&values])?;
        self.values.push(values);
        Ok(())
    }
}
impl PagedVisibleMechanisms for Visible<'_> {
    type Cursor = usize;
    type Block = usize;
    type Output = [WorkspaceTensor; 2];
    type Error = Error;
    fn error(&self, cause: PagedVisibleError) -> Error {
        self.context.metadata_source(cause)
    }
    fn open_blocks(&mut self, _: PagedVisiblePlan) -> Result<usize, Error> {
        Ok(0)
    }
    fn next_block(&mut self, cursor: &mut usize) -> Result<Option<usize>, Error> {
        while let Some(block) = self.state.blocks.get(*cursor) {
            let index = *cursor;
            *cursor += 1;
            if self.plan.selects(block.start, block.end) {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
    fn block_range(&self, index: &usize) -> (i64, i64) {
        let b = &self.state.blocks[*index];
        (b.start, b.end)
    }
    fn push_block(&mut self, index: &usize, range: Range<i32>) -> Result<(), Error> {
        let pair = self.state.load_block_for_invocation(
            *index,
            self.invocation.0,
            self.invocation.1,
            0,
            self.context,
        )?;
        self.push(&pair, range)
    }
    fn submit_block(&mut self) -> Result<(), Error> {
        if self.context.completion_strategy()
            == eredu_nn::workspace::WorkspaceCompletionStrategy::OperationSubmissions
        {
            return Ok(());
        }
        self.context.charge_metadata(size_of::<(
            &mut Self,
            [&WorkspaceTensor; 2],
            Result<(), Error>,
        )>())?;
        let pair = [
            self.keys
                .last()
                .ok_or(WorkspaceMetadataError::Unqualified)?,
            self.values
                .last()
                .ok_or(WorkspaceMetadataError::Unqualified)?,
        ];
        self.context.complete_values(&pair)
    }
    fn tail_range(&self) -> Option<(i64, i64)> {
        self.state
            .tail
            .as_ref()
            .map(|_| (self.state.geometry.tail_start, self.state.geometry.offset))
    }
    fn push_tail(&mut self, range: Range<i32>) -> Result<(), Error> {
        self.context
            .charge_metadata(size_of::<[WorkspaceTensor; 2]>())?;
        let pair = self
            .state
            .tail
            .as_ref()
            .expect("validated independent tail")
            .clone();
        self.push(&pair, range)
    }
    fn finish(&mut self, tokens: i64) -> Result<Self::Output, Error> {
        self.context.charge_metadata(size_of::<(
            &mut Self,
            i64,
            [WorkspaceTensor; 2],
            Result<WorkspaceTensor, Error>,
        )>())?;
        if self.keys.is_empty() {
            return Err(self.error(PagedVisibleError::History));
        }
        let keys = if self.keys.len() == 1 {
            self.keys[0].clone()
        } else {
            WorkspaceTensor::concatenate(&self.keys, -2, self.context)?
        };
        self.context.retain_values(&[&keys])?;
        let values = if self.values.len() == 1 {
            self.values[0].clone()
        } else {
            WorkspaceTensor::concatenate(&self.values, -2, self.context)?
        };
        self.context.retain_values(&[&values])?;
        if i64::from(keys.shape()[2]) != tokens || i64::from(values.shape()[2]) != tokens {
            return Err(self.error(PagedVisibleError::History));
        }
        Ok([keys, values])
    }
}
struct Update<'a> {
    invocation: (i64, i64),
    state: &'a mut WorkspacePagedAppendState,
    input: Option<[WorkspaceTensor; 2]>,
    context: &'a WorkspaceContext,
}
impl PagedLocalUpdateMechanisms for Update<'_> {
    type Pair = [WorkspaceTensor; 2];
    type Error = Error;
    fn visible(&mut self, plan: PagedVisiblePlan) -> Result<Self::Pair, Error> {
        self.context.charge_metadata(
            PagedVisiblePlan::control_bytes::<Visible<'_>>()
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        &Self,
                        PagedVisiblePlan,
                        usize,
                        &WorkspacePagedBlock,
                        Option<&[WorkspaceTensor; 2]>,
                    )>())
                })
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut count = 0usize;
        for block in &self.state.blocks {
            if plan.selects(block.start, block.end) {
                count = count
                    .checked_add(1)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        if self.state.tail.is_some()
            && plan.selects(self.state.geometry.tail_start, self.state.geometry.offset)
        {
            count = count
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        let keys = self.context.metadata_vec(count)?;
        let values = self.context.metadata_vec(count)?;
        plan.run(&mut Visible {
            invocation: self.invocation,
            state: self.state,
            plan,
            keys,
            values,
            context: self.context,
        })
    }
    fn join_input(&mut self, [past_keys, past_values]: Self::Pair) -> Result<Self::Pair, Error> {
        self.context.charge_metadata(size_of::<(
            &mut Self,
            [WorkspaceTensor; 2],
            [WorkspaceTensor; 2],
            Result<WorkspaceTensor, Error>,
        )>())?;
        let [keys, values] = self.input.as_ref().expect("input precedes append");
        let joined_keys =
            WorkspaceTensor::concatenate(&[past_keys, keys.clone()], -2, self.context)?;
        self.context.retain_values(&[&joined_keys])?;
        let joined_values =
            WorkspaceTensor::concatenate(&[past_values, values.clone()], -2, self.context)?;
        self.context.retain_values(&[&joined_values])?;
        Ok([joined_keys, joined_values])
    }
    fn copy_input(&mut self) -> Result<Self::Pair, Error> {
        let pair = self.input.as_ref().expect("input precedes append").clone();
        self.context.retain_values(&[&pair[0]])?;
        self.context.retain_values(&[&pair[1]])?;
        Ok(pair)
    }
    fn prepare_append(&mut self, visible: &Self::Pair) -> Result<(), Error> {
        if self.context.completion_strategy()
            == eredu_nn::workspace::WorkspaceCompletionStrategy::OperationSubmissions
        {
            return Ok(());
        }
        self.context.charge_metadata(size_of::<(
            &mut Self,
            &[WorkspaceTensor; 2],
            Result<(), Error>,
        )>())?;
        self.context.complete_values(&[&visible[0], &visible[1]])
    }
    fn append(&mut self) -> Result<(), Error> {
        self.state
            .append_normalized(self.input.take().expect("one append"), false, self.context)
    }
}
