//! Exact projected block traversal through the existing workspace accumulator.
use super::*;
use crate::cache::{PagedScanMechanisms, PagedScanPlan};
use eredu_nn::{
    workspace::{WorkspaceBackend, WorkspaceBlockwiseAccumulator},
    BlockwiseAttentionBackend, BlockwiseAttentionOptions, BlockwiseAttentionSpec,
};

impl WorkspacePagedAppendState {
    /// Scans actual retained blocks and the independent tail using the same
    /// pass/lease-order driver as the native cache. The caller supplies any
    /// relative bias through its existing source-qualified tensor producer;
    /// returning `None` explicitly selects no extra bias for that block.
    ///
    /// This records numerical work and does not create native manager, source,
    /// completion or execution authority. Accepted host promotions remain in
    /// the state on later failure, matching their completed native publication;
    /// already accepted trace work remains spent.
    pub fn scan_attention<F>(
        &mut self,
        spec: BlockwiseAttentionSpec<'_, WorkspaceTensor>,
        options: BlockwiseAttentionOptions,
        bias: F,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error>
    where
        F: FnMut(i64, i64, &WorkspaceContext) -> Result<Option<WorkspaceTensor>, Error>,
    {
        context.charge_metadata(
            PagedScanPlan::control_bytes::<Scan<'_, F>>()
                .and_then(|bytes| {
                    bytes.checked_add(std::mem::size_of::<(
                        &mut Self,
                        BlockwiseAttentionSpec<'_, WorkspaceTensor>,
                        BlockwiseAttentionOptions,
                        &WorkspaceContext,
                        Option<&i32>,
                        i32,
                        PagedScanPlan,
                    )>())
                })
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if !context.shares_trace(&self.context) {
            return Err(context.metadata_error(format_args!("paged scan uses another context")));
        }
        if self.geometry.key_only {
            return Err(context.metadata_error(format_args!(
                "key-only paged cache requires architecture-owned attention"
            )));
        }
        let query_len = *spec.queries.shape().get(2).ok_or_else(|| {
            context.metadata_error(format_args!("paged scan requires rank-four queries"))
        })?;
        let plan = PagedScanPlan::new(
            self.geometry.offset,
            query_len,
            self.geometry.window,
            i64::from(self.geometry.prefix_tokens),
            options,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        if spec.context_end != self.geometry.offset
            || spec.query_start != plan.query_start()
            || spec.sliding_window != self.geometry.window
            || spec.prefix_tokens != i64::from(self.geometry.prefix_tokens)
        {
            return Err(context.metadata_error(format_args!(
                "paged scan policy differs from retained source"
            )));
        }
        // A finite conditional transfer inventory covers every possible policy
        // victim. Native execution selects through the existing manager policy;
        // it may skip these source-qualified stores and loads for hot rows.
        self.prepare_host_sources(plan.query_start(), self.geometry.offset, context)?;
        let accumulator =
            WorkspaceBackend::begin_blockwise_attention_with_options(spec, options, context)?;
        plan.run(&mut Scan {
            state: self,
            accumulator: Some(accumulator),
            plan,
            pass: 0,
            bias,
            context,
        })
    }
}
struct Scan<'a, F> {
    state: &'a mut WorkspacePagedAppendState,
    accumulator: Option<WorkspaceBlockwiseAccumulator>,
    plan: PagedScanPlan,
    pass: usize,
    bias: F,
    context: &'a WorkspaceContext,
}
impl<F> Scan<'_, F>
where
    F: FnMut(i64, i64, &WorkspaceContext) -> Result<Option<WorkspaceTensor>, Error>,
{
    fn prepare_block_controls(&self) -> Result<(), Error> {
        // Callback arguments, shallow source handles and the actual optional
        // bias/result coexist with the shared traversal/accumulator controls.
        self.context.charge_metadata(std::mem::size_of::<(
            &Self,
            i64,
            i64,
            [WorkspaceTensor; 2],
            Option<WorkspaceTensor>,
            Result<Option<WorkspaceTensor>, Error>,
            Result<u64, Error>,
            Result<(), Error>,
        )>())?;
        Ok(())
    }
    fn accumulate(
        &mut self,
        start: i64,
        end: i64,
        [keys, values]: [WorkspaceTensor; 2],
    ) -> Result<(), Error> {
        let bias = (self.bias)(start, end, self.context)?;
        WorkspaceBackend::accumulate_blockwise_attention_with_bias(
            self.accumulator.as_mut().expect("scan has not finished"),
            start,
            end,
            keys,
            values,
            bias.as_ref(),
            self.context,
        )?;
        Ok(())
    }
}
impl<F> PagedScanMechanisms for Scan<'_, F>
where
    F: FnMut(i64, i64, &WorkspaceContext) -> Result<Option<WorkspaceTensor>, Error>,
{
    type Cursor = usize;
    type Block = usize;
    type Output = WorkspaceTensor;
    type Error = Error;
    fn begin_pass(&mut self, pass: usize) -> Result<(), Error> {
        self.pass = pass;
        if pass == 1 {
            WorkspaceBackend::begin_blockwise_value_pass(
                self.accumulator.as_mut().expect("scan has not finished"),
                self.context,
            )?;
        }
        Ok(())
    }
    fn open_blocks(&mut self) -> Result<usize, Error> {
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
    fn consume_block(&mut self, index: &usize) -> Result<(), Error> {
        self.prepare_block_controls()?;
        let values = self.state.load_block_for_invocation(
            *index,
            self.plan.query_start(),
            self.state.geometry.offset,
            self.pass,
            self.context,
        )?;
        let block = &self.state.blocks[*index];
        let (start, end) = (block.start, block.end);
        self.accumulate(start, end, values)
    }
    fn submit(&mut self) -> Result<(), Error> {
        // The existing blockwise descriptors carry numerical frontiers. A
        // metadata traversal cannot manufacture a native completion receipt.
        Ok(())
    }
    fn consume_tail(&mut self) -> Result<(), Error> {
        if self.state.tail.is_some() {
            self.prepare_block_controls()?;
        }
        if let Some(values) = &self.state.tail {
            self.accumulate(
                self.state.geometry.tail_start,
                self.state.geometry.offset,
                values.clone(),
            )?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<WorkspaceTensor, Error> {
        let output = WorkspaceBackend::finish_blockwise_attention(
            self.accumulator.take().expect("one shared scan finish"),
            self.context,
        )?;
        self.context.retain_values(&[&output])?;
        self.context.complete_cache_scan(&output)?;
        self.state.discard_after_attention(self.context)?;
        Ok(output)
    }
}
