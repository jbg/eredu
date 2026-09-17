use super::*;
use eredu_nn::{PoolingAttentionCache, PoolingOverlap, PoolingWindows};

impl WorkspaceResidentLayerState {
    fn pooling(&self) -> Result<&WorkspacePoolingLayerState, Error> {
        match self {
            Self::Pooling(state) => Ok(state),
            _ => Err(Error::backend("layer has no pooling attention state")),
        }
    }
    fn pooling_mut(&mut self) -> Result<&mut WorkspacePoolingLayerState, Error> {
        match self {
            Self::Pooling(state) => Ok(state),
            _ => Err(Error::backend("layer has no pooling attention state")),
        }
    }
}

impl PoolingAttentionCache<WorkspaceTensor> for WorkspaceResidentLayerState {
    type Checkpoint = WorkspacePoolingLayerState;
    fn offset(&self) -> i32 {
        self.position()
    }
    fn pooling_ratio(&self, stream: u32) -> Option<i32> {
        self.pooling().ok()?.pooling_ratio(stream)
    }
    fn append_local(
        &mut self,
        keys: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.pooling_mut()?.append_local(keys, context)
    }
    fn local_mask(
        &self,
        queries: i32,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.pooling()?.local_mask(queries, offset, context)
    }
    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        gates: WorkspaceTensor,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<PoolingWindows<WorkspaceTensor>, Error> {
        self.pooling_mut()?
            .accumulate_pooling_windows(stream, values, gates, offset, context)
    }
    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        gates: WorkspaceTensor,
    ) -> Result<PoolingOverlap<WorkspaceTensor>, Error> {
        self.pooling_mut()?
            .replace_pooling_overlap(stream, values, gates)
    }
    fn append_pooled(
        &mut self,
        stream: u32,
        values: WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        self.pooling_mut()?.append_pooled(stream, values, context)
    }
    fn pooling_mask(
        &self,
        stream: u32,
        queries: i32,
        offset: i32,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.pooling()?
            .pooling_mask(stream, queries, offset, context)
    }
    fn checkpoint(&self) -> Result<Self::Checkpoint, Error> {
        self.pooling()?.checkpoint()
    }
    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        self.pooling_mut()?.restore(checkpoint, context)
    }
    fn finalize(&mut self) -> Result<(), Error> {
        self.pooling_mut()?.finalize()
    }
    fn clear(&mut self) -> Result<(), Error> {
        self.pooling_mut()?.clear()
    }
}
