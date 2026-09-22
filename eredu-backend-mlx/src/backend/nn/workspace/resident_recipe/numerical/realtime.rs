//! The same retained native boundary source for one non-text frame.
use super::*;

impl SpeculativeNumericalRecipe {
    /// Adds the actual selected graph's finite group submissions and consumer
    /// waits to this complete numerical cut. This is descriptive native fit,
    /// never a model occurrence, frame admission or operation-bank grant.
    pub(crate) fn with_realtime_boundaries(
        mut self,
        submissions: usize,
        consumers: usize,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("realtime graph boundary source is incomplete"));
        let dispatch = self.completion.dispatch.ok_or_else(invalid)?;
        if self.completion.traversal.limits().streams
            != usize::from(dispatch.gpu_entries != 0) + usize::from(dispatch.cpu_entries != 0)
            || (dispatch.gpu_entries == 0 && dispatch.cpu_model.is_none())
        {
            return Err(invalid());
        }
        let old_graph = graph_capacity::ResidentGraphStorage::for_completion(self.completion)
            .ok_or_else(invalid)?;
        self.completion.nested_completions = self
            .completion
            .nested_completions
            .checked_add(submissions)
            .ok_or_else(invalid)?;
        let waits = submissions.checked_mul(consumers).ok_or_else(invalid)?;
        let graph = graph_capacity::ResidentGraphStorage::for_completion(self.completion)
            .ok_or_else(invalid)?;
        let records = record_capacity::ResidentRecordStorage::for_completion_with_waits(
            self.completion,
            waits,
        )
        .ok_or_else(invalid)?;
        self.graph_capacity =
            usize::try_from(graph.full_capacity.ok_or_else(invalid)?).map_err(|_| invalid())?;
        self.record_capacity =
            usize::try_from(records.full_capacity.ok_or_else(invalid)?).map_err(|_| invalid())?;
        self.controls = self
            .controls
            .checked_sub(old_graph.control_bytes().ok_or_else(invalid)?)
            .and_then(|n| n.checked_add(graph.control_bytes()?))
            .ok_or_else(invalid)?;
        let frames = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<ResidentDispatchPopulation>(),
            std::mem::size_of::<graph_capacity::ResidentGraphStorage>() * 2,
            std::mem::size_of::<record_capacity::ResidentRecordStorage>(),
            std::mem::size_of::<(usize, usize, usize, &WorkspaceContext)>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or_else(invalid)?;
        context.charge_metadata(controls)?;
        self.controls = self
            .controls
            .checked_add(u64::try_from(controls).map_err(|_| invalid())?)
            .ok_or_else(invalid)?;
        Ok(self)
    }
}
