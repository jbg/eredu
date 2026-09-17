//! Exact ordinary logical subgroup pair plus its completed arithmetic stages.
use super::*;
use crate::backend::nn::logical_collective::{self, Operations};
use crate::backend::runtime::distributed::topology::original_source::OwnedOriginalExchangeLayoutRound;
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LogicalCollectiveKind {
    Sum,
    Gather,
}
pub(crate) struct LogicalCollectiveQuote {
    pub(crate) group: eredu_core::CollectiveGroupId,
    pub(crate) order: usize,
    pub(crate) rank: usize,
    pub(crate) kind: LogicalCollectiveKind,
    pub(crate) protocol: eredu_runtime::CommunicationOperation,
    pub(crate) input: WorkspaceLayout,
    pub(crate) native_dtype: safemlx::Dtype,
    pub(crate) stages: LogicalCollectiveStages,
    pub(crate) output: u64,
    pub(crate) scratch: u64,
    source: OriginalParallelSource,
}
pub(crate) enum LogicalCollectiveStages {
    Exchange {
        pairs: Vec<OwnedOriginalExchangeLayoutRound>,
        peer_recipe: SpeculativeNumericalRecipe,
        peer_capacity: BoundaryStageCapacity,
        result_recipe: SpeculativeNumericalRecipe,
        result_capacity: BoundaryStageCapacity,
    },
    Packed(packed::PackedWorldQuote),
    Routed(routed::RoutedCollectiveQuote),
    RoutedPeer { value: usize, step: usize, pairs: [OwnedOriginalExchangeLayoutRound; 1],
        peer_recipe: SpeculativeNumericalRecipe, peer_capacity: BoundaryStageCapacity },
}
mod packed;
mod routed;
impl LogicalCollectiveQuote {
    #[inline(never)]
    pub(crate) fn prepare(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<Option<Self>, Error> {
        if source.logical_group_id().is_none()
            || !matches!(
                operation.kind,
                WorkspaceOperationKindView::Collective(
                    WorkspaceCollectiveView::Sum { .. }
                        | WorkspaceCollectiveView::GatherFirstAxis { .. }
                )
            )
        {
            return Ok(None);
        }
        source
            .funding()
            .reserve_metadata(size_of::<(
                Self,
                Option<Self>,
                Result<Option<Self>, Error>,
                WorkspaceContext,
                WorkspaceTraceReport,
                SpeculativeNumericalRecipe,
                [WorkspaceTensor; 3],
            )>())
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let Some((order, input, dtype, native)) = source
            .logical_collective_layout(operation)
            .map_err(|cause| source.neural_error(cause))?
        else {
            return Ok(None);
        };
        let kind = match native {
            safemlx::distributed::GroupWorkerOperation::Sum => LogicalCollectiveKind::Sum,
            safemlx::distributed::GroupWorkerOperation::Gather => LogicalCollectiveKind::Gather,
            _ => return Err(source.neural_error(crate::backend::error::Error::PrefillScopeUnavailable)),
        };
        Self::prepare_group(source, order, kind, input, dtype, mechanism).map(Some)
    }
    /// Shared selected-group producer for model occurrences and completed count
    /// sources. The retained manifest supplies group identity and ordinary plan.
    pub(crate) fn prepare_group(source: &OriginalParallelSource, order: usize,
        kind: LogicalCollectiveKind, input: eredu_nn::workspace::WorkspaceLayoutView<'_>,
        dtype: safemlx::Dtype, mechanism: ResidentExecutionMechanisms) -> Result<Self, Error> {
        let protocol=match kind{LogicalCollectiveKind::Sum=>eredu_runtime::CommunicationOperation::AllReduceSum,
            LogicalCollectiveKind::Gather=>eredu_runtime::CommunicationOperation::AllGatherUneven};
        Self::prepare_group_protocol(source,order,kind,input,dtype,mechanism,protocol)
    }
    pub(crate) fn prepare_agreement(source:&OriginalParallelSource,order:usize,
        input:eredu_nn::workspace::WorkspaceLayoutView<'_>,mechanism:ResidentExecutionMechanisms)->Result<Self,Error>{
        let context=WorkspaceContext::new_with_metadata_funding(mechanism,source.funding().clone())?;
        context.charge_metadata(size_of::<(WorkspaceContext,usize,Result<Self,Error>)>())?;
        let invalid=||context.metadata_error(format_args!("provider agreement differs from its selected control declaration"));
        let actual=source.communication_source().map_err(|cause|source.neural_error(cause))?;
        let descriptor=actual.group(order).ok_or_else(invalid)?.1;
        let selected=actual.source().manifest().select_group_operation(descriptor.id(),eredu_runtime::CommunicationOperation::FailureAgreement)
            .map_err(|cause|context.metadata_source(cause))?;
        if selected.order()!=order||input.shape()!=[1]||input.dtype()!=WorkspaceDtype::Int32 {return Err(invalid());}
        Self::prepare_group_protocol(source,order,LogicalCollectiveKind::Sum,input,safemlx::Dtype::Int32,
            mechanism,eredu_runtime::CommunicationOperation::FailureAgreement)
    }
    fn prepare_group_protocol(source:&OriginalParallelSource,order:usize,kind:LogicalCollectiveKind,
        input:eredu_nn::workspace::WorkspaceLayoutView<'_>,dtype:safemlx::Dtype,
        mechanism:ResidentExecutionMechanisms,protocol:eredu_runtime::CommunicationOperation)->Result<Self,Error>{
        source.funding().reserve_metadata(size_of::<(Self, Result<Self, Error>,
            WorkspaceContext, WorkspaceTraceReport, SpeculativeNumericalRecipe, [WorkspaceTensor; 3])>())
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        let invalid = || context.metadata_error(format_args!(
            "logical collective differs from its exact selected subgroup"));
        let actual = source.communication_source().map_err(|cause| source.neural_error(cause))?;
        let (selected, descriptor, _) = actual.group(order).ok_or_else(invalid)?;
        if !selected.is_logical() || descriptor.local_index() != Some(selected.rank()) {
            return Err(invalid());
        }
        let group = descriptor.id();
        let layout = context.layout(input.shape(), input.dtype())?
            .with_representation(input.representation());
        if selected.logical_routed_plan().map_err(|_| invalid())?.is_some() {
            let value = routed::RoutedCollectiveQuote::prepare(source, &actual, order, kind,
                &layout, dtype, mechanism, protocol, &context)?;
            let output = u64::try_from(value.result_capacity.backing).map_err(|_| invalid())?;
            let scratch = u64::try_from(value.scratch().ok_or_else(invalid)?).map_err(|_| invalid())?;
            return Ok(Self { group, order, rank: selected.rank(), kind, protocol, input: layout,
                native_dtype: dtype, stages: LogicalCollectiveStages::Routed(value),
                output, scratch, source: source.clone() });
        }
        let Some(exchange)=actual.group_exchange(order).map_err(|cause|source.neural_error(cause))? else {
            let value=packed::PackedWorldQuote::prepare(source,&actual,order,kind,&layout,dtype,mechanism,&context)?;
            // Sum extracts a view: its whole packed world backing remains live
            // after this logical operation, even though the view is smaller.
            let retained=match kind {LogicalCollectiveKind::Sum=>value.world.backing_capacity().max(value.result_capacity.backing),
                LogicalCollectiveKind::Gather=>value.result_capacity.backing};
            let output=u64::try_from(retained).map_err(|_|invalid())?;
            let scratch=value.scratch().and_then(|n|u64::try_from(n).ok()).ok_or_else(invalid)?;
            let rank=actual.group(order).ok_or_else(invalid)?.0.rank();
            return Ok(Self {group,order,rank,kind,protocol,input:layout,native_dtype:dtype,
                stages:LogicalCollectiveStages::Packed(value),output,scratch,source:source.clone()});
        };
        let rank=exchange.group().rank();
        let mut pairs=context.metadata_vec(exchange.rounds())?;
        for round in 0..exchange.rounds() {
            pairs.push(exchange.round_layout_storage(&actual,round,input.shape(),dtype)
                .map_err(|cause|source.neural_error(cause))?.try_into_owned(source)
                .map_err(|cause|source.neural_error(cause))?);
        }
        let prototype = WorkspaceTensor::existing(layout.clone(), &context)?;
        // Both accepted output leaves are separate in the conservative source.
        // Actual aliasing can reduce this population but never add another input.
        let sent = WorkspaceTensor::existing(layout.clone(), &context)?;
        let received = WorkspaceTensor::existing(layout.clone(), &context)?;
        context.begin_span();
        let ops = logical_collective::Workspace(&context);
        let zero = ops.zero(&sent)?;
        let peer = logical_collective::peer(&ops, &sent, &received, &zero)?;
        let peer_report = context.finish_report(&[peer.clone()])?;
        let peer_recipe =
            super::numerical(&peer_report, 1, mechanism, &context)?;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let peer_capacity = boundary::capacity(peer_recipe, runtime, &context)?;
        // Ordinary execution completes the dependency arithmetic before the
        // final sum/stack. Retain that boundary instead of fusing two stages.
        let peer = WorkspaceTensor::existing(layout.clone(), &context)?;
        context.begin_span();
        let result = match kind {
            LogicalCollectiveKind::Sum => logical_collective::sum(&ops, &prototype, &peer)?,
            LogicalCollectiveKind::Gather => {
                logical_collective::gather(&ops, &prototype, &peer, rank == 0)?
            }
        };
        let result_report = context.finish_report(&[result])?;
        let result_recipe =
            super::numerical(&result_report, 1, mechanism, &context)?;
        let result_capacity = boundary::capacity(result_recipe, runtime, &context)?;
        let output = u64::try_from(result_capacity.backing).map_err(|_| invalid())?;
        let scratch=pairs.iter().try_fold(0usize,|total,pair|
            total.checked_add(pair.backing_capacity())?.checked_add(peer_capacity.backing))
            .and_then(|n|u64::try_from(n).ok()).ok_or_else(invalid)?;
        Ok(Self {
            group,
            order,
            rank,
            kind, protocol,
            input: layout,
            native_dtype: dtype,
            stages: LogicalCollectiveStages::Exchange { pairs,peer_recipe,peer_capacity,result_recipe,result_capacity },
            output,
            scratch,
            source: source.clone(),
        })
    }
    /// One actual explicit-route dependency stage. Its owning model or count
    /// source supplies final assembly after the completed peer row.
    pub(crate) fn prepare_routed_peer(source: &OriginalParallelSource, order: usize,
        value: usize, step: usize, input: eredu_nn::workspace::WorkspaceLayoutView<'_>,
        dtype: safemlx::Dtype, mechanism: ResidentExecutionMechanisms) -> Result<Self, Error> {
        Self::prepare_routed_peer_protocol(source,order,value,step,input,dtype,mechanism,
            eredu_runtime::CommunicationOperation::AllGatherUneven)
    }
    fn prepare_routed_peer_protocol(source:&OriginalParallelSource,order:usize,value:usize,step:usize,
        input:eredu_nn::workspace::WorkspaceLayoutView<'_>,dtype:safemlx::Dtype,
        mechanism:ResidentExecutionMechanisms,protocol:eredu_runtime::CommunicationOperation)->Result<Self,Error>{
        source.funding().reserve_metadata(size_of::<(Self, Result<Self, Error>,
            WorkspaceContext, WorkspaceTraceReport, SpeculativeNumericalRecipe, [WorkspaceTensor; 3])>())
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        let invalid = || context.metadata_error(format_args!("routed count peer differs from its retained itinerary"));
        let actual = source.communication_source().map_err(|cause| source.neural_error(cause))?;
        let (group, descriptor, _) = actual.group(order).ok_or_else(invalid)?;
        if descriptor.local_index() != Some(group.rank()) {
            return Err(invalid());
        }
        let pair = actual.group_routed_exchange_layout(order, value, step, input.shape(), dtype)
            .map_err(|cause| source.neural_error(cause))?.ok_or_else(invalid)?
            .try_into_owned(source).map_err(|cause| source.neural_error(cause))?;
        let layout = context.layout(input.shape(), input.dtype())?.with_representation(input.representation());
        let sent = WorkspaceTensor::existing(layout.clone(), &context)?;
        let received = WorkspaceTensor::existing(layout.clone(), &context)?;
        context.begin_span();
        let ops = logical_collective::Workspace(&context);
        let zero = ops.zero(&sent)?;
        let peer = logical_collective::peer(&ops, &sent, &received, &zero)?;
        let report = context.finish_report(&[peer])?;
        let peer_recipe = super::numerical(&report, 1, mechanism, &context)?;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let peer_capacity = boundary::capacity(peer_recipe, runtime, &context)?;
        let output = u64::try_from(peer_capacity.backing).map_err(|_| invalid())?;
        let scratch = u64::try_from(pair.backing_capacity()).map_err(|_| invalid())?;
        Ok(Self { group: descriptor.id(), order, rank: group.rank(), kind: LogicalCollectiveKind::Gather, protocol,
            input: layout, native_dtype: dtype, stages: LogicalCollectiveStages::RoutedPeer {
                value, step, pairs: [pair], peer_recipe, peer_capacity }, output, scratch, source: source.clone() })
    }
    pub(crate) fn routed(&self) -> Option<&routed::RoutedCollectiveQuote> {
        match &self.stages { LogicalCollectiveStages::Routed(value) => Some(value), _ => None }
    }
    pub(crate) fn routed_step(&self) -> Option<(usize, usize)> {
        match &self.stages { LogicalCollectiveStages::RoutedPeer { value, step, .. } => Some((*value, *step)), _ => None }
    }
    pub(crate) fn matches_source(&self, source: &OriginalParallelSource) -> bool {
        self.source.same_source(source)
    }
    pub(crate) fn packed(&self)->Option<&packed::PackedWorldQuote>{
        match &self.stages {LogicalCollectiveStages::Packed(value)=>Some(value),_=>None}
    }
    pub(crate) fn exchange_pairs(&self)->Option<&[OwnedOriginalExchangeLayoutRound]>{
        match &self.stages {LogicalCollectiveStages::Exchange{pairs,..}=>Some(pairs),
            LogicalCollectiveStages::RoutedPeer{pairs,..}=>Some(pairs.as_slice()),_=>None}
    }
    pub(crate) fn arithmetic_capacity(&self,peer:bool)->Option<BoundaryStageCapacity>{
        match &self.stages {LogicalCollectiveStages::Exchange{peer_capacity,result_capacity,..}=>
            Some(if peer{*peer_capacity}else{*result_capacity}),
            LogicalCollectiveStages::RoutedPeer{peer_capacity,..} if peer=>Some(*peer_capacity),_=>None}
    }
    pub(crate) fn maximum_backing_births(&self) -> Option<usize> {
        match &self.stages {
            LogicalCollectiveStages::Exchange{pairs,peer_recipe,result_recipe,..}=>pairs.iter().try_fold(
                result_recipe.storage.maximum_births(),|total,pair|total.checked_add(pair.maximum_backing_births()?)?
                    .checked_add(peer_recipe.storage.maximum_births())),
            LogicalCollectiveStages::Packed(value)=>value.maximum_births(),
            LogicalCollectiveStages::Routed(value)=>value.maximum_births(),
            LogicalCollectiveStages::RoutedPeer { pairs, peer_recipe, .. } => pairs.iter().try_fold(
                peer_recipe.storage.maximum_births(), |total, pair| total.checked_add(pair.maximum_backing_births()?)),
        }
    }
}
