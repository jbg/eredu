//! Cold source of the same native frame encoder, logical pair and decoder.
use super::*;
use crate::backend::nn::boundary_frame;
use crate::backend::runtime::distributed::topology::original_source::OwnedOriginalRouteLayoutRound;
use eredu_nn::Tensor;
use eredu_runtime::{CommunicationRouteId, CommunicationTensorMetadata};
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundaryStageCapacity {
    pub(crate) graph: usize,
    pub(crate) records: usize,
    pub(crate) backing: usize,
}
/// Every recipe belongs to this actual source/role/shape. Holding it allocates
/// no native tensor, reserves no role and makes no completion claim.
pub(crate) struct PipelineBoundaryQuote {
    pub(crate) route: CommunicationRouteId,
    pub(crate) ordinal: usize,
    pub(crate) header_bytes: usize,
    pub(crate) input: WorkspaceLayout,
    pub(crate) receiving: bool,
    pub(crate) encoding: SpeculativeNumericalRecipe,
    pub(crate) decoding: Option<SpeculativeNumericalRecipe>,
    pub(crate) encode_capacity: BoundaryStageCapacity,
    pub(crate) decode_capacity: Option<BoundaryStageCapacity>,
    pub(crate) rounds: Vec<OwnedOriginalRouteLayoutRound>,
    pub(crate) output: Option<u64>,
    pub(crate) scratch: u64,
    source: OriginalParallelSource,
}
impl PipelineBoundaryQuote {
    #[inline(never)]
    pub(crate) fn prepare(
        source: &OriginalParallelSource,
        operation: WorkspaceOperationView<'_>,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<Self, Error> {
        let funding = source.funding();
        funding
            .reserve_metadata(size_of::<(
                Self,
                Result<Self, Error>,
                WorkspaceContext,
                WorkspaceTraceReport,
                SpeculativeNumericalRecipe,
                BoundaryStageCapacity,
            )>())
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "pipeline boundary differs from its retained route source"
            ))
        };
        let WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary {
            route,
            ordinal,
            header_bytes,
        }) = operation.kind
        else {
            return Err(invalid());
        };
        if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
            return Err(invalid());
        }
        let input = operation.inputs.get(0).ok_or_else(invalid)?;
        if input != operation.outputs.get(0).ok_or_else(invalid)? {
            return Err(invalid());
        }
        let route = CommunicationRouteId::new(route);
        let actual = source
            .communication_source()
            .map_err(|cause| source.neural_error(cause))?;
        let order = actual
            .source()
            .manifest()
            .routes()
            .iter()
            .position(|value| value.id() == route)
            .ok_or_else(invalid)?;
        let descriptor = actual
            .source()
            .manifest()
            .routes()
            .get(order)
            .ok_or_else(invalid)?;
        let receiving = descriptor.destination() == actual.source().manifest().rank();
        if !receiving && descriptor.source() != actual.source().manifest().rank() {
            return Err(invalid());
        }
        let layout = context
            .layout(input.shape(), input.dtype())?
            .with_representation(input.representation());
        let prototype = WorkspaceTensor::existing(layout.clone(), &context)?;
        let dtype = eredu_runtime::working_memory::WorkspaceCommunicationMetadata.dtype(&prototype);
        let expected = actual
            .source()
            .boundary_header_length(route, ordinal, input.shape(), &dtype, funding)
            .map_err(|cause| context.metadata_source(cause))?;
        if expected != header_bytes {
            return Err(invalid());
        }
        let header_length = i32::try_from(header_bytes).map_err(|_| invalid())?;
        let header = WorkspaceTensor::existing(
            context.layout(&[header_length], WorkspaceDtype::Uint8)?,
            &context,
        )?;
        context.begin_span();
        let frame =
            boundary_frame::encode(&boundary_frame::Workspace(&context), &prototype, &header)?;
        let report = context.finish_report(&[frame.clone()])?;
        let encoding = super::numerical(&report, 1, mechanism, &context)?;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let encode_capacity = capacity(encoding, runtime, &context)?;
        let exchange = actual
            .framed_route_exchange(order)
            .map_err(|cause| source.neural_error(cause))?
            .ok_or_else(invalid)?;
        if exchange.rounds()>1 {
            let manifest=actual.source().manifest();
            let wave=manifest.route_submission_waves_with_metadata(&context)?.into_iter().find(|wave|wave.contains(&order)).ok_or_else(invalid)?;
            let contract=descriptor.boundary_contract().ok_or_else(invalid)?;
            for candidate in &manifest.routes()[wave] {
                if candidate.boundary_contract()!=Some(contract)
                    || actual.source().boundary_header_length(candidate.id(),ordinal,input.shape(),&dtype,funding)
                        .map_err(|cause|context.metadata_source(cause))?!=header_bytes {
                    return Err(invalid());
                }
            }
        }
        let mut rounds=context.metadata_vec(exchange.rounds())?;
        for ordinal in 0..exchange.rounds(){
            rounds.push(exchange.round_layout_storage(&actual,ordinal,frame.shape(),safemlx::Dtype::Uint8)
                .map_err(|cause|source.neural_error(cause))?.try_into_owned(source)
                .map_err(|cause|source.neural_error(cause))?);
        }
        let (decoding, decode_capacity) = if receiving {
            let (recipe, capacity) =
                decode(source, input, frame.shape(), header_length, mechanism)?;
            (Some(recipe), Some(capacity))
        } else {
            (None, None)
        };
        let transported = u64::try_from(rounds.iter().try_fold(0usize,|sum,round|sum.checked_add(round.backing_capacity())).ok_or_else(invalid)?).map_err(|_| invalid())?;
        let decoded = decode_capacity.map_or(0, |value| value.backing);
        let encoded = u64::try_from(encode_capacity.backing).map_err(|_| invalid())?;
        // Source returns its original activation. A receiver may retain either
        // the received frame backing or a separately aligned decoded payload.
        // Both actual populations stay in the conservative role overlap.
        let output = receiving
            .then(|| transported.checked_add(u64::try_from(decoded).ok()?))
            .flatten();
        if receiving && output.is_none() {
            return Err(invalid());
        }
        let scratch = if receiving {
            encoded
        } else {
            encoded.checked_add(transported).ok_or_else(invalid)?
        };
        Ok(Self {
            route,
            ordinal,
            header_bytes,
            input: layout,
            receiving,
            encoding,
            decoding,
            encode_capacity,
            decode_capacity,
            rounds,
            output,
            scratch,
            source: source.clone(),
        })
    }
    pub(crate) fn wire_shape(&self)->&[i32]{self.rounds.first().expect("nonempty retained route path").shape()}
    pub(crate) fn matches_source(&self, source: &OriginalParallelSource) -> bool {
        self.source.same_source(source)
    }
    /// All real child Data births share the request allocator. They are absent
    /// from the parent GPU worker population because each child completes them.
    pub(crate) fn maximum_backing_births(&self) -> Option<usize> {
        self.encoding
            .storage
            .maximum_births()
            .checked_add(self.rounds.iter().try_fold(0usize,|sum,round|sum.checked_add(round.maximum_backing_births()?))?)?
            .checked_add(
                self.decoding
                    .map_or(0, |recipe| recipe.storage.maximum_births()),
            )
    }
}
pub(super) fn capacity(
    recipe: SpeculativeNumericalRecipe,
    runtime: &safemlx::PreparedInputRuntime,
    context: &WorkspaceContext,
) -> Result<BoundaryStageCapacity, Error> {
    context.charge_metadata(size_of::<(
        BoundaryStageCapacity,
        Result<BoundaryStageCapacity, Error>,
    )>())?;
    let bytes = usize::try_from(recipe.storage.mutable_bytes())
        .map_err(|_| context.metadata_error(format_args!("boundary native capacity overflow")))?;
    let backing = safemlx::OriginalBufferBudget::metal_population_layout(
        runtime,
        bytes,
        recipe.storage.maximum_births(),
    )
    .map_err(|cause| context.metadata_source(cause))?;
    Ok(BoundaryStageCapacity {
        graph: recipe.graph_capacity,
        records: recipe.record_capacity,
        backing: backing.capacity(),
    })
}
#[inline(never)]
fn decode(
    source: &OriginalParallelSource,
    input: WorkspaceLayoutView<'_>,
    frame: &[i32],
    header: i32,
    mechanism: ResidentExecutionMechanisms,
) -> Result<(SpeculativeNumericalRecipe, BoundaryStageCapacity), Error> {
    let context = WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
    context.charge_metadata(size_of::<(
        WorkspaceContext,
        WorkspaceTraceReport,
        SpeculativeNumericalRecipe,
        BoundaryStageCapacity,
        Result<(SpeculativeNumericalRecipe, BoundaryStageCapacity), Error>,
    )>())?;
    let prototype = WorkspaceTensor::existing(
        context
            .layout(input.shape(), input.dtype())?
            .with_representation(input.representation()),
        &context,
    )?;
    let frame = WorkspaceTensor::existing(context.layout(frame, WorkspaceDtype::Uint8)?, &context)?;
    context.begin_span();
    let (header, payload) = boundary_frame::split(
        &boundary_frame::Workspace(&context),
        &frame,
        header,
        &prototype,
    )?;
    let report = context.finish_report(&[header, payload])?;
    let recipe = super::numerical(&report, 2, mechanism, &context)?;
    let runtime = source
        .agreement_inputs()
        .ok_or_else(|| {
            context.metadata_error(format_args!("boundary source has no input runtime"))
        })?
        .runtime();
    let capacity = capacity(recipe, runtime, &context)?;
    Ok((recipe, capacity))
}
