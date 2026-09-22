//! Descriptive native variable ceilings for the retained semantic itinerary.
use super::*;
use eredu_nn::workspace::WorkspaceExpertTransfer;
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertTransferProfile {
    pub(crate) payload: WorkspaceExpertTransfer,
    pub(crate) order: usize,
    pub(crate) world: bool,
    pub(crate) dtype: safemlx::Dtype,
    pub(crate) width: i32,
    pub(crate) send_rows: usize,
    pub(crate) receive_rows: usize,
    pub(crate) capacity: BoundaryStageCapacity,
    pub(crate) births: usize,
    pub(crate) input_reorder: Option<ExpertReorderEnvelope>,
    pub(crate) output_reorder: Option<ExpertReorderEnvelope>,
    pub(crate) ordinary: Option<OrdinaryParallelControls>,
}
#[derive(Clone, Copy)]
pub(super) struct ExpertTransferGeometry {
    pub group: eredu_core::CollectiveGroupId,
    pub rank: usize,
    pub peers: usize,
    pub selected_rows: usize,
    pub input_width: i32,
    pub output_width: i32,
    pub input_dtype: safemlx::Dtype,
    pub scores_dtype: safemlx::Dtype,
    pub coefficients_dtype: safemlx::Dtype,
    pub output_dtype: safemlx::Dtype,
    pub bias_dtype: Option<safemlx::Dtype>,
    pub transfers: eredu_nn::workspace::WorkspaceExpertTransfers,
}
pub(crate) struct ExpertTransportQuote {
    profiles: Vec<ExpertTransferProfile>,
    local: Vec<Option<ExpertLocalTransport>>,
    scratch: Vec<Option<eredu_nn::workspace::WorkspaceAllocationPopulation>>,
}
impl ExpertTransportQuote {
    pub(super) fn prepare(local: &ExpertLocalQuote) -> Result<Self, Error> {
        let source = &local.source;
        let context =
            WorkspaceContext::new_with_metadata_funding(local.mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(ExpertTransferGeometry, WorkspaceContext)>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert transfer has no selected payload representation"
            ))
        };
        let declaration = local.declaration.as_view();
        let (input_width, output_width) = declaration.kernel.dimensions();
        let geometry = ExpertTransferGeometry {
            group: declaration.group,
            rank: declaration.rank,
            peers: declaration.peers,
            selected_rows: declaration.selected_rows().ok_or_else(invalid)?,
            input_width,
            output_width,
            input_dtype: native_dtype(&local.inputs[0]).ok_or_else(invalid)?,
            scores_dtype: native_dtype(&local.inputs[2]).ok_or_else(invalid)?,
            coefficients_dtype: native_dtype(&local.inputs[3]).ok_or_else(invalid)?,
            output_dtype: native_dtype(local.outputs.first().ok_or_else(invalid)?)
                .ok_or_else(invalid)?,
            bias_dtype: local.outputs.get(1).and_then(native_dtype),
            transfers: declaration.transfers,
        };
        Self::prepare_for(source, local.mechanism, geometry)
    }
    pub(super) fn prepare_for(
        source: &OriginalParallelSource,
        mechanism: ResidentExecutionMechanisms,
        geometry: ExpertTransferGeometry,
    ) -> Result<Self, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(
            Self,
            ExpertTransferGeometry,
            ExpertTransferProfile,
            Result<Self, Error>,
            [i32; 2],
            WorkspaceContext,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert transfer has no retained native itinerary source"
            ))
        };
        let actual = source
            .communication_source()
            .map_err(|cause| source.neural_error(cause))?;
        let selection = actual
            .source()
            .manifest()
            .select_group_operation(
                geometry.group,
                eredu_runtime::CommunicationOperation::VariableAllToAll,
            )
            .map_err(|cause| context.metadata_source(cause))?;
        let order = selection.order();
        let (logical, descriptor, wave) = actual.group(order).ok_or_else(invalid)?;
        if descriptor.local_index() != Some(geometry.rank)
            || descriptor.members().len() != geometry.peers
        {
            return Err(invalid());
        }
        let local_route = logical.is_logical() && logical.logical_variable_world_plan().is_none();
        let world = logical.is_logical() && !local_route;
        let native = if local_route {
            None
        } else {
            let (native, persistent) = if world {
                let plan = logical.logical_variable_world_plan().ok_or_else(invalid)?;
                if !wave
                    || plan.members() != descriptor.members()
                    || !std::ptr::eq(plan.group(), logical)
                    || !logical
                        .native_group()
                        .shares_native_handle(actual.world().native_group())
                {
                    return Err(invalid());
                }
                (
                    actual.world().native_group(),
                    actual
                        .world_persistent()
                        .map_err(|cause| source.neural_error(cause))?,
                )
            } else {
                (
                    logical.native_group(),
                    actual
                        .group_persistent(order)
                        .map_err(|cause| source.neural_error(cause))?,
                )
            };
            if persistent.native().has_unqualified_storage()
                || !persistent.native().is_for(native)
                || !persistent.source().same_source(actual.source())
            {
                return Err(invalid());
            }
            Some((native, persistent))
        };
        let selected = geometry.selected_rows;
        // This is a global receiver cap, including ranks with no local bank.
        // The current rank's maximum_received_rows()==0 cannot stand in for it.
        let gathered = selected.checked_mul(geometry.peers).ok_or_else(invalid)?;
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let mut scratch = context.metadata_vec(geometry.transfers.len())?;
        let mut profiles = context.metadata_vec(geometry.transfers.len())?;
        let mut local_quotes = context.metadata_vec(geometry.transfers.len())?;
        for payload in geometry.transfers.iter() {
            let (width, dtype) = match payload {
                WorkspaceExpertTransfer::ForwardIndex | WorkspaceExpertTransfer::ReverseIndex => {
                    (1, safemlx::Dtype::Int32)
                }
                WorkspaceExpertTransfer::ForwardInput => {
                    (geometry.input_width, geometry.input_dtype)
                }
                WorkspaceExpertTransfer::ForwardScores => (1, geometry.scores_dtype),
                WorkspaceExpertTransfer::ForwardCoefficients => (1, geometry.coefficients_dtype),
                WorkspaceExpertTransfer::ReverseOutput => {
                    (geometry.output_width, geometry.output_dtype)
                }
                WorkspaceExpertTransfer::ReverseBias => (
                    geometry.output_width,
                    geometry.bias_dtype.ok_or_else(invalid)?,
                ),
            };
            let (send_rows, receive_rows) = if payload.reverse() {
                (gathered, selected)
            } else {
                (selected, gathered)
            };
            if local_route {
                let quote = ExpertLocalTransport::prepare(
                    source,
                    mechanism,
                    order,
                    width,
                    dtype,
                    send_rows,
                    receive_rows,
                )?;
                profiles.push(ExpertTransferProfile {
                    payload,
                    order,
                    world: false,
                    dtype,
                    width,
                    send_rows,
                    receive_rows,
                    capacity: quote.capacity,
                    births: quote.births,
                    input_reorder: None,
                    output_reorder: None,
                    ordinary: quote.ordinary_controls().and_then(|value| {
                        value.communication(logical, &[geometry.peers, geometry.peers])
                    }),
                });
                scratch.push(quote.scratch(&context)?);
                local_quotes.push(Some(quote));
                continue;
            }
            local_quotes.push(None);
            let native = native.as_ref().ok_or_else(invalid)?.0;
            // The actual accepted worker inserts one diagonal row only for a
            // completely idle participant. Include it without synthesizing IDs
            // or a completed count matrix in cold quotation.
            let shape = [
                i32::try_from(send_rows.max(1)).map_err(|_| invalid())?,
                width,
            ];
            context.charge_metadata(
                native
                    .variable_cpu_envelope_storage_control_bytes()
                    .ok_or_else(invalid)?,
            )?;
            let envelope = native
                .variable_cpu_envelope_storage(&shape, dtype, receive_rows.max(1))
                .map_err(|cause| context.metadata_source(cause))?;
            if !envelope.is_for_group(native) {
                return Err(invalid());
            }
            context.charge_metadata(envelope.backing_control_bytes().ok_or_else(invalid)?)?;
            let capacity = BoundaryStageCapacity {
                graph: envelope.graph_capacity(),
                records: envelope.record_capacity(),
                backing: envelope
                    .backing_capacity(runtime)
                    .map_err(|cause| context.metadata_source(cause))?,
            };
            let (mut births, _) = envelope.evaluation().logical_backing_population();
            let native_scratch =
                super::super::scratch::native_cpu(&context, mechanism, capacity.backing, births)?;
            let mut input_scratch = None;
            let mut output_scratch = None;
            let (input_reorder, output_reorder) = if let Some(plan) = logical
                .logical_variable_world_plan()
                .filter(|plan| !plan.canonical_order())
            {
                // Source payloads arrive from the selected Gather/Scatter or
                // accepted variable output, all of which publish row-contiguity.
                let (input, input_rows) = ExpertReorderEnvelope::prepare_with_scratch(
                    source,
                    mechanism,
                    send_rows,
                    width,
                    dtype,
                    descriptor.members().len(),
                )?;
                let (output, output_rows) = ExpertReorderEnvelope::prepare_with_scratch(
                    source,
                    mechanism,
                    receive_rows,
                    width,
                    dtype,
                    plan.world_size(),
                )?;
                input_scratch = super::super::scratch::alternatives(&context, &input_rows)?;
                output_scratch = super::super::scratch::alternatives(&context, &output_rows)?;
                births = births
                    .checked_add(input.births)
                    .and_then(|n| n.checked_add(output.births))
                    .ok_or_else(invalid)?;
                (Some(input), Some(output))
            } else {
                (None, None)
            };
            let mut rows = context.metadata_vec(3)?;
            rows.push((&native_scratch, 1));
            let qualified =
                input_reorder.is_none() || (input_scratch.is_some() && output_scratch.is_some());
            if let Some(source) = &input_scratch {
                rows.push((source, 1));
            }
            if let Some(source) = &output_scratch {
                rows.push((source, 1));
            }
            scratch.push(if qualified {
                Some(super::super::scratch::simultaneous(&context, &rows)?)
            } else {
                None
            });
            profiles.push(ExpertTransferProfile {
                payload,
                order,
                world,
                dtype,
                width,
                send_rows,
                receive_rows,
                capacity,
                births,
                input_reorder,
                output_reorder,
                ordinary: envelope
                    .ordinary_controls()
                    .and_then(OrdinaryParallelControls::group)
                    .and_then(|value| {
                        value.communication(logical, &[geometry.peers, geometry.peers])
                    }),
            });
        }
        Ok(Self {
            profiles,
            local: local_quotes,
            scratch,
        })
    }
    pub(crate) fn scratch(&self) -> &[Option<eredu_nn::workspace::WorkspaceAllocationPopulation>] {
        &self.scratch
    }
    pub(crate) fn profile(&self, index: usize) -> Option<ExpertTransferProfile> {
        self.profiles.get(index).copied()
    }
    pub(crate) fn profiles(&self) -> &[ExpertTransferProfile] {
        &self.profiles
    }
    pub(crate) fn input_requires_parent_completion(&self, index: usize) -> bool {
        self.local.get(index).is_some_and(Option::is_some)
            || self.profiles.get(index).is_some_and(|profile| {
                matches!(
                    profile.payload,
                    WorkspaceExpertTransfer::ForwardIndex | WorkspaceExpertTransfer::ReverseIndex
                )
            })
    }
    pub(crate) fn parent_input_completions(&self) -> Option<usize> {
        (0..self.profiles.len()).try_fold(0usize, |count, index| {
            count.checked_add(usize::from(self.input_requires_parent_completion(index)))
        })
    }
    pub(crate) fn local(&self, index: usize) -> Option<&ExpertLocalTransport> {
        self.local.get(index)?.as_ref()
    }
    pub(crate) fn ordinary_controls(&self) -> Option<OrdinaryParallelControls> {
        self.profiles
            .iter()
            .try_fold(OrdinaryParallelControls::default(), |total, profile| {
                let mut value = profile.ordinary?;
                for reorder in [profile.input_reorder, profile.output_reorder]
                    .into_iter()
                    .flatten()
                {
                    value = value.append(reorder.ordinary?)?;
                }
                total.append(value)
            })
    }
}
impl ExpertTransferProfile {
    pub(crate) fn matches(
        &self,
        order: usize,
        world: bool,
        input: &safemlx::Array,
        matrix: &[usize],
        peers: usize,
        rank: usize,
        reverse: bool,
        axis: usize,
    ) -> bool {
        if order != self.order
            || world != self.world
            || axis != 0
            || reverse != self.payload.reverse()
            || input.dtype() != self.dtype
            || input.shape().len() != 2
            || input.shape()[1] != self.width
            || rank >= peers
            || peers.checked_mul(peers) != Some(matrix.len())
        {
            return false;
        }
        for member in 0..peers {
            let mut sent = 0usize;
            let mut received = 0usize;
            for peer in 0..peers {
                let (a, b) = if reverse {
                    (matrix[peer * peers + member], matrix[member * peers + peer])
                } else {
                    (matrix[member * peers + peer], matrix[peer * peers + member])
                };
                let (Some(a), Some(b)) = (sent.checked_add(a), received.checked_add(b)) else {
                    return false;
                };
                sent = a;
                received = b;
            }
            if sent > self.send_rows
                || received > self.receive_rows
                || (member == rank && usize::try_from(input.shape()[0]).ok() != Some(sent))
            {
                return false;
            }
        }
        true
    }
    pub(crate) fn aggregate_backing(self) -> Option<usize> {
        self.capacity
            .backing
            .checked_add(self.input_reorder.map_or(0, |p| p.capacity.backing))?
            .checked_add(self.output_reorder.map_or(0, |p| p.capacity.backing))
    }
    pub(crate) fn covers(self, capacity: BoundaryStageCapacity) -> bool {
        capacity.graph <= self.capacity.graph
            && capacity.records <= self.capacity.records
            && capacity.backing <= self.capacity.backing
    }
}
fn native_dtype(layout: &WorkspaceLayout) -> Option<safemlx::Dtype> {
    super::super::super::byte_view::Dtype::from_layout(layout.as_view()).map(|dtype| dtype.native())
}
