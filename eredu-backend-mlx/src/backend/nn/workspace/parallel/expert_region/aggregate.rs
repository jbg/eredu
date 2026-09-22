//! One complete source-derived region population; every child remains separate.
use super::*;
pub(crate) struct ExpertRegionAggregate {
    pub(crate) child_bytes: u64,
    pub(crate) child_scratch: Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
    pub(crate) child_births: usize,
    pub(crate) host_bytes: u64,
    pub(crate) parent_completions: usize,
    pub(crate) indexed_parent: bool,
    pub(crate) empty_slices: usize,
    pub(crate) empty_slice: WorkspaceOperation,
    pub(crate) extra_parents: Vec<WorkspaceOperation>,
    pub(crate) output_bytes: Vec<u64>,
}
impl ExpertRegionAggregate {
    pub(super) fn prepare(local: &ExpertLocalQuote) -> Result<Self, Error> {
        let source = &local.source;
        let context =
            WorkspaceContext::new_with_metadata_funding(local.mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(
            Self,
            Result<Self, Error>,
            WorkspaceContext,
            [i32; 1],
            usize,
            u64,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert region requires its complete count/parent/child source"
            ))
        };
        let view = local.declaration.as_view();
        let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
        let mut bytes = u64::try_from(local.capacity.backing).map_err(|_| invalid())?;
        let mut births = local.maximum_births;
        let movement = local.movement.as_ref().ok_or_else(invalid)?;
        for (kind, count) in view.movement.counts().into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let (backing, created) = movement.per_kind_population(kind).ok_or_else(invalid)?;
            bytes = bytes
                .checked_add(
                    u64::try_from(backing.checked_mul(count).ok_or_else(invalid)?)
                        .map_err(|_| invalid())?,
                )
                .ok_or_else(invalid)?;
            births = births
                .checked_add(created.checked_mul(count).ok_or_else(invalid)?)
                .ok_or_else(invalid)?;
        }
        let transport = local.transport.as_ref().ok_or_else(invalid)?;
        for profile in transport.profiles() {
            bytes = bytes
                .checked_add(
                    u64::try_from(profile.aggregate_backing().ok_or_else(invalid)?)
                        .map_err(|_| invalid())?,
                )
                .ok_or_else(invalid)?;
            births = births.checked_add(profile.births).ok_or_else(invalid)?;
        }
        let count = local.counts.as_ref().ok_or_else(invalid)?;
        bytes = bytes.checked_add(count.backing).ok_or_else(invalid)?;
        births = births.checked_add(count.births).ok_or_else(invalid)?;
        let provider = local.provider.as_ref().ok_or_else(invalid)?;
        bytes = bytes.checked_add(provider.backing).ok_or_else(invalid)?;
        births = births.checked_add(provider.births).ok_or_else(invalid)?;
        // Initial selected-ID read plus each returned metadata column use the
        // actual parent's existing one-root completion worker. Movement child
        // inputs each complete before crossing into another native arena.
        // The transport source separately declares input completion for each
        // metadata column (which may contain a parent empty Slice), or every
        // local-route payload. Completed numerical child outputs need none.
        let metadata = view
            .transfers
            .iter()
            .filter(|payload| {
                matches!(
                    payload,
                    eredu_nn::workspace::WorkspaceExpertTransfer::ForwardIndex
                        | eredu_nn::workspace::WorkspaceExpertTransfer::ReverseIndex
                )
            })
            .count();
        let parent_completions = view
            .movement
            .parent_completions()
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(metadata))
            .and_then(|n| n.checked_add(transport.parent_input_completions()?))
            .and_then(|n| n.checked_add(usize::from(local.addressable.is_some())))
            .ok_or_else(invalid)?;
        // Each optional zero-row prepared integer uses its source's paid unit
        // seed followed by this exact empty Slice. Its possible parent metadata
        // constructor must be retained even when this rank's rows are nonzero.
        let empty_slices = view
            .movement
            .row_gathers
            .checked_add(view.movement.scalar_gathers)
            .and_then(|n| n.checked_add(view.movement.indexed_adds))
            .and_then(|n| n.checked_add(metadata))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(invalid)?;
        let parent =
            WorkspaceContext::new_with_metadata_funding(local.mechanism, source.funding().clone())?;
        let seed =
            WorkspaceTensor::existing(parent.layout(&[1, 1], WorkspaceDtype::Int32)?, &parent)?;
        parent.begin_span();
        let empty = seed.narrow_axis(0, 0, 0, &parent)?;
        let mut report = parent.finish_report(&[empty])?;
        if report.operations.len() != 1 {
            return Err(invalid());
        }
        let empty_slice = report.operations.pop().ok_or_else(invalid)?;
        let mut output_bytes = context.metadata_vec(local.outputs.len())?;
        for layout in &local.outputs {
            let dtype =
                crate::backend::nn::workspace::byte_view::Dtype::from_layout(layout.as_view())
                    .ok_or_else(invalid)?;
            let shape = [
                view.source_rows.max(1),
                usize::try_from(view.kernel.dimensions().1).map_err(|_| invalid())?,
            ];
            context.charge_metadata(safemlx::PreparedInputRuntime::zeros_plan_control_bytes())?;
            // The final zero/add worker creates this same dtype/row geometry.
            let plan = runtime
                .zeros(dtype.native(), &shape)
                .map_err(|cause| context.metadata_source(cause))?;
            output_bytes.push(u64::try_from(plan.backing_bytes()).map_err(|_| invalid())?);
        }
        // The shared sequential driver creates one order and one destination
        // directory; activation and optional bias both borrow those same rows.
        let host_bytes =
            if view.kernel.reduction() == eredu_nn::GroupReduction::SequentialGroupOrder {
                u64::try_from(
                    view.selected_rows()
                        .and_then(|n| n.checked_mul(size_of::<usize>()))
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(invalid)?,
                )
                .map_err(|_| invalid())?
            } else {
                0
            };
        let source_host = if local.mechanism.allocation().original_storage {
            local.addressable.as_ref().map_or(0, |quote| quote.host_capacity())
                .checked_add(u64::try_from(crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection::expert_local_control_bytes()
                    .ok_or_else(invalid)?).map_err(|_| invalid())?).ok_or_else(invalid)?
        } else {
            // Indexed source metadata belongs to this operation. Ordinary
            // caller and native graph controls are supplied by their producers.
            match &local.ordinary_addressable {
                Some(program) => program.host_bytes().ok_or_else(invalid)?,
                None => 0,
            }
        };
        let host_bytes = host_bytes.checked_add(source_host).ok_or_else(invalid)?;
        let child_scratch = super::super::scratch::region(&context, local)?;
        if let Some(source) = &child_scratch {
            bytes = source.backing_bytes().ok_or_else(invalid)?;
        }
        Ok(Self {
            child_bytes: bytes,
            child_scratch,
            child_births: births,
            host_bytes,
            parent_completions,
            indexed_parent: local.addressable.is_some(),
            empty_slices,
            empty_slice,
            extra_parents: Vec::new(),
            output_bytes,
        })
    }
}
