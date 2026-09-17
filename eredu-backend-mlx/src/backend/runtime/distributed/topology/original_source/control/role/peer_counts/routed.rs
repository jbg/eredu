//! Existing routed collective itinerary, delivering completed integer rows.
use super::*;
use super::super::super::super::inputs::CompletedCommunicationInput;
use super::super::super::super::parallel::RetainedLogicalCollective;

pub(super) fn execute(owner: &OriginalParallelControlOwner, first: ParallelControlClaim,
    group: &Group, order: usize, input: &CompletedCommunicationInput, local_wire: &[i32],
    count_source: Option<&super::super::super::super::parallel::RetainedExpertCount>,
    stream: &Stream, c: &Custody) -> Result<SharedStorageOwner<table::PeerCountTable>, Error> {
    reserve(&c.funding, &[size_of::<(Array, Option<CompletedCommunicationWords>)>(),
        size_of::<Result<(Array, Option<CompletedCommunicationWords>), Error>>(),
        size_of::<RetainedLogicalCollective>(), size_of::<Option<ParallelControlClaim>>(),
        size_of::<Vec<bool>>(), size_of::<Vec<i32>>()])?;
    let source = owner.owner().request.source.communication_source()?;
    if !source.matches_group(order, group) || !input.belongs_to(&source) {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let (_, descriptor, _) = source.group(order).ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    let peers = descriptor.members().len();
    let plan = group.logical_routed_plan().map_err(|_| fail(PeerCountCause::Identity, c))?
        .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    if plan.len() != peers || input.value().size() != local_wire.len() {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let identity = first.identity(); let ordinal = first.ordinal(); let mut first = Some(first);
    let mut seen = table::destination::<bool>(peers, c)?; seen.resize(peers, false);
    let mut rows = table::destination::<i32>(peers.checked_mul(local_wire.len()).ok_or_else(overflow)?, c)?;
    rows.resize(peers * local_wire.len(), 0);
    let model = owner.owner().request.source.model().ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    for (value, route) in plan.values().enumerate() {
        let source_rank = route.source_rank();
        if source_rank >= peers || std::mem::replace(&mut seen[source_rank], true) {
            return Err(fail(PeerCountCause::Identity, c));
        }
        let advance = |step: usize, _exchange, (previous, _words): (Array, Option<CompletedCommunicationWords>)| {
            let quote = match count_source {
                Some(count)=>count.logical(Some((value,step))).ok_or_else(||fail(PeerCountCause::Identity,c))?,
                None=>RetainedLogicalCollective::prepare_count_routed_step(model, order, value, step, input, stream)?,
            };
            logical::execute_count_route_step(owner, first.take(), previous, quote, stream, c)
                .map(|(array, words)| (array, Some(words)))
        };
        reserve(&c.funding, &[route.iteration_control_bytes::<(Array, Option<CompletedCommunicationWords>), Error, _>(
            &advance).ok_or_else(overflow)?])?;
        let (array, words) = route.fold_exchanges((logical::retain_count_input(input.value(), c)?, None), advance)?;
        let row = match &words {
            Some(words) => words.as_slice(),
            None if source_rank == group.rank() => local_wire,
            None => return Err(fail(PeerCountCause::Identity, c)),
        };
        if row.len() != local_wire.len() { return Err(fail(PeerCountCause::Identity, c)); }
        let start = source_rank * local_wire.len();
        rows[start..start + local_wire.len()].copy_from_slice(row);
        drop(words); drop(array);
    }
    if seen.iter().any(|seen| !seen) { return Err(fail(PeerCountCause::Identity, c)); }
    table::PeerCountTable::new(&rows, descriptor.members(), source.source().manifest().world_size(),
        None, order, identity, ordinal, c)
}
