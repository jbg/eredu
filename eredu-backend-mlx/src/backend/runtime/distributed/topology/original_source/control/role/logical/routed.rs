//! Actual routed Pair/Peer steps followed by the same admitted arithmetic role.
use super::*;

pub(super) fn destination<T>(count: usize, c: &Custody) -> Result<Vec<T>, Error> {
    reserve(&c.funding, &[size_of::<Vec<T>>(), size_of::<Result<Vec<T>, Error>>(),
        Layout::array::<T>(count).map_err(|_| overflow())?.size()])?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| overflow())?;
    if values.capacity() != count { return Err(overflow()); }
    Ok(values)
}
pub(super) fn result<O: logical_collective::packed::PackedOperations>(ops: &O,
    values: Vec<(usize, O::Value)>, quote: &LogicalCollectiveQuote) -> Result<O::Value, O::Error> {
    ops.charge(size_of::<(&O, Vec<(usize, O::Value)>, &LogicalCollectiveQuote,
        Result<O::Value, O::Error>)>())?;
    match quote.kind {
        LogicalCollectiveKind::Sum => logical_collective::routed::sum(ops, values),
        LogicalCollectiveKind::Gather => logical_collective::routed::gather(ops, values, quote.input.shape()),
    }
}
pub(super) fn execute_finish<T, F>(owner: &OriginalParallelControlOwner,
    quote: RetainedLogicalCollective, input: &Array, stream: &Stream,
    first: ParallelControlClaim, c: &Custody, finish: F) -> Result<T, Error>
where F: FnOnce(Array, &OriginalScopeObserver) -> Result<T, Error> {
    reserve(&c.funding, &[size_of::<F>(), size_of::<T>(), size_of::<Result<T, Error>>(),
        size_of::<Vec<(usize, Array)>>(), size_of::<Option<ParallelControlClaim>>(),
        size_of::<Pair>(), size_of::<[Array; 2]>(), size_of::<Result<[Array; 2], Error>>()])?;
    let source = owner.owner().request.source.communication_source()?;
    let group = source.group(quote.value().order).ok_or_else(|| fail(LogicalCause::Identity, c))?.0;
    let plan = group.logical_routed_plan().map_err(|_| fail(LogicalCause::Identity, c))?
        .ok_or_else(|| fail(LogicalCause::Identity, c))?;
    let selected = quote.value().routed().ok_or_else(|| fail(LogicalCause::Identity, c))?;
    if plan.len() != selected.values.len() || group.size() != plan.len() {
        return Err(fail(LogicalCause::Identity, c));
    }
    let mut first = Some(first);
    let mut values = destination(plan.len(), c)?;
    for (value, (route, expected)) in plan.values().zip(&selected.values).enumerate() {
        if route.source_rank() != expected.source_rank { return Err(fail(LogicalCause::Identity, c)); }
        let mut source_index = 0;
        let step = |step: usize, _exchange, previous: Array| {
            let source = expected.steps.get(source_index).ok_or_else(|| fail(LogicalCause::Identity, c))?;
            if source.step != step || source.source.value().routed_step() != Some((value, step)) {
                return Err(fail(LogicalCause::Identity, c));
            }
            source_index += 1;
            let selected = source.source.value().exchange_pairs().and_then(|pairs| pairs.first())
                .ok_or_else(|| fail(LogicalCause::Identity, c))?;
            let capacity = AgreementCapacity { graph: selected.graph_capacity(),
                records: selected.record_capacity(), backing: selected.backing_capacity() };
            let pair = Pair { input: previous, round: 0, quote: source.source.clone(),
                claim: match first.take() { Some(first) => first, None => claim(owner.owner(), c)? },
                owner: OriginalParallelControlOwner(owner.0.clone()), custody: c.clone() };
            let outputs = run_native_role(pair, capacity, &owner.owner().native, c,
                |pair, observer| Ok(pair.run(observer)))
                .map_err(|cause| Error::with_original_control_source(cause, false))??;
            run_arithmetic(owner, source.source.clone(), ArithmeticStage::Peer, outputs, stream, c)
        };
        reserve(&c.funding, &[route.iteration_control_bytes::<Array, Error, _>(&step).ok_or_else(overflow)?])?;
        let output = route.fold_exchanges(retained_array(input, c)?, step)?;
        if source_index != expected.steps.len() { return Err(fail(LogicalCause::Identity, c)); }
        values.push((route.source_rank(), output));
    }
    let members = plan.len();
    drop(source);
    let first = match first { Some(first) => first, None => claim(owner.owner(), c)? };
    run_arithmetic_finish_inputs(owner, quote.clone(), ArithmeticStage::RoutedResult,
        ArithmeticInputs::Routed(values), stream, first, c, |output, observer| {
            let expected = quote.value();
            let count = i32::try_from(members).map_err(|_| overflow())?;
            let shape = match expected.kind {
                LogicalCollectiveKind::Sum => output.shape() == input.shape(),
                LogicalCollectiveKind::Gather if input.shape().is_empty() => output.shape() == [count],
                LogicalCollectiveKind::Gather => output.shape().len() == input.shape().len()
                    && input.shape()[0].checked_mul(count) == output.shape().first().copied()
                    && output.shape().iter().skip(1).eq(input.shape().iter().skip(1)),
            };
            if !shape || output.dtype() != expected.native_dtype { return Err(fail(LogicalCause::Identity, c)); }
            finish(output, observer)
        })
}
