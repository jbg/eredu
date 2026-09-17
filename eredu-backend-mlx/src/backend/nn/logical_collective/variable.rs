//! One local variable route/slice/assembly worker for ordinary and funded calls.
use crate::backend::runtime::distributed::{LogicalExchangePlan, LogicalVariableRoute, LogicalVariableRoutePlan};
use std::mem::{size_of, size_of_val};
pub(crate) trait LocalVariableOperations {
    type Value;
    type Error;
    fn invalid(&self) -> Self::Error;
    fn charge(&self, bytes: usize) -> Result<(), Self::Error>;
    fn values(&self, capacity: usize) -> Result<Vec<(usize, Self::Value)>, Self::Error>;
    fn rows(&self, value: &Self::Value) -> Option<usize>;
    fn slice(&mut self, value: &Self::Value, counts: &[usize], destination: usize)
        -> Result<Self::Value, Self::Error>;
    fn exchange(&mut self, value: usize, step: usize, exchange: LogicalExchangePlan<'_>, input: Self::Value)
        -> Result<Self::Value, Self::Error>;
    fn concatenate(&mut self, values: Vec<(usize, Self::Value)>) -> Result<Self::Value, Self::Error>;
}
pub(crate) fn controls<O: LocalVariableOperations>() -> Option<usize> {
    let frames = [size_of::<(&mut O, &O::Value, &[usize], &[usize], LogicalVariableRoutePlan<'_>)>(),
        size_of::<Vec<(usize, O::Value)>>(), size_of::<O::Value>(), size_of::<Result<O::Value, O::Error>>(),
        size_of::<LogicalVariableRoute<'_>>(),
        size_of::<Option<LogicalExchangePlan<'_>>>(), size_of::<[usize; 6]>(),
        size_of::<std::ops::Range<usize>>() * 2];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn execute<O: LocalVariableOperations>(ops: &mut O, input: &O::Value,
    send: &[usize], receive: &[usize], plan: LogicalVariableRoutePlan<'_>) -> Result<O::Value, O::Error> {
    ops.charge(controls::<O>().ok_or_else(|| ops.invalid())?)?;
    let sent = send.iter().copied().try_fold(0usize, usize::checked_add).ok_or_else(|| ops.invalid())?;
    if send.len() != plan.group().size() || receive.len() != send.len()
        || Some(sent) != ops.rows(input) {
        return Err(ops.invalid());
    }
    let mut values = ops.values(plan.len())?;
    for (value, route) in plan.values().enumerate() {
        let source_rank = route.source_rank();
        if source_rank >= receive.len() { return Err(ops.invalid()); }
        let destination = route.destination_rank();
        if destination >= send.len()
            || values.iter().any(|(source, _)| *source == source_rank) { return Err(ops.invalid()); }
        let mut routed = ops.slice(input, send, destination)?;
        for step in 0..route.steps() {
            if let Some(exchange) = route.exchange(step) {
                routed = ops.exchange(value, step, exchange, routed)?;
            }
        }
        if ops.rows(&routed) != Some(receive[source_rank]) { return Err(ops.invalid()); }
        // The retained plan bounds this insertion; a bad internal plan cannot
        // grow an unpriced destination vector.
        if values.len() == plan.len() { return Err(ops.invalid()); }
        values.push((source_rank, routed));
    }
    if values.len() != receive.len() { return Err(ops.invalid()); }
    // The admitted unique source ranks form a permutation. Place each cycle
    // directly, without a recursive sort or another output vector.
    for index in 0..values.len() {
        while values[index].0 != index {
            let destination = values[index].0;
            values.swap(index, destination);
        }
    }
    ops.concatenate(values)
}
