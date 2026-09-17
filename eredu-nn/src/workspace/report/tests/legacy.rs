use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn report(
    context: &WorkspaceContext,
    retained: &[WorkspaceTensor],
) -> Result<WorkspaceTraceReport, Error> {
    for value in retained {
        value.validate_context(context)?;
    }
    let mut reachable = BTreeSet::new();
    let mut retained_state = Some(0_u64);
    let mut maximum_allocations = 0usize;
    let mut pending: Vec<_> = retained.iter().map(|value| value.storage.clone()).collect();
    while let Some(storage) = pending.pop() {
        if reachable.insert(Rc::as_ptr(&storage)) {
            retained_state = add_bound(retained_state, storage.bytes)?;
            maximum_allocations = maximum_allocations
                .checked_add(storage.maximum_allocations)
                .ok_or_else(|| workspace_overflow("workspace allocation sum overflow"))?;
            pending.extend(storage.possible_aliases.iter().cloned());
        }
    }
    let trace = context.trace.borrow();
    let mut total = Some(trace.scratch);
    let mut persistent = Some(0_u64);
    for storage in &trace.allocations {
        total = add_bound(total, storage.bytes)?;
        if reachable.contains(&Rc::as_ptr(storage)) {
            persistent = add_bound(persistent, storage.bytes)?;
        }
    }
    if !trace.missing.is_empty() {
        total = None;
    }
    let tensor_buffers = WorkspaceTensorBufferReport {
        total_bytes: total,
        retained_bytes: persistent,
        transient_bytes: total.zip(persistent).map(|(total, state)| total - state),
    };
    let host_workspace_bytes = trace
        .missing_host
        .is_empty()
        .then_some(trace.host_workspace);
    let total = add_bound(total, host_workspace_bytes)?;
    let transient = total.zip(persistent).map(|(total, state)| total - state);
    let mut opening_storage = None;
    let state = trace
        .opening_state
        .as_ref()
        .map(|opening| {
            let mut pending = opening.clone();
            let mut seen = BTreeSet::new();
            let mut displaced = Some(0_u64);
            let mut opening_bytes = Some(0u64);
            let mut opening_allocations = 0usize;
            while let Some(storage) = pending.pop() {
                if seen.insert(Rc::as_ptr(&storage)) {
                    opening_bytes = opening_bytes
                        .zip(storage.bytes)
                        .and_then(|(a, b)| a.checked_add(b));
                    opening_allocations = opening_allocations
                        .checked_add(storage.maximum_allocations)
                        .ok_or_else(|| workspace_overflow("workspace allocation sum overflow"))?;
                    if !reachable.contains(&Rc::as_ptr(&storage)) {
                        displaced = add_bound(displaced, storage.bytes)?;
                    }
                    pending.extend(storage.possible_aliases.iter().cloned());
                }
            }
            opening_storage = Some(WorkspaceStoragePopulation {
                bytes: opening_bytes,
                maximum_allocations: opening_allocations,
            });
            Ok::<_, Error>(WorkspaceStateSpanReport {
                retained_bytes: retained_state,
                displaced_bytes: displaced,
                transient_bytes: add_bound(transient, displaced)?,
            })
        })
        .transpose()?;
    Ok(WorkspaceTraceReport {
        opening_storage,
        closing_storage: WorkspaceStoragePopulation {
            bytes: retained_state,
            maximum_allocations,
        },
        residual: context
            .borrowed
            .borrow()
            .as_ref()
            .map(|borrowed| residual_report(&trace, retained, borrowed))
            .transpose()?,
        state,
        total_bytes: total,
        retained_bytes: persistent,
        transient_bytes: transient,
        tensor_buffers,
        host_workspace_bytes,
        tensor_handle_clones: context.identity.tensor_handle_clones.get(),
        fact_construction_bytes: None,
        metadata_construction: None,
        operations: trace.operations.clone(),
        unpriced_operations: trace.missing.clone(),
        unpriced_host_operations: trace.missing_host.clone(),
        assumptions: trace.assumptions.iter().cloned().collect(),
    })
}

fn residual_report(
    trace: &Trace,
    retained: &[WorkspaceTensor],
    borrowed: &WorkspaceBorrowedStorage,
) -> Result<WorkspaceResidualReport, Error> {
    fn reachable(
        roots: impl IntoIterator<Item = Rc<Storage>>,
    ) -> BTreeMap<*const Storage, Rc<Storage>> {
        let mut result = BTreeMap::new();
        let mut pending: Vec<_> = roots.into_iter().collect();
        while let Some(storage) = pending.pop() {
            let identity = Rc::as_ptr(&storage);
            if let std::collections::btree_map::Entry::Vacant(entry) = result.entry(identity) {
                pending.extend(storage.possible_aliases.iter().cloned());
                entry.insert(storage);
            }
        }
        result
    }

    let mut report = WorkspaceResidualReport {
        opening_storage: None,
        closing_storage: None,
        borrowed_storage: borrowed.clone(),
        total_bytes: None,
        retained_bytes: None,
        displaced_bytes: None,
        transient_bytes: None,
    };
    let Some(opening) = &trace.opening_state else {
        return Ok(report);
    };
    let excluded = borrowed.storage_identities();
    let closing = reachable(retained.iter().map(|value| value.storage.clone()));
    let opening = reachable(opening.iter().cloned());
    let mut union = reachable(trace.allocations.iter().cloned());
    union.extend(opening.iter().map(|(key, value)| (*key, value.clone())));
    union.extend(closing.iter().map(|(key, value)| (*key, value.clone())));

    let mut persistent = Some(0);
    let mut displaced = Some(0);
    let mut total = Some(trace.scratch);
    let mut transient = Some(trace.scratch);
    for (identity, storage) in &union {
        if excluded.contains(identity) {
            continue;
        }
        total = add_bound(total, storage.bytes)?;
        if closing.contains_key(identity) {
            persistent = add_bound(persistent, storage.bytes)?;
        } else {
            transient = add_bound(transient, storage.bytes)?;
            if opening.contains_key(identity) {
                displaced = add_bound(displaced, storage.bytes)?;
            }
        }
    }
    if !trace.missing.is_empty() {
        total = None;
        transient = None;
    }
    let host = trace
        .missing_host
        .is_empty()
        .then_some(trace.host_workspace);
    let population = |roots: &BTreeMap<*const Storage, Rc<Storage>>| {
        let mut result = WorkspaceStoragePopulation::EMPTY;
        for (identity, storage) in roots {
            if excluded.contains(identity) { continue; }
            result.bytes = result.bytes.zip(storage.bytes).and_then(|(a,b)| a.checked_add(b));
            result.maximum_allocations = result.maximum_allocations.checked_add(storage.maximum_allocations)
                .ok_or_else(|| workspace_overflow("residual population overflow"))?;
        }
        Ok::<_, Error>(result)
    };
    report.opening_storage = Some(population(&opening)?);
    report.closing_storage = Some(population(&closing)?);
    report.total_bytes = add_bound(total, host)?;
    report.retained_bytes = persistent;
    report.displaced_bytes = displaced;
    report.transient_bytes = add_bound(transient, host)?;
    Ok(report)
}

fn add_bound(left: Option<u64>, right: Option<u64>) -> Result<Option<u64>, Error> {
    left.zip(right)
        .map(|(a, b)| {
            a.checked_add(b)
                .ok_or_else(|| workspace_overflow("workspace allocation sum overflow"))
        })
        .transpose()
}
