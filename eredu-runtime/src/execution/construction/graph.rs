//! The existing graph validator with shared ordinary/counted destinations.
use super::*;
use std::{cmp::Reverse, collections::BinaryHeap};

pub(in crate::execution) fn construct(
    groups: Vec<ExecutionGroupSpec>,
    output_name: &str,
    destination: Destination<'_>,
) -> Result<ExecutionGraph, Failure> {
    destination.controls::<(
        ExecutionGraph,
        ExecutionGraphError,
        Vec<(&str, usize)>,
        Vec<Vec<usize>>,
        Vec<usize>,
        Vec<bool>,
        BinaryHeap<Reverse<usize>>,
    )>()?;
    if groups.is_empty() {
        return Err(destination.graph_error(ExecutionGraphError::EmptyGraph));
    }
    let mut by_id = destination.vector(groups.len())?;
    by_id.extend(
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.id(), index)),
    );
    by_id.sort_unstable();
    let lookup = |id: &str| {
        let index = by_id.partition_point(|&(name, _)| name < id);
        by_id
            .get(index)
            .filter(|&&(name, _)| name == id)
            .map(|&(_, index)| index)
    };
    // Check declaration order, including an earlier duplicate before a later
    // empty name. The sorted borrowed index picks the first declared duplicate.
    for (index, group) in groups.iter().enumerate() {
        if group.id.trim().is_empty() {
            return Err(destination.graph_error(ExecutionGraphError::EmptyGroupId));
        }
        if lookup(group.id()) != Some(index) {
            return Err(destination.graph_error(ExecutionGraphError::DuplicateGroup(
                destination.text(group.id())?,
            )));
        }
    }
    let output = match lookup(output_name) {
        Some(index) => index,
        None => {
            return Err(destination.graph_error(ExecutionGraphError::UnknownOutput(
                destination.text(output_name)?,
            )));
        }
    };
    let mut dependencies = destination.vector(groups.len())?;
    let mut dependents = destination.vector(groups.len())?;
    dependents.resize_with(groups.len(), Vec::<usize>::new);
    let mut indegree = destination.vector(groups.len())?;
    indegree.resize(groups.len(), 0usize);
    for (index, group) in groups.iter().enumerate() {
        let mut resolved = destination.vector(group.dependencies.len())?;
        for dependency in &group.dependencies {
            let dependency_index = match lookup(dependency) {
                Some(index) => index,
                None => {
                    return Err(
                        destination.graph_error(ExecutionGraphError::UnknownDependency {
                            group: destination.text(group.id())?,
                            dependency: destination.text(dependency)?,
                        }),
                    );
                }
            };
            if dependency_index == index {
                return Err(destination.graph_error(ExecutionGraphError::SelfDependency(
                    destination.text(group.id())?,
                )));
            }
            if resolved.contains(&dependency_index) {
                return Err(
                    destination.graph_error(ExecutionGraphError::DuplicateDependency {
                        group: destination.text(group.id())?,
                        dependency: destination.text(dependency)?,
                    }),
                );
            }
            resolved.push(dependency_index);
            destination.reserve(&mut dependents[dependency_index], 1)?;
            dependents[dependency_index].push(index);
        }
        indegree[index] = resolved.len();
        dependencies.push(resolved);
    }
    // Each group enters this min-heap at most once. Move the prepaid vector's
    // capacity into it; no push can exceed the actual group count.
    let mut ready_values = destination.vector(groups.len())?;
    ready_values.extend(
        indegree
            .iter()
            .enumerate()
            .filter_map(|(index, &degree)| (degree == 0).then_some(Reverse(index))),
    );
    let mut ready = BinaryHeap::from(ready_values);
    let mut execution_order = destination.vector(groups.len())?;
    while let Some(Reverse(index)) = ready.pop() {
        execution_order.push(index);
        for &dependent in &dependents[index] {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                ready.push(Reverse(dependent));
            }
        }
    }
    if execution_order.len() != groups.len() {
        return Err(destination.graph_error(ExecutionGraphError::Cycle));
    }
    let mut contributes = destination.vector(groups.len())?;
    contributes.resize(groups.len(), false);
    let mut pending = destination.vector(groups.len())?;
    contributes[output] = true;
    pending.push(output);
    while let Some(index) = pending.pop() {
        for &dependency in &dependencies[index] {
            if !contributes[dependency] {
                contributes[dependency] = true;
                pending.push(dependency);
            }
        }
    }
    let missing = contributes.iter().filter(|&&present| !present).count();
    if missing != 0 {
        let mut disconnected = destination.vector(missing)?;
        for (group, present) in groups.iter().zip(contributes) {
            if !present {
                disconnected.push(destination.text(group.id())?);
            }
        }
        return Err(destination.graph_error(ExecutionGraphError::Disconnected { disconnected }));
    }
    drop(by_id);
    Ok(ExecutionGraph {
        groups,
        dependencies,
        dependents,
        execution_order,
        output,
    })
}
