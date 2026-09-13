//! Immutable lookup metadata retained with the checked partition declaration.
use std::collections::BTreeMap;

use eredu_core::parameters::ParameterError;
use eredu_runtime::{
    ArchitectureParameterDescription, OwnedParameterGroupSpec, ParameterMemberSpec,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
};

/// Indices refer only to the exact immutable tasks and declaration used at binding.
/// This contains no tensors, loaded authority, source handles or parameter values.
#[derive(Debug)]
pub(crate) struct ParameterMemberIndex {
    outputs: BTreeMap<String, Option<(usize, Option<usize>)>>,
    members: BTreeMap<String, (usize, usize)>,
}

impl ParameterMemberIndex {
    pub(crate) fn new(
        tasks: &[ReplicatedTextMaterializationTask],
        parameters: &ArchitectureParameterDescription,
    ) -> Self {
        let mut outputs = BTreeMap::new();
        for (task_index, task) in tasks.iter().enumerate() {
            for (name, companion) in std::iter::once((task.name(), None))
                .chain(task.aliases().iter().map(|alias| (alias.as_str(), None)))
                .chain(
                    task.output_companions()
                        .iter()
                        .enumerate()
                        .map(|(index, companion)| (companion.name(), Some(index))),
                )
            {
                let location = (task_index, companion);
                outputs
                    .entry(name.to_owned())
                    .and_modify(|previous| {
                        if *previous != Some(location) {
                            *previous = None;
                        }
                    })
                    .or_insert(Some(location));
            }
        }
        // ArchitectureParameterDescription already rejects duplicate members.
        let members = parameters
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(group_index, group)| {
                group
                    .members()
                    .iter()
                    .enumerate()
                    .map(move |(member_index, member)| {
                        (member.target().to_owned(), (group_index, member_index))
                    })
            })
            .collect();
        Self { outputs, members }
    }

    pub(crate) fn output<'a>(
        &self,
        tasks: &'a [ReplicatedTextMaterializationTask],
        parameter: &str,
    ) -> Result<
        (
            &'a ReplicatedTextMaterializationTask,
            Option<&'a ReplicatedTextOutputCompanion>,
        ),
        ParameterError,
    > {
        let (task, companion) = self
            .outputs
            .get(parameter)
            .ok_or_else(|| ParameterError::Missing(parameter.into()))?
            .ok_or_else(|| {
                super::invalid("parameter identity resolves to multiple selected tasks")
            })?;
        let task = &tasks[task];
        Ok((
            task,
            companion.map(|index| &task.output_companions()[index]),
        ))
    }

    pub(crate) fn member<'a>(
        &self,
        parameters: &'a ArchitectureParameterDescription,
        task: &ReplicatedTextMaterializationTask,
        companion: Option<&ReplicatedTextOutputCompanion>,
        parameter: &str,
    ) -> Result<(&'a OwnedParameterGroupSpec, &'a ParameterMemberSpec), ParameterError> {
        let primary = companion.map_or(task.name(), ReplicatedTextOutputCompanion::name);
        let mut found = self.members.get(primary).copied();
        if companion.is_none() {
            for alias in task.aliases() {
                if let Some(location) = self.members.get(alias).copied() {
                    if found.is_some_and(|previous| previous != location) {
                        return Err(super::invalid(
                            "selected parameter aliases resolve to multiple module slots",
                        ));
                    }
                    found = Some(location);
                }
            }
        }
        let (group, member) = found.ok_or_else(|| ParameterError::Missing(parameter.into()))?;
        let group = &parameters.groups()[group];
        Ok((group, &group.members()[member]))
    }
}
