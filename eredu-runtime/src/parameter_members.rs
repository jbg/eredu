//! Canonical parameter lookup over immutable selected tasks and declarations.
use std::{collections::BTreeMap, sync::Arc};

use crate::{
    ArchitectureParameterDescription, OwnedParameterGroupSpec, ParameterMemberSpec,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
};
use eredu_core::parameters::ParameterError;

/// Retained lookup over exact selected tasks and a validated canonical declaration.
/// The index owns immutable metadata, with no tensors, parameter values, loaded
/// session authority or checkpoint source handles. Aliases and companions remain
/// distinct outputs; ambiguity is rejected when the affected identity is resolved.
#[derive(Debug)]
pub struct ParameterMemberIndex {
    tasks: Arc<[ReplicatedTextMaterializationTask]>,
    parameters: Arc<ArchitectureParameterDescription>,
    outputs: BTreeMap<String, Option<(usize, Option<usize>)>>,
    members: BTreeMap<String, (usize, usize)>,
}

impl ParameterMemberIndex {
    /// Resolves an identity against the exact retained snapshot, including aliases,
    /// generated or stored companions, effective shape and canonical module slot.
    pub fn resolve(&self, parameter: &str) -> Result<ResolvedParameterMember<'_>, ParameterError> {
        resolve_parameter_member_inner(&self.tasks, &self.parameters, parameter, Some(self))
    }

    /// Retains a snapshot of the selected tasks and the immutable declaration.
    pub fn new(
        tasks: &[ReplicatedTextMaterializationTask],
        parameters: Arc<ArchitectureParameterDescription>,
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
        Self {
            tasks: tasks.to_vec().into(),
            parameters,
            outputs,
            members,
        }
    }

    fn output<'a>(
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
            .ok_or_else(|| invalid("parameter identity resolves to multiple selected tasks"))?;
        let task = &tasks[task];
        Ok((
            task,
            companion.map(|index| &task.output_companions()[index]),
        ))
    }

    fn member<'a>(
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
                        return Err(invalid(
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

/// Borrowed selected output and its canonical parameter slot. This metadata grants
/// no permission to read values, replace parameters or select native resources.
#[derive(Debug)]
pub struct ResolvedParameterMember<'a> {
    /// Primary selected materialization task.
    pub(crate) task: &'a crate::ReplicatedTextMaterializationTask,
    /// Exact output companion when resolving a companion identity.
    pub(crate) companion: Option<&'a crate::ReplicatedTextOutputCompanion>,
    /// Canonical owning parameter group.
    pub(crate) group: &'a crate::OwnedParameterGroupSpec,
    /// Canonical module member selected through the task aliases.
    pub(crate) member: &'a crate::ParameterMemberSpec,
}

impl<'a> ResolvedParameterMember<'a> {
    /// Primary selected materialization task.
    pub const fn task(&self) -> &'a ReplicatedTextMaterializationTask {
        self.task
    }

    /// Exact output companion when resolving a companion identity.
    pub const fn companion(&self) -> Option<&'a ReplicatedTextOutputCompanion> {
        self.companion
    }

    /// Canonical owning parameter group.
    pub const fn group(&self) -> &'a OwnedParameterGroupSpec {
        self.group
    }

    /// Canonical module member selected through the task aliases.
    pub const fn member(&self) -> &'a ParameterMemberSpec {
        self.member
    }
}

fn resolve_parameter_member_inner<'a>(
    tasks: &'a [crate::ReplicatedTextMaterializationTask],
    parameters: &'a crate::ArchitectureParameterDescription,
    parameter: &str,
    index: Option<&ParameterMemberIndex>,
) -> Result<ResolvedParameterMember<'a>, eredu_core::parameters::ParameterError> {
    use eredu_core::parameters::ParameterError;
    let (task, companion) = if let Some(index) = index {
        index.output(tasks, parameter)?
    } else {
        let mut matching = tasks
            .iter()
            .flat_map(|task| {
                std::iter::once((task, None)).chain(
                    task.output_companions()
                        .iter()
                        .map(move |companion| (task, Some(companion))),
                )
            })
            .filter(|(task, companion)| match companion {
                Some(companion) => companion.name() == parameter,
                None => {
                    task.name() == parameter
                        || task.aliases().iter().any(|alias| alias == parameter)
                }
            });
        let (task, companion) = matching
            .next()
            .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
        if matching.next().is_some() {
            return Err(invalid(
                "parameter identity resolves to multiple selected tasks",
            ));
        }
        (task, companion)
    };
    let logical = companion.map_or_else(
        || task.logical_shape(),
        |companion| companion.logical_shape(),
    );
    if logical.len() > 32 || logical.contains(&0) {
        return Err(invalid("selected effective parameter geometry is invalid"));
    }
    let (group, member) = if let Some(index) = index {
        index.member(parameters, task, companion, parameter)?
    } else {
        let mut matching = parameters
            .groups()
            .iter()
            .flat_map(|group| group.members().iter().map(move |member| (group, member)))
            .filter(|(_, member)| match companion {
                Some(companion) => member.target() == companion.name(),
                None => {
                    member.target() == task.name()
                        || task.aliases().iter().any(|alias| alias == member.target())
                }
            });
        let (group, member) = matching
            .next()
            .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
        if matching.next().is_some() {
            return Err(invalid(
                "selected parameter aliases resolve to multiple module slots",
            ));
        }
        (group, member)
    };
    Ok(ResolvedParameterMember {
        task,
        companion,
        group,
        member,
    })
}

/// Resolves a selected parameter without retaining an index. The caller's slices
/// remain borrowed; no task catalog or canonical declaration is cloned.
pub fn resolve_parameter_member<'a>(
    tasks: &'a [ReplicatedTextMaterializationTask],
    parameters: &'a ArchitectureParameterDescription,
    parameter: &str,
) -> Result<ResolvedParameterMember<'a>, ParameterError> {
    resolve_parameter_member_inner(tasks, parameters, parameter, None)
}

fn invalid(message: &str) -> ParameterError {
    ParameterError::Invalid(message.into())
}
