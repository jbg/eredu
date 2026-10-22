//! Unambiguous lookup of independently materialized selected outputs.
use crate::{ReplicatedTextMaterializationTask, ReplicatedTextParameterOwner};
use std::collections::BTreeMap;

/// Multiple exact selected tasks claim one canonical output identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("materialization output {target:?} has multiple tasks owned by {previous:?} and {owner:?}")]
pub struct MaterializationTaskIndexError {
    /// Repeated canonical output, including a standalone companion identity.
    pub target: String,
    /// Owner of the first task claiming the output.
    pub previous: ReplicatedTextParameterOwner,
    /// Owner of the second task claiming the output.
    pub owner: ReplicatedTextParameterOwner,
}

/// Borrowed exact tasks keyed by canonical primary or standalone companion output.
/// Generated companions remain part of their primary operation and are excluded;
/// aliases do not become independent resources. Distinct outputs remain distinct
/// even when their source recipes or physical checkpoint inputs are equal.
#[derive(Debug)]
pub struct MaterializationTaskIndex<'a> {
    outputs: BTreeMap<&'a str, &'a ReplicatedTextMaterializationTask>,
}
impl<'a> MaterializationTaskIndex<'a> {
    /// Indexes one or more immutable selected task sequences without cloning tasks,
    /// acquiring sources or reconstructing standalone companion lowering.
    pub fn new(
        tasks: impl IntoIterator<Item = &'a ReplicatedTextMaterializationTask>,
    ) -> Result<Self, MaterializationTaskIndexError> {
        let mut outputs = BTreeMap::<&'a str, &'a ReplicatedTextMaterializationTask>::new();
        for task in tasks {
            for (target, exact) in std::iter::once((task.name(), task)).chain(
                task.output_companions().iter().filter_map(|companion| {
                    companion
                        .materialization_task()
                        .map(|exact| (companion.name(), exact))
                }),
            ) {
                if let Some(previous) = outputs.insert(target, exact) {
                    return Err(MaterializationTaskIndexError {
                        target: target.into(),
                        previous: previous.owner().clone(),
                        owner: exact.owner().clone(),
                    });
                }
            }
        }
        Ok(Self { outputs })
    }
    /// Resolves only independently materialized output identities. A companion's
    /// topology name may be an alias of the exact retained standalone task name.
    pub fn get(&self, target: &str) -> Option<&'a ReplicatedTextMaterializationTask> {
        self.outputs.get(target).copied()
    }
}
