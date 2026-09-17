//! Exact completed task selection, retained without copying recipe/provenance maps.
use super::*;
use std::sync::Arc;

/// Immutable materialization result of a successfully validated text contract.
/// Cloning aliases its closed owner; this grants no source readiness or execution authority.
#[derive(Debug, Clone)]
pub struct PreparedContractMaterialization(Option<Arc<Materialization>>);

#[derive(Debug)]
struct Materialization {
    tasks: Vec<ReplicatedTextMaterializationTask>,
    declared: Vec<String>,
    addressable: Vec<String>,
    selected: SelectedReplicatedTextRealization,
}
impl Drop for PreparedContractMaterialization {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl PreparedContractMaterialization {
    pub(super) fn new(
        selected: SelectedReplicatedTextRealization,
        tasks: Vec<ReplicatedTextMaterializationTask>,
        declared: Vec<String>,
        addressable: Vec<String>,
    ) -> Self {
        Self(Some(Arc::new(Materialization {
            tasks,
            declared,
            addressable,
            selected,
        })))
    }
    fn data(&self) -> &Materialization {
        self.0.as_deref().expect("materialization source is live")
    }
    pub(super) fn tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.data().tasks
    }
    pub(super) fn addressable(&self) -> &[String] {
        &self.data().addressable
    }

    pub(super) fn validate<'a>(
        &self,
        selected: &SelectedReplicatedTextRealization,
        declarations: impl Iterator<Item = &'a str>,
        metadata: ContractMetadata<'_>,
    ) -> Result<(), PreparedTextContractError> {
        metadata.controls::<(Self, Vec<bool>, Option<usize>, Result<usize, usize>)>()?;
        if let Some(context) = metadata.context() {
            context
                .charge_metadata(std::mem::size_of_val(&declarations))
                .map_err(eredu_nn::Error::from)?;
        }
        if !std::ptr::eq(self.data().selected.requirements(), selected.requirements())
            || !std::ptr::eq(
                self.data().selected.materialization_tasks(),
                selected.materialization_tasks(),
            )
        {
            return Err(metadata.message(format_args!(
                "materialization source belongs to a different selected realization"
            )));
        }
        // Input order and duplicate spelling never change the canonical set.
        // The only temporary allocation is bounded by this exact retained set.
        let expected = &self.data().declared;
        let mut seen = metadata.vector(expected.len())?;
        seen.resize(expected.len(), false);
        for name in declarations {
            let index = expected
                .binary_search_by(|expected| expected.as_str().cmp(name))
                .map_err(|_| {
                    metadata.message(format_args!(
                        "materialization source does not declare addressable parameter {name:?}"
                    ))
                })?;
            seen[index] = true;
        }
        if seen.iter().any(|seen| !seen) {
            return Err(metadata.message(format_args!(
                "materialization source requires its complete declared addressable set"
            )));
        }
        Ok(())
    }

    // Only ordinary native materialization consumes owned task metadata. Checked
    // equation consumers retain this source or consume execution geometry.
    pub(super) fn into_parts(mut self) -> (Vec<ReplicatedTextMaterializationTask>, Vec<String>) {
        match Arc::try_unwrap(self.0.take().expect("materialization source is live")) {
            Ok(source) => (source.tasks, source.addressable),
            Err(source) => (source.tasks.clone(), source.addressable.clone()),
        }
    }
}
