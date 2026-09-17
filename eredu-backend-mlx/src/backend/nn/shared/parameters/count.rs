//! Fixed observation of an existing source under the caller's runtime owner.
use crate::MlxTensor;
use eredu_nn::{
    ParameterMetadataView, ParameterSourceError, ParameterSourceVisitor, Parameterized,
};
use safemlx::{ArrayDescriptorError, RuntimeCallGuard};
use std::collections::BTreeMap;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParameterSourceCounts {
    pub(crate) named_slots: usize,
    pub(crate) auxiliary_slots: usize,
    pub(crate) metadata_name_bytes: usize,
    pub(crate) shape_elements: usize,
    pub(crate) unknown_backings: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ParameterCountError {
    #[error("borrowed parameter source traversal failed")]
    Traversal(#[source] ParameterSourceError),
    #[error("borrowed parameter descriptor failed at slot {slot}")]
    Descriptor {
        slot: usize,
        #[source]
        source: ArrayDescriptorError,
    },
    #[error("borrowed parameter source count overflow")]
    CountOverflow,
}

pub(crate) fn add_parameter_count(
    total: &mut usize,
    amount: usize,
) -> Result<(), ParameterCountError> {
    *total = total
        .checked_add(amount)
        .ok_or(ParameterCountError::CountOverflow)?;
    Ok(())
}

/// All rows are actual retained map entries; keys are not ParameterSpec values.
/// Both ordinary inventory and fixed counting use this same borrowed worker.
pub(crate) fn visit_parameter_map<'a, E>(
    values: &'a BTreeMap<String, MlxTensor>,
    mut visitor: impl FnMut(&'a str, &'a MlxTensor) -> Result<(), E>,
) -> Result<(), E> {
    for (key, value) in values {
        visitor(key, value)?;
    }
    Ok(())
}

pub(crate) struct ParameterSourceCounter<'g> {
    pub(crate) guard: &'g mut RuntimeCallGuard,
    pub(crate) counts: ParameterSourceCounts,
    pub(crate) slot: usize,
    pub(crate) failure: Option<ParameterCountError>,
}
impl<'g> ParameterSourceCounter<'g> {
    pub(crate) fn new(guard: &'g mut RuntimeCallGuard) -> Self {
        Self {
            guard,
            counts: Default::default(),
            slot: 0,
            failure: None,
        }
    }

    pub(crate) fn observe_source<T: Parameterized<MlxTensor> + ?Sized>(
        &mut self,
        source: &T,
    ) -> Result<(), ParameterCountError> {
        let coverage = source.visit_parameter_sources(self);
        // Independent channels retain observer-first precedence. A later
        // source failure cannot discard an actual earlier descriptor cause.
        if let Some(error) = self.failure {
            return Err(error);
        }
        coverage.map_err(ParameterCountError::Traversal)
    }

    pub(crate) fn observe_auxiliary(
        &mut self,
        value: &MlxTensor,
    ) -> Result<(), ParameterCountError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self.observe(None, value);
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    fn observe(
        &mut self,
        metadata: Option<ParameterMetadataView<'_>>,
        value: &MlxTensor,
    ) -> Result<(), ParameterCountError> {
        if let Some(metadata) = metadata {
            add_parameter_count(&mut self.counts.named_slots, 1)?;
            add_parameter_count(
                &mut self.counts.metadata_name_bytes,
                metadata.id().as_str().len(),
            )?;
            for name in [
                metadata.alias_of().map(|id| id.as_str()),
                metadata.group(),
                metadata.linear_companion_of().map(|id| id.as_str()),
            ]
            .into_iter()
            .flatten()
            {
                add_parameter_count(&mut self.counts.metadata_name_bytes, name.len())?;
            }
        } else {
            add_parameter_count(&mut self.counts.auxiliary_slots, 1)?;
        }
        let descriptor = self.guard.descriptor(value.as_array()).map_err(|source| {
            ParameterCountError::Descriptor {
                slot: self.slot,
                source,
            }
        })?;
        let facts = descriptor.facts();
        add_parameter_count(&mut self.counts.shape_elements, facts.rank())?;
        // Some includes both certified no-storage empty and physical zero-byte
        // storage. Counts never deduplicate aliases or grant storage credit.
        add_parameter_count(
            &mut self.counts.unknown_backings,
            usize::from(facts.allocation().is_none()),
        )?;
        add_parameter_count(&mut self.slot, 1)
    }

    fn visit(&mut self, metadata: Option<ParameterMetadataView<'_>>, value: &MlxTensor) {
        if self.failure.is_none() {
            self.failure = self.observe(metadata, value).err();
        }
    }
}
impl<'a> ParameterSourceVisitor<'a, MlxTensor> for ParameterSourceCounter<'_> {
    fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a MlxTensor) {
        self.visit(Some(metadata), value);
    }
    fn retained(&mut self, value: &'a MlxTensor) {
        self.visit(None, value);
    }
}
