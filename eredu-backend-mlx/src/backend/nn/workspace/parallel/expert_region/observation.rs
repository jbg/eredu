//! Capture source aliases inside the existing local expert numerical child.
use super::*;
use crate::backend::array_copy::CaptureNativePopulation;
use eredu_nn::{GroupedUnitBatch, GroupedUnitObserver};

/// Per-invocation input completion and per-delivered-batch sources. The caller
/// supplies populations from its actual observer program; these are not grants.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertLocalObservationSource {
    input: CaptureNativePopulation,
    before: CaptureNativePopulation,
    after: CaptureNativePopulation,
    dtype: Option<eredu_nn::workspace::WorkspaceFloatingType>,
    parent_input: CaptureNativePopulation,
}
impl ExpertLocalObservationSource {
    pub(crate) fn new(
        input: CaptureNativePopulation,
        before: CaptureNativePopulation,
        after: CaptureNativePopulation,
    ) -> Option<Self> {
        if input.publications != 0 || input.completions != input.retained_roots {
            return None;
        }
        for value in [before, after] {
            if value.retained_roots % 5 != 0
                || value.completions != value.retained_roots
                || value.publications % 5 != 0
                || value.publications > value.retained_roots
            {
                return None;
            }
        }
        Some(Self {
            input,
            before,
            after,
            dtype: None,
            parent_input: CaptureNativePopulation::default(),
        })
    }
    pub(crate) fn from_addressable(
        source: eredu_nn::workspace::WorkspaceAddressableObservationSource,
    ) -> Option<Self> {
        let result =
            Self::from_descriptor(eredu_nn::workspace::WorkspaceExpertObservationSource {
                before: source.before,
                after: source.after,
                unit_dtype: source.unit_dtype,
                ..Default::default()
            })?;
        Some(result)
    }
    pub(super) fn from_descriptor(
        source: eredu_nn::workspace::WorkspaceExpertObservationSource,
    ) -> Option<Self> {
        source.validate().ok()?;
        let batch = |value: eredu_nn::workspace::WorkspaceGroupedSourceRetention| {
            Some(CaptureNativePopulation {
                publications: value.partition_copies.checked_mul(5)?,
                completions: value.source_loans()?,
                retained_roots: value.source_loans()?,
                controls: value.callback_control_bytes,
            })
        };
        let mut result = Self::new(
            CaptureNativePopulation::default(),
            batch(source.before)?,
            batch(source.after)?,
        )?;
        // The ordinary invocation callback precedes entry to the local child.
        result.parent_input = CaptureNativePopulation {
            publications: 0,
            completions: source.input_settlements,
            retained_roots: source.input_settlements,
            controls: source.input_control_bytes,
        };
        result.dtype = source.unit_dtype;
        Some(result)
    }
    pub(super) fn parent_population(self, publications: usize) -> CaptureNativePopulation {
        CaptureNativePopulation {
            publications,
            ..self.parent_input
        }
    }
    pub(super) fn publications(
        self,
        bank: &WorkspaceGroupedBank,
        rows: usize,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<usize, Error> {
        if rows == 0 {
            return Ok(0);
        }
        let schedule = mechanism
            .grouped_observation_schedule(
                bank,
                u32::try_from(rows)
                    .map_err(|_| eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
            )?
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        let calls = match schedule {
            eredu_nn::workspace::WorkspaceGroupedObservationSchedule::WholeBatch => 1,
            eredu_nn::workspace::WorkspaceGroupedObservationSchedule::TokenChunks(chunk) => {
                rows.div_ceil(chunk.get() as usize)
            }
        };
        self.before
            .publications
            .checked_add(self.after.publications)
            .and_then(|n| n.checked_mul(calls))
            .ok_or_else(|| eredu_nn::workspace::WorkspaceMetadataError::Overflow.into())
    }
}

pub(crate) struct SourceObserver {
    source: ExpertLocalObservationSource,
    roots: Vec<WorkspaceTensor>,
    groups: Option<WorkspaceTensor>,
    population: CaptureNativePopulation,
    context: WorkspaceContext,
    representation: Option<WorkspaceRepresentation>,
}
impl SourceObserver {
    pub(crate) fn new(
        source: ExpertLocalObservationSource,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(size_of::<(
            Self,
            ExpertLocalObservationSource,
            CaptureNativePopulation,
            Result<Self, Error>,
        )>())?;
        let mut roots = context.metadata_vec(source.input.retained_roots)?;
        for _ in 0..source.input.retained_roots {
            roots.push(input.clone());
        }
        Ok(Self {
            source,
            roots,
            groups: None,
            population: source.input,
            context: context.clone(),
            representation: None,
        })
    }
    pub(super) fn bind_groups(
        &mut self,
        routes: &GroupSelection<WorkspaceTensor>,
    ) -> Result<(), Error> {
        if self.groups.is_some() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        self.groups = Some(routes.group_indices().clone());
        Ok(())
    }
    pub(crate) fn bind_source_groups(&mut self, groups: &WorkspaceTensor) -> Result<(), Error> {
        if self.groups.is_some() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        self.groups = Some(groups.clone());
        Ok(())
    }
    pub(crate) fn representation(&self) -> Option<WorkspaceRepresentation> {
        self.representation
    }
    pub(crate) fn finish_report(
        mut self,
        outputs: &[WorkspaceTensor],
    ) -> Result<(WorkspaceTraceReport, CaptureNativePopulation), Error> {
        self.context
            .reserve_metadata_vec(&mut self.roots, outputs.len())?;
        self.roots.extend(outputs.iter().cloned());
        Ok((self.context.finish_report(&self.roots)?, self.population))
    }
    fn record(
        &mut self,
        batch: &GroupedUnitBatch<'_, WorkspaceTensor>,
        population: CaptureNativePopulation,
    ) -> Result<(), Error> {
        if self.source.dtype.is_some_and(|expected| {
            batch.values.layout().representation().map(|v| v.dtype()) != Some(expected)
        }) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        let actual = batch.values.layout().representation();
        if self
            .representation
            .is_some_and(|prior| Some(prior) != actual)
        {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        self.representation = actual;
        let groups = self
            .groups
            .as_ref()
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?;
        self.context
            .reserve_metadata_vec(&mut self.roots, population.retained_roots)?;
        for _ in 0..population.retained_roots / 5 {
            self.roots.extend([
                batch.values.clone(),
                batch.token_indices.clone(),
                batch.selection_indices.clone(),
                batch.coefficients.clone(),
                groups.clone(),
            ]);
        }
        self.population = self
            .population
            .checked_add(population)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        Ok(())
    }
}
impl GroupedUnitObserver<WorkspaceTensor> for SourceObserver {
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
        self.record(batch, self.source.before)
    }
    fn observe_effective(
        &mut self,
        batch: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.record(batch, self.source.after)
    }
}

pub(super) fn finish(
    context: &WorkspaceContext,
    outputs: &[WorkspaceTensor],
    observed: Option<SourceObserver>,
    mechanism: ResidentExecutionMechanisms,
) -> Result<SpeculativeNumericalRecipe, Error> {
    let Some(mut observed) = observed else {
        let report = context.finish_report(outputs)?;
        return SpeculativeNumericalRecipe::inspect_owned_child(
            &report,
            outputs.len(),
            mechanism,
            context,
        );
    };
    context.reserve_metadata_vec(&mut observed.roots, outputs.len())?;
    observed.roots.extend(outputs.iter().cloned());
    let report = context.finish_report(&observed.roots)?;
    SpeculativeNumericalRecipe::inspect_owned_child_with_capture(
        &report,
        outputs.len(),
        mechanism,
        context,
        observed.population,
    )
}
