//! Actual resident program populations and one replacement of their old peaks.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use crate::backend::{error::Error, nn::workspace::{ResidentNativeRecipe, ResidentSamplingProgram}};
use eredu_core::{ExecutionWorkspaceEstimate, WorkspaceBound};
use eredu_nn::workspace::WorkspaceTraceReport;
use eredu_runtime::working_memory::{
    InferenceSpanWorkspace, InferenceSpanWorkspacePlan, NativeStorageCarryoverReport,
    NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder,
    PreparedNativeStoragePlan, RegisteredWorkspaceCopy, SamplingWorkspaceReport,
    TextPromptWorkspaceReport, WorkingMemoryError,
};

struct ConstructedPreparationPopulation {
    bytes: u64,
    births: usize,
    retained_roots: usize,
    controls: u64,
    credit: u64,
    replacement: u64,
}
/// Completion and new construction are distinct source profiles. A completed
/// B contributes no new input birth to Q; its exact source pin stays in the
/// enclosing residual quote. Only that closed source witness selects this arm.
enum PreparationPopulation<'a> {
    Constructed(ConstructedPreparationPopulation),
    Completed(&'a eredu_runtime::input::OriginalPreparedWorkspaceSource),
}
pub(crate) struct NativeProgramStorage {
    plan: InferenceSpanWorkspacePlan,
    selection: NativeStorageSelection,
    capacity: u64,
    carryover: NativeStorageCarryoverReport,
    envelope: NativePrefillEnvelope,
    prompt_bytes: u64,
    prompt_replacement_bytes: u64,
    sampling_bytes: u64,
    prompt_credit: u64,
    sampling_credit: u64,
    population_controls: u64,
    attempts: usize,
    rows: usize,
    works: usize,
}
impl NativeProgramStorage {
    pub(crate) fn capacity_bytes(&self) -> u64 { self.capacity }
    pub(crate) fn direct_control_bytes(&self, mechanism: &MlxNativeStorage,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>)
        -> Result<Option<u64>, Error> {
        metadata.admit::<(super::control_storage::NativePublicationOwnerLayout, Option<super::control_storage::NativePublicationOwnerLayout>,
            Result<Option<super::control_storage::NativePublicationOwnerLayout>, NativeStorageCause>,
            Result<Option<u64>, Error>, &Self, &MlxNativeStorage, usize)>()
            .map_err(|cause| Error::Neural(metadata.error(cause)))?;
        let capacity = usize::try_from(self.capacity)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        mechanism.publication_owner_layout(capacity)
            .map_err(|cause| metadata.source(cause)).map_err(Error::Neural)
            .map(|layout| layout.and_then(|layout| layout.control_bytes(self.attempts, self.rows)))
    }
    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn works(&self) -> usize {
        self.works
    }
    pub(crate) fn replace_enclosing(
        &self,
        outside: &mut ExecutionWorkspaceEstimate,
    ) -> Result<(), Error> {
        self.replace_enclosing_metadata(
            outside,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
    }
    pub(crate) fn replace_enclosing_metadata(
        &self,
        outside: &mut ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<(), Error> {
        if outside.geometry != self.plan.geometry() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        replace(
            &mut outside.activations,
            self.prompt_credit,
            self.prompt_replacement_bytes,
            "; original preparation native term replaced by its actual selected source capacity",
            metadata,
        )?;
        replace(
            &mut outside.vocabulary,
            self.sampling_credit,
            self.sampling_bytes,
            "; original sampling native peak replaced by all actual sampled generations, excluding equation-owned input roots",
            metadata,
        )
    }
}
fn replace(
    bound: &mut WorkspaceBound,
    old: u64,
    new: u64,
    reason: &str,
    metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
) -> Result<(), Error> {
    let WorkspaceBound::Bounded { bytes, assumptions } = bound else {
        return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
    };
    *bytes = bytes
        .checked_sub(old)
        .and_then(|n| n.checked_add(new))
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
    metadata
        .append(assumptions, reason)
        .map_err(|cause| Error::Neural(metadata.error(cause)))?;
    Ok(())
}
fn add(left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right)
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
}
pub(crate) struct NativeSamplingPopulation {
    pub(crate) bytes: u64,
    pub(crate) controls: u64,
    pub(crate) carryover: Option<u64>,
    work_rows: super::work_rows::WorkRows,
}
impl MlxNativeStorage {
    pub(crate) fn sampling_population(&self, program: &ResidentSamplingProgram) -> Result<Option<NativeSamplingPopulation>, Error> {
        self.sampling_population_with(program, |_| {})
    }
    fn sampling_population_with(&self, program: &ResidentSamplingProgram,
        mut generation: impl FnMut(u64)) -> Result<Option<NativeSamplingPopulation>, Error> {
        if let Some(funding) = program.planning_metadata() {
            funding.reserve_metadata(std::mem::size_of::<(
                super::work_rows::WorkRows, &super::work_rows::WorkRows,
                Option<eredu_nn::workspace::WorkspaceStoragePopulation>, Option<usize>, usize,
            )>()).map_err(Error::WorkspacePlanning)?;
        }
        let mut sampling_bytes = 0u64;
        let mut population_controls = std::mem::size_of::<super::work_rows::WorkRows>() as u64;
        let mut work_rows = super::work_rows::WorkRows::default();
        let mut sampling_carryover = Some(0u64);
        for row in program.rows() {
            let closing =
                self.carryover_population(row.closing_roots(), &mut population_controls)?;
            sampling_carryover = sampling_carryover.zip(closing).map(|(a, b)| a.max(b));
            let Some(storage) = row.mutable_storage() else {
                return Ok(None);
            };
            let Some(closing) = row.closing_roots() else {
                return Ok(None);
            };
            work_rows.sampling(storage.maximum_births(), closing.maximum_allocations)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let Some(population) =
                self.original_population(storage.mutable_bytes(), storage.maximum_births())?
            else {
                return Ok(None);
            };
            generation(population.capacity() as u64);
            sampling_bytes = add(sampling_bytes, population.capacity() as u64)?;
            population_controls = add(population_controls, population.control_bytes() as u64)?;
        }
        Ok(Some(NativeSamplingPopulation {
            bytes: sampling_bytes,
            work_rows,
            controls: population_controls,
            carryover: sampling_carryover,
        }))
    }

    fn original_population(
        &self,
        requested_bytes: u64,
        maximum_births: usize,
    ) -> Result<Option<safemlx::OriginalBufferPopulationLayout>, Error> {
        let runtime = self
            .runtime
            .as_ref()
            .map_err(|cause| Error::Other(Box::new(NativeStorageCause::Cold(cause.retained()))))?;
        let requested = usize::try_from(requested_bytes)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        match OriginalBufferBudget::population_layout(runtime, requested, maximum_births) {
            Ok(layout) => Ok(Some(layout)),
            Err(OriginalBufferCause::Unsupported) => Ok(None),
            Err(cause) => Err(Error::Other(Box::new(NativeStorageCause::Fixed(cause)))),
        }
    }

    fn carryover_population(
        &self,
        population: Option<eredu_nn::workspace::WorkspaceStoragePopulation>,
        controls: &mut u64,
    ) -> Result<Option<u64>, Error> {
        let frames = [
            std::mem::size_of::<Option<eredu_nn::workspace::WorkspaceStoragePopulation>>(),
            std::mem::size_of::<eredu_nn::workspace::WorkspaceStoragePopulation>(),
            std::mem::size_of::<Result<Option<u64>, Error>>(),
            std::mem::size_of::<&mut u64>(),
        ];
        let frame_bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        *controls = add(*controls, frame_bytes)?;
        let Some(population) = population else {
            return Ok(None);
        };
        let Some(bytes) = population.bytes else {
            return Ok(None);
        };
        let Some(layout) = self.original_population(bytes, population.maximum_allocations)? else {
            return Ok(None);
        };
        *controls = add(*controls, layout.control_bytes() as u64)?;
        Ok(Some(layout.capacity() as u64))
    }

    /// The caller selects this only for the no-capture registered resident
    /// program. Inputs are the actual paired trace and enclosing reports, not
    /// geometry-generated native occurrences or a requested arena capacity.
    pub(crate) fn resident_program(
        &self,
        recipe: &ResidentNativeRecipe,
        prompt: &TextPromptWorkspaceReport,
        sampling: &SamplingWorkspaceReport,
        opening_rows: Option<usize>,
        paged_sources: bool,
    ) -> Result<Option<NativeProgramStorage>, Error> {
        let geometry = recipe.plan().geometry();
        let Some(prompt_facts) = self
            .prompt_input_facts(geometry)
            .map_err(|cause| Error::Other(Box::new(cause)))?
        else {
            return Ok(None);
        };
        let Some(prompt_credit) = prompt.tensor_peak_bytes() else {
            return Ok(None);
        };
        let prompt_bytes = u64::try_from(prompt_facts.mutable_bytes())
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        self.program(
            recipe,
            sampling,
            opening_rows,
            paged_sources,
            PreparationPopulation::Constructed(ConstructedPreparationPopulation {
                bytes: prompt_bytes,
                births: prompt_facts.maximum_births(),
                retained_roots: 1, // actual constructed prompt array
                controls: 0,
                credit: prompt_credit,
                replacement: prompt_bytes,
            }),
        )
    }

    /// The same native partition for a fresh saved-source resume. The closed
    /// copy and pending reports already price selected allocator capacity and
    /// copy scratch. Add only the amount by which actual original backing
    /// exceeds those same new-tensor terms; old source credit is unchanged.
    pub(crate) fn resume_program(
        &self,
        recipe: &ResidentNativeRecipe,
        source: &RegisteredWorkspaceCopy<StorageIdentity>,
        pending: &WorkspaceTraceReport,
        sampling: &SamplingWorkspaceReport,
        opening_rows: Option<usize>,
        paged_sources: bool,
    ) -> Result<Option<NativeProgramStorage>, Error> {
        let Some(copy) = recipe.resume_copy() else {
            return Ok(None);
        };
        let Some(population) =
            self.original_population(copy.logical_bytes() as u64, copy.births())?
        else {
            return Ok(None);
        };
        let Some((source_new, pending_new)) = source
            .report()
            .tensor_buffers
            .total_bytes
            .zip(pending.tensor_buffers.total_bytes)
        else {
            return Ok(None);
        };
        // The immutable registered program supplies decoder/key demand. The
        // actual pending trace supplies its independent program, excluding its
        // already charged opening root. Both include the ordinary emitter's
        // allocator allowance and copy-compaction scratch; neither is raw N.
        let priced = add(source_new, pending_new)?;
        let physical = u64::try_from(population.capacity())
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let controls = [
            population.control_bytes(),
            std::mem::size_of::<(
                &RegisteredWorkspaceCopy<StorageIdentity>,
                &WorkspaceTraceReport,
            )>(),
            std::mem::size_of::<Option<(u64, u64)>>(),
            std::mem::size_of::<u64>() * 2,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        self.program(
            recipe,
            sampling,
            opening_rows,
            paged_sources,
            PreparationPopulation::Constructed(ConstructedPreparationPopulation {
                bytes: physical.max(priced),
                births: copy.births(),
                retained_roots: copy.roots(),
                controls,
                credit: 0,
                replacement: physical.saturating_sub(priced),
            }),
        )
    }

    /// Uses the actual completed input profile. No eager token constructor,
    /// descriptor, host staging, or input identity is repeated in this request.
    pub(crate) fn completed_input_program(
        &self,
        recipe: &ResidentNativeRecipe,
        source: &eredu_runtime::input::OriginalPreparedWorkspaceSource,
        sampling: &SamplingWorkspaceReport,
        opening_rows: Option<usize>,
        paged_sources: bool,
    ) -> Result<Option<NativeProgramStorage>, Error> {
        self.program(
            recipe,
            sampling,
            opening_rows,
            paged_sources,
            PreparationPopulation::Completed(source),
        )
    }

    fn program(
        &self,
        recipe: &ResidentNativeRecipe,
        sampling: &SamplingWorkspaceReport,
        opening_rows: Option<usize>,
        paged_sources: bool,
        preparation: PreparationPopulation<'_>,
    ) -> Result<Option<NativeProgramStorage>, Error> {
        if let Some(funding) = recipe.planning_metadata() {
            funding.reserve_metadata(std::mem::size_of::<(
                super::work_rows::WorkRows, &super::work_rows::WorkRows,
                Option<eredu_nn::workspace::WorkspaceStoragePopulation>, Option<usize>, usize, bool,
            )>()).map_err(Error::WorkspacePlanning)?;
        }
        let preparation = match preparation {
            PreparationPopulation::Constructed(population) => population,
            PreparationPopulation::Completed(source) => {
                if source.borrowed_storage().is_none() {
                    return Ok(None);
                }
                // These zeros describe an absent producer, not an upload with
                // fabricated zero-valued native facts. B owns all existing leaves.
                ConstructedPreparationPopulation {
                    bytes: 0,
                    births: 0,
                    retained_roots: 0,
                    controls: 0,
                    credit: 0,
                    replacement: 0,
                }
            }
        };
        let geometry = recipe.plan().geometry();
        let Some(opening_rows) = opening_rows else {
            return Ok(None);
        };
        let Some(sampling_credit) = sampling.tensor_peak_bytes else {
            return Ok(None);
        };
        let steps = usize::try_from(geometry.max_output_tokens)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        if sampling.steps != geometry.max_output_tokens
            || recipe.sampling_records().len()
                != steps
                    .checked_add(1)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?
        {
            return Ok(None);
        }
        let mut capacity = 0u64;
        let mut work_rows = super::work_rows::WorkRows::default();
        let profile_controls = u64::try_from(
            std::mem::size_of::<PreparationPopulation<'_>>()
                .checked_add(std::mem::size_of::<ConstructedPreparationPopulation>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<super::work_rows::WorkRows>()))
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let mut population_controls = add(
            add(preparation.controls, profile_controls)?,
            u64::try_from(
                NativePrefillEnvelopeBuilder::control_bytes()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )?;
        // Actual prefill finish and model completion both require an exact
        // Record drain before the shared driver can advance this request.
        let mut envelope = NativePrefillEnvelopeBuilder::new_completed_equations(recipe.plan());
        let mut state_carryover = Some(0u64);
        let mut output_carryover = Some(0u64);
        let mut validation_carryover = Some(0u64);
        for (native, trace) in recipe.records().iter().zip(recipe.plan().records()) {
            let opening =
                self.carryover_population(native.opening_state(), &mut population_controls)?;
            let state =
                self.carryover_population(Some(native.closing_state()), &mut population_controls)?;
            state_carryover = state_carryover.zip(state).map(|(a, b)| a.max(b));
            let output =
                self.carryover_population(native.current_output(), &mut population_controls)?;
            output_carryover = output_carryover.zip(output).map(|(a, b)| a.max(b));
            let population = native.validation_producers().map(|p| {
                eredu_nn::workspace::WorkspaceStoragePopulation {
                    bytes: Some(p.mutable_bytes()),
                    maximum_allocations: p.maximum_births(),
                }
            });
            let validation = self.carryover_population(population, &mut population_controls)?;
            if matches!(
                native.span(),
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Prefill(_)
            ) {
                validation_carryover = validation_carryover
                    .zip(validation)
                    .map(|(a, b)| add(a, b))
                    .transpose()?;
            }
            let (Some(storage), Some(old_native)) = (
                native.mutable_storage(),
                trace.new_tensor_allocation_bytes(),
            ) else {
                return Ok(None);
            };
            let (Some(opening_roots), Some(validations)) =
                (native.opening_state(), native.validation_roots()) else {
                return Ok(None);
            };
            let prefill = match native.span() {
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Prefill(_) => true,
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Decode { .. } => false,
                eredu_runtime::working_memory::InferenceWorkspaceSpan::Sampling(_) => return Ok(None),
            };
            work_rows.equation(
                prefill,
                storage.maximum_births(), native.capture_roots(), validations,
                opening_roots.maximum_allocations, native.closing_state().maximum_allocations,
            ).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
            let Some(population) =
                self.original_population(storage.mutable_bytes(), storage.maximum_births())?
            else {
                return Ok(None);
            };
            // Ordinary emitters bound requested bytes (including their cache
            // allowance), while original backing rounds every positive birth
            // independently. Keep the whole current equation until exact
            // successful retirement; no per-primitive release is assumed.
            let construction = (population.capacity() as u64).max(old_native);
            envelope
                .push(
                    native.span(),
                    NativeEquationStorage {
                        construction_bytes: construction,
                        opening_state_bytes: opening,
                        closing_state_bytes: state,
                        output_bytes: output,
                        validation_producer_bytes: validation,
                    },
                )
                .map_err(Error::PrefillControl)?;
            capacity = add(capacity, construction)?;
            population_controls = add(population_controls, population.control_bytes() as u64)?;
        }
        let envelope = envelope.finish().map_err(Error::PrefillControl)?;
        if envelope.original_equation_bytes() != capacity {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        capacity = envelope.equation_bytes();
        if let Some(source)=recipe.parallel_control_source()? {
            // Controls are synchronous at the shared policy boundary. Their
            // one actual status program can coexist with all retained model
            // roots, so add its physical peak to this SAME request-wide bank.
            // A failed/pending vote forbids another role until terminal proof;
            // it is not treated as released capacity by the native worker.
            let control=source.control_capacity()?;
            capacity=add(capacity,u64::try_from(control.backing)
                .map_err(|_|Error::PrefillControl(WorkingMemoryError::Overflow))?)?;
        }
        let Some(sampling_population) = self.sampling_population(recipe.sampling_program())? else {
            return Ok(None);
        };
        let sampling_bytes = sampling_population.bytes;
        let sampling_carryover = sampling_population.carryover;
        population_controls = add(population_controls, sampling_population.controls)?;
        work_rows.include_sampling(&sampling_population.work_rows);
        let prompt_bytes = preparation.bytes;
        capacity = add(add(capacity, prompt_bytes)?, sampling_bytes)?;
        let capture_publications = recipe.records().iter().try_fold(0usize, |n, row| {
            n.checked_add(row.capture_publications())
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
        })?;
        // Work has one prompt and one sampler preparation, and at most one
        // inference owner per reserved output. Each inference may publish its
        // actual nonstate and decoder inventories once. Every accepted actual
        // capture callback also consumes one pure-native source publication.
        let attempts = steps
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_add(capture_publications))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let works = steps
            .checked_add(2)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        // This same bound sizes each real Work's clone/metadata inventories and
        // each publication's registry/attachment destinations. Other Work owners
        // retain their own histories; none is copied into the current inventory.
        let rows = work_rows.finish(opening_rows, preparation.births,
            preparation.retained_roots, recipe.foreground_source_publication_rows(), paged_sources)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        Ok(Some(NativeProgramStorage {
            plan: recipe.plan().clone(),
            selection: self.selection.clone(),
            capacity,
            envelope,
            carryover: NativeStorageCarryoverReport {
                prepared_input_bytes: prompt_bytes,
                equation_state_bytes: state_carryover,
                equation_output_bytes: output_carryover,
                sampling_roots_bytes: sampling_carryover,
                prefill_validation_producers_bytes: validation_carryover,
                equation_records: recipe.records().len(),
                sampling_records: recipe.sampling_records().len(),
            },
            prompt_bytes,
            prompt_replacement_bytes: preparation.replacement,
            sampling_bytes,
            prompt_credit: preparation.credit,
            sampling_credit,
            population_controls,
            attempts,
            rows,
            works,
        }))
    }

    pub(crate) fn selected_plan(
        &self,
        workspace: &InferenceSpanWorkspace,
        recipe: Option<&ResidentNativeRecipe>,
        program: Option<&NativeProgramStorage>,
        collector_controls: Option<u64>,
    ) -> Result<PreparedNativeStoragePlan<Self>, Error> {
        if let Some(recipe) = recipe {
            if !recipe.plan().same_plan(workspace.plan())
                || recipe.records().len() != workspace.plan().records().len()
                || recipe
                    .records()
                    .iter()
                    .zip(workspace.plan().records())
                    .any(|(native, equation)| native.span() != equation.span())
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        if let Some(program) = program {
            if !program.plan.same_plan(workspace.plan())
                || !program.selection.same_selection(&self.selection)
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        let capacity = program.map(|p| p.capacity);
        let population = program.map(|p| (p.attempts, p.rows));
        let direct = if let Some(program) = program {
            let capacity = usize::try_from(program.capacity)
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
            self.publication_owner_layout(capacity)
                .map_err(|cause| Error::Other(Box::new(cause)))?
                .and_then(|layout| layout.control_bytes(program.attempts, program.rows))
        } else {
            None
        };
        let provider = direct
            .zip(collector_controls)
            .map(|(a, b)| {
                a.checked_add(b)
                    .and_then(|n| {
                        n.checked_add(
                            (2 * std::mem::size_of::<NativeProgramStorage>()
                                + std::mem::size_of::<PreparationPopulation>())
                                as u64,
                        )
                        .and_then(|n| n.checked_add(program?.population_controls))
                    })
                    .ok_or(WorkingMemoryError::Overflow)
            })
            .transpose()
            .map_err(Error::PrefillControl)?;
        let covered = workspace
            .plan()
            .records()
            .iter()
            .enumerate()
            .map(|(index, trace)| {
                recipe?.records()[index].mutable_storage()?;
                trace
                    .new_tensor_allocation_bytes()?
                    .checked_add(program?.prompt_bytes)
            });
        let plan = PreparedNativeStoragePlan::prepare_qualified(
            workspace, self, capacity, population, covered, provider,
        )
        .map_err(Error::PrefillControl)?;
        if let Some(program) = program.filter(|_| recipe.is_some()) {
            plan.with_completed_prefill_envelope(program.envelope.clone())
                .and_then(|plan| plan.with_carryover_report(program.carryover))
                .map_err(Error::PrefillControl)
        } else {
            Ok(plan)
        }
    }
}

/// Sampling-only projection of the same physical population worker. Every
/// generation is retained; no retirement credit is inferred from a cold trace.
pub(crate) struct NativeSamplingStorage {
    plan: InferenceSpanWorkspacePlan,
    selection: NativeStorageSelection,
    generations: Vec<u64>,
    population: NativeSamplingPopulation,
    attempts: usize,
    rows: usize,
    works: usize,
}
impl NativeSamplingStorage {
    pub(crate) fn attempts(&self) -> usize { self.attempts }
    pub(crate) fn rows(&self) -> usize { self.rows }
    pub(crate) fn works(&self) -> usize { self.works }
}
impl MlxNativeStorage {
    pub(crate) fn sampling_program(&self, workspace: &InferenceSpanWorkspace,
        program: &ResidentSamplingProgram, reseed: bool, funding: &eredu_core::HostMetadataFunding) -> Result<NativeSamplingStorage, Error> {
        use eredu_nn::workspace::WorkspaceMetadataAllocation;
        funding.reserve_metadata(std::mem::size_of::<NativeSamplingStorage>()
            + std::mem::size_of::<Result<NativeSamplingStorage, Error>>()
            + std::mem::size_of::<NativeSamplingPopulation>())
            .map_err(Error::WorkspacePlanning)?;
        if workspace.plan().records().len() != program.rows().len()
            || workspace.plan().records().iter().zip(program.rows()).any(|(neutral, native)|
                neutral.span() != &eredu_runtime::working_memory::InferenceWorkspaceSpan::Sampling(native.phase())) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut generations = funding.metadata_vec(program.rows().len()).map_err(Error::Neural)?;
        let population = self.sampling_population_with(program, |bytes| generations.push(bytes))?
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let works = program.steps().and_then(|n| n.checked_add(usize::from(reseed)))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        // Each actual replacement Work contains one phase; its invocation
        // sources and both explicit handoff retains remain covered separately.
        let rows = population.work_rows.sampling_only()
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        Ok(NativeSamplingStorage { plan: workspace.plan().clone(), selection: self.selection.clone(),
            generations, population, attempts: works, rows, works })
    }
    pub(crate) fn sampling_plan(&self, workspace: &InferenceSpanWorkspace,
        program: &NativeSamplingStorage, collector_controls: u64)
        -> Result<PreparedNativeStoragePlan<Self>, Error> {
        if !program.plan.same_plan(workspace.plan()) || !program.selection.same_selection(&self.selection) {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let capacity = usize::try_from(program.population.bytes)
            .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let direct = self.publication_owner_layout(capacity)
            .map_err(|cause| Error::Other(Box::new(cause)))?
            .and_then(|layout| layout.control_bytes(program.attempts, program.rows))
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let provider = direct.checked_add(collector_controls)
            .and_then(|n| n.checked_add(program.population.controls))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        PreparedNativeStoragePlan::prepare_qualified(workspace, self, Some(program.population.bytes),
            Some((program.attempts, program.rows)),
            workspace.plan().records().iter().map(|row| row.new_tensor_allocation_bytes()), Some(provider))
            .and_then(|plan| plan.with_retained_equation_generations(program.generations.iter().copied().map(Some)))
            .map_err(Error::PrefillControl)
    }
}
