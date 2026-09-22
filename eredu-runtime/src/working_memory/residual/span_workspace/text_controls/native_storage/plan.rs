//! Cold selected facts, retaining no native object and creating no future payload.
use super::*;

/// Descriptive source-derived carryover contributions for the exact selected
/// resident program. Independent maxima may overlap. This report intentionally
/// does not certify successful retirement, replace capacity, or issue byte credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeStorageCarryoverReport {
    /// Whole prepared-input/native preparation population remains conservative.
    pub prepared_input_bytes: u64,
    /// Maximum complete equation-closing state backing, including possible aliases.
    pub equation_state_bytes: Option<u64>,
    /// Maximum current semantic output's complete backing before sampling.
    pub equation_output_bytes: Option<u64>,
    /// Maximum advanced RNG plus all emitted-token alias backing across actual phases.
    pub sampling_roots_bytes: Option<u64>,
    /// Whole native producers of every retained prefill validation result.
    /// This intentionally overbounds the results; it is not scalar-shape pricing.
    pub prefill_validation_producers_bytes: Option<u64>,
    /// Exact retained equation record count.
    pub equation_records: usize,
    /// Exact sampling preparation plus permitted step record count.
    pub sampling_records: usize,
}

#[derive(Debug)]
pub(in crate::working_memory) struct NativeStorageLayout {
    pub(super) mechanism: TypeId,
    pub(super) selection: NativeStorageSelection,
    // Full admitted workspace and covered native portion are distinct. The
    // coverage is never inferred from a cap, a row, current usage or headroom.
    // Scalar semantic coordinates only: retaining SpanPlan here would create
    // plan -> original host -> control binding -> plan after promotion.
    spans: Vec<(InferenceWorkspaceSpan, Option<u64>, Option<u64>)>,
    pub(super) capacity: Option<u64>,
    pub(in crate::working_memory) placement: Option<Arc<eredu_core::MemoryPlacement>>,
    carryover: Option<NativeStorageCarryoverReport>,
    // Explicit original-program equation envelope: either all generations or
    // a complete carried-root envelope at actual successful retirement cuts.
    // It replaces the registered equation peak at seal.
    pub(in crate::working_memory) equation_generations: Option<u64>,
    pub(super) population: Option<(usize, usize)>,
    control_bytes: Option<u64>,
    pub(super) exact_storage: bool,
    _funding: Option<eredu_core::HostMetadataFunding>,
}

#[derive(Debug, Clone)]
pub(in crate::working_memory) struct NativeStorageLayoutOwner(Option<Arc<NativeStorageLayout>>);
impl std::ops::Deref for NativeStorageLayoutOwner {
    type Target = NativeStorageLayout;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live native storage layout")
    }
}
impl NativeStorageLayoutOwner {
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live layout"),
            other.0.as_ref().expect("live layout"),
        )
    }
    pub(super) fn get_mut(&mut self) -> Option<&mut NativeStorageLayout> {
        Arc::get_mut(self.0.as_mut().expect("live native storage layout"))
    }
}
impl Drop for NativeStorageLayoutOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
/// Cold facts tied to the exact equation plan and selected mechanism. This is
/// not an allocation grant. Unknown allocator/control or span coverage keeps
/// the original quote incomplete even when the requested cap is known.
pub struct PreparedNativeStoragePlan<M: OriginalNativeStorageMechanism> {
    pub(super) layout: NativeStorageLayoutOwner,
    pub(super) plan: InferenceSpanWorkspacePlan,
    _mechanism: PhantomData<fn() -> M>,
}
impl<M: OriginalNativeStorageMechanism> PreparedNativeStoragePlan<M> {
    /// Bind actual selected full-span requirements and their native decomposition.
    /// `covered` must supply exactly one entry for every actual plan record; no
    /// iterator size hint is trusted. `provider_controls` covers the concrete
    /// native budget/sidecars/borrowed observations and their error/move overlap.
    /// `allocator_and_keys` separately covers managed Arc/Rc headers, exposed
    /// spare capacity and dynamic key/metadata payload beyond requested layouts.
    /// Allocator-private bookkeeping and unrelated process storage are excluded.
    /// `population` supplies the complete finite (attempts, maximum rows) pair;
    /// absent population is unknown, not an assumed zero. Neither term includes
    /// the native payload cap itself.
    pub fn prepare(
        workspace: &InferenceSpanWorkspace,
        selection: &NativeStorageSelection,
        capacity: Option<u64>,
        population: Option<(usize, usize)>,
        covered: impl IntoIterator<Item = Option<u64>>,
        provider_controls: Option<u64>,
        allocator_and_keys: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        let funding = workspace.plan().metadata_funding();
        cold_controls::<(Self, NativeStorageLayout, Result<Self, WorkingMemoryError>)>(
            funding.as_ref(),
        )?;
        if let Some(funding) = &funding {
            funding
                .reserve_metadata(shared_shell::<NativeStorageLayout>()?)
                .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        }
        if let Some(funding) = &funding {
            funding
                .reserve_metadata(std::mem::size_of_val(&covered))
                .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        }
        let mut covered = covered.into_iter();
        if let Some(funding) = &funding {
            funding
                .reserve_metadata(std::mem::size_of_val(&covered))
                .map_err(crate::working_memory::reservation_metadata::funding_error)?;
        }
        let mut spans = match &funding {
            Some(funding) => funding
                .metadata_vec(workspace.plan().records().len())
                .map_err(|error| {
                    crate::working_memory::reservation_metadata::neural_error(error, funding)
                })?,
            None => Vec::with_capacity(workspace.plan().records().len()),
        };
        for index in 0..workspace.plan().records().len() {
            let native = covered.next().ok_or(WorkingMemoryError::IdentityMismatch)?;
            let full = workspace.span_bytes(index);
            if let Some(native) = native {
                if full.is_some_and(|full| native > full)
                    || capacity.is_some_and(|capacity| native > capacity)
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            spans.push((
                workspace.plan().records()[index].span().clone(),
                full,
                native,
            ));
        }
        if covered.next().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let requested = population
            .map(|(attempts, rows)| Self::requested_control_bytes(attempts, rows))
            .transpose()?;
        let payload = u64::try_from(spans.capacity())
            .ok()
            .and_then(|n| {
                n.checked_mul(
                    size_of::<(InferenceWorkspaceSpan, Option<u64>, Option<u64>)>() as u64,
                )
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        let known = [
            requested,
            Some(payload),
            provider_controls,
            allocator_and_keys,
        ]
        .into_iter()
        .flatten()
        .try_fold(0u64, |a, b| {
            a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
        })?;
        let complete = capacity.is_some()
            && population.is_some()
            && provider_controls.is_some()
            && allocator_and_keys.is_some()
            && spans.iter().enumerate().all(|(index, (_, full, native))| {
                (full.is_some()
                    || workspace.physical_outside.is_some()
                        && workspace.plan().records()[index]
                            .domain_allocations()
                            .is_some())
                    && native.is_some()
            });
        Ok(Self {
            layout: NativeStorageLayoutOwner(Some(Arc::new(NativeStorageLayout {
                mechanism: TypeId::of::<M>(),
                selection: selection.clone(),
                spans,
                capacity,
                placement: None,
                carryover: None,
                equation_generations: None,
                population,
                control_bytes: complete.then_some(known),
                exact_storage: false,
                _funding: funding,
            }))),
            plan: workspace.plan().clone(),
            _mechanism: PhantomData,
        })
    }

    /// Retain a producer-certified uniform allocation placement before sealing.
    /// This is descriptive allocation strategy, never an execution grant.
    pub fn with_uniform_placement(mut self, placement: Arc<eredu_core::MemoryPlacement>) -> Self {
        self.layout.get_mut().expect("unpublished layout").placement = Some(placement);
        self
    }

    /// Prepare the same selected plan using qualified finite control producers.
    /// The actual mechanism supplies its per-key nested-storage fact. Missing
    /// qualification, key facts, native coverage or provider owners stays typed
    /// unknown; no population/capacity or allocator-private term is invented.
    ///
    /// `provider_controls` must include the complete concrete budget, attachment,
    /// observation (including any keys it owns), mechanism/budget clone and
    /// error owners, including managed headers and finite overlap. Observation
    /// payloads are additional to the three registry key copies counted here. It excludes the native payload cap itself.
    pub fn prepare_qualified(
        workspace: &InferenceSpanWorkspace,
        mechanism: &M,
        capacity: Option<u64>,
        population: Option<(usize, usize)>,
        covered: impl IntoIterator<Item = Option<u64>>,
        provider_controls: Option<u64>,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(
            Self,
            Result<Self, WorkingMemoryError>,
            Option<u64>,
            Option<(usize, usize)>,
        )>(workspace.plan().metadata_funding().as_ref())?;
        let qualified = crate::working_memory::qualified_storage::qualified();
        let nested = mechanism.key_clone_storage_bytes();
        let complete = qualified && nested.is_some() && population.is_some();
        // Reuse exactly the same coverage selection, cardinality and arithmetic.
        // The term below accounts for the managed header/key
        // contribution; it never supplies a native payload or span bound.
        let mut plan = Self::prepare(
            workspace,
            mechanism.selection(),
            capacity,
            population,
            covered,
            provider_controls,
            complete.then_some(0),
        )?;
        plan.layout.get_mut().expect("unpublished layout").placement =
            mechanism.uniform_budget_placement();
        if let (true, Some(nested), Some((attempts, rows))) = (qualified, nested, population) {
            let managed = Self::qualified_control_bytes(attempts, rows, nested)?;
            let layout = plan.layout.get_mut().expect("unpublished control recipe");
            let payload = crate::working_memory::qualified_storage::array_bytes::<(
                InferenceWorkspaceSpan,
                Option<u64>,
                Option<u64>,
            )>(layout.spans.capacity())?;
            let known = [Some(managed), Some(payload), provider_controls]
                .into_iter()
                .flatten()
                .try_fold(0u64, |a, b| {
                    a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
                })?;
            layout.control_bytes = layout.control_bytes.map(|_| known);
            layout.exact_storage = true;
        }
        Ok(plan)
    }

    /// Retains descriptive carryover populations for the exact consumed plan.
    /// It leaves capacity, covered spans, control completeness and all-generation
    /// retention unchanged. It cannot promote partial liveness into admission.
    pub fn with_carryover_report(
        mut self,
        report: NativeStorageCarryoverReport,
    ) -> Result<Self, WorkingMemoryError> {
        let sampling = usize::try_from(self.plan.geometry().max_output_tokens)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(WorkingMemoryError::Overflow)?;
        if report.equation_records != self.plan.records().len()
            || report.sampling_records != sampling
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.layout
            .get_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .carryover = Some(report);
        Ok(self)
    }
    /// Source-derived diagnostics retained by this actual native plan, if supplied.
    pub fn carryover_report(&self) -> Option<&NativeStorageCarryoverReport> {
        self.layout.carryover.as_ref()
    }

    /// Retain every certified equation generation through the whole program.
    /// One bound is required for each exact original span, independent of its
    /// old peak coverage. The complete sum must fit this plan's native capacity.
    /// Sealing replaces the already composed registered equation contribution;
    /// it does not add a second native allowance or discard disjoint host work.
    pub fn with_retained_equation_generations(
        mut self,
        generations: impl IntoIterator<Item = Option<u64>>,
    ) -> Result<Self, WorkingMemoryError> {
        cold_controls::<(Self, Result<Self, WorkingMemoryError>, u64, bool)>(
            self.plan.metadata_funding().as_ref(),
        )?;
        let mut generations = generations.into_iter();
        let mut sum = 0u64;
        let mut complete = true;
        for record in self.plan.records() {
            let next = generations
                .next()
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if let (Some(next), Some(covered)) = (next, record.new_tensor_allocation_bytes()) {
                if next < covered {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            if let Some(next) = next {
                sum = sum.checked_add(next).ok_or(WorkingMemoryError::Overflow)?;
            } else {
                complete = false;
            }
        }
        if generations.next().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let total = complete.then_some(sum);
        if let (Some(total), Some(capacity)) = (total, self.layout.capacity) {
            if total > capacity {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        let layout = self
            .layout
            .get_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if layout.equation_generations.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        layout.equation_generations = total;
        if total.is_none() {
            layout.control_bytes = None;
        }
        Ok(self)
    }

    fn qualified_control_bytes(
        attempts: usize,
        rows: usize,
        nested_key_bytes: u64,
    ) -> Result<u64, WorkingMemoryError> {
        use crate::working_memory::qualified_storage as storage;
        let ordinary_registry = PreparedNativePublication::<M::Key>::requested_control_bytes(rows)?;
        let exact_registry =
            PreparedNativePublication::<M::Key>::qualified_control_bytes(rows, nested_key_bytes)?;
        let attempts64 = u64::try_from(attempts).map_err(|_| WorkingMemoryError::Overflow)?;
        let without_registry = Self::requested_control_bytes(attempts, rows)?
            .checked_sub(
                ordinary_registry
                    .checked_mul(attempts64)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .ok_or(WorkingMemoryError::Overflow)?;
        let frames = [
            storage::vector_control_bytes::<Option<M::Attachment>>()?,
            storage::vector_control_bytes::<RootPublicationInput>()?,
            storage::vector_control_bytes::<usize>()?,
            storage::vector_control_bytes::<M::Root<'static>>()?,
            storage::vector_control_bytes::<Option<M::Observation<'static>>>()?,
            storage::vector_control_bytes::<WorkingMemoryStorage<M::Key>>()?,
            storage::vector_control_bytes::<bool>()?,
        ]
        .into_iter()
        .try_fold(0u64, |a, b| {
            a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
        })?
        .checked_mul(attempts64)
        .ok_or(WorkingMemoryError::Overflow)?;
        // A zero-attempt bank still returns this fixed nonowning refusal frame;
        // it never constructs a publication, clones custody or observes a root.
        let refusal = (size_of::<Result<OriginalNativePublication<M>, WorkingMemoryError>>()
            as u64)
            .checked_add(size_of::<&mut OriginalNativeStorageBank<M>>() as u64)
            .and_then(|n| n.checked_add(size_of::<&mut WorkingMemoryFundingScope>() as u64))
            .ok_or(WorkingMemoryError::Overflow)?;
        let fixed_headers =
            crate::working_memory::funding::native_partition::qualified_header_bytes(attempts)?
                .checked_add(storage::shared_header_bytes::<NativeStorageLayout>()?)
                // Selection's payload-free Arc is retained by this plan and provider.
                .and_then(|n| {
                    storage::shared_bytes::<()>()
                        .ok()
                        .and_then(|selection| n.checked_add(selection))
                })
                .ok_or(WorkingMemoryError::Overflow)?;
        without_registry
            .checked_add(
                exact_registry
                    .checked_mul(attempts64)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .and_then(|n| n.checked_add(fixed_headers))
            .and_then(|n| n.checked_add(frames))
            .and_then(|n| n.checked_add(refusal))
            .ok_or(WorkingMemoryError::Overflow)
    }

    /// Requested fixed representations and backing layouts, excluding native
    /// producer controls, managed shared headers, dynamic keys and selected span
    /// rows. This requested extent alone does not qualify actual constructors.
    pub fn requested_control_bytes(
        maximum_publications: usize,
        maximum_rows: usize,
    ) -> Result<u64, WorkingMemoryError> {
        let partition =
            crate::working_memory::funding::native_partition::requested_control_bytes()?;
        let registry = PreparedNativePublication::<M::Key>::requested_control_bytes(maximum_rows)?;
        let each = (size_of::<NativeStorageRegistration<M::Key>>() as u64)
            .checked_add(size_of::<M::Key>() as u64)
            .and_then(|n| n.checked_add(size_of::<Option<M::Attachment>>() as u64))
            .and_then(|n| n.checked_add(size_of::<M::Observation<'static>>() as u64))
            .and_then(|n| n.checked_add(size_of::<Option<M::Observation<'static>>>() as u64))
            .and_then(|n| n.checked_add(size_of::<NativeStorageObservation<M::Key>>() as u64))
            .and_then(|n| n.checked_add(size_of::<RootPublicationInput>() as u64))
            .and_then(|n| n.checked_add(size_of::<usize>() as u64))
            .and_then(|n| n.checked_add(size_of::<M::Root<'static>>() as u64))
            .and_then(|n| n.checked_add(size_of::<WorkingMemoryStorage<M::Key>>() as u64))
            .and_then(|n| n.checked_add(size_of::<bool>() as u64))
            .ok_or(WorkingMemoryError::Overflow)?;
        let per = each
            .checked_mul(u64::try_from(maximum_rows).map_err(|_| WorkingMemoryError::Overflow)?)
            .and_then(|n| n.checked_add(registry))
            .and_then(|n| n.checked_add(size_of::<NativeStorageError<M::Error>>() as u64))
            .and_then(|n| {
                n.checked_add(size_of::<
                    Option<crate::working_memory::gguf_source::SourceInventoryOrigin>,
                >() as u64)
            })
            .and_then(|n| {
                n.checked_add(size_of::<
                    Result<
                        Option<crate::working_memory::gguf_source::SourceInventoryOrigin>,
                        WorkingMemoryError,
                    >,
                >() as u64)
            })
            .and_then(|n| {
                n.checked_add(
                    size_of::<Option<&eredu_checkpoint::store::SourceStorageIdentity>>() as u64,
                )
            })
            .and_then(|n| n.checked_add(size_of::<OriginalNativePublication<M>>() as u64))
            .and_then(|n| {
                n.checked_add(size_of::<(
                    Option<usize>,
                    usize,
                    usize,
                    std::slice::Iter<'static, RootPublicationInput>,
                    &M::Observation<'static>,
                    &'static str,
                    WorkingMemoryError,
                )>() as u64)
            })
            .and_then(|n| n.checked_add(size_of::<Vec<M::Root<'static>>>() as u64))
            .and_then(|n| n.checked_add(size_of::<Vec<Option<M::Observation<'static>>>>() as u64))
            // Borrow/clone attachment and consuming source-vector transfer
            // happen after publication, while the prepared registry stays live.
            .and_then(|n| n.checked_add(3 * size_of::<Vec<WorkingMemoryStorage<M::Key>>>() as u64))
            .and_then(|n| {
                n.checked_add(3 * size_of::<Option<Vec<WorkingMemoryStorage<M::Key>>>>() as u64)
            })
            .and_then(|n| {
                n.checked_add(2 * size_of::<Option<WorkingMemoryStorage<M::Key>>>() as u64)
            })
            .and_then(|n| n.checked_add(size_of::<Option<&WorkingMemoryStorage<M::Key>>>() as u64))
            .and_then(|n| {
                n.checked_add(size_of::<std::iter::Enumerate<std::slice::Iter<'_, usize>>>() as u64)
            })
            .ok_or(WorkingMemoryError::Overflow)?;
        per.checked_mul(
            u64::try_from(maximum_publications).map_err(|_| WorkingMemoryError::Overflow)?,
        )
        .and_then(|n| n.checked_add(size_of::<OriginalNativeStorageBank<M>>() as u64))
        .and_then(|n| n.checked_add(size_of::<Self>() as u64))
        .and_then(|n| n.checked_add(size_of::<NativeStorageLayout>() as u64))
        .and_then(|n| n.checked_add(partition))
        .and_then(|n| n.checked_add(size_of::<NativeStorageError<M::Error>>() as u64))
        // The scalar-provider path separately supplies managed shared headers,
        // exposed spare capacity, nested keys and provider overlap. The qualified
        // path below computes its own concrete finite header/key contribution.
        .ok_or(WorkingMemoryError::Overflow)
    }

    /// Complete separate control contribution, or unknown. The payload cap is
    /// already part of the selected workspace and must not be added again to Q.
    pub fn control_bytes(&self) -> Option<u64> {
        self.layout.control_bytes
    }
}

impl NativeStorageLayout {
    pub(in crate::working_memory) fn domain_remainder(
        &self,
        workspace: &InferenceSpanWorkspace,
        index: usize,
        domain: eredu_core::MemoryDomainId,
        native_held: Option<u64>,
        native_registered: u64,
    ) -> Result<u64, WorkingMemoryError> {
        let record = workspace
            .plan()
            .records()
            .get(index)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let (expected, full_diagnostic, native) = self
            .spans
            .get(index)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if expected != record.span() || *full_diagnostic != workspace.span_bytes(index) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let full = workspace
            .span_domain_bytes(index, domain)?
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let native = native.ok_or(WorkingMemoryError::UnknownBound)?;
        let capacity = self.capacity.ok_or(WorkingMemoryError::UnknownBound)?;
        let placement = self
            .placement
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let covered = if placement.domains().contains(&domain) {
            native
        } else {
            0
        };
        let expected_hold = if placement.domains().contains(&domain) {
            capacity
        } else {
            0
        };
        // Publication converts part of the live partition into retained rows.
        // Its issuing producer still owns the full immutable selected allowance;
        // a retired or reduced partition cannot reopen a new equation span.
        let live_partition = native_held
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .checked_add(native_registered)
            .ok_or(WorkingMemoryError::Overflow)?;
        if live_partition != expected_hold || native > capacity {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        full.checked_sub(covered)
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
}

impl PreparedTextControlWorkspace {
    /// Install exact selected native facts before sealing, once. The physical
    /// cap stays in the original workspace; only the separate controls join Q.
    pub fn with_native_storage<M: OriginalNativeStorageMechanism>(
        mut self,
        plan: PreparedNativeStoragePlan<M>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.native_storage.is_some() || !self.plan.same_plan(&plan.plan) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let known = [self.binding.facts.work, plan.layout.control_bytes]
            .into_iter()
            .flatten()
            .try_fold(0u64, |a, b| {
                a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
            })?;
        self.binding.facts.work = self
            .binding
            .facts
            .work
            .zip(plan.layout.control_bytes)
            .map(|_| known);
        self.binding.native_storage = Some(plan.layout);
        Ok(self)
    }
}

impl OwnedTextSpanWorkspace {
    /// Consume the exact accepted native selection once, before construction.
    /// No scalar capacity is accepted here. Rejection, including wrong selection
    /// or health, cannot reconstruct the bank from a cloned diagnostic or guard.
    pub fn take_native_storage_bank<M: OriginalNativeStorageMechanism>(
        &mut self,
        run: &WorkingMemoryFundingRun,
        selection: &NativeStorageSelection,
    ) -> Result<Option<OriginalNativeStorageBank<M>>, WorkingMemoryError> {
        if self.native_storage_taken {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        let Some(layout) = self
            .workspace()
            .control_binding()
            .and_then(|b| b.native_storage.as_ref())
            .cloned()
        else {
            return Ok(None);
        };
        self.native_storage_taken = true;
        if layout.mechanism != TypeId::of::<M>() || !layout.selection.same_selection(selection) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        layout
            .control_bytes
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let capacity = layout.capacity.ok_or(WorkingMemoryError::UnknownBound)?;
        let receipt = crate::working_memory::funding::native_partition::ReservedNativePartition::from_selected_span(run, self, capacity, layout.placement.clone().ok_or(WorkingMemoryError::UnknownBound)?)?;
        let partition = run.take_native_partition(receipt)?;
        Ok(Some(OriginalNativeStorageBank {
            budget: None,
            mechanism: None,
            attempts: layout.population.ok_or(WorkingMemoryError::UnknownBound)?.0,
            layout,
            installation_started: false,
            reservation: self.reservation().clone(),
            controls: self.controls.clone(),
            partition,
        }))
    }
}
