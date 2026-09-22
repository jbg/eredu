//! Accepted role installation shared by text-independent execution paths.
use super::*;
use crate::backend::nn::workspace::SpeculativeNumericalRecipe;
use crate::backend::submission_recovery::native_role::realtime::RealtimeOperationClaim;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, MemoryLedger, OriginalHostSourceBank,
    OriginalOperationMetadataCustody, OriginalRealtimeBudgetCustody,
};

impl<U: 'static> OriginalOperationPlan<'_, U> {
    /// The caller has already consumed its unique neutral bank claim. This
    /// worker only installs actual accepted source/storage in the existing slot.
    pub(super) fn prepare_accepted_bank(
        self,
        observer: safemlx::OriginalScopeObserver,
        custody: OriginalOperationMetadataCustody,
        request: OperationRequest,
        controls: OperationControls,
        disk_source: Option<storage::ForegroundDiskSourceTicket>,
        materialized: Option<materialization::MaterializationOwner>,
    ) -> Result<Box<dyn ErasedOwner>, Error> {
        let source = self.retained_source().ok_or_else(unknown)?;
        let retained_sources = Some(source.retain_ready_host_sources(custody.clone())?);
        let PreparedContainers {
            requests,
            mut scopes,
            pending,
        } = self.prepare_containers(&custody)?;
        let materialized_recipe =
            materialized.map(|owner| materialization::MaterializationSource::Role {
                owner,
                observer: observer.clone(),
            });
        scopes.push(observer);
        let population = self.residency.ok_or_else(unknown)?;
        let mut prepared = PreparedStorage::new_with_custody(
            self.units,
            population,
            self.neural,
            None,
            self.window_sources.as_ref().ok_or_else(unknown)?.as_slice(),
            &self.manager,
            self.sources.as_slice(),
            &custody,
            None,
            self.foreground_disk_plan(),
            None,
            None,
            disk_source,
        )?;
        let registry = Rc::new(Registry {
            materialized_recipe: RefCell::new(materialized_recipe),
            request,
            scopes: RefCell::new(scopes),
            scope_limit: 1,
            registered_scopes: Cell::new(1),
            active: Cell::new(true),
            // These banks are constructed inside one already accepted
            // speculative/realtime invocation and retire with that owner.
            entered: Cell::new(true),
            retirement_failure: Cell::new(None),
        });
        let background = prepared.background.take();
        let bank = Rc::new(Bank {
            replacements: self
                .retained_source()
                .map(|source| source.replacement_values().clone())
                .unwrap_or_default(),
            background: RefCell::new(background),
            prepared: RefCell::new(prepared),
            pending: RefCell::new(pending),
            pending_limit: self.pending,
            binding_row_limit: population.unprepared.transfer.binding_rows,
            requests,
            registry,
            _retained_sources: retained_sources,
            selected_stream: self.selected_stream.ok_or_else(identity)?,
            identity: self.identity,
        });
        let replaced = self.slot.replace(Some(OriginalOperationProjection {
            value: Rc::downgrade(&bank),
            identity: Arc::clone(&bank.identity),
            controls,
        }));
        drop(replaced);
        Ok(Box::new(OwnedBank {
            bank,
            slot: self.slot,
            registration: None,
        }))
    }
}

/// One owned selected policy source. All request members are descriptive until
/// the frame's private one-use RealtimeOperationClaim supplies its native scope.
pub(crate) struct RealtimeLayerwisePlan<U: 'static> {
    plan: OriginalOperationPlan<'static, U>,
}
pub(crate) struct QualifiedRealtimeLayerwisePlan<U: 'static> {
    plan: OriginalOperationPlan<'static, U>,
    recipe: SpeculativeNumericalRecipe,
}
impl<U: 'static, P> MlxLayerwisePolicy<U, P> {
    pub(crate) fn realtime_plan(
        &self,
        source: super::super::host_workspace::LayerwiseWorkspace,
        stream: &Stream,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
    ) -> Result<RealtimeLayerwisePlan<U>, Error> {
        let funding = context.metadata_funding().ok_or_else(unknown)?;
        context
            .charge_metadata(
                rc_layout::<super::super::host_workspace::LayerwiseWorkspace>()
                    .and_then(|n| n.checked_add(size_of::<RealtimeLayerwisePlan<U>>()))
                    .and_then(|n| {
                        n.checked_add(size_of::<Result<RealtimeLayerwisePlan<U>, Error>>())
                    })
                    .and_then(|n| {
                        n.checked_add(safemlx::StreamCopyPlan::<()>::capture_control_bytes().ok()?)
                    })
                    .ok_or_else(overflow)?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let windows = self.operation_source.as_ref().map_err(|_| unknown())?;
        source.validate_original_policy(
            &self.residency,
            &self.workspace_identity,
            self.unit_ids.as_slice(),
        )?;
        if !source.has_original_ready_host_source(&self.residency, &windows.window_owner())
            && !source.has_original_foreground_source(&self.residency, &windows.window_owner())
        {
            return Err(unknown());
        }
        // Bounded policy finish submits the final output after the shared
        // graph boundaries, exactly as ordinary bounded text execution does.
        let neural = NeuralPopulation::single_forward(&self.layout, true)?;
        let residency = ResidencyPopulation::from_window_forwards(
            windows.controller_units,
            windows.windows(),
            1,
        )?;
        let mut plan = self.operation_plan_from_population(
            None,
            OperationCounts {
                units: self.layout.len(),
                scopes: 1,
            },
            neural,
            Some(residency),
            None,
        )?;
        plan.owned_sources = Some(Rc::new(source));
        plan = plan
            .with_preparation_funding(Some(&funding))
            .with_selected_stream(stream)?
            .with_foreground_disk_reads(pool)?;
        Ok(RealtimeLayerwisePlan { plan })
    }
}
impl<U: 'static> RealtimeLayerwisePlan<U> {
    pub(crate) fn qualify(
        self,
        recipe: SpeculativeNumericalRecipe,
        actual: &super::super::host_workspace::LayerwiseWorkspace,
        context: &WorkspaceContext,
    ) -> Result<QualifiedRealtimeLayerwisePlan<U>, Error> {
        let mut plan = self.plan;
        actual.validate_original_policy(&plan.manager, &plan.identity, plan.sources.as_slice())?;
        let windows = plan.window_sources.as_ref().ok_or_else(unknown)?;
        if !actual.has_original_ready_host_source(&plan.manager, windows)
            && !(plan.foreground_disk_plan().is_some()
                && actual.has_original_foreground_source(&plan.manager, windows))
        {
            return Err(identity());
        }
        if let Some(disk) = plan
            .foreground_disk
            .as_mut()
            .filter(|disk| disk.background().is_some())
        {
            let select = |ordinals: Option<&[usize]>| {
                disk.select_background_execution_ordinals(ordinals.ok_or_else(identity)?)
            };
            context
                .charge_metadata(std::mem::size_of_val(&select))
                .map_err(|cause| Error::Neural(cause.into()))?;
            actual.with_execution_ordinals(select)?;
        }
        let mut recipe = recipe
            .with_realtime_boundaries(
                plan.neural.submissions,
                plan.neural.shape.consumers(),
                context,
            )
            .and_then(|recipe| recipe.with_layerwise_source(actual, windows.as_slice(), context))
            .map_err(Error::Neural)?;
        if let Some(disk) = plan.foreground_disk_plan() {
            recipe = recipe
                .with_materialized_storage(
                    disk.population().ok_or_else(unknown)?.materialized,
                    context,
                )
                .map_err(Error::Neural)?;
        }
        let fit = NeuralProducerFit::pending(plan.neural).ok_or_else(unknown)?;
        plan.neural_fit = NeuralProducerFit::Recipe(fit.requirement());
        plan.control_bytes().ok_or_else(unknown)?;
        Ok(QualifiedRealtimeLayerwisePlan { plan, recipe })
    }
}
impl<U: 'static> QualifiedRealtimeLayerwisePlan<U> {
    pub(crate) fn recipe(&self) -> &SpeculativeNumericalRecipe {
        &self.recipe
    }
    pub(crate) fn source_facts(&self) -> Option<HostSourceConstructionFacts> {
        self.plan.source_construction_facts()
    }
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        let dynamic = match self.plan.foreground_disk_plan() {
            Some(disk) => u64::try_from(disk.population()?.materialized.controls).ok()?,
            None => 0,
        };
        self.prepared_control_bytes()?.checked_add(dynamic)
    }
    fn prepared_control_bytes(&self) -> Option<u64> {
        let frames = [
            size_of::<Self>(),
            size_of::<RealtimeNeuralOwner>(),
            size_of::<RealtimeOperationClaim<'_>>(),
            size_of::<materialization::MaterializationSource>(),
            size_of::<materialization::MaterializationOwner>(),
            size_of::<Option<materialization::MaterializationOwner>>(),
            size_of::<crate::backend::runtime::residency::manager::OriginalMaterializedLoan<'_>>(),
            size_of::<(
                &crate::backend::submission_recovery::native_role::NativeRoleContext<'_>,
                &eredu_nn::workspace::HostMetadataFunding,
            )>(),
            size_of::<OriginalRealtimeBudgetCustody>(),
            size_of::<OriginalOperationMetadataCustody>(),
            size_of::<Option<OriginalHostSourceBank>>(),
            size_of::<Option<storage::ForegroundDiskSourceTicket>>(),
            size_of::<Result<RealtimeNeuralOwner, Error>>(),
            size_of::<(
                safemlx::OriginalScopeObserver,
                OriginalRealtimeBudgetCustody,
            )>(),
            size_of::<
                Result<
                    (
                        safemlx::OriginalScopeObserver,
                        OriginalRealtimeBudgetCustody,
                    ),
                    Error,
                >,
            >(),
        ];
        let extra = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)?
            .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                neural::NeuralBoundaryFailure<OriginalRealtimeBudgetCustody>,
            >()?)?;
        self.plan
            .control_bytes()?
            .checked_add(u64::try_from(extra).ok()?)
    }
    pub(crate) fn prepare(
        self,
        claim: RealtimeOperationClaim<'_>,
        source: Option<OriginalHostSourceBank>,
    ) -> Result<RealtimeNeuralOwner, Error> {
        let (native, funding) = claim.materialization_source();
        let materialized = materialization::MaterializationOwner::from_role(native, funding);
        let (observer, custody) = claim.prepare(
            usize::try_from(self.prepared_control_bytes().ok_or_else(unknown)?)
                .map_err(|_| overflow())?,
        )?;
        let result = (|| {
            let previous = self.plan.slot.take();
            let busy = previous
                .as_ref()
                .is_some_and(|view| view.value.strong_count() != 0);
            self.plan.slot.set(previous);
            if busy {
                return Err(Error::PrefillScopeReentrant);
            }
            let source = match (self.plan.foreground_disk_plan(), source) {
                (Some(plan), Some(bank))
                    if bank.matches_facts(plan.source_facts().ok_or_else(unknown)?) =>
                {
                    Some(storage::ForegroundDiskSourceTicket::Source {
                        bank,
                        custody: custody.clone().into(),
                    })
                }
                (None, None) => None,
                _ => return Err(identity()),
            };
            let value = self.plan.prepare_accepted_bank(
                observer,
                custody.clone().into(),
                OperationRequest::Realtime(custody.clone()),
                OperationControls::Realtime(custody.clone()),
                source,
                Some(materialized),
            )?;
            Ok(RealtimeNeuralOwner::from_bounded(value, custody.clone()))
        })();
        result.map_err(|cause| neural::boundary_error(cause, &custody))
    }
}
