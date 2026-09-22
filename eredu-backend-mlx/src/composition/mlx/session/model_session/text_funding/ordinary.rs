//! Finite ordinary collectors and their existing generic publication worker.
use super::*;
use eredu_runtime::working_memory::{MemoryLedger, StorageMetadataFunding};
mod native;

#[derive(Clone, Debug)]
pub(in crate::composition::mlx::session::model_session) struct Plan {
    rows: usize,
    host_owner_bytes: usize,
    additional_bytes: u64,
    quote_components: std::cell::Cell<crate::backend::error::WorkspaceQuoteComponents>,
    native: Option<std::rc::Rc<native::Native>>,
    checkpoint:
        Option<crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointWork>,
    // The shared plan allocation retires before its cold metadata owner.
    native_custody: Option<eredu_nn::workspace::HostMetadataFunding>,
}
impl Plan {
    pub(in crate::composition::mlx::session::model_session) fn inspect(
        session: &MlxModelSession,
        equations: &eredu_runtime::working_memory::InferenceWorkspaceReport,
        sampling: &eredu_runtime::working_memory::SamplingWorkspaceReport,
        predictions: u64,
    ) -> Result<Self, Error> {
        let pool = &session.payload.memory_ledger;
        let mut nonstate = RetainedStorage::generic_census(pool);
        let mut decoder = RetainedStorage::generic_census(pool);
        session
            .payload
            .collect_retained_idle_storage(&mut nonstate, &mut decoder)?;
        let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
        let opening = nonstate
            .generic_publication_rows(pool)?
            .checked_add(decoder.generic_publication_rows(pool)?)
            .ok_or_else(overflow)?;
        let roots = equations
            .maximum_closing_storage_allocations()
            .checked_add(
                sampling
                    .maximum_closing_storage_allocations
                    .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?,
            )
            .ok_or_else(overflow)?;
        Self::for_closing_population(
            opening,
            roots,
            predictions.checked_add(2).ok_or_else(overflow)?,
        )
    }

    pub(in crate::composition::mlx::session::model_session) fn for_closing_population(
        opening_rows: usize,
        closing_allocations: usize,
        work_population: u64,
    ) -> Result<Self, Error> {
        let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
        // A physical root publishes its payload and optional host-control row.
        // Prompt construction adds one token root and one shared input identity.
        let rows = closing_allocations
            .checked_add(1)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(opening_rows))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(overflow)?;
        Self::for_rows(rows, work_population)
    }

    /// The enclosing source supplies its actual simultaneous publication rows
    /// and Work population. This is the same constructor used by text planning.
    pub(super) fn for_rows(rows: usize, work_population: u64) -> Result<Self, Error> {
        let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
        let work = usize::try_from(work_control_bytes()?).map_err(|_| overflow())?;
        let collectors =
            usize::try_from(collectors::work_control_bytes(rows).ok_or_else(overflow)?)
                .map_err(|_| overflow())?;
        let observer = usize::try_from(
            crate::backend::managed_memory::scoped_observer_bytes()
                .map_err(Error::PrefillControl)?,
        )
        .map_err(|_| overflow())?;
        let execution = crate::backend::nn::shared::OrdinaryExecutionRegistration::control_bytes()
            .and_then(|bytes| bytes.checked_add(ordinary_paged_install_frames()))
            .ok_or_else(overflow)?;
        let host_owner_bytes = work
            .checked_add(collectors)
            .and_then(|n| n.checked_add(observer))
            .and_then(|n| n.checked_add(execution))
            .ok_or_else(overflow)?;
        let host = StorageMetadataFunding::host_owner_bytes(host_owner_bytes)
            .map_err(Error::WorkspacePlanning)?;
        let publication =
            crate::backend::runtime::residency::storage::generic_storage_publication_layout(rows)
                .map_err(Error::PrefillControl)?
                .requested_bytes();
        let constructor =
            MemoryLedger::storage_metadata_control_bytes().map_err(Error::PrefillControl)?;
        // Model work publishes nonstate and decoder separately. Both may remain
        // live through escaped aliases; prompt/sampler preparation need at most
        // the same two destinations. Existing text controls already price work.
        let per_work = u64::try_from(host.checked_sub(work).ok_or_else(overflow)?)
            .map_err(|_| overflow())?
            .checked_add(
                WorkingMemoryFundingScope::allocation_funding_control_bytes()
                    .map_err(Error::PrefillControl)?,
            )
            .and_then(|n| n.checked_add(constructor))
            .and_then(|n| {
                publication
                    .checked_add(constructor)
                    .and_then(|p| p.checked_mul(2))
                    .and_then(|p| n.checked_add(p))
            })
            .ok_or_else(overflow)?;
        let additional_bytes = work_population.checked_mul(per_work).ok_or_else(overflow)?;
        Ok(Self {
            rows,
            host_owner_bytes,
            additional_bytes,
            quote_components: std::cell::Cell::new(Default::default()),
            native: None,
            checkpoint: None,
            native_custody: None,
        })
    }
    pub(in crate::composition::mlx::session::model_session) fn quote_components(
        &self,
    ) -> crate::backend::error::WorkspaceQuoteComponents {
        self.quote_components.get()
    }
    pub(in crate::composition::mlx::session::model_session) fn record_quote_components(
        &self,
        mut value: crate::backend::error::WorkspaceQuoteComponents,
    ) {
        if let Some(native) = &self.native {
            let actual = native.diagnostics;
            value.ordinary_native_observed_host = actual.ordinary_native_observed_host;
            value.ordinary_native_control_allocations = actual.ordinary_native_control_allocations;
            value.ordinary_caller_observed_host = actual.ordinary_caller_observed_host;
            value.ordinary_caller_control_allocations = actual.ordinary_caller_control_allocations;
            value.ordinary_ledger_controls = actual.ordinary_ledger_controls;
            value.ordinary_native_host_requirement = actual.ordinary_native_host_requirement;
            value.kernel_attempts = actual.kernel_attempts;
            value.equation_rows = actual.equation_rows;
            value.mixed_rows = actual.mixed_rows;
            value.gpu_entries = actual.gpu_entries;
            value.cpu_entries = actual.cpu_entries;
            value.nested_frontiers = actual.nested_frontiers;
            value.sampling_gpu_entries = actual.sampling_gpu_entries;
            value.sampling_cpu_entries = actual.sampling_cpu_entries;
            value.sampling_nested_frontiers = actual.sampling_nested_frontiers;
        }
        value.ordinary_publication_controls = Some(self.additional_bytes);
        self.quote_components.set(value);
    }
    pub(in crate::composition::mlx::session::model_session) fn additional_bytes(&self) -> u64 {
        self.additional_bytes
    }

    pub(in crate::composition::mlx::session::model_session) fn with_native(
        mut self,
        recipe: crate::backend::nn::workspace::ResidentNativeRecipe,
        context: &eredu_nn::workspace::WorkspaceContext,
        ledger: &MemoryLedger,
        session: &MlxModelSession,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        paged_source: Option<&crate::backend::nn::workspace::ProjectedPagedSources>,
    ) -> Result<Self, Error> {
        if self.native.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let custody = context
            .metadata_funding()
            .ok_or_else(|| Error::text_admission(WorkingMemoryError::UnknownBound))?;
        let native =
            native::Native::prepare(recipe, context, ledger, session, source, paged_source)?;
        self.additional_bytes = self
            .additional_bytes
            .checked_add(native.wrapper_metadata())
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        self.native = Some(native);
        self.native_custody = Some(custody);
        Ok(self)
    }

    pub(in crate::composition::mlx::session::model_session) fn native_requirements(
        &self,
    ) -> Option<&eredu_core::DomainMemoryRequirements> {
        self.native.as_ref().map(|native| native.requirements())
    }

    pub(in crate::composition::mlx::session::model_session) fn for_step(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        prefill: bool,
    ) -> Result<Self, Error> {
        let mut plan = self.clone();
        if let Some(native) = &self.native {
            plan.checkpoint = native.checkpoint_work(step, prefill)?;
            let bytes = usize::try_from(native.step_metadata(step, prefill)?)
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
            plan.host_owner_bytes = plan
                .host_owner_bytes
                .checked_add(bytes)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        }
        Ok(plan)
    }

    pub(in crate::composition::mlx::session::model_session) fn for_preparation(
        &self,
    ) -> Result<Self, Error> {
        let mut plan = self.clone();
        if let Some(native) = &self.native {
            let bytes = usize::try_from(native.preparation_metadata()?)
                .map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?;
            plan.host_owner_bytes = plan
                .host_owner_bytes
                .checked_add(bytes)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        }
        Ok(plan)
    }

    pub(in crate::composition::mlx::session::model_session) fn paged_work(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        prefill: bool,
    ) -> Result<Option<crate::backend::nn::workspace::OrdinaryPagedWork>, Error> {
        self.native
            .as_ref()
            .map(|native| native.paged_work(step, prefill))
            .transpose()
            .map(Option::flatten)
    }

    pub(in crate::composition::mlx::session::model_session) fn indexed_program(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        prefill: bool,
    ) -> Result<Option<std::rc::Rc<dyn crate::backend::runtime::residency::parameter_bank::OrdinaryIndexedRequestProgram>>, Error>{
        self.native
            .as_ref()
            .map(|native| native.indexed_program(step, prefill))
            .transpose()
            .map(Option::flatten)
    }

    #[cfg(test)]
    pub(super) fn fixture(rows: usize) -> (Self, u64) {
        let plan = Self::for_rows(rows, 1).unwrap();
        let complete = plan
            .additional_bytes
            .checked_add(work_control_bytes().unwrap())
            .unwrap();
        (plan, complete)
    }
}

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session) fn new_ordinary(
        scope: WorkingMemoryFundingScope,
        plan: Plan,
    ) -> Result<FundedWorkOwner, Error> {
        Self::new_with_publication(scope, None, None, None, None, None, None, Some(plan))
    }
}

pub(super) fn prepare(
    scope: &mut WorkingMemoryFundingScope,
    plan: Plan,
) -> Result<
    (
        usize,
        eredu_core::HostPreparationAuthority,
        safemlx::ScopedPhysicalBackingObserver,
        crate::backend::nn::shared::OrdinaryExecutionRegistration,
    ),
    Error,
> {
    let metadata = scope
        .prepare_storage_metadata()
        .map_err(Error::WorkspacePlanning)?;
    let host = metadata
        .prepare_host_owner(plan.host_owner_bytes)
        .map_err(Error::WorkspacePlanning)?;
    let observer =
        crate::backend::managed_memory::prepare_fresh_scoped_observer(scope, host.clone())?;
    let execution = crate::backend::nn::shared::OrdinaryExecutionRegistration::new(
        crate::backend::nn::shared::OrdinaryExecutionOwner::new(host.clone(), observer.clone())
            .with_checkpoint(plan.checkpoint, metadata.funding().clone()),
    )?;
    Ok((plan.rows, host, observer, execution))
}

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session) fn configure_ordinary_scope(
        &self,
        scope: &mut safemlx::SubmissionScope,
    ) -> Result<(), safemlx::error::Exception> {
        if let Some(observer) = self.ordinary_observer.borrow().as_ref() {
            scope.bind_physical_observer(observer)?;
        }
        Ok(())
    }
}
