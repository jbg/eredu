//! Ordinary native producers retained from the same funded equation traversal.
use super::*;
use crate::backend::{
    nn::workspace::{
        OrdinaryIndexedPrograms, OrdinaryPagedProgram, OrdinaryPagedWork, ProjectedPagedSources,
        ResidentNativeRecipe,
    },
    runtime::residency::parameter_bank::OrdinaryIndexedRequestProgram,
};
use eredu_core::DomainMemoryRequirements;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{InferenceTextStep, WorkspaceReportMetadata};
use std::rc::Rc;

#[derive(Debug)]
pub(super) struct Native {
    recipe: ResidentNativeRecipe,
    indexed: OrdinaryIndexedPrograms,
    paged: Option<OrdinaryPagedProgram>,
    checkpoint: Option<
        crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram,
    >,
    requirements: DomainMemoryRequirements,
    wrapper_metadata: u64,
    pub(super) diagnostics: crate::backend::error::WorkspaceQuoteComponents,
}
#[track_caller]
fn unknown() -> Error {
    Error::text_admission(WorkingMemoryError::UnknownBound)
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
impl Native {
    pub(super) fn prepare(
        mut recipe: ResidentNativeRecipe,
        context: &WorkspaceContext,
        ledger: &MemoryLedger,
        session: &MlxModelSession,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        paged_source: Option<&ProjectedPagedSources>,
    ) -> Result<Rc<Self>, Error> {
        context
            .charge_metadata(size_of::<(
                Self,
                ResidentNativeRecipe,
                Option<&ProjectedPagedSources>,
                Option<OrdinaryPagedProgram>,
                &WorkspaceContext,
                &MemoryLedger,
                &MlxModelSession,
                Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
                Result<Option<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>, Error>,
                Option<eredu_nn::workspace::HostMetadataFunding>,
                (&MemoryLedger, &mut ResidentNativeRecipe, Option<&eredu_nn::workspace::HostMetadataFunding>),
                Option<DomainMemoryRequirements>,
                [crate::backend::error::WorkspaceQuoteComponents; 2],
                crate::backend::nn::workspace::OrdinaryNativeControls,
                u64,
                Result<Rc<Self>, Error>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        if let Some(cause) = recipe.take_ordinary_call_failure() {
            return Err(Error::Neural(context.metadata_source(cause)));
        }
        if recipe.ordinary_requires_paged_source() && paged_source.is_none() {
            return Err(unknown());
        }
        // The selected policy supplies its real submit/consumer frontiers for
        // ordinary storage as well as native arenas. Bind those descriptive
        // facts before the recipe moves into this publication owner.
        session
            .payload
            .model
            .erased()
            .bind_layerwise_neural_recipe(ledger, &mut recipe, context.metadata_funding().as_ref())
            .map_err(|cause| cause.at_text_admission())?;
        recipe
            .bind_ordinary_model_completion_calls(
                crate::composition::mlx::replicated_text::ordinary_model_completion_call_controls,
            )
            .map_err(|cause| cause.at_text_admission())?;
        let materialization = if let Some(source) = source {
            let native = session
                .payload
                .model
                .erased()
                .native_storage_mechanism()?
                .ok_or_else(|| unknown())?;
            let runtime = native
                .prepared_input_runtime()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let allocation = session
                .payload
                .model
                .resident_workspace_mechanisms()
                .ok_or_else(|| unknown())?
                .ordinary_storage()
                .allocation();
            recipe.bind_ordinary_materialization(source, runtime, allocation, ledger, context)?
        } else {
            None
        };
        let checkpoint = Some(
            session
                .payload
                .model
                .erased()
                .ordinary_checkpoint_program(recipe.plan(), context)?
                .ok_or_else(unknown)?,
        );
        let paged = paged_source
            .map(|source| source.prepare_ordinary(recipe.plan(), ledger, context))
            .transpose()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let mut calls = recipe.ordinary_call_controls().ok_or_else(|| unknown())?;
        if let Some(paged) = &paged {
            calls = calls
                .append(paged.caller_controls().ok_or_else(unknown)?)
                .ok_or_else(overflow)?;
        }
        let native_observed = recipe
            .ordinary_native_controls(context)
            .map_err(Error::Neural)?
            .ok_or_else(|| unknown())?;
        let observed = native_observed
            .append(calls.observed)
            .ok_or_else(overflow)?;
        let indexed = recipe
            .ordinary_indexed_programs(context)
            .map_err(Error::Neural)?;
        let mut host = observed
            .host_allowance_with_ledger_metadata()
            .map_err(Error::PrefillControl)?;
        let ledger_controls = host
            .checked_sub(observed.observed_host_bytes)
            .ok_or_else(overflow)?;
        for program in indexed.programs() {
            host = host
                .checked_add(
                    super::super::ordinary_indexed_control_bytes(program).ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
        }
        let metadata = WorkspaceReportMetadata::new(context);
        let mut requirements = metadata
            .placed_requirements(ledger.topology(), host, ledger.host_placement())
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        if let Some(materialization) = materialization {
            requirements = metadata
                .combine_domain_requirements(&requirements, &materialization, true)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        }
        if let Some(transfers) = paged
            .as_ref()
            .and_then(|program| program.transfer_requirements())
        {
            requirements = metadata
                .combine_domain_requirements(&requirements, transfers, true)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        }
        // Each transfer source retains its actual domain attribution. Existing
        // indexed source storage is pinned by its provider, never reserved here.
        for program in recipe.ordinary_transfer_programs() {
            let transfers = program.transfer_requirements().ok_or_else(|| unknown())?;
            requirements = metadata
                .combine_domain_requirements(&requirements, transfers, true)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        }
        let mut diagnostics = crate::backend::error::WorkspaceQuoteComponents::default();
        diagnostics.ordinary_native_observed_host = Some(native_observed.observed_host_bytes);
        diagnostics.ordinary_native_control_allocations = Some(native_observed.control_allocations);
        diagnostics.ordinary_caller_observed_host = Some(calls.observed.observed_host_bytes);
        diagnostics.ordinary_caller_control_allocations = Some(calls.observed.control_allocations);
        diagnostics.ordinary_ledger_controls = Some(ledger_controls);
        diagnostics.ordinary_native_host_requirement = Some(
            requirements
                .get(ledger.topology().host_domain())
                .and_then(|charge| charge.total())
                .map_err(|cause| Error::PrefillControl(cause.into()))?,
        );
        // Record source populations through the existing funded diagnostic
        // producer before the recipe becomes immutable in this ordinary owner.
        recipe.record_quote_components(diagnostics)?;
        let diagnostics = recipe.quote_components;
        context
            .metadata_rc(Self {
                recipe,
                diagnostics,
                indexed,
                paged,
                wrapper_metadata: calls
                    .metadata_bytes
                    .checked_add(
                        checkpoint
                            .as_ref()
                            .map(|p| p.metadata_bytes().ok_or_else(overflow))
                            .transpose()?
                            .unwrap_or(0),
                    )
                    .ok_or_else(overflow)?,
                checkpoint,
                requirements,
            })
            .map_err(|cause| Error::Neural(cause.into()))
    }

    pub(super) fn requirements(&self) -> &DomainMemoryRequirements {
        &self.requirements
    }
    pub(super) fn wrapper_metadata(&self) -> u64 {
        self.wrapper_metadata
    }
    pub(super) fn step_metadata(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<u64, Error> {
        // This validates the real step coordinates even without indexed work.
        self.indexed.for_step(step, prefill)?;
        let bytes = self
            .recipe
            .ordinary_step_call_metadata(step, prefill)
            .ok_or_else(|| unknown())?;
        let paged = self
            .paged
            .as_ref()
            .map(|program| program.step_call_controls(step, prefill))
            .transpose()?
            .map_or(0, |controls| controls.metadata_bytes);
        bytes.checked_add(paged).ok_or_else(overflow)
    }
    pub(super) fn preparation_metadata(&self) -> Result<u64, Error> {
        self.recipe
            .ordinary_preparation_call_metadata()
            .ok_or_else(|| unknown())
    }
    pub(super) fn checkpoint_work(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<
        Option<crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointWork>,
        Error,
    > {
        self.checkpoint
            .as_ref()
            .map(|p| p.for_step(step, prefill))
            .transpose()
    }
    pub(super) fn paged_work(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<Option<OrdinaryPagedWork>, Error> {
        self.paged
            .as_ref()
            .map(|program| program.for_step(step, prefill))
            .transpose()
    }
    pub(super) fn indexed_program(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Result<Option<Rc<dyn OrdinaryIndexedRequestProgram>>, Error> {
        self.indexed.for_step(step, prefill)
    }
}
