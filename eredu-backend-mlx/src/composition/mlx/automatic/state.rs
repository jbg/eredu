//! One state-configured load source through the existing factory workers.
use super::*;
use eredu_runtime::CacheResidencyPolicy;
type Inspection =
    eredu_core::ArtifactInspection<eredu_architectures::processor_plan::ArtifactArchitecturePlan>;

/// An MLX execution-plan factory with an explicit mutable-cache residency policy.
/// The default factory remains `Copy`; this configured owner retains the actual
/// finite policy and shares all target, drafting and resource realization paths.
#[derive(Debug, Clone)]
pub struct MlxStateBackendFactory {
    base: MlxBackendFactory,
    state: CacheResidencyPolicy,
}
impl MlxStateBackendFactory {
    pub(super) fn new(base: MlxBackendFactory, state: CacheResidencyPolicy) -> Self {
        Self { base, state }
    }
    /// Produces the exact portable request used for selection and realization.
    pub fn load_request_for_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<MlxLoadRequest, AutomaticPlanningError> {
        let ordinary = self.base.load_request_for_plan(plan)?;
        Ok(MlxLoadRequest::from_normalized(
            ordinary
                .normalized()
                .clone()
                .with_state_residency(self.state.clone()),
        ))
    }
}
impl AutomaticPlanningBackend for MlxStateBackendFactory {
    type Inspection = Inspection;
    fn backend_id(&self) -> BackendId {
        self.base.backend_id()
    }
    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        self.base.discover_hardware()
    }
    fn inspect_resources(
        &self,
        model_path: &Path,
    ) -> Result<(ModelResourceProfile, Inspection), AutomaticPlanningError> {
        inspect_resources(
            model_path,
            MlxInspectionOptions::new(MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default()
                    .with_state_residency(self.state.clone()),
            )),
        )
    }
    fn admit_candidate(
        &self,
        inspection: &Inspection,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        admit_candidate(inspection, self.load_request_for_plan(plan)?)
    }
    fn bounded_residency_requirement(
        &self,
        inspection: &Inspection,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
        bounded_residency_requirement(inspection, plan, || self.load_request_for_plan(plan))
    }
}
impl ExecutionPlanBackendFactory for MlxStateBackendFactory {
    type Backend = MlxBackend<'static>;
    type DrafterPreparation = eredu_architectures::ExternalDraftPreparation;
    type SelectedDrafterPreparation = eredu_architectures::PreparedExternalDraft;
    type Drafter = MlxDrafter;
    fn inspect_loading_artifact<R: eredu_core::ModelConfigurationResolver>(
        &self, path: &Path, resolver: &R,
    ) -> Result<eredu_core::ArtifactInspection<R::ArtifactPlan>, AutomaticPlanningError>
    where R::ArtifactPlan: Send + Sync + 'static,
    {
        self.base.inspect_loading_artifact(path, resolver)
    }
    fn select_target(
        &self,
        inspection: &Inspection,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionPlanTargetSelection<Self::Backend>, AutomaticPlanningError> {
        select_target(inspection, self.load_request_for_plan(plan)?)
    }
    fn realize_target(
        &self,
        selected: SelectedExecutionPlanTarget<Self::Backend>,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError> {
        self.base.realize_target(selected)
    }
    fn select_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &SelectedExecutionPlanTarget<Self::Backend>,
        artifact: Option<ExternalDraftArtifact<Self::DrafterPreparation>>,
    ) -> Result<
        Option<ExternalDraftArtifact<Self::SelectedDrafterPreparation>>,
        AutomaticPlanningError,
    > {
        self.base.select_drafting(plan, target, artifact)
    }
    fn realize_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &ModelRuntime<Self::Backend>,
        selected: eredu_core::SelectedExecutionPlanDrafting<Self::SelectedDrafterPreparation>,
    ) -> Result<RealizedDrafting<MlxDrafter>, AutomaticPlanningError> {
        self.base.realize_drafting(plan, target, selected)
    }
}
pub(super) fn inspect_resources(
    model_path: &Path,
    options: MlxInspectionOptions,
) -> Result<(ModelResourceProfile, Inspection), AutomaticPlanningError> {
    let inspection = MlxBackendFactory::default().inspect_loading_artifact(
        model_path, &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
    )?;
    let report = super::super::inspection::inspect_selected_artifact(&inspection, options);
    Ok((report.resources, inspection))
}

pub(super) fn admit_candidate(
    inspection: &Inspection,
    load: MlxLoadRequest,
) -> Result<CandidateAdmission, AutomaticPlanningError> {
    match super::super::loading::select_preparation(inspection, load) {
        Ok(_) => Ok(CandidateAdmission {
            supported: true,
            rejection: None,
        }),
        Err(error) => Ok(CandidateAdmission {
            supported: false,
            rejection: Some(error.to_string()),
        }),
    }
}

pub(super) fn bounded_residency_requirement(
    inspection: &Inspection,
    plan: &ExecutionPlan,
    options: impl FnOnce() -> Result<MlxLoadRequest, AutomaticPlanningError>,
) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
    if matches!(plan.residency(), ResidencyPlan::FullyResident) {
        return Err(AutomaticPlanningError::Invalid(
            "fully resident execution has no bounded device window".into(),
        ));
    }
    let options =
        options().map_err(|error| planning_backend_error("bounded_residency_options", error))?;
    let selected = super::super::loading::select_preparation(inspection, options)
        .map_err(|error| planning_backend_error("select_model_preparation", error))?;
    let (text, excluded) = selected.neutral().selected_bounded_residency();
    selected_text_bounded_requirement(&text, &excluded)
        .map_err(|error| planning_backend_error("selected_text_residency", error))
}

pub(super) fn select_target(
    inspection: &Inspection,
    options: MlxLoadRequest,
) -> Result<ExecutionPlanTargetSelection<MlxBackend<'static>>, AutomaticPlanningError> {
    let selected = super::super::loading::select_preparation(inspection, options)
        .map_err(|error| planning_backend_error("select_model_preparation", error))?;
    let policy = selected.neutral().admission().request().policy();
    let capabilities = selected.session_capabilities();
    Ok(ExecutionPlanTargetSelection::new(
        policy,
        selected,
        capabilities,
    ))
}
