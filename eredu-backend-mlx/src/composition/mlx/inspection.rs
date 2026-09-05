//! Side-effect-free MLX adapter for architecture-owned model inspection.

use std::path::Path;

use eredu_core::ModelInspectionReport;

use super::*;

/// Options applied while inspecting a model artifact.
#[derive(Debug, Clone, Default)]
pub struct MlxInspectionOptions {
    /// The exact loading policy that admission should validate.
    load: MlxLoadRequest,
}

impl MlxInspectionOptions {
    /// Creates inspection options for an exact load request.
    pub const fn new(load: MlxLoadRequest) -> Self {
        Self { load }
    }

    /// Returns the load request whose feasibility is being inspected.
    pub fn load(&self) -> MlxLoadRequest {
        self.load.clone()
    }
}

/// Inspects a local SafeTensors model directory or GGUF checkpoint without
/// instantiating a model, materializing tensor payloads, or creating an MLX
/// execution stream.
pub fn inspect_model(
    path: impl AsRef<Path>,
    options: MlxInspectionOptions,
) -> Result<ModelInspectionReport, Error> {
    Ok(inspect_model_preparation(path, options)?.into_report())
}

/// Inspects and retains the exact neutral selection for callers that will
/// proceed directly to source preparation.
pub fn inspect_model_preparation(
    path: impl AsRef<Path>,
    options: MlxInspectionOptions,
) -> Result<eredu_architectures::ModelInspectionOutcome, Error> {
    let (request, _rank) = options.load.checked_normalized()?;
    let mechanisms = preparation_mechanisms();
    Ok(eredu_architectures::inspect_model(
        path,
        request,
        &mechanisms,
        media_feature_availability(),
    ))
}

/// Inspects backend mechanisms against one already admitted portable artifact.
pub(crate) fn inspect_selected_artifact(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    options: MlxInspectionOptions,
) -> ModelInspectionReport {
    inspect_selected_preparation(inspection.clone(), &options)
        .expect("automatic inspection already validated its MLX load request")
        .into_report()
}

/// Retains the exact neutral preparation selected while producing the report.
pub(crate) fn inspect_selected_preparation(
    inspection: eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    options: &MlxInspectionOptions,
) -> Result<eredu_architectures::ModelInspectionOutcome, Error> {
    let (request, _rank) = options.load.checked_normalized()?;
    let mechanisms = preparation_mechanisms();
    Ok(eredu_architectures::inspect_selected_model(
        inspection,
        request,
        &mechanisms,
        media_feature_availability(),
    ))
}

fn preparation_mechanisms() -> super::loading::MlxPreparationMechanisms<'static> {
    super::loading::MlxPreparationMechanisms::new(
        &super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
    )
}

const fn media_feature_availability() -> eredu_core::MediaFeatureAvailability {
    eredu_core::MediaFeatureAvailability {
        image: cfg!(feature = "image"),
        audio: cfg!(feature = "audio"),
    }
}

#[cfg(test)]
mod tests;
