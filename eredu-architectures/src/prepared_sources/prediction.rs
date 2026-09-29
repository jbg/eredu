//! Header-only admission for separately supplied embedded prediction weights.

use std::{collections::BTreeMap, path::Path};

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{
    artifact::{ArtifactError, ArtifactFile},
    checkpoint::TensorCatalog,
    ArtifactFormat, ArtifactInspection, GgufCompanionRequirement, ModelConfiguration,
    ModelConfigurationResolver, ResolvedModelConfiguration, ValidatedGguf,
};

use crate::{
    configuration::{ModelConfigurations, PortableSafetensorsCatalog, SafetensorsModelConfig},
    qwen4_exp::{
        checkpoint::schema::SafetensorsEncoding, config::Config,
        prepared::SafetensorsPredictionPlan,
    },
};

#[derive(Debug, Clone)]
pub(crate) struct InspectedPredictionSource {
    inspection: ArtifactInspection<SafetensorsPredictionPlan>,
    files: Vec<ArtifactFile>,
}

impl InspectedPredictionSource {
    pub(crate) fn inspect(path: &Path) -> Result<Self, ArtifactError> {
        let inspection = eredu_core::artifact::inspect_artifact(path, &PredictionResolver)?;
        let PredictionPlan::Admitted(header) = inspection.architecture_plan() else {
            return Err(invalid("prediction companion header was not admitted"));
        };
        let header = header.clone();
        let shards = inspection
            .safetensors_shards()
            .ok_or_else(|| invalid("prediction companion has no retained SafeTensors shards"))?;
        let mut files = shards
            .logical_payload_paths()
            .iter()
            .map(|(role, path)| ArtifactFile::new(format!("prediction/{role}"), path))
            .collect::<Vec<_>>();
        files.push(ArtifactFile::new(
            "prediction/config.json",
            inspection.path().join("config.json"),
        ));
        Ok(Self {
            inspection: inspection.map_architecture_plan(|_| header),
            files,
        })
    }

    pub(crate) fn header_plan(&self) -> &SafetensorsPredictionPlan {
        self.inspection.architecture_plan()
    }

    pub(crate) fn open_source(
        &self,
        max_cached_sources: usize,
    ) -> Result<SharedCheckpointSource, ArtifactError> {
        let shards = self
            .inspection
            .safetensors_shards()
            .ok_or_else(|| invalid("prediction companion has no retained SafeTensors shards"))?;
        eredu_core::artifact::open_prepared_safetensors_artifact(
            self.inspection.tensors(),
            shards.clone(),
            self.header_plan().resolution().clone(),
            max_cached_sources,
        )
    }

    pub(crate) fn files(&self) -> Vec<ArtifactFile> {
        self.files.clone()
    }
}

#[derive(Debug, Clone)]
enum PredictionPlan {
    Unadmitted(Config, SafetensorsEncoding),
    Admitted(SafetensorsPredictionPlan),
}

struct PredictionResolver;

impl ModelConfigurationResolver for PredictionResolver {
    type ArtifactPlan = PredictionPlan;

    fn resolve_safetensors(
        &self,
        json: &serde_json::Value,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let (configuration, plan) = ModelConfigurations.resolve_safetensors(json)?.into_parts();
        let Some(architecture) = plan.safetensors_architecture() else {
            return Err(invalid(
                "prediction companion requires a SafeTensors configuration",
            ));
        };
        let SafetensorsModelConfig::Qwen4Exp(config, encoding) = architecture.model() else {
            return Err(invalid(
                "this prediction companion requires a Qwen4Exp configuration",
            ));
        };
        Ok(ResolvedModelConfiguration::new(
            configuration,
            PredictionPlan::Unadmitted(config.clone(), encoding.clone()),
        ))
    }

    fn resolve_gguf(
        &self,
        _architecture: &str,
        _checkpoint: &eredu_gguf::Checkpoint,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        Err(invalid("separate prediction weights must use SafeTensors"))
    }

    fn gguf_companion_requirements(
        &self,
        _architecture: &str,
        _checkpoint: &eredu_gguf::Checkpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
        Err(invalid("separate prediction weights must use SafeTensors"))
    }

    fn artifact_plan(
        &self,
        _sidecars: &BTreeMap<String, Vec<u8>>,
        format: ArtifactFormat,
        _configuration: &ModelConfiguration,
        tensors: &TensorCatalog,
        _validated_gguf: Option<&ValidatedGguf>,
        resolved_plan: Self::ArtifactPlan,
    ) -> Result<Self::ArtifactPlan, ArtifactError> {
        let PredictionPlan::Unadmitted(config, encoding) = resolved_plan else {
            return Err(invalid("prediction companion header is already admitted"));
        };
        if format != ArtifactFormat::SafeTensors {
            return Err(invalid("separate prediction weights must use SafeTensors"));
        }
        let physical = tensors
            .descriptors()
            .filter(|tensor| tensor.name.starts_with("mtp."))
            .map(|tensor| {
                let storage = tensor.storage.as_ref().ok_or_else(|| {
                    invalid(format!(
                        "prediction tensor {:?} has no storage provenance",
                        tensor.name
                    ))
                })?;
                let dtype = crate::replicated_text::stored_dtype(&tensor.dtype)
                    .map_err(|error| invalid(error.to_string()))?;
                let source = eredu_runtime::ReplicatedTextPhysicalSource::new(
                    &tensor.name,
                    &tensor.name,
                    std::path::PathBuf::from(&storage.member),
                    &tensor.name,
                    eredu_checkpoint::SourceTensorEncoding::Safetensors(dtype),
                    storage.length,
                )
                .map_err(|error| invalid(error.to_string()))?;
                Ok((tensor.name.clone(), source))
            })
            .collect::<Result<BTreeMap<_, _>, ArtifactError>>()?;
        SafetensorsPredictionPlan::prepare(
            &PortableSafetensorsCatalog(tensors),
            config,
            encoding,
            physical,
        )
        .map(PredictionPlan::Admitted)
        .map_err(|error| invalid(error.to_string()))
    }
}

fn invalid(message: impl Into<String>) -> ArtifactError {
    ArtifactError::InvalidArchitecturePlan(message.into())
}
