//! Caller-owned catalogues retain exact forecasts and immutable artifact evidence.
use crate::{
    forecast::{self, DomainBudget, Inspection},
    Candidate, ModelVariant, Quality, Suitability,
};
use eredu::api::{
    BackendId, ForecastCalibration, GenerationMemoryOptions, GenerationMemoryPlacement,
    InputTokenCount,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Placement {
    Unified,
    Host,
    Separate { device: String },
    Unknown,
}

/// Reproducible text-only, cold, fully resident builder workload.
/// Direct inspection also accepts arbitrary facade load and memory options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub backend: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub batch_size: u64,
    pub prefill_chunk_tokens: u64,
    pub placement: Placement,
    pub allocator_cache_bytes: u64,
}

impl Profile {
    pub fn validate(&self) -> Result<(), CatalogueError> {
        BackendId::new(&self.backend).map_err(|e| CatalogueError::Invalid(e.to_string()))?;
        if self.input_tokens == 0
            || self.output_tokens == 0
            || self.batch_size == 0
            || self.prefill_chunk_tokens == 0
        {
            return Err(CatalogueError::Invalid(
                "workload dimensions must be positive".into(),
            ));
        }
        if matches!(&self.placement, Placement::Separate {device} if device.is_empty()) {
            return Err(CatalogueError::Invalid(
                "separate memory needs a device identity".into(),
            ));
        }
        Ok(())
    }

    pub fn memory_options(&self) -> Result<GenerationMemoryOptions, CatalogueError> {
        self.validate()?;
        let placement = match &self.placement {
            Placement::Unified => GenerationMemoryPlacement::Unified,
            Placement::Host => GenerationMemoryPlacement::Host,
            Placement::Separate { device } => GenerationMemoryPlacement::Separate {
                device: device.clone(),
            },
            Placement::Unknown => GenerationMemoryPlacement::Unknown,
        };
        let mut options =
            GenerationMemoryOptions::new(InputTokenCount::text(self.input_tokens), placement);
        options.max_output_tokens = Some(self.output_tokens);
        options.batch_size = self.batch_size;
        options.prefill_chunk_tokens = self.prefill_chunk_tokens;
        options.backend_overhead =
            ForecastCalibration::default().allocator_overhead(self.allocator_cache_bytes, 0);
        Ok(options)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Catalogue {
    pub schema_version: u32,
    /// Caller-supplied immutable engine build/revision identifier.
    pub engine_revision: String,
    pub profile: Profile,
    pub benchmark_version: Option<String>,
    pub models: Vec<Model>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub repo_id: String,
    pub revision: String,
    pub quality: Option<Quality>,
    pub publisher: Option<crate::sources::PublisherSource>,
    pub adoption: Option<crate::ranking::Adoption>,
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Variant {
    pub artifact: ModelVariant,
    pub lineage: Option<crate::lineage::Verification>,
    pub suitability: Suitability,
    pub suitability_evidence: Vec<String>,
    pub inspection: Inspection,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogueError {
    #[error("invalid recommendation catalogue: {0}")]
    Invalid(String),
    #[error("catalogue engine identity differs: expected {expected}, found {found}")]
    EngineMismatch { expected: String, found: String },
    #[error(transparent)]
    Forecast(#[from] forecast::ForecastError),
}

impl Catalogue {
    pub fn validate(&self, expected_engine: &str) -> Result<(), CatalogueError> {
        if self.engine_revision != expected_engine {
            return Err(CatalogueError::EngineMismatch {
                expected: expected_engine.into(),
                found: self.engine_revision.clone(),
            });
        }
        let invalid = |s: &str| CatalogueError::Invalid(s.into());
        if self.schema_version != SCHEMA_VERSION
            || self.engine_revision.is_empty()
            || self
                .benchmark_version
                .as_ref()
                .is_some_and(|s| s.is_empty())
        {
            return Err(invalid("missing provenance or unsupported schema"));
        }
        self.profile.validate()?;
        let mut repos = HashSet::new();
        for model in &self.models {
            if !crate::valid_repo(&model.repo_id)
                || !crate::valid_revision(&model.revision)
                || !repos.insert(&model.repo_id)
                || model.variants.is_empty()
            {
                return Err(invalid("invalid repository, revision or duplicate model"));
            }
            if model
                .publisher
                .as_ref()
                .is_some_and(|p| !p.valid(&model.repo_id))
                || model
                    .adoption
                    .as_ref()
                    .is_some_and(|a| !a.valid(&model.repo_id))
                || (model.quality.is_some() && self.benchmark_version.is_none())
            {
                return Err(invalid("invalid publisher or ranking provenance"));
            }
            if model
                .quality
                .as_ref()
                .is_some_and(|q| !crate::quality::has_recommendation_evidence(Some(q)))
            {
                return Err(invalid("incomplete benchmark attribution"));
            }
            if model.quality.as_ref().is_some_and(|q| {
                !crate::valid_repo(&q.evaluated_model_id)
                    || (!q.inherited && q.evaluated_model_id != model.repo_id)
            }) {
                return Err(invalid(
                    "benchmark attribution differs from checkpoint identity",
                ));
            }
            let mut ids = HashSet::new();
            for item in &model.variants {
                if !ids.insert(&item.artifact.variant_id)
                    || item.artifact.variant_id.is_empty()
                    || item.artifact.files.is_empty()
                    || !matches!(item.artifact.format.as_str(), "gguf" | "safetensors")
                {
                    return Err(invalid("invalid or duplicate artifact variant"));
                }
                let mut names = HashSet::new();
                for file in &item.artifact.files {
                    if !crate::safe_member(&file.name) || !names.insert(file.name.as_str()) {
                        return Err(invalid("invalid or duplicate artifact member"));
                    }
                }
                if item
                    .artifact
                    .filename
                    .as_ref()
                    .is_some_and(|n| !names.contains(n.as_str()))
                    || item
                        .artifact
                        .projector
                        .as_ref()
                        .is_some_and(|n| !names.contains(n.as_str()))
                {
                    return Err(invalid("representative or companion absent from manifest"));
                }
                if model.quality.as_ref().is_some_and(|q| q.inherited) {
                    let lineage = item
                        .lineage
                        .as_ref()
                        .ok_or_else(|| invalid("inherited benchmark lacks identity evidence"))?;
                    if !crate::valid_revision(&lineage.evaluated_revision) {
                        return Err(invalid("invalid evaluated revision"));
                    }
                    lineage
                        .artifact
                        .matches(&lineage.evaluated, &item.artifact.format)
                        .map_err(|e| CatalogueError::Invalid(e.to_string()))?;
                }
                if !item.inspection.provenance.as_ref().is_some_and(|p| {
                    p.revision == model.revision
                        && p.source == format!("{}:{}", model.repo_id, item.artifact.variant_id)
                }) {
                    return Err(invalid(
                        "inspection provenance differs from pinned artifact",
                    ));
                }
                match (&item.inspection.forecast, item.inspection.compatibility) {
                    (Some(f), crate::Compatibility::Compatible) => {
                        if f.request.input != InputTokenCount::text(self.profile.input_tokens)
                            || f.request.max_output_tokens != Some(self.profile.output_tokens)
                            || f.request.batch_size != self.profile.batch_size
                            || f.requested_chunk_tokens != self.profile.prefill_chunk_tokens
                            || f.request.domains.is_empty()
                        {
                            return Err(invalid("stored forecast workload differs from profile"));
                        }
                        // A stored availability observation must never become reusable fit advice.
                        if f.request.domains.iter().any(|d| {
                            d.already_resident_bytes != 0 || d.budget != Default::default()
                        }) {
                            return Err(invalid(
                                "catalogue forecasts require cold, capacity-free domains",
                            ));
                        }
                    }
                    (None, crate::Compatibility::Incompatible | crate::Compatibility::Unknown) => {}
                    _ => return Err(invalid("compatibility contradicts forecast evidence")),
                }
            }
        }
        Ok(())
    }

    /// Reassesses stored forecasts against fresh domain budgets without network access.
    pub fn assess(
        &self,
        expected_engine: &str,
        budgets: &[DomainBudget],
    ) -> Result<Vec<Candidate>, CatalogueError> {
        self.validate(expected_engine)?;
        self.models
            .iter()
            .flat_map(|model| model.variants.iter().map(move |item| (model, item)))
            .map(|(model, item)| {
                let mut assessment = forecast::assess(&item.inspection, item.suitability, budgets)?;
                assessment
                    .reasons
                    .extend(item.suitability_evidence.iter().cloned());
                Ok(Candidate {
                    repo_id: model.repo_id.clone(),
                    revision: model.revision.clone(),
                    variant: item.artifact.clone(),
                    quality: model.quality.clone(),
                    publisher: model.publisher.clone(),
                    adoption: model.adoption.clone(),
                    assessment,
                })
            })
            .collect()
    }
}
