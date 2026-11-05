//! Catalogue construction from pinned metadata and an explicit benchmark index.
use crate::{
    catalogue::{Catalogue, Model, Profile, Variant, SCHEMA_VERSION},
    forecast::Inspection,
    transport::Hub,
    worker::{self, WorkerLimits},
    Compatibility, Suitability,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Exclusion {
    pub repo_id: String,
    pub variant_id: Option<String>,
    /// Distinguishes a proven incompatibility from missing evidence or resource limits.
    pub compatibility: Compatibility,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildReport {
    pub exclusions: Vec<Exclusion>,
}

/// Build-time coverage and input formats. These never replace artifact inspection.
pub struct CatalogueSelection {
    pub max_parameters: u64,
    pub formats: Vec<String>,
    pub public_only: bool,
}

pub struct CatalogueBuilder {
    pub hub: Hub,
    pub executable: PathBuf,
    pub limits: WorkerLimits,
    pub profile: Profile,
    pub engine_revision: String,
}

impl CatalogueBuilder {
    async fn inspect(&self, metadata: eredu::api::ArtifactMetadata) -> Result<Inspection> {
        let (path, profile, limits) = (self.executable.clone(), self.profile.clone(), self.limits);
        tokio::task::spawn_blocking(move || worker::inspect(&path, &profile, metadata, limits))
            .await?
    }

    /// Source membership establishes candidates; measured benchmarks only enrich evidence.
    pub async fn build_sources(
        &self,
        sources: &crate::sources::SourceIndex,
        benchmarks: Option<&crate::quality::BenchmarkIndex>,
        selection: &CatalogueSelection,
    ) -> Result<(Catalogue, BuildReport)> {
        self.profile.validate()?;
        sources.validate()?;
        if let Some(index) = benchmarks {
            index.validate()?;
        }
        ensure!(
            !self.engine_revision.is_empty()
                && selection.max_parameters > 0
                && !selection.formats.is_empty()
                && selection
                    .formats
                    .iter()
                    .all(|f| matches!(f.as_str(), "safetensors" | "gguf")),
            "Invalid explicit-source build selection"
        );
        let mut catalogue = Catalogue {
            schema_version: SCHEMA_VERSION,
            engine_revision: self.engine_revision.clone(),
            profile: self.profile.clone(),
            benchmark_version: benchmarks.map(|b| b.version.clone()),
            models: Vec::new(),
        };
        let mut report = BuildReport::default();
        for entry in &sources.entries {
            match self
                .model(&entry.repo_id, benchmarks, entry, selection)
                .await
            {
                Ok((model, exclusions)) => {
                    if let Some(model) = model {
                        catalogue.models.push(model);
                    }
                    report.exclusions.extend(exclusions);
                }
                Err(error) if source_failure(&error) => return Err(error),
                Err(error) => report.exclusions.push(Exclusion {
                    repo_id: entry.repo_id.clone(),
                    variant_id: None,
                    compatibility: Compatibility::Unknown,
                    reason: error.to_string(),
                }),
            }
        }
        catalogue.models.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
        ensure!(
            !catalogue.models.is_empty(),
            "No compatible source candidates; existing output preserved"
        );
        catalogue.validate(&self.engine_revision)?;
        Ok((catalogue, report))
    }

    async fn model(
        &self,
        repo: &str,
        index: Option<&crate::quality::BenchmarkIndex>,
        source: &crate::sources::SourceEntry,
        selection: &CatalogueSelection,
    ) -> Result<(Option<Model>, Vec<Exclusion>)> {
        let (repository, manifest) = self.hub.repository(repo, None).await?;
        let config = self
            .hub
            .file(repo, &repository.revision, "config.json")
            .await?;
        let identity = eredu_architectures::configuration::resolve_model_identity(
            &serde_json::from_slice(&config)?,
        )?;
        ensure!(
            identity.kind == source.publisher.family && source.publisher.valid(repo),
            "Pinned architecture does not match its official publisher declaration"
        );
        let quality = index
            .and_then(|index| index.for_model(repo, &manifest))
            .filter(|q| !q.inherited);
        ensure!(
            !selection.public_only || (manifest["gated"] == false && manifest["private"] != true),
            "Source is gated or private"
        );
        let parameters = manifest["safetensors"]["total"].as_u64();
        ensure!(
            parameters.is_some_and(|n| n > 0 && n <= selection.max_parameters),
            "Outside {}-parameter coverage or missing parameter count: {:?}",
            selection.max_parameters,
            parameters
        );
        let (purpose, evidence) =
            crate::suitability::assess_checked(&self.hub, &manifest, quality.as_ref()).await?;
        let mut model = Model {
            repo_id: repo.into(),
            revision: repository.revision.clone(),
            quality,
            publisher: Some(source.publisher.clone()),
            adoption: source.adoption.clone(),
            variants: Vec::new(),
        };
        let mut exclusions = Vec::new();
        for artifact in &repository.variants {
            if !selection.formats.contains(&artifact.format) {
                exclusions.push(Exclusion {
                    repo_id: repo.into(),
                    variant_id: Some(artifact.variant_id.clone()),
                    compatibility: Compatibility::Unknown,
                    reason: "Outside selected catalog artifact formats".into(),
                });
                continue;
            }
            let result: Result<Variant> = async {
                let metadata = self.hub.metadata(&repository, artifact, &manifest).await?;
                let inspection = match metadata {
                    Ok(metadata) => self.inspect(metadata).await?,
                    Err(reasons) => Inspection {
                        provenance: None,
                        compatibility: Compatibility::Incompatible,
                        reasons,
                        forecast: None,
                    },
                };
                let mut suitability = purpose;
                if suitability == Suitability::Unknown {
                    let template = crate::suitability::fetch_chat_template_checked(
                        &self.hub,
                        repo,
                        &repository.revision,
                        &manifest,
                    )
                    .await?;
                    suitability =
                        crate::suitability::classify(&manifest, model.quality.as_ref(), template).0;
                }
                Ok(Variant {
                    artifact: artifact.clone(),
                    lineage: None,
                    suitability,
                    suitability_evidence: vec![evidence.clone()],
                    inspection,
                })
            }
            .await;
            match result {
                Ok(item) if item.inspection.compatibility == Compatibility::Compatible => {
                    model.variants.push(item)
                }
                Ok(item) => exclusions.push(Exclusion {
                    repo_id: repo.into(),
                    variant_id: Some(artifact.variant_id.clone()),
                    compatibility: item.inspection.compatibility,
                    reason: item.inspection.reasons.join("; "),
                }),
                Err(error) if source_failure(&error) => return Err(error),
                Err(error) => exclusions.push(Exclusion {
                    repo_id: repo.into(),
                    variant_id: Some(artifact.variant_id.clone()),
                    compatibility: Compatibility::Unknown,
                    reason: error.to_string(),
                }),
            }
        }
        Ok(((!model.variants.is_empty()).then_some(model), exclusions))
    }

    /// Recheck the existing immutable artifacts with a new engine revision/profile.
    /// Benchmark attribution and source revisions are retained. Rejections remain in the report.
    pub async fn refresh(&self, original: &Catalogue) -> Result<(Catalogue, BuildReport)> {
        self.profile.validate()?;
        ensure!(
            !self.engine_revision.is_empty(),
            "Engine revision must be explicit"
        );
        original.validate(&original.engine_revision)?;
        let mut catalogue = Catalogue {
            schema_version: SCHEMA_VERSION,
            engine_revision: self.engine_revision.clone(),
            profile: self.profile.clone(),
            benchmark_version: original.benchmark_version.clone(),
            models: Vec::new(),
        };
        let mut report = BuildReport::default();
        for old in &original.models {
            let (repository, manifest) = self
                .hub
                .repository(&old.repo_id, Some(&old.revision))
                .await?;
            let mut model = Model {
                variants: Vec::new(),
                ..old.clone()
            };
            for item in &old.variants {
                let result: Result<Inspection> = async {
                    match self
                        .hub
                        .metadata(&repository, &item.artifact, &manifest)
                        .await?
                    {
                        Ok(metadata) => self.inspect(metadata).await,
                        Err(reasons) => Ok(Inspection {
                            provenance: None,
                            compatibility: Compatibility::Incompatible,
                            reasons,
                            forecast: None,
                        }),
                    }
                }
                .await;
                match result {
                    Ok(inspection) if inspection.compatibility == Compatibility::Compatible => {
                        model.variants.push(Variant {
                            inspection,
                            ..item.clone()
                        })
                    }
                    Ok(inspection) => report.exclusions.push(Exclusion {
                        repo_id: old.repo_id.clone(),
                        variant_id: Some(item.artifact.variant_id.clone()),
                        compatibility: inspection.compatibility,
                        reason: inspection.reasons.join("; "),
                    }),
                    Err(error) if source_failure(&error) => return Err(error),
                    Err(error) => report.exclusions.push(Exclusion {
                        repo_id: old.repo_id.clone(),
                        variant_id: Some(item.artifact.variant_id.clone()),
                        compatibility: Compatibility::Unknown,
                        reason: error.to_string(),
                    }),
                }
            }
            if !model.variants.is_empty() {
                catalogue.models.push(model);
            }
        }
        ensure!(
            !catalogue.models.is_empty(),
            "No verified catalogue candidates; existing output is preserved"
        );
        catalogue.validate(&self.engine_revision)?;
        Ok((catalogue, report))
    }
}

fn source_failure(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        source
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|error| {
                // A discovered repository may be gated, removed, or inaccessible to
                // this caller. Report that candidate as unverified. Rate limits,
                // service failures, and transport errors still abort publication.
                !error
                    .status()
                    .is_some_and(|status| matches!(status.as_u16(), 401 | 403 | 404 | 410))
            })
    })
}

/// Atomic replacement uses a temporary file in the destination directory.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), value)?;
    temporary.as_file_mut().sync_all()?;
    temporary.persist(path).map_err(|e| e.error)?;
    Ok(())
}
