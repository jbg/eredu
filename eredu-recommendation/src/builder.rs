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
    /// Expand measured publisher checkpoints into lineage-verified quantizations.
    pub include_quantizations: bool,
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

    /// Repository selection and benchmark policy belong to the supplied index.
    /// Each candidate is pinned before inspection; failed candidates are reported.
    pub async fn build(
        &self,
        index: &crate::quality::BenchmarkIndex,
    ) -> Result<(Catalogue, BuildReport)> {
        self.build_selected(
            index,
            &CatalogueSelection {
                max_parameters: u64::MAX,
                formats: vec!["safetensors".into(), "gguf".into()],
                public_only: false,
                include_quantizations: true,
            },
        )
        .await
    }

    /// Build from measured checkpoint identities, with explicit coverage limits.
    /// Disabling quantization expansion inspects only benchmark-mapped publishers.
    pub async fn build_selected(
        &self,
        index: &crate::quality::BenchmarkIndex,
        selection: &CatalogueSelection,
    ) -> Result<(Catalogue, BuildReport)> {
        self.profile.validate()?;
        index.validate()?;
        ensure!(
            selection.max_parameters > 0
                && !selection.formats.is_empty()
                && selection
                    .formats
                    .iter()
                    .all(|f| matches!(f.as_str(), "safetensors" | "gguf")),
            "Invalid catalog selection"
        );
        ensure!(
            !self.engine_revision.is_empty(),
            "Engine revision must be explicit"
        );
        let mut catalogue = Catalogue {
            schema_version: SCHEMA_VERSION,
            engine_revision: self.engine_revision.clone(),
            profile: self.profile.clone(),
            benchmark_version: index.version.clone(),
            models: Vec::new(),
        };
        let mut report = BuildReport::default();
        let mut repos = std::collections::HashSet::new();
        for parent in &index.candidates {
            let mut candidates = if selection.include_quantizations {
                crate::quality::derived_models(&self.hub, parent).await?
            } else {
                Vec::new()
            };
            candidates.insert(0, serde_json::json!({"id":parent}));
            for candidate in candidates {
                let Some(repo) = candidate["id"].as_str() else {
                    continue;
                };
                if !repos.insert(repo.to_owned()) {
                    continue;
                }
                match self.model(repo, index, Some(selection)).await {
                    Ok((model, exclusions)) => {
                        if let Some(model) = model {
                            catalogue.models.push(model);
                        }
                        report.exclusions.extend(exclusions);
                    }
                    Err(error) if source_failure(&error) => {
                        // Source failures abort publication instead of silently shrinking coverage.
                        return Err(error.context(format!("Building {repo}")));
                    }
                    Err(error) => report.exclusions.push(Exclusion {
                        repo_id: repo.into(),
                        variant_id: None,
                        compatibility: Compatibility::Unknown,
                        reason: error.to_string(),
                    }),
                }
            }
        }
        catalogue.models.sort_by(|a, b| {
            crate::quality::compare_quality(a.quality.as_ref(), b.quality.as_ref())
                .then_with(|| a.repo_id.cmp(&b.repo_id))
        });
        ensure!(
            !catalogue.models.is_empty(),
            "No verified catalogue candidates; existing output is preserved"
        );
        catalogue.validate(&self.engine_revision)?;
        Ok((catalogue, report))
    }

    async fn model(
        &self,
        repo: &str,
        index: &crate::quality::BenchmarkIndex,
        selection: Option<&CatalogueSelection>,
    ) -> Result<(Option<Model>, Vec<Exclusion>)> {
        let (repository, manifest) = self.hub.repository(repo, None).await?;
        let quality = index.for_model(repo, &manifest);
        if quality.is_none() {
            return Ok((
                None,
                vec![Exclusion {
                    repo_id: repo.into(),
                    variant_id: None,
                    compatibility: Compatibility::Unknown,
                    reason: "No unambiguous measured benchmark lineage".into(),
                }],
            ));
        }
        let original = quality
            .as_ref()
            .filter(|q| q.inherited)
            .map(|q| q.evaluated_model_id.as_str());
        if let Some(selection) = selection {
            ensure!(
                !selection.public_only
                    || (manifest["gated"] == false && manifest["private"] != true),
                "Source is gated or private"
            );
            let metadata = match original {
                Some(parent) => self.hub.model(parent).await?,
                None => manifest.clone(),
            };
            let parameters = metadata["safetensors"]["total"].as_u64();
            ensure!(
                parameters.is_some_and(|n| n > 0 && n <= selection.max_parameters),
                "Outside {}-parameter coverage or missing original parameter count: {:?}",
                selection.max_parameters,
                parameters
            );
        }
        let parent = match original {
            Some(original) => {
                Some(crate::lineage::evaluated(&self.hub, repo, &manifest, original).await?)
            }
            None => None,
        };
        let (purpose, evidence) =
            crate::suitability::assess_checked(&self.hub, &manifest, quality.as_ref()).await?;
        let mut model = Model {
            repo_id: repo.into(),
            revision: repository.revision.clone(),
            quality,
            variants: Vec::new(),
        };
        let mut exclusions = Vec::new();
        for artifact in &repository.variants {
            if selection.is_some_and(|s| !s.formats.contains(&artifact.format)) {
                exclusions.push(Exclusion {
                    repo_id: repo.into(),
                    variant_id: Some(artifact.variant_id.clone()),
                    compatibility: Compatibility::Unknown,
                    reason: "Outside selected catalog artifact formats".into(),
                });
                continue;
            }
            let result: Result<Variant> = async {
                let lineage = match &parent {
                    Some(p) => Some(
                        crate::lineage::verify(&self.hub, repo, &repository.revision, artifact, p)
                            .await?,
                    ),
                    None => None,
                };
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
                    lineage,
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
