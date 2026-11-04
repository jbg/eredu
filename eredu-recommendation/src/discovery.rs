//! Progressive remote discovery with explicit credentials and immutable revisions.
use crate::{transport::Hub, ModelVariant};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Repository {
    pub repo_id: String,
    pub revision: String,
    pub downloads: u64,
    pub variants: Vec<ModelVariant>,
}

impl Hub {
    /// Search returns repository identifiers; expansion and inspection are explicit.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let value = self
            .json(
                "api/models",
                &[
                    ("search", query),
                    ("sort", "downloads"),
                    ("direction", "-1"),
                    ("limit", &limit.min(100).to_string()),
                ],
            )
            .await?;
        let rows = value.as_array().context("Expected model search results")?;
        Ok(rows
            .iter()
            .filter_map(|m| m["id"].as_str())
            .filter(|id| crate::valid_repo(id))
            .take(limit)
            .map(str::to_owned)
            .collect())
    }

    /// Resolve a moving reference once, then fetch and enumerate its pinned manifest.
    pub async fn repository(
        &self,
        repo: &str,
        revision: Option<&str>,
    ) -> Result<(Repository, Value)> {
        ensure!(crate::valid_repo(repo), "Invalid repository identity");
        let revision = match revision {
            Some(r) => r.to_owned(),
            None => self.model(repo).await?["sha"]
                .as_str()
                .context("Missing immutable repository revision")?
                .to_owned(),
        };
        ensure!(
            crate::valid_revision(&revision),
            "Expected an immutable 40-digit repository revision"
        );
        let manifest = self
            .json(
                &format!("api/models/{repo}/revision/{revision}"),
                &[("blobs", "true")],
            )
            .await?;
        ensure!(
            manifest["sha"].as_str() == Some(&revision),
            "Repository revision changed"
        );
        let variants = crate::variants::from_manifest(&manifest)?;
        let repository = Repository {
            repo_id: repo.into(),
            revision,
            downloads: manifest["downloads"].as_u64().unwrap_or(0),
            variants,
        };
        Ok((repository, manifest))
    }

    /// Fetch only container headers and small processor sidecars, never tensor payloads.
    pub async fn metadata(
        &self,
        repository: &Repository,
        variant: &ModelVariant,
        manifest: &Value,
    ) -> Result<Result<eredu::api::ArtifactMetadata, Vec<String>>> {
        ensure!(
            crate::valid_repo(&repository.repo_id) && crate::valid_revision(&repository.revision),
            "Invalid pinned repository"
        );
        ensure!(
            manifest["sha"].as_str() == Some(&repository.revision),
            "Manifest revision differs from repository"
        );
        crate::metadata::fetch_metadata(
            self,
            &repository.repo_id,
            &repository.revision,
            variant,
            manifest,
        )
        .await
    }
}
