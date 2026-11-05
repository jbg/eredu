//! Explicit publisher membership, separate from optional ranking observations.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublisherSource {
    pub publisher: String,
    pub family: eredu_architectures::ModelKind,
    pub source_url: String,
    pub observed_at: u64,
}
impl PublisherSource {
    pub fn valid(&self, repo: &str) -> bool {
        crate::valid_repo(repo)
            && repo.split('/').next() == Some(self.publisher.as_str())
            && self.source_url == format!("https://huggingface.co/{}", self.publisher)
            && self
                .family
                .official_hf_organisations()
                .contains(&self.publisher.as_str())
            && self.observed_at > 0
    }
}

#[cfg(feature = "hub")]
mod import {
    use super::*;
    use crate::{ranking::Adoption, transport::Hub};
    use anyhow::{ensure, Context, Result};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PublisherSelection {
        pub publishers: Vec<String>,
        pub max_parameters: u64,
        pub per_publisher: usize,
        pub tasks: Vec<String>,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SourceEntry {
        pub repo_id: String,
        pub publisher: PublisherSource,
        pub adoption: Option<Adoption>,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SourceIndex {
        pub entries: Vec<SourceEntry>,
    }
    impl PublisherSelection {
        /// Publisher scope comes from the same exhaustive registry as architecture admission.
        pub fn supported(max_parameters: u64, per_publisher: usize) -> Self {
            let publishers: std::collections::BTreeSet<_> = eredu_architectures::ModelKind::ALL
                .iter()
                .flat_map(|k| {
                    k.official_hf_organisations()
                        .iter()
                        .map(|s| (*s).to_owned())
                })
                .collect();
            Self {
                publishers: publishers.into_iter().collect(),
                max_parameters,
                per_publisher,
                tasks: vec![
                    "text-generation".into(),
                    "image-text-to-text".into(),
                    "any-to-any".into(),
                ],
            }
        }
    }
    impl SourceIndex {
        pub fn validate(&self) -> Result<()> {
            let mut seen = std::collections::HashSet::new();
            ensure!(
                !self.entries.is_empty() && self.entries.len() <= 4096,
                "Invalid source count"
            );
            for entry in &self.entries {
                ensure!(
                    entry.publisher.valid(&entry.repo_id)
                        && seen.insert(entry.repo_id.to_lowercase())
                        && entry
                            .adoption
                            .as_ref()
                            .is_none_or(|a| a.valid(&entry.repo_id)),
                    "Invalid source evidence"
                );
            }
            Ok(())
        }
        /// Enumerate only caller-selected namespaces, including every paginated
        /// result before shortlisting. No unscoped Hub search or derivative search.
        /// Membership proves namespace ownership, not an independent quality review.
        pub async fn publishers(hub: &Hub, selection: &PublisherSelection) -> Result<Self> {
            ensure!(
                !selection.publishers.is_empty()
                    && selection.publishers.len() <= 64
                    && selection.max_parameters > 0
                    && (1..=128).contains(&selection.per_publisher)
                    && !selection.tasks.is_empty(),
                "Invalid publisher selection"
            );
            let mut entries = Vec::new();
            let mut publishers = std::collections::HashSet::new();
            for publisher in &selection.publishers {
                ensure!(
                    crate::valid_repo(&format!("{publisher}/model"))
                        && !publisher.contains('/')
                        && publishers.insert(publisher.to_lowercase()),
                    "Invalid or duplicate publisher"
                );
                let rows = hub
                    .uncached()
                    .pages(
                        "api/models",
                        &[
                            ("author", publisher),
                            ("limit", "100"),
                            ("expand[]", "createdAt"),
                            ("expand[]", "config"),
                            ("expand[]", "downloads"),
                            ("expand[]", "pipeline_tag"),
                            ("expand[]", "safetensors"),
                            ("expand[]", "gated"),
                            ("expand[]", "private"),
                        ],
                    )
                    .await?;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                let mut candidates = Vec::new();
                for row in rows {
                    let repo = row["id"]
                        .as_str()
                        .context("Missing publisher repository ID")?;
                    ensure!(
                        repo.split('/').next() == Some(publisher.as_str()),
                        "Publisher listing escaped its namespace"
                    );
                    if row["private"] != false
                        || row["gated"] != false
                        || !row["pipeline_tag"]
                            .as_str()
                            .is_some_and(|t| selection.tasks.iter().any(|x| x == t))
                        || !row["safetensors"]["total"]
                            .as_u64()
                            .is_some_and(|n| n > 0 && n <= selection.max_parameters)
                    {
                        continue;
                    }
                    let Some(family) = row["config"]["model_type"]
                        .as_str()
                        .and_then(|t| eredu_architectures::ModelKind::resolve_model_type(t).ok())
                        .filter(|k| k.official_hf_organisations().contains(&publisher.as_str()))
                    else {
                        continue;
                    };
                    let adoption = row["downloads"].as_u64().map(|downloads| Adoption {
                        provider: "Hugging Face".into(),
                        source_url: format!("https://huggingface.co/{repo}"),
                        observed_at: now,
                        window_days: 30,
                        downloads,
                        repository_created_at: row["createdAt"]
                            .as_str()
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .and_then(|d| u64::try_from(d.timestamp()).ok()),
                    });
                    candidates.push(SourceEntry {
                        repo_id: repo.into(),
                        publisher: PublisherSource {
                            publisher: publisher.clone(),
                            family,
                            source_url: format!("https://huggingface.co/{publisher}"),
                            observed_at: now,
                        },
                        adoption,
                    });
                }
                let policy = crate::ranking::RankingPolicy::default();
                candidates.sort_by(|a, b| {
                    let rate = |e: &SourceEntry| {
                        e.adoption.as_ref().map(|v| {
                            v.daily_rate(policy.adoption_prior_daily, policy.adoption_prior_days)
                        })
                    };
                    rate(b)
                        .partial_cmp(&rate(a))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.repo_id.cmp(&b.repo_id))
                });
                candidates.truncate(selection.per_publisher);
                entries.extend(candidates);
            }
            let index = Self { entries };
            index.validate()?;
            Ok(index)
        }
    }
}
#[cfg(feature = "hub")]
pub use import::{PublisherSelection, SourceEntry, SourceIndex};
