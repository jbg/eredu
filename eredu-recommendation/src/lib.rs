#![doc = include_str!("../README.md")]

#[cfg(feature = "builder")]
pub mod builder;
pub mod catalogue;
#[cfg(feature = "hub")]
pub mod discovery;
pub mod forecast;
pub mod gguf;
pub mod lineage;
#[cfg(feature = "hub")]
mod metadata;
pub mod quality;
pub mod ranking;
pub mod sources;
pub mod suitability;
#[cfg(feature = "hub")]
pub mod transport;
#[cfg(feature = "hub")]
pub mod variants;
#[cfg(feature = "builder")]
pub mod worker;

use eredu::api::{GenerationMemoryEstimate, MemoryFit};
use serde::{Deserialize, Serialize};

/// A logical checkpoint, including every shard and explicitly selected companion.
/// Discovery establishes file identity, not backend compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelVariant {
    pub variant_id: String,
    pub format: String,
    pub files: Vec<File>,
    pub filename: Option<String>,
    pub projector: Option<String>,
    /// Stable preference within a model's encodings; this is not a benchmark score.
    pub precision_rank: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct File {
    pub name: String,
    pub size_bytes: u64,
}

/// One measured benchmark configuration and its checkpoint attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Quality {
    pub provider: String,
    pub evaluation_id: String,
    pub evaluated_model_id: String,
    pub evaluation_name: String,
    pub source_url: String,
    pub score: Option<f64>,
    /// One-based order within a single benchmark index, lower is better.
    pub rank: u64,
    pub inherited: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Compatibility {
    Compatible,
    Incompatible,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Suitability {
    Suitable,
    Unsuitable,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assessment {
    pub compatibility: Compatibility,
    pub suitability: Suitability,
    /// Retains separate physical memory domains and explicit uncertainty.
    pub memory: Option<GenerationMemoryEstimate>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub repo_id: String,
    pub revision: String,
    pub variant: ModelVariant,
    pub quality: Option<Quality>,
    pub publisher: Option<sources::PublisherSource>,
    pub adoption: Option<ranking::Adoption>,
    pub assessment: Assessment,
}

/// Application eligibility choices; compatibility and a proven likely fit are mandatory.
#[derive(Debug, Clone)]
pub struct RecommendationPolicy {
    pub ranking: ranking::RankingPolicy,
    pub require_conversational: bool,
}

impl Default for RecommendationPolicy {
    fn default() -> Self {
        Self {
            ranking: ranking::RankingPolicy::default(),
            require_conversational: true,
        }
    }
}

impl RecommendationPolicy {
    fn meets_requirements(&self, candidate: &Candidate) -> bool {
        candidate.assessment.compatibility == Compatibility::Compatible
            && (!self.require_conversational
                || candidate.assessment.suitability == Suitability::Suitable)
    }

    pub fn eligible(&self, candidate: &Candidate) -> bool {
        self.meets_requirements(candidate)
            && candidate
                .assessment
                .memory
                .as_ref()
                .is_some_and(|m| m.fit == MemoryFit::LikelyFit)
    }

    /// Candidates must use a common benchmark index and comparable forecast profiles.
    /// Unknown or straddling memory bounds never become affirmative fit advice.
    pub fn rank<'a>(
        &self,
        candidates: &'a [Candidate],
    ) -> Result<Vec<&'a Candidate>, ranking::RankingError> {
        self.ordered(candidates.iter().filter(|c| self.eligible(c)).collect())
    }

    /// Separate availability for supported candidates whose memory fit is unverified.
    /// These are not recommendations. Known shortfalls remain excluded; callers
    /// preserve load/request admission warnings before executing an unverified model.
    pub fn memory_unverified<'a>(
        &self,
        candidates: &'a [Candidate],
    ) -> Result<Vec<&'a Candidate>, ranking::RankingError> {
        self.ordered(
            candidates
                .iter()
                .filter(|c| {
                    self.meets_requirements(c)
                        && c.assessment
                            .memory
                            .as_ref()
                            .is_some_and(|m| m.fit == MemoryFit::InsufficientInformation)
                })
                .collect(),
        )
    }

    fn ordered<'a>(
        &self,
        mut ranked: Vec<&'a Candidate>,
    ) -> Result<Vec<&'a Candidate>, ranking::RankingError> {
        let scores = self.ranking.scores(&ranked).ok_or(ranking::RankingError)?;
        let mut scored: Vec<_> = ranked.drain(..).zip(scores).collect();
        scored.sort_by(|(a, sa), (b, sb)| {
            sb.total_cmp(sa)
                .then_with(|| b.variant.precision_rank.cmp(&a.variant.precision_rank))
                .then_with(|| a.repo_id.cmp(&b.repo_id))
                .then_with(|| a.variant.variant_id.cmp(&b.variant.variant_id))
        });
        ranked = scored.into_iter().map(|(c, _)| c).collect();
        Ok(ranked)
    }
}

pub fn valid_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

pub(crate) fn safe_member(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && name
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

pub(crate) fn valid_repo(repo: &str) -> bool {
    repo.split('/').count() == 2
        && repo.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        })
}
