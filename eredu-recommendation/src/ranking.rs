//! Portable ranking observations, independent of eligibility and benchmark coverage.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Adoption {
    pub provider: String,
    pub source_url: String,
    /// UTC Unix seconds at collection time. Counts belong to a repository, not a revision.
    pub observed_at: u64,
    pub window_days: u16,
    pub downloads: u64,
    /// Hub repository creation, not the date of its latest commit or a proven release date.
    pub repository_created_at: Option<u64>,
}

impl Adoption {
    pub fn valid(&self, repo: &str) -> bool {
        self.provider == "Hugging Face"
            && self.source_url == format!("https://huggingface.co/{repo}")
            && self.observed_at > 0
            && self.window_days == 30
            && self
                .repository_created_at
                .is_none_or(|t| t > 0 && t <= self.observed_at)
    }
    /// Only young repositories have a shorter exposure window. Missing age uses
    /// the full window; an edit never renews exposure. The prior damps launch spikes.
    pub fn daily_rate(&self, prior_daily: f64, prior_days: f64) -> f64 {
        let days = self
            .repository_created_at
            .map(|t| {
                ((self.observed_at.saturating_sub(t)) as f64 / 86_400.0)
                    .min(f64::from(self.window_days))
            })
            .unwrap_or(f64::from(self.window_days));
        (self.downloads as f64 + prior_daily * prior_days) / (days + prior_days)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RankingSignal {
    MeasuredBenchmark,
    RecentAdoption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightedSignal {
    pub signal: RankingSignal,
    pub weight: f64,
}

/// Signal values are percentiles among observed eligible candidates. Missing
/// evidence has neutral percentile 0.5, not a fabricated measurement of zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankingPolicy {
    pub signals: Vec<WeightedSignal>,
    pub adoption_prior_daily: f64,
    pub adoption_prior_days: f64,
}
impl Default for RankingPolicy {
    fn default() -> Self {
        Self {
            signals: vec![
                WeightedSignal {
                    signal: RankingSignal::MeasuredBenchmark,
                    weight: 1.0,
                },
                WeightedSignal {
                    signal: RankingSignal::RecentAdoption,
                    weight: 1.0,
                },
            ],
            adoption_prior_daily: 100.0,
            adoption_prior_days: 7.0,
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("invalid ranking policy: weights and adoption prior must be finite and positive, with unique signals")]
pub struct RankingError;

impl RankingPolicy {
    pub fn valid(&self) -> bool {
        !self.signals.is_empty()
            && self
                .signals
                .iter()
                .all(|s| s.weight.is_finite() && s.weight > 0.0)
            && self
                .signals
                .iter()
                .enumerate()
                .all(|(i, s)| !self.signals[..i].iter().any(|p| p.signal == s.signal))
            && self
                .signals
                .iter()
                .map(|s| s.weight)
                .sum::<f64>()
                .is_finite()
            && self.adoption_prior_daily.is_finite()
            && self.adoption_prior_daily >= 0.0
            && self.adoption_prior_days.is_finite()
            && self.adoption_prior_days > 0.0
            && ((u64::MAX as f64 + self.adoption_prior_daily * self.adoption_prior_days)
                / self.adoption_prior_days)
                .is_finite()
    }
    /// Returned scores are ranking contributions, never model quality estimates.
    pub fn scores(&self, candidates: &[&crate::Candidate]) -> Option<Vec<f64>> {
        if !self.valid() {
            return None;
        }
        let mut scores = vec![0.0; candidates.len()];
        let weight: f64 = self.signals.iter().map(|s| s.weight).sum();
        for signal in &self.signals {
            let values: Vec<_> = candidates
                .iter()
                .map(|c| match signal.signal {
                    RankingSignal::MeasuredBenchmark => c
                        .quality
                        .as_ref()
                        .filter(|q| crate::quality::has_recommendation_evidence(Some(q)))
                        .map(|q| -(q.rank as f64)),
                    RankingSignal::RecentAdoption => c
                        .adoption
                        .as_ref()
                        .filter(|a| a.valid(&c.repo_id))
                        .map(|a| a.daily_rate(self.adoption_prior_daily, self.adoption_prior_days)),
                })
                .collect();
            // Multiple encodings of a checkpoint do not create extra popularity votes.
            let observed: std::collections::HashMap<_, _> = candidates
                .iter()
                .zip(&values)
                .filter_map(|(c, v)| v.map(|v| (c.repo_id.as_str(), v)))
                .collect();
            let observed: Vec<_> = observed.into_values().collect();
            for (i, value) in values.iter().enumerate() {
                let percentile = value
                    .filter(|_| observed.len() > 1)
                    .map(|v| {
                        let below = observed.iter().filter(|&&x| x < v).count() as f64;
                        let equal = observed.iter().filter(|&&x| x == v).count() as f64;
                        (below + (equal - 1.0) / 2.0) / (observed.len() - 1) as f64
                    })
                    .unwrap_or(0.5);
                scores[i] += percentile * (signal.weight / weight);
            }
        }
        Some(scores)
    }
}
