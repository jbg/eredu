#[cfg(feature = "hub")]
use super::transport::Hub;
use crate::Quality;
#[cfg(feature = "hub")]
use anyhow::Context;
#[cfg(feature = "hub")]
use anyhow::Result;
#[cfg(feature = "hub")]
use futures::{stream, StreamExt};
#[cfg(feature = "hub")]
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(feature = "hub")]
use std::collections::HashMap;

// Benchmark policy is shared across models; model IDs are always discovered.
pub const BENCHMARKS: &[&str] = &["Artificial Analysis Intelligence Index"];

#[cfg(feature = "hub")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkIndex {
    pub models: HashMap<String, Quality>,
    pub candidates: Vec<String>,
    pub version: String,
}

#[cfg(feature = "hub")]
impl BenchmarkIndex {
    /// Retrieves measured Artificial Analysis evaluations using an explicit credential.
    pub async fn artificial_analysis(key: &str) -> Result<Self> {
        let mut header = reqwest::header::HeaderValue::from_str(key.trim())?;
        header.set_sensitive(true);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let methodology = client
            .get("https://artificialanalysis.ai/methodology/intelligence-benchmarking")
            .send()
            .await?
            .error_for_status()?;
        let methodology = Hub::bounded_body(methodology, 16 * 1024 * 1024).await?;
        let version = regex::Regex::new(
            r"Artificial Analysis Intelligence Index v([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
        )?
        .captures(std::str::from_utf8(&methodology)?)
        .context("AA methodology version could not be identified")?[1]
            .to_owned();
        let response = client
            .get("https://artificialanalysis.ai/api/v2/data/llms/models")
            .header("x-api-key", header)
            .send()
            .await?
            .error_for_status()?;
        let data: Value =
            serde_json::from_slice(&Hub::bounded_body(response, 16 * 1024 * 1024).await?)?;
        let rows = data["data"].as_array().context("Expected AA model data")?;
        let mut scored = Vec::new();
        for row in rows {
            let Some(score) = row["evaluations"]["artificial_analysis_intelligence_index"]
                .as_f64()
                .filter(|score| score.is_finite())
            else {
                continue;
            };
            let id = row["id"].as_str().context("Missing AA model ID")?;
            let slug = row["slug"].as_str().context("Missing AA model slug")?;
            anyhow::ensure!(
                !slug.is_empty()
                    && slug != "."
                    && slug != ".."
                    && slug
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)),
                "Invalid AA model slug"
            );
            scored.push((
                score,
                id.to_owned(),
                slug.to_owned(),
                row["name"].as_str().unwrap_or(slug).to_owned(),
            ));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let results = stream::iter(scored.into_iter().map(|(score, id, slug, name)| {
            let client = &client;
            async move {
                let url = format!("https://artificialanalysis.ai/models/{slug}");
                let response = client.get(&url).send().await?.error_for_status()?;
                let bytes = Hub::bounded_body(response, 16 * 1024 * 1024).await?;
                let page = std::str::from_utf8(&bytes)?;
                let repo = weights_repo(page)?;
                let estimated = if repo.is_some() {
                    index_is_estimated(page, &id)
                        .with_context(|| format!("Checking AA evaluation status for {url}"))?
                } else {
                    false
                };
                Ok::<_, anyhow::Error>((repo, estimated, id, url, name, score))
            }
        }))
        .buffered(4)
        .collect::<Vec<_>>()
        .await;
        let evaluations = results.len();
        let mut unmapped = 0;
        let mut estimates = 0;
        let mut index = Self {
            version,
            ..Default::default()
        };
        for result in results {
            let (repo, estimated, id, url, name, score) = result?;
            let Some(repo) = repo else {
                unmapped += 1;
                continue;
            };
            if estimated {
                estimates += 1;
                continue;
            }
            // Multiple reasoning configurations can point to one checkpoint. Keep the
            // highest scoring measured configuration and retain its name for attribution.
            if index.models.contains_key(&repo) {
                continue;
            }
            let order = index.candidates.len() as u64 + 1;
            index.candidates.push(repo.clone());
            index.models.insert(
                repo.clone(),
                Quality {
                    evaluated_model_id: repo,
                    inherited: false,
                    evaluation_id: id,
                    provider: "Artificial Analysis".into(),
                    score: Some(score),
                    source_url: url,
                    evaluation_name: name,
                    rank: order,
                },
            );
        }
        anyhow::ensure!(
            !index.models.is_empty(),
            "AA returned no mapped open-weight models; existing catalogue preserved"
        );
        eprintln!("AA: {evaluations} scored configurations, {} distinct HF checkpoints, {unmapped} configurations without HF weights links, {estimates} estimated configurations excluded", index.models.len());
        Ok(index)
    }

    /// All entries share one ordered evaluation index and direct checkpoint attribution.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.version.is_empty() && !self.candidates.is_empty(),
            "Benchmark index has no version or candidates"
        );
        let mut repos = std::collections::HashSet::new();
        let mut ranks = std::collections::HashSet::new();
        let mut provider = None;
        for repo in &self.candidates {
            let quality = self
                .models
                .get(repo)
                .context("Benchmark candidate has no evaluation")?;
            anyhow::ensure!(
                crate::valid_repo(repo)
                    && repos.insert(repo)
                    && quality.evaluated_model_id == *repo
                    && !quality.inherited
                    && has_recommendation_evidence(Some(quality))
                    && ranks.insert(quality.rank)
                    && quality.score.is_none_or(f64::is_finite),
                "Invalid benchmark attribution or duplicate rank"
            );
            anyhow::ensure!(
                provider.is_none_or(|p| p == &quality.provider),
                "Benchmark index mixes evaluation providers"
            );
            provider = Some(&quality.provider);
        }
        Ok(())
    }

    pub fn for_model(&self, repo: &str, info: &Value) -> Option<Quality> {
        if let Some(quality) = self.models.get(repo) {
            return Some(quality.clone());
        }
        let parent = quantized_parent(info)?;
        let mut quality = self.models.get(parent)?.clone();
        quality.inherited = true;
        Some(quality)
    }
}

pub fn quantized_parent(info: &Value) -> Option<&str> {
    let card = &info["cardData"];
    let explicit = card["base_model_relation"].as_str();
    if explicit.is_some_and(|relation| relation != "quantized") {
        return None;
    }
    if info["tags"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|tag| {
            [
                "base_model:finetune:",
                "base_model:adapter:",
                "base_model:merge:",
                "base_model:distill:",
                "base_model:distillation:",
            ]
            .iter()
            .any(|prefix| tag.starts_with(prefix))
        })
    {
        return None;
    }
    let tagged: std::collections::BTreeSet<_> = info["tags"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|tag| tag.strip_prefix("base_model:quantized:"))
        .collect();
    let parent = match &card["base_model"] {
        Value::String(parent) => Some(parent.as_str()),
        Value::Array(parents) if parents.len() == 1 => parents[0].as_str(),
        Value::Null if tagged.len() == 1 => tagged.first().copied(),
        _ => None,
    }?;
    if tagged.iter().any(|tag| *tag != parent) {
        return None;
    }
    (explicit == Some("quantized") || tagged.contains(parent)).then_some(parent)
}

#[cfg(feature = "hub")]
pub async fn derived_models(hub: &Hub, parent: &str) -> Result<Vec<Value>> {
    let filter = format!("base_model:quantized:{parent}");
    hub.pages(
        "api/models",
        &[
            ("filter", &filter),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", "100"),
            ("full", "true"),
            ("config", "true"),
        ],
    )
    .await
}

/// New datasets can describe either additional training or quantization calibration.
/// Without distinguishing evidence, they cannot establish unchanged benchmark lineage.
pub fn preserves_evaluated_datasets(quantization: &Value, evaluated: &Value) -> bool {
    fn datasets(info: &Value) -> std::collections::BTreeSet<&str> {
        let mut ids: std::collections::BTreeSet<_> = info["tags"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter_map(|tag| tag.strip_prefix("dataset:"))
            .collect();
        match &info["cardData"]["datasets"] {
            Value::String(id) => {
                ids.insert(id.as_str());
            }
            Value::Array(values) => {
                ids.extend(values.iter().filter_map(Value::as_str));
            }
            _ => {}
        }
        ids
    }
    datasets(quantization).is_subset(&datasets(evaluated))
}

pub fn compare_quality(a: Option<&Quality>, b: Option<&Quality>) -> std::cmp::Ordering {
    let order = |quality: Option<&Quality>| {
        quality
            .filter(|q| has_recommendation_evidence(Some(q)) && q.rank > 0)
            .map(|q| q.rank)
            .unwrap_or(u64::MAX)
    };
    order(a).cmp(&order(b))
}

pub fn has_recommendation_evidence(quality: Option<&Quality>) -> bool {
    quality.is_some_and(|q| {
        !q.provider.is_empty()
            && !q.evaluation_id.is_empty()
            && !q.evaluated_model_id.is_empty()
            && !q.evaluation_name.is_empty()
            && q.source_url.starts_with("https://")
            && q.rank > 0
    })
}

#[cfg(feature = "hub")]
fn weights_repo(page: &str) -> Result<Option<String>> {
    let links = regex::Regex::new(
        r#"href="https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?">Hugging Face"#,
    )?;
    let repos: std::collections::BTreeSet<_> = links
        .captures_iter(page)
        .map(|capture| capture[1].to_owned())
        .collect();
    anyhow::ensure!(repos.len() <= 1, "Ambiguous AA weights links");
    Ok(repos.into_iter().next())
}

#[cfg(feature = "hub")]
fn index_is_estimated(page: &str, model_id: &str) -> Result<bool> {
    // AA does not expose estimate status in its API. Read the page's identified
    // currentModel, never related-model entries or a page-wide estimate label.
    let chunks = regex::Regex::new(r#"self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)"#)?;
    let mut payload = String::new();
    for capture in chunks.captures_iter(page) {
        payload.push_str(&serde_json::from_str::<String>(&capture[1])?);
    }
    let mut status = None;
    for (offset, key) in payload.match_indices("\"currentModel\":") {
        let suffix = payload
            .get(offset + key.len()..)
            .expect("ASCII match ends on a character boundary");
        let mut values = serde_json::Deserializer::from_str(suffix).into_iter::<Value>();
        let model = values.next().context("Missing AA current model")??;
        if model["id"].as_str() != Some(model_id) {
            continue;
        }
        let estimated = model["intelligenceIndexIsEstimated"]
            .as_bool()
            .context("AA evaluation has no explicit estimate status")?;
        anyhow::ensure!(
            status.is_none_or(|previous| previous == estimated),
            "Conflicting AA estimate status"
        );
        status = Some(estimated);
    }
    status.context("AA page has no matching current model with explicit estimate status")
}
