//! Curator-selected checkpoint sources, independent of benchmark measurements.
use serde::{Deserialize, Serialize};

pub const LM_STUDIO_CATALOG: &str = "https://lmstudio.ai/models";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Curation {
    pub provider: String,
    pub model_id: String,
    pub source_url: String,
    pub revision: u64,
    pub definition_sha256: String,
    /// Position in the curator's catalog, not a performance measurement.
    pub rank: u64,
}

impl Curation {
    pub fn valid(&self) -> bool {
        self.provider == "LM Studio"
            && crate::valid_repo(&self.model_id)
            && self.source_url == format!("{LM_STUDIO_CATALOG}/{}", self.model_id)
            && self.revision > 0
            && self.rank > 0
            && self.definition_sha256.len() == 64
            && self
                .definition_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
    }
}

#[cfg(feature = "hub")]
mod importer {
    use super::*;
    use anyhow::{ensure, Context, Result};
    use futures::{stream, StreamExt};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{collections::HashSet, time::Duration};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CuratedRepository {
        pub repo_id: String,
        pub curation: Curation,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CurationIndex {
        pub catalog_url: String,
        pub catalog_sha256: String,
        pub repositories: Vec<CuratedRepository>,
    }

    #[derive(Deserialize)]
    struct Definition {
        model: String,
        #[serde(default)]
        base: Vec<Value>,
    }

    impl CurationIndex {
        pub fn version(&self) -> String {
            format!("lm-studio:sha256:{}", self.catalog_sha256)
        }

        pub fn validate(&self) -> Result<()> {
            ensure!(
                self.catalog_url == LM_STUDIO_CATALOG
                    && self.catalog_sha256.len() == 64
                    && self.catalog_sha256.bytes().all(|b| b.is_ascii_hexdigit())
                    && !self.repositories.is_empty(),
                "Invalid LM Studio catalog snapshot"
            );
            let mut repositories = HashSet::new();
            for entry in &self.repositories {
                ensure!(
                    crate::valid_repo(&entry.repo_id)
                        && entry.curation.valid()
                        && repositories.insert(entry.repo_id.to_ascii_lowercase()),
                    "Invalid or duplicate curated repository"
                );
            }
            Ok(())
        }

        /// Import only local entries on LM Studio's featured catalog and the exact
        /// HF sources referenced by their model.yaml files. Never search HF or the Hub.
        pub async fn lm_studio(cache_dir: Option<std::path::PathBuf>) -> Result<Self> {
            let client = reqwest::Client::builder()
                .user_agent("eredu-recommendation-builder")
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(90))
                .redirect(reqwest::redirect::Policy::none())
                .build()?;
            let transport = Transport { client, cache_dir };
            let client = &transport;
            let page = fetch(client, LM_STUDIO_CATALOG).await?;
            let catalog_records = records(&page)?;
            let mut featured = Vec::new();
            for value in &catalog_records.json {
                walk(value, &mut |value| {
                    if value["data-model-page-has-downloads"] == true {
                        if let Some(path) = value["href"].as_str().filter(|p| valid_path(p)) {
                            featured.push(path.to_owned());
                        }
                    }
                });
            }
            unique(&mut featured);
            ensure!(
                !featured.is_empty() && featured.len() <= 256,
                "LM Studio featured catalog is missing or exceeds its expected bounds"
            );
            eprintln!("LM Studio: {} featured local collections", featured.len());
            let pages = stream::iter(featured.iter().map(|path| async move {
                let url = format!("https://lmstudio.ai{path}");
                let body = fetch(client, &url).await?;
                let values = records(&body)?;
                let mut paths = Vec::new();
                if path.trim_start_matches("/models/").split('/').count() == 2 {
                    paths.push(path.clone());
                } else {
                    for value in &values.json {
                        walk(value, &mut |value| {
                            for option in value["downloadOptions"].as_array().into_iter().flatten()
                            {
                                if let Some(id) = option["identifier"]
                                    .as_str()
                                    .filter(|id| crate::valid_repo(id))
                                {
                                    paths.push(format!("/models/{id}"));
                                }
                            }
                        });
                    }
                }
                unique(&mut paths);
                ensure!(
                    !paths.is_empty(),
                    "LM Studio local collection {url} has no model definitions"
                );
                Ok::<_, anyhow::Error>(paths)
            }))
            .buffered(4)
            .collect::<Vec<_>>()
            .await;
            let mut models = Vec::new();
            for paths in pages {
                models.extend(paths?);
            }
            unique(&mut models);
            ensure!(
                !models.is_empty() && models.len() <= 1024,
                "Invalid curated model count"
            );
            eprintln!("LM Studio: {} selected model definitions", models.len());
            let results = stream::iter(models.iter().enumerate().map(|(rank, path)| async move {
                let model_id = path.trim_start_matches("/models/");
                let url = format!("https://lmstudio.ai{path}");
                let body = fetch(client, &url).await?;
                parse_model(&body, model_id, rank as u64 + 1)
                    .with_context(|| format!("Reading LM Studio definition {url}"))
            }))
            .buffered(4)
            .collect::<Vec<_>>()
            .await;
            let mut repositories = Vec::new();
            let mut seen = HashSet::new();
            for result in results {
                for entry in result? {
                    if seen.insert(entry.repo_id.to_ascii_lowercase()) {
                        repositories.push(entry);
                    }
                }
            }
            let result = Self {
                catalog_url: LM_STUDIO_CATALOG.into(),
                catalog_sha256: digest(page.as_bytes()),
                repositories,
            };
            result.validate()?;
            Ok(result)
        }
    }

    struct Transport {
        client: reqwest::Client,
        cache_dir: Option<std::path::PathBuf>,
    }

    async fn fetch(transport: &Transport, url: &str) -> Result<String> {
        const LIMIT: usize = 16 * 1024 * 1024;
        let cache = transport
            .cache_dir
            .as_ref()
            .map(|dir| dir.join(digest(url.as_bytes())));
        if let Some(path) = &cache {
            if std::fs::metadata(path).ok().is_some_and(|m| {
                m.len() <= LIMIT as u64
                    && m.modified()
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age < Duration::from_secs(3600))
            }) {
                return Ok(std::fs::read_to_string(path)?);
            }
        }
        for attempt in 0..4 {
            let response = transport
                .client
                .get(url)
                .send()
                .await
                .and_then(reqwest::Response::error_for_status);
            match response {
                Ok(response) => {
                    ensure!(
                        response.status().is_success(),
                        "Unexpected LM Studio HTTP status {}",
                        response.status()
                    );
                    let bytes = crate::transport::Hub::bounded_body(response, LIMIT).await?;
                    let body = String::from_utf8(bytes)?;
                    if let Some(path) = &cache {
                        let directory = path.parent().unwrap();
                        std::fs::create_dir_all(directory)?;
                        let temporary = tempfile::NamedTempFile::new_in(directory)?;
                        std::fs::write(temporary.path(), &body)?;
                        temporary.persist(path).map_err(|error| error.error)?;
                    }
                    return Ok(body);
                }
                Err(error)
                    if attempt < 3
                        && error
                            .status()
                            .is_none_or(|s| s.is_server_error() || s.as_u16() == 429) =>
                {
                    tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt + 1))).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
        unreachable!()
    }

    fn valid_path(path: &str) -> bool {
        path.strip_prefix("/models/").is_some_and(|s| {
            !s.is_empty()
                && s.split('/').count() <= 2
                && s.split('/').all(|p| {
                    !p.is_empty()
                        && p != "."
                        && p != ".."
                        && p.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                })
        })
    }

    fn unique(values: &mut Vec<String>) {
        let mut seen = HashSet::new();
        values.retain(|value| seen.insert(value.clone()));
    }

    fn digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn walk(value: &Value, visit: &mut impl FnMut(&Value)) {
        visit(value);
        match value {
            Value::Array(values) => {
                for value in values {
                    walk(value, visit);
                }
            }
            Value::Object(values) => {
                for value in values.values() {
                    walk(value, visit);
                }
            }
            _ => {}
        }
    }

    struct Records {
        json: Vec<Value>,
        text: std::collections::BTreeMap<String, String>,
    }

    // The public site renders model.yaml as a UTF-8 length-prefixed RSC text record.
    // Decode the framing rather than matching YAML-looking text from descriptions.
    fn records(page: &str) -> Result<Records> {
        let chunks = regex::Regex::new(r#"self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)"#)?;
        let mut payload = String::new();
        for capture in chunks.captures_iter(page) {
            payload.push_str(&serde_json::from_str::<String>(&capture[1])?);
        }
        ensure!(
            !payload.is_empty(),
            "LM Studio page has no structured records"
        );
        let mut result = Records {
            json: Vec::new(),
            text: std::collections::BTreeMap::new(),
        };
        let mut offset = 0;
        while offset < payload.len() {
            let suffix = &payload[offset..];
            let Some(colon) = suffix.find(':') else {
                break;
            };
            ensure!(
                suffix[..colon].bytes().all(|b| b.is_ascii_hexdigit()),
                "Invalid LM Studio record framing"
            );
            let start = offset + colon + 1;
            if payload.as_bytes().get(start) == Some(&b'T') {
                let comma = payload[start + 1..]
                    .find(',')
                    .context("Missing LM Studio text length")?
                    + start
                    + 1;
                let size = usize::from_str_radix(&payload[start + 1..comma], 16)?;
                let end = (comma + 1)
                    .checked_add(size)
                    .context("LM Studio record length overflow")?;
                let text = payload
                    .get(comma + 1..end)
                    .context("Truncated LM Studio text record")?;
                result.text.insert(suffix[..colon].into(), text.into());
                offset = end;
            } else {
                let end = payload[start..]
                    .find('\n')
                    .map(|n| start + n)
                    .unwrap_or(payload.len());
                let record = &payload[start..end];
                if record.starts_with('[') || record.starts_with('{') {
                    result.json.push(serde_json::from_str(record)?);
                }
                offset = end;
            }
            if payload.as_bytes().get(offset) == Some(&b'\n') {
                offset += 1;
            }
        }
        Ok(result)
    }

    fn parse_model(page: &str, model_id: &str, rank: u64) -> Result<Vec<CuratedRepository>> {
        let records = records(page)?;
        let mut revision = None;
        for value in &records.json {
            walk(value, &mut |value| {
                if value["artifact"]["identifier"] == model_id {
                    if let Some(number) = value["artifact"]["current"]["revisionNumber"].as_u64() {
                        revision = Some(number);
                    }
                }
            });
        }
        let revision = revision.context("Missing LM Studio artifact revision")?;
        let mut yaml_sources = Vec::new();
        for value in &records.json {
            walk(value, &mut |value| {
                if let Some(component) = value.get("rawComponent") {
                    let mut is_definition = false;
                    walk(component, &mut |v| {
                        if v.as_str() == Some("model.yaml") {
                            is_definition = true;
                        }
                    });
                    if is_definition {
                        walk(component, &mut |v| {
                            if let Some(content) = v["content"].as_str() {
                                let text = content
                                    .strip_prefix('$')
                                    .and_then(|id| records.text.get(id).map(String::as_str))
                                    .unwrap_or(content);
                                if text.len() <= 256 * 1024 {
                                    yaml_sources.push(text.to_owned());
                                }
                            }
                        });
                    }
                }
            });
        }
        unique(&mut yaml_sources);
        let mut definitions = Vec::new();
        for text in &yaml_sources {
            if let Ok(definition) = serde_saphyr::from_str::<Definition>(text) {
                if definition.model == model_id {
                    definitions.push((text, definition));
                }
            }
        }
        ensure!(
            definitions.len() == 1,
            "Missing or ambiguous LM Studio model.yaml"
        );
        let (yaml, definition) = definitions.pop().unwrap();
        let curation = Curation {
            provider: "LM Studio".into(),
            model_id: model_id.into(),
            source_url: format!("{LM_STUDIO_CATALOG}/{model_id}"),
            revision,
            definition_sha256: digest(yaml.as_bytes()),
            rank,
        };
        ensure!(curation.valid(), "Invalid LM Studio model provenance");
        let mut repositories = Vec::new();
        for variant in definition.base {
            for source in variant["sources"].as_array().into_iter().flatten() {
                if source["type"] == "huggingface" {
                    let user = source["user"]
                        .as_str()
                        .context("Missing curated source owner")?;
                    let repo = source["repo"]
                        .as_str()
                        .context("Missing curated source repository")?;
                    let repo_id = format!("{user}/{repo}");
                    ensure!(crate::valid_repo(&repo_id), "Invalid curated HF identity");
                    repositories.push(CuratedRepository {
                        repo_id,
                        curation: curation.clone(),
                    });
                }
            }
        }
        ensure!(
            !repositories.is_empty(),
            "Curated model has no direct HF sources"
        );
        Ok(repositories)
    }
}

#[cfg(feature = "hub")]
pub use importer::{CuratedRepository, CurationIndex};
