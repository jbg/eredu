//! Logical checkpoint discovery; filename hints never establish compatibility.
use crate::{File, ModelVariant};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub fn from_manifest(manifest: &Value) -> Result<Vec<ModelVariant>> {
    let siblings = manifest["siblings"]
        .as_array()
        .context("Missing artifact file listing")?;
    let mut listed = BTreeMap::new();
    for item in siblings {
        let name = item["rfilename"]
            .as_str()
            .context("Missing artifact member name")?;
        ensure!(crate::safe_member(name), "Invalid artifact member {name}");
        let size = item["size"]
            .as_u64()
            .or_else(|| item["lfs"]["size"].as_u64())
            .unwrap_or(0);
        ensure!(
            listed.insert(name.to_owned(), size).is_none(),
            "Duplicate artifact member {name}"
        );
    }
    let shards = regex::Regex::new(r"^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$")?;
    let mut groups: BTreeMap<String, Vec<(usize, usize, File)>> = BTreeMap::new();
    let mut projectors = Vec::new();
    for (name, size) in &listed {
        if !name.ends_with(".gguf") {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        let base = lower.rsplit('/').next().unwrap();
        if base.contains("mmproj") {
            projectors.push(name.clone());
            continue;
        }
        if lower.split('/').any(|p| p == "mtp" || p == "draft")
            || base.starts_with("mtp-")
            || base.starts_with("mtp_")
        {
            continue;
        }
        let (key, index, count) = match shards.captures(name) {
            Some(c) => (c[1].to_string(), c[2].parse()?, c[3].parse()?),
            None => (name.trim_end_matches(".gguf").into(), 1, 1),
        };
        groups.entry(key).or_default().push((
            index,
            count,
            File {
                name: name.clone(),
                size_bytes: *size,
            },
        ));
    }
    let hints = regex::Regex::new(
        r"(?i)(?:^|[-.])((?:IQ|Q|TQ)[0-9]+(?:_[A-Z0-9]+)*|BF16|F16|F32|FP8|MXFP4)(?:[-.]|$)",
    )?;
    let mut variants = Vec::new();
    let mut used_ids = BTreeSet::new();
    for (stem, mut members) in groups {
        members.sort_by_key(|m| m.0);
        let count = members[0].1;
        if count == 0
            || members.len() != count
            || members
                .iter()
                .enumerate()
                .any(|(i, m)| m.0 != i + 1 || m.1 != count)
        {
            continue;
        }
        let quant = hints.captures(&stem).map(|c| c[1].to_ascii_uppercase());
        let mut id = quant.clone().unwrap_or_else(|| stem.clone());
        if !used_ids.insert(id.clone()) {
            id = stem.clone();
            ensure!(used_ids.insert(id.clone()), "Duplicate artifact identity");
        }
        let filename = members[0].2.name.clone();
        let directory = filename.rsplit_once('/').map(|p| p.0).unwrap_or("");
        let matching: Vec<_> = projectors
            .iter()
            .filter(|p| p.rsplit_once('/').map(|p| p.0).unwrap_or("") == directory)
            .collect();
        let projector = (matching.len() == 1).then(|| matching[0].clone());
        let mut files: Vec<_> = members.into_iter().map(|m| m.2).collect();
        if let Some(name) = &projector {
            files.push(File {
                name: name.clone(),
                size_bytes: listed[name],
            });
        }
        add_sidecars(&listed, &mut files);
        variants.push(ModelVariant {
            variant_id: id,
            format: "gguf".into(),
            files,
            filename: Some(filename),
            projector,
            precision_rank: precision_rank(quant.as_deref()),
        });
    }
    let tensors: Vec<_> = listed
        .iter()
        .filter(|(name, _)| name.ends_with(".safetensors") && !name.contains('/'))
        .collect();
    if !tensors.is_empty() && listed.contains_key("config.json") {
        let mut files: Vec<_> = tensors
            .into_iter()
            .map(|(name, size)| File {
                name: name.clone(),
                size_bytes: *size,
            })
            .collect();
        add_sidecars(&listed, &mut files);
        let quant = manifest["config"]["quantization_config"]["bits"].as_u64();
        variants.push(ModelVariant {
            variant_id: "safetensors".into(),
            format: "safetensors".into(),
            files,
            filename: None,
            projector: None,
            precision_rank: quant.map_or(160, |b| (b.min(32) * 10) as u16),
        });
    }
    Ok(variants)
}

fn add_sidecars(listed: &BTreeMap<String, u64>, files: &mut Vec<File>) {
    for (name, size) in listed {
        if !name.contains('/')
            && matches!(
                name.as_str(),
                "config.json"
                    | "model.safetensors.index.json"
                    | "tokenizer.json"
                    | "tokenizer_config.json"
                    | "tokenizer.model"
                    | "special_tokens_map.json"
                    | "chat_template.jinja"
                    | "preprocessor_config.json"
                    | "processor_config.json"
                    | "video_preprocessor_config.json"
            )
        {
            files.push(File {
                name: name.clone(),
                size_bytes: *size,
            });
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files.dedup_by(|a, b| a.name == b.name);
}

fn precision_rank(quant: Option<&str>) -> u16 {
    let Some(quant) = quant else {
        return 0;
    };
    if quant == "F32" {
        return 320;
    }
    if quant == "BF16" || quant == "F16" {
        return 160;
    }
    if quant == "FP8" {
        return 80;
    }
    if quant == "MXFP4" {
        return 40;
    }
    let digits: String = quant
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    let base = digits.parse::<u16>().unwrap_or(0).saturating_mul(10);
    base.saturating_add(if quant.ends_with("_M") {
        2
    } else if quant.ends_with("_L") || quant.ends_with("_XL") {
        3
    } else {
        0
    })
}

pub(crate) fn gguf_members(variant: &ModelVariant) -> Result<(Vec<String>, Option<String>)> {
    let mut members: Vec<_> = variant
        .files
        .iter()
        .filter(|f| f.name.ends_with(".gguf") && Some(&f.name) != variant.projector.as_ref())
        .map(|f| f.name.clone())
        .collect();
    members.sort();
    ensure!(!members.is_empty(), "Missing primary GGUF members");
    Ok((members, variant.projector.clone()))
}
