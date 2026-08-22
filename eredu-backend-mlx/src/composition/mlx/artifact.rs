//! Composite-artifact discovery performed before MLX materialization.

use std::path::{Path, PathBuf};

use crate::backend::mlx::error::Error;

/// Finds the preferred sibling multimodal-projector GGUF for a model file.
///
/// Dense projectors are preferred when both dense and quantized conversions
/// are present. A sole quantized projector is retained for architectures that
/// admit checkpoint-native projector quantization.
pub fn find_sibling_mmproj(gguf_file: &Path, architecture: &str) -> Result<Option<PathBuf>, Error> {
    let parent = gguf_file.parent().unwrap_or_else(|| Path::new("."));
    let mut search_dirs = vec![parent];
    let parent_name = parent
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_uppercase();
    let is_quantization_directory = matches!(parent_name.as_str(), "F16" | "F32" | "BF16")
        || parent_name.starts_with("Q2_")
        || parent_name.starts_with("Q3_")
        || parent_name.starts_with("Q4_")
        || parent_name.starts_with("Q5_")
        || parent_name.starts_with("Q6_")
        || parent_name.starts_with("Q8_")
        || parent_name.starts_with("IQ")
        || parent_name.starts_with("UD-")
        || parent_name.starts_with("MXFP4");
    if let Some(grandparent) = parent.parent().filter(|_| is_quantization_directory) {
        if grandparent != parent {
            search_dirs.push(grandparent);
        }
    }
    let mut candidates = Vec::new();
    for directory in &search_dirs {
        candidates.extend(
            std::fs::read_dir(directory)?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    let name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    name.starts_with("mmproj") && name.ends_with(".gguf")
                }),
        );
    }
    candidates.sort();
    candidates.dedup();
    let dense = candidates
        .iter()
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            name.contains("f16") || name.contains("bf16") || name.contains("f32")
        })
        .cloned()
        .collect::<Vec<_>>();
    match (dense.as_slice(), candidates.as_slice()) {
        ([path], _) => Ok(Some(path.clone())),
        ([], [path]) => Ok(Some(path.clone())),
        ([], []) => Ok(None),
        _ => Err(Error::UnsupportedArchitecture(format!(
            "{architecture} GGUF requires an unambiguous nearby mmproj file; found {} candidates while searching {}",
            candidates.len(),
            search_dirs
                .iter()
                .map(|directory| directory.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}
