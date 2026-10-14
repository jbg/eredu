//! Print exact layout-independent GGUF identities without decoding parameters.
use std::{error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let paths = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("usage: gguf_catalog_fingerprint ARTIFACT ...".into());
    }
    for path in paths {
        let checkpoint = eredu_gguf::Checkpoint::open_with_representation(
            &path,
            eredu_gguf::Limits::default(),
            eredu_gguf::QuantizedTensorRepresentation::Encoded,
        )?;
        let digest = eredu_checkpoint::gguf_fingerprint::fingerprint_gguf_catalog(&checkpoint)?;
        let digest = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        println!(
            "{}",
            serde_json::json!({"path":path, "catalog_sha256":digest})
        );
    }
    Ok(())
}
