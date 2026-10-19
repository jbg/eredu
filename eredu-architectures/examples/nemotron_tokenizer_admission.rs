//! Validates all seven pinned released tokenizer policies without a native backend.
use eredu_architectures::nemotron_labs_diffusion::{tokenizer, FamilyConfig};
use eredu_text::tokenizer::Tokenizer;
use std::{error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: nemotron_tokenizer_admission RELEASE_ROOT")?,
    );
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/nemotron_labs_diffusion/tokenizer-cases.json"
    ))?;
    for name in [
        "3B", "3B-Base", "8B", "8B-Base", "14B", "14B-Base", "VLM-8B",
    ] {
        let directory = root.join(name);
        let config = FamilyConfig::from_hf_json(&std::fs::read(directory.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        tokenizer::validate(&config, &tokenizer)?;
        compare_cases(&tokenizer, &cases)?;
        let encoded = tokenizer
            .encode("Hello, world!", false)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "{name}: reserved policy admitted; text IDs={:?}",
            encoded.get_ids()
        );
    }
    for artifact in args {
        let (checkpoint, plan) =
            eredu_architectures::nemotron_labs_diffusion::gguf::TextGgufPlan::open(PathBuf::from(
                artifact,
            ))?;
        let metadata = checkpoint
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let reconstructed = eredu_text::gguf::from_metadata(&metadata)?
            .ok_or("GGUF tokenizer cannot be reconstructed")?;
        let tokenizer = Tokenizer::from_tokenizer(reconstructed.tokenizer);
        tokenizer::validate(plan.config(), &tokenizer)?;
        compare_cases(&tokenizer, &cases)?;
        println!(
            "{}: GGUF tokenizer matches pinned Unicode, digits and reserved-token cases",
            plan.provenance().repository()
        );
    }
    Ok(())
}

fn compare_cases(tokenizer: &Tokenizer, cases: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    for case in cases["cases"].as_array().ok_or("missing tokenizer cases")? {
        let text = case["text"].as_str().ok_or("missing tokenizer text")?;
        let expected = case["token_ids"]
            .as_array()
            .ok_or("missing tokenizer IDs")?
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let actual = tokenizer
            .encode(text, false)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        if actual.get_ids() != expected {
            return Err(format!(
                "tokenizer mismatch for {text:?}: {:?} versus {expected:?}",
                actual.get_ids()
            )
            .into());
        }
    }
    Ok(())
}
