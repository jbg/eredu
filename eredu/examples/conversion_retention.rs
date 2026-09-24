//! Bound optional weight conversions, observe their scope, trim, and forecast a fresh request.
use std::path::PathBuf;

use eredu::api::{
    default_local_device, local_device_plan, GenerationForecastOptions, LoadedModel,
    PreparedChatGenerationSettings,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    residency::ParameterConversionRetentionPolicy, ExecutionPlan, GenerationConfigOverrides,
    TextGenerationConfig,
};

fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let artifact = arguments.first().map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!(
            "usage: cargo run -p eredu --example conversion_retention -- <artifact-with-tokenizer> [prompt]"
        )
    })?;
    let prompt = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("Explain gravity briefly.");
    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?)
        .with_parameter_conversion_retention(Some(ParameterConversionRetentionPolicy::Bounded {
            max_bytes: 32 * 1024 * 1024,
        }));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &artifact, &plan)?
            .into_parts();

    // The report supplies requested/effective policy, scope and eligibility;
    // consumers do not need a parallel ledger or native cache access.
    println!(
        "loaded retention: {}",
        serde_json::to_string_pretty(&model.parameter_conversion_retention()?)?
    );
    let token_ids = model.encode(prompt, false)?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        ..Default::default()
    };
    let sampling = model.resolve_generation_config(settings.overrides)?;
    for token in model.generate_tokens(token_ids.clone(), TextGenerationConfig::new(sampling))? {
        println!("token: {}", token?.token_id()?);
    }
    println!(
        "used retention: {}",
        serde_json::to_string_pretty(&model.parameter_conversion_retention()?)?
    );

    // Ordinary trimming settles work and preserves weights and request state.
    // Released claims are not a promise of reclaimed allocator backing or RSS.
    println!(
        "trim: {}",
        serde_json::to_string_pretty(&model.trim_parameter_conversions()?)?
    );
    println!(
        "after trim: {}",
        serde_json::to_string_pretty(&model.parameter_conversion_retention()?)?
    );

    // Reset is separate: clear request state to forecast a fresh next request.
    // It preserves any admitted conversions; this example has just trimmed them.
    model.reset()?;
    let forecast =
        model.forecast_token_ids(&token_ids, settings, &GenerationForecastOptions::default())?;
    // Includes current retention facts and pending casts; unknown bounds stay unknown.
    println!("next request: {}", serde_json::to_string_pretty(&forecast)?);
    Ok(())
}
