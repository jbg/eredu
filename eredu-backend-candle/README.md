# Eredu Candle backend

Candle is an unoptimised backend primarily intended to validate the shared
backend contract. Use the direct context API below when the application owns
device binding. For factory-based
loading, see the [facade example](../eredu/README.md).

```rust,ignore
use eredu::api::{inspect_artifact, ArtifactSource, PreparedTextModel, TextModelOptions};
use eredu_runtime::ModelLoadFacts;
use eredu_backend_candle::CandleContext;
use eredu_core::{GenerationConfigOverrides, ModelLoadOptions, TextGenerationConfig};

let artifact = inspect_artifact(&ArtifactSource::new(checkpoint_path))?;
let device = eredu_core::DevicePlan::new("candle", "cpu:0")?;
let context = CandleContext::open(&device)?;
let mut model = PreparedTextModel::from_artifact(artifact, TextModelOptions::default())?
    .load_on::<_, ModelLoadFacts<_>>(context, ModelLoadOptions::default())?;
let prompt = model.encode("The capital of France is", true)?;
let config = model.resolve_generation_config(GenerationConfigOverrides {
    do_sample: Some(false),
    max_new_tokens: Some(8),
    ..Default::default()
})?;
let ids = model.generate_tokens(prompt, TextGenerationConfig::new(config))?
    .map(|token| token.and_then(|value| value.token_id()))
    .collect::<Result<Vec<_>, _>>()?;
println!("{}", model.decode(&ids, true)?);
```

Disable the facade's default features to omit MLX. Use `cpu:0` for CPU,
`metal:0` with the backend's `metal` feature on macOS, or `cuda:0` with
the backend's `cuda` feature and a CUDA toolkit. Distributed placement
is unavailable on Candle.

For media, enable the corresponding `image` or `audio` feature in both crates
when depending on the backend directly. Factory-based loading uses the
facade's `candle` feature together with its `image` or `audio` feature.

See [execution control](../doc/execution-control.md#candle-local-execution)
for admission and control semantics and
[family acceptance](../doc/candle-cpu-validation.md) for measured coverage.
