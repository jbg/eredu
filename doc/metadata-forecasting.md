# Forecasting from supplied checkpoint metadata

Applications can inspect checkpoint metadata without downloading weight payloads
or creating a native device. `eredu::api::inspect_model_metadata` accepts an
`ArtifactMetadata` bundle and `MetadataInspectionOptions`. The options require an
explicit `BackendId` and carry a `NormalizedLoadRequest` for quantization,
residency, speculation and parallel policy. The facade obtains cold capability
facts from that backend and invokes the shared architecture inspection driver.
The returned `ModelInspectionOutcome` feeds `forecast_inspected_generation` and
`estimate_inspected_generation_memory`.

Applications depend only on `eredu` and import these contracts through
`eredu::api`. They do not construct capability providers or import backend crates.
Selecting `BackendId::new("mlx")` requires eredu's `mlx` Cargo feature. A known
backend without its feature returns `MetadataInspectionError::BackendNotCompiled`;
an unknown identifier returns `MetadataInspectionError::UnknownBackend`. There is
no default backend, automatic selection or fallback. Compiling an adapter makes
its capabilities available; it does not select it or establish hardware
availability. Inspection does not discover hardware, fetch objects, create a
device or access local checkpoint files.

## Inputs and provenance

`ArtifactMetadata` contains a source identifier, an immutable revision identifier,
format-specific metadata and applicable processor sidecars. Inspection retains
that provenance in `ModelInspectionReport::metadata_provenance`. The application
owns transport, credentials, object caching and verification that every supplied
object belongs to the declared revision. Source identifiers are diagnostic labels;
inspection never opens them as filesystem paths.

For SafeTensors, supply:

- Exact `config.json` bytes.
- Exact index bytes when the checkpoint is indexed. Multiple shards require an
  index; its tensor-to-member map must exactly match the supplied headers.
- Every referenced shard's relative member name, eight-byte header-length prefix,
  complete JSON header and full object length. Header buffers contain no payload.
- Applicable processor sidecars, keyed by relative name. An absent sidecar key
  declares that the sidecar is absent from the revision.

For GGUF, supply:

- Each shard's relative member name, full object length and complete byte prefix
  through the metadata and tensor descriptors. Alignment padding is optional.
- Shard zero first, with all members named according to the GGUF split convention.
- Complete header sets for any architecture-required companions, such as a media
  projector, each with its explicit semantic role. Companion encoding and
  architecture binding are validated by eredu.

The format parsers validate lengths, offsets, shapes, encodings, duplicate names
and shard membership. Architecture analysis validates the exact tensor contract,
quantization companions and materialization recipes. Full object lengths are
caller-supplied facts: structural validation of headers does not prove payload
availability or integrity. Configuration, a quantization label or parameter count
alone cannot substitute for a complete tensor catalog.

## Application integration

This example receives fetched SafeTensors metadata and an explicit memory budget.
All model interpretation and memory arithmetic remain in eredu:

```rust,ignore
use eredu::api::*;
use std::collections::BTreeMap;

let metadata = ArtifactMetadata {
    provenance: MetadataProvenance {
        source: repository_and_variant,
        revision: immutable_revision,
    },
    checkpoint: CheckpointMetadata::SafeTensors {
        config: config_json_bytes,
        index: index_json_bytes, // Option<Vec<u8>>
        headers: shard_headers, // Vec<SafetensorsHeader>
    },
    sidecars: BTreeMap::new(), // This revision has no applicable processor sidecars.
};
let load_request = NormalizedLoadRequest::default()
    .with_parameter_conversion_retention(Some(ParameterConversionRetentionPolicy::Unlimited));
let inspection_options = MetadataInspectionOptions::new(BackendId::new("mlx")?)
    .with_load_request(load_request);
let inspection = inspect_model_metadata(&metadata, &inspection_options)?;
if !inspection.report().is_compatible() {
    // Present structured issues or exclude this candidate.
    return Ok(());
}
let mut options = GenerationMemoryOptions::new(
    InputTokenCount::text(2_000),
    GenerationMemoryPlacement::Unified,
);
options.max_output_tokens = Some(256);
options.budget.application_limit_bytes = Some(8 * 1024 * 1024 * 1024);
options.budget.reserve_bytes = 512 * 1024 * 1024;
let forecast = forecast_inspected_generation(
    &inspection,
    &options,
    &Default::default(),
)?;
```

`GenerationMemoryOptions::for_hardware_device` accepts an existing neutral
`HardwareProfile` and device selection without performing discovery. Supply the
memory placement and capacities of the target hardware, plus application limits
and reserves. Request geometry, output allowance, concurrency, residency and
conversion-retention policy are part of the candidate being evaluated.

Target memory placement, capacity and application reserves remain separate from
backend capabilities. All types needed to supply hardware facts, configure the
portable load request, inspect metadata and forecast memory are available through
`eredu::api`. Backend integrations implement `PreparationMechanismProvider` for
the lower-level `eredu-architectures` driver; this is not an application input.

## Results and loading

`is_compatible()` means the catalog and requested execution policy pass cold
selection. `is_loadable()` also requires local artifact admission and is false
for supplied metadata. Tokenizer, chat-template and tool behavior are separate
readiness contracts; this API does not infer them from tensor headers.

A successful metadata selection supports forecasts but grants no payload-loading
authority. Both model-preparation constructors return `ArtifactError::MetadataOnly`
for it. GGUF materializers and conversion iterators also reject metadata-only
checkpoints before filesystem access. After downloading the chosen artifact,
inspect the actual local files before preparing sources or executing a model.

Local and supplied-header inspection share architecture finalization, cold
selection, materialization sizing and forecasting. The SafeTensors parser shares
header validation with local admission; GGUF shares its reader and shard catalog
assembly. Neither entry point claims native layout or dispatch eligibility from
headers. Cold promotion allowances, unknown workspaces and unsupported execution
paths remain explicit. Applications preserve indeterminate fit results and use
the forecast's bounds and assumptions when ranking candidates.

## Validation

`eredu/tests/fixtures/metadata-client` has `eredu` as its only dependency. Its
behavioral tests exercise unknown and disabled backend errors, explicit MLX
inspection, metadata rejection and forecasting using only facade imports:

```sh
cargo test --manifest-path eredu/tests/fixtures/metadata-client/Cargo.toml
cargo test --manifest-path eredu/tests/fixtures/metadata-client/Cargo.toml --features mlx
```

`cargo test -p eredu --no-default-features --test metadata_forecast` compares
local and supplied-header selection, complete forecast requests and estimates at
4, 128 and 2,000 positions. SafeTensors coverage includes single and indexed split
checkpoints with resident, host-layerwise and disk-streamed requests. GGUF
coverage includes single and split checkpoints with F32, F16, BF16, Q4_0 and Q8_0
weights. The tests delete checkpoint files before supplied-header inspection and
check that source preparation fails.

`eredu-checkpoint` and `eredu-gguf` tests cover incomplete and contradictory shard
sets, duplicate members and tensors, invalid lengths and offsets, truncated
headers and forbidden payload reads. These checks validate metadata planning;
they do not execute inference or measure native performance.

Affine SafeTensors coverage includes packed weights and their scale/bias
companions. Released-header parity coverage uses
`LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf`. The portable test supplies identical mock
mechanism facts to both inspection paths and compares complete requests and
estimates at the same three position counts. It reads only the checkpoint headers.
Run it against the pinned local artifact with:

```sh
EREDU_METADATA_GGUF=/path/to/LFM2.5-1.2B-Instruct-BF16.gguf \
  cargo test -p eredu --no-default-features --test metadata_forecast \
  released_gguf_metadata_forecast_parity -- --ignored
```
