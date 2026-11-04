# eredu-recommendation

Revision-pinned checkpoint discovery, benchmark attribution, metadata-only
Eredu assessment, and offline recommendation catalogues.

The crate owns recommendation policy and ordinary serializable Rust types.
It consumes Eredu's inspection and forecasting APIs for model-family support,
execution selection and memory accounting. It never loads checkpoint payloads,
opens a device, or implements model-family memory formulas.

## Features

| Feature | Surface |
| --- | --- |
| Default, with no features | Catalogue schemas, validation, offline budget reassessment, ranking, lineage and suitability policy, facade forecast integration |
| `hub` | Hugging Face discovery, pinned metadata transport, variant enumeration and Artificial Analysis benchmark ingestion |
| `builder` | Catalogue construction and supervised inspection workers; includes `hub` |
| `cli` | Command-line tools; includes `builder` |
| `mlx` | Explicit facade adapter selection for live metadata inspection |
| `metal`, `cuda` | Forward the corresponding facade configuration; select `mlx` separately |
| `image`, `audio` | Forward portable media capability configuration |

No native adapter is enabled by default. `forecast::from_inspection` consumes
an existing architecture selection built with caller-supplied mechanism facts.
`forecast::inspect` resolves the explicitly selected facade backend and returns
its typed error when that adapter is unavailable. Offline catalogue assessment
works without compiling the adapter used to produce the catalogue.

## Offline assessment

Applications supply a catalogue, the exact engine build identity they accept,
and fresh budgets for each physical memory domain:

```rust,no_run
use eredu_recommendation::{catalogue::Catalogue, forecast::DomainBudget, RecommendationPolicy};
use eredu::api::{MemoryBudget, MemoryDomain};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let catalogue: Catalogue = serde_json::from_reader(std::fs::File::open("models.json")?)?;
let budgets = vec![DomainBudget {
    domain: MemoryDomain::Unified,
    budget: MemoryBudget {
        available_bytes: Some(12 * 1024 * 1024 * 1024),
        application_limit_bytes: None,
        reserve_bytes: 512 * 1024 * 1024,
    },
}];
let candidates = catalogue.assess("accepted-engine-build-identity", &budgets)?;
let policy = RecommendationPolicy::default();
for candidate in policy.rank(&candidates) {
    println!("{} @ {}: {}", candidate.repo_id, candidate.revision, candidate.variant.variant_id);
}
# Ok(())
# }
```

The engine identity is supplied by the application. It identifies the exact
runtime build, feature set and forecasting assumptions that the application
accepts; a Git revision alone does not distinguish different compiled features.
A mismatch rejects the catalogue. Schema validation also checks immutable
repository revisions, safe unique files, benchmark attribution, inherited
checkpoint identity evidence, and the forecast workload profile.

Each entry retains the complete `GenerationForecast`, including its physical
domains, assumptions, uncertainty, speculative plans and refinement plans.
Assessment clears old capacity observations, applies the supplied budgets, and
uses the facade's forecast recomputation. An omitted domain has no supplied
budget. Unknown availability stays unknown; zero available bytes is a real
zero-capacity observation. Host and device capacities are never added together.
Changing the workload or preparation policy requires new inspection evidence.

The default recommendation policy requires compatible metadata, likely memory
fit, conversational suitability and measured benchmark evidence. Applications
can relax the benchmark or conversational requirement explicitly. Compatibility
and likely fit remain mandatory. Eligible candidates sort by benchmark order,
then encoding precision preference, with deterministic identity tie-breakers.
All candidates passed to one ranking operation use the same benchmark index
and comparable workload profiles. Precision preference is a filename-derived
encoding heuristic, not a measured quality score. Neither file size nor model
popularity proves fit, compatibility, tool support or conversational quality.

## Remote discovery and direct assessment

With `hub`, credentials and disk-cache location are explicit:

```rust,no_run
# #[cfg(feature = "hub")]
# async fn example() -> anyhow::Result<()> {
use eredu_recommendation::transport::Hub;
let hub = Hub::new(None, None)?;
let repositories = hub.search("SmolLM", 10).await?;
let (repository, manifest) = hub.repository(&repositories[0], None).await?;
for variant in &repository.variants {
    let metadata = hub.metadata(&repository, variant, &manifest).await?;
    // Pass successful metadata to forecast::inspect, or to a supervised worker.
    println!("{}: metadata fetched: {}", variant.variant_id, metadata.is_ok());
}
# Ok(())
# }
```

A moving reference is resolved once; discovery then reads an immutable manifest.
Model sizes, shard grouping and encoding preferences are discovery hints.
Compatibility comes from the explicit Eredu backend inspection. Only complete
GGUF shard groups are enumerated. A single unambiguous media projector in the
same directory is retained; ambiguous companions require an explicit artifact
selection before inspection. SafeTensors shard completeness is validated by
Eredu using the pinned index and headers.

HTTP responses and container headers have byte limits, range responses must
match the requested interval, and pagination stays on the Hub origin. Retry
logic honors rate limits. Public disk-cache entries expire after 24 hours;
authenticated requests do not use the public disk cache. Process-local metadata caching has a 10-minute TTL. Incompatible artifacts and unresolved
metadata are separate outcomes.

## Benchmark evidence

`Quality` records provider, evaluation identity, evaluated checkpoint, source
URL, optional score and order within a single benchmark index. Applications can
construct `quality::BenchmarkIndex` directly or deserialize one from JSON.
The index validates direct checkpoint attribution, a common provider and unique
ranks. A derived checkpoint inherits evidence only through an unambiguous
quantization relationship, compatible modification/dataset declarations, and
matching architecture and dimension evidence. Such evidence is conservative
metadata attribution, not a cryptographic proof that weights are unchanged.

`BenchmarkIndex::artificial_analysis(api_key)` fetches measured Artificial
Analysis configurations and retains their evaluation URLs and configuration
names. It excludes estimates, unverified measurement status, and ambiguous checkpoint mappings.
The exact model configuration must match the API score. Pages without an explicit
estimate flag qualify only when they publish a complete set of finite, uniquely
identified per-evaluation results for the current methodology; a missing flag alone
never establishes measurement. The adapter
also reads public methodology and model pages; their structure and mapping
availability are external dependencies. API credentials are arguments rather
than implicit environment lookups. Applications retain provider attribution
when presenting the results and are responsible for applicable source terms.
The crate contains no bundled benchmark dataset or API credential.

## Catalogue CLI

Build the portable tools:

```sh
cargo build -p eredu-recommendation --features cli
cargo run -p eredu-recommendation --features cli -- search SmolLM --limit 10
```

Live inspection and building require the selected adapter. For MLX on Apple
silicon, build with `--features cli,mlx,metal`. Metadata inspection uses cold
capability facts; it does not create a Metal device.

A builder profile is explicit JSON. For example:

```json
{
  "backend": "mlx",
  "inputTokens": 2048,
  "outputTokens": 512,
  "batchSize": 1,
  "prefillChunkTokens": 256,
  "placement": {"kind": "unified"},
  "allocatorCacheBytes": 268435456
}
```

The builder profile describes text-only, cold, fully resident preparation with
Eredu's default forecast calibration. The direct library API accepts arbitrary
facade inspection options, normalized preparation requests, memory options and
calibration. Catalogues retain their actual descriptive forecasts.

```sh
target/debug/eredu-recommendation inspect owner/model \
  --profile profile.json --engine-revision exact-engine-build

target/debug/eredu-recommendation build \
  --index benchmark-index.json --profile profile.json \
  --engine-revision exact-engine-build --output models.json --report report.json

target/debug/eredu-recommendation refresh models.json \
  --profile profile.json --engine-revision new-exact-engine-build --report report.json

target/debug/eredu-recommendation check models.json --engine-revision exact-engine-build

target/debug/eredu-recommendation rank models.json \
  --engine-revision exact-engine-build --budgets budgets.json
```

`build` also accepts `--aa-key-file` instead of `--index`. Network commands
accept global `--hf-token-file` and `--cache-dir` arguments. Credentials never
appear in catalogue output. `check` and `rank` do not initialize HTTP transport
or resolve credentials.

`budgets.json` is an array, for example:

```json
[{
  "domain": {"kind": "unified"},
  "budget": {
    "application_limit_bytes": null,
    "available_bytes": 12884901888,
    "reserve_bytes": 536870912
  }
}]
```

Catalogue building uses an explicit worker executable. The CLI selects its own
worker implementation; library callers supply a path implementing the same
protocol. Worker input is supplied through temporary files, never command-line
payloads. The supervisor samples resident memory and output sizes every 25 ms
and enforces elapsed-time limits. These are sampled process bounds, not hard
operating-system allocation limits. Defaults are 60 seconds, 2 GiB resident
memory, and 128 MiB per I/O stream; CLI flags configure time and memory limits.
Unavailable memory monitoring, resource exhaustion and backend errors remain
unverified outcomes. They never become claims of model incompatibility.

Build reports preserve exclusions and their compatibility state. Repository-level
lineage and metadata rejections exclude that checkpoint without interrupting
assessment of other repositories. Source failures
abort publication rather than silently reducing coverage. Refresh retains
immutable artifact revisions and benchmark evidence while re-inspecting the
entries for the requested engine/profile. Catalogue and report files each use
atomic replacement, and an empty result never replaces an existing catalogue.

## Verification

Portable boundary and public APIs:

```sh
cargo check -p eredu-recommendation --no-default-features
cargo clippy -p eredu-recommendation --features cli --all-targets --no-deps -- -D warnings
cargo test -p eredu-recommendation --doc --features hub
```

Native feature checks compile the selected adapter separately. Recommendation
and memory estimates are advisory; successful metadata inspection does not grant
payload-loading authority or establish numerical inference parity.
