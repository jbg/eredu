use std::process::Command;

fn rust_sources(root: &std::path::Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(root).expect("source directory must be readable") {
        let path = entry.expect("source entry must be readable").path();
        if path.is_dir() {
            rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn portable_facade_graph_has_no_accelerator_runtime() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("cargo tree must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8(output.stdout).unwrap();
    for forbidden in [
        "safemlx ",
        "safemlx-sys ",
        "windows-sys ",
        "image ",
        "rustfft ",
    ] {
        assert!(
            !tree.lines().any(|line| line.starts_with(forbidden)),
            "forbidden dependency {forbidden:?} in:\n{tree}"
        );
    }
}

#[test]
fn shared_architectures_do_not_depend_on_accelerator_runtimes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let architecture = root.join("eredu-architectures");
    let manifest = std::fs::read_to_string(architecture.join("Cargo.toml"))
        .expect("architecture manifest must be readable");
    assert!(!manifest.contains("safemlx"));

    let mut sources = Vec::new();
    rust_sources(&architecture.join("src"), &mut sources);
    for source in sources {
        let text = std::fs::read_to_string(&source).expect("architecture source must be readable");
        for forbidden in ["safemlx", "backend::mlx", "integrations::"] {
            assert!(
                !text.contains(forbidden),
                "backend dependency {forbidden:?} leaked into {source:?}"
            );
        }
    }
}

#[test]
fn multimodal_decoder_cutover_has_neutral_ownership_and_no_legacy_trees() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_root.parent().expect("workspace root");
    let families = ["gemma4", "inkling", "muse_glimmer"];

    for family in families {
        let legacy = crate_root
            .join("src/composition/mlx_architectures")
            .join(family);
        assert!(
            !legacy.exists(),
            "legacy multimodal architecture implementation remains at {legacy:?}"
        );
        let neutral = workspace.join("eredu-architectures/src").join(family);
        assert!(
            neutral.is_dir(),
            "neutral multimodal family source is missing at {neutral:?}"
        );
        let mut sources = Vec::new();
        rust_sources(&neutral, &mut sources);
        for source in sources {
            let text = std::fs::read_to_string(&source).expect("neutral source must be readable");
            for forbidden in ["safemlx", "backend::mlx", "MlxBackend"] {
                assert!(
                    !text.contains(forbidden),
                    "backend dependency {forbidden:?} leaked into {source:?}"
                );
            }
            for other in families {
                if other != family {
                    assert!(
                        !text.contains(&format!("{other}::")),
                        "cross-family dependency {other:?} leaked into {source:?}"
                    );
                }
            }
            for qwen_dependency in ["qwen::", "qwen3_vl", "mlx_architectures::"] {
                assert!(
                    !text.contains(qwen_dependency),
                    "architecture implementation dependency {qwen_dependency:?} leaked into {source:?}"
                );
            }
        }
    }

    for general_root in [
        crate_root.join("src/backend"),
        workspace.join("eredu-nn/src"),
        workspace.join("eredu-runtime/src"),
        workspace.join("eredu-checkpoint/src"),
    ] {
        let mut sources = Vec::new();
        rust_sources(&general_root, &mut sources);
        for source in sources {
            let text = std::fs::read_to_string(&source)
                .expect("general implementation source must be readable")
                .to_ascii_lowercase();
            for forbidden in ["gemma", "inkling", "muse_glimmer", "muse-glimmer"] {
                assert!(
                    !text.contains(forbidden),
                    "family identifier {forbidden:?} leaked into general layer {source:?}"
                );
            }
        }
    }

    let pipeline = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/pipeline.rs"),
    )
    .expect("pipeline source must be readable");
    for required in [
        "struct NeutralGemma4Stage",
        "struct NeutralInklingStage",
        "struct MuseGlimmerStage",
    ] {
        assert!(
            pipeline.contains(required),
            "neutral pipeline stage is missing: {required}"
        );
    }
    for legacy in [
        "struct GemmaStage",
        "Gemma4LayerwiseAdapter",
        "InklingLayerwiseAdapter",
        "MuseGlimmerLayerwiseAdapter",
    ] {
        assert!(
            !pipeline.contains(legacy),
            "legacy distributed implementation remains: {legacy}"
        );
    }

    let expert = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/expert.rs"),
    )
    .expect("distributed expert source must be readable");
    for legacy in [
        "ExpertArchitecture::Gemma4",
        "ExpertArchitecture::Inkling",
        "ExpertArchitecture::MuseGlimmer",
        "ExpertParallelCache::Gemma4",
        "ExpertParallelCache::Inkling",
        "ExpertParallelCache::MuseGlimmer",
    ] {
        assert!(
            !expert.contains(legacy),
            "legacy expert execution representation remains: {legacy}"
        );
    }
}

#[test]
fn hybrid_decoder_cutover_has_neutral_ownership_and_no_legacy_trees() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_root.parent().expect("workspace root");
    for family in ["kimi_linear", "lfm2", "nemotron_h"] {
        let legacy = crate_root
            .join("src/composition/mlx_architectures")
            .join(family);
        assert!(
            !legacy.exists(),
            "legacy hybrid architecture implementation remains at {legacy:?}"
        );
        let neutral = workspace.join("eredu-architectures/src").join(family);
        assert!(
            neutral.is_dir(),
            "neutral family source is missing at {neutral:?}"
        );
        let mut sources = Vec::new();
        rust_sources(&neutral, &mut sources);
        for source in sources {
            let text = std::fs::read_to_string(&source).expect("neutral source must be readable");
            for forbidden in ["safemlx", "backend::mlx", "MlxBackend"] {
                assert!(
                    !text.contains(forbidden),
                    "backend dependency {forbidden:?} leaked into {source:?}"
                );
            }
            for other in ["kimi_linear", "lfm2", "nemotron_h"] {
                if other != family {
                    assert!(
                        !text.contains(&format!("{other}::")),
                        "cross-family dependency {other:?} leaked into {source:?}"
                    );
                }
            }
        }
        let module = std::fs::read_to_string(neutral.join("mod.rs"))
            .expect("neutral family module must be readable");
        assert!(
            module.contains("hybrid_decoder::HybridDecoder")
                || (family == "nemotron_h"
                    && std::fs::read_to_string(neutral.join("model.rs"))
                        .expect("Nemotron model source must be readable")
                        .contains("hybrid_decoder::HybridDecoder")),
            "{family} does not consume the shared hybrid decoder assembly"
        );
    }

    for general_root in [
        crate_root.join("src/backend"),
        workspace.join("eredu-nn/src"),
        workspace.join("eredu-runtime/src"),
        workspace.join("eredu-checkpoint/src"),
    ] {
        let mut sources = Vec::new();
        rust_sources(&general_root, &mut sources);
        for source in sources {
            let text = std::fs::read_to_string(&source)
                .expect("general implementation source must be readable")
                .to_ascii_lowercase();
            for forbidden in ["kimi", "lfm2", "nemotron"] {
                assert!(
                    !text.contains(forbidden),
                    "family identifier {forbidden:?} leaked into general layer {source:?}"
                );
            }
        }
    }

    let pipeline = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/pipeline.rs"),
    )
    .expect("pipeline source must be readable");
    assert!(pipeline.contains("struct NeutralHybridPipelineStage<"));
    for legacy_wrapper in [
        "struct KimiLinearStage",
        "struct Lfm2Stage",
        "struct NemotronHStage",
    ] {
        assert!(
            !pipeline.contains(legacy_wrapper),
            "family pipeline stage wrapper remains: {legacy_wrapper}"
        );
    }

    let model = std::fs::read_to_string(crate_root.join("src/composition/mlx/model.rs"))
        .expect("MLX model facade must be readable");
    let expert = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/expert.rs"),
    )
    .expect("distributed expert source must be readable");
    for legacy_cache in [
        "ModelCache::KimiLinear",
        "ModelCache::Lfm2",
        "ModelCache::NemotronH",
        "ExpertParallelCache::KimiLinear",
        "ExpertParallelCache::Lfm2",
        "ExpertParallelCache::NemotronH",
    ] {
        assert!(
            !model.contains(legacy_cache) && !expert.contains(legacy_cache),
            "family-specific heterogeneous cache variant remains: {legacy_cache}"
        );
    }
    assert!(expert.contains("NeutralHybrid(Box<dyn NeutralHybridExpertArchitecture>)"));
    for legacy_architecture in [
        "ExpertArchitecture::KimiLinear",
        "ExpertArchitecture::Lfm2",
        "ExpertArchitecture::NemotronH",
    ] {
        assert!(
            !expert.contains(legacy_architecture),
            "family-specific distributed execution match remains: {legacy_architecture}"
        );
    }
}

#[test]
fn deepseek_cutover_has_one_neutral_source_of_model_semantics() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_root.parent().expect("workspace root");

    for deleted in [
        "eredu/src/composition/mlx_architectures/deepseek_v3",
        "eredu/src/composition/mlx_architectures/deepseek_v4",
    ] {
        assert!(
            !workspace.join(deleted).exists(),
            "legacy DeepSeek implementation still exists at {deleted}"
        );
    }

    let mut neutral_sources = Vec::new();
    rust_sources(
        &workspace.join("eredu-architectures/src/deepseek"),
        &mut neutral_sources,
    );
    for source in neutral_sources {
        let text = std::fs::read_to_string(&source).expect("DeepSeek source must be readable");
        for forbidden in [
            "safemlx",
            "backend::mlx",
            "MlxBackend",
            "CompressedLatentCache",
        ] {
            assert!(
                !text.contains(forbidden),
                "backend dependency {forbidden:?} leaked into {source:?}"
            );
        }
    }

    let mut backend_sources = Vec::new();
    rust_sources(&crate_root.join("src/backend/mlx"), &mut backend_sources);
    for source in backend_sources {
        let text = std::fs::read_to_string(&source).expect("MLX source must be readable");
        assert!(
            !text.to_ascii_lowercase().contains("deepseek"),
            "DeepSeek policy leaked into backend implementation {source:?}"
        );
    }

    let mut facade_sources = Vec::new();
    rust_sources(&crate_root.join("src"), &mut facade_sources);
    for source in facade_sources {
        let text = std::fs::read_to_string(&source).expect("facade source must be readable");
        for forbidden in [
            concat!("DeepSeekV3", "LayerwiseAdapter"),
            concat!("DeepSeekV4", "LayerwiseAdapter"),
            concat!("mlx_architectures::", "deepseek_v3"),
            concat!("mlx_architectures::", "deepseek_v4"),
            concat!("load_deepseek_", "pipeline"),
            concat!("load_deepseek_v4_", "pipeline"),
        ] {
            assert!(
                !text.contains(forbidden),
                "legacy DeepSeek dependency {forbidden:?} remains in {source:?}"
            );
        }
    }

    let architecture_root = crate_root.join("src/composition/mlx_architectures");
    let mut architecture_sources = Vec::new();
    rust_sources(&architecture_root, &mut architecture_sources);
    for source in architecture_sources {
        if source
            .strip_prefix(&architecture_root)
            .is_ok_and(|relative| {
                relative
                    .components()
                    .next()
                    .is_some_and(|component| component.as_os_str() == "qwen")
            })
        {
            continue;
        }
        let text = std::fs::read_to_string(&source).expect("architecture source must be readable");
        for forbidden in ["QwenLinear", "QwenWeightFormat", "qwen::hybrid::qwen3_5"] {
            assert!(
                !text.contains(forbidden),
                "Qwen-hybrid linear dependency {forbidden:?} leaked into {source:?}"
            );
        }
    }
}

#[test]
fn mlx_neural_operator_binding_is_architecture_agnostic() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(root.join("src/backend/mlx/nn/shared.rs"))
        .expect("MLX neural operator binding must be readable");
    for forbidden in ["eredu_architectures", "ModelArgs", "llama::", "Llama"] {
        assert!(
            !source.contains(forbidden),
            "architecture dependency {forbidden:?} leaked into MLX operators"
        );
    }
}

#[test]
fn mlx_backend_contains_no_llama_knowledge() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let backend = root.join("src/backend/mlx");
    let mut sources = Vec::new();
    rust_sources(&backend, &mut sources);
    for source in sources {
        let text = std::fs::read_to_string(&source).expect("MLX source must be readable");
        for forbidden in ["llama", "Llama", "ModelArgs", "eredu_architectures::llama"] {
            assert!(
                !text.contains(forbidden),
                "Llama dependency {forbidden:?} leaked into MLX backend source {source:?}"
            );
        }
    }
}

#[test]
fn qwen_neutral_cutover_has_no_legacy_or_backend_model_knowledge() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = crate_root.parent().expect("workspace root");
    let deleted = "eredu/src/composition/mlx_architectures/qwen";
    assert!(
        !workspace.join(deleted).exists(),
        "legacy Qwen implementation still exists at {deleted}"
    );
    let composition_modules =
        std::fs::read_to_string(workspace.join("eredu/src/composition/mod.rs"))
            .expect("composition module must be readable");
    for split in [
        "mod qwen_hybrid;",
        "mod qwen_vl;",
        "mod qwen_processor;",
        "mod qwen_expert;",
    ] {
        assert!(
            !composition_modules.contains(split),
            "Qwen composition retained a split top-level loader {split:?}"
        );
    }
    let qwen_binder = std::fs::read_to_string(workspace.join("eredu/src/composition/qwen.rs"))
        .expect("Qwen composition binder must be readable");
    for component in ["mod hybrid", "mod vl", "mod expert"] {
        assert!(
            qwen_binder.contains(component),
            "Qwen binder is missing consolidated component {component:?}"
        );
    }

    let neutral_qwen = workspace.join("eredu-architectures/src/qwen");
    assert!(
        neutral_qwen.is_dir(),
        "neutral Qwen architecture is missing"
    );
    let mut neutral_sources = Vec::new();
    rust_sources(&neutral_qwen, &mut neutral_sources);
    assert!(
        !neutral_sources.is_empty(),
        "neutral Qwen architecture is empty"
    );
    for source in neutral_sources {
        let text = std::fs::read_to_string(&source).expect("neutral source must be readable");
        for forbidden in ["safemlx", "backend::mlx", "MlxBackend"] {
            assert!(
                !text.contains(forbidden),
                "backend dependency {forbidden:?} leaked into neutral Qwen source {source:?}"
            );
        }
    }

    let mut sources = Vec::new();
    rust_sources(&workspace.join("eredu/src"), &mut sources);
    for source in sources {
        let text = std::fs::read_to_string(&source).expect("Rust source must be readable");
        for forbidden in [
            "mlx_architectures::qwen",
            "QwenHybridLayerwiseAdapter",
            "Qwen3VlLayerwiseAdapter",
            "struct QwenHybridStage",
            "struct Qwen3VlStage",
        ] {
            assert!(
                !text.contains(forbidden),
                "legacy Qwen dependency {forbidden:?} remains in {source:?}"
            );
        }
    }

    for relative in ["eredu-nn/src", "eredu-runtime/src", "eredu-checkpoint/src"] {
        let mut shared_sources = Vec::new();
        rust_sources(&workspace.join(relative), &mut shared_sources);
        for source in shared_sources {
            let text = std::fs::read_to_string(&source).expect("shared source must be readable");
            assert!(
                !text.to_ascii_lowercase().contains("qwen"),
                "Qwen-specific knowledge leaked into shared source {source:?}"
            );
        }
    }

    for relative in ["src/backend/mlx"] {
        let mut backend_sources = Vec::new();
        rust_sources(&crate_root.join(relative), &mut backend_sources);
        for source in backend_sources {
            let text = std::fs::read_to_string(&source).expect("MLX source must be readable");
            assert!(
                !text.to_ascii_lowercase().contains("qwen"),
                "Qwen model semantics leaked into backend implementation {source:?}"
            );
        }
    }

    let pipeline = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/pipeline.rs"),
    )
    .expect("distributed pipeline source must be readable");
    assert!(!pipeline.contains("struct QwenStage"));
    assert!(pipeline.contains("type QwenStage = NeutralDecoderStage"));
    assert!(pipeline.contains("type NeutralQwenHybridStage = NeutralHybridPipelineStage"));
    assert!(pipeline.contains("struct NeutralQwenVlStage"));
    assert!(pipeline.contains("struct NeutralQwenConditionalStage"));
    for forbidden in ["std::any::Any", "dyn Any", "downcast_ref", "downcast_mut"] {
        assert!(
            !pipeline.contains(forbidden),
            "pipeline ingress retained type-erased state through {forbidden:?}"
        );
    }
    assert!(
        !pipeline.contains("distributed::all_sum(&forwarded"),
        "composition retained a Qwen row-parallel reduction"
    );

    for relative in [
        "src/composition/qwen.rs",
        "src/composition/qwen_hybrid.rs",
        "src/composition/qwen_vl.rs",
        "src/composition/mlx_architectures/distributed/pipeline.rs",
    ] {
        let source = std::fs::read_to_string(crate_root.join(relative))
            .expect("Qwen composition source must be readable");
        assert!(
            !source.contains(".router.route("),
            "Qwen composition bypassed the routed-expert provider in {relative}"
        );
    }

    let expert_composition =
        std::fs::read_to_string(crate_root.join("src/composition/qwen_expert.rs"))
            .expect("Qwen expert composition source must be readable");
    for removed_provider in [
        "struct CachedQwenExpertProvider",
        "struct QwenExpertExecutorProvider",
        "struct QwenResidentExpertExecutorProvider",
    ] {
        assert!(
            !expert_composition.contains(removed_provider),
            "Qwen composition still owns backend provider {removed_provider:?}"
        );
    }
    assert!(expert_composition.contains("CachedSwiGluExpertProvider"));

    let distributed_expert = std::fs::read_to_string(
        crate_root.join("src/composition/mlx_architectures/distributed/expert.rs"),
    )
    .expect("distributed expert source must be readable");
    for legacy in [
        "QwenHybridLayerwise",
        "Qwen3VlLayerwise",
        "execute_cached_qwen_hybrid",
        "mlx_architectures::qwen",
    ] {
        assert!(
            !distributed_expert.contains(legacy),
            "legacy Qwen expert branch {legacy:?} remains"
        );
    }

    let inspection =
        std::fs::read_to_string(crate_root.join("src/backend/mlx/runtime/execution/inspection.rs"))
            .expect("MLX inspection utilities must be readable");
    for duplicate in [
        "pub trait ActivationObserver",
        "pub struct MoeRoutingObservation",
    ] {
        assert!(
            !inspection.contains(duplicate),
            "MLX redefined neutral inspection contract {duplicate:?}"
        );
    }
    assert!(inspection.contains("pub use eredu_runtime::{NoopObserver, RoutingObservation"));
}

#[test]
fn moshi_composition_uses_the_neutral_layered_runtime() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source =
        std::fs::read_to_string(root.join("src/composition/mlx_architectures/moshi/layerwise.rs"))
            .expect("Moshi layered composition must be readable");
    for required in [
        "impl LayeredArchitecture<MlxBackend, MoshiCache> for MoshiArchitecture",
        "impl ParallelLayeredArchitecture<MlxBackend, MoshiCache> for MoshiArchitecture",
        "prepare_layerwise_policy_with_bindings",
    ] {
        assert!(
            source.contains(required),
            "Moshi composition is missing neutral runtime contract {required:?}"
        );
    }
    for legacy in [
        "impl ArchitectureAdapter",
        "LayerwiseModel<Moshi",
        "load_layerwise_model_with_quantization",
        "load_tensor_parallel_layerwise_model(",
    ] {
        assert!(
            !source.contains(legacy),
            "Moshi composition still depends on legacy execution contract {legacy:?}"
        );
    }
}

#[test]
fn mlx_sampling_contains_only_backend_primitives() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let sampler =
        std::fs::read_to_string(root.join("src/backend/mlx/runtime/generation/sampler.rs"))
            .expect("MLX sampling facade must be readable");
    for runtime_owned in [
        "pub struct DefaultSampler",
        "pub struct GenerationSampler",
        "pub struct MirostatV2Sampler",
        "pub struct ConstrainedSampler",
    ] {
        assert!(
            !sampler.contains(runtime_owned),
            "backend reimplemented runtime sampling policy {runtime_owned:?}"
        );
    }

    let backend =
        std::fs::read_to_string(root.join("src/backend/mlx/runtime/generation/backend.rs"))
            .expect("MLX sampling capabilities must be readable");
    assert!(backend.contains("impl SamplingBackend for MlxSamplingBackend"));
}

#[test]
fn cache_execution_algorithms_are_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let core_cache = workspace.join("eredu-core/src/cache");
    for removed in ["executor.rs", "lifecycle.rs", "storage.rs"] {
        assert!(
            !core_cache.join(removed).exists(),
            "cache execution algorithm {removed} remains in eredu-core"
        );
    }

    let core = std::fs::read_to_string(workspace.join("eredu-core/src/cache.rs"))
        .expect("core cache schema module must be readable");
    for runtime_owned in [
        "CacheResidencyPool",
        "CacheBlockLifecycle",
        "CacheBlockStorage",
        "CacheIoExecutionState",
    ] {
        assert!(
            !core.contains(runtime_owned),
            "runtime cache implementation {runtime_owned} leaked into eredu-core"
        );
    }

    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/cache.rs"))
        .expect("runtime cache module must be readable");
    assert!(runtime.contains("pub struct CacheResidencyPool"));
    assert!(runtime.contains("mod executor;"));
    assert!(runtime.contains("mod lifecycle;"));
    assert!(runtime.contains("mod persistence;"));
    assert!(runtime.contains("mod storage;"));
    assert!(runtime.contains("mod telemetry;"));
    assert!(runtime.contains("mod worker;"));

    let worker = std::fs::read_to_string(workspace.join("eredu-runtime/src/cache/worker.rs"))
        .expect("runtime cache worker must be readable");
    for runtime_owned in [
        "pub struct CacheIoWorker",
        "pub struct CacheIoSubmission",
        "pub struct CacheIoTicket",
    ] {
        assert!(
            worker.contains(runtime_owned),
            "runtime does not own cache worker primitive {runtime_owned}"
        );
    }

    let telemetry = std::fs::read_to_string(workspace.join("eredu-runtime/src/cache/telemetry.rs"))
        .expect("runtime cache telemetry must be readable");
    for runtime_owned in [
        "pub struct CacheLayerResidencyStats",
        "pub struct CacheLayerResidencyReport",
        "pub struct CacheResidencyReport",
        "pub struct CacheResidencyTelemetry",
    ] {
        assert!(
            telemetry.contains(runtime_owned),
            "runtime does not own cache telemetry schema {runtime_owned}"
        );
    }

    let persistence =
        std::fs::read_to_string(workspace.join("eredu-runtime/src/cache/persistence.rs"))
            .expect("runtime prompt-cache persistence must be readable");
    for runtime_owned in [
        "pub struct PromptCachePublication",
        "pub struct LiveCacheBlockPublication",
        "pub fn inspect_prompt_cache",
        "pub fn validate_prompt_cache_manifest",
        "pub fn hash_prompt_cache_shard_payload",
    ] {
        assert!(
            persistence.contains(runtime_owned),
            "runtime does not own prompt-cache algorithm {runtime_owned}"
        );
    }

    let mlx =
        std::fs::read_to_string(workspace.join("eredu/src/backend/mlx/runtime/cache/residency.rs"))
            .expect("MLX cache realization must be readable");
    for runtime_owned in [
        "pub enum CacheResidencyPolicy",
        "pub enum LiveCacheDiskPolicy",
        "pub struct PagedCacheOptions",
        "pub struct CacheLayerResidencyStats",
        "pub struct CacheLayerResidencyReport",
        "pub struct CacheResidencyReport",
        "pub struct CacheResidencyTelemetry",
        "fn inspect_prompt_cache",
        "fn validate_prompt_cache_manifest",
        "fn hash_prompt_cache_shard_payload",
        "fn publish_prompt_cache_generation",
        "fn live_block_paths",
        "fn publish_live_block_file",
        "struct TemporaryFileGuard",
        "enum DiskRequest",
        "struct DiskCompletion",
        "struct DiskWorkerShared",
    ] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX cache realization redefined neutral policy {runtime_owned}"
        );
    }
}

#[test]
fn immutable_weight_residency_policy_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/weight_residency.rs"))
        .expect("runtime weight-residency policy must be readable");
    for runtime_owned in [
        "pub struct LayerwiseLoadOptions",
        "pub struct DenseDiskStreamLoadOptions",
        "pub struct DenseTransferSchedule",
        "pub enum LayerWeightResidency",
        "pub struct ExpertCacheLoadOptions",
        "pub struct ExpertIdentity",
        "pub enum ExpertPass",
        "pub enum NonExpertWeightResidency",
        "pub enum ExpertWeightResidency",
        "pub enum WeightResidency",
        "pub enum ExecutionResidency",
        "pub struct LayerwiseModelMetadata",
        "pub struct StaticUnitBindings",
    ] {
        assert!(
            runtime.contains(runtime_owned),
            "runtime does not own immutable-weight policy {runtime_owned}"
        );
    }

    for relative in [
        "eredu/src/backend/mlx/runtime/execution/layerwise.rs",
        "eredu/src/backend/mlx/runtime/residency/dense_stream.rs",
        "eredu/src/backend/mlx/runtime/residency/expert_cache.rs",
    ] {
        let source = std::fs::read_to_string(workspace.join(relative))
            .expect("MLX weight-residency realization must be readable");
        for runtime_owned in [
            "pub struct LayerwiseLoadOptions",
            "pub struct DenseDiskStreamLoadOptions",
            "pending: VecDeque<usize>",
            "ready: VecDeque<DensePreparedTransfer>",
            "pub enum LayerWeightResidency",
            "pub struct ExpertCacheLoadOptions",
            "pub struct ExpertIdentity",
            "pub enum ExpertPass",
            "pub enum NonExpertWeightResidency",
            "pub enum ExpertWeightResidency",
            "pub enum WeightResidency",
            "pub enum ExecutionResidency",
            "pub struct LayerwiseModelMetadata",
            "pub struct StaticUnitBindings",
        ] {
            assert!(
                !source.contains(runtime_owned),
                "MLX redefined immutable-weight policy {runtime_owned} in {relative}"
            );
        }
    }
}

#[test]
fn dense_stream_telemetry_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/dense.rs"))
        .expect("runtime dense-stream telemetry must be readable");
    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/execution/layerwise.rs"),
    )
    .expect("MLX layerwise realization must be readable");
    for runtime_owned in [
        "pub struct DenseDiskStreamReport",
        "pub struct DenseCacheMetrics",
        "pub struct DenseTierResidencyReport",
        "pub struct DenseExecutionGroupReport",
        "pub struct DensePassReport",
        "pub struct DensePassCounterSnapshot",
        "pub struct DenseStreamTelemetry",
    ] {
        assert!(
            runtime.contains(runtime_owned),
            "runtime does not own dense-stream telemetry {runtime_owned}"
        );
        assert!(
            !mlx.contains(runtime_owned),
            "MLX redefined dense-stream telemetry {runtime_owned}"
        );
    }
    for mlx_owned_state in [
        "struct DensePassState",
        "struct DensePassActivity",
        "struct DenseExecutionGroupPlan",
        "struct DenseExecutionGroupState",
    ] {
        assert!(
            !mlx.contains(mlx_owned_state),
            "MLX retained dense-stream policy state {mlx_owned_state}"
        );
    }
}

#[test]
fn execution_group_orchestration_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/execution.rs"))
        .expect("runtime execution scheduler must be readable");
    assert!(runtime.contains("pub struct ExecutionGroupSchedule"));
    assert!(runtime.contains("remaining_consumers"));
    let layered = std::fs::read_to_string(workspace.join("eredu-runtime/src/layered.rs"))
        .expect("runtime layered execution must be readable");
    assert!(layered.contains("pub struct LayeredForwardState"));
    assert!(layered.contains("fn group_unit_count"));
    assert!(layered.contains("fn begin_execution_group"));
    assert!(layered.contains("ExecutionGroupSchedule::new"));
    assert!(layered.contains("B::fork_executors"));
    assert!(layered.contains("B::order_after"));
    assert!(layered.contains("B::submit"));

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/execution/layerwise.rs"),
    )
    .expect("MLX layerwise realization must be readable");
    for runtime_owned in [
        "ExecutionGroupReadySet",
        "let mut remaining_consumers",
        ".consumer_counts()",
        "ArchitectureCheckpointPlan",
        "pub struct LayerwiseForwardState",
    ] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX layerwise realization retained scheduler algorithm {runtime_owned}"
        );
    }
}

#[test]
fn parallel_model_metadata_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/parallel.rs"))
        .expect("runtime parallel policy must be readable");
    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/execution/layerwise.rs"),
    )
    .expect("MLX layerwise realization must be readable");
    assert!(runtime.contains("pub struct ParallelModelInfo"));
    assert!(!mlx.contains("pub struct ParallelModelInfo"));
}

#[test]
fn weight_prefetch_execution_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/prefetch.rs"))
        .expect("runtime background prefetch worker must be readable");
    assert!(runtime.contains("pub struct BackgroundPrefetchWorker"));
    for behavior in [
        "catch_unwind",
        "record_backpressure",
        "cancel_all",
        "worker.join",
    ] {
        assert!(
            runtime.contains(behavior),
            "runtime prefetch worker does not own {behavior}"
        );
    }

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/dense_stream.rs"),
    )
    .expect("MLX dense-stream realization must be readable");
    for runtime_owned in [
        "enum WorkerMessage",
        "fn worker_loop",
        "PrefetchExecutionState",
        "catch_unwind",
        "JoinHandle",
    ] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX dense-stream realization retained worker algorithm {runtime_owned}"
        );
    }
}

#[test]
fn weight_window_policy_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/residency.rs"))
        .expect("runtime residency controller must be readable");
    assert!(runtime.contains("pub fn commit_group_window"));
    assert!(runtime.contains("pub fn protect_group_window"));

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/manager.rs"),
    )
    .expect("MLX residency realization must be readable");
    for runtime_owned in [".set_group_window(", "let mut seen = BTreeSet::new()"] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX residency realization retained window policy {runtime_owned}"
        );
    }
}

#[test]
fn weight_acquisition_transitions_are_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/residency.rs"))
        .expect("runtime residency controller must be readable");
    for runtime_owned in [
        "pub struct ResidencyAcquisition",
        "pub fn plan_acquisition",
        "pub fn reserve_acquisition",
        "pub fn touch_acquisition_hits",
        "pub fn rollback_acquisition",
    ] {
        assert!(runtime.contains(runtime_owned));
    }

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/manager.rs"),
    )
    .expect("MLX residency realization must be readable");
    for runtime_owned in [".reserve_copies(", ".rollback_reserved("] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX residency realization retained acquisition transition {runtime_owned}"
        );
    }
}

#[test]
fn weight_publication_and_prefetch_policy_are_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/residency.rs"))
        .expect("runtime residency controller must be readable");
    assert!(runtime.contains("pub fn publish_acquisition_copy"));
    assert!(runtime.contains("pub fn begin_prefetch"));

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/manager.rs"),
    )
    .expect("MLX residency realization must be readable");
    for runtime_owned in [
        ".publish_reserved(",
        ".record_prefetch(",
        ".record_transfer(",
    ] {
        assert!(
            !mlx.contains(runtime_owned),
            "MLX residency realization retained publication policy {runtime_owned}"
        );
    }
}

#[test]
fn weight_lease_ownership_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/residency.rs"))
        .expect("runtime residency lease must be readable");
    for runtime_owned in [
        "pub trait ResidencyLeaseStorage",
        "pub trait ResidencyLeaseOwner",
        "pub struct ResidencyLease",
        "impl<S, O> Drop for ResidencyLease",
    ] {
        assert!(runtime.contains(runtime_owned));
    }

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/manager.rs"),
    )
    .expect("MLX residency realization must be readable");
    assert!(!mlx.contains("pub struct ResidentUnitLease"));
    assert!(!mlx.contains("impl Drop for ResidentUnitLease"));
}

#[test]
fn shared_checkpoint_source_is_checkpoint_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let checkpoint = std::fs::read_to_string(workspace.join("eredu-checkpoint/src/store.rs"))
        .expect("checkpoint source definitions must be readable");
    assert!(checkpoint.contains("pub type SharedCheckpointSource = Arc<dyn CheckpointSource>;"));

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/execution/layerwise.rs"),
    )
    .expect("MLX layerwise realization must be readable");
    assert!(!mlx.contains("pub type SharedCheckpointSource"));
}

#[test]
fn weight_transfer_ownership_is_runtime_owned() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let runtime = std::fs::read_to_string(workspace.join("eredu-runtime/src/residency.rs"))
        .expect("runtime residency transfer must be readable");
    for runtime_owned in [
        "pub trait ResidencyTransferOwner",
        "pub struct ResidencyTransfer",
        "impl<L, C, R, O> Drop for ResidencyTransfer",
    ] {
        assert!(runtime.contains(runtime_owned));
    }

    let mlx = std::fs::read_to_string(
        workspace.join("eredu/src/backend/mlx/runtime/residency/manager.rs"),
    )
    .expect("MLX residency realization must be readable");
    assert!(!mlx.contains("pub struct ResidentTransfer"));
    assert!(!mlx.contains("impl Drop for ResidentTransfer"));
    assert!(!mlx.contains("fn finish(&mut self, report_error"));
    assert!(!mlx.contains(".ledger_mut()\n            .resolve_transfer"));
}

#[test]
fn llama_hot_path_remains_statically_dispatched_and_device_native() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    for relative in [
        "eredu-runtime/src/layered.rs",
        "eredu-architectures/src/llama.rs",
    ] {
        let path = workspace.join(relative);
        let source = std::fs::read_to_string(&path).expect("hot-path source must be readable");
        for forbidden in [
            "Box<dyn",
            "&dyn Architecture",
            "dyn AttentionCache",
            ".to_bytes(",
            ".as_bytes(",
            ".synchronize(",
            "eval(",
        ] {
            assert!(
                !source.contains(forbidden),
                "hot path {path:?} contains forbidden operation {forbidden:?}"
            );
        }
    }

    let composition = std::fs::read_to_string(workspace.join("eredu/src/composition/llama.rs"))
        .expect("Llama composition source must be readable");
    for removed in [
        "ArchitectureAdapter",
        "LayerwiseModel<",
        "LlamaAdapterInput",
    ] {
        assert!(
            !composition.contains(removed),
            "Llama composition retains legacy runtime path {removed:?}"
        );
    }
}

#[test]
fn backend_and_example_dependencies_have_the_correct_manifest_ownership() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--manifest-path",
            manifest.to_str().unwrap(),
            "--no-deps",
            "--format-version",
            "1",
        ])
        .output()
        .expect("cargo metadata must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let package = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["name"] == env!("CARGO_PKG_NAME"))
        .expect("facade package must be present");
    let dependencies = package["dependencies"].as_array().unwrap();
    let mlx_features = package["features"]["mlx"].as_array().unwrap();

    for name in ["safetensors", "memmap2", "tempfile", "half", "libc"] {
        let dependency = dependencies
            .iter()
            .find(|dependency| dependency["name"] == name && dependency["kind"].is_null())
            .unwrap_or_else(|| panic!("missing normal dependency {name}"));
        assert_eq!(
            dependency["optional"], true,
            "backend dependency {name} must be optional"
        );
        assert!(
            mlx_features
                .iter()
                .any(|feature| feature == &format!("dep:{name}")),
            "MLX feature must enable backend dependency {name}"
        );
    }

    let runtime = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["name"] == "eredu-runtime")
        .expect("neutral runtime package must be present");
    let windows = runtime["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|dependency| dependency["name"] == "windows-sys")
        .expect("runtime persistence must own its Windows publication primitive");
    assert_eq!(windows["optional"], false);
    assert_eq!(windows["target"], "cfg(windows)");

    let features = package["features"].as_object().unwrap();
    for removed in ["media-processing", "image-processing", "audio-processing"] {
        assert!(
            !features.contains_key(removed),
            "backend-agnostic feature name {removed} must not hide MLX materialization"
        );
    }
    for (feature, required) in [
        ("mlx-media", &["mlx"][..]),
        ("mlx-image", &["mlx-media", "dep:image"][..]),
        ("mlx-audio", &["mlx-media", "dep:rustfft"][..]),
    ] {
        let enabled = features[feature]
            .as_array()
            .unwrap_or_else(|| panic!("missing feature {feature}"));
        for required in required {
            assert!(
                enabled.iter().any(|entry| entry == required),
                "feature {feature} must enable {required}"
            );
        }
    }

    for name in ["anyhow", "clap"] {
        assert!(
            dependencies
                .iter()
                .any(|dependency| dependency["name"] == name && dependency["kind"] == "dev"),
            "example dependency {name} must be development-only"
        );
        assert!(
            !dependencies
                .iter()
                .any(|dependency| dependency["name"] == name && dependency["kind"].is_null()),
            "example dependency {name} must not be a normal facade dependency"
        );
    }
}

#[test]
fn backend_implementation_does_not_import_facade_orchestration() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(
        !manifest.join("src/api/mlx.rs").exists(),
        "MLX speculative execution must not live in the facade source tree"
    );
    let backend = manifest.join("src/backend");
    let mut sources = Vec::new();
    rust_sources(&backend, &mut sources);
    let offenders = sources
        .into_iter()
        .filter(|path| {
            let source = std::fs::read_to_string(path).expect("backend source must be readable");
            let production = source
                .split_once("#[cfg(test)]\nmod tests")
                .map_or(source.as_str(), |(production, _)| production);
            ["crate::api", "crate::runtime"]
                .iter()
                .any(|forbidden| production.contains(forbidden))
        })
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "backend implementations must not depend upward on facade orchestration: {offenders:#?}"
    );
}

#[test]
fn facade_runtime_contains_only_backend_independent_orchestration() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime = manifest.join("src/runtime");
    for removed in [
        "attention",
        "cache",
        "checkpoint",
        "distributed",
        "execution",
        "media",
        "residency",
    ] {
        assert!(
            !runtime.join(removed).exists(),
            "backend implementation directory leaked into facade runtime: {removed}"
        );
    }
    assert!(
        !runtime.join("generation/sampler.rs").exists(),
        "MLX sampling must be owned by backend::mlx"
    );

    let mut sources = Vec::new();
    rust_sources(&runtime, &mut sources);
    let offenders = sources
        .into_iter()
        .filter(|path| {
            let source = std::fs::read_to_string(path).expect("runtime source must be readable");
            source.contains("safemlx::") || source.contains("safemlx_sys::")
        })
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "facade runtime must not name accelerator runtime types: {offenders:#?}"
    );
}

#[test]
fn crate_root_does_not_reexport_mlx_implementation_types() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let library = manifest.join("src/lib.rs");
    let source = std::fs::read_to_string(library).expect("crate root must be readable");
    assert!(source.contains("pub mod backend;"));
    assert!(!source.contains("pub use backend::mlx"));
    assert!(!source.contains("pub mod error;"));
    for removed in [
        "pub mod architectures;",
        "pub mod nn;",
        "ModelInputBuilder",
        "trait ModelInput",
        "trait ModelOutput",
    ] {
        assert!(
            !source.contains(removed),
            "crate root retains MLX implementation API {removed:?}"
        );
    }
    for removed in ["architectures", "nn"] {
        assert!(
            !manifest.join("src").join(removed).exists(),
            "MLX implementation tree remains at crate root: {removed}"
        );
    }
    assert!(
        manifest.join("src/backend/mlx/nn").is_dir(),
        "MLX backend does not own reusable neural capabilities"
    );
    assert!(
        manifest.join("src/composition/mlx_architectures").is_dir(),
        "high-level MLX model composition is missing"
    );
    assert!(
        !manifest.join("src/backend/mlx/architectures").exists(),
        "model-family composition leaked back into the MLX backend"
    );
}

#[test]
fn generic_loaded_model_source_does_not_import_backend_implementations() {
    let loaded = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/loaded.rs");
    let source = std::fs::read_to_string(loaded).expect("loaded-model source must be readable");
    for forbidden in [
        "safemlx::",
        "safemlx_sys::",
        "crate::backend::mlx",
        "crate::composition::mlx_architectures",
        "cfg(feature = \"mlx\")",
    ] {
        assert!(
            !source.contains(forbidden),
            "generic LoadedModel orchestration contains backend implementation reference {forbidden:?}"
        );
    }
}

#[test]
fn portable_api_tests_do_not_depend_on_backend_implementations() {
    let api = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api");
    let portable =
        std::fs::read_to_string(api.join("tests.rs")).expect("portable API tests must be readable");
    for forbidden in [
        "safemlx::",
        "safemlx_sys::",
        "crate::backend::mlx",
        "crate::composition::mlx_architectures",
        "crate::backend::mlx::nn",
    ] {
        assert!(
            !portable.contains(forbidden),
            "portable API tests contain backend implementation reference {forbidden:?}"
        );
    }
    assert!(
        api.join("tests/mlx.rs").is_file(),
        "MLX integration tests must have a dedicated module"
    );
}

#[test]
fn backend_conformance_suite_remains_backend_independent() {
    let suite =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend_conformance.rs");
    let source = std::fs::read_to_string(suite).expect("conformance suite must be readable");
    for forbidden in [
        "safemlx::",
        "safemlx_sys::",
        "backend::mlx",
        "cfg(feature = \"mlx\")",
        "tempfile",
    ] {
        assert!(
            !source.contains(forbidden),
            "backend conformance suite contains MLX implementation reference {forbidden:?}"
        );
    }
}
