fn run_ring_pipeline(dense_stream: bool, family: FixtureFamily) {
    run_ring_pipeline_mode(dense_stream, family, WorkerMode::Standard);
}

fn run_ring_cartesian_pipeline(dense_stream: bool, family: FixtureFamily, axes: &'static str) {
    run_ring_cartesian_pipeline_mode(dense_stream, family, axes, WorkerMode::Standard);
}

fn run_ring_cartesian_pipeline_mode(
    dense_stream: bool,
    family: FixtureFamily,
    axes: &'static str,
    mode: WorkerMode,
) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::GptOssGguf {
        let path = checkpoint.path().join("model.gguf");
        write_gpt_oss_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::Lfm2MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_lfm2_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::NemotronHGguf {
        let path = checkpoint.path().join("model.gguf");
        write_nemotron_h_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::Qwen3MoeTied => {
                write_qwen_fixture_with_tied_head(checkpoint.path(), "qwen3_moe", true)
            }
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekDsparkTarget => {
                write_deepseek_v4_dspark_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(
                checkpoint.path(),
                if matches!(
                    mode,
                    WorkerMode::OpaqueSessionPredictionFree
                        | WorkerMode::OpaqueSessionAddressableParameterBank
                ) {
                    0
                } else {
                    1
                },
            ),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::AddressableParameterBankRequantize | WorkerMode::Requantize
                ) =>
            {
                write_nemotron_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::NemotronH if mode == WorkerMode::OpaqueNemotronHMtp => {
                write_nemotron_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::AddressableParameterBankRequantize | WorkerMode::Requantize
                ) =>
            {
                write_inkling_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling
                if matches!(
                    mode,
                    WorkerMode::OpaqueInklingMtp
                        | WorkerMode::OpaqueInklingMtpAddressableParameterBank
                ) =>
            {
                write_inkling_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            _ => panic!("Cartesian pipeline helper received unsupported {family:?}"),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::from_dense_stream(dense_stream),
        family,
        mode,
        checkpoint,
        checkpoint_path,
        Some(axes),
    );
}

fn run_ring_layerwise_host_cartesian_pipeline_mode(
    family: FixtureFamily,
    axes: &'static str,
    mode: WorkerMode,
) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path(), 1),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            _ => panic!("host-layerwise Cartesian helper received unsupported {family:?}"),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::LayerwiseHost,
        family,
        mode,
        checkpoint,
        checkpoint_path,
        Some(axes),
    );
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WorkerMode {
    Standard,
    FinalOutputIntervention,
    AddressableParameterBank,
    AddressableParameterBankRequantize,
    Requantize,
    OpaqueSession,
    OpaqueSessionPredictionFree,
    OpaquePreparedSpeculativeCapability,
    OpaqueUnsupportedDirectPartition,
    OpaqueSessionRequantize,
    OpaqueInspection,
    OpaqueTextGeneration,
    OpaqueSessionAddressableParameterBank,
    OpaqueSessionEvictingAddressableParameterBank,
    OpaqueMuseImage,
    OpaqueInklingMedia,
    OpaqueInklingMtp,
    OpaqueInklingMtpAddressableParameterBank,
    OpaqueQwenHybridMtp,
    OpaqueNemotronHMtp,
    OpaqueDeepSeekMtpTarget,
    OpaqueDeepSeekDsparkTarget,
    OpaqueGemma4Media,
    OpaqueGemma4MediaInspection,
    OpaqueQwen3VlMedia,
    OpaqueQwenConditionalMedia,
    QwenHybridPromptCache,
    PromptCachePrepareFailure,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum WorkerResidency {
    FullyResident,
    LayerwiseHost,
    DenseDiskStream,
}

impl WorkerResidency {
    const fn from_dense_stream(enabled: bool) -> Self {
        if enabled {
            Self::DenseDiskStream
        } else {
            Self::FullyResident
        }
    }
}

fn run_ring_pipeline_mode(dense_stream: bool, family: FixtureFamily, mode: WorkerMode) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if family == FixtureFamily::DeepSeekGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_gguf_fixture(&path);
        path
    } else if matches!(family, FixtureFamily::Qwen2Gguf | FixtureFamily::Qwen3Gguf) {
        let path = checkpoint.path().join("model.gguf");
        write_qwen_gguf_fixture(
            &path,
            if family == FixtureFamily::Qwen2Gguf {
                "qwen2"
            } else {
                "qwen3"
            },
        );
        path
    } else if family == FixtureFamily::Qwen3MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_qwen3_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::GptOssGguf {
        let path = checkpoint.path().join("model.gguf");
        write_gpt_oss_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::Lfm2MoeGguf {
        let path = checkpoint.path().join("model.gguf");
        write_lfm2_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::NemotronHGguf {
        let path = checkpoint.path().join("model.gguf");
        write_nemotron_h_moe_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(&path);
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(&path);
        path
    } else {
        match family {
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Llama => write_fixture(checkpoint.path()),
            FixtureFamily::Mistral => write_mistral_fixture(checkpoint.path()),
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_prediction(checkpoint.path(), 2, 1)
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekDsparkTarget => {
                write_deepseek_v4_dspark_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path(), 1),
            FixtureFamily::Gemma => write_gemma_fixture(checkpoint.path()),
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Qwen3
                if matches!(
                    mode,
                    WorkerMode::Requantize | WorkerMode::OpaqueSessionRequantize
                ) =>
            {
                write_qwen_requantized_tp_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen3 => write_qwen_fixture(checkpoint.path(), "qwen3"),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::Qwen3MoeTied => {
                write_qwen_fixture_with_tied_head(checkpoint.path(), "qwen3_moe", true)
            }
            FixtureFamily::GptOss => write_gpt_oss_fixture(checkpoint.path()),
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen3NextMoe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_next")
            }
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
            FixtureFamily::Qwen35Moe => {
                write_qwen_hybrid_moe_fixture(checkpoint.path(), "qwen3_5_moe_text")
            }
            FixtureFamily::Qwen35Multimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), false)
            }
            FixtureFamily::Qwen35ZeroPrediction => {
                write_qwen35_zero_prediction_fixture(checkpoint.path())
            }
            FixtureFamily::Qwen35MoeMultimodal => {
                write_qwen35_multimodal_fixture(checkpoint.path(), true)
            }
            FixtureFamily::Qwen3Vl => write_qwen3_vl_fixture(checkpoint.path(), false),
            FixtureFamily::Qwen3VlMoe => write_qwen3_vl_fixture(checkpoint.path(), true),
            FixtureFamily::Inkling if mode == WorkerMode::OpaqueInklingMtp => {
                write_inkling_pipeline_mtp_fixture(checkpoint.path())
            }
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::MuseGlimmer => {
                write_muse_glimmer_tensor_parallel_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekGguf
            | FixtureFamily::Qwen2Gguf
            | FixtureFamily::Qwen3Gguf
            | FixtureFamily::Qwen3MoeGguf
            | FixtureFamily::GptOssGguf
            | FixtureFamily::Lfm2MoeGguf
            | FixtureFamily::NemotronHGguf
            | FixtureFamily::KimiLinearGguf
            | FixtureFamily::InklingGguf => unreachable!(),
        }
        checkpoint.path().to_path_buf()
    };
    run_ring_pipeline_processes(
        WorkerResidency::from_dense_stream(dense_stream),
        family,
        mode,
        checkpoint,
        checkpoint_path,
        None,
    );
}

fn run_ring_pipeline_processes(
    residency: WorkerResidency,
    family: FixtureFamily,
    mode: WorkerMode,
    _checkpoint: tempfile::TempDir,
    checkpoint_path: PathBuf,
    cartesian_axes: Option<&'static str>,
) {
    let prompt_cache = tempfile::tempdir().unwrap();
    let world_size = match cartesian_axes {
        Some("tp-pp-ep") => 8,
        Some("tp" | "ep") => 2,
        Some(_) => 4,
        None => 2,
    };
    let sockets = (0..world_size)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(world_size),
    };
    for rank in 0..world_size {
        let mut command = Command::new(&executable);
        command
            .args([
                "--exact",
                "tests::distributed_pipeline_ring::pipeline_ring_worker",
                "--nocapture",
            ])
            .env(WORKER_RANK, rank.to_string())
            .env(CHECKPOINT_DIR, &checkpoint_path)
            .env(FIXTURE_FAMILY, family.name())
            .env(PROMPT_CACHE_ROOT, prompt_cache.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped());
        if family == FixtureFamily::Qwen3VlMoe && cartesian_axes == Some("tp-pp-ep") {
            command.env("EREDU_TEST_PARTITION_COLLECTIVE_TRACE", "1");
        }
        command.stderr(Stdio::piped());
        if let Some(axes) = cartesian_axes {
            command.env(CARTESIAN_AXES, axes);
        }
        match residency {
            WorkerResidency::FullyResident => {}
            WorkerResidency::LayerwiseHost => {
                command.env(LAYERWISE_HOST, "1");
            }
            WorkerResidency::DenseDiskStream => {
                command.env(DENSE_STREAM, "1");
            }
        }
        match mode {
            WorkerMode::Standard => {}
            WorkerMode::FinalOutputIntervention => {
                command.env(FINAL_OUTPUT_INTERVENTION, "1");
            }
            WorkerMode::AddressableParameterBank => {
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::AddressableParameterBankRequantize => {
                command.env(EXPERT_CACHE, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::Requantize => {
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueSession => {
                command.env(OPAQUE_SESSION, "1");
            }
            WorkerMode::OpaqueSessionPredictionFree => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PREDICTION_FREE_TARGET, "1");
            }
            WorkerMode::OpaquePreparedSpeculativeCapability => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PREPARED_SPECULATIVE_CAPABILITY, "1");
            }
            WorkerMode::OpaqueUnsupportedDirectPartition => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPECTED_UNSUPPORTED_DIRECT_PARTITION, "1");
            }
            WorkerMode::OpaqueSessionRequantize => {
                command.env(OPAQUE_SESSION, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INSPECTION, "1");
            }
            WorkerMode::OpaqueTextGeneration => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_TEXT_GENERATION, "1");
            }
            WorkerMode::OpaqueSessionAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::OpaqueSessionEvictingAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(EXPERT_CACHE, "1");
                command.env(EXPERT_CACHE_EVICTION, "1");
            }
            WorkerMode::OpaqueMuseImage => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_MUSE_IMAGE, "1");
            }
            WorkerMode::OpaqueInklingMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MEDIA, "1");
            }
            WorkerMode::OpaqueInklingMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
            }
            WorkerMode::OpaqueInklingMtpAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::OpaqueQwenHybridMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN_HYBRID_MTP, "1");
            }
            WorkerMode::OpaqueNemotronHMtp => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_NEMOTRON_H_MTP, "1");
            }
            WorkerMode::OpaqueDeepSeekMtpTarget => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
            }
            WorkerMode::OpaqueDeepSeekDsparkTarget => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
                command.env(OPAQUE_DEEPSEEK_DSPARK_TARGET, "1");
            }
            WorkerMode::OpaqueGemma4Media => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
            }
            WorkerMode::OpaqueGemma4MediaInspection => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_GEMMA4_MEDIA, "1");
                command.env(OPAQUE_INSPECTION, "1");
            }
            WorkerMode::OpaqueQwen3VlMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN3_VL_MEDIA, "1");
            }
            WorkerMode::OpaqueQwenConditionalMedia => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN_CONDITIONAL_MEDIA, "1");
            }
            WorkerMode::QwenHybridPromptCache => {
                command.env(QWEN_HYBRID_PROMPT_CACHE, "1");
            }
            WorkerMode::PromptCachePrepareFailure => {
                command.env(OPAQUE_SESSION, "1");
                command.env(PROMPT_CACHE_PREPARE_FAILURE, "1");
            }
        }
        children.children.push(command.spawn().unwrap());
    }
    let deadline = Instant::now()
        + Duration::from_secs(match world_size {
            8 => 180,
            4 => 90,
            _ => 45,
        });
    let mut timed_out = false;
    loop {
        let statuses = children
            .children
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        let peer_failed = statuses.iter().flatten().any(|status| !status.success());
        if timed_out || peer_failed {
            // A peer often reports the global failure agreement before the
            // originating worker has flushed its local architecture error.
            // Preserve that causal diagnostic without waiting for the full Ring
            // deadline or allowing a failed process set to run unbounded.
            if peer_failed && !timed_out {
                thread::sleep(Duration::from_millis(500));
            }
            for child in &mut children.children {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let outputs = children.finish();
    let failures = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{world_size}-process pipeline Ring test failed:\n{}",
        if timed_out {
            format!(
                "timed out waiting for Ring workers\n\n{}",
                failures.join("\n\n")
            )
        } else {
            failures.join("\n\n")
        }
    );
}
