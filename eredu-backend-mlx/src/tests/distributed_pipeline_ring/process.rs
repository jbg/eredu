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
        if matches!(
            mode,
            WorkerMode::OpaqueComponentCapture | WorkerMode::OpaqueComponentCaptureAddressableBank
        ) {
            write_deepseek_gguf_fixture_with_components(&path, true);
        } else {
            write_deepseek_gguf_fixture(&path);
        }
        path
    } else if let FixtureFamily::MuseGlimmerGguf(routed) = family {
        let path = checkpoint.path().join("model.gguf");
        write_muse_glimmer_gguf_component_fixture(&path, routed);
        path
    } else if family == FixtureFamily::DeepSeekDenseGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_dense_gguf_fixture(&path);
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
        write_qwen3_moe_gguf_fixture(&path, mode == WorkerMode::OpaqueComponentCapture);
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
        write_kimi_linear_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else {
        match family {
            FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe if mode.is_component_capture() => {
                write_qwen3_vl_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen3VlMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::MuseGlimmer | FixtureFamily::MuseGlimmerMoe
                if mode.is_component_capture() =>
            {
                write_muse_glimmer_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::MuseGlimmerMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::Inkling | FixtureFamily::InklingDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueInklingComponents
                        | WorkerMode::OpaqueInklingComponentsRequantize
                        | WorkerMode::OpaqueInklingComponentsMxFp4
                ) =>
            {
                prediction_components::inkling_component_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Inkling,
                    matches!(
                        mode,
                        WorkerMode::OpaqueInklingComponents
                            | WorkerMode::OpaqueInklingComponentsRequantize
                            | WorkerMode::OpaqueInklingComponentsMxFp4
                    )
                    .then_some(true),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                            | WorkerMode::OpaqueInklingComponents
                    ),
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueQwenHybridMtp
                        | WorkerMode::OpaqueQwenHybridMtpFp8
                        | WorkerMode::OpaqueQwenHybridMtpPreparationFailure
                        | WorkerMode::OpaqueQwenHybridMtpSchedulerFailure
                        | WorkerMode::OpaqueQwenHybridMtpControlDelivery(_)
                        | WorkerMode::OpaqueQwenHybridMtpControlSetupFailure
                ) =>
            {
                write_qwen35_multimodal_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                    1,
                    mode == WorkerMode::OpaqueQwenHybridMtpFp8,
                    true,
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen35_conditional_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                )
            }
            FixtureFamily::Qwen3Next
            | FixtureFamily::Qwen35
            | FixtureFamily::Qwen3NextMoe
            | FixtureFamily::Qwen35Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen_hybrid_component_fixture(checkpoint.path(), family)
            }
            FixtureFamily::K2Fp8(partial) => write_k2_fp8_fixture(checkpoint.path(), partial),
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Nanbeige => write_nanbeige_fixture(checkpoint.path()),
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
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 1, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_transform_fixture_with_prediction(checkpoint.path(), 1, 64, 1)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 0, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_mixed_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_dense_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekDense => write_deepseek_dense_fixture(checkpoint.path()),
            FixtureFamily::DeepSeekV4 if mode.is_dspark() => write_deepseek_v4_dspark_fixture(
                checkpoint.path(),
                mode != WorkerMode::OpaqueDeepSeekDsparkTarget,
            ),
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_v4_target_component_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueComponentCaptureRequantize
                            | WorkerMode::OpaqueComponentCaptureMxFp4
                            | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                    ),
                );
            }
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_v4_transform_fixture(checkpoint.path());
            }
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_v4_component_fixture(checkpoint.path())
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
            FixtureFamily::Lfm2 | FixtureFamily::Lfm2Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_lfm2_component_fixture(checkpoint.path(), family == FixtureFamily::Lfm2Moe)
            }
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_kimi_linear_component_fixture(
                    checkpoint.path(),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                    ),
                )
            }
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::AddressableParameterBankRequantize | WorkerMode::Requantize
                ) =>
            {
                write_nemotron_quantizable_fixture(checkpoint.path())
            }
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::OpaqueNemotronHMtp
                        | WorkerMode::OpaqueNemotronHMtpRouted
                        | WorkerMode::OpaqueNemotronHMtpMxFp4
                        | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                ) =>
            {
                write_nemotron_prediction_components_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpRouted
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpMxFp4
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                )
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
            FixtureFamily::GptOss => write_gpt_oss_fixture_with_patterns(
                checkpoint.path(),
                mode == WorkerMode::OpaqueComponentCapture,
            ),
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
        if matches!(
            mode,
            WorkerMode::OpaqueComponentCapture | WorkerMode::OpaqueComponentCaptureAddressableBank
        ) {
            write_deepseek_gguf_fixture_with_components(&path, true);
        } else {
            write_deepseek_gguf_fixture(&path);
        }
        path
    } else if let FixtureFamily::MuseGlimmerGguf(routed) = family {
        let path = checkpoint.path().join("model.gguf");
        write_muse_glimmer_gguf_component_fixture(&path, routed);
        path
    } else if family == FixtureFamily::DeepSeekDenseGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_dense_gguf_fixture(&path);
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
        write_qwen3_moe_gguf_fixture(&path, mode == WorkerMode::OpaqueComponentCapture);
        path
    } else if family == FixtureFamily::KimiLinearGguf {
        let path = checkpoint.path().join("model.gguf");
        write_kimi_linear_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else {
        match family {
            FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe if mode.is_component_capture() => {
                write_qwen3_vl_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen3VlMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::MuseGlimmer | FixtureFamily::MuseGlimmerMoe
                if mode.is_component_capture() =>
            {
                write_muse_glimmer_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::MuseGlimmerMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::Inkling | FixtureFamily::InklingDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueInklingComponents
                        | WorkerMode::OpaqueInklingComponentsRequantize
                        | WorkerMode::OpaqueInklingComponentsMxFp4
                ) =>
            {
                prediction_components::inkling_component_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Inkling,
                    matches!(
                        mode,
                        WorkerMode::OpaqueInklingComponents
                            | WorkerMode::OpaqueInklingComponentsRequantize
                            | WorkerMode::OpaqueInklingComponentsMxFp4
                    )
                    .then_some(true),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                            | WorkerMode::OpaqueInklingComponents
                    ),
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueQwenHybridMtp
                        | WorkerMode::OpaqueQwenHybridMtpFp8
                        | WorkerMode::OpaqueQwenHybridMtpPreparationFailure
                        | WorkerMode::OpaqueQwenHybridMtpSchedulerFailure
                        | WorkerMode::OpaqueQwenHybridMtpControlDelivery(_)
                        | WorkerMode::OpaqueQwenHybridMtpControlSetupFailure
                ) =>
            {
                write_qwen35_multimodal_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                    1,
                    mode == WorkerMode::OpaqueQwenHybridMtpFp8,
                    true,
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen35_conditional_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                )
            }
            FixtureFamily::Qwen3Next
            | FixtureFamily::Qwen35
            | FixtureFamily::Qwen3NextMoe
            | FixtureFamily::Qwen35Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen_hybrid_component_fixture(checkpoint.path(), family)
            }
            FixtureFamily::K2Fp8(partial) => write_k2_fp8_fixture(checkpoint.path(), partial),
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Qwen2 => write_qwen_fixture(checkpoint.path(), "qwen2"),
            FixtureFamily::Nanbeige => write_nanbeige_fixture(checkpoint.path()),
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
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 1, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_transform_fixture_with_prediction(checkpoint.path(), 1, 64, 1)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 0, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_mixed_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_dense_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekDense => write_deepseek_dense_fixture(checkpoint.path()),
            FixtureFamily::DeepSeekV4 if mode.is_dspark() => write_deepseek_v4_dspark_fixture(
                checkpoint.path(),
                mode != WorkerMode::OpaqueDeepSeekDsparkTarget,
            ),
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_v4_target_component_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueComponentCaptureRequantize
                            | WorkerMode::OpaqueComponentCaptureMxFp4
                            | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                    ),
                );
            }
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_v4_transform_fixture(checkpoint.path());
            }
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_v4_component_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekV4 => write_deepseek_v4_fixture(checkpoint.path(), 1),
            FixtureFamily::Qwen3Moe => write_qwen_fixture(checkpoint.path(), "qwen3_moe"),
            FixtureFamily::GptOss => write_gpt_oss_fixture_with_patterns(
                checkpoint.path(),
                mode == WorkerMode::OpaqueComponentCapture,
            ),
            FixtureFamily::Lfm2 | FixtureFamily::Lfm2Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_lfm2_component_fixture(checkpoint.path(), family == FixtureFamily::Lfm2Moe)
            }
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_kimi_linear_component_fixture(
                    checkpoint.path(),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                    ),
                )
            }
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::OpaqueNemotronHMtp
                        | WorkerMode::OpaqueNemotronHMtpRouted
                        | WorkerMode::OpaqueNemotronHMtpMxFp4
                        | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                ) =>
            {
                write_nemotron_prediction_components_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpRouted
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpMxFp4
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                )
            }
            FixtureFamily::NemotronH => write_nemotron_fixture(checkpoint.path()),
            FixtureFamily::Inkling => write_inkling_fixture(checkpoint.path()),
            FixtureFamily::InklingDense => write_inkling_dense_fixture(checkpoint.path()),
            FixtureFamily::InklingDenseMultimodal => {
                write_inkling_dense_multimodal_fixture(checkpoint.path())
            }
            FixtureFamily::InklingMultimodal => write_inkling_multimodal_fixture(checkpoint.path()),
            FixtureFamily::Qwen3Next => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_next"),
            FixtureFamily::Qwen35 => write_qwen_hybrid_fixture(checkpoint.path(), "qwen3_5_text"),
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
    OpaqueComponentCapture,
    OpaqueComponentCaptureRequantize,
    OpaqueComponentCaptureMxFp4,
    OpaqueComponentCaptureAddressableBank,
    OpaqueComponentCaptureAddressableBankRequantize,
    OpaqueProviderFailure,
    OpaqueSessionAddressableParameterBank,
    OpaqueSessionEvictingAddressableParameterBank,
    OpaqueMuseImage,
    OpaqueInklingMedia,
    OpaqueInklingMtp,
    OpaqueInklingComponents,
    OpaqueInklingComponentsRequantize,
    OpaqueInklingComponentsMxFp4,
    OpaqueInklingMtpAddressableParameterBank,
    OpaqueQwenHybridMtp,
    OpaqueQwenHybridMtpFp8,
    OpaqueQwenHybridMtpPreparationFailure,
    OpaqueQwenHybridMtpSchedulerFailure,
    OpaqueQwenHybridMtpControlSetupFailure,
    OpaqueQwenHybridMtpControlDelivery(&'static str),
    OpaqueNemotronHMtp,
    OpaqueNemotronHMtpRouted,
    OpaqueNemotronHMtpMxFp4,
    OpaqueNemotronHMtpRoutedMxFp4,
    OpaqueDeepSeekMtpTarget,
    OpaqueDeepSeekMtpTargetRequantize,
    OpaqueDeepSeekMtpTargetMxFp4,
    OpaqueDeepSeekDsparkTarget,
    OpaqueDeepSeekDsparkTargetRequantize,
    OpaqueDeepSeekDsparkTargetMxFp4,
    OpaqueGemma4Media,
    OpaqueGemma4MediaInspection,
    OpaqueQwen3VlMedia,
    OpaqueQwenConditionalMedia,
    QwenHybridPromptCache,
    PromptCachePrepareFailure,
}

impl WorkerMode {
    fn runs_prediction_components(self) -> bool {
        matches!(
            self,
            Self::OpaqueInklingComponents
                | Self::OpaqueInklingComponentsRequantize
                | Self::OpaqueInklingComponentsMxFp4
                | Self::OpaqueInklingMtpAddressableParameterBank
                | Self::OpaqueQwenHybridMtp
                | Self::OpaqueQwenHybridMtpFp8
                | Self::OpaqueNemotronHMtp
                | Self::OpaqueNemotronHMtpRouted
                | Self::OpaqueNemotronHMtpMxFp4
                | Self::OpaqueNemotronHMtpRoutedMxFp4
                | Self::OpaqueDeepSeekMtpTarget
                | Self::OpaqueDeepSeekMtpTargetRequantize
                | Self::OpaqueDeepSeekMtpTargetMxFp4
                | Self::OpaqueDeepSeekDsparkTarget
                | Self::OpaqueDeepSeekDsparkTargetRequantize
                | Self::OpaqueDeepSeekDsparkTargetMxFp4
        )
    }

    fn is_component_capture(self) -> bool {
        matches!(
            self,
            Self::OpaqueComponentCapture | Self::OpaqueComponentCaptureAddressableBank
        ) || self.is_quantized_component_capture()
    }

    fn is_quantized_component_capture(self) -> bool {
        matches!(
            self,
            Self::OpaqueComponentCaptureRequantize
                | Self::OpaqueComponentCaptureMxFp4
                | Self::OpaqueComponentCaptureAddressableBankRequantize
        )
    }

    fn is_dspark(self) -> bool {
        matches!(
            self,
            Self::OpaqueDeepSeekDsparkTarget
                | Self::OpaqueDeepSeekDsparkTargetRequantize
                | Self::OpaqueDeepSeekDsparkTargetMxFp4
        )
    }
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
        if matches!(
            mode,
            WorkerMode::OpaqueComponentCapture | WorkerMode::OpaqueComponentCaptureAddressableBank
        ) {
            write_deepseek_gguf_fixture_with_components(&path, true);
        } else {
            write_deepseek_gguf_fixture(&path);
        }
        path
    } else if let FixtureFamily::MuseGlimmerGguf(routed) = family {
        let path = checkpoint.path().join("model.gguf");
        write_muse_glimmer_gguf_component_fixture(&path, routed);
        path
    } else if family == FixtureFamily::DeepSeekDenseGguf {
        let path = checkpoint.path().join("model.gguf");
        write_deepseek_dense_gguf_fixture(&path);
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
        write_qwen3_moe_gguf_fixture(&path, mode == WorkerMode::OpaqueComponentCapture);
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
        write_kimi_linear_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else if family == FixtureFamily::InklingGguf {
        let path = checkpoint.path().join("model.gguf");
        write_inkling_gguf_fixture(
            &path,
            matches!(
                mode,
                WorkerMode::OpaqueComponentCapture
                    | WorkerMode::OpaqueComponentCaptureAddressableBank
            ),
        );
        path
    } else {
        match family {
            FixtureFamily::Qwen3Vl | FixtureFamily::Qwen3VlMoe if mode.is_component_capture() => {
                write_qwen3_vl_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen3VlMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::MuseGlimmer | FixtureFamily::MuseGlimmerMoe
                if mode.is_component_capture() =>
            {
                write_muse_glimmer_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::MuseGlimmerMoe,
                    mode.is_quantized_component_capture(),
                )
            }
            FixtureFamily::Inkling | FixtureFamily::InklingDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueInklingComponents
                        | WorkerMode::OpaqueInklingComponentsRequantize
                        | WorkerMode::OpaqueInklingComponentsMxFp4
                ) =>
            {
                prediction_components::inkling_component_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Inkling,
                    matches!(
                        mode,
                        WorkerMode::OpaqueInklingComponents
                            | WorkerMode::OpaqueInklingComponentsRequantize
                            | WorkerMode::OpaqueInklingComponentsMxFp4
                    )
                    .then_some(true),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                            | WorkerMode::OpaqueInklingComponents
                    ),
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueQwenHybridMtp
                        | WorkerMode::OpaqueQwenHybridMtpFp8
                        | WorkerMode::OpaqueQwenHybridMtpPreparationFailure
                        | WorkerMode::OpaqueQwenHybridMtpSchedulerFailure
                        | WorkerMode::OpaqueQwenHybridMtpControlDelivery(_)
                        | WorkerMode::OpaqueQwenHybridMtpControlSetupFailure
                ) =>
            {
                write_qwen35_multimodal_fixture_config(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                    1,
                    mode == WorkerMode::OpaqueQwenHybridMtpFp8,
                    true,
                )
            }
            FixtureFamily::Qwen35Multimodal | FixtureFamily::Qwen35MoeMultimodal
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen35_conditional_component_fixture(
                    checkpoint.path(),
                    family == FixtureFamily::Qwen35MoeMultimodal,
                )
            }
            FixtureFamily::Qwen3Next
            | FixtureFamily::Qwen35
            | FixtureFamily::Qwen3NextMoe
            | FixtureFamily::Qwen35Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_qwen_hybrid_component_fixture(checkpoint.path(), family)
            }
            FixtureFamily::K2Fp8(partial) => write_k2_fp8_fixture(checkpoint.path(), partial),
            FixtureFamily::K2Dense | FixtureFamily::K2Mova => {
                write_k2_fixture(checkpoint.path(), family == FixtureFamily::K2Mova)
            }
            FixtureFamily::Llama => write_fixture(checkpoint.path()),
            FixtureFamily::Mistral => write_mistral_fixture(checkpoint.path()),
            FixtureFamily::Nanbeige => write_nanbeige_fixture(checkpoint.path()),
            FixtureFamily::DeepSeek if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 1, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_transform_fixture_with_prediction(checkpoint.path(), 1, 64, 1)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                ) =>
            {
                write_deepseek_fixture_with_values(checkpoint.path(), 2, 0, true)
            }
            FixtureFamily::DeepSeek
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_mixed_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeek => write_deepseek_fixture(checkpoint.path(), 2),
            FixtureFamily::DeepSeekDense
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_dense_transform_fixture(checkpoint.path())
            }
            FixtureFamily::DeepSeekDense => write_deepseek_dense_fixture(checkpoint.path()),
            FixtureFamily::DeepSeekV4 if mode.is_dspark() => write_deepseek_v4_dspark_fixture(
                checkpoint.path(),
                mode != WorkerMode::OpaqueDeepSeekDsparkTarget,
            ),
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_deepseek_v4_target_component_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueComponentCaptureRequantize
                            | WorkerMode::OpaqueComponentCaptureMxFp4
                            | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                    ),
                );
            }
            FixtureFamily::DeepSeekV4
                if matches!(
                    mode,
                    WorkerMode::OpaqueDeepSeekMtpTargetRequantize
                        | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4
                ) =>
            {
                write_deepseek_v4_transform_fixture(checkpoint.path());
            }
            FixtureFamily::DeepSeekV4 if mode == WorkerMode::OpaqueDeepSeekMtpTarget => {
                write_deepseek_v4_component_fixture(checkpoint.path())
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
            FixtureFamily::GptOss => write_gpt_oss_fixture_with_patterns(
                checkpoint.path(),
                mode == WorkerMode::OpaqueComponentCapture,
            ),
            FixtureFamily::Lfm2 | FixtureFamily::Lfm2Moe
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_lfm2_component_fixture(checkpoint.path(), family == FixtureFamily::Lfm2Moe)
            }
            FixtureFamily::Lfm2 => write_lfm2_pipeline_fixture(checkpoint.path(), false),
            FixtureFamily::Lfm2Moe => write_lfm2_pipeline_fixture(checkpoint.path(), true),
            FixtureFamily::KimiLinear
                if matches!(
                    mode,
                    WorkerMode::OpaqueComponentCapture
                        | WorkerMode::OpaqueComponentCaptureAddressableBank
                        | WorkerMode::OpaqueComponentCaptureRequantize
                        | WorkerMode::OpaqueComponentCaptureMxFp4
                        | WorkerMode::OpaqueComponentCaptureAddressableBankRequantize
                ) =>
            {
                write_kimi_linear_component_fixture(
                    checkpoint.path(),
                    !matches!(
                        mode,
                        WorkerMode::OpaqueComponentCapture
                            | WorkerMode::OpaqueComponentCaptureAddressableBank
                    ),
                )
            }
            FixtureFamily::KimiLinear => write_kimi_linear_fixture(checkpoint.path()),
            FixtureFamily::NemotronH
                if matches!(
                    mode,
                    WorkerMode::OpaqueNemotronHMtp
                        | WorkerMode::OpaqueNemotronHMtpRouted
                        | WorkerMode::OpaqueNemotronHMtpMxFp4
                        | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                ) =>
            {
                write_nemotron_prediction_components_fixture(
                    checkpoint.path(),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpRouted
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                    matches!(
                        mode,
                        WorkerMode::OpaqueNemotronHMtpMxFp4
                            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                    ),
                )
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
            FixtureFamily::MuseGlimmerMoe => {
                write_muse_glimmer_component_fixture(checkpoint.path(), true, false)
            }
            FixtureFamily::DeepSeekGguf
            | FixtureFamily::MuseGlimmerGguf(_)
            | FixtureFamily::DeepSeekDenseGguf
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
    if let Ok(device) = std::env::var("EREDU_TEST_RING_DEVICE") {
        eprintln!("Ring fixture device={device}");
    }
    let prompt_cache = tempfile::tempdir().unwrap();
    let world_size = match cartesian_axes {
        Some("tp-pp-ep") => 8,
        Some("tp" | "pp" | "ep") => 2,
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
        output_readers: Vec::with_capacity(world_size),
    };
    let media_fixture = checkpoint_path
        .is_file()
        .then(|| {
            let path = checkpoint_path
                .parent()
                .unwrap()
                .join("component-media-fixture.json");
            path.exists().then(|| {
                serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).unwrap()).unwrap()
            })
        })
        .flatten();
    let v4_fp8_fixture = checkpoint_path
        .join("component-v4-fp8-fixture.json")
        .exists()
        || checkpoint_path
            .join("component-v3-fp8-fixture.json")
            .exists();
    let mixed_fp8_fixture = checkpoint_path
        .join("component-v4-mixed-fp8-fixture.json")
        .exists();
    for rank in 0..world_size {
        let mut command = Command::new(&executable);
        if mixed_fp8_fixture {
            command.env("EREDU_RING_DEEPSEEK_MIXED_FP8", "1");
        }
        if v4_fp8_fixture {
            command.env("EREDU_RING_DEEPSEEK_FP8", "1");
        }
        if checkpoint_path
            .join("component-independent-experts.json")
            .exists()
        {
            command.env(EXPERT_CACHE, "1");
        }
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
        command.stderr(Stdio::piped());
        if let Some(protocol) = &media_fixture {
            command.env(
                COMPONENT_CAPTURE_PATCH_WIDTH,
                protocol["patch_width"].as_u64().unwrap().to_string(),
            );
        }
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
        if matches!(
            family,
            FixtureFamily::MuseGlimmer
                | FixtureFamily::MuseGlimmerMoe
                | FixtureFamily::Qwen3Vl
                | FixtureFamily::Qwen3VlMoe
                | FixtureFamily::Gemma
        ) && mode.is_component_capture()
        {
            command.env(COMPONENT_CAPTURE_MEDIA, "1");
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
            WorkerMode::OpaqueComponentCaptureAddressableBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_COMPONENT_CAPTURE, "1");
                command.env(EXPERT_CACHE, "1");
                if matches!(
                    family,
                    FixtureFamily::Qwen35Multimodal
                        | FixtureFamily::Qwen35MoeMultimodal
                        | FixtureFamily::MuseGlimmerMoe
                ) {
                    command.env(COMPONENT_CAPTURE_MEDIA, "1");
                }
            }
            WorkerMode::OpaqueComponentCapture => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_COMPONENT_CAPTURE, "1");
                if matches!(
                    family,
                    FixtureFamily::Qwen35Multimodal
                        | FixtureFamily::Qwen35MoeMultimodal
                        | FixtureFamily::MuseGlimmer
                        | FixtureFamily::MuseGlimmerMoe
                ) {
                    command.env(COMPONENT_CAPTURE_MEDIA, "1");
                }
            }
            WorkerMode::OpaqueComponentCaptureAddressableBankRequantize => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_COMPONENT_CAPTURE, "1");
                command.env(EXPERT_CACHE, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueComponentCaptureRequantize => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_COMPONENT_CAPTURE, "1");
                command.env(REQUANTIZE, "1");
            }
            WorkerMode::OpaqueComponentCaptureMxFp4 => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_COMPONENT_CAPTURE, "1");
                command.env(REQUANTIZE, "mxfp4");
            }
            WorkerMode::OpaqueProviderFailure => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_PROVIDER_FAILURE, "1");
                command.env(EXPERT_CACHE, "1");
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
            WorkerMode::OpaqueInklingComponents
            | WorkerMode::OpaqueInklingComponentsRequantize
            | WorkerMode::OpaqueInklingComponentsMxFp4 => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
                command.env(OPAQUE_INKLING_COMPONENTS, "1");
                if mode == WorkerMode::OpaqueInklingComponentsRequantize {
                    command.env(REQUANTIZE, "1");
                } else if mode == WorkerMode::OpaqueInklingComponentsMxFp4 {
                    command.env(REQUANTIZE, "mxfp4");
                }
            }
            WorkerMode::OpaqueInklingMtpAddressableParameterBank => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_INKLING_MTP, "1");
                command.env(EXPERT_CACHE, "1");
            }
            WorkerMode::OpaqueQwenHybridMtp
            | WorkerMode::OpaqueQwenHybridMtpFp8
            | WorkerMode::OpaqueQwenHybridMtpPreparationFailure
            | WorkerMode::OpaqueQwenHybridMtpSchedulerFailure
            | WorkerMode::OpaqueQwenHybridMtpControlDelivery(_)
            | WorkerMode::OpaqueQwenHybridMtpControlSetupFailure => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_QWEN_HYBRID_MTP, "1");
                if mode == WorkerMode::OpaqueQwenHybridMtpPreparationFailure {
                    command.env("EREDU_TEST_SPECULATIVE_PREPARATION_FAILURE", "1");
                }
                if let WorkerMode::OpaqueQwenHybridMtpControlDelivery(case) = mode {
                    command.env("EREDU_TEST_SPECULATIVE_CONTROL_DELIVERY", case);
                }
                if matches!(
                    mode,
                    WorkerMode::OpaqueQwenHybridMtpSchedulerFailure
                        | WorkerMode::OpaqueQwenHybridMtpControlSetupFailure
                ) {
                    command.env(
                        "EREDU_TEST_SPECULATIVE_SCHEDULER_FAILURE",
                        if mode == WorkerMode::OpaqueQwenHybridMtpControlSetupFailure {
                            "controlled"
                        } else {
                            "ordinary"
                        },
                    );
                }
                if mode == WorkerMode::OpaqueQwenHybridMtpFp8 {
                    command.env("EREDU_RING_PREDICTION_FP8", "1");
                }
            }
            WorkerMode::OpaqueNemotronHMtp
            | WorkerMode::OpaqueNemotronHMtpRouted
            | WorkerMode::OpaqueNemotronHMtpMxFp4
            | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4 => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_NEMOTRON_H_MTP, "1");
                if matches!(
                    mode,
                    WorkerMode::OpaqueNemotronHMtpMxFp4 | WorkerMode::OpaqueNemotronHMtpRoutedMxFp4
                ) {
                    command.env(REQUANTIZE, "mxfp4");
                }
            }
            WorkerMode::OpaqueDeepSeekMtpTarget
            | WorkerMode::OpaqueDeepSeekMtpTargetRequantize
            | WorkerMode::OpaqueDeepSeekMtpTargetMxFp4 => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
                if mode == WorkerMode::OpaqueDeepSeekMtpTargetRequantize {
                    command.env(REQUANTIZE, "1");
                } else if mode == WorkerMode::OpaqueDeepSeekMtpTargetMxFp4 {
                    command.env(REQUANTIZE, "mxfp4");
                }
            }
            WorkerMode::OpaqueDeepSeekDsparkTarget
            | WorkerMode::OpaqueDeepSeekDsparkTargetRequantize
            | WorkerMode::OpaqueDeepSeekDsparkTargetMxFp4 => {
                command.env(OPAQUE_SESSION, "1");
                command.env(OPAQUE_DEEPSEEK_MTP_TARGET, "1");
                command.env(OPAQUE_DEEPSEEK_DSPARK_TARGET, "1");
                if mode == WorkerMode::OpaqueDeepSeekDsparkTargetRequantize {
                    command.env(REQUANTIZE, "1");
                } else if mode == WorkerMode::OpaqueDeepSeekDsparkTargetMxFp4 {
                    command.env(REQUANTIZE, "mxfp4");
                }
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
        children.push(command.spawn().unwrap());
    }
    // Component trials include complete parameter/query passes, coordinated
    // edits and many snapshot branches. Keep a bounded whole-case watchdog
    // separate from the unchanged per-collective completion deadline.
    let case_seconds = match world_size {
        8 => 180,
        4 => 90,
        _ => 45,
    } * if mode.is_component_capture() || mode.runs_prediction_components() {
        4
    } else {
        1
    };
    // Published-size projector fixtures may spend much longer in bounded host
    // decoding than tiny synthetic models. This affects only the test watchdog.
    let case_seconds = media_fixture
        .as_ref()
        .and_then(|fixture| fixture.get("case_timeout_seconds"))
        .map(|seconds| seconds.as_u64().expect("fixture timeout must be seconds"))
        .unwrap_or(case_seconds);
    assert!(case_seconds > 0, "fixture timeout must be positive");
    let deadline = Instant::now() + Duration::from_secs(case_seconds);
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
