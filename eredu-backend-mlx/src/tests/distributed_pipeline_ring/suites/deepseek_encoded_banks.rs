// Published encodings through independently resident target and prediction banks.
#[derive(Clone, Copy, Debug)]
enum DeepSeekEncodedBankFixture {
    V3,
    V4Float,
    V4Ue8m0,
    V4Mixed,
    DsparkFloat,
    DsparkUe8m0,
    DsparkMixed,
}

fn run_deepseek_encoded_bank_case(
    encoding: DeepSeekEncodedBankFixture,
    axes: &'static str,
    residency: WorkerResidency,
) {
    use DeepSeekEncodedBankFixture::*;
    let checkpoint = tempfile::tempdir().unwrap();
    let dspark = matches!(encoding, DsparkFloat | DsparkUe8m0 | DsparkMixed);
    let v3 = matches!(encoding, V3);
    if v3 {
        write_deepseek_v3_fp8_fixture(checkpoint.path());
    } else {
        write_deepseek_v4_fp8_fixture_kind(
            checkpoint.path(),
            dspark,
            matches!(encoding, V4Ue8m0 | V4Mixed | DsparkUe8m0 | DsparkMixed),
            matches!(encoding, V4Mixed | DsparkMixed),
        );
    }
    std::fs::write(
        checkpoint.path().join("component-independent-experts.json"),
        b"{}",
    )
    .unwrap();
    let path = checkpoint.path().to_owned();
    eprintln!("DeepSeek independent encoded experts {encoding:?} {axes} {residency:?}");
    run_ring_pipeline_processes(
        residency,
        if v3 {
            FixtureFamily::DeepSeek
        } else {
            FixtureFamily::DeepSeekV4
        },
        if dspark {
            WorkerMode::OpaqueDeepSeekDsparkTarget
        } else {
            WorkerMode::OpaqueDeepSeekMtpTarget
        },
        checkpoint,
        path,
        Some(axes),
    );
}

#[test]
#[ignore = "requires native MLX and local Ring ranks; run explicitly"]
fn ring_deepseek_encoded_independent_banks_focused() {
    use DeepSeekEncodedBankFixture::*;
    for encoding in [
        V3,
        V4Float,
        V4Ue8m0,
        V4Mixed,
        DsparkFloat,
        DsparkUe8m0,
        DsparkMixed,
    ] {
        run_deepseek_encoded_bank_case(encoding, "tp", WorkerResidency::FullyResident);
    }
}

macro_rules! deepseek_encoded_bank_matrix {
    ($name:ident, $encoding:ident) => {
        #[test]
        #[ignore = "requires native MLX and up to eight local Ring ranks; run explicitly"]
        fn $name() {
            for axes in ["tp", "pp", "ep", "tp-pp", "tp-ep", "pp-ep", "tp-pp-ep"] {
                for residency in [
                    WorkerResidency::FullyResident,
                    WorkerResidency::LayerwiseHost,
                    WorkerResidency::DenseDiskStream,
                ] {
                    run_deepseek_encoded_bank_case(
                        DeepSeekEncodedBankFixture::$encoding,
                        axes,
                        residency,
                    );
                }
            }
        }
    };
}
deepseek_encoded_bank_matrix!(ring_deepseek_v3_fp8_independent_banks_matrix, V3);
deepseek_encoded_bank_matrix!(ring_deepseek_v4_fp8_float_independent_banks_matrix, V4Float);
deepseek_encoded_bank_matrix!(ring_deepseek_v4_fp8_ue8m0_independent_banks_matrix, V4Ue8m0);
deepseek_encoded_bank_matrix!(ring_deepseek_v4_mixed_fp8_independent_banks_matrix, V4Mixed);
deepseek_encoded_bank_matrix!(
    ring_deepseek_dspark_fp8_float_independent_banks_matrix,
    DsparkFloat
);
deepseek_encoded_bank_matrix!(
    ring_deepseek_dspark_fp8_ue8m0_independent_banks_matrix,
    DsparkUe8m0
);
deepseek_encoded_bank_matrix!(
    ring_deepseek_dspark_mixed_fp8_independent_banks_matrix,
    DsparkMixed
);
