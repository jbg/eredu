//! Existing biased MXFP4 fixture through the shared public parallel driver.
use super::*;
fn source() -> Fixture {
    let root=fixture(false);
    eredu_evaluation::fixtures::write_gpt_oss_checkpoint(&root.0,true).unwrap();
    root
}
fn run<const TENSOR:usize,const PIPELINE:usize>(mode:&str)->serde_json::Value {
    run_partitioned_with_fixture(mode,
        eredu_core::ParallelTopology::new(TENSOR,PIPELINE,1,1).unwrap(),None,source)
}
#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_managed_gpt_oss_mxfp4_tensor_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::gpt_oss::native_managed_gpt_oss_mxfp4_tensor_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_GPT_OSS_MXFP4_TP_MODE","PUBLIC_GPT_OSS_MXFP4_TP_RESULT:",
        "biased MXFP4 GPT-OSS TP",run::<2,1>);
}
#[test]
#[ignore="requires Metal and two local Ring processes; gate after the TP source check"]
fn native_managed_gpt_oss_mxfp4_pipeline_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::gpt_oss::native_managed_gpt_oss_mxfp4_pipeline_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_GPT_OSS_MXFP4_PP_MODE","PUBLIC_GPT_OSS_MXFP4_PP_RESULT:",
        "biased MXFP4 GPT-OSS PP",run::<1,2>);
}
#[test]
#[ignore="requires Metal and four local Ring processes; gate after the TP source check"]
fn native_managed_gpt_oss_mxfp4_combined_matches_ordinary_and_controlled() {
    compare_modes_with_world(
        "managed_plain::parallel::gpt_oss::native_managed_gpt_oss_mxfp4_combined_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_GPT_OSS_MXFP4_COMBINED_MODE","PUBLIC_GPT_OSS_MXFP4_COMBINED_RESULT:",
        "biased MXFP4 GPT-OSS TP/PP",4,run::<2,2>);
}
