//! Same local host window and global model equations in the public TP driver.
use super::*;
const CASE:&str="managed_plain::parallel::layerwise::native_managed_host_layerwise_tensor_parallel_matches_ordinary_and_controlled";
const MODE:&str="EREDU_PUBLIC_MANAGED_HOST_TP_MODE";
const RESULT:&str="PUBLIC_MANAGED_HOST_TP_RESULT:";
fn run(mode:&str)->serde_json::Value {
    // The nonzero two-unit model exceeds its one-unit device window. Every
    // 2/2/1 prefill chunk and four-token continuation crosses both acquires.
    let weights=eredu_runtime::WeightResidency::layerwise_host(
        eredu_runtime::LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(8<<20),Some(8<<20),1).unwrap()));
    run_partitioned_with_load(mode,eredu_core::ParallelTopology::new(2,1,1,1).unwrap(),
        None,||fixture(false),eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(weights))
}
#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_managed_host_layerwise_tensor_parallel_matches_ordinary_and_controlled() {
    compare_modes(CASE,MODE,RESULT,"Host layerwise TP",run)
}

// Six units leave three per pipeline stage: both the one-unit host window and
// the ordinary two-unit disk transfer window must turn over on every chunk.
fn bounded_weights_fixture()->Fixture {
    let root=fixture(false);
    let path=root.0.join("config.json");
    let mut config:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["num_hidden_layers"]=6.into();
    std::fs::write(path,serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved=eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0,resolved.architecture.checkpoint());
    root
}
fn run_bounded_weights(mode:&str,tp:usize,pp:usize,disk:bool)->serde_json::Value {
    let weights=if disk {
        eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(8<<20,0,0,0).unwrap())
    } else {
        eredu_runtime::WeightResidency::layerwise_host(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(8<<20),Some(8<<20),1).unwrap()))
    };
    run_partitioned_with_load(mode,eredu_core::ParallelTopology::new(tp,pp,1,1).unwrap(),
        None,bounded_weights_fixture,
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(weights))
}

#[test]
#[ignore="requires Metal and local Ring processes"]
fn native_managed_host_pipeline_parallel_turns_over_bounded_weight_windows(){
    compare_modes_with_world(
        "managed_plain::parallel::layerwise::native_managed_host_pipeline_parallel_turns_over_bounded_weight_windows",
        "EREDU_PUBLIC_MANAGED_HOST_PIPELINE_WEIGHTS_MODE",
        "PUBLIC_MANAGED_HOST_PIPELINE_WEIGHTS_RESULT:",
        "host_pipeline bounded weights",2,|mode|run_bounded_weights(mode,1,2,false))
}

#[test]
#[ignore="requires Metal and local Ring processes"]
fn native_managed_host_combined_parallel_turns_over_bounded_weight_windows(){
    compare_modes_with_world(
        "managed_plain::parallel::layerwise::native_managed_host_combined_parallel_turns_over_bounded_weight_windows",
        "EREDU_PUBLIC_MANAGED_HOST_COMBINED_WEIGHTS_MODE",
        "PUBLIC_MANAGED_HOST_COMBINED_WEIGHTS_RESULT:",
        "host_combined bounded weights",4,|mode|run_bounded_weights(mode,2,2,false))
}

#[test]
#[ignore="requires Metal and local Ring processes"]
fn native_managed_disk_tensor_parallel_turns_over_bounded_weight_windows(){
    compare_modes_with_world(
        "managed_plain::parallel::layerwise::native_managed_disk_tensor_parallel_turns_over_bounded_weight_windows",
        "EREDU_PUBLIC_MANAGED_DISK_TENSOR_WEIGHTS_MODE",
        "PUBLIC_MANAGED_DISK_TENSOR_WEIGHTS_RESULT:",
        "disk_tensor bounded weights",2,|mode|run_bounded_weights(mode,2,1,true))
}

#[test]
#[ignore="requires Metal and local Ring processes"]
fn native_managed_disk_pipeline_parallel_turns_over_bounded_weight_windows(){
    compare_modes_with_world(
        "managed_plain::parallel::layerwise::native_managed_disk_pipeline_parallel_turns_over_bounded_weight_windows",
        "EREDU_PUBLIC_MANAGED_DISK_PIPELINE_WEIGHTS_MODE",
        "PUBLIC_MANAGED_DISK_PIPELINE_WEIGHTS_RESULT:",
        "disk_pipeline bounded weights",2,|mode|run_bounded_weights(mode,1,2,true))
}

#[test]
#[ignore="requires Metal and local Ring processes"]
fn native_managed_disk_combined_parallel_turns_over_bounded_weight_windows(){
    compare_modes_with_world(
        "managed_plain::parallel::layerwise::native_managed_disk_combined_parallel_turns_over_bounded_weight_windows",
        "EREDU_PUBLIC_MANAGED_DISK_COMBINED_WEIGHTS_MODE",
        "PUBLIC_MANAGED_DISK_COMBINED_WEIGHTS_RESULT:",
        "disk_combined bounded weights",4,|mode|run_bounded_weights(mode,2,2,true))
}
