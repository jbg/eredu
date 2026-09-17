//! Actual selected four-member logical TP groups across two pipeline stages.
use super::*;
const CASE:&str="managed_plain::parallel::larger::native_managed_larger_combined_parallel_matches_ordinary_and_controlled";
const MODE:&str="EREDU_PUBLIC_MANAGED_LARGER_COMBINED_MODE";
const RESULT:&str="PUBLIC_MANAGED_LARGER_COMBINED_RESULT:";
fn run(mode:&str)->serde_json::Value {
    run_partitioned_with_fixture(mode,eredu_core::ParallelTopology::new(4,2,1,1).unwrap(),None,larger_fixture)
}
#[test]
#[ignore="requires Metal and eight local Ring ranks"]
fn native_managed_larger_combined_parallel_matches_ordinary_and_controlled(){
    compare_modes_with_world(CASE,MODE,RESULT,"larger combined TP/PP",8,run)
}

// Four actual head units keep every selected TP4 rank nonempty. Recreate the
// same deterministic nonzero weights from this fixture's exact neutral schema;
// the serial oracle and all ordinary/managed/controlled ranks use this source.
fn larger_fixture()->Fixture {
    let fixture=fixture(false);
    let path=fixture.0.join("config.json");
    let mut config:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["num_key_value_heads"]=4.into();
    let resolved=eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    std::fs::write(path,serde_json::to_vec(&config).unwrap()).unwrap();
    write_tensor_plan(&fixture.0,resolved.architecture.checkpoint());
    fixture
}
