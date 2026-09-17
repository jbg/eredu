//! Actual load-time conversion and the retained transformed source manager.
use super::*;

// Three 64x64 expert matrices: packed 4-bit weights plus one F32 scale
// and affine bias per 64-value row. The source checkpoint remains F32.
const PACKED_MEMBER_BYTES:u64=3*64*(64/2+2*4);
const DENSE_MEMBER_BYTES:u64=3*64*64*4;

fn transformed_fixture()->Fixture {
    let fixture=fixture();
    let path=fixture.0.join("config.json");
    let mut config:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["hidden_size"]=64.into();
    config["intermediate_size"]=128.into();
    config["moe_intermediate_size"]=64.into();
    std::fs::write(path,serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved=eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    super::super::super::write_tensor_plan(&fixture.0,resolved.architecture.checkpoint());
    fixture
}
fn transformed_pressure(model:&LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>) {
    assert!(PACKED_MEMBER_BYTES<DENSE_MEMBER_BYTES);
    // Dense storage cannot pass this exact packed ownership and one-member peak.
    pressure_with_member_bytes(model,PACKED_MEMBER_BYTES);
}

#[test]
#[ignore="run after original independent-cache generation and transformed source handoff pass"]
fn native_original_transformed_independent_cache_evicts_and_decodes_in_all_drivers() {
    const CASE:&str="managed_plain::expert_cache::transformed::native_original_transformed_independent_cache_evicts_and_decodes_in_all_drivers";
    verify_mode_results(CASE,"EREDU_PUBLIC_TRANSFORMED_CACHE_MODE",|mode|{
        let execution=Residency::Resident.execution()
            .with_weight_transformation(eredu_core::WeightTransformationPlan::Affine{bits:4,group_size:64})
            .with_expert_cache(Some(eredu_core::ExpertCachePlan::new(
                Some(PACKED_MEMBER_BYTES),Some(PACKED_MEMBER_BYTES),PACKED_MEMBER_BYTES,PACKED_MEMBER_BYTES,
                eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
            )));
        run_mode_with_execution_observed(mode,transformed_fixture(),0.0,
            eredu_core::TextSamplingStrategy::Standard,execution,
            eredu_runtime::CacheResidencyPolicy::Device,Some(&transformed_pressure))
    });
}
