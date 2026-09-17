//! Same public distributed intervention driver with actual two-side evidence.
use super::*;
use eredu_core::intervention::InterventionEvidence;
fn loaded<const TP:usize,const PP:usize,const SUMMARY:bool>(mode:&str,
    model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture,partitioned:bool)->serde_json::Value {
    super::super::super::super::capture::run_intervention_evidence_loaded(mode,model,root,partitioned,
        "model.layers.0.feed_forward.units",TP*PP,16u64<<30,
        if SUMMARY{InterventionEvidence::Summary}else{InterventionEvidence::Preview{max_elements:7}})
}
fn ordinary<const TP:usize,const PP:usize,const SUMMARY:bool>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {loaded::<TP,PP,SUMMARY>("ordinary",model,root,true)}
fn managed<const TP:usize,const PP:usize,const SUMMARY:bool>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {loaded::<TP,PP,SUMMARY>("managed",model,root,true)}
fn controlled<const TP:usize,const PP:usize,const SUMMARY:bool>(model:LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,root:Fixture)->serde_json::Value {loaded::<TP,PP,SUMMARY>("controlled",model,root,true)}
fn run<const TP:usize,const PP:usize,const SUMMARY:bool>(mode:&str)->serde_json::Value {
    if mode=="serial" {
        let root=managed_fixture(fixture(false));
        let execution=ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx","metal:0").unwrap());
        let (model,_)=LoadedModel::load_execution_plan(&MlxBackendFactory::default(),&root.0,&execution)
            .unwrap_or_else(report_failure).into_parts();
        return loaded::<TP,PP,SUMMARY>("ordinary",model,root,false);
    }
    let worker=match mode {"ordinary"=>ordinary::<TP,PP,SUMMARY>,"managed"=>managed::<TP,PP,SUMMARY>,
        "controlled"=>controlled::<TP,PP,SUMMARY>,_=>panic!("evidence mode")};
    run_partitioned_with_fixture(mode,eredu_core::ParallelTopology::new(TP,PP,1,1).unwrap(),Some(worker),source::<1>)
}
fn compare(actual:&serde_json::Value,expected:&serde_json::Value,mode:&str,rank:usize) {
    super::compare(actual,expected,mode,rank);
    super::super::compare(&actual["evidence"],&expected["evidence"],mode,rank);
}
macro_rules! case {($name:ident,$tp:literal,$pp:literal,$summary:literal)=>{
    #[test]
    #[ignore="requires Metal and local Ring processes; gate PP/combined after corresponding TP evidence"]
    fn $name(){compare_selected_modes_by(concat!("managed_plain::parallel::capture::interventions::evidence::",stringify!($name)),
        "EREDU_PUBLIC_PARTITION_INTERVENTION_EVIDENCE_MODE","PUBLIC_PARTITION_INTERVENTION_EVIDENCE_RESULT:",
        "partition intervention evidence",$tp*$pp,&["ordinary","managed","controlled"],run::<$tp,$pp,$summary>,compare);}
};}
case!(native_tp_preview_evidence_preserves_original_component_outcome,2,1,false);
case!(native_pp_preview_evidence_preserves_original_component_outcome,1,2,false);
case!(native_combined_preview_evidence_preserves_original_component_outcome,2,2,false);
case!(native_tp_summary_evidence_preserves_original_component_outcome,2,1,true);
case!(native_pp_summary_evidence_preserves_original_component_outcome,1,2,true);
case!(native_combined_summary_evidence_preserves_original_component_outcome,2,2,true);
