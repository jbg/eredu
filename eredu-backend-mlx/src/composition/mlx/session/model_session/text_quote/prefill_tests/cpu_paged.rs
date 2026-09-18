//! Original paged CPU execution uses the actual installed manager/scan issuer.
use super::*;

#[test]
#[ignore="requires qualified CPU execution and actual paged native source publication"]
fn original_cpu_paged_prefill_and_decode_match_ordinary_nonzero_state() {
    run_cpu_paged(false);
}
#[test]
#[ignore="requires original CPU Host store/load publication and actual paged source execution"]
fn original_cpu_paged_host_spill_and_reload_match_ordinary_nonzero_state() {
    run_cpu_paged(true);
}
fn run_cpu_paged(spill:bool) {
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let artifact=crate::composition::mlx::replicated_text::tests::tiny_artifact("llama",true);
    let mut expected=None;
    for original in [false,true] {
        let inspection=eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(artifact.path()).unwrap();
        let factory=crate::MlxBackendFactory::default().with_state_residency(
            eredu_runtime::CacheResidencyPolicy::Paged(// The fixture declares one KV head of width8, F32, two positions
                // per page, and separate K/V: one page is128 logical bytes.
                // Two-page capacity forces real Host traffic across six inputs
                // and four outputs while preserving the protected current page.
                eredu_runtime::PagedCacheOptions::new(2,if spill{2*2*1*2*8*4}else{1<<20},1<<20,1)
                .unwrap().with_full_attention(true)));
        let plan=eredu_core::ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx","cpu:0").unwrap());
        let selected=eredu_core::select_execution_plan_target(&factory,&plan,inspection).unwrap();
        let target=eredu_core::realize_execution_plan_target(&factory,&plan,selected).unwrap();
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        // The CPU factory's enclosing control and selected-context copy own
        // local wrappers. Their actual charges retire with this backend, whereas
        // registered process stream/worker births survive. Measure both before
        // loading any model; never accept a post-generation usage baseline.
        let retiring=target.backend().retiring_stream_wrapper_control_bytes();
        assert!(retiring>0);
        let baseline=pool.used_bytes().unwrap().checked_sub(retiring).unwrap();
        let mut runtime=target.into_runtime().unwrap();let stream=runtime.backend().stream().clone();
        let input=[2,5,7,3,11,13];let mut request=config(original);
        let mut policy=request.inference_policy();
        policy.submission_tracking_capacity_bytes=None;policy.graph_metadata_capacity_bytes=None;
        request=request.with_inference_policy(policy);
        let mut generation=if original {
            TextGeneration::from_token_ids_with_sequence(&mut runtime,
                eredu_core::TokenIdsInputPlan::new(&input).unwrap(),request,TokenFilter::All,None,
                GenerationSequenceRequest::new(4,&[]))
        }else{TextGeneration::new(&mut runtime,input.to_vec(),request)}.unwrap();
        let mut sequence=original.then(||generation.take_prepared_sequence().unwrap().prepare_storage().unwrap());
        let mut tokens=Vec::new();
        for (step, output) in (&mut generation).enumerate() {
            let output=output.unwrap_or_else(|error|panic!("paged CPU original={original}, spill={spill}, output={step}: {error:?}"));
            let token=output.token_id().unwrap();tokens.push(token);
            if let Some(sequence)=&mut sequence {sequence.commit(token,eredu_core::TokenTerminalSignals::default()).unwrap();}
        }
        drop(generation);assert_eq!(tokens.len(),4);
        if spill {
            let report=runtime.session().cache_residency_report().unwrap().unwrap();
            assert!(report.host_demotions>0,"the actual store worker must run");
            assert!(report.host_promotions>0,"the actual reload worker must run");
        }
        if let Some(sequence)=&sequence {assert_eq!(sequence.tokens().len(),4);}
        let source=runtime.session().payload.model.erased().resident_reset_source().unwrap();
        let state=source.state();
        let numeric=state.retained_arrays().into_iter().map(|value|(
            value.shape().to_vec(),value.evaluated().unwrap().try_to_vec::<f32>().unwrap())).collect::<Vec<_>>();
        assert!(!numeric.is_empty());assert!(numeric.iter().flat_map(|(_,v)|v).all(|x|x.is_finite()));
        assert!(numeric.iter().flat_map(|(_,v)|v).any(|x|*x!=0.0));
        let positions=runtime.session().payload.model.erased().state_snapshot();
        assert!(positions.iter().all(|(position,_)|*position==9));
        if let Some(reference)=&expected {assert_eq!(&(tokens,numeric),reference);}else{expected=Some((tokens,numeric));}
        drop(sequence);drop(source);fixture::finish(runtime,&stream);
        fixture::settle(&pool,baseline);
    }
}
