//! Paid full-attention pager branch; no new model/cache selection is implied.
use super::*;
use crate::backend::runtime::cache::state::RealtimeKvBranchPlan;
use eredu_runtime::working_memory::{InferenceExecutionIdentity,WorkingMemoryPool};
use safemlx::PreparedInputRuntime;

fn append(state:&mut MlxKeyValueState,rows:usize,base:f32,stream:&Stream) {
    let keys=(0..rows*8).map(|index|base+index as f32/16.0).collect::<Vec<_>>();
    let values=keys.iter().map(|value|1.0-3.0*value).collect::<Vec<_>>();
    let keys=Array::from_slice(&keys,&[1,1,rows as i32,8]);
    let values=Array::from_slice(&values,&[1,1,rows as i32,8]);
    keys.evaluated().unwrap();values.evaluated().unwrap();
    state.layers.slots_mut()[0].update_and_fetch(keys,values,stream).unwrap();
}
fn tail(state:&MlxKeyValueState)->Vec<Vec<f32>> {
    let MlxKeyValueLayerState::Paged(cache)=&state.layers.slots()[0] else {panic!("actual pager")};
    let mut values=Vec::new();
    cache.visit_realtime_tail_operands(&mut |array,copy| {
        assert!(!copy,"full-pager mutable tail branches retain immutable array aliases");
        values.push(array.evaluated().unwrap().as_slice::<f32>().to_vec());
    });
    values
}
fn ids(manager:&CacheResidencyManager)->Vec<eredu_core::cache::CacheBlockId> {
    manager.layer_block_ids(0,CacheRepresentation::KeyValue,0,i64::MAX,0).unwrap()
}
fn paid_branch(source:&MlxKeyValueState,runtime:&PreparedInputRuntime,stream:&Stream)
    ->(MlxKeyValueTransactionBranch,WorkingMemoryPool,u64) {
    let plan=RealtimeKvBranchPlan::inspect(source).unwrap();
    let quoted=u64::try_from(plan.host_bytes().unwrap()).unwrap();
    assert!(plan.copy_plan(runtime,stream).unwrap().is_none(),
        "alias-only full pager does not fabricate a numerical copy phase");
    let pool=WorkingMemoryPool::new(1<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),1<<20).unwrap();
    let initial=pool.used_bytes().unwrap();
    let branch=plan.prepare(None,None,runtime,stream,&funding,None).unwrap();
    assert!(pool.used_bytes().unwrap()>initial);
    assert!(pool.used_bytes().unwrap()<=initial+quoted,"source quote bounds actual branch controls");
    drop(funding);
    assert!(pool.used_bytes().unwrap()>0,"branch owns its original Host source");
    (branch,pool,initial+quoted)
}

#[test]
#[ignore="requires local MLX CPU execution; focused paid full-pager transaction adapter"]
fn original_full_paged_realtime_branch_preserves_pages_tail_and_host_custody() {
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let _roots=safemlx::PrefillRootsRuntime::prepare_for_stream(&stream,&stream).unwrap();
    let runtime=PreparedInputRuntime::prepare().unwrap();
    let pager=manager();
    let mut canonical=MlxKeyValueState::paged(layout(None),pager.clone(),None).unwrap();

    // Preserve None for the actual untouched tail; no synthetic empty allocation.
    let (empty,pool,_)=paid_branch(&canonical,&runtime,&stream);
    assert!(tail(&empty.state).is_empty());
    MlxKeyValueState::discard_branch(empty).unwrap();
    assert!(pool.used_bytes().unwrap()>0,"canonical reporting retains the source after empty discard");
    let first_source_pool=pool;
    let first_source_charge=first_source_pool.used_bytes().unwrap();
    // The first paid clone also initializes the canonical source's lazy
    // revision. That shared identity owns this account independently of the
    // replaceable reporting tables, including after an empty discard.
    assert_eq!(pager.report().unwrap().logical_cached_tokens,0);
    assert_eq!(pager.report().unwrap().mutable_tail_bytes,0);

    append(&mut canonical,5,0.75,&stream);
    let original_tail=tail(&canonical);
    let original_ids=ids(&pager);
    let original_report=pager.report().unwrap();
    assert_eq!(canonical.offset(),5);assert_eq!(original_ids.len(),1);
    assert_eq!(original_tail.len(),2);assert_eq!(original_tail[0].len(),8);
    assert!(original_report.mutable_tail_bytes>0);

    let (mut branch,pool,ceiling)=paid_branch(&canonical,&runtime,&stream);
    assert_eq!(first_source_pool.used_bytes().unwrap(),first_source_charge,
        "canonical revision preserves the first source account after reporting replacement");
    assert_eq!(tail(&branch.state),original_tail);
    append(&mut branch.state,4,11.0,&stream);
    assert_eq!(branch.state.offset(),9);assert!(ids(&pager).len()>original_ids.len());
    assert_eq!(canonical.offset(),5);assert_eq!(tail(&canonical),original_tail);
    MlxKeyValueState::discard_branch(branch).unwrap();
    assert_eq!(ids(&pager),original_ids);
    assert_eq!(tail(&canonical),original_tail);
    assert_eq!(pager.report().unwrap().logical_cached_tokens,5);
    assert_eq!(pager.report().unwrap().mutable_tail_bytes,original_report.mutable_tail_bytes);
    assert!(pool.used_bytes().unwrap()>0,"rollback retains canonical reporting storage under its actual source");
    assert!(pool.peak_bytes().unwrap()<=ceiling,"quoted controls include actual rollback work");

    // The same canonical source can resume a fresh paid branch after discard.
    // Compare its unchanged ordinary transaction worker numerically as well.
    let ordinary_manager=manager();
    let mut ordinary=MlxKeyValueState::paged(layout(None),ordinary_manager,None).unwrap();
    append(&mut ordinary,5,0.75,&stream);
    let mut expected=ordinary.branch().unwrap();
    append(&mut expected.state,5,31.0,&stream);
    ordinary.commit_branch(expected).unwrap();

    let discarded_reporting_pool=pool;
    let (mut resumed,pool,_)=paid_branch(&canonical,&runtime,&stream);
    assert_eq!(discarded_reporting_pool.used_bytes().unwrap(),0,"fresh reporting replaces and retires discarded-frame storage");
    append(&mut resumed.state,5,31.0,&stream);
    canonical.commit_branch(resumed).unwrap();
    assert_eq!(canonical.offset(),10);assert_eq!(canonical.offset(),ordinary.offset());
    assert_eq!(tail(&canonical),tail(&ordinary));
    assert_eq!(pager.report().unwrap().logical_cached_tokens,10);
    assert!(original_ids.iter().all(|id|ids(&pager).contains(id)),"canonical sealed pages survive commit");
    assert!(pool.used_bytes().unwrap()>0,"committed paid state retains its Host storage source");
    let retained_revision=eredu_runtime::working_memory::InferenceStateRetention::inference_retention(&canonical).revision().clone();
    drop(canonical);
    assert_eq!(first_source_pool.used_bytes().unwrap(),first_source_charge,
        "escaped source revision keeps its original account after canonical state retirement");
    assert!(pool.used_bytes().unwrap()>0,"remaining manager alias still owns paid reporting storage");
    drop(pager);
    assert_eq!(pool.used_bytes().unwrap(),0,"last paid canonical and reporting owner retires the current account");
    assert_eq!(first_source_pool.used_bytes().unwrap(),first_source_charge,
        "source revision custody is independent of the retired manager and reporting tables");
    drop(retained_revision);
    assert_eq!(first_source_pool.used_bytes().unwrap(),0,
        "last original revision owner retires the first cumulative source account");

    let sliding=MlxKeyValueState::paged(layout(Some(4)),manager(),None).unwrap();
    let error=RealtimeKvBranchPlan::inspect(&sliding).err().expect("sliding rollback remains unqualified");
    assert!(matches!(error,crate::backend::Error::PrefillControl(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)));
}
