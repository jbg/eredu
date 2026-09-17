use super::*;
use eredu_core::{AttentionPolicy,InputModality,InputTensorIdentity,PreparedInputError,checkpoint::TensorDtype};
use eredu_nn::workspace::*;
use eredu_runtime::{PreparedInputPart,PreparedInputPayload,PreparedModelInput};
use eredu_runtime::PreparedInputInspector;
use eredu_core::CapabilityError;
use std::{convert::Infallible,sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}}};

#[derive(Debug)]
struct AccountState {remaining:AtomicUsize,retired:AtomicBool}
#[derive(Debug)]
struct Account(Arc<AccountState>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),WorkspaceMetadataFundingError>{
        let n=self.0.remaining.load(Ordering::SeqCst);
        self.0.remaining.store(n.checked_sub(bytes).ok_or(WorkspaceMetadataFundingError::Capacity{
            required:bytes as u64,available:n as u64})?,Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {fn drop(&mut self){self.0.retired.store(true,Ordering::SeqCst);}}
#[derive(Clone,Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{panic!("text ingress only borrows existing token values")}
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error=Infallible;
    fn operation_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{panic!("no equation")}
    fn write_operation_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{panic!("no equation")}
    fn host_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{panic!("no equation")}
    fn write_host_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{panic!("no equation")}
}
fn context()->(WorkspaceContext,Arc<AccountState>){
    let state=Arc::new(AccountState{remaining:AtomicUsize::new(usize::MAX),retired:AtomicBool::new(false)});
    let funding=WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (WorkspaceContext::new_with_metadata_funding(NoEquations,funding).unwrap(),state)
}
fn config()->FamilyConfig {
    FamilyConfig::from_hf_json(br#"{
      "model_type":"gemma4","text_config":{"model_type":"gemma4_text",
      "hidden_size":8,"num_hidden_layers":4,"intermediate_size":12,
      "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
      "rms_norm_eps":0.00001,"vocab_size":16,"max_position_embeddings":64,
      "sliding_window":16,"num_kv_shared_layers":2,
      "layer_types":["full_attention","sliding_attention","full_attention","sliding_attention"]}}
    "#).unwrap()
}
struct Inspector;
impl PreparedInputInspector<WorkspaceTensor> for Inspector {
    fn identity(&self,value:&WorkspaceTensor)->Result<InputTensorIdentity,PreparedInputError>{
        InputTensorIdentity::new(TensorDtype::I32,value.shape().iter().map(|n|*n as usize).collect())
    }
    fn i32_values(&self,_:&WorkspaceTensor)->Result<Vec<i32>,CapabilityError>{panic!("no media")}
    fn bool_values(&self,_:&WorkspaceTensor)->Result<Vec<bool>,CapabilityError>{panic!("no media")}
}
#[test]
fn gemma_counted_ingress_preserves_order_and_retains_its_destination(){
    let plain=WorkspaceContext::new(NoEquations);
    let tokens=[3,2].into_iter().map(|n|WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1,n],WorkspaceDtype::Int32).unwrap(),&plain).unwrap());
    let parts=tokens.map(|t|PreparedInputPart::new(InputModality::Text,PreparedInputPayload::TokenIds(t),[]).unwrap()).collect();
    let input=PreparedModelInput::new(parts,|t|Inspector.identity(t)).unwrap();
    let admitted=crate::media_plan::admit_gemma4_input(&config(),&input,&Inspector).unwrap();
    let ordinary=prepare_composite_ingress::<WorkspaceBackend>(PreparedCompositeInput::new(&input,&admitted).unwrap(),&plain).unwrap();
    let (context,account)=context();
    let prepared=PreparedCompositeInput::new_with_metadata(&input,&admitted,&context).unwrap();
    let before=account.remaining.load(Ordering::SeqCst);
    let counted=prepare_composite_ingress::<WorkspaceBackend>(prepared,&plain).unwrap();
    assert!(account.remaining.load(Ordering::SeqCst)<before);
    let parts=counted.decoder_parts_with_metadata(Metadata::new(Some(&context))).unwrap();
    let lengths=|parts:&[DecoderInputPart<'_,WorkspaceTensor>]|parts.iter().map(|p|match p{
        DecoderInputPart::Text(tokens)=>tokens.dim(1),_=>panic!("same text modality")}).collect::<Vec<_>>();
    assert_eq!(lengths(&parts),[3,2]);assert_eq!(lengths(&parts),lengths(&ordinary.decoder_parts()));
    drop(parts);drop(context);drop(input);drop(admitted);drop(ordinary);
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(counted);assert!(account.retired.load(Ordering::SeqCst));
}
#[test]
fn gemma_counted_publications_replace_exact_policies_and_keep_failed_source(){
    let args=config();let (context,account)=context();
    let metadata=Metadata::new(Some(&context));
    let mut actual=SharedAttentionPublications::prepare(&args.text,metadata).unwrap();
    let mut ordinary=SharedAttentionStates::new();
    let mut policies=Vec::new();
    for i in 0..args.text.num_hidden_layers(){let p=args.text.layer_policy(i).unwrap();
        if matches!(p.key_value,eredu_nn::AttentionStateSource::Publish{..}){policies.push(p.attention);}}
    assert_eq!(policies.len(),2);assert!(actual.is_empty());
    for (i,policy) in policies.iter().enumerate(){let values=(13+i as i32,29+i as i32);
        actual.publish(*policy,values).unwrap();ordinary.publish(*policy,values).unwrap();}
    actual.publish(policies[0],(31,47)).unwrap();ordinary.publish(policies[0],(31,47)).unwrap();
    assert_eq!(actual.len(),ordinary.len());for (policy,value) in &ordinary{assert_eq!(actual.get(policy),Some(value));}
    let absent=AttentionPolicy::sliding(7).unwrap();
    assert!(!policies.contains(&absent));
    let error=actual.publish(absent,(53,59)).unwrap_err();
    assert!(error.to_string().contains("no declared publisher"));
    drop(actual);drop(context);assert!(!account.retired.load(Ordering::SeqCst));
    drop(error);assert!(account.retired.load(Ordering::SeqCst));
}


#[test]
fn gemma_paid_boundary_round_trip_keeps_exact_policies_and_receiver_custody() {
    use eredu_runtime::ArchitectureBoundary;
    let schema=TextBoundarySchema{hidden_size:8,per_layer_geometry:Some((4,3)),
        shared_geometry:vec![(AttentionPolicy::Full,1,4),(AttentionPolicy::sliding(16).unwrap(),1,4)]};
    let values=vec![17,-3,23,31,-41];
    let ordinary=schema.encode(schema.decode(values.clone()).unwrap()).unwrap();
    let (context,account)=context();
    let before=account.remaining.load(Ordering::SeqCst);
    let wire=schema.wire_schema_with_metadata(&context).unwrap();
    assert_eq!(wire,schema.wire_schema().unwrap());
    let resolved=wire.resolve_with_metadata(2,3,&context).unwrap();
    let copied=resolved.clone_with_metadata(&context).unwrap();
    assert_eq!(copied,resolved);
    assert_ne!(copied.primary().shape().as_ptr(),resolved.primary().shape().as_ptr());
    let decoded=schema.decode_with_metadata(values.clone(),&context).unwrap();
    assert_eq!(decoded.per_layer_input,Some(17));
    assert_eq!(decoded.shared.get(&AttentionPolicy::Full),Some(&(-3,23)));
    let encoded=schema.encode_with_metadata(decoded,&context).unwrap();
    for (actual,expected) in encoded.iter().zip(&ordinary) {
        assert_eq!(actual.role(),expected.role());assert_eq!(actual.tensor(),expected.tensor());
    }
    let receiver=schema.decode_with_metadata(values,&context).unwrap();
    assert!(account.remaining.load(Ordering::SeqCst)<before);
    assert_eq!(context.operation_count(),0);
    drop((wire,resolved,copied,encoded,ordinary));drop(context);
    assert!(!account.retired.load(Ordering::SeqCst));
    assert_eq!(receiver.shared.get(&AttentionPolicy::sliding(16).unwrap()),Some(&(31,-41)));
    drop(receiver);assert!(account.retired.load(Ordering::SeqCst));
    let (context,account)=self::context();account.remaining.store(0,Ordering::SeqCst);
    assert!(schema.decode_with_metadata(vec![19],&context).is_err());
    assert_eq!(account.remaining.load(Ordering::SeqCst),0);
}
