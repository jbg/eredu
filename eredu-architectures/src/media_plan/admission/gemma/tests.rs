use super::*;
use eredu_core::{checkpoint::TensorDtype, PreparedInputError};
use eredu_nn::workspace::*;
use std::{convert::Infallible, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}};

#[derive(Debug)]
struct AccountState { remaining: Mutex<usize>, retired: AtomicBool }
#[derive(Debug)]
struct Account(Arc<AccountState>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let mut remaining=self.0.remaining.lock().unwrap();
        *remaining=remaining.checked_sub(bytes).ok_or(WorkspaceMetadataFundingError::Capacity {
            required:bytes as u64,available:*remaining as u64,
        })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {self.0.retired.store(true,Ordering::SeqCst);}
}
#[derive(Clone,Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{
        panic!("admission must not execute a tensor equation")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error=Infallible;
    fn operation_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{panic!("no equation")}
    fn write_operation_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{panic!("no equation")}
    fn host_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{panic!("no equation")}
    fn write_host_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{panic!("no equation")}
}
struct Inspector {changed:bool}
impl PreparedInputInspector<Vec<usize>> for Inspector {
    fn identity(&self,t:&Vec<usize>)->Result<InputTensorIdentity,PreparedInputError>{
        let mut shape=t.clone();
        if self.changed {shape[1]+=1;}
        InputTensorIdentity::new(if shape.len()==2 {TensorDtype::U32}else{TensorDtype::F32},shape)
    }
    fn identity_with_metadata(&self,t:&Vec<usize>,context:&WorkspaceContext)->Result<InputTensorIdentity,Error>{
        context.charge_metadata(std::mem::size_of::<(InputTensorIdentity,Result<InputTensorIdentity,PreparedInputError>)>())?;
        let mut shape=context.metadata_vec(t.len())?;
        shape.extend_from_slice(t);
        if self.changed {shape[1]+=1;}
        InputTensorIdentity::new(if shape.len()==2 {TensorDtype::U32}else{TensorDtype::F32},shape)
            .map_err(|e|context.metadata_source(e))
    }
    fn i32_values(&self,_:&Vec<usize>)->Result<Vec<i32>,CapabilityError>{panic!("no media metadata")}
    fn bool_values(&self,_:&Vec<usize>)->Result<Vec<bool>,CapabilityError>{panic!("no media metadata")}
}
fn config()->FamilyConfig {
    FamilyConfig::from_hf_json(br#"{
      "model_type":"gemma4",
      "text_config":{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":2,
        "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":4,"rms_norm_eps":0.00001,"vocab_size":16,
        "max_position_embeddings":64,"layer_types":["full_attention","full_attention"]}
    }"#).unwrap()
}
fn input(parts:Vec<(InputModality,bool,Vec<usize>)>)->PreparedModelInput<Vec<usize>> {
    let parts=parts.into_iter().map(|(modality,embeddings,shape)|{
        PreparedInputPart::new(modality,if embeddings {
            eredu_runtime::PreparedInputPayload::Embeddings(shape)
        }else{eredu_runtime::PreparedInputPayload::TokenIds(shape)},[]).unwrap()
    }).collect();
    PreparedModelInput::new(parts,|t|Inspector{changed:false}.identity(t)).unwrap()
}
fn context()->(WorkspaceContext,Arc<AccountState>){
    let state=Arc::new(AccountState{remaining:Mutex::new(usize::MAX),retired:AtomicBool::new(false)});
    let funding=WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (WorkspaceContext::new_with_metadata_funding(NoEquations,funding).unwrap(),state)
}

#[test]
fn gemma_counted_admission_preserves_ordered_positions_and_exact_input_identity(){
    let args=config();
    let input=input(vec![(InputModality::Text,false,vec![1,3]),(InputModality::Text,true,vec![1,2,8])]);
    let expected=crate::media_plan::admit_gemma4_input(&args,&input,&Inspector{changed:false}).unwrap();
    let (context,state)=context();
    let before=*state.remaining.lock().unwrap();
    let actual=admit(&args,&input,&Inspector{changed:false},&context).unwrap();
    assert_eq!(actual.decoder_shape(),[1,5]);
    assert_eq!(actual.parts(),expected.parts());
    assert_eq!(actual.identity(),expected.identity());
    assert!(*state.remaining.lock().unwrap()<before);
    let error=admit(&args,&input,&Inspector{changed:true},&context).err().unwrap();
    assert!(error.to_string().contains("identity changed"));
    // Metadata errors charge their source; the enclosing caller keeps H, just
    // as the native speculative source-retained error envelope does.
    let escaped=(error,context.metadata_funding().unwrap());
    drop(actual);drop(expected);drop(context);
    assert!(!state.retired.load(Ordering::SeqCst),"escaped rejection retains its paid source");
    drop(escaped);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn gemma_counted_admission_matches_shape_refusal_and_refuses_before_unfunded_construction(){
    let args=config();
    for shape in [vec![2,3,8],vec![1,3,7],vec![1,3]] {
        let input=input(vec![(InputModality::Text,true,shape)]);
        let expected=crate::media_plan::admit_gemma4_input(&args,&input,&Inspector{changed:false}).err().unwrap();
        let (context,_)=context();
        let actual=admit(&args,&input,&Inspector{changed:false},&context).err().unwrap();
        assert_eq!(actual.to_string(),expected.to_string());
    }
    let input=input(vec![(InputModality::Text,false,vec![1,3])]);
    let (context,state)=context();
    *state.remaining.lock().unwrap()=0;
    let error=admit(&args,&input,&Inspector{changed:false},&context).err().unwrap();
    assert!(error.into_metadata_funding_error().is_ok());
    assert_eq!(*state.remaining.lock().unwrap(),0);
}
