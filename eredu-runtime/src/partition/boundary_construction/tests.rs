use super::*;
use eredu_nn::workspace::*;
use std::{convert::Infallible,sync::{Arc,Mutex,atomic::{AtomicBool,Ordering}}};
#[derive(Debug)]
struct Account {remaining:Arc<Mutex<usize>>,retired:Arc<AtomicBool>}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
        let mut available=self.remaining.lock().unwrap();
        let next=available.checked_sub(bytes).ok_or(HostMetadataFundingError::Capacity{
            required:bytes as u64,available:*available as u64})?;
        *available=next;Ok(())
    }
}
impl Drop for Account {fn drop(&mut self){self.retired.store(true,Ordering::SeqCst);}}
#[derive(Debug)]struct NoTensorFacts;
impl WorkspaceMechanisms for NoTensorFacts {
    fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{Ok(None)}
}
impl WorkspaceFactMechanisms for NoTensorFacts {
    type Error=Infallible;
    fn operation_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{Ok(None)}
    fn write_operation_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{Ok(None)}
    fn host_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{Ok(None)}
    fn write_host_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{Ok(None)}
}
fn context()->(WorkspaceContext,Arc<Mutex<usize>>,Arc<AtomicBool>){
    let remaining=Arc::new(Mutex::new(16*1024*1024));let retired=Arc::new(AtomicBool::new(false));
    let funding=HostMetadataFunding::new(Account{remaining:remaining.clone(),retired:retired.clone()}).unwrap();
    (WorkspaceContext::new_with_metadata_funding(NoTensorFacts,funding).unwrap(),remaining,retired)
}
fn spec(role:&str,width:i32,context:&WorkspaceContext)->BoundaryTensorSpec{
    BoundaryTensorSpec::new_with_metadata(role,&[BoundaryTensorDimension::Batch,
        BoundaryTensorDimension::Sequence,BoundaryTensorDimension::Fixed(width)],BoundaryTensorDtype::Activation,context).unwrap()
}
#[test]
fn paid_pipeline_schema_preserves_order_unicode_and_per_tensor_geometry(){
    let (context,remaining,retired)=context();let before=*remaining.lock().unwrap();
    let role=String::from("état/主");
    let primary=spec(&role,7,&context);
    let auxiliary=vec![spec("résidu/次",5,&context)];
    let ordinary=BoundaryWireSchema::new("séquence/予測",primary.clone(),auxiliary.clone()).unwrap();
    let paid=BoundaryWireSchema::from_owned_with_metadata("séquence/予測",primary,auxiliary,&context).unwrap();
    assert_eq!(paid,ordinary);
    let resolved=paid.resolve_each_with_metadata(2,&[3,11],&context).unwrap();
    assert_eq!(resolved,ordinary.resolve_each(2,[3,11]).unwrap());
    assert_eq!(resolved.primary().shape(),[2,3,7]);assert_eq!(resolved.auxiliary()[0].shape(),[2,11,5]);
    assert_ne!(resolved.primary.role.as_ptr(),role.as_ptr());
    assert_ne!(resolved.primary.role.as_ptr(),paid.primary.role.as_ptr());
    drop((role,paid,ordinary));
    assert_eq!(resolved.primary.role,"état/主");assert!(*remaining.lock().unwrap()<before);
    let no_aux=NoAuxiliaryBoundarySchema::new(13);
    assert_eq!(no_aux.wire_schema_with_metadata(&context).unwrap(),no_aux.wire_schema().unwrap());
    assert!(no_aux.encode_with_metadata::<i32>(NoAuxiliaryBoundary,&context).unwrap().is_empty());
    no_aux.decode_with_metadata::<i32>(vec![],&context).unwrap();
    let tagged=ArchitectureBoundaryValue::new_with_metadata("résidu/次",vec![17,-4,29],&context).unwrap();
    assert_eq!(tagged.tensor(),&[17,-4,29]);
    assert_eq!(context.operation_count(),0); // metadata completion is no native fit.
    drop((resolved,tagged));drop(context);assert!(retired.load(Ordering::SeqCst));
}
#[test]
fn paid_pipeline_schema_refuses_before_destinations_and_keeps_error_custody(){
    let (context,remaining,retired)=context();
    let no_aux=NoAuxiliaryBoundarySchema::new(9);
    let schema=no_aux.wire_schema_with_metadata(&context).unwrap();
    let before=*remaining.lock().unwrap();
    assert!(schema.resolve_each_with_metadata(2,&[],&context).is_err());
    assert!(schema.resolve_with_metadata(0,3,&context).is_err());
    assert!(no_aux.decode_with_metadata(vec![19i32],&context).is_err());
    assert!(*remaining.lock().unwrap()<before);
    let failure=BoundaryWireSchema::from_owned_with_metadata("dup",spec("主",7,&context),
        vec![spec("主",5,&context)],&context).unwrap_err();
    assert!(std::error::Error::source(&failure).is_some());
    // Low-level metadata destinations require the enclosing result/error owner
    // to retain their exact account. Keep the cause before that custody, as the
    // shared partition caller does; metadata_source itself carries no account.
    let failure=(failure,context.metadata_funding().unwrap());
    drop(schema);drop(context);assert!(!retired.load(Ordering::SeqCst));
    drop(failure);assert!(retired.load(Ordering::SeqCst));
    let (context,remaining,_)=self::context();let schema=no_aux.wire_schema_with_metadata(&context).unwrap();
    *remaining.lock().unwrap()=0;
    assert!(schema.resolve_with_metadata(2,3,&context).is_err());
    assert_eq!(*remaining.lock().unwrap(),0);assert_eq!(context.operation_count(),0);
}
