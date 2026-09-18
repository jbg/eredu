use super::*;
use eredu_nn::workspace::{HostMetadataAccount,HostMetadataFunding,HostMetadataFundingError};
use std::sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}};
#[derive(Debug)]
struct Account {live:Arc<AtomicBool>,used:Arc<AtomicUsize>,limit:Arc<AtomicUsize>}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
        self.used.fetch_update(Ordering::SeqCst,Ordering::SeqCst,|used|
            used.checked_add(bytes).filter(|&next|next<=self.limit.load(Ordering::SeqCst)))
            .map(|_|()).map_err(|_|HostMetadataFundingError::Unavailable)
    }
}
impl Drop for Account {fn drop(&mut self){self.live.store(false,Ordering::SeqCst);}}
struct SourceOwner(Arc<AtomicBool>);
impl Drop for SourceOwner {fn drop(&mut self){self.0.store(false,Ordering::SeqCst);}}
#[derive(Debug)]
struct Payload {values:[i32;8],source:Arc<AtomicBool>,funding:Arc<AtomicBool>,drops:Arc<AtomicUsize>}
impl Drop for Payload {
    fn drop(&mut self){
        assert!(self.source.load(Ordering::SeqCst));assert!(self.funding.load(Ordering::SeqCst));
        self.drops.fetch_add(1,Ordering::SeqCst);
    }
}
fn prepared_source()->RetainedCommunicationSource {
    let transfer=CommunicationOperationRequirement::tensors(CommunicationOperation::SendReceive,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,2,32,None).unwrap(),true).unwrap();
    let contract=RoleExactBoundaryContract::new("séquence/边界",[
        BoundaryRoleContract::symbolic("résidu/主",TensorDtype::F32,vec![
            BoundaryDimensionContract::Variable{maximum:8},BoundaryDimensionContract::Fixed(4)]).unwrap(),
    ]).unwrap();
    let route=CommunicationRouteDescriptor::new(CommunicationRouteId::new(19),0,0,1,transfer.clone()).unwrap()
        .with_boundary_contract(contract).unwrap();
    let completion=CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete).unwrap();
    let mut proposals=proposals();
    for (rank,proposal) in proposals.iter_mut().enumerate(){
        proposal.manifest=CommunicationManifest::new(2,rank,vec![],vec![route.clone()]).unwrap()
            .with_completion_policy(completion);
    }
    let agreed=establish_communication_session(&transport(0,&proposals),&proposals[0].manifest,
        proposals[0].nonce).unwrap();
    let capability=CommunicationCapabilities::new([transfer]).unwrap()
        .with_completion_capabilities(CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete]).unwrap())
        .with_boundary_framing([BoundaryFramingProtocol::RoleExactV1]).unwrap();
    prepare_communication_realization(&proposals[0].manifest,agreed.manifests(),&capability,
        CommunicationTopologyCapabilities::RingWithWorldWaves).unwrap()
        .into_retained_source(agreed.identity()).unwrap()
}
#[test]
fn prepared_boundary_headers_preserve_unicode_wire_and_keep_source_after_payload_escape(){
    let source_live=Arc::new(AtomicBool::new(true));let funding_live=Arc::new(AtomicBool::new(true));
    let used=Arc::new(AtomicUsize::new(0));let limit=Arc::new(AtomicUsize::new(usize::MAX));
    let drops=Arc::new(AtomicUsize::new(0));
    let funding=HostMetadataFunding::new(Account{live:funding_live.clone(),used:used.clone(),limit:limit.clone()}).unwrap();
    let host=eredu_core::HostPreparationAuthority::retain(SourceOwner(source_live.clone()));
    let source=prepared_source().with_host_preparation(&host).unwrap();
    let selected=source.prepare_boundary_source(CommunicationRouteId::new(19),&funding).unwrap();
    let other=prepared_source();
    assert!(!selected.source().same_source(&other));
    let roles=[BoundaryRoleContract::new("résidu/主",TensorDtype::F32,vec![2,4]).unwrap()];
    let ordinary=selected.descriptor().boundary_contract().unwrap()
        .frame_values(CommunicationRouteId::new(19),&roles,vec![[11,-23,37,41,-59,67,71,-83]]).unwrap();
    let frames=selected.frame_values(&roles,vec![Payload{values:[11,-23,37,41,-59,67,71,-83],
        source:source_live.clone(),funding:funding_live.clone(),drops:drops.clone()}]).unwrap();
    assert_eq!(frames.values()[0].header(),ordinary[0].header());
    assert_eq!(frames.values()[0].tensor().values,[11,-23,37,41,-59,67,71,-83]);
    let header=frames.values()[0].header();
    assert_eq!(&header[..8],b"EREDUBND");
    assert!(header.windows("séquence/边界".len()).any(|value|value=="séquence/边界".as_bytes()));
    assert!(header.windows("résidu/主".len()).any(|value|value=="résidu/主".as_bytes()));
    assert_eq!(u64::from_le_bytes(header[header.len()-8..].try_into().unwrap()),32);
    let spent=used.load(Ordering::SeqCst);assert!(spent>header.len());
    drop((source,other,selected,host,funding,ordinary));
    assert!(source_live.load(Ordering::SeqCst)&&funding_live.load(Ordering::SeqCst));
    assert_eq!(used.load(Ordering::SeqCst),spent);
    drop(frames);assert_eq!(drops.load(Ordering::SeqCst),1);
    assert!(!source_live.load(Ordering::SeqCst)&&!funding_live.load(Ordering::SeqCst));
}
#[test]
fn prepared_boundary_refusal_spends_no_retry_credit_and_retains_error_source(){
    let source_live=Arc::new(AtomicBool::new(true));let funding_live=Arc::new(AtomicBool::new(true));
    let used=Arc::new(AtomicUsize::new(0));let limit=Arc::new(AtomicUsize::new(usize::MAX));
    let funding=HostMetadataFunding::new(Account{live:funding_live.clone(),used:used.clone(),limit:limit.clone()}).unwrap();
    let host=eredu_core::HostPreparationAuthority::retain(SourceOwner(source_live.clone()));
    let source=prepared_source().with_host_preparation(&host).unwrap();
    let selected=source.prepare_boundary_source(CommunicationRouteId::new(19),&funding).unwrap();
    let invalid=[BoundaryRoleContract::new("résidu/主",TensorDtype::F32,vec![9,4]).unwrap()];
    let before=used.load(Ordering::SeqCst);
    let error=selected.frame_values(&invalid,vec![13]).unwrap_err();
    assert!(used.load(Ordering::SeqCst)>before,"failed checked call keeps its paid constructor attempt");
    let spent=used.load(Ordering::SeqCst);limit.store(spent,Ordering::SeqCst);
    let valid=[BoundaryRoleContract::new("résidu/主",TensorDtype::F32,vec![1,4]).unwrap()];
    let refused=selected.frame_values(&valid,vec![17]).unwrap_err();
    assert_eq!(used.load(Ordering::SeqCst),spent,"refused reservation is nonmutating");
    drop((source,selected,host,funding,error));
    assert!(source_live.load(Ordering::SeqCst)&&funding_live.load(Ordering::SeqCst));
    drop(refused);assert!(!source_live.load(Ordering::SeqCst)&&!funding_live.load(Ordering::SeqCst));
}
