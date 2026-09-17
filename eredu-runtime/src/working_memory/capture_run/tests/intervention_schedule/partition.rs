use super::*;
use crate::capture::partition::*;
use crate::intervention::PreparedPartitionInterventionProjection;
use eredu_core::consensus::{ConsensusTransport,BoundedConsensusTransport};
use eredu_nn::workspace::{WorkspaceMetadataFunding,WorkspaceMetadataAccount,WorkspaceMetadataFundingError};
#[derive(Debug)]
struct Account;
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self,_bytes:usize)->Result<(),WorkspaceMetadataFundingError> {Ok(())}
}
struct Settled;
impl Completion for Settled {
    type Error=std::io::Error;
    fn is_complete(&self)->Result<bool,Self::Error> {Ok(true)}
    fn wait(&self)->Result<(),Self::Error> {Ok(())}
}
impl BoundedCompletion for Settled {
    fn wait_bounded(self,_wait:BoundedCompletionWait)->Result<BoundedCompletionOutcome,Self::Error> {Ok(BoundedCompletionOutcome::Completed)}
}
struct Transport {rank:usize,reject_peer:bool}
impl ConsensusTransport for Transport {
    type Error=std::io::Error;
    fn participant_count(&self)->usize {2}
    fn all_gather_words(&self,_words:&[u32])->Result<Vec<u32>,Self::Error> {panic!("unbounded transport")}
}
impl BoundedConsensusTransport for Transport {
    type Completion=Settled;
    type GatherOutput=Vec<u32>;
    fn submit_all_gather_words(&self,words:&[u32])->Result<Submission<Vec<u32>,Settled>,Self::Error> {
        assert_eq!(words.len(),18,"existing outcome protocol");
        let mut output=Vec::new();
        for rank in 0..2 {
            let mut peer=words.to_vec();peer[2]=rank as u32;
            if rank!=self.rank {peer[7]=if rank==0 {u32::from(!self.reject_peer)}else{2};}
            output.extend(peer);
        }
        Ok(Submission {output,completion:Settled})
    }
    fn resolve_all_gather_words(&self,output:Vec<u32>)->Result<Vec<u32>,Self::Error> {Ok(output)}
}
impl PartitionCaptureTransport for Transport {
    fn capture_rank(&self)->usize {self.rank}
    fn capture_wait(&self)->Result<BoundedCompletionWait,CaptureError> {
        Ok(BoundedCompletionWait::new(std::time::Duration::from_secs(1),CompletionCancellationMode::QuarantineUntilComplete).unwrap())
    }
    fn ensure_capture_active(&self)->Result<(),BackendFailure> {Ok(())}
    fn estimate_capture_gather(&self,words:usize)->Result<CaptureUsage,CaptureError> {
        Ok(CaptureUsage {retained_bytes:(words*8) as u64,host_bytes:(words*4) as u64,..Default::default()})
    }
    fn fail_capture_exchange(&self,_error:&PartitionCaptureExchangeError) {}
}
impl PartitionCaptureHookTransport for Transport {
    type HookOutput=();
    fn estimate_capture_hook(&self,members:&[usize])->Result<CaptureUsage,CaptureError> {
        assert_eq!(members,&[0]);Ok(CaptureUsage::default())
    }
    fn submit_capture_hook(&self,_members:&[usize],_success:bool)->Result<Submission<(),Settled>,Self::Error> {panic!("singleton or absent rank submitted member transport")}
    fn resolve_capture_hook(&self,_output:())->Result<bool,Self::Error> {panic!("singleton or absent rank resolved member transport")}
}
#[test]
fn partition_outcome_requires_original_member_completion_and_never_refunds_local_credits() {
    for (rank,rejected) in [(0,false),(1,false),(0,true),(1,true)] {
        let (capture,admitted)=sources();
        let pool=WorkingMemoryPool::new(1<<26,0).unwrap();
        let original=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
        let plan=CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&original).unwrap();
        let (reservation,run)=fresh(&pool,plan.initialization_peak_bytes());
        let native=run.scope().unwrap();
        let mut bank=run.prepare_capture_run(&reservation,plan).unwrap();
        let initial=finish(bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap());
        let mut frame=bank.begin_step(CapturePhase::Decode,1).unwrap().prepare().unwrap();
        let baseline=frame.interventions()[0].charged;
        let funding=WorkspaceMetadataFunding::new(Account).unwrap();
        let projection=PreparedPartitionInterventionProjection::prepare(&original,0,CapturePhase::Decode,1,None,
            &[1,4],1,&eredu_core::component::ComponentCoordinateMap::range(4,0..4).unwrap(),None,4,funding.clone()).unwrap();
        let usage=CaptureUsage {retained_bytes:32,host_bytes:16,..Default::default()};
        let projected=[CaptureUsage {host_bytes:24,..Default::default()};2];
        let dependencies=CaptureUsage {retained_bytes:64,host_bytes:8,..Default::default()};
        let member=PartitionInterventionMemberSource {rank:0,projection:&projection,shape:&[1,4],execution_identity:[7;32],
            usage,projection_usage:projected,source_usage:dependencies};
        let source=PreparedPartitionInterventionSource::new(&original,0,CapturePhase::Decode,1,2,
            &[PartitionInterventionInvocationSource {window:None,members:&[member]}],&funding).unwrap();
        let transport=Transport {rank,reject_peer:rejected};
        let mut ledger=CaptureLedger::new(capture.admission());
        let mut work=source.prepare(&transport,DistributedCommitEpoch::FIRST,&mut ledger).unwrap();
        let global=work.global_reserved();assert_eq!(ledger.total(),global);
        if rank==0 {
            let claim=frame.take_intervention(0).unwrap();claim.validate_native_custody(&native).unwrap();
            let mut loan=work.begin(None,true).unwrap().unwrap();
            assert!(loan.charge_execution(usage).is_err(),"source validation precedes spending");
            loan.validate(&claim,&projection,None,[7;32],&[1,4],usage,projected,dependencies).unwrap();
            let local=loan.reserved();loan.charge_projection(projected).unwrap();loan.charge_source(dependencies).unwrap();
            loan.charge_execution(usage).unwrap();assert!(loan.charge_execution(usage).is_err());
            if rejected {drop(loan);drop(claim);} else {
                work.finish(loan,true).unwrap();frame.record_intervention(claim.finish(local).unwrap()).unwrap();
            }
        } else {
            assert!(work.begin(None,true).unwrap().is_none());
            assert_eq!(frame.interventions()[0].outcome,InterventionOutcome::Missing,"absent rank has no local edit result");
        }
        assert_eq!(ledger.total(),global,"local child spending must not recharge the parent");
        if rejected {
            assert!(work.deliver(baseline).is_err());
            assert_eq!(frame.interventions()[0].outcome,InterventionOutcome::Missing);
            drop(frame);
        } else {
            let receipt=work.deliver(baseline).unwrap();frame.record_partition_intervention(receipt).unwrap();
            assert_eq!(frame.interventions()[0].charged,baseline.checked_add(global).unwrap());
            assert_eq!(frame.interventions()[0].outcome,InterventionOutcome::Applied);
            assert!(frame.take_intervention(0).is_err());assert!(work.deliver(baseline).is_err());
            drop(finish(frame));
        }
        drop(work);assert_eq!(ledger.total(),global,"dropping original work never refunds its ledger");
        drop((initial,bank,reservation,run,projection,original,funding));native.certify().unwrap();
    }
}
