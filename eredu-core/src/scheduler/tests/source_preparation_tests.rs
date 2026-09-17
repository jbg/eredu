//! Source admission reuses the exact distributed transaction and custody rules.
use super::*;

#[derive(Debug)]
struct SourceState { value: u32, discarded: Rc<Cell<usize>> }
impl SemanticStateTransaction for SourceState {
    type Branch = (u32, Rc<Cell<usize>>, bool);
    type Error = std::io::Error;
    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        panic!("prepared distributed turn must use its actual source callback")
    }
    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.value = branch.0;
        Ok(())
    }
    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        branch.1.set(branch.1.get() + 1);
        if branch.2 { Err(std::io::Error::other("secondary discard failure")) } else { Ok(()) }
    }
}

#[test]
fn distributed_source_refusal_discards_partial_admission_and_preserves_canonical_state() {
    for local_refusal in [true, false] {
        let discarded = Rc::new(Cell::new(0));
        let mut scheduler = Scheduler::<u32,SourceState,Output>::new(
            SchedulerLimits::with_execution_bounds(2,2,2,2,1,usize::MAX).unwrap()).unwrap();
        let requests = [RequestId::new(701), RequestId::new(702)];
        let mut work = Vec::new();
        for request in requests {
            scheduler.register(request,SourceState{value:11,discarded:discarded.clone()}).unwrap();
            work.push(scheduler.enqueue(request,3).unwrap());
        }
        let transport = ScriptedTransport::new(2,vec![
            GatherStep::default(), GatherStep::default(),
            GatherStep { replacements: if local_refusal { vec![] } else { vec![(1,7,0)] } },
        ]);
        let prepared = Cell::new(0);
        let dispatched = Cell::new(0);
        let error = scheduler.run_distributed_bounded_with_preparation(
            0x534f_5552,&transport,consensus_wait(),Instant::now(),2,
            |id, value, state| {
                prepared.set(prepared.get()+1);
                if local_refusal && id.request()==requests[1] {
                    Err(std::io::Error::other("original source refusal"))
                } else { Ok((state.value+*value,state.discarded.clone(),local_refusal)) }
            },
            |_,_,_| {
                dispatched.set(dispatched.get()+1);
                Ok::<_,Infallible>(Output { complete:Rc::new(Cell::new(true)),fail:false })
            },
        ).unwrap_err();
        if local_refusal { assert_eq!(error,SchedulerError::State("original source refusal".into())); }
        assert_eq!(prepared.get(),2);
        assert_eq!(dispatched.get(),0);
        assert_eq!(discarded.get(),if local_refusal {1}else{2});
        assert!(scheduler.report().poisoned);
        assert_eq!(scheduler.report().completed_work,0);
        for (request,work) in requests.into_iter().zip(work) {
            assert_eq!(scheduler.request_state(request).unwrap().value,11);
            assert_eq!(scheduler.queued_for_request(request),1);
            assert_eq!(scheduler.work_lifecycle(work),Some(WorkLifecycle::Queued));
        }
        assert!(transport.steps.borrow().is_empty(),"all ranks reached preparation agreement");
    }
}

#[test]
fn distributed_source_success_commits_only_after_existing_completion_consensus() {
    let discarded = Rc::new(Cell::new(0));
    let mut scheduler = Scheduler::<u32,SourceState,Output>::new(SchedulerLimits::default()).unwrap();
    let request = RequestId::new(703);
    scheduler.register(request,SourceState{value:11,discarded:discarded.clone()}).unwrap();
    scheduler.enqueue(request,3).unwrap();
    let transport = ScriptedTransport::new(2,vec![]);
    let ready = Rc::new(Cell::new(false));
    let prepared = Cell::new(0);
    for complete in [false,true] {
        ready.set(complete);
        let progress = scheduler.run_distributed_bounded_with_preparation(
            0x534f_5552,&transport,consensus_wait(),Instant::now(),1,
            |_,value,state| {
                prepared.set(prepared.get()+1);
                Ok::<_,Infallible>((state.value+*value,state.discarded.clone(),false))
            },
            |_,_,_| Ok::<_,Infallible>(Output { complete:ready.clone(),fail:false }),
        ).unwrap();
        assert_eq!(progress.committed.len(),usize::from(complete));
        assert_eq!(scheduler.request_state(request).unwrap().value,if complete {14}else{11});
    }
    assert_eq!(prepared.get(),1);
    assert_eq!(discarded.get(),0);
}

#[derive(Debug,thiserror::Error)]
#[error("original completion source {code}")]
struct OriginalCompletionFailure { code:u32 }
struct FailingSourceOutput { polls:Rc<Cell<u32>> }
impl TransitionOutput for FailingSourceOutput {
    type Error=OriginalCompletionFailure;
    fn is_complete(&self)->Result<bool,Self::Error> {
        let code=self.polls.get()+1;self.polls.set(code);
        Err(OriginalCompletionFailure{code})
    }
    fn retained_resources(&self)->usize {1}
}
impl DistributedTransitionOutput for FailingSourceOutput {
    fn encode_distributed_output(&self,_:&mut Vec<u32>)->Result<(),String> {
        panic!("failed original source cannot publish a descriptor")
    }
}

#[test]
fn distributed_completion_source_survives_pending_rank_and_final_resolution() {
    let discarded=Rc::new(Cell::new(0));
    let mut scheduler=Scheduler::<u32,SourceState,FailingSourceOutput>::new(SchedulerLimits::default()).unwrap();
    let request=RequestId::new(704);
    scheduler.register(request,SourceState{value:11,discarded:discarded.clone()}).unwrap();
    let work=scheduler.enqueue(request,3).unwrap();
    let transport=ScriptedTransport::new(2,vec![]);
    let polls=Rc::new(Cell::new(0));
    let mut retained=None;
    for turn in 0..3 {
        if turn==1 {
            // Deadline agreement then the exact output-completion frame.
            // Local Failed(2) with remote Incomplete(0) must retain the output.
            transport.steps.borrow_mut().extend([
                GatherStep::default(),GatherStep{replacements:vec![(1,7,0)]},
            ]);
        }
        let progress=scheduler.run_distributed_bounded_with_preparation_and_errors(
            0x534f_5552,&transport,consensus_wait(),Instant::now(),1,
            |_,value,state|Ok::<_,Infallible>((state.value+*value,state.discarded.clone(),false)),
            |_,_,_|Ok::<_,Infallible>(FailingSourceOutput{polls:polls.clone()}),
            |id,cause|{
                assert_eq!(id,work);assert_eq!(discarded.get(),0);
                if retained.is_none(){retained=Some(crate::BackendFailure::from_error(cause));}
            },
        ).unwrap();
        assert!(progress.committed.is_empty());
        assert_eq!(scheduler.report().completed_work,0);
        if turn<2 {
            assert!(progress.failed.is_empty());
            assert_eq!(scheduler.request_state(request).unwrap().value,11);
            assert_eq!(discarded.get(),0);
        } else {
            assert_eq!(progress.failed.len(),1);
            assert_eq!(discarded.get(),1);
        }
        if turn!=0 {
            let source=std::error::Error::source(retained.as_ref().unwrap()).unwrap();
            assert_eq!(source.downcast_ref::<OriginalCompletionFailure>().unwrap().code,1,
                "later polling failure must not replace the first original source");
        }
    }
    assert_eq!(polls.get(),2);
    assert_eq!(scheduler.work_lifecycle(work),Some(WorkLifecycle::Failed));
}
