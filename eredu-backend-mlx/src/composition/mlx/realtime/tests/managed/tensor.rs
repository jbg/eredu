//! Two real local Ring ranks, sharing one Metal device, using the same frame driver.
use super::*;
use crate::tests::distributed_realtime_ring::{MODEL_WORKER_RANK,MODEL_WORKER_FIXTURE};

const PROTOCOL: u64 = 0x4d4f_5348_4954_5032;

const CASE:&str="composition::mlx::realtime::tests::managed::tensor::native_managed_realtime_tp2_matches_ordinary_across_forced_and_cached_frames";

fn selected(path:&Path,rank:usize)->MlxPreparedRealtimeExecution {
    let topology=eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2,1,1,1).unwrap(),rank).unwrap();
    let options=MlxLoadRequest::with_parallel(topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Gpu,0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,32,MlxLoadRequest::test_communication_completion_policy()).unwrap();
    MlxRealtimeExecutionContext::select_realtime_execution(prepare(path),&options,true).unwrap()
}

#[test]
#[ignore="requires Metal and two real local Ring processes; follows resident realtime admission"]
fn native_managed_realtime_tp2_matches_ordinary_across_forced_and_cached_frames() {
    let Some(rank)=std::env::var_os(MODEL_WORKER_RANK) else {
        let fixture=tempfile::tempdir().unwrap();
        write_tiny_native_artifact_with_values(fixture.path(),None,true);
        crate::tests::distributed_realtime_ring::run_model_parity_worker(
            fixture.path(),"moshi",false,CASE,true,std::time::Duration::from_secs(120));
        return;
    };
    let rank:usize=rank.to_string_lossy().parse().unwrap();
    let path=std::env::var_os(MODEL_WORKER_FIXTURE).unwrap();
    let path=Path::new(&path);
    let sampling=RealtimeSampling::greedy();
    let request=RequestId::new(943);
    let mut serial=load(path);
    let mut ordinary=selected_scheduler(&serial,request,sampling);
    let expected=inputs().into_iter().map(|frame|tokens(&drive_selected_frame(
        &mut serial,&mut ordinary,request,frame))).collect::<Vec<_>>();
    drop(ordinary);
    let SelectedTestModel{backend,model}=serial;
    drop(model);
    crate::backend::ordinary_retirement::reclaim();

    // Selection finishes before the real group is created. Reuse the actual
    // prepared Metal stream owner supplied by the ordinary public loader.
    let source=selected(path,rank);
    let native=safemlx::distributed::init(true,safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!((native.rank(),native.size()),(rank,2));
    let world=std::sync::Arc::new(crate::backend::runtime::distributed::Group::uncontracted(&native));
    let consensus = crate::backend::distributed::MlxRealtimeConsensusTransport::new(&native, backend.stream());
    let backend=backend.with_tensor_parallel_group(world);
    let model=backend.materialize_realtime_execution(source).unwrap();
    assert_eq!(model.selected().topology(),eredu_core::ParallelTopology::new(2,1,1,1).unwrap());
    assert!(model.executor().parallel_communication().is_some());
    let mut parallel=SelectedTestModel{backend,model};
    let mut ordinary=selected_scheduler(&parallel,request,sampling);
    crate::backend::runtime::distributed::reset_native_collective_submissions();
    for (index,frame) in inputs().into_iter().enumerate() {
        ordinary.enqueue(request, frame).unwrap();
        let actual = finish_distributed_frame(|| ordinary.run_distributed_turn(
            PROTOCOL, &consensus, std::time::Instant::now(),
            |_, frame, branch| parallel.backend.submit_realtime_frame(&mut parallel.model, frame, branch)));
        assert_eq!(tokens(&actual),expected[index],"ordinary TP rank {rank} frame {index}");
    }
    let ordinary_model_collectives=crate::backend::runtime::distributed::contracted_collective_submissions();
    assert!(ordinary_model_collectives>0,"ordinary TP must submit selected model collectives");
    assert_eq!(crate::backend::runtime::distributed::original_model_collective_submissions(),0,
        "ordinary TP must not consume an original frame occurrence");
    drop(ordinary);
    let SelectedTestModel{backend,model}=parallel;
    drop(model);
    crate::backend::ordinary_retirement::reclaim();

    let source=selected(path,rank);
    let model=backend.materialize_realtime_execution(source).unwrap();
    assert_eq!(model.selected().topology(),eredu_core::ParallelTopology::new(2,1,1,1).unwrap());
    assert!(model.executor().parallel_communication().is_some());
    let mut parallel=SelectedTestModel{backend,model};
    let mut managed=managed_scheduler(&parallel,request,sampling);
    crate::backend::runtime::distributed::reset_native_collective_submissions();
    for (index,frame) in inputs().into_iter().enumerate() {
        managed.enqueue(request, frame).unwrap();
        let actual = finish_distributed_frame(|| parallel.backend.run_realtime_distributed_bounded(
            &mut parallel.model, &mut managed, PROTOCOL,
            std::time::Instant::now(), 1, CAPACITY, None));
        assert_eq!(tokens(&actual),expected[index],"managed TP rank {rank} frame {index}");
        assert_eq!(managed.request_state(request).unwrap().generation().schedule_state().frontier(),index+1);
    }
    let managed_model_collectives=crate::backend::runtime::distributed::original_model_collective_submissions();
    assert_eq!(managed_model_collectives,ordinary_model_collectives,
        "the same four frames must execute the same model Sum/Gather occurrences; scheduler consensus is separate");
    assert_eq!(crate::backend::runtime::distributed::contracted_collective_submissions(),managed_model_collectives,
        "prepared model occurrences retain their selected communication contracts");
    eprintln!("managed realtime TP rank {rank}: four frames matched serial and ordinary TP");
}

fn finish_distributed_frame<E:std::fmt::Debug>(
    mut turn: impl FnMut() -> Result<eredu_core::scheduler::SchedulerProgress<RealtimeInputFrame,MlxPrepublicationFrame>,E>,
) -> RealtimeOutputFrame {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let mut progress = turn().unwrap();
        assert!(progress.failed.is_empty(), "distributed frame failed: {:?}", progress.failed);
        if let Some((_, _, output)) = progress.committed.pop() {
            return output.into_host_output().unwrap();
        }
        assert!(std::time::Instant::now() < deadline, "distributed frame did not finish");
        std::thread::yield_now();
    }
}
