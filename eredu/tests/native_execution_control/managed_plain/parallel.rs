//! Actual public TP startup, model execution and controlled advancement.
//! Each mode/rank is a fresh process with a real local Ring communicator.
use super::*;
use std::{io::Read, net::TcpListener, process::{Child,Command,Stdio},time::{Duration,Instant}};

const CASE: &str = "managed_plain::parallel::native_managed_tensor_parallel_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_MANAGED_TP_MODE";
const RESULT: &str = "PUBLIC_MANAGED_TP_RESULT:";

fn run(mode:&str)->serde_json::Value {
    run_partitioned(mode,eredu_core::ParallelTopology::new(2,1,1,1).unwrap())
}
fn run_partitioned(mode:&str,topology:eredu_core::ParallelTopology)->serde_json::Value {
    run_partitioned_with_lifecycle(mode, topology, None)
}
fn run_partitioned_with_lifecycle(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>)
    ->serde_json::Value {
    run_partitioned_with_fixture(mode, topology, lifecycle, || fixture(false))
}
fn run_partitioned_with_fixture(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture)->serde_json::Value {
    run_partitioned_with_state(mode, topology, lifecycle, fixture_source,
        eredu_runtime::CacheResidencyPolicy::Device)
}
fn run_partitioned_with_fixture_settings(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture,generation:PreparedChatGenerationSettings)->serde_json::Value {
    run_partitioned_with_load_on_and_settings(mode,topology,lifecycle,fixture_source,
        eredu_runtime::NormalizedLoadRequest::default(),safemlx::DeviceType::Gpu,generation)
}
fn run_partitioned_with_state(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture,state:eredu_runtime::CacheResidencyPolicy)->serde_json::Value {
    run_partitioned_with_load(mode,topology,lifecycle,fixture_source,
        eredu_runtime::NormalizedLoadRequest::default().with_state_residency(state))
}
fn run_partitioned_with_load(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture,load:eredu_runtime::NormalizedLoadRequest)->serde_json::Value {
    run_partitioned_with_load_on(mode, topology, lifecycle, fixture_source, load, safemlx::DeviceType::Gpu)
}
fn run_partitioned_with_load_on(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture,load:eredu_runtime::NormalizedLoadRequest,
    device:safemlx::DeviceType)->serde_json::Value {
    run_partitioned_with_load_on_and_settings(mode,topology,lifecycle,fixture_source,load,device,settings(0.0))
}
fn run_partitioned_with_load_on_and_settings(mode:&str,topology:eredu_core::ParallelTopology,
    lifecycle:Option<fn(LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,Fixture)->serde_json::Value>,
    fixture_source:fn()->Fixture,load:eredu_runtime::NormalizedLoadRequest,
    device:safemlx::DeviceType,generation:PreparedChatGenerationSettings)->serde_json::Value {
    // The serial resident result is the independent numerical oracle for the
    // selected parallel placement, including paged cache state.
    if mode == "serial" { return run_mode("ordinary",fixture_source(),0.0); }
    let world=safemlx::distributed::Group::init(true,safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(),topology.world_size());
    let backend=eredu_backend_mlx::native::prepared_distributed_backend_on(&world, device).unwrap_or_else(report_failure)
        .expect("qualified prepared distributed backend");
    let topology=eredu_core::ParallelRankTopology::new(
        topology,world.rank()).unwrap();
    let options=eredu_backend_mlx::MlxLoadRequest::from_normalized(
        load,
    ).with_parallel_topology(topology,
        eredu_backend_mlx::native::DeviceAssignment::new(device,0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),1,32,
        eredu_runtime::CommunicationCompletionPolicy::new(Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete).unwrap()).unwrap();
    let root=managed_fixture(fixture_source());
    let mut model=LoadedModel::load(backend,&root.0,options).unwrap_or_else(report_failure);
    if let Some(lifecycle)=lifecycle { return lifecycle(model, root); }
    let settings=generation;
    if mode == "ordinary" {
        let prompt=model.encode(PROMPT,false).unwrap();
        let config=model.resolve_generation_config(settings.overrides).unwrap();
        let config=TextGenerationConfig::new(config).with_seed(settings.seed)
            .with_inference_policy(TextInferencePolicy{prefill_chunk_positions:NonZeroU64::new(2),..Default::default()});
        let ids:Vec<u32>=model.generate_tokens(prompt,config).unwrap_or_else(report_failure)
            .map(|token|token.unwrap_or_else(report_failure).token_id().unwrap()).collect();
        assert_eq!(ids.len(),4);
        return serde_json::json!({"ids":ids,"text":model.decode(&ids,true).unwrap()});
    }
    assert!(matches!(mode,"managed"|"controlled"));
    let source=model.compile_managed_plain_text_source(std::fs::File::open(root.0.join("tokenizer.json")).unwrap())
        .unwrap_or_else(report_failure);
    let cancellation=GenerationCancellationToken::new();
    let mut visible=String::new();let mut finishes=Vec::new();
    let mut emit=|event:GenerationPlainTextEvent<'_>|match event{
        GenerationPlainTextEvent::TextDelta(text)=>visible.push_str(text),
        GenerationPlainTextEvent::Finished{reason}=>finishes.push(reason),
    };
    let request=ManagedPlainTextRequest::new(PROMPT,settings);
    let output=if mode=="controlled" {
        let mut session=model.start_managed_plain_text(&source,request,&cancellation)
            .unwrap_or_else(report_failure).expect("live controlled request");
        while session.finish_reason().is_none(){
            session=session.advance(&cancellation,&mut emit).unwrap_or_else(report_failure);
        }
        session.into_output().unwrap_or_else(|_|panic!("terminal controlled request"))
    } else {
        model.generate_managed_plain_text(&source,request,&cancellation,&mut emit)
            .unwrap_or_else(report_failure).expect("live managed request")
    };
    assert_eq!(output.finish_reason,FinishReason::MaxTokens);
    assert_eq!(finishes,[FinishReason::MaxTokens]);
    assert_eq!(output.token_ids.as_ref().len(),4);
    assert_eq!(output.text.as_str(),visible);
    let address=output.text.as_str().as_ptr();
    drop(source);drop(model);
    assert_eq!(address,output.text.as_str().as_ptr());
    serde_json::json!({"ids":output.token_ids.as_ref(),"text":output.text.as_str()})
}
struct Workers(Vec<(Child,std::thread::JoinHandle<Vec<u8>>,std::thread::JoinHandle<Vec<u8>>)>);
impl Drop for Workers {
    fn drop(&mut self){for (child,_,_) in &mut self.0{let _=child.kill();let _=child.wait();}}
}
fn result(bytes:&[u8],marker:&str)->serde_json::Value{
    String::from_utf8_lossy(bytes).lines().find_map(|line|line.strip_prefix(marker))
        .map(|line|serde_json::from_str(line).unwrap()).expect("public TP result marker")
}
#[test]
#[ignore="requires Metal and two local Ring processes"]
fn native_managed_tensor_parallel_matches_ordinary_and_controlled(){
    compare_modes(CASE,MODE,RESULT,"TP",run)
}
fn compare_modes(case:&str,mode_env:&str,marker:&str,label:&str,run:fn(&str)->serde_json::Value){
    compare_modes_with_world(case,mode_env,marker,label,2,run)
}
fn compare_modes_with_world(case:&str,mode_env:&str,marker:&str,label:&str,world:usize,run:fn(&str)->serde_json::Value){
    compare_selected_modes(case, mode_env, marker, label, world,
        &["ordinary", "managed", "controlled"], run)
}
fn compare_selected_modes(case:&str,mode_env:&str,marker:&str,label:&str,world:usize,
    modes:&[&str],run:fn(&str)->serde_json::Value){
    compare_selected_modes_by(case, mode_env, marker, label, world, modes, run,
        |actual, expected, mode, rank| assert_eq!(actual, expected, "{mode} rank{rank}"))
}
fn compare_selected_modes_by(case:&str,mode_env:&str,marker:&str,label:&str,world:usize,
    modes:&[&str],run:fn(&str)->serde_json::Value,
    compare:fn(&serde_json::Value,&serde_json::Value,&str,usize)){
    if let Ok(mode)=std::env::var(mode_env){println!("\n{marker}{}",run(&mode));return;}
    let executable=std::env::current_exe().unwrap();
    let serial=Command::new(&executable).args(["--exact",case,"--ignored","--nocapture"])
        .env(mode_env,"serial").output().unwrap();
    assert!(serial.status.success(),"serial: {}\n{}",String::from_utf8_lossy(&serial.stdout),String::from_utf8_lossy(&serial.stderr));
    let expected=result(&serial.stdout,marker);
    for &mode in modes {
        // A fresh hostfile/group for each mode avoids cross-run native authority.
        let scratch=fixture(false);
        let sockets:Vec<TcpListener>=(0..world).map(|_|TcpListener::bind(("127.0.0.1",0)).unwrap()).collect();
        let hosts:Vec<_>=sockets.iter().map(|socket|vec![format!("127.0.0.1:{}",socket.local_addr().unwrap().port())]).collect();
        let hostfile=scratch.0.join("ring-hosts.json");
        std::fs::write(&hostfile,serde_json::to_vec(&hosts).unwrap()).unwrap();drop(sockets);
        let mut workers=Workers(Vec::new());
        for rank in 0..world {
            let mut child=Command::new(&executable).args(["--exact",case,"--ignored","--nocapture"])
                .env(mode_env,mode).env("MLX_RANK",rank.to_string()).env("MLX_HOSTFILE",&hostfile)
                .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
            let mut out=child.stdout.take().unwrap();let mut err=child.stderr.take().unwrap();
            let stdout=std::thread::spawn(move||{let mut bytes=Vec::new();out.read_to_end(&mut bytes).unwrap();bytes});
            let stderr=std::thread::spawn(move||{let mut bytes=Vec::new();err.read_to_end(&mut bytes).unwrap();bytes});
            workers.0.push((child,stdout,stderr));
        }
        let deadline=Instant::now()+Duration::from_secs(120);
        loop {
            if workers.0.iter_mut().all(|(child,_,_)|child.try_wait().unwrap().is_some()){break;}
            assert!(Instant::now()<deadline,"{mode}: bounded public {label} worker deadline");
            std::thread::sleep(Duration::from_millis(20));
        }
        let results:Vec<_>=workers.0.drain(..).enumerate().map(|(rank,(mut child,stdout,stderr))| {
            (rank,child.wait().unwrap(),stdout.join().unwrap(),stderr.join().unwrap())
        }).collect();
        // Report every failed rank before asserting: a lower rank may carry
        // only propagated rejection while the actual cause belongs to a peer.
        for (rank,status,stdout,stderr) in &results {
            if !status.success() {
                eprintln!("{mode} rank{rank}: {}\n{}",String::from_utf8_lossy(stdout),String::from_utf8_lossy(stderr));
            }
        }
        assert!(results.iter().all(|(_,status,_,_)|status.success()),"{mode}: public {label} rank failures reported above");
        for (rank,_,stdout,_) in results {
            let actual=result(&stdout,marker);compare(&actual,&expected,mode,rank);
            eprintln!("{marker}{mode} rank{rank}: {actual}");
        }
    }
}

#[path = "parallel/pipeline.rs"]
mod pipeline;

#[path = "parallel/combined.rs"]
mod combined;

#[path = "parallel/saved.rs"]
mod saved;

#[path = "parallel/larger.rs"]
mod larger;

#[path = "parallel/paged.rs"]
mod paged;

#[path = "parallel/layerwise.rs"]
mod layerwise;

#[path = "parallel/capture.rs"]
mod capture;


#[path = "parallel/cpu.rs"]
mod cpu;

#[path = "parallel/gpt_oss.rs"]
mod gpt_oss;

#[path = "parallel/composite.rs"]
mod composite;

#[path = "parallel/expert.rs"]
mod expert;
