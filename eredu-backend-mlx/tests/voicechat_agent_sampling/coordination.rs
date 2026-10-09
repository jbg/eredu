//! The designated owner resolves policy and draws once for every participant.
use super::*;
use eredu_architectures::voicechat::{
    channel_decision::{
        select_channels, ChannelDecisionPolicy, DecisionEvidence, DecisionInput, DecisionResolver,
        DecisionSource, ReplicatedChannelDecisions,
    },
    channels::DualChannelLogits,
};
use eredu_backend_mlx::native::MlxReplicatedCoordinator;
use eredu_core::{
    checkpoint::TensorDtype, scheduler::SemanticStateTransaction, BackendFailure,
    CollectiveGroupId, Completion, CompletionCancellationMode,
};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeScheduleState, TokenDomain,
};
use std::{cell::Cell, convert::Infallible};

const WORKER: &str = "EREDU_AGENT_SAMPLING_WORLD_SIZE";

#[derive(Clone)]
struct Host;
impl SemanticStateTransaction for Host {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Self::Error> {
        Ok(self.clone())
    }
    fn commit_branch(&mut self, _: Self) -> Result<(), Self::Error> {
        unreachable!()
    }
}
impl RealtimeScheduleState for Host {
    type Identity = ();
    type Error = Infallible;
    fn identity(&self) -> &() {
        &()
    }
    fn validate_successor(&self, _: &Self) -> Result<(), Self::Error> {
        unreachable!()
    }
}
enum Unsubmitted {}
impl Completion for Unsubmitted {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        match *self {}
    }
    fn wait(&self) -> Result<(), Self::Error> {
        match *self {}
    }
}

struct Resolver {
    owner: bool,
    policy: ChannelDecisionPolicy,
    preflights: Cell<usize>,
    resolutions: Cell<usize>,
    evidence: DecisionEvidence,
}
impl DecisionResolver for Resolver {
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure> {
        assert!(self.owner, "a follower must not inspect owner policy");
        self.preflights.set(self.preflights.get() + 1);
        self.policy
            .preflight(domain)
            .map_err(BackendFailure::from_error)
    }
    fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        assert!(self.owner, "a follower must not resolve turn policy");
        self.resolutions.set(self.resolutions.get() + 1);
        assert_eq!(input.evidence, self.evidence);
        assert!(input.raw_agent < 16);
        assert_eq!(input.raw_function, 6);
        Ok(self.policy)
    }
}

fn coordinator(world: &safemlx::distributed::Group, stream: &Stream) -> MlxReplicatedCoordinator {
    let id = CollectiveGroupId::new(98);
    let group = CommunicationGroupDescriptor::new(
        id,
        0,
        (0..world.size()).collect(),
        Some(world.rank()),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::I32],
                CommunicationTensorLimits::new(1, 1, 5, None).unwrap(),
                true,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(10),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    MlxReplicatedCoordinator::from_manifest(manifest, id, 1, world, stream).unwrap()
}

#[test]
fn native_agent_sampling_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut agreement = coordinator(&world, &stream);
    let owner = world.rank() == 1;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "nucleus_crossing")
        .unwrap();
    let mut sampler = AgentSampler::new(AgentSamplingPolicy::new(0.9, 0.7, 1.4, 64).unwrap());
    let history = case["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    sampler.set_history(&history).unwrap();
    let state = |has_random: bool| {
        RealtimeGenerationState::<_, _, _, _, _, Unsubmitted>::new(
            Host,
            RealtimeGenerationContract::new((), 1),
            Host,
            (),
            vec![sampler.clone()],
            has_random.then(|| RandomState::with_seed(317).unwrap()),
        )
        .unwrap()
    };
    let mut ordinary = state(true).branch().unwrap();
    let mut distributed = state(owner).branch().unwrap();
    let values = case["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let mut function = vec![-3.; 16];
    function[6] = 4.;
    let logits = DualChannelLogits {
        agent: MlxTensor::from_f32_slice(&values, &[1, 1, 16], &stream).unwrap(),
        function: MlxTensor::from_f32_slice(&function, &[1, 1, 16], &stream).unwrap(),
    };
    let cases = [
        (
            ChannelDecisionPolicy::Tool {
                function_feedback: Some(9),
                spoken_text: Some(5),
            },
            [12, 9, 5],
        ),
        (
            ChannelDecisionPolicy::Main {
                agent_feedback: Some(4),
                function_feedback: Some(8),
                spoken_text: Some(7),
            },
            [4, 8, 7],
        ),
    ];
    for (index, (policy, expected)) in cases.into_iter().enumerate() {
        let evidence = DecisionEvidence {
            first_blank: Some(index == 0),
            audio_frame_index: Some(23),
        };
        let resolver = Resolver {
            owner,
            policy,
            preflights: Cell::new(0),
            resolutions: Cell::new(0),
            evidence,
        };
        let reference = select_channels::<Backend, _, _, _, _>(
            &mut ordinary,
            &logits,
            policy.into(),
            evidence,
            TokenDomain::new(16),
            None,
            &stream,
        )
        .unwrap();
        let mut authority = ReplicatedChannelDecisions::new(&mut agreement, [31, 7, index as u64]);
        let decision = select_channels::<Backend, _, _, _, _>(
            &mut distributed,
            &logits,
            DecisionSource::Resolved(&resolver),
            evidence,
            TokenDomain::new(16),
            Some(&mut authority),
            &stream,
        )
        .unwrap();
        assert_eq!(decision.raw_agent_id, reference.raw_agent_id);
        assert_eq!(decision.raw_function_id, 6);
        assert_eq!(
            Backend::token_id(&decision.raw_agent, &stream).unwrap(),
            reference.raw_agent_id
        );
        assert_eq!(
            Backend::token_id(&decision.raw_function, &stream).unwrap(),
            6
        );
        assert_eq!(
            [
                decision.feedback.agent(),
                decision.feedback.function(),
                decision.spoken_text
            ],
            expected
        );
        assert_eq!(decision.feedback, reference.feedback);
        assert_eq!(
            distributed.samplers()[0].history(),
            ordinary.samplers()[0].history()
        );
        assert_eq!(
            distributed.samplers()[0].history().last(),
            Some(&expected[0])
        );
        assert_eq!(resolver.preflights.get(), usize::from(owner));
        assert_eq!(resolver.resolutions.get(), usize::from(owner));
    }
    assert_eq!(distributed.samplers()[0].history().len(), history.len() + 2);
    let expected_next = ordinary
        .draw_standard_normal::<Backend>(&[7], &stream)
        .unwrap();
    let next = distributed.draw_standard_normal::<Backend>(&[7], &stream);
    if owner {
        assert_eq!(
            next.unwrap().to_f32_vec(&stream).unwrap(),
            expected_next.to_f32_vec(&stream).unwrap()
        );
    } else {
        assert!(
            next.is_err(),
            "followers retain no RNG after adopting decisions"
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_agent_sampling_owner_decisions_preserve_feedback_and_speech_rng() {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let sockets = (0..2)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = Children(Vec::new());
    for rank in 0..2 {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "coordination::native_agent_sampling_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, "2")
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if Instant::now() >= deadline || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                let _ = child.kill();
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    assert!(
        outputs
            .iter()
            .all(|output| output.as_ref().is_ok_and(|output| output.status.success())),
        "agent sampling rank outputs:\n{}",
        outputs
            .iter()
            .enumerate()
            .map(|(rank, output)| match output {
                Ok(output) => format!(
                    "rank {rank}: {}\n{}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ),
                Err(error) => format!("rank {rank}: failed to reap worker: {error}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    );
}
