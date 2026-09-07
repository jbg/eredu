use super::*;
use eredu_core::{intervention::*, *};
use eredu_runtime::capture::CaptureSession;
use std::{cell::Cell, collections::VecDeque, io, rc::Rc, sync::Arc};

mod records;
use records::*;

#[derive(Clone, Default)]
struct Host {
    forwards: Rc<Cell<u64>>,
    fault: Rc<Cell<Option<&'static str>>>,
    copies: Rc<Cell<u64>>,
}
impl Host {
    fn copying(&self, stage: &'static str) -> io::Result<()> {
        self.copies.set(self.copies.get() + 1);
        if self.fault.get() == Some(stage) {
            Err(io::Error::other(stage))
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
struct NativeState {
    ring: VecDeque<u32>,
    offset: u64,
    recurrent: u64,
}
struct Session {
    identity: Arc<()>,
    native: NativeState,
}
struct Saved {
    identity: Arc<()>,
    native: NativeState,
}
#[derive(Clone, Debug, PartialEq)]
struct Sampling {
    temperature: f32,
    rng: u64,
    history: Vec<u32>,
    adaptive: u64,
    next: u64,
}
struct Generation {
    sampling: Sampling,
    capture: Option<CaptureSession>,
}
#[derive(Clone, Copy)]
struct Token(u32);
impl TokenOutput for Token {
    type Error = io::Error;
    fn token_id(&self) -> io::Result<u32> {
        Ok(self.0)
    }
}
struct Done;
impl Completion for Done {
    type Error = io::Error;
    fn is_complete(&self) -> io::Result<bool> {
        Ok(true)
    }
    fn wait(&self) -> io::Result<()> {
        Ok(())
    }
}
impl BackendProvider for Host {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = io::Error;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("host-continuation", "1")
    }
    fn devices(&self) -> io::Result<Vec<(DeviceDescriptor, DeviceCapabilities)>> {
        Ok(vec![])
    }
    fn prepare_model(&self, _: ()) -> io::Result<PreparedModel<()>> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> io::Result<Session> {
        Ok(Session {
            identity: Arc::new(()),
            native: NativeState::default(),
        })
    }
}
impl BackendSession<Host> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = u64;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(&mut self, host: &Host, input: Vec<u32>) -> io::Result<Submission<u64, Done>> {
        host.forwards.set(host.forwards.get() + 1);
        for token in input {
            self.native.offset += 1;
            self.native.recurrent = self
                .native
                .recurrent
                .wrapping_mul(31)
                .wrapping_add(u64::from(token));
            self.native.ring.push_back(token);
            if self.native.ring.len() > 8 {
                self.native.ring.pop_front();
            }
        }
        Ok(Submission {
            output: self.native.recurrent
                ^ self.native.ring.iter().map(|v| u64::from(*v)).sum::<u64>(),
            completion: Done,
        })
    }
    fn decode(&mut self, host: &Host, input: Token) -> io::Result<Submission<u64, Done>> {
        self.prefill(host, vec![input.0])
    }
    fn observe_output(&self, _: &Host, _: &u64) -> io::Result<ObservationSet> {
        Ok(ObservationSet::new())
    }
}

fn sample(
    value: u64,
    state: &mut Generation,
    filter: &TokenFilter,
) -> io::Result<Submission<Token, Done>> {
    let sampling = &mut state.sampling;
    let mut value = vec![(value % 65536) as f32];
    if let Some(run) = &mut state.capture {
        run.begin_step(
            if sampling.next == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            },
            sampling.next,
        )
        .map_err(io::Error::other)?;
        run.observe(&mut Records, "state", &value)
            .map_err(io::Error::other)?;
        if let Some(changed) = run
            .intervene(&mut Records, "state", &value)
            .map_err(io::Error::other)?
        {
            value = changed;
        }
        run.finish_interventions().map_err(io::Error::other)?;
    }
    // Deliberately stateful fixture: recurrent input, RNG, adaptive feedback and
    // full penalty history each affect the sampled canonical token.
    sampling.rng ^= sampling.rng << 13;
    sampling.rng ^= sampling.rng >> 7;
    sampling.rng ^= sampling.rng << 17;
    let scaled = (f64::from(value[0]) / f64::from(sampling.temperature)) as u64;
    let mut token = ((sampling.rng ^ sampling.adaptive ^ scaled) % 64) as u32;
    while filter
        .allowed_mask()
        .is_some_and(|mask| !mask[token as usize])
    {
        token = (token + 1) % 64;
    }
    sampling.history.push(token);
    sampling.adaptive = sampling
        .adaptive
        .wrapping_mul(17)
        .wrapping_add(sampling.history.iter().map(|v| u64::from(*v)).sum::<u64>());
    sampling.next += 1;
    Ok(Submission {
        output: Token(token),
        completion: Done,
    })
}
impl TextGenerationBackend for Host {
    fn text_sampling_control_support(_: &ModelRuntime<Self>) -> ControlSupport {
        ControlSupport::Supported
    }
    type Prompt = Vec<u32>;
    type Token = Token;
    type TextGenerationState = Generation;
    type TextCompletion = Done;
    fn start_text_generation(_: &Self, config: TextGenerationConfig) -> io::Result<Generation> {
        Ok(Generation {
            sampling: Sampling {
                temperature: 0.8,
                rng: config.seed(),
                history: vec![],
                adaptive: 0,
                next: 0,
            },
            capture: None,
        })
    }
    fn prepare_text_prompt(_: &Self, ids: Vec<u32>) -> io::Result<Vec<u32>> {
        Ok(ids)
    }
    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Vec<u32>,
        filter: &TokenFilter,
        state: &mut Generation,
    ) -> io::Result<Submission<Token, Done>> {
        let submitted = runtime.prefill(prompt)?;
        sample(submitted.output, state, filter)
    }
    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Token,
        filter: &TokenFilter,
        state: &mut Generation,
    ) -> io::Result<Submission<Token, Done>> {
        let submitted = runtime.decode(token)?;
        sample(submitted.output, state, filter)
    }
    fn capture_discovery(_: &ModelRuntime<Self>) -> Result<CaptureDiscovery, CaptureError> {
        Ok(discovery())
    }
    fn intervention_discovery(
        _: &ModelRuntime<Self>,
    ) -> Result<InterventionDiscovery, CaptureError> {
        Ok(interventions())
    }
    fn configure_text_interventions(
        runtime: &ModelRuntime<Self>,
        state: &mut Generation,
        capture: AdmittedCapturePlan,
        intervention: AdmittedInterventionPlan,
    ) -> Result<(), CaptureError> {
        eredu_runtime::capture::validate_session(
            &capture,
            &Self::capture_discovery(runtime)?,
            estimate,
        )?;
        eredu_runtime::intervention::validate_session(
            &capture,
            &intervention,
            &Self::intervention_discovery(runtime)?,
            &Records,
        )?;
        eredu_runtime::intervention::install_session(
            &mut state.capture,
            capture,
            Some((intervention, Arc::new(Records))),
        )
    }
    fn take_text_capture(state: &mut Generation) -> Option<CapturedStep> {
        state.capture.as_mut().and_then(CaptureSession::take_step)
    }
}
fn estimate_copy(bytes: u64) -> Option<SnapshotEstimate> {
    Some(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}
impl NativeTextStateBackend for Host {
    type NativeTextState = Saved;
    fn estimate_native_text_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Saved,
        _: u64,
    ) -> io::Result<Option<u64>> {
        Self::validate_native_text_state(runtime, saved)?;
        Ok(if runtime.backend().fault.get() == Some("growth") {
            None
        } else {
            Some(256)
        })
    }
    fn native_text_state_support(_: &ModelRuntime<Self>) -> ControlSupport {
        ControlSupport::Supported
    }
    fn estimate_native_text_state(
        runtime: &ModelRuntime<Self>,
        _: Option<&Saved>,
    ) -> io::Result<Option<SnapshotEstimate>> {
        Ok(if runtime.backend().fault.get() == Some("estimate") {
            None
        } else {
            estimate_copy(256)
        })
    }
    fn capture_native_text_state(runtime: &mut ModelRuntime<Self>) -> io::Result<Saved> {
        runtime.backend().copying("native")?;
        Ok(Saved {
            identity: runtime.session().identity.clone(),
            native: runtime.session().native.clone(),
        })
    }
    fn copy_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &Saved,
    ) -> io::Result<Saved> {
        Self::validate_native_text_state(runtime, saved)?;
        runtime.backend().copying("native")?;
        Ok(Saved {
            identity: saved.identity.clone(),
            native: saved.native.clone(),
        })
    }
    fn validate_native_text_state(runtime: &ModelRuntime<Self>, saved: &Saved) -> io::Result<()> {
        if Arc::ptr_eq(&runtime.session().identity, &saved.identity) {
            Ok(())
        } else {
            Err(io::Error::other("foreign executable"))
        }
    }
    fn exchange_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        slot: &mut Saved,
    ) -> io::Result<()> {
        Self::validate_native_text_state(runtime, slot)?;
        if runtime.backend().fault.get() == Some("exchange") {
            return Err(io::Error::other("exchange"));
        }
        std::mem::swap(&mut runtime.session_mut().native, &mut slot.native);
        Ok(())
    }
}
impl TextSnapshotBackend for Host {
    type SamplingState = Sampling;
    fn continuation_input_tokens(
        input: Option<PendingTextInput<&Vec<u32>, &Token>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        match input? {
            PendingTextInput::Prefill(prompt) => (prompt.len() as u64).checked_add(predictions - 1),
            PendingTextInput::Decode(_) => Some(predictions),
        }
    }
    fn estimate_sampling_growth(
        _: &ModelRuntime<Self>,
        _: &Sampling,
        predictions: u64,
    ) -> io::Result<Option<u64>> {
        Ok(predictions.checked_mul(4))
    }
    fn sampling_state(state: &Generation) -> &Sampling {
        &state.sampling
    }
    fn install_sampling_state(state: &mut Generation, sampling: Sampling) {
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: Sampling,
        capture: Option<CaptureSession>,
    ) -> Generation {
        Generation { sampling, capture }
    }
    fn sampling_prediction(sampling: &Sampling) -> u64 {
        sampling.next
    }
    fn estimate_sampling_state(
        _: &ModelRuntime<Self>,
        state: &Sampling,
    ) -> io::Result<Option<SnapshotEstimate>> {
        Ok(estimate_copy(128 + 4 * state.history.len() as u64))
    }
    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self>,
        state: &Sampling,
    ) -> io::Result<Sampling> {
        runtime.backend().copying("sampling")?;
        Ok(state.clone())
    }
    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        _: Option<PendingTextInput<&Vec<u32>, &Token>>,
    ) -> io::Result<Option<SnapshotEstimate>> {
        Ok(estimate_copy(128))
    }
    fn copy_pending_input(
        runtime: &mut ModelRuntime<Self>,
        input: Option<PendingTextInput<&Vec<u32>, &Token>>,
    ) -> io::Result<Option<PendingTextInput<Vec<u32>, Token>>> {
        runtime.backend().copying("pending")?;
        Ok(input.map(|input| match input {
            PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(prompt.clone()),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(*token),
        }))
    }
    fn capture_run(state: &Generation) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    fn capture_run_mut(state: &mut Generation) -> Option<&mut CaptureSession> {
        state.capture.as_mut()
    }
    fn estimate_child_capture(
        _: &ModelRuntime<Self>,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(shape, selection, slice)
    }
    fn child_intervention_estimator(
        _: &ModelRuntime<Self>,
    ) -> Result<Arc<dyn InterventionEstimator>, CaptureError> {
        Ok(Arc::new(Records))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Controller(Vec<u32>, Rc<Cell<Option<&'static str>>>);
impl SnapshotTokenController for Controller {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        if self.1.get() == Some("controller-estimate") {
            return None;
        }
        (std::mem::size_of::<Self>() as u64).checked_add(4 * self.0.len() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        if self.1.get() == Some("controller-copy") {
            return Err("injected host copy failure".into());
        }
        Ok(self.clone())
    }
}
impl TokenFilterController for Controller {
    type Error = io::Error;
    fn current_filter(&mut self) -> io::Result<TokenFilter> {
        let mut allowed = vec![true; 64];
        allowed[self.0.len() % 64] = false;
        Ok(TokenFilter::allowed(allowed).unwrap())
    }
    fn commit_token(&mut self, id: u32) -> io::Result<()> {
        self.0.push(id);
        Ok(())
    }
    fn is_complete(&mut self) -> io::Result<bool> {
        Ok(false)
    }
}
impl TextSamplingControlBackend for Host {
    fn sampling_control_facts(
        state: &Generation,
    ) -> eredu_runtime::execution_control::SamplingStateFacts {
        eredu_runtime::execution_control::SamplingStateFacts {
            temperature: state.sampling.temperature,
            requires_positive_temperature: true,
            has_rng: true,
        }
    }
    fn install_sampling_override(
        runtime: &mut ModelRuntime<Self>,
        state: &mut Generation,
        request: eredu_runtime::execution_control::ValidatedSamplingOverride,
    ) -> io::Result<()> {
        runtime.backend().copying("reseed")?;
        if let Some(seed) = request.reseed() {
            state.sampling.rng = seed;
        }
        state.sampling.temperature = request.temperature();
        Ok(())
    }
}
fn start(
    driver: &mut TextGenerationDriver<'_, Host>,
    mode: u8,
) -> ManagedTextContinuation<Host, Controller> {
    start_with_controller(driver, mode, Controller::default())
}
fn start_with_controller<C: TokenFilterController>(
    driver: &mut TextGenerationDriver<'_, Host>,
    mode: u8,
    controller: C,
) -> ManagedTextContinuation<Host, C> {
    let config = TextGenerationConfig::new(
        resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(20),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(12345);
    let mut state = driver.start(vec![1, 2], config, controller).unwrap();
    let (capture, intervention) = plans(mode);
    driver
        .enable_interventions(&mut state, capture, intervention)
        .unwrap();
    ManagedTextContinuation::root(state)
}

#[test]
fn host_continuations_conform_for_all_record_modes() {
    for mode in 0..4 {
        let host = Host::default();
        let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start_with_controller(
            &mut driver,
            mode,
            TokenChoiceController::new(Controller::default(), eredu_runtime::TokenDomain::new(64)),
        );
        forced_choice_conformance(
            &mut driver,
            &mut state,
            &ContinuationFixtureLimits {
                host_bytes: 4096,
                growth_bytes: 4096,
                max_predictions: 20,
                capture: Some(limits()),
            },
            64,
            || host.forwards.get(),
        );
        sampling_override_conformance(
            &mut driver,
            &mut state,
            &ContinuationFixtureLimits {
                host_bytes: 4096,
                growth_bytes: 4096,
                max_predictions: 20,
                capture: Some(limits()),
            },
            || host.forwards.get(),
        );
        continuation_conformance(
            &mut driver,
            state,
            ContinuationFixtureLimits {
                host_bytes: 4096,
                growth_bytes: 4096,
                max_predictions: 20,
                capture: Some(limits()),
            },
            || host.forwards.get(),
        );
    }
}

#[test]
fn staged_failures_preserve_source_snapshot_and_monotone_copy_accounting() {
    for mode in 0..4 {
        let host = Host::default();
        let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start(&mut driver, mode);
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 2,
            retained_bytes: 64_000_000,
            cumulative_copy_bytes: 512_000_000,
        });
        for _ in 0..3 {
            step(&mut driver, &mut state);
        }
        let saved = TextContinuationSnapshot::capture(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            Some(4096),
        )
        .unwrap();
        let baseline: Vec<_> = (0..3).map(|_| step(&mut driver, &mut state)).collect();
        for stage in [
            "estimate",
            "controller-estimate",
            "facade-host",
            "controller-copy",
            "pending",
            "sampling",
            "native",
            "exchange",
        ] {
            let native = driver.runtime().session().native.clone();
            let controller = state.controller().clone();
            let boundary = state.boundary(&mut driver).unwrap();
            let sampling = Host::sampling_state(boundary.parts().1).clone();
            drop(boundary);
            let usage = budget.usage();
            let copies = host.copies.get();
            host.fault.set(Some(stage));
            state.controller().1.set(Some(stage));
            assert!(saved
                .restore_with(&mut state.boundary(&mut driver).unwrap(), &budget, || {
                    if stage == "facade-host" {
                        Err("injected semantic copy failure".into())
                    } else {
                        Ok(())
                    }
                })
                .is_err());
            assert_eq!(driver.runtime().session().native, native);
            assert_eq!(state.controller(), &controller);
            assert_eq!(
                Host::sampling_state(state.boundary(&mut driver).unwrap().parts().1),
                &sampling
            );
            assert_eq!(budget.usage().retained_bytes, usage.retained_bytes);
            if stage == "estimate" || stage == "controller-estimate" {
                assert_eq!(budget.usage(), usage);
                assert_eq!(host.copies.get(), copies);
            } else {
                assert!(budget.usage().cumulative_copy_bytes > usage.cumulative_copy_bytes);
            }
            host.fault.set(None);
            state.controller().1.set(None);
            saved
                .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
                .unwrap();
            let actual: Vec<_> = (0..3).map(|_| step(&mut driver, &mut state)).collect();
            assert_eq!(actual, baseline);
        }
        let native = driver.runtime().session().native.clone();
        let usage = budget.usage();
        let copies = host.copies.get();
        host.fault.set(Some("growth"));
        assert!(matches!(
            saved.native_continuation_growth(driver.runtime(), 8),
            Err(TextSnapshotError::Control(
                ExecutionControlError::UnknownEstimate
            ))
        ));
        assert_eq!(host.copies.get(), copies);
        assert_eq!(budget.usage(), usage);
        host.fault.set(None);
        let failed: Result<(_, ()), _> = saved.fork_with(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "failed-host-child",
                max_predictions: 8,
                capture_limits: Some(limits()),
                intervention: None,
                host_bytes: Some(4096),
                continuation_growth_bytes: Some(4096),
            },
            |_, _| {
                Err(TextSnapshotError::Host(
                    "injected branch semantic copy failure".into(),
                ))
            },
        );
        assert!(matches!(failed, Err(TextSnapshotError::Host(_))));
        assert_eq!(driver.runtime().session().native, native);
        assert_eq!(budget.usage().branches, usage.branches);
        assert_eq!(budget.usage().retained_bytes, usage.retained_bytes);
        assert!(budget.usage().cumulative_copy_bytes > usage.cumulative_copy_bytes);
    }
}
