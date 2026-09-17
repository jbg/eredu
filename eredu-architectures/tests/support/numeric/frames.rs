struct Bind<'a> {
    values: &'a BTreeMap<String, NumericTensor>,
    seen: &'a mut BTreeSet<String>,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let value = self
            .values
            .get(name)
            .unwrap_or_else(|| panic!("unbound actual parameter {name}"));
        assert_eq!(tensor.shape, value.shape);
        assert!(value.data.iter().all(|x| x.is_finite()));
        assert!(value.data.iter().any(|x| x.abs() > 1e-9));
        tensor.data.clone_from(&value.data);
        self.seen.insert(name.to_owned());
    }
}
impl StaticParameterVisitorMut<NumericBackend> for Bind<'_> {
    type Error = Error;
    fn visit_mut<M>(&mut self, _: &str, module: &mut M) -> Result<(), Error>
    where
        M: Parameterized<NumericTensor>,
    {
        module.visit_parameters_mut(self);
        Ok(())
    }
}
struct Construction {
    store: RetainedCheckpointSource,
    bound: Rc<Cell<usize>>,
    residency: Rc<RefCell<residency::Evidence>>,
}
impl<A> RealtimeModelConstructionMechanisms<A, NumericBackend> for Construction
where
    A: LayeredArchitecture<NumericBackend, State>,
    A::Error: std::fmt::Display,
{
    type State = State;
    type PolicyError = Error;
    type ResidentPolicy = residency::Resident<A::Unit>;
    type BoundedPolicy = residency::Policy<A::Unit>;
    type Error = Error;
    fn prepare_resident_materialization(
        &mut self,
        architecture: &mut A,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &NumericContext,
    ) -> Result<(), Error> {
        assert!(source_architecture.is_none() && source_units.is_none());
        assert_eq!(selected.residency(), LayerWeightResidency::FullyResident);
        residency::validate_tasks(tasks, &self.store)?;
        let (pinned, unit_bindings) =
            eredu_runtime::realtime_task_binding_plan(tasks, self.store.as_ref())
                .map_err(Error::backend)?
                .into_parts();
        let mut values = residency::materialize(&pinned, &self.store, context)?;
        for bindings in unit_bindings.values() {
            for (name, value) in residency::materialize(bindings, &self.store, context)? {
                assert!(values.insert(name, value).is_none());
            }
        }
        assert!(!values.is_empty());
        let mut seen = BTreeSet::new();
        let mut binder = Bind {
            values: &values,
            seen: &mut seen,
        };
        architecture
            .visit_static_parameters_mut(&mut binder)
            .map_err(Error::backend)?;
        for unit in units {
            unit.visit_parameters_mut(&mut binder);
        }
        assert_eq!(seen, values.keys().cloned().collect());
        self.bound.set(values.len());
        Ok(())
    }
    fn resident_policy(
        &mut self,
        _: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedRealtimeRealization,
        _: &NumericContext,
    ) -> Result<RealizedRealtimePolicy<Self::ResidentPolicy>, Error> {
        assert_eq!(units.len(), selected.execution_units().len());
        Ok(RealizedRealtimePolicy::new(
            residency::Resident(ResidentUnitWindow::new(units)),
            selected.residency(),
        ))
    }
    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        source_architecture: Option<&mut A>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &NumericContext,
    ) -> Result<RealizedRealtimePolicy<Self::BoundedPolicy>, Error> {
        assert!(source_architecture.is_none());
        let policy = residency::Policy::prepare(
            architecture,
            tasks,
            selected,
            self.store.clone(),
            context,
            self.bound.clone(),
            self.residency.clone(),
        )?;
        Ok(RealizedRealtimePolicy::new(policy, selected.residency()))
    }
    fn realize_state(
        &mut self,
        selected: &eredu_runtime::SelectedRealtimeStateRealization,
        _: &NumericContext,
    ) -> Result<RealizedRealtimeState<State>, Error> {
        let state = DeviceState::create(selected.layout().clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .map_err(Error::backend)?;
        Ok(RealizedRealtimeState::new(state, selected.clone()))
    }
}

// The terminal scheduler removes canonical state. Observe its actual Drop rather
// than pretending the terminal request still exposes a mutable/live generation.
struct TransactionalState {
    state: State,
    retired: Rc<RefCell<Option<State>>>,
}
impl Drop for TransactionalState {
    fn drop(&mut self) {
        assert!(self
            .retired
            .borrow_mut()
            .replace(self.state.clone())
            .is_none());
    }
}
#[derive(Clone)]
struct Branch(State);
impl Deref for Branch {
    type Target = State;
    fn deref(&self) -> &State {
        &self.0
    }
}
impl DerefMut for Branch {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.0
    }
}
impl SemanticStateTransaction for TransactionalState {
    type Branch = Branch;
    type Error = Infallible;
    fn branch(&self) -> Result<Branch, Infallible> {
        Ok(Branch(self.state.clone()))
    }
    fn commit_branch(&mut self, branch: Branch) -> Result<(), Infallible> {
        self.state = branch.0;
        Ok(())
    }
    fn discard_branch(_: Branch) -> Result<(), Infallible> {
        Ok(())
    }
    fn permits_parallel_branches(&self) -> bool {
        false
    }
}
struct Tensors;
impl RealtimeHostTokenMaterializer for Tensors {
    type Tensor = NumericTensor;
    type Error = Error;
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<NumericTensor, Error> {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        Ok(NumericTensor::new(
            shape.map(|n| n as i32),
            values.iter().map(|x| *x as f32).collect(),
        ))
    }
}
// The marker belongs to each actual host-produced tensor. Only the explicit
// initialization-rejection case enables it; the recorder retains Weak handles.
struct RecordingHost {
    enabled: bool,
    probes: Vec<std::sync::Weak<()>>,
}
impl RealtimeHostTokenMaterializer for RecordingHost {
    type Tensor = NumericTensor;
    type Error = Error;
    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<NumericTensor, Error> {
        let mut value = Tensors.materialize_i32(values, shape)?;
        if self.enabled {
            let owner = Arc::new(());
            self.probes.push(Arc::downgrade(&owner));
            value.retirement_probe = Some(owner);
        }
        Ok(value)
    }
}
impl RealtimeFrameTensorMechanisms for Tensors {
    type Tensor = NumericTensor;
    type Error = Error;
    fn column(&mut self, matrix: &NumericTensor, column: usize) -> Result<NumericTensor, Error> {
        Ok(matrix.axis_slice(1, column, column + 1))
    }
    fn filled_column(&mut self, token: i32, batch: usize) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            [batch as i32, 1],
            vec![token as f32; batch],
        ))
    }
    fn stack_columns(
        &mut self,
        columns: &[NumericTensor],
        batch: usize,
    ) -> Result<NumericTensor, Error> {
        if columns.is_empty() {
            return Ok(NumericTensor::new([batch as i32, 0], vec![]));
        }
        for c in columns {
            assert_eq!(c.shape, [batch as i32, 1]);
        }
        NumericTensor::concatenate(columns, 1, &NumericContext::default())
    }
}
#[derive(Default)]
struct Trace {
    values: Vec<(String, NumericTensor)>,
    fail: Option<String>,
    failure: Option<Error>,
    checked_failure: bool,
    final_rows: bool,
    intervene_text: bool,
}
#[derive(Debug, thiserror::Error)]
#[error("actual Moshi callback failed at {path}")]
struct CallbackFailure {
    path: String,
}
struct Observer(Rc<RefCell<Trace>>);
impl ActivationObserver<NumericTensor, Error> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        !self.0.borrow().final_rows
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        if self.0.borrow().intervene_text && path == "text_linear.logits" {
            let mut replacement = value.clone();
            let width = *replacement.shape.last().unwrap() as usize;
            for row in replacement.data.chunks_mut(width) {
                row.fill(-100.0);
                row[3] = 100.0;
            }
            Ok(Some(replacement))
        } else {
            Ok(None)
        }
    }
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        let mut trace = self.0.borrow_mut();
        assert_eq!(value.shape[1], 1, "canonical frame is not a packed prompt");
        assert!(value.data.iter().all(|v| v.is_finite()));
        assert!(value.data.iter().any(|v| v.abs() > 1e-9));
        trace.values.push((path.to_owned(), value.clone()));
        if trace.fail.as_deref() == Some(path) {
            let failure = Error::backend_source(CallbackFailure {
                path: path.to_owned(),
            });
            trace.failure = Some(failure.clone());
            return Err(failure);
        }
        Ok(())
    }
}
struct Complete {
    context: NumericContext,
    calls: usize,
    executions: usize,
}
impl<T>
    RealtimeFrameCompletionMechanism<
        NumericTensor,
        T,
        (Option<NumericTensor>, moshi::ForwardContext<NumericTensor>),
    > for Complete
{
    type Completion = NumericCompletion;
    type Error = Error;
    fn complete(
        &mut self,
        _: MaterializedRealtimeInput<NumericTensor>,
        output: &CompletedRealtimeFrame<NumericTensor, NumericTensor>,
        _: &T,
        _: &RealtimePayloadHistory<NumericTensor>,
        execution: Option<(Option<NumericTensor>, moshi::ForwardContext<NumericTensor>)>,
    ) -> Result<NumericCompletion, RealtimeCompletionCreationError<NumericCompletion, Error>> {
        self.calls += 1;
        if let Some((logits, forward)) = execution {
            self.executions += 1;
            if let Some(logits) = logits {
                assert_eq!(logits.shape[1], 1);
                assert_tensor_exact(
                    &logits,
                    forward.text_logits().unwrap(),
                    "actual retained text output",
                );
            }
            let temporal = forward.temporal_output().unwrap();
            assert_eq!(temporal.shape[1], 1);
            assert!(temporal.data.iter().any(|v| v.abs() > 1e-9));
        }
        NumericBackend::submit(
            &self.context,
            [
                output.text(),
                output.decision_audio(),
                output.sampled_audio(),
            ],
        )
        .map_err(RealtimeCompletionCreationError::before_submission)
    }
}

#[derive(Clone)]
struct Snapshot {
    state: State,
    schedule: RealtimeFrameScheduleState,
    history: Vec<(RealtimeSlotCoordinate, NumericTensor)>,
    calls: Vec<usize>,
    random: Option<i32>,
}
fn snapshot(sessions: &Sessions) -> Snapshot {
    let s = sessions.request_state(REQUEST).unwrap();
    let g = s.generation();
    let history = g.model_state().payload_history();
    if let Some(contract) = history.contract() {
        assert_eq!(contract.owner().value(), s.incarnation().value());
        assert_eq!(
            contract.generation().value(),
            s.history_generation().value()
        );
        for envelope in history.envelopes() {
            assert_eq!(envelope.contract(), contract);
        }
    }
    Snapshot {
        state: g.model_state().model_state().state.clone(),
        schedule: g.schedule_state().clone(),
        history: history.entries().map(|(c, v)| (c, v.clone())).collect(),
        calls: g.samplers().iter().map(|s| s.calls).collect(),
        random: g.random_state().copied(),
    }
}
fn same_tensor_option(a: &Option<NumericTensor>, b: &Option<NumericTensor>, label: &str) {
    assert_eq!(a.is_some(), b.is_some(), "{label}");
    if let (Some(a), Some(b)) = (a, b) {
        assert_tensor_exact(a, b, label);
    }
}
fn same_state(a: &State, b: &State) {
    assert_eq!(a.layout(), b.layout());
    assert_eq!(a.as_ref().len(), b.as_ref().len());
    for (a, b) in a.as_ref().iter().zip(b.as_ref()) {
        assert_eq!(a.resets, b.resets);
        assert_eq!(a.fixed_offset, b.fixed_offset);
        assert!(a.compressed.is_none() && b.compressed.is_none());
        assert!(a.pooling.is_none() && b.pooling.is_none());
        assert!(a.fixed.is_empty() && b.fixed.is_empty());
        let (a, b) = (a.attention.as_ref().unwrap(), b.attention.as_ref().unwrap());
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.window, b.window);
        same_tensor_option(&a.keys, &b.keys, "complete frame keys");
        same_tensor_option(&a.values, &b.values, "complete frame values");
        assert!(a.attention_history.is_none() && b.attention_history.is_none());
    }
}
fn same_snapshot(a: &Snapshot, b: &Snapshot) {
    same_state(&a.state, &b.state);
    assert_eq!(a.schedule, b.schedule);
    assert_eq!(a.random, b.random);
    assert_eq!(a.calls, b.calls);
    assert_eq!(a.history.len(), b.history.len());
    for ((ac, av), (bc, bv)) in a.history.iter().zip(&b.history) {
        assert_eq!(ac, bc);
        assert_tensor_exact(av, bv, "complete delayed coordinate payload");
    }
}
struct Frame {
    snapshot: Snapshot,
    values: Vec<(String, NumericTensor)>,
    outputs: Vec<NumericTensor>,
    diagnostics: Vec<NumericTensor>,
}
struct Report {
    frames: Vec<Frame>,
    // Last complete canonical snapshot, taken before terminal teardown if any.
    final_state: Snapshot,
    retired_state: Option<State>,
    committed_work: u64,
    status: RequestStatus,
    bound_parameters: usize,
    completion_calls: usize,
    executions: usize,
    failed_values: Vec<(String, NumericTensor)>,
    residency: residency::Evidence,
    projections: Vec<(String, Vec<i32>)>,
}
#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    residency: ExecutionResidency,
    bounded: bool,
    diagnostics: bool,
    forced: bool,
    fail: Option<&'static str>,
    cancel: bool,
    readout: eredu_core::OutputDemand,
    required_observer: bool,
    force_text_only: bool,
    intervene_text: bool,
    convention: eredu_core::RealtimeFrameConvention,
    chunks: [usize; 3],
    observe_after_initialization: bool,
    reject_initialization: bool,
}
impl Default for Case {
    fn default() -> Self {
        Self {
            batch: 1,
            residency: ExecutionResidency::FullyResident,
            bounded: false,
            diagnostics: true,
            forced: false,
            fail: None,
            cancel: false,
            readout: eredu_core::OutputDemand::Sequence,
            required_observer: false,
            force_text_only: false,
            intervene_text: false,
            convention: eredu_core::RealtimeFrameConvention::FeedbackAlignedHistory,
            chunks: [2, 5, 3],
            observe_after_initialization: false,
            reject_initialization: false,
        }
    }
}
struct Visitor {
    case: Case,
    config: moshi::MoshiConfig,
    context: NumericContext,
    started: Rc<Cell<bool>>,
}
impl moshi::MoshiRealtimeArchitectureVisitor<NumericBackend, State> for Visitor {
    type Output = Report;
    type Error = Error;
    fn construction_started(&mut self) {
        assert!(!self.started.replace(true));
    }
    fn visit<A>(
        self,
        mut prepared: moshi::PreparedMoshiRealtimeArchitecture<A>,
        store: RetainedCheckpointSource,
    ) -> Result<Report, Error>
    where
        A: moshi::MoshiRealtimeExecutionArchitecture<NumericBackend, State>
            + RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        assert!(self.started.get());
        let architecture = prepared.take_architecture();
        assert!(prepared.take_source_architecture().is_none());
        assert!(prepared.take_parallel().is_none());
        let contract = prepared.take_contract();
        let identity = RealtimeModelSessionIdentity::from_selected(contract.selected());
        assert_eq!(
            contract.selected().execution_units().len(),
            self.config.temporal().num_hidden_layers() as usize
                + self.config.frame_schedule().depth_audio_codebooks()
        );
        assert_eq!(
            contract.selected().state().layout().len(),
            (self.config.temporal().num_hidden_layers()
                + self.config.depth_template().num_hidden_layers()) as usize
        );
        let bound = Rc::new(Cell::new(0));
        let residency = Rc::new(RefCell::new(residency::Evidence::default()));
        let source_diagnostics = store.clone();
        let constructed = construct_realtime_model::<A, NumericBackend, _>(
            architecture,
            None,
            contract,
            Construction {
                store,
                bound: bound.clone(),
                residency: residency.clone(),
            },
            &self.context,
        )
        .map_err(Error::backend)?;
        residency
            .borrow_mut()
            .construction_complete(&source_diagnostics)?;
        let (execution, state) = constructed.into_execution_and_state();
        let ingress = moshi::realtime_ingress_contract(&self.config).map_err(Error::backend)?;
        let schedule = ingress.schedule().clone();
        assert_eq!(schedule.frame_convention(), self.case.convention);
        let retired = Rc::new(RefCell::new(None));
        let generation = RealtimeGenerationState::new(
            RealtimePayloadState::fresh(
                TransactionalState {
                    state,
                    retired: retired.clone(),
                },
                schedule.clone(),
            ),
            schedule.clone(),
            RealtimeSampling::new(0.7, 0.9, 11).map_err(Error::backend)?,
            vec![
                StatefulNumericSampler {
                    calls: 0,
                    invalid: false
                };
                schedule.depth_audio_codebooks() + 1
            ],
            Some(11),
        )
        .map_err(Error::backend)?;
        let mut sessions = Sessions::new(
            identity,
            SchedulerLimits::with_execution_bounds(1, 20, 3, 1, 1, usize::MAX).unwrap(),
        )
        .map_err(Error::backend)?;
        sessions
            .register(REQUEST, generation)
            .map_err(Error::backend)?;
        let trace = Rc::new(RefCell::new(Trace::default()));
        trace.borrow_mut().final_rows = self.case.required_observer;
        trace.borrow_mut().intervene_text = self.case.intervene_text;
        let mut executor =
            if self.case.required_observer && !self.case.observe_after_initialization {
                moshi::MoshiPreparedRealtimeFrameExecutor::with_observer(
                    execution,
                    Observer(trace.clone()),
                )
            } else {
                moshi::MoshiPreparedRealtimeFrameExecutor::with_opportunistic_observer(
                    execution,
                    Observer(trace.clone()),
                )
            }
            .with_readout(self.case.readout);
        self.context.projections.lock().unwrap().clear();
        let mut host = RecordingHost {
            enabled: self.case.reject_initialization,
            probes: Vec::new(),
        };
        let mut tensors = Tensors;
        let mut complete = Complete {
            context: self.context.clone(),
            calls: 0,
            executions: 0,
        };
        let initial_state = snapshot(&sessions);
        let total = self.case.chunks.iter().sum::<usize>();
        let prefix = self.case.chunks[0];
        let text_card = self.config.text_vocabulary_size() as usize;
        let audio_card = self.config.audio_vocabulary_size() as usize;
        let frames = (0..total)
            .map(|position| {
                let input = (0..self.case.batch)
                    .flat_map(|batch| {
                        (0..schedule.input_audio_codebooks())
                            .map(move |slot| ((position + batch + slot + 1) % audio_card) as i32)
                    })
                    .collect();
                let mut frame = RealtimeInputFrame::new(self.case.batch, input);
                if self.case.forced {
                    frame = frame
                        .with_forced_text(
                            (0..self.case.batch)
                                .map(|b| ((b + position + 1) % text_card) as i32)
                                .collect(),
                        )
                        .with_forced_generated_audio(
                            (0..self.case.batch)
                                .flat_map(|b| {
                                    (0..schedule.generated_audio_codebooks())
                                        .map(move |q| ((position + b + q + 1) % audio_card) as i32)
                                })
                                .collect(),
                        );
                }
                if self.case.force_text_only {
                    frame = frame.with_forced_text(
                        (0..self.case.batch)
                            .map(|b| ((b + position + 1) % text_card) as i32)
                            .collect(),
                    );
                }
                if self.case.diagnostics {
                    frame = frame.with_diagnostics();
                }
                frame
            })
            .collect::<Vec<_>>();
        let mut pending = frames.into_iter();
        sessions
            .enqueue_batch(REQUEST, pending.by_ref().take(prefix).collect())
            .map_err(Error::backend)?;
        let mut queued = prefix;
        let mut records: Vec<Frame> = Vec::new();
        for turn in 0..30 {
            // Reuse the canonical scheduler; only admission batch boundaries vary.
            let additional = if records.len() == queued && queued == prefix {
                self.case.chunks[1]
            } else if records.len() == queued && queued == prefix + self.case.chunks[1] {
                self.case.chunks[2]
            } else {
                0
            };
            if records.len() == 1 && self.case.observe_after_initialization {
                executor = moshi::MoshiPreparedRealtimeFrameExecutor::with_observer(
                    executor.into_execution(),
                    Observer(trace.clone()),
                )
                .with_readout(self.case.readout);
            }
            if additional != 0 {
                sessions
                    .enqueue_batch(REQUEST, pending.by_ref().take(additional).collect())
                    .map_err(Error::backend)?;
                queued += additional;
            }
            if records.len() == prefix {
                trace.borrow_mut().fail = self.case.fail.map(str::to_owned);
                if self.case.cancel {
                    sessions.cancel(REQUEST).map_err(Error::backend)?;
                }
            }
            trace.borrow_mut().values.clear();
            let mut execute = |_,
                               frame: &RealtimeInputFrame,
                               branch: &mut eredu_runtime::RealtimeSessionBranch<
                eredu_runtime::RealtimePayloadBranch<Branch, NumericTensor>,
                StatefulNumericSampler,
                i32,
                NumericCompletion,
            >| {
                let payload = branch.payload_contract(&ingress).map_err(Error::backend)?;
                eredu_runtime::execute_realtime_frame::<NumericBackend, _, _, _, _, _, _, _, _>(
                    &ingress,
                    &payload,
                    frame,
                    branch.generation_mut(),
                    &moshi::realtime_decision_execution(),
                    &mut host,
                    &mut tensors,
                    &mut executor,
                    &mut complete,
                    &self.context,
                )
                .map_err(|error| {
                    if self.case.reject_initialization {
                        assert!(matches!(
                            &error,
                            eredu_runtime::RealtimeFrameCoordinatorError::Model(
                                moshi::MoshiRealtimeExecutionError::ObservationWithoutModel
                            )
                        ));
                        trace.borrow_mut().checked_failure = true;
                    }
                    if let Some(expected) = self.case.fail {
                        // The coordinator retains the concrete model error. The
                        // scheduler intentionally publishes only its Display.
                        let eredu_runtime::RealtimeFrameCoordinatorError::Model(
                            moshi::MoshiRealtimeExecutionError::Execution(
                                eredu_runtime::LayerwiseRuntimeError::Architecture(cause),
                            ),
                        ) = &error
                        else {
                            panic!("unexpected failure before callback {expected}: {error:?}");
                        };
                        let actual = cause
                            .source()
                            .and_then(|source| source.downcast_ref::<CallbackFailure>())
                            .expect("original typed callback failure");
                        let mut trace = trace.borrow_mut();
                        let injected = trace
                            .failure
                            .as_ref()
                            .and_then(|error| error.source())
                            .and_then(|source| source.downcast_ref::<CallbackFailure>())
                            .expect("actual observer injected the sentinel");
                        assert_eq!(actual.path, expected);
                        assert!(std::ptr::eq(actual, injected));
                        assert!(!trace.checked_failure);
                        trace.checked_failure = true;
                    }
                    Error::backend(error)
                })
            };
            let progress = if self.case.bounded {
                sessions.run_local_bounded(Instant::now(), [2, 1, 2][turn % 3], &mut execute)
            } else {
                sessions.run_local_turn(Instant::now(), &mut execute)
            };
            let progress = match progress {
                Ok(progress) => progress,
                Err(SchedulerError::Submission(message)) if self.case.reject_initialization => {
                    assert!(trace.borrow().checked_failure);
                    assert_eq!(message, "realtime prepared model execution failed");
                    assert!(records.is_empty());
                    assert_eq!(complete.calls, 0);
                    assert_eq!(complete.executions, 0);
                    assert_eq!(
                        sessions.request_status(REQUEST),
                        Some(RequestStatus::Failed)
                    );
                    break;
                }
                Err(SchedulerError::Submission(message)) if self.case.fail.is_some() => {
                    assert!(trace.borrow().checked_failure);
                    assert_eq!(message, "realtime prepared model execution failed");
                    assert_eq!(records.len(), prefix);
                    assert_eq!(
                        sessions.request_status(REQUEST),
                        Some(RequestStatus::Failed)
                    );
                    break;
                }
                Err(error) => return Err(Error::backend(error)),
            };
            assert!(
                progress.committed.len() <= 1,
                "one canonical branch per request"
            );
            for (_, _, output) in progress.committed {
                assert!(output.completion().is_complete().unwrap());
                let frame = output.frame();
                let mut outputs = vec![
                    frame.text().clone(),
                    frame.decision_audio().clone(),
                    frame.sampled_audio().clone(),
                ];
                outputs.extend(frame.aligned_audio().cloned());
                let snapshot = snapshot(&sessions);
                assert_eq!(snapshot.schedule.frontier(), records.len() + 1);
                records.push(Frame {
                    snapshot,
                    values: trace.borrow().values.clone(),
                    outputs,
                    diagnostics: frame.diagnostics().to_vec(),
                });
            }
            let status = sessions.request_status(REQUEST).unwrap();
            if status != RequestStatus::Active || records.len() == total {
                break;
            }
        }
        let status = sessions.request_status(REQUEST).unwrap();
        let final_state = if status == RequestStatus::Active {
            snapshot(&sessions)
        } else {
            assert!(sessions.request_state(REQUEST).is_none());
            records
                .last()
                .map_or_else(|| initial_state.clone(), |r| r.snapshot.clone())
        };
        let failed_values = trace.borrow().values.clone();
        if self.case.fail.is_some() || self.case.cancel || self.case.reject_initialization {
            let before = trace.borrow().values.len();
            let progress = sessions
                .run_local_turn::<Error>(Instant::now(), |_, _, _| {
                    panic!("terminal request resubmitted")
                })
                .unwrap();
            assert_eq!(progress.newly_submitted, 0);
            assert_eq!(trace.borrow().values.len(), before);
            assert!(sessions.request_state(REQUEST).is_none());
            assert_eq!(
                sessions.report().completed_work,
                if self.case.reject_initialization {
                    0
                } else {
                    prefix as u64
                }
            );
            same_state(retired.borrow().as_ref().unwrap(), &final_state.state);
        } else {
            assert_eq!(records.len(), total);
        }
        let retired_state = retired.borrow().clone();
        residency
            .borrow_mut()
            .execution_complete(&source_diagnostics)?;
        let residency = residency.borrow().clone();
        let report = Report {
            frames: records,
            final_state,
            retired_state,
            committed_work: sessions.report().completed_work,
            status,
            bound_parameters: bound.get(),
            completion_calls: complete.calls,
            executions: complete.executions,
            failed_values,
            residency,
            projections: self.context.projections.lock().unwrap().clone(),
        };
        if self.case.reject_initialization {
            assert!(
                !host.probes.is_empty(),
                "input materialization preceded transition rejection"
            );
            let probes = std::mem::take(&mut host.probes);
            drop(host);
            drop(executor);
            drop(sessions);
            assert!(
                probes.iter().all(|probe| probe.upgrade().is_none()),
                "actual host tensors retired after rejected branch and fixture owners"
            );
        }
        Ok(report)
    }
}
fn run(path: &std::path::Path, config: &moshi::MoshiConfig, case: Case) -> Report {
    let source = moshi::prepare_selected_moshi_realtime_source(selected_with_residency(
        path,
        config,
        case.residency,
    ))
    .unwrap();
    // Resolve the actual content identity, never the metadata-only debug handoff.
    let _actual_content = source.artifact_identity().unwrap();
    let started = Rc::new(Cell::new(false));
    let report = moshi::visit_selected_moshi_realtime_architecture::<NumericBackend, State, _>(
        source,
        &NumericContext::default(),
        Visitor {
            case,
            config: config.clone(),
            context: NumericContext::default(),
            started: started.clone(),
        },
    )
    .unwrap();
    assert!(started.get());
    assert!(report.bound_parameters > 0);
    report
}
fn same_values(a: &[(String, NumericTensor)], b: &[(String, NumericTensor)]) {
    assert_eq!(a.len(), b.len());
    for ((ap, av), (bp, bv)) in a.iter().zip(b) {
        assert_eq!(ap, bp);
        assert_tensor_exact(av, bv, "real selected callback");
    }
}
fn same_outputs(a: &[NumericTensor], b: &[NumericTensor]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_tensor_exact(a, b, "actual completed frame");
    }
}
