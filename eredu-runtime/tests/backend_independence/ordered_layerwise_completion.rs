use super::*;
use eredu_runtime::OrderedLayerwiseCompletion;

#[derive(Debug, PartialEq, Eq)]
enum Call {
    Submit(usize, Vec<i32>),
    Begin,
    Acquire(usize),
    Complete(usize),
    Order(usize, usize),
    PolicyFinish,
    Finish(usize),
    Drop(usize),
    Abort,
}
#[derive(Debug, thiserror::Error)]
enum HookError {
    #[error("injected submission admission failure")]
    Submit,
    #[error("injected ordered boundary failure")]
    Order,
    #[error("injected terminal boundary failure")]
    Finish,
}
struct Trace {
    calls: RefCell<Vec<Call>>,
    next: Cell<usize>,
    fail_submit: Cell<Option<usize>>,
    fail_order: Option<(usize, usize)>,
    fail_finish: Option<usize>,
    shared_executor: Cell<bool>,
}
struct Boundary {
    id: usize,
    orders: Cell<usize>,
    trace: Rc<Trace>,
}
impl Completion for Boundary {
    type Error = HookError;
    fn is_complete(&self) -> Result<bool, HookError> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), HookError> {
        Ok(())
    }
}
impl OrderedLayerwiseCompletion<()> for Boundary {
    fn order_after(&self, _: &()) -> Result<(), HookError> {
        let nth = self.orders.get() + 1;
        self.orders.set(nth);
        self.trace
            .calls
            .borrow_mut()
            .push(Call::Order(self.id, nth));
        if self.trace.fail_order == Some((self.id, nth)) {
            Err(HookError::Order)
        } else {
            Ok(())
        }
    }
    fn finish(self) -> Result<(), HookError> {
        self.trace.calls.borrow_mut().push(Call::Finish(self.id));
        if self.trace.fail_finish == Some(self.id) {
            Err(HookError::Finish)
        } else {
            Ok(())
        }
    }
}
impl Drop for Boundary {
    fn drop(&mut self) {
        self.trace.calls.borrow_mut().push(Call::Drop(self.id));
    }
}
struct Policy {
    inner: RecordingPolicy,
    trace: Rc<Trace>,
}
impl LayerwisePolicy<FakeBackend, FakeUnit> for Policy {
    type Lease = RecordingLease;
    type Error = &'static str;
    fn uses_shared_group_executor(&self, _: &()) -> Result<bool, Self::Error> {
        Ok(self.trace.shared_executor.get())
    }
    fn submit_group(
        &mut self,
        _: &(),
        value: &FakeTensor,
    ) -> Result<
        impl OrderedLayerwiseCompletion<()> + 'static,
        impl std::error::Error + Send + Sync + 'static,
    > {
        let id = self.trace.next.get();
        self.trace.next.set(id + 1);
        self.trace
            .calls
            .borrow_mut()
            .push(Call::Submit(id, value.0.clone()));
        if self.trace.fail_submit.get() == Some(id) {
            return Err(HookError::Submit);
        }
        Ok(Boundary {
            id,
            orders: Cell::new(0),
            trace: Rc::clone(&self.trace),
        })
    }
    fn begin(&mut self, initial: &FakeTensor, context: &()) -> Result<(), Self::Error> {
        self.trace.calls.borrow_mut().push(Call::Begin);
        self.inner.begin(initial, context)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &(),
    ) -> Result<Self::Lease, eredu_runtime::LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&()) -> Result<FakeUnit, E>,
    {
        self.trace.calls.borrow_mut().push(Call::Acquire(ordinal));
        self.inner.acquire(ordinal, address, build, context)
    }
    fn complete<'a, SV, CV>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a FakeTensor,
        state: SV,
        values: CV,
        context: &(),
    ) -> Result<(), Self::Error>
    where
        FakeTensor: 'a,
        SV: Iterator<Item = &'a FakeTensor>,
        CV: Iterator<Item = &'a FakeTensor>,
    {
        self.trace.calls.borrow_mut().push(Call::Complete(ordinal));
        self.inner
            .complete(ordinal, address, lease, output, state, values, context)
    }
    fn finish(&mut self, output: &FakeTensor, context: &()) -> Result<(), Self::Error> {
        self.trace.calls.borrow_mut().push(Call::PolicyFinish);
        self.inner.finish(output, context)
    }
    fn abort(&mut self, active: Option<(usize, ExecutionUnitAddress, Self::Lease)>, context: &()) {
        self.trace.calls.borrow_mut().push(Call::Abort);
        self.inner.abort(active, context);
    }
}
fn units() -> Vec<FakeUnit> {
    [0, 10, 20, 21]
        .into_iter()
        .map(|marker| FakeUnit { marker })
        .collect()
}
fn architecture() -> GroupedFixture {
    GroupedFixture {
        static_modules: FakeOperator,
        trace: Vec::new(),
    }
}
type Runtime =
    LayerwiseRuntime<GroupedFixture, FakeBackend, DeviceState<FakeBackend, FakeLayerState>, Policy>;
fn fixture(fail_order: Option<(usize, usize)>, fail_finish: Option<usize>) -> (Runtime, Rc<Trace>) {
    let trace = Rc::new(Trace {
        calls: RefCell::new(Vec::new()),
        next: Cell::new(0),
        fail_submit: Cell::new(None),
        fail_order,
        fail_finish,
        shared_executor: Cell::new(false),
    });
    let runtime = LayerwiseRuntime::new(
        architecture(),
        Policy {
            inner: RecordingPolicy::new(units()),
            trace: Rc::clone(&trace),
        },
    );
    (runtime, trace)
}
fn run(
    runtime: &mut Runtime,
    parallel: bool,
) -> Result<FakeTensor, eredu_runtime::LayerwiseRuntimeError<Error, &'static str>> {
    let mut state = fixture_state();
    if parallel {
        runtime.forward_parallel(None, &mut state, &(), &())
    } else {
        runtime.forward(None, &mut state, &())
    }
}


fn submission_cause<'a>(
    error: &'a eredu_runtime::LayerwiseRuntimeError<Error, &'static str>,
) -> &'a HookError {
    let retained = std::error::Error::source(error)
        .expect("submission error lost its neutral source owner")
        .downcast_ref::<eredu_core::BackendFailure>()
        .expect("submission source changed type");
    std::error::Error::source(retained)
        .expect("neutral failure lost original cause")
        .downcast_ref::<HookError>()
        .expect("submission cause was formatted or replaced")
}

#[test]
fn policy_creation_error_stops_before_work_and_keeps_previous_owners() {
    // FakeBackend's ordinary creation error is Infallible. This distinct
    // concrete policy error must reach the shared driver without conversion.
    for parallel in [false, true] {
        for failed in [0, 2] {
            let (mut runtime, trace) = fixture(None, None);
            trace.fail_submit.set(Some(failed));
            let error = run(&mut runtime, parallel).unwrap_err();
            assert!(matches!(submission_cause(&error), HookError::Submit));
            assert!(error.to_string().contains("injected submission admission failure"));
            assert_eq!(runtime.architecture().trace.len(), failed);
            assert_eq!(runtime.policy().inner.aborts, usize::from(failed != 0));
            assert!(!runtime.policy().inner.forward_active);
            let calls = trace.calls.borrow();
            assert!(!calls
                .iter()
                .any(|call| matches!(call, Call::PolicyFinish | Call::Finish(_))));
            assert_eq!(
                calls
                    .iter()
                    .filter(|call| matches!(call, Call::Submit(_, _)))
                    .count(),
                failed + 1
            );
            let mut dropped = calls
                .iter()
                .filter_map(|call| match call {
                    Call::Drop(id) => Some(*id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            dropped.sort_unstable();
            assert_eq!(dropped, (0..failed).collect::<Vec<_>>());
        }
    }
}

#[test]
fn policy_ordered_completion_routes_initial_dependencies_readout_and_terminal_retirement() {
    for parallel in [false, true] {
        SUBMIT_COUNT.set(0);
        ORDER_COUNT.set(0);
        let (mut runtime, trace) = fixture(None, None);
        assert_eq!(
            run(&mut runtime, parallel).unwrap(),
            FakeTensor(vec![30, 20, 21])
        );
        assert_eq!(
            runtime.architecture().trace,
            [(0, 0), (1, 0), (2, 0), (2, 1)]
        );
        assert_eq!(
            (SUBMIT_COUNT.get(), ORDER_COUNT.get()),
            (0, 0),
            "custom policy owns actual submission and ordering"
        );
        assert_eq!(
            *trace.calls.borrow(),
            [
                Call::Submit(0, vec![0]),
                Call::Begin,
                Call::Order(0, 1),
                Call::Acquire(0),
                Call::Complete(0),
                Call::Submit(1, vec![10, 0]),
                Call::Order(0, 2),
                Call::Acquire(1),
                Call::Complete(1),
                Call::Submit(2, vec![20, 10]),
                Call::Order(1, 1),
                Call::Order(2, 1),
                Call::Acquire(2),
                Call::Complete(2),
                Call::Acquire(3),
                Call::Complete(3),
                Call::Submit(3, vec![30, 20, 21]),
                Call::Order(3, 1),
                Call::PolicyFinish,
                Call::Finish(1),
                Call::Drop(1),
                Call::Finish(2),
                Call::Drop(2),
                Call::Finish(3),
                Call::Drop(3),
                Call::Finish(0),
                Call::Drop(0),
            ]
        );
        assert_eq!(runtime.policy().inner.aborts, 0);
        assert!(!runtime.policy().inner.forward_active);
    }
}

#[test]
fn policy_order_failure_stops_later_units_and_preserves_completion_drop_owners() {
    for parallel in [false, true] {
        for (producer, expected_units, owners) in [(0, 0, 1), (1, 2, 3), (3, 4, 4)] {
            let (mut runtime, trace) = fixture(Some((producer, 1)), None);
            let error = run(&mut runtime, parallel).unwrap_err();
            assert!(matches!(submission_cause(&error), HookError::Order));
            assert!(error.to_string().contains("injected ordered boundary failure"));
            assert_eq!(runtime.architecture().trace.len(), expected_units);
            assert_eq!(runtime.policy().inner.aborts, 1);
            assert!(!runtime.policy().inner.forward_active);
            assert!(runtime.policy().inner.units.iter().all(Option::is_some));
            let calls = trace.calls.borrow();
            assert!(!calls
                .iter()
                .any(|call| matches!(call, Call::PolicyFinish | Call::Finish(_))));
            let mut dropped = calls
                .iter()
                .filter_map(|call| match call {
                    Call::Drop(id) => Some(*id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            dropped.sort_unstable();
            assert_eq!(dropped, (0..owners).collect::<Vec<_>>());
        }
    }
}

#[test]
fn policy_terminal_failure_keeps_remaining_completions_on_their_actual_drop_path() {
    for parallel in [false, true] {
        let (mut runtime, trace) = fixture(None, Some(2));
        let error = run(&mut runtime, parallel).unwrap_err();
        assert!(matches!(submission_cause(&error), HookError::Finish));
        assert!(error.to_string().contains("injected terminal boundary failure"));
        assert_eq!(runtime.architecture().trace.len(), 4);
        let calls = trace.calls.borrow();
        let terminal = calls
            .iter()
            .position(|call| matches!(call, Call::PolicyFinish))
            .unwrap();
        assert_eq!(
            &calls[terminal..],
            [
                Call::PolicyFinish,
                Call::Finish(1),
                Call::Drop(1),
                Call::Finish(2),
                Call::Drop(2),
                Call::Drop(3),
                Call::Drop(0)
            ]
        );
        assert!(!runtime.policy().inner.forward_active);
    }
}

#[test]
fn default_ordered_policy_preserves_ordinary_submissions_and_dependency_results() {
    for parallel in [false, true] {
        FORK_COUNT.set(0);
        SUBMIT_COUNT.set(0);
        ORDER_COUNT.set(0);
        let mut runtime = LayerwiseRuntime::<_, FakeBackend, _, _>::new(
            architecture(),
            RecordingPolicy::new(units()),
        );
        let mut state = fixture_state();
        let output = if parallel {
            runtime.forward_parallel(None, &mut state, &(), &())
        } else {
            runtime.forward(None, &mut state, &())
        }
        .unwrap();
        assert_eq!(output, FakeTensor(vec![30, 20, 21]));
        assert_eq!(
            (FORK_COUNT.get(), SUBMIT_COUNT.get(), ORDER_COUNT.get()),
            (1, 4, 5)
        );
        assert_eq!(runtime.policy().aborts, 0);
    }
}

#[test]
fn explicit_shared_executor_preserves_group_results_and_failures_without_backend_forks() {
    for parallel in [false, true] {
        for failure in [None, Some((1, 1))] {
            FORK_COUNT.set(0);
            let (mut ordinary, ordinary_trace) = fixture(failure, None);
            let expected = run(&mut ordinary, parallel);
            assert_eq!(FORK_COUNT.get(), 1);
            FORK_COUNT.set(0);
            let (mut shared, shared_trace) = fixture(failure, None);
            shared_trace.shared_executor.set(true);
            let actual = run(&mut shared, parallel);
            assert_eq!(
                FORK_COUNT.get(),
                0,
                "explicit prepared choice must not construct group executors"
            );
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => assert_eq!(expected, actual),
                (Err(expected), Err(actual)) => {
                    assert!(matches!(submission_cause(&expected), HookError::Order));
                    assert!(matches!(submission_cause(&actual), HookError::Order));
                }
                _ => panic!("shared executor changed the group outcome"),
            }
            assert_eq!(ordinary.architecture().trace, shared.architecture().trace);
            assert_eq!(*ordinary_trace.calls.borrow(), *shared_trace.calls.borrow());
            assert_eq!(ordinary.policy().inner.aborts, shared.policy().inner.aborts);
        }
    }
}
