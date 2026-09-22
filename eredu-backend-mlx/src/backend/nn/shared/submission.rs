#[path = "submission/prepared_collectives.rs"]
mod prepared_collectives;
use super::*;
use crate::backend::runtime::distributed::completion::MlxNeuralCommunicationCompletion;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
pub(crate) use prepared_collectives::control_bytes as value_completion_control_bytes;
use safemlx::OperationEvent;
use std::{cell::Cell, rc::Rc};

mod ordinary_boundary;
mod ordinary_broadcast;
mod ordinary_controls;
mod ordinary_local;
pub(crate) use ordinary_local::ordinary_expert_local_control_bytes;
mod ordinary_owner;
mod original;
pub(crate) use ordinary_owner::{
    OrdinaryExecutionOwner, OrdinaryExecutionRegistration, current_ordinary_execution_owner,
};
pub(crate) use original::{
    NeuralSubmissionShape, OriginalArraySubmissionCause, OriginalArraySubmissionFailure,
    OriginalNeuralSubmissionCompletion, OriginalObservationFailure, OriginalObservationSite,
    OriginalSubmissionFailure, PointwisePreparationFailure, PointwiseSubmissionPlan,
    PreparedNeuralSubmission, PreparedPointwiseSubmission, SubmissionPreparationCause,
    SubmissionPreparationError,
};

struct HostResourceNode {
    values: Vec<Box<dyn Send>>,
    next: Option<Box<HostResourceNode>>,
}

struct HostResources(Option<Box<HostResourceNode>>);

impl HostResources {
    fn new(values: Vec<Box<dyn Send>>) -> Self {
        Self(Some(Box::new(HostResourceNode { values, next: None })))
    }

    fn push(&mut self, value: Box<dyn Send>) {
        self.0
            .as_mut()
            .expect("live host resource owner")
            .values
            .push(value);
    }
}

#[derive(Default)]
struct RetiredHostResources(Option<Box<HostResourceNode>>);

impl Drop for RetiredHostResources {
    fn drop(&mut self) {
        // Arbitrary user destructors are not safe to invoke during TLS teardown.
        if let Some(node) = self.0.take() {
            std::mem::forget(node);
        }
    }
}

thread_local! {
    static RETIRED_HOST_RESOURCES: RefCell<RetiredHostResources> = RefCell::default();
    static RECLAIMING_HOST_RESOURCES: Cell<bool> = const { Cell::new(false) };
}

impl Drop for HostResources {
    fn drop(&mut self) {
        let mut node = self.0.take();
        if node.as_ref().is_some_and(|node| node.values.is_empty()) {
            return;
        }
        let _ = RETIRED_HOST_RESOURCES.try_with(|retired| {
            if let Ok(mut retired) = retired.try_borrow_mut() {
                let mut owned = node.take().expect("one preallocated host resource node");
                owned.next = retired.0.take();
                retired.0 = Some(owned);
            }
        });
        if let Some(node) = node {
            std::mem::forget(node);
        }
    }
}

impl MlxNeuralBackend {
    /// Reclaims this thread's retired application-owned submission resources.
    ///
    /// This ordinary host operation may run arbitrary Rust destructors and block.
    /// Polling, completion Drop, and TLS teardown never invoke those destructors.
    /// New submissions also call this method. Reentrant calls while inside the
    /// native runtime, calls while unwinding, and recursive reclamation are deferred.
    /// If the owner thread exits before reclamation, these resources remain retained.
    /// If a resource destructor panics, the rest of that detached batch remains
    /// permanently retained instead of running more destructors during unwinding.
    /// Ready submission, host, ordinary and native-owner queues are drained until
    /// no owner retires. Unresolved submissions remain quarantined without waiting.
    pub fn reclaim_retired_resources() {
        if !safemlx::can_reclaim_submission_resources() {
            return;
        }
        let _ = RECLAIMING_HOST_RESOURCES.try_with(|reclaiming| {
            if reclaiming.replace(true) {
                return;
            }
            struct Reset<'a>(&'a Cell<bool>);
            impl Drop for Reset<'_> {
                fn drop(&mut self) {
                    self.0.set(false);
                }
            }
            let _reset = Reset(reclaiming);
            loop {
                let mut reclaimed = crate::backend::submission_recovery::reap_with_progress();
                // Recovery can settle the final native Record after the last
                // numerical runtime entry. Its registry alias still owns the
                // Scope and physical observer until this ordinary closed pass.
                // Busy/error retains every unresolved native owner unchanged.
                let _ = safemlx::try_retire_completed_submissions();
                let pending = RETIRED_HOST_RESOURCES
                    .try_with(|retired| {
                        retired
                            .try_borrow_mut()
                            .ok()
                            .and_then(|mut retired| retired.0.take())
                    })
                    .ok()
                    .flatten();
                let mut pending = RetiredHostResources(pending);
                while let Some(node) = pending.0.as_mut() {
                    if let Some(value) = node.values.pop() {
                        // If this destructor panics, the snapshot's Drop retains
                        // every remaining value rather than invoking more user
                        // destructors during unwinding.
                        drop(value);
                        reclaimed = true;
                    } else {
                        let mut node = pending.0.take().expect("current retired node");
                        pending.0 = node.next.take();
                        drop(node);
                        reclaimed = true;
                    }
                }
                reclaimed |= crate::backend::ordinary_retirement::reclaim();
                // Array retirement callbacks can stage more host/ordinary owners,
                // and their destructors can in turn retire more native backing.
                reclaimed |= safemlx::reclaim_allocation_owners() != 0;
                if !reclaimed {
                    break;
                }
            }
        });
    }
}

struct SubmissionResources {
    ordinary: Option<OrdinaryExecutionOwner>,
    event: RefCell<Option<Rc<OperationEvent>>>,
    _arrays: Vec<Array>,
    additional: RefCell<HostResources>,
    children: Cell<usize>,
    failed: Cell<bool>,
}

impl Retention for SubmissionResources {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.failed.set(true);
        }
    }
}

struct ConsumerResources {
    owner: Rc<SubmissionResources>,
    _stream: Stream,
}

impl ConsumerResources {
    fn new(owner: &Rc<SubmissionResources>, stream: &Stream) -> Self {
        owner.children.set(
            owner
                .children
                .get()
                .checked_add(1)
                .expect("consumer ticket overflow"),
        );
        Self {
            owner: Rc::clone(owner),
            _stream: stream.clone(),
        }
    }
}

impl Retention for ConsumerResources {
    fn observe(&self, status: Status) {
        self.owner.observe(status);
    }
}

impl Drop for ConsumerResources {
    fn drop(&mut self) {
        self.owner.children.set(self.owner.children.get() - 1);
    }
}

struct ConsumerUnwind<'a>(&'a Cell<bool>);
impl Drop for ConsumerUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.set(true);
        }
    }
}

/// Exact MLX completion retaining every Rust-side submission resource.
pub struct MlxSubmissionCompletion {
    event: Rc<OperationEvent>,
    retained: Recovery<Rc<SubmissionResources>>,
}

impl MlxSubmissionCompletion {
    fn new(event: OperationEvent, retained: Recovery<Rc<SubmissionResources>>) -> Self {
        let event = Rc::new(event);
        *retained.retention().event.borrow_mut() = Some(Rc::clone(&event));
        Self { event, retained }
    }
    fn check_native_status(&self) -> Result<bool, safemlx::error::Exception> {
        crate::backend::submission_recovery::reap();
        let status = self.retained.progress();
        if status.failed || status.blocked || self.retained.retention().failed.get() {
            Err(safemlx::error::Exception::custom(
                "native submission failed; unresolved resources remain retained",
            ))
        } else {
            Ok(status.settled && self.retained.retention().children.get() == 0)
        }
    }

    fn retain<T: Send + 'static>(&self, value: T) {
        self.retained
            .retention()
            .additional
            .borrow_mut()
            .push(Box::new(value));
    }

    /// Orders a consumer stream after this exact completion without blocking.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), safemlx::error::Exception> {
        self.check_native_status()?;
        let owner = self.retained.retention();
        let mut child = ordinary_owner::begin(
            ConsumerResources::new(owner, stream),
            owner.ordinary.as_ref(),
        )?;
        let _unwind = ConsumerUnwind(&owner.failed);
        let result = self.event.wait_on(stream);
        if result.is_err() {
            owner.failed.set(true);
        }
        child.seal();
        let status = child.progress();
        if status.failed || status.blocked {
            return Err(safemlx::error::Exception::custom(
                "consumer dependency failed; native resources remain retained",
            ));
        }
        result
    }
}

impl Completion for MlxSubmissionCompletion {
    type Error = safemlx::error::Exception;

    fn resources_releasable(&self) -> bool {
        safemlx::try_with_submission_retirement(|| {
            crate::backend::submission_recovery::reap();
            self.retained.progress().settled && self.retained.retention().children.get() == 0
        })
        .unwrap_or(false)
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            let complete = self.event.is_complete()?;
            Ok(complete && self.check_native_status()?)
        })
        .unwrap_or(Ok(false))
    }

    fn wait(&self) -> Result<(), Self::Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl SubmissionBackend for MlxNeuralBackend {
    type Executor = Stream;
    type OwnedExecutor = Stream;
    type Completion = MlxSubmissionCompletion;

    fn fork_executors(
        executor: &Self::Executor,
        count: usize,
    ) -> Result<Vec<Self::OwnedExecutor>, safemlx::error::Exception> {
        if count <= 1 {
            return Ok((0..count).map(|_| executor.clone()).collect());
        }
        let device = executor.get_device()?;
        Ok((0..count)
            .map(|_| Stream::new_with_device(&device))
            .collect())
    }

    fn submit<'a, I>(
        _executor: &Self::Executor,
        values: I,
    ) -> Result<Self::Completion, safemlx::error::Exception>
    where
        MlxTensor: 'a,
        I: IntoIterator<Item = &'a MlxTensor>,
    {
        Self::reclaim_retired_resources();
        let ordinary = current_ordinary_execution_owner()?;
        let arrays = values
            .into_iter()
            .map(|value| value.as_array().clone())
            .collect();
        let resources = Rc::new(SubmissionResources {
            ordinary: ordinary.clone(),
            event: RefCell::new(None),
            _arrays: arrays,
            additional: RefCell::new(HostResources::new(Vec::new())),
            children: Cell::new(0),
            failed: Cell::new(false),
        });
        let mut retained = ordinary_owner::begin(resources, ordinary.as_ref())?;
        let event = safemlx::transforms::async_eval_with_operation_event(
            retained.retention()._arrays.iter(),
        );
        retained.seal();
        let completion = MlxSubmissionCompletion::new(event?, retained);
        completion.check_native_status()?;
        Ok(completion)
    }

    fn order_after(
        completion: &Self::Completion,
        executor: &Self::Executor,
    ) -> Result<(), safemlx::error::Exception> {
        completion.wait_on(executor)
    }

    fn retain_until_complete<T: Send + 'static>(
        _executor: &Self::Executor,
        completion: &Self::Completion,
        value: T,
    ) -> Result<(), <Self::Completion as Completion>::Error> {
        completion.retain(value);
        Ok(())
    }
}

const MLX_COMMUNICATION_MAX_ELEMENTS: usize = i32::MAX as usize;

#[cfg(test)]
mod consumer_scope_tests {
    use super::*;
    use crate::backend::submission_recovery::Probe;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeProbe(Rc<Cell<Status>>);
    impl Probe for FakeProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.0.get()
        }
    }
    struct DropWitness(Arc<AtomicUsize>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn explicit_reclamation_finishes_cross_queue_retirement_without_waiting_for_pending_work() {
        use crate::backend::ordinary_retirement::OrdinaryRetirement;

        struct HostToOrdinary(Arc<AtomicUsize>);
        impl Drop for HostToOrdinary {
            fn drop(&mut self) {
                drop(OrdinaryRetirement::new(DropWitness(Arc::clone(&self.0))));
            }
        }
        struct NativeToHost(Arc<AtomicUsize>);
        impl Drop for NativeToHost {
            fn drop(&mut self) {
                assert!(safemlx::can_reclaim_submission_resources());
                drop(HostResources::new(vec![Box::new(HostToOrdinary(
                    Arc::clone(&self.0),
                ))]));
            }
        }
        struct PendingRetention(Arc<AtomicUsize>);
        impl Retention for PendingRetention {
            fn observe(&self, _: Status) {}
        }
        impl Drop for PendingRetention {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let pending_drops = Arc::new(AtomicUsize::new(0));
        let pending_status = Rc::new(Cell::new(Status {
            settled: false,
            failed: false,
            blocked: false,
        }));
        drop(Recovery::with_probe(
            PendingRetention(Arc::clone(&pending_drops)),
            FakeProbe(Rc::clone(&pending_status)),
        ));

        let array = Array::from_slice(&[1.25f32, -2.5], &[2]);
        array.evaluated().unwrap();
        array
            .retain_allocation_owner(NativeToHost(Arc::clone(&drops)))
            .unwrap();
        drop(OrdinaryRetirement::new(array));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        MlxNeuralBackend::reclaim_retired_resources();
        // Cached backing retains its allocation owners after the array retires.
        // Eviction makes that backing eligible for cross-queue reclamation.
        safemlx::memory::clear_cache().unwrap();
        MlxNeuralBackend::reclaim_retired_resources();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(pending_drops.load(Ordering::SeqCst), 0);

        pending_status.set(Status {
            settled: true,
            failed: false,
            blocked: false,
        });
        MlxNeuralBackend::reclaim_retired_resources();
        assert_eq!(pending_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn completed_poll_and_drop_defer_arbitrary_destructors_until_unlocked_reclamation() {
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let tensor = MlxTensor::from_array(Array::ones::<f32>(&[4], &stream).unwrap());
        let completion = MlxNeuralBackend::submit(&stream, [&tensor]).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        completion.retain(DropWitness(Arc::clone(&drops)));
        completion.wait().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(completion);
        crate::backend::submission_recovery::reap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while safemlx::try_with_submission_retirement(MlxNeuralBackend::reclaim_retired_resources)
            .is_none()
        {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            MlxNeuralBackend::reclaim_retired_resources();
            drops.load(Ordering::SeqCst) == 1
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn thread_exit_never_runs_arbitrary_retired_resource_destructors() {
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_drops = Arc::clone(&drops);
        std::thread::spawn(move || {
            drop(HostResources::new(vec![Box::new(DropWitness(
                worker_drops,
            ))]));
        })
        .join()
        .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn panicking_resource_destructor_does_not_unwind_through_other_retired_values() {
        struct Panics;
        impl Drop for Panics {
            fn drop(&mut self) {
                panic!("injected retained-resource destructor panic");
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        drop(HostResources::new(vec![
            Box::new(DropWitness(Arc::clone(&drops))),
            Box::new(Panics),
        ]));
        let result = std::panic::catch_unwind(MlxNeuralBackend::reclaim_retired_resources);
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        MlxNeuralBackend::reclaim_retired_resources();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn pending_consumer_retains_additional_resources_after_primary_owner_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = Rc::new(SubmissionResources {
            ordinary: None,
            event: RefCell::new(None),
            _arrays: Vec::new(),
            additional: RefCell::new(HostResources::new(vec![Box::new(DropWitness(Arc::clone(
                &drops,
            )))])),
            children: Cell::new(0),
            failed: Cell::new(false),
        });
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let native = Rc::new(Cell::new(Status {
            settled: false,
            failed: false,
            blocked: false,
        }));
        let child = Recovery::with_probe(
            ConsumerResources::new(&owner, &stream),
            FakeProbe(Rc::clone(&native)),
        );
        assert_eq!(owner.children.get(), 1);
        drop(owner);
        drop(child);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        native.set(Status {
            settled: true,
            failed: false,
            blocked: false,
        });
        crate::backend::submission_recovery::reap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            MlxNeuralBackend::reclaim_retired_resources();
            drops.load(Ordering::SeqCst) == 1
        });
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn healthy_parent_observation_does_not_clear_consumer_failure() {
        let owner = Rc::new(SubmissionResources {
            ordinary: None,
            event: RefCell::new(None),
            _arrays: Vec::new(),
            additional: RefCell::new(HostResources::new(Vec::new())),
            children: Cell::new(0),
            failed: Cell::new(false),
        });
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let child = Recovery::with_probe(
            ConsumerResources::new(&owner, &stream),
            FakeProbe(Rc::new(Cell::new(Status {
                settled: true,
                failed: true,
                blocked: false,
            }))),
        );
        child.progress();
        owner.observe(Status {
            settled: true,
            failed: false,
            blocked: false,
        });
        assert!(owner.failed.get());
        drop(child);
        crate::backend::submission_recovery::wait_for_retirement(|| owner.children.get() == 0);
        assert_eq!(owner.children.get(), 0);
        assert!(owner.failed.get());
    }
}

#[derive(Clone, Copy)]
enum MlxCommunicationDtypes {
    Floating,
    FloatingAndI32,
    FloatingI32AndU32,
}

impl MlxCommunicationDtypes {
    fn admits(self, dtype: Dtype) -> bool {
        let floating = matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
        match self {
            Self::Floating => floating,
            Self::FloatingAndI32 => floating || dtype == Dtype::Int32,
            Self::FloatingI32AndU32 => floating || matches!(dtype, Dtype::Int32 | Dtype::Uint32),
        }
    }
}

fn validate_communication_tensor(
    value: &Array,
    dtypes: MlxCommunicationDtypes,
) -> Result<(), safemlx::error::Exception> {
    if !dtypes.admits(value.dtype()) {
        return Err(safemlx::error::Exception::custom(format!(
            "MLX communication does not advertise dtype {:?}",
            value.dtype()
        )));
    }
    if value.ndim() > i32::MAX as usize || value.size() > MLX_COMMUNICATION_MAX_ELEMENTS {
        return Err(safemlx::error::Exception::custom(
            "MLX communication tensor exceeds advertised rank or element limits",
        ));
    }
    Ok(())
}

fn validate_route_bundle(
    values: &[MlxTensor],
    route: &CommunicationRouteRealization,
) -> Result<(), safemlx::error::Exception> {
    let requirement = route.descriptor().requirement();
    let limits = requirement.limits().ok_or_else(|| {
        safemlx::error::Exception::custom("point-to-point route has no tensor limits")
    })?;
    if values.is_empty() || values.len() > limits.max_tensors() {
        return Err(safemlx::error::Exception::custom(format!(
            "point-to-point route bundle has {} tensors, expected 1..={}",
            values.len(),
            limits.max_tensors()
        )));
    }
    for value in values {
        let array = value.as_array();
        validate_communication_tensor(array, MlxCommunicationDtypes::FloatingI32AndU32)?;
        let dtype = crate::tensor::portable_dtype(array.dtype());
        if !requirement.dtypes().contains(&dtype) {
            return Err(safemlx::error::Exception::custom(format!(
                "point-to-point route does not admit dtype {dtype:?}"
            )));
        }
        if array.ndim() > limits.max_tensor_rank() || array.size() > limits.max_tensor_elements() {
            return Err(safemlx::error::Exception::custom(format!(
                "point-to-point placeholder shape {:?} exceeds route limits",
                array.shape()
            )));
        }
    }
    Ok(())
}

fn collective_completion(
    input: Array,
    output: &Array,
    group: &Group,
    executor: &Stream,
    count_buffers: Vec<Vec<usize>>,
) -> Result<MlxCommunicationCompletion, safemlx::error::Exception> {
    MlxCommunicationCompletion::submit(
        [output],
        vec![input, output.clone()],
        count_buffers,
        vec![group.clone()],
        Vec::new(),
        vec![executor.clone()],
    )
}

impl CommunicationBackend for MlxNeuralBackend {
    fn with_expert_inactive_wave<E, F>(
        source: Option<eredu_nn::workspace::WorkspaceExpertInactiveWave>,
        context: Option<&Group>,
        executor: &Stream,
        run: F,
    ) -> Result<Result<(), E>, Self::CommunicationError>
    where
        F: FnOnce() -> Result<(), E>,
    {
        let Some(context) = context else {
            return Ok(run());
        };
        Self::with_parallel_control_context(context, |bound| {
            if bound.is_none() {
                return Ok(run());
            }
            let source = source.ok_or_else(|| {
                crate::backend::error::Error::Neural(context.model_source_missing())
            })?;
            context
                .original_control_request()
                .ok_or_else(|| {
                    crate::backend::error::Error::Neural(context.model_source_missing())
                })?
                .with_expert_inactive_wave(source, context, executor, run)
        })?
    }
    fn with_expert_provider_wave<E, F>(
        source: eredu_nn::workspace::WorkspaceExpertProviderWave,
        context: Option<&Group>,
        executor: &Stream,
        run: F,
    ) -> Result<Result<(), E>, Self::CommunicationError>
    where
        F: FnOnce() -> Result<(), E>,
    {
        let Some(context) = context else {
            return Ok(run());
        };
        Self::with_parallel_control_context(context, |bound| {
            if bound.is_none() {
                return Ok(run());
            }
            context
                .original_control_request()
                .ok_or_else(|| {
                    crate::backend::error::Error::Neural(context.model_source_missing())
                })?
                .with_expert_provider_wave(source, context, executor, run)
        })?
    }
    fn with_expert_route_region<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P,
        input: &MlxTensor,
        routes: &eredu_nn::GroupSelection<MlxTensor>,
        context: Option<&Group>,
        executor: &Stream,
        run: F,
    ) -> Result<
        Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
        Self::CommunicationError,
    >
    where
        P: eredu_nn::Parameterized<MlxTensor>,
        F: FnOnce(
            &mut P,
            Option<eredu_runtime::PreparedExpertMovementLoan<'_>>,
        ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
    {
        let _ = (input, routes);
        let Some(context) = context else {
            return Ok(run(bank, None));
        };
        Self::with_parallel_control_context(context, |bound| {
            if bound.is_none() {
                return Ok(run(bank, None));
            }
            context
                .original_control_request()
                .ok_or_else(|| {
                    crate::backend::error::Error::Neural(context.model_source_missing())
                })?
                .with_expert_region(source, bank, context, executor, run)
        })?
    }
    fn with_ordinary_expert_route_local<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P,
        input: &MlxTensor,
        scores: &MlxTensor,
        coefficients: &MlxTensor,
        local_indices: &[i32],
        context: Option<&Group>,
        executor: &Stream,
        run: F,
    ) -> Result<
        Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
        Self::CommunicationError,
    >
    where
        P: eredu_nn::Parameterized<MlxTensor>,
        F: FnOnce(&mut P) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
    {
        let _ = (context, executor);
        ordinary_local::run(
            source,
            bank,
            input,
            scores,
            coefficients,
            local_indices,
            run,
        )
    }

    fn with_expert_route_local<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P,
        input: &MlxTensor,
        scores: &MlxTensor,
        coefficients: &MlxTensor,
        completed: &eredu_core::ErasedSharedStorageOwner,
        local_rows: &[usize],
        context: &Group,
        executor: &Stream,
        run: F,
    ) -> Result<
        Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
        Self::CommunicationError,
    >
    where
        P: eredu_nn::Parameterized<MlxTensor>,
        F: FnOnce(&mut P) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<MlxTensor>, E>,
    {
        context
            .original_control_request()
            .ok_or_else(|| crate::backend::error::Error::Neural(context.model_source_missing()))?
            .with_expert_local(
                source,
                bank,
                input,
                scores,
                coefficients,
                completed,
                local_rows,
                context,
                executor,
                run,
            )
    }

    fn prepare_expert_route_input(
        values: &[i32],
        source: &eredu_core::ErasedSharedStorageOwner,
        context: &Group,
        executor: &Stream,
    ) -> Result<Option<MlxTensor>, Self::CommunicationError> {
        Self::with_parallel_control_context(context, |prepared| {
            let Some(_) = prepared else {
                return Ok(None);
            };
            context
                .original_control_request()
                .ok_or_else(|| {
                    crate::backend::error::Error::Neural(context.model_source_missing())
                })?
                .prepare_expert_route_input(values, source, context, executor)
                .map(|value| Some(MlxTensor::from_array(value)))
        })?
    }

    fn with_prepared_expert_route_indices<T, E, F>(
        value: &MlxTensor,
        context: &Group,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: for<'loan> FnOnce(
            Option<(
                &'loan [i32],
                &'loan eredu_nn::workspace::HostMetadataFunding,
            )>,
        ) -> Result<T, E>,
    {
        prepared_collectives::expert_input::with_indices(value, context, executor, run)
    }
    fn with_prepared_peer_count_consensus<T, E, F>(
        local: &[i32],
        group: &Group,
        context: &Group,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: for<'loan> FnOnce(
            Option<(
                &'loan [i32],
                &'loan eredu_nn::workspace::HostMetadataFunding,
            )>,
        ) -> Result<T, E>,
    {
        Self::with_parallel_control_context(context, |source| {
            let Some(_) = source else {
                return Ok(run(None));
            };
            let projection = context.original_control_request().ok_or_else(|| {
                crate::backend::error::Error::Neural(context.model_source_missing())
            })?;
            projection.with_peer_count_consensus(local, group, context, executor, run)
        })?
    }
    fn with_prepared_peer_count_source<T, E, F>(
        local: &[i32],
        group: &Group,
        context: &Group,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: for<'loan> FnOnce(Option<eredu_runtime::PreparedPeerCountLoan<'loan>>) -> Result<T, E>,
    {
        Self::with_parallel_control_context(context, |source| {
            let Some(_) = source else {
                return Ok(run(None));
            };
            let projection = context.original_control_request().ok_or_else(|| {
                crate::backend::error::Error::Neural(context.model_source_missing())
            })?;
            projection.with_peer_count_source(local, group, context, executor, run)
        })?
    }
    fn complete_model_dependencies(
        values: &[&MlxTensor],
        context: &Group,
        executor: &Stream,
    ) -> Result<Option<()>, Self::CommunicationError> {
        prepared_collectives::complete(values, context, executor)
    }
    fn submit_prepared_boundary_dependencies(
        values: &[eredu_runtime::ArchitectureBoundaryValue<MlxTensor>],
        source: &eredu_runtime::PreparedBoundarySource,
        route: &CommunicationRouteRealization,
        context: &Group,
        executor: &Stream,
    ) -> Result<Option<Submission<(), Self::CommunicationCompletion>>, Self::CommunicationError>
    {
        let request = context
            .original_control_request()
            .ok_or_else(|| crate::backend::error::Error::Neural(context.model_source_missing()))?;
        let completion =
            request.submit_boundary_dependencies(context, values, source, route, executor)?;
        Ok(Some(Submission {
            output: (),
            completion: completion.into(),
        }))
    }

    fn prepare_boundary_source(
        prepared: &Group,
        route: &CommunicationRouteRealization,
    ) -> Result<Option<eredu_runtime::PreparedBoundarySource>, eredu_core::BackendFailure> {
        if let Some(request) = prepared.original_control_request() {
            return request
                .prepare_boundary_source(prepared, route)
                .map(Some)
                .map_err(crate::backend::error::Error::into_backend_failure);
        }
        if prepared.has_original_parallel() || prepared.has_original_control() {
            return Err(
                crate::backend::error::Error::Neural(prepared.model_source_missing())
                    .into_backend_failure(),
            );
        }
        Ok(None)
    }
    fn with_prepared_publication_group<T, E, F>(
        group: &Group,
        prepared: &Group,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        prepared.with_prepared_publication_group(group, funding, executor, run)
    }

    fn with_parallel_control_context<T, E, F>(
        prepared: &Group,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: FnOnce(Option<(&Group, &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        if let Some(request) = prepared.original_control_request() {
            return request.with_context(prepared, run);
        }
        if let Some(binding) = prepared.original_control() {
            return binding.with_context(prepared, run);
        }
        if prepared.has_original_parallel() {
            return Err(crate::backend::error::Error::Neural(
                prepared.model_source_missing(),
            ));
        }
        Ok(run(None))
    }
    fn with_model_control_context<T, E, F>(
        prepared: &Group,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: FnOnce(Option<(&Group, &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        match prepared.original_control_request() {
            Some(request) => request.with_model_context(prepared, run),
            None => Self::with_parallel_control_context(prepared, run),
        }
    }
    fn with_prepared_control_group<T, E, F>(
        event: eredu_runtime::replicated_session::ParallelControlEvent,
        group: &Group,
        prepared: &Group,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        executor: &Stream,
        run: F,
    ) -> Result<Result<T, E>, Self::CommunicationError>
    where
        F: FnOnce(Option<&Group>) -> Result<T, E>,
    {
        match prepared.original_control() {
            Some(binding) => binding.with_group(event, group, prepared, funding, executor, run),
            None => match prepared.original_control_request() {
                Some(request) => request.with_group(event, group, funding, executor, run),
                None => Ok(run(None)),
            },
        }
    }

    type CommunicationGroup = Group;
    type CommunicationRoute = CommunicationRouteRealization;
    type CommunicationCompletion = MlxNeuralCommunicationCompletion;
    type CommunicationError = crate::backend::error::Error;

    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>,
    {
        let outputs = values
            .into_iter()
            .map(|value| value.as_array().clone())
            .collect::<Vec<_>>();
        #[cfg(test)]
        if std::env::var_os("EREDU_TEST_PARTITION_COLLECTIVE_TRACE").is_some() {
            eprintln!(
                "partition-schedule rank={} operation=local-dependencies shapes={:?}",
                std::env::var("MLX_RANK").unwrap_or_else(|_| "?".into()),
                outputs.iter().map(Array::shape).collect::<Vec<_>>(),
            );
        }
        let retained = outputs.clone();
        let completion = MlxCommunicationCompletion::submit(
            outputs.iter(),
            retained,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![executor.clone()],
        )?;
        Ok(Submission {
            output: (),
            completion: completion.into(),
        })
    }
}

/// Mechanical tensor metadata used by the backend-neutral partition driver.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MlxCommunicationTensorMetadata;

impl eredu_runtime::CommunicationTensorMetadata<MlxNeuralBackend>
    for MlxCommunicationTensorMetadata
{
    fn matches_shape_with_funding(
        &self,
        tensor: &MlxTensor,
        shape: &[i32],
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Option<bool>, eredu_nn::workspace::HostMetadataFundingError> {
        let parts = [
            std::mem::size_of::<(
                &Self,
                &MlxTensor,
                &[i32],
                &eredu_nn::workspace::HostMetadataFunding,
            )>(),
            std::mem::size_of::<Result<Option<bool>, eredu_nn::workspace::HostMetadataFundingError>>(
            ),
        ];
        funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or(eredu_nn::workspace::HostMetadataFundingError::Overflow)?,
        )?;
        Ok(Some(tensor.as_array().shape() == shape))
    }
    fn fixed_metadata_with_funding(
        &self,
        tensor: &MlxTensor,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        Option<(TensorDtype, usize, Option<usize>)>,
        eredu_nn::workspace::HostMetadataFundingError,
    > {
        use eredu_nn::workspace::HostMetadataFundingError;
        let frames = [
            std::mem::size_of::<(&Self, &MlxTensor, &eredu_nn::workspace::HostMetadataFunding)>(),
            std::mem::size_of::<(TensorDtype, usize, Option<usize>)>(),
            std::mem::size_of::<std::slice::Iter<'_, i32>>(),
            std::mem::size_of::<
                Result<Option<(TensorDtype, usize, Option<usize>)>, HostMetadataFundingError>,
            >(),
        ];
        funding.reserve_metadata(
            frames
                .into_iter()
                .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                .ok_or(HostMetadataFundingError::Overflow)?,
        )?;
        let array = tensor.as_array();
        let elements = array.shape().iter().try_fold(1usize, |count, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| count.checked_mul(dim))
        });
        Ok(Some((
            crate::tensor::portable_dtype(array.dtype()),
            array.ndim(),
            elements,
        )))
    }

    fn dtype(&self, tensor: &MlxTensor) -> TensorDtype {
        crate::tensor::portable_dtype(tensor.as_array().dtype())
    }

    fn shape(&self, tensor: &MlxTensor) -> Vec<usize> {
        tensor
            .as_array()
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or(usize::MAX))
            .collect()
    }
}

impl SumReductionBackend for MlxNeuralBackend {
    fn complete_model_sum_wave<E, V>(
        values: &[MlxTensor],
        group: &Group,
        context: &Group,
        executor: &Stream,
        validate: V,
    ) -> Result<Option<Vec<MlxTensor>>, eredu_core::BackendFailure>
    where
        E: std::error::Error + Send + Sync + 'static,
        V: FnMut(&[MlxTensor], &eredu_nn::workspace::HostMetadataFunding, bool) -> Result<(), E>,
    {
        prepared_collectives::sum_wave(values, group, context, executor, validate)
    }
    fn complete_model_sum(
        value: &MlxTensor,
        group: &Group,
        context: &Group,
        executor: &Stream,
    ) -> Result<Option<MlxTensor>, Self::CommunicationError> {
        prepared_collectives::sum(value, group, context, executor)
    }
    fn all_reduce_sum(
        value: MlxTensor,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("sum", &input, group, "");
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        let output = crate::backend::runtime::distributed::all_sum(&input, group, executor)?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion: completion.into(),
        })
    }
}

impl EvenGatherBackend for MlxNeuralBackend {
    fn all_gather_even(
        value: MlxTensor,
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("gather", &input, group, &format!("axis={axis}"));
        validate_communication_tensor(&input, MlxCommunicationDtypes::FloatingAndI32)?;
        let axis = i32::try_from(axis).map_err(|_| {
            safemlx::error::Exception::custom("all-gather axis does not fit in i32")
        })?;
        let output = crate::backend::distributed::all_gather_axis(&input, axis, group, executor)?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion: completion.into(),
        })
    }
}

impl UnevenGatherBackend for MlxNeuralBackend {
    fn complete_model_gather(
        value: &MlxTensor,
        counts: &[usize],
        axis: usize,
        group: &Group,
        context: &Group,
        executor: &Stream,
    ) -> Result<Option<MlxTensor>, Self::CommunicationError> {
        prepared_collectives::gather(value, counts, axis, group, context, executor)
    }
    fn all_gather_uneven(
        value: MlxTensor,
        counts: &[usize],
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective(
            "gather_uneven",
            &input,
            group,
            &format!("axis={axis} counts={counts:?}"),
        );
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        let axis = i32::try_from(axis).map_err(|_| {
            safemlx::error::Exception::custom("uneven all-gather axis does not fit in i32")
        })?;
        let output = crate::backend::distributed::all_gather_uneven_axis(
            &input, axis, counts, group, executor,
        )?;
        let completion =
            collective_completion(input, &output, group, executor, vec![counts.to_vec()])?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion: completion.into(),
        })
    }
}

impl VariableAllToAllBackend for MlxNeuralBackend {
    fn complete_prepared_variable_all_to_all(
        value: &MlxTensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        matrix: &eredu_runtime::CommunicationPeerMatrix<'_>,
        group: &Group,
        context: &Group,
        executor: &Stream,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Option<MlxTensor>, Self::CommunicationError> {
        Self::with_parallel_control_context(context, |source| {
            let Some((_, actual)) = source else {
                return Ok(None);
            };
            if !actual.same_account(funding) {
                return Err(crate::backend::error::Error::Neural(
                    context.model_source_missing(),
                ));
            }
            let projection = context.original_control_request().ok_or_else(|| {
                crate::backend::error::Error::Neural(context.model_source_missing())
            })?;
            #[cfg(test)]
            crate::tests::support::path_instrumentation::variable_all_to_all_submission();
            // The selected transfer owns input readiness. Numerical region
            // children have already settled and published their outputs; only
            // the source-declared metadata/local/inactive inputs need another
            // parent completion before entering the communication child.
            projection
                .complete_variable_all_to_all(
                    value.as_array(),
                    counts,
                    axis,
                    matrix,
                    group,
                    context,
                    executor,
                    funding,
                )
                .map(|value| Some(MlxTensor::from_array(value)))
        })?
    }
    fn variable_all_to_all(
        value: MlxTensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::variable_all_to_all_submission();
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective(
            "exchange",
            &input,
            group,
            &format!(
                "axis={axis} send={:?} receive={:?}",
                counts.send(),
                counts.receive()
            ),
        );
        validate_communication_tensor(&input, MlxCommunicationDtypes::FloatingAndI32)?;
        if counts.group_size() != group.size() {
            return Err(safemlx::error::Exception::custom(format!(
                "variable all-to-all has {} peer counts for group size {}",
                counts.group_size(),
                group.size()
            ))
            .into());
        }
        if counts
            .send()
            .iter()
            .chain(counts.receive())
            .any(|count| *count > i32::MAX as usize)
        {
            return Err(safemlx::error::Exception::custom(
                "variable all-to-all peer count exceeds advertised i32 limit",
            )
            .into());
        }
        let all_zero = counts
            .send()
            .iter()
            .chain(counts.receive())
            .all(|count| *count == 0);
        if all_zero && group.size() > 1 {
            // A zero-element MLX result may be considered already evaluated,
            // which would omit the native collective and let this subgroup
            // advance ahead of non-empty subgroups in the same world wave.
            // Submit one private self-directed sentinel per member while
            // preserving the selected logical zero-count result exactly.
            let mut sentinel_shape = input.shape().to_vec();
            sentinel_shape[axis] = 1;
            let sentinel = zeros_dtype(&sentinel_shape, input.dtype(), executor)?;
            let mut sentinel_counts = vec![0usize; group.size()];
            sentinel_counts[group.rank()] = 1;
            let sentinel_output = crate::backend::distributed::all_to_all_v_axis(
                &sentinel,
                axis,
                &sentinel_counts,
                &sentinel_counts,
                group,
                executor,
            )?;
            let completion = collective_completion(
                sentinel,
                &sentinel_output,
                group,
                executor,
                vec![sentinel_counts],
            )?;
            return Ok(Submission {
                output: MlxTensor::from_array(input),
                completion: completion.into(),
            });
        }
        let output = crate::backend::distributed::all_to_all_v_axis(
            &input,
            axis,
            counts.send(),
            counts.receive(),
            group,
            executor,
        )?;
        let completion = collective_completion(
            input,
            &output,
            group,
            executor,
            vec![counts.send().to_vec(), counts.receive().to_vec()],
        )?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion: completion.into(),
        })
    }
}

impl BroadcastBackend for MlxNeuralBackend {
    fn broadcast(
        value: MlxTensor,
        root: usize,
        group: &Group,
        executor: &Stream,
    ) -> Result<Submission<MlxTensor, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        if let Some(funding) = group.model_funding() {
            let frames = [
                std::mem::size_of::<MlxTensor>(),
                std::mem::size_of::<Submission<MlxTensor, Self::CommunicationCompletion>>(),
                std::mem::size_of::<
                    Result<
                        Submission<MlxTensor, Self::CommunicationCompletion>,
                        Self::CommunicationError,
                    >,
                >(),
            ];
            funding
                .reserve_metadata(
                    frames
                        .into_iter()
                        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                        .ok_or(crate::backend::error::Error::WorkspacePlanning(
                            eredu_nn::workspace::HostMetadataFundingError::Overflow,
                        ))?,
                )
                .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        }
        if let Some(result) = group.publish_original(value.as_array(), root, executor) {
            let (output, completion) = result?;
            return Ok(Submission {
                output: MlxTensor::from_array(output),
                completion: completion.into(),
            });
        }
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        if root >= group.size() {
            return Err(safemlx::error::Exception::custom(format!(
                "broadcast root {root} is outside group size {}",
                group.size()
            ))
            .into());
        }
        let input = value.into_array();
        #[cfg(test)]
        trace_partition_collective("broadcast", &input, group, &format!("root={root}"));
        validate_communication_tensor(&input, MlxCommunicationDtypes::Floating)?;
        // Preserve every rank's lazy predecessor graph in the submitted event.
        // A zero-valued expression keeps non-roots in the same dependency order
        // without performing an unbounded host synchronization before returning
        // the exact communication completion.
        let contribution = if group.rank() == root {
            input.clone()
        } else {
            input.multiply(Array::try_from_f32(0.0)?, executor)?
        };
        let output = crate::backend::runtime::distributed::all_sum_for(
            eredu_runtime::CommunicationOperation::Broadcast,
            &contribution,
            group,
            executor,
        )?;
        let completion = MlxCommunicationCompletion::submit(
            [&output],
            ordinary_broadcast::retained_arrays(input, contribution, output.clone()),
            Vec::new(),
            vec![group.clone()],
            Vec::new(),
            vec![executor.clone()],
        )?;
        Ok(Submission {
            output: MlxTensor::from_array(output),
            completion: completion.into(),
        })
    }
}

impl BarrierBackend for MlxNeuralBackend {
    fn barrier(
        group: &Group,
        executor: &Stream,
    ) -> Result<Self::CommunicationCompletion, Self::CommunicationError> {
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let token = zeros_dtype(&[], Dtype::Float32, executor)?;
        let completed = crate::backend::runtime::distributed::payload_free_all_sum_for(
            eredu_runtime::CommunicationOperation::Barrier,
            &token,
            group,
            executor,
        )?;
        MlxCommunicationCompletion::submit(
            [&completed],
            vec![token, completed.clone()],
            Vec::new(),
            vec![group.clone()],
            Vec::new(),
            vec![executor.clone()],
        )
        .map(Into::into)
        .map_err(Into::into)
    }
}

impl FailureAgreementBackend for MlxNeuralBackend {
    type FailureAgreementOutput = MlxFailureAgreement;

    fn agree_success(
        local_success: bool,
        group: &Group,
        executor: &Stream,
    ) -> Result<
        Submission<MlxFailureAgreement, Self::CommunicationCompletion>,
        Self::CommunicationError,
    > {
        if let Some(binding) = group.original_control() {
            let (output, completion) = binding.agree(group, local_success, executor)?;
            return Ok(Submission {
                output: output.into_agreement(),
                completion,
            });
        }
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let member_count = i32::try_from(group.size()).map_err(|_| {
            safemlx::error::Exception::custom(
                "failure-agreement group size exceeds the advertised i32 status count",
            )
        })?;
        let input = Array::try_from_slice(&[i32::from(local_success)], &[1])?;
        #[cfg(test)]
        trace_partition_collective(
            "agreement",
            &input,
            group,
            &format!("success={local_success}"),
        );
        let output = crate::backend::runtime::distributed::payload_free_all_sum_for(
            eredu_runtime::CommunicationOperation::FailureAgreement,
            &input,
            group,
            executor,
        )?;
        let completion = collective_completion(input, &output, group, executor, Vec::new())?;
        let (agreement, completion) = completion.with_failure_agreement(output, member_count);
        Ok(Submission {
            output: agreement,
            completion: completion.into(),
        })
    }

    fn agree_success_from_source(
        local_success: bool,
        group: &Group,
        phase: eredu_runtime::DistributedExecutionPhase,
        executor: &Stream,
        source: &Group,
    ) -> Result<Option<bool>, Self::CommunicationError> {
        match source.original_control_request() {
            Some(request) => {
                request.complete_provider_vote(local_success, group, phase, executor, source)
            }
            None => Ok(None),
        }
    }

    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError> {
        output.resolve_neural()
    }
}

impl PointToPointBackend for MlxNeuralBackend {
    fn send_receive_prepared(
        values: eredu_runtime::PreparedBoundaryFrames<MlxTensor>,
        route: &CommunicationRouteRealization,
        context: &Group,
        executor: &Stream,
    ) -> Result<
        Option<Submission<Vec<MlxTensor>, Self::CommunicationCompletion>>,
        Self::CommunicationError,
    > {
        let request = context
            .original_control_request()
            .ok_or_else(|| crate::backend::error::Error::Neural(context.model_source_missing()))?;
        request
            .transfer_boundary_frames(context, values, route, executor)
            .map(Some)
    }

    fn send_receive(
        values: Vec<RoleExactBoundaryValue<MlxTensor>>,
        route: &CommunicationRouteRealization,
        executor: &Stream,
    ) -> Result<Submission<Vec<MlxTensor>, Self::CommunicationCompletion>, Self::CommunicationError>
    {
        let retained_capacity = ordinary_boundary::validate_count(route, values.len())?;
        let group = route.group().ok_or_else(|| {
            safemlx::error::Exception::custom(format!(
                "world rank is not an endpoint of communication route {}",
                route.descriptor().id().value()
            ))
        })?;
        let _setup = group.begin_bounded_setup()?;
        if let Some(setup) = &_setup {
            setup.check()?;
        }
        let descriptor = route.descriptor();
        let receiving = matches!(
            route.endpoint(),
            Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination)
        );
        let peer_rank = route.peer_rank().ok_or_else(|| {
            safemlx::error::Exception::custom(format!(
                "communication route {} endpoint has no local peer rank",
                descriptor.id().value()
            ))
        })?;
        let logical = values
            .iter()
            .map(|value| value.tensor().clone())
            .collect::<Vec<_>>();
        validate_route_bundle(&logical, route)?;
        let mut inputs = Vec::with_capacity(retained_capacity);
        let mut frames = Vec::with_capacity(values.len());
        let mut expected_headers = Vec::with_capacity(values.len());
        for value in values {
            let (header, tensor) = value.into_parts();
            let input = tensor.into_array();
            let header_len = i32::try_from(header.len()).map_err(|_| {
                safemlx::error::Exception::custom(
                    "boundary frame header length exceeds MLX dimensions",
                )
            })?;
            let header_array = Array::try_from_slice(&header, &[header_len])?;
            let frame = super::super::boundary_frame::encode(
                &super::super::boundary_frame::Native(executor),
                &input,
                &header_array,
            )?;
            inputs.push(input);
            frames.push(frame);
            expected_headers.push(header);
        }
        let (submitted, outputs, received_headers) = if !receiving {
            let mut submitted = Vec::with_capacity(frames.len());
            for frame in &frames {
                submitted.push(send(frame, peer_rank, group, executor)?);
            }
            (submitted, inputs.clone(), Vec::new())
        } else {
            let mut received = Vec::with_capacity(frames.len());
            for placeholder in &frames {
                received.push(recv_like(placeholder, peer_rank, group, executor)?);
            }
            let mut outputs = Vec::with_capacity(received.len());
            let mut received_headers = Vec::with_capacity(received.len());
            for ((input, expected), received) in inputs.iter().zip(&expected_headers).zip(&received)
            {
                let header_len = i32::try_from(expected.len()).map_err(|_| {
                    safemlx::error::Exception::custom(
                        "boundary frame header length exceeds MLX indexing",
                    )
                })?;
                let (received_header, payload) = crate::backend::nn::boundary_frame::split(
                    &crate::backend::nn::boundary_frame::Native(executor),
                    received,
                    header_len,
                    input,
                )?;
                outputs.push(payload);
                received_headers.push((received_header, expected.clone()));
            }
            (received.clone(), outputs, received_headers)
        };
        let mut retained = inputs;
        retained.extend(frames);
        retained.extend(outputs.iter().cloned());
        retained.extend(submitted.iter().cloned());
        let mut completion_outputs = Vec::with_capacity(retained_capacity);
        completion_outputs.extend(submitted.iter());
        completion_outputs.extend(outputs.iter());
        completion_outputs.extend(received_headers.iter().map(|(header, _)| header));
        let completion = MlxCommunicationCompletion::submit(
            completion_outputs,
            retained,
            Vec::new(),
            vec![group.clone()],
            vec![route.clone()],
            vec![executor.clone()],
        )?
        .with_boundary_headers(received_headers);
        Ok(Submission {
            output: outputs.into_iter().map(MlxTensor::from_array).collect(),
            completion: completion.into(),
        })
    }
}

/// Byte framing permits arbitrary header lengths. Copy the byte slice before
/// reinterpreting an unaligned payload: native typed kernels may round a view's
/// byte offset to an element offset even though byte-wise host reads succeed.
pub(super) fn decode_boundary_payload(
    received: &Array,
    header_len: i32,
    prototype: &Array,
    executor: &Stream,
) -> Result<Array, safemlx::error::Exception> {
    super::super::boundary_frame::decode(
        &super::super::boundary_frame::Native(executor),
        received,
        header_len,
        prototype,
    )
}

impl TransferBackend for MlxNeuralBackend {
    type HostBuffer = Arc<ImmutableHostTransferBuffer>;
    type Transfer = MlxSubmissionCompletion;
    type TransferError = safemlx::error::Exception;

    fn promote(
        executor: &Self::Executor,
        host: &Self::HostBuffer,
    ) -> Result<(Self::MaterializedWeight, Self::Transfer), Self::TransferError> {
        Self::reclaim_retired_resources();
        let mut retained = Recovery::begin(Rc::new(SubmissionResources {
            ordinary: None,
            event: RefCell::new(None),
            _arrays: Vec::new(),
            additional: RefCell::new(HostResources::new(vec![Box::new(Arc::clone(host))])),
            children: Cell::new(0),
            failed: Cell::new(false),
        }))?;
        let submitted = host.copy_to_array_operation(executor);
        retained.seal();
        let (weight, event) = submitted?;
        let completion = MlxSubmissionCompletion::new(event, retained);
        completion.retain(weight.clone());
        completion.check_native_status()?;
        Ok((MlxTensor::from_array(weight), completion))
    }

    fn demote(
        executor: &Self::Executor,
        weight: &Self::MaterializedWeight,
    ) -> Result<(Self::HostBuffer, Self::Transfer), Self::TransferError> {
        Self::reclaim_retired_resources();
        let mut retained = Recovery::begin(Rc::new(SubmissionResources {
            ordinary: None,
            event: RefCell::new(None),
            _arrays: vec![weight.as_array().clone()],
            additional: RefCell::new(HostResources::new(Vec::new())),
            children: Cell::new(0),
            failed: Cell::new(false),
        }))?;
        let submitted = HostTransferBuffer::copy_from_array_operation(
            weight.as_array(),
            HostTransferPolicy::Transfer,
            executor,
        );
        retained.seal();
        let (host, event) = submitted?;
        let host = Arc::new(host.freeze());
        let completion = MlxSubmissionCompletion::new(event, retained);
        completion.retain(Arc::clone(&host));
        completion.check_native_status()?;
        Ok((host, completion))
    }
}

impl eredu_runtime::TerminalCommunicationBackend for MlxNeuralBackend {
    fn mark_terminal_submission(group: &Group) {
        group.mark_terminal_submission();
    }
}
