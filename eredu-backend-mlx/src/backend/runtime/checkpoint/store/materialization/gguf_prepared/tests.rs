use super::*;
use crate::backend::runtime::checkpoint::store::test_support::open_gguf_checkpoint_source_for_test;
use eredu_checkpoint::store::{CheckpointSource, ReadPolicy, TensorReadRequest, WeightStore};
use safemlx::{Device, DeviceType, PrefillRootsRuntime, ScopedSubmissionProgress};

thread_local! { static UNWIND_NEXT: Cell<bool> = const { Cell::new(false) }; }
pub(super) fn inject_pre_native_unwind() {
    if UNWIND_NEXT.with(|next| next.replace(false)) {
        std::panic::panic_any("GGUF unpublished lease unwind");
    }
}

#[derive(Debug, PartialEq)]
enum Values {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    U32(Vec<u32>),
    U8(Vec<u8>),
}
fn values(array: &Array, observer: Option<&OriginalScopeObserver>) -> Values {
    let evaluated = match observer {
        Some(observer) => array.completed_in_original_scope(observer).unwrap(),
        None => array.evaluated().unwrap(),
    };
    match array.dtype() {
        safemlx::Dtype::Float32 => Values::F32(evaluated.try_as_slice::<f32>().unwrap().to_vec()),
        safemlx::Dtype::Float16 => {
            Values::F16(evaluated.try_as_slice::<half::f16>().unwrap().to_vec())
        }
        safemlx::Dtype::Uint8 => Values::U8(evaluated.try_as_slice::<u8>().unwrap().to_vec()),
        safemlx::Dtype::Uint32 => Values::U32(evaluated.try_as_slice::<u32>().unwrap().to_vec()),
        dtype => panic!("unexpected fixture dtype: {dtype:?}"),
    }
}

/// Constructed before Source entry; contains no manufactured admission authority.
pub(crate) struct OriginalGgufMissFixture {
    context: MlxParameterMaterializationContext,
    source: NeutralGgufWeightStore,
    expected: BTreeMap<String, Values>,
    _native_runtime: PrefillRootsRuntime,
    host_runtime: Rc<safemlx::PreparedInputRuntime>,
    stream: Stream,
    directory: tempfile::TempDir,
}
impl OriginalGgufMissFixture {
    pub(crate) fn source_planning_inputs(
        &self,
    ) -> (
        eredu_checkpoint::store::SharedCheckpointSource,
        Rc<safemlx::PreparedInputRuntime>,
        Stream,
    ) {
        (
            Arc::new(self.source.clone()),
            Rc::clone(&self.host_runtime),
            self.stream.clone(),
        )
    }

    pub(crate) fn source_physical_reads(&self) -> u64 {
        self.source.diagnostics().unwrap().physical_reads
    }

    pub(crate) fn new() -> Self {
        Self::new_kind(eredu_gguf::GgmlType::F32)
    }
    pub(crate) fn new_affine() -> Self {
        Self::new_kind(eredu_gguf::GgmlType::Q4_0)
    }
    pub(crate) fn new_mxfp4() -> Self {
        Self::new_kind(eredu_gguf::GgmlType::MxFp4)
    }
    pub(crate) fn new_packed() -> Self {
        Self::new_kind(eredu_gguf::GgmlType::IQ4NL)
    }
    fn new_kind(kind: eredu_gguf::GgmlType) -> Self {
        use eredu_gguf::GgmlType;
        let affine = kind == GgmlType::Q4_0;
        let dense = kind == GgmlType::F32;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("original.gguf");
        let values = [1.5_f32, -2.5, 3.5, 4.5, -5.5, 6.5, 7.5, -8.5];
        let bytes: Vec<_> = if affine {
            (0..2)
                .flat_map(|row| {
                    let mut block = [0u8; 18];
                    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                    for (i, byte) in block.iter_mut().enumerate().skip(2) {
                        *byte = 0x21 + row + i as u8;
                    }
                    block
                })
                .collect()
        } else if dense {
            values.into_iter().flat_map(f32::to_le_bytes).collect()
        } else {
            let (_, block_bytes) = kind.block_and_bytes().unwrap();
            (0..2 * block_bytes).map(|i| (i * 13 + 37) as u8).collect()
        };
        eredu_gguf::Writer::default()
            .write(
                std::fs::File::create(path.clone()).unwrap(),
                &BTreeMap::new(),
                &[eredu_gguf::TensorInput {
                    name: "bank.weight",
                    dimensions: if dense { &[4, 2] } else { &[32, 2] },
                    ggml_type: kind,
                    data: &bytes,
                }],
            )
            .unwrap();
        let source = open_gguf_checkpoint_source_for_test(
            GgufCheckpoint::open(path).unwrap(),
            str::to_owned,
        )
        .unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let native_runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let context = MlxParameterMaterializationContext::new(&stream, &stream);
        let mut expected = BTreeMap::new();
        for key in source.keys() {
            let mut request = Self::request();
            request.key = key.clone();
            let ordinary = context
                .weight_lease(source.acquire_lease(request).unwrap())
                .unwrap()
                .prepare_materialization(&stream, &stream)
                .unwrap()
                .finish()
                .unwrap();
            expected.insert(key, self::values(&ordinary, None));
        }
        if affine {
            assert_eq!(expected.len(), 3);
            assert!(
                matches!(&expected["bank.weight"], Values::U32(weights) if weights.iter().any(|x| *x != 0))
            );
        } else if dense {
            assert_eq!(expected["bank.weight"], Values::F32(values.to_vec()));
        } else {
            assert_eq!(expected.len(), if kind == GgmlType::MxFp4 { 2 } else { 1 });
            assert!(match &expected["bank.weight"] {
                Values::U8(v) => v.iter().any(|v| *v != 0),
                Values::U32(v) => v.iter().any(|v| *v != 0),
                _ => false,
            });
        }
        Self {
            context,
            source,
            expected,
            _native_runtime: native_runtime,
            host_runtime: Rc::new(safemlx::PreparedInputRuntime::prepare().unwrap()),
            stream,
            directory,
        }
    }

    fn ready(&self, controls: &OriginalTextControlGuard) -> PreparedPendingWeight {
        PreparedPendingWeight::new_with_gguf_host_copy(
            controls.clone(),
            Rc::clone(&self.host_runtime),
        )
    }

    fn request() -> TensorReadRequest {
        TensorReadRequest {
            key: "bank.weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        }
    }
    fn lease(&self) -> WeightLease {
        self.context
            .weight_lease(self.source.acquire_lease(Self::request()).unwrap())
            .unwrap()
    }
    fn address(lease: &WeightLease) -> *const NeutralGgufLease {
        let WeightLeaseSource::Gguf(source) = &lease.source else {
            unreachable!()
        };
        std::ptr::from_ref(source.lease.as_ref())
    }
    fn settle(&self, pending: &PendingWeightMaterialization, observer: &OriginalScopeObserver) {
        let event = safemlx::transforms::async_eval_with_original_operation_event_on_stream(
            [pending.output()],
            observer,
            &self.stream,
        )
        .unwrap();
        event.synchronize().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let (outcome, status) = observer.progress().unwrap();
            assert_eq!(outcome, ScopedSubmissionProgress::Observed);
            assert!(!status.failed() && !status.blocked());
            if status.is_settled() {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(
            values(pending.output(), Some(observer)),
            self.expected[pending.key()]
        );
        drop(event);
    }
    fn finish(pending: PendingWeightMaterialization) {
        let observed = pending.retained.finish().unwrap();
        assert_eq!(observed.outcome, ScopedSubmissionProgress::Observed);
        assert!(observed.status.settled && !observed.status.failed && !observed.status.blocked);
    }

    pub(crate) fn cache_miss_hit_and_retry(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) {
        let before = self.source.diagnostics().unwrap().physical_reads;
        let first = self.lease();
        let pointer = Self::address(&first);
        let first = first
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        assert_eq!(Self::address(first.lease()), pointer);
        let facts = first
            .retained
            .retention()
            .gguf_host
            .as_ref()
            .unwrap()
            .facts();
        assert_eq!(facts.iter().flatten().count(), 1);
        let copied = facts[0].unwrap();
        assert_eq!(copied.copy_bytes(), 8 * std::mem::size_of::<f32>());
        assert!(copied.backing_bytes() >= copied.copy_bytes());
        assert!(copied.metadata_bytes() > 0);
        assert_eq!(
            copied.maximum_input_bytes(),
            copied.maximum_input_capacity() * std::mem::size_of::<f32>()
        );
        self.settle(&first, observer);
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        let first_group = first.retained.retention().group.as_ref().unwrap().clone();
        let second = self
            .lease()
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        assert!(
            second
                .retained
                .retention()
                .gguf_host
                .as_ref()
                .unwrap()
                .facts()
                .iter()
                .all(Option::is_none),
            "cache hit prepares no source copy"
        );
        let hit_host = second.retained.retention().gguf_host.as_ref().unwrap();
        assert!(hit_host.transform_facts().iter().all(Option::is_none));
        assert!(
            !hit_host.has_source_plan(),
            "cache hit cannot prepare typed payload metadata"
        );

        assert!(second
            .retained
            .retention()
            .group
            .as_ref()
            .unwrap()
            .same(&first_group));
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        self.settle(&second, observer);
        let weak = first_group.downgrade();
        drop(first_group);
        Self::finish(second);
        Self::finish(first);
        assert!(weak.upgrade().is_none(), "no new strong cache pool");
        let third = self
            .lease()
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 2
        );
        self.settle(&third, observer);
        Self::finish(third);
    }

    pub(crate) fn affine_companions(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) {
        let before = self.source.diagnostics().unwrap();
        let mut pending = Vec::with_capacity(3);
        for key in ["bank.weight", "bank.scales", "bank.biases"] {
            let mut request = Self::request();
            request.key = key.into();
            let lease = self
                .context
                .weight_lease(self.source.acquire_lease(request).unwrap())
                .unwrap();
            let pointer = Self::address(&lease);
            let ready = lease
                .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
                .unwrap();
            assert_eq!(Self::address(ready.lease()), pointer);
            self.settle(&ready, observer);
            pending.push(ready);
        }
        let after = self.source.diagnostics().unwrap();
        assert_eq!(after.physical_reads, before.physical_reads + 1);
        assert_eq!(after.cache_hits, before.cache_hits + 1);
        assert_eq!(after.coalesced_group_hits, before.coalesced_group_hits + 2);
        let group = pending[0].retained.retention().group.as_ref().unwrap();
        assert!(pending
            .iter()
            .all(|p| p.retained.retention().group.as_ref().unwrap().same(group)));
        let weak = group.downgrade();
        Self::finish(pending.remove(0));
        assert!(weak.upgrade().is_some());
        for value in pending {
            Self::finish(value);
        }
        assert!(weak.upgrade().is_none());
    }

    pub(crate) fn failed_read(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> PreparedGgufMaterializationFailure {
        eprintln!(
            "gguf_prepared_error_owner_bytes={} boxed_stage_bytes={} materialization_error_bytes={}",
            std::mem::size_of::<PreparedGgufMaterializationFailure>(),
            std::mem::size_of::<PreparedGgufBoxedFailure>(),
            std::mem::size_of::<CheckpointMaterializationError>(),
        );
        let lease = self.lease();
        let pointer = Self::address(&lease);
        let before = self.source.diagnostics().unwrap().physical_reads;
        // The actual warm reader now encounters an EOF; no source-policy mock.
        std::fs::OpenOptions::new()
            .write(true)
            .open(self.directory.path().join("original.gguf"))
            .unwrap()
            .set_len(0)
            .unwrap();
        let failure = lease
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .err()
            .expect("truncated payload must fail");
        let CheckpointMaterializationError::PreparedGguf(failure) = failure else {
            panic!("typed prepared source failure: {failure:?}")
        };
        assert_eq!(std::ptr::from_ref(failure.cause().lease()), pointer);
        assert!(failure.cause().store_error().is_some());
        assert!(std::error::Error::source(&failure)
            .unwrap()
            .downcast_ref::<PreparedGgufBoxedFailure>()
            .is_some());
        assert_eq!(self.source.diagnostics().unwrap().physical_reads, before);
        assert!(self
            .context
            .converted_groups
            .lock()
            .unwrap()
            .all_values(|group| group.upgrade().is_none()));
        failure
    }

    pub(crate) fn unwind_unpublished(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) {
        let mut pending = PendingWeightMaterialization::begin_with_original(
            self.lease(),
            &self.stream,
            &self.stream,
            Some((self.ready(controls), observer.clone())),
        )
        .unwrap();
        UNWIND_NEXT.with(|next| next.set(true));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pending.materialize_gguf_prepared()
        }))
        .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&'static str>(),
            Some(&"GGUF unpublished lease unwind")
        );
        let value = pending.retained.retention();
        assert!(value.output.is_none() && value.source.is_none() && value.group.is_none());
        assert!(value.gguf_custody.is_none());
        // Destruction must not borrow the absent lease. The original node still
        // owns its independent custody and exact observer through retirement.
        drop(pending);
        let (outcome, status) = observer.progress().unwrap();
        assert_eq!(outcome, ScopedSubmissionProgress::Observed);
        assert!(status.is_settled() && !status.failed() && !status.blocked());
    }
}

#[path = "tests/host_copy.rs"]
mod host_copy;

impl OriginalGgufMissFixture {
    pub(crate) fn admitted_dense_layouts(&self) -> [u64; 9] {
        let plan = self.source.conversion_plan(&Self::request()).unwrap();
        let conversion = plan.conversion();
        assert_eq!(conversion.outputs().len(), 1);
        let name = conversion.descriptor().name.len() as u64;
        let dimensions =
            (conversion.descriptor().dimensions.len() * std::mem::size_of::<u64>()) as u64;
        let requests = conversion.requested_elements();
        // G1; selected descriptor; G2 shape+bytes; G3 base+one actual name;
        // reached f32 destination. This fixture's exact Full dense population.
        [
            plan.requested_read_bytes(),
            name,
            dimensions,
            (requests[0] * 8) as u64,
            requests[2] as u64,
            name,
            dimensions,
            plan.output_name().len() as u64,
            requests[2] as u64,
        ]
    }
    fn admitted_ready(
        &self,
        controls: &OriginalTextControlGuard,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) -> PreparedPendingWeight {
        PreparedPendingWeight::new_with_gguf_host_destinations(
            controls.clone(),
            Rc::clone(&self.host_runtime),
            bank,
        )
        .unwrap_or_else(|(_, cause)| panic!("admitted pending: {cause}"))
    }
    pub(crate) fn admitted_miss(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) -> Array {
        let before = self.source.diagnostics().unwrap().physical_reads;
        let lease = self.lease();
        let pointer = Self::address(&lease);
        let pending = lease
            .prepare_original_gguf_fixture(
                &self.stream,
                self.admitted_ready(controls, bank),
                observer.clone(),
            )
            .unwrap_or_else(|cause| panic!("admitted GGUF miss: {cause}"));
        assert_eq!(Self::address(pending.lease()), pointer);
        let host = pending.retained.retention().gguf_host.as_ref().unwrap();
        let bank = host.host_destinations.as_ref().unwrap();
        assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
        assert_eq!(host.transform_facts()[0].unwrap().output_capacity, 8);
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        self.settle(&pending, observer);
        let group = pending.retained.retention().group.as_ref().unwrap();
        let CachedGgufArrays::Supplied { names, values } = &group.arrays else {
            panic!("admitted final names must stay in their actual owner")
        };
        assert_eq!(names.get(0), Some("bank.weight"));
        assert_eq!(values.iter().flatten().count(), 1);
        let name_pointer = names.get(0).unwrap().as_ptr();
        let hit = self
            .lease()
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        assert!(group.same(hit.retained.retention().group.as_ref().unwrap()));
        let CachedGgufArrays::Supplied { names, .. } =
            &hit.retained.retention().group.as_ref().unwrap().arrays
        else {
            unreachable!()
        };
        assert_eq!(names.get(0).unwrap().as_ptr(), name_pointer);
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        self.settle(&hit, observer);
        Self::finish(hit);
        let output = pending.output().clone();
        Self::finish(pending);
        output
    }
    pub(crate) fn admitted_refusal(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
        after_read: bool,
    ) -> CheckpointMaterializationError {
        let before = self.source.diagnostics().unwrap().physical_reads;
        let lease = self.lease();
        let pointer = Self::address(&lease);
        let error = match lease.prepare_original_gguf_fixture(
            &self.stream,
            self.admitted_ready(controls, bank),
            observer.clone(),
        ) {
            Ok(_) => panic!("one-byte-short original destination unexpectedly succeeded"),
            Err(error) => error,
        };
        if after_read {
            assert!(
                matches!(&error, CheckpointMaterializationError::PreparedGgufHostCopy(failure)
                if matches!(failure.cause(), super::super::gguf_host::GgufHostCopyCause::HostDestination(
                    eredu_runtime::working_memory::HostDestinationCause::Capacity { required: 32, remaining: 31 })))
            );
        } else {
            let CheckpointMaterializationError::PreparedGgufAdmitted(failure) = &error else {
                panic!("actual G1 failure: {error}");
            };
            assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
            let mut source = std::error::Error::source(&error);
            let mut host_cause = None;
            while let Some(current) = source {
                if let Some(cause) =
                    current.downcast_ref::<eredu_runtime::working_memory::HostDestinationCause>()
                {
                    host_cause = Some(cause);
                    break;
                }
                source = current.source();
            }
            assert!(matches!(
                host_cause,
                Some(
                    eredu_runtime::working_memory::HostDestinationCause::Capacity {
                        required: 32,
                        remaining: 31
                    }
                )
            ));

            assert!(matches!(
                failure.host_destination_cause(),
                Some(
                    eredu_runtime::working_memory::HostDestinationCause::Capacity {
                        required: 32,
                        remaining: 31
                    }
                )
            ));
        }
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + u64::from(after_read)
        );
        error
    }
    pub(crate) fn admitted_warm_hit(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) {
        let first = self
            .lease()
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        self.settle(&first, observer);
        let before = self.source.diagnostics().unwrap().physical_reads;
        let hit = self
            .lease()
            .prepare_original_gguf_fixture(
                &self.stream,
                self.admitted_ready(controls, bank),
                observer.clone(),
            )
            .unwrap();
        assert!(first
            .retained
            .retention()
            .group
            .as_ref()
            .unwrap()
            .same(hit.retained.retention().group.as_ref().unwrap()));
        let host = hit.retained.retention().gguf_host.as_ref().unwrap();
        let bank = host.host_destinations.as_ref().unwrap();
        assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 1));
        assert!(!host.raw_attempted && host.allocation_source.is_none());
        assert!(host.transform_facts().iter().all(Option::is_none));
        assert_eq!(self.source.diagnostics().unwrap().physical_reads, before);
        self.settle(&hit, observer);
        Self::finish(hit);
        Self::finish(first);
    }
}

impl OriginalGgufMissFixture {
    pub(crate) fn admitted_metadata_refusal(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
        required: u64,
    ) -> CheckpointMaterializationError {
        let before = self.source.diagnostics().unwrap().physical_reads;
        let lease = self.lease();
        let pointer = Self::address(&lease);
        let error = match lease.prepare_original_gguf_fixture(
            &self.stream,
            self.admitted_ready(controls, bank),
            observer.clone(),
        ) {
            Ok(_) => panic!("short G2/G3 destination unexpectedly succeeded"),
            Err(e) => e,
        };
        let CheckpointMaterializationError::PreparedGgufAdmitted(failure) = &error else {
            panic!("actual supplied failure: {error}")
        };
        assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
        assert!(
            matches!(failure.host_destination_cause(),Some(eredu_runtime::working_memory::HostDestinationCause::Capacity {required:r,remaining})if *r==required && *remaining==required-1)
        );
        assert_eq!(self.source.diagnostics().unwrap().physical_reads, before);
        assert!(self
            .context
            .converted_groups
            .lock()
            .unwrap()
            .all_values(|g| g.upgrade().is_none()));
        error
    }
}

#[path = "tests/source_arenas.rs"]
mod source_arenas;

#[path = "tests/funded_component.rs"]
mod funded_component;

#[path = "tests/cache_capsule.rs"]
mod cache_capsule;
