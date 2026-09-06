use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use std::{cell::Cell, rc::Rc};

struct PendingResources {
    output: Option<Array>,
    source: Option<Array>,
    group: Option<Arc<CachedGgufGroup>>,
    lease: WeightLease,
    _source_stream: Stream,
    _execution_stream: Stream,
}

impl Retention for PendingResources {
    fn observe(&self, _: Status) {}
}

/// Prepared tensor materialization retaining its exact checkpoint source.
pub struct PendingWeightMaterialization {
    retained: Recovery<PendingResources>,
}

impl PendingWeightMaterialization {
    pub(super) fn begin(
        lease: WeightLease,
        source_stream: &Stream,
        execution_stream: &Stream,
    ) -> Result<Self, CheckpointMaterializationError> {
        let key = lease.key().to_owned();
        let retained = Recovery::begin(PendingResources {
            output: None,
            source: None,
            group: None,
            lease,
            _source_stream: source_stream.clone(),
            _execution_stream: execution_stream.clone(),
        })
        .map_err(|source| materialization_error(&key, "prepare recovery", source))?;
        Ok(Self { retained })
    }

    pub(super) fn set_source(&mut self, source: Array) {
        self.retained.retention_mut().source = Some(source);
    }

    pub(super) fn source(&self) -> &Array {
        self.retained
            .retention()
            .source
            .as_ref()
            .expect("prepared source")
    }

    pub(super) fn set_group(&mut self, group: Arc<CachedGgufGroup>) {
        self.retained.retention_mut().group = Some(group);
    }

    pub(super) fn prepared(
        mut self,
        output: Array,
    ) -> Result<Self, CheckpointMaterializationError> {
        self.retained.retention_mut().output = Some(output);
        self.retained.seal();
        let status = self.retained.progress();
        if status.failed || status.blocked {
            return Err(materialization_error(
                self.key(),
                "prepare",
                safemlx::error::Exception::custom(
                    "native preparation failed; unresolved resources remain retained",
                ),
            ));
        }
        Ok(self)
    }

    fn key(&self) -> &str {
        self.retained.retention().lease.key()
    }

    /// Returns the lazy materialized output.
    pub fn output(&self) -> &Array {
        self.retained
            .retention()
            .output
            .as_ref()
            .expect("prepared output")
    }

    #[cfg(test)]
    pub(super) fn finish(self) -> Result<Array, CheckpointMaterializationError> {
        self.submit()?.synchronize()
    }

    pub(super) fn submit(self) -> Result<WeightMaterialization, CheckpointMaterializationError> {
        let output = self.output().clone();
        WeightMaterialization::submit_retained(output, vec![self])
    }

    /// Releases a prepared source after its enclosing submission has completed.
    ///
    /// This never waits: an unresolved preparation retains its owner independently.
    pub fn complete(self) {}
}

struct MaterializationResources {
    inputs: Vec<Array>,
    outputs: Vec<Array>,
    _sources: Vec<PendingWeightMaterialization>,
    event: Option<Event>,
    children: Cell<usize>,
    failed: Cell<bool>,
}

impl Retention for MaterializationResources {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.failed.set(true);
        }
    }
}

struct MaterializationObservation {
    owner: Rc<MaterializationResources>,
    _stream: Option<Stream>,
}

impl MaterializationObservation {
    fn new(
        owner: &Rc<MaterializationResources>,
        stream: Option<Stream>,
    ) -> Result<Self, safemlx::error::Exception> {
        let next = owner.children.get().checked_add(1).ok_or_else(|| {
            safemlx::error::Exception::custom("materialization observation count exhausted")
        })?;
        owner.children.set(next);
        Ok(Self {
            owner: Rc::clone(owner),
            _stream: stream,
        })
    }
}

impl Retention for MaterializationObservation {
    fn observe(&self, status: Status) {
        self.owner.observe(status);
    }
}

impl Drop for MaterializationObservation {
    fn drop(&mut self) {
        self.owner.children.set(self.owner.children.get() - 1);
    }
}

struct MaterializationUnwind<'a>(&'a Cell<bool>);
impl Drop for MaterializationUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.set(true);
        }
    }
}

/// Exact native completion retaining source leases, outputs and consumer tickets.
///
/// Polling and Drop never wait. Failed or unobservable native work retains its
/// ownership independently, including when submission fails before publication.
#[must_use = "checkpoint resources remain retained until exact native completion"]
pub struct WeightMaterialization {
    key: String,
    retained: Recovery<Rc<MaterializationResources>>,
}

fn materialization_error(
    key: &str,
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CheckpointMaterializationError {
    CheckpointMaterializationError::Mlx {
        key: key.to_owned(),
        operation,
        source,
    }
}

impl WeightMaterialization {
    /// Arms ownership before a potentially eager conversion or native submission.
    pub(crate) fn prepare_retained(
        inputs: Vec<Array>,
        sources: Vec<PendingWeightMaterialization>,
    ) -> Result<Self, CheckpointMaterializationError> {
        let key = sources
            .first()
            .map(|source| source.key().to_owned())
            .unwrap_or_else(|| "<derived checkpoint materialization>".into());
        let retained = Recovery::begin(Rc::new(MaterializationResources {
            inputs,
            outputs: Vec::new(),
            _sources: sources,
            event: None,
            children: Cell::new(0),
            failed: Cell::new(false),
        }))
        .map_err(|source| materialization_error(&key, "prepare recovery", source))?;
        Ok(Self { key, retained })
    }

    pub(crate) fn inputs(&self) -> &[Array] {
        &self.retained.retention().inputs
    }

    pub(crate) fn outputs(&self) -> &[Array] {
        &self.retained.retention().outputs
    }

    /// Returns preparation dependencies only after its eager native work is
    /// provably healthy and terminal. This is an explicit host-side boundary,
    /// never used by polling or Drop.
    pub(crate) fn finish_preparation(
        mut self,
    ) -> Result<Vec<PendingWeightMaterialization>, CheckpointMaterializationError> {
        self.retained.seal();
        while !self.check_native_status()? {
            std::thread::yield_now();
        }
        Ok(std::mem::take(
            &mut Rc::get_mut(self.retained.retention_mut())
                .expect("unpublished preparation")
                ._sources,
        ))
    }

    pub(crate) fn submit_outputs(
        mut self,
        outputs: Vec<Array>,
    ) -> Result<Self, CheckpointMaterializationError> {
        Rc::get_mut(self.retained.retention_mut())
            .expect("unpublished owner")
            .outputs = outputs;
        let result = async_eval_with_event(self.outputs().iter());
        self.retained.seal();
        if result.is_err() {
            self.retained.retention().failed.set(true);
        }
        let event = result.map_err(|source| self.mlx_error("evaluation submission", source))?;
        Rc::get_mut(self.retained.retention_mut())
            .expect("unpublished owner")
            .event = Some(event);
        self.check_native_status()?;
        Ok(self)
    }

    /// Submits an output and retains its source materializations until completion.
    pub fn submit_retained(
        output: Array,
        sources: Vec<PendingWeightMaterialization>,
    ) -> Result<Self, CheckpointMaterializationError> {
        Self::prepare_retained(Vec::new(), sources)?.submit_outputs(vec![output])
    }

    /// Returns the materialized output while this owner retains its sources.
    pub fn output(&self) -> &Array {
        self.outputs()
            .first()
            .expect("materialization has an output")
    }

    fn event(&self) -> &Event {
        self.retained
            .retention()
            .event
            .as_ref()
            .expect("submitted materialization")
    }

    fn check_native_status(&self) -> Result<bool, CheckpointMaterializationError> {
        crate::backend::submission_recovery::reap();
        let status = self.retained.progress();
        let resources = self.retained.retention();
        if status.failed || status.blocked || resources.failed.get() {
            Err(self.mlx_error(
                "completion",
                safemlx::error::Exception::custom(
                    "native materialization failed; unresolved resources remain retained",
                ),
            ))
        } else {
            Ok(status.settled && resources.children.get() == 0)
        }
    }

    /// Orders a compatible consumer while retaining its independent completion ticket.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), CheckpointMaterializationError> {
        self.check_native_status()?;
        let owner = self.retained.retention();
        let observation = MaterializationObservation::new(owner, Some(stream.clone()))
            .map_err(|source| self.mlx_error("prepare consumer recovery", source))?;
        let mut child = Recovery::begin(observation)
            .map_err(|source| self.mlx_error("prepare consumer recovery", source))?;
        let _unwind = MaterializationUnwind(&owner.failed);
        let result = self.event().wait_on(stream);
        if result.is_err() {
            owner.failed.set(true);
        }
        child.seal();
        let status = child.progress();
        if status.failed || status.blocked {
            return Err(self.mlx_error(
                "consumer stream wait",
                safemlx::error::Exception::custom(
                    "native consumer dependency failed; resources remain retained",
                ),
            ));
        }
        result.map_err(|source| self.mlx_error("consumer stream wait", source))
    }

    /// Polls the whole submission and every consumer without waiting for the runtime.
    pub fn is_complete(&self) -> Result<bool, CheckpointMaterializationError> {
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            self.event().is_complete().map_err(|source| {
                self.retained.retention().failed.set(true);
                self.mlx_error("completion query", source)
            })
        })
        .unwrap_or(Ok(false))
    }

    pub(crate) fn wait(&self) -> Result<(), CheckpointMaterializationError> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }

    /// Waits explicitly for exact completion and returns the independently owned output.
    pub fn synchronize(self) -> Result<Array, CheckpointMaterializationError> {
        self.wait()?;
        Ok(self.output().clone())
    }

    fn mlx_error(
        &self,
        operation: &'static str,
        source: safemlx::error::Exception,
    ) -> CheckpointMaterializationError {
        materialization_error(&self.key, operation, source)
    }
}

pub(super) fn materialize_contiguous(
    key: &str,
    source: &Array,
    offset_elements: usize,
    shape: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let elements = shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            CheckpointMaterializationError::ArithmeticOverflow {
                context: format!("contiguous materialization size for tensor {key:?}"),
            }
        })
    })?;
    let end = offset_elements.checked_add(elements).ok_or_else(|| {
        CheckpointMaterializationError::ArithmeticOverflow {
            context: format!("contiguous materialization end for tensor {key:?}"),
        }
    })?;
    let flattened = source.reshape(&[-1], source_stream).map_err(|source| {
        CheckpointMaterializationError::Mlx {
            key: key.to_string(),
            operation: "flatten contiguous selection",
            source,
        }
    })?;
    let selected = materialize_range(
        key,
        flattened,
        &[source.size()],
        0,
        offset_elements,
        end,
        source_stream,
        execution_stream,
    )?;
    let shape = shape
        .iter()
        .map(|dimension| to_i32(key, "contiguous output dimension", *dimension))
        .collect::<Result<Vec<_>, _>>()?;
    selected
        .reshape(&shape, execution_stream)
        .map_err(|source| CheckpointMaterializationError::Mlx {
            key: key.to_string(),
            operation: "reshape contiguous selection",
            source,
        })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_range(
    key: &str,
    source: Array,
    source_shape: &[usize],
    axis: usize,
    start: usize,
    end: usize,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let axis_i32 = to_i32(key, "axis", axis)?;
    let front = if axis == 0 {
        source
    } else {
        source
            .move_axis(axis_i32, 0, source_stream)
            .map_err(|source| mlx_error(key, "move range axis", source))?
    };
    let start = to_i32(key, "range start", start)?;
    let end = to_i32(key, "range end", end)?;
    let selected = front
        .try_index_device(start..end, source_stream)
        .map_err(|source| mlx_error(key, "range selection", source))?;
    let selected = if axis == 0 {
        selected
    } else {
        selected
            .move_axis(0, axis_i32, source_stream)
            .map_err(|source| mlx_error(key, "restore range axis", source))?
    };
    let selected = if axis == 0 {
        selected
    } else {
        // Inner-axis ranges are non-contiguous views. Compact only the selected
        // result, keeping the temporary bounded by the output shape.
        let mut output_shape = source_shape.to_vec();
        output_shape[axis] = usize::try_from(end - start).map_err(|_| {
            CheckpointMaterializationError::ArithmeticOverflow {
                context: format!("selected range length for tensor {key:?}"),
            }
        })?;
        let row_major_shape = output_shape
            .iter()
            .map(|dimension| to_i32(key, "selected dimension", *dimension))
            .collect::<Result<Vec<_>, _>>()?;
        selected
            .flatten(None, None, source_stream)
            .and_then(|value| value.reshape(&row_major_shape, source_stream))
            .map_err(|source| mlx_error(key, "range compaction", source))?
    };
    selected
        .copy(execution_stream)
        .map_err(|source| mlx_error(key, "copy", source))
}

pub(super) fn materialize_indices(
    key: &str,
    source: &Array,
    axis: usize,
    indices: &[usize],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Array, CheckpointMaterializationError> {
    let axis = to_i32(key, "axis", axis)?;
    let indices = indices
        .iter()
        .map(|index| to_i32(key, "tensor index", *index))
        .collect::<Result<Vec<_>, _>>()?;
    let count = to_i32(key, "index count", indices.len())?;
    let index_array = Array::from_slice(&indices, &[count])
        .copy(source_stream)
        .map_err(|source| mlx_error(key, "index upload", source))?;
    source
        .take_axis(&index_array, axis, source_stream)
        .and_then(|selected| selected.copy(execution_stream))
        .map_err(|source| mlx_error(key, "ordered index selection", source))
}

fn to_i32(
    key: &str,
    what: &'static str,
    value: usize,
) -> Result<i32, CheckpointMaterializationError> {
    i32::try_from(value).map_err(|_| CheckpointMaterializationError::ArithmeticOverflow {
        context: format!("{what} for tensor {key:?} does not fit in i32"),
    })
}

fn mlx_error(
    key: &str,
    operation: &'static str,
    source: safemlx::error::Exception,
) -> CheckpointMaterializationError {
    CheckpointMaterializationError::Mlx {
        key: key.to_string(),
        operation,
        source,
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::backend::submission_recovery::Probe;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    struct Controlled {
        settled: Arc<AtomicBool>,
        failed: bool,
    }
    impl Probe for Controlled {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: self.settled.load(Ordering::Acquire),
                failed: self.failed,
                blocked: false,
            }
        }
    }

    fn fixture() -> (tempfile::TempDir, SafetensorsWeightStore, Stream) {
        let dir = tempfile::tempdir().unwrap();
        for (key, value) in [("one", 1i32), ("two", 2i32)] {
            let bytes = value.to_le_bytes();
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::I32, vec![1], &bytes)
                    .unwrap();
            safetensors::tensor::serialize_to_file(
                [(key, view)],
                None,
                &dir.path().join(format!("{key}.safetensors")),
            )
            .unwrap();
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"one":"one.safetensors","two":"two.safetensors"}}"#,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        (dir, store, stream)
    }

    fn assert_capacity_pinned(store: &SafetensorsWeightStore) {
        assert!(matches!(
            acquire(store, "two"),
            Err(CheckpointMaterializationError::Store(
                StoreError::CapacityExhausted { .. }
            ))
        ));
    }

    fn acquire(
        store: &SafetensorsWeightStore,
        key: &str,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        let lease = store.acquire_lease(TensorReadRequest {
            key: key.to_owned(),
            selection: TensorSelection::Full,
            policy: WeightReadPolicy::RequireBounded,
        })?;
        WeightLease::from_checkpoint_lease(lease, Arc::new(Mutex::new(BTreeMap::new())))
    }

    fn reap_until_available(store: &SafetensorsWeightStore) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            crate::backend::submission_recovery::reap();
            if acquire(store, "two").is_ok() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "terminal source lease was not retired"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn failed_preparation_before_output_publication_retains_exact_checkpoint_lease() {
        let (_dir, store, stream) = fixture();
        let settled = Arc::new(AtomicBool::new(false));
        let prepared = Recovery::with_probe(
            PendingResources {
                lease: acquire(&store, "one").unwrap(),
                output: None,
                source: None,
                group: None,
                _source_stream: stream.clone(),
                _execution_stream: stream,
            },
            Controlled {
                settled: Arc::clone(&settled),
                failed: true,
            },
        );
        let started = Instant::now();
        drop(prepared); // A failed producer has not established a terminal frontier.
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_capacity_pinned(&store);
        settled.store(true, Ordering::Release);
        reap_until_available(&store);
    }

    #[test]
    fn independent_consumer_ticket_keeps_source_after_parent_drop() {
        let (_dir, store, stream) = fixture();
        let source = acquire(&store, "one")
            .unwrap()
            .prepare_materialization(&stream, &stream)
            .unwrap();
        let resources = Rc::new(MaterializationResources {
            inputs: Vec::new(),
            outputs: vec![source.output().clone()],
            _sources: vec![source],
            event: None,
            children: Cell::new(0),
            failed: Cell::new(false),
        });
        let parent = Recovery::with_probe(
            Rc::clone(&resources),
            Controlled {
                settled: Arc::new(AtomicBool::new(true)),
                failed: false,
            },
        );
        let settled = Arc::new(AtomicBool::new(false));
        let child = Recovery::with_probe(
            MaterializationObservation::new(&resources, Some(stream)).unwrap(),
            Controlled {
                settled: Arc::clone(&settled),
                failed: false,
            },
        );
        drop(resources);
        drop(parent);
        drop(child);
        assert_capacity_pinned(&store);
        settled.store(true, Ordering::Release);
        reap_until_available(&store);
    }

    #[test]
    fn materialization_poll_and_drop_do_not_wait_for_a_contended_runtime() {
        let (_dir, store, stream) = fixture();
        let materialized = acquire(&store, "one")
            .unwrap()
            .materialize(&stream, &stream)
            .unwrap();
        materialized.wait().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || loop {
            if safemlx::try_with_submission_retirement(|| {
                ready_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let started = Instant::now();
        assert!(!materialized.is_complete().unwrap());
        drop(materialized);
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_capacity_pinned(&store);
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        reap_until_available(&store);
    }
}
