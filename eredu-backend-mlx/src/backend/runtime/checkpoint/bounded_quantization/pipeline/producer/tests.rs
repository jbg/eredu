use super::*;
use crate::backend::runtime::checkpoint::store::{
    ColdMaterializationSlot, MaterializationPayloadShape, PreparedEncodedInputPlan,
};
use crate::backend::submission_recovery::native_role::NativeRoleCapacity;
use eredu_checkpoint::{recipe::EncodedRecipeRead, AffineQuantization};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use safemlx::{Device, DeviceType, OperationEvent, PrefillRootsRuntime, PreparedInputRuntime};
use std::{cell::Cell, rc::Rc};

#[derive(Debug)]
struct Invocation(Rc<Cell<usize>>);
impl Drop for Invocation {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

#[derive(Debug, thiserror::Error)]
enum ProducerFailure {
    #[error("tile pipeline: {0}")]
    Backend(#[from] Error),
    #[error("cold tile admission: {0}")]
    Admission(
        #[source]
        eredu_runtime::working_memory::SharedNativeInitializationFailure<
            cold::Output<Invocation, WeightMaterialization, encoded_affine::ConstructionError>,
            eredu_core::BackendFailure,
        >,
    ),
    #[error("cold tile invocation: {0}")]
    Invocation(#[source] cold::FailedSubmission<Invocation, encoded_affine::ConstructionError>),
}
impl From<eredu_checkpoint::recipe::RecipeError> for ProducerFailure {
    fn from(value: eredu_checkpoint::recipe::RecipeError) -> Self {
        Self::Backend(value.into())
    }
}

// Runtime/stream/read metadata, final output and pipeline metadata are explicit
// fixture prerequisites. The pool funds each actual native submission, input
// and fixed materialization slot; this is not whole-producer admission.
struct FundedProducer<'a> {
    pool: WorkingMemoryPool,
    runtime: &'a PreparedInputRuntime,
    streams: &'a [Stream; 2],
    slots: Vec<usize>,
    live: Rc<Cell<usize>>,
    peak_live: usize,
    peak_used: Cell<u64>,
    largest_tile: u64,
    fail_at: Option<usize>,
    truncate_at: Option<(usize, PathBuf)>,
}
impl TileProducer for FundedProducer<'_> {
    type Completion = cold::Submission<Invocation, WeightMaterialization, Infallible>;
    type Error = ProducerFailure;

    fn submit(
        &mut self,
        source: &dyn CheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<Self::Completion, ProducerFailure> {
        self.slots.push(slot);
        if self.fail_at == Some(self.slots.len() - 1) {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound).into());
        }
        let WeightQuantization::Affine(quantization) = quantization else {
            panic!("affine fixture");
        };
        let read = recipe.prepare_encoded_read(source).unwrap().unwrap();
        let shape = read
            .output()
            .shape()
            .iter()
            .map(|&n| i32::try_from(n).unwrap())
            .collect::<Vec<_>>();
        let columns = *shape.last().unwrap() as usize;
        let rows = shape[..shape.len() - 1]
            .iter()
            .map(|&n| n as usize)
            .product();
        let layout = OperationEvent::cpu_affine_quantize_submission_layout(
            Dtype::Float32,
            Dtype::Float16,
            shape.len(),
            rows,
            columns,
            quantization.group_size,
            quantization.bits,
        )
        .expect("qualified CPU affine layout");
        let capacity = NativeRoleCapacity {
            graph: layout.graph_capacity(),
            records: layout.record_capacity(),
            backing: layout.physical_capacity(self.runtime).unwrap(),
        };
        let payload = MaterializationPayloadShape {
            inputs: 1,
            outputs: 3,
            pending_sources: 0,
        };
        let input =
            PreparedEncodedInputPlan::new(&read, self.runtime, &shape, Dtype::Float32).unwrap();
        let input_bytes = input.required_bytes().unwrap();
        let slot_bytes = ColdMaterializationSlot::required_bytes(payload).unwrap();
        self.live.set(self.live.get() + 1);
        self.peak_live = self.peak_live.max(self.live.get());
        let pool = &self.pool;
        let peak_used = &self.peak_used;
        let stream = &self.streams[slot];
        if let Some((index, path)) = &self.truncate_at {
            if *index == self.slots.len() - 1 {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
            }
        }
        let plan = encoded_affine::plan(
            pool,
            self.runtime,
            capacity,
            Invocation(self.live.clone()),
            input,
            quantization,
            target,
            stream,
            layout,
        );
        self.largest_tile = self
            .largest_tile
            .max(plan.required_bytes().unwrap() + input_bytes + slot_bytes);
        let submitted = plan.submit(pool).map_err(|failure| {
            let (uncalled, failure) = failure.into_parts();
            drop(uncalled);
            ProducerFailure::Admission(failure)
        })?;
        peak_used.set(peak_used.get().max(pool.used_bytes().unwrap()));
        submitted.into_result().map_err(ProducerFailure::Invocation)
    }
}

fn drain(producer: &FundedProducer<'_>) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        producer.pool.used_bytes() == Ok(0)
    });
    assert_eq!(producer.live.get(), 0);
}

fn bytes(source: &dyn CheckpointSource, key: &str) -> Vec<u8> {
    let read = DerivedWeightRecipe::source(key, TensorSelection::Full)
        .prepare_encoded_read(source)
        .unwrap()
        .unwrap();
    let mut bytes = vec![0; read.output().byte_len() as usize];
    EncodedRecipeRead::read_many_borrowed_into(std::iter::once(&read), &mut [&mut bytes]).unwrap();
    bytes
}

fn exercise(
    shape: Vec<usize>,
    budget: u64,
    targets: usize,
    expected_tiles: usize,
    expected_slots: usize,
) {
    let device = Device::new(DeviceType::Cpu, 0);
    let streams = [
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    ];
    let _roots = PrefillRootsRuntime::prepare_for_stream(&streams[0], &streams[1]).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let elements = shape.iter().product::<usize>();
    let source = Arc::new(
        MemoryWeightStore::from_safetensors((0..targets).map(|target| {
            let values = (0..elements)
                .flat_map(|i| ((i % 16) as f32 + target as f32).to_le_bytes())
                .collect();
            (
                format!("weight{target}"),
                SafeDtype::F32,
                shape.clone(),
                values,
            )
        }))
        .unwrap(),
    );
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::new(32, 4).unwrap(),
        budget,
        (0..targets).map(|n| {
            BoundedQuantizationTarget::direct(
                format!("weight{n}"),
                format!("scales{n}"),
                Some(format!("biases{n}")),
            )
            .unwrap()
            .with_affine_companion_dtype(RecipeDtype::F16)
            .unwrap()
        }),
    )
    .unwrap();
    let ordinary =
        BoundedQuantizedWeightStore::create(source.clone(), plan.clone(), &streams[0]).unwrap();
    let prepared = ColdQuantization::prepare(source.into(), plan)
        .unwrap()
        .allocate_ordinary(&streams[0])
        .unwrap();
    let mut producer = FundedProducer {
        pool: WorkingMemoryPool::new(1 << 20, 0).unwrap(),
        runtime: &runtime,
        streams: &streams,
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: None,
    };
    let (result, _) = prepared
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap();
    assert_eq!(result.report(), ordinary.report());
    assert_eq!(result.report().source_tiles, expected_tiles);
    assert_eq!(result.report().peak_in_flight_tiles, expected_slots);
    assert_eq!(
        producer.slots,
        (0..expected_tiles)
            .map(|n| n % expected_slots)
            .collect::<Vec<_>>()
    );
    assert_eq!(producer.peak_live, expected_slots);
    assert!(producer.peak_used.get() <= expected_slots as u64 * producer.largest_tile);
    drain(&producer);
    for target in 0..targets {
        let expected_words = (0..elements / 8)
            .flat_map(|n| {
                (if n % 2 == 0 {
                    0x89abcdef_u32
                } else {
                    0x01234567_u32
                })
                .to_le_bytes()
            })
            .collect::<Vec<_>>();
        let expected_scales = (0..elements / 32)
            .flat_map(|_| half::f16::from_f32(-1.0).to_le_bytes())
            .collect::<Vec<_>>();
        let expected_biases = (0..elements / 32)
            .flat_map(|_| half::f16::from_f32(15.0 + target as f32).to_le_bytes())
            .collect::<Vec<_>>();
        for (key, expected) in [
            (format!("weight{target}"), expected_words),
            (format!("scales{target}"), expected_scales),
            (format!("biases{target}"), expected_biases),
        ] {
            assert_eq!(bytes(&result, &key), expected);
            assert_eq!(bytes(&ordinary, &key), expected);
        }
    }
}

#[test]
fn cold_completions_share_tiling_writeback_and_cross_target_window() {
    exercise(vec![8, 64], 320, 1, 8, 1);
    exercise(vec![8, 64], 640, 1, 8, 2);
    exercise(vec![2, 4, 64], 640, 1, 8, 2);
    exercise(vec![8, 64], 100_000, 2, 2, 2);
}

#[test]
fn producer_failure_retires_the_already_queued_cold_submission() {
    let device = Device::new(DeviceType::Cpu, 0);
    let streams = [
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    ];
    let _roots = PrefillRootsRuntime::prepare_for_stream(&streams[0], &streams[1]).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let source = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "weight".into(),
            SafeDtype::F32,
            vec![8, 64],
            (0..512)
                .flat_map(|n| ((n % 16) as f32).to_le_bytes())
                .collect(),
        )])
        .unwrap(),
    );
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::new(32, 4).unwrap(),
        640,
        [
            BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                .unwrap()
                .with_affine_companion_dtype(RecipeDtype::F16)
                .unwrap(),
        ],
    )
    .unwrap();
    let prepared = ColdQuantization::prepare(source.into(), plan)
        .unwrap()
        .allocate_ordinary(&streams[0])
        .unwrap();
    let mut producer = FundedProducer {
        pool: WorkingMemoryPool::new(1 << 20, 0).unwrap(),
        runtime: &runtime,
        streams: &streams,
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: Some(1),
        truncate_at: None,
    };
    let error = prepared
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(
        error,
        ProducerFailure::Backend(Error::PrefillControl(WorkingMemoryError::UnknownBound))
    ));
    assert_eq!(producer.slots, [0, 1]);
    assert_eq!(producer.peak_live, 1);
    drain(&producer);
}

#[test]
fn failed_encoded_read_keeps_typed_cause_and_native_role_after_queue_unwinds() {
    use eredu_checkpoint::store::{
        EncodedReadFailure, EncodedReadFailureCause, SafetensorsWeightStore,
    };
    use safetensors::tensor::{serialize_to_file, TensorView};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let values = (0..512)
        .flat_map(|n| ((n % 16) as f32).to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(SafeDtype::F32, vec![8, 64], &values).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let device = Device::new(DeviceType::Cpu, 0);
    let streams = [
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    ];
    let _roots = PrefillRootsRuntime::prepare_for_stream(&streams[0], &streams[1]).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::new(32, 4).unwrap(),
        640,
        [
            BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                .unwrap()
                .with_affine_companion_dtype(RecipeDtype::F16)
                .unwrap(),
        ],
    )
    .unwrap();
    let prepared = ColdQuantization::prepare(source.into(), plan)
        .unwrap()
        .allocate_ordinary(&streams[0])
        .unwrap();
    let mut producer = FundedProducer {
        pool: WorkingMemoryPool::new(1 << 20, 0).unwrap(),
        runtime: &runtime,
        streams: &streams,
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: Some((1, path)),
    };
    let error = prepared
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(&error, ProducerFailure::Invocation(_)));
    assert_eq!(producer.slots, [0, 1]);
    assert_eq!(producer.peak_live, 2);
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    let input_error = loop {
        if let Some(input) = cause.downcast_ref::<crate::backend::runtime::checkpoint::store::EncodedInputConstructionError>() {
            break input;
        }
        cause = cause
            .source()
            .expect("original typed input cause in error chain");
    };
    assert!(matches!(
        input_error,
        crate::backend::runtime::checkpoint::store::EncodedInputConstructionError::Read(
            EncodedReadFailure {
                cause: EncodedReadFailureCause::Changed,
                ..
            }
        )
    ));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        producer.live.get() == 1
    });
    // The first tile retires independently. The failed second invocation and
    // its source error account remain live while the returned error is held.
    assert_eq!(producer.live.get(), 1);
    assert!(producer.pool.used_bytes().unwrap() > 0);
    drop(error);
    drain(&producer);
}

#[test]
fn cold_slot_admission_refusal_retains_invocation_until_error_retirement() {
    let device = Device::new(DeviceType::Cpu, 0);
    let streams = [
        Stream::new_with_device(&device),
        Stream::new_with_device(&device),
    ];
    let _roots = PrefillRootsRuntime::prepare_for_stream(&streams[0], &streams[1]).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let source = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "weight".into(),
            SafeDtype::F32,
            vec![8, 64],
            (0..512)
                .flat_map(|n| ((n % 16) as f32).to_le_bytes())
                .collect(),
        )])
        .unwrap(),
    );
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::new(32, 4).unwrap(),
        640,
        [
            BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                .unwrap()
                .with_affine_companion_dtype(RecipeDtype::F16)
                .unwrap(),
        ],
    )
    .unwrap();
    let prepare = || {
        ColdQuantization::prepare(source.clone().into(), plan.clone())
            .unwrap()
            .allocate_ordinary(&streams[0])
            .unwrap()
    };
    let mut producer = FundedProducer {
        pool: WorkingMemoryPool::new(0, 0).unwrap(),
        runtime: &runtime,
        streams: &streams,
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: None,
    };
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    let ProducerFailure::Admission(failure) = error else {
        panic!("role admission must refuse before callback")
    };
    let Some(WorkingMemoryError::BudgetExceeded {
        required_bytes,
        available_bytes: 0,
    }) = failure.accounting_failure()
    else {
        panic!("exact role comparison")
    };
    let role_bytes = *required_bytes;
    assert!(failure.constructor_failure().is_none());
    assert_eq!(producer.live.get(), 0);
    assert_eq!(producer.pool.used_bytes().unwrap(), 0);
    drop(failure);
    let slot_bytes = ColdMaterializationSlot::required_bytes(MaterializationPayloadShape {
        inputs: 1,
        outputs: 3,
        pending_sources: 0,
    })
    .unwrap();
    producer.pool = WorkingMemoryPool::new(role_bytes + slot_bytes - 1, 0).unwrap();
    producer.slots.clear();
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(&error, ProducerFailure::Invocation(_)));
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    let refusal = loop {
        if let Some(cause) = cause.downcast_ref::<WorkingMemoryError>() {
            break cause;
        }
        cause = cause.source().expect("typed slot admission cause");
    };
    assert!(
        matches!(refusal, WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }
        if *required_bytes == slot_bytes && *available_bytes == slot_bytes - 1)
    );
    assert_eq!(producer.slots, [0]);
    assert_eq!(producer.live.get(), 1);
    assert_eq!(producer.pool.used_bytes().unwrap(), role_bytes);
    drop(error);
    drain(&producer);
}
