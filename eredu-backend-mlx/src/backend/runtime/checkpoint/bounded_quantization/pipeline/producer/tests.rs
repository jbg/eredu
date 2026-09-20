use super::*;
use crate::backend::runtime::checkpoint::store::{
    ColdMaterializationSlot, MaterializationPayloadShape,
};
use eredu_checkpoint::{recipe::EncodedRecipeRead, AffineQuantization};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use safemlx::{Device, DeviceType, PrefillRootsRuntime, PreparedInputRuntime};
use std::{cell::Cell, rc::Rc};

#[derive(Debug)]
struct Invocation(Rc<Cell<usize>>);
impl Drop for Invocation {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

use encoded_affine::TileError;

// Runtime/stream/source birth, recipe declarations and final output are fixture
// prerequisites. The pool funds metadata, keys, compiled reads, native work,
// input, completion slots and queue; this is not full producer admission.
struct FundedProducer<'a> {
    pool: WorkingMemoryPool,
    runtime: &'a PreparedInputRuntime,
    streams: [&'a Stream; 2],
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
    type Error = TileError<Invocation>;

    fn submit(
        &mut self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<(Self::Completion, u64), TileError<Invocation>> {
        self.slots.push(slot);
        if self.fail_at == Some(self.slots.len() - 1) {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound).into());
        }
        let before_host = self.pool.used_bytes().unwrap();
        let tile = encoded_affine::Tile::prepare::<Invocation>(
            &self.pool,
            self.runtime,
            source,
            recipe,
            target,
            quantization,
        )?;
        let host_bytes = self.pool.used_bytes().unwrap() - before_host;
        let source_bytes = tile.source_bytes();
        let payload = MaterializationPayloadShape {
            inputs: 1,
            outputs: 3,
            pending_sources: 0,
        };
        let input = tile.input().unwrap();
        let input_bytes = input.required_bytes().unwrap();
        let slot_bytes = ColdMaterializationSlot::required_bytes(payload).unwrap();
        self.live.set(self.live.get() + 1);
        self.peak_live = self.peak_live.max(self.live.get());
        let pool = &self.pool;
        let peak_used = &self.peak_used;
        let stream = self.streams[slot];
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
        let plan = tile.plan(Invocation(self.live.clone()), stream)?;
        self.largest_tile = self
            .largest_tile
            .max(plan.required_bytes().unwrap() + input_bytes + slot_bytes + host_bytes);
        let submitted = plan.submit(pool).map_err(|failure| {
            let (uncalled, failure) = failure.into_parts();
            drop(uncalled);
            TileError::Admission(failure)
        })?;
        peak_used.set(peak_used.get().max(pool.used_bytes().unwrap()));
        submitted
            .into_result()
            .map(|completion| (completion, source_bytes))
            .map_err(TileError::Invocation)
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
        QuantizedCheckpoint::create(source.clone(), plan.clone(), &streams[0]).unwrap();
    let prepared = ColdQuantization::prepare(source.into(), plan)
        .unwrap()
        .allocate_ordinary(&streams[0])
        .unwrap();
    let mut producer = FundedProducer {
        pool: WorkingMemoryPool::new(1 << 20, 0).unwrap(),
        runtime: &runtime,
        streams: streams.each_ref(),
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: None,
    };
    let (result, _) = prepared
        .materialize_with_admitted_producer(&producer.pool.clone(), DeviceType::Cpu, &mut producer)
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
    let controls = BoundedAllocatorCache::required_original_bytes(
        super::super::controls::required_bytes::<<FundedProducer<'_> as TileProducer>::Completion>(
        )
        .unwrap(),
    )
    .unwrap();
    assert!(producer.peak_used.get() <= controls + expected_slots as u64 * producer.largest_tile);
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
            assert_eq!(bytes(result.source(), &key), expected);
            assert_eq!(bytes(ordinary.source(), &key), expected);
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
        streams: streams.each_ref(),
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: Some(1),
        truncate_at: None,
    };
    let error = prepared
        .materialize_with_admitted_producer(&producer.pool.clone(), DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(
        error,
        PipelineAdmissionError::Producer(TileError::Backend(Error::PrefillControl(
            WorkingMemoryError::UnknownBound
        )))
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
        streams: streams.each_ref(),
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: Some((1, path)),
    };
    let error = prepared
        .materialize_with_admitted_producer(&producer.pool.clone(), DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(
        &error,
        PipelineAdmissionError::Producer(TileError::Invocation(_))
    ));
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
        100_000,
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
        streams: streams.each_ref(),
        slots: Vec::new(),
        live: Rc::new(Cell::new(0)),
        peak_live: 0,
        peak_used: Cell::new(0),
        largest_tile: 0,
        fail_at: None,
        truncate_at: None,
    };
    let error = prepare()
        .materialize_with_admitted_producer(&producer.pool.clone(), DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(&error, PipelineAdmissionError::Admission(_)));
    assert!(producer.slots.is_empty());
    assert_eq!(producer.live.get(), 0);
    assert_eq!(producer.pool.used_bytes().unwrap(), 0);
    drop(error);
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    let TileError::Metadata(failure) = error else {
        panic!("metadata admission must refuse before inference")
    };
    let Some(WorkingMemoryError::BudgetExceeded {
        required_bytes,
        available_bytes: 0,
    }) = failure.accounting_failure()
    else {
        panic!("exact metadata comparison")
    };
    let metadata_bytes = *required_bytes;
    assert!(failure.constructor_failure().is_none());
    assert_eq!(producer.live.get(), 0);
    assert_eq!(producer.pool.used_bytes().unwrap(), 0);
    drop(failure);
    producer.pool = WorkingMemoryPool::new(metadata_bytes, 0).unwrap();
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    let TileError::Read(eredu_runtime::working_memory::EncodedRecipeSourceError::Keys(failure)) =
        error
    else {
        panic!("key admission must refuse after the funded metadata");
    };
    assert!(matches!(
        failure.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(failure);
    assert_eq!(producer.pool.used_bytes().unwrap(), 0);

    // Construct the actual host prerequisites in a separate sizing run. The
    // retained quote excludes scratch that has already retired after compilation.
    let sizing = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let target = &plan.targets[0];
    let root: eredu_checkpoint::store::RetainedCheckpointSource = source.clone().into();
    let tile = encoded_affine::Tile::prepare::<Invocation>(
        &sizing,
        &runtime,
        &root,
        &target.source,
        target,
        plan.quantization,
    )
    .unwrap();
    let prerequisites = sizing.used_bytes().unwrap();
    let construction_peak = sizing.peak_bytes().unwrap();
    let role = tile
        .plan(Invocation(Rc::new(Cell::new(1))), &streams[0])
        .unwrap();
    let role_bytes = role.required_bytes().unwrap();
    drop(role);
    drop(tile);
    assert_eq!(sizing.used_bytes().unwrap(), 0);
    assert!(prerequisites + role_bytes - 1 >= construction_peak);
    producer.pool = WorkingMemoryPool::new(prerequisites + role_bytes - 1, 0).unwrap();
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    let TileError::Admission(failure) = error else {
        panic!("role comparison must precede native callback")
    };
    assert!(
        matches!(failure.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == role_bytes && *available_bytes == role_bytes - 1)
    );
    assert_eq!(producer.live.get(), 0);
    assert_eq!(producer.pool.used_bytes().unwrap(), 0);
    drop(failure);
    let slot_bytes = ColdMaterializationSlot::required_bytes(MaterializationPayloadShape {
        inputs: 1,
        outputs: 3,
        pending_sources: 0,
    })
    .unwrap();
    producer.pool = WorkingMemoryPool::new(prerequisites + role_bytes + slot_bytes - 1, 0).unwrap();
    producer.slots.clear();
    let error = prepare()
        .materialize_with_producer(DeviceType::Cpu, &mut producer)
        .unwrap_err();
    assert!(matches!(&error, TileError::Invocation(_)));
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

#[test]
fn admitted_cpu_resources_drive_tiles_without_ordinary_runtime_setup() {
    const CHILD: &str = "EREDU_ADMITTED_CPU_TILE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                concat!(
                "backend::runtime::checkpoint::bounded_quantization::pipeline::producer::tests::",
                "admitted_cpu_resources_drive_tiles_without_ordinary_runtime_setup"
            ),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("ADMITTED_CPU_TILES_OK"));
        return;
    }
    let pool = crate::backend::managed_memory::domain();
    let baseline = pool.used_bytes().unwrap();
    let resources = cpu_resources::CpuTileResources::prepare(&pool).unwrap();
    resources.validate_pool(&pool).unwrap();
    let controls = resources.original_control_bytes();
    let persistent = pool.used_bytes().unwrap();
    assert!(persistent > baseline + controls);
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    assert!(resources.validate_pool(&foreign).is_err());
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), persistent);
    check_cold_metadata_refusals(&pool, &resources, persistent);
    assert_ne!(
        resources.streams()[0].get_index().unwrap(),
        resources.streams()[1].get_index().unwrap()
    );
    fn encode(dtype: SafeDtype, value: f32) -> Vec<u8> {
        match dtype {
            SafeDtype::F32 => value.to_le_bytes().to_vec(),
            SafeDtype::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
            SafeDtype::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
            _ => unreachable!(),
        }
    }
    for input_dtype in [SafeDtype::F32, SafeDtype::F16, SafeDtype::BF16] {
        for (companion_dtype, companion) in [
            (SafeDtype::F16, RecipeDtype::F16),
            (SafeDtype::BF16, RecipeDtype::BF16),
            (SafeDtype::F32, RecipeDtype::F32),
        ] {
            for (budget, expected_slots) in [(320, 1), (640, 2)] {
                let source = Arc::new(
                    MemoryWeightStore::from_safetensors([(
                        "weight".into(),
                        input_dtype,
                        vec![8, 64],
                        (0..512)
                            .flat_map(|n| encode(input_dtype, (n % 16) as f32))
                            .collect(),
                    )])
                    .unwrap(),
                );
                let plan = BoundedQuantizationPlan::new(
                    AffineQuantization::new(32, 4).unwrap(),
                    budget,
                    [
                        BoundedQuantizationTarget::direct("weight", "scales", Some("biases"))
                            .unwrap()
                            .with_affine_companion_dtype(companion.clone())
                            .unwrap(),
                    ],
                )
                .unwrap();
                let source = source.into();
                let metadata_policy = eredu_runtime::working_memory::DependencyMemoryPolicy::default();
                let conversion = admission::ColdConversion {
                    source: &source, plan: &plan, pool: &pool, resources: &resources, metadata_policy,
                };
                let metadata_bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&conversion).unwrap();
                let companion_bytes = 16 * plan.targets()[0].affine_companion_bytes();
                let output_bytes: u64 = [
                    ("weight", [8, 8], 256),
                    ("scales", [8, 2], companion_bytes),
                    ("biases", [8, 2], companion_bytes),
                ].into_iter().map(|(name, shape, bytes)| {
                    WorkingMemoryPool::memory_tensor_buffer_quote(name, &shape, bytes, metadata_policy)
                        .unwrap().total_bytes()
                }).sum();
                let with_outputs = persistent + metadata_bytes + output_bytes;
                // Original source/header birth remains a fixture prerequisite.
                // The initializer admits cold clones, inference/provenance metadata,
                // conversion and the completed handoff through the common worker.
                let result = conversion.prepare().unwrap();
                result.validate(&source, &plan).unwrap();
                let (source, report) = result.into_parts();
                assert_eq!(report.source_tiles, 8);
                assert_eq!(report.peak_in_flight_tiles, expected_slots);
                let expected_words = (0..64)
                    .flat_map(|n| {
                        (if n % 2 == 0 {
                            0x89abcdef_u32
                        } else {
                            0x01234567_u32
                        })
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(bytes(source.as_ref(), "weight"), expected_words);
                assert_eq!(
                    bytes(source.as_ref(), "scales"),
                    (0..16)
                        .flat_map(|_| encode(companion_dtype, -1.0))
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    bytes(source.as_ref(), "biases"),
                    (0..16)
                        .flat_map(|_| encode(companion_dtype, 15.0))
                        .collect::<Vec<_>>()
                );
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    safemlx::reclaim_allocation_owners();
                    pool.used_bytes() == Ok(with_outputs)
                });
                assert_eq!(pool.used_bytes().unwrap(), with_outputs);
                let identity = source.identity();
                let catalog = source.source_keys().into_iter().map(|key| {
                    let row = eredu_checkpoint::store::PreparedTensorSource {
                        metadata: source.source_metadata(&key).unwrap(),
                        provenance: source.source_provenance(&key).unwrap(),
                    };
                    (key, row)
                }).collect();
                let source = Arc::new(eredu_checkpoint::store::PreparedCheckpointSource::new(
                    source, catalog,
                ).unwrap()).into();
                let keys = [String::from("weight")];
                let read = eredu_checkpoint::store::MemoryEncodedReadPlan::from_source(&source, &keys)
                    .unwrap().unwrap().construct(()).unwrap();
                let selected_output_bytes = WorkingMemoryPool::memory_tensor_buffer_quote(
                    "weight", &[8, 8], 256, metadata_policy,
                ).unwrap().total_bytes();
                drop(source);
                assert_eq!(pool.used_bytes().unwrap(), persistent + metadata_bytes + selected_output_bytes);
                drop(identity);
                assert_eq!(pool.used_bytes().unwrap(), persistent + selected_output_bytes);
                let mut output = vec![0; expected_words.len()];
                read.read_into(&mut output).unwrap();
                assert_eq!(output, expected_words);
                assert!(pool.used_bytes().unwrap() > persistent);
                drop(read);
                safemlx::reclaim_allocation_owners();
                assert_eq!(pool.used_bytes().unwrap(), persistent);
            }
        }
    }
    drop(resources);
    safemlx::reclaim_allocation_owners();
    // The native scheduler, allocator, stream registrations and worker threads
    // really survive these wrappers. Only the composing controls retire here.
    assert_eq!(pool.used_bytes().unwrap(), persistent - controls);
    assert!(pool.used_bytes().unwrap() > baseline);
    println!("ADMITTED_CPU_TILES_OK");
}

fn check_cold_metadata_refusals(
    pool: &WorkingMemoryPool,
    resources: &cpu_resources::CpuTileResources,
    persistent: u64,
) {
    use eredu_runtime::working_memory::{DependencyMemoryPolicy, SharedNativeInitializer};
    let source = Arc::new(MemoryWeightStore::from_safetensors([
        ("weight".into(), SafeDtype::F32, vec![1, 32], vec![0; 128]),
        ("scales".into(), SafeDtype::U8, vec![1], vec![7]),
    ]).unwrap()).into();
    let plan = BoundedQuantizationPlan::new(
        AffineQuantization::new(32, 4).unwrap(), 320,
        [BoundedQuantizationTarget::direct("weight", "scales", Some("biases")).unwrap()],
    ).unwrap();
    let metadata_policy = DependencyMemoryPolicy::default();
    let conversion = || admission::ColdConversion { source: &source, plan: &plan, pool, resources, metadata_policy };
    let required = WorkingMemoryPool::shared_native_initialization_required_bytes(&conversion()).unwrap();
    let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
    let refusal = admission::ColdConversion { pool: &short, ..conversion() }.prepare().unwrap_err();
    assert!(matches!(refusal.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { .. })));
    assert!(refusal.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(refusal);

    let overflow = admission::ColdConversion {
        metadata_policy: DependencyMemoryPolicy { fixed_bytes: usize::MAX, bytes_per_input_byte: 1 },
        ..conversion()
    }.prepare().unwrap_err();
    assert!(matches!(overflow.accounting_failure(), Some(WorkingMemoryError::Overflow)));
    assert_eq!(pool.used_bytes().unwrap(), persistent);
    drop(overflow);

    let enlarged = admission::ColdConversion {
        metadata_policy: DependencyMemoryPolicy { fixed_bytes: metadata_policy.fixed_bytes + 123, ..metadata_policy },
        ..conversion()
    };
    assert_eq!(enlarged.required_storage_bytes().unwrap() - conversion().required_storage_bytes().unwrap(), 123);
    // Known metadata permits admission, then ordinary cold collision checking
    // fails. Its formatted cause keeps the original account until error drop.
    let failure = conversion().prepare().unwrap_err();
    assert!(matches!(failure.constructor_failure(), Some(admission::ConstructionError::Preparation(Error::Quantization(_)))));
    assert_eq!(pool.used_bytes().unwrap(), persistent + required);
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), persistent);
}

#[test]
fn cpu_resource_admission_rejects_foreign_and_unquoted_domains_before_runtime_birth() {
    const CHILD: &str = "EREDU_CPU_TILE_RESOURCE_REFUSAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                concat!(
                "backend::runtime::checkpoint::bounded_quantization::pipeline::producer::tests::",
                "cpu_resource_admission_rejects_foreign_and_unquoted_domains_before_runtime_birth"
            ),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("CPU_TILE_RESOURCE_REFUSALS_OK"));
        return;
    }
    fn policy(error: &cpu_resources::ResourceError) -> &WorkingMemoryError {
        let mut cause: &(dyn std::error::Error + 'static) = error;
        loop {
            if let Some(policy) = cause.downcast_ref::<WorkingMemoryError>() {
                return policy;
            }
            cause = cause.source().expect("original policy cause");
        }
    }
    let pool = crate::backend::managed_memory::domain();
    let before = pool.used_bytes().unwrap();
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let error = cpu_resources::CpuTileResources::prepare(&foreign).unwrap_err();
    assert!(matches!(
        policy(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert!(crate::backend::managed_memory::input_allocator::admitted_initializer(&pool).is_err());
    drop(error);
    let ordinary = pool.acquire_unquoted().unwrap();
    let error = cpu_resources::CpuTileResources::prepare(&pool).unwrap_err();
    assert!(matches!(policy(&error), WorkingMemoryError::UnknownBound));
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert!(crate::backend::managed_memory::input_allocator::admitted_initializer(&pool).is_err());
    drop(error);
    drop(ordinary);
    let resources = cpu_resources::CpuTileResources::prepare(&pool).unwrap();
    resources.validate_pool(&pool).unwrap();
    println!("CPU_TILE_RESOURCE_REFUSALS_OK");
}
