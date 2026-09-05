mod support {
    pub mod numeric_backend;
}

use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
use eredu_codec::mimi::{
    construct, prepare_checkpoint, released_checkpoint_requirements, Config,
    MimiParameterRequirement,
};
use eredu_nn::{
    Parameter, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized, Tensor,
};
use eredu_runtime::{bind_materialized_unit, materialize_bindings, WeightBinding};
use serde_json::json;
use support::numeric_backend::{activate, Context, FailAt, NumericTensor, ReferenceBackend};
use tempfile::TempDir;

struct Fixture {
    _directory: TempDir,
    checkpoint: PathBuf,
    active_parameters: usize,
    transpose_parameters: usize,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| build_fixture().expect("sparse Mimi fixture builds"))
}

fn build_fixture() -> Result<Fixture, Box<dyn std::error::Error>> {
    let config = Config::v0_1(Some(1));
    let requirements = released_checkpoint_requirements(&config)?;
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("mimi");
    fs::create_dir(&checkpoint)?;
    let mut weight_map = serde_json::Map::new();
    let mut transpose_parameters = 0;
    for (index, requirement) in requirements.iter().enumerate() {
        let file_name = format!("tensor-{index:03}.safetensors");
        write_tensor_shard(&checkpoint.join(&file_name), requirement)?;
        weight_map.insert(
            requirement.checkpoint_key().to_owned(),
            serde_json::Value::String(file_name),
        );
        if requirement.is_active()
            && matches!(requirement.recipe(), DerivedWeightRecipe::Transpose { .. })
        {
            transpose_parameters += 1;
        }
    }
    fs::write(
        checkpoint.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": weight_map}))?,
    )?;
    Ok(Fixture {
        _directory: directory,
        checkpoint,
        active_parameters: requirements.iter().filter(|item| item.is_active()).count(),
        transpose_parameters,
    })
}

fn write_tensor_shard(
    path: &Path,
    requirement: &MimiParameterRequirement,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        requirement.source_bytes(),
        requirement.physical_shape().iter().product::<usize>() as u64 * 4
    );
    let header = json!({
        requirement.checkpoint_key(): {
            "dtype": "F32",
            "shape": requirement.physical_shape(),
            "data_offsets": [0, requirement.source_bytes()],
        }
    });
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let payload_start = 8 + header.len() as u64;
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.set_len(payload_start + requirement.source_bytes())?;
    for (logical_coordinates, value) in nonzero_values(requirement) {
        let physical = physical_coordinates(requirement, &logical_coordinates);
        let offset = flat_index(&physical, requirement.physical_shape()) as u64 * 4;
        file.seek(SeekFrom::Start(payload_start + offset))?;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn nonzero_values(requirement: &MimiParameterRequirement) -> Vec<(Vec<usize>, f32)> {
    let name = requirement.logical_name();
    let connected_convolution = name == "encoder.init_conv1d.weight"
        || name == "encoder.final_conv1d.weight"
        || name == "decoder.init_conv1d.weight"
        || name == "decoder.final_conv1d.weight"
        || name == "downsample.weight"
        || name == "upsample.weight"
        || (name.starts_with("encoder.layers.") && name.ends_with(".downsample.weight"))
        || (name.starts_with("decoder.layers.") && name.ends_with(".upsample.weight"));
    if connected_convolution {
        let transposed = name == "upsample.weight"
            || (name.starts_with("decoder.layers.") && name.ends_with(".upsample.weight"));
        let kernel = if transposed {
            0
        } else {
            requirement.logical_shape()[1] - 1
        };
        return vec![(vec![0, kernel, 0], 1.0)];
    }
    if name == "quantizer.rvq_first.input_proj.weight"
        || name == "quantizer.rvq_first.output_proj.weight"
    {
        return vec![(vec![0, 0, 0], 1.0)];
    }
    if name == "quantizer.rvq_first.vq.layers.0._codebook.cluster_usage" {
        return vec![(vec![0], 1.0), (vec![1], 1.0)];
    }
    if name == "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum" {
        return vec![(vec![0, 0], -1.0), (vec![1, 0], 1.0)];
    }
    Vec::new()
}

fn physical_coordinates(requirement: &MimiParameterRequirement, logical: &[usize]) -> Vec<usize> {
    match requirement.recipe() {
        DerivedWeightRecipe::Source { .. } => logical.to_vec(),
        DerivedWeightRecipe::Transpose { axes, .. } => {
            let mut physical = vec![0; axes.len()];
            for (logical_axis, &physical_axis) in axes.iter().enumerate() {
                physical[physical_axis] = logical[logical_axis];
            }
            physical
        }
        recipe => panic!("released Mimi fixture contains unexpected recipe {recipe:?}"),
    }
}

fn flat_index(coordinates: &[usize], shape: &[usize]) -> usize {
    let mut result = 0;
    for (&coordinate, &dimension) in coordinates.iter().zip(shape) {
        result = result * dimension + coordinate;
    }
    result
}

fn prepare(context: &Context) -> eredu_codec::mimi::PreparedMimiArtifact {
    activate(context);
    prepare_checkpoint(&fixture().checkpoint, Config::v0_1(Some(1)))
        .expect("fixture passes exact released catalog validation")
}

fn fingerprint_f32(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn constructs_real_mimi_and_runs_numeric_latent_pcm_and_streaming_paths() {
    let context = Context::new(FailAt::None);
    let prepared = prepare(&context);
    assert_eq!(prepared.requirements().len(), 318);
    assert_eq!(prepared.bindings().len(), fixture().active_parameters);
    let mut mimi = construct::<ReferenceBackend>(prepared, &context, &context)
        .expect("numeric reference backend constructs actual Mimi");

    let latent = NumericTensor::f32(
        (0..512)
            .map(|channel| if channel == 0 { 0.75 } else { 0.0 })
            .collect(),
        &[1, 512, 1],
    );
    let codes = mimi
        .encode_latent(&latent, &context)
        .expect("latent encode");
    assert_eq!(codes.shape(), &[1, 1, 1]);
    assert_eq!(codes.to_i32_vec(&context).unwrap(), vec![1]);
    let decoded = mimi.decode_latent(&codes, &context).expect("latent decode");
    assert_eq!(decoded.shape(), &[1, 512, 1]);
    assert_eq!(
        fingerprint_f32(&decoded.to_f32_vec(&context).unwrap()),
        0xdd53_f575_b485_4b25,
        "decoded latent values changed"
    );

    let pcm = NumericTensor::f32(vec![0.25; 1_920], &[1, 1, 1_920]);
    let offline_codes = mimi.encode(&pcm, &context).expect("offline PCM encode");
    assert_eq!(offline_codes.shape(), &[1, 1, 1]);
    assert_eq!(offline_codes.to_i32_vec(&context).unwrap(), vec![2]);
    let decode_codes = NumericTensor::from_i32_slice(&[1], &[1, 1, 1], &context).unwrap();
    let offline_pcm = mimi
        .decode(&decode_codes, &context)
        .expect("offline PCM decode");
    assert_eq!(offline_pcm.shape(), &[1, 1, 1_920]);
    assert_eq!(
        fingerprint_f32(&offline_pcm.to_f32_vec(&context).unwrap()),
        0x3f83_673a_fba7_f925,
        "offline decoded PCM values changed"
    );

    mimi.reset_encode_state();
    let first = NumericTensor::f32(vec![0.25; 960], &[1, 1, 960]);
    let second = NumericTensor::f32(vec![0.25; 960], &[1, 1, 960]);
    let first_result = mimi
        .encode_step(&first, &context)
        .expect("first encode step");
    let second_result = mimi
        .encode_step(&second, &context)
        .expect("second encode step");
    assert!(
        first_result.is_none(),
        "half a Mimi frame must remain buffered"
    );
    assert_eq!(
        second_result
            .expect("the second half-frame must emit one token")
            .to_i32_vec(&context)
            .unwrap(),
        vec![2]
    );
    mimi.reset_encode_state();
    assert!(mimi.encode_step(&first, &context).unwrap().is_none());
    assert_eq!(
        mimi.encode_step(&second, &context)
            .unwrap()
            .expect("reset stream must reproduce the emitted token")
            .to_i32_vec(&context)
            .unwrap(),
        vec![2]
    );

    let one_code = NumericTensor::from_i32_slice(&[1], &[1, 1], &context).unwrap();
    mimi.reset_decode_state();
    let stream_a = mimi
        .decode_step(&one_code, &context)
        .expect("first decode step");
    let stream_b = mimi
        .decode_step(&one_code, &context)
        .expect("second decode step");
    assert_eq!(stream_a.shape(), &[1, 1, 1_920]);
    assert_eq!(stream_b.shape(), &[1, 1, 1_920]);
    mimi.reset_decode_state();
    let stream_after_reset = mimi.decode_step(&one_code, &context).unwrap();
    let stream_values_a = stream_a.to_f32_vec(&context).unwrap();
    let stream_values_b = stream_b.to_f32_vec(&context).unwrap();
    assert_eq!(fingerprint_f32(&stream_values_a), 0x3f83_673a_fba7_f925);
    assert_eq!(fingerprint_f32(&stream_values_b), 0x3f83_673a_fba7_f925);
    let streamed =
        NumericTensor::concatenate(&[stream_a.clone(), stream_b.clone()], 2, &context).unwrap();
    let offline_two_codes = NumericTensor::from_i32_slice(&[1, 1], &[1, 1, 2], &context).unwrap();
    let offline_two = mimi.decode(&offline_two_codes, &context).unwrap();
    let streamed_values = streamed.to_f32_vec(&context).unwrap();
    let offline_two_values = offline_two.to_f32_vec(&context).unwrap();
    assert_eq!(fingerprint_f32(&streamed_values), 0x03b7_4671_1441_cf25);
    assert_eq!(streamed_values, offline_two_values);
    assert_eq!(
        stream_values_a,
        stream_after_reset.to_f32_vec(&context).unwrap()
    );

    mimi.reset_encode_state();
    context.set_failure(FailAt::Execution);
    assert!(mimi.encode_step(&first, &context).is_err());
    context.set_failure(FailAt::None);
    mimi.reset_encode_state();
    assert!(mimi.encode_step(&first, &context).unwrap().is_none());
    assert_eq!(
        mimi.encode_step(&second, &context)
            .unwrap()
            .expect("reset after execution failure must recover")
            .to_i32_vec(&context)
            .unwrap(),
        vec![2]
    );

    let counters = context.counters();
    assert_eq!(counters.preflight, fixture().active_parameters);
    assert_eq!(counters.payload_reads, fixture().active_parameters);
    assert_eq!(counters.direct_materializations, 0);
    assert_eq!(counters.direct_recipes, fixture().active_parameters);
    assert_eq!(counters.materialized_cache_hits, 0);
    assert_eq!(counters.materializations, fixture().active_parameters);
    assert_eq!(counters.completions, fixture().active_parameters);
    assert_eq!(
        counters.completions_with_live_leases,
        fixture().active_parameters
    );
    assert_eq!(counters.validations, fixture().active_parameters);
    assert_eq!(counters.binds, fixture().active_parameters);
    assert_eq!(counters.transpose_recipes, fixture().transpose_parameters);
    assert_eq!(counters.guard_drops, fixture().active_parameters);
    assert_eq!(counters.encoded_lease_drops, fixture().active_parameters);
    assert_eq!(counters.live_encoded_leases, 0);
    assert_eq!(counters.max_live_encoded_leases, 1);
    assert_eq!(counters.execution_failures, 1);
}

#[test]
fn every_active_codebook_count_constructs_and_executes_the_same_neutral_path() {
    let latent = (0..512)
        .map(|channel| if channel == 0 { 0.75 } else { 0.0 })
        .collect::<Vec<_>>();
    let context = Context::new(FailAt::None);
    activate(&context);
    let warm = prepare_checkpoint(&fixture().checkpoint, Config::v0_1(Some(32))).unwrap();
    let warm = construct::<ReferenceBackend>(warm, &context, &context).unwrap();
    assert_eq!(warm.mimi_config().num_codebooks, 32);
    let mut expected_materializations = 318;
    for active in 1..=32 {
        let prepared = prepare_checkpoint(&fixture().checkpoint, Config::v0_1(Some(active)))
            .expect("the exact released artifact admits every supported active count");
        assert_eq!(prepared.requirements().len(), 318);
        assert_eq!(prepared.bindings().len(), 3 * active as usize + 222);
        let mut mimi = construct::<ReferenceBackend>(prepared, &context, &context)
            .expect("every supported active count constructs through generic mechanisms");

        let latent = NumericTensor::f32(latent.clone(), &[1, 512, 1]);
        let codes = mimi.encode_latent(&latent, &context).unwrap();
        assert_eq!(codes.shape(), &[1, active, 1]);
        let mut expected_codes = vec![0; active as usize];
        expected_codes[0] = 1;
        assert_eq!(codes.to_i32_vec(&context).unwrap(), expected_codes);

        let decoded = mimi.decode_latent(&codes, &context).unwrap();
        assert_eq!(decoded.shape(), &[1, 512, 1]);
        assert_eq!(
            fingerprint_f32(&decoded.to_f32_vec(&context).unwrap()),
            0xdd53_f575_b485_4b25
        );
        expected_materializations += 3 * active as usize + 222;
        let counters = context.counters();
        assert_eq!(counters.binds, expected_materializations);
        assert_eq!(counters.payload_reads, 318);
        assert_eq!(counters.encoded_lease_drops, 318);
        assert_eq!(
            counters.materialized_cache_hits,
            expected_materializations - 318
        );
    }
}

#[test]
fn failures_are_causal_and_never_partially_publish() {
    for failure in [
        FailAt::Capability,
        FailAt::Read,
        FailAt::Recipe,
        FailAt::Completion,
        FailAt::Validate,
    ] {
        let context = Context::new(failure);
        let prepared = prepare(&context);
        let result = construct::<ReferenceBackend>(prepared, &context, &context);
        assert!(result.is_err(), "{failure:?} must fail");
        let counters = context.counters();
        assert_eq!(counters.binds, 0, "{failure:?} must not publish");
        match failure {
            FailAt::Capability => {
                assert_eq!(counters.payload_reads, 0);
                assert_eq!(counters.unloaded_allocations, 0);
            }
            FailAt::Read => assert_eq!(counters.materializations, 0),
            FailAt::Recipe => {
                assert!(counters.transpose_recipes > 0);
                assert!(counters.materializations < fixture().active_parameters);
            }
            FailAt::Completion => {
                assert_eq!(counters.materializations, 1);
                assert_eq!(counters.guard_drops, 1);
            }
            FailAt::Validate => {
                assert_eq!(counters.materializations, fixture().active_parameters);
                assert_eq!(counters.validations, 1);
            }
            FailAt::None | FailAt::DelayedCompletion | FailAt::Execution => unreachable!(),
        }
        assert_eq!(counters.live_encoded_leases, 0);
        assert_eq!(counters.encoded_lease_drops, counters.payload_reads);
    }
}

#[test]
fn encoded_leases_remain_live_through_delayed_completion() {
    let context = Context::new(FailAt::DelayedCompletion);
    let prepared = prepare(&context);
    let worker_context = context.clone();
    let worker = std::thread::spawn(move || {
        activate(&worker_context);
        construct::<ReferenceBackend>(prepared, &worker_context, &worker_context).is_ok()
    });

    context.wait_for_delayed_completion();
    let waiting = context.counters();
    assert_eq!(waiting.completions, 1);
    assert_eq!(waiting.completions_with_live_leases, 1);
    assert_eq!(waiting.live_encoded_leases, 1);
    assert_eq!(waiting.encoded_lease_drops, 0);
    context.release_delayed_completion();
    assert!(worker.join().unwrap());

    let completed = context.counters();
    assert_eq!(completed.delayed_completion_waits, 1);
    assert_eq!(completed.completions, fixture().active_parameters);
    assert_eq!(
        completed.completions_with_live_leases,
        fixture().active_parameters
    );
    assert_eq!(completed.live_encoded_leases, 0);
    assert_eq!(completed.encoded_lease_drops, fixture().active_parameters);
    assert_eq!(completed.guard_drops, fixture().active_parameters);
}

#[derive(Debug)]
struct ExtensionUnit {
    weight: Parameter<NumericTensor>,
}

impl Parameterized<NumericTensor> for ExtensionUnit {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, NumericTensor>,
    {
        self.weight.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, NumericTensor>,
    {
        self.weight.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight.set_trainable(trainable);
    }
}

#[test]
fn generic_parameterized_extension_uses_the_same_backend_mechanisms() {
    let context = Context::new(FailAt::None);
    activate(&context);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("extension.safetensors");
    let header = json!({"extension.source": {
        "dtype": "F32", "shape": [2], "data_offsets": [0, 8]
    }});
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&1.25f32.to_le_bytes()).unwrap();
    file.write_all(&(-0.5f32).to_le_bytes()).unwrap();
    drop(file);

    let store = SafetensorsWeightStore::open(&path).unwrap();
    let bindings = [WeightBinding::new(
        "extension.weight",
        "extension.source",
        TensorSelection::Full,
        8,
    )
    .unwrap()];
    let unit = materialize_bindings::<ReferenceBackend>(&store, &bindings, &context).unwrap();
    let mut extension = ExtensionUnit {
        weight: Parameter::unloaded(
            ParameterSpec::trainable("extension.weight").unwrap(),
            &[2],
            &context,
        )
        .unwrap(),
    };
    bind_materialized_unit::<ReferenceBackend, _>(&mut extension, unit).unwrap();
    assert_eq!(
        extension.weight.as_ref().to_f32_vec(&context).unwrap(),
        vec![1.25, -0.5]
    );
    let counters = context.counters();
    assert_eq!(counters.payload_reads, 1);
    assert_eq!(counters.materializations, 1);
    assert_eq!(counters.validations, 1);
    assert_eq!(counters.binds, 1);
}
