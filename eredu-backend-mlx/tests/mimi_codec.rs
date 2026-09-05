use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
};

use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::checkpoint::store::MlxParameterMaterializationContext,
    },
    native::ExecutionContext,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    safetensors::SafetensorsMetadataCatalog,
    store::{SafetensorsWeightStore, WeightStore},
};
use eredu_codec::mimi::{
    construct, prepare_checkpoint, prepare_source, released_checkpoint_requirements, Config, Mimi,
    MimiParameterRequirement,
};
use eredu_codec::AudioTokenizer;
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, Tensor};
use safemlx::{ops::indexing::TryIndexOp, transforms::eval, Array, Device, DeviceType};

fn write_sparse_v01_checkpoint(
    path: &Path,
    requirements: &[MimiParameterRequirement],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tensors = serde_json::Map::new();
    let mut offsets = BTreeMap::new();
    let mut next_offset = 0u64;
    for requirement in requirements {
        let end = next_offset.checked_add(requirement.source_bytes()).unwrap();
        tensors.insert(
            requirement.checkpoint_key().to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": requirement.physical_shape(),
                "data_offsets": [next_offset, end],
            }),
        );
        offsets.insert(requirement.checkpoint_key().to_owned(), next_offset);
        next_offset = end;
    }
    let mut header = serde_json::to_vec(&tensors)?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let payload_start = 8 + u64::try_from(header.len())?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.set_len(payload_start.checked_add(next_offset).unwrap())?;

    for requirement in requirements {
        let values: &[(Vec<usize>, f32)] = match requirement.logical_name() {
            "encoder.init_conv1d.bias" => &[(vec![0], 3.25), (vec![63], -1.5)],
            "encoder.init_conv1d.weight" => &[(vec![2, 5, 0], 7.0), (vec![3, 1, 0], -2.0)],
            _ => &[],
        };
        for (logical, value) in values {
            let physical = physical_coordinates(requirement, logical);
            let element = flat_index(&physical, requirement.physical_shape());
            let tensor_offset = offsets[requirement.checkpoint_key()];
            let byte_offset = u64::try_from(element)?.checked_mul(4).unwrap();
            file.seek(SeekFrom::Start(
                payload_start
                    .checked_add(tensor_offset)
                    .and_then(|offset| offset.checked_add(byte_offset))
                    .unwrap(),
            ))?;
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn physical_coordinates(requirement: &MimiParameterRequirement, logical: &[usize]) -> Vec<usize> {
    match requirement.recipe() {
        DerivedWeightRecipe::Source { .. } => logical.to_vec(),
        DerivedWeightRecipe::Transpose { axes, .. } => {
            let mut physical = vec![0; axes.len()];
            for (logical_axis, physical_axis) in axes.iter().copied().enumerate() {
                physical[physical_axis] = logical[logical_axis];
            }
            physical
        }
        recipe => panic!("released Mimi fixture contains unexpected recipe {recipe:?}"),
    }
}

fn flat_index(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0usize, |index, (coordinate, dimension)| {
            index * dimension + coordinate
        })
}

struct ParameterCapture<'a> {
    stream: &'a safemlx::Stream,
    direct: Option<Vec<f32>>,
    transposed: Option<Vec<f32>>,
}

impl<'value> ParameterVisitor<'value, MlxTensor> for ParameterCapture<'_> {
    fn visit(&mut self, metadata: ParameterMetadata, parameter: &'value MlxTensor) {
        let destination = match metadata.id.as_str() {
            "encoder.init_conv1d.bias" => &mut self.direct,
            "encoder.init_conv1d.weight" => &mut self.transposed,
            _ => return,
        };
        *destination = Some(parameter.to_f32_vec(self.stream).unwrap());
    }
}

#[test]
fn generic_cpu_mimi_construction_materializes_identity_and_transpose_recipes() {
    let config = Config::v0_1(Some(1));
    let requirements = released_checkpoint_requirements(&config).unwrap();
    assert_eq!(requirements.len(), 318);
    assert!(requirements.iter().any(|requirement| {
        requirement.is_active()
            && matches!(requirement.recipe(), DerivedWeightRecipe::Source { .. })
    }));
    assert!(requirements.iter().any(|requirement| {
        requirement.is_active()
            && matches!(requirement.recipe(), DerivedWeightRecipe::Transpose { .. })
    }));

    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("mimi-v0.1.safetensors");
    write_sparse_v01_checkpoint(&checkpoint, &requirements).unwrap();
    let catalog = SafetensorsMetadataCatalog::discover(&checkpoint).unwrap();
    let store =
        Arc::new(SafetensorsWeightStore::open_admitted(catalog.into_admitted_shards(), 1).unwrap());
    let prepared = prepare_source(store.clone(), config).unwrap();
    let active_parameters = prepared.bindings().len();
    let active_source_bytes = requirements
        .iter()
        .filter(|requirement| requirement.is_active())
        .map(MimiParameterRequirement::source_bytes)
        .sum::<u64>();
    let prepared_diagnostics = store.diagnostics().unwrap();
    assert_eq!(prepared_diagnostics.physical_reads, 0);
    assert_eq!(prepared_diagnostics.physical_read_bytes, 0);

    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let materialization = MlxParameterMaterializationContext::new(stream, stream);
    let mimi: Mimi<MlxTensor> =
        construct::<MlxNeuralBackend>(prepared, stream, &materialization).unwrap();
    assert_eq!(mimi.mimi_config().num_codebooks, 1);
    assert!(active_parameters > 0);
    let materialized_diagnostics = store.diagnostics().unwrap();
    // Exact admitted-range reads use two matching physical passes so a
    // concurrent payload change cannot publish a mixed lease.
    assert_eq!(
        materialized_diagnostics.physical_reads,
        u64::try_from(active_parameters).unwrap() * 2
    );
    assert_eq!(
        materialized_diagnostics.physical_read_bytes,
        active_source_bytes * 2
    );
    assert_eq!(materialized_diagnostics.payload_shard_paths.len(), 1);

    let mut capture = ParameterCapture {
        stream,
        direct: None,
        transposed: None,
    };
    mimi.visit_parameters(&mut capture);
    let direct = capture.direct.expect("identity parameter was bound");
    assert_eq!(direct[0], 3.25);
    assert_eq!(direct[63], -1.5);
    let transposed = capture.transposed.expect("transposed parameter was bound");
    assert_eq!(transposed[2 * 7 + 5], 7.0);
    assert_eq!(transposed[3 * 7 + 1], -2.0);
}

#[test]
#[ignore = "requires EREDU_MIMI_PATH with a released Mimi safetensors checkpoint and Metal"]
fn local_mimi_checkpoint_encode_decode_smoke() {
    let path = std::env::var("EREDU_MIMI_PATH")
        .expect("EREDU_MIMI_PATH must point to a Mimi safetensors checkpoint");
    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let prepared = prepare_checkpoint(path, Config::v0_1(Some(8))).unwrap();
    let materialization = MlxParameterMaterializationContext::new(stream, stream);
    let mut mimi = construct::<MlxNeuralBackend>(prepared, stream, &materialization).unwrap();
    let cfg = mimi.config();
    assert_eq!(cfg.codebooks, 8);
    assert_eq!(cfg.cardinality, 2_048);

    let codes = MlxTensor::from_array(Array::zeros::<i32>(&[1, 8, 2], stream).unwrap());
    let latent = mimi.decode_latent(&codes, stream).unwrap();
    assert_eq!(latent.shape(), &[1, 512, 2]);
    let recoded = mimi.encode_latent(&latent, stream).unwrap();
    assert_eq!(recoded.shape(), &[1, 8, 2]);
    let pcm = mimi.decode(&codes, stream).unwrap();
    assert_eq!(pcm.shape(), &[1, 1, 3840]);
    let alternate_codes = MlxTensor::from_array(Array::ones::<i32>(&[1, 8, 2], stream).unwrap());
    let alternate_pcm = mimi.decode(&alternate_codes, stream).unwrap();
    eval([pcm.as_array(), alternate_pcm.as_array()]).unwrap();
    stream.synchronize().unwrap();
    let pcm_values = pcm.as_array().evaluated().unwrap();
    let alternate_values = alternate_pcm.as_array().evaluated().unwrap();
    let difference = pcm_values
        .as_slice::<f32>()
        .iter()
        .zip(alternate_values.as_slice::<f32>())
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>();
    assert!(difference > 1e-3, "Mimi decode ignored token values");
    let encoded = mimi.encode(&pcm, stream).unwrap();
    assert_eq!(encoded.shape(), &[1, 8, 2]);

    // PyTorch Mimi oracle for x[n] = ((n mod 17) - 8) / 64. This catches
    // architecture drift that a shape-only checkpoint smoke test cannot.
    let parity_pcm = (0..7680)
        .map(|sample| ((sample % 17) as f32 - 8.0) / 64.0)
        .collect::<Vec<_>>();
    let parity_pcm = MlxTensor::from_array(
        Array::from_slice(&parity_pcm, &[1, 1, 7680])
            .copy(stream)
            .unwrap(),
    );
    let actual_codes = mimi.encode(&parity_pcm, stream).unwrap();
    let expected_codes = Array::from_slice(
        &[
            1049, 605, 1964, 1964, 74, 712, 712, 712, 1441, 1441, 1441, 1441, 1820, 1820, 1820,
            1820, 1711, 1711, 1711, 1711, 1386, 818, 818, 1418, 127, 755, 755, 127, 130, 1228,
            1228, 1115,
        ],
        &[1, 8, 4],
    )
    .copy(stream)
    .unwrap();
    assert!(
        actual_codes
            .as_array()
            .all_close(&expected_codes, 0.0, 0.0, None, stream)
            .unwrap()
            .item::<bool>(stream),
        "Mimi encode tokens differ from the released PyTorch checkpoint oracle"
    );

    mimi.reset_encode_state();
    let encoded_first = mimi
        .encode_step(
            &MlxTensor::from_array(
                pcm.as_array()
                    .try_index_device((.., .., 0..1920), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap()
        .expect("first PCM frame should encode to one Mimi frame");
    let encoded_second = mimi
        .encode_step(
            &MlxTensor::from_array(
                pcm.as_array()
                    .try_index_device((.., .., 1920..3840), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap()
        .expect("second PCM frame should encode to one Mimi frame");
    assert_eq!(encoded_first.shape(), &[1, 8]);
    assert_eq!(encoded_second.shape(), &[1, 8]);

    mimi.reset_decode_state();
    let first = mimi
        .decode_step(
            &MlxTensor::from_array(
                codes
                    .as_array()
                    .try_index_device((.., .., 0), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap();
    let second = mimi
        .decode_step(
            &MlxTensor::from_array(
                codes
                    .as_array()
                    .try_index_device((.., .., 1), stream)
                    .unwrap(),
            ),
            stream,
        )
        .unwrap();
    assert_eq!(first.shape(), &[1, 1, 1920]);
    assert_eq!(second.shape(), &[1, 1, 1920]);
    let streamed = MlxTensor::concatenate(&[first, second], 2, stream).unwrap();
    assert_eq!(streamed.shape(), pcm.shape());
}
