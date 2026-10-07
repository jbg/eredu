//! Dense draws share the retained categorical RNG and supplied uniforms are deterministic.

use eredu_backend_mlx::{
    backend::{random::RandomState, runtime::generation::MlxSamplingBackend as Backend},
    MlxTensor,
};
use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
use eredu_nn::Tensor;
use eredu_runtime::execution_control::{SnapshotBudget, SnapshotReservation, SnapshotValue};
use eredu_runtime::{
    DefaultSampler, DenseSamplingBackend, GumbelTruncation, PredictionDirective,
    SequentialDecisionDriver, SequentialDecisionPlan, TokenDomain,
};
use safemlx::{Array, Device, DeviceType, Stream};

fn tensor(values: &[f32], shape: &[i32]) -> MlxTensor {
    MlxTensor::from_array(Array::from_slice(values, shape))
}

fn snapshot_random(random: &RandomState) -> (RandomState, SnapshotReservation) {
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let retention = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(random.estimate_snapshot_value().unwrap()),
        )
        .unwrap();
    (random.copy_snapshot_value(&retention).unwrap(), retention)
}

#[test]
fn dense_draws_replay_and_survive_decision_driver_adoption() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let seed = RandomState::with_seed(317).unwrap();
    let plan = SequentialDecisionPlan::new(
        [PredictionDirective::Force(
            MlxTensor::from_i32_slice(&[1], &[1], &stream).unwrap(),
        )],
        false,
        false,
    )
    .unwrap();
    let mut driver = SequentialDecisionDriver::<Backend, _>::new(
        plan,
        vec![DefaultSampler],
        vec![0.0],
        Some(seed.clone()),
    )
    .unwrap();
    assert!(driver.draw_uniform(&[0], &stream).is_err());
    let first = driver.draw_uniform(&[2, 3], &stream).unwrap();
    let mut replay = seed;
    let expected = Backend::uniform_f32(&[2, 3], &mut replay, &stream).unwrap();
    assert_eq!(
        first.to_f32_vec(&stream).unwrap(),
        expected.to_f32_vec(&stream).unwrap()
    );
    let (snapshot, _retention) = snapshot_random(driver.random_state().unwrap());
    let source_key = MlxTensor::from_array(driver.random_state().unwrap().as_array().clone());
    let snapshot_key = MlxTensor::from_array(snapshot.as_array().clone());
    let backing = MlxTensor::inspect_storage(&[&source_key, &snapshot_key]).unwrap();
    assert!(backing.values.iter().all(Option::is_some));
    assert_ne!(
        backing.values[0], backing.values[1],
        "snapshot owns independent key storage"
    );
    let normal = driver.draw_standard_normal(&[7], &stream).unwrap();
    let logits = tensor(&[0.0, 1.0, -1.0], &[1, 3]);
    let selected = driver
        .sample_gumbel(&logits, GumbelTruncation::TopP(0.9), 1e-8, &stream)
        .unwrap();
    assert!(driver.decisions().is_empty());
    driver
        .resolve(0, &logits, TokenDomain::new(3), &stream)
        .unwrap();
    let (_, mut adopted) = driver.finish_into_sampling_state().unwrap();
    let after = Backend::uniform_f32(&[5], adopted.as_mut().unwrap(), &stream).unwrap();

    let mut replay = snapshot;
    let replay_normal = Backend::standard_normal_f32(&[7], &mut replay, &stream).unwrap();
    let replay_uniform = Backend::uniform_f32(&[1, 3], &mut replay, &stream).unwrap();
    let replay_selected = Backend::gumbel_max(
        &logits,
        &replay_uniform,
        GumbelTruncation::TopP(0.9),
        1e-8,
        &stream,
    )
    .unwrap();
    let replay_after = Backend::uniform_f32(&[5], &mut replay, &stream).unwrap();
    assert_eq!(
        normal.to_f32_vec(&stream).unwrap(),
        replay_normal.to_f32_vec(&stream).unwrap()
    );
    assert_eq!(
        selected.to_i32_vec(&stream).unwrap(),
        replay_selected.to_i32_vec(&stream).unwrap()
    );
    assert_eq!(
        after.to_f32_vec(&stream).unwrap(),
        replay_after.to_f32_vec(&stream).unwrap()
    );
}

#[test]
fn isolated_rng_snapshot_preserves_strided_key_words() {
    use safemlx::ops::indexing::TryIndexOp;

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let words = Array::from_slice(&[11_u32, 99, 22, 99], &[2, 2]);
    let strided = words.try_index_device((.., 0), &stream).unwrap();
    strided.evaluated().unwrap();
    assert_eq!(strided.strides(), &[2]);
    let original = RandomState::from_key(strided);
    let (mut saved, _retention) = snapshot_random(&original);
    assert_eq!(
        saved
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<u32>()
            .unwrap(),
        [11, 22]
    );
    let mut reference = RandomState::from_key(Array::from_slice(&[11_u32, 22], &[2]));
    let actual = Backend::standard_normal_f32(&[9], &mut saved, &stream).unwrap();
    let expected = Backend::standard_normal_f32(&[9], &mut reference, &stream).unwrap();
    assert_eq!(
        actual.to_f32_vec(&stream).unwrap(),
        expected.to_f32_vec(&stream).unwrap()
    );
}

#[test]
fn supplied_gumbel_draws_respect_ties_and_always_keep_one_candidate() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let logits = tensor(&[0.0, 0.0, -10.0], &[1, 3]);
    let uniforms = tensor(&[0.1, 0.9, 0.99999], &[1, 3]);
    let selected =
        Backend::gumbel_max(&logits, &uniforms, GumbelTruncation::TopK(1), 1e-8, &stream).unwrap();
    assert_eq!(selected.to_i32_vec(&stream).unwrap(), vec![1]);
    let logits = tensor(&[-3.0, -2.0, -1.0, 0.0], &[1, 4]);
    let uniforms = tensor(&[0.99999, 0.1, 0.1, 0.1], &[1, 4]);
    for (policy, expected) in [
        (GumbelTruncation::None, 0),
        (GumbelTruncation::TopP(0.0), 3),
    ] {
        let selected = Backend::gumbel_max(&logits, &uniforms, policy, 1e-8, &stream).unwrap();
        assert_eq!(selected.to_i32_vec(&stream).unwrap(), vec![expected]);
    }
}
