use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    DistributedNeuralBackend, EmbeddingLookupPolicy, EmbeddingSpec, LinearFormatSpec, LinearSpec,
    ParameterMetadata, ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
    VocabularyParallelRange,
};

pub(super) fn local_setup_failures(native: &distributed::Group, stream: &Stream) {
    let rank = native.rank();
    let id = CollectiveGroupId::new(92);
    let tensor = |operation, dtypes| {
        CommunicationOperationRequirement::tensors(
            operation,
            dtypes,
            CommunicationTensorLimits::new(1, 2, 16, None).unwrap(),
            true,
        )
        .unwrap()
    };
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![CommunicationGroupDescriptor::new(
            id,
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
                tensor(CommunicationOperation::Broadcast, vec![TensorDtype::I32]),
                tensor(CommunicationOperation::AllReduceSum, vec![TensorDtype::F32]),
                tensor(
                    CommunicationOperation::AllGatherUneven,
                    vec![TensorDtype::F32],
                ),
            ])
            .unwrap(),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let coordinator = super::super::coordinator::MlxReplicatedCoordinator::from_manifest(
        manifest, id, 1, native, stream,
    )
    .unwrap();
    let group = coordinator.parallel_group();
    let range = VocabularyParallelRange {
        global_vocabulary: 4,
        local: rank * 2..rank * 2 + 2,
    };
    let format = LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
    let mut embedding = MlxNeuralBackend::vocabulary_parallel_embedding(
        EmbeddingSpec {
            vocabulary: 4,
            dimensions: 2,
            weight: ParameterSpec::trainable("lookup").unwrap(),
            format: format.clone(),
        },
        range.clone(),
        stream,
    )
    .unwrap();
    let mut linear = MlxNeuralBackend::vocabulary_parallel_linear(
        LinearSpec {
            input: 2,
            output: 4,
            weight: ParameterSpec::trainable("projection").unwrap(),
            bias: None,
            format,
        },
        range,
        stream,
    )
    .unwrap();
    struct Bind(MlxTensor);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind {
        fn visit_mut(&mut self, _: ParameterMetadata, value: &'a mut MlxTensor) {
            *value = self.0.clone();
        }
    }
    let local_weights: Vec<_> = (rank * 4..rank * 4 + 4).map(|x| x as f32 + 1.).collect();
    let mut binding = Bind(MlxTensor::from_array(Array::from_slice(
        &local_weights,
        &[2, 2],
    )));
    embedding.visit_parameters_mut(&mut binding);
    linear.visit_parameters_mut(&mut binding);
    let token = MlxTensor::from_array(Array::from_slice(
        &[if rank == 0 { 4_i32 } else { 2 }],
        &[1],
    ));
    assert!(
        MlxNeuralBackend::vocabulary_parallel_lookup(
            &mut embedding,
            &token,
            EmbeddingLookupPolicy::Strict,
            group,
            stream,
        )
        .is_err(),
        "one invalid token must reject every rank before all-reduce"
    );
    let token = MlxTensor::from_array(Array::from_slice(&[2_i32], &[1]));
    let value = MlxNeuralBackend::vocabulary_parallel_lookup(
        &mut embedding,
        &token,
        EmbeddingLookupPolicy::Strict,
        group,
        stream,
    )
    .unwrap();
    assert_eq!(value.to_f32_vec(stream).unwrap(), vec![5., 6.]);
    let bad = MlxTensor::from_array(Array::from_slice(&[1f32, 1., 1.], &[1, 3]));
    let good = MlxTensor::from_array(Array::from_slice(&[1f32, 1.], &[1, 2]));
    assert!(
        MlxNeuralBackend::vocabulary_parallel_project(
            &mut linear,
            if rank == 0 { &bad } else { &good },
            group,
            stream,
        )
        .is_err(),
        "one invalid affine input must reject every rank before gather"
    );
    let value =
        MlxNeuralBackend::vocabulary_parallel_project(&mut linear, &good, group, stream).unwrap();
    assert_eq!(value.to_f32_vec(stream).unwrap(), vec![3., 7., 11., 15.]);
    assert!(
        MlxNeuralBackend::vocabulary_parallel_embedding_project(
            &mut embedding,
            if rank == 0 { &bad } else { &good },
            group,
            stream,
        )
        .is_err(),
        "one invalid tied affine input must reject every rank before gather"
    );
    let value = MlxNeuralBackend::vocabulary_parallel_embedding_project(
        &mut embedding,
        &good,
        group,
        stream,
    )
    .unwrap();
    assert_eq!(value.to_f32_vec(stream).unwrap(), vec![3., 7., 11., 15.]);
}
