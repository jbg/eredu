use super::*;
use crate::{DistributedNeuralBackend, LinearSpec, VocabularyParallelRange};

// This fixture executes every floating result as contiguous float32. Publishing
// that fact lets uneven gathers qualify their padding constructor exactly.
#[derive(Debug)]
struct ParallelMechanism;
impl WorkspaceMechanisms for ParallelMechanism {
    fn operation_bound(&self, operation: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>, Error> {
        AllocatingMechanism.operation_bound(operation)
    }
    fn host_workspace_bound(&self, operation: &WorkspaceOperation) -> Result<Option<WorkspaceHostBound>, Error> {
        AllocatingMechanism.host_workspace_bound(operation)
    }
    fn output_representation(&self, operation: WorkspaceOperationView<'_>, output: usize) -> Option<WorkspaceRepresentation> {
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32)
            .then_some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true))
    }
}
fn context() -> WorkspaceContext { WorkspaceContext::new(ParallelMechanism) }

fn formats() -> Vec<LinearFormatSpec> {
    vec![
        LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        LinearFormatSpec::affine(
            LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 3).unwrap()),
            ParameterSpec::trainable("weight.scales").unwrap(),
            ParameterSpec::trainable("weight.biases").unwrap(),
        )
        .unwrap(),
        LinearFormatSpec::scaled(
            LinearFormat::MxFp4,
            ParameterSpec::trainable("weight.scales").unwrap(),
        )
        .unwrap(),
    ]
}

fn embedding_spec(format: LinearFormatSpec) -> EmbeddingSpec {
    EmbeddingSpec {
        vocabulary: 35,
        dimensions: 64,
        weight: ParameterSpec::trainable("weight").unwrap(),
        format,
    }
}

#[test]
fn vocabulary_ownership_prices_local_dense_or_packed_parameters_and_exact_uneven_gather() {
    for format in formats() {
        for (rank, local) in [(0, 0..12), (1, 12..24), (2, 24..35)] {
            for tied in [false, true] {
                let context = context();
                let parallel = WorkspaceParallelContext::new(rank, 3).unwrap();
                let range = VocabularyParallelRange {
                    global_vocabulary: 35,
                    local: local.clone(),
                };
                let mut embedding = WorkspaceBackend::vocabulary_parallel_embedding(
                    embedding_spec(format.clone()),
                    range.clone(),
                    &context,
                )
                .unwrap();
                let mut head = WorkspaceBackend::vocabulary_parallel_linear(
                    LinearSpec {
                        input: 64,
                        output: 35,
                        weight: ParameterSpec::trainable("head.weight").unwrap(),
                        bias: None,
                        format: format.clone(),
                    },
                    range.clone(),
                    &context,
                )
                .unwrap();
                let ids = WorkspaceTensor::from_i32_slice(&[2, 14, 29, 34, -1], &[1, 5], &context)
                    .unwrap();
                let hidden = WorkspaceBackend::vocabulary_parallel_lookup(
                    &mut embedding,
                    &ids,
                    crate::EmbeddingLookupPolicy::ZeroSentinel(-1),
                    &parallel,
                    &context,
                )
                .unwrap();
                assert_eq!(hidden.shape(), [1, 5, 64]);
                let last = hidden
                    .index(&[Index::Full, Index::Range(4, 5)], &context)
                    .unwrap();
                let scores = if tied {
                    WorkspaceBackend::vocabulary_parallel_embedding_project(
                        &mut embedding,
                        &last,
                        &parallel,
                        &context,
                    )
                } else {
                    WorkspaceBackend::vocabulary_parallel_project(
                        &mut head, &last, &parallel, &context,
                    )
                }
                .unwrap();
                assert_eq!(scores.shape(), [1, 1, 35]);
                let report = context.report(&[scores]).unwrap();
                assert!(report.total_bytes.is_some());
                let lookup = report
                    .operations
                    .iter()
                    .find(|op| {
                        matches!(
                            op.kind,
                            WorkspaceOperationKind::VocabularyParallelLookup { .. }
                        )
                    })
                    .unwrap();
                assert!(
                    matches!(&lookup.kind, WorkspaceOperationKind::VocabularyParallelLookup { range: r, policy: crate::EmbeddingLookupPolicy::ZeroSentinel(-1), format: f } if *r == range && *f == format)
                );
                let projection = report
                    .operations
                    .iter()
                    .find(|op| matches!(op.kind, WorkspaceOperationKind::Projection(_)))
                    .unwrap();
                assert_eq!(projection.inputs[0].shape(), [1, 1, 64]);
                assert_eq!(projection.outputs[0].shape(), [1, 1, local.len() as i32]);
                let expected = match format.encoding() {
                    LinearFormat::Dense => vec![vec![local.len() as i32, 64]],
                    LinearFormat::Affine(_) => vec![
                        vec![local.len() as i32, 6],
                        vec![local.len() as i32, 2],
                        vec![local.len() as i32, 2],
                    ],
                    LinearFormat::MxFp4 => {
                        vec![vec![local.len() as i32, 8], vec![local.len() as i32, 2]]
                    }
                    _ => unreachable!(),
                };
                for op in [lookup, projection] {
                    assert_eq!(
                        op.inputs[1..]
                            .iter()
                            .map(|p| p.shape().to_vec())
                            .collect::<Vec<_>>(),
                        expected
                    );
                }
                let gather = report.operations.iter().find(|op|matches!(op.kind,
                    WorkspaceOperationKind::Collective(WorkspaceCollective::GatherFirstAxis{..}))).unwrap();
                assert!(
                    matches!(&gather.kind, WorkspaceOperationKind::Collective(WorkspaceCollective::GatherFirstAxis { axis: 2, rank: r, peer_widths }) if *r == rank && *peer_widths == [12,12,11])
                );
                assert_eq!(gather.inputs[0].shape(), [1, 1, 12]);
                assert_eq!(gather.outputs[0].shape(), [3, 1, 12]);
                assert_eq!(report.operations.last().unwrap().outputs[0].shape(), [1, 1, 35]);
                assert!(report.operations.iter().any(|op| matches!(op.kind, WorkspaceOperationKind::Collective(WorkspaceCollective::Sum { partitions: 3, rank: r }) if r == rank)));
            }
        }
    }
}

#[test]
fn ownership_and_rank_mismatch_fail_before_tracing_projection_or_collective() {
    for args in [(0, 0), (1, 1), (3, 2)] {
        assert!(WorkspaceParallelContext::new(args.0, args.1).is_err());
    }
    let context = context();
    let input = existing_f32(&[2, 1, 64], &context).unwrap();
    let ids = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Int32).unwrap(),
        &context,
    )
    .unwrap();
    let range = VocabularyParallelRange {
        global_vocabulary: 35,
        local: 12..24,
    };
    let mut embedding = WorkspaceBackend::vocabulary_parallel_embedding(
        embedding_spec(LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()),
        range.clone(),
        &context,
    )
    .unwrap();
    let before = context.report(&[]).unwrap().operations.len();
    for (rank, size) in [(0, 3), (1, 2), (0, usize::MAX)] {
        let parallel = WorkspaceParallelContext::new(rank, size).unwrap();
        assert!(WorkspaceBackend::vocabulary_parallel_embedding_project(
            &mut embedding,
            &input,
            &parallel,
            &context
        )
        .is_err());
        assert!(WorkspaceBackend::vocabulary_parallel_lookup(
            &mut embedding,
            &ids,
            crate::EmbeddingLookupPolicy::Strict,
            &parallel,
            &context
        )
        .is_err());
        assert_eq!(context.report(&[]).unwrap().operations.len(), before);
    }
    let mut wrong_global = embedding_spec(LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap());
    wrong_global.vocabulary = 34;
    assert!(
        WorkspaceBackend::vocabulary_parallel_embedding(wrong_global, range, &context).is_err()
    );
    let mut unpartitioned = WorkspaceBackend::embedding(
        embedding_spec(LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()),
        &context,
    )
    .unwrap();
    let before = context.report(&[]).unwrap().operations.len();
    assert!(WorkspaceBackend::vocabulary_parallel_embedding_project(
        &mut unpartitioned,
        &input,
        &WorkspaceParallelContext::new(0, 1).unwrap(),
        &context
    )
    .is_err());
    assert_eq!(context.report(&[]).unwrap().operations.len(), before);
}

#[test]
fn absent_collective_facts_preserve_unknown_even_when_local_projection_is_priced() {
    #[derive(Debug)]
    struct MissingTransport;
    impl WorkspaceMechanisms for MissingTransport {
        fn output_representation(&self, operation: WorkspaceOperationView<'_>, output: usize) -> Option<WorkspaceRepresentation> {
            ParallelMechanism.output_representation(operation, output)
        }

        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if matches!(operation.kind, WorkspaceOperationKind::Collective(_)) {
                Ok(None)
            } else {
                AllocatingMechanism.operation_bound(operation)
            }
        }
    }
    let context = WorkspaceContext::new(MissingTransport);
    let mut embedding = WorkspaceBackend::vocabulary_parallel_embedding(
        embedding_spec(LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()),
        VocabularyParallelRange {
            global_vocabulary: 35,
            local: 24..35,
        },
        &context,
    )
    .unwrap();
    let input = existing_f32(&[2, 1, 64], &context).unwrap();
    context.begin_span();
    let output = WorkspaceBackend::vocabulary_parallel_embedding_project(
        &mut embedding,
        &input,
        &WorkspaceParallelContext::new(2, 3).unwrap(),
        &context,
    )
    .unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.total_bytes, None);
    assert_eq!(report.transient_bytes, None);
    assert_eq!(report.retained_bytes, Some(2 * 35 * 4));
    assert_eq!(report.unpriced_operations.len(), 1);
    assert!(matches!(report.operations[report.unpriced_operations[0]].kind, WorkspaceOperationKind::Collective(_)));
    assert!(matches!(
        report.operations[0].kind,
        WorkspaceOperationKind::Projection(_)
    ));
    assert!(report.operations.iter().any(|operation| matches!(operation.kind,
        WorkspaceOperationKind::Elementwise("zeros_f32"))));
}

#[test]
fn observed_parallel_readout_reports_the_selected_input_and_preserves_collective_geometry() {
    struct Observer(Vec<Vec<i32>>);
    impl crate::ProjectionInputObserver<WorkspaceTensor> for Observer {
        fn observe(&mut self, input: &WorkspaceTensor) -> Result<(), Error> {
            self.0.push(input.shape().to_vec());
            Ok(())
        }
        fn observe_generated(
            &mut self,
            _: &WorkspaceTensor,
            _: &crate::GeneratedTensorSource,
            generate: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
        ) -> Result<(), Error> {
            self.observe(&generate()?)
        }
    }
    let context = context();
    let mut embedding = WorkspaceBackend::vocabulary_parallel_embedding(
        embedding_spec(LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()),
        VocabularyParallelRange {
            global_vocabulary: 35,
            local: 0..18,
        },
        &context,
    )
    .unwrap();
    let mut observer = Observer(Vec::new());
    let input = existing_f32(&[2, 1, 64], &context).unwrap();
    context.begin_span();
    let output = WorkspaceBackend::vocabulary_parallel_embedding_project_with_input_observer(
        &mut embedding,
        &input,
        &WorkspaceParallelContext::new(0, 2).unwrap(),
        &context,
        Some(&mut observer),
    )
    .unwrap();
    assert_eq!(observer.0, [vec![2, 1, 64]]);
    assert_eq!(output.shape(), [2, 1, 35]);
    let before = context.report(&[]).unwrap();
    assert_eq!(before.operations.iter().filter(|operation| matches!(operation.kind, WorkspaceOperationKind::Projection(_))).count(), 1);
    assert_eq!(before.operations.iter().filter(|operation| matches!(operation.kind, WorkspaceOperationKind::Collective(_))).count(), 1);
    assert!(
        WorkspaceBackend::vocabulary_parallel_embedding_project_with_input_observer(
            &mut embedding,
            &input,
            &WorkspaceParallelContext::new(1, 2).unwrap(),
            &context,
            Some(&mut observer),
        )
        .is_err()
    );
    assert_eq!(observer.0.len(), 1);
    assert_eq!(context.report(&[]).unwrap().operations.len(), before.operations.len());
}
