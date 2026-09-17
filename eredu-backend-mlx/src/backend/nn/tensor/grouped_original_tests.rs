//! Component exercise supplied by the existing genuine original Source fixture.
use super::*;
use crate::MlxTensor;
use eredu_runtime::working_memory::OriginalTextControlGuard;
use safemlx::{OperationEvent, OriginalScopeObserver};

pub(crate) struct GroupedOriginalTestPlan {
    graph: safemlx::ResidentGraphLayout,
    traversal: safemlx::OperationEvalTraversalLayout,
    controls: u64,
}
impl GroupedOriginalTestPlan {
    pub(crate) fn new() -> Self {
        // Actual dense/Sum grouped producer, no bias/activation: 111 potential
        // primitives, eight scalar seeds, four completed leaves, two retained
        // validation roots and the output. Include one Synchronizer.
        let graph = OperationEvent::resident_graph_layout(111, 8, 3).unwrap();
        let base = OperationEvent::eval_record_layout(112, 2, 112).unwrap();
        let traversal =
            OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots: 3,
                arrays: 1 + 111 + 8 + 4 + 3,
                tape_entries: 112,
                input_edges: 130 + 3,
                output_slots: 112,
                streams: 2,
                captures: base.capture_slots().max(1),
            })
            .unwrap();
        let query = graph
            .control_bytes()
            .unwrap()
            .checked_add(base.query_control_bytes().unwrap())
            .unwrap()
            .checked_add(traversal.query_control_bytes().unwrap())
            .unwrap()
            .checked_add(super::super::matrix::grouped_projection_control_bytes().unwrap())
            .unwrap()
            .checked_add(std::mem::size_of::<Self>())
            .unwrap()
            .checked_add(std::mem::size_of::<[&Array; 3]>())
            .unwrap();
        let controls = validation_storage::grouped_test_control_bytes(2)
            .unwrap()
            .checked_add(u64::try_from(query).unwrap())
            .unwrap();
        Self {
            graph,
            traversal,
            controls,
        }
    }
    pub(crate) fn chunk_storage_test_control_bytes() -> u64 {
        validation_storage::grouped_test_control_bytes(0)
            .unwrap()
            .checked_add(
                u64::try_from(
                    GroupedOutputStorage {
                        calls: 1,
                        chunks: 2,
                        ..GroupedOutputStorage::default()
                    }
                    .control_bytes()
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
    }

    pub(crate) fn run_chunk_storage_test(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        values: [Array; 3],
    ) -> GroupedChunkOutputs {
        let custody = validation_storage::TokenValidationCustody::Text(controls.metadata_custody());
        let mut bank = PreparedGroupedOutputs::prepare(
            GroupedOutputStorage {
                calls: 1,
                chunks: 2,
                        ..GroupedOutputStorage::default()
            },
            custody.clone(),
        )
        .unwrap();
        // Refusal before the take leaves the original destination available.
        assert!(bank.take(3).is_none());
        let scope = TokenValidationScope::begin_prepared(PreparedTokenValidations(
            TokenValidationBatch {
                validations: Vec::new(),
                _original: Some(custody),
            },
            bank,
        ))
        .unwrap();
        let mut output = GroupedChunkOutputs::prepare(2).unwrap();
        let address = output.values.as_ptr();
        let [first, second, extra] = values;
        output.push(first).unwrap();
        output.push(second).unwrap();
        assert!(output.push(extra).is_err());
        assert_eq!(output.values.len(), 2);
        assert_eq!(output.values.capacity(), 2);
        assert_eq!(output.values.as_ptr(), address);
        assert!(GroupedChunkOutputs::prepare(2).is_err());
        let batch = scope.finish();
        assert!(batch.is_empty());
        drop(batch);
        assert!(!observer.status().has_work());
        // The original Scope and directory may retire first. This detached
        // initialized destination carries its own raw original custody.
        output
    }

    pub(crate) fn with_component_outputs<T>(
        storage: GroupedOutputStorage,
        controls: &OriginalTextControlGuard,
        construct: impl FnOnce() -> T,
    ) -> T {
        Self::with_component_validation_outputs(storage, 0, controls, construct, |result, batch| {
            assert!(batch.is_empty());
            drop(batch);
            result
        })
    }

    pub(crate) fn component_output_controls(storage: GroupedOutputStorage, validations: usize) -> u64 {
        validation_storage::grouped_test_control_bytes(validations).unwrap()
            .checked_add(storage.control_bytes().unwrap() as u64).unwrap()
            .checked_add(std::mem::size_of::<smallvec::SmallVec<[&Array; 4]>>() as u64).unwrap()
    }

    pub(crate) fn with_component_validation_outputs<T, U>(
        storage: GroupedOutputStorage,
        validation_count: usize,
        controls: &OriginalTextControlGuard,
        construct: impl FnOnce() -> T,
        complete: impl FnOnce(T, TokenValidationBatch) -> U,
    ) -> U {
        // These exact destinations enter the genuine fixture's quote before
        // admission. They use the same custody and storage as production.
        let custody = validation_storage::TokenValidationCustody::Text(controls.metadata_custody());
        let grouped = PreparedGroupedOutputs::prepare(storage, custody.clone()).unwrap();
        let mut validations = Vec::new();
        validations.try_reserve_exact(validation_count).unwrap();
        assert_eq!(validations.capacity(), validation_count);
        let scope = TokenValidationScope::begin_prepared(PreparedTokenValidations(
            TokenValidationBatch {
                validations,
                _original: Some(custody),
            },
            grouped,
        )).unwrap();
        let result = construct();
        let batch = scope.finish();
        assert_eq!(batch.validations.len(), validation_count);
        complete(result, batch)
    }

    pub(crate) fn control_bytes(&self) -> u64 {
        self.controls
    }

    pub(crate) fn run(
        self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        stream: &Stream,
        construct: impl FnOnce() -> MlxTensor,
    ) -> Result<MlxTensor, Exception> {
        // Same owning batch/type/allocation strategy as the production ingress;
        // the genuine Source callback supplies its previously admitted custody.
        // This test capacity is declared before Source admission, not inferred
        // from available Vec capacity or a native arena's spare bytes.
        let custody = validation_storage::TokenValidationCustody::Text(controls.metadata_custody());
        let mut validations = Vec::new();
        validations.try_reserve_exact(2).unwrap();
        assert_eq!(validations.capacity(), 2);
        let address = validations.as_ptr();
        let prepared = PreparedTokenValidations(
            TokenValidationBatch {
                validations,
                _original: Some(custody),
            },
            PreparedGroupedOutputs::default(),
        );
        let validation = TokenValidationScope::begin_prepared(prepared).unwrap();
        assert!(!observer.status().has_work());
        let graph = OperationEvent::prepare_resident_graph(self.graph, observer).unwrap();
        let output = construct();
        assert!(
            !observer.status().has_work(),
            "BF16 construction evaluated eagerly"
        );
        drop(graph); // the host reservation ends before any Eval worker runs
        let batch = validation.finish();
        assert_eq!(batch.validations.len(), 2);
        assert_eq!(batch.validations.capacity(), 2);
        assert_eq!(batch.validations.as_ptr(), address);
        let roots = [
            output.as_array(),
            &batch.validations[0].invalid,
            &batch.validations[1].invalid,
        ];
        let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
            roots.iter().copied(),
            observer,
            stream,
            &self.traversal,
        )
        .unwrap();
        event.synchronize().unwrap();
        assert!(roots
            .iter()
            .all(|root| root.completed_in_original_scope(observer).is_ok()));
        // Publication is after semantic validation even though the masked
        // result has completed safely. Invalid IDs never escape as a result.
        let result = match batch.validate_completed() {
            Ok(()) => Ok(output),
            Err(error) => {
                safemlx::try_with_submission_retirement(|| drop(output)).unwrap();
                Err(error)
            }
        };
        safemlx::try_with_submission_retirement(|| drop((batch, event))).unwrap();
        assert!(!observer.status().failed());
        result
    }
}
