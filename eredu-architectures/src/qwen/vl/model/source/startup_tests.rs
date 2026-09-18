//! The request's paid metadata destination survives serial and TP startup.
use super::*;
use crate::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureStateFactory, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
};
use std::num::NonZeroU32;

struct Inspector;
impl eredu_runtime::PreparedInputInspector<WorkspaceTensor> for Inspector {
    fn identity(
        &self,
        value: &WorkspaceTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::InputTensorIdentity::new(
            eredu_core::checkpoint::TensorDtype::U32,
            value.shape().iter().map(|&n| n as usize).collect(),
        )
    }
    fn i32_values(&self, _: &WorkspaceTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        panic!("text has no grid")
    }
    fn bool_values(&self, _: &WorkspaceTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        panic!("text has no mask")
    }
}
fn fixture(
    rank: Option<usize>,
    context: &WorkspaceContext,
) -> (Model, DeviceState, PreparedModelInput<WorkspaceTensor>) {
    let args = config();
    let model = match rank {
        None => Model::new(args, context).unwrap(),
        Some(rank) => {
            let global = Model::new(args.clone(), context).unwrap();
            let description = global.parameter_description(context).unwrap();
            let topology = eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
            let rank = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
            let layout =
                crate::partitioned_execution::derive_partitioned_local_layout(&description, rank)
                    .unwrap();
            let geometry = crate::qwen::vl::local_geometry(&args, &layout).unwrap();
            Model::new_parallel(args, geometry, context).unwrap()
        }
    };
    let layout = model.state_layout(None).unwrap();
    let mut factory = eredu_runtime::working_memory::WorkspaceResidentStateFactory::new(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(4).unwrap(),
        context,
    )
    .unwrap();
    let state = factory.realize(&layout).unwrap();
    let input = PreparedModelInput::new(
        vec![
            PreparedInputPart::new(
                eredu_core::InputModality::Text,
                PreparedInputPayload::TokenIds(
                    WorkspaceTensor::full_u32(3, &[1, 5], context).unwrap(),
                ),
                [],
            )
            .unwrap(),
        ],
        |value| eredu_runtime::PreparedInputInspector::identity(&Inspector, value),
    )
    .unwrap();
    (model, state, input)
}

#[test]
fn qwen_vl_serial_and_tp_startup_preserve_paid_request_and_prepared_rotary() {
    for rank in [None, Some(0), Some(1)] {
        // Tensor construction has no checked metadata. The admitted request's
        // independent paid destination must survive the parallel adapter.
        let tensors = WorkspaceContext::new(Facts);
        let (mut model, mut state, input) = fixture(rank, &tensors);
        let admission =
            crate::media_plan::admit_qwen_vl_input(&config(), &input, &Inspector).unwrap();
        let (metadata, account) = funded();
        let input =
            PreparedCompositeInput::new_with_metadata(&input, &admission, &metadata).unwrap();
        let before = tensors.operation_count();
        let forward = match rank {
            None => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward(
                &mut model, input, &mut state, &tensors,
            ),
            Some(rank) => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward_parallel(
                &mut model, input, &mut state, &WorkspaceParallelContext::new(rank, 2).unwrap(), &tensors,
            ),
        }.unwrap();
        assert_eq!(forward.hidden.shape(), [1, 5, 32]);
        assert_eq!(forward.context.tokens.as_ref().unwrap().shape(), [1, 5]);
        assert_eq!(
            forward.context.position_delta.as_ref().unwrap().shape(),
            [1]
        );
        let rotary = forward.context.rotary.as_ref().unwrap();
        assert_eq!(rotary.0.shape(), [5, 8]);
        assert_eq!(rotary.1.shape(), [5, 8]);
        assert!(
            forward
                .context
                .metadata
                .as_ref()
                .unwrap()
                .metadata_funding()
                .unwrap()
                .same_account(&metadata.metadata_funding().unwrap())
        );
        let report = tensors.report(&[]).unwrap();
        let operations = &report.operations[before..];
        assert_eq!(
            operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::PreparedMultiAxisRotary(_)))
                .count(),
            1
        );
        assert!(
            !operations
                .iter()
                .any(|op| matches!(op.kind, WorkspaceOperationKind::MultiAxisRotary(_)))
        );
        assert_eq!(
            operations
                .iter()
                .any(|op| matches!(op.kind, WorkspaceOperationKind::Collective(_))),
            rank.is_some()
        );
        drop(metadata);
        assert!(!account.retired.load(Ordering::SeqCst));
        drop(forward);
        assert!(account.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn qwen_vl_serial_and_tp_startup_refuse_before_any_tensor_work() {
    for rank in [None, Some(0), Some(1)] {
        let tensors = WorkspaceContext::new(Facts);
        let (mut model, mut state, input) = fixture(rank, &tensors);
        let admission =
            crate::media_plan::admit_qwen_vl_input(&config(), &input, &Inspector).unwrap();
        let (metadata, account) = funded();
        let input =
            PreparedCompositeInput::new_with_metadata(&input, &admission, &metadata).unwrap();
        account
            .refuse
            .store(account.calls.load(Ordering::SeqCst), Ordering::SeqCst);
        let before = tensors.operation_count();
        let result = match rank {
            None => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward(
                &mut model, input, &mut state, &tensors,
            ),
            Some(rank) => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward_parallel(
                &mut model, input, &mut state, &WorkspaceParallelContext::new(rank, 2).unwrap(), &tensors,
            ),
        };
        let error = result.err().expect("reached first allocation refusal");
        assert_eq!(tensors.operation_count(), before);
        assert!(matches!(
            error.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        drop(metadata);
        assert!(account.retired.load(Ordering::SeqCst));
    }
}


#[test]
fn qwen_vl_dense_observation_preserves_every_collective_geometry() {
    use eredu_runtime::ParallelLayeredArchitecture;
    struct Observer(usize);
    impl eredu_runtime::ActivationObserver<WorkspaceTensor, Error> for Observer {
        fn observe(&mut self, _: &str, _: &WorkspaceTensor) -> Result<(), Error> {
            self.0 += 1;
            Ok(())
        }
    }
    for rank in [None, Some(0), Some(1)] {
        let mut traces = Vec::new();
        for observed in [false, true] {
            let context = WorkspaceContext::new(Facts);
            let (mut model, mut state, input) = fixture(rank, &context);
            let admitted = crate::media_plan::admit_qwen_vl_input(&config(), &input, &Inspector).unwrap();
            let input = PreparedCompositeInput::new(&input, &admitted).unwrap();
            let parallel = rank.map(|rank| WorkspaceParallelContext::new(rank, 2).unwrap());
            let mut forward = match &parallel {
                Some(parallel) => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward_parallel(
                    &mut model, input, &mut state, parallel, &context),
                None => <Model as CompositeArchitecture<WorkspaceBackend, DeviceState>>::begin_composite_forward(
                    &mut model, input, &mut state, &context),
            }.unwrap();
            let mut unit = model.construct_unit(1, 0, &context).unwrap();
            let start = context.operation_count();
            let mut observer = Observer(0);
            let output = match (&parallel, observed) {
                (Some(parallel), true) => model.forward_unit_parallel_observed(1, 0, &mut unit,
                    &forward.hidden, &mut state, &mut forward.context, parallel, &context, &mut observer),
                (Some(parallel), false) => model.forward_unit_parallel(1, 0, &mut unit,
                    &forward.hidden, &mut state, &mut forward.context, parallel, &context),
                (None, true) => model.forward_unit_observed(1, 0, &mut unit,
                    &forward.hidden, &mut state, &mut forward.context, &context, &mut observer),
                (None, false) => model.forward_unit(1, 0, &mut unit,
                    &forward.hidden, &mut state, &mut forward.context, &context),
            }.unwrap();
            assert_eq!(output.shape(), [1, 5, 32]);
            assert_eq!(observer.0 > 0, observed);
            let report = context.report(&[output]).unwrap();
            let collectives = report.operations[start..].iter().filter_map(|operation| {
                matches!(operation.kind, WorkspaceOperationKind::Collective(_)).then(||
                    operation.inputs.iter().map(|layout| layout.shape().to_vec()).collect::<Vec<_>>())
            }).collect::<Vec<_>>();
            if rank.is_some() {
                assert_eq!(collectives, vec![vec![vec![1, 5, 32]], vec![vec![1, 5, 32]]]);
            } else { assert!(collectives.is_empty()); }
            // Also compare the real projection and pointwise shapes, so a
            // numerically harmless flatten cannot silently change its recipe.
            let shapes = report.operations[start..].iter().map(|operation|
                (format!("{:?}", operation.kind), operation.inputs.iter().map(|l|l.shape().to_vec()).collect::<Vec<_>>(),
                    operation.outputs.iter().map(|l|l.shape().to_vec()).collect::<Vec<_>>()))
                .collect::<Vec<_>>();
            traces.push(shapes);
        }
        assert_eq!(traces[0], traces[1], "rank {rank:?}");
    }
}
