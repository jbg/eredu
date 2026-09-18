use super::*;
use crate::composite_execution::{CompositeArchitecture, CompositeMediaIngressArchitecture};
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureBoundary, PreparedInputInspector, PreparedInputPart, PreparedInputPayload,
    PreparedModelInput,
};

struct Inspector;
impl PreparedInputInspector<WorkspaceTensor> for Inspector {
    fn identity(
        &self,
        value: &WorkspaceTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::InputTensorIdentity::new(
            match value.layout().dtype() {
                WorkspaceDtype::Uint32 => eredu_core::checkpoint::TensorDtype::U32,
                WorkspaceDtype::Int32 => eredu_core::checkpoint::TensorDtype::I32,
                _ => eredu_core::checkpoint::TensorDtype::F32,
            },
            value.shape().iter().map(|&n| n as usize).collect(),
        )
    }
    fn i32_values(&self, value: &WorkspaceTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        assert_eq!(value.shape(), [1, 3]);
        Ok(vec![1, 4, 4])
    }
    fn bool_values(&self, _: &WorkspaceTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        panic!("no Boolean metadata")
    }
}
fn media_input(context: &WorkspaceContext) -> PreparedModelInput<WorkspaceTensor> {
    let text = |n| {
        PreparedInputPart::new(
            eredu_core::InputModality::Text,
            PreparedInputPayload::TokenIds(WorkspaceTensor::full_u32(1, &[1, n], context).unwrap()),
            [],
        )
        .unwrap()
    };
    let pixels = WorkspaceTensor::full_f32(0.5, &[16, 24], context).unwrap();
    let grid = WorkspaceTensor::full_i32(0, &[1, 3], context).unwrap();
    PreparedModelInput::new(
        vec![
            text(2),
            PreparedInputPart::new_with_extents(
                eredu_core::InputModality::Image,
                PreparedInputPayload::Tensor(pixels),
                [(eredu_core::InputMetadataKey::PatchGrid, grid)],
                [eredu_core::InputExtent::PatchGrid {
                    time: 1,
                    height: 4,
                    width: 4,
                }],
            )
            .unwrap(),
            text(3),
        ],
        |value| Inspector.identity(value),
    )
    .unwrap()
}

#[test]
fn qwen_vl_paid_partition_waves_match_eager_and_retained_ingress_with_reached_refusals() {
    let ordinary = WorkspaceContext::new(Facts);
    let model = Model::new(config(), &ordinary).unwrap();
    let input = media_input(&ordinary);
    let admission = crate::media_plan::admit_qwen_vl_input(&config(), &input, &Inspector).unwrap();
    let prepared = PreparedCompositeInput::new(&input, &admission).unwrap();
    for (tensor, pipeline) in [(2, 1), (1, 2), (2, 2)] {
        let expected=<Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::prepared_group_collective_waves(
            &model,0,prepared,tensor,pipeline,None).unwrap();
        let operation = |context: &WorkspaceContext| {
            <Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::prepared_group_collective_waves(
            &model,0,prepared,tensor,pipeline,Some(context))
        };
        let (context, account) = funded();
        let start = account.calls.load(Ordering::SeqCst);
        assert_eq!(operation(&context).unwrap(), expected);
        let calls = account.calls.load(Ordering::SeqCst) - start;
        assert!(calls > 0);
        drop(context);
        assert!(account.retired.load(Ordering::SeqCst));
        for cut in 0..calls {
            let (context, account) = funded();
            let start = account.calls.load(Ordering::SeqCst);
            account.refuse.store(start + cut, Ordering::SeqCst);
            let error = operation(&context).expect_err("every reached wave allocation is funded");
            assert!(matches!(
                error.into_metadata_funding_error(),
                Ok(HostMetadataFundingError::Capacity { .. })
            ));
            drop(context);
            assert!(account.retired.load(Ordering::SeqCst));
        }
    }
    let (paths_context, _) = funded();
    for (group, count) in [(0, 4), (1, 3)] {
        for index in 0..count {
            assert_eq!(<Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::unit_path(&model,group,index, Some(&paths_context)).unwrap(),
                <Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::unit_path(&model,group,index, None).unwrap());
        }
        assert_eq!(<Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::group_input_observation_path(&model,group, Some(&paths_context)).unwrap(),
            <Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::group_input_observation_path(&model,group, None).unwrap());
        assert_eq!(<Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::group_output_observation_path(&model,group, Some(&paths_context)).unwrap(),
            <Model as LayeredArchitecture<WorkspaceBackend,DeviceState>>::group_output_observation_path(&model,group, None).unwrap());
    }
    assert!(<Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::prepared_primary_ingress_collectives(
        &model,prepared,2,Some(&paths_context)).unwrap().is_none());
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 9,
        max_output_tokens: 2,
        prefill_chunk_positions: 3,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let plan = MediaPrefillPlan::prepare(&config(), input, &Inspector, geometry).unwrap();
    let (context, _) = funded();
    let waves=<Model as CompositeMediaIngressArchitecture<WorkspaceBackend,DeviceState>>::media_group_collective_waves_with_metadata(
        &model,&plan,0,2,2,&context).unwrap().unwrap();
    assert_eq!(waves.iter().map(Vec::len).collect::<Vec<_>>(), [5, 6]);
    assert_eq!(waves[0][0].shape(), [16, 16]);
    assert_eq!(waves[0][4].shape(), [4, 32]);
    for (range, expected) in [
        (0..3, vec![vec![1, 2, 32]]),
        (3..6, vec![]),
        (6..9, vec![vec![1, 3, 32]]),
    ] {
        let span = eredu_runtime::prefill::PrefillChunk {
            position: range.start,
            input: range,
            output: eredu_core::OutputDemand::StateOnly,
        };
        let actual=<Model as CompositeMediaIngressArchitecture<WorkspaceBackend,DeviceState>>::media_primary_ingress_collectives_with_metadata(
            &model,&plan,&span,2,&context).unwrap().unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|operation| operation.shape().to_vec())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(Some(actual),<Model as CompositeMediaIngressArchitecture<WorkspaceBackend,DeviceState>>::media_primary_ingress_collectives(
            &model,&plan,&span,2).unwrap());
    }
}

#[test]
fn qwen_vl_paid_partition_boundaries_keep_shapes_roles_and_refuse_each_allocation() {
    let ordinary = WorkspaceContext::new(Facts);
    let model = Model::new(config(), &ordinary).unwrap();
    let vision = vision_partition_boundary_schema(&config(), false, 2)
        .unwrap()
        .resolve(1, 4)
        .unwrap();
    let decoder = PipelineBoundarySchema::from_args(&config())
        .wire_schema()
        .unwrap()
        .resolve(1, 3)
        .unwrap();
    for (from, to, selected, sequence, continuation) in [
        (0, 0, &vision, 4, Some((16, 16))),
        (0, 1, &vision, 4, None),
        (1, 1, &decoder, 3, None),
    ] {
        let expected=<Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::partition_boundary_schema(
            &model,from,to,selected,1,sequence,&[4,3],continuation,None).unwrap();
        let operation = |context: &WorkspaceContext| {
            <Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::partition_boundary_schema(
            &model,from,to,selected,1,sequence,&[4,3],continuation,Some(context))
        };
        let (context, account) = funded();
        let start = account.calls.load(Ordering::SeqCst);
        assert_eq!(operation(&context).unwrap(), expected);
        let calls = account.calls.load(Ordering::SeqCst) - start;
        assert!(calls > 10);
        drop(context);
        assert!(account.retired.load(Ordering::SeqCst));
        for cut in 0..calls {
            let (context, account) = funded();
            let start = account.calls.load(Ordering::SeqCst);
            account.refuse.store(start + cut, Ordering::SeqCst);
            let failure = operation(&context).expect_err("every boundary allocation is funded");
            assert!(matches!(
                failure.into_metadata_funding_error(),
                Ok(HostMetadataFundingError::Capacity { .. })
            ));
            drop(context);
            assert!(account.retired.load(Ordering::SeqCst));
        }
    }
    let hidden = WorkspaceTensor::full_f32(0.5, &[1, 4, 32], &ordinary).unwrap();
    let deepstack = vec![hidden.clone(), hidden.clone()];
    let forward = ForwardContext {
        mask: None,
        tokens: None,
        parts: Vec::new(),
        rotary: None,
        position_delta: None,
        pending_media: None,
        media_span: true,
        span_assembled: false,
        vision_state: None,
        vision_initial: None,
        vision_output: None,
        deepstack,
        visual_mask: None,
        metadata: None,
    };
    let operation = |context: &WorkspaceContext| {
        <Model as CompositeArchitecture<WorkspaceBackend,DeviceState>>::partition_boundary_values(
        &model,0,1,&vision,&hidden,&forward,Some(context))
    };
    let (context, account) = funded();
    let start = account.calls.load(Ordering::SeqCst);
    let values = operation(&context).unwrap().unwrap();
    assert_eq!(
        values.iter().map(|value| value.role()).collect::<Vec<_>>(),
        ["hidden", "deepstack.0", "deepstack.1"]
    );
    assert!(
        values
            .iter()
            .all(|value| value.tensor().same_context(&hidden))
    );
    let boundary_roots = values
        .iter()
        .map(|value| value.tensor().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        ordinary.report(&boundary_roots).unwrap().retained_bytes,
        ordinary.report(&[hidden.clone()]).unwrap().retained_bytes,
        "the three logical fields retain exactly one existing tensor backing"
    );
    let calls = account.calls.load(Ordering::SeqCst) - start;
    drop(values);
    drop(context);
    assert!(account.retired.load(Ordering::SeqCst));
    for cut in 0..calls {
        let (context, account) = funded();
        let start = account.calls.load(Ordering::SeqCst);
        account.refuse.store(start + cut, Ordering::SeqCst);
        let failure =
            operation(&context).expect_err("every boundary value/role allocation is funded");
        assert!(matches!(
            failure.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        drop(context);
        assert!(account.retired.load(Ordering::SeqCst));
    }
}
