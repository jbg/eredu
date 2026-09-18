use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::{ArchitectureParameters, LayeredArchitecture};
use std::{
    convert::Infallible,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

type Model = LayeredModel<WorkspaceBackend>;
type DeviceState = eredu_runtime::DeviceState<
    WorkspaceBackend,
    eredu_runtime::working_memory::WorkspaceResidentLayerState,
>;
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        // This neutral fixture selects F32 storage for floating equations.
        // Views preserve that dtype without claiming contiguous backing.
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32).then_some(
            WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,
                !matches!(operation.kind, WorkspaceOperationKindView::View(_)
                    | WorkspaceOperationKindView::Transpose(_)
                    | WorkspaceOperationKindView::Index { .. }
                    | WorkspaceOperationKindView::StaticSlice { .. })),
        )
    }
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("absent facts")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("absent host facts")
    }
}
fn config() -> ModelArgs {
    crate::qwen::vl::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62, "tie_word_embeddings":true,
        "text_config":{"model_type":"qwen3_vl_text", "hidden_size":32, "num_hidden_layers":3,
            "intermediate_size":64, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":8, "rms_norm_eps":0.000001, "vocab_size":64,
            "max_position_embeddings":128, "rope_theta":1000000.0,
            "rope_scaling":{"mrope_section":[2,1,1], "mrope_interleaved":true}},
        "vision_config":{"depth":4, "hidden_size":16, "intermediate_size":24, "num_heads":4,
            "num_position_embeddings":16, "in_channels":3, "patch_size":2,
            "spatial_merge_size":2, "temporal_patch_size":2, "out_hidden_size":32,
            "deepstack_visual_indexes":[1,3]}
    })).unwrap()
}
fn partition(rank: usize, context: &WorkspaceContext) -> Model {
    let args = config();
    let global = Model::new(args.clone(), context).unwrap();
    let description = global.parameter_description(context).unwrap();
    let topology = eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap();
    let rank = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
    let layout =
        crate::partitioned_execution::derive_partitioned_local_layout(&description, rank).unwrap();
    let local = crate::qwen::vl::local_geometry(&args, &layout).unwrap();
    let first = rank.pipeline_parallel_rank() == 0;
    let (groups, roles) = if first {
        (
            vec![(VISION_EXECUTION_GROUP, 0..4), (TEXT_EXECUTION_GROUP, 0..1)],
            vec!["vision", "embedding"],
        )
    } else {
        (vec![(TEXT_EXECUTION_GROUP, 1..3)], vec!["norm", "output"])
    };
    let ownership = eredu_runtime::PartitionOwnership::new(first, !first, roles).unwrap();
    let partition =
        crate::qwen::vl::partition_local_geometry(&args, &layout, groups, &ownership).unwrap();
    Model::new_parallel(args, local, context)
        .unwrap()
        .with_partition_geometry(partition)
}

#[test]
fn qwen_vl_completed_partition_source_keeps_graph_geometry_and_identity() {
    let context = WorkspaceContext::new(Facts);
    for rank in 0..4 {
        let mut initial = partition(rank, &context);
        let expected = initial.parameter_description(&context).unwrap().into_owned();
        let source = initial.prepare_source(&context).unwrap();
        let cold = Model::new_with_source(source.clone(), &context).unwrap();
        assert!(std::ptr::eq(&*initial.args, &*cold.args));
        assert!(same_owner(
            initial.parallel_geometry.as_ref(),
            cold.parallel_geometry.as_ref()
        ));
        assert!(same_owner(
            initial.partition_geometry.as_ref(),
            cold.partition_geometry.as_ref()
        ));
        let description = cold.parameter_description(&context).unwrap();
        assert!(matches!(description, std::borrow::Cow::Borrowed(_)));
        assert_eq!(description.graph(), expected.graph());
        assert_eq!(description.groups(), expected.groups());
        assert_eq!(
            cold.state_layout(Some(&context)).unwrap(),
            initial.state_layout(None).unwrap()
        );
        for group in 0..2 {
            let transport =
                <Model as LayeredArchitecture<WorkspaceBackend, DeviceState>>::group_transport(
                    &initial, group,
                );
            assert!(<Model as LayeredArchitecture<
                WorkspaceBackend,
                DeviceState,
            >>::group_transport_matches(
                &cold, group, &transport
            ));
        }
        drop(description);
        drop(initial);
        assert_eq!(cold.args.text.hidden_size, 32);
        let mut replaced = Model::new_with_source(source.clone(), &context).unwrap();
        replaced.args = SharedCompositeConfig::new(config(), None).unwrap();
        assert!(
            replaced
                .parameter_description(&context)
                .is_err()
        );
        let mut replaced = Model::new_with_source(source.clone(), &context).unwrap();
        replaced.parallel_geometry = Some(Arc::new(
            (**replaced.parallel_geometry.as_ref().unwrap()).clone(),
        ));
        assert!(replaced.state_layout(Some(&context)).is_err());
        let mut replaced = Model::new_with_source(source, &context).unwrap();
        replaced.partition_geometry = Some(Arc::new(
            (**replaced.partition_geometry.as_ref().unwrap()).clone(),
        ));
        assert!(
            replaced
                .parameter_description(&context)
                .is_err()
        );
    }
}

#[derive(Debug)]
struct FundingState {
    calls: AtomicUsize,
    refuse: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<FundingState>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let index = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if index == self.0.refuse.load(Ordering::SeqCst) {
            Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn funded() -> (WorkspaceContext, Arc<FundingState>) {
    let state = Arc::new(FundingState {
        calls: AtomicUsize::new(0),
        refuse: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap(),
        state,
    )
}
#[test]
fn qwen_vl_completed_partition_source_funds_cold_constructor_and_reached_refusals() {
    let initial_context = WorkspaceContext::new(Facts);
    let mut initial = partition(0, &initial_context);
    let expected = initial.parameter_description(&initial_context).unwrap().into_owned();
    let source = initial.prepare_source(&initial_context).unwrap();
    drop((initial, initial_context));
    let (context, account) = funded();
    let start = account.calls.load(Ordering::SeqCst);
    let cold = Model::new_with_source(source.clone(), &context).unwrap();
    let constructor_calls = account.calls.load(Ordering::SeqCst) - start;
    let description = cold.parameter_description(&context).unwrap();
    assert!(matches!(&description, std::borrow::Cow::Borrowed(_)));
    assert_eq!(description.groups(), expected.groups());
    assert_eq!(description.graph(), expected.graph());
    assert!(context.metadata_census().unwrap().context_bytes() > 0);
    drop(description);
    // The same cold result owner retains its real metadata account and tensors.
    let retained = (cold, context.metadata_funding().unwrap());
    drop(context);
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(account.retired.load(Ordering::SeqCst));
    assert!(constructor_calls > 10);
    for cut in [0, constructor_calls / 2, constructor_calls - 1] {
        let (context, account) = funded();
        let start = account.calls.load(Ordering::SeqCst);
        account.refuse.store(start + cut, Ordering::SeqCst);
        let failure = Model::new_with_source(source.clone(), &context)
            .err()
            .expect("reached refusal");
        assert!(matches!(
            failure.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        assert!(account.calls.load(Ordering::SeqCst) > start + cut);
        drop(context);
        assert!(account.retired.load(Ordering::SeqCst));
    }
}

#[path = "../partition_metadata/tests.rs"]
mod partition_controls_tests;

#[path = "startup_tests.rs"]
mod startup_tests;
