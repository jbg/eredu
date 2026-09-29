//! Cold rank selection preserves source authority, stream geometry and row collectives.
use super::*;
use eredu_architectures::qwen4_exp::prepared::{
    GgufTargetPlan, SafetensorsTargetPlan, TargetPartitionExecutionPlan,
};
use eredu_runtime::*;

pub(super) fn request(tp: usize, pp: usize, ep: usize, rank: usize) -> NormalizedLoadRequest {
    let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
    load_policy::request(LayerWeightResidency::FullyResident)
        .with_media_execution(MediaLoadRequest::Disabled)
        .with_drafting(DraftingLoadRequest::Disabled)
        .with_parallel_execution(
            ParallelLoadRequest::new(
                ParallelRankTopology::new(topology, rank).unwrap(),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
                2,
                32,
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(5),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
}

fn check(
    plan: TargetPartitionExecutionPlan,
    source: &SharedCheckpointSource,
    request: &NormalizedLoadRequest,
) {
    let before = source.source_diagnostics().unwrap().physical_reads;
    let rank = request.parallel_topology().unwrap();
    assert_eq!(plan.requirements().topology(), rank);
    assert_eq!(
        (
            plan.tensor_partition().rank(),
            plan.tensor_partition().ranks()
        ),
        (rank.tensor_parallel_rank(), rank.tensor_parallel_size())
    );
    let selected_state = plan.requirements().state().unwrap();
    assert_eq!(plan.state_requirements().layout(), selected_state.layout());
    for stream in plan.state_requirements().append_streams() {
        assert!(stream.layer < selected_state.layout().len());
    }
    assert!(plan
        .row_requirements()
        .iter()
        .all(|row| selected_state.global_layers().contains(&row.spec().unit)));
    assert_eq!(plan.table_owner(), 0);
    assert_eq!(
        plan.requirements().tensor_group().is_some(),
        rank.tensor_parallel_size() > 1 || rank.expert_parallel_size() > 1,
        "EP rows retain singleton data authority; pure PP keeps ordinary local lookup",
    );
    if let Some(id) = plan.requirements().tensor_group() {
        let group = plan
            .requirements()
            .communication()
            .groups()
            .iter()
            .find(|group| group.id() == id)
            .unwrap();
        assert_eq!(group.members().len(), rank.tensor_parallel_size());
        let sum = group
            .requirements()
            .operations()
            .iter()
            .find(|op| op.operation() == CommunicationOperation::AllReduceSum)
            .unwrap();
        assert!(sum
            .dtypes()
            .contains(&eredu_core::checkpoint::TensorDtype::I32));
        for row in plan.row_requirements() {
            assert!(
                sum.limits().unwrap().max_tensor_elements()
                    >= row.limits().requests * row.spec().dimensions as usize
            );
        }
    }
    let selection = plan.load_selection_request().unwrap().clone();
    let facts = cold::capabilities(plan.requirements().execution(), None);
    assert!(
        plan.clone()
            .select(
                &selection,
                &facts,
                None,
                &CommunicationCapabilities::new([]).unwrap()
            )
            .is_err(),
        "missing collective mechanisms reject during cold admission"
    );
    let selected = plan
        .select(&selection, &facts, None, &numeric_partition_capabilities())
        .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        before,
        "cold rank selection reads no payload"
    );
    let bound = selected.bind(source.clone(), None).unwrap();
    assert_eq!(bound.selected().requirements().topology(), rank);
    assert_eq!(
        bound.tensor_target().partition().rank(),
        rank.tensor_parallel_rank()
    );
    assert_eq!(
        bound
            .tensor_target()
            .bound_spec()
            .geometry()
            .state_layout()
            .unwrap(),
        *bound.selected().base().text().state().layout()
    );
    assert!(
        source.source_diagnostics().unwrap().physical_reads <= before + 3,
        "source binding reads only literal controls, never ordinary/expert/table payloads"
    );
}

#[test]
fn qwen4_cold_partition_safetensors_tp2_tp4_pp2_and_combined_retains_exact_sources() {
    let directory = tempfile::tempdir().unwrap();
    let target = prepared_tensor_parallel::small_safetensors(directory.path(), false);
    let source = target.artifact();
    let header = SafetensorsTargetPlan::prepare(
        source.as_ref(),
        target.spec().configuration().clone(),
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
    )
    .unwrap();
    let physical: BTreeMap<_, _> = header
        .resolution()
        .source_keys()
        .iter()
        .map(|key| {
            let provenance = source.source_provenance(key).unwrap();
            let metadata = source.source_metadata(key).unwrap();
            (
                key.clone(),
                ReplicatedTextPhysicalSource::new(
                    provenance.catalog_key,
                    provenance.physical_tensor,
                    provenance.backing_shard.unwrap(),
                    provenance.output,
                    provenance.source_encoding,
                    metadata.encoded_byte_len,
                )
                .unwrap(),
            )
        })
        .collect();
    for (tp, pp, ep) in [
        (2, 1, 1),
        (4, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (1, 5, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ] {
        for rank in 0..tp * pp * ep {
            let request = request(tp, pp, ep, rank);
            let before = source.source_diagnostics().unwrap().physical_reads;
            let plan = header
                .partition_execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    physical.clone(),
                )
                .unwrap();
            assert_eq!(source.source_diagnostics().unwrap().physical_reads, before);
            check(plan, source, &request);
        }
    }
    let base = request(2, 1, 1, 0);
    let embedded = base
        .clone()
        .with_drafting(DraftingLoadRequest::embedded(2).unwrap());
    assert!(header
        .partition_execution_plan_for_load(
            &embedded,
            eredu_nn::TensorElementType::F32,
            &cold::Support,
            physical.clone()
        )
        .is_err());
    for invalid in [request(1, 6, 1, 0), request(1, 1, 4, 0)] {
        assert!(
            header
                .partition_execution_plan_for_load(
                    &invalid,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    physical.clone(),
                )
                .is_err(),
            "empty pipeline or expert ownership must reject during cold admission"
        );
    }
}

#[test]
fn qwen4_cold_partition_gguf_keeps_published_mapping_and_stage_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    eredu_evaluation::qwen4_exp::Fixture::new().write(&path);
    let header = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(&path).unwrap()).unwrap();
    let source = super::super::qwen4_gguf::open_text_source(header.text_plan());
    for (tp, pp, ep) in [
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ] {
        for rank in 0..tp * pp * ep {
            let request = request(tp, pp, ep, rank);
            let plan = header
                .partition_execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                )
                .unwrap();
            check(plan, &source, &request);
        }
    }
}
