use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::{ArchitectureParameters, LayeredArchitecture};

#[derive(Debug,Clone)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, operation: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>,Error> {
        // Constructing declared parameters creates real scalar placeholder
        // operations in the workspace backend. The logical weight shape is
        // not allocated by this initialization.
        assert!(matches!(operation.kind,WorkspaceOperationKind::ParameterPlaceholder));
        assert!(operation.inputs.is_empty());
        assert_eq!(operation.outputs.len(),1);
        assert!(operation.outputs[0].shape().is_empty());
        Ok(Some(WorkspaceOperationBound {
            outputs:vec![WorkspaceOutputStorage::Allocate(operation.outputs[0].bytes()?)],
            scratch_bytes:0,
            assumptions:"fixture initializes one scalar in its declared dtype".into(),
        }))
    }
    fn host_workspace_bound(&self, operation:&WorkspaceOperation)->Result<Option<WorkspaceHostBound>,Error>{
        assert!(matches!(operation.kind,WorkspaceOperationKind::ParameterPlaceholder));
        Ok(Some(WorkspaceHostBound{bytes:0,assumptions:"fixture scalar initialization has no separate staging".into()}))
    }
}

fn config() -> FamilyConfig {
    FamilyConfig::from_hf_json(br#"{
      "model_type":"gemma4", "tie_word_embeddings":false,
      "text_config":{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":2,
        "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":4,"rms_norm_eps":0.00001,"vocab_size":16,
        "max_position_embeddings":64,"layer_types":["full_attention","full_attention"]}
    }"#).unwrap()
}

#[test]
fn gemma_completed_source_borrows_exact_graph_and_refuses_equal_replacement_config() {
    type Model=LayeredModel<WorkspaceBackend>;
    type State=eredu_runtime::DeviceState<WorkspaceBackend,eredu_runtime::working_memory::WorkspaceResidentLayerState>;
    let context=WorkspaceContext::new(Facts);
    let mut initial=Model::new(config(),&context).unwrap();
    let expected=initial.parameter_description(&context).unwrap();
    let state=initial.state_layout().unwrap();
    let source=initial.prepare_source(&context).unwrap();
    let cold=Model::new_with_source(source.clone(),&context).unwrap();
    assert!(std::ptr::eq(&*initial.args,&*cold.args));
    let description=cold.parameter_description_with_metadata(&context).unwrap();
    assert!(matches!(&description,std::borrow::Cow::Borrowed(_)));
    assert_eq!(description.graph(),expected.graph());
    assert_eq!(description.groups(),expected.groups());
    assert_eq!(cold.checked_graph(Metadata::new(None)).unwrap().state,state);
    for group in 0..3 {
        let transport=<Model as LayeredArchitecture<WorkspaceBackend,State>>::group_transport(&initial,group);
        assert!(<Model as LayeredArchitecture<WorkspaceBackend,State>>::group_transport_matches(&cold,group,&transport));
    }
    drop(description);
    drop(initial);
    assert_eq!(cold.args.text.hidden_size,8);
    let mut replaced=Model::new_with_source(source,&context).unwrap();
    replaced.args=SharedCompositeConfig::new(config(),None).unwrap();
    assert!(replaced.checked_graph(Metadata::new(None)).is_err());
    assert!(replaced.parameter_description_with_metadata(&context).is_err());
}
