use super::*;

#[test]
fn selected_rank_contract_keeps_exact_order_membership_and_distinct_result_limits() {
    let sum=CommunicationOperationRequirement::tensors(CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,2,6,None).unwrap(),true).unwrap();
    let gather=CommunicationOperationRequirement::tensors(CommunicationOperation::AllGatherEven,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,2,6,None).unwrap()
            .with_output_tensor_elements(12).unwrap(),true).unwrap();
    let member=CommunicationGroupDescriptor::new(CollectiveGroupId::new(91),0,vec![0,1],Some(0),
        CommunicationGroupRequirements::new([sum,gather]).unwrap()).unwrap();
    let remote=CommunicationGroupDescriptor::new(CollectiveGroupId::new(92),1,vec![1,2],None,
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let manifest=CommunicationManifest::new(3,0,vec![member,remote],vec![]).unwrap();
    let selected=manifest.select_group_operation(CollectiveGroupId::new(91),CommunicationOperation::AllGatherEven).unwrap();
    assert_eq!(selected.order(),0);
    assert!(std::ptr::eq(selected.descriptor(),&manifest.groups()[0]));
    assert!(std::ptr::eq(selected.requirement(),&manifest.groups()[0].requirements().operations()[1]));
    let requirement=selected.requirement();
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::F32,2,Some(6),false),Ok(()));
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::F32,2,Some(12),true),Ok(()));
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::F32,2,Some(12),false),Err(CommunicationTensorContractError::Limits));
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::F32,3,Some(6),false),Err(CommunicationTensorContractError::Limits));
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::F32,2,None,true),Err(CommunicationTensorContractError::Limits));
    assert_eq!(requirement.validate_tensor_metadata(&TensorDtype::I32,2,Some(6),false),Err(CommunicationTensorContractError::Dtype));
    assert!(matches!(manifest.select_group_operation(CollectiveGroupId::new(90),CommunicationOperation::AllReduceSum),
        Err(CommunicationGroupOperationError::Unknown(_))));
    assert!(matches!(manifest.select_group_operation(CollectiveGroupId::new(92),CommunicationOperation::Barrier),
        Err(CommunicationGroupOperationError::NotMember(_))));
    assert!(matches!(manifest.select_group_operation(CollectiveGroupId::new(91),CommunicationOperation::Broadcast),
        Err(CommunicationGroupOperationError::NotSelected{..})));
}
