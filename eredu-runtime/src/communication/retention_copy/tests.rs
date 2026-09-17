use super::*;

#[test]
fn retained_declaration_copies_preserve_order_unicode_and_independent_payloads() {
    let limits=CommunicationTensorLimits::new(3,4,4097,None).unwrap();
    let tensor=CommunicationOperationRequirement::tensors(CommunicationOperation::AllReduceSum,
        [TensorDtype::I32,TensorDtype::F32,TensorDtype::Encoded("形式/q4".into())],limits,true).unwrap();
    let original=CommunicationGroupRequirements::new([
        CommunicationOperationRequirement::barrier(true),tensor]).unwrap();
    let copy=original.try_copy_for_retention().unwrap();
    assert_eq!(copy,original.clone());
    assert_ne!(copy.operations.as_ptr(),original.operations.as_ptr());
    assert_ne!(copy.operations[1].dtypes.as_ptr(),original.operations[1].dtypes.as_ptr());
    assert!(copy.retention_copy_bytes().unwrap()>size_of_val(&copy));
    let (TensorDtype::Encoded(a),TensorDtype::Encoded(b))=(&copy.operations[1].dtypes[2],&original.operations[1].dtypes[2]) else { panic!("encoded source"); };
    assert_ne!(a.as_ptr(),b.as_ptr());
    drop(original);
    assert_eq!(copy.operations()[1].dtypes(),[TensorDtype::I32,TensorDtype::F32,TensorDtype::Encoded("形式/q4".into())]);

    let requirement=CommunicationOperationRequirement::tensors(CommunicationOperation::SendReceive,
        [TensorDtype::F32],CommunicationTensorLimits::new(3,4,4097*7,None).unwrap(),true).unwrap();
    let boundary=RoleExactBoundaryContract::new("séquence/予測",[
        BoundaryRoleContract::symbolic("état/主",TensorDtype::F32,vec![
            BoundaryDimensionContract::Variable{maximum:4097},BoundaryDimensionContract::Fixed(7)]).unwrap(),
        BoundaryRoleContract::new("résidu/次",TensorDtype::F32,vec![3,5]).unwrap(),
    ]).unwrap();
    let route=CommunicationRouteDescriptor::new(CommunicationRouteId::new(93),2,1,3,requirement)
        .unwrap().with_boundary_contract(boundary).unwrap();
    let copy=route.try_copy_for_retention().unwrap();
    assert_eq!(copy,route.clone());
    let a=route.boundary.as_ref().unwrap();let b=copy.boundary.as_ref().unwrap();
    assert_ne!(a.schema.as_ptr(),b.schema.as_ptr());
    assert_ne!(a.roles.as_ptr(),b.roles.as_ptr());
    assert_ne!(a.roles[0].role.as_ptr(),b.roles[0].role.as_ptr());
    assert_ne!(a.roles[0].shape.as_ptr(),b.roles[0].shape.as_ptr());
    assert!(copy.retention_copy_bytes().unwrap()>a.schema.len()+a.roles[0].role.len());
    drop(route);
    let boundary=copy.boundary.as_ref().unwrap();
    assert_eq!(boundary.schema(),"séquence/予測");
    assert_eq!(boundary.roles()[0].role(),"état/主");
    boundary.validate_invocation(&[
        BoundaryRoleContract::new("état/主",TensorDtype::F32,vec![17,7]).unwrap(),
        BoundaryRoleContract::new("résidu/次",TensorDtype::F32,vec![3,5]).unwrap(),
    ]).unwrap();
    assert!(boundary.validate_invocation(&[
        BoundaryRoleContract::new("état/主",TensorDtype::F32,vec![4098,7]).unwrap(),
        BoundaryRoleContract::new("résidu/次",TensorDtype::F32,vec![3,5]).unwrap(),
    ]).is_err());
}
