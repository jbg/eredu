use super::*;
use eredu_nn::{Parameterized, ParameterSpec, ParameterVisitor, ParameterVisitorMut, ParameterMetadataView};
use eredu_nn::workspace::*;

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(&self, _: &WorkspaceOperation) -> std::result::Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("source projection does not execute tensor operations")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(&self, _: WorkspaceOperationView<'_>) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> { unreachable!() }
    fn write_operation_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceEffectDestination<'_>) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> { unreachable!() }
    fn host_facts(&self, _: WorkspaceOperationView<'_>) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> { unreachable!() }
    fn write_host_facts(&self, _: WorkspaceOperationView<'_>, _: WorkspaceHostDestination<'_>) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> { unreachable!() }
}
struct Module { specs: Vec<ParameterSpec>, values: Vec<WorkspaceTensor> }
impl Parameterized<WorkspaceTensor> for Module {
    fn visit_parameters<'a,V:ParameterVisitor<'a,WorkspaceTensor>>(&'a self, visitor:&mut V) {
        for (spec,value) in self.specs.iter().zip(&self.values) {
            visitor.visit_borrowed(ParameterMetadataView::from_spec(spec,true),value);
        }
    }
    fn visit_parameters_mut<'a,V:ParameterVisitorMut<'a,WorkspaceTensor>>(&'a mut self, visitor:&mut V) {
        for (spec,value) in self.specs.iter().zip(&mut self.values) {
            visitor.visit_mut_borrowed(ParameterMetadataView::from_spec(spec,true),value);
        }
    }
    fn set_trainable(&mut self,_:bool) {}
}
fn module(source:&Source,unit:usize,context:&WorkspaceContext)->Module {
    Module {
        specs:source.units[unit].1.iter().map(|row|ParameterSpec::trainable(row.binding.name()).unwrap()).collect(),
        values:source.units[unit].1.iter().map(|row|WorkspaceTensor::existing(context.layout(&row.shape,row.dtype).unwrap(),context).unwrap()).collect(),
    }
}
fn retained(context:&WorkspaceContext,values:&[WorkspaceTensor])->u64 {
    context.begin_state_span(values.iter()).unwrap();
    context.report_scalars(values).unwrap().state.unwrap().retained_bytes.unwrap()
}

#[test]
fn paid_projection_preserves_actual_float_types_and_alias_lifetimes_at_global_addresses() {
    let mut source=source();
    let f16=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,false));
    let bf16=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Bfloat16,false));
    source.units[0].1[0].representation=f16;
    source.units[0].1[1].representation=bf16;
    source.units[1].1[0].representation=f16;
    source.addresses=Some(vec![source.layout.address(0).unwrap().with_index(7),source.layout.address(1).unwrap().with_index(8),source.layout.address(2).unwrap().with_index(12)]);
    let context=WorkspaceContext::new_recording_facts(Facts);
    let mut projected=WorkspaceParameterSourceLoan::new(&source).prepare_projection(&context).unwrap();
    let mut first=module(&source,0,&context);
    projected.bind(&mut first,0,source.execution_address(0).unwrap(),&context).unwrap();
    assert_eq!(first.values[0].layout().representation(),f16);
    assert_eq!(first.values[1].layout().representation(),bf16);
    let mut alias=module(&source,1,&context);
    projected.bind(&mut alias,1,source.execution_address(1).unwrap(),&context).unwrap();
    let mut repeated=module(&source,0,&context);
    projected.bind(&mut repeated,0,source.execution_address(0).unwrap(),&context).unwrap();
    let values=first.values.into_iter().chain(alias.values).chain(repeated.values).collect::<Vec<_>>();
    // Three live physical prototypes: one shared trace owner, two independent
    // invocation copies. Equal names/capacities never collapse the latter.
    assert_eq!(retained(&context,&values),12);
    assert!(context.metadata_census().unwrap().context_bytes()>0);
    drop(projected);
    assert_eq!(retained(&context,&values),12,"escaped values retain their paid roots");
}

#[test]
fn projection_refuses_foreign_context_and_local_address_before_replacing_slots() {
    let mut source=source();
    source.addresses=Some(vec![source.layout.address(0).unwrap().with_index(7),source.layout.address(1).unwrap().with_index(8),source.layout.address(2).unwrap().with_index(12)]);
    let context=WorkspaceContext::new_recording_facts(Facts);
    let other=WorkspaceContext::new_recording_facts(Facts);
    let mut projection=WorkspaceParameterSourceLoan::new(&source).prepare_projection(&context).unwrap();
    let mut unit=module(&source,0,&context);
    let before=unit.values.iter().cloned().collect::<Vec<_>>();
    assert!(projection.bind(&mut unit,0,source.layout.address(0).unwrap(),&context).is_err());
    assert!(projection.bind(&mut unit,0,source.execution_address(0).unwrap(),&other).is_err());
    let values=before.into_iter().chain(unit.values).collect::<Vec<_>>();
    assert_eq!(retained(&context,&values),16,"both unmodified slots keep original eight-byte backing");
    drop(projection);
    source.addresses.as_mut().unwrap()[1]=source.layout.address(0).unwrap().with_index(7);
    assert!(WorkspaceParameterSourceLoan::new(&source).count().is_err());
}
