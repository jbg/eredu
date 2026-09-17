use super::*;
use crate::parameter::{bind_parameter_values, ParameterOrchestrationError};
use eredu_nn::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceLayout,
        WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    },
    ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut,
};
use std::{cell::Cell, error::Error as _};

#[derive(Debug)]
struct NoOperations;
impl WorkspaceMechanisms for NoOperations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("parameter binding must not execute even a metadata operation")
    }
}

struct Slots {
    values: Vec<WorkspaceTensor>,
    immutable: Vec<ParameterId>,
    mutable: Vec<Option<ParameterId>>,
}
impl Slots {
    fn new(context: &WorkspaceContext) -> Self {
        Self {
            values: vec![value(context, Some(7)), value(context, Some(11))],
            immutable: vec![id("a"), id("b")],
            mutable: vec![Some(id("a")), Some(id("b"))],
        }
    }
}
fn metadata(id: &ParameterId) -> ParameterMetadata {
    ParameterMetadata::from_spec(&ParameterSpec::trainable(id.as_str()).unwrap(), true)
}
impl Parameterized<WorkspaceTensor> for Slots {
    fn visit_parameters<'a, V: ParameterVisitor<'a, WorkspaceTensor>>(&'a self, visitor: &mut V) {
        for (id, value) in self.immutable.iter().zip(&self.values) {
            visitor.visit(metadata(id), value);
        }
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, WorkspaceTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        for (id, value) in self.mutable.iter().zip(&mut self.values) {
            if let Some(id) = id {
                visitor.visit_mut(metadata(id), value);
            }
        }
    }
    fn set_trainable(&mut self, _: bool) {}
}

fn id(name: &str) -> ParameterId {
    ParameterId::new(name).unwrap()
}
fn view(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    shape: &[i32],
    dtype: WorkspaceDtype,
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(shape, dtype).unwrap(),
        root,
        context,
    )
    .unwrap()
}
fn value(context: &WorkspaceContext, capacity: Option<u64>) -> WorkspaceTensor {
    view(
        context,
        &WorkspaceExistingStorage::new(capacity, context),
        &[2],
        WorkspaceDtype::Float32,
    )
}
fn weights(context: &WorkspaceContext) -> BTreeMap<ParameterId, WorkspaceTensor> {
    [
        (id("a"), value(context, Some(101))),
        (id("b"), value(context, Some(103))),
    ]
    .into()
}
fn retained(context: &WorkspaceContext, values: &[WorkspaceTensor]) -> Option<u64> {
    context
        .report(values)
        .unwrap()
        .state
        .unwrap()
        .retained_bytes
}
fn unchanged(context: &WorkspaceContext, module: &Slots) {
    assert_eq!(retained(context, &module.values), Some(18));
    let report = context.report(&module.values).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
}
fn cause(error: &Error) -> &ParameterOrchestrationError<Error> {
    error
        .source()
        .unwrap()
        .downcast_ref::<ParameterOrchestrationError<Error>>()
        .expect("binding retains the neutral orchestration error as its source")
}

#[test]
fn binding_preserves_supplied_aliases_without_allocating_or_reading_payloads() {
    let context = WorkspaceContext::new(NoOperations);
    let mut module = Slots::new(&context);
    let root = WorkspaceExistingStorage::new(Some(64), &context);
    let first = view(&context, &root, &[2], WorkspaceDtype::Float32);
    let second = view(&context.clone(), &root, &[2], WorkspaceDtype::Float32);
    assert!(first.same_context(&second));
    context.begin_state_span([]).unwrap();
    bind_workspace_parameters(
        &mut module,
        [(id("a"), first.clone()), (id("b"), second)].into(),
    )
    .unwrap();
    // The provider's original handle and both bound slots share one capacity.
    let mut roots = module.values.clone();
    roots.push(first);
    assert_eq!(retained(&context, &roots), Some(64));
    let report = context.report(&roots).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
}

#[test]
fn missing_and_unexpected_bindings_reject_before_replacing_a_valid_first_slot() {
    for missing in [true, false] {
        let context = WorkspaceContext::new(NoOperations);
        let mut module = Slots::new(&context);
        context.begin_state_span([]).unwrap();
        let mut provided = weights(&context);
        if missing {
            provided.remove(&id("b"));
        } else {
            provided.insert(id("extra"), value(&context, Some(107)));
        }
        let error = bind_workspace_parameters(&mut module, provided).unwrap_err();
        if missing {
            assert!(
                matches!(cause(&error), ParameterOrchestrationError::MissingBinding { parameter } if parameter == &id("b"))
            );
        } else {
            assert!(
                matches!(cause(&error), ParameterOrchestrationError::UnexpectedBindings { parameters } if parameters == &[id("extra")])
            );
        }
        unchanged(&context, &module);
    }
}

#[test]
fn duplicate_immutable_or_mutable_destinations_reject_without_partial_binding() {
    for mutable in [false, true] {
        let context = WorkspaceContext::new(NoOperations);
        let mut module = Slots::new(&context);
        let mut provided = weights(&context);
        if mutable {
            module.mutable[1] = Some(id("a"));
        } else {
            module.immutable[1] = id("a");
            provided.remove(&id("b"));
        }
        context.begin_state_span([]).unwrap();
        let error = bind_workspace_parameters(&mut module, provided).unwrap_err();
        assert!(
            matches!(cause(&error), ParameterOrchestrationError::DuplicateParameter { parameter } if parameter == &id("a"))
        );
        unchanged(&context, &module);
    }
}

#[test]
fn changed_mutable_topology_rejects_before_publication() {
    for missing in [false, true] {
        let context = WorkspaceContext::new(NoOperations);
        let mut module = Slots::new(&context);
        module.mutable[1] = (!missing).then(|| id("unexpected"));
        context.begin_state_span([]).unwrap();
        let error = bind_workspace_parameters(&mut module, weights(&context)).unwrap_err();
        let ParameterOrchestrationError::ParameterTraversalMismatch { parameters } = cause(&error)
        else {
            panic!("wrong failure: {error}");
        };
        assert_eq!(
            parameters,
            &if missing {
                vec![id("b")]
            } else {
                vec![id("b"), id("unexpected")]
            }
        );
        unchanged(&context, &module);
    }
}

#[test]
fn shape_context_and_dtype_are_validated_before_any_slot_changes() {
    for invalid in ["shape", "context", "dtype"] {
        let context = WorkspaceContext::new(NoOperations);
        let foreign = WorkspaceContext::new(NoOperations);
        let mut module = Slots::new(&context);
        let supplied_context = if invalid == "context" {
            &foreign
        } else {
            &context
        };
        let replacement = view(
            supplied_context,
            &WorkspaceExistingStorage::new(Some(103), supplied_context),
            if invalid == "shape" { &[3] } else { &[2] },
            if invalid == "dtype" {
                WorkspaceDtype::Int32
            } else {
                WorkspaceDtype::Float32
            },
        );
        let mut provided = weights(&context);
        provided.insert(id("b"), replacement);
        context.begin_state_span([]).unwrap();
        let error = bind_workspace_parameters(&mut module, provided).unwrap_err();
        assert!(matches!(
            cause(&error),
            ParameterOrchestrationError::Backend(_)
        ));
        unchanged(&context, &module);
    }
}

#[test]
fn unknown_and_sublogical_backing_remain_provider_facts_after_binding() {
    let context = WorkspaceContext::new(NoOperations);
    let mut module = Slots::new(&context);
    context.begin_state_span([]).unwrap();
    // A two-element F32 metadata projection may describe native F16 backing or
    // a broadcast allocation. Binding must not replace its four-byte evidence.
    bind_workspace_parameters(
        &mut module,
        [
            (id("a"), value(&context, Some(4))),
            (id("b"), value(&context, None)),
        ]
        .into(),
    )
    .unwrap();
    assert_eq!(retained(&context, &module.values[..1]), Some(4));
    assert_eq!(retained(&context, &module.values), None);
    assert!(context
        .report(&module.values)
        .unwrap()
        .operations
        .is_empty());
}

#[test]
fn generic_binding_preserves_exclusion_and_calls_validation_before_mutation() {
    let context = WorkspaceContext::new(NoOperations);
    let mut module = Slots::new(&context);
    context.begin_state_span([]).unwrap();
    let validations = Cell::new(0);
    let bindings = Cell::new(0);
    bind_parameter_values(
        &mut module,
        [(id("a"), value(&context, Some(64)))].into(),
        |id| id.as_str() == "b",
        |_: &WorkspaceTensor, _: &WorkspaceTensor| {
            assert_eq!(bindings.get(), 0);
            validations.set(validations.get() + 1);
            Ok::<_, Error>(())
        },
        |parameter, weight| {
            assert_eq!(validations.get(), 1);
            bindings.set(bindings.get() + 1);
            *parameter = weight;
        },
    )
    .unwrap();
    assert_eq!((validations.get(), bindings.get()), (1, 1));
    assert_eq!(retained(&context, &module.values), Some(75));
    assert_eq!(retained(&context, &module.values[1..]), Some(11));
}

#[test]
fn empty_modules_accept_only_empty_bindings() {
    let mut module = Slots {
        values: vec![],
        immutable: vec![],
        mutable: vec![],
    };
    bind_workspace_parameters(&mut module, BTreeMap::new()).unwrap();
    let context = WorkspaceContext::new(NoOperations);
    let error = bind_workspace_parameters(&mut module, weights(&context)).unwrap_err();
    assert!(
        matches!(cause(&error), ParameterOrchestrationError::UnexpectedBindings { parameters } if parameters == &[id("a"), id("b")])
    );
}

#[test]
fn finite_workspace_rows_reject_late_mismatch_then_publish_shared_aliases_atomically() {
    struct PreparedSlots { specs: [ParameterSpec;2], values: [WorkspaceTensor;2] }
    impl Parameterized<WorkspaceTensor> for PreparedSlots {
        fn visit_parameters<'a,V:ParameterVisitor<'a,WorkspaceTensor>>(&'a self, visitor:&mut V) {
            for (spec,value) in self.specs.iter().zip(&self.values) {
                visitor.visit_borrowed(eredu_nn::ParameterMetadataView::from_spec(spec,true),value);
            }
        }
        fn visit_parameters_mut<'a,V:ParameterVisitorMut<'a,WorkspaceTensor>>(&'a mut self,visitor:&mut V) {
            for (spec,value) in self.specs.iter().zip(&mut self.values) {
                visitor.visit_mut_borrowed(eredu_nn::ParameterMetadataView::from_spec(spec,true),value);
            }
        }
        fn set_trainable(&mut self,_:bool) {}
    }
    let context=WorkspaceContext::new(NoOperations);
    let mut module=PreparedSlots {specs:[ParameterSpec::trainable("a").unwrap(),ParameterSpec::trainable("b").unwrap()],
        values:[value(&context,Some(7)),value(&context,Some(11))]};
    let root=WorkspaceExistingStorage::new(Some(64),&context);
    let first=view(&context,&root,&[2],WorkspaceDtype::Float32);
    let wrong=view(&context,&root,&[2],WorkspaceDtype::Uint32);
    let mut rows=[crate::PreparedParameterBinding::new("a",first.clone()),
        crate::PreparedParameterBinding::new("b",wrong)];
    context.begin_state_span([]).unwrap();
    let error=bind_prepared_workspace_parameters(&mut module,&mut rows,&context).unwrap_err();
    assert!(matches!(error.source().unwrap().downcast_ref::<crate::PreparedParameterBindingError<PreparedWorkspaceBindingCause>>(),
        Some(crate::PreparedParameterBindingError::Backend(PreparedWorkspaceBindingCause::Dtype))));
    assert_eq!(retained(&context,&module.values),Some(18));
    // The first valid row was not consumed by a later failed companion; the
    // same fixed destination can be corrected and validated before publication.
    rows[1]=crate::PreparedParameterBinding::new("b",view(&context,&root,&[2],WorkspaceDtype::Float32));
    bind_prepared_workspace_parameters(&mut module,&mut rows,&context).unwrap();
    assert_eq!(retained(&context,&module.values),Some(64));
    let report=context.report(&[first,module.values[0].clone(),module.values[1].clone()]).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.state.unwrap().retained_bytes,Some(64));
    assert_eq!(report.tensor_buffers.total_bytes,Some(0));
}
