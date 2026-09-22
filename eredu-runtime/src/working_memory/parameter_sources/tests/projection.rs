use super::*;
use eredu_nn::workspace::*;
use eredu_nn::{
    ParameterMetadataView, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized,
};

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> std::result::Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("source projection does not execute tensor operations")
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!()
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> std::result::Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!()
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!()
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> std::result::Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!()
    }
}
struct Module {
    specs: Vec<ParameterSpec>,
    values: Vec<WorkspaceTensor>,
}
impl Parameterized<WorkspaceTensor> for Module {
    fn visit_parameter_sources<'a, V: eredu_nn::ParameterSourceVisitor<'a, WorkspaceTensor>>(
        &'a self,
        visitor: &mut V,
    ) -> std::result::Result<(), eredu_nn::ParameterSourceError> {
        let mut __source_result = Ok(());

        for (spec, value) in self.specs.iter().zip(&self.values) {
            visitor.parameter(ParameterMetadataView::from_spec(spec, true), value);
        }

        __source_result
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, WorkspaceTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        for (spec, value) in self.specs.iter().zip(&mut self.values) {
            visitor.visit_mut(ParameterMetadataView::from_spec(spec, true), value);
        }
    }
    fn set_trainable(&mut self, _: bool) {}
}
fn module(source: &Source, unit: usize, context: &WorkspaceContext) -> Module {
    Module {
        specs: source.units[unit]
            .1
            .iter()
            .map(|row| ParameterSpec::trainable(row.binding.name()).unwrap())
            .collect(),
        values: source.units[unit]
            .1
            .iter()
            .map(|row| {
                WorkspaceTensor::existing(context.layout(&row.shape, row.dtype).unwrap(), context)
                    .unwrap()
            })
            .collect(),
    }
}
fn retained(context: &WorkspaceContext, values: &[WorkspaceTensor]) -> u64 {
    context.begin_state_span(values.iter()).unwrap();
    context
        .report_scalars(values)
        .unwrap()
        .state
        .unwrap()
        .retained_bytes
        .unwrap()
}

#[test]
fn paid_projection_preserves_actual_float_types_and_alias_lifetimes_at_global_addresses() {
    let mut source = source();
    let f16 = Some(WorkspaceRepresentation::new(
        WorkspaceFloatingType::Float16,
        false,
    ));
    let bf16 = Some(WorkspaceRepresentation::new(
        WorkspaceFloatingType::Bfloat16,
        false,
    ));
    source.units[0].1[0].representation = f16;
    source.units[0].1[1].representation = bf16;
    source.units[1].1[0].representation = f16;
    source.addresses = Some(vec![
        source.layout.address(0).unwrap().with_index(7),
        source.layout.address(1).unwrap().with_index(8),
        source.layout.address(2).unwrap().with_index(12),
    ]);
    let context = WorkspaceContext::new_recording_facts(Facts);
    let mut projected = WorkspaceParameterSourceLoan::new(&source)
        .prepare_projection(&context)
        .unwrap();
    let mut first = module(&source, 0, &context);
    projected
        .bind(
            &mut first,
            0,
            source.execution_address(0).unwrap(),
            &context,
        )
        .unwrap();
    assert_eq!(first.values[0].layout().representation(), f16);
    assert_eq!(first.values[1].layout().representation(), bf16);
    let mut alias = module(&source, 1, &context);
    projected
        .bind(
            &mut alias,
            1,
            source.execution_address(1).unwrap(),
            &context,
        )
        .unwrap();
    let mut repeated = module(&source, 0, &context);
    projected
        .bind(
            &mut repeated,
            0,
            source.execution_address(0).unwrap(),
            &context,
        )
        .unwrap();
    let values = first
        .values
        .into_iter()
        .chain(alias.values)
        .chain(repeated.values)
        .collect::<Vec<_>>();
    // Three live physical prototypes: one shared trace owner, two independent
    // invocation copies. Equal names/capacities never collapse the latter.
    assert_eq!(retained(&context, &values), 12);
    assert!(context.metadata_census().unwrap().context_bytes() > 0);
    drop(projected);
    assert_eq!(
        retained(&context, &values),
        12,
        "escaped values retain their paid roots"
    );
}

#[test]
fn projection_refuses_foreign_context_and_local_address_before_replacing_slots() {
    let mut source = source();
    source.addresses = Some(vec![
        source.layout.address(0).unwrap().with_index(7),
        source.layout.address(1).unwrap().with_index(8),
        source.layout.address(2).unwrap().with_index(12),
    ]);
    let context = WorkspaceContext::new_recording_facts(Facts);
    let other = WorkspaceContext::new_recording_facts(Facts);
    let mut projection = WorkspaceParameterSourceLoan::new(&source)
        .prepare_projection(&context)
        .unwrap();
    let mut unit = module(&source, 0, &context);
    let before = unit.values.iter().cloned().collect::<Vec<_>>();
    assert!(
        projection
            .bind(&mut unit, 0, source.layout.address(0).unwrap(), &context)
            .is_err()
    );
    assert!(
        projection
            .bind(&mut unit, 0, source.execution_address(0).unwrap(), &other)
            .is_err()
    );
    let values = before.into_iter().chain(unit.values).collect::<Vec<_>>();
    assert_eq!(
        retained(&context, &values),
        16,
        "both unmodified slots keep original eight-byte backing"
    );
    drop(projection);
    source.addresses.as_mut().unwrap()[1] = source.layout.address(0).unwrap().with_index(7);
    assert!(WorkspaceParameterSourceLoan::new(&source).count().is_err());
}

#[test]
fn projection_uses_explicit_independent_ownership_without_hiding_missing_bindings() {
    use std::error::Error as _;
    let mut source = source();
    source.excluded.push("independent.bank".into());
    let context = WorkspaceContext::new_recording_facts(Facts);
    let mut projected = WorkspaceParameterSourceLoan::new(&source)
        .prepare_projection(&context)
        .unwrap();
    let make = || {
        let mut unit = module(&source, 0, &context);
        unit.specs
            .push(ParameterSpec::trainable("independent.bank").unwrap());
        unit.values.push(
            WorkspaceTensor::existing(
                context.layout(&[3], WorkspaceDtype::Float32).unwrap(),
                &context,
            )
            .unwrap(),
        );
        unit
    };
    let mut unit = make();
    let bank = unit.values[2].clone();
    projected
        .bind(&mut unit, 0, source.execution_address(0).unwrap(), &context)
        .unwrap();
    assert_eq!(retained(&context, &unit.values), 20);
    assert_eq!(
        retained(&context, &[bank.clone(), unit.values[2].clone()]),
        12,
        "the independent bank slot retains its existing owner without a replacement"
    );

    let mut ordinary = make();
    let weights = source.units[0]
        .1
        .iter()
        .zip(&unit.values)
        .map(|(row, value)| {
            (
                eredu_nn::ParameterId::new(row.binding.name()).unwrap(),
                value.clone(),
            )
        })
        .collect();
    crate::working_memory::bind_workspace_parameters(&mut ordinary, weights, |id| {
        source.excludes_parameter(id.as_str())
    })
    .unwrap();
    assert_eq!(retained(&context, &ordinary.values), 20);

    let mut missing = make();
    missing.specs[1] = ParameterSpec::trainable("missing.unit.weight").unwrap();
    let before = missing.values.clone();
    let error = projected
        .bind(
            &mut missing,
            0,
            source.execution_address(0).unwrap(),
            &context,
        )
        .unwrap_err();
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<crate::PreparedParameterBindingError<
                crate::working_memory::PreparedWorkspaceBindingCause,
            >>(),
        Some(crate::PreparedParameterBindingError::MissingBinding)
    ));
    assert_eq!(
        retained(
            &context,
            &before.into_iter().chain(missing.values).collect::<Vec<_>>()
        ),
        28,
        "a missing nonbank parameter refuses before any earlier slot is replaced"
    );

    drop(projected);
    drop(source);
    assert_eq!(
        retained(&context, &unit.values),
        20,
        "bound and excluded values retain their owners after the source projection retires"
    );
    assert_eq!(
        context
            .report_scalars(&unit.values)
            .unwrap()
            .tensor_buffers
            .total_bytes,
        Some(0)
    );
}

#[test]
fn retained_parameter_rows_require_actual_installed_backings_and_share_their_full_capacity() {
    let mut source = source();
    source.retained = vec![(0, 0), (0, 1), (1, 0)];
    for &(unit, row) in &source.retained {
        let value = &mut source.units[unit].1[row];
        value.capacity = 64;
        value.physical = 8;
        value.owner.lifetime = WorkspaceParameterLifetime::Trace;
        value.representation = Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true,
        ));
    }
    for installed in [
        None,
        Some((32, WorkspaceFloatingType::Float32)),
        Some((64, WorkspaceFloatingType::Float16)),
    ] {
        let context = WorkspaceContext::new_recording_facts(Facts);
        let mut first = module(&source, 0, &context);
        let before = first.values.clone();
        if let Some((capacity, dtype)) = installed {
            let backing = WorkspaceExistingStorage::try_new(Some(capacity), &context).unwrap();
            let rows = source
                .retained
                .iter()
                .map(|&(unit, row)| {
                    let row = &source.units[unit].1[row];
                    WorkspaceParameterRepresentation::new(
                        eredu_nn::ParameterId::new(row.binding.name()).unwrap(),
                        context
                            .layout(&row.shape, row.dtype)
                            .unwrap()
                            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                    )
                    .with_backing(backing.clone())
                })
                .collect();
            context.install_parameter_representations(rows).unwrap();
        }
        let mut projection = WorkspaceParameterSourceLoan::new(&source)
            .prepare_projection(&context)
            .unwrap();
        assert!(
            projection
                .bind(
                    &mut first,
                    0,
                    source.execution_address(0).unwrap(),
                    &context
                )
                .is_err()
        );
        assert_eq!(
            retained(
                &context,
                &before.into_iter().chain(first.values).collect::<Vec<_>>()
            ),
            16,
            "missing or mismatched retained backing refuses before replacing any slot"
        );
    }
    let context = WorkspaceContext::new_recording_facts(Facts);
    let mut first = module(&source, 0, &context);
    let backing = WorkspaceExistingStorage::try_new(Some(64), &context).unwrap();
    let declarations = source
        .retained
        .iter()
        .map(|&(unit, row)| {
            let source = &source.units[unit].1[row];
            WorkspaceParameterRepresentation::new(
                eredu_nn::ParameterId::new(source.binding.name()).unwrap(),
                context
                    .layout(&source.shape, source.dtype)
                    .unwrap()
                    .with_representation(source.representation),
            )
            .with_backing(backing.clone())
        })
        .collect();
    context
        .install_parameter_representations(declarations)
        .unwrap();
    let mut projection = WorkspaceParameterSourceLoan::new(&source)
        .prepare_projection(&context)
        .unwrap();
    projection
        .bind(
            &mut first,
            0,
            source.execution_address(0).unwrap(),
            &context,
        )
        .unwrap();
    let mut alias = module(&source, 1, &context);
    projection
        .bind(
            &mut alias,
            1,
            source.execution_address(1).unwrap(),
            &context,
        )
        .unwrap();
    let mut repeated = module(&source, 0, &context);
    projection
        .bind(
            &mut repeated,
            0,
            source.execution_address(0).unwrap(),
            &context,
        )
        .unwrap();
    let values = first
        .values
        .into_iter()
        .chain(alias.values)
        .chain(repeated.values)
        .collect::<Vec<_>>();
    assert_eq!(
        retained(&context, &values),
        64,
        "the real imported root is shared across names, units and invocations"
    );
    drop(projection);
    drop(source);
    drop(backing);
    assert_eq!(
        retained(&context, &values),
        64,
        "escaped views retain the full installed source capacity"
    );
}
