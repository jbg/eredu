use super::*;
use eredu_nn::workspace::*;
use eredu_runtime::{
    parameter_operations::{
        ParameterReplacementValues, PreparedBankParameterMember, PreparedParameterPublication,
    },
    working_memory::InferenceExecutionIdentity,
};

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("publication does not execute equations")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("no equation facts")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("no equation emission")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("no equation host facts")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("no equation host emission")
    }
}
fn context() -> (WorkspaceContext, HostMetadataFunding) {
    let pool = crate::backend::managed_memory::ledger();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            pool.configured_limits().clone(),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    (context, funding)
}
pub(super) fn replacement_values(value: MlxTensor) -> ParameterReplacementValues<MlxTensor> {
    let (context, funding) = context();
    let mut rows = context.metadata_vec(1).unwrap();
    rows.push((
        context.metadata_string(format_args!("weight")).unwrap(),
        value,
    ));
    ParameterReplacementValues::from_prepared_rows(rows, funding, &context).unwrap()
}
fn bank() -> (tempfile::TempDir, SharedAddressableParameterBank) {
    let (dir, store) = fixture();
    let entries = entries();
    let members = entries
        .iter()
        .flat_map(|entry| {
            entry
                .unit
                .bindings()
                .iter()
                .map(|binding| PreparedBankParameterMember {
                    key: entry.identity,
                    binding: binding.name().to_owned(),
                    parameter: binding.name().to_owned(),
                    materialized: binding.source_recipe().infer(store.as_ref()).unwrap(),
                })
        })
        .collect();
    let mut bank = AddressableParameterBank::new(
        store,
        entries,
        ParameterBankOptions::new(OffloadConfig::new(Some(64), Some(64), 1).unwrap(), 64, 64)
            .unwrap(),
        stream(),
        stream(),
    )
    .unwrap();
    bank.parameter_members = members;
    (dir, SharedAddressableParameterBank::new(bank))
}

#[test]
fn prepared_bank_publication_exchanges_each_physical_pool_once_and_restores_counters() {
    let (_one, one) = bank();
    let (_two, two) = bank();
    let scoped = one.scoped(0).unwrap();
    let (context, funding) = context();
    let values = replacement_values(MlxTensor::from_array(Array::from_slice(
        &[half::f16::from_f32(1.25); 6],
        &[3, 2],
    )));
    let handles = [&two, &scoped, &one];
    let mut prepared = with_bank_parameter_publication(handles.into_iter(), &context, |visit| {
        PreparedParameterPublication::prepare(
            values.clone(),
            true,
            &mut *visit,
            |_| panic!("banks lend shared sources, not loaded slots"),
            &context,
            funding.clone(),
        )
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))
    })
    .unwrap();
    assert_eq!(one.inner.lock().unwrap().parameter_revision, 0);
    with_bank_parameter_publication(handles.into_iter(), &context, |visit| {
        assert!(one.inner.try_lock().is_err() && two.inner.try_lock().is_err());
        prepared
            .validate(&mut *visit, |_, _| panic!("no loaded slots"))
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        prepared.exchange(|visitor| assert!(visit(visitor).unwrap()));
        Ok(())
    })
    .unwrap();
    for bank in [&one, &two] {
        let bank = bank.inner.lock().unwrap();
        assert_eq!(
            bank.parameter_revision, 1,
            "scoped duplicate is one participant"
        );
        assert!(bank.parameter_replacements.same_source(&values));
        assert!(bank.effective_member_bytes.values().all(|&n| n == 12));
    }
    let invalid = replacement_values(MlxTensor::from_array(Array::from_slice(
        &[7f32; 4],
        &[2, 2],
    )));
    assert!(
        with_bank_parameter_publication(handles.into_iter(), &context, |visit| {
            PreparedParameterPublication::prepare(
                invalid,
                true,
                visit,
                |_| unreachable!(),
                &context,
                funding.clone(),
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))
        })
        .is_err()
    );
    assert_eq!(one.inner.lock().unwrap().parameter_revision, 1);
    assert_eq!(two.inner.lock().unwrap().parameter_revision, 1);
    let mut restore = with_bank_parameter_publication(handles.into_iter(), &context, |visit| {
        PreparedParameterPublication::prepare(
            ParameterReplacementValues::default(),
            false,
            visit,
            |_| unreachable!(),
            &context,
            funding.clone(),
        )
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))
    })
    .unwrap();
    with_bank_parameter_publication(handles.into_iter(), &context, |visit| {
        restore
            .validate(&mut *visit, |_, _| unreachable!())
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        restore.exchange(|visitor| assert!(visit(visitor).unwrap()));
        Ok(())
    })
    .unwrap();
    for bank in [&one, &two] {
        let bank = bank.inner.lock().unwrap();
        assert_eq!(bank.parameter_revision, 2);
        assert!(bank.parameter_replacements.is_empty());
        assert!(bank.effective_member_bytes.values().all(|&n| n == 16));
    }
}
