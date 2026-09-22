use super::*;
use eredu_nn::workspace::*;
use std::{
    cell::Cell,
    convert::Infallible,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
struct Account(Arc<AtomicBool>);
impl eredu_core::HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        unreachable!()
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        Ok(None)
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        Ok(None)
    }
}
fn context() -> (WorkspaceContext, HostMetadataFunding, Arc<AtomicBool>) {
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account(retired.clone())).unwrap();
    (
        WorkspaceContext::new_with_metadata_funding(Facts, funding.clone()).unwrap(),
        funding,
        retired,
    )
}
struct Owner {
    a: i32,
    b: i32,
    source: ParameterReplacementValues<i32>,
}
impl Owner {
    fn visit(&mut self, visitor: &mut dyn ParameterPublication<i32>) -> Result<bool, Infallible> {
        visitor.slot("a", &mut self.a);
        visitor.slot("b", &mut self.b);
        visitor.replacement_source(&mut self.source);
        Ok(true)
    }
}
fn values(
    context: &WorkspaceContext,
    funding: &HostMetadataFunding,
) -> ParameterReplacementValues<i32> {
    let mut rows = context.metadata_vec(2).unwrap();
    rows.push((context.metadata_string(format_args!("a")).unwrap(), 3));
    rows.push((context.metadata_string(format_args!("b")).unwrap(), 7));
    ParameterReplacementValues::from_prepared_rows(rows, funding.clone(), context).unwrap()
}
#[test]
fn failed_value_preparation_and_final_slot_validation_publish_nothing() {
    let (context, funding, _) = context();
    let values = values(&context, &funding);
    let mut owner = Owner {
        a: 1,
        b: 2,
        source: Default::default(),
    };
    let calls = Cell::new(0);
    let failed = PreparedParameterPublication::prepare(
        values.clone(),
        true,
        |v| owner.visit(v),
        |value| {
            calls.set(calls.get() + 1);
            if calls.get() == 4 {
                Err(WorkspaceMetadataError::Unqualified.into())
            } else {
                Ok(*value)
            }
        },
        &context,
        funding.clone(),
    );
    assert!(matches!(failed, Err(ParameterPublicationError::Value(_))));
    assert_eq!((owner.a, owner.b), (1, 2));
    assert!(owner.source.is_empty());
    let mut prepared = PreparedParameterPublication::prepare(
        values,
        true,
        |v| owner.visit(v),
        |v| Ok(*v),
        &context,
        funding,
    )
    .unwrap();
    owner.b = 8;
    assert!(matches!(
        prepared.validate(|v| owner.visit(v), |a, b| Ok(a == b)),
        Err(ParameterPublicationError::Contract(
            ParameterPublicationFailure::Identity
        ))
    ));
    assert_eq!((owner.a, owner.b), (1, 8));
    assert!(owner.source.is_empty());
    owner.b = 2;
    prepared
        .validate(|v| owner.visit(v), |a, b| Ok(a == b))
        .unwrap();
    prepared.exchange(|v| {
        owner.visit(v).unwrap();
    });
    assert_eq!((owner.a, owner.b), (3, 7));
}
#[test]
fn rollback_moves_same_handles_and_last_future_source_retains_metadata() {
    let (context, funding, retired) = context();
    let values = values(&context, &funding);
    let mut owner = Owner {
        a: 1,
        b: 2,
        source: Default::default(),
    };
    let clone_calls = Cell::new(0);
    let mut prepared = PreparedParameterPublication::prepare(
        values,
        true,
        |v| owner.visit(v),
        |value| {
            clone_calls.set(clone_calls.get() + 1);
            Ok(*value)
        },
        &context,
        funding.clone(),
    )
    .unwrap();
    assert_eq!(clone_calls.get(), 4);
    prepared
        .validate(|v| owner.visit(v), |a, b| Ok(a == b))
        .unwrap();
    prepared.exchange(|v| {
        owner.visit(v).unwrap();
    });
    let alias = owner.source.clone();
    assert_eq!(alias.get("b"), Some(&7));
    prepared
        .validate(|v| owner.visit(v), |a, b| Ok(a == b))
        .unwrap();
    prepared.exchange(|v| {
        owner.visit(v).unwrap();
    });
    assert_eq!((owner.a, owner.b), (1, 2));
    assert!(owner.source.is_empty());
    assert_eq!(clone_calls.get(), 4);
    drop((prepared, context, funding));
    assert!(!retired.load(Ordering::SeqCst));
    drop(alias);
    assert!(retired.load(Ordering::SeqCst));
}
#[test]
fn equal_values_from_another_source_are_stale_participants() {
    let (context, funding, _) = context();
    let original = values(&context, &funding);
    let mut owner = Owner {
        a: 1,
        b: 2,
        source: original,
    };
    let replacements = values(&context, &funding);
    let prepared = PreparedParameterPublication::prepare(
        replacements,
        false,
        |v| owner.visit(v),
        |v| Ok(*v),
        &context,
        funding.clone(),
    )
    .unwrap();
    owner.source = values(&context, &funding);
    assert!(matches!(
        prepared.validate(|v| owner.visit(v), |a, b| Ok(a == b)),
        Err(ParameterPublicationError::Contract(
            ParameterPublicationFailure::Identity
        ))
    ));
    assert_eq!((owner.a, owner.b), (1, 2));
    assert_eq!(owner.source.get("a"), Some(&3));
}

#[test]
fn counters_prepare_once_and_final_counter_refusal_is_atomic() {
    let (context, funding, _) = context();
    let values = values(&context, &funding);
    let mut first = 4u64;
    let mut last = u64::MAX;
    let calls = Cell::new(0usize);
    let mut visit = |v: &mut dyn ParameterPublication<i32>| {
        v.counter(&mut first, &mut |_, _| {
            calls.set(calls.get() + 1);
            Ok(8)
        });
        v.counter(&mut last, &mut |_, _| {
            calls.set(calls.get() + 1);
            u64::MAX
                .checked_add(1)
                .ok_or_else(|| WorkspaceMetadataError::Overflow.into())
        });
        Ok::<_, Infallible>(true)
    };
    let result = PreparedParameterPublication::prepare(
        values.clone(),
        true,
        &mut visit,
        |v| Ok(*v),
        &context,
        funding.clone(),
    );
    assert!(matches!(result, Err(ParameterPublicationError::Value(_))));
    assert_eq!(calls.get(), 2);
    assert_eq!((first, last), (4, u64::MAX));
    let mut visit = |v: &mut dyn ParameterPublication<i32>| {
        v.counter(&mut first, &mut |_, _| {
            calls.set(calls.get() + 1);
            Ok(8)
        });
        v.counter(&mut last, &mut |_, _| {
            calls.set(calls.get() + 1);
            Ok(9)
        });
        Ok::<_, Infallible>(true)
    };
    let mut result = PreparedParameterPublication::prepare(
        values,
        true,
        &mut visit,
        |v| Ok(*v),
        &context,
        funding,
    )
    .unwrap();
    result.validate(&mut visit, |a, b| Ok(a == b)).unwrap();
    result.exchange(|v| {
        visit(v).unwrap();
    });
    result.validate(&mut visit, |a, b| Ok(a == b)).unwrap();
    result.exchange(|v| {
        visit(v).unwrap();
    });
    assert_eq!(
        calls.get(),
        4,
        "validation and rollback never reconstruct proposals"
    );
    assert_eq!((first, last), (4, u64::MAX));
}

#[test]
fn concurrent_replacement_aliases_retire_payload_before_last_payer() {
    use std::sync::{atomic::AtomicUsize, Barrier};
    struct Payload {
        retired: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            assert!(!self.retired.load(Ordering::SeqCst));
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let (context, funding, retired) = context();
    let drops = Arc::new(AtomicUsize::new(0));
    let mut rows = context.metadata_vec(1).unwrap();
    rows.push((
        context.metadata_string(format_args!("a")).unwrap(),
        Payload {
            retired: retired.clone(),
            drops: drops.clone(),
        },
    ));
    let value =
        ParameterReplacementValues::from_prepared_rows(rows, funding.clone(), &context).unwrap();
    let alias = value.clone();
    let barrier = Arc::new(Barrier::new(2));
    drop((context, funding));
    std::thread::scope(|scope| {
        let shared = &barrier;
        scope.spawn(move || {
            shared.wait();
            drop(value);
        });
        scope.spawn(move || {
            shared.wait();
            drop(alias);
        });
    });
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn foreign_metadata_payer_cannot_replace_the_actual_constructor_account() {
    let (context, funding, _) = context();
    let foreign = HostMetadataFunding::new(Account(Arc::new(AtomicBool::new(false)))).unwrap();
    assert!(matches!(
        ParameterReplacementValues::<i32>::from_prepared_rows(
            Vec::new(),
            foreign.clone(),
            &context
        ),
        Err(ParameterPublicationFailure::Identity)
    ));
    let values = values(&context, &funding);
    let mut visited = false;
    let result = PreparedParameterPublication::prepare(
        values,
        true,
        |_| {
            visited = true;
            Ok::<_, Infallible>(true)
        },
        |value| Ok(*value),
        &context,
        foreign,
    );
    assert!(matches!(
        result,
        Err(ParameterPublicationError::Contract(
            ParameterPublicationFailure::Identity
        ))
    ));
    assert!(!visited);
}
