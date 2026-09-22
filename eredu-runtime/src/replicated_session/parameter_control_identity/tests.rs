use super::*;
use crate::parameter_operations::{
    ParameterPublication, ParameterPublicationError, ParameterReplacementValues,
    PreparedParameterPublication,
};
use crate::working_memory::{InferenceExecutionIdentity, InferenceRetention, MediaSessionBinding};
use eredu_nn::workspace::*;
use std::convert::Infallible;

#[derive(Debug)]
struct Metadata;
impl eredu_core::HostMetadataAccount for Metadata {
    fn reserve_metadata(&self, _: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        Ok(())
    }
}
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        unreachable!("identity publication executes no tensor equations")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
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
fn context() -> (WorkspaceContext, HostMetadataFunding) {
    let funding = HostMetadataFunding::new(Metadata).unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    (context, funding)
}
struct Owner {
    identity: ParameterControlIdentity,
    value: i32,
    source: ParameterReplacementValues<i32>,
}
impl Owner {
    fn visit(&mut self, visitor: &mut dyn ParameterPublication<i32>) -> Result<bool, Infallible> {
        visitor.slot("weight", &mut self.value);
        visitor.replacement_source(&mut self.source);
        self.identity.visit(visitor);
        Ok(true)
    }
    fn origin(&self) -> ReplicatedTextControlOrigin {
        ReplicatedTextControlOrigin {
            owner: self.identity.clone(),
            captured: None,
        }
    }
}
fn replacement(
    context: &WorkspaceContext,
    funding: &HostMetadataFunding,
) -> ParameterReplacementValues<i32> {
    let mut rows = context.metadata_vec(1).unwrap();
    rows.push((context.metadata_string(format_args!("weight")).unwrap(), 17));
    ParameterReplacementValues::from_prepared_rows(rows, funding.clone(), context).unwrap()
}

#[test]
fn prepared_generation_rejects_old_snapshot_and_media_origins_and_rolls_back_with_values() {
    let (context, funding) = context();
    let mut owner = Owner {
        identity: ParameterControlIdentity::new(),
        value: 5,
        source: Default::default(),
    };
    let snapshot = owner.origin();
    let foreign = ReplicatedTextControlOrigin {
        owner: ParameterControlIdentity::new(),
        captured: None,
    };
    let media = MediaSessionBinding {
        execution: InferenceExecutionIdentity::default(),
        revision: InferenceRetention::new().revision().clone(),
        control: owner.identity.clone(),
        frontier: 3,
    };
    let mut live_media = MediaSessionBinding {
        execution: media.execution.clone(),
        revision: media.revision.clone(),
        control: media.control.clone(),
        frontier: media.frontier,
    };
    let mut prepared = PreparedParameterPublication::prepare(
        replacement(&context, &funding),
        true,
        |v| owner.visit(v),
        |value| Ok(*value),
        &context,
        funding,
    )
    .unwrap();
    assert!(snapshot.same_origin(&owner.origin()));
    assert!(!foreign.same_origin(&owner.origin()));
    assert_eq!(owner.value, 5);
    prepared
        .validate(|v| owner.visit(v), |a, b| Ok(a == b))
        .unwrap();
    prepared.exchange(|v| {
        owner.visit(v).unwrap();
    });
    assert_eq!(owner.value, 17);
    assert_eq!(owner.source.get("weight"), Some(&17));
    assert_eq!(owner.identity.generation, 1);
    assert!(
        snapshot.owner.same_owner(&owner.identity),
        "publication retains its existing owner allocation"
    );
    assert!(!snapshot.same_origin(&owner.origin()));
    let updated = owner.origin();
    assert!(updated.same_origin(&owner.origin()));
    live_media.control = owner.identity.clone();
    assert!(!media.same_origin(&live_media));
    assert!(!media.matches(&live_media));

    prepared
        .validate(|v| owner.visit(v), |a, b| Ok(a == b))
        .unwrap();
    prepared.exchange(|v| {
        owner.visit(v).unwrap();
    });
    assert_eq!(owner.value, 5);
    assert!(owner.source.is_empty());
    assert_eq!(owner.identity.generation, 0);
    assert!(snapshot.same_origin(&owner.origin()));
    assert!(!updated.same_origin(&owner.origin()));
    live_media.control = owner.identity.clone();
    assert!(media.matches(&live_media));
}

#[test]
fn generation_overflow_refuses_before_values_sources_or_snapshot_identity_change() {
    let (context, funding) = context();
    let mut owner = Owner {
        identity: ParameterControlIdentity::new(),
        value: 5,
        source: Default::default(),
    };
    owner.identity.generation = u64::MAX;
    let snapshot = owner.origin();
    let result = PreparedParameterPublication::prepare(
        replacement(&context, &funding),
        true,
        |v| owner.visit(v),
        |value| Ok(*value),
        &context,
        funding,
    );
    let Err(ParameterPublicationError::Value(error)) = result else {
        panic!("generation overflow must refuse prepared publication")
    };
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|cause| cause.downcast_ref::<WorkspaceMetadataError>()),
        Some(WorkspaceMetadataError::Overflow)
    ));
    assert_eq!(owner.value, 5);
    assert!(owner.source.is_empty());
    assert_eq!(owner.identity.generation, u64::MAX);
    assert!(snapshot.same_origin(&owner.origin()));
}
