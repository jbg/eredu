use super::*;

#[test]
fn immutable_source_bounds_count_duplicate_and_hidden_physical_branches() {
    let (source, _) = memory();
    let before = Arc::strong_count(&source);
    assert_eq!(source.source_storage_slot_bound().unwrap(), Some(2));
    assert_eq!(Arc::strong_count(&source), before);
    let selected: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            source.clone(),
            "selected",
            BTreeSet::from(["selected".into()]),
        )
        .unwrap(),
    );
    let hidden: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(source.clone(), "hidden", BTreeSet::new()).unwrap(),
    );
    assert_eq!(hidden.source_storage_slot_bound().unwrap(), Some(2));
    let composite = CompositeCheckpointSource::new([selected, hidden]).unwrap();
    assert_eq!(composite.source_keys(), ["selected"]);
    assert_eq!(composite.source_storage_slot_bound().unwrap(), Some(4));
    let (complete, owners) = visit(&composite);
    assert!(complete);
    assert_eq!(owners.len(), 4);
    assert_eq!(capacities(&owners).len(), 2);
}

#[test]
fn complete_current_visitor_and_legacy_storage_do_not_imply_stable_bounds() {
    assert_eq!(LegacyOnly.source_storage_slot_bound().unwrap(), None);
    let source = PrefixSource {
        payload: Arc::new(vec![3, 5]),
        outcome: Outcome::Panic,
    };
    // Calling the visitor would panic; the default count performs no visit.
    assert_eq!(source.source_storage_slot_bound().unwrap(), None);
    let source = PrefixSource {
        payload: Arc::new(vec![7, 11]),
        outcome: Outcome::Complete,
    };
    assert!(visit(&source).0);
    assert_eq!(source.source_storage_slot_bound().unwrap(), None);
}

struct CountSource(Option<usize>);
impl CheckpointSource for CountSource {
    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        Ok(self.0)
    }
    fn source_keys(&self) -> Vec<String> {
        Vec::new()
    }
    fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
        panic!("no metadata")
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("no payload")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("no diagnostics")
    }
}
#[test]
fn unknown_composite_terms_do_not_hide_known_checked_overflow() {
    let sources: Vec<SharedCheckpointSource> = vec![
        Arc::new(CountSource(None)),
        Arc::new(CountSource(Some(usize::MAX))),
        Arc::new(CountSource(Some(1))),
    ];
    let source = CompositeCheckpointSource::new(sources).unwrap();
    assert!(matches!(
        source.source_storage_slot_bound(),
        Err(StoreError::Overflow { .. })
    ));
    let sources: Vec<SharedCheckpointSource> =
        vec![Arc::new(CountSource(None)), Arc::new(CountSource(Some(1)))];
    assert_eq!(
        CompositeCheckpointSource::new(sources)
            .unwrap()
            .source_storage_slot_bound()
            .unwrap(),
        None
    );
}
