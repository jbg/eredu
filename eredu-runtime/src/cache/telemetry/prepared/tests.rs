use super::*;
use eredu_nn::workspace::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
struct Account {
    closed: Arc<AtomicBool>,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("telemetry performs no equations")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("telemetry has no tensor facts")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("telemetry emits no tensor facts")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("telemetry has no equation host facts")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("telemetry emits no equation host facts")
    }
}
fn collector() -> CacheResidencyTelemetry {
    let mut result = CacheResidencyTelemetry::new(2);
    result.layer_activity_mut(5000).demand_hits = 3;
    result.layer_activity_mut(7000).failures = 2;
    result.report.current_device_bytes = 40;
    result.report.current_host_bytes = 20;
    result.report.in_flight_write_bytes = 8;
    result
}

#[test]
fn prepared_snapshot_preserves_history_overflow_peaks_and_empty_scratch_custody() {
    let closed = Arc::new(AtomicBool::new(false));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        closed: closed.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoEquations, funding).unwrap();
    let mut ordinary = collector();
    let mut prepared = collector();
    let limit = prepared.prepared_layer_count(0..130);
    assert_eq!(limit, CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
    drop(
        prepared
            .install_storage(PreparedCacheTelemetry::prepare(limit, &context).unwrap())
            .unwrap(),
    );
    let mut current = CacheTelemetryRows::new();
    drop(
        current
            .install(PreparedCacheTable::prepare(130, &context).unwrap())
            .unwrap(),
    );
    let mut reference = BTreeMap::new();
    for layer in 0..130 {
        let stats = CacheLayerResidencyStats {
            logical_cached_tokens: layer as u64 + 1,
            current_device_bytes: layer as u64 + 2,
            ..Default::default()
        };
        reference.insert(layer, stats.clone());
        current.insert_prepared(layer, stats).unwrap();
    }
    closed.store(true, Ordering::SeqCst);
    prepared
        .validate_prepared_layers(current.iter().map(|(layer, _)| *layer))
        .unwrap();
    ordinary.layer_activity_mut(0).demand_misses = 4;
    prepared.layer_activity_mut(0).demand_misses = 4;
    ordinary.finalize_snapshot(reference, 40, 19, None);
    prepared.finalize_snapshot_with_rows(&mut current, 40, 19, None);
    assert_eq!(ordinary.report, prepared.report);
    assert_eq!(prepared.report.per_layer.len(), 128);
    assert_eq!(prepared.report.per_layer[125].global_layer, 125);
    assert_eq!(prepared.report.per_layer[126].global_layer, 5000);
    assert_eq!(prepared.report.per_layer[127].global_layer, 7000);
    assert_eq!(prepared.report.per_layer_overflow_layers, 4);
    assert_eq!(
        prepared.report.per_layer_overflow.current_device_bytes,
        128 + 129 + 130 + 131
    );
    assert_eq!(prepared.report.peak_device_bytes, 40);
    assert_eq!(prepared.report.peak_host_bytes, 0);
    assert_eq!(prepared.report.peak_in_flight_write_bytes, 8);
    assert!(current.is_empty());
    assert!(current.validate_prepared_population(130).is_ok());
    assert!(current.validate_prepared_population(131).is_err());
    drop(context);
    assert!(!retired.load(Ordering::SeqCst));
    drop(prepared);
    assert!(!retired.load(Ordering::SeqCst));
    drop(current);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn borrowed_recent_counts_preserve_prefix_uniqueness_and_atomic_refusal() {
    use crate::cache::{CacheBlockLifecycle, CacheLifecycleError};
    use eredu_core::cache::{CacheBlockId, CacheRepresentation};
    let id = |layer, start| CacheBlockId {
        session_id: 9,
        global_layer: layer,
        representation: CacheRepresentation::KeyValue,
        start,
        end: start + 2,
        rank: None,
    };
    let mut source = CacheBlockLifecycle::new();
    let ids = [id(2, 0), id(2, 2), id(2, 4), id(2, 6), id(3, 0)];
    for (index, value) in ids.iter().enumerate() {
        source.insert(value.clone(), index == 0).unwrap();
    }
    let repeated = [&ids[0], &ids[1], &ids[1], &ids[2], &ids[3], &ids[4]];
    let mut actual = BTreeMap::new();
    source
        .accumulate_recent_protection_counts(repeated.into_iter(), 2, &mut actual)
        .unwrap();
    let expected = source
        .recent_protection_counts(repeated.into_iter().rev().cloned(), 2)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual, BTreeMap::from([(2, 2), (3, 1)]));
    let prior = actual.clone();
    assert_eq!(
        source.accumulate_recent_protection_counts([&ids[2], &ids[1]].into_iter(), 2, &mut actual),
        Err(CacheLifecycleError::UnorderedCandidates)
    );
    assert_eq!(actual, prior);
    let missing = id(4, 0);
    assert!(matches!(
        source.accumulate_recent_protection_counts([&ids[1], &missing].into_iter(), 0, &mut actual),
        Err(CacheLifecycleError::MissingBlock(_))
    ));
    assert_eq!(actual, prior);
}
