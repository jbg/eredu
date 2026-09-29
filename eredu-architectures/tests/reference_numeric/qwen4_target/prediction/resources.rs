//! Aggregate cold coverage uses actual retained target/prediction mechanisms.
use super::*;
use eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution;
use eredu_core::{resources::*, Observed};
use eredu_runtime::execution_resources::PreparedResourceQuery;

pub(super) fn check(selected: &SelectedTargetExecution, source: &SharedCheckpointSource) {
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let mut query = PreparedResourceQuery {
        scope: ResourceIdentity {
            scope: "numeric".into(),
            key: "paired-instance".into(),
        },
        batch_size: 2,
        prefix_positions: 5,
        additional_positions: 4,
        device_pool: Observed::Unavailable {
            reason: "neutral fixture has no physical pool".into(),
        },
    };
    let report = selected.describe_prepared_resources(&query).unwrap();
    let states = &report.resources.allocations;
    let expected_count = selected.realization().text().state().components().len()
        + selected.prediction_state().unwrap().components().len();
    assert_eq!(states.len(), expected_count);
    // Four target streams and two predictor streams, two lanes each. Shared
    // vocabulary does not eliminate independent prediction state/stream limits.
    assert_eq!(report.stream_allowances.payload_bytes, 12 * 65536);
    assert_eq!(report.stream_allowances.scratch_bytes, 12 * 65536);
    assert_eq!(report.stream_allowances.catalog_bytes, 12 * 65536);
    let rows = selected.realization().row_lookups().unwrap();
    assert_eq!(report.rows, Some(rows.requirements()));
    assert_eq!(report.row_policy, Some(rows.options()));
    assert_eq!(&report.row_decoders, rows.decode_memory());
    assert_eq!(report.row_decoders.len(), 1);
    for (id, memory) in &report.row_decoders {
        assert_eq!(
            memory.values,
            rows.descriptors().entries()[id]
                .decode_memory()
                .unwrap()
                .values
        );
        assert!(
            memory.storage.is_empty() && !memory.missing.is_empty(),
            "neutral fixture does not invent native allocations"
        );
    }

    assert_eq!(report.rows.unwrap().rows.source_bytes, 24 * 2 * 4);
    assert_eq!(report.rows.unwrap().rows.scalar_bytes, 0);
    let ResourceCoverage::Partial { reasons } = &report.resources.coverage else {
        panic!()
    };
    assert!(reasons.iter().any(|r| r.contains("target-feature")));
    assert!(reasons.iter().any(|r| r.contains("row tables")));
    // The tiny fixture has one K/V head of width two, F32, batch two.
    let kv: Vec<_> = states
        .iter()
        .filter(|a| {
            a.identity.key.ends_with("attention.keys")
                || a.identity.key.ends_with("attention.values")
        })
        .collect();
    assert_eq!(kv.len(), 6); // two target indexed layers plus one prediction layer
    for value in kv {
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = &value.size
        else {
            panic!()
        };
        assert_eq!(current.payload.lower_bytes, 2 * 5 * 2 * 4);
        assert_eq!(horizon_peak.payload.upper_bytes, Some(2 * 9 * 2 * 4));
        assert!(horizon_peak.capacity.upper_bytes.is_none());
    }
    let retained = report.resources;
    query.prefix_positions = 9;
    query.additional_positions = 8;
    let grown = selected.describe_prepared_resources(&query).unwrap();
    // Sparse selection budgets never cap the original K/V history.
    for after in &grown.resources.allocations {
        let before = retained
            .allocations
            .iter()
            .find(|a| a.identity == after.identity)
            .unwrap();
        let ResourceSize::ContextDependent {
            current: a,
            horizon_peak,
        } = &after.size
        else {
            panic!()
        };
        let ResourceSize::ContextDependent { current: b, .. } = &before.size else {
            panic!()
        };
        if after.identity.key.ends_with("attention.keys")
            || after.identity.key.ends_with("attention.values")
        {
            assert!(a.payload.lower_bytes > b.payload.lower_bytes);
            assert_eq!(horizon_peak.payload.upper_bytes, Some(2 * 17 * 2 * 4));
        } else if after.identity.key.contains("state.recurrent")
            || after.identity.key.contains("state.convolution")
            || after.identity.key.contains("state.integer_history")
        {
            assert_eq!(
                a.payload, b.payload,
                "fixed history stays bounded while K/V grows"
            );
        }
    }
    query.batch_size = 3;
    assert!(selected.describe_prepared_resources(&query).is_err());
    query.batch_size = 0;
    assert!(selected.describe_prepared_resources(&query).is_err());
    query.batch_size = 1;
    query.prefix_positions = u64::MAX;
    assert!(selected.describe_prepared_resources(&query).is_err());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
}
