use super::*;
use crate::{checkpoint::TensorDtype, InputPartDescriptor, InputTensorIdentity};

fn mixed() -> PreparedPromptAttribution {
    let parts = [
        (
            InputModality::Text,
            InputPayloadKind::TokenIds,
            vec![1, 2],
            TensorDtype::U32,
        ),
        (
            InputModality::Image,
            InputPayloadKind::Embeddings,
            vec![1, 3, 8],
            TensorDtype::F32,
        ),
        (
            InputModality::Text,
            InputPayloadKind::Embeddings,
            vec![1, 2, 8],
            TensorDtype::F32,
        ),
    ]
    .into_iter()
    .map(|(modality, kind, shape, dtype)| {
        InputPartDescriptor::new(
            modality,
            kind,
            InputTensorIdentity::new(dtype, shape).unwrap(),
            [],
        )
        .unwrap()
    })
    .collect();
    PreparedPromptAttribution {
        schema_version: PREPARED_PROMPT_ATTRIBUTION_VERSION,
        prepared: PreparedInputIdentity::new(parts).unwrap(),
        semantic_content_identity: "actual-source".into(),
        opening_position: 11,
        decoder_positions: 7,
        batch: 1,
        canonical_token_ids: vec![7, 9],
        segments: vec![
            PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: 0,
                    modality: InputModality::Text,
                    payload: InputPayloadKind::TokenIds,
                    decoder_range: [0, 2],
                },
                tokens: PromptTokenAttribution::Canonical { range: [0, 2] },
            },
            PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: 1,
                    modality: InputModality::Image,
                    payload: InputPayloadKind::Embeddings,
                    decoder_range: [2, 5],
                },
                tokens: PromptTokenAttribution::NotTokenized,
            },
            PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: 2,
                    modality: InputModality::Text,
                    payload: InputPayloadKind::Embeddings,
                    decoder_range: [5, 7],
                },
                tokens: PromptTokenAttribution::NotTokenized,
            },
        ],
    }
}

#[test]
fn mixed_positions_keep_nonzero_origin_and_only_real_canonical_ids() {
    let source = mixed();
    source.validate().unwrap();
    assert_eq!(source.complete_token_ids(), None);
    assert_eq!(source.canonical_token_ids, [7, 9]);
    assert_eq!(source.input_range(0).unwrap(), [11, 18]);
    assert_eq!(source.input_range(1).unwrap(), [18, 19]);
    assert_eq!(source.input_range(3).unwrap(), [20, 21]);
    assert_eq!(
        source.input_range(u64::MAX),
        Err(PreparedControlInputError::Overflow)
    );
}
#[test]
fn malformed_part_coverage_and_fabricated_projected_tokens_reject() {
    let mut source = mixed();
    source.segments[1].plan.decoder_range = [3, 6];
    assert_eq!(
        source.validate(),
        Err(PreparedControlInputError::InvalidAttribution)
    );
    let mut source = mixed();
    source.segments[2].tokens = PromptTokenAttribution::Canonical { range: [2, 4] };
    source.canonical_token_ids.extend([0, 0]);
    assert_eq!(
        source.validate(),
        Err(PreparedControlInputError::InvalidAttribution)
    );
    let mut source = mixed();
    source.opening_position = u64::MAX - 6;
    assert_eq!(source.validate(), Err(PreparedControlInputError::Overflow));
}

#[test]
fn last_concurrent_attribution_owner_keeps_real_host_custody() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let host = HostPreparationAuthority::retain(Owner(dropped.clone()));
    let source = SharedPromptAttribution::from_prepared(mixed(), host).unwrap();
    let aliases: Vec<_> = (0..8).map(|_| source.clone()).collect();
    drop(source);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for alias in aliases {
            joins.push(scope.spawn(move || {
                assert_eq!(alias.attribution().canonical_token_ids, [7, 9]);
                drop(alias);
            }));
        }
        for join in joins {
            join.join().unwrap();
        }
    });
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
