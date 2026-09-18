use super::*;
use crate::input::host::HostTensorView;

fn text(values: &[u32]) -> HostInputPart<'_> {
    HostInputPart {
        modality: InputModality::Text,
        kind: InputPayloadKind::TokenIds,
        payload: HostTensorView {
            shape: &[1, 2],
            values: HostTensorValues::U32(values),
        },
        metadata: &[],
        extents: &[],
    }
}
fn image() -> HostInputPart<'static> {
    HostInputPart {
        modality: InputModality::Image,
        kind: InputPayloadKind::Embeddings,
        payload: HostTensorView {
            shape: &[1, 1, 2],
            values: HostTensorValues::F32(&[0.25, -0.75]),
        },
        metadata: &[],
        extents: &[],
    }
}

fn records() -> [CompositeSemanticPartRecord; 3] {
    use crate::working_memory::{CompositeChatProjection as P, CompositeSemanticRole as R};
    [
        CompositeSemanticPartRecord {
            source_part: 0,
            start: 0,
            end: 2,
            ..Default::default()
        },
        CompositeSemanticPartRecord {
            source_part: 1,
            start: 2,
            end: 5,
            role: R::Projected,
            modality: InputModality::Image,
            chat_projection: P::Markers(&["<image>"]),
            ..Default::default()
        },
        CompositeSemanticPartRecord {
            source_part: 2,
            start: 5,
            end: 7,
            ..Default::default()
        },
    ]
}
fn marker(value: &str) -> Option<u32> {
    match value {
        "<image>" => Some(41),
        "<end>" => Some(43),
        _ => None,
    }
}

#[test]
fn ordered_text_and_exact_markers_retain_distinct_decoder_coordinates() {
    let parts = [text(&[11, 17]), image(), text(&[23, 29])];
    let mut coordinates = Vec::new();
    compare_parts(
        parts.into_iter(),
        &[11, 17, 41, 23, 29],
        Some(&records()),
        marker,
        |value| coordinates.push(value),
    )
    .unwrap();
    assert_eq!(
        coordinates
            .iter()
            .map(|v| (v.part(), v.text_start(), v.text_length()))
            .collect::<Vec<_>>(),
        [(0, 0, 2), (1, 2, 0), (2, 2, 2)]
    );
    assert_eq!(
        coordinates
            .iter()
            .map(|v| (v.rendered_range(), v.decoder_range()))
            .collect::<Vec<_>>(),
        [
            ([0, 2], Some([0, 2])),
            ([2, 3], Some([2, 5])),
            ([3, 5], Some([5, 7]))
        ]
    );
    assert_eq!(coordinates[1].modality(), InputModality::Image);
}

#[test]
fn reordered_missing_extra_and_equal_length_foreign_text_refuse() {
    for (values, offset) in [
        (&[23, 29, 11, 17][..], 0),
        (&[11, 17, 23][..], 3),
        (&[11, 17, 23, 29, 31][..], 4),
        (&[11, 19, 23, 29][..], 1),
    ] {
        assert_eq!(
            compare_parts(
                [text(values)].into_iter(),
                &[11, 17, 23, 29],
                None,
                marker,
                |_| {}
            ),
            Err(PreparedChatInputRejection::TextMismatch { token: offset })
        );
    }
}

#[test]
fn media_cannot_skip_undeclared_or_incorrect_rendered_tokens() {
    let parts = [text(&[11, 17]), image(), text(&[23, 29])];
    assert_eq!(
        compare_parts(parts.into_iter(), &[11, 17, 23, 29], None, marker, |_| {}),
        Err(PreparedChatInputRejection::ProjectionUnavailable { part: 1 })
    );
    for tokens in [
        &[11, 17, 23, 29][..],
        &[11, 17, 42, 23, 29][..],
        &[11, 17, 41, 41, 23, 29][..],
    ] {
        assert!(
            compare_parts(parts.into_iter(), tokens, Some(&records()), marker, |_| {}).is_err()
        );
    }
    let mut wrong = records();
    wrong[1].source_part = 2;
    assert_eq!(
        compare_parts(
            parts.into_iter(),
            &[11, 17, 41, 23, 29],
            Some(&wrong),
            marker,
            |_| {}
        ),
        Err(PreparedChatInputRejection::ProjectionUnavailable { part: 1 })
    );
}

#[test]
fn architecture_sequence_compares_every_marker_and_generated_framing_consumes_none() {
    let mut sequence = records();
    sequence[1].chat_projection = CompositeChatProjection::Markers(&["<image>", "<end>"]);
    let parts = [text(&[11, 17]), image(), text(&[23, 29])];
    compare_parts(
        parts.into_iter(),
        &[11, 17, 41, 43, 23, 29],
        Some(&sequence),
        marker,
        |_| {},
    )
    .unwrap();
    assert!(
        compare_parts(
            parts.into_iter(),
            &[11, 17, 41, 42, 23, 29],
            Some(&sequence),
            marker,
            |_| {}
        )
        .is_err()
    );
    let framing = CompositeSemanticPartRecord {
        end: 2,
        chat_projection: CompositeChatProjection::Markers(&[]),
        ..Default::default()
    };
    let mut count = 0;
    compare_parts(
        [text(&[51, 53])].into_iter(),
        &[],
        Some(&[framing]),
        marker,
        |c| {
            assert_eq!(c.rendered_range(), [0, 0]);
            assert_eq!(c.decoder_range(), Some([0, 2]));
            count += 1;
        },
    )
    .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn projected_text_cannot_replace_original_token_identity() {
    let mut projected = image();
    projected.modality = InputModality::Text;
    assert_eq!(
        compare_parts(
            [text(&[11, 17]), projected].into_iter(),
            &[11, 17],
            None,
            marker,
            |_| {}
        ),
        Err(PreparedChatInputRejection::TextIdentityUnavailable { part: 1 })
    );
}

#[test]
fn signed_token_values_preserve_identity_without_wrapping_negative_ids() {
    let mut part = text(&[]);
    part.payload.values = HostTensorValues::I32(&[11, 17]);
    compare_parts([part].into_iter(), &[11, 17], None, marker, |_| {}).unwrap();
    part.payload.values = HostTensorValues::I32(&[11, -1]);
    assert_eq!(
        compare_parts([part].into_iter(), &[11, u32::MAX], None, marker, |_| {}),
        Err(PreparedChatInputRejection::TextMismatch { token: 1 })
    );
}

#[test]
fn video_group_accepts_exact_compact_or_expanded_rendering_without_mixing_modes() {
    use crate::working_memory::{CompositeGeneratedText, CompositeSemanticRole};
    let origin = |index| eredu_core::InputExtent::VideoFrame {
        group: 2,
        index,
        count: 2,
        first_source_frame: index,
        last_source_frame: index,
        source_fps_bits: 1.0f64.to_bits(),
    };
    let mut media = image();
    media.modality = InputModality::Video;
    let parts = [
        text(&[70, 81]),
        media,
        text(&[82]),
        text(&[71, 81]),
        media,
        text(&[82]),
    ];
    let mut records = Vec::new();
    let mut position = 0;
    for index in 0..2 {
        for (rows, modality, role, projection) in [
            (
                2,
                InputModality::Text,
                CompositeSemanticRole::Tokens,
                CompositeChatProjection::VideoPrefix {
                    origin: origin(index),
                    text: CompositeGeneratedText::ClockSeconds {
                        seconds: index as u64,
                        leading_space: index != 0,
                    },
                    framing: 81,
                },
            ),
            (
                3,
                InputModality::Video,
                CompositeSemanticRole::Projected,
                CompositeChatProjection::VideoMarker {
                    origin: origin(index),
                    compact: &["<start>", "<video>", "<stop>"],
                    expanded: &["<video>"],
                },
            ),
            (
                1,
                InputModality::Text,
                CompositeSemanticRole::Tokens,
                CompositeChatProjection::VideoSuffix {
                    origin: origin(index),
                    framing: 82,
                },
            ),
        ] {
            records.push(CompositeSemanticPartRecord {
                source_part: records.len(),
                start: position,
                end: position + rows,
                modality,
                role,
                chat_projection: projection,
                ..Default::default()
            });
            position += rows;
        }
    }
    let marker = |s: &str| match s {
        "<start>" => Some(81),
        "<video>" => Some(83),
        "<stop>" => Some(82),
        _ => None,
    };
    for expected in [&[81, 83, 82][..], &[70, 81, 83, 82, 71, 81, 83, 82][..]] {
        let mut coordinates = Vec::new();
        compare_parts(parts.into_iter(), expected, Some(&records), marker, |c| {
            coordinates.push(c)
        })
        .unwrap();
        assert_eq!(coordinates.last().unwrap().decoder_range(), Some([11, 12]));
        assert_eq!(
            coordinates.last().unwrap().rendered_range()[1],
            expected.len()
        );
    }
    for expected in [
        &[70, 81, 83, 82][..],
        &[81, 83, 82, 71, 81, 83, 82][..],
        &[70, 81, 83, 82, 71, 81, 84, 82][..],
    ] {
        assert!(
            compare_parts(parts.into_iter(), expected, Some(&records), marker, |_| {}).is_err()
        );
    }
    records[4].chat_projection = CompositeChatProjection::VideoMarker {
        origin: origin(0),
        compact: &["<start>", "<video>", "<stop>"],
        expanded: &["<video>"],
    };
    assert!(
        compare_parts(
            parts.into_iter(),
            &[81, 83, 82],
            Some(&records),
            marker,
            |_| {}
        )
        .is_err()
    );
}
