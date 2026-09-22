//! Native fixture conversion through the public original host-source constructor.
use crate::composition::mlx::MlxBackend;
use crate::native::MlxModelInput;
use eredu_core::{ModelRuntime, PreparedPromptAttribution};
use eredu_runtime::input::host::{
    HostInputPart, HostInputPartView, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use eredu_runtime::input::{OriginalModelInputBackend, OriginalModelInputCustody};
use safemlx::{Array, Dtype};

enum Values {
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}
struct Slot {
    shape: Vec<usize>,
    values: Values,
}
impl Slot {
    fn new(array: &Array) -> Self {
        let evaluated = array.evaluated().unwrap();
        let values = match array.dtype() {
            Dtype::Uint32 => Values::U32(evaluated.as_slice::<u32>().to_vec()),
            Dtype::Int32 => Values::I32(evaluated.as_slice::<i32>().to_vec()),
            Dtype::Float32 => Values::F32(evaluated.as_slice::<f32>().to_vec()),
            Dtype::Bool => Values::Bool(evaluated.as_slice::<bool>().to_vec()),
            dtype => panic!("unsupported original source fixture dtype {dtype:?}"),
        };
        Self {
            shape: array
                .shape()
                .iter()
                .map(|n| usize::try_from(*n).unwrap())
                .collect(),
            values,
        }
    }
    fn view(&self) -> HostTensorView<'_> {
        HostTensorView {
            shape: &self.shape,
            values: match &self.values {
                Values::U32(v) => HostTensorValues::U32(v),
                Values::I32(v) => HostTensorValues::I32(v),
                Values::F32(v) => HostTensorValues::F32(v),
                Values::Bool(v) => HostTensorValues::Bool(v),
            },
        }
    }
}

/// Reads caller-owned numerical fixture values before constructing independently
/// accounted source buffers, native leaves and authenticated selected semantics.
pub(crate) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: MlxModelInput,
) -> (
    MlxModelInput,
    OriginalModelInputCustody,
    PreparedPromptAttribution,
) {
    let (mut prompt, custody) = input.with_borrowed(|input| {
        let payload = input
            .parts
            .iter()
            .map(|p| Slot::new(p.payload().value()))
            .collect::<Vec<_>>();
        let metadata = input
            .parts
            .iter()
            .map(|p| {
                p.metadata()
                    .iter()
                    .map(|(k, v)| (*k, Slot::new(v)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = metadata
            .iter()
            .map(|p| p.iter().map(|(k, v)| (*k, v.view())).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let parts = input
            .parts
            .iter()
            .enumerate()
            .map(|(i, p)| HostInputPart {
                modality: p.modality(),
                kind: p.payload().kind(),
                payload: payload[i].view(),
                metadata: &views[i],
                extents: p.extents(),
            })
            .collect::<Vec<_>>();
        MlxBackend::prepare_original_model_input(
            runtime,
            PreparedHostInputPlan::prepare(&parts).unwrap(),
            &Default::default(),
        )
        .unwrap()
        .into_parts()
    });
    if let Some(chunk) = input.with_borrowed(|i| i.prefill_chunk_positions()) {
        prompt = prompt.with_prefill_chunk_positions(chunk);
    }
    let attribution = attribution(&prompt);
    (prompt, custody, attribution)
}

pub(crate) fn attribution(prompt: &MlxModelInput) -> PreparedPromptAttribution {
    use eredu_core::{
        InputPayloadKind, PreparedPromptSegment, PreparedPromptSegmentPlan, PromptTokenAttribution,
    };
    let semantics = MlxBackend::original_model_input_semantics(prompt)
        .expect("canonical original fixture source");
    let mut canonical_token_ids = Vec::new();
    let segments = semantics
        .records()
        .iter()
        .map(|record| {
            let part = semantics.source().part(record.source_part).unwrap();
            let tokens = if part.kind() == InputPayloadKind::TokenIds {
                let start = canonical_token_ids.len() as u64;
                match part.payload_view().values {
                    HostTensorValues::U32(values) => canonical_token_ids.extend_from_slice(values),
                    HostTensorValues::I32(values) => canonical_token_ids
                        .extend(values.iter().map(|v| u32::try_from(*v).unwrap())),
                    _ => panic!("original token type"),
                }
                PromptTokenAttribution::Canonical {
                    range: [start, canonical_token_ids.len() as u64],
                }
            } else {
                PromptTokenAttribution::NotTokenized
            };
            PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: record.source_part as u64,
                    modality: record.modality,
                    payload: part.kind(),
                    decoder_range: [record.start, record.end],
                },
                tokens,
            }
        })
        .collect();
    let (prepared, semantic_content_identity) = prompt.with_borrowed(|i| {
        let identity = i.cache_identity().unwrap();
        (
            identity.prepared().clone(),
            identity.semantic_content_fingerprint().to_owned(),
        )
    });
    let result = PreparedPromptAttribution {
        schema_version: 1,
        prepared,
        semantic_content_identity,
        opening_position: semantics.binding().frontier(),
        decoder_positions: semantics.layout().positions() as u64,
        batch: 1,
        segments,
        canonical_token_ids,
    };
    result.validate().unwrap();
    result
}
