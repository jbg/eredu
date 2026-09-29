//! Native realization of the portable indexed unit over resident and paged state.
use super::*;
use eredu_architectures::{
    decoder::ComponentInstrumentation,
    qwen4_exp::{
        config::Config,
        indexed::{IndexedSublayer, IndexedSublayerInput, IndexedSublayerSpec},
        qsa::QsaExecutionLimits,
    },
};
use eredu_core::LayerSchedule;
use eredu_nn::{
    ParameterMetadata, ParameterVisitorMut, Parameterized, RotaryPosition, Tensor,
    TensorElementType,
};
use eredu_runtime::{
    AppendOnlyStream, AppendStreamBinding, AppendStreamLimits, RuntimeAppendStreams,
};

struct Load;
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Load {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut MlxTensor) {
        let seed = meta
            .id
            .as_str()
            .bytes()
            .fold(0usize, |s, b| (s * 17 + b as usize) % 101);
        let count = value.shape().iter().map(|d| *d as usize).product();
        let values: Vec<_> = (0..count)
            .map(|i| (((i * 7 + seed) % 29) as f32 - 14.) * 0.03)
            .collect();
        *value = Array::from_slice(&values, value.shape()).into();
    }
}
fn specification() -> IndexedSublayerSpec {
    let mut c = Config::from_json(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eredu-architectures/src/qwen4_exp/config/released.json"
        )))
        .unwrap(),
    )
    .unwrap();
    c.hidden_size = 2;
    c.residual.streams = 2;
    c.residual.rank = 2;
    c.norm_epsilon = 1e-4;
    c.attention.heads = 2;
    c.attention.kv_heads = 1;
    c.attention.head_dim = 4;
    c.attention.rotary_dim = 2;
    c.attention.rotary.dimensions = 2;
    c.attention.index_heads = 2;
    c.attention.index_head_dim = 4;
    c.attention.ratio = 3;
    c.attention.budget = 6;
    IndexedSublayerSpec::from_config(
        &c,
        "model.layers.0",
        2,
        16384,
        QsaExecutionLimits {
            batch: 2,
            tokens: 32,
            workspace_bytes: 1048576,
        },
        TensorElementType::F32,
        TensorElementType::F32,
        |_| eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense),
    )
    .unwrap()
}
fn fixture(spec: &IndexedSublayerSpec) -> (StateLayout, Vec<AppendStreamBinding>) {
    let layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![spec
                .state
                .cache_policy(spec.kv_heads, spec.head_dim)
                .unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let bindings = spec
        .state
        .streams()
        .into_iter()
        .map(|spec| AppendStreamBinding {
            layer: 0,
            lanes: 2,
            spec,
            limits: AppendStreamLimits {
                entries: 32,
                page_entries: 2,
                read_entries: 2,
            },
            payload_bytes: 65536,
            scratch_bytes: 4096,
            catalog_bytes: 65536,
        })
        .collect();
    (layout, bindings)
}
fn run(
    layer: &mut IndexedSublayer<MlxNeuralBackend>,
    state: &mut MlxHybridState,
    a: usize,
    b: usize,
    stream: &Stream,
) -> MlxTensor {
    let values: Vec<_> = (0..2)
        .flat_map(|lane| {
            (a..b).flat_map(move |t| {
                (0..4).map(move |i| (((lane * 88 + t * 4 + i) * 11 % 43) as f32 - 21.) / 17.)
            })
        })
        .collect();
    let input: Array = Array::from_slice(&values, &[2, (b - a) as i32, 2, 2]);
    let cosine: Vec<_> = (0..2)
        .flat_map(|lane| (a..b).flat_map(move |t| [((t + lane * 3) as f32 / 5.).cos(); 2]))
        .collect();
    let sine: Vec<_> = (0..2)
        .flat_map(|lane| (a..b).flat_map(move |t| [((t + lane * 3) as f32 / 5.).sin(); 2]))
        .collect();
    // Includes the singleton-head media representation and a partial rotary prefix.
    let cos: MlxTensor = Array::from_slice(&cosine, &[2, 1, (b - a) as i32, 2]).into();
    let sin: MlxTensor = Array::from_slice(&sine, &[2, 1, (b - a) as i32, 2]).into();
    let visibility: Vec<_> = (0..2)
        .flat_map(|lane| {
            (a..b).map(move |t| {
                if lane == 0 {
                    t != 0 && t != 7
                } else {
                    t != 2 && t != 3 && t != 11
                }
            })
        })
        .collect();
    layer
        .forward(
            IndexedSublayerInput {
                residual: &input.into(),
                visible: Some(&visibility),
                rotary: Some(RotaryPosition::Embeddings {
                    cosine: &cos,
                    sine: &sin,
                }),
            },
            &mut state.layers_mut()[0],
            None,
            stream,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap()
}
fn close(a: &MlxTensor, b: &MlxTensor, stream: &Stream) {
    assert_eq!(a.shape(), b.shape());
    for (a, b) in a
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .zip(b.to_f32_vec(stream).unwrap())
    {
        assert!((a - b).abs() <= 2e-4, "{a} != {b}");
    }
}
fn exercise(stream: &Stream) {
    let start = std::time::Instant::now();
    let spec = specification();
    let (layout, bindings) = fixture(&spec);
    let mut layer = IndexedSublayer::<MlxNeuralBackend>::new(spec.clone(), stream).unwrap();
    layer.visit_parameters_mut(&mut Load);
    let mut whole = MlxHybridState::device(layout.clone(), &bindings).unwrap();
    let expected = run(&mut layer, &mut whole, 0, 22, stream);
    for paged in [false, true] {
        let options = eredu_runtime::PagedCacheOptions::new(2, 512, 1048576, 1)
            .unwrap()
            .with_full_attention(true);
        let selected = super::semantic_transaction_tests::selected_state(
            layout.clone(),
            eredu_runtime::ReplicatedTextStateAccess::AttentionWithStreams,
            if paged {
                eredu_runtime::CacheResidencyPolicy::Paged(options.clone())
            } else {
                eredu_runtime::CacheResidencyPolicy::Device
            },
            bindings.clone(),
        );
        let manager = paged.then(|| CacheResidencyManager::new(options.clone()).unwrap());
        let mut state = MlxHybridState::from_selected(&selected, manager, None).unwrap();
        let mut parts = vec![
            run(&mut layer, &mut state, 0, 2, stream),
            run(&mut layer, &mut state, 2, 6, stream),
        ];
        for t in 6..22 {
            parts.push(run(&mut layer, &mut state, t, t + 1, stream));
        }
        close(
            &MlxTensor::concatenate(&parts, 1, stream).unwrap(),
            &expected,
            stream,
        );
        assert_eq!(state.offset(), 22);
        for lane in 0..2 {
            for spec in spec.state.streams() {
                let expected_stream = whole.layers_mut()[0]
                    .append_stream(spec.slot, lane)
                    .unwrap();
                let actual = state.layers_mut()[0]
                    .append_stream(spec.slot, lane)
                    .unwrap();
                assert_eq!(actual.len(), expected_stream.len());
                for row in 0..actual.len() {
                    close(
                        &actual.read(row..row + 1, stream).unwrap(),
                        &expected_stream.read(row..row + 1, stream).unwrap(),
                        stream,
                    );
                }
            }
        }
        let checkpoint = state.deep_clone_state().unwrap();
        let first = run(&mut layer, &mut state, 22, 23, stream);
        let charged = state.residency_report().unwrap();
        state.restore_checkpoint(&checkpoint, stream).unwrap();
        close(&run(&mut layer, &mut state, 22, 23, stream), &first, stream);
        if let Some(before) = charged {
            let report = state.residency_report().unwrap().unwrap();
            assert!(report.append_stream_read_bytes >= before.append_stream_read_bytes);
            assert!(report.selected_attention_bytes > 0);
            assert_eq!(
                report.prefill_full_attention_bytes + report.decode_full_attention_bytes,
                0
            );
            assert!(report.append_stream_scratch_peak_bytes <= 4096);
            println!("qsa indexed paged: selected_bytes={} summary_bytes={} summary_scratch_peak={} attention_scratch_peak={}",report.selected_attention_bytes,report.append_stream_read_bytes,report.append_stream_scratch_peak_bytes,report.attention_scratch_peak_bytes);
        }
    }
    println!("qsa indexed native fixture elapsed={:?}", start.elapsed());
}
#[test]
fn cached_qsa_indexed_native_resident_paged_and_rollback() {
    exercise(&Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Cpu,
        0,
    )));
}
#[test]
#[ignore = "requires Metal device access"]
fn cached_qsa_indexed_native_resident_paged_and_rollback_metal() {
    exercise(&Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Gpu,
        0,
    )));
}
