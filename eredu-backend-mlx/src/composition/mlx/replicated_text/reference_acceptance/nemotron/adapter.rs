use super::*;
#[test]
#[ignore = "requires a pinned released text checkpoint and independent adapter reference"]
fn normal_loader_released_adapter_matches_independent_causal_and_draft_rows() {
    use eredu_core::decoding::AttentionInvocation;
    use safemlx::Dtype;
    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_NEMOTRON_RELEASE_ROOT").expect("pinned released checkpoint root"),
    );
    let (stream, weights) = execution_streams();
    let backend = crate::native::backend(&stream, &weights);
    let source = eredu_core::DraftAdapterSource::new(
        root.join("linear_spec_lora"),
        match root.file_name().and_then(|name| name.to_str()).unwrap() {
            "3B" => "nvidia/Nemotron-Labs-Diffusion-3B",
            "8B" => "nvidia/Nemotron-Labs-Diffusion-8B",
            "14B" => "nvidia/Nemotron-Labs-Diffusion-14B",
            _ => panic!("select a pinned official text geometry"),
        },
    );
    let policy = match std::env::var("EREDU_NEMOTRON_RELEASE_RESIDENCY").as_deref() {
        Ok("resident") => eredu_runtime::WeightResidency::fully_resident(),
        Ok("disk") => eredu_runtime::WeightResidency::dense_disk_stream(Default::default()),
        _ => eredu_runtime::WeightResidency::layerwise_host(Default::default()),
    };
    let model = eredu_core::load_model(
        &backend,
        &root,
        crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default()
                .with_draft_adapter(source)
                .with_weight_residency(policy),
        ),
    )
    .unwrap();
    let reference = Array::load_safetensors(
        root.join("reference-adapter-bf16-components.safetensors"),
        &weights,
    )
    .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    struct Precision<'a> {
        reference: &'a std::collections::HashMap<String, Array>,
        label: &'a str,
        stream: &'a safemlx::Stream,
        values: std::collections::HashMap<String, Array>,
    }
    impl eredu_runtime::ActivationObserver<Array, Error> for Precision<'_> {
        fn observe(&mut self, path: &str, value: &Array) -> Result<(), Error> {
            let suffix = path.strip_prefix("model.layers.");
            let key = suffix.and_then(|suffix| {
                let (index, field) = suffix.split_once('.')?;
                if field == "output" {
                    Some(format!("{}.layer.{index}", self.label))
                } else {
                    Some(format!("{}.component.{index}.{field}", self.label))
                }
            });
            if let Some(key) = key.filter(|key| self.reference.contains_key(key)) {
                let expected = &self.reference[&key];
                if self.label == "causal.0" || key.contains(".component.0.attention.write") {
                    self.values.insert(key.clone(), value.clone());
                }
                let actual = value
                    .as_dtype(Dtype::Float32, self.stream)?
                    .into_evaluated()?;
                let expected = expected
                    .as_dtype(Dtype::Float32, self.stream)?
                    .into_evaluated()?;
                let (mut max, mut dot, mut a2, mut b2) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
                for (&a, &b) in actual
                    .as_slice::<f32>()
                    .iter()
                    .zip(expected.as_slice::<f32>())
                {
                    max = max.max((a - b).abs());
                    dot += f64::from(a) * f64::from(b);
                    a2 += f64::from(a).powi(2);
                    b2 += f64::from(b).powi(2);
                }
                println!(
                    "{key}: dtype={:?} max_abs={max:.8} cosine={:.8}",
                    value.dtype(),
                    dot / (a2 * b2).sqrt()
                );
            }
            Ok(())
        }
    }
    runtime.session_mut().with_model_operation(|model| {
        let model = model.erased_mut();
        let mut cache = model.prepare_autoregressive_cache()?;
        let mut frontier = 0;
        let mut failures = Vec::new();
        for (step, tokens) in [&[1u32,100,200][..], &[17], &[31], &[421]].into_iter().enumerate() {
            let tokens = Array::from_slice(tokens, &[1, tokens.len() as i32]);
            for (label, tokens, invocation) in [
                (format!("causal.{step}"), tokens, AttentionInvocation::CAUSAL),
                (format!("block.{step}"), reference[&format!("block.{step}.tokens")].clone(), AttentionInvocation::DIFFUSION),
                (format!("draft.{step}"), reference[&format!("block.{step}.tokens")].clone(), AttentionInvocation::DRAFT),
            ] {
                if invocation == AttentionInvocation::CAUSAL { frontier += tokens.shape()[1] as u64; }
                let mut observer = Precision { reference: &reference, label: &label, stream: &stream, values: Default::default() };
                let value = model.attention_forward_with_observer(&tokens, None, invocation, Some(&mut cache), step==0 && invocation==AttentionInvocation::CAUSAL, &stream, &mut observer)?;
                observer.values.insert(format!("{label}.logits"), value.clone());
                Array::save_safetensors(&observer.values, None, root.join(format!("normal-loader-{label}-bf16-components.safetensors"))).unwrap();
                if !compare_bfloat_scores(&label, &value, &reference[&format!("{label}.logits")], invocation, &stream)? { failures.push(label); }
                assert_eq!(cache.generation()?, frontier, "transient block or adapter draft advances causal frontier");
            }
        }
        assert!(failures.is_empty(), "normal loaded computation differs from the pinned independent BF16 eager reference: {failures:?}");
        Ok(())
    }).unwrap();
}

pub(super) fn compare_bfloat_scores(
    label: &str,
    value: &Array,
    expected: &Array,
    invocation: eredu_core::decoding::AttentionInvocation,
    stream: &Stream,
) -> Result<bool, Error> {
    use eredu_core::decoding::AttentionInvocation;
    use safemlx::Dtype;
    let dtype = value.dtype();
    let actual = value
        .as_dtype(Dtype::Float32, stream)?
        .into_evaluated()?
        .as_slice::<f32>()
        .to_vec();
    let expected = expected
        .as_dtype(Dtype::Float32, stream)?
        .into_evaluated()?
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0f32;
    let mut absolute = 0.0f64;
    let mut dot = 0.0f64;
    let mut a2 = 0.0f64;
    let mut b2 = 0.0f64;
    let mut admitted = true;
    for (&a, &b) in actual.iter().zip(&expected) {
        maximum = maximum.max((a - b).abs());
        absolute += f64::from((a - b).abs());
        dot += f64::from(a) * f64::from(b);
        a2 += f64::from(a).powi(2);
        b2 += f64::from(b).powi(2);
        admitted &= a.is_finite() && b.is_finite();
    }
    assert_eq!(dtype, Dtype::Bfloat16);
    let vocabulary = value.dim(-1) as usize;
    // BF16 base equations and FP32 low-rank updates have separate
    // cross-kernel rounding envelopes. Raw errors remain reported.
    let (kl_limit, cosine_limit) = if invocation == AttentionInvocation::DRAFT {
        (0.005, 0.998)
    } else {
        (0.002, 0.9995)
    };
    let mut max_kl = 0.0f64;
    let mut matches = 0usize;
    for (a, b) in actual
        .chunks_exact(vocabulary)
        .zip(expected.chunks_exact(vocabulary))
    {
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 {
                        (i, x)
                    } else {
                        best
                    }
                })
        };
        let (ai, am) = argmax(a);
        let (bi, bm) = argmax(b);
        matches += usize::from(ai == bi);
        let za = a
            .iter()
            .map(|&x| (f64::from(x) - f64::from(am)).exp())
            .sum::<f64>()
            .ln();
        let zb = b
            .iter()
            .map(|&x| (f64::from(x) - f64::from(bm)).exp())
            .sum::<f64>()
            .ln();
        let kl = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| {
                let log_b = f64::from(b) - f64::from(bm) - zb;
                log_b.exp() * (log_b - (f64::from(a) - f64::from(am) - za))
            })
            .sum::<f64>();
        admitted &= kl.is_finite() && kl >= -1e-12 && kl <= kl_limit;
        max_kl = max_kl.max(kl);
    }
    admitted &= matches == actual.len() / vocabulary;
    let cosine = dot / (a2 * b2).sqrt();
    println!(
        "{label}: output={dtype:?} max_abs={maximum:.8} mean_abs={:.8} cosine={cosine:.8} max_KL={max_kl:.10} matching_rows={matches}/{} admitted={admitted}",
        absolute / actual.len() as f64,
        actual.len() / vocabulary
    );
    Ok(admitted && cosine >= cosine_limit)
}
