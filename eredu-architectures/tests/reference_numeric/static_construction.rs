//! Independent old constructors versus actual shared ordinary B adapters.
use super::*;
use eredu_architectures::decoder::{StaticModuleSpec, StaticModules};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{LinearFormatSpec, ParameterSpec, VocabularyParallelRange};
fn old_format(weight_name: &str, format: LinearFormat) -> Result<LinearFormatSpec, Error> {
    fn companion(weight: &str, name: String, component: &str) -> Result<ParameterSpec, Error> {
        let mut p = ParameterSpec::trainable(name).map_err(Error::backend)?;
        p.group = Some(weight.to_owned());
        if p.id.as_str() == weight {
            return Err(Error::backend(format!(
                "linear {component} companion reuses weight identity {weight:?}"
            )));
        }
        Ok(p)
    }
    let prefix = weight_name.strip_suffix(".weight").ok_or_else(|| {
        Error::backend(format!(
            "encoded ordinary linear parameter {weight_name:?} must end in .weight"
        ))
    });
    match format {
        LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => LinearFormatSpec::unscaled(format),
        LinearFormat::E4M3BlockFp8(_) => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.weight_scale_inv"), "scale")?,
            )
        }
        LinearFormat::MxFp4 => {
            let prefix = prefix?;
            LinearFormatSpec::scaled(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
            )
        }
        LinearFormat::Affine(_) => {
            let prefix = prefix?;
            LinearFormatSpec::affine(
                format,
                companion(weight_name, format!("{prefix}.scales"), "scale")?,
                companion(weight_name, format!("{prefix}.biases"), "affine-bias")?,
            )
        }
    }
}
fn old_construct<B: NeuralBackend>(
    spec: StaticModuleSpec,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModules<B>, Error> {
    let embeddings = B::embedding(
        EmbeddingSpec {
            vocabulary: spec.vocabulary,
            dimensions: spec.hidden_size,
            weight: ParameterSpec::trainable(&spec.embedding_weight).map_err(Error::backend)?,
            format: old_format(&spec.embedding_weight, spec.embedding_quantization.into())?,
        },
        context,
    )?;
    let normalization_weight =
        ParameterSpec::trainable(&spec.normalization_weight).map_err(Error::backend)?;
    let norm = B::normalization(
        eredu_nn::NormalizationConstructionSpec {
            groups: spec.normalization_groups,
            dimensions: spec.hidden_size,
            epsilon: spec.normalization_epsilon,
            scale: if spec.normalization_offset == 0.0 {
                eredu_nn::NormalizationScale::Learned(normalization_weight)
            } else {
                eredu_nn::NormalizationScale::LearnedOffset {
                    weight: normalization_weight,
                    offset: spec.normalization_offset,
                }
            },
        },
        context,
    )?;
    let lm_head = if spec.tied_head {
        None
    } else {
        Some(B::linear(
            LinearSpec {
                input: spec.hidden_size,
                output: spec.vocabulary,
                weight: ParameterSpec::trainable(&spec.head_weight).map_err(Error::backend)?,
                bias: None,
                format: old_format(&spec.head_weight, spec.head_format)?,
            },
            context,
        )?)
    };
    Ok(StaticModules {
        embeddings,
        norm,
        lm_head,
    })
}
fn old_parallel<B: eredu_nn::DistributedNeuralBackend>(
    spec: StaticModuleSpec,
    embedding_range: VocabularyParallelRange,
    output_range: Option<VocabularyParallelRange>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModules<B>, Error> {
    embedding_range.validate_global_rows(spec.vocabulary)?;
    let embeddings = B::vocabulary_parallel_embedding(
        EmbeddingSpec {
            vocabulary: spec.vocabulary,
            dimensions: spec.hidden_size,
            weight: ParameterSpec::trainable(&spec.embedding_weight).map_err(Error::backend)?,
            format: old_format(&spec.embedding_weight, spec.embedding_quantization.into())?,
        },
        embedding_range,
        context,
    )?;
    let normalization_weight =
        ParameterSpec::trainable(&spec.normalization_weight).map_err(Error::backend)?;
    let norm = B::normalization(
        eredu_nn::NormalizationConstructionSpec {
            groups: spec.normalization_groups,
            dimensions: spec.hidden_size,
            epsilon: spec.normalization_epsilon,
            scale: if spec.normalization_offset == 0.0 {
                eredu_nn::NormalizationScale::Learned(normalization_weight)
            } else {
                eredu_nn::NormalizationScale::LearnedOffset {
                    weight: normalization_weight,
                    offset: spec.normalization_offset,
                }
            },
        },
        context,
    )?;
    let lm_head = match (spec.tied_head, output_range) {
        (true, None) => None,
        (true, Some(_)) => {
            return Err(Error::backend(
                "tied decoder output must not declare separate vocabulary ownership",
            ));
        }
        (false, None) => {
            return Err(Error::backend(
                "untied decoder output is missing vocabulary ownership",
            ));
        }
        (false, Some(range)) => {
            range.validate_global_rows(spec.vocabulary)?;
            Some(B::vocabulary_parallel_linear(
                LinearSpec {
                    input: spec.hidden_size,
                    output: spec.vocabulary,
                    weight: ParameterSpec::trainable(&spec.head_weight).map_err(Error::backend)?,
                    bias: None,
                    format: old_format(&spec.head_weight, spec.head_format)?,
                },
                range,
                context,
            )?)
        }
    };
    Ok(StaticModules {
        embeddings,
        norm,
        lm_head,
    })
}

fn spec(tied: bool, affine: bool, offset: f32) -> StaticModuleSpec {
    StaticModuleSpec {
        normalization_groups: Some(2),
        embedding_weight: "static.embedding.weight".into(),
        normalization_weight: "static.norm.weight".into(),
        head_weight: "static.head.weight".into(),
        vocabulary: 11,
        hidden_size: 64,
        normalization_epsilon: 1e-5,
        normalization_offset: offset,
        embedding_quantization: affine.then(|| WeightQuantization::Affine(Default::default())),
        head_format: if affine {
            LinearFormat::Affine(Default::default())
        } else {
            LinearFormat::Dense
        },
        tied_head: tied,
    }
}
fn readout(
    model: &mut StaticModules<NumericBackend>,
    tokens: &[usize],
    context: &NumericContext,
) -> NumericTensor {
    let embedded = model
        .embeddings
        .forward(&NumericTensor::token_ids(tokens), context)
        .unwrap();
    let hidden = model.norm.forward(&embedded, context).unwrap();
    match &mut model.lm_head {
        Some(head) => head.forward(&hidden, context).unwrap(),
        None => model.embeddings.as_linear(&hidden, context).unwrap(),
    }
}
#[test]
fn static_shared_ordinary_constructor_matches_nonzero_independent_legacy_readout() {
    let context = NumericContext::default();
    for tied in [false, true] {
        for affine in [false, true] {
            for offset in [0.0, -0.0, 1.0] {
                let source = spec(tied, affine, offset);
                let mut expected =
                    old_construct::<NumericBackend>(source.clone(), &context).unwrap();
                let mut actual =
                    StaticModules::<NumericBackend>::from_spec(source, &context).unwrap();
                let actual_metadata = eredu_nn::validate_parameter_topology(&actual).unwrap();
                assert_eq!(
                    actual_metadata,
                    eredu_nn::validate_parameter_topology(&expected).unwrap()
                );
                assert_eq!(
                    actual_metadata.len(),
                    if affine {
                        if tied {
                            4
                        } else {
                            7
                        }
                    } else if tied {
                        2
                    } else {
                        3
                    }
                );
                let reference = readout(&mut expected, &[2, 7], &context);
                let observed = readout(&mut actual, &[2, 7], &context);
                assert_eq!(observed.shape, [1, 2, 11]);
                assert!(observed.data.iter().any(|v| v.abs() > 1e-6));
                assert_tensor_exact(&observed, &reference, "shared static construction");
                assert_finite_values(&observed.data, "shared static construction");
            }
        }
    }
}
#[test]
fn static_shared_parallel_constructor_matches_legacy_local_rows_and_error_precedence() {
    let context = NumericContext::default();
    let range = VocabularyParallelRange {
        global_vocabulary: 11,
        local: 4..8,
    };
    for tied in [false, true] {
        let source = spec(tied, false, 1.0);
        let output = (!tied).then(|| range.clone());
        let mut expected =
            old_parallel::<NumericBackend>(source.clone(), range.clone(), output.clone(), &context)
                .unwrap();
        let mut actual = StaticModules::<NumericBackend>::from_parallel_spec(
            source,
            range.clone(),
            output,
            &context,
        )
        .unwrap();
        assert_eq!(actual.embeddings.weight.shape, [4, 64]);
        if let Some(head) = &actual.lm_head {
            assert_eq!(head.weight.shape, [4, 64]);
        }
        assert_eq!(
            eredu_nn::validate_parameter_topology(&actual).unwrap(),
            eredu_nn::validate_parameter_topology(&expected).unwrap()
        );
        let reference = readout(&mut expected, &[4, 7], &context);
        let observed = readout(&mut actual, &[4, 7], &context);
        assert_eq!(observed.shape, [1, 2, 4]);
        assert!(observed.data.iter().any(|v| v.abs() > 1e-6));
        assert_tensor_exact(
            &observed,
            &reference,
            "shared vocabulary-local static construction",
        );
    }
    for case in 0..7 {
        let mut source = spec(false, false, 0.0);
        let mut embedding = range.clone();
        let mut output = None;
        match case {
            0 => {
                embedding.local = 8..4;
                source.embedding_weight = " ".into();
            }
            1 => {
                source.embedding_weight = " ".into();
                source.embedding_quantization = Some(WeightQuantization::Affine(
                    eredu_checkpoint::AffineQuantization {
                        group_size: 0,
                        ..Default::default()
                    },
                ));
            }
            2 => {
                source.normalization_weight = " ".into();
            }
            3 => {
                source.normalization_groups = Some(0);
            }
            4 => {}
            5 => {
                source.tied_head = true;
                output = Some(range.clone());
            }
            6 => {
                source.head_weight = " ".into();
                output = Some(VocabularyParallelRange {
                    global_vocabulary: 12,
                    local: 4..8,
                });
            }
            _ => unreachable!(),
        }
        let expected = old_parallel::<NumericBackend>(
            source.clone(),
            embedding.clone(),
            output.clone(),
            &context,
        )
        .err()
        .expect("legacy rejection");
        let actual = StaticModules::<NumericBackend>::from_parallel_spec(
            source, embedding, output, &context,
        )
        .err()
        .expect("shared rejection");
        assert_eq!(actual.to_string(), expected.to_string(), "case {case}");
        assert_eq!(
            std::error::Error::source(&actual).is_some(),
            std::error::Error::source(&expected).is_some()
        );
    }
}
