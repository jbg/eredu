mod v4_components {
    use super::*;
    use eredu_core::{
        capture::*, intervention::*, parameters::*, ModelConfigurationResolver,
        TextGenerationBackend,
    };

    fn tensor<'a>(step: &'a CapturedStep, path: &str) -> &'a [f32] {
        let record = step
            .records
            .iter()
            .find(|record| record.path == path)
            .unwrap_or_else(|| panic!("missing planned reconstruction capture {path}"));
        assert_eq!(record.outcome, CaptureOutcome::Captured, "{path}");
        let Some(CapturePayload::Tensor(value)) = &record.payload else {
            panic!("full tensor {path}")
        };
        let eredu_core::TensorObservationData::F32(values) = value.data() else {
            panic!("F32 {path}")
        };
        values
    }

    fn reconstruct_readout(
        step: &CapturedStep,
        readout: &eredu_core::component::ComponentReadoutEquation,
        gain: &[f32],
        weight: &[f32],
        score_weights: &[Vec<f32>],
        rows: usize,
    ) {
        let mixing = readout.stream_residual.as_ref().unwrap();
        let hidden = gain.len();
        let streams = mixing.streams;
        let mut terms = vec![match &mixing.base {
            eredu_core::component::ComponentStreamBase::Broadcast { input } => {
                let base = tensor(step, input);
                (0..rows * streams * hidden)
                    .map(|i| base[i / (streams * hidden) * hidden + i % hidden] as f64)
                    .collect::<Vec<_>>()
            }
            eredu_core::component::ComponentStreamBase::Streams { input } => {
                tensor(step, input).iter().map(|v| *v as f64).collect()
            }
        }];
        let gamma = |operations: usize| {
            let nu = operations as f64 * f64::from(f32::EPSILON) / 2.;
            assert!(nu < 1.);
            nu / (1. - nu)
        };
        let mut stream_error = vec![0_f64; rows * streams * hidden];
        for cycle in &mixing.cycles {
            let pre = tensor(step, &cycle.pre);
            let incoming = tensor(step, &cycle.input);
            let collapsed = tensor(step, &cycle.collapsed);
            for row in 0..rows {
                for h in 0..hidden {
                    let sum = (0..streams)
                        .map(|s| {
                            incoming[(row * streams + s) * hidden + h] as f64
                                * pre[row * streams + s] as f64
                        })
                        .sum::<f64>();
                    let magnitude = (0..streams)
                        .map(|s| {
                            (f64::from(incoming[(row * streams + s) * hidden + h])
                                * f64::from(pre[row * streams + s]))
                            .abs()
                        })
                        .sum::<f64>();
                    let bound = (gamma(2 * streams) * magnitude).max(2e-5);
                    assert!(
                        (sum - f64::from(collapsed[row * hidden + h])).abs() <= bound,
                        "{}: signed sum {sum}, native {}, bound {bound}",
                        cycle.collapsed,
                        collapsed[row * hidden + h]
                    );
                }
            }
            let coefficients = tensor(step, &cycle.combination);
            for term in &mut terms {
                let mut mixed = vec![0.0; term.len()];
                for row in 0..rows {
                    for to in 0..streams {
                        for h in 0..hidden {
                            mixed[(row * streams + to) * hidden + h] = (0..streams)
                                .map(|from| {
                                    term[(row * streams + from) * hidden + h]
                                        * coefficients[(row * streams + from) * streams + to] as f64
                                })
                                .sum();
                        }
                    }
                }
                *term = mixed;
            }
            let post = tensor(step, &cycle.post);
            let write = tensor(step, &cycle.write);
            let mut next_error = vec![0_f64; stream_error.len()];
            for row in 0..rows {
                for to in 0..streams {
                    for h in 0..hidden {
                        let mut propagated = 0.;
                        let mut magnitude = (f64::from(post[row * streams + to])
                            * f64::from(write[row * hidden + h]))
                        .abs();
                        for from in 0..streams {
                            let coefficient =
                                f64::from(coefficients[(row * streams + from) * streams + to]);
                            let source = (row * streams + from) * hidden + h;
                            propagated += coefficient.abs() * stream_error[source];
                            magnitude += (coefficient * f64::from(incoming[source])).abs();
                        }
                        next_error[(row * streams + to) * hidden + h] =
                            propagated + gamma(2 * streams + 2) * magnitude;
                    }
                }
            }
            stream_error = next_error;
            terms.push(
                (0..rows * streams * hidden)
                    .map(|i| {
                        post[i / hidden] as f64
                            * write[i / (streams * hidden) * hidden + i % hidden] as f64
                    })
                    .collect(),
            );
            for (i, actual) in tensor(step, &cycle.output).iter().enumerate() {
                let signed = terms.iter().map(|t| t[i]).sum::<f64>();
                let bound = stream_error[i].max(3e-5);
                assert!(
                    (signed - f64::from(*actual)).abs() <= bound,
                    "V4 stream cycle {}: signed {signed}, native {actual}, bound {bound}",
                    cycle.output
                );
            }
        }
        let collapse = tensor(step, &mixing.head.coefficients);
        let residual = tensor(step, &readout.residual);
        let normalized = tensor(step, &readout.normalized);
        let head_input = tensor(step, readout.projection_input.as_ref().unwrap());
        let scores = tensor(
            step,
            if readout.score_writes.is_empty() {
                &readout.linear_scores
            } else {
                assert_eq!(
                    readout.output_transform,
                    eredu_core::component::ComponentOutputTransform::Identity
                );
                &readout.logits
            },
        );
        let vocabulary = weight.len() / hidden;
        assert_eq!(score_weights.len(), readout.score_writes.len());
        for row in 0..rows {
            let rms = (residual[row * hidden..(row + 1) * hidden]
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                / hidden as f64
                + readout.normalization.epsilon.value() as f64)
                .sqrt();
            let mut reconstructed = [0.0; 2];
            let mut score_error = [0_f64; 2];
            let mut score_magnitude = [0_f64; 2];
            for term in &terms {
                for (selected, token) in [2usize, 7].into_iter().enumerate() {
                    reconstructed[selected] += (0..hidden)
                        .map(|h| {
                            let value = (0..streams)
                                .map(|s| {
                                    term[(row * streams + s) * hidden + h]
                                        * collapse[row * streams + s] as f64
                                })
                                .sum::<f64>();
                            value / rms * gain[h] as f64 * weight[token * hidden + h] as f64
                        })
                        .sum::<f64>();
                }
            }
            for h in 0..hidden {
                let collapsed = terms
                    .iter()
                    .map(|term| {
                        (0..streams)
                            .map(|s| {
                                term[(row * streams + s) * hidden + h]
                                    * collapse[row * streams + s] as f64
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                let mut bound = 0.;
                let mut magnitude = 0.;
                for s in 0..streams {
                    let i = (row * streams + s) * hidden + h;
                    let coefficient = f64::from(collapse[row * streams + s]);
                    bound += coefficient.abs() * stream_error[i];
                    magnitude +=
                        (coefficient * terms.iter().map(|term| term[i]).sum::<f64>()).abs();
                }
                bound += gamma(2 * streams) * magnitude;
                assert!(
                    (collapsed - f64::from(residual[row * hidden + h])).abs() <= bound.max(3e-5),
                    "stream head collapse"
                );
                let mathematical = f64::from(residual[row * hidden + h]) / rms * f64::from(gain[h]);
                assert!(
                    (mathematical - f64::from(normalized[row * hidden + h])).abs()
                        <= (gamma(2 * hidden + 8) * mathematical.abs()).max(2e-5),
                    "native RMS normalization"
                );
                for (selected, token) in [2usize, 7].into_iter().enumerate() {
                    let w = f64::from(weight[token * hidden + h]);
                    // Preserve the explicit normalization-rounding/input-
                    // quantization term instead of attributing it to components.
                    reconstructed[selected] +=
                        (f64::from(head_input[row * hidden + h]) - mathematical) * w;
                    score_error[selected] += bound / rms * (f64::from(gain[h]) * w).abs();
                    score_magnitude[selected] +=
                        (f64::from(head_input[row * hidden + h]) * w).abs();
                }
            }
            for (write, matrix) in readout.score_writes.iter().zip(score_weights) {
                assert_eq!(write.broadcast_axes, ["sequence"]);
                let input = tensor(step, &write.projection_input);
                let original = tensor(step, &write.output);
                let effective = tensor(step, &write.effective_output);
                assert_eq!(matrix.len(), vocabulary * input.len());
                assert_eq!(effective.len(), vocabulary);
                for (selected, token) in [2usize, 7].into_iter().enumerate() {
                    let projected = input
                        .iter()
                        .zip(&matrix[token * input.len()..(token + 1) * input.len()])
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>();
                    let magnitude = input
                        .iter()
                        .zip(&matrix[token * input.len()..(token + 1) * input.len()])
                        .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
                        .sum::<f64>();
                    assert!(
                        (projected - original[token] as f64).abs()
                            <= (gamma(2 * input.len()) * magnitude).max(3e-5),
                        "dynamic score projection {}",
                        write.weight
                    );
                    reconstructed[selected] += effective[token] as f64;
                    score_magnitude[selected] += f64::from(effective[token]).abs();
                }
            }
            let expected = [
                scores[row * vocabulary + 2] as f64,
                scores[row * vocabulary + 7] as f64,
            ];
            for selected in 0..2 {
                score_error[selected] +=
                    gamma(2 * hidden + 2 * readout.score_writes.len()) * score_magnitude[selected];
                assert!(
                    (reconstructed[selected] - expected[selected]).abs()
                        <= score_error[selected].max(3e-5),
                    "V4 native selected score: {} != {}, bound {}",
                    reconstructed[selected],
                    expected[selected],
                    score_error[selected]
                );
            }
            assert!(
                ((reconstructed[0] - reconstructed[1]) - (expected[0] - expected[1])).abs()
                    <= (score_error[0] + score_error[1]).max(3e-5),
                "V4 native target-alternative score"
            );
        }
    }

    pub(super) fn verify_partitioned_readout(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        readout: &eredu_core::component::ComponentReadoutEquation,
        steps: &[SharedCapturedStep],
    ) {
        let facts = MlxBackend::parameter_discovery(runtime).unwrap();
        // Each query exchanges the complete catalog before reading its selected
        // tensor. Reserve bounded discovery envelopes as well as the small
        // outputs; the eight-rank quantized catalog alone exceeds 8 MiB. These
        // are cumulative operation charges, not simultaneously resident weights.
        let queries = 2 + readout.score_writes.len() as u64;
        let limits = facts
            .usage
            .checked_add(CaptureUsage {
                captures: 128,
                retained_bytes: (256 << 20) * queries + (8 << 20),
                host_bytes: (1 << 30) * queries + (64 << 20),
                encoded_bytes: (128 << 20) * queries + (8 << 20),
            })
            .unwrap();
        let parameters: Vec<_> = [
            readout.normalization.gain.as_ref().unwrap(),
            &readout.weight,
        ]
        .into_iter()
        .chain(readout.score_writes.iter().map(|write| &write.weight))
        .map(|id| {
            let slot = facts.parameters.iter().find(|p| &p.id == id).unwrap();
            MlxBackend::query_parameter(
                runtime,
                &facts.identity,
                id,
                ParameterRegion {
                    starts: vec![0; slot.shape.len()],
                    shape: slot.shape.clone(),
                },
                limits,
            )
            .unwrap_or_else(|error| panic!("partitioned readout query {id}: {error}"))
            .values
        })
        .collect();
        let hidden = parameters[0].len();
        for step in steps {
            let rows = tensor(step, &readout.residual).len() / hidden;
            reconstruct_readout(
                step,
                readout,
                &parameters[0],
                &parameters[1],
                &parameters[2..],
                rows,
            );
        }
    }

    pub(super) fn verify_independent_overlay(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        checkpoint: &Path,
        stream: &Stream,
        edits: &[ParameterEdit],
    ) {
        let bytes = std::fs::read(checkpoint.join("model.safetensors")).unwrap();
        let source = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let mut tensors = source
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    name,
                    (view.dtype(), view.shape().to_vec(), view.data().to_vec()),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        // Rewrite the published individual expert matrices independently of
        // the runtime's packed bank binding and overlay publication machinery.
        for edit in edits {
            let ParameterUpdate::Add { values } = &edit.update else {
                panic!("fixture additive edit")
            };
            for (linear, delta) in values.iter().enumerate() {
                let mut remainder = linear;
                let mut coordinates = vec![0; edit.region.shape.len()];
                for axis in (0..coordinates.len()).rev() {
                    coordinates[axis] = edit.region.starts[axis] as usize
                        + remainder % edit.region.shape[axis] as usize;
                    remainder /= edit.region.shape[axis] as usize;
                }
                let (name, coordinates) = if let Some(root) =
                    edit.parameter.strip_suffix(".switch_mlp.gate_up_proj")
                {
                    assert_eq!(coordinates.len(), 3);
                    let half = edit.parameter_shape[1] as usize / 2;
                    let field = if coordinates[1] < half { "w1" } else { "w3" };
                    (
                        format!("{root}.experts.{}.{field}.weight", coordinates[0]),
                        vec![coordinates[1] % half, coordinates[2]],
                    )
                } else if let Some(root) = edit.parameter.strip_suffix(".switch_mlp.down_proj") {
                    assert_eq!(coordinates.len(), 3);
                    (
                        format!("{root}.experts.{}.w2.weight", coordinates[0]),
                        vec![coordinates[1], coordinates[2]],
                    )
                } else {
                    (edit.parameter.clone(), coordinates)
                };
                let (dtype, shape, data) = tensors.get_mut(&name).unwrap();
                assert_eq!(*dtype, safetensors::Dtype::F32, "{name}");
                assert_eq!(coordinates.len(), shape.len());
                let offset =
                    coordinates
                        .iter()
                        .zip(shape.iter())
                        .fold(0, |offset, (coordinate, extent)| {
                            assert!(coordinate < extent);
                            offset * extent + coordinate
                        })
                        * 4;
                let before = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                data[offset..offset + 4].copy_from_slice(&(before + delta).to_le_bytes());
            }
        }
        let independent = tempfile::tempdir().unwrap();
        std::fs::copy(
            checkpoint.join("config.json"),
            independent.path().join("config.json"),
        )
        .unwrap();
        safetensors::serialize_to_file(
            tensors.iter().map(|(name, (dtype, shape, bytes))| {
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            }),
            None,
            &independent.path().join("model.safetensors"),
        )
        .unwrap();
        let weights_stream = fixture_weights_stream(stream);
        let backend = MlxBackend::new(stream, &weights_stream);
        let model = load_model(&backend, independent.path(), MlxLoadRequest::default()).unwrap();
        let mut reference = ModelRuntime::from_prepared(backend, model).unwrap();
        runtime.reset().unwrap();
        for prefill in [true, false, false] {
            let expected = parameter_fixture_forward(&mut reference, prefill);
            assert_parameter_predictions(
                parameter_fixture_forward(runtime, prefill),
                &expected,
                3e-5,
            );
        }
        runtime.reset().unwrap();
    }

    fn prove(device: DeviceType) {
        if !crate::tests::support::native_process::enter("capture") { return; }
        #[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
        let _streams = crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &crate::backend::managed_memory::domain(), device,
        ).unwrap().expect("qualified native stream factory");
        for residency in [
            WeightResidency::default(),
            WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
                OffloadConfig::new(None, None, 1).unwrap(),
            )),
            WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            ),
        ] {
            prove_residency(device, residency);
        }
    }

    fn prove_residency(device: DeviceType, residency: WeightResidency) {
        let root = tempfile::tempdir().unwrap();
        write_deepseek_v4_fixture(root.path(), 0);
        let config =
            serde_json::from_slice(&std::fs::read(root.path().join("config.json")).unwrap())
                .unwrap();
        let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let group = graph
            .components
            .iter()
            .find(|g| g.layer_index == 0 && g.write_input_projection.is_some())
            .unwrap();
        let factor = group.write_input_projection.as_ref().unwrap();
        let mut paths = vec![
            group.activation.clone(),
            group.effective_activation.clone(),
            factor.input.clone(),
            factor.output.clone(),
            factor.final_input.clone(),
            group.write_output.clone().unwrap(),
            "layers.0.hyper.attention.pre".into(),
            "layers.0.hyper.attention.post".into(),
            "layers.0.hyper.attention.combination".into(),
        ];
        let readout = graph.component_readout.as_ref().unwrap();
        let mixing = readout.stream_residual.as_ref().unwrap();
        let eredu_core::component::ComponentStreamBase::Broadcast { input: base } = &mixing.base
        else {
            panic!("target embedding broadcast")
        };
        paths.extend([
            base.clone(),
            mixing.head.input.clone(),
            mixing.head.coefficients.clone(),
            readout.residual.clone(),
            readout.normalized.clone(),
            readout.projection_input.clone().unwrap(),
            readout.linear_scores.clone(),
        ]);
        for cycle in &mixing.cycles {
            paths.extend([
                cycle.input.clone(),
                cycle.output.clone(),
                cycle.collapsed.clone(),
                cycle.write.clone(),
                cycle.pre.clone(),
                cycle.post.clone(),
                cycle.combination.clone(),
            ]);
        }
        paths.sort();
        paths.dedup();
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        // Keep the schema fixture private to this trial while making all read
        // and write factors, embeddings and stream coefficients distinguishable.
        let file = root.path().join("model.safetensors");
        let arrays = Array::load_safetensors(&file, &weights).unwrap();
        let arrays: Vec<_> = arrays
            .into_iter()
            .map(|(name, value)| {
                let value =
                    if value.dtype() == safemlx::Dtype::Int32 || name.ends_with("norm.weight") {
                        value
                    } else {
                        let seed = name
                            .bytes()
                            .fold(0usize, |n, b| (n * 17 + b as usize) % 101);
                        let values: Vec<_> = (0..value.size())
                            .map(|i| ((i * 37 + seed) % 101) as f32 / 1000.0 - 0.05)
                            .collect();
                        Array::from_slice(&values, value.shape())
                    };
                (name, value)
            })
            .collect();
        let staged = root.path().join("component-values.safetensors");
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), value)),
            None,
            &staged,
        )
        .unwrap();
        std::fs::rename(staged, file).unwrap();
        let backend = MlxBackend::new(&stream, &weights);
        let model = load_model(
            &backend,
            root.path(),
            MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
        for path in &paths {
            let support = discovery
                .support
                .points
                .iter()
                .find(|p| &p.path == path)
                .unwrap();
            assert_eq!(
                support.prefill,
                eredu_core::ObservationSupportStatus::Supported,
                "{path}"
            );
            assert_eq!(
                support.decode,
                eredu_core::ObservationSupportStatus::Supported,
                "{path}"
            );
        }
        let usage = CaptureUsage {
            captures: 128,
            retained_bytes: 8 << 20,
            host_bytes: 8 << 20,
            encoded_bytes: 8 << 20,
        };
        let facts = MlxBackend::parameter_discovery(&mut runtime).unwrap();
        let mut matrices = Vec::new();
        for name in [&factor.weight, &group.write_weight] {
            let slot = facts.parameters.iter().find(|p| &p.id == name).unwrap();
            matrices.push(
                MlxBackend::query_parameter(
                    &mut runtime,
                    &facts.identity,
                    name,
                    ParameterRegion {
                        starts: vec![0, 0],
                        shape: slot.shape.clone(),
                    },
                    usage,
                )
                .unwrap()
                .values,
            );
        }
        assert!(matrices
            .iter()
            .all(|m| m.iter().any(|v| *v < 0.0) && m.iter().any(|v| *v > 0.0)));
        let column = group
            .write_column(
                &eredu_core::component::ComponentId {
                    group: group.id.clone(),
                    index: 3,
                },
                &facts,
            )
            .unwrap();
        for selection in [Some(column.output), column.input].into_iter().flatten() {
            let query = MlxBackend::query_parameter(
                &mut runtime,
                &facts.identity,
                &selection.parameter.id,
                selection.region.clone(),
                usage,
            )
            .unwrap();
            let matrix = if selection.parameter.id == group.write_weight {
                &matrices[1]
            } else {
                &matrices[0]
            };
            let stride = selection.parameter.shape[1] as usize;
            let starts = &selection.region.starts;
            let shape = &selection.region.shape;
            let expected: Vec<_> = (0..shape[0] as usize)
                .flat_map(|r| {
                    let start = (starts[0] as usize + r) * stride + starts[1] as usize;
                    matrix[start..start + shape[1] as usize].iter().copied()
                })
                .collect();
            assert_eq!(query.values, expected);
        }
        let mut readout_parameters = Vec::new();
        for name in [
            readout.normalization.gain.as_ref().unwrap(),
            &readout.weight,
        ] {
            let slot = facts.parameters.iter().find(|p| &p.id == name).unwrap();
            readout_parameters.push(
                MlxBackend::query_parameter(
                    &mut runtime,
                    &facts.identity,
                    name,
                    ParameterRegion {
                        starts: vec![0; slot.shape.len()],
                        shape: slot.shape.clone(),
                    },
                    usage,
                )
                .unwrap()
                .values,
            );
        }
        let capture = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: paths
                .iter()
                .map(|path| CaptureSelection {
                    id: path.clone(),
                    path: path.clone(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::FullTensor,
                })
                .collect(),
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage.checked_mul(32).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 3,
            },
        )
        .unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(3),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let mut runs = Vec::new();
        for keep in [None, Some(false), Some(true)] {
            runtime.reset().unwrap();
            let intervention = keep.map(|keep_selected| {
                InterventionPlan {
                    schema_version: INTERVENTION_SCHEMA_VERSION,
                    operations: vec![InterventionOperation {
                        id: "v4-channel".into(),
                        target: group.activation.clone(),
                        schedule: CaptureSchedule {
                            prefill: true,
                            decode: false,
                            ..Default::default()
                        },
                        slices: vec![CaptureSlice {
                            axis: "sequence".into(),
                            start: 1,
                            end: 2,
                            stride: 1,
                        }],
                        action: InterventionAction::MaskComponents {
                            dtype: InterventionDtype::Float32,
                            indices: vec![3],
                            keep_selected,
                        },
                        evidence: InterventionEvidence::Preview { max_elements: 64 },
                    }],
                }
                .admit(
                    &MlxBackend::intervention_discovery(&runtime).unwrap(),
                    capture.request(),
                    "V4 grouped channel trial",
                )
                .unwrap()
            });
            let mut generation = eredu_core::ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2, 3],
                TextGenerationConfig::new(sampling),
                ComponentCaptureController::default(),
            )
            .unwrap();
            if let Some(intervention) = intervention {
                generation
                    .enable_interventions(capture.clone(), intervention)
                    .unwrap();
            } else {
                generation.enable_capture(capture.clone()).unwrap();
            }
            let mut steps = Vec::new();
            for prediction in 0..3 {
                generation.next().unwrap().unwrap();
                let step = generation.take_captured_delivery().unwrap().unwrap();
                let tokens = if prediction == 0 { 3 } else { 1 };
                let channels = tensor(&step, &group.effective_activation);
                let input = tensor(&step, &factor.input);
                let write = tensor(&step, group.write_output.as_ref().unwrap());
                let per_group = group.count / factor.groups;
                let latent = factor.groups * factor.rank;
                let hidden = write.len() / tokens;
                assert_eq!(channels.len(), tokens * group.count);
                assert!(channels.iter().any(|v| v.abs() > 1e-6));
                assert_eq!(
                    tensor(&step, &factor.output),
                    tensor(&step, &factor.final_input)
                );
                for t in 0..tokens {
                    for out in 0..hidden {
                        let mut sum = 0.0f64;
                        for g in 0..factor.groups {
                            for c in 0..per_group {
                                let value = channels[t * group.count + g * per_group + c];
                                assert_eq!(value, input[(g * tokens + t) * per_group + c]);
                                for r in 0..factor.rank {
                                    sum += value as f64
                                        * matrices[0][(g * factor.rank + r) * per_group + c] as f64
                                        * matrices[1][out * latent + g * factor.rank + r] as f64;
                                }
                            }
                        }
                        assert!(
                            (sum - write[t * hidden + out] as f64).abs() < 2e-5,
                            "V4 grouped reconstruction"
                        );
                    }
                }
                reconstruct_readout(
                    &step,
                    readout,
                    &readout_parameters[0],
                    &readout_parameters[1],
                    &[],
                    tokens,
                );
                steps.push(step);
            }
            runs.push(steps);
        }
        for (run, keep) in [(1, false), (2, true)] {
            let before = tensor(&runs[run][0], &group.activation);
            let after = tensor(&runs[run][0], &group.effective_activation);
            assert_eq!(before, tensor(&runs[0][0], &group.activation));
            for i in 0..before.len() {
                assert_eq!(
                    after[i],
                    if i / group.count == 1 && ((i % group.count == 3) != keep) {
                        0.0
                    } else {
                        before[i]
                    }
                );
            }
            assert_ne!(
                tensor(&runs[run][0], group.write_output.as_ref().unwrap()),
                tensor(&runs[0][0], group.write_output.as_ref().unwrap())
            );
        }
    }

    #[test]
    #[ignore = "requires native MLX CPU execution"]
    fn native_v4_grouped_component_capture_cpu() {
        prove(DeviceType::Cpu);
    }
    #[test]
    #[ignore = "requires native MLX Metal execution"]
    fn native_v4_grouped_component_capture_metal() {
        prove(DeviceType::Gpu);
    }
}
