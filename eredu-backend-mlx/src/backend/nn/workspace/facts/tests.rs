use super::*;

type Worker = for<'a, 'b> fn(
    WorkspaceOperationView<'a>,
    MetalAllocationFacts,
    &mut Emitter<'b>,
) -> FactResult<Option<WorkspaceOperationFacts>>;

fn operation(kind: WorkspaceOperationKind, input: &[i32], output: &[i32]) -> WorkspaceOperation {
    WorkspaceOperation {
        kind,
        inputs: vec![WorkspaceLayout::new(input, WorkspaceDtype::Float32).unwrap()],
        outputs: vec![WorkspaceLayout::new(output, WorkspaceDtype::Float32).unwrap()],
    }
}

fn compare(worker: Worker, operation: &WorkspaceOperation, scratch: u64, capacity: u64) {
    let allocation = MetalAllocationFacts { page_size: 4096 };
    let run = |sink: &mut Emitter<'_>| worker(operation.as_view(), allocation, sink);
    let expected = run(&mut Emitter::count()).unwrap().unwrap();
    assert_eq!(expected.scratch_bytes, scratch);
    let mut outputs = vec![WorkspaceOutputEffect::AliasInput(usize::MAX); expected.layout.outputs];
    let mut aliases = vec![usize::MAX; expected.layout.aliases];
    let mut assumptions = vec![0xFF; expected.layout.assumption_bytes];
    assert_eq!(
        write(
            run,
            WorkspaceEffectDestination {
                outputs: &mut outputs,
                aliases: &mut aliases,
                assumptions: &mut assumptions,
            }
        )
        .unwrap(),
        Some(expected)
    );
    let legacy = ordinary(run).unwrap().unwrap();
    assert_eq!(legacy.scratch_bytes, scratch);
    assert_eq!(legacy.assumptions.as_bytes(), assumptions);
    for (flat, old) in outputs.iter().zip(&legacy.outputs) {
        assert_eq!(flat.as_view(&aliases), Some(old.as_view()));
        match flat {
            WorkspaceOutputEffect::Allocate(bytes)
            | WorkspaceOutputEffect::AllocateOrAliasInputs { bytes, .. } => {
                assert_eq!(*bytes, capacity)
            }
            _ => panic!("this fixture declares a possible allocation"),
        }
    }
}

#[test]
fn actual_reduction_softmax_and_storage_workers_emit_the_same_exact_ordinary_facts() {
    // Independent small-buffer arithmetic: native capacity is 2*n-1 below
    // one page. Input [2,3] is 24 bytes ->47; [2] is8 ->15; scalar4 ->7.
    compare(
        super::super::storage::emit,
        &operation(WorkspaceOperationKind::DeepCopy, &[2, 3], &[2, 3]),
        47,
        47,
    );
    compare(
        super::super::storage::emit,
        &operation(WorkspaceOperationKind::Contiguous, &[2, 3], &[2, 3]),
        0,
        47,
    );
    compare(
        super::super::reduction::emit,
        &operation(
            WorkspaceOperationKind::Reduction("sum", 1, false),
            &[2, 3],
            &[2],
        ),
        47,
        15,
    );
    // Mean adds three output buffers plus a scalar to the sum envelope.
    compare(
        super::super::reduction::emit,
        &operation(
            WorkspaceOperationKind::Reduction("mean", 1, true),
            &[2, 3],
            &[2, 1],
        ),
        99,
        15,
    );
    compare(
        super::super::softmax::emit,
        &operation(
            WorkspaceOperationKind::Reduction("softmax", 1, false),
            &[2, 3],
            &[2, 3],
        ),
        0,
        47,
    );
    // General-axis softmax: 8*47 +2*23 +2*(23+47), minus output47.
    compare(
        super::super::softmax::emit,
        &operation(
            WorkspaceOperationKind::Reduction("softmax", 0, false),
            &[2, 3],
            &[2, 3],
        ),
        515,
        47,
    );
}

#[test]
fn actual_worker_rejects_every_inexact_destination_and_late_geometry_without_writes() {
    let allocation = MetalAllocationFacts { page_size: 4096 };
    let mut op = operation(
        WorkspaceOperationKind::SliceUpdate { starts: vec![0, 0] },
        &[2, 3],
        &[2, 3],
    );
    op.inputs
        .push(WorkspaceLayout::new(&[1, 3], WorkspaceDtype::Float32).unwrap());
    let run = |sink: &mut Emitter<'_>| super::super::storage::emit(op.as_view(), allocation, sink);
    let facts = run(&mut Emitter::count()).unwrap().unwrap();
    assert_eq!(facts.layout.outputs, 1);
    assert_eq!(facts.layout.aliases, 2);
    assert_eq!(facts.scratch_bytes, 69);
    for slot in 0..3 {
        for short in [true, false] {
            let mut lengths = [
                facts.layout.outputs,
                facts.layout.aliases,
                facts.layout.assumption_bytes,
            ];
            lengths[slot] = if short {
                lengths[slot] - 1
            } else {
                lengths[slot] + 1
            };
            let mut outputs = vec![WorkspaceOutputEffect::AliasOutput(123); lengths[0]];
            let mut aliases = vec![456; lengths[1]];
            let mut text = vec![0xA5; lengths[2]];
            assert!(matches!(
                write(
                    run,
                    WorkspaceEffectDestination {
                        outputs: &mut outputs,
                        aliases: &mut aliases,
                        assumptions: &mut text,
                    }
                ),
                Err(error) if matches!(error.cause(), MlxWorkspaceFactCause::Destination(_))
            ));
            assert!(outputs
                .iter()
                .all(|value| *value == WorkspaceOutputEffect::AliasOutput(123)));
            assert!(aliases.iter().all(|value| *value == 456));
            assert!(text.iter().all(|value| *value == 0xA5));
        }
    }
    // Source/output bytes are valid; the later update extent is invalid. It
    // must win over deliberately wrong destination lengths, before any write.
    op.inputs[1] = WorkspaceLayout::new(&[3, 3], WorkspaceDtype::Float32).unwrap();
    let mut outputs = [WorkspaceOutputEffect::AliasOutput(123); 2];
    let mut aliases = [456; 3];
    let mut text = [0xA5; 7];
    let error = write(
        |sink| super::super::storage::emit(op.as_view(), allocation, sink),
        WorkspaceEffectDestination {
            outputs: &mut outputs,
            aliases: &mut aliases,
            assumptions: &mut text,
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        MlxWorkspaceFactError::descriptor("invalid Metal storage-transform descriptor")
    );
    assert_eq!(outputs, [WorkspaceOutputEffect::AliasOutput(123); 2]);
    assert_eq!(aliases, [456; 3]);
    assert_eq!(text, [0xA5; 7]);
}

#[test]
fn borrowed_reduction_has_no_rank_cap_and_unknown_emission_preserves_destinations() {
    let mut input = vec![1; 513];
    input[256] = 3;
    let mut output = input.clone();
    output.remove(256);
    // Input12 ->23 and output4 ->7, with no partial accumulator.
    compare(
        super::super::reduction::emit,
        &operation(
            WorkspaceOperationKind::Reduction("sum", 256, false),
            &input,
            &output,
        ),
        23,
        7,
    );
    let unknown = operation(
        WorkspaceOperationKind::Elementwise("not_a_storage_transform"),
        &[1],
        &[1],
    );
    let mut outputs = [WorkspaceOutputEffect::AliasOutput(123)];
    let mut aliases = [456];
    let mut text = [0xA5];
    assert_eq!(
        write(
            |sink| super::super::storage::emit(
                unknown.as_view(),
                MetalAllocationFacts { page_size: 4096 },
                sink
            ),
            WorkspaceEffectDestination {
                outputs: &mut outputs,
                aliases: &mut aliases,
                assumptions: &mut text
            }
        )
        .unwrap(),
        None
    );
    assert_eq!(outputs, [WorkspaceOutputEffect::AliasOutput(123)]);
    assert_eq!(aliases, [456]);
    assert_eq!(text, [0xA5]);
}

#[test]
fn closed_assumption_count_and_fill_preserve_utf8_and_scalar_formatting() {
    let expected = format!(
        "μ={}; page={}; optional={:?}; flag={}",
        i64::MIN,
        u64::MAX,
        Some(32_u32),
        true
    );
    let len = text_length(format_args!(
        "μ={}; page={}; optional={:?}; flag={}",
        i64::MIN,
        u64::MAX,
        Some(32_u32),
        true
    ))
    .unwrap();
    let mut bytes = vec![0xFF; len];
    assert_eq!(
        write_text(
            format_args!(
                "μ={}; page={}; optional={:?}; flag={}",
                i64::MIN,
                u64::MAX,
                Some(32_u32),
                true
            ),
            &mut bytes
        )
        .unwrap(),
        len
    );
    assert_eq!(expected.as_bytes(), bytes);
}

#[test]
fn basic_fact_companion_preserves_full_alias_domain_empty_broadcast_and_seed_bytes() {
    let allocation = MetalAllocationFacts { page_size: 4096 };
    let mut concatenate = operation(WorkspaceOperationKind::Concatenate, &[1], &[257]);
    concatenate.inputs = (0..257)
        .map(|_| WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap())
        .collect();
    // 257 possible input casts each retain capacity(4)=7. The full output
    // stores 1028 bytes and has capacity2055; no bounded input-count shortcut.
    compare(super::super::basic::emit, &concatenate, 1799, 2055);
    let run =
        |sink: &mut Emitter<'_>| super::super::basic::emit(concatenate.as_view(), allocation, sink);
    let facts = run(&mut Emitter::count()).unwrap().unwrap();
    assert_eq!(facts.layout.aliases, 257);
    let mut output = [WorkspaceOutputEffect::AliasOutput(99)];
    let mut aliases = vec![usize::MAX; 257];
    let mut text = vec![0; facts.layout.assumption_bytes];
    write(
        run,
        WorkspaceEffectDestination {
            outputs: &mut output,
            aliases: &mut aliases,
            assumptions: &mut text,
        },
    )
    .unwrap();
    assert!(aliases.iter().copied().eq(0..257));
    assert_eq!(
        output,
        [WorkspaceOutputEffect::AllocateOrAliasInputs {
            bytes: 2055,
            alias_start: 0,
            alias_count: 257,
        }]
    );

    // Empty broadcast results can retain a larger pre-broadcast input cast.
    // Three F32 values have capacity23, even though output elements are zero.
    compare(
        super::super::basic::emit,
        &operation(
            WorkspaceOperationKind::Elementwise("square"),
            &[1, 3],
            &[0, 3],
        ),
        23,
        23,
    );
    let mut shape = vec![1; 513];
    shape[512] = 3;
    compare(
        super::super::basic::emit,
        &operation(
            WorkspaceOperationKind::Elementwise("square"),
            &shape,
            &shape,
        ),
        23,
        23,
    );
    let seed = WorkspaceOperation {
        kind: WorkspaceOperationKind::ParameterPlaceholder,
        inputs: Vec::new(),
        outputs: vec![WorkspaceLayout::new(&[], WorkspaceDtype::Bool).unwrap()],
    };
    compare(super::super::basic::emit, &seed, 0, 1);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn observed_packed_fact_worker_preserves_compact_outputs_and_whole_envelope_at_high_rank() {
    let encoding = eredu_checkpoint::LinearFormat::E4M3BlockFp8(
        eredu_checkpoint::BlockFp8Format::new(
            128,
            128,
            eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
        )
        .unwrap(),
    );
    let format = eredu_nn::LinearFormatSpec::scaled(
        encoding,
        eredu_nn::ParameterSpec::trainable("projection.scales").unwrap(),
    )
    .unwrap();
    let mut input_shape = vec![1; 259];
    input_shape[257] = 2;
    input_shape[258] = 128;
    let mut score_shape = input_shape.clone();
    score_shape[258] = 3;
    let input = WorkspaceLayout::new(&input_shape, WorkspaceDtype::Float32).unwrap();
    let weight = WorkspaceLayout::new(&[3, 128], WorkspaceDtype::Uint8).unwrap();
    let scales = WorkspaceLayout::new(&[1, 1], WorkspaceDtype::Float32).unwrap();
    let score = WorkspaceLayout::new(&score_shape, WorkspaceDtype::Float32).unwrap();
    let values = WorkspaceLayout::new(&[2, 128], WorkspaceDtype::Uint8).unwrap();
    let activation_scales = WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Float32).unwrap();
    let whole = WorkspaceOperation {
        kind: WorkspaceOperationKind::Projection(format.clone()),
        inputs: vec![input.clone(), weight.clone(), scales.clone()],
        outputs: vec![score.clone()],
    };
    // Capacity: input2047, weight767, weight-scale7, values511, scales15,
    // score47. Prepare=2*2047+7; finish=511+15+767+2*47.
    compare(super::super::packed::emit, &whole, 6014, 47);
    let prepare = WorkspaceOperation {
        kind: WorkspaceOperationKind::ProjectionPrepare(format.clone()),
        inputs: whole.inputs.clone(),
        outputs: vec![values.clone(), activation_scales.clone()],
    };
    let finish = WorkspaceOperation {
        kind: WorkspaceOperationKind::ProjectionFinish(format),
        inputs: vec![input, weight, scales, values, activation_scales],
        outputs: vec![score],
    };
    compare(super::super::packed::emit, &finish, 1387, 47);
    let allocation = MetalAllocationFacts { page_size: 4096 };
    let run =
        |sink: &mut Emitter<'_>| super::super::packed::emit(prepare.as_view(), allocation, sink);
    let measured = run(&mut Emitter::count()).unwrap().unwrap();
    assert_eq!(measured.scratch_bytes, 4101);
    assert_eq!(measured.layout.outputs, 2);
    assert_eq!(measured.layout.aliases, 0);
    let mut outputs = [WorkspaceOutputEffect::AliasOutput(99); 2];
    let mut text = vec![0xAA; measured.layout.assumption_bytes];
    assert_eq!(
        write(
            run,
            WorkspaceEffectDestination {
                outputs: &mut outputs,
                aliases: &mut [],
                assumptions: &mut text,
            },
        )
        .unwrap(),
        Some(measured)
    );
    assert_eq!(
        outputs,
        [
            WorkspaceOutputEffect::Allocate(511),
            WorkspaceOutputEffect::Allocate(15)
        ]
    );
    let ordinary = ordinary(run).unwrap().unwrap();
    assert_eq!(ordinary.scratch_bytes, 4101);
    assert_eq!(ordinary.assumptions.as_bytes(), text);
    assert!(outputs
        .iter()
        .zip(&ordinary.outputs)
        .all(|(a, b)| a.as_view(&[]) == Some(b.as_view())));
}

#[test]
fn masked_readout_fixed_failures_preserve_sources_and_leave_every_destination_untouched() {
    let f = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    let mut op = WorkspaceOperation {
        kind: WorkspaceOperationKind::MaskedOutputProjection {
            top_centroids: 1,
            mask_margin: 1.0,
        },
        inputs: vec![
            f(&[0, 3, 7]),
            f(&[8, 7]),
            f(&[0, 3, 2]),
            WorkspaceLayout::new(&[8], WorkspaceDtype::Int32).unwrap(),
        ],
        outputs: vec![f(&[0, 3, 8])],
    };
    // Empty readout still prices its explicit scalar/fill pair: each native
    // F32 minimum is four bytes with capacity seven on this mechanism.
    compare(super::super::readout::emit, &op, 7, 7);
    op.kind = WorkspaceOperationKind::MaskedOutputProjection {
        top_centroids: 0,
        mask_margin: f32::NAN,
    };
    let allocation = MetalAllocationFacts { page_size: 4096 };
    let mut outputs = [WorkspaceOutputEffect::AliasOutput(91)];
    let mut aliases = [73];
    let mut text = [0xA7; 3];
    let result = write(
        |sink| super::super::readout::emit(op.as_view(), allocation, sink),
        WorkspaceEffectDestination {
            outputs: &mut outputs,
            aliases: &mut aliases,
            assumptions: &mut text,
        },
    );
    assert_eq!(
        result,
        Err(MlxWorkspaceFactError::masked_output(
            eredu_nn::operation_geometry::MaskedOutputGeometryError
        ))
    );
    assert_eq!(outputs, [WorkspaceOutputEffect::AliasOutput(91)]);
    assert_eq!(aliases, [73]);
    assert_eq!(text, [0xA7; 3]);
    let detail = eredu_nn::operation_geometry::validate_masked_output_geometry(
        &[0, 3, 7],
        &[8, 7],
        &[0, 3, 2],
        &[8],
        0,
        f32::NAN,
    )
    .unwrap_err();
    assert_eq!(
        super::super::readout::operation_bound(&op, allocation)
            .unwrap_err()
            .to_string(),
        detail.to_string()
    );
}

#[test]
fn selected_public_companion_retains_separate_host_payload_and_exact_destinations() {
    let selected = super::super::MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 4096 },
        sdpa_blocks: None,
    };
    for (shape, bytes) in [(&[2, 3][..], 24), (&[][..], 4), (&[0, 7][..], 0)] {
        // All source and destination storage is established by this ordinary
        // test. The companion conveys facts, not an original funding witness.
        let op = WorkspaceOperation {
            kind: WorkspaceOperationKind::GeneratedF32Initialization,
            inputs: vec![],
            outputs: vec![WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap()],
        };
        let tensor = selected.operation_facts(op.as_view()).unwrap().unwrap();
        let host = selected.host_facts(op.as_view()).unwrap().unwrap();
        assert_eq!(host.bytes, bytes);
        assert!(host.assumption_bytes > 0);
        let ordinary = selected.operation_bound(&op).unwrap().unwrap();
        let ordinary_host = selected.host_workspace_bound(&op).unwrap().unwrap();
        let mut outputs = vec![WorkspaceOutputEffect::AliasOutput(97); tensor.layout.outputs];
        let mut aliases = vec![83; tensor.layout.aliases];
        let mut text = vec![0xA5; tensor.layout.assumption_bytes];
        assert_eq!(
            selected
                .write_operation_facts(
                    op.as_view(),
                    WorkspaceEffectDestination {
                        outputs: &mut outputs,
                        aliases: &mut aliases,
                        assumptions: &mut text,
                    }
                )
                .unwrap(),
            Some(tensor)
        );
        assert_eq!(tensor.scratch_bytes, ordinary.scratch_bytes);
        assert_eq!(text, ordinary.assumptions.as_bytes());
        assert_eq!(outputs.len(), ordinary.outputs.len());
        for (actual, expected) in outputs.iter().zip(&ordinary.outputs) {
            assert_eq!(actual.as_view(&aliases), Some(expected.as_view()));
        }
        let mut host_text = vec![0xA5; host.assumption_bytes];
        assert_eq!(
            selected
                .write_host_facts(
                    op.as_view(),
                    WorkspaceHostDestination {
                        assumptions: &mut host_text,
                    }
                )
                .unwrap(),
            Some(host)
        );
        assert_eq!(host_text, ordinary_host.assumptions.as_bytes());
        assert_eq!(ordinary_host.bytes, bytes);
        for length in [host.assumption_bytes - 1, host.assumption_bytes + 1] {
            let mut rejected = vec![0xA5; length];
            let error = selected
                .write_host_facts(
                    op.as_view(),
                    WorkspaceHostDestination {
                        assumptions: &mut rejected,
                    },
                )
                .unwrap_err();
            let MlxWorkspaceFactCause::Destination(cause) = error.cause() else {
                panic!("wrong cause")
            };
            assert_eq!(cause.kind, WorkspaceFactDestinationKind::HostAssumptions);
            assert_eq!(cause.expected, host.assumption_bytes);
            assert_eq!(cause.actual, length);
            assert!(rejected.iter().all(|byte| *byte == 0xA5));
        }
    }
    // This selected primitive owns a real five-byte-per-logit expansion plus
    // three U32 history entries. Tensor buffers are a separate domain.
    let penalties = operation(
        WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Penalties {
            history_positions: 3,
            repetition: true,
            additive: true,
        }),
        &[2, 3],
        &[2, 3],
    );
    let host = selected.host_facts(penalties.as_view()).unwrap().unwrap();
    assert_eq!(host.bytes, 42);
    assert_eq!(
        selected
            .host_workspace_bound(&penalties)
            .unwrap()
            .unwrap()
            .bytes,
        42
    );
}

#[test]
fn selected_public_companion_keeps_unknown_and_late_error_before_any_destination_write() {
    let selected = super::super::MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 4096 },
        sdpa_blocks: None,
    };
    let unknown = operation(WorkspaceOperationKind::Elementwise("unpriced"), &[3], &[3]);
    let mut outputs = [WorkspaceOutputEffect::AliasOutput(97)];
    let mut aliases = [83];
    let mut text = [0xA5; 7];
    assert_eq!(
        selected
            .write_operation_facts(
                unknown.as_view(),
                WorkspaceEffectDestination {
                    outputs: &mut outputs,
                    aliases: &mut aliases,
                    assumptions: &mut text,
                }
            )
            .unwrap(),
        None
    );
    assert_eq!(
        selected
            .write_host_facts(
                unknown.as_view(),
                WorkspaceHostDestination {
                    assumptions: &mut text,
                }
            )
            .unwrap(),
        None
    );
    assert_eq!(outputs, [WorkspaceOutputEffect::AliasOutput(97)]);
    assert_eq!(aliases, [83]);
    assert_eq!(text, [0xA5; 7]);
    let mut malformed = operation(
        WorkspaceOperationKind::SliceUpdate { starts: vec![0, 0] },
        &[2, 3],
        &[2, 3],
    );
    malformed
        .inputs
        .push(WorkspaceLayout::new(&[3, 3], WorkspaceDtype::Float32).unwrap());
    let fixed = selected
        .write_host_facts(
            malformed.as_view(),
            WorkspaceHostDestination {
                assumptions: &mut text,
            },
        )
        .unwrap_err();
    assert_eq!(
        fixed,
        MlxWorkspaceFactError::descriptor("invalid Metal storage-transform descriptor")
    );
    assert_eq!(text, [0xA5; 7]);
    let ordinary = selected.host_workspace_bound(&malformed).unwrap_err();
    assert_eq!(ordinary.to_string(), fixed.to_string());
    assert!(std::error::Error::source(&ordinary).is_none());
    assert!(std::error::Error::source(&fixed).is_some());
}
