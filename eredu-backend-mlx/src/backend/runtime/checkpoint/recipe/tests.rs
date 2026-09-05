use std::sync::Arc;

use safemlx::{Device, DeviceType};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

use super::*;
use eredu_checkpoint::store::SafetensorsWeightStore;

fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let dir = tempfile::tempdir().unwrap();
    let left = [1i32, 2, 3, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let right = [5i32, 6, 7, 8]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let cube = [0i32; 8]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "left",
                TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
            ),
            (
                "right",
                TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
            ),
            (
                "cube",
                TensorView::new(SafeDtype::I32, vec![2, 2, 2], &cube).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    (dir, store)
}

#[test]
fn bitwise_view_preserves_checkpoint_bytes() {
    let (_dir, store) = fixture();
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let recipe = DerivedWeightRecipe::View {
        input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
        dtype: RecipeDtype::U8,
        shape: vec![2, 8],
    };
    let metadata = recipe.infer(store.as_ref()).unwrap();
    assert_eq!(metadata.shape(), &[2, 8]);
    assert_eq!(metadata.dtype(), &RecipeDtype::U8);
    let output = recipe
        .materialize(store.as_ref(), context.stream())
        .unwrap();
    assert_eq!(
        output.evaluated().unwrap().as_slice::<u8>(),
        &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
    );
}

#[test]
fn recipe_preflight_reads_only_checkpoint_metadata() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Transpose {
        input: Box::new(DerivedWeightRecipe::source("cube", TensorSelection::Full)),
        axes: vec![0, 2, 1],
    };

    preflight_mlx_recipe(&recipe, store.as_ref()).unwrap();

    let diagnostics =
        eredu_checkpoint::store::CheckpointSource::source_diagnostics(store.as_ref()).unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
    assert!(diagnostics.payload_shard_paths.is_empty());
}

#[test]
fn lowers_logical_mxfp4_values_to_mlx_u32_storage() {
    let (_dir, store) = fixture();
    let logical = DerivedWeightRecipe::View {
        input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
        dtype: RecipeDtype::F4,
        shape: vec![1, 32],
    };

    let lowered = lower_mxfp4_recipe(logical, store.as_ref()).unwrap();
    let metadata = lowered.infer(store.as_ref()).unwrap();
    assert_eq!(metadata.shape(), &[1, 4]);
    assert_eq!(metadata.dtype(), &RecipeDtype::U32);
    assert!(matches!(
        lowered,
        DerivedWeightRecipe::View {
            dtype: RecipeDtype::U32,
            ref shape,
            ..
        } if shape == &[1, 4]
    ));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_lowered_mxfp4_recipe_materializes_u32_storage() {
    let (_dir, store) = fixture();
    let context =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let logical = DerivedWeightRecipe::View {
        input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
        dtype: RecipeDtype::F4,
        shape: vec![1, 32],
    };
    let lowered = lower_mxfp4_recipe(logical, store.as_ref()).unwrap();

    let output = lowered
        .materialize(store.as_ref(), context.stream())
        .unwrap();
    assert_eq!(output.shape(), &[1, 4]);
    assert_eq!(output.dtype(), Dtype::Uint32);
    assert_eq!(output.evaluated().unwrap().as_slice::<u32>(), &[1, 2, 3, 4]);
}

fn one_mapping_cross_shard_fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let dir = tempfile::tempdir().unwrap();
    let left = [1i32, 2, 3, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let right = [5i32, 6, 7, 8]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "left",
            TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
        )],
        None,
        &dir.path().join("model-00001-of-00002.safetensors"),
    )
    .unwrap();
    serialize_to_file(
        [(
            "right",
            TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
        )],
        None,
        &dir.path().join("model-00002-of-00002.safetensors"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "weight_map": {
                "left": "model-00001-of-00002.safetensors",
                "right": "model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let store =
        Arc::new(SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap());
    (dir, store)
}

fn source(key: &str) -> DerivedWeightRecipe {
    DerivedWeightRecipe::source(key, TensorSelection::Full)
}

#[test]
fn conservative_peak_counts_retained_inputs_and_outputs() {
    let (_directory, store) = fixture();
    let reshape = DerivedWeightRecipe::Reshape {
        input: Box::new(source("left")),
        shape: vec![4],
    };
    assert_eq!(
        reshape.peak_materialization_bytes(store.as_ref()).unwrap(),
        32
    );

    let concatenate = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![source("left"), source("right")],
    };
    assert_eq!(
        concatenate
            .peak_materialization_bytes(store.as_ref())
            .unwrap(),
        64
    );
}

#[test]
fn infers_nested_stack_concatenate_slice_and_cast() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Cast {
        input: Box::new(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![source("left"), source("right")],
                },
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![
                        DerivedWeightRecipe::source(
                            "left",
                            TensorSelection::Indices {
                                axis: 0,
                                indices: vec![1, 0],
                            },
                        ),
                        source("right"),
                    ],
                },
            ],
        }),
        dtype: RecipeDtype::F32,
    };
    let metadata = recipe.infer(store.as_ref()).unwrap();
    assert_eq!(metadata.shape(), &[2, 4, 2]);
    assert_eq!(metadata.dtype(), &RecipeDtype::F32);
    assert_eq!(metadata.byte_len(), 64);
    assert_eq!(recipe.source_keys(), vec!["left", "right"]);
}

#[test]
fn materializes_ordered_expert_stack_on_cpu() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![source("right"), source("left")],
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let output = recipe.materialize(store.as_ref(), &stream).unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    assert_eq!(output.nbytes(), 32);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<i32>(),
        &[5, 6, 7, 8, 1, 2, 3, 4]
    );
}

#[test]
fn materializes_cross_shard_join_with_one_mapping() {
    let (_dir, store) = one_mapping_cross_shard_fixture();
    let recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![source("left"), source("right")],
        }],
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let output = recipe.materialize(store.as_ref(), &stream).unwrap();
    assert_eq!(output.shape(), &[1, 4, 2]);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<i32>(),
        &[1, 2, 3, 4, 5, 6, 7, 8]
    );
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert_eq!(diagnostics.touched_shard_paths.len(), 2);
    assert!(diagnostics.evictions >= 1);
}

#[test]
fn selects_an_axis_from_a_derived_source_selection() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Select {
        input: Box::new(DerivedWeightRecipe::source(
            "left",
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )),
        selection: TensorSelection::Indices {
            axis: 1,
            indices: vec![1],
        },
    };
    let metadata = recipe.infer(store.as_ref()).unwrap();
    assert_eq!(metadata.shape(), &[1, 1]);
    assert_eq!(recipe.source_keys(), vec!["left"]);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let output = recipe.materialize(store.as_ref(), &stream).unwrap();
    assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), &[2]);
}

#[test]
fn materialized_recipe_selection_has_row_major_storage() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Select {
        input: Box::new(source("cube")),
        selection: TensorSelection::Indices {
            axis: 1,
            indices: vec![1, 0],
        },
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let output = recipe.materialize(store.as_ref(), &stream).unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    assert_eq!(output.strides(), &[4, 2, 1]);
    output.evaluated().unwrap();
}

#[test]
fn bounded_selection_composes_at_the_source() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::source(
        "left",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0],
        },
    );
    let rewritten = recipe
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )
        .unwrap();
    assert_eq!(
        rewritten,
        DerivedWeightRecipe::source(
            "left",
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            }
        )
    );
}

#[test]
fn independent_axis_selections_collapse_to_one_contiguous_source_span() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Select {
        input: Box::new(DerivedWeightRecipe::source(
            "cube",
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )),
        selection: TensorSelection::Range {
            axis: 1,
            start: 0,
            end: 1,
        },
    };
    let rewritten = recipe
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 2,
                start: 0,
                end: 1,
            },
        )
        .unwrap();
    assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[1, 1, 1]);
    assert_eq!(
        rewritten,
        DerivedWeightRecipe::source(
            "cube",
            TensorSelection::Contiguous {
                offset_elements: 0,
                shape: vec![1, 1, 1],
            }
        )
    );
}

#[test]
fn bounded_selection_prunes_and_reorders_join_children() {
    let (_dir, store) = fixture();
    let concatenate = DerivedWeightRecipe::Concatenate {
        axis: 0,
        inputs: vec![source("left"), source("right")],
    };
    let rewritten = concatenate
        .select_bounded(
            store.as_ref(),
            TensorSelection::Indices {
                axis: 0,
                indices: vec![3, 0, 2],
            },
        )
        .unwrap();
    assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[3, 2]);
    assert_eq!(
        rewritten,
        DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source(
                    "right",
                    TensorSelection::Range {
                        axis: 0,
                        start: 1,
                        end: 2,
                    },
                ),
                DerivedWeightRecipe::source(
                    "left",
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
                DerivedWeightRecipe::source(
                    "right",
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
            ],
        }
    );
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let output = rewritten.materialize(store.as_ref(), &stream).unwrap();
    assert_eq!(
        output.evaluated().unwrap().as_slice::<i32>(),
        &[7, 8, 1, 2, 5, 6]
    );

    let stack = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![source("left"), source("right")],
    };
    let rewritten = stack
        .select_bounded(
            store.as_ref(),
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
        )
        .unwrap();
    assert_eq!(rewritten.source_keys(), vec!["left", "right"]);
    assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[3, 2, 2]);
    assert!(matches!(
        rewritten,
        DerivedWeightRecipe::Stack { inputs, .. }
            if inputs == vec![source("right"), source("left"), source("right")]
    ));
}

#[test]
fn bounded_selection_crosses_transpose_reshape_and_view() {
    let (_dir, store) = fixture();
    let transpose = DerivedWeightRecipe::Transpose {
        input: Box::new(source("left")),
        axes: vec![1, 0],
    };
    let rewritten = transpose
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        )
        .unwrap();
    assert!(matches!(
        rewritten,
        DerivedWeightRecipe::Transpose { input, .. }
            if matches!(
                *input,
                DerivedWeightRecipe::Source {
                    selection: TensorSelection::Range {
                        axis: 1,
                        start: 0,
                        end: 1,
                    },
                    ..
                }
            )
    ));

    let reshape = DerivedWeightRecipe::Reshape {
        input: Box::new(source("left")),
        shape: vec![4],
    };
    let rewritten = reshape
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 2,
            },
        )
        .unwrap();
    assert!(matches!(
        rewritten,
        DerivedWeightRecipe::Reshape { input, shape }
            if shape == vec![2]
                && matches!(
                    *input,
                    DerivedWeightRecipe::Source {
                        selection: TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                        ..
                    }
                )
    ));

    let view = DerivedWeightRecipe::View {
        input: Box::new(source("left")),
        dtype: RecipeDtype::U8,
        shape: vec![2, 8],
    };
    let rewritten = view
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 1,
                start: 0,
                end: 4,
            },
        )
        .unwrap();
    assert!(matches!(
        rewritten,
        DerivedWeightRecipe::View { input, shape, .. }
            if shape == vec![2, 4]
                && matches!(
                    *input,
                    DerivedWeightRecipe::Source {
                        selection: TensorSelection::Range {
                            axis: 1,
                            start: 0,
                            end: 1,
                        },
                        ..
                    }
                )
    ));
}

#[test]
fn bounded_selection_splits_a_stacked_expert_reshape_at_sources() {
    let (_dir, store) = fixture();
    let recipe = DerivedWeightRecipe::Reshape {
        input: Box::new(DerivedWeightRecipe::Stack {
            axis: 2,
            inputs: vec![source("cube"), source("cube")],
        }),
        shape: vec![2, 4, 2],
    };
    let rewritten = recipe
        .select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 1,
                start: 2,
                end: 4,
            },
        )
        .unwrap();
    assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[2, 2, 2]);
    assert!(matches!(
        rewritten,
        DerivedWeightRecipe::Reshape { input, .. }
            if matches!(
                &*input,
                DerivedWeightRecipe::Stack { inputs, .. }
                    if inputs.iter().all(|input| matches!(
                        input,
                        DerivedWeightRecipe::Source {
                            selection: TensorSelection::Range {
                                axis: 1,
                                start: 1,
                                end: 2,
                            },
                            ..
                        }
                    ))
            )
    ));
}

#[test]
fn bounded_selection_rejects_unaligned_view_geometry() {
    let (_dir, store) = fixture();
    let view = DerivedWeightRecipe::View {
        input: Box::new(source("left")),
        dtype: RecipeDtype::U8,
        shape: vec![2, 8],
    };
    assert!(matches!(
        view.select_bounded(
            store.as_ref(),
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 2,
            },
        ),
        Err(
            eredu_checkpoint::recipe::RecipeError::SelectionPushdownUnsupported {
                operation: "view",
                ..
            }
        )
    ));
}

#[test]
fn rejects_shape_and_permutation_errors_before_materialization() {
    let (_dir, store) = fixture();
    let reshape = DerivedWeightRecipe::Reshape {
        input: Box::new(source("left")),
        shape: vec![3],
    };
    assert!(matches!(
        reshape.infer(store.as_ref()),
        Err(eredu_checkpoint::recipe::RecipeError::ElementCountMismatch { .. })
    ));
    let transpose = DerivedWeightRecipe::Transpose {
        input: Box::new(source("left")),
        axes: vec![0, 0],
    };
    assert!(matches!(
        transpose.infer(store.as_ref()),
        Err(eredu_checkpoint::recipe::RecipeError::InvalidPermutation { .. })
    ));
}

#[test]
fn neg_log_materializes_negative_rates_and_rejects_nonnegative_values() {
    let dir = tempfile::tempdir().unwrap();
    let negative = [-1.0f32, -4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let invalid = [-1.0f32, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "negative",
                TensorView::new(SafeDtype::F32, vec![2], &negative).unwrap(),
            ),
            (
                "invalid",
                TensorView::new(SafeDtype::F32, vec![2], &invalid).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let recipe = |key| DerivedWeightRecipe::NegLog {
        input: Box::new(source(key)),
    };

    let output = recipe("negative").materialize(&store, &stream).unwrap();
    let output = output.evaluated().unwrap();
    assert_eq!(output.as_slice::<f32>()[0], 0.0);
    assert!((output.as_slice::<f32>()[1] - 4.0f32.ln()).abs() < 1e-6);
    assert!(matches!(
        recipe("invalid").materialize(&store, &stream),
        Err(WeightRecipeError::NonNegativeNegLogInput)
    ));
}
