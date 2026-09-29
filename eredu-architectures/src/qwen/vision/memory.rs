//! Shared tower invocation geometry. Storage facts remain invocation-local: lazy
//! execution, returned features and parameter owners do not share one lifetime.
use super::{VisionAttentionPolicy, VisionConfig, VisionMode};
use eredu_nn::{
    mechanism_memory::{element_bytes, MechanismInvocation, MechanismMemoryContract},
    AttentionArithmetic, Error, TensorElementType,
};
use std::collections::BTreeMap;

/// A compact group of equal invocations in one architecture-owned component.
#[derive(Debug, Clone)]
pub struct VisionMemoryInvocation {
    /// Stable relative component identity; independent from checkpoint container.
    pub component: String,
    /// Number of invocations with the same geometry (e.g. equal video frames).
    pub repetitions: u64,
    /// Exact operator geometry under the report's declared activation representation.
    pub invocation: MechanismInvocation,
    /// Selected native facts for one invocation, including unknowns and retention.
    pub memory: MechanismMemoryContract,
}

/// Request-specific shared vision facts, not a physical live-memory admission.
#[derive(Debug, Clone)]
pub struct VisionMemoryReport {
    /// Assumed floating activation representation. Promotion is not inferred from weights.
    pub activation_element: TensorElementType,
    /// Architecture-owned host array payload bytes across setup phases.
    /// Neither peak storage nor Vec capacity; excludes backend helper host arrays,
    /// objects and allocator overhead.
    pub host_setup_logical_bytes: u64,
    /// Logical final and DeepStack output bytes under the declared representation.
    pub retained_output_logical_bytes: u64,
    /// Compact native invocation groups. Do not sum these as simultaneous allocations.
    pub invocations: Vec<VisionMemoryInvocation>,
    /// Whole-tower facts that remain unresolved even when operator facts are available.
    pub missing: Vec<String>,
}

fn overflow() -> Error {
    Error::backend("vision memory geometry overflowed")
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or_else(overflow)
}
fn count(map: &mut BTreeMap<u64, u64>, size: u64, repetitions: u64) -> Result<(), Error> {
    if size != 0 && repetitions != 0 {
        let previous = map.get(&size).copied().unwrap_or(0);
        map.insert(size, add(previous, repetitions)?);
    }
    Ok(())
}

/// Builds full/window segment histograms without one allocation per patch or frame.
fn segments(
    config: &VisionConfig,
    grids: &[(i32, i32, i32)],
) -> Result<(u64, BTreeMap<u64, u64>, BTreeMap<u64, u64>), Error> {
    eredu_nn::sequence_layout::validate_patch_grid(grids, config.spatial_merge_size, None)
        .map_err(Error::backend)?;
    if config.mode == VisionMode::WindowScheduled && config.window_size <= 0 {
        return Err(Error::backend("vision window size must be positive"));
    }
    let merge = config.spatial_merge_size as u64;
    let unit = mul(merge, merge)?;
    let mut patches = 0;
    let mut full = BTreeMap::new();
    let mut windows = BTreeMap::new();
    for &(time, height, width) in grids {
        let (time, height, width) = (time as u64, height as u64, width as u64);
        let area = mul(height, width)?;
        patches = add(patches, mul(time, area)?)?;
        count(&mut full, area, time)?;
        if config.mode == VisionMode::WindowScheduled {
            let window = config.window_size as u64 / merge / config.patch_size as u64;
            if window == 0 {
                return Err(Error::backend("vision window has no merged patches"));
            }
            let (height, width) = (height / merge, width / merge);
            // Full rectangles and at most three partial edge/corner shapes.
            for (h, nh) in [(window, height / window), (height % window, 1)] {
                for (w, nw) in [(window, width / window), (width % window, 1)] {
                    count(
                        &mut windows,
                        mul(mul(h, w)?, unit)?,
                        mul(mul(time, nh)?, nw)?,
                    )?;
                }
            }
        }
    }
    if config.mode == VisionMode::DeepStack {
        windows = full.clone();
    }
    Ok((patches, full, windows))
}

impl VisionConfig {
    /// Logical architecture-owned host array payload across setup phases.
    /// This geometry-only calculation allocates no per-patch/frame catalog and
    /// queries no native mechanism. Pass all grids in a joint encoder invocation;
    /// their combined patch extent must fit the encoder's signed geometry.
    /// Excludes Vec capacity, backend helper arrays, native storage and pools.
    pub fn host_setup_logical_bytes(
        &self,
        grids: impl IntoIterator<Item = (i32, i32, i32)>,
    ) -> Result<u64, Error> {
        self.validate().map_err(Error::backend)?;
        let merge = self.spatial_merge_size as u64;
        let unit = mul(merge, merge)?;
        let mut patches = 0u64;
        let mut full_chunks = 0u64;
        let mut window_chunks = 0u64;
        for grid in grids {
            let n = eredu_nn::sequence_layout::validate_patch_grid(
                std::slice::from_ref(&grid),
                self.spatial_merge_size,
                None,
            )
            .map_err(Error::backend)?;
            patches = add(patches, n as u64)?;
            i32::try_from(patches).map_err(|_| overflow())?;
            let (time, height, width) = (grid.0 as u64, grid.1 as u64, grid.2 as u64);
            full_chunks = add(full_chunks, time)?;
            let windows = if self.mode == VisionMode::WindowScheduled {
                if self.window_size <= 0 {
                    return Err(Error::backend("vision window size must be positive"));
                }
                let window = self.window_size as u64 / merge / self.patch_size as u64;
                if window == 0 {
                    return Err(Error::backend("vision window has no merged patches"));
                }
                mul(
                    mul(
                        (height / merge).div_ceil(window),
                        (width / merge).div_ceil(window),
                    )?,
                    time,
                )?
            } else {
                time
            };
            window_chunks = add(window_chunks, windows)?;
        }
        // Interpolated samples hold four u32 IDs and four f32 weights, followed
        // by four corner ID/weight arrays. Gathered positions use one ID array.
        let position_bytes = mul(
            patches,
            if self.mode == VisionMode::DeepStack {
                64
            } else {
                4
            },
        )?;
        // Rotary [time,y,x] and [y,x] arrays; permutation/inverse; full/window
        // chunk vectors. These are payload totals across phases, not peak bytes.
        add(
            add(position_bytes, mul(patches, 20)?)?,
            add(
                mul(patches / unit, 8)?,
                mul(add(full_chunks, window_chunks)?, 4)?,
            )?,
        )
    }

    /// Composes exact shared-tower normalization, projection, rotary and segmented-attention geometry
    /// with side-effect-free backend facts. `element` is explicit: promotion and
    /// physical dtype are unresolved until the selected binding establishes them.
    /// This report must never be treated as a total physical-memory upper bound.
    pub fn memory_report(
        &self,
        grids: &[(i32, i32, i32)],
        element: TensorElementType,
        mut mechanism: impl FnMut(&MechanismInvocation) -> Result<MechanismMemoryContract, Error>,
        mut segment: impl FnMut(&MechanismInvocation) -> Result<MechanismMemoryContract, Error>,
    ) -> Result<VisionMemoryReport, Error> {
        self.validate().map_err(Error::backend)?;
        if !matches!(
            element,
            TensorElementType::F16
                | TensorElementType::Bf16
                | TensorElementType::F32
                | TensorElementType::F64
        ) {
            return Err(Error::backend(
                "vision activation representation must be floating point",
            ));
        }
        let mut report = VisionMemoryReport {
            activation_element: element,
            host_setup_logical_bytes: 0,
            retained_output_logical_bytes: 0,
            invocations: Vec::new(),
            missing: Vec::new(),
        };
        if grids.is_empty() {
            return Ok(report);
        }
        let (patches, full, windows) = segments(self, grids)?;
        let hidden = self.hidden_size as u64;
        let unit = mul(
            self.spatial_merge_size as u64,
            self.spatial_merge_size as u64,
        )?;
        let merged = patches / unit;
        report.host_setup_logical_bytes = self.host_setup_logical_bytes(grids.iter().copied())?;
        report.retained_output_logical_bytes = mul(
            mul(
                mul(merged, self.out_hidden_size as u64)?,
                self.deepstack_layer_count() as u64 + 1,
            )?,
            element_bytes(element),
        )?;
        let invocation = eredu_nn::multimodal::FlattenedPatchSpec {
            channels: self.in_channels,
            temporal: self.temporal_patch_size,
            height: self.patch_size,
            width: self.patch_size,
            output: self.hidden_size,
        }
        .memory_invocation(patches, element, true)?;
        // Patch embedding uses Tensor::linear after reshaping a dense kernel,
        // not the backend's constructed/optimized linear module. Keep the
        // generic unbound projection query and unknown weight representation.
        let memory = mechanism(&invocation)?;
        memory.validate()?;
        report.invocations.push(VisionMemoryInvocation {
            component: "patch_embed.proj".into(),
            repetitions: 1,
            invocation,
            memory,
        });
        let invocation = self
            .rotary_spec()?
            .memory_invocation(&[patches, 2], TensorElementType::I32)?;
        let memory = mechanism(&invocation)?;
        memory.validate()?;
        report.invocations.push(VisionMemoryInvocation {
            component: "rotary.position_embeddings".into(),
            repetitions: 1,
            invocation,
            memory,
        });
        let projected = |report: &mut VisionMemoryReport,
                         mechanism: &mut dyn FnMut(
            &MechanismInvocation,
        )
            -> Result<MechanismMemoryContract, Error>,
                         name: String,
                         rows,
                         input,
                         output|
         -> Result<(), Error> {
            let invocation = MechanismInvocation::Projection {
                rows,
                input,
                output,
                format: self.linear_format(&format!("{name}.weight")),
                element,
                weight_element: None,
                bias: true,
                bias_element: None,
            };
            let memory = mechanism(&invocation)?;
            memory.validate()?;
            report.invocations.push(VisionMemoryInvocation {
                component: name,
                repetitions: 1,
                invocation,
                memory,
            });
            Ok(())
        };
        let normalized = |report: &mut VisionMemoryReport,
                          mechanism: &mut dyn FnMut(
            &MechanismInvocation,
        )
            -> Result<MechanismMemoryContract, Error>,
                          name: String,
                          rows,
                          width|
         -> Result<(), Error> {
            let invocation = MechanismInvocation::LayerNormalization {
                rows,
                width,
                element,
                weight: true,
                weight_element: None,
                bias: true,
                bias_element: None,
            };
            invocation.logical_values()?;
            let memory = mechanism(&invocation)?;
            memory.validate()?;
            report.invocations.push(VisionMemoryInvocation {
                component: name,
                repetitions: 1,
                invocation,
                memory,
            });
            Ok(())
        };
        for layer in 0..self.layer_count() {
            let prefix = format!("blocks.{layer}");
            normalized(
                &mut report,
                &mut mechanism,
                format!("{prefix}.norm1"),
                patches,
                hidden,
            )?;
            projected(
                &mut report,
                &mut mechanism,
                format!("{prefix}.attn.qkv"),
                patches,
                hidden,
                mul(hidden, 3)?,
            )?;
            let policy = self.layer_policy(layer).expect("validated schedule");
            let histogram = match policy.attention {
                VisionAttentionPolicy::Full => &full,
                VisionAttentionPolicy::Windowed => &windows,
            };
            for (&length, &repetitions) in histogram {
                let invocation = MechanismInvocation::Attention {
                    batch: 1,
                    query_heads: self.num_heads as u64,
                    kv_heads: self.num_heads as u64,
                    queries: length,
                    keys: length,
                    key_width: hidden / self.num_heads as u64,
                    value_width: hidden / self.num_heads as u64,
                    element,
                    arithmetic: AttentionArithmetic::Fused,
                    softcap: false,
                    sinks: false,
                };
                let memory = segment(&invocation)?;
                memory.validate()?;
                report.invocations.push(VisionMemoryInvocation {
                    component: format!("{prefix}.attn.segment.{length}"),
                    repetitions,
                    invocation,
                    memory,
                });
            }
            projected(
                &mut report,
                &mut mechanism,
                format!("{prefix}.attn.proj"),
                patches,
                hidden,
                hidden,
            )?;
            normalized(
                &mut report,
                &mut mechanism,
                format!("{prefix}.norm2"),
                patches,
                hidden,
            )?;
            projected(
                &mut report,
                &mut mechanism,
                format!("{prefix}.mlp.linear_fc1"),
                patches,
                hidden,
                self.intermediate_size as u64,
            )?;
            projected(
                &mut report,
                &mut mechanism,
                format!("{prefix}.mlp.linear_fc2"),
                patches,
                self.intermediate_size as u64,
                hidden,
            )?;
            if let Some(index) = policy.deepstack_merger {
                let width = mul(hidden, unit)?;
                normalized(
                    &mut report,
                    &mut mechanism,
                    format!("deepstack_merger_list.{index}.norm"),
                    merged,
                    width,
                )?;
                projected(
                    &mut report,
                    &mut mechanism,
                    format!("deepstack_merger_list.{index}.linear_fc1"),
                    merged,
                    width,
                    width,
                )?;
                projected(
                    &mut report,
                    &mut mechanism,
                    format!("deepstack_merger_list.{index}.linear_fc2"),
                    merged,
                    width,
                    self.out_hidden_size as u64,
                )?;
            }
        }
        let width = mul(hidden, unit)?;
        normalized(
            &mut report,
            &mut mechanism,
            "merger.norm".into(),
            patches,
            hidden,
        )?;
        projected(
            &mut report,
            &mut mechanism,
            "merger.linear_fc1".into(),
            merged,
            width,
            width,
        )?;
        projected(
            &mut report,
            &mut mechanism,
            "merger.linear_fc2".into(),
            merged,
            width,
            self.out_hidden_size as u64,
        )?;
        report.missing = [
            "activation representation is caller-declared; parameter dtype, promotion and mixed intermediate representations are not established",
            "patch kernel reshape/layout and unbound Tensor::linear lowering, learned-position lookup/interpolation, rotary application, activation, residual and index/concatenation native storage are undescribed",
            "backend helper host arrays (including rotary frequencies), host Vec capacities, allocator/object overhead and report construction storage are outside architecture host payload bytes",
            "lazy evaluation, layer retirement, returned output aliases and snapshot lifetimes lack a complete physical pool/owner schedule",
        ].into_iter().map(str::to_owned).collect();
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(window: bool) -> VisionConfig {
        let mut source = serde_json::json!({
            "depth": 2, "hidden_size": 16, "intermediate_size": 24, "num_heads": 4,
            "num_position_embeddings": 64, "in_channels": 3, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": [0]
        });
        if window {
            source["window_size"] = serde_json::json!(8);
            source["fullatt_block_indexes"] = serde_json::json!([1]);
        }
        let source: super::super::VisionConfigSource = serde_json::from_value(source).unwrap();
        if window {
            source.normalize_qwen3_5().unwrap()
        } else {
            source.normalize_qwen3_vl().unwrap()
        }
    }
    fn report(config: &VisionConfig, grids: &[(i32, i32, i32)]) -> VisionMemoryReport {
        config
            .memory_report(
                grids,
                TensorElementType::F32,
                MechanismMemoryContract::unknown,
                MechanismMemoryContract::unknown,
            )
            .unwrap()
    }
    #[test]
    fn compact_geometry_matches_actual_full_and_partial_window_segments() {
        let config = config(true);
        let grids = [(3, 6, 10), (2, 2, 4)];
        let (patches, full, windows) = segments(&config, &grids).unwrap();
        assert_eq!(patches, 196);
        for (actual, expected) in [
            (
                eredu_nn::sequence_layout::attention_chunk_lengths(&grids).unwrap(),
                full,
            ),
            (
                eredu_nn::sequence_layout::window_partition(&grids, 2, 8, 2)
                    .unwrap()
                    .chunk_lengths,
                windows,
            ),
        ] {
            let mut histogram = BTreeMap::new();
            for length in actual {
                *histogram.entry(length as u64).or_insert(0) += 1;
            }
            assert_eq!(expected, histogram);
        }
        let memory = report(&config, &grids);
        assert!(matches!(
            memory.invocations[0].invocation,
            MechanismInvocation::Projection {
                rows: 196,
                input: 24,
                output: 16,
                format: eredu_checkpoint::LinearFormat::Dense,
                weight_element: None,
                bias: true,
                bias_element: None,
                ..
            }
        ));
        assert_eq!(memory.invocations[0].component, "patch_embed.proj");
        assert_eq!(memory.invocations[0].repetitions, 1);
        assert_eq!(memory.retained_output_logical_bytes, 49 * 32 * 4 * 2);
        let segments: Vec<_> = memory
            .invocations
            .iter()
            .filter(|g| matches!(g.invocation, MechanismInvocation::Attention { .. }))
            .collect();
        assert_eq!(segments.len(), 5); // three window lengths; two full frame lengths
        assert_eq!(segments.iter().map(|g| g.repetitions).sum::<u64>(), 25);
        assert!(memory
            .invocations
            .iter()
            .all(|g| !g.memory.missing.is_empty()));
        assert!(!memory.missing.is_empty());
    }
    #[test]
    fn equal_video_frames_remain_compact_and_formats_survive() {
        let mut config = config(false);
        config.linear_formats.insert(
            "model.visual.blocks.0.attn.qkv.weight".into(),
            eredu_checkpoint::LinearFormat::MxFp4,
        );
        let memory = report(&config, &[(10000, 4, 4)]);
        assert_eq!(memory.invocations.len(), 22); // patch, rotary, two blocks, deepstack and final merger
        assert_eq!(
            memory
                .invocations
                .iter()
                .filter(|g| matches!(g.invocation, MechanismInvocation::Attention { .. }))
                .map(|g| g.repetitions)
                .collect::<Vec<_>>(),
            [10000, 10000]
        );
        assert!(matches!(
            memory
                .invocations
                .iter()
                .find(|g| g.component == "blocks.0.attn.qkv")
                .unwrap()
                .invocation,
            MechanismInvocation::Projection {
                format: eredu_checkpoint::LinearFormat::MxFp4,
                ..
            }
        ));
        let normalizations: Vec<_> = memory
            .invocations
            .iter()
            .filter_map(|group| {
                let MechanismInvocation::LayerNormalization {
                    rows,
                    width,
                    element,
                    weight,
                    weight_element,
                    bias,
                    bias_element,
                } = &group.invocation
                else {
                    return None;
                };
                assert_eq!(*element, TensorElementType::F32);
                assert!(*weight && *bias);
                assert_eq!((*weight_element, *bias_element), (None, None));
                assert_eq!(group.repetitions, 1);
                let values = group.invocation.logical_values().unwrap();
                assert_eq!(values.len(), 2);
                assert!(values.iter().all(|value| value.shape == [*rows, *width]
                    && value.element == TensorElementType::F32));
                Some((group.component.as_str(), *rows, *width))
            })
            .collect();
        assert_eq!(
            normalizations,
            [
                ("blocks.0.norm1", 160000, 16),
                ("blocks.0.norm2", 160000, 16),
                ("deepstack_merger_list.0.norm", 40000, 64),
                ("blocks.1.norm1", 160000, 16),
                ("blocks.1.norm2", 160000, 16),
                ("merger.norm", 160000, 16),
            ]
        );
        assert_eq!(
            memory.host_setup_logical_bytes,
            160000 * 84 + 40000 * 8 + 20000 * 4
        );
        assert_eq!(
            memory.host_setup_logical_bytes,
            config.host_setup_logical_bytes([(10000, 4, 4)]).unwrap()
        );
    }
    #[test]
    fn text_only_skips_native_queries_and_bad_geometry_fails_before_them() {
        let config = config(false);
        let empty = config
            .memory_report(
                &[],
                TensorElementType::F32,
                |_| panic!("projection"),
                |_| panic!("attention"),
            )
            .unwrap();
        assert_eq!(empty.host_setup_logical_bytes, 0);
        assert!(empty.invocations.is_empty());
        assert_eq!(config.host_setup_logical_bytes([]).unwrap(), 0);
        assert!(config
            .host_setup_logical_bytes([(1, 32768, 32768); 2])
            .is_err());
        for grids in [vec![(1, 3, 4)], vec![(i32::MAX, 4, 4)]] {
            assert!(config
                .memory_report(
                    &grids,
                    TensorElementType::F32,
                    |_| panic!("projection"),
                    |_| panic!("attention")
                )
                .is_err());
        }
        let mut windowed = config.clone();
        windowed.mode = VisionMode::WindowScheduled;
        for size in [0, -1] {
            windowed.window_size = size;
            assert!(windowed
                .memory_report(
                    &[(1, 4, 4)],
                    TensorElementType::F32,
                    |_| panic!("projection"),
                    |_| panic!("attention")
                )
                .is_err());
        }
    }
}
