use super::*;
use crate::qwen4_exp::{
    config::{Config, LayerKind},
    mtp::PredictionLimits,
    qsa::QsaExecutionLimits,
    target::MixerSpec,
};
use eredu_checkpoint::{AffineQuantization, BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_core::ParallelTopology;
use eredu_nn::{
    GatedProductGroupLayout, GatedProductPolicy, GroupReduction, GroupedProjectionSpec,
    LinearFormatSpec, LinearRowLayout, ParameterSpec, TensorElementType,
};
use eredu_runtime::RoutedBankId;

fn parameter(name: impl AsRef<str>) -> ParameterSpec {
    ParameterSpec::trainable(name.as_ref()).unwrap()
}

fn spec(encoding: Option<LinearFormat>) -> PredictionSpec {
    let mut config = Config::from_json(
        &serde_json::from_str(include_str!("../../config/released.json")).unwrap(),
    )
    .unwrap();
    // Keep the released attention geometry, including 24 query / two K/V heads,
    // while making the ownership boundaries small and deliberately uneven.
    config.experts.count = 5;
    config.experts.selected = 2;
    config.experts.intermediate = 128;
    config.experts.shared_intermediate = 128;
    config.prediction.as_mut().unwrap().layers =
        eredu_core::LayerSchedule::new(2, vec![LayerKind::Recurrent, LayerKind::Indexed]).unwrap();
    let dense = || LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
    PredictionSpec::from_prepared(
        &config,
        PredictionLimits {
            qsa: QsaExecutionLimits {
                batch: 1,
                tokens: 4,
                workspace_bytes: 1 << 24,
            },
            tile_blocks: 16,
            selection_workspace: 1 << 20,
            element: TensorElementType::F32,
        },
        RoutedBankId::new(2),
        |depth| {
            let projection = |suffix: &str, fused: bool| {
                let name = format!("mtp.layers.{depth}.mlp.experts.{suffix}");
                let format = match encoding {
                    Some(LinearFormat::Affine(format)) => LinearFormatSpec::affine(
                        LinearFormat::Affine(format),
                        parameter(format!("{name}.scales")),
                        parameter(format!("{name}.biases")),
                    )?,
                    Some(format) => {
                        LinearFormatSpec::scaled(format, parameter(format!("{name}.scales")))?
                    }
                    None => dense(),
                };
                let format = if fused && matches!(encoding, Some(LinearFormat::E4M3BlockFp8(_))) {
                    format.with_row_layout(LinearRowLayout::equal_partitions(2)?)?
                } else {
                    format
                };
                GroupedProjectionSpec::new(parameter(&name), None, format)
            };
            GroupedGatedProductSpec::new(
                5,
                config.hidden_size,
                128,
                config.hidden_size,
                GatedProductPolicy::ordinary_silu(),
                GatedProductGroupLayout::Packed {
                    gate_up: projection("gate_up_proj", true)?,
                    down: projection("down_proj", false)?,
                },
            )
        },
        |_| Ok(dense()),
    )
    .unwrap()
}

fn range(axis: usize, start: usize, end: usize) -> TensorSelection {
    TensorSelection::Range { axis, start, end }
}

#[test]
fn prediction_tp4_preserves_replicated_kv_and_recurrent_component_ranges() {
    let spec = spec(None);
    for rank in 0..4 {
        let partition = spec.tensor_partition(rank, 4).unwrap();
        assert_eq!((partition.rank(), partition.ranks()), (rank, 4));
        let MixerSpec::Indexed(indexed) = &partition.local_spec().units[1].mixer else {
            panic!("indexed prediction depth")
        };
        assert_eq!(
            (indexed.heads, indexed.kv_heads, indexed.head_dim),
            (6, 1, 256)
        );
        assert_eq!(indexed.projections[0].output, 3072);
        for suffix in ["k_proj", "v_proj"] {
            assert_eq!(
                partition.parameter_selections(&format!("mtp.layers.1.self_attn.{suffix}.weight")),
                Some([range(0, rank / 2 * 256, (rank / 2 + 1) * 256)].as_slice())
            );
        }
        assert_eq!(
            partition.parameter_selections("mtp.layers.1.self_attn.q_proj.weight"),
            Some([range(0, rank * 3072, (rank + 1) * 3072)].as_slice())
        );
        assert_eq!(
            partition.parameter_selections("mtp.layers.0.linear_attn.in_proj_qkv.weight"),
            Some(
                [
                    range(0, rank * 512, (rank + 1) * 512),
                    range(0, 2048 + rank * 512, 2048 + (rank + 1) * 512),
                    range(0, 4096 + rank * 1536, 4096 + (rank + 1) * 1536),
                ]
                .as_slice()
            )
        );
        let MixerSpec::Recurrent(recurrent) = &partition.local_spec().units[0].mixer else {
            panic!("recurrent prediction depth")
        };
        assert_eq!(
            (recurrent.mixer.key_heads, recurrent.mixer.value_heads),
            (4, 12)
        );
        for name in [
            "mtp.fc_hidden.weight",
            "mtp.fc_embedding.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.1.mlp.gate.weight",
        ] {
            assert!(partition.parameter_selections(name).is_none(), "{name}");
        }
    }
}

#[test]
fn prediction_fp8_ranges_preserve_fused_rows_and_scale_blocks() {
    let spec = spec(Some(LinearFormat::E4M3BlockFp8(
        BlockFp8Format::new(64, 64, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
    )));
    let partition = spec.tensor_partition(1, 2).unwrap();
    for depth in 0..2 {
        let root = format!("mtp.layers.{depth}.mlp.experts");
        for (suffix, expected) in [
            ("gate_up_proj", vec![range(1, 64, 128), range(1, 192, 256)]),
            ("gate_up_proj.scales", vec![range(1, 1, 2), range(1, 3, 4)]),
            ("down_proj", vec![range(2, 64, 128)]),
            ("down_proj.scales", vec![range(2, 1, 2)]),
        ] {
            assert_eq!(
                partition.parameter_selections(&format!("{root}.{suffix}")),
                Some(expected.as_slice())
            );
        }
    }
    // TP4 would divide a 64-value inverse-scale block into 32-value pieces.
    assert!(spec.tensor_partition(0, 4).is_err());
}

#[test]
fn prediction_affine_ranges_preserve_packing_and_both_companions() {
    let spec = spec(Some(LinearFormat::Affine(
        AffineQuantization::new(64, 4).unwrap(),
    )));
    let partition = spec.tensor_partition(1, 2).unwrap();
    for depth in 0..2 {
        let root = format!("mtp.layers.{depth}.mlp.experts");
        for (suffix, expected) in [
            ("gate_up_proj", vec![range(1, 64, 128), range(1, 192, 256)]),
            (
                "gate_up_proj.scales",
                vec![range(1, 64, 128), range(1, 192, 256)],
            ),
            (
                "gate_up_proj.biases",
                vec![range(1, 64, 128), range(1, 192, 256)],
            ),
            ("down_proj", vec![range(2, 8, 16)]),
            ("down_proj.scales", vec![range(2, 1, 2)]),
            ("down_proj.biases", vec![range(2, 1, 2)]),
        ] {
            assert_eq!(
                partition.parameter_selections(&format!("{root}.{suffix}")),
                Some(expected.as_slice())
            );
        }
    }
    // The packed words divide by four; the 64-value affine groups do not.
    assert!(spec.tensor_partition(0, 4).is_err());
}

#[test]
fn prediction_ep_uneven_owners_preserve_router_and_reject_altered_realizations() {
    let tensor = spec(None).tensor_partition(1, 2).unwrap();
    let spec = tensor.local_spec();
    for (rank, expected) in [(0, vec![0, 1, 2]), (1, vec![3, 4])] {
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 2, 1).unwrap(), rank).unwrap();
        let realization = spec.expert_realization(topology).unwrap();
        assert_eq!(realization.local_global_group_indices(), expected);
        let local = spec.with_expert_realization(&realization).unwrap();
        for unit in &local.units {
            let ff = &unit.feed_forward.feed_forward;
            assert_eq!(ff.experts.group_count() as usize, expected.len());
            assert_eq!(ff.experts.intermediate_dimensions(), 64);
            assert_eq!(ff.router.selection().group_count(), 5);
            assert_eq!(ff.bank, RoutedBankId::new(2));
        }
        assert!(local.with_expert_realization(&realization).is_err());
        let wrong_owner = realization
            .unit_specs()
            .iter()
            .map(|((_, depth), unit)| {
                (
                    (ExecutionGroupId::new("target").unwrap(), *depth),
                    unit.clone(),
                )
            })
            .collect();
        let wrong_owner = ExpertRealizationPlan::balanced(5, topology, wrong_owner).unwrap();
        assert!(spec.with_expert_realization(&wrong_owner).is_err());
        let wrong_equation = realization
            .unit_specs()
            .iter()
            .map(|(address, unit)| {
                (
                    address.clone(),
                    unit.clone().with_reduction(GroupReduction::Sum),
                )
            })
            .collect();
        let wrong_equation = ExpertRealizationPlan::balanced(5, topology, wrong_equation).unwrap();
        assert!(spec.with_expert_realization(&wrong_equation).is_err());
        // A failed check cannot mutate the reusable source specification.
        assert!(spec.with_expert_realization(&realization).is_ok());
    }
    for (rank, ranks) in [(0, 0), (2, 2), (0, 5)] {
        assert!(spec.tensor_partition(rank, ranks).is_err());
    }
    let mut malformed = spec.clone();
    malformed.units[1].depth = 0;
    assert!(malformed.tensor_partition(0, 2).is_err());
}
