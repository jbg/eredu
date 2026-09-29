//! Checks the exact layout retained by architecture preparation without replanning it.
use eredu_runtime::{ArchitectureParameterDescription, LocalModelLayout, TensorPlacement};

pub(super) fn validate(
    parameters: &ArchitectureParameterDescription,
    layout: &LocalModelLayout,
) -> Result<(), String> {
    let mut count = 0;
    for group in parameters.groups() {
        for member in group.members() {
            count += 1;
            let name = member.target();
            let tensor = layout
                .tensor(name)
                .ok_or_else(|| format!("retained layout has no parameter {name}"))?;
            if tensor.global_shape() != member.global_shape()
                || tensor.logical_name() != group.group().logical_name()
                || tensor.role() != group.group().role()
            {
                return Err(format!(
                    "retained layout changed the global declaration of {name}"
                ));
            }
            let mut shape = tensor.global_shape().to_vec();
            for placement in tensor
                .additional_placements()
                .iter()
                .chain(std::iter::once(tensor.placement()))
            {
                apply(&mut shape, placement)
                    .map_err(|error| format!("retained layout for {name}: {error}"))?;
            }
            if shape != tensor.local_shape() {
                return Err(format!(
                    "retained layout for {name} produces shape {shape:?}, expected {:?}",
                    tensor.local_shape()
                ));
            }
        }
    }
    if count != layout.len() {
        return Err("retained layout contains undeclared parameters".into());
    }
    Ok(())
}

fn apply(shape: &mut [usize], placement: &TensorPlacement) -> Result<(), String> {
    let (axis, width) = match placement {
        TensorPlacement::Replicated | TensorPlacement::Local | TensorPlacement::Rank { .. } => {
            return Ok(());
        }
        TensorPlacement::Omit => {
            return Err("a complete parameter layout cannot omit a tensor".into());
        }
        TensorPlacement::Shard { axis, index, parts } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("shard axis is outside the source")?;
            if *parts == 0
                || index >= parts
                || !extent.is_multiple_of(*parts)
                || extent / parts == 0
            {
                return Err("invalid equal shard".into());
            }
            (*axis, extent / parts)
        }
        TensorPlacement::Range { axis, start, end } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("range axis is outside the source")?;
            if start >= end || *end > extent {
                return Err("invalid physical range".into());
            }
            (*axis, end - start)
        }
        TensorPlacement::Indices { axis, indices } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("index axis is outside the source")?;
            if indices.is_empty()
                || indices.iter().any(|index| *index >= extent)
                || indices
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != indices.len()
            {
                return Err("invalid physical indices".into());
            }
            (*axis, indices.len())
        }
    };
    shape[axis] = width;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_rank_projection_preserves_shared_kv_and_rejects_other_ranks() {
        use eredu_runtime::{
            ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout, LocalTensorLayout,
            MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
            ParameterMemberSpec, ParameterRole,
        };
        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let group = ParameterGroupSpec::new(
            "keys",
            ParameterRole::AttentionHeads,
            [ParameterMemberSpec::new(
                "keys.weight",
                vec![4, 8],
                MemberSharding::Replicated,
            )],
        )
        .unwrap();
        let description = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [group.clone()],
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_role("keys"),
                group,
            )],
        )
        .unwrap();
        let mut layout = LocalModelLayout::default();
        layout.insert(
            "keys.weight".into(),
            LocalTensorLayout::new(
                "keys",
                ParameterRole::AttentionHeads,
                vec![4, 8],
                vec![2, 8],
                TensorPlacement::Range {
                    axis: 0,
                    start: 2,
                    end: 4,
                },
                Some(2),
                Some(1..2),
                false,
            ),
        );
        let topology = eredu_core::ParallelTopology::new(4, 1, 1, 1).unwrap();
        let rank = eredu_core::ParallelRankTopology::new(topology, 3).unwrap();
        let description = description
            .with_partition_layout(rank, layout.clone())
            .unwrap();
        assert_eq!(
            super::super::derive_partitioned_local_layout(&description, rank).unwrap(),
            layout
        );
        assert!(super::super::derive_partitioned_local_layout(
            &description,
            eredu_core::ParallelRankTopology::new(topology, 0).unwrap()
        )
        .unwrap_err()
        .contains("projection onto rank 0 is not implemented"));
        let tensor = super::super::retained_partition_layout(&description, rank)
            .unwrap()
            .unwrap()
            .tensor("keys.weight")
            .unwrap();
        assert_eq!(tensor.local_shape(), [2, 8]);
        assert_eq!(tensor.logical_range(), Some(&(1..2)));
    }

    #[test]
    fn exact_physical_placements_preserve_replicas_and_combined_axes() {
        let mut shape = [4, 12, 8];
        apply(
            &mut shape,
            &TensorPlacement::Range {
                axis: 0,
                start: 2,
                end: 4,
            },
        )
        .unwrap();
        apply(
            &mut shape,
            &TensorPlacement::Indices {
                axis: 1,
                indices: vec![1, 3, 5, 7],
            },
        )
        .unwrap();
        assert_eq!(shape, [2, 4, 8]);
        for _rank in 0..4 {
            let mut kv = [2, 8];
            apply(
                &mut kv,
                &TensorPlacement::Range {
                    axis: 0,
                    start: _rank / 2,
                    end: _rank / 2 + 1,
                },
            )
            .unwrap();
            assert_eq!(kv, [1, 8]);
        }
        for invalid in [
            TensorPlacement::Range {
                axis: 1,
                start: 0,
                end: 9,
            },
            TensorPlacement::Indices {
                axis: 0,
                indices: vec![1, 1],
            },
            TensorPlacement::Shard {
                axis: 0,
                index: 2,
                parts: 2,
            },
            TensorPlacement::Shard {
                axis: 0,
                index: 0,
                parts: 0,
            },
        ] {
            assert!(apply(&mut [2, 8], &invalid).is_err());
        }
    }
}
