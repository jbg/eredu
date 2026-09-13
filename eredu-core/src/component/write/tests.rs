use super::*;
use crate::{
    capture::CaptureUsage,
    component::{
        ComponentActivation, ComponentNormalization, ComponentNormalizationKind, ComponentScalar,
        ComponentWritePartition,
    },
    intervention::InterventionDtype,
    parameters::ProjectionInputTransform,
};

fn fixture() -> (ComponentGroup, ParameterDiscovery) {
    let group = ComponentGroup {
        id: "attention.channels".into(),
        node_id: "attention".into(),
        layer_index: 0,
        count: 6,
        activation: "channels".into(),
        effective_activation: "channels.effective".into(),
        write_input: None,
        write_output: Some("write".into()),
        output: Some("output".into()),
        write_partition: ComponentWritePartition::Complete,
        input: "normalized".into(),
        reads: vec![],
        routed_reads: vec![],
        write_weight: "outer".into(),
        write_parameter_group: "attention.parameters".into(),
        shared_write_weight: "shared.outer".into(),
        write_bias: None,
        write_input_projection: Some(ComponentGroupedWriteProjection {
            weight: "inner".into(),
            parameter_group: "attention.parameters".into(),
            shared_weight: "shared.inner".into(),
            bias: None,
            groups: 2,
            rank: 2,
            input: "grouped_input".into(),
            output: "latent".into(),
            final_input: "latent.write_input".into(),
        }),
        activation_equation: ComponentActivation::Attention {
            query_heads: 2,
            key_value_heads: 1,
            head_width: 3,
            output_gate: None,
        },
        input_normalization: ComponentNormalization {
            kind: ComponentNormalizationKind::Rms,
            epsilon: ComponentScalar::new(1e-5),
            gain: None,
            gain_offset: ComponentScalar::new(0.0),
            bias: None,
            groups: 1,
        },
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    };
    let loaded = ParameterDiscovery {
        identity: "loaded-overlay".into(),
        artifact_identity: "artifact".into(),
        overlay_identity: Some("edit".into()),
        usage: CaptureUsage::default(),
        coordination_usage: Default::default(),
        parameters: [("outer", vec![3, 4]), ("inner", vec![4, 3])]
            .into_iter()
            .map(|(name, shape)| LoadedParameter {
                id: name.into(),
                shared_id: format!("actual-shared:{name}"),
                shape,
                dtype: Some(InterventionDtype::Float32),
                supported: true,
                access: None,
                condition: "effective edited values".into(),
                input_transform: ProjectionInputTransform::Identity,
            })
            .collect(),
    };
    (group, loaded)
}

#[test]
fn grouped_columns_reconstruct_nonzero_writes_and_signed_directions() {
    let (group, loaded) = fixture();
    let inner = [
        [1.0, 2.0, -1.0],
        [-2.0, 0.5, 3.0],
        [0.25, -1.0, 2.0],
        [4.0, -0.5, 1.0],
    ];
    let outer = [
        [1.0, -1.0, 2.0, 0.5],
        [-0.5, 3.0, -2.0, 1.0],
        [2.0, 0.25, 0.5, -1.0],
    ];
    let values = [0.5, -1.0, 2.0, -0.25, 1.5, 0.75];
    let direction = [1.0, -2.0, 0.5];
    let latent: Vec<f64> = inner
        .iter()
        .enumerate()
        .map(|(row, weights)| {
            weights
                .iter()
                .zip(&values[row / 2 * 3..])
                .map(|(w, a)| w * a)
                .sum()
        })
        .collect();
    let expected: Vec<f64> = outer
        .iter()
        .map(|weights| weights.iter().zip(&latent).map(|(w, a)| w * a).sum())
        .collect();
    let mut reconstructed = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        let column = group
            .write_column(
                &ComponentId {
                    group: group.id.clone(),
                    index,
                },
                &loaded,
            )
            .unwrap();
        let input = column.input.unwrap();
        assert_eq!(input.parameter.shared_id, "actual-shared:inner");
        assert_eq!(column.output.parameter.shared_id, "actual-shared:outer");
        assert_eq!(input.region.shape, [2, 1]);
        assert_eq!(column.output.region.shape, [3, 2]);
        let start = input.region.starts[0] as usize;
        let local = input.region.starts[1] as usize;
        assert_eq!(column.output.region.starts, [0, start as u64]);
        let projected: Vec<f64> = (start..start + 2)
            .map(|k| (0..3).map(|h| direction[h] * outer[h][k]).sum())
            .collect();
        let signed: f64 = projected
            .iter()
            .enumerate()
            .map(|(k, d)| d * inner[start + k][local])
            .sum();
        let mut direct_signed = 0.0;
        for h in 0..3 {
            let write: f64 = (start..start + 2)
                .map(|k| outer[h][k] * inner[k][local])
                .sum();
            reconstructed[h] += value * write;
            direct_signed += direction[h] * write;
        }
        assert_eq!(signed, direct_signed);
    }
    assert_eq!(reconstructed.as_slice(), expected);
    assert!(expected.iter().all(|value| *value != 0.0));
}

#[test]
fn direct_columns_preserve_loaded_support_and_old_serialization() {
    let (mut group, mut loaded) = fixture();
    group.write_input_projection = None;
    loaded.parameters[0].shape = vec![3, 6];
    loaded.parameters[0].supported = false;
    let column = group
        .write_column(
            &ComponentId {
                group: group.id.clone(),
                index: 5,
            },
            &loaded,
        )
        .unwrap();
    assert!(column.input.is_none());
    assert!(!column.output.parameter.access().query);
    assert_eq!(column.output.region.starts, [0, 5]);
    assert_eq!(column.output.region.shape, [3, 1]);
    let wire = serde_json::to_value(&group).unwrap();
    assert!(wire.get("write_input_projection").is_none());
    assert_eq!(
        serde_json::from_value::<ComponentGroup>(wire).unwrap(),
        group
    );
    let (group, _) = fixture();
    assert_eq!(
        serde_json::from_slice::<ComponentGroup>(&serde_json::to_vec(&group).unwrap()).unwrap(),
        group
    );
}

#[test]
fn malformed_factor_geometry_and_missing_loaded_slots_are_typed() {
    let (group, loaded) = fixture();
    let id = ComponentId {
        group: group.id.clone(),
        index: 4,
    };
    for (groups, rank) in [(0, 2), (2, 0), (4, 2), (2, usize::MAX)] {
        let mut invalid = group.clone();
        let stage = invalid.write_input_projection.as_mut().unwrap();
        stage.groups = groups;
        stage.rank = rank;
        assert_eq!(
            invalid.write_column(&id, &loaded),
            Err(ComponentWriteError::Geometry)
        );
    }
    let mut invalid = loaded.clone();
    invalid.parameters[1].shape = vec![4, 6];
    assert_eq!(
        group.write_column(&id, &invalid),
        Err(ComponentWriteError::Geometry)
    );
    invalid = loaded.clone();
    invalid.parameters.push(loaded.parameters[1].clone());
    assert_eq!(
        group.write_column(&id, &invalid),
        Err(ComponentWriteError::Geometry)
    );
    invalid = loaded.clone();
    invalid.parameters.pop();
    assert_eq!(
        group.write_column(&id, &invalid),
        Err(ComponentWriteError::MissingParameter("inner".into()))
    );
    for id in [
        ComponentId {
            group: "wrong".into(),
            index: 0,
        },
        ComponentId {
            group: group.id.clone(),
            index: 6,
        },
    ] {
        assert_eq!(
            group.write_column(&id, &loaded),
            Err(ComponentWriteError::Identity)
        );
    }
}
