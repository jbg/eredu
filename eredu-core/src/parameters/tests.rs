use super::*;
fn discovery() -> ParameterDiscovery {
    ParameterDiscovery {
        identity: "loaded-v0".into(),
        artifact_identity: "weights".into(),
        overlay_identity: None,
        parameters: ["weight", "tied"]
            .into_iter()
            .map(|id| LoadedParameter {
                id: id.into(),
                shared_id: "weight".into(),
                shape: vec![3, 4],
                dtype: Some(InterventionDtype::Float32),
                supported: true,
                access: None,
                condition: String::new(),
                input_transform: ProjectionInputTransform::Identity,
            })
            .collect(),
        usage: CaptureUsage::default(),
        coordination_usage: Default::default(),
    }
}
fn plan() -> ParameterOverlayPlan {
    ParameterOverlayPlan {
        schema_version: PARAMETER_SCHEMA_VERSION,
        base_identity: "loaded-v0".into(),
        provenance: "nonzero fixture".into(),
        edits: vec![ParameterEdit {
            id: "read".into(),
            parameter: "weight".into(),
            parameter_shape: vec![3, 4],
            dtype: InterventionDtype::Float32,
            region: ParameterRegion {
                starts: vec![1, 0],
                shape: vec![1, 4],
            },
            update: ParameterUpdate::Replace {
                values: vec![1.0, -2.0, 3.0, -4.0],
            },
        }],
    }
}
#[test]
fn peer_intent_keeps_local_authority_and_binds_source_version_and_ordered_edits() {
    let facts = discovery();
    let host = plan();
    let admitted = AdmittedParameterOverlay::admit(host.clone(), &facts).unwrap();
    let prior_digest = Sha256::digest(serde_json::to_vec(&host).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(admitted.identity(), format!("overlay-{prior_digest}"));
    let mut peer = facts.clone();
    peer.identity = "independent-loaded-rank-v0".into();
    let mut peer_host = host.clone();
    peer_host.base_identity = peer.identity.clone();
    let peer_admitted = AdmittedParameterOverlay::admit(peer_host, &peer).unwrap();
    assert_eq!(admitted.intent_identity(), peer_admitted.intent_identity());
    assert_ne!(admitted.identity(), peer_admitted.identity());
    assert!(matches!(
        admitted.validate(&peer),
        Err(ParameterError::StaleIdentity)
    ));
    for variant in 0..5 {
        let mut changed_facts = facts.clone();
        let mut changed_host = host.clone();
        match variant {
            0 => changed_facts.artifact_identity.push_str("-different"),
            1 => changed_facts.overlay_identity = Some("prior-overlay".into()),
            2 => changed_host.provenance.push_str(" changed"),
            3 => {
                changed_host.edits[0].update = ParameterUpdate::Add {
                    values: vec![1.0, -2.0, 3.0, -4.0],
                }
            }
            _ => changed_facts.parameters[0].shared_id = "other-sharing-class".into(),
        }
        let changed = AdmittedParameterOverlay::admit(changed_host, &changed_facts).unwrap();
        assert_ne!(admitted.intent_identity(), changed.intent_identity());
        if variant <= 1 {
            assert!(matches!(
                admitted.validate(&changed_facts),
                Err(ParameterError::StaleIdentity)
            ));
        }
    }
    let mut wire = serde_json::to_value(&facts).unwrap();
    wire.as_object_mut().unwrap().remove("coordination_usage");
    assert_eq!(
        serde_json::from_value::<ParameterDiscovery>(wire)
            .unwrap()
            .coordination_usage,
        ParameterCoordinationUsage::default()
    );
}
#[test]
fn overlay_authority_binds_exact_identity_geometry_dtype_and_alias_conflicts() {
    let facts = discovery();
    let host = plan();
    let encoded = serde_json::to_string(&host).unwrap();
    let admitted =
        AdmittedParameterOverlay::admit(serde_json::from_str(&encoded).unwrap(), &facts).unwrap();
    assert_eq!(admitted.shared_targets(), &["weight"]);
    let mut stale = facts.clone();
    stale.identity = "loaded-v1".into();
    assert!(matches!(
        admitted.validate(&stale),
        Err(ParameterError::StaleIdentity)
    ));
    for variant in 0..4 {
        let mut invalid = host.clone();
        match variant {
            0 => invalid.edits[0].dtype = InterventionDtype::Float16,
            1 => invalid.edits[0].parameter_shape[0] = 4,
            2 => invalid.edits[0].region.starts[0] = u64::MAX,
            _ => {
                invalid.edits[0].update = ParameterUpdate::Add {
                    values: vec![f32::NAN; 4],
                }
            }
        }
        assert!(AdmittedParameterOverlay::admit(invalid, &facts).is_err());
    }
    let mut overlap = host.clone();
    let mut alias = host.edits[0].clone();
    alias.id = "write".into();
    alias.parameter = "tied".into();
    alias.region = ParameterRegion {
        starts: vec![0, 2],
        shape: vec![3, 1],
    };
    alias.update = ParameterUpdate::Add {
        values: vec![1.0; 3],
    };
    overlap.edits.push(alias);
    assert!(matches!(
        AdmittedParameterOverlay::admit(overlap, &facts),
        Err(ParameterError::Conflict(_))
    ));
    let mut disjoint = host;
    let mut alias = disjoint.edits[0].clone();
    alias.id = "another".into();
    alias.parameter = "tied".into();
    alias.region.starts[0] = 2;
    disjoint.edits.push(alias);
    AdmittedParameterOverlay::admit(disjoint, &facts).unwrap();
}

#[test]
fn projections_check_axis_payload_finiteness_and_overflow() {
    let projection = ParameterProjection {
        region: ParameterRegion {
            starts: vec![0, 1, 0],
            shape: vec![2, 3, 4],
        },
        axis: 1,
        directions: 2,
        coefficients: vec![1.0, -2.0, 0.5, -1.0, 3.0, 4.0],
    };
    assert_eq!(projection.output_shape(&[2, 5, 4]).unwrap(), [2, 4, 2]);
    for invalid in 0..5 {
        let mut p = projection.clone();
        match invalid {
            0 => p.axis = 3,
            1 => p.directions = u64::MAX,
            2 => p.coefficients[0] = f32::INFINITY,
            3 => {
                p.coefficients.pop();
            }
            _ => p.region.starts[1] = u64::MAX,
        }
        assert!(p.output_shape(&[2, 5, 4]).is_err());
    }
}

#[test]
fn selected_projection_input_arithmetic_roundtrips_and_old_records_remain_unknown() {
    let mut facts = discovery();
    facts.parameters[0].input_transform = ProjectionInputTransform::BlockFp8E4m3 {
        block_width: 128,
        floor: crate::component::ComponentScalar::new(1e-4),
        magnitude: crate::component::ComponentScalar::new(448.0),
    };
    let mut wire = serde_json::to_value(&facts).unwrap();
    assert_eq!(
        serde_json::from_value::<ParameterDiscovery>(wire.clone()).unwrap(),
        facts
    );
    for parameter in wire["parameters"].as_array_mut().unwrap() {
        parameter.as_object_mut().unwrap().remove("input_transform");
    }
    let old = serde_json::from_value::<ParameterDiscovery>(wire).unwrap();
    assert!(old
        .parameters
        .iter()
        .all(|p| p.input_transform == ProjectionInputTransform::Unspecified));
}

#[test]
fn exact_operation_capabilities_override_legacy_aggregate_admission() {
    let mut facts = discovery();
    let admitted = AdmittedParameterOverlay::admit(plan(), &facts).unwrap();
    assert!(facts.parameters[0].access().replacement);
    facts.parameters[0].access = Some(ParameterAccess {
        query: true,
        projection: true,
        replacement: false,
    });
    assert!(facts.parameters[0].supported);
    assert!(matches!(
        AdmittedParameterOverlay::admit(plan(), &facts),
        Err(ParameterError::Unsupported(_))
    ));
    assert!(matches!(
        admitted.validate(&facts),
        Err(ParameterError::Unsupported(_))
    ));
    let encoded = serde_json::to_vec(&facts).unwrap();
    let decoded: ParameterDiscovery = serde_json::from_slice(&encoded).unwrap();
    assert!(decoded.parameters[0].access().query);
    assert!(!decoded.parameters[0].access().replacement);
    assert!(decoded.parameters[1].access().replacement);
    facts.parameters[0].supported = false;
    facts.parameters[0].access.as_mut().unwrap().replacement = true;
    admitted.validate(&facts).unwrap();
}
