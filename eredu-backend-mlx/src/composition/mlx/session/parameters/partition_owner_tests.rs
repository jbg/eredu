//! Native local-owner checks within the existing prepared Ring fixtures.
//! The public peer transaction tests separately cover version/cache lifecycle.
use super::*;
#[path = "partition_coordination_tests.rs"]
mod coordination;

fn host(value: &MlxTensor, stream: &Stream) -> Result<Vec<f32>, Error> {
    Ok(value
        .as_array()
        .as_dtype(Dtype::Float32, stream)?
        .contiguous(false, stream)?
        .evaluated()?
        .as_slice::<f32>()
        .to_vec())
}

impl MlxModelSession {
    pub(crate) fn verify_partition_parameter_owner_for_test(
        &mut self,
        stream: &Stream,
        reference: &mut ModelRuntime<MlxBackend<'_>>,
    ) {
        let facts = self.parameter_facts().unwrap();
        assert!(facts
            .parameters
            .iter()
            .any(|parameter| parameter.access().query));
        let slots = self
            .payload
            .model
            .erased()
            .prepared_parameter_slots()
            .to_vec();
        assert!(!slots.is_empty());
        assert!(slots.iter().any(|slot| matches!(
            slot.location,
            eredu_runtime::parameter_operations::PreparedParameterLocation::Unit { .. }
        )));
        let global_addresses = {
            let layout = self
                .payload
                .model
                .erased()
                .partition_parameter_description()
                .expect("prepared partition declaration")
                .unit_layout();
            (0..layout.len())
                .map(|ordinal| layout.address(ordinal).unwrap())
                .collect::<Vec<_>>()
        };
        let mut original = BTreeMap::new();
        let mut changed = BTreeMap::new();
        let mut expected = BTreeMap::new();
        let mut observed_nonzero = false;
        for slot in &slots {
            use eredu_runtime::parameter_operations::PreparedParameterLocation;
            if let PreparedParameterLocation::Unit { ordinal, address } = slot.location {
                let foreign = *global_addresses
                    .iter()
                    .find(|other| **other != address)
                    .expect("two-layer fixture has another semantic unit");
                for location in [
                    PreparedParameterLocation::Unit {
                        ordinal,
                        address: foreign,
                    },
                    PreparedParameterLocation::Unit {
                        ordinal: usize::MAX,
                        address,
                    },
                ] {
                    self.with_model_operation(|model| {
                        let result = model.erased_mut().with_parameter_slots(
                            &location,
                            &Default::default(),
                            &mut |_| {
                                panic!(
                                    "invalid owner must fail before traversal or materialization"
                                )
                            },
                            stream,
                        );
                        assert!(
                            matches!(result, Err(Error::ArchitectureModel(_))),
                            "{result:?}"
                        );
                        Ok(())
                    })
                    .unwrap();
                }
            }
            let id = slot.parameter.id.as_str();
            let ids = BTreeSet::from([id.to_owned()]);
            let (value, values) = self
                .with_model_operation(|model| {
                    // Complete an actual native read before a semantic callback
                    // rejection. The next loan must still acquire the same owner.
                    let rejected = with_selected_parameter_values(
                        model.erased_mut(),
                        &ids,
                        stream,
                        |selected| {
                            let _ = host(&selected[id], stream)?;
                            Err::<(), _>(Error::ArchitectureModel(
                                "injected parameter callback rejection".into(),
                            ))
                        },
                    );
                    assert!(
                        matches!(rejected, Err(Error::ArchitectureModel(ref message))
                    if message == "injected parameter callback rejection"),
                        "{id}: {rejected:?}"
                    );
                    with_selected_parameter_values(model.erased_mut(), &ids, stream, |selected| {
                        let value = &selected[id];
                        assert_eq!(
                            value
                                .shape()
                                .iter()
                                .map(|n| *n as usize)
                                .collect::<Vec<_>>(),
                            slot.materialized.shape
                        );
                        Ok((value.clone(), host(value, stream)?))
                    })
                })
                .unwrap();
            assert_eq!(
                value.as_array().dtype(),
                Dtype::Float32,
                "dense Qwen fixture"
            );
            observed_nonzero |= values.iter().any(|n| n.abs() > 1e-4);
            let replacement = values.iter().map(|value| value + 0.125).collect::<Vec<_>>();
            let tensor = Array::from_slice(&replacement, &value.shape())
                .as_dtype(value.as_array().dtype(), stream)
                .unwrap();
            tensor.evaluated().unwrap();
            changed.insert(id.to_owned(), MlxTensor::from_array(tensor));
            expected.insert(id.to_owned(), replacement);
            original.insert(id.to_owned(), value);
        }
        assert!(observed_nonzero);
        assert_eq!(original.len(), slots.len());
        assert_eq!(changed.len(), slots.len());
        assert_eq!(expected.len(), slots.len());
        self.verify_partition_parameter_coordination_for_test(
            stream, reference, &original, &changed, false,
        );
        self.ensure_no_submission_in_flight().unwrap();
        let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
            .unwrap()
            .enter()
            .unwrap();
        let count = self
            .payload
            .parameter_state
            .parameter_sources()
            .count(&mut guard)
            .unwrap();
        assert_eq!(
            count
                .counts()
                .role(ParameterOwnerRole::DisplacedOriginal)
                .parameters
                .auxiliary_slots,
            self.payload.parameter_state.originals.len()
        );
        assert_eq!(
            count
                .counts()
                .role(ParameterOwnerRole::PublishedOverlay)
                .parameters
                .auxiliary_slots,
            self.payload.parameter_state.published.len()
        );
        // This transaction installs each declared model slot directly; it does
        // not create entries in NativeParameterState's separate overlay maps.
        // The exact map counts above describe those owners. Every installed
        // slot is checked against its nonzero replacement below, and again
        // against its original value after coordinated restoration.
        drop(count);
        let owner = crate::composition::mlx::replicated_text::NativeParameterOwnerSource::new(
            self.payload.model.erased(),
        )
        .count(&mut guard)
        .unwrap();
        assert!(std::ptr::addr_eq(
            owner.source().owner(),
            self.payload.model.erased()
        ));
        assert_eq!(
            owner.counts(),
            crate::composition::mlx::replicated_text::NativeParameterOwnerSource::new(
                self.payload.model.erased()
            )
            .count(&mut guard)
            .unwrap()
            .counts()
        );
        drop(owner);
        drop(guard);

        for slot in &slots {
            let id = slot.parameter.id.as_str();
            self.with_model_operation(|model| {
                with_selected_parameter_values(
                    model.erased_mut(),
                    &BTreeSet::from([id.to_owned()]),
                    stream,
                    |selected| {
                        assert_eq!(
                            host(&selected[id], stream)?,
                            expected[id],
                            "replacement {id}"
                        );
                        Ok(())
                    },
                )
            })
            .unwrap();
        }
        self.verify_partition_parameter_coordination_for_test(
            stream, reference, &original, &changed, true,
        );
        for slot in &slots {
            let id = slot.parameter.id.as_str();
            self.with_model_operation(|model| {
                with_selected_parameter_values(
                    model.erased_mut(),
                    &BTreeSet::from([id.to_owned()]),
                    stream,
                    |selected| {
                        assert_eq!(
                            host(&selected[id], stream)?,
                            host(&original[id], stream)?,
                            "restoration {id}"
                        );
                        Ok(())
                    },
                )
            })
            .unwrap();
        }
        self.ensure_no_submission_in_flight().unwrap();
        let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
            .unwrap()
            .enter()
            .unwrap();
        let counted = self
            .payload
            .parameter_state
            .parameter_sources()
            .count(&mut guard)
            .unwrap();
        assert_eq!(
            counted
                .counts()
                .role(ParameterOwnerRole::DisplacedOriginal)
                .parameters
                .auxiliary_slots,
            self.payload.parameter_state.originals.len()
        );
        assert_eq!(
            counted
                .counts()
                .role(ParameterOwnerRole::PublishedOverlay)
                .parameters
                .auxiliary_slots,
            self.payload.parameter_state.published.len()
        );
        drop(counted);
        drop(guard);
        let restored = self.parameter_facts().unwrap();
        assert_eq!(restored.identity, facts.identity);
        assert_eq!(restored.parameters, facts.parameters);
        self.ensure_no_submission_in_flight().unwrap();
    }
}
