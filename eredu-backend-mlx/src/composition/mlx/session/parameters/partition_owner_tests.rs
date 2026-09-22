//! Native local-owner checks within the existing prepared Ring fixtures.
//! The public peer transaction tests separately cover version/cache lifecycle.
use super::*;
#[path = "partition_coordination_tests.rs"]
mod coordination;

impl MlxModelSession {
    pub(crate) fn verify_partition_parameter_owner_for_test(
        &mut self,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        reference: &mut ModelRuntime<MlxBackend<'_>>,
    ) {
        let stream = environment.stream();
        let catalog = self.partition_parameter_catalog(None).unwrap();
        let (facts, layouts) = (catalog.discovery, catalog.layouts);
        let operation = numerical::PreparedOperation::new(self, environment).unwrap();
        assert!(
            facts
                .parameters
                .iter()
                .any(|parameter| parameter.access().query)
        );
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
        let mut original = operation.context.metadata_vec(slots.len()).unwrap();
        let mut changed = operation.context.metadata_vec(slots.len()).unwrap();
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
                    let ids = [Some(slot.parameter.id.as_str())];
                    let preparation = operation.preparation(&ids);
                    self.with_model_operation_funded(operation.funding.clone(), |model| {
                        let result = model.erased_mut().with_parameter_slots(
                            &location,
                            &Default::default(),
                            &mut |_| {
                                panic!(
                                    "invalid owner must fail before traversal or materialization"
                                )
                            },
                            stream,
                            Some(&preparation),
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
            let layout = &layouts[id];
            let region = ParameterRegion {
                starts: vec![0; layout.shape.len()],
                shape: layout.shape.clone(),
            };
            let values = self
                .read_resident_effective_parameter(
                    id,
                    layout,
                    numerical::Request::Read(&region),
                    environment,
                )
                .unwrap()
                .unwrap();
            assert_eq!(
                values.len(),
                slot.materialized.shape.iter().product::<usize>()
            );
            let rejected = self.reject_parameter_callback_for_test(id, layout, &region, &operation);
            let rejected = rejected.unwrap_err();
            let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&rejected);
            let mut injected = false;
            while let Some(cause) = source {
                injected |= cause.is::<numerical::InjectedParameterCallbackFailure>();
                source = cause.source();
            }
            assert!(
                injected,
                "callback failure must preserve its exact typed cause: {rejected:?}"
            );
            assert_eq!(
                self.read_resident_effective_parameter(
                    id,
                    layout,
                    numerical::Request::Read(&region),
                    environment
                )
                .unwrap()
                .unwrap(),
                values
            );
            observed_nonzero |= values.iter().any(|n| n.abs() > 1e-4);
            let update = ParameterUpdate::Add {
                values: vec![0.125; values.len()],
            };
            let (before, after) = self
                .prepare_parameter_update(id, layout, &[(&region, &update)], &operation)
                .unwrap();
            operation.context.charge_metadata(id.len() * 2).unwrap();
            original.push((id.to_owned(), before));
            changed.push((id.to_owned(), after));
            expected.insert(
                id.to_owned(),
                values.iter().map(|value| value + 0.125).collect::<Vec<_>>(),
            );
        }
        assert!(observed_nonzero);
        assert_eq!(original.len(), slots.len());
        assert_eq!(changed.len(), slots.len());
        assert_eq!(expected.len(), slots.len());
        let original = parameter_rows(original, &operation).unwrap();
        let changed = parameter_rows(changed, &operation).unwrap();
        self.verify_partition_parameter_coordination_for_test(
            environment,
            reference,
            &original,
            &changed,
            false,
            &operation,
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
            let layout = &layouts[id];
            let region = ParameterRegion {
                starts: vec![0; layout.shape.len()],
                shape: layout.shape.clone(),
            };
            let values = self
                .read_resident_effective_parameter(
                    id,
                    layout,
                    numerical::Request::Read(&region),
                    environment,
                )
                .unwrap()
                .unwrap();
            assert_eq!(values, expected[id], "replacement {id}");
        }
        self.verify_partition_parameter_coordination_for_test(
            environment,
            reference,
            &original,
            &changed,
            true,
            &operation,
        );
        for slot in &slots {
            let id = slot.parameter.id.as_str();
            let layout = &layouts[id];
            let region = ParameterRegion {
                starts: vec![0; layout.shape.len()],
                shape: layout.shape.clone(),
            };
            let values = self
                .read_resident_effective_parameter(
                    id,
                    layout,
                    numerical::Request::Read(&region),
                    environment,
                )
                .unwrap()
                .unwrap();
            let before = expected[id]
                .iter()
                .map(|value| value - 0.125)
                .collect::<Vec<_>>();
            for (actual, expected) in values.iter().zip(before) {
                assert!((actual - expected).abs() < 1e-6, "restoration {id}");
            }
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
