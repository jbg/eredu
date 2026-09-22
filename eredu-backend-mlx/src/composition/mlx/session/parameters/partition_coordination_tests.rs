//! Actual Ring transport, prepared loans, native contractions and rollback.
//! These exercise the live driver beneath the public global parameter API.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use eredu_runtime::parameter_operations::*;
use std::cell::RefCell;

#[derive(Default)]
struct Budget {
    used: CaptureUsage,
    denied: bool,
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        if self.denied {
            return Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: true,
            });
        }
        self.used = self.used.checked_add(cost)?;
        assert!(self.used.host_bytes < 16 << 20 && self.used.retained_bytes < 16 << 20);
        Ok(None)
    }
}

impl MlxModelSession {
    pub(super) fn verify_partition_parameter_coordination_for_test(
        &mut self,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        reference: &mut ModelRuntime<MlxBackend<'_>>,
        original: &ParameterReplacementValues<MlxTensor>,
        changed: &ParameterReplacementValues<MlxTensor>,
        remove: bool,
        operation: &numerical::PreparedOperation<'_>,
    ) {
        let retained = operation.sources.borrow();
        let mut source_rows = operation.context.metadata_vec(retained.len()).unwrap();
        source_rows.extend(retained.iter().cloned());
        drop(retained);
        let sources = CompletedParameterSources::from_prepared(
            source_rows,
            &operation.context,
            operation.funding.clone(),
        )
        .unwrap();
        let source_values = sources
            .select(
                if remove { original } else { changed },
                &operation.context,
                operation.funding.clone(),
            )
            .unwrap();
        let transport = self.payload.distributed.clone().unwrap();
        let owner = transport.parameter_operations().unwrap();
        let prepared_transport = self.prepared_parameter_transport(&transport).unwrap();
        let binding = self.parameter_operation_binding().unwrap();
        let rank = transport.parameter_rank();
        let before = owner.usage().unwrap();
        if !remove {
            for failure_mode in 0..2 {
                let mut denied = Budget {
                    denied: failure_mode == 0 && rank == 1,
                    ..Default::default()
                };
                let local = if failure_mode == 1 && rank == 1 {
                    Err(ParameterError::Invalid(
                        "local preparation failed before read geometry was available".into(),
                    ))
                } else {
                    Ok(ParameterReadPreparation::new((), [17; 32], 1))
                };
                let rejected = owner.read(
                    &prepared_transport,
                    binding,
                    &[17; 32],
                    ParameterOperationKind::Query,
                    local,
                    &mut denied,
                    |_| panic!("preparation rejection must precede parameter work on every rank"),
                    |_| Ok(()),
                );
                assert!(matches!(rejected, Err(ParameterError::Coordination(_))));
                assert_eq!(
                    owner.usage().unwrap().attempts,
                    before.attempts + failure_mode + 1
                );
            }
            self.verify_native_global_parameter_read_for_test(environment, reference, binding, 0.0);
        }
        for fail in if remove {
            vec![false]
        } else {
            vec![true, false]
        } {
            let session = RefCell::new(&mut *self);
            let result = owner.transaction(
                &prepared_transport,
                binding,
                &[if remove { 23 } else { 19 }; 32],
                if remove {
                    ParameterOperationKind::Removal
                } else {
                    ParameterOperationKind::Activation
                },
                Ok(()),
                |_| {
                    session
                        .borrow_mut()
                        .prepare_parameter_publication(
                            if remove {
                                original.clone()
                            } else {
                                changed.clone()
                            },
                            source_values.clone(),
                            !remove,
                            operation,
                        )
                        .map_err(failure)
                },
                |(publication, reset)| {
                    session
                        .borrow_mut()
                        .commit_parameter_publication(publication, reset)
                        .map_err(failure)?;
                    if fail && rank == 1 {
                        Err(ParameterError::Invalid(
                            "injected rejection after parameter publication".into(),
                        ))
                    } else {
                        Ok(())
                    }
                },
                |(publication, reset)| {
                    session
                        .borrow_mut()
                        .commit_parameter_publication(publication, reset)
                        .map_err(failure)
                },
            );
            drop(session);
            if fail {
                assert!(matches!(result, Err(ParameterError::Coordination(_))));
                self.verify_native_global_parameter_read_for_test(
                    environment,
                    reference,
                    binding,
                    0.0,
                );
            } else {
                result.unwrap();
                self.verify_native_global_parameter_read_for_test(
                    environment,
                    reference,
                    binding,
                    if remove { 0.0 } else { 0.125 },
                );
            }
        }
        assert!(owner.usage().unwrap().reserved.host_bytes > before.reserved.host_bytes);
    }

    fn verify_native_global_parameter_read_for_test(
        &mut self,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        reference: &mut ModelRuntime<MlxBackend<'_>>,
        binding: ParameterOperationBinding,
        delta: f32,
    ) {
        let target = "model.layers.0.self_attn.o_proj.weight";
        let transport = self.payload.distributed.clone().unwrap();
        let owner = transport.parameter_operations().unwrap();
        let prepared_transport = self.prepared_parameter_transport(&transport).unwrap();
        let mut metadata = Budget::default();
        let prepared = self.capture_discovery.as_ref().unwrap();
        let layouts = (0..transport.parameter_setup().participant_count())
            .map(|rank| {
                prepared
                    .parameter_partition_layout_for_rank(target, rank, &mut metadata)
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let global = layouts[0].global_shape().to_vec();
        let region = ParameterRegion {
            starts: vec![1, 1],
            shape: vec![2, global[1] - 2],
        };
        let coordinates = layouts
            .iter()
            .map(|layout| layout.coordinates())
            .collect::<Vec<_>>();
        let facts = MlxBackend::parameter_discovery(reference).unwrap();
        let limits = CaptureUsage {
            captures: 100_000,
            retained_bytes: 512 << 20,
            host_bytes: 512 << 20,
            encoded_bytes: 512 << 20,
        };
        // The same collective catalog owns each local materialization layout.
        // Complete that exchange before entering the lower-level read protocol.
        let catalog = self.partition_parameter_catalog(Some(limits)).unwrap();
        for axis in [None, Some(0), Some(1)] {
            let projection = axis.map(|axis| ParameterProjection {
                region: region.clone(),
                axis,
                directions: 2,
                coefficients: (0..region.shape[axis] * 2)
                    .map(|i| if i % 2 == 0 { 1.25 } else { -0.5 })
                    .collect(),
            });
            let plan = PartitionParameterReadPlan::new(
                &global,
                &region,
                projection.as_ref(),
                &coordinates,
                128,
                &mut metadata,
            )
            .unwrap();
            let mut native_fragments = Vec::new();
            for fragment in plan
                .fragments()
                .iter()
                .filter(|fragment| fragment.rank() == transport.parameter_rank())
            {
                let projection = projection.as_ref().map(|projection| {
                    fragment
                        .source()
                        .project_contraction(projection, &global, &mut metadata)
                        .unwrap()
                });
                native_fragments.push((fragment.source().local().clone(), projection));
            }
            // Prepay fixture source work, host outputs, coefficient copies and
            // assembly independently from the driver's mandatory control frames.
            metadata
                .reserve_quota(CaptureUsage {
                    retained_bytes: 1 << 20,
                    host_bytes: 1 << 20,
                    ..Default::default()
                })
                .unwrap();
            let assembly_budget = RefCell::new(Budget::default());
            let values = owner
                .read(
                    &prepared_transport,
                    binding,
                    plan.identity(),
                    if projection.is_some() {
                        ParameterOperationKind::Projection
                    } else {
                        ParameterOperationKind::Query
                    },
                    Ok(ParameterReadPreparation::new(
                        (),
                        *plan.identity(),
                        plan.max_rank_words(),
                    )),
                    &mut metadata,
                    |_| {
                        if native_fragments.is_empty() {
                            return Ok(Vec::new());
                        }
                        let mut output = prepared_transport
                            .funding()
                            .metadata_vec(plan.rank_counts()[transport.parameter_rank()])
                            .map_err(|cause| failure(Error::Neural(cause)))?;
                        for (region, projection) in &native_fragments {
                            let request = match projection {
                                Some(projection) => {
                                    numerical::Request::Project(projection.projection())
                                }
                                None => numerical::Request::Read(region),
                            };
                            let values = self
                                .read_resident_effective_parameter(
                                    target,
                                    &catalog.layouts[target],
                                    request,
                                    environment,
                                )
                                .map_err(failure)?
                                .expect("actual prepared fragment");
                            assert!(output.len() + values.len() <= output.capacity());
                            output.extend(values.into_iter().map(f32::to_bits));
                        }
                        Ok(output)
                    },
                    |rows| {
                        let rows = rows
                            .iter()
                            .map(|row| row.iter().copied().map(f32::from_bits).collect::<Vec<_>>())
                            .collect::<Vec<_>>();
                        plan.assemble(&rows, &mut *assembly_budget.borrow_mut())
                    },
                )
                .unwrap();
            let mut expected = match &projection {
                Some(projection) => MlxBackend::project_parameter(
                    reference,
                    &facts.identity,
                    target,
                    projection.clone(),
                    limits,
                )
                .unwrap()
                .values
                .clone(),
                None => MlxBackend::query_parameter(
                    reference,
                    &facts.identity,
                    target,
                    region.clone(),
                    limits,
                )
                .unwrap()
                .values
                .clone(),
            };
            if let Some(projection) = &projection {
                let width = region.shape[projection.axis] as usize;
                for (index, value) in expected.iter_mut().enumerate() {
                    let direction = index % projection.directions as usize;
                    *value += delta
                        * projection.coefficients[direction * width..(direction + 1) * width]
                            .iter()
                            .sum::<f32>();
                }
            } else {
                for value in &mut expected {
                    *value += delta;
                }
            }
            assert_eq!(values.len(), expected.len());
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "global parameter value {actual} vs {expected}"
                );
            }
        }
    }
}
