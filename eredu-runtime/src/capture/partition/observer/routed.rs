//! Sparse provider callbacks spend the ordinary partition observer's authority.
use super::*;

impl<'a, B, T, L, N, F> PartitionCaptureObserver<'a, B, T, L, N, F>
where
    B: CaptureBackend,
    T: PartitionCaptureHookTransport,
    L: PartitionCaptureLayout,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    pub(super) fn prepare_routed_selection(
        &mut self,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Selection<'a, T>, PartitionCaptureObserverError<B::Error>> {
        let placement = self
            .layout
            .routed_capture_placement_at(
                self.session.plan(),
                index,
                phase,
                prediction,
                self.session.invocation,
                self.limits,
            )
            .map_err(CaptureExecutionError::Admission)?;
        let selection = &self.session.plan().plan().selections[index];
        let point = &self.session.plan().points()[index];
        let eredu_core::ObservationValueType::RoutedUnits {
            geometry,
            ref routing,
        } = point.value_type
        else {
            return Err(CaptureExecutionError::Admission(CaptureError::Invalid(
                "routed selection has no bank geometry".into(),
            ))
            .into());
        };
        if placement.routing.is_empty()
            || placement.routing.len() > 4096
            || &placement.routing != routing
            || placement.effective
                != (point.position == eredu_core::ObservationPosition::AfterIntervention)
        {
            return Err(CaptureExecutionError::Admission(CaptureError::Invalid(
                "routed capture placement differs from the admitted invocation or timing".into(),
            ))
            .into());
        }
        let mut limits = self.limits;
        if self.size_receipts {
            let bound = placement
                .producers
                .iter()
                .try_fold(0u64, |largest, producer| {
                    let bytes = producer
                        .projection
                        .fragments()
                        .iter()
                        .enumerate()
                        .try_fold(16_384u64, |bytes, (fragment, _)| {
                            let slice = super::super::routed::global_fragment_slice(
                                &producer.projection,
                                fragment,
                            )?;
                            let native = self.backend.estimate_partition_routed_units(
                                &PartitionRoutedUnitCaptureRequest {
                                    geometry,
                                    source_tokens: producer.projection.global_shape()[0],
                                    ownership: &producer.ownership,
                                    slice: &slice,
                                },
                            )?;
                            let provenance = add(
                                super::super::routed::ownership_bytes(&producer.ownership)?,
                                mul(
                                    producer.ownership.maximum_source_rows(
                                        producer.projection.global_shape()[0],
                                        geometry.routes_per_token,
                                    )?,
                                    64,
                                )?,
                            )?;
                            add(
                                bytes,
                                add(
                                    provenance,
                                    add(
                                        fragment_metadata_usage(selection, point, 3)?.encoded_bytes,
                                        native.encoded_bytes,
                                    )?,
                                )?,
                            )
                        })?;
                    Ok::<_, CaptureError>(largest.max(bytes))
                })
                .map_err(CaptureExecutionError::Admission)?;
            if bound > limits.max_record_bytes {
                return Err(CaptureExecutionError::Admission(CaptureError::Limit {
                    budget: CaptureBudget::Encoded,
                    cumulative: false,
                })
                .into());
            }
            limits.max_record_bytes = bound;
        }
        let mut work = self.session.prepare_partition_routed_capture(
            self.transport,
            index,
            placement.producers,
            limits,
            |request| self.backend.estimate_partition_routed_units(request),
        )?;
        let members = placement.sources.iter().map(|source| source.rank).collect();
        let ownership = placement
            .sources
            .iter()
            .find(|source| source.rank == self.transport.capture_rank())
            .map(|source| source.ownership.clone());
        let hook = self.session.prepare_partition_hook(&mut work, members)?;
        let source = self.session.prepare_partition_routed_source(
            &mut work,
            &self.backend,
            placement.sources,
        )?;
        Ok(Selection {
            work,
            hook: Some(hook),
            source,
            routed: Some(RoutedSelection {
                routing: placement.routing,
                effective: placement.effective,
                ownership,
            }),
        })
    }
}

impl<B, T, L, N, F> crate::RoutedUnitObserver<B::Tensor>
    for PartitionCaptureObserver<'_, B, T, L, N, F>
where
    B: CaptureBackend,
    B::Error: Send + Sync,
    T: PartitionCaptureHookTransport,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn invocation_active(&self) -> bool {
        self.routed_active
    }

    fn begin_invocation(
        &mut self,
        invocation: &crate::RoutedUnitInvocation<'_, B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        let started = std::time::Instant::now();
        let result = (|| {
            if self.routed_active {
                return Err(eredu_nn::Error::backend(
                    "routed capture invocation already started",
                ));
            }
            let path = self
                .routed_path
                .as_deref()
                .ok_or_else(|| eredu_nn::Error::backend("missing routed capture invocation"))?;
            self.routed_active = true;
            let mut failure = None;
            // Every selected before/after record has prepaid source votes. Settle
            // them all in plan order even after a preceding selection rejects.
            for selection in &mut self.selections {
                let Some(routed) = &selection.routed else {
                    continue;
                };
                if routed.routing != path {
                    continue;
                }
                let local = (|| {
                    let owned = routed.ownership.as_ref().ok_or_else(|| {
                        eredu_nn::Error::backend("routed capture reached a nonmember invocation")
                    })?;
                    let source = selection.source.take().ok_or_else(|| {
                        eredu_nn::Error::backend("routed source was already prepared")
                    })?;
                    self.session
                        .ready_partition_routed_source(
                            &mut selection.work,
                            source,
                            &mut self.backend,
                            invocation,
                            failure.is_none(),
                        )
                        .map_err(eredu_nn::Error::backend_retained_source)?;
                    let shape = self
                        .backend
                        .shape(invocation.input)
                        .map_err(eredu_nn::Error::backend_retained_source)?;
                    let rows = super::super::session::routed_input_rows(&shape)
                        .map_err(eredu_nn::Error::backend_retained_source)?;
                    let dtype = self
                        .backend
                        .source_dtype(invocation.input)
                        .ok_or_else(|| eredu_nn::Error::backend("routed source lost its dtype"))?;
                    self.session
                        .begin_partition_routed_capture(
                            &mut selection.work,
                            rows,
                            dtype,
                            owned.coordinates.units(),
                            invocation
                                .origins
                                .map(|origins| origins.capture_coordinates()),
                        )
                        .map_err(eredu_nn::Error::backend_retained_source)
                })();
                if let Err(error) = local {
                    self.session.failed_partition_source(&mut selection.work);
                    failure.get_or_insert(error);
                }
            }
            for work in &mut self.interventions {
                if work.routed_path() != Some(path) {
                    continue;
                }
                let result = (self
                    .intervention_callbacks
                    .as_ref()
                    .expect("prepared sparse mechanism")
                    .begin_routed)(
                    self.session,
                    work,
                    &mut self.backend,
                    invocation,
                    failure.is_none(),
                )
                .map_err(eredu_nn::Error::backend_retained_source);
                if let Err(error) = result {
                    failure.get_or_insert(error);
                }
            }
            failure.map_or(Ok(()), Err)
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        self.routed_result(result)
    }

    fn observe(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.capture_routed_batch(batch, false)
    }
    fn observe_effective(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.capture_routed_batch(batch, true)
    }

    fn intervene(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, eredu_nn::Error> {
        let started = std::time::Instant::now();
        let result = (|| {
            let path = self
                .routed_path
                .as_deref()
                .ok_or_else(|| eredu_nn::Error::backend("missing sparse edit invocation"))?;
            if !self.routed_active {
                return Err(eredu_nn::Error::backend(
                    "sparse edit invocation is inactive",
                ));
            }
            let mut effective = None;
            for work in &mut self.interventions {
                if work.routed_path() != Some(path) {
                    continue;
                }
                let mut source = batch
                    .partition_capture_source()
                    .map_err(eredu_nn::Error::backend_retained_source)?;
                source.source.values = effective.as_ref().unwrap_or(batch.units.values);
                let output = (self
                    .intervention_callbacks
                    .as_ref()
                    .expect("prepared sparse mechanism")
                    .apply_routed)(
                    self.session, work, &mut self.backend, &source
                )
                .map_err(eredu_nn::Error::backend_retained_source)?;
                if output.is_some() {
                    effective = output;
                }
            }
            Ok(effective)
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        self.routed_result(result)
    }

    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        let started = std::time::Instant::now();
        let result = (|| {
            let path = self
                .routed_path
                .as_deref()
                .ok_or_else(|| eredu_nn::Error::backend("missing routed capture invocation"))?;
            let active = std::mem::take(&mut self.routed_active);
            let mut failure = None;
            // Finish every record and its prepaid group vote before a provider
            // can enter reverse exchange. No record is published here.
            for selection in &mut self.selections {
                let Some(routed) = &selection.routed else {
                    continue;
                };
                if routed.routing != path {
                    continue;
                }
                let local = self
                    .session
                    .finish_partition_routed_capture(&mut selection.work, success && active)
                    .map_err(eredu_nn::Error::backend_retained_source);
                if local.is_err() {
                    self.session.failed_partition_source(&mut selection.work);
                }
                let agreed = match selection.hook.take() {
                    Some(hook) => self
                        .session
                        .agree_partition_hook(&mut selection.work, hook, local.is_ok())
                        .map_err(eredu_nn::Error::backend_retained_source)
                        .and_then(|agreed| {
                            if agreed {
                                Ok(())
                            } else {
                                Err(eredu_nn::Error::backend_retained_source(
                                    PartitionCaptureObserverError::<B::Error>::HookRejected,
                                ))
                            }
                        }),
                    None => Err(eredu_nn::Error::backend(
                        "routed capture invocation already finished",
                    )),
                };
                if let Err(error) = local.and(agreed) {
                    failure.get_or_insert(error);
                }
            }
            for work in &mut self.interventions {
                if work.routed_path() != Some(path) {
                    continue;
                }
                let result = (self
                    .intervention_callbacks
                    .as_ref()
                    .expect("prepared sparse mechanism")
                    .finish_routed)(
                    self.session, work, success && active && failure.is_none()
                )
                .map_err(eredu_nn::Error::backend_retained_source);
                if let Err(error) = result {
                    failure.get_or_insert(error);
                }
            }
            failure.map_or(Ok(()), Err)
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        self.routed_result(result)
    }
}

impl<B, T, L, N, F> PartitionCaptureObserver<'_, B, T, L, N, F>
where
    B: CaptureBackend,
    B::Error: Send + Sync,
    T: PartitionCaptureHookTransport,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn capture_routed_batch(
        &mut self,
        batch: &crate::RoutedUnitBatch<'_, B::Tensor>,
        effective: bool,
    ) -> Result<(), eredu_nn::Error> {
        let started = std::time::Instant::now();
        let result = (|| {
            if !self.routed_active {
                return Err(eredu_nn::Error::backend(
                    "routed capture has no active provider",
                ));
            }
            let path = self.routed_path.as_deref().expect("active invocation");
            let source = batch
                .partition_capture_source()
                .map_err(eredu_nn::Error::backend_retained_source)?;
            for selection in &mut self.selections {
                let Some(routed) = &selection.routed else {
                    continue;
                };
                if routed.routing != path || routed.effective != effective {
                    continue;
                }
                self.session
                    .observe_partition_routed_units(&mut selection.work, &mut self.backend, &source)
                    .map_err(eredu_nn::Error::backend_retained_source)?;
            }
            Ok(())
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        self.routed_result(result)
    }
}
