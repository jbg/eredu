//! Cold-path architecture/backend composition selected by public loaders.

use safemlx::Array;

pub(crate) use crate::backend::nn::shared::MlxNeuralBackend;

impl From<eredu_runtime::ParameterBankLoadOptions>
    for crate::backend::runtime::residency::parameter_bank::ParameterBankOptions
{
    fn from(options: eredu_runtime::ParameterBankLoadOptions) -> Self {
        Self::new(
            options.offload(),
            options.compact_bank_scratch_bytes(),
            options.prefill_compact_bank_target_bytes(),
        )
        .expect("architecture-owned expert-cache options were already validated")
    }
}

pub(crate) fn neural_observer_error(error: crate::backend::error::Error) -> eredu_nn::Error {
    eredu_nn::Error::backend_retained_source(error)
}

/// Canonical NN source constructor plus this adapter's native argument and
/// neural return transports. The actual cause retains its existing payer.
pub(crate) fn retained_observation_error_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let caller = [
        size_of::<crate::backend::error::Error>(),
        size_of::<eredu_nn::Error>(),
        size_of::<Result<(), eredu_nn::Error>>(),
    ];
    caller.into_iter().try_fold(
        eredu_nn::Error::retained_source_construction_bytes::<crate::backend::error::Error>()?,
        usize::checked_add,
    )
}

/// Adapts public MLX-array observation to the neutral tensor/error contract.
pub(crate) struct NeutralActivationObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, crate::backend::error::Error>,
    routed_path: Option<String>,
    routed_invocation_active: bool,
}

impl<'a> NeutralActivationObserver<'a> {
    pub(crate) fn new(
        inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, crate::backend::error::Error>,
    ) -> Self {
        Self {
            inner,
            routed_path: None,
            routed_invocation_active: false,
        }
    }
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralActivationObserver<'_>
{
    fn requires_prepared_traversal(&self) -> bool {
        self.inner.requires_prepared_traversal()
    }

    fn requires_sequence_readout(&self) -> bool {
        self.inner.requires_sequence_readout()
    }
    fn original_speculative_capture(&self) -> Option<eredu_runtime::capture::OriginalSpeculativeCaptureInvocation<'_>> { self.inner.original_speculative_capture() }
    fn retain_original_speculative_capture(&mut self, capture: eredu_core::speculative::SpeculativeActivationCapture) -> Result<(), eredu_runtime::capture::CaptureProtocolError> { self.inner.retain_original_speculative_capture(capture) }
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedPrefillCapture<'_>> {
        self.inner.admitted_prefill_capture()
    }
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&eredu_runtime::working_memory::AdmittedCaptureContinuation<'_>> {
        self.inner.admitted_capture_continuation()
    }
    fn ordinary_prefill_capture(&self) -> Option<&eredu_runtime::capture::OrdinaryPrefillCapture> {
        self.inner.ordinary_prefill_capture()
    }
    fn supports_prefill_spans(&self) -> bool {
        self.inner.supports_prefill_spans()
    }
    fn supports_prefill_context(&self) -> bool {
        self.inner.supports_prefill_context()
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), eredu_nn::Error> {
        self.inner
            .begin_prefill_context(frontier)
            .map_err(neural_observer_error)
    }

    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn eredu_runtime::RoutedUnitObserver<crate::MlxTensor>>, eredu_nn::Error>
    {
        let Some(observer) = self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        else {
            self.routed_path = None;
            self.routed_invocation_active = false;
            return Ok(None);
        };
        // The mutable getter lends the inner scope. Retain only its nesting flag
        // so the immutable query below requires neither a second borrow nor work.
        self.routed_invocation_active = observer.invocation_active();
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(path.into());
        }
        Ok(Some(self))
    }
    fn transactional(&self) -> bool {
        self.inner.transactional()
    }
    fn begin_prefill_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .begin_prefill_chunk(chunk)
            .map_err(neural_observer_error)
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.inner.finish_prefill(committed);
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.inner.requires_prefill_opening_state()
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
        opening: &eredu_runtime::inspection::PrefillOpeningState<'_, crate::MlxTensor>,
    ) -> Result<Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>, eredu_nn::Error>
    {
        opening.with_tensor_adapter(crate::MlxTensor::as_array, |opening| {
            self.inner
                .prepare_prefill_chunk_with_opening(context, opening)
                .map_err(neural_observer_error)
        })
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &eredu_runtime::inspection::PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<eredu_runtime::inspection::PreparedPrefillChunkRetention>, eredu_nn::Error>
    {
        self.inner
            .prepare_prefill_chunk_retention(context)
            .map_err(neural_observer_error)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        settled: eredu_runtime::inspection::SettledPrefillChunkRetention,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .retire_prefill_chunk_retention(settled)
            .map_err(neural_observer_error)
    }

    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: eredu_runtime::ExpertPass,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .prepare_transaction(epoch, pass)
            .map_err(neural_observer_error)
    }
    fn coordinate_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .coordinate_transaction(epoch)
            .map_err(neural_observer_error)
    }
    fn complete_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .complete_transaction(epoch)
            .map_err(neural_observer_error)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.inner.finish_transaction(epoch, committed)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, eredu_nn::Error>
    {
        self.inner
            .routing_control(path, rows)
            .map_err(neural_observer_error)
    }

    fn routing_unmodified_interest(&self, path: &str) -> eredu_runtime::RoutingUnmodifiedInterest {
        self.inner.routing_unmodified_interest(path)
    }
    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: eredu_runtime::RoutingDecision<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .routing_unmodified(
                path,
                eredu_runtime::RoutingDecision {
                    ids: effective.ids.as_array(),
                    coefficients: effective.coefficients.as_array(),
                },
            )
            .map_err(neural_observer_error)
    }

    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<eredu_runtime::RoutingDecision<'_, crate::MlxTensor>>,
        effective: eredu_runtime::RoutingDecision<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .routing_applied(
                path,
                original.map(|value| eredu_runtime::RoutingDecision {
                    ids: value.ids.as_array(),
                    coefficients: value.coefficients.as_array(),
                }),
                eredu_runtime::RoutingDecision {
                    ids: effective.ids.as_array(),
                    coefficients: effective.coefficients.as_array(),
                },
            )
            .map_err(neural_observer_error)
    }

    fn routing_failed(&mut self, path: &str, message: &str) {
        self.inner.routing_failed(path, message);
    }

    fn observe(&mut self, path: &str, value: &crate::MlxTensor) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value.as_array())
            .map_err(neural_observer_error)
    }
    fn observe_replica(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_replica(path, value.as_array())
            .map_err(neural_observer_error)
    }

    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &crate::MlxTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<crate::MlxTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_generated(path, prototype.as_array(), source, &mut || {
                generate()
                    .map(crate::MlxTensor::into_array)
                    .map_err(crate::backend::error::Error::from)
            })
            .map_err(neural_observer_error)
    }

    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &crate::MlxTensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<
            crate::MlxTensor,
            eredu_nn::Error,
        >,
    ) -> Result<(), eredu_nn::Error> {
        let mut mapped = eredu_nn::MappedGeneratedTensorFactory::new(
            factory,
            crate::MlxTensor::as_array,
            crate::MlxTensor::into_array,
            crate::backend::error::Error::from,
            |_: &crate::backend::error::Error| {
                eredu_nn::Error::backend_retained_source(eredu_nn::GeneratedTensorRetentionSignal)
            },
        );
        self.inner
            .observe_generated_retained(path, prototype.as_array(), source, &mut mapped)
            .map_err(neural_observer_error)
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        self.inner
            .intervene(path, value.as_array())
            .map(|value| value.map(crate::MlxTensor::from_array))
            .map_err(neural_observer_error)
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: routing.selected_experts.as_array(),
                selected_scores: routing.selected_scores.as_array(),
                coefficients: routing.coefficients.as_array(),
                routed_output: routing.routed_output.as_array(),
                local_routed_output: routing.local_routed_output.map(crate::MlxTensor::as_array),
                reduced_routed_output: routing
                    .reduced_routed_output
                    .map(crate::MlxTensor::as_array),
                shared_output: routing.shared_output.map(crate::MlxTensor::as_array),
                combined_output: routing.combined_output.map(crate::MlxTensor::as_array),
                expert_count: routing.expert_count,
            })
            .map_err(neural_observer_error)
    }
}

impl eredu_runtime::RoutedUnitObserver<crate::MlxTensor> for NeutralActivationObserver<'_> {
    fn begin_invocation(
        &mut self,
        invocation: &eredu_runtime::RoutedUnitInvocation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.routed_invocation_active = true;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing neutral routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        {
            Some(observer) => observer.begin_invocation(&eredu_runtime::RoutedUnitInvocation {
                input: invocation.input.as_array(),
                origins: invocation.origins,
                unit_coordinates: invocation.unit_coordinates,
            }),
            None => Ok(()),
        }
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), eredu_nn::Error> {
        self.routed_invocation_active = false;
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing neutral routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        {
            Some(observer) => observer.finish_invocation(success),
            None => Ok(()),
        }
    }
    fn invocation_active(&self) -> bool {
        self.routed_invocation_active
    }
    fn observe(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing neutral routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        {
            Some(observer) => observer.observe(&batch.map_tensors(crate::MlxTensor::as_array)),
            None => Ok(()),
        }
    }
    fn intervene(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, crate::MlxTensor>,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing neutral routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        {
            Some(observer) => observer
                .intervene(&batch.map_tensors(crate::MlxTensor::as_array))
                .map(|value| value.map(crate::MlxTensor::from_array)),
            None => Ok(None),
        }
    }
    fn observe_effective(
        &mut self,
        batch: &eredu_runtime::RoutedUnitBatch<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        let path = self
            .routed_path
            .as_deref()
            .ok_or_else(|| eredu_nn::Error::backend("missing neutral routed observer path"))?;
        match self
            .inner
            .routed_unit_observer(path)
            .map_err(neural_observer_error)?
        {
            Some(observer) => {
                observer.observe_effective(&batch.map_tensors(crate::MlxTensor::as_array))
            }
            None => Ok(()),
        }
    }
}

#[cfg(test)]
fn select_architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    mut owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
    mut owns_expert: impl FnMut(eredu_runtime::ParameterBankKey) -> bool,
) -> impl Iterator<Item = eredu_architectures::ExpertResidencyUnit> {
    units.into_iter().filter(move |unit| {
        owns_unit(unit.owner_group(), unit.owner_unit())
            && match unit.distribution() {
                eredu_architectures::ExpertResidencyDistribution::ExpertParallel => {
                    owns_expert(unit.identity())
                }
                eredu_architectures::ExpertResidencyDistribution::Replicated => true,
                _ => false,
            }
    })
}

pub mod mlx;
pub mod moshi;

#[cfg(test)]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;

#[cfg(test)]
mod expert_selection_tests {
    use eredu_architectures::{
        ExpertParameterRecipe, ExpertParameterRole, ExpertResidencyDistribution,
        ExpertResidencyUnit,
    };
    use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
    use eredu_runtime::{ExecutionGroupId, ParameterBankKey};

    fn expert_unit(
        identity: ParameterBankKey,
        group: &str,
        owner_unit: usize,
        distribution: ExpertResidencyDistribution,
    ) -> ExpertResidencyUnit {
        let parameter = ExpertParameterRecipe::new(
            "weight",
            format!("{group}.{owner_unit}.weight"),
            DerivedWeightRecipe::source("source", TensorSelection::Full),
            ExpertParameterRole::Preserved,
        )
        .unwrap();
        ExpertResidencyUnit::new(
            identity,
            ExecutionGroupId::new(group).unwrap(),
            owner_unit,
            format!("{group}.{owner_unit}"),
            distribution,
            [parameter],
        )
        .unwrap()
    }

    #[test]
    fn expert_selection_uses_owner_address_and_distribution_before_lowering() {
        let units = vec![
            expert_unit(
                ParameterBankKey::new(0, 1, 0),
                "target",
                7,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(0, 7, 1),
                "mtp.0",
                1,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(0, 9, 2),
                "mtp.0",
                1,
                ExpertResidencyDistribution::Replicated,
            ),
        ];

        let selected = super::select_architecture_expert_units(
            units,
            |group, unit| group.as_str() == "mtp.0" && unit == 1,
            |identity| identity.member() == 1,
        )
        .map(|unit| unit.identity())
        .collect::<Vec<_>>();

        assert_eq!(
            selected,
            vec![
                ParameterBankKey::new(0, 7, 1),
                ParameterBankKey::new(0, 9, 2)
            ]
        );
    }
}

#[cfg(test)]
#[path = "tests/neutral_observer.rs"]
mod neutral_observer_tests;

#[cfg(test)]
#[path = "tests/retained_generated_factory.rs"]
mod retained_generated_factory_tests;
