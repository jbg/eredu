//! Native placement and resources for the portable execution-plan driver.

use eredu_core::{
    automatic::{
        ColdPlanPlacement, HardwareBackendProfile, HardwareDeviceKind, HardwareDeviceProfile,
    },
    preparation::{PlanMechanismFacts, ResidencyDiagnostics},
    AutomaticPlanningError, BackendFailure, BackendHardware, BackendId, DevicePlan,
    HardwareProfile, MediaFeatureAvailability,
};

use crate::{CandleContext, CandlePlacement, CandlePreparation};

/// Opens the exact configured Candle device for a portable execution plan.
#[derive(Clone, Copy, Debug, Default)]
pub struct CandleBackendFactory {
    pub(crate) placement: CandlePlacement,
    diagnostics: ResidencyDiagnostics,
}

impl CandleBackendFactory {
    /// Configures device-specific cold mechanism facts without creating a device.
    pub fn for_device(device: &DevicePlan) -> Result<Self, AutomaticPlanningError> {
        Ok(Self {
            placement: CandlePlacement::from_plan(device)?,
            diagnostics: ResidencyDiagnostics::default(),
        })
    }

    /// Enables device and current-process observations for bounded residency.
    pub const fn with_residency_diagnostics(mut self, backend: bool, process: bool) -> Self {
        self.diagnostics = ResidencyDiagnostics::new(backend, process);
        self
    }

    fn draft_placement(
        &self,
        device: &DevicePlan,
    ) -> Result<CandlePlacement, AutomaticPlanningError> {
        let draft = CandlePlacement::from_plan(device)?;
        if !self.placement.supports_local_text_extensions()
            || !draft.supports_local_text_extensions()
        {
            return Err(AutomaticPlanningError::Invalid(
                "Candle external drafting requires CPU or Metal devices".into(),
            ));
        }
        Ok(draft)
    }

    /// Opens the configured device for the shared prepared loader.
    pub fn open(self) -> Result<CandleContext, BackendFailure> {
        CandleContext::open(&self.placement.device_plan())
    }
}

impl BackendHardware for CandleBackendFactory {
    fn backend_id(&self) -> BackendId {
        BackendId::new("candle").expect("valid backend identifier")
    }

    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        let device = self.placement.device_plan();
        let (family, index) = self.placement.family_index();
        let (memory, detail) = match crate::memory::discover_memory(self.placement) {
            Ok(memory) => (memory, None),
            Err(error) => (
                eredu_core::host_memory::observe(eredu_core::PhysicalMemorySemantics::Unknown),
                Some(error.to_string()),
            ),
        };
        let capacity = memory.selected_capacity(self.placement.is_cpu());
        let devices = if detail.is_none() {
            vec![HardwareDeviceProfile {
                id: device.device().to_owned(),
                kind: if self.placement.is_cpu() {
                    HardwareDeviceKind::Cpu
                } else {
                    HardwareDeviceKind::Gpu
                },
                family: family.to_owned(),
                index,
                total_memory_bytes: capacity.total_bytes,
                available_memory_bytes: capacity.available_bytes,
            }]
        } else {
            Vec::new()
        };
        Ok(HardwareProfile::observe_host(
            memory.physical_memory_bytes,
            memory.available_memory_bytes,
            memory.physical_semantics.into(),
            vec![HardwareBackendProfile {
                backend: self.backend_id(),
                available: detail.is_none(),
                detail,
                devices,
            }],
        ))
    }
}

impl PlanMechanismFacts for CandleBackendFactory {
    type Preparation = CandlePreparation;

    fn preparation_mechanisms(&self) -> Self::Preparation {
        CandlePreparation::new(self.placement)
    }

    fn draft_preparation_mechanisms(
        &self,
        device: &DevicePlan,
    ) -> Result<CandlePreparation, AutomaticPlanningError> {
        Ok(CandlePreparation::new(self.draft_placement(device)?))
    }

    fn residency_diagnostics(&self) -> ResidencyDiagnostics {
        self.diagnostics
    }

    fn media_features(&self) -> MediaFeatureAvailability {
        self.preparation_mechanisms().media_features()
    }
}

impl eredu_core::preparation::BackendInspection for CandleBackendFactory {
    fn device_uses_host_memory(&self, device: &DevicePlan) -> Result<bool, BackendFailure> {
        self.plan_placement(device, None)
            .map(CandlePlacement::is_cpu)
            .map_err(|error| {
                BackendFailure::new(eredu_core::BackendFailureKind::InvalidInput, error)
            })
    }

    fn proposed_allocator_cache_limit(&self, bytes: u64) -> Result<u64, BackendFailure> {
        if self.placement.is_cpu() && bytes == 0 {
            return Ok(0);
        }
        Err(BackendFailure::new(
            eredu_core::BackendFailureKind::Unsupported,
            std::io::Error::other(if self.placement.is_cpu() {
                "Candle CPU has no backend-owned allocator cache"
            } else {
                "Candle does not expose accelerator allocator cache controls"
            }),
        )
        .with_operation("allocator cache policy"))
    }

    fn set_allocator_cache_limit(&self, bytes: u64) -> Result<(), BackendFailure> {
        self.proposed_allocator_cache_limit(bytes).map(|_| ())
    }

    fn allocator_observation(
        &self,
    ) -> Result<Option<eredu_core::AllocatorObservation>, BackendFailure> {
        Ok(Some(crate::memory::observation(self.placement)))
    }
}

impl ColdPlanPlacement for CandleBackendFactory {
    type Placement = CandlePlacement;

    fn plan_placement(
        &self,
        device: &DevicePlan,
        rank: Option<(usize, usize)>,
    ) -> Result<Self::Placement, AutomaticPlanningError> {
        let placement = CandlePlacement::from_plan(device)?;
        if placement != self.placement {
            return Err(AutomaticPlanningError::Invalid(
                "Candle device differs from the configured placement".into(),
            ));
        }
        if rank.is_some() {
            return Err(AutomaticPlanningError::Invalid(
                "Candle distributed placement is unavailable".into(),
            ));
        }
        Ok(placement)
    }
}

impl eredu_core::automatic::ExecutionPlanNativeFactory for CandleBackendFactory {
    type Backend<D: 'static> = CandleContext;
    type AssistantBinder = CandleContext;
    type DraftBinding<D: 'static, P: 'static> = CandleDraftBinding<P>;

    fn realize_placement<D: 'static>(
        &self,
        device: &DevicePlan,
        placement: &CandlePlacement,
    ) -> Result<CandleContext, AutomaticPlanningError> {
        if self.plan_placement(device, None)? != *placement {
            return Err(AutomaticPlanningError::Invalid(
                "Candle device differs from the retained placement".into(),
            ));
        }
        self.open()
            .map_err(|error| planning_error("open_candle_device", error))
    }

    fn open_draft<D: 'static, P: 'static>(
        &self,
        binding: eredu_core::automatic::ExternalDraftBinding<'_, CandleContext>,
    ) -> Result<CandleDraftBinding<P>, AutomaticPlanningError> {
        use eredu_core::DraftPlacementPlan;

        let (placement, target) = binding.into_parts();
        if target.placement() != self.placement {
            return Err(AutomaticPlanningError::Invalid(
                "Candle draft binding has a different target placement".into(),
            ));
        }
        target
            .check_healthy()
            .map_err(|error| planning_error("open_candle_draft", error))?;
        let device = self.placement.device_plan();
        let draft = self.draft_placement(placement.execution_device(&device))?;
        let context = match placement {
            DraftPlacementPlan::Target => target.clone(),
            DraftPlacementPlan::Device { .. } => CandleContext::open(&draft.device_plan())
                .map_err(|error| planning_error("open_candle_draft", error))?,
            _ => {
                return Err(AutomaticPlanningError::Invalid(
                    "unsupported draft placement".into(),
                ))
            }
        };
        Ok(CandleDraftBinding {
            context,
            payload: std::marker::PhantomData,
        })
    }
}

#[doc(hidden)]
pub struct CandleDraftBinding<P> {
    context: CandleContext,
    payload: std::marker::PhantomData<fn() -> P>,
}

impl<P: 'static> eredu_core::automatic::DraftBindingResources for CandleDraftBinding<P> {
    type Generation = eredu_core::backend::CausalTextGeneration;
    type Backend = CandleContext;
    type AssistantBinder = CandleContext;
    type Payload = P;
    type Resources = eredu_core::eager_completion::EagerRetainedPayload<P, CandleContext>;

    fn assistant_binder(&self) -> CandleContext {
        self.context.clone()
    }

    fn model_binding(&self) -> (&CandleContext, CandlePlacement) {
        (&self.context, self.context.placement())
    }

    fn finish_binding(self, payload: P) -> Result<Self::Resources, AutomaticPlanningError> {
        let payload = crate::device::CandleSettlement::new(&self.context, payload)
            .finish()
            .map_err(|error| planning_error("complete_candle_draft_load", error))?;
        Ok(eredu_core::eager_completion::EagerRetainedPayload::new(
            payload,
            self.context,
        ))
    }
}

fn planning_error(operation: &'static str, error: BackendFailure) -> AutomaticPlanningError {
    AutomaticPlanningError::Backend {
        operation,
        message: error.to_string(),
    }
}
