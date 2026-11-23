//! Host and device memory observations for Candle placements.

use eredu_core::{
    capability::{ExecutionMemoryPlacement, NativeMemoryFacts, ParameterMemoryFacts},
    AllocatorCachePolicyReport, AllocatorCachePolicySource, AllocatorObservation,
    AllocatorTelemetry, AvailableMemory, BackendFailure, Observed, PhysicalMemorySemantics,
};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use eredu_core::{capability::DeviceMemoryCapacity, ObservationKind};

use crate::{CandleContext, CandlePlacement};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn metal_device(index: usize) -> Option<candle_metal_kernels::metal::Device> {
    candle_metal_kernels::metal::Device::all()
        .into_iter()
        .nth(index)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_semantics(device: &candle_metal_kernels::metal::Device) -> PhysicalMemorySemantics {
    use objc2_metal::MTLDevice;
    if device.as_ref().hasUnifiedMemory() {
        PhysicalMemorySemantics::Unified
    } else {
        PhysicalMemorySemantics::SeparateTiers
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_memory(device: &candle_metal_kernels::metal::Device) -> AvailableMemory {
    let mut memory = eredu_core::host_memory::observe(metal_semantics(device));
    if memory.physical_semantics == PhysicalMemorySemantics::SeparateTiers {
        let recommended = device.recommended_max_working_set_size() as u64;
        let allocated = device.current_allocated_size() as u64;
        memory.device = Some(DeviceMemoryCapacity {
            total_bytes: Observed::unsupported("Metal does not report physical VRAM capacity"),
            available_bytes: if recommended == 0 {
                Observed::unavailable("Metal did not report a recommended working set")
            } else {
                Observed::Available {
                    value: recommended.saturating_sub(allocated),
                    kind: ObservationKind::Estimated,
                    source: "Metal recommended working set less this process's device allocation"
                        .into(),
                }
            },
        });
    }
    memory
}

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{sys::CUdevice_attribute, CudaContext};

#[cfg(feature = "cuda")]
fn cuda_semantics(context: &CudaContext) -> PhysicalMemorySemantics {
    match context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED) {
        Ok(0) => PhysicalMemorySemantics::SeparateTiers,
        Ok(1) => PhysicalMemorySemantics::Unified,
        _ => PhysicalMemorySemantics::Unknown,
    }
}

#[cfg(feature = "cuda")]
fn cuda_memory(context: &CudaContext) -> AvailableMemory {
    let mut memory = eredu_core::host_memory::observe(cuda_semantics(context));
    if memory.physical_semantics == PhysicalMemorySemantics::SeparateTiers {
        memory.device = Some(DeviceMemoryCapacity {
            total_bytes: context.total_mem().map_or_else(
                |error| Observed::unavailable(error.to_string()),
                |total| Observed::exact(total as u64, "CUDA driver device capacity"),
            ),
            available_bytes: context.mem_get_info().map_or_else(
                |error| Observed::unavailable(error.to_string()),
                |(free, _)| Observed::Available {
                    value: free as u64,
                    kind: ObservationKind::Observational,
                    source: "CUDA driver free memory for the selected device".into(),
                },
            ),
        });
    }
    memory
}

pub(crate) fn discover_memory(
    placement: CandlePlacement,
) -> Result<AvailableMemory, BackendFailure> {
    match placement {
        CandlePlacement::Cpu => Ok(eredu_core::host_memory::observe(
            PhysicalMemorySemantics::Unified,
        )),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        CandlePlacement::Metal(index) => metal_device(index)
            .as_ref()
            .map(metal_memory)
            .ok_or_else(|| crate::text::invalid("Metal device index is unavailable")),
        #[cfg(feature = "cuda")]
        CandlePlacement::Cuda(index) => CudaContext::new(index)
            .map(|context| cuda_memory(&context))
            .map_err(|error| BackendFailure::from_error(candle_core::Error::wrap(error))),
        #[allow(unreachable_patterns)]
        _ => Err(crate::text::invalid(
            "accelerator is unavailable in this build",
        )),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_facts(device: &candle_metal_kernels::metal::Device) -> NativeMemoryFacts {
    NativeMemoryFacts {
        allocator_cache_bytes: Observed::unsupported(
            "Candle does not expose Metal buffer pool occupancy",
        ),
        device_allocated_bytes: Observed::Available {
            value: device.current_allocated_size() as u64,
            kind: ObservationKind::Observational,
            source:
                "Metal currentAllocatedSize for this process and device, including cached buffers"
                    .into(),
        },
        physical_semantics: metal_semantics(device),
        ..unplaced_native_facts()
    }
}

pub(crate) fn cache_policy(placement: CandlePlacement) -> Option<AllocatorCachePolicyReport> {
    placement.is_cpu().then_some(AllocatorCachePolicyReport {
        limit_bytes: 0,
        source: AllocatorCachePolicySource::NoBackendCache,
    })
}

fn unplaced_native_facts() -> NativeMemoryFacts {
    NativeMemoryFacts {
        active_allocation_bytes: Observed::unsupported("Candle has no active-allocation counter"),
        allocator_cache_bytes: Observed::unavailable("the execution device has not been selected"),
        device_allocated_bytes: Observed::unavailable("the execution device has not been selected"),
        physical_semantics: PhysicalMemorySemantics::Unknown,
    }
}

fn native_facts(placement: CandlePlacement) -> NativeMemoryFacts {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let CandlePlacement::Metal(index) = placement {
        return metal_device(index)
            .as_ref()
            .map_or_else(unplaced_native_facts, metal_facts);
    }
    NativeMemoryFacts {
        allocator_cache_bytes: if placement.is_cpu() {
            Observed::exact(0, "Candle CPU has no backend-owned allocation cache")
        } else {
            Observed::unsupported("Candle does not expose accelerator buffer pool occupancy")
        },
        device_allocated_bytes: Observed::unsupported(
            "no device allocation counter for this placement",
        ),
        physical_semantics: if placement.is_cpu() {
            PhysicalMemorySemantics::Unified
        } else {
            PhysicalMemorySemantics::Unknown
        },
        ..unplaced_native_facts()
    }
}

pub(crate) fn context_facts(context: &CandleContext) -> NativeMemoryFacts {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let candle_core::Device::Metal(device) = &context.device {
        return metal_facts(device.device());
    }
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(device) = &context.device {
        return NativeMemoryFacts {
            physical_semantics: cuda_semantics(device.cuda_stream().context()),
            ..native_facts(context.placement())
        };
    }
    native_facts(context.placement())
}

pub(crate) fn placement_memory(context: &CandleContext) -> ParameterMemoryFacts {
    let placement = context.placement();
    let native = context_facts(context);
    let available = match &context.device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        candle_core::Device::Metal(device) => metal_memory(device.device()),
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(device) => cuda_memory(device.cuda_stream().context()),
        _ => eredu_core::host_memory::observe(native.physical_semantics),
    };
    let allocator = match &context.device {
        candle_core::Device::Cpu => Some("candle:cpu".into()),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        candle_core::Device::Metal(device) => Some(format!("candle:metal:{:?}", device.id())),
        _ => None,
    };
    ParameterMemoryFacts {
        placement: ExecutionMemoryPlacement {
            device: placement.device_plan(),
            host_execution: placement.is_cpu(),
            allocator,
        },
        native,
        available,
        allocator_cache_limit: if placement.is_cpu() {
            Observed::exact(0, "Candle CPU has no backend-owned allocation cache")
        } else {
            Observed::unsupported("Candle does not expose an accelerator allocator cache limit")
        },
    }
}

pub(crate) fn telemetry(native: NativeMemoryFacts) -> AllocatorTelemetry {
    AllocatorTelemetry {
        active_bytes: native.active_allocation_bytes.value().copied(),
        cache_bytes: native.allocator_cache_bytes.value().copied(),
        device_allocated_bytes: native.device_allocated_bytes.value().copied(),
        ..AllocatorTelemetry::default()
    }
}

pub(crate) fn observation(placement: CandlePlacement) -> AllocatorObservation {
    AllocatorObservation {
        cache_policy: cache_policy(placement),
        telemetry: telemetry(native_facts(placement)),
    }
}
